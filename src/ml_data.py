"""B0 supervised data protocol; simulation stays complex128, no normalization.

Schedule functions enumerate metadata only: no formal training, validation
performance scan, formal-test channel or OOD materialization is implemented.
"""

from pathlib import Path

import numpy as np

from src.estimators import ls_linear_interpolate
from src.link import PILOT_SYMBOLS, apply_frequency_mimo_channel
from src.lmmse_train import read_manifest, sha256_file
from src.numerics import complex_gaussian_noise, noise_variance_from_snr
from src.pilots import map_pilot_symbols


SNR_DB = (0, 5, 10, 15, 20)
DENSITIES = (2, 4, 8, 16)
TRAINING_SEED = 10261001
SNR_DOMAIN = 12001
TRAIN_NOISE_DOMAIN = 12002
CALIBRATION_NOISE_DOMAIN = 12003
MANIFEST_SHA256 = "03c4fc237c374741df0d3aa43956461e4870885cf851e1f63640f00041c488f4"
TENSOR_MAPPING = tuple(f"{'Re' if component == 0 else 'Im'} H[RX{rx},TX{tx}]"
                       for rx in range(2) for tx in range(2) for component in range(2))


def _integer(name, value, upper=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < 0 or (upper is not None and value >= upper)):
        raise ValueError(f"invalid {name}")


def frozen_manifest(path):
    if sha256_file(Path(path)) != MANIFEST_SHA256:
        raise ValueError("frozen train/validation manifest SHA256 mismatch")
    return read_manifest(path)


def _check_manifest(rows):
    # Check the existing specification without writing or resplitting anything.
    from src.lmmse_train import formal_manifest_rows
    if rows != formal_manifest_rows():
        raise ValueError("exact frozen train/validation manifest required")


def balanced_training_snr(training_seed, epoch):
    """One SNR per canonical train frame, 4000 of each; epoch-specific permutation."""
    _integer("training_seed", training_seed, 2**32)
    _integer("epoch", epoch, 2**32)
    labels = np.repeat(np.array(SNR_DB, dtype=np.int64), 4000)
    rng = np.random.default_rng(np.random.SeedSequence([SNR_DOMAIN, training_seed, epoch]))
    return rng.permutation(labels)


def training_plan(manifest, training_seed, epoch):
    _check_manifest(manifest)
    snrs = balanced_training_snr(training_seed, epoch)
    return [{"split": "train", "trajectory_id": tid, "frame_index": frame,
             "snr_db": int(snrs[10*tid+frame])}
            for tid in range(2000) for frame in range(10)]


def validation_plan(manifest):
    _check_manifest(manifest)
    return [{"split": "validation_id", "trajectory_id": tid, "frame_index": frame,
             "snr_db": snr} for tid in range(2000, 2400)
            for frame in range(10) for snr in SNR_DB]


def training_unit_noise(training_seed, epoch, trajectory_id, frame_index):
    for name, value, upper in (("training_seed", training_seed, 2**32),
            ("epoch", epoch, 2**32), ("training trajectory", trajectory_id, 2000),
            ("frame_index", frame_index, 10)):
        _integer(name, value, upper)
    seed = np.random.SeedSequence([TRAIN_NOISE_DOMAIN, training_seed, epoch,
                                  trajectory_id, frame_index])
    return complex_gaussian_noise((2, 64, 16), 1.0, np.random.default_rng(seed))


def validation_unit_noise(trajectory_id, frame_index):
    _integer("validation trajectory", trajectory_id, 2400)
    _integer("frame_index", frame_index, 10)
    if trajectory_id < 2000:
        raise ValueError("validation requires trajectory_id in [2000,2400)")
    # Exact fixed T10D full-grid Z, reused across all SNRs, densities and model seeds.
    seed = 90261001 + 100*trajectory_id + frame_index
    return complex_gaussian_noise((2, 64, 16), 1.0, np.random.default_rng(seed))


def calibration_unit_noise(training_seed, trajectory_id, frame_index):
    _integer("training_seed", training_seed, 2**32)
    _integer("calibration trajectory", trajectory_id, 4)
    _integer("frame_index", frame_index, 10)
    seed = np.random.SeedSequence([CALIBRATION_NOISE_DOMAIN, training_seed,
                                  trajectory_id, frame_index])
    return complex_gaussian_noise((2, 64, 16), 1.0, np.random.default_rng(seed))


def direct_pilot_ls(h_true, layout, unit_noise, snr_db, pilot_symbols=PILOT_SYMBOLS):
    """Truth is used by the simulator only, at the TX owner's actual pilot REs."""
    h = np.asarray(h_true)
    z = np.asarray(unit_noise)
    if (h.shape != (2, 2, 64, 16) or h.dtype != np.complex128
            or z.shape != (2, 64, 16) or z.dtype != np.complex128
            or not np.all(np.isfinite(h)) or not np.all(np.isfinite(z))):
        raise ValueError("finite complex128 H[2,2,64,16] and Z[2,64,16] required")
    if snr_db not in SNR_DB:
        raise ValueError("B0 training/validation SNR must be one of 0/5/10/15/20")
    if layout.offsets_by_tx not in ((0, 1), (0, 2), (0, 4), (0, 8)):
        raise ValueError("existing fixed-density FDM layout required")
    known = map_pilot_symbols(layout, pilot_symbols)
    noise = np.sqrt(noise_variance_from_snr(snr_db))*z
    ls = np.full(h.shape, np.nan+1j*np.nan, dtype=np.complex128)
    for tx in range(2):
        at_pilot = layout.pilot_mask[tx]
        ls[:, tx, at_pilot] = (h[:, tx, at_pilot]
            + noise[:, at_pilot]/known[tx, at_pilot][None, :])
    return ls


def r3_from_pilot_ls(pilot_ls, layout):
    """Receiver interpolation only: no H_true, data bits or SNR argument."""
    return ls_linear_interpolate(pilot_ls, layout.pilot_mask)


def full_link_pilot_sentinel(h_true, layout, unit_noise, snr_db):
    from src.estimators import pilot_ls_estimate
    x = map_pilot_symbols(layout, PILOT_SYMBOLS)
    y = apply_frequency_mimo_channel(h_true, x) + np.sqrt(noise_variance_from_snr(snr_db))*unit_noise
    actual, _ = pilot_ls_estimate(y, x, layout.pilot_mask)
    direct = direct_pilot_ls(h_true, layout, unit_noise, snr_db)
    selected = np.broadcast_to(layout.pilot_mask[None], actual.shape)
    error = float(np.max(np.abs(actual[selected]-direct[selected])))
    scale = float(np.max(np.abs(actual[selected])))
    tolerance = 32*np.finfo(np.float64).eps*max(1.0, scale)
    if error > tolerance:
        raise ValueError("direct observations differ from actual Xpilot -> HX+N -> Pilot-LS")
    return {"max_absolute_error": error, "tolerance": tolerance,
            "machine_precision_match": True, "bits_generated": False,
            "pilot_symbol_power": float(np.abs(PILOT_SYMBOLS[0])**2)}


def complex_to_tensor(h_complex, device=None):
    import torch
    h = np.asarray(h_complex)
    if h.shape == (2, 2, 64, 16):
        h = h[None]
    if (h.ndim != 5 or h.shape[1:] != (2, 2, 64, 16) or not len(h)
            or h.dtype != np.complex128 or not np.all(np.isfinite(h))):
        raise ValueError("finite complex128 [B,2,2,64,16] required")
    channels = np.stack((h.real, h.imag), axis=3).reshape(-1, 8, 64, 16)
    return torch.from_numpy(np.ascontiguousarray(channels, dtype=np.float32)).to(device=device)


def tensor_to_complex(tensor):
    import torch
    if (tensor.ndim != 4 or tuple(tensor.shape[1:]) != (8, 64, 16)
            or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())):
        raise ValueError("finite float32 [B,8,64,16] required")
    parts = tensor.detach().cpu().numpy().astype(np.float64).reshape(-1, 2, 2, 2, 64, 16)
    return np.asarray(parts[:, :, :, 0] + 1j*parts[:, :, :, 1], dtype=np.complex128)


def quantization_diagnostics(h_complex):
    h = np.asarray(h_complex)
    if h.ndim == 4:
        h = h[None]
    reconstructed = tensor_to_complex(complex_to_tensor(h))
    error = reconstructed-h
    return {"max_complex_absolute_error": float(np.max(np.abs(error))),
            "RMS_complex_error": float(np.sqrt(np.mean(np.abs(error)**2))),
            "simulation_dtype": "complex128", "model_dtype": "float32",
            "reconstruction_accumulator_dtype": str(reconstructed.dtype),
            "normalization_used": False}


def complex_data_error_energies(prediction, target, data_mask):
    """Future NMSE energies use complex128/float64 accumulation, no frame ratio."""
    from src.b0_model import spatial_data_mask
    mask = spatial_data_mask(data_mask).cpu().numpy()
    predicted = tensor_to_complex(prediction)
    truth = np.asarray(target)
    if truth.ndim == 4:
        truth = truth[None]
    if truth.shape != predicted.shape or truth.dtype != np.complex128:
        raise ValueError("original complex128 target must match reconstructed prediction")
    return {"error_energy": float(np.sum(np.abs(predicted[..., mask]-truth[..., mask])**2, dtype=np.float64)),
            "reference_energy": float(np.sum(np.abs(truth[..., mask])**2, dtype=np.float64)),
            "num_complex_observations": int(predicted.shape[0]*4*np.count_nonzero(mask)),
            "accumulator_dtype": "complex128 / float64"}
