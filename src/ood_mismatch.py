"""T13A independent OOD namespaces, frozen estimators and actual paired link.

Existing communication/channel/training modules are reused without modification.
There is no formal-test generator or covariance fitting/training path here.
"""
import csv
import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src import link, lmmse, ml_data
from src.b0_development import state_hash
from src.b0_model import ResidualCNN, predict_channel
from src.lmmse_train import load_frozen_covariance, prepare_train_lmmse, raw_array_sha256, sha256_file
from src.lmmse_train_validation import batched_data_zf
from src.metrics import channel_error_energies, count_bit_errors
from src.numerics import complex_gaussian_noise, noise_variance_from_snr
from src.pilots import build_fdm_pilot_layout, count_layout_resources
from src.tdl import fractional_delay_frequency_response, rms_delay_spread_seconds, tdl_a_profile
from src.tdl_awgn_scan import scale_unit_noise
from src.trajectories import prepare_trajectory, slice_channel_frames

PREFIX = "t13a_ood_mismatch"
T12E_COMMIT = "e594d44fe9921f93140f7525842cc32a927b5623"
BASE_SEED = 20261301
NUM_TRAJECTORIES = 200
FRAMES = 5
SNRS = (5, 10, 20)
SHIFTS = {"speed120": {"code": 1, "speed_kmh": 120., "delay_spread_ns": 100.},
          "DS300": {"code": 2, "speed_kmh": 30., "delay_spread_ns": 300.}}
METHODS = ("R3", "R4_frozen", "R5_ood_oracle", "B0")
LAYOUT = build_fdm_pilot_layout(2, 64, 16, 16, 4)
LIMITS = {"power_abs": .10, "temporal_correlation_abs": .05,
          "frequency_covariance_abs": .10, "frequency_correlation_abs": .05,
          "rms_relative": .10, "factor_reconstruction_abs": 1e-12}


def seed_words(role, shift, tid, frame=0, tx=0, purpose=1):
    if (role not in (13001, 13002, 13003) or shift not in SHIFTS or purpose not in (0, 1)
            or any(isinstance(v, bool) or not isinstance(v, int) for v in (tid, frame, tx))
            or not 0 <= tid < NUM_TRAJECTORIES or not 0 <= frame < FRAMES or tx not in (0, 1)):
        raise ValueError("dedicated OOD coordinates/role/purpose required")
    return [role, BASE_SEED, purpose, SHIFTS[shift]["code"], tid, frame, tx]


def seed(role, shift, tid, frame=0, tx=0, purpose=1):
    return int(np.random.SeedSequence(seed_words(role, shift, tid, frame, tx, purpose)).generate_state(1, dtype=np.uint64)[0])


def manifest():
    return [{"split": "ood_dev", "shift": shift, "trajectory_id": tid,
             "channel_seed": seed(13001, shift, tid), "frames_per_trajectory": FRAMES,
             "symbols_per_trajectory": 16*FRAMES, **{k: spec[k] for k in ("speed_kmh", "delay_spread_ns")}}
            for shift, spec in SHIFTS.items() for tid in range(NUM_TRAJECTORIES)]


def prepare_channel(shift):
    if shift not in SHIFTS:
        raise ValueError("exactly two single-factor OOD conditions supported")
    spec = SHIFTS[shift]
    model = prepare_trajectory(frames_per_trajectory=FRAMES, speed_kmh=spec["speed_kmh"])
    delays, powers = tdl_a_profile(spec["delay_spread_ns"])
    phase = fractional_delay_frequency_response(np.eye(23, dtype=np.complex128), delays, model.frequencies_hz)
    for a in (delays, powers, phase):
        a.setflags(write=False)
    return replace(model, delays_s=delays, powers=powers, phase_matrix=phase)


def draw_channel(model, shift, tid, purpose=1):
    taps = model.generate_taps(seed(13001, shift, tid, purpose=purpose))
    h = model.frequency_response(taps)
    taps.setflags(write=False); h.setflags(write=False)
    frames = slice_channel_frames(h)
    if frames.shape != (5, 2, 2, 64, 16) or not np.shares_memory(frames, h):
        raise ValueError("five continuous shared-memory frames required")
    return taps, h, frames


def complex_scalar(value):
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


def calibrate_channel(model, shift):
    """Independent purpose0 pool; empirical checks never construct R5."""
    power_sum, count = 0., 0
    tap_power = np.zeros(23)
    temporal = {lag: np.zeros(3, dtype=np.complex128) for lag in (1, 4)}
    frequency = {lag: np.zeros(4, dtype=np.complex128) for lag in (1, 4, 8, 16, 32)}
    for tid in range(NUM_TRAJECTORIES):
        taps, h, _ = draw_channel(model, shift, tid, purpose=0)
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(taps)):
            raise ValueError("nonfinite calibration channel")
        power_sum += float(np.sum(np.abs(h)**2)); count += h.size
        tap_power += np.mean(np.abs(taps)**2, axis=(0, 1, 3))
        unit = taps / np.sqrt(model.powers)[None, None, :, None]
        for lag, values in temporal.items():
            a, b = unit[..., lag:], unit[..., :-lag]
            values += [np.sum(a*b.conj()), np.sum(np.abs(a)**2), np.sum(np.abs(b)**2)]
        for lag, values in frequency.items():
            a, b = h[:, :, lag:], h[:, :, :-lag]
            values += [np.sum(a*b.conj()), np.sum(np.abs(a)**2), np.sum(np.abs(b)**2), a.size]
    rf = lmmse.build_frequency_covariance(desired_rms_delay_spread_ns=SHIFTS[shift]["delay_spread_ns"])
    time_records, frequency_records = [], []
    for lag, (cross, p1, p0) in temporal.items():
        empirical = cross/np.sqrt(p1.real*p0.real)
        theory = model.covariance[lag, 0]
        error = float(abs(empirical-theory))
        time_records.append({"lag": lag, "empirical": complex_scalar(empirical), "theory": float(theory),
            "absolute_error": error, "passed": error <= LIMITS["temporal_correlation_abs"]})
    for lag, (cross, p1, p0, n) in frequency.items():
        raw = cross/n.real; correlation = cross/np.sqrt(p1.real*p0.real); theory = rf[lag, 0]
        raw_error, corr_error = float(abs(raw-theory)), float(abs(correlation-theory))
        frequency_records.append({"distance": lag, "empirical_covariance": complex_scalar(raw),
            "empirical_correlation": complex_scalar(correlation), "theory": complex_scalar(theory),
            "covariance_error": raw_error, "correlation_error": corr_error,
            "passed": raw_error <= LIMITS["frequency_covariance_abs"] and corr_error <= LIMITS["frequency_correlation_abs"]})
    true_rms = rms_delay_spread_seconds(model.delays_s, model.powers)
    empirical_pdp = tap_power/tap_power.sum()
    measured_rms = rms_delay_spread_seconds(model.delays_s, empirical_pdp)
    delays, powers = tdl_a_profile(SHIFTS[shift]["delay_spread_ns"])
    checks = {"channel_power": abs(power_sum/count-1) <= LIMITS["power_abs"],
        "lag1_lag4_Jakes": all(r["passed"] for r in time_records),
        "frequency_covariance_correlation": all(r["passed"] for r in frequency_records),
        "empirical_RMS_delay": abs(measured_rms/true_rms-1) <= LIMITS["rms_relative"],
        "exact_existing_profile_scaling": bool(np.array_equal(delays, model.delays_s) and np.array_equal(powers, model.powers)),
        "factor_reconstruction": model.factorization_summary["factor_reconstruction_max_error"] <= LIMITS["factor_reconstruction_abs"]}
    return {"shift": shift, "purpose": "independent channel calibration only", "seed_purpose": 0,
        "num_independent_trajectories": NUM_TRAJECTORIES, "continuous_frames": FRAMES,
        "channel_power": power_sum/count, "power_target": 1., "limits": LIMITS,
        "nominal_delay_spread_ns": SHIFTS[shift]["delay_spread_ns"],
        "deterministic_RMS_delay_ns": true_rms*1e9, "empirical_RMS_delay_ns": measured_rms*1e9,
        "temporal_correlations": time_records, "frequency_correlations": frequency_records,
        "mobility": model.mobility, "factorization": model.factorization_summary,
        "checks": checks, "all_checks_passed": all(checks.values())}


def require_calibration(records):
    if set(records) != set(SHIFTS) or any(not r["all_checks_passed"] for r in records.values()):
        raise ValueError("OOD calibration failed; performance scan prohibited")


def load_frozen_estimators(outputs, device):
    outputs = Path(outputs)
    t10c = json.loads((outputs/"t10c_lmmse_train_results.json").read_text(encoding="utf8"))
    frozen = load_frozen_covariance(outputs/"t10c_lmmse_train_covariance.npz", t10c["manifest"]["sha256"])
    if (frozen.file_sha256 != t10c["covariance_sha256"] or frozen.raw_array_sha256 != t10c["R_train_raw_array_sha256"]
            or sha256_file(outputs/"t10c_train_validation_manifest.csv") != frozen.manifest_sha256):
        raise ValueError("R4 frozen file/raw/manifest provenance changed")
    t12d = json.loads((outputs/"t12d_b0_df16_dev60_results.json").read_text(encoding="utf8"))
    reference = t12d["checkpoints"]["best60"]
    path = outputs/"t12d_b0_df16_dev60_best60.pt"
    if not t12d["all_checks_passed"] or reference["epoch"] != 57 or sha256_file(path) != reference["file_sha256"]:
        raise ValueError("exact mixed T12D best57 checkpoint required")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload["epoch"] != 57 or payload["training_seed"] != 13261001 or payload["density"] != 16
            or payload["protocol_hash"] != reference["protocol_hash"]):
        raise ValueError("B0 checkpoint identity changed")
    with torch.random.fork_rng(devices=[]):
        model = ResidualCNN()
    model.load_state_dict(payload["model_state_dict"], strict=True); model.to(device).eval()
    model.requires_grad_(False)
    if state_hash(model) != reference["model_tensor_sha256"]:
        raise ValueError("frozen B0 model tensor hash changed")
    return frozen, model, {"file_sha256": frozen.file_sha256, "raw_array_sha256": frozen.raw_array_sha256,
        "manifest_sha256": frozen.manifest_sha256, "reestimated": False}, {
        "checkpoint": path.name, "file_sha256": reference["file_sha256"], "model_state_sha256": state_hash(model),
        "epoch": 57, "mixed_SNR_model": True, "T12E_model_used": False, "adapted": False}


def prepare_estimators(frozen, shift):
    spec = SHIFTS[shift]
    rf = lmmse.build_frequency_covariance(desired_rms_delay_spread_ns=spec["delay_spread_ns"])
    rt = lmmse.build_time_covariance(speed_kmh=spec["speed_kmh"])
    theory = lmmse.build_full_grid_covariance(rf, rt)
    pools = {snr: {"R4_frozen": tuple(prepare_train_lmmse(frozen, LAYOUT.pilot_mask[q], noise_variance_from_snr(snr)) for q in range(2)),
        "R5_ood_oracle": tuple(lmmse.prepare_oracle_lmmse(theory, lmmse.pilot_flat_indices(LAYOUT.pilot_mask[q]),
            noise_variance_from_snr(snr), (64, 16)) for q in range(2))} for snr in SNRS}
    record = {"shift": shift, "source": "theoretical TDL-A PDP and Jakes covariance only",
        "speed_kmh": spec["speed_kmh"], "delay_spread_ns": spec["delay_spread_ns"],
        "frequency_raw_sha256": raw_array_sha256(rf), "time_raw_sha256": raw_array_sha256(rt),
        "full_grid_raw_sha256": raw_array_sha256(theory), "empirical_data_used": False,
        "preparation": [{"snr_db": snr, "method": m, "tx": q, **obj.diagnostics}
            for snr, pairs in pools.items() for m, pair in pairs.items() for q, obj in enumerate(pair)]}
    return pools, record


def infer_b0(model, r3_batch, timing):
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    base = ml_data.complex_to_tensor(np.stack(r3_batch), device=device)
    with torch.inference_mode():
        prediction = predict_channel(model, base, LAYOUT.data_mask)
        union = torch.tensor(LAYOUT.pilot_union, device=device, dtype=torch.bool)
        if not torch.equal(prediction[..., union], base[..., union]):
            raise ValueError("B0 modified pilot-union entries")
        h = ml_data.tensor_to_complex(prediction)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timing["B0_inference_seconds"] += perf_counter()-started
    return h


def evaluate_trajectory(model_channel, shift, tid, frozen_model, pools, historical, timing):
    started = perf_counter(); _, h_long, frames = draw_channel(model_channel, shift, tid)
    timing["channel_generation_seconds"] += perf_counter()-started
    details, bases, frame_hashes = [], [], []
    for frame, h in enumerate(frames):
        bits, symbols = link.generate_master_qpsk(bits_seeds=tuple(seed(13002, shift, tid, frame, q) for q in range(2)))
        z = complex_gaussian_noise((2, 64, 16), 1, np.random.default_rng(seed(13003, shift, tid, frame)), dtype=np.complex128)
        digests = {"frame_index": frame, "H_sha256": raw_array_sha256(h), "bits_sha256": raw_array_sha256(bits),
            "unit_noise_sha256": raw_array_sha256(z), "Y_by_snr": {}}
        for snr in SNRS:
            x = link.assemble_pilot_data_frame(symbols, LAYOUT)
            noise = scale_unit_noise(z, noise_variance_from_snr(snr))
            y = link.add_received_noise(link.apply_frequency_mimo_channel(h, x), noise)
            accepted_x = historical["assemble_pilot_data_frame"](symbols, LAYOUT)
            accepted_y = historical["add_received_noise"](historical["apply_frequency_mimo_channel"](h, accepted_x), noise)
            if not np.array_equal(x, accepted_x) or not np.array_equal(y, accepted_y):
                raise ValueError("current actual X/Y differ from frozen T10D physical link")
            y.setflags(write=False)
            started = perf_counter(); pilot_ls, r3 = link.estimate_ls_linear_channel(y, LAYOUT)
            timing["R3_seconds"] += perf_counter()-started
            channels = {"R3": r3}
            for method, key in (("R4_frozen", "R4_seconds"), ("R5_ood_oracle", "R5_seconds")):
                started = perf_counter(); channels[method] = lmmse.estimate_mimo_from_pilot_ls(pilot_ls, pools[snr][method])
                timing[key] += perf_counter()-started
            digests["Y_by_snr"][str(snr)] = raw_array_sha256(y)
            bases.append(r3); details.append((frame, snr, h, bits, z, y, channels, raw_array_sha256(pilot_ls)))
        frame_hashes.append(digests)
    predictions = infer_b0(frozen_model, bases, timing)
    sums = {snr: {"nmse_truth_energy": 0., "total_bits": 0, **{f"{m}_error_energy": 0. for m in METHODS},
                 **{f"{m}_bit_errors": 0 for m in (*METHODS, "Perfect")}} for snr in SNRS}
    for index, (frame, snr, h, bits, z, y, channels, pilot_digest) in enumerate(details):
        channels["B0"] = predictions[index]
        reference = link.extract_reference_data_bits(bits, LAYOUT.data_mask)
        totals = sums[snr]; truth = None; y_ids = []
        for method, channel in {**channels, "Perfect": h}.items():
            if method != "Perfect":
                error, channel_truth = channel_error_energies(channel, h, LAYOUT.data_mask)
                if truth is not None and channel_truth != truth:
                    raise ValueError("different NMSE denominators across methods")
                truth = channel_truth; totals[f"{method}_error_energy"] += error
            started = perf_counter(); y_ids.append(id(y)); detected = batched_data_zf(y, channel, LAYOUT.data_mask)
            timing["ZF_seconds"] += perf_counter()-started
            errors, num_bits = count_bit_errors(link.demodulate_data_bits(np.sqrt(2)*detected, LAYOUT.data_mask), reference)
            if num_bits != 2*np.count_nonzero(LAYOUT.data_mask):
                raise ValueError("BER denominator differs from actual data mask")
            totals[f"{method}_bit_errors"] += errors
        totals["nmse_truth_energy"] += truth; totals["total_bits"] += num_bits
        digests = frame_hashes[frame]
        if (len(set(y_ids)) != 1 or raw_array_sha256(y) != digests["Y_by_snr"][str(snr)]
                or raw_array_sha256(h) != digests["H_sha256"] or raw_array_sha256(bits) != digests["bits_sha256"]
                or raw_array_sha256(z) != digests["unit_noise_sha256"]):
            raise ValueError("common random number inputs mutated")
        digests.setdefault("shared_pilot_LS_by_snr", {})[str(snr)] = pilot_digest
    rows = [{"shift": shift, "trajectory_id": tid, "split": "ood_dev", "snr_db": snr, "dF": 16,
        "pilot_ratio": count_layout_resources(LAYOUT)["union_ratio"], "channel_seed": seed(13001, shift, tid),
        "frames_aggregated": FRAMES, "same_H_bits_noise_Y": True, **sums[snr]} for snr in SNRS]
    return rows, {"shift": shift, "trajectory_id": tid, "H_long_sha256": raw_array_sha256(h_long),
        "frame_realizations": frame_hashes, "exact_T10D_link_checks": FRAMES*len(SNRS)}


def read_csv(path):
    with Path(path).open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def id_references(outputs):
    id_lmmse = read_csv(Path(outputs)/"t10d_lmmse_train_validation_scan.csv")
    mixed = read_csv(Path(outputs)/"t12d_b0_df16_dev60_validation.csv")
    result = {}
    for snr in SNRS:
        r = next(r for r in id_lmmse if int(r["dF"]) == 16 and int(r["snr_db"]) == snr)
        b = next(r for r in mixed if int(r["snr_db"]) == snr)
        result[snr] = {"R3": float(r["ls_nmse_db"]), "R4_frozen": float(r["train_nmse_db"]), "B0": float(b["b0_best60_nmse_db"])}
    return result


def aggregate(rows, references, expected_units=NUM_TRAJECTORIES):
    result = []
    for shift in SHIFTS:
        for snr in SNRS:
            group = [r for r in rows if r["shift"] == shift and r["snr_db"] == snr]
            if (len(group) != expected_units or {r["trajectory_id"] for r in group} != set(range(expected_units))
                    or any(r["frames_aggregated"] != FRAMES or not r["same_H_bits_noise_Y"] for r in group)):
                raise ValueError("complete five-frame paired trajectory pool required")
            truth = sum(r["nmse_truth_energy"] for r in group); bits = sum(r["total_bits"] for r in group)
            a = {"shift": shift, "snr_db": snr, "dF": 16, "pilot_ratio": 1/32,
                "num_trajectories": len(group), "frames_aggregated": len(group)*FRAMES,
                "nmse_truth_energy": truth, "total_bits": bits}
            for m in METHODS:
                error = sum(r[f"{m}_error_energy"] for r in group)
                a[f"{m}_error_energy"] = error; a[f"{m}_nmse_db"] = float(10*np.log10(error/truth))
            for m in (*METHODS, "Perfect"):
                errors = sum(r[f"{m}_bit_errors"] for r in group)
                a[f"{m}_bit_errors"] = errors; a[f"{m}_ber"] = errors/bits
            a["R4_mismatch_penalty_db"] = a["R4_frozen_nmse_db"]-a["R5_ood_oracle_nmse_db"]
            a["B0_gap_to_oracle_db"] = a["B0_nmse_db"]-a["R5_ood_oracle_nmse_db"]
            a["B0_minus_R4_db"] = a["B0_nmse_db"]-a["R4_frozen_nmse_db"]
            for m in ("R3", "R4_frozen", "B0"):
                a[f"{m}_ID_nmse_db"] = references[snr][m]
                a[f"{m}_OOD_minus_ID_db"] = a[f"{m}_nmse_db"]-references[snr][m]
            result.append(a)
    return result


def direction_flags(rows):
    penalties = [r["R4_mismatch_penalty_db"] for r in rows]
    return {"frozen_statistics_mismatch_weak": all(p < .5 for p in penalties),
        "statistics_mismatch_material": any(p >= 1. for p in penalties),
        "learned_baseline_outperforms_frozen_lmmse": any(r["B0_minus_R4_db"] < 0 for r in rows)}


def paired_diagnostics(rows):
    result = []
    for shift in SHIFTS:
        for snr in SNRS:
            group = [r for r in rows if r["shift"] == shift and r["snr_db"] == snr]
            for a, b in combinations(METHODS, 2):
                ea = np.array([r[f"{a}_error_energy"] for r in group]); eb = np.array([r[f"{b}_error_energy"] for r in group])
                db = 10*np.log10(ea/eb)
                ber = np.array([(r[f"{a}_bit_errors"]-r[f"{b}_bit_errors"])/r["total_bits"] for r in group])
                def summary(v):
                    q = np.quantile(v, [.25, .5, .75])
                    return {"Q25": float(q[0]), "median": float(q[1]), "Q75": float(q[2]),
                        "fraction_lower_error": float(np.mean(v < 0)), "fraction_tied": float(np.mean(v == 0))}
                result.append({"shift": shift, "snr_db": snr, "comparison": f"{a} minus {b}",
                    "num_independent_trajectory_units": len(group), "five_frames_accumulated_first": True,
                    "NMSE_difference_db": summary(db), "BER_difference": summary(ber), "inferential_test": None})
    return result
