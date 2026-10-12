"""T12E only: dedicated 20-dB exposure, frozen B0/Adam/receiver math."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import b0_development as old, b0_dev60 as dev, b0_perf as perf, ml_data as md
from src.b0_model import predict_channel
from src.lmmse_train import sha256_file
from src.numerics import noise_variance_from_snr

PREFIX = "t12e_b0_df16_20db"
T12D_COMMIT = "c60877a397ac3df03970c003f9d8d196a0115ac7"
LAYOUT = dev.LAYOUT
CROSS_EPOCHS = (0, 12, 30, 60)
SELECTION_METRIC = "fixed 20-dB validation pooled data-RE complex MSE"


def training_snrs(epoch):
    dev.epoch_order(epoch)
    return np.full(20000, 20, dtype=np.int64)


def training_batch(h_cache, epoch, ids):
    dev.epoch_order(epoch)
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or not len(ids) or np.any(ids < 0) or np.any(ids >= 20000):
        raise ValueError("canonical TRAIN IDs required")
    timings = {}; stamp = perf_counter()
    h = np.asarray(h_cache[ids], dtype=np.complex128)
    timings["H_true_cache_access"] = perf_counter()-stamp
    stamp = perf_counter()
    z = np.stack([md.training_unit_noise(old.SEED, epoch, int(i)//10, int(i)%10) for i in ids])
    timings["unit_noise_generation"] = perf_counter()-stamp
    snrs = training_snrs(epoch)[ids]
    stamp = perf_counter(); ls = perf.batch_pilot_ls(h, z, snrs, LAYOUT)
    timings["Pilot_LS"] = perf_counter()-stamp
    hc = perf.vectorized_r3(ls, LAYOUT, timings)
    stamp = perf_counter(); base, target = md.complex_to_tensor(hc), md.complex_to_tensor(h)
    timings["complex_to_float32"] = perf_counter()-stamp
    return {"base": base, "target": target, "IDs": torch.from_numpy(ids.copy()),
        "SNR": torch.from_numpy(snrs.copy()), "timings": timings,
        "hashes": {"H_true": perf.raw_hash(h), "unit_noise": perf.raw_hash(z), "H_base": perf.raw_hash(base),
            "target": perf.raw_hash(target), "sample_IDs": perf.raw_hash(ids), "SNR": perf.raw_hash(snrs)}}


class ProbeDataset(dev.Dev60Dataset):
    def __getitem__(self, key):
        epoch, ids = key
        if self.h is None:
            self.h = np.load(self.cache_path, mmap_mode="r", allow_pickle=False)
        return training_batch(self.h, epoch, ids)


def training_loader(path, sampler):
    return DataLoader(ProbeDataset(path), sampler=sampler, batch_size=None, collate_fn=perf.identity_collate,
        num_workers=2, pin_memory=False, in_order=True, multiprocessing_context="spawn",
        worker_init_fn=perf.worker_init, prefetch_factor=2, persistent_workers=True)


class ProbeBridge(dev.TrainingBridge):
    def finish(self, order, epoch):
        try:
            next(self.iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("extra training batch")
        ids = np.asarray(self.ids, dtype=np.int64); snrs = np.asarray(self.snrs, dtype=np.int64)
        if not np.array_equal(ids, order) or not np.array_equal(snrs, training_snrs(epoch)[order]):
            raise ValueError("same original order and all20dB required")
        return {"ordered_ID_hash": perf.raw_hash(ids), "ordered_SNR_hash": perf.raw_hash(snrs),
            "batch_hashes_SHA256": hashlib.sha256(json.dumps(self.hashes, sort_keys=True).encode()).hexdigest(),
            "DataLoader_waiting_seconds": self.wait_seconds, "CPU_stage_seconds_sum": self.cpu_stage_seconds,
            "samples": len(ids), "order_exact": True, "all_training_examples_20db": True}


def train_epoch(model, optimizer, batches, order, epoch, resources=None):
    stamp = perf_counter(); bridge = ProbeBridge(batches, resources)
    stats = old.train_epoch(model, optimizer, *bridge.inputs(), LAYOUT.data_mask, order)
    source = bridge.finish(order, epoch)
    torch.cuda.synchronize() if next(model.parameters()).is_cuda else None
    return stats, source, perf_counter()-stamp


def evaluate_20db(model, cache, epoch_zero=False):
    """Only SNR20 is read here; cross-SNR scores cannot select checkpoints."""
    model.eval(); device = next(model.parameters()).device
    error, truth, count, maximum = 0.0, 0.0, 0, 0.0
    identity = True
    with torch.inference_mode():
        for start in range(0, len(cache["h"]), 128):
            end = min(start+128, len(cache["h"]))
            base = cache["base"][4, start:end].to(device)
            pred = predict_channel(model, base, LAYOUT.data_mask)
            if epoch_zero:
                identity = identity and torch.equal(pred, base)
            union = torch.as_tensor(LAYOUT.pilot_union, device=device)
            maximum = max(maximum, float((pred[..., union]-base[..., union]).abs().max().cpu()))
            energies = md.complex_data_error_energies(pred, cache["h"][start:end], LAYOUT.data_mask)
            error += energies["error_energy"]; truth += energies["reference_energy"]
            count += energies["num_complex_observations"]
    if maximum != 0 or (epoch_zero and not identity):
        raise ValueError("pilot preservation or epoch0 identity failed")
    return {"snr_db": 20, "error_energy": error, "truth_energy": truth, "num_complex_coefficients": count,
        "mse": error/count, "nmse_linear": error/truth, "nmse_db": float(10*np.log10(error/truth))}


def best_epoch(log):
    # No cross-SNR/NMSE/BER/reference field participates.
    values = np.asarray([r["val_mse_20db"] for r in log], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise FloatingPointError("invalid20dB MSE")
    return int(np.argmin(values))


def specialization_flags(gain):
    if not np.isfinite(gain):
        raise ValueError("finite dedicated gain required")
    return {"strong_specialization_gain": gain >= 1.0,
        "moderate_specialization_gain": .2 <= gain < 1.0,
        "high_snr_limitation_persists_under_dedicated_training": gain < .2}


def checkpoint_save(path, model, optimizer, epoch, val_mse, protocol_hash, architecture):
    path = Path(path); rng = np.random.get_state()
    payload = {"model_state_dict": dev.cpu_copy(model.state_dict()), "optimizer_state_dict": dev.cpu_copy(optimizer.state_dict()),
        "epoch": epoch, "training_seed": old.SEED, "density": 16, "batch_size": 128,
        "optimizer_config": {"name": "Adam", "lr": .001, "weight_decay": 0}, "optimizer_total_steps": dev.optimizer_steps(optimizer),
        "protocol_hash": protocol_hash, "architecture": architecture, "diagnostic_model": True, "formal_seed": False,
        "counts_toward_formal_five_seed_results": False, "training_snr": [20],
        "validation_mse_20db": float(val_mse), "selection_metric": SELECTION_METRIC,
        "torch_rng_state": torch.random.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(),
        "numpy_rng_state": {"generator": rng[0], "keys": torch.from_numpy(rng[1].astype(np.int64)),
            "pos": rng[2], "has_gauss": rng[3], "cached_gaussian": rng[4]}}
    temporary = path.with_suffix(".tmp"); torch.save(payload, temporary); temporary.replace(path)
    loaded, restored, _ = checkpoint_load(path, next(model.parameters()).device, protocol_hash)
    model_hash, optimizer_hash = old.state_hash(model), dev.structured_hash(optimizer.state_dict())
    if old.state_hash(loaded) != model_hash or dev.structured_hash(restored.state_dict()) != optimizer_hash:
        raise ValueError("complete checkpoint reload changed model/Adam")
    return {"path": str(path), "file_sha256": sha256_file(path), "epoch": epoch,
        "model_tensor_sha256": model_hash, "optimizer_tensor_sha256": optimizer_hash,
        "optimizer_total_steps": dev.optimizer_steps(optimizer), "protocol_hash": protocol_hash,
        "weights_only_reload": True, "model_and_optimizer_exact_reload": True}


def checkpoint_load(path, device, protocol_hash):
    model, optimizer, payload = dev.checkpoint_load(path, device, protocol_hash)
    if (payload["training_snr"] != [20] or payload["selection_metric"] != SELECTION_METRIC
            or not payload["diagnostic_model"] or payload["formal_seed"]):
        raise ValueError("wrong20dB diagnostic identity")
    return model, optimizer, payload


def distance_by_tx(layout=LAYOUT):
    distances = []
    for tx in range(2):
        tones = np.flatnonzero(np.any(layout.pilot_mask[tx], axis=1))
        distances.append(np.min(np.abs(np.arange(64)[:, None]-tones), axis=1))
    return np.asarray(distances)


class RegionDiagnostics:
    """Original data masks and energy sums; separate audit groups retain all REs."""
    def __init__(self, layout=LAYOUT):
        self.layout = layout; self.sums = {}
        times = np.arange(16)[None, None, None, :]
        distance = distance_by_tx(layout)[None, :, :, None]
        data = np.broadcast_to(layout.data_mask[0], (2, 2, 64, 16))
        self.masks = {}
        for label, ts in (("interior_1_11", range(1, 12)), ("trailing_13_15", range(13, 16)), ("remaining_0_12", (0, 12))):
            self.masks["time", label] = data & np.isin(times, list(ts))
        for value in range(int(distance.max())+1):
            self.masks["frequency_distance", value] = data & (distance == value)
    def add(self, h, r3, probe):
        for method, estimate in (("R3", r3), ("B0_20db_only", probe)):
            e, truth = np.abs(estimate-h)**2, np.abs(h)**2
            for (group, label), mask in self.masks.items():
                values = self.sums.setdefault((method, group, label), [0., 0., 0])
                values[0] += float(np.sum(e[mask], dtype=np.float64))
                values[1] += float(np.sum(truth[mask], dtype=np.float64)); values[2] += int(mask.sum())
    def rows(self):
        rows = []
        for (method, group, label), (error, truth, count) in self.sums.items():
            rows.append({"method": method, "snr_db": 20, "group": group, "region_or_distance": label,
                "requested_group": label != "remaining_0_12" and label != 0,
                "error_energy": error, "truth_energy": truth, "complex_coefficient_count": count,
                "nmse_linear": error/truth, "nmse_db": float(10*np.log10(error/truth)) if error else None})
        return rows


class FullLinkDiagnostics:
    """Instrument frozen full-link results; never alter estimates or return values."""
    def __init__(self, root, mixed_model):
        self.root, self.mixed_model = Path(root), mixed_model
        self.estimate_count = self.energy_count = 0; self.pending = {}
        self.regions = RegionDiagnostics(); self.fixed_sample = None
        self.original_estimate = old.link.estimate_ls_linear_channel
        self.original_energy = old.channel_error_energies
    def estimate(self, y, layout):
        ls, r3 = self.original_estimate(y, layout)
        index = self.estimate_count; self.estimate_count += 1
        if index % 5 == 4:
            # At most ten20dB frames retained for the existing trajectory batch.
            self.pending[index] = (r3, ls, y)
        return ls, r3
    def energy(self, probe, h, mask):
        result = self.original_energy(probe, h, mask)
        index = self.energy_count; self.energy_count += 1
        if index % 5 == 4:
            r3, ls, y = self.pending.pop(index)
            self.regions.add(h, r3, probe)
            if index == 4:
                self.fixed_sample = fixed_sample(self.root, h, r3, probe, ls, y, self.mixed_model)
        return result
    def verify(self, evaluation):
        if self.pending or self.estimate_count != 20000 or self.energy_count != 20000 or not self.fixed_sample:
            raise ValueError("full-link instrumentation count/order changed")
        high = next(r for r in evaluation["aggregates"] if r["snr_db"] == 20)
        rows = self.regions.rows()
        for method, key in (("B0_20db_only", "b0_nmse_error_energy"), ("R3", "r3_nmse_error_energy")):
            reference = high[key] if method != "R3" else sum(r[key] for r in evaluation["trajectory_rows"] if r["snr_db"] == 20)
            for group in ("time", "frequency_distance"):
                selected = [r for r in rows if r["method"] == method and r["group"] == group]
                if (sum(r["complex_coefficient_count"] for r in selected) != 15872000
                    or not np.isclose(sum(r["error_energy"] for r in selected), reference, rtol=1e-12)
                    or not np.isclose(sum(r["truth_energy"] for r in selected), high["nmse_truth_energy"], rtol=1e-12)):
                    raise ValueError("region totals do not match full original data grid")
        return rows


def fixed_sample(root, h, r3, probe, ls, y, mixed_model):
    from src.lmmse_train import load_frozen_covariance, prepare_train_lmmse
    from src.lmmse import estimate_mimo_from_pilot_ls
    frozen = load_frozen_covariance(Path(root)/"outputs/t10c_lmmse_train_covariance.npz", md.MANIFEST_SHA256)
    estimators = tuple(prepare_train_lmmse(frozen, LAYOUT.pilot_mask[q], noise_variance_from_snr(20)) for q in range(2))
    r4 = estimate_mimo_from_pilot_ls(ls, estimators)
    with torch.inference_mode():
        mixed = md.tensor_to_complex(predict_channel(mixed_model, md.complex_to_tensor(r3, next(mixed_model.parameters()).device), LAYOUT.data_mask))[0]
    channels = {"R3": r3, "mixed_B0_best57": mixed, "B0_20db_only": probe, "R4": r4}
    points = [{"k": k, "t": t, "H_true": old.complex_record(h[:, :, k, t]),
        **{f"H_{name}": old.complex_record(value[:, :, k, t]) for name, value in channels.items()},
        **{f"absolute_error_{name}": np.abs(value[:, :, k, t]-h[:, :, k, t]).tolist() for name, value in channels.items()}}
        for k, t in ((1, 2), (1, 15))]
    return {"trajectory_id": 2000, "frame_index": 0, "snr_db": 20, "dF": 16, "points": points,
        "Y_sha256": perf.raw_hash(y), "H_R3_sha256": perf.raw_hash(r3), "H_R4_sha256": perf.raw_hash(r4),
        "mixed_B0_evaluated_frames": 1, "R4_evaluated_frames": 1, "mixed_or_R4_performance_scan": False}
