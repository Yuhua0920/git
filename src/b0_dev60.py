"""T12D budget extension; frozen model, trainer, receiver and PERF math reused."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import b0_development as old, b0_perf as perf, ml_data as md
from src.b0_model import ResidualCNN
from src.lmmse_train import sha256_file
from src.pilots import build_fdm_pilot_layout

PREFIX = "t12d_b0_df16_dev60"
T12C_COMMIT = "cfabed19201b617dad20920f669dcbe325c9bd49"
EPOCH30_HASH = "e5a77cc6448a9afabf09688880ac659a6753f9bc33f12e2d626a715acdcec07f"
LAYOUT = build_fdm_pilot_layout(2, 64, 16, 16, 4)
PROFILE = dict(cpu_threads=4, workers=2, prefetch=2, persistent_workers=True,
               pin_memory=False, non_blocking=False, batch_size=128)


def tensor_stream_hash(tensor):
    """Same raw-C SHA256 as PERF, with bounded CPU scratch for GPU residency."""
    digest = hashlib.sha256()
    if tensor.ndim == 5:
        for group in tensor:
            for start in range(0, len(group), 128):
                digest.update(group[start:start+128].detach().cpu().numpy().tobytes())
    elif tensor.ndim == 4:
        for start in range(0, len(tensor), 128):
            digest.update(tensor[start:start+128].detach().cpu().numpy().tobytes())
    else:
        raise ValueError("original base tensor dimensions required")
    return digest.hexdigest()


def prepare_H_cache(path, prepared, rows):
    """Write exact continuous H once; do not retain a second full RAM H copy."""
    from src.lmmse_train import formal_trajectory, raw_array_sha256
    path = Path(path)
    ids = [r["trajectory_id"] for r in rows]
    if ids != list(range(len(rows))) or not rows or len(rows) > 2000 or any(r["split"] != "train" for r in rows):
        raise ValueError("original consecutive TRAIN rows required")
    mapped = np.lib.format.open_memmap(path, mode="w+", dtype=np.complex128, shape=(len(rows)*10, 2, 2, 64, 16))
    hashes = []
    for index, row in enumerate(rows):
        draw = formal_trajectory(prepared, row)
        mapped[index*10:index*10+10] = draw.h_frames
        hashes.append({"trajectory_id": row["trajectory_id"], "H_long_sha256": raw_array_sha256(draw.h_long)})
    mapped.flush(); del mapped
    return hashes


def epoch_order(epoch):
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= 60:
        raise ValueError("T12D training epoch must be 1..60")
    return np.random.default_rng(np.random.SeedSequence([old.SHUFFLE_DOMAIN, old.SEED, epoch])).permutation(20000)


def selected_profile(report):
    if not report["all_checks_passed"]:
        raise ValueError("accepted T12C-PERF required")
    profile = report["selected_profile"]
    if any(profile[k] != value for k, value in PROFILE.items()) or profile["validation_strategy"] != "GPU-resident":
        raise ValueError("exact fixed PERF-Popt required; no rebenchmark/profile change")
    return {**PROFILE, "name": "PERF-Popt", "cache_strategy": profile["cache_strategy"],
            "validation_strategy": "GPU-resident"}


def training_batch(h_cache, epoch, ids):
    epoch_order(epoch)  # Same strict run budget for all entry points.
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or not len(ids) or np.any(ids < 0) or np.any(ids >= 20000):
        raise ValueError("canonical original training IDs required")
    snrs = md.balanced_training_snr(old.SEED, epoch)[ids]
    timings = {}; stamp = perf_counter()
    h = np.asarray(h_cache[ids], dtype=np.complex128)
    timings["H_true_cache_access"] = perf_counter()-stamp
    stamp = perf_counter()
    z = np.stack([md.training_unit_noise(old.SEED, epoch, int(i)//10, int(i)%10) for i in ids])
    timings["unit_noise_generation"] = perf_counter()-stamp
    stamp = perf_counter(); ls = perf.batch_pilot_ls(h, z, snrs, LAYOUT)
    timings["Pilot_LS"] = perf_counter()-stamp
    hc = perf.vectorized_r3(ls, LAYOUT, timings)
    stamp = perf_counter(); base, target = md.complex_to_tensor(hc), md.complex_to_tensor(h)
    timings["complex_to_float32"] = perf_counter()-stamp
    return {"base": base, "target": target, "IDs": torch.from_numpy(ids.copy()),
            "SNR": torch.from_numpy(snrs.copy()), "timings": timings,
            "hashes": {"H_true": perf.raw_hash(h), "unit_noise": perf.raw_hash(z), "H_base": perf.raw_hash(base),
                       "target": perf.raw_hash(target), "sample_IDs": perf.raw_hash(ids), "SNR": perf.raw_hash(snrs)}}


class Dev60Sampler(perf.EpochBatchSampler):
    def ordered_ids(self):
        order = epoch_order(self.epoch)
        if self.fixed_subset is not None:
            order = order[np.isin(order, self.fixed_subset)]
        return order[:self.count]


class Dev60Dataset(perf.BatchDataset):
    def __getitem__(self, key):
        epoch, ids = key
        if self.h is None:
            self.h = np.load(self.cache_path, mmap_mode="r", allow_pickle=False)
        return training_batch(self.h, epoch, ids)


def training_loader(path, sampler):
    return DataLoader(Dev60Dataset(path), sampler=sampler, batch_size=None, collate_fn=perf.identity_collate,
        num_workers=2, pin_memory=False, in_order=True, multiprocessing_context="spawn",
        worker_init_fn=perf.worker_init, prefetch_factor=2, persistent_workers=True)


class IndexedInput:
    def __init__(self, bridge, name):
        self.bridge, self.name = bridge, name
    def __getitem__(self, ids):
        if self.name == "base":
            self.bridge.advance(ids)
        elif not np.array_equal(ids, self.bridge.batch["IDs"].numpy()):
            raise ValueError("P0 target/SNR index mismatch")
        value = self.bridge.batch[self.name]
        return value.numpy() if self.name == "SNR" else value


class TrainingBridge:
    """Adapt original indexed trainer to FIFO batches; do not replace its math."""
    def __init__(self, batches, resource_sampler=None):
        self.iterator = iter(batches)
        if resource_sampler:
            resource_sampler.children = [w.pid for w in getattr(self.iterator, "_workers", [])]
        self.resources = resource_sampler; self.batch = None; self.wait_seconds = 0.0
        self.ids, self.snrs, self.hashes = [], [], []
        self.cpu_stage_seconds = {}
    def advance(self, ids):
        stamp = perf_counter(); self.batch = next(self.iterator)
        self.wait_seconds += perf_counter()-stamp
        if not np.array_equal(ids, self.batch["IDs"].numpy()):
            raise ValueError("optimizer batch order differs from P0")
        if self.resources and self.resources.rows[-1]["available_RAM_bytes"] < 1.5*2**30:
            raise MemoryError("physical RAM below fixed PERF-Popt safety gate")
        self.ids.extend(self.batch["IDs"].tolist()); self.snrs.extend(self.batch["SNR"].tolist())
        self.hashes.append(self.batch["hashes"])
        for name, seconds in self.batch["timings"].items():
            self.cpu_stage_seconds[name] = self.cpu_stage_seconds.get(name, 0.0)+seconds
    def inputs(self):
        return tuple(IndexedInput(self, k) for k in ("base", "target", "SNR"))
    def finish(self, order, epoch):
        try:
            next(self.iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("extra optimizer batch")
        ids = np.asarray(self.ids, dtype=np.int64); snrs = np.asarray(self.snrs, dtype=np.int64)
        expected_snrs = md.balanced_training_snr(old.SEED, epoch)[order]
        if not np.array_equal(ids, order) or not np.array_equal(snrs, expected_snrs):
            raise ValueError("P0 ordered IDs/SNR mismatch")
        return {"ordered_ID_hash": perf.raw_hash(ids), "ordered_SNR_hash": perf.raw_hash(snrs),
            "batch_hashes_SHA256": hashlib.sha256(json.dumps(self.hashes, sort_keys=True).encode()).hexdigest(),
            "DataLoader_waiting_seconds": self.wait_seconds, "CPU_stage_seconds_sum": self.cpu_stage_seconds,
            "samples": len(ids), "order_exact": True}


def train_epoch(model, optimizer, batches, order, epoch, resources=None):
    stamp = perf_counter(); bridge = TrainingBridge(batches, resources)
    stats = old.train_epoch(model, optimizer, *bridge.inputs(), LAYOUT.data_mask, order)
    source = bridge.finish(order, epoch)
    torch.cuda.synchronize() if next(model.parameters()).is_cuda else None
    return stats, source, perf_counter()-stamp


def optimizer_steps(optimizer):
    steps = [int(state["step"].item()) for state in optimizer.state.values()]
    if not steps:
        return 0
    if len(set(steps)) != 1 or len(steps) != 8:
        raise ValueError("all original Adam parameter step counters must agree")
    return steps[0]


def structured_hash(value):
    digest = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().numpy()
            digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
        elif isinstance(item, dict):
            for k in sorted(item, key=str):
                digest.update(str(k).encode()); add(item[k])
        elif isinstance(item, (list, tuple)):
            digest.update(str(len(item)).encode())
            for v in item:
                add(v)
        else:
            digest.update(json.dumps(item, sort_keys=True).encode())
    add(value); return digest.hexdigest()


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return value


def checkpoint_load(path, device, protocol_hash):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload["training_seed"] != old.SEED or payload["density"] != 16 or payload["protocol_hash"] != protocol_hash
        or payload["counts_toward_formal_five_seed_results"] or not 0 <= payload["epoch"] <= 60
        or payload["optimizer_config"] != {"name": "Adam", "lr": .001, "weight_decay": 0}):
        raise ValueError("wrong T12D checkpoint identity/protocol")
    with torch.random.fork_rng(devices=[]):
        model = ResidualCNN()
    model.load_state_dict(payload["model_state_dict"], strict=True); model.to(device).eval()
    optimizer = old.optimizer_for(model); optimizer.load_state_dict(payload["optimizer_state_dict"])
    if optimizer_steps(optimizer) != payload["optimizer_total_steps"]:
        raise ValueError("Adam step provenance mismatch")
    return model, optimizer, payload


def checkpoint_save(path, model, optimizer, epoch, val_mse, protocol_hash, architecture):
    path = Path(path)
    state = np.random.get_state()
    payload = {"model_state_dict": cpu_copy(model.state_dict()), "optimizer_state_dict": cpu_copy(optimizer.state_dict()),
        "epoch": epoch, "training_seed": old.SEED, "density": 16, "batch_size": 128,
        "optimizer_config": {"name": "Adam", "lr": .001, "weight_decay": 0}, "optimizer_total_steps": optimizer_steps(optimizer),
        "protocol_hash": protocol_hash, "architecture": architecture, "development_model": True,
        "counts_toward_formal_five_seed_results": False, "validation_mse_pooled": float(val_mse),
        "selection_metric": "pooled validation data-RE complex MSE",
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng_state": {"generator": state[0], "keys": torch.from_numpy(state[1].astype(np.int64)),
                            "pos": state[2], "has_gauss": state[3], "cached_gaussian": state[4]}}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary); temporary.replace(path)
    loaded, restored, _ = checkpoint_load(path, next(model.parameters()).device, protocol_hash)
    model_hash, optimizer_hash = old.state_hash(model), structured_hash(optimizer.state_dict())
    if model_hash != old.state_hash(loaded) or optimizer_hash != structured_hash(restored.state_dict()):
        raise ValueError("full checkpoint roundtrip failed")
    return {"path": str(path), "file_sha256": sha256_file(path), "epoch": epoch,
        "model_tensor_sha256": model_hash, "optimizer_tensor_sha256": optimizer_hash,
        "optimizer_total_steps": payload["optimizer_total_steps"], "weights_only_reload": True,
        "model_and_optimizer_exact_reload": True, "protocol_hash": protocol_hash}


def replay_epoch(epoch, training, validation, source, original):
    if not 0 <= epoch <= 30:
        raise ValueError("replay gate only epochs0..30")
    ref = original["epoch_records"][epoch]
    val_exact = validation == ref["validation"]
    schedule = epoch == 0 or (source["ordered_ID_hash"] == ref["shuffle_order_sha256"] and
        perf.raw_hash(md.balanced_training_snr(old.SEED, epoch)) == ref["SNR_assignment_sha256"])
    train_exact = epoch == 0 or training["train_mse"] == ref["training"]["train_mse"]
    within = epoch == 0 or bool(np.isclose(training["train_mse"], ref["training"]["train_mse"], rtol=1e-12, atol=1e-14))
    if not val_exact or not schedule or not within:
        raise ValueError(f"T12B replay gate mismatch at epoch{epoch}; STOP before epoch31")
    return {"epoch": epoch, "validation_all_fields_exact": val_exact, "schedule_hashes_exact": schedule,
            "train_MSE_exact": train_exact, "train_MSE_tolerance_passed": within}


def require_extension_gate(rows, model):
    if ([r["epoch"] for r in rows] != list(range(31)) or not all(r["validation_all_fields_exact"] and
            r["schedule_hashes_exact"] and r["train_MSE_tolerance_passed"] for r in rows)
            or old.state_hash(model) != EPOCH30_HASH):
        raise ValueError("epoch0..30 replay/state gate failed; epoch31 prohibited")
    return {"all_checks_passed": True, "epochs_verified": list(range(31)), "epoch30_model_tensor_sha256": EPOCH30_HASH,
            "epoch30_state_hash_exact": True, "all_train_MSE_exact": all(r["train_MSE_exact"] for r in rows),
            "per_epoch": rows, "checked_before_epoch31": True}


def convergence_diagnosis(log, epochs):
    values = np.array([r["val_mse_pooled"] for r in log]); best = int(np.argmin(values))
    relative = float((values[30]-values[best])/values[30])
    refs, final = epochs[30]["validation"]["per_snr"], epochs[best]["validation"]["per_snr"]
    gains = [{"snr_db": a["snr_db"], "epoch30_nmse_db": a["nmse_db"], "best60_nmse_db": b["nmse_db"],
              "improvement_db": a["nmse_db"]-b["nmse_db"]} for a, b in zip(refs, final)]
    high = gains[-1]["improvement_db"]
    last = values[51:61]; endpoint = float((last[0]-last[-1])/last[0]); width = float(np.ptp(last)/np.mean(last))
    frequency = sum(r["best_so_far"] for r in log[51:61]); worsening = float(values[-1]/values[best]-1)
    if worsening >= .02 and best < 51:
        category = "overfitting indication"
    elif endpoint >= .01 and frequency >= 3:
        category = "still improving"
    elif width <= .01 and abs(endpoint) < .005:
        category = "plateau likely"
    else:
        category = "oscillatory"
    return {"best_epoch": best, "best_validation_mse": float(values[best]),
        "epoch30_to_best_improvement": {"relative": relative, "absolute": float(values[30]-values[best]),
                                        "epoch30_mse": float(values[30]), "best60_mse": float(values[best])},
        "per_snr_epoch30_to_best": gains, "diagnostic_scope": "original batch128 fixed checkpoint validation path",
        "under_training_evidence": best > 35 and relative >= .02,
        "high_snr_undertraining_contribution": "meaningful" if high >= .5 else "not_meaningful",
        "high_snr_limitation_persists": relative > 0 and high < .2,
        "plateau_diagnosis": {"category": category, "window": [51, 60], "relative_improvement": endpoint,
            "global_best_frequency": frequency, "validation_range": float(np.ptp(last)), "relative_range": width,
            "final_relative_worsening_from_best": worsening, "automatic_continuation": False}}


def paired_diagnostics(best_rows, epoch30_rows):
    old_index = {(int(r["trajectory_id"]), int(r["snr_db"])): r for r in epoch30_rows}
    output = []
    for snr in md.SNR_DB:
        selected = [r for r in best_rows if r["snr_db"] == snr]
        if len(selected) != 400:
            raise ValueError("exact400 paired independent trajectory units required")
        for reference in ("epoch30", "R3"):
            for metric, current, previous in (("error_energy", "b0_nmse_error_energy", "r3_nmse_error_energy"),
                                               ("bit_errors", "b0_bit_errors", "r3_bit_errors")):
                differences = np.array([float(r[current])-float(old_index[r["trajectory_id"], snr][current])
                    if reference == "epoch30" else float(r[current])-float(r[previous]) for r in selected])
                q = np.quantile(differences, [.25, .5, .75])
                output.append({"snr_db": snr, "comparison": f"best60-minus-{reference}", "metric": metric,
                    "trajectory_units": 400, "Q25": float(q[0]), "median": float(q[1]), "Q75": float(q[2]),
                    "fraction_improved": float(np.mean(differences < 0)), "p_value": None})
    return output
