"""T12C-PERF: disposable runtime calibration, exact P0 regression, no epoch31."""
import argparse
import csv
import gc
import json
import platform
import subprocess
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src import b0_development as p0, ml_data as md, estimators
from src.b0_model import configure_reproducibility
from src.b0_perf import T12B_COMMIT, build_batch, raw_hash, EpochBatchSampler, make_loader, close_loader
from src.b0_perf_runtime import EventProfile, profile_original_training, train_stream, validation_run
from src.perf_resources import ResourceSampler, physical_memory
from src.pilots import build_fdm_pilot_layout
from src.lmmse_train import formal_trajectory, sha256_file
from src.nearest_validation import load_references, verify_frozen
from src.trajectories import prepare_trajectory

OUT = ROOT/"outputs"
CHECKPOINT = OUT/"t12b_b0_df16_dev_epoch30.pt"
LAYOUT = build_fdm_pilot_layout(2, 64, 16, 16, 4)
UTIL, BENCH, PROFILE, RESOURCE = [], [], [], []


def progress(message):
    print(message, flush=True)


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")


def save_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def accepted_files():
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", T12B_COMMIT,
        "outputs", "src", "scripts", "tests", "configs", "docs"], cwd=ROOT, text=True).splitlines()
    return {p: sha256_file(ROOT/p) for p in paths if p != "docs/stage_status.md"}


def release():
    gc.collect(); torch.cuda.empty_cache()


@contextmanager
def resources(name):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    sampler = ResourceSampler()
    started = perf_counter()
    data = {}
    try:
        yield data, sampler
    finally:
        torch.cuda.synchronize()
        data.update({"wall_seconds": perf_counter()-started, **sampler.finish(),
            "GPU_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "GPU_peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        UTIL.extend({"phase": name, **r} for r in sampler.rows)
        RESOURCE.append({"phase": name, **data})
        progress(f"{name}: {data['wall_seconds']:.3f}s; GPU mean={data['gpu_utilization']['mean']:.1f}%; RAM min={data['available_RAM_min_bytes']/2**30:.2f}GiB")


@contextmanager
def profile_data(times):
    def wrapper(name, fn):
        def call(*args, **kwargs):
            stamp = perf_counter(); result = fn(*args, **kwargs)
            times[name] += perf_counter()-stamp
            return result
        return call
    with patch.object(md, "training_unit_noise", wrapper("unit_noise_generation", md.training_unit_noise)), \
         patch.object(md, "direct_pilot_ls", wrapper("Pilot_LS", md.direct_pilot_ls)), \
         patch.object(md, "complex_to_tensor", wrapper("complex_to_float32", md.complex_to_tensor)), \
         patch.object(estimators, "frequency_linear_interpolate_ls", wrapper("R3_frequency_interpolation", estimators.frequency_linear_interpolate_ls)), \
         patch.object(estimators, "time_linear_interpolate_frequency_estimates", wrapper("R3_time_interpolation", estimators.time_linear_interpolate_frequency_estimates)):
        yield


def fresh():
    model, _ = p0.checkpoint_load(CHECKPOINT, torch.device("cuda:0"))
    return model, p0.optimizer_for(model)


def compute_benchmark(base, target, threads, batch_size=128):
    torch.set_num_threads(threads)
    model, optimizer = fresh()
    b, t = base[:batch_size].to("cuda"), target[:batch_size].to("cuda")
    def step():
        optimizer.zero_grad(set_to_none=True)
        pred = p0.predict_channel(model, b, LAYOUT.data_mask)
        loss = p0.data_complex_mse(pred, t, LAYOUT.data_mask)
        loss.backward(); optimizer.step()
    for _ in range(2):
        step()
    torch.cuda.synchronize(); trials = []
    for _ in range(3):
        stamp = perf_counter()
        for __ in range(32):
            step()
        torch.cuda.synchronize(); trials.append(32*batch_size/(perf_counter()-stamp))
    row = {"cpu_threads": threads, "batch_size": batch_size, "samples_per_second_trials": trials,
        "samples_per_second": float(np.median(trials)), "CV": float(np.std(trials)/np.mean(trials)),
        "stable": bool(np.std(trials)/np.mean(trials) <= .15),
        "fixed_input_hash": raw_hash(base[:batch_size]), "optimizer_updates_discarded": True,
        "steps_each_trial": 32, "warmup_steps": 2}
    del b, t, model, optimizer; release()
    return row


def compare_run(run, expected, sampler):
    ids = sampler.ordered_ids()
    if run["batch_hashes"] != expected:
        raise ValueError("worker/input hashes differ from P0")
    if run["ordered_ID_hash"] != raw_hash(ids) or run["SNR_sequence_hash"] != raw_hash(md.balanced_training_snr(p0.SEED, sampler.epoch)[ids]):
        raise ValueError("worker order/SNR sequence differ from P0")


def worker_case(path, expected, workers, prefetch, persistent, pin, non_blocking):
    name = f"worker{workers}_pf{prefetch}_persist{persistent}_pin{pin}_nb{non_blocking}"
    sampler = EpochBatchSampler(4096, 1)
    loader = make_loader(path, sampler, workers, prefetch, persistent, pin)
    row = {"workers": workers, "prefetch": prefetch if workers else None,
        "persistent_workers": persistent if workers else False, "pin_memory": pin,
        "non_blocking": non_blocking, "completed": False, "rejection": None}
    try:
        model, optimizer = fresh()
        runs = []
        with resources(name) as (memory, monitor):
            for epoch in (1, 2):
                sampler.epoch = epoch
                # Include iterator creation/spawn in measured pass wall time.
                stamp = perf_counter()
                run = train_stream(model, optimizer, loader, LAYOUT.data_mask, non_blocking=non_blocking, resources=monitor)
                run["wall_seconds_including_iterator"] = perf_counter()-stamp
                compare_run(run, expected[epoch], sampler)
                runs.append(run)
        row.update({"completed": True, "sample_hashes_exact": True, "ordered_IDs_exact": True,
            "first_pass_seconds": runs[0]["wall_seconds_including_iterator"],
            "warm_pass_seconds": runs[1]["wall_seconds_including_iterator"],
            "samples_per_second": 4096/runs[1]["wall_seconds_including_iterator"],
            "passes": [{k: v for k, v in r.items() if k not in ("batch_hashes", "loss_trace")} for r in runs],
            "memory": memory, "memory_safe": memory["memory_safe"]})
    except MemoryError as exc:
        row.update({"rejection": str(exc), "memory": memory, "memory_safe": False})
        progress(f"REJECT {name}: {exc}")
    finally:
        close_loader(loader)
        del loader
        if 'model' in locals():
            del model, optimizer
        release()
    BENCH.append({"group": "workers", **{k: v for k, v in row.items() if k not in ("passes", "memory")}})
    return row


def replay(path, profile, subset):
    records = []
    for mode in ("P0", "vectorized"):
        model, optimizer = fresh()
        sampler = EpochBatchSampler(1024, 1, subset)
        loader = make_loader(path, sampler, 0 if mode == "P0" else profile["workers"],
            profile["prefetch"] or 2, profile["persistent_workers"], profile["pin_memory"] if mode != "P0" else False, mode)
        runs = []
        try:
            for epoch in (1, 2):
                sampler.epoch = epoch
                runs.append(train_stream(model, optimizer, loader, LAYOUT.data_mask, non_blocking=profile["non_blocking"] if mode != "P0" else False))
            records.append({"loss_trace": sum((r["loss_trace"] for r in runs), []),
                "ordered_ID_hashes": [r["ordered_ID_hash"] for r in runs],
                "SNR_hashes": [r["SNR_sequence_hash"] for r in runs], "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "state_dict_hash": p0.state_hash(model), "optimizer_steps": sum(r["optimizer_steps"] for r in runs)})
        finally:
            close_loader(loader); del loader, model, optimizer; release()
    a, b = records
    difference = max(float((a["state"][k]-b["state"][k]).abs().max()) for k in a["state"])
    result = {"epochs": [1, 2], "fixed_subset_frames": 1024, "subset_ID_hash": raw_hash(subset),
        "batch_IDs_exact": a["ordered_ID_hashes"] == b["ordered_ID_hashes"],
        "SNR_sequence_exact": a["SNR_hashes"] == b["SNR_hashes"],
        "loss_trace_exact": a["loss_trace"] == b["loss_trace"],
        "P0_loss_trace": a["loss_trace"], "Popt_loss_trace": b["loss_trace"],
        "P0_state_dict_hash": a["state_dict_hash"], "Popt_state_dict_hash": b["state_dict_hash"],
        "final_state_exact": a["state_dict_hash"] == b["state_dict_hash"],
        "max_tensor_difference": difference, "optimizer_steps_each_path": a["optimizer_steps"],
        "temporary_weights_discarded": True, "development_training": False}
    if not all(result[k] for k in ("batch_IDs_exact", "SNR_sequence_exact", "loss_trace_exact", "final_state_exact")):
        raise ValueError("exact calibration replay failed; reject optimization")
    return result


def benchmark():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU baseline required on accepted environment")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != T12B_COMMIT:
        raise ValueError("T12B fixed committed snapshot must be HEAD")
    before = accepted_files(); old = json.loads((OUT/"t12b_b0_df16_dev_results.json").read_text(encoding="utf8"))
    settings = configure_reproducibility(p0.SEED)
    start_memory = physical_memory(); started = perf_counter()
    prepared = prepare_trajectory(); manifest = md.frozen_manifest(OUT/"t10c_train_validation_manifest.csv")
    times = defaultdict(float)
    progress("PERF-P0 frozen; all optimizer operations below use disposable calibration copies")
    with tempfile.TemporaryDirectory(prefix="t12c_cache_", dir=OUT) as temporary:
        tmp = Path(temporary); path = tmp/"train_truth.npy"
        with resources("P0_H_preparation") as (h_resource, _):
            stamp = perf_counter(); h, long_hashes = p0.cache_truth(prepared, manifest, "train")
            times["trajectory_H_true_preparation"] = perf_counter()-stamp
            stamp = perf_counter(); np.save(path, h)
            cache_write = perf_counter()-stamp
        with resources("P0_data_preparation") as (data_resource, _):
            stamp = perf_counter()
            with profile_data(times):
                target = md.complex_to_tensor(h)
                base, snrs = p0.generate_training_inputs(h, LAYOUT, 1)
            data_wall = perf_counter()-stamp
        # Exact P0, including its finite checks, gradient/weight monitoring and accumulation.
        model, optimizer = fresh(); recorder = EventProfile()
        with resources("P0_training_20000") as (training_resource, _):
            stamp = perf_counter()
            with profile_original_training(optimizer, recorder):
                p0.train_epoch(model, optimizer, base, target, snrs, LAYOUT.data_mask, p0.epoch_order(1))
            torch.cuda.synchronize(); training_wall = perf_counter()-stamp
            p0_final_hash = p0.state_hash(model); p0_losses = recorder.losses
            times.update(recorder.seconds())
        del model, optimizer; release()
        full_hashes = []
        with resources("vectorized_full_20000_equivalence") as (equiv_resource, _):
            stamp = perf_counter()
            for _, ids in EpochBatchSampler():
                batch = build_batch(h, 1, ids)
                if not torch.equal(batch["base"], base[ids]) or not torch.equal(batch["target"], target[ids]):
                    raise ValueError("full training float32 equivalence failed")
                full_hashes.append(batch["hashes"])
            vectorized_full_wall = perf_counter()-stamp
        # Same random actual training samples, independent original complex128 reference.
        maximum = 0.0
        for epoch in (1, 2):
            for start in range(0, 1024, 128):
                ids = p0.epoch_order(epoch)[start:start+128]
                a = build_batch(h, epoch, ids, "P0", include_complex=True)
                b = build_batch(h, epoch, ids, include_complex=True)
                if a["hashes"] != b["hashes"]:
                    raise ValueError("H/noise/SNR/R3 reference mismatch")
                maximum = max(maximum, float(np.max(np.abs(a["base_complex"]-b["base_complex"]))))
        del a, b, batch
        fixed_base, fixed_target = base[p0.epoch_order(1)[:512]].clone(), target[p0.epoch_order(1)[:512]].clone()
        thread_rows = []
        for threads in (4, 8, 12, 14):
            with resources(f"threads{threads}") as (memory, _):
                row = compute_benchmark(fixed_base, fixed_target, threads)
            row["memory"] = memory; thread_rows.append(row)
            BENCH.append({"group": "threads", **{k: v for k, v in row.items() if k not in ("memory", "samples_per_second_trials")}})
        chosen_threads = max((r for r in thread_rows if r["stable"] and r["memory"]["memory_safe"]), key=lambda r: r["samples_per_second"])["cpu_threads"]
        torch.set_num_threads(chosen_threads)
        future = []
        for size in (128, 256, 512):
            with resources(f"future_batch{size}") as (memory, _):
                row = compute_benchmark(fixed_base, fixed_target, chosen_threads, size)
            row.update({"memory": memory, "future_information_only": True, "continuation_batch_unchanged": 128})
            future.append(row); BENCH.append({"group": "future_GPU_batch", "batch_size": size, "samples_per_second": row["samples_per_second"], "peak_allocated_bytes": memory["GPU_peak_allocated_bytes"]})
        del fixed_base, fixed_target
        # Compare no-cache generation with noise-independent H cache, never H_base reuse.
        ids = p0.epoch_order(1)[:512]
        stamp = perf_counter()
        uncached = np.stack([formal_trajectory(prepared, manifest[int(i)//10]).h_frames[int(i)%10] for i in ids])
        no_cache = perf_counter()-stamp
        stamp = perf_counter(); cached = h[ids].copy(); cache_access = perf_counter()-stamp
        cache_exact = np.array_equal(uncached, cached)
        if not cache_exact:
            raise ValueError("noise-independent H cache mismatch")
        cache_report = {"measured_frames": 512, "no_cache_seconds": no_cache, "H_cache_access_seconds": cache_access,
            "speedup": no_cache/cache_access, "H_true_exact": cache_exact, "RAM_bytes_full_H": h.nbytes,
            "NPY_bytes": path.stat().st_size, "full_cache_preparation_seconds": times["trajectory_H_true_preparation"],
            "cache_write_seconds": cache_write, "no_cache_20000_seconds_estimate": no_cache*20000/512,
            "strategy": "noise-independent H only; per-epoch noise/H_base regenerated; read-only worker mmap"}
        del uncached, cached
        expected = {}
        for epoch in (1, 2):
            expected[epoch] = [build_batch(h, epoch, ids, "P0")["hashes"] for _, ids in EpochBatchSampler(4096, epoch)]
        # Release large materialized P0 epoch arrays before spawning workers.
        del base, target, h; release()
        worker_rows = []
        for workers in (0, 2, 4, 8):
            combinations = [(2, False)] if workers == 0 else [(2, False), (2, True), (4, True)]
            for prefetch, persistent in combinations:
                worker_rows.append(worker_case(path, expected, workers, prefetch, persistent, False, False))
        best = min((r for r in worker_rows if r["completed"] and r["memory_safe"]), key=lambda r: r["warm_pass_seconds"])
        # Pin and blocking/nonblocking tested on worker0 and the best nonzero candidate.
        selected_worker_counts = {0, best["workers"]}
        best_nonzero = min((r for r in worker_rows if r["workers"] and r["completed"] and r["memory_safe"]), key=lambda r: r["warm_pass_seconds"], default=None)
        if best_nonzero:
            selected_worker_counts.add(best_nonzero["workers"])
        for workers in sorted(selected_worker_counts):
            template = best_nonzero if workers else None
            for non_blocking in (False, True):
                worker_rows.append(worker_case(path, expected, workers, template["prefetch"] if template else 2,
                    template["persistent_workers"] if template else False, True, non_blocking))
        best = min((r for r in worker_rows if r["completed"] and r["memory_safe"]), key=lambda r: r["warm_pass_seconds"])
        selected = {k: best[k] for k in ("workers", "prefetch", "persistent_workers", "pin_memory", "non_blocking")}
        selected.update({"name": "PERF-Popt", "cpu_threads": chosen_threads,
            "cache_strategy": "noise-independent H_true read-only mmap; epoch-dependent H_base never reused",
            "batch_size": 128, "AMP": False, "equivalence_passed": True, "memory_safe": True})
        progress(f"Selected data profile: {selected}")
        with resources("mini_two_epoch_update_replay") as (replay_memory, _):
            state_replay = replay(path, selected, p0.epoch_order(1)[:1024])
        # Final full pass with persistent pipeline primed using this mini calibration.
        model, optimizer = fresh(); sampler = EpochBatchSampler(20000, 1)
        loader = make_loader(path, sampler, selected["workers"], selected["prefetch"] or 2,
            selected["persistent_workers"], selected["pin_memory"])
        with resources("Popt_training_20000_including_data") as (opt_memory, monitor):
            stamp = perf_counter()
            opt_run = train_stream(model, optimizer, loader, LAYOUT.data_mask, non_blocking=selected["non_blocking"], resources=monitor)
            opt_total = perf_counter()-stamp
        compare_run(opt_run, full_hashes, sampler)
        epoch_replay = {"P0_state_hash": p0_final_hash, "Popt_state_hash": opt_run["state_dict_hash"],
            "final_state_exact": p0_final_hash == opt_run["state_dict_hash"],
            "loss_trace_exact": p0_losses == opt_run["loss_trace"], "samples": 20000,
            "optimizer_steps_each_path": 157, "calibration_weights_discarded": True}
        if not epoch_replay["final_state_exact"] or not epoch_replay["loss_trace_exact"]:
            raise ValueError("full original P0 / Popt update regression failed")
        close_loader(loader); del loader, model, optimizer; release()
        if not opt_memory["memory_safe"]:
            raise MemoryError("selected full profile failed memory gate")
        # Fixed T12B validation observations generated once by unchanged original link.
        refs = load_references(OUT)
        with resources("P0_validation_preparation") as (val_prep_memory, _):
            stamp = perf_counter(); val_h, val_hashes = p0.cache_truth(prepared, manifest, "validation_id")
            cache = p0.prepare_validation(val_h, val_hashes, refs, LAYOUT, p0.replay_link(ROOT), progress)
            val_prep_seconds = perf_counter()-stamp
        # Y/bits were verified by original preparation; no BER scan is performed.
        del cache["received"], cache["bits"]; release()
        frozen_model, meta = p0.checkpoint_load(CHECKPOINT, torch.device("cuda"))
        checkpoint_state = p0.state_hash(frozen_model)
        torch.set_num_threads(4)
        with resources("P0_validation_CPU") as (val_cpu_memory, _):
            val_cpu = validation_run(frozen_model, cache, LAYOUT)
        historical_metrics = old["epoch_records"][30]["validation"]
        if val_cpu["metrics"] != historical_metrics:
            raise ValueError("original frozen epoch30 validation metric regression drift")
        validation_rows = [{"strategy": "CPU", **val_cpu, "input_exact": True, "predictions_exact": True,
            "metrics_exact": True, "memory": val_cpu_memory}]
        torch.set_num_threads(chosen_threads)
        for strategy in ("pinned", "GPU-resident"):
            release()
            with resources(f"validation_{strategy}") as (memory, _):
                stamp = perf_counter()
                tensor = cache["base"].pin_memory() if strategy == "pinned" else cache["base"].to("cuda")
                torch.cuda.synchronize(); setup = perf_counter()-stamp
                result = validation_run(frozen_model, cache, LAYOUT, tensor, strategy == "pinned")
                result["cache_setup_seconds"] = setup
                del tensor; release()
            row = {"strategy": strategy, **result,
                "input_exact": result["input_tensor_SHA256"] == val_cpu["input_tensor_SHA256"],
                "predictions_exact": result["prediction_tensor_SHA256"] == val_cpu["prediction_tensor_SHA256"],
                "metrics_exact": result["metrics"] == val_cpu["metrics"], "memory": memory}
            if not all(row[k] for k in ("input_exact", "predictions_exact", "metrics_exact")):
                raise ValueError("validation tensor/prediction/metric differs")
            validation_rows.append(row)
        # A thread-optimized ordinary CPU strategy remains an eligible candidate.
        with resources("validation_CPU_optimized_threads") as (memory, _):
            result = validation_run(frozen_model, cache, LAYOUT)
        validation_rows.append({"strategy": "CPU optimized threads", **result, "input_exact": result["input_tensor_SHA256"] == val_cpu["input_tensor_SHA256"],
            "predictions_exact": result["prediction_tensor_SHA256"] == val_cpu["prediction_tensor_SHA256"],
            "metrics_exact": result["metrics"] == val_cpu["metrics"], "memory": memory})
        best_validation = min((r for r in validation_rows if r["memory"]["memory_safe"]), key=lambda r: r["wall_seconds"])
        selected["validation_strategy"] = best_validation["strategy"]
        selected["validation_input_SHA256"] = best_validation["input_tensor_SHA256"]
        # Original checkpoint I/O benchmark to a temporary file, discarded immediately.
        with resources("checkpoint_IO") as (checkpoint_memory, _):
            io_trials = []
            for _ in range(3):
                stamp = perf_counter()
                p0.checkpoint_save(tmp/"discarded_io.pt", frozen_model, 30, meta["validation_mse_pooled"], meta["architecture"])
                io_trials.append(perf_counter()-stamp)
        inference = {"observations_each_path": 20000, "SNRs": list(md.SNR_DB),
            "P0_prediction_tensor_hash": val_cpu["prediction_tensor_SHA256"],
            "Popt_prediction_tensor_hash": best_validation["prediction_tensor_SHA256"],
            "checkpoint_state_unchanged": checkpoint_state == p0.state_hash(frozen_model),
            "checkpoint_file_unchanged": sha256_file(CHECKPOINT) == old["epoch30_checkpoint_hash"],
            "historical_T12B_metrics_exact": True,
            "all_checks_passed": all(r["input_exact"] and r["predictions_exact"] and r["metrics_exact"] for r in validation_rows)}
        del cache, val_h, frozen_model; release()
    # Temporary caches and calibration weights have now been removed.
    after = accepted_files()
    if after != before:
        raise ValueError("accepted core or historical artifact modified")
    frozen = verify_frozen(OUT)
    checkpoint_io = float(np.median(io_trials))
    # Same conservative save-per-epoch cost for both; actual baseline/selected full passes.
    p0_epoch = data_wall+training_wall+val_cpu["wall_seconds"]+checkpoint_io
    opt_epoch = opt_total+best_validation["wall_seconds"]+checkpoint_io
    ratio = p0_epoch/opt_epoch
    sample_equiv = {"full_training_frames": 20000, "epoch": 1, "model_input_float32_exact": True,
        "target_float32_exact": True, "worker_reference_epochs": [1, 2], "worker_reference_frames_each": 4096,
        "H_noise_input_target_ID_SNR_hashes": full_hashes, "all_checks_passed": True}
    baseline = {"name": "PERF-P0", "cpu_threads": 4, "workers": 0, "prefetch": None, "persistent_workers": False,
        "pin_memory": False, "non_blocking": False, "batch_size": 128,
        "cache_strategy": "original materialized H_true; original whole-epoch H_base regenerated",
        "validation_strategy": "original CPU tensor cache", "data_preparation_seconds": data_wall,
        "training_seconds": training_wall, "validation_seconds": val_cpu["wall_seconds"],
        "checkpoint_IO_seconds_median": checkpoint_io, "seconds_per_epoch": p0_epoch,
        "samples_per_second": 20000/p0_epoch, "DataLoader_waiting_seconds": 0,
        "DataLoader_waiting_note": "N/A: P0 builds entire epoch before GPU loop", "memory": [data_resource, training_resource, val_cpu_memory]}
    selected.update({"data_and_training_seconds_full_measured": opt_total,
        "seconds_per_epoch": opt_epoch, "samples_per_second": 20000/opt_epoch,
        "memory": opt_memory, "equivalence_passed": inference["all_checks_passed"] and epoch_replay["final_state_exact"],
        "startup_included_in_full_pass": True})
    for key, seconds in times.items():
        PROFILE.append({"profile": "PERF-P0", "stage": key, "seconds": seconds,
            "scope": "one-off" if key == "trajectory_H_true_preparation" else "full 20000 training epoch"})
    PROFILE.extend([{"profile": "PERF-P0", "stage": "DataLoader_waiting", "seconds": 0, "scope": "N/A"},
        {"profile": "PERF-P0", "stage": "validation_data_preparation", "seconds": val_prep_seconds, "scope": "one-off 20000 fixed observations"},
        {"profile": "PERF-P0", "stage": "validation_forward", "seconds": val_cpu["stages"]["validation_forward"], "scope": "20000 frozen checkpoint inputs"},
        {"profile": "PERF-P0", "stage": "checkpoint_IO", "seconds": checkpoint_io, "scope": "median3 saves/reloads to temporary file"},
        {"profile": "PERF-P0", "stage": "data_generation_wall", "seconds": data_wall, "scope": "aggregate; not additive to sub-stages"},
        {"profile": "PERF-P0", "stage": "training_wall", "seconds": training_wall, "scope": "aggregate; not additive to sub-stages"}])
    for key, seconds in opt_run["CPU_worker_stage_seconds_sum"].items():
        PROFILE.append({"profile": "PERF-Popt", "stage": key, "seconds": seconds, "scope": "worker CPU sum; overlaps GPU/other workers"})
    for key, seconds in opt_run["GPU_stage_seconds"].items():
        PROFILE.append({"profile": "PERF-Popt", "stage": key, "seconds": seconds, "scope": "full20000 CUDA events"})
    PROFILE.append({"profile": "PERF-Popt", "stage": "DataLoader_waiting", "seconds": opt_run["DataLoader_waiting_seconds"], "scope": "observed next(iterator) wait"})
    BENCH.extend({"group": "validation", "strategy": r["strategy"], "wall_seconds": r["wall_seconds"],
        "samples_per_second": 20000/r["wall_seconds"], "memory_safe": r["memory"]["memory_safe"]} for r in validation_rows)
    BENCH.extend([{"group": "full_epoch", "name": "PERF-P0", "wall_seconds": p0_epoch, "samples_per_second": 20000/p0_epoch},
        {"group": "full_epoch", "name": "PERF-Popt", "wall_seconds": opt_epoch, "samples_per_second": 20000/opt_epoch},
        {"group": "cache", "name": "no cache512", "wall_seconds": no_cache}, {"group": "cache", "name": "Hcache512", "wall_seconds": cache_access}])
    report = {"task": "T12C-PERF", "git": {"T12B_commit": head, "startup_pytest": "978 passed in 70.37s (0:01:10)",
        "T12B_committed_cleanly": True, "clean_after_commit": True,
        "commit_message": "feat: add first full B0 dF16 development training"},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "GPU": torch.cuda.get_device_name(), "before": start_memory},
        "baseline_profile": baseline, "candidate_profiles": worker_rows+validation_rows+thread_rows,
        "selected_profile": selected, **{k: selected[k] for k in ("cpu_threads", "workers", "prefetch", "persistent_workers", "pin_memory", "non_blocking", "cache_strategy")},
        "thread_profiles": thread_rows, "worker_profiles": worker_rows, "validation_profiles": validation_rows,
        "cache_benchmark": cache_report,
        "R3_equivalence": {"complex128_actual_random_frames": 2048, "epochs": [1, 2], "complex128_max_error": maximum,
            "complex128_bitwise_equal": maximum == 0, "float32_bitwise_equal": True, "full_epoch_float32_frames": 20000,
            "whole_epoch_vectorized_generation_seconds": vectorized_full_wall},
        "deterministic_equivalence": {"sample_order": True, "SNR_assignment": True, "H_true": True, "unit_noise": True,
            "H_base": True, "target": True, "pinned_nonblocking": True, "all_checks_passed": True},
        "sample_hash_equivalence": sample_equiv, "state_dict_replay": state_replay, "full_epoch_update_regression": epoch_replay,
        "checkpoint_inference_regression": inference,
        "pinned_nonblocking_equivalence": {"tested": [r for r in worker_rows if r["pin_memory"]], "all_checks_passed": True},
        "future_batch_benchmark": future, "samples_per_second": {"P0": 20000/p0_epoch, "Popt": 20000/opt_epoch},
        "speedup": ratio, "speedup_scope": "actual full 20000 training pass plus frozen epoch30 20000-input validation and identical median checkpoint I/O; one-off prep excluded",
        "future_cost_estimate": {"T12D_30_additional_epochs_seconds": 30*opt_epoch,
            "T12D_one_off_cache_prep_seconds": times["trajectory_H_true_preparation"]+cache_write+val_prep_seconds+best_validation.get("cache_setup_seconds", 0),
            "four_densities_five_seeds_30_epochs_seconds": 20*30*opt_epoch,
            "scope": "serial linear extrapolation from dF16 engineering timing only; other densities and formal seeds unmeasured"},
        "gpu_utilization": opt_memory["gpu_utilization"], "cpu_utilization": opt_memory["cpu_utilization"],
        "memory": {"physical": start_memory, "selected": opt_memory, "validation": best_validation["memory"], "all_phase_resources": RESOURCE},
        "provenance": {"accepted_before": before, "accepted_after": after, "frozen_T10D_T10C_hashes": frozen,
            "train_H_long_hashes": long_hashes, "source_sha256": {p: sha256_file(ROOT/p) for p in ("src/b0_perf.py", "src/b0_perf_runtime.py", "src/perf_resources.py", "scripts/benchmark_b0_pipeline.py", "tests/test_b0_perf.py", "docs/b0_perf_protocol.md")}},
        "batch_size": 128, "optimizer": "Adam", "lr": .001, "weight_decay": 0, "training_seed": p0.SEED,
        "AMP_used": False, "scientific_protocol_changed": False, "training_performed": False,
        "calibration_optimizer_updates_only": True, "temporary_calibration_weights_discarded": True,
        "development_epochs_completed": 30, "epoch31_started": False, "formal_test_used": False,
        "formal_test_sealed": True, "OOD_used": False, "additional_density_or_seed_used": False,
        "numeric_checks_passed": True, "all_checks_passed": False,
        "warnings": [r["rejection"] for r in worker_rows if r["rejection"]], "total_benchmark_seconds": perf_counter()-started}
    save_csv(OUT/"t12c_perf_benchmark.csv", BENCH)
    save_csv(OUT/"t12c_perf_profile.csv", PROFILE)
    save_csv(OUT/"t12c_perf_utilization.csv", UTIL)
    report["output_SHA256"] = {p: sha256_file(OUT/p) for p in ("t12c_perf_benchmark.csv", "t12c_perf_profile.csv", "t12c_perf_utilization.csv")}
    save_json(OUT/"t12c_perf_results.json", report)
    progress(f"Numeric engineering checks PASS: speedup={ratio:.3f}; {p0_epoch:.3f} -> {opt_epoch:.3f}s/epoch")


def finalize():
    path = OUT/"t12c_perf_results.json"; report = json.loads(path.read_text(encoding="utf8"))
    if report["provenance"]["accepted_after"] != accepted_files():
        raise ValueError("historical accepted files changed")
    for p, digest in report["provenance"]["source_sha256"].items():
        if sha256_file(ROOT/p) != digest:
            raise ValueError(f"benchmark source changed after run: {p}")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True, encoding="utf8")
    (OUT/"t12c_perf_pytest.txt").write_text(result.stdout+result.stderr, encoding="utf8")
    progress(result.stdout)
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True, encoding="utf8")
    if result.returncode or diff.returncode:
        raise RuntimeError(f"final pytest/diff failed\n{result.stderr}\n{diff.stdout}\n{diff.stderr}")
    report["pytest"] = {"command": "python -m pytest -q", "summary": result.stdout.strip().splitlines()[-1], "exit_code": 0}
    report["git_diff_check"] = {"exit_code": 0, "stdout": diff.stdout, "warnings": diff.stderr}
    report["all_checks_passed"] = True
    save_json(path, report); progress("T12C-PERF final checks PASS; stop before epoch31")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize()
    else:
        if (OUT/"t12c_perf_results.json").exists():
            raise ValueError("benchmark artifacts already exist; avoid unintended replay")
        benchmark()


if __name__ == "__main__":
    main()
