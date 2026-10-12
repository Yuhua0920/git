"""Measure co-resident selected caches/workers and uninstrumented validation cost."""
import csv
import gc
import json
import sys
import tempfile
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import benchmark_b0_pipeline as bench
from src import b0_development as p0, ml_data as md
from src.b0_model import configure_reproducibility
from src.b0_perf import EpochBatchSampler, make_loader, close_loader, raw_hash
from src.b0_perf_runtime import train_stream
from src.lmmse_train import sha256_file
from src.nearest_validation import load_references
from src.trajectories import prepare_trajectory


def csv_rows(path):
    with path.open(encoding="utf8", newline="") as file:
        return list(csv.DictReader(file))


def main():
    path = bench.OUT/"t12c_perf_results.json"
    report = json.loads(path.read_text(encoding="utf8"))
    if "combined_residency_verification" in report:
        raise ValueError("combined profile already measured")
    if bench.accepted_files() != report["provenance"]["accepted_after"]:
        raise ValueError("accepted files changed")
    configure_reproducibility(p0.SEED, report["cpu_threads"])
    prepared = prepare_trajectory(); manifest = md.frozen_manifest(bench.OUT/"t10c_train_validation_manifest.csv")
    old = json.loads((bench.OUT/"t12b_b0_df16_dev_results.json").read_text(encoding="utf8"))
    with tempfile.TemporaryDirectory(prefix="t12c_residency_", dir=bench.OUT) as temporary:
        tmp = Path(temporary).resolve()
        if not tmp.is_relative_to(bench.OUT.resolve()):
            raise ValueError("temporary cache must stay inside outputs")
        with bench.resources("combined_fixed_validation_preparation") as (prep_memory, _):
            h_val, val_hashes = p0.cache_truth(prepared, manifest, "validation_id")
            cache = p0.prepare_validation(h_val, val_hashes, load_references(bench.OUT), bench.LAYOUT, p0.replay_link(ROOT), bench.progress)
        del cache["received"], cache["bits"]; bench.release()
        frozen_model, _ = p0.checkpoint_load(bench.CHECKPOINT, torch.device("cuda"))
        with bench.resources("P0_validation_without_prediction_hashing") as (cpu_memory, _):
            stamp = perf_counter(); cpu_metrics = p0.evaluate_validation(frozen_model, cache, bench.LAYOUT)
            torch.cuda.synchronize(); cpu_seconds = perf_counter()-stamp
        if cpu_metrics != old["epoch_records"][30]["validation"]:
            raise ValueError("uninstrumented P0 validation metrics drift")
        original_input_hash = raw_hash(cache["base"])
        gpu_base = cache["base"].to("cuda")
        if raw_hash(gpu_base) != original_input_hash:
            raise ValueError("GPU validation cache changed")
        # Release redundant CPU H_base before measuring the GPU-resident profile.
        cache["base"] = gpu_base
        bench.release()
        with bench.resources("combined_training_H_preparation") as (h_memory, _):
            h_train, train_hashes = p0.cache_truth(prepared, manifest, "train")
            if train_hashes != old["provenance"]["train_H_long_hashes"]:
                raise ValueError("all 2000 original T12B H_long hashes must match")
            np.save(tmp/"h.npy", h_train)
            del h_train; bench.release()
        profile = report["selected_profile"]
        sampler = EpochBatchSampler()
        loader = make_loader(tmp/"h.npy", sampler, profile["workers"], profile["prefetch"] or 2,
            profile["persistent_workers"], profile["pin_memory"])
        model, optimizer = bench.fresh()
        try:
            with bench.resources("Popt_combined_full_epoch") as (combined, monitor):
                stamp = perf_counter()
                run = train_stream(model, optimizer, loader, bench.LAYOUT.data_mask, non_blocking=profile["non_blocking"], resources=monitor)
                train_seconds = perf_counter()-stamp
                bench.compare_run(run, report["sample_hash_equivalence"]["H_noise_input_target_ID_SNR_hashes"], sampler)
                if run["state_dict_hash"] != report["full_epoch_update_regression"]["P0_state_hash"]:
                    raise ValueError("co-resident full-epoch final state drift")
                # Keep persistent workers, training model/Adam, H mapping and GPU cache live.
                stamp = perf_counter(); gpu_metrics = p0.evaluate_validation(frozen_model, cache, bench.LAYOUT)
                torch.cuda.synchronize(); gpu_seconds = perf_counter()-stamp
                if gpu_metrics != cpu_metrics:
                    raise ValueError("co-resident validation metrics drift")
            if not combined["memory_safe"]:
                raise MemoryError("selected co-resident profile failed predeclared memory gate")
        finally:
            close_loader(loader); del loader, model, optimizer, frozen_model, gpu_base, cache, h_val; bench.release()
    # All numerical performance/BER data remain untouched. Only engineering timings added.
    checkpoint_io = report["baseline_profile"]["checkpoint_IO_seconds_median"]
    baseline = report["baseline_profile"]
    p0_epoch = baseline["data_preparation_seconds"]+baseline["training_seconds"]+cpu_seconds+checkpoint_io
    opt_epoch = train_seconds+gpu_seconds+checkpoint_io
    report["profiled_baseline_profile"] = dict(baseline)
    report["profiled_selected_profile"] = dict(report["selected_profile"])
    report["profiled_speedup"] = report["speedup"]
    report["instrumentation_note"] = "Initial validation profiles include prediction SHA256 verification/D2H copies. Final epoch planning uses unchanged original metric path without these extra checks. Other profiles retain timing/resource/required finite-check overhead."
    baseline.update({"validation_seconds": cpu_seconds, "seconds_per_epoch": p0_epoch, "samples_per_second": 20000/p0_epoch})
    profile.update({"validation_strategy": "GPU-resident", "redundant_CPU_validation_H_base_released": True,
        "data_and_training_seconds_full_measured": train_seconds, "seconds_per_epoch": opt_epoch,
        "samples_per_second": 20000/opt_epoch, "validation_seconds": gpu_seconds, "memory": combined,
        "combined_residency_verified": True, "memory_safe": combined["memory_safe"]})
    report["samples_per_second"] = {"P0": 20000/p0_epoch, "Popt": 20000/opt_epoch}
    report["speedup"] = p0_epoch/opt_epoch
    report["speedup_scope"] = "actual full training pass with original required finite/metric checks plus original uninstrumented fixed validation and identical checkpoint I/O; Popt workers and GPU cache coexist, first spawn included; one-off prep excluded"
    report["combined_residency_verification"] = {"full_train_samples": 20000, "validation_inputs": 20000,
        "workers_remained_live_during_validation": profile["persistent_workers"], "input_hashes_exact": True,
        "full_update_state_exact": True, "validation_metrics_exact": True, "train_H_long_hashes_matched": 2000,
        "CPU_validation_seconds": cpu_seconds, "GPU_validation_seconds": gpu_seconds, "training_seconds": train_seconds,
        "memory": combined, "memory_safe": combined["memory_safe"], "all_checks_passed": True}
    report["gpu_utilization"], report["cpu_utilization"] = combined["gpu_utilization"], combined["cpu_utilization"]
    report["memory"]["selected_combined"] = combined
    report["memory"]["all_phase_resources"].extend(bench.RESOURCE)
    report["calibration_initial_state"] = {"checkpoint": "outputs/t12b_b0_df16_dev_epoch30.pt",
        "state_dict_hash": old["checkpoints"]["epoch30"]["state_dict_tensor_sha256"],
        "Adam_state": "fresh identical disposable optimizer; not continuation state",
        "epochs_of_input_noise": [1, 2], "weights_saved": False}
    report["measurement_limits"] = ["Single local hardware session; system-wide CPU/GPU utilization includes background activity.",
        "Thread4/12/14 medians differ by less than 0.02%; 4 chosen by predeclared highest stable measured median, not a claim of statistical superiority.",
        "Worker RSS sums shared mapped pages more than once; available physical RAM is the admission gate.",
        "Future 20-model cost extrapolates dF16/30 epochs only; other densities/seeds and their cache preparation unmeasured."]
    report["future_cost_estimate"]["T12D_30_additional_epochs_seconds"] = 30*opt_epoch
    report["future_cost_estimate"]["four_densities_five_seeds_30_epochs_seconds"] = 600*opt_epoch
    report["provenance"]["source_sha256"].update({p: sha256_file(ROOT/p) for p in
        ("scripts/verify_b0_perf_profile.py", "tests/test_b0_perf_residency.py")})
    benchmark_path, profile_path, util_path = (bench.OUT/p for p in ("t12c_perf_benchmark.csv", "t12c_perf_profile.csv", "t12c_perf_utilization.csv"))
    rows = csv_rows(benchmark_path)
    # Preserve original profiled rows; final rows have explicit measurement scope.
    rows.extend([{"group": "full_epoch_without_validation_hashing", "name": "PERF-P0", "wall_seconds": p0_epoch, "samples_per_second": 20000/p0_epoch},
        {"group": "full_epoch_without_validation_hashing", "name": "PERF-Popt", "wall_seconds": opt_epoch, "samples_per_second": 20000/opt_epoch},
        {"group": "validation_without_prediction_hashing", "strategy": "CPU", "wall_seconds": cpu_seconds},
        {"group": "validation_without_prediction_hashing", "strategy": "GPU-resident co-resident", "wall_seconds": gpu_seconds}])
    bench.save_csv(benchmark_path, rows)
    rows = csv_rows(profile_path)
    rows.extend([{"profile": "PERF-P0", "stage": "validation_wall_without_prediction_hashing", "seconds": cpu_seconds, "scope": "original pure fixed validation metric path"},
        {"profile": "PERF-Popt", "stage": "validation_wall_without_prediction_hashing", "seconds": gpu_seconds, "scope": "GPU cache + persistent workers co-resident"},
        {"profile": "PERF-Popt", "stage": "training_wall_combined", "seconds": train_seconds, "scope": "full20000 first spawn included; fixed GPU validation cache resident"}])
    bench.save_csv(profile_path, rows); bench.save_csv(util_path, csv_rows(util_path)+bench.UTIL)
    report["output_SHA256"] = {p.name: sha256_file(p) for p in (benchmark_path, profile_path, util_path)}
    report["provenance"]["accepted_after"] = bench.accepted_files()
    if report["provenance"]["accepted_before"] != report["provenance"]["accepted_after"]:
        raise ValueError("historical file changed")
    bench.save_json(path, report)
    bench.progress(f"Combined profile PASS: {p0_epoch:.4f} -> {opt_epoch:.4f}s/epoch; speedup {report['speedup']:.4f}")


if __name__ == "__main__":
    main()
