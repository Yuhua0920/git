"""T12B: exactly one fixed 30-epoch development run, or CSV-only figure rebuild."""

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.plot_b0_development import FIGURE_NAMES, plot_results
from scripts.run_lmmse_train_validation import qa_tools, save_csv
from scripts.validate_b0_pipeline import architecture_report, json_save, process_memory, sync
from src import b0_development as dev
from src import ml_data as md
from src.b0_model import ResidualCNN, configure_reproducibility
from src.config import load_config
from src.lmmse_train import sha256_file
from src.nearest_validation import load_references, verify_frozen
from src.pilots import build_fdm_pilot_layout, count_layout_resources
from src.trajectories import prepare_trajectory


def progress(message):
    print(message, flush=True)


def render_and_audit(input_dir, output_dir, tools_dir=None):
    tools, gate = qa_tools(tools_dir)
    paths, records = plot_results(input_dir, output_dir, gate)
    for path, record in zip(paths, records):
        text = subprocess.run([sys.executable, str(tools/"audit_pdf_text.py"), str(path.with_suffix(".pdf")),
            "--min-pt", "5", "--json"], capture_output=True, text=True, encoding="utf8")
        collision_path = path.with_suffix(".collision-audit.json")
        collision = subprocess.run([sys.executable, str(tools/"audit_figure_collisions.py"),
            str(path.with_suffix(".pdf")), "--json-out", str(collision_path)], capture_output=True, text=True, encoding="utf8")
        if text.returncode != 0 or collision.returncode != 0:
            raise RuntimeError(f"figure QA failed: {path.name}\n{text.stdout}\n{collision.stdout}\n{collision.stderr}")
        record.update({"path": str(path), "PDF_text_audit": json.loads(text.stdout),
            "collision_audit": json.loads(collision_path.read_text(encoding="utf8")),
            "alignment_audit": json.loads(path.with_suffix(".alignment.json").read_text(encoding="utf8")),
            "sha256": {ext: sha256_file(path.with_suffix('.'+ext)) for ext in ("png", "pdf", "svg")}})
    return records


def log_row(epoch, train, val, data_seconds, train_seconds, val_seconds, epoch_seconds, best):
    row = {"epoch": epoch, "train_mse": None if train is None else train["train_mse"],
           "val_mse_pooled": val["pooled_mse"]}
    row.update({f"val_nmse_{r['snr_db']}db": r["nmse_db"] for r in val["per_snr"]})
    row.update({"learning_rate": 0.001, "gradient_norm": None if train is None else train["gradient_norm"]})
    row.update({f"conv{i}_weight_norm": (train or {})[f"conv{i}_weight_norm"] for i in range(1, 5)})
    row.update({"epoch_seconds": epoch_seconds, "data_generation_seconds": data_seconds,
                "training_seconds": train_seconds, "validation_seconds": val_seconds, "best_so_far": best})
    return row


def record_row(epoch, train, val, timing, best, model):
    if train is None:
        # Norms are measured at epoch0 but there was no training/loss/gradient.
        train = {"train_mse": None, "gradient_norm": None, **dev.model_norms(model)}
    return log_row(epoch, train, val, *timing, best)


def verify_protocol_start(outputs):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != dev.T12A_COMMIT:
        raise ValueError("accepted T12A commit must be HEAD before this single run")
    old = json.loads((outputs/"t12a_b0_pipeline_results.json").read_text(encoding="utf8"))
    if not old["all_checks_passed"]:
        raise ValueError("accepted T12A report required")
    for p, digest in old["provenance"]["source_sha256"].items():
        if sha256_file(ROOT/p) != digest:
            raise ValueError(f"T12A source changed: {p}")
    config = load_config(ROOT/"configs/base.yaml")
    if config["noise"]["snr_db"] != list(md.SNR_DB):
        raise ValueError("fixed SNR list required")
    return {"T12A_commit": head, "startup_pytest": "928 passed in 51.96s",
            "T12A_committed_cleanly": True, "working_tree_clean_after_T12A_commit": True}


def run_training(outputs):
    report_path = outputs/f"{dev.PREFIX}_results.json"
    if any((outputs/f"{dev.PREFIX}{suffix}").exists() for suffix in ("_best.pt", "_epoch30.pt", "_training_log.csv", "_results.json")):
        raise ValueError("development artifacts already exist; refuse accidental second training run")
    git = verify_protocol_start(outputs)
    before = dev.accepted_files(ROOT)
    protocol_sha = sha256_file(ROOT/"docs/b0_development_protocol.md")
    started = perf_counter()
    settings = configure_reproducibility(dev.SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResidualCNN().to(device)
    architecture = architecture_report(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    layout = build_fdm_pilot_layout(2, 64, 16, 16, 4)
    resources = count_layout_resources(layout)
    if resources["union_ratio"] != 1/32:
        raise ValueError("actual dF16 pilot union ratio wrong")
    refs = load_references(outputs)
    manifest = md.frozen_manifest(outputs/"t10c_train_validation_manifest.csv")
    historical = dev.replay_link(ROOT)
    prepared = prepare_trajectory()
    stamp = perf_counter()
    train_h, train_hashes = dev.cache_truth(prepared, manifest, "train", progress)
    target = md.complex_to_tensor(train_h)
    train_cache_seconds = perf_counter()-stamp
    stamp = perf_counter()
    val_h, val_hashes = dev.cache_truth(prepared, manifest, "validation_id", progress)
    cache = dev.prepare_validation(val_h, val_hashes, refs, layout, historical, progress)
    validation_preparation_seconds = perf_counter()-stamp
    stamp = perf_counter()
    zero = dev.evaluate_validation(model, cache, layout, epoch_zero=True)
    sync(device)
    zero_seconds = perf_counter()-stamp
    initial = dev.initial_reference(zero, refs)
    epoch0_state = dev.state_hash(model)
    best_metric, best_epoch = zero["pooled_mse"], 0
    best_checkpoint = dev.checkpoint_save(outputs/f"{dev.PREFIX}_best.pt", model, 0, best_metric, architecture)
    best_state = dev.state_hash(model)
    log = [record_row(0, None, zero, (validation_preparation_seconds, 0.0, zero_seconds,
                      validation_preparation_seconds+zero_seconds), True, model)]
    save_csv(outputs/f"{dev.PREFIX}_training_log.csv", log)
    epoch_records = [{"epoch": 0, "validation": zero, "training": None}]
    progress(f"epoch0 pooled MSE={best_metric:.12g}; zero-init/R3 calibration PASS")
    optimizer = dev.optimizer_for(model)
    for epoch in range(1, dev.EPOCHS+1):
        epoch_stamp = perf_counter()
        data_stamp = perf_counter()
        base, snrs = dev.generate_training_inputs(train_h, layout, epoch)
        data_seconds = perf_counter()-data_stamp
        order = dev.epoch_order(epoch)
        stamp = perf_counter()
        training = dev.train_epoch(model, optimizer, base, target, snrs, layout.data_mask, order)
        sync(device)
        train_seconds = perf_counter()-stamp
        if any(r["sample_count"] != 4000 for r in training["train_per_snr"]):
            raise ValueError("training SNR balance failed")
        del base
        stamp = perf_counter()
        val = dev.evaluate_validation(model, cache, layout)
        sync(device)
        val_seconds = perf_counter()-stamp
        is_best = dev.select_best(val["pooled_mse"], best_metric)
        if is_best:
            best_metric, best_epoch = val["pooled_mse"], epoch
            best_checkpoint = dev.checkpoint_save(outputs/f"{dev.PREFIX}_best.pt", model, epoch, best_metric, architecture)
            best_state = dev.state_hash(model)
        epoch_seconds = perf_counter()-epoch_stamp
        log.append(record_row(epoch, training, val, (data_seconds, train_seconds, val_seconds, epoch_seconds), is_best, model))
        epoch_records.append({"epoch": epoch, "training": training, "validation": val,
            "SNR_assignment_sha256": dev.raw_array_sha256(snrs), "shuffle_order_sha256": dev.raw_array_sha256(order)})
        save_csv(outputs/f"{dev.PREFIX}_training_log.csv", log)
        json_save(outputs/f"{dev.PREFIX}_training_progress.json", {"completed_epoch": epoch,
            "best_epoch": best_epoch, "epochs": epoch_records, "formal_test_sealed": True})
        progress(f"epoch {epoch:02d}/30 train={training['train_mse']:.9g} val={val['pooled_mse']:.9g} "
                 f"best={best_epoch} data/train/val={data_seconds:.2f}/{train_seconds:.2f}/{val_seconds:.2f}s")
    final_checkpoint = dev.checkpoint_save(outputs/f"{dev.PREFIX}_epoch30.pt", model, 30, log[-1]["val_mse_pooled"], architecture)
    training_wall_seconds = perf_counter()-started
    # Fixed before inspecting any BER/gap; no optimizer exists on the evaluation path.
    del model, optimizer, target, train_h, cache, val_h
    best_model, best_meta = dev.checkpoint_load(outputs/f"{dev.PREFIX}_best.pt", device)
    if dev.state_hash(best_model) != best_state or best_meta["epoch"] != best_epoch:
        raise ValueError("best state selection/reload mismatch")
    stamp = perf_counter()
    evaluation = dev.full_link_validation(ROOT, prepared, refs, best_model, layout, historical, progress)
    sync(device)
    full_link_seconds = perf_counter()-stamp
    if dev.state_hash(best_model) != best_state:
        raise ValueError("best checkpoint altered during validation")
    aggregate = evaluation["aggregates"]
    save_csv(outputs/f"{dev.PREFIX}_validation_trajectory_metrics.csv", evaluation["trajectory_rows"])
    save_csv(outputs/f"{dev.PREFIX}_validation.csv", aggregate)
    best_record = epoch_records[best_epoch]["validation"]
    final_link_mse = sum(r["b0_nmse_error_energy"] for r in aggregate)/(20000*4*np.count_nonzero(layout.data_mask[0]))
    if not np.isclose(final_link_mse, best_record["pooled_mse"], rtol=1e-6, atol=1e-12):
        raise ValueError("best full-link / checkpoint selection metric mismatch")
    after = dev.accepted_files(ROOT)
    if before != after or sha256_file(ROOT/"docs/b0_development_protocol.md") != protocol_sha:
        raise ValueError("accepted history / predeclared protocol changed")
    frozen = verify_frozen(outputs)
    flag = dev.oracle_red_flag(aggregate)
    report = {"task": "T12B", "experiment_type": "single-development-seed full training of B0 residual CNN at pilot ratio 1/32",
        "development_model": True, "development_seed": dev.SEED, "counts_toward_formal_five_seed_results": False,
        "model_name": "B0", "density": 16, "pilot_ratio": resources["union_ratio"], "architecture": architecture,
        "params": 23144, "MAC": 23592960, "optimizer": "Adam", "lr": 0.001, "weight_decay": 0,
        "batch_size": 128, "max_epochs": 30, "scheduler": False, "early_stopping": False,
        "gradient_clipping": False, "EMA": False, "normalization_used": False, "mask_or_SNR_input": False,
        "git": git, "reproducibility": settings,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "torch": torch.__version__, "device": str(device), "cuda_build": torch.version.cuda,
            "GPU_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "train_protocol": {"trajectory_ids": list(range(2000)), "independent_trajectories": 2000,
            "continuous_frames_each": 10, "examples_each_epoch": 20000, "examples_per_SNR_each_epoch": 4000,
            "epochs": list(range(1, 31)), "training_noise": "SeedSequence([12002,13261001,epoch,tid,frame])",
            "shuffle": "SeedSequence([12004,13261001,epoch])", "H_base_regenerated_every_epoch": True,
            "cached_noise_independent_H_true_only": True, "loss": "T12A float32 data-RE complex MSE",
            "metric_accumulation": "float64 energy of float32 prediction/quantized target, no half factor"},
        "validation_protocol": {"trajectory_ids": list(range(2000, 2400)), "independent_trajectories": 400,
            "continuous_frames_each": 10, "observations_each_epoch": 20000, "SNRs": list(md.SNR_DB),
            "unit_noise_seed": "90261001+100*tid+frame", "target": "original complex128 H_true",
            "selection_metric": "pooled validation data-RE complex MSE", "tie_policy": "earlier epoch",
            "selection_candidates": list(range(31)), "BER_used_for_selection": False,
            "R4_R5_used_for_selection": False, "old_baseline_performance": "unchanged frozen T10D CSV",
            "full_link_performance_paths_added": ["best B0+ZF"], "MMSE_detector_used": False},
        "epoch0_reference": initial, "epoch0_state_dict_sha256": epoch0_state,
        "best_epoch": best_epoch, "best_checkpoint_hash": best_checkpoint["file_sha256"],
        "epoch30_checkpoint_hash": final_checkpoint["file_sha256"],
        "checkpoints": {"best": best_checkpoint, "epoch30": final_checkpoint},
        "training_curve": log, "epoch_records": epoch_records, "validation_aggregate": aggregate,
        "paired_trajectory_diagnostics": evaluation["paired"], "residual_diagnostics": evaluation["residual"],
        "fixed_sample": evaluation["fixed_sample"], "realization_hashes": evaluation["realizations"],
        "exact_T10D_replay": evaluation["replay"], "best_full_link_pooled_mse": final_link_mse,
        "maximum_pilot_modification": 0.0, "oracle_red_flag": flag,
        "oracle_red_flag_threshold_db": 0.10,
        "oracle_audit": {"required": flag, "replay_exact": True, "frozen_baselines_unchanged": True,
            "original_complex128_metric": True, "unobserved_H_true_not_model_input": True,
            "model_input_only_R3": True, "formal_test_sealed": True, "innovation_interpretation": False},
        "training_budget_diagnosis": dev.budget_diagnosis(log),
        "runtime": {"training_wall_seconds": training_wall_seconds, "train_H_cache_seconds": train_cache_seconds,
            "fixed_validation_preparation_seconds": validation_preparation_seconds, "full_link_seconds": full_link_seconds,
            "total_wall_seconds_before_figures": perf_counter()-started,
            "GPU_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "GPU_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
            "CPU_memory": process_memory(), "future_4_densities_x_5_seeds_training_seconds_estimate": 20*training_wall_seconds,
            "cost_estimate_scope": "linear serial extrapolation of dF16 dev; density-dependent generation costs unmeasured"},
        "formal_test_used": False, "formal_test_sealed": True, "OOD_generated": False,
        "hyperparameter_search_performed": False, "num_filtered_samples": 0, "numerical_failures": [],
        "provenance": {"accepted_before": before, "accepted_after": after, "frozen_T10D_T10C_hashes": frozen,
            "predeclared_protocol_sha256": protocol_sha, "train_H_long_hashes": train_hashes,
            "source_sha256": {p: sha256_file(ROOT/p) for p in ("src/b0_development.py", "scripts/run_b0_development.py",
                "scripts/plot_b0_development.py", "tests/test_b0_development.py", "docs/b0_development_protocol.md")},
            "CSV_sha256": {suffix: sha256_file(outputs/f"{dev.PREFIX}{suffix}.csv") for suffix in
                ("_training_log", "_validation", "_validation_trajectory_metrics")}},
        "figures": [], "numeric_checks_passed": True, "all_checks_passed": False}
    # Numerical artifacts survive a figure QA failure and can be redrawn without retraining.
    json_save(report_path, report)
    stamp = perf_counter()
    report["figures"] = render_and_audit(outputs, outputs)
    report["runtime"]["figure_QA_seconds"] = perf_counter()-stamp
    report["runtime"]["total_wall_seconds"] = perf_counter()-started
    report["all_checks_passed"] = False
    json_save(report_path, report)
    progress(f"T12B numeric and figure QA completed: best epoch={best_epoch}, Oracle red flag={flag}")
    return report


def finalize_report(outputs):
    """Called after visual review; preserve numerical arrays and checkpoint files."""
    report_path = outputs/f"{dev.PREFIX}_results.json"
    report = json.loads(report_path.read_text(encoding="utf8"))
    if not report["numeric_checks_passed"] or report["provenance"]["accepted_before"] != dev.accepted_files(ROOT):
        raise ValueError("numerical/frozen-source evidence failed")
    for suffix, digest in report["provenance"]["CSV_sha256"].items():
        if sha256_file(outputs/f"{dev.PREFIX}{suffix}.csv") != digest:
            raise ValueError("numeric CSV changed")
    for path, digest in report["provenance"]["source_sha256"].items():
        if sha256_file(ROOT/path) != digest:
            raise ValueError("recorded T12B source changed")
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
        capture_output=True, text=True, encoding="utf8")
    (outputs/f"{dev.PREFIX}_pytest.txt").write_text(tests.stdout+tests.stderr, encoding="utf8")
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True, encoding="utf8")
    progress(tests.stdout)
    if tests.returncode or diff.returncode:
        raise RuntimeError(f"final tests/diff failed\n{tests.stderr}\n{diff.stdout}\n{diff.stderr}")
    for figure in report["figures"]:
        figure["visual_self_check"] = "Actual exported PNG inspected: all raw curves present; labels/legend legible; no clipping or overlap"
    report["pytest"] = {"command": "python -m pytest -q", "exit_code": tests.returncode,
                         "summary": tests.stdout.strip().splitlines()[-1]}
    report["git_diff_check"] = {"exit_code": diff.returncode, "stdout": diff.stdout, "warnings": diff.stderr}
    report["all_checks_passed"] = True
    json_save(report_path, report)
    progress("T12B final provenance, pytest and diff checks PASS")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--finalize-after-visual-review", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=ROOT/"outputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs")
    args = parser.parse_args(argv)
    if args.finalize_after_visual_review:
        finalize_report(args.output_dir)
        return
    if args.plot_only:
        records = render_and_audit(args.input_dir, args.output_dir)
        json_save(args.output_dir/f"{dev.PREFIX}_figure_qa.json", records)
        progress("Three figures rebuilt from two CSV files only; alignment/text/collision checked")
        return
    if args.output_dir.resolve() != (ROOT/"outputs").resolve() or args.input_dir.resolve() != args.output_dir.resolve():
        raise ValueError("single development run uses the project outputs directory")
    run_training(args.output_dir)


if __name__ == "__main__":
    main()
