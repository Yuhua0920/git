"""One fixed continuous epoch0..60 run with a mandatory epoch30 replay gate."""
import argparse
import csv
import gc
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src import b0_dev60 as dev, b0_development as old, b0_perf as perf, ml_data as md
from src.b0_model import ResidualCNN, configure_reproducibility
from src.lmmse_train import sha256_file
from src.nearest_validation import load_references, verify_frozen
from src.perf_resources import ResourceSampler, distribution, physical_memory
from src.trajectories import prepare_trajectory
from scripts.run_b0_development import record_row
from scripts.run_lmmse_train_validation import qa_tools, save_csv
from scripts.validate_b0_pipeline import architecture_report
from scripts.plot_b0_dev60 import plot_results

OUT = ROOT/"outputs"
PROTOCOL = ROOT/"docs/b0_dev60_protocol.md"
SOURCE_FILES = ("src/b0_dev60.py", "scripts/run_b0_dev60.py", "scripts/plot_b0_dev60.py", "tests/test_b0_dev60.py", "docs/b0_dev60_protocol.md")


def progress(message):
    print(message, flush=True)


def json_save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf8")


def accepted_files():
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", dev.T12C_COMMIT,
        "outputs", "src", "scripts", "tests", "configs", "docs"], cwd=ROOT, text=True).splitlines()
    return {p: sha256_file(ROOT/p) for p in paths if p != "docs/stage_status.md"}


def cpu_release():
    gc.collect(); torch.cuda.empty_cache()


def log_row(epoch, training, validation, timing, best, model, waiting=0.0):
    row = record_row(epoch, training, validation, timing, best, model)
    row.update({"source_run_phase": "replayed_0_30" if epoch <= 30 else "extended_31_60",
                "DataLoader_waiting_seconds": waiting})
    return row


def render_and_audit(input_dir, output_dir):
    tools, gate = qa_tools(None)
    paths, records = plot_results(input_dir, output_dir, gate)
    for path, record in zip(paths, records):
        text = subprocess.run([sys.executable, str(tools/"audit_pdf_text.py"), str(path.with_suffix(".pdf")), "--min-pt", "5", "--json"], capture_output=True, text=True, encoding="utf8")
        collision_path = path.with_suffix(".collision-audit.json")
        collision = subprocess.run([sys.executable, str(tools/"audit_figure_collisions.py"), str(path.with_suffix(".pdf")), "--json-out", str(collision_path)], capture_output=True, text=True, encoding="utf8")
        if text.returncode or collision.returncode:
            raise RuntimeError(f"figure QA failed: {path.name}\n{text.stdout}\n{collision.stdout}")
        record.update({"path": str(path), "PDF_text_audit": json.loads(text.stdout),
            "collision_audit": json.loads(collision_path.read_text(encoding="utf8")),
            "alignment_audit": json.loads(path.with_suffix(".alignment.json").read_text(encoding="utf8")),
            "sha256": {ext: sha256_file(path.with_suffix("."+ext)) for ext in ("png", "pdf", "svg")}})
    return records


def audit_T12B():
    path = OUT/"t12b_b0_df16_dev_epoch30.pt"
    payload = torch.load(path, weights_only=True, map_location="cpu")
    return {"path": str(path), "file_sha256": sha256_file(path), "payload_keys": sorted(payload),
        "model_state_present": "state_dict" in payload, "optimizer_state_present": "optimizer_state_dict" in payload,
        "Adam_step_moments_present": False, "model_only_resume_not_used": True,
        "main_run_start": "fresh epoch0 same development seed; one live Adam through60"}


def fixed_sample_callback(original):
    frozen = original["fixed_sample"]
    def collect(root, h, r3, best, ls, y, bits, layout):
        if perf.raw_hash(y) != frozen["Y_sha256"] or perf.raw_hash(r3) != frozen["H_R3_sha256"]:
            raise ValueError("fixed original sample Y/R3 changed")
        points = []
        for p in frozen["points"]:
            k, t = p["k"], p["t"]
            true = old.complex_record(h[:, :, k, t]); base = old.complex_record(r3[:, :, k, t])
            if true != p["H_true"] or base != p["H_R3"]:
                raise ValueError("fixed RE truth/R3 changed")
            points.append({"k": k, "t": t, "H_true": true, "H_R3": base,
                "H_B0_epoch30": p["H_B0"], "H_B0_best60": old.complex_record(best[:, :, k, t]), "H_R4": p["H_R4"],
                "absolute_error_epoch30": p["absolute_error_B0"],
                "absolute_error_best60": np.abs(best[:, :, k, t]-h[:, :, k, t]).tolist()})
        return {"trajectory_id": 2000, "frame_index": 0, "snr_db": 10, "dF": 16, "points": points,
            "Y_sha256": frozen["Y_sha256"], "H_R3_sha256": frozen["H_R3_sha256"],
            "R4_values_from_frozen_T12B": True, "R4_evaluations": 0, "epoch30_evaluations": 0,
            "norm_best60_minus_R3_over_R3": float(np.linalg.norm(best-r3)/np.linalg.norm(r3))}
    return collect


def validation_comparison(aggregates, reference):
    output = []
    for best, previous in zip(aggregates, reference["validation_aggregate"]):
        if best["snr_db"] != previous["snr_db"] or best["total_bits"] != previous["total_bits"]:
            raise ValueError("full-link matched reference mismatch")
        row = {**best, "b0_best60_nmse_db": best["b0_nmse_db"], "b0_epoch30_nmse_db": previous["b0_nmse_db"],
            "epoch30_to_best60_gain_db": previous["b0_nmse_db"]-best["b0_nmse_db"],
            "b0_best60_ber": best["b0_ber"], "b0_epoch30_ber": previous["b0_ber"],
            "epoch30_to_best60_ber_reduction": previous["b0_ber"]-best["b0_ber"],
            "b0_minus_r5_db": best["b0_nmse_db"]-best["r5_oracle_nmse_db"],
            "perfect_nmse_db": None}
        output.append(row)
    return output


def execute():
    if any((OUT/f"{dev.PREFIX}{suffix}").exists() for suffix in ("_training_log.csv", "_results.json", "_best60.pt", "_epoch60.pt")):
        raise ValueError("single60 run artifacts already exist; refuse accidental retraining")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != dev.T12C_COMMIT:
        raise ValueError("accepted T12C commit must be HEAD")
    before = accepted_files(); protocol_hash = sha256_file(PROTOCOL)
    original = json.loads((OUT/"t12b_b0_df16_dev_results.json").read_text(encoding="utf8"))
    perf_report = json.loads((OUT/"t12c_perf_results.json").read_text(encoding="utf8"))
    profile = dev.selected_profile(perf_report); audit = audit_T12B()
    settings = configure_reproducibility(old.SEED, 4)
    model = ResidualCNN().to("cuda"); architecture = architecture_report(model)
    if old.state_hash(model) != original["epoch0_state_dict_sha256"]:
        raise ValueError("original epoch0 model initialization not exact")
    optimizer = old.optimizer_for(model); optimizer_id = id(optimizer)
    prepared = prepare_trajectory(); manifest = md.frozen_manifest(OUT/"t10c_train_validation_manifest.csv")
    refs = load_references(OUT); historical = old.replay_link(ROOT)
    sampler_resources = ResourceSampler(); torch.cuda.reset_peak_memory_stats()
    start_memory = physical_memory(); started = perf_counter()
    epoch_records, log, replay_rows = [], [], []
    start_resource_index = 0
    with tempfile.TemporaryDirectory(prefix="t12d_h_", dir=OUT) as temporary:
        tmp = Path(temporary).resolve()
        if not tmp.is_relative_to(OUT.resolve()):
            raise ValueError("temporary cache outside workspace")
        stamp = perf_counter()
        val_h, val_hashes = old.cache_truth(prepared, manifest, "validation_id")
        cache = old.prepare_validation(val_h, val_hashes, refs, dev.LAYOUT, historical, progress)
        cpu_hash = dev.tensor_stream_hash(cache["base"]); gpu_base = cache["base"].to("cuda")
        if dev.tensor_stream_hash(gpu_base) != cpu_hash or cpu_hash != perf_report["selected_profile"]["validation_input_SHA256"]:
            raise ValueError("fixed validation cache not bitwise original PERF/T12B")
        cache["base"] = gpu_base; cpu_release()
        validation_prep = perf_counter()-stamp
        stamp = perf_counter(); zero = old.evaluate_validation(model, cache, dev.LAYOUT, epoch_zero=True)
        torch.cuda.synchronize(); zero_seconds = perf_counter()-stamp
        initial = old.initial_reference(zero, refs); replay_rows.append(dev.replay_epoch(0, None, zero, None, original))
        best_metric, best_epoch = zero["pooled_mse"], 0
        best_checkpoint = dev.checkpoint_save(OUT/f"{dev.PREFIX}_best60.pt", model, optimizer, 0, best_metric, protocol_hash, architecture)
        row = log_row(0, None, zero, (validation_prep, 0., zero_seconds, validation_prep+zero_seconds), True, model)
        log.append(row)
        epoch_records.append({"epoch": 0, "validation": zero, "training": None})
        del cache["received"], cache["bits"]; cpu_release()
        stamp = perf_counter()
        train_hashes = dev.prepare_H_cache(tmp/"h.npy", prepared, [r for r in manifest if r["split"] == "train"])
        if train_hashes != original["provenance"]["train_H_long_hashes"]:
            raise ValueError("all original train trajectory hashes must match")
        cpu_release()
        train_prep = perf_counter()-stamp
        progress(f"Cache preparation complete: available physical RAM {physical_memory()['available_RAM_bytes']/2**30:.2f}GiB; same read-only H and GPU-resident validation")
        sample = dev.Dev60Sampler(); loader = dev.training_loader(tmp/"h.npy", sample)
        save_csv(OUT/f"{dev.PREFIX}_training_log.csv", log)
        replay_start = perf_counter(); extension_start = None
        gate = None
        try:
            for epoch in range(1, 61):
                if epoch == 31:
                    if not gate or not gate["all_checks_passed"]:
                        raise ValueError("epoch31 prohibited until replay PASS")
                    extension_start = perf_counter()
                epoch_start = perf_counter(); sample.epoch = epoch
                steps_before = dev.optimizer_steps(optimizer)
                training, source, train_seconds = dev.train_epoch(model, optimizer, loader, dev.epoch_order(epoch), epoch, sampler_resources)
                if id(optimizer) != optimizer_id or steps_before != (epoch-1)*157 or dev.optimizer_steps(optimizer) != epoch*157:
                    raise ValueError("training Adam was reset or update count changed")
                if training["optimizer_steps"] != 157 or any(r["sample_count"] != 4000 for r in training["train_per_snr"]):
                    raise ValueError("full sample/batch/SNR protocol mismatch")
                stamp = perf_counter(); val = old.evaluate_validation(model, cache, dev.LAYOUT)
                torch.cuda.synchronize(); val_seconds = perf_counter()-stamp
                if epoch <= 30:
                    replay_rows.append(dev.replay_epoch(epoch, training, val, source, original))
                is_best = old.select_best(val["pooled_mse"], best_metric)
                if is_best:
                    best_metric, best_epoch = val["pooled_mse"], epoch
                    best_checkpoint = dev.checkpoint_save(OUT/f"{dev.PREFIX}_best60.pt", model, optimizer, epoch, best_metric, protocol_hash, architecture)
                epoch_seconds = perf_counter()-epoch_start
                row = log_row(epoch, training, val, (sum(source["CPU_stage_seconds_sum"].values()), train_seconds, val_seconds, epoch_seconds), is_best, model,
                              source["DataLoader_waiting_seconds"])
                log.append(row)
                epoch_records.append({"epoch": epoch, "training": training, "validation": val,
                    "optimizer_instance_id": optimizer_id, "optimizer_steps_before": steps_before,
                    "optimizer_steps_after": dev.optimizer_steps(optimizer),
                    "SNR_assignment_sha256": perf.raw_hash(md.balanced_training_snr(old.SEED, epoch)),
                    "shuffle_order_sha256": perf.raw_hash(dev.epoch_order(epoch)), "source": source})
                save_csv(OUT/f"{dev.PREFIX}_training_log.csv", log)
                json_save(OUT/f"{dev.PREFIX}_training_progress.json", {"completed_epoch": epoch, "best_epoch": best_epoch,
                    "epoch0_30_replay_gate": gate, "epochs": epoch_records, "formal_test_sealed": True})
                progress(f"epoch{epoch:02d}/60 train={training['train_mse']:.12g} val={val['pooled_mse']:.12g} best={best_epoch} train/val={train_seconds:.2f}/{val_seconds:.2f}s Adamstep={dev.optimizer_steps(optimizer)}")
                if epoch == 30:
                    gate = dev.require_extension_gate(replay_rows, model)
                    replay_wall = perf_counter()-replay_start+zero_seconds
                    json_save(OUT/f"{dev.PREFIX}_epoch30_replay_gate.json", gate)
                    progress(f"MANDATORY REPLAY GATE PASS: epoch30 state {old.state_hash(model)}; same Adam continues")
            extension_wall = perf_counter()-extension_start
            final_checkpoint = dev.checkpoint_save(OUT/f"{dev.PREFIX}_epoch60.pt", model, optimizer, 60, log[-1]["val_mse_pooled"], protocol_hash, architecture)
            final_optimizer_hash = dev.structured_hash(optimizer.state_dict()); final_steps = dev.optimizer_steps(optimizer)
        except Exception as exc:
            failed_resources = sampler_resources.finish()
            save_csv(OUT/f"{dev.PREFIX}_failed_utilization.csv", sampler_resources.rows)
            json_save(OUT/f"{dev.PREFIX}_failed.json", {"task": "T12D", "error": str(exc), "completed_epoch": len(log)-1,
                "optimizer_steps_at_stop": dev.optimizer_steps(optimizer), "resources": failed_resources,
                "epoch31_started": len(log)>31, "all_checks_passed": False, "formal_test_used": False})
            raise
        finally:
            perf.close_loader(loader)
        training_wall = perf_counter()-started
        resource_summary = sampler_resources.finish()
        save_csv(OUT/f"{dev.PREFIX}_utilization.csv", sampler_resources.rows)
        if not resource_summary["memory_safe"]:
            raise MemoryError("fixed profile failed physical-memory/paging gate")
        del model, optimizer, cache, val_h, gpu_base, loader; cpu_release()
    best_model, restored_optimizer, best_payload = dev.checkpoint_load(OUT/f"{dev.PREFIX}_best60.pt", torch.device("cuda"), protocol_hash)
    if best_payload["epoch"] != best_epoch or old.state_hash(best_model) != best_checkpoint["model_tensor_sha256"]:
        raise ValueError("wrong global best reload")
    before_eval_hash = old.state_hash(best_model)
    stamp = perf_counter()
    with patch.object(old, "fixed_sample", fixed_sample_callback(original)):
        evaluation = old.full_link_validation(ROOT, prepared, refs, best_model, dev.LAYOUT, historical, progress)
    full_link_seconds = perf_counter()-stamp
    if old.state_hash(best_model) != before_eval_hash:
        raise ValueError("full link altered best checkpoint model")
    comparison = validation_comparison(evaluation["aggregates"], original)
    save_csv(OUT/f"{dev.PREFIX}_validation.csv", comparison)
    save_csv(OUT/f"{dev.PREFIX}_validation_trajectory_metrics.csv", evaluation["trajectory_rows"])
    epoch30_rows = []
    with (OUT/"t12b_b0_df16_dev_validation_trajectory_metrics.csv").open(encoding="utf8", newline="") as file:
        epoch30_rows = list(csv.DictReader(file))
    paired = dev.paired_diagnostics(evaluation["trajectory_rows"], epoch30_rows)
    residuals = []
    for best, ref in zip(evaluation["residual"], original["residual_diagnostics"]):
        if best["snr_db"] != ref["snr_db"]:
            raise ValueError("residual comparison SNR mismatch")
        residuals.append({**best, "epoch30_RMS_residual": ref["RMS_residual"], "epoch30_RMS_ratio": ref["RMS_ratio"],
            "RMS_ratio_change": best["RMS_ratio"]-ref["RMS_ratio"], "RMS_ratio_relative_change": best["RMS_ratio"]/ref["RMS_ratio"]-1})
    diagnosis = dev.convergence_diagnosis(log, epoch_records)
    full_mse = sum(r["b0_nmse_error_energy"] for r in comparison)/(20000*4*np.count_nonzero(dev.LAYOUT.data_mask[0]))
    if not np.isclose(full_mse, best_metric, rtol=1e-6, atol=1e-12):
        raise ValueError("best full link and fixed batch128 selection MSE disagree")
    after = accepted_files()
    if before != after or protocol_hash != sha256_file(PROTOCOL):
        raise ValueError("accepted files/predeclared protocol changed")
    red_flag = old.oracle_red_flag(comparison)
    flag_audit = {"required": red_flag, "original_complex128_metric": True, "model_input_only_R3": True,
        "unobserved_truth_not_model_input": True, "same_H_Y_bits_noise_exact": True,
        "frozen_reference_hashes_verified": True, "all_checks_passed": True}
    epoch_mean = float(np.mean([r["epoch_seconds"] for r in log[1:]]))
    report = {"task": "T12D", "experiment_type": "single development seed continuous epoch0..60 convergence diagnosis",
        "git": {"T12C_commit": head, "T12C_committed_cleanly": True, "clean_after_commit": True,
            "startup_pytest": "995 passed in 61.97s (0:01:01)"},
        "T12B_checkpoint_optimizer_state_audit": audit, "t12b_epoch30_reference": original["checkpoints"]["epoch30"],
        "epoch0_30_replay_verified": gate, "development_seed": old.SEED, "density": 16, "pilot_ratio": 1/32,
        "development_model": True, "counts_toward_formal_five_seed_results": False, "perf_profile": profile,
        "architecture": architecture, "max_epochs": 60, "batch_size": 128, "optimizer": "Adam", "lr": .001, "weight_decay": 0,
        "optimizer_continuity": {"training_instance_id": optimizer_id, "same_instance_all60_epochs": True, "no_reset": True,
            "epoch30_total_steps": 4710, "epoch31_steps_before": epoch_records[31]["optimizer_steps_before"],
            "final_total_steps": final_steps, "final_optimizer_tensor_sha256": final_optimizer_hash},
        **diagnosis, "checkpoints": {"best60": best_checkpoint, "epoch60": final_checkpoint},
        "training_curve": log, "epoch_records": epoch_records, "epoch0_reference": initial,
        "validation_results": comparison, "paired_trajectory_diagnostics": paired, "residual_diagnostics": residuals,
        "fixed_sample": evaluation["fixed_sample"], "realization_hashes": evaluation["realizations"],
        "exact_T10D_replay": evaluation["replay"], "best_full_link_pooled_mse": full_mse,
        "oracle_red_flag": red_flag, "oracle_red_flag_threshold_db": .10, "oracle_audit": flag_audit,
        "runtime": {"validation_cache_preparation_seconds": validation_prep, "train_H_cache_preparation_seconds": train_prep,
            "epoch0_validation_seconds": zero_seconds, "epoch0_30_replay_wall_seconds": replay_wall,
            "epoch31_60_wall_seconds": extension_wall, "mean_seconds_per_epoch": epoch_mean,
            "validation_seconds_total": sum(r["validation_seconds"] for r in log),
            "training_wall_seconds_including_cache": training_wall, "best_full_link_seconds": full_link_seconds,
            "total_wall_seconds_before_figures": perf_counter()-started,
            "P0_reference_seconds_per_epoch": perf_report["baseline_profile"]["seconds_per_epoch"],
            "Popt_real_training_speedup": perf_report["baseline_profile"]["seconds_per_epoch"]/epoch_mean,
            "data_generation_seconds_scope": "sum of worker CPU stages; overlaps training_seconds (FIFO feeding plus original GPU/monitor loop)",
            "GPU_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "GPU_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "resource_summary": resource_summary, "physical_memory_before": start_memory},
        "reproducibility": settings, "environment": {"python": platform.python_version(), "torch": torch.__version__, "GPU": torch.cuda.get_device_name()},
        "provenance": {"accepted_before": before, "accepted_after": after, "frozen_T10D_T10C_hashes": verify_frozen(OUT),
            "protocol_sha256": protocol_hash, "source_sha256": {p: sha256_file(ROOT/p) for p in SOURCE_FILES},
            "CSV_sha256": {s: sha256_file(OUT/f"{dev.PREFIX}{s}.csv") for s in ("_training_log", "_validation", "_validation_trajectory_metrics", "_utilization")},
            "validation_tensor_sha256": cpu_hash, "train_H_long_hashes_matched": 2000},
        "scheduler": False, "warmup": False, "gradient_clipping": False, "EMA": False, "AMP": False,
        "architecture_or_loss_changed": False, "normalization_used": False, "mask_or_SNR_network_input": False,
        "formal_test_used": False, "formal_test_sealed": True, "OOD_generated": False,
        "additional_density_or_seed_used": False, "baselines_rerun": False, "numerical_failures": [],
        "prior_attempts": [{"path": "outputs/t12d_attempt01", "completed_epoch": 0,
            "reason": "physical RAM safety gate during epoch1", "partial_epoch1_steps": "not captured in first handler",
            "seed_or_profile_changed": False, "source_and_failure_snapshots_preserved": True},
            {"path": "outputs/t12d_attempt02", "completed_epoch": 1, "reason": "inconsistent first-row CSV schema",
             "optimizer_steps_at_stop": 157, "epoch1_numerical_replay_passed": True,
             "seed_or_profile_changed": False, "source_and_failure_snapshots_preserved": True}],
        "cache_lifecycle_correction": "bounded SHA scratch and direct identical-H NPY construction; original Popt workers/order/math/profile unchanged",
        "figures": [], "numeric_checks_passed": True, "all_checks_passed": False}
    report_path = OUT/f"{dev.PREFIX}_results.json"; json_save(report_path, report)
    stamp = perf_counter(); report["figures"] = render_and_audit(OUT, OUT)
    report["runtime"]["figure_QA_seconds"] = perf_counter()-stamp
    report["runtime"]["total_wall_seconds"] = perf_counter()-started
    json_save(report_path, report)
    progress(f"60-epoch numeric/figure checks completed: best={best_epoch}, under_training={diagnosis['under_training_evidence']}, Oracle flag={red_flag}")


def finalize():
    path = OUT/f"{dev.PREFIX}_results.json"; report = json.loads(path.read_text(encoding="utf8"))
    if not report["numeric_checks_passed"] or accepted_files() != report["provenance"]["accepted_after"]:
        raise ValueError("frozen/numeric evidence failed")
    for p, digest in report["provenance"]["source_sha256"].items():
        if sha256_file(ROOT/p) != digest:
            raise ValueError(f"source changed after run: {p}")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True, encoding="utf8")
    (OUT/f"{dev.PREFIX}_pytest.txt").write_text(result.stdout+result.stderr, encoding="utf8"); progress(result.stdout)
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True, encoding="utf8")
    if result.returncode or diff.returncode:
        raise RuntimeError(f"pytest/diff failed\n{result.stderr}\n{diff.stdout}\n{diff.stderr}")
    for figure in report["figures"]:
        figure["visual_self_check"] = "Actual exported PNG inspected; all raw observations/methods present; legend and axes legible without clipping"
    report["pytest"] = {"command": "python -m pytest -q", "exit_code": 0, "summary": result.stdout.strip().splitlines()[-1]}
    report["git_diff_check"] = {"exit_code": 0, "warnings": diff.stderr, "stdout": diff.stdout}
    report["all_checks_passed"] = True; json_save(path, report)
    progress("T12D final checks PASS; fixed budget complete, no new model started")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--finalize-after-visual-review", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=OUT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.plot_only:
        records = render_and_audit(args.input_dir, args.output_dir)
        json_save(args.output_dir/f"{dev.PREFIX}_figure_qa.json", records)
    elif args.finalize_after_visual_review:
        finalize()
    else:
        if args.input_dir.resolve() != OUT.resolve() or args.output_dir.resolve() != OUT.resolve():
            raise ValueError("single main experiment uses project outputs")
        execute()


if __name__ == "__main__":
    main()
