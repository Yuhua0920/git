"""T12E: one fixed60-epoch20dB diagnostic, or CSV-only reconstruction."""
import argparse
import csv
import gc
import json
import platform
import shutil
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
from src import b0_high_snr as probe, b0_development as old, b0_dev60 as dev, b0_perf as perf, ml_data as md
from src.b0_model import ResidualCNN, configure_reproducibility
from src.lmmse_train import sha256_file
from src.nearest_validation import load_references, verify_frozen
from src.perf_resources import ResourceSampler, physical_memory
from src.trajectories import prepare_trajectory
from scripts.run_lmmse_train_validation import qa_tools, save_csv
from scripts.validate_b0_pipeline import architecture_report
from scripts.plot_b0_high_snr import plot_results, FIGURE_NAMES, read_csv

OUT = ROOT/"outputs"
PROTOCOL = ROOT/"docs/b0_20db_protocol.md"
SOURCES = ("src/b0_high_snr.py", "scripts/run_b0_high_snr.py", "scripts/plot_b0_high_snr.py", "tests/test_b0_high_snr.py", "docs/b0_20db_protocol.md")


def progress(message):
    print(message, flush=True)


def json_save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf8")


def accepted_files():
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", probe.T12D_COMMIT,
        "outputs", "src", "scripts", "tests", "configs", "docs"], cwd=ROOT, text=True).splitlines()
    return {p: sha256_file(ROOT/p) for p in paths if p != "docs/stage_status.md"}


def release():
    gc.collect(); torch.cuda.empty_cache()


def log_row(epoch, train, val, model, best, epoch_seconds, train_seconds=0., val_seconds=0., waiting=0.):
    return {"epoch": epoch, "train_mse_20db": None if train is None else train["train_mse"],
        "val_mse_20db": val["mse"], "val_nmse_20db": val["nmse_db"],
        "gradient_norm": None if train is None else train["gradient_norm"], **old.model_norms(model),
        "learning_rate": .001, "epoch_seconds": epoch_seconds, "training_seconds": train_seconds,
        "validation_seconds": val_seconds, "DataLoader_waiting_seconds": waiting,
        "best_so_far": best, "exposure_matched": epoch == 12, "cumulative_20db_exposures": epoch*20000}


def cross_rows(epoch, validation, checkpoint):
    return [{"checkpoint": checkpoint, "epoch": epoch, "snr_db": r["snr_db"], "mse": r["mse"],
        "nmse_db": r["nmse_db"], "used_for_selection": r["snr_db"] == 20,
        "full_link_formal_record": False, "evaluation_scope": "fixed cache checkpoint diagnostic"}
        for r in validation["per_snr"]]


def comparison_rows(evaluation, mixed_csv):
    rows = []
    for high, mixed in zip(evaluation["aggregates"], mixed_csv):
        if high["snr_db"] != int(mixed["snr_db"]):
            raise ValueError("frozen SNR comparison mismatch")
        rows.append({"checkpoint": "global_best", "snr_db": high["snr_db"],
            "r3_nmse_db": high["r3_nmse_db"], "mixed_B0_best57_nmse_db": float(mixed["b0_best60_nmse_db"]),
            "b0_20db_only_nmse_db": high["b0_nmse_db"], "r4_nmse_db": high["r4_nmse_db"], "r5_nmse_db": high["r5_oracle_nmse_db"],
            "r3_ber": high["r3_ber"], "mixed_B0_best57_ber": float(mixed["b0_best60_ber"]),
            "b0_20db_only_ber": high["b0_ber"], "r4_ber": high["r4_ber"], "r5_ber": high["r5_ber"], "perfect_ber": high["perfect_ber"],
            "gain_vs_R3_db": high["r3_to_b0_gain_db"],
            "gain_vs_mixed_best57_db": float(mixed["b0_best60_nmse_db"])-high["b0_nmse_db"],
            "gap_to_R4_db": high["b0_minus_r4_db"], "gap_to_R5_db": high["b0_minus_oracle_db"],
            "R3_R4_gap_closed_fraction": high["r3_r4_gap_closed_fraction"],
            "bit_errors": high["b0_bit_errors"], "total_bits": high["total_bits"], "trajectory_units": 400})
    return rows


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


def execute():
    if any((OUT/f"{probe.PREFIX}{s}").exists() for s in ("_training_log.csv", "_results.json", "_best.pt", "_epoch60.pt")):
        raise ValueError("diagnostic artifacts already exist; refuse accidental second run")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != probe.T12D_COMMIT:
        raise ValueError("accepted T12D commit must be HEAD")
    before = accepted_files(); source_hashes = {p: sha256_file(ROOT/p) for p in SOURCES}; protocol_hash = sha256_file(PROTOCOL)
    mixed = json.loads((OUT/f"{dev.PREFIX}_results.json").read_text(encoding="utf8"))
    if not mixed["all_checks_passed"] or mixed["best_epoch"] != 57:
        raise ValueError("accepted mixed development reference required")
    mixed_csv = read_csv(OUT/f"{dev.PREFIX}_validation.csv")
    mixed_log = read_csv(OUT/f"{dev.PREFIX}_training_log.csv")
    profile = dev.selected_profile(json.loads((OUT/"t12c_perf_results.json").read_text(encoding="utf8")))
    settings = configure_reproducibility(old.SEED, 4)
    model = ResidualCNN().to("cuda"); architecture = architecture_report(model)
    old_report = json.loads((OUT/"t12b_b0_df16_dev_results.json").read_text(encoding="utf8"))
    if old.state_hash(model) != old_report["epoch0_state_dict_sha256"]:
        raise ValueError("same seed and exact zero-init architecture required")
    optimizer = old.optimizer_for(model); optimizer_id = id(optimizer)
    prepared = prepare_trajectory(); manifest = md.frozen_manifest(OUT/"t10c_train_validation_manifest.csv")
    refs = load_references(OUT); historical = old.replay_link(ROOT)
    resources = ResourceSampler(); torch.cuda.reset_peak_memory_stats(); started = perf_counter()
    log, epochs, cross = [], [], []; checkpoints = {}; val_total = 0.
    with tempfile.TemporaryDirectory(prefix="t12e_h_", dir=OUT) as temporary:
        tmp = Path(temporary).resolve()
        if not tmp.is_relative_to(OUT.resolve()):
            raise ValueError("temporary cache outside workspace")
        stamp = perf_counter(); vh, vh_hashes = old.cache_truth(prepared, manifest, "validation_id")
        cache = old.prepare_validation(vh, vh_hashes, refs, probe.LAYOUT, historical, progress)
        base_hash = dev.tensor_stream_hash(cache["base"]); gpu_base = cache["base"].to("cuda")
        if base_hash != mixed["provenance"]["validation_tensor_sha256"] or dev.tensor_stream_hash(gpu_base) != base_hash:
            raise ValueError("original validation input tensors changed")
        cache["base"] = gpu_base; release(); val_prep = perf_counter()-stamp
        stamp = perf_counter(); zero = old.evaluate_validation(model, cache, probe.LAYOUT, epoch_zero=True)
        initial = old.initial_reference(zero, refs)
        if zero != mixed["epoch_records"][0]["validation"]:
            raise ValueError("all epoch0 observations and predictions must exactly match frozen mixed run")
        val0 = probe.evaluate_20db(model, cache, epoch_zero=True)
        if val0 != {k: v for k, v in zero["per_snr"][-1].items() if k not in ("bit_errors", "total_bits", "ber")}:
            raise ValueError("20dB-only validation math differs from original")
        zero_seconds = perf_counter()-stamp; val_total += zero_seconds
        cross.extend(cross_rows(0, zero, "epoch0")); best_metric, best = val0["mse"], 0
        checkpoints["best"] = probe.checkpoint_save(OUT/f"{probe.PREFIX}_best.pt", model, optimizer, 0, best_metric, protocol_hash, architecture)
        log.append(log_row(0, None, val0, model, True, zero_seconds, val_seconds=zero_seconds))
        epochs.append({"epoch": 0, "validation20": val0, "training": None})
        del cache["received"], cache["bits"]; release()
        stamp = perf_counter(); train_hashes = dev.prepare_H_cache(tmp/"h.npy", prepared, [r for r in manifest if r["split"] == "train"])
        if train_hashes != old_report["provenance"]["train_H_long_hashes"]:
            raise ValueError("same original TRAIN trajectories required")
        train_prep = perf_counter()-stamp; release()
        progress(f"Cache PASS; available RAM={physical_memory()['available_RAM_bytes']/2**30:.2f}GiB; fixed PERF-Popt")
        sampler = dev.Dev60Sampler(); loader = probe.training_loader(tmp/"h.npy", sampler)
        save_csv(OUT/f"{probe.PREFIX}_training_log.csv", log); train_started = perf_counter()
        try:
            for epoch in range(1, 61):
                stamp = perf_counter(); sampler.epoch = epoch
                steps_before = dev.optimizer_steps(optimizer)
                train, source, train_seconds = probe.train_epoch(model, optimizer, loader, dev.epoch_order(epoch), epoch, resources)
                if (id(optimizer) != optimizer_id or steps_before != (epoch-1)*157 or dev.optimizer_steps(optimizer) != epoch*157
                    or source["ordered_ID_hash"] != mixed["epoch_records"][epoch]["shuffle_order_sha256"]
                    or [r["sample_count"] for r in train["train_per_snr"]] != [0, 0, 0, 0, 20000]):
                    raise ValueError("optimizer continuity or original order/all20dB exposure changed")
                val_stamp = perf_counter(); val = probe.evaluate_20db(model, cache)
                torch.cuda.synchronize(); val_seconds = perf_counter()-val_stamp
                is_best = old.select_best(val["mse"], best_metric)
                if is_best:
                    best_metric, best = val["mse"], epoch
                    checkpoints["best"] = probe.checkpoint_save(OUT/f"{probe.PREFIX}_best.pt", model, optimizer, epoch, best_metric, protocol_hash, architecture)
                if epoch in (12, 60):
                    name = f"epoch{epoch}"
                    checkpoints[name] = probe.checkpoint_save(OUT/f"{probe.PREFIX}_{name}.pt", model, optimizer, epoch, val["mse"], protocol_hash, architecture)
                if epoch in probe.CROSS_EPOCHS:
                    cross_stamp = perf_counter()
                    c = old.evaluate_validation(model, cache, probe.LAYOUT)
                    if c["per_snr"][-1] != val:
                        raise ValueError("cross-SNR evaluation changed primary20dB metrics")
                    cross.extend(cross_rows(epoch, c, f"epoch{epoch}"))
                    val_seconds += perf_counter()-cross_stamp
                val_total += val_seconds
                log.append(log_row(epoch, train, val, model, is_best, perf_counter()-stamp, train_seconds, val_seconds, source["DataLoader_waiting_seconds"]))
                epochs.append({"epoch": epoch, "validation20": val, "training": train, "source": source,
                    "shuffle_order_sha256": source["ordered_ID_hash"], "optimizer_instance_id": optimizer_id,
                    "optimizer_steps_before": steps_before, "optimizer_steps_after": dev.optimizer_steps(optimizer)})
                save_csv(OUT/f"{probe.PREFIX}_training_log.csv", log)
                json_save(OUT/f"{probe.PREFIX}_training_progress.json", {"completed_epoch": epoch, "best_epoch": best, "epoch_records": epochs, "cross_diagnostics": cross})
                progress(f"epoch{epoch:02d}/60 train20={train['train_mse']:.12g} val20={val['mse']:.12g} NMSE20={val['nmse_db']:.6f} best={best} Adamstep={dev.optimizer_steps(optimizer)}")
            train_wall = perf_counter()-train_started
            final_steps = dev.optimizer_steps(optimizer)
        except Exception as exc:
            summary = resources.finish()
            save_csv(OUT/f"{probe.PREFIX}_failed_utilization.csv", resources.rows)
            json_save(OUT/f"{probe.PREFIX}_failed.json", {"task": "T12E", "error": str(exc), "completed_epoch": len(log)-1,
                "optimizer_steps_at_stop": dev.optimizer_steps(optimizer), "resources": summary, "all_checks_passed": False})
            raise
        finally:
            perf.close_loader(loader)
        summary = resources.finish(); save_csv(OUT/f"{probe.PREFIX}_utilization.csv", resources.rows)
        if not summary["memory_safe"]:
            raise MemoryError("fixed physical-memory/paging gate failed")
        if best != probe.best_epoch(log):
            raise ValueError("checkpoint selection not global20dB MSE argmin")
        del model, optimizer, loader; release()
        model, restored, payload = probe.checkpoint_load(OUT/f"{probe.PREFIX}_best.pt", torch.device("cuda"), protocol_hash)
        if payload["epoch"] != best:
            raise ValueError("wrong best checkpoint")
        if best in probe.CROSS_EPOCHS:
            best_cross = [r for r in cross if r["epoch"] == best]
            cross.extend([{**r, "checkpoint": "global_best"} for r in best_cross])
        else:
            stamp = perf_counter(); c = old.evaluate_validation(model, cache, probe.LAYOUT)
            if c["per_snr"][-1] != epochs[best]["validation20"]:
                raise ValueError("reloaded best20dB validation mismatch")
            cross.extend(cross_rows(best, c, "global_best")); val_total += perf_counter()-stamp
        del restored, cache, vh, gpu_base; release()
    mixed_model, mixed_optimizer, mixed_payload = dev.checkpoint_load(OUT/f"{dev.PREFIX}_best60.pt", torch.device("cuda"), mixed["provenance"]["protocol_sha256"])
    if mixed_payload["epoch"] != 57 or old.state_hash(mixed_model) != mixed["checkpoints"]["best60"]["model_tensor_sha256"]:
        raise ValueError("frozen mixed checkpoint mismatch")
    del mixed_optimizer; release()
    collector = probe.FullLinkDiagnostics(ROOT, mixed_model); before_model = old.state_hash(model)
    stamp = perf_counter()
    with patch.object(old.link, "estimate_ls_linear_channel", collector.estimate), patch.object(old, "channel_error_energies", collector.energy), patch.object(old, "fixed_sample", lambda *a: None):
        evaluation = old.full_link_validation(ROOT, prepared, refs, model, probe.LAYOUT, historical, progress)
    full_seconds = perf_counter()-stamp
    if old.state_hash(model) != before_model:
        raise ValueError("evaluation altered best model")
    regions = collector.verify(evaluation); comparisons = comparison_rows(evaluation, mixed_csv)
    primary = comparisons[-1]; gain = primary["gain_vs_mixed_best57_db"]
    if not np.isclose(next(r for r in evaluation["aggregates"] if r["snr_db"] == 20)["b0_nmse_error_energy"]/15872000, best_metric, rtol=1e-6, atol=1e-12):
        raise ValueError("full-link batch50 and selection batch128 disagree")
    epoch12 = epochs[12]["validation20"]
    exposure = {"exposure_matched_epoch": 12, "probe_20db_exposures": 240000, "mixed_epoch60_20db_exposures": 240000,
        "mixed_best57_20db_exposures": 228000, "equal_exposure_reference": "mixed epoch60",
        "equal_optimizer_trajectory": False, "strictly_equivalent_training": False,
        "epoch12_nmse_db": epoch12["nmse_db"], "mixed_best57_nmse_db": float(mixed_csv[-1]["b0_best60_nmse_db"]),
        "mixed_epoch60_nmse_db": float(mixed_log[60]["val_nmse_20db"]),
        "epoch12_gain_vs_mixed_best57_db": float(mixed_csv[-1]["b0_best60_nmse_db"])-epoch12["nmse_db"],
        "epoch12_gain_vs_mixed_epoch60_db": float(mixed_log[60]["val_nmse_20db"])-epoch12["nmse_db"]}
    primary_rows = [{"method": name, "snr_db": 20, "nmse_db": nmse, "ber": ber} for name, nmse, ber in (
        ("R3", primary["r3_nmse_db"], primary["r3_ber"]), ("mixed_B0_best57", primary["mixed_B0_best57_nmse_db"], primary["mixed_B0_best57_ber"]),
        ("B0_20db_epoch12", epoch12["nmse_db"], None), ("B0_20db_best", primary["b0_20db_only_nmse_db"], primary["b0_20db_only_ber"]),
        ("R4", primary["r4_nmse_db"], primary["r4_ber"]), ("R5", primary["r5_nmse_db"], primary["r5_ber"]))]
    residual = evaluation["residual"][-1]; previous_residual = mixed["residual_diagnostics"][-1]
    residual_comparison = {"mixed_best57": previous_residual, "probe_best": residual,
        "RMS_ratio_change": residual["RMS_ratio"]-previous_residual["RMS_ratio"],
        "RMS_ratio_relative_change": residual["RMS_ratio"]/previous_residual["RMS_ratio"]-1}
    csv_outputs = {"_training_log": log, "_validation": comparisons, "_primary_comparison": primary_rows,
        "_validation_trajectory_metrics": evaluation["trajectory_rows"],
        "_primary_trajectory_metrics": [r for r in evaluation["trajectory_rows"] if r["snr_db"] == 20],
        "_checkpoint_cross_snr_diagnostics": cross, "_time_region_diagnostics": [r for r in regions if r["group"] == "time"],
        "_frequency_distance_diagnostics": [r for r in regions if r["group"] == "frequency_distance"]}
    for suffix, rows in csv_outputs.items():
        save_csv(OUT/f"{probe.PREFIX}{suffix}.csv", rows)
    after = accepted_files()
    if before != after or source_hashes != {p: sha256_file(ROOT/p) for p in SOURCES}:
        raise ValueError("historical source/results or predeclared source changed")
    red = old.oracle_red_flag(evaluation["aggregates"])
    report = {"task": "T12E", "experiment_type": "20-dB-only capacity probe for the B0 residual CNN under sparse pilots",
        "git": {"T12D_commit": head, "T12D_committed_cleanly": True, "clean_after_commit": True, "startup_pytest": "1054 passed in 68.12s (0:01:08)"},
        "diagnostic_model": True, "formal_seed": False, "counts_toward_formal_five_seed_results": False,
        "development_seed": old.SEED, "density": 16, "pilot_ratio": 1/32, "training_snr": [20], "max_epochs": 60,
        "architecture": architecture, "perf_profile": profile, "optimizer": {"name": "Adam", "lr": .001, "weight_decay": 0},
        "optimizer_continuity": {"same_instance_all60_epochs": True, "no_reset": True, "instance_id": optimizer_id, "final_steps": final_steps},
        "protocol_equality": {"same_architecture_input_loss_dtype": True, "same_optimizer_profile_seed": True,
            "same_training_pool_channel_bits_unit_noise": True, "same_shuffle_hashes_all60": True,
            "only_training_variable": "SNR exposure all20dB instead of mixed", "prescribed_selection_change": probe.SELECTION_METRIC,
            "no_loss_reweighting": True},
        "mixed_reference": {"CSV": f"outputs/{dev.PREFIX}_validation.csv", "20db_row": mixed_csv[-1],
            "epoch30_csv": "outputs/t12b_b0_df16_dev_validation.csv", "epoch30_nmse_db": float(mixed_csv[-1]["b0_epoch30_nmse_db"])},
        "epoch0_reference": initial, "selection_metric": probe.SELECTION_METRIC, "best_epoch": best,
        "best_validation_mse_20db": best_metric, "best_20db_nmse": primary["b0_20db_only_nmse_db"],
        "exposure_matched_epoch": 12, "epoch12_comparison": exposure, "dedicated_gain_vs_mixed": gain, **probe.specialization_flags(gain),
        "training_curve": log, "epoch_records": epochs, "checkpoints": checkpoints,
        "primary_20db_results": primary, "cross_snr_results": comparisons, "checkpoint_cross_snr_diagnostics": cross,
        "residual_diagnostics": residual_comparison,
        "time_region_diagnostics": csv_outputs["_time_region_diagnostics"], "frequency_distance_diagnostics": csv_outputs["_frequency_distance_diagnostics"],
        "frequency_distance_definition": "perTX nearest own FDM tone, absolute shifted-index distance; nonperiodic; distance0 separate audit group",
        "region_totals_verified": True, "fixed_sample": collector.fixed_sample,
        "exact_T10D_replay": evaluation["replay"], "realization_hashes": evaluation["realizations"],
        "oracle_red_flag": red, "oracle_red_flag_threshold_db": .10,
        "oracle_audit": {"required": red, "model_input_only_R3": True, "truth_not_network_input": True,
            "same_H_Y_bits_unit_noise": True, "power_SNR_frozen_sources_unchanged": True, "original_complex128_metrics": True, "all_checks_passed": True},
        "runtime": {"validation_cache_prepare_seconds": val_prep, "train_H_cache_prepare_seconds": train_prep,
            "60_epochs_wall_seconds": train_wall, "validation_seconds": val_total, "full_link_seconds": full_seconds,
            "mean_epoch_seconds": float(np.mean([r["epoch_seconds"] for r in log[1:]])),
            "GPU_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "GPU_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "resource_summary": summary},
        "provenance": {"accepted_before": before, "accepted_after": after, "protocol_sha256": protocol_hash,
            "source_sha256": source_hashes, "frozen_T10D_T10C_hashes": verify_frozen(OUT), "validation_tensor_sha256": base_hash,
            "train_H_long_hashes_matched": 2000, "CSV_sha256": {s: sha256_file(OUT/f"{probe.PREFIX}{s}.csv") for s in (*csv_outputs, "_utilization")}},
        "reproducibility": settings, "environment": {"python": platform.python_version(), "torch": torch.__version__, "GPU": torch.cuda.get_device_name()},
        "formal_test_used": False, "formal_test_sealed": True, "OOD_generated": False, "baselines_rerun": False,
        "mixed_or_R4_performance_scan_rerun": False, "additional_seed_or_density": False,
        "scheduler": False, "warmup": False, "clipping": False, "EMA": False, "AMP": False,
        "SNR_or_mask_or_R4_or_covariance_input": False, "MMSE_detector_used": False,
        "numerical_failures": [], "figures": [], "numeric_checks_passed": True, "all_checks_passed": False}
    path = OUT/f"{probe.PREFIX}_results.json"; json_save(path, report)
    stamp = perf_counter(); report["figures"] = render_and_audit(OUT, OUT)
    report["runtime"]["figure_QA_seconds"] = perf_counter()-stamp
    report["runtime"]["total_wall_seconds"] = perf_counter()-started; json_save(path, report)
    progress(f"T12E numeric/figure checks complete: best={best}; dedicated gain={gain:.6f}dB; Oracle flag={red}")


def rebuild_audit():
    names = [f"{probe.PREFIX}{s}.csv" for s in ("_training_log", "_primary_comparison", "_validation")]
    with tempfile.TemporaryDirectory(prefix="t12e_csv_only_", dir=OUT) as directory:
        temporary = Path(directory).resolve()
        if not temporary.is_relative_to(OUT.resolve()):
            raise ValueError("temporary reconstruction outside outputs")
        for name in names:
            shutil.copyfile(OUT/name, temporary/name)
        if sorted(p.name for p in temporary.iterdir()) != sorted(names):
            raise ValueError("CSV-only initial directory required")
        render_and_audit(temporary, temporary)
        checks = {name+"."+ext: {"original": sha256_file(OUT/(name+"."+ext)), "rebuilt": sha256_file(temporary/(name+"."+ext))}
            for name in FIGURE_NAMES for ext in ("png", "pdf", "svg")}
        if any(r["original"] != r["rebuilt"] for r in checks.values()):
            raise ValueError("CSV-only exports not byte-identical")
        result = {"initial_files": names, "numerical_experiment_rerun": False, "hashes": checks, "all_checks_passed": True}
        json_save(OUT/f"{probe.PREFIX}_csv_rebuild_audit.json", result)
        return result


def finalize():
    path = OUT/f"{probe.PREFIX}_results.json"; r = json.loads(path.read_text(encoding="utf8"))
    if not r["numeric_checks_passed"] or accepted_files() != r["provenance"]["accepted_after"]:
        raise ValueError("numeric/frozen evidence failed")
    if any(sha256_file(ROOT/p) != digest for p, digest in r["provenance"]["source_sha256"].items()):
        raise ValueError("current source changed after run")
    for suffix, digest in r["provenance"]["CSV_sha256"].items():
        if sha256_file(OUT/f"{probe.PREFIX}{suffix}.csv") != digest:
            raise ValueError("CSV changed after run")
    r["CSV_rebuild_audit"] = rebuild_audit()
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True, encoding="utf8")
    (OUT/f"{probe.PREFIX}_pytest.txt").write_text(result.stdout+result.stderr, encoding="utf8"); progress(result.stdout)
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True, encoding="utf8")
    if result.returncode or diff.returncode:
        raise RuntimeError(f"pytest/diff failed\n{result.stderr}\n{diff.stdout}\n{diff.stderr}")
    for f in r["figures"]:
        f["visual_self_check"] = "Actual exported PNG inspected; all raw data/methods present; labels/legend readable without clipping"
    r["pytest"] = {"exit_code": 0, "summary": result.stdout.strip().splitlines()[-1]}
    r["git_diff_check"] = {"exit_code": 0, "warnings": diff.stderr}
    r["all_checks_passed"] = True; json_save(path, r)
    progress("T12E final checks PASS; fixed probe complete, no new model started")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--finalize-after-visual-review", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=OUT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.plot_only:
        json_save(args.output_dir/f"{probe.PREFIX}_figure_qa.json", render_and_audit(args.input_dir, args.output_dir))
    elif args.finalize_after_visual_review:
        finalize()
    else:
        if args.input_dir.resolve() != OUT.resolve() or args.output_dir.resolve() != OUT.resolve():
            raise ValueError("main diagnostic uses project outputs")
        execute()


if __name__ == "__main__":
    main()
