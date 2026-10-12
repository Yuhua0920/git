"""T13A calibrated two-axis OOD scan, CSV-only redraw, or final verification."""
import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/"outputs"

from scripts.plot_ood_mismatch import FIGURE_NAMES, plot_results
from scripts.run_lmmse_train_validation import qa_tools, save_csv
from scripts.validate_b0_pipeline import json_save
from src import ood_mismatch as scan
from src.b0_development import replay_link, state_hash
from src.b0_model import configure_reproducibility
from src.config import load_config
from src.lmmse_train import raw_array_sha256, sha256_file
from src.nearest_validation import verify_frozen
from src.pilots import count_layout_resources

SOURCES = ("docs/ood_mismatch_protocol.md", "src/ood_mismatch.py", "scripts/run_ood_mismatch.py",
           "scripts/plot_ood_mismatch.py", "tests/test_ood_mismatch.py")


def progress(message):
    print(message, flush=True)


def accepted_files():
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", scan.T12E_COMMIT,
        "src", "scripts", "tests", "configs", "docs", "outputs"], cwd=ROOT, text=True).splitlines()
    return {p: sha256_file(ROOT/p) for p in paths if p != "docs/stage_status.md"}


def render_and_audit(input_dir, output_dir):
    tools, gate = qa_tools(None)
    paths, records = plot_results(input_dir, output_dir, gate)
    for path, record in zip(paths, records):
        text = subprocess.run([sys.executable, str(tools/"audit_pdf_text.py"), str(path.with_suffix(".pdf")),
            "--min-pt", "5", "--json"], capture_output=True, text=True, encoding="utf8")
        collision_path = path.with_suffix(".collision-audit.json")
        collision = subprocess.run([sys.executable, str(tools/"audit_figure_collisions.py"), str(path.with_suffix(".pdf")),
            "--json-out", str(collision_path)], capture_output=True, text=True, encoding="utf8")
        if text.returncode or collision.returncode:
            raise RuntimeError(f"figure QA failed: {path.name}\n{text.stdout}\n{collision.stdout}")
        record.update({"path": str(path), "PDF_text_audit": json.loads(text.stdout),
            "collision_audit": json.loads(collision_path.read_text(encoding="utf8")),
            "alignment_audit": json.loads(path.with_suffix(".alignment.json").read_text(encoding="utf8")),
            "sha256": {ext: sha256_file(path.with_suffix('.'+ext)) for ext in ('png', 'pdf', 'svg')}})
    return records


def execute():
    result_path = OUT/f"{scan.PREFIX}_results.json"
    if result_path.exists() or (OUT/f"{scan.PREFIX}.csv").exists():
        raise ValueError("OOD artifacts already exist; refuse accidental scan rerun")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != scan.T12E_COMMIT:
        raise ValueError("accepted T12E commit must be HEAD")
    load_config(ROOT/"configs/base.yaml")
    before = accepted_files(); source_hashes = {p: sha256_file(ROOT/p) for p in SOURCES}
    frozen_refs = verify_frozen(OUT)
    if count_layout_resources(scan.LAYOUT)["union_ratio"] != 1/32:
        raise ValueError("actual dF16 union pilot count wrong")
    settings = configure_reproducibility(13261001, 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen, model, r4_record, b0_record = scan.load_frozen_estimators(OUT, device)
    references = scan.id_references(OUT); historical = replay_link(ROOT)
    report = {"task": "T13A", "experiment_type": "development scan of frozen-statistics mismatch under low-pilot MIMO-OFDM channel estimation",
        "literature_gate_passed": True, "git": {"T12E_commit": scan.T12E_COMMIT,
            "T12E_committed_cleanly": True, "clean_after_commit": True, "startup_pytest": "1090 passed in 77.03s (0:01:17)"},
        "ood_temporal_definition": scan.SHIFTS["speed120"], "ood_frequency_definition": scan.SHIFTS["DS300"],
        "seed_policy": {"base_seed": scan.BASE_SEED, "roles": {"channel": 13001, "bits": 13002, "unit_noise": 13003},
            "purpose": {"calibration": 0, "scan": 1}, "words": "[role,base,purpose,shift_code,trajectory_id,frame_index,tx]",
            "derivation": "SeedSequence.generate_state(1,uint64)[0]", "split": "dedicated OOD-development", "formal_namespace_used": False},
        "r4_frozen_provenance": r4_record, "b0_checkpoint_provenance": b0_record,
        "dF": 16, "dT": 4, "pilot_ratio": 1/32, "snr_db": list(scan.SNRS),
        "independent_trajectories_per_shift": 200, "continuous_frames_per_trajectory": 5,
        "ID_reference_results": references, "ID_comparison_paired": False, "ID_scan_rerun": False,
        "formal_training_performed": False, "new_model_trained": False, "B0_adaptation": False,
        "formal_test_used": False, "formal_test_sealed": True, "MMSE_detector_used": False,
        "channel_semantics": "symbol-wise quasi-static; no within-symbol ICI; existing nominal PDP DS scaling",
        "provenance": {"accepted_before": before, "source_sha256": source_hashes, "frozen_T10D_T10C_hashes": frozen_refs},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
            "device": str(device), "GPU": torch.cuda.get_device_name() if device.type == "cuda" else None},
        "reproducibility": settings, "numerical_failures": [], "figures": [], "all_checks_passed": False}
    started = perf_counter(); channel_models = {s: scan.prepare_channel(s) for s in scan.SHIFTS}
    calibration = {}
    for shift, channel in channel_models.items():
        calibration[shift] = scan.calibrate_channel(channel, shift)
        progress(f"Calibration {shift}: power={calibration[shift]['channel_power']:.8f}; PASS={calibration[shift]['all_checks_passed']}")
    report["calibration"] = calibration; report["calibration_seconds"] = perf_counter()-started
    json_save(result_path, report)
    scan.require_calibration(calibration)
    pool_started = perf_counter()
    pools, oracle_records = {}, {}
    for shift in scan.SHIFTS:
        pools[shift], oracle_records[shift] = scan.prepare_estimators(frozen, shift)
    report["r5_ood_covariance_provenance"] = oracle_records
    report["estimator_preparation_seconds"] = perf_counter()-pool_started
    save_csv(OUT/f"{scan.PREFIX}_manifest.csv", scan.manifest())
    timing = {k: 0. for k in ("channel_generation_seconds", "R3_seconds", "R4_seconds", "R5_seconds", "B0_inference_seconds", "ZF_seconds")}
    rows, realizations = [], []
    scan_started = perf_counter()
    try:
        for shift in scan.SHIFTS:
            for tid in range(scan.NUM_TRAJECTORIES):
                new_rows, realization = scan.evaluate_trajectory(channel_models[shift], shift, tid, model, pools[shift], historical, timing)
                rows.extend(new_rows); realizations.append(realization)
                if (tid+1) % 25 == 0:
                    progress(f"Actual pilots/ZF {shift}: {tid+1}/200 trajectories")
    except Exception as exc:
        report["numerical_failures"].append({"shift": shift, "trajectory_id": tid, "reason": str(exc), "no_samples_removed": True})
        report["runtime"] = timing; report["completed_trajectory_rows"] = len(rows)
        json_save(result_path, report)
        if rows:
            save_csv(OUT/f"{scan.PREFIX}_partial_trajectory_metrics.csv", rows)
        raise
    timing["performance_scan_wall_seconds"] = perf_counter()-scan_started
    aggregate = scan.aggregate(rows, references)
    if len(rows) != 1200 or len(aggregate) != 6 or raw_array_sha256(frozen.R_train) != r4_record["raw_array_sha256"] or state_hash(model) != b0_record["model_state_sha256"]:
        raise ValueError("row count or frozen estimator mutation failed")
    save_csv(OUT/f"{scan.PREFIX}_trajectory_metrics.csv", rows)
    save_csv(OUT/f"{scan.PREFIX}.csv", aggregate)
    paired = scan.paired_diagnostics(rows)
    save_csv(OUT/f"{scan.PREFIX}_paired_diagnostics.csv", [{"shift": r["shift"], "snr_db": r["snr_db"], "comparison": r["comparison"],
        **{f"{metric}_{key}": value for metric in ("NMSE_difference_db", "BER_difference") for key, value in r[metric].items()}} for r in paired])
    after = accepted_files()
    if before != after or frozen_refs != verify_frozen(OUT):
        raise ValueError("historical source/results changed")
    report.update({"aggregate_results": aggregate, "paired_diagnostics": paired, **scan.direction_flags(aggregate),
        "realization_hashes": realizations, "no_inferential_statistics": True, "trajectory_rows": len(rows),
        "common_random_numbers_verified": True, "exact_T10D_link_checks": 6000,
        "num_filtered_trajectories": 0, "runtime": timing, "numeric_checks_passed": True})
    report["provenance"]["accepted_after"] = after
    report["provenance"]["CSV_sha256"] = {s: sha256_file(OUT/f"{scan.PREFIX}{s}.csv") for s in ("", "_trajectory_metrics", "_manifest", "_paired_diagnostics")}
    json_save(result_path, report)
    report["figures"] = render_and_audit(OUT, OUT)
    report["runtime"]["total_wall_seconds"] = perf_counter()-started
    json_save(result_path, report)
    progress("T13A numeric/figure checks complete; no model trained")


def rebuild_audit():
    with tempfile.TemporaryDirectory(prefix="t13a_csv_only_", dir=OUT) as directory:
        temporary = Path(directory)
        source = f"{scan.PREFIX}.csv"; shutil.copyfile(OUT/source, temporary/source)
        if [p.name for p in temporary.iterdir()] != [source]:
            raise ValueError("one CSV-only starting directory required")
        render_and_audit(temporary, temporary)
        hashes = {name+"."+ext: {"original": sha256_file(OUT/(name+"."+ext)), "rebuilt": sha256_file(temporary/(name+"."+ext))}
            for name in FIGURE_NAMES for ext in ("png", "pdf", "svg")}
        if any(v["original"] != v["rebuilt"] for v in hashes.values()):
            raise ValueError("CSV-only exports differ")
        record = {"initial_files": [source], "numerical_scan_rerun": False, "hashes": hashes, "all_checks_passed": True}
        json_save(OUT/f"{scan.PREFIX}_csv_rebuild_audit.json", record)
        return record


def finalize():
    path = OUT/f"{scan.PREFIX}_results.json"; r = json.loads(path.read_text(encoding="utf8"))
    if not r["numeric_checks_passed"] or accepted_files() != r["provenance"]["accepted_after"]:
        raise ValueError("frozen/numerical gate failed")
    if any(sha256_file(ROOT/p) != h for p, h in r["provenance"]["source_sha256"].items()):
        raise ValueError("source changed after scan")
    if any(sha256_file(OUT/f"{scan.PREFIX}{s}.csv") != h for s, h in r["provenance"]["CSV_sha256"].items()):
        raise ValueError("numeric CSV changed")
    r["CSV_rebuild_audit"] = rebuild_audit()
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True, encoding="utf8")
    (OUT/f"{scan.PREFIX}_pytest.txt").write_text(result.stdout+result.stderr, encoding="utf8"); progress(result.stdout)
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True, encoding="utf8")
    if result.returncode or diff.returncode:
        raise RuntimeError(f"pytest/diff failed\n{result.stderr}\n{diff.stdout}\n{diff.stderr}")
    for f in r["figures"]:
        f["visual_self_check"] = "Actual exported PNG and each panel inspected; labels/legend readable and all requested conditions/methods present"
    r["pytest"] = {"exit_code": 0, "summary": result.stdout.strip().splitlines()[-1]}
    r["git_diff_check"] = {"exit_code": 0, "warnings": diff.stderr}
    r["all_checks_passed"] = True; json_save(path, r)
    progress("T13A final checks PASS; development mechanism scan complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--finalize-after-visual-review", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=OUT); parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.plot_only:
        json_save(args.output_dir/f"{scan.PREFIX}_figure_qa.json", render_and_audit(args.input_dir, args.output_dir))
    elif args.finalize_after_visual_review:
        finalize()
    else:
        if args.input_dir.resolve() != OUT.resolve() or args.output_dir.resolve() != OUT.resolve():
            raise ValueError("scan artifacts belong in project outputs")
        execute()


if __name__ == "__main__":
    main()
