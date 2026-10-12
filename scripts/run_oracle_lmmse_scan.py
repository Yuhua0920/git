"""T10B entry: 200 paired trajectories, or 5-trajectory benchmark / CSV-only plots."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_oracle_lmmse_scan import plot_results  # noqa: E402
from src.oracle_lmmse_scan import (load_theoretical_table, read_csv, run_oracle_lmmse_scan,  # noqa: E402
                                   verify_t09b_regression)
from src.lmmse import LMMSEPreparationError  # noqa: E402
from src.tdl_noiseless_scan import NoiselessLinkFailure  # noqa: E402
from scripts.validate_lmmse_oracle import check_config  # noqa: E402
from src.provenance import recorded_source_matches


OUTPUTS = PROJECT_ROOT / "outputs"
PREFIX = "t10b_oracle_lmmse"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance():
    git = ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}"]
    history = subprocess.check_output(git + ["log", "--format=%H%x09%s"], text=True).splitlines()
    commits = [line.split("\t")[0] for line in history if line.endswith("\tfeat: add calibrated Oracle LMMSE channel estimator")]
    if not commits:
        raise ValueError("accepted T10A standalone commit is missing")
    files = sorted((PROJECT_ROOT / "src").glob("*.py")) + [Path(__file__),
        PROJECT_ROOT / "scripts/plot_oracle_lmmse_scan.py", PROJECT_ROOT / "tests/test_oracle_lmmse_scan.py",
        PROJECT_ROOT / "configs/base.yaml", OUTPUTS / "t09b_tdl_awgn_scan.csv",
        OUTPUTS / "t09b_tdl_awgn_results.json", OUTPUTS / "t09b_tdl_awgn_trajectory_metrics.csv",
        OUTPUTS / "t10a_lmmse_oracle_theory.csv", OUTPUTS / "t10a_lmmse_oracle_results.json"]
    return {"git_head": subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip(),
        "git_status_at_run": subprocess.check_output(git + ["status", "--short"], text=True).strip(),
        "t10a_commit": commits[0], "python_version": platform.python_version(), "platform": platform.platform(),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "source_and_reference_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): sha256(p) for p in files}}


def frozen_source_checks(t10a):
    return {name: recorded_source_matches(PROJECT_ROOT, name, digest, "t10a", current_hash=sha256)
            for name, digest in t10a["source_file_hashes"].items()}


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_failure(record):
    OUTPUTS.mkdir(exist_ok=True)
    path = OUTPUTS / f"{PREFIX}_failure.json"
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"FAIL: recorded in {path}; comparison plots not generated")


def alignment_gate(tools_dir):
    if tools_dir is None:
        return None
    sys.path.insert(0, str(tools_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    return require_matplotlib_panel_alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--benchmark", action="store_true", help="5 trajectories / 100 conditions; no result files")
    modes.add_argument("--plot-only", action="store_true", help="rebuild four PNG/PDF figures using only aggregate CSV")
    parser.add_argument("--figure-qa-tools-dir", type=Path, help="optional render-time figure alignment auditor")
    args = parser.parse_args()
    if args.plot_only:
        before = {name: sha256(OUTPUTS / f"{name}.png") if (OUTPUTS / f"{name}.png").exists() else None for name in (
            "t10b_nmse_comparison", "t10b_ber_comparison", "t10b_oracle_theory_vs_empirical", "t10b_correct_bit_rate_comparison")}
        paths = plot_results(OUTPUTS / f"{PREFIX}_scan.csv", OUTPUTS, alignment_gate(args.figure_qa_tools_dir))
        checks = {p.name: before[p.stem] == sha256(p) for p in paths if before[p.stem] is not None}
        created = [p.name for p in paths if before[p.stem] is None]
        if created:
            print("Created from aggregate CSV without previous images:", json.dumps(created))
        print("CSV-only PNG regeneration exact SHA256 matches:", json.dumps(checks))
        if not all(checks.values()):
            raise SystemExit(1)
        return
    config = check_config()
    t10a = json.loads((OUTPUTS / "t10a_lmmse_oracle_results.json").read_text(encoding="utf-8"))
    frozen_checks = frozen_source_checks(t10a)
    if not all(frozen_checks.values()) or sha256(OUTPUTS / "t10a_lmmse_oracle_theory.csv") != t10a["theory_csv_sha256"]:
        raise ValueError("frozen T10A source/config/theory conflict")
    theory = load_theoretical_table(OUTPUTS / "t10a_lmmse_oracle_theory.csv")
    metadata = provenance()
    started = perf_counter()
    try:
        report, rows = run_oracle_lmmse_scan(theory, 5 if args.benchmark else 200,
            progress=lambda n: print(f"Completed {n}/200 paired trajectories", flush=True))
    except (NoiselessLinkFailure, LMMSEPreparationError, ValueError, np.linalg.LinAlgError) as exc:
        write_failure({"task": "T10B", "all_checks_passed": False, "num_filtered_trajectories": 0,
            "numerical_failures": [getattr(exc, "record", {"reason": str(exc)})], "provenance": metadata})
        raise SystemExit(1) from exc
    if args.benchmark:
        timing = report["runtime"]
        print(f"Benchmark: 5 trajectories, 100 conditions, 40 prepared estimators")
        print("Runtime:", json.dumps(timing))
        print(f"Projected 200-trajectory simulation: {timing['simulation_seconds'] * 40:.6f} s; no result files saved")
        return
    regression = verify_t09b_regression(report["aggregate_results"], read_csv(OUTPUTS / "t09b_tdl_awgn_scan.csv"))
    archived = json.loads((OUTPUTS / "t09b_tdl_awgn_results.json").read_text(encoding="utf-8"))
    hashes = {r["trajectory_id"]: r["H_sha256_complex128_C_order"] for r in archived["channel_realizations"]}
    channel_checks = [{"trajectory_id": r["trajectory_id"],
                       "passed": r["H_sha256_complex128_C_order"] == hashes[r["trajectory_id"]]}
                      for r in report["realization_hashes"]]
    regression["all_200_H_hash_checks"] = channel_checks
    regression["passed"] = regression["passed"] and all(r["passed"] for r in channel_checks)
    report.update({"config": config, "t09b_regression": regression, "frozen_source_checks": frozen_checks,
                   "provenance": metadata, "t10a_committed_cleanly": True})
    report["all_checks_passed"] = bool(report["all_checks_passed"] and regression["passed"])
    if not report["all_checks_passed"]:
        write_failure(report)
        raise SystemExit(1)
    for row in report["aggregate_results"]:
        row["all_checks_passed"] = report["all_checks_passed"]
    OUTPUTS.mkdir(exist_ok=True)
    save_csv(OUTPUTS / f"{PREFIX}_trajectory_metrics.csv", rows)
    save_csv(OUTPUTS / f"{PREFIX}_scan.csv", report["aggregate_results"])
    figures = plot_results(OUTPUTS / f"{PREFIX}_scan.csv", OUTPUTS, alignment_gate(args.figure_qa_tools_dir))
    report["figures"] = [{"path": p.relative_to(PROJECT_ROOT).as_posix(),
        "source_csv": f"outputs/{PREFIX}_scan.csv", "sha256": sha256(p)} for p in figures]
    report["output_csv_sha256"] = {p.name: sha256(p) for p in (
        OUTPUTS / f"{PREFIX}_scan.csv", OUTPUTS / f"{PREFIX}_trajectory_metrics.csv")}
    report["runtime"]["total_seconds_including_outputs"] = perf_counter() - started
    report_path = OUTPUTS / f"{PREFIX}_results.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("T09B full regression:", regression["passed"])
    print("Oracle empirical/theory comparisons:", json.dumps(report["t10a_theory_regression"], indent=2))
    print("Manual diagnostics:", json.dumps(report["manual_diagnostics"], indent=2))
    print("Runtime:", json.dumps(report["runtime"]))
    print("T10B all_checks_passed:", report["all_checks_passed"])
    print(f"Generated: {report_path}; {len(rows)} trajectory rows / 20 aggregate rows / 4 PNGs")


if __name__ == "__main__":
    main()
