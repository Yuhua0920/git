"""T09B entry: fixed main-scenario paired AWGN calibration, not final E1."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_tdl_awgn_scan import plot_results  # noqa: E402
from scripts.run_tdl_noiseless_scan import check_config, save_csv  # noqa: E402
from src.tdl_awgn_scan import run_tdl_awgn_scan, verify_noiseless_regression  # noqa: E402
from src.tdl_noiseless_scan import NoiselessLinkFailure  # noqa: E402
from src.provenance import recorded_source_matches


OUTPUTS = PROJECT_ROOT / "outputs"
PREFIX = "t09b_tdl_awgn"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def frozen_source_checks(baseline: dict) -> dict:
    return {path: recorded_source_matches(PROJECT_ROOT, path, digest, "t09a")
            for path, digest in baseline["provenance"]["source_sha256"].items()
            if path.startswith("src/") or path == "configs/base.yaml"}


def verify_t09a_channels(report, archived_rows, baseline) -> dict:
    # All trajectories' previously archived maxima plus exact fixed H entries.
    expected = {int(row["trajectory_id"]): row for row in archived_rows if int(row["dF"]) == 2}
    maxima = [{"trajectory_id": row["trajectory_id"],
               "passed": bool(np.isclose(row["condition_number_max"],
                                         float(expected[row["trajectory_id"]]["condition_number_max"]),
                                         rtol=1e-12, atol=1e-12))}
              for row in report["channel_realizations"]]
    traces = []
    for actual, old in zip(report["manual_diagnostics"], baseline["manual_diagnostics"]):
        matched = actual["k"] == old["k"] and actual["t"] == old["t"] and all(
            np.array_equal(actual["H_true"][part], old["H_true"][part]) for part in ("real", "imag")
        )
        traces.append({"trajectory_id": 0, "k": actual["k"], "t": actual["t"], "exact_array_equal": matched})
    return {"archived_trajectory_condition_max_checks": maxima, "fixed_H_exact_array_checks": traces,
            "new_full_H_hash_count": len(report["channel_realizations"]),
            "passed": len(maxima) == 200 and len(traces) == 3 and all(row["passed"] for row in maxima)
            and all(row["exact_array_equal"] for row in traces)}


def provenance() -> dict:
    git = ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}"]
    revision = subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    status = subprocess.check_output(git + ["status", "--short"], cwd=PROJECT_ROOT, text=True).strip()
    files = sorted((PROJECT_ROOT / "src").glob("*.py")) + [Path(__file__),
        PROJECT_ROOT / "scripts" / "plot_tdl_awgn_scan.py", PROJECT_ROOT / "scripts" / "run_tdl_noiseless_scan.py",
        PROJECT_ROOT / "tests" / "test_tdl_awgn_scan.py", PROJECT_ROOT / "configs" / "base.yaml",
        OUTPUTS / "t09a_tdl_noiseless_scan.csv", OUTPUTS / "t09a_tdl_noiseless_results.json",
        OUTPUTS / "t09a_tdl_noiseless_trajectory_metrics.csv"]
    return {"git_head": revision, "git_status_at_run": status, "python_version": platform.python_version(),
            "platform": platform.platform(), "numpy_version": np.__version__,
            "source_and_reference_sha256": {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                            for path in files}}


def write_failure(record):
    OUTPUTS.mkdir(exist_ok=True)
    path = OUTPUTS / f"{PREFIX}_failure.json"
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"FAIL: context saved to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--benchmark", action="store_true", help="5 trajectories / 100 noisy evaluations; no result files")
    modes.add_argument("--plot-only", action="store_true", help="rebuild three PNGs solely from aggregate CSV")
    args = parser.parse_args()
    if args.plot_only:
        for path in plot_results(OUTPUTS / f"{PREFIX}_scan.csv", OUTPUTS):
            print(f"Regenerated {path}")
        return
    config = check_config()
    if config["noise"]["snr_db"] != [0, 5, 10, 15, 20]:
        raise ValueError("T09B requires SNR=[0,5,10,15,20]")
    baseline = json.loads((OUTPUTS / "t09a_tdl_noiseless_results.json").read_text(encoding="utf-8"))
    frozen_checks = frozen_source_checks(baseline)
    if not all(frozen_checks.values()):
        raise ValueError(f"frozen T09A source/config conflict: {frozen_checks}")
    started = perf_counter()
    count = 5 if args.benchmark else 200
    try:
        report, rows = run_tdl_awgn_scan(count, include_noiseless_regression=not args.benchmark,
                                       progress=lambda n: print(f"Completed {n}/{count} trajectories", flush=True))
    except NoiselessLinkFailure as exc:
        write_failure({"task": "T09B", "all_checks_passed": False, "numerical_failures": [exc.record]})
        print(json.dumps(exc.record, allow_nan=False))
        raise SystemExit(1) from exc
    elapsed = perf_counter() - started
    if args.benchmark:
        print(f"Benchmark: {count} trajectories, {len(rows)} noisy evaluations, {elapsed:.6f} s; no result files saved")
        print(f"Estimated 200-trajectory noisy scan: {elapsed * 200 / count:.6f} s (full regression adds 800 zero-noise evaluations)")
        return
    regression = verify_noiseless_regression(report["zero_noise_aggregate_results"], read_csv(OUTPUTS / "t09a_tdl_noiseless_scan.csv"))
    channel_checks = verify_t09a_channels(report, read_csv(OUTPUTS / "t09a_tdl_noiseless_trajectory_metrics.csv"), baseline)
    report.update({"config": config, "t09a_noiseless_regression": regression,
                   "t09a_channel_checks": channel_checks, "frozen_source_checks": frozen_checks,
                   "provenance": provenance(), "runtime": {"scan_and_zero_noise_regression_seconds": elapsed}})
    report["all_checks_passed"] = bool(report["all_checks_passed"] and regression["passed"] and channel_checks["passed"])
    for row in report["aggregate_results"]:
        row["all_checks_passed"] = bool(row["all_checks_passed"] and report["all_checks_passed"])
    if not report["all_checks_passed"]:
        write_failure(report)
        raise SystemExit(1)
    OUTPUTS.mkdir(exist_ok=True)
    save_csv(OUTPUTS / f"{PREFIX}_trajectory_metrics.csv", rows)
    save_csv(OUTPUTS / f"{PREFIX}_scan.csv", report["aggregate_results"])
    save_csv(OUTPUTS / "t09b_noise_statistics.csv", report["noise_statistics"])
    plot_results(OUTPUTS / f"{PREFIX}_scan.csv", OUTPUTS)
    report["runtime"]["total_seconds_including_outputs"] = perf_counter() - started
    (OUTPUTS / f"{PREFIX}_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Noise statistics:", json.dumps(report["noise_statistics"], indent=2))
    print("Pilot-LS MSE checks:", json.dumps(report["pilot_ls_theoretical_mse_checks"], indent=2))
    print("T09A noiseless regression:", json.dumps(regression, indent=2))
    print("Manual diagnostics:", json.dumps(report["manual_diagnostics"], indent=2))
    print("Runtime:", json.dumps(report["runtime"]))
    print("T09B all_checks_passed:", report["all_checks_passed"])


if __name__ == "__main__":
    main()
