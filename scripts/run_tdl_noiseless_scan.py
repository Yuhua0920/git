"""T09A official entry: 200 trajectories, or a five-trajectory timing benchmark."""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, validate_config  # noqa: E402
from src.doppler import mobility_parameters, ofdm_time_grid_seconds  # noqa: E402
from src.tdl_noiseless_scan import NoiselessLinkFailure, run_noiseless_scan  # noqa: E402


OUTPUTS = PROJECT_ROOT / "outputs"
PREFIX = "t09a_tdl_noiseless"
STANDARD_CROSSCHECK = {
    "implementation_version": "3GPP TR 38.901 V17.0.0",
    "implementation_url": "https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/17.00.00_60/tr_138901v170000p.pdf",
    "checked_version": "ETSI TR 138 901 V19.5.0 / 3GPP TR 38.901 Release 19",
    "checked_url": "https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/19.05.00_60/tr_138901v190500p.pdf",
    "date": "2026-10-01", "method": "direct inspection of official PDFs",
    "table_7_7_2_1": {"normalized_delays": "unchanged", "powers_db": "unchanged",
                      "fading": "all 23 Rayleigh, unchanged", "row_order": "unchanged",
                      "v17_printed_page": 74, "v19_printed_page": 92},
    "delay_scaling": {"clause": "7.7.3 / Equation 7.7-1", "result": "unchanged",
                      "v17_printed_page": 76, "v19_printed_page": 94},
    "doppler_statement": {"clause": "7.7.2", "result": "classical/Jakes; fD=v/lambda, unchanged",
                          "v17_printed_page": 73, "v19_printed_page": 91},
    "relevant_numerical_fields_unchanged": True, "numerical_implementation_modified": False,
    "record": "docs/tdl_standard_crosscheck.md",
}


def check_config() -> dict:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    validate_config(config)
    expected = {
        "nt": 2, "nr": 2, "nfft": 64, "active_carriers": 64, "num_symbols": 16,
        "delta_f_hz": 30000, "sample_rate_hz": 1920000, "cp_len": 16,
        "fft_norm": "ortho", "modulation": "qpsk_gray", "total_power": 1.0,
    }
    for key, value in expected.items():
        if config["system"][key] != value:
            raise ValueError(f"T09A requires system.{key}={value}")
    if config["pilot"]["spacing_time"] != 4 or config["pilot"]["spacing_frequency_options"] != [2, 4, 8, 16]:
        raise ValueError("T09A requires dT=4 and dF=[2,4,8,16]")
    return config


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_symbol_nmse(csv_path: Path, png_path: Path) -> None:
    """Regenerate only from raw CSV; no smoothing or synthetic no-data values."""
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 64:
        raise ValueError("per-symbol CSV must contain exactly 64 rows")
    with tempfile.TemporaryDirectory(prefix="t09a_matplotlib_") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axis = plt.subplots(figsize=(9, 5.2))
            fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.82)
            for t in (0, 4, 8, 12):
                axis.axvline(t, color="0.82", linewidth=1, linestyle=":",
                             label="Pilot time" if t == 0 else None, zorder=0)
            for dF, marker in zip((2, 4, 8, 16), ("o", "s", "^", "D")):
                group = sorted((row for row in rows if int(row["dF"]) == dF),
                               key=lambda row: int(row["symbol_index"]))
                if [int(row["symbol_index"]) for row in group] != list(range(16)):
                    raise ValueError(f"dF={dF}: missing or duplicate symbol rows")
                x = [int(row["symbol_index"]) for row in group]
                y = [float(row["nmse_db"]) if row["nmse_db"] else np.nan for row in group]
                axis.plot(x, y, marker=marker, markersize=4, linewidth=1.2, label=f"dF = {dF}")
            axis.set(xlabel="OFDM symbol index", ylabel="Data-RE NMSE (dB)", xticks=range(16), xlim=(-0.3, 15.3))
            fig.suptitle("Noiseless dynamic TDL-A: LS-LIN interpolation error", y=0.97)
            axis.grid(axis="y", alpha=0.25)
            handles, labels = axis.get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncols=5, fontsize=9)
            fig.text(0.5, 0.035, "dF=2: no data RE at pilot times; undefined NMSE is shown as gaps.",
                     ha="center", fontsize=8)
            fig.savefig(png_path, dpi=180)
            plt.close(fig)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def provenance() -> dict:
    git_args = ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}"]
    revision = subprocess.check_output(git_args + ["rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    status = subprocess.check_output(git_args + ["status", "--short"], cwd=PROJECT_ROOT, text=True).strip()
    sources = sorted((PROJECT_ROOT / "src").glob("*.py")) + [
        Path(__file__), PROJECT_ROOT / "configs" / "base.yaml",
        PROJECT_ROOT / "docs" / "tdl_standard_crosscheck.md",
        PROJECT_ROOT / "tests" / "test_tdl_noiseless_scan.py",
    ]
    return {"git_head": revision, "git_status_at_run": status, "python_version": platform.python_version(),
            "platform": platform.platform(), "numpy_version": np.__version__,
            "source_sha256": {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sources}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="five trajectories, no output files")
    parser.add_argument("--plot-only", action="store_true", help="rebuild PNG from the 64-row CSV only")
    args = parser.parse_args()
    if args.benchmark and args.plot_only:
        parser.error("--benchmark and --plot-only are mutually exclusive")
    if args.plot_only:
        plot_symbol_nmse(OUTPUTS / f"{PREFIX}_symbol_nmse.csv", OUTPUTS / "t09a_symbol_nmse.png")
        print("Regenerated outputs/t09a_symbol_nmse.png from raw per-symbol CSV")
        return
    config = check_config()
    started = perf_counter()
    count = 5 if args.benchmark else 200
    try:
        report, rows = run_noiseless_scan(count, progress=lambda n: print(f"Completed {n}/{count} trajectories", flush=True)
                                         if n % 25 == 0 or n == count else None)
    except NoiselessLinkFailure as exc:
        OUTPUTS.mkdir(exist_ok=True)
        failure_path = OUTPUTS / f"{PREFIX}_failure.json"
        failure_path.write_text(json.dumps({"task": "T09A", "all_checks_passed": False, "failure": exc.record},
                                           indent=2, allow_nan=False), encoding="utf-8")
        print(f"FAIL: {exc}; context saved to {failure_path}")
        raise SystemExit(1) from exc
    elapsed = perf_counter() - started
    if args.benchmark:
        print(f"Benchmark: {count} trajectories, {len(rows)} evaluations, {elapsed:.6f} s; no result files saved")
        print(f"Estimated 200-trajectory scan: {elapsed * 200 / count:.6f} s")
        return
    report.update({"standard_crosscheck": STANDARD_CROSSCHECK, "config": config,
                   "mobility": mobility_parameters(), "symbol_start_times_s": ofdm_time_grid_seconds().tolist(),
                   "runtime_seconds_scan": elapsed, "provenance": provenance()})
    OUTPUTS.mkdir(exist_ok=True)
    save_csv(OUTPUTS / f"{PREFIX}_trajectory_metrics.csv", rows)
    save_csv(OUTPUTS / f"{PREFIX}_scan.csv", report["aggregate_results"])
    save_csv(OUTPUTS / f"{PREFIX}_symbol_nmse.csv", report["per_symbol_results"])
    plot_symbol_nmse(OUTPUTS / f"{PREFIX}_symbol_nmse.csv", OUTPUTS / "t09a_symbol_nmse.png")
    report["runtime_seconds_total"] = perf_counter() - started
    (OUTPUTS / f"{PREFIX}_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Aggregate results:")
    for row in report["aggregate_results"]:
        print(json.dumps(row, allow_nan=False))
    print("Condition numbers:", json.dumps(report["condition_number_statistics"]))
    print("Maximum condition location:", json.dumps(report["maximum_condition_location"]))
    print("Manual diagnostics (complex arrays stored as separate real/imag):")
    print(json.dumps(report["manual_diagnostics"], indent=2, allow_nan=False))
    print(f"Scan runtime: {elapsed:.6f} s; total including outputs: {report['runtime_seconds_total']:.6f} s")
    print("T09A all_checks_passed:", report["all_checks_passed"])
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
