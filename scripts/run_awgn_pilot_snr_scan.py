"""Run the fixed T07A 200-frame scan or regenerate figures from raw aggregate CSV."""

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.awgn_scan import (  # noqa: E402
    BITS_SEED_BASE, DF_OPTIONS, NOISE_SEED_BASE, NUM_FRAMES,
    SEED_STRIDE, SNR_POINTS, run_scan, zero_noise_regression,
)
from src.config import load_config, validate_config  # noqa: E402
from scripts.plot_awgn_pilot_snr_scan import plot_results  # noqa: E402


def _write_csv(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_aggregation(conditions: list, frames: list) -> bool:
    """Check reported metrics against saved per-frame sufficient statistics."""
    if len(conditions) != 20 or len(frames) != 4000:
        return False
    if len({(r["snr_db"], r["frame_index"], r["dF"]) for r in frames}) != 4000:
        return False
    for row in conditions:
        subset = [f for f in frames if (f["dF"], f["snr_db"]) == (row["dF"], row["snr_db"])]
        if len(subset) != 200 or row["num_frames"] != 200:
            return False
        for field in ("nmse_error_energy", "nmse_truth_energy"):
            if not np.isclose(row[field], sum(f[field] for f in subset), rtol=1e-12, atol=1e-12):
                return False
        if row["total_data_bits"] != sum(f["data_bits"] for f in subset):
            return False
        if not np.isclose(row["nmse_data_linear"], row["nmse_error_energy"] / row["nmse_truth_energy"], rtol=1e-12):
            return False
        for prefix in ("perfect_csi", "ls_lin"):
            if row[f"{prefix}_bit_errors"] != sum(f[f"{prefix}_bit_errors"] for f in subset):
                return False
            if row[f"{prefix}_ber"] != row[f"{prefix}_bit_errors"] / row["total_data_bits"]:
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="Time 5 frames/condition; write no performance results")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate PNGs from existing aggregate CSV")
    args = parser.parse_args()
    if args.benchmark and args.plot_only:
        parser.error("--benchmark and --plot-only are mutually exclusive")
    output_dir = PROJECT_ROOT / "outputs"
    aggregate_path = output_dir / "t07a_awgn_pilot_snr_scan.csv"
    if args.plot_only:
        for path in plot_results(aggregate_path, output_dir):
            print(f"Plot regenerated: {path}")
        return
    config = load_config(PROJECT_ROOT / "configs/base.yaml")
    validate_config(config)
    expected_system = {"nt": 2, "nr": 2, "nfft": 64, "active_carriers": 64,
                       "num_symbols": 16, "total_power": 1.0, "modulation": "qpsk_gray"}
    if any(config["system"][key] != value for key, value in expected_system.items()):
        raise ValueError("T07A requires unchanged T06B system parameters")
    if config["noise"]["snr_db"] != list(SNR_POINTS) or config["pilot"]["spacing_time"] != 4 or config["pilot"]["spacing_frequency_options"] != list(DF_OPTIONS):
        raise ValueError("T07A requires specified five SNR points and four pilot spacings")
    if args.benchmark:
        started = perf_counter()
        _, frames, _ = run_scan(num_frames=5)
        elapsed = perf_counter() - started
        print(f"Benchmark only: 5 frames/condition, {len(frames)} evaluations, {elapsed:.3f} s")
        print(f"Estimated 4000-evaluation scan time: {elapsed * 40:.3f} s; full run remains 200 frames/condition")
        return

    baseline_path = output_dir / "t06b_noiseless_full_link_results.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    regression = zero_noise_regression(baseline)
    if not regression["passed"]:
        raise ValueError("T06B zero-noise regression failed")
    print("T07A AWGN calibration scan: fixed T06B channel, 20 conditions, 200 frames/condition", flush=True)
    print("SNR reference: 1 per RX branch; sigma_n^2 = 10^(-SNR_dB/10)", flush=True)
    print("Zero-noise T06B regression: PASS", flush=True)
    started = perf_counter()
    rows, frames, noise_checks = run_scan(progress=lambda message: print(message, flush=True))
    scan_seconds = perf_counter() - started
    aggregation_passed = verify_aggregation(rows, frames)
    all_checks = bool(aggregation_passed and regression["passed"] and all(r["all_checks_passed"] for r in rows))
    output_dir.mkdir(exist_ok=True)
    frame_path = output_dir / "t07a_awgn_frame_metrics.csv"
    json_path = output_dir / "t07a_awgn_pilot_snr_scan_results.json"
    _write_csv(aggregate_path, rows)
    _write_csv(frame_path, frames)
    plot_started = perf_counter()
    figure_paths = plot_results(aggregate_path, output_dir)
    plot_seconds = perf_counter() - plot_started
    report = {
        "task": "T07A", "experiment_type": "fixed deterministic channel AWGN calibration; not final TDL-A Monte Carlo",
        "channel_definition": {**baseline["channel_definition"], "link_phase_definitions": baseline["link_phase_definitions"],
                               "noise_added": True, "channel_reused_from": "T06B"},
        "snr_definition": {"reference_signal_power_per_rx": 1.0, "sigma_n_squared": "10**(-snr_db/10)",
                           "noise_position": "frequency-domain Y=HX+N, all pilot and data REs",
                           "per_frame_normalization": False},
        "snr_points": list(SNR_POINTS),
        "pilot_densities": [{"dF": r["dF"], "pilot_ratio": r["pilot_ratio"]} for r in rows if r["snr_db"] == 0],
        "num_frames_per_condition": NUM_FRAMES,
        "seed_policy": {"bits_seed_base": BITS_SEED_BASE, "noise_seed_base": NOISE_SEED_BASE,
                        "snr_index_order": list(SNR_POINTS), "stride": SEED_STRIDE,
                        "bits_seed_txq": "bits_seed_base + snr_index*100000 + 2*frame_index + q",
                        "noise_seed": "noise_seed_base + snr_index*100000 + frame_index",
                        "frame_index_range": [0, 199], "tx_index_range": [0, 1],
                        "shared_master_data_across_df": True, "shared_noise_across_df": True,
                        "same_noisy_y_for_both_methods": True, "channel_seed": None, "training_seed": None},
        "perfect_csi_comparison_note": "Perfect CSI knows H but uses noisy Y; BER is evaluated on each dF's data mask, so masks and BER can differ.",
        "aggregation_definition": {"nmse": "sum_frame error_energy / sum_frame truth_energy; then 10*log10",
                                   "ber": "sum_frame bit_errors / sum_frame data_bits", "data_mask_only": True},
        "noise_power_checks": noise_checks, "conditions": rows, "zero_noise_regression": regression,
        "frame_metrics_rows": len(frames), "aggregate_rows": len(rows),
        "aggregation_verified_against_frame_rows": aggregation_passed,
        "runtime_seconds": {"scan": scan_seconds, "plots": plot_seconds},
        "plot_zero_ber_policy": "display zero-error points at 0.5/total_data_bits with downward markers; raw CSV unchanged",
        "all_checks_passed": all_checks,
    }
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for noise in noise_checks:
        print(f"SNR={noise['snr_db']}: target noise={noise['target_noise_variance']:.8g}, empirical={noise['empirical_noise_power']:.8g}, relative error={noise['noise_relative_error']:.3%}")
    for row in rows:
        print(f"dF={row['dF']:2}, SNR={row['snr_db']:2}: NMSE={row['nmse_data_db']:.6f} dB, Perfect BER={row['perfect_csi_ber']:.8f}, LS-LIN BER={row['ls_lin_ber']:.8f}, bits={row['total_data_bits']}")
    print(f"Scan runtime: {scan_seconds:.3f} s; plotting: {plot_seconds:.3f} s")
    print(f"Aggregate rows: {len(rows)}; frame metrics rows: {len(frames)}")
    print(f"Aggregation from frame rows: {'PASS' if aggregation_passed else 'FAIL'}")
    print(f"All checks: {'PASS' if all_checks else 'FAIL'}")
    for path in (aggregate_path, frame_path, json_path, *figure_paths):
        print(f"Saved: {path}")
    if not all_checks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
