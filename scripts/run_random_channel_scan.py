"""T07B 200 independent Gaussian-tap channel trajectories and pilot trade-off."""

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_random_channel_scan import plot_results  # noqa: E402
from src.awgn_scan import BITS_SEED_BASE, DF_OPTIONS, NOISE_SEED_BASE, SNR_POINTS  # noqa: E402
from src.config import load_config, validate_config  # noqa: E402
from src.random_channels import DELAY_SAMPLES, NORMALIZED_PDP, POWERS_DB  # noqa: E402
from src.stochastic_scan import (  # noqa: E402
    CHANNEL_SEED_BASE, NUM_TRAJECTORIES, ScanFailure, fixed_channel_regressions,
    frame_duration_seconds, run_random_scan, verify_trajectory_aggregation,
)


def _read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--benchmark", action="store_true", help="Time 5 trajectories without saving performance results")
    group.add_argument("--plot-only", action="store_true", help="Rebuild three PNGs using the aggregate CSV")
    args = parser.parse_args()
    output_dir = PROJECT_ROOT / "outputs"
    aggregate_path = output_dir / "t07b_random_channel_scan.csv"
    trajectory_path = output_dir / "t07b_random_channel_trajectory_metrics.csv"
    json_path = output_dir / "t07b_random_channel_scan_results.json"
    if args.plot_only:
        for path in plot_results(aggregate_path, output_dir):
            print(f"Plot regenerated: {path}")
        return
    config = load_config(PROJECT_ROOT / "configs/base.yaml")
    validate_config(config)
    expected = {"nt": 2, "nr": 2, "nfft": 64, "active_carriers": 64, "num_symbols": 16,
                "modulation": "qpsk_gray", "total_power": 1.0, "delta_f_hz": 30000, "cp_len": 16, "fft_norm": "ortho"}
    if any(config["system"][key] != value for key, value in expected.items()):
        raise ValueError("T07B requires the unchanged system parameters")
    if config["noise"]["snr_db"] != list(SNR_POINTS) or config["pilot"]["spacing_time"] != 4 or config["pilot"]["spacing_frequency_options"] != list(DF_OPTIONS):
        raise ValueError("T07B requires specified SNR/pilot settings")
    if args.benchmark:
        started = perf_counter()
        _, rows, _ = run_random_scan(num_trajectories=5)
        elapsed = perf_counter() - started
        print(f"Benchmark only: 5 trajectories, {len(rows)} evaluations, {elapsed:.3f} s")
        print(f"Projected 200-trajectory / 4000-evaluation time: {elapsed * 40:.3f} s")
        return
    baseline = json.loads((output_dir / "t06b_noiseless_full_link_results.json").read_text(encoding="utf-8"))
    regressions = fixed_channel_regressions(baseline, _read_csv(output_dir / "t07a_awgn_frame_metrics.csv"))
    if not regressions["passed"]:
        raise ValueError("fixed-channel regression failed")
    print("T07B random independent complex-Gaussian tapped fading calibration; not 3GPP TDL-A", flush=True)
    print("200 independent trajectories; 20 conditions each; no realization power normalization", flush=True)
    print("Fixed-channel T06B/T07A regressions: PASS", flush=True)
    started = perf_counter()
    try:
        aggregates, trajectories, metadata = run_random_scan(progress=lambda message: print(message, flush=True))
    except ScanFailure as exc:
        report = {"task": "T07B", "all_checks_passed": False, "status": "needs numerical inspection",
                  "numerical_failures": [exc.record], "runtime": perf_counter() - started}
        json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"FAIL: {exc.record}", flush=True)
        raise SystemExit(1) from exc
    elapsed = perf_counter() - started
    aggregation_passed = verify_trajectory_aggregation(aggregates, trajectories)
    finite = bool(all(all(np.isfinite(value) for value in row.values()) for row in (*aggregates, *trajectories)))
    # A predeclared descriptive Monte Carlo check, never a channel rescaling step.
    power_mean_tolerance = 0.20
    power_check = bool(all(abs(stat["mean"] - 1) <= power_mean_tolerance for stat in metadata["channel_power_statistics"].values()))
    all_checks = bool(aggregation_passed and finite and power_check and regressions["passed"]
                      and all(row["all_checks_passed"] for row in aggregates))
    _write_csv(aggregate_path, aggregates)
    _write_csv(trajectory_path, trajectories)
    plot_started = perf_counter()
    plots = plot_results(aggregate_path, output_dir)
    plot_elapsed = perf_counter() - plot_started
    report = {
        "task": "T07B", "experiment_type": "random independent complex-Gaussian tapped fading calibration; not 3GPP TDL-A",
        "channel_model_description": "Independent CN(0,p_l) taps per RX/TX/path; static over 16 symbols; ensemble-only PDP normalization",
        "delay_samples": list(DELAY_SAMPLES), "powers_db": list(POWERS_DB), "normalized_pdp": NORMALIZED_PDP.tolist(),
        "per_realization_channel_normalization": False, "num_trajectories": NUM_TRAJECTORIES,
        "snr_points": list(SNR_POINTS),
        "pilot_densities": [{"dF": row["dF"], "pilot_ratio": row["pilot_ratio"]} for row in aggregates if row["snr_db"] == 0],
        "seed_policy": {"channel_seed_base": CHANNEL_SEED_BASE, "bits_seed_base": BITS_SEED_BASE,
                        "noise_seed_base": NOISE_SEED_BASE, "channel_seed": "channel_seed_base + trajectory_id",
                        "bits_seed_txq": "bits_seed_base + 2*trajectory_id + q",
                        "noise_seed": "noise_seed_base + snr_index*100000 + trajectory_id",
                        "trajectory_id_range": [0, 199], "snr_index_order": list(SNR_POINTS),
                        "bits_shared_across_snr": True, "paired_H_data_noise_across_df": True,
                        "same_noisy_Y_for_both_methods": True, "training_seed": None},
        "snr_definition": {"ensemble_signal_power_per_rx": 1.0, "noise_variance": "10**(-snr_db/10)",
                           "instantaneous_normalization": False},
        **metadata, "channel_power_sanity_check": {"absolute_mean_deviation_tolerance": power_mean_tolerance, "passed": power_check},
        "aggregate_conditions": aggregates, "aggregate_rows": len(aggregates), "trajectory_rows": len(trajectories),
        "aggregation_rebuilt_from_trajectory_rows": aggregation_passed,
        "all_outputs_finite": finite, "fixed_channel_regressions": regressions,
        "proxy_definition": {"method": "LS-LIN + ZF", "correct_bits_per_frame_proxy": "data_bits_per_frame*(1-BER)",
                             "correct_bit_rate_proxy_bps": "correct_bits_per_frame_proxy/frame_duration_seconds",
                             "frame_duration_seconds": frame_duration_seconds()},
        "perfect_csi_note": "True H is known but Y contains fading and AWGN; ZF noise enhancement remains. Data masks differ across dF.",
        "condition_statistics_unit": "200 trajectories x 64 frequency bins; no time replication counted",
        "runtime": {"scan_seconds": elapsed, "plot_seconds": plot_elapsed}, "all_checks_passed": all_checks,
    }
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Channel energy statistics: {metadata['channel_power_statistics']}")
    print(f"Condition number statistics: {metadata['condition_number_statistics']}")
    for row in aggregates:
        print(f"dF={row['dF']:2}, SNR={row['snr_db']:2}: NMSE={row['nmse_data_db']:.6f} dB, "
              f"Perfect BER={row['perfect_csi_ber']:.8f}, LS-LIN BER={row['ls_lin_ber']:.8f}, "
              f"correct-bit-rate proxy={row['correct_bit_rate_proxy_bps']:.3f} bit/s")
    print(f"Runtime: scan={elapsed:.3f} s, plots={plot_elapsed:.3f} s")
    print(f"Aggregate rows: {len(aggregates)}; trajectory rows: {len(trajectories)}; filtered trajectories: 0")
    print(f"All checks: {'PASS' if all_checks else 'FAIL'}")
    for path in (aggregate_path, trajectory_path, json_path, *plots):
        print(f"Saved: {path}")
    if not all_checks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
