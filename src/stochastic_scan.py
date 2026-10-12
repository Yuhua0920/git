"""T07B paired trajectory scan; every generated trajectory is retained."""

import re

import numpy as np

from src.awgn_scan import BITS_SEED_BASE, DF_OPTIONS, NOISE_SEED_BASE, SNR_POINTS
from src.link import (
    add_received_noise, apply_frequency_mimo_channel, assemble_pilot_data_frame,
    demodulate_data_bits, estimate_ls_linear_channel, extract_reference_data_bits,
    generate_master_qpsk, receive_perfect_and_ls_linear,
)
from src.metrics import MetricAccumulator, channel_error_energies, count_bit_errors
from src.numerics import complex_gaussian_noise, noise_variance_from_snr
from src.pilots import build_fdm_pilot_layout, count_layout_resources
from src.random_channels import build_random_mimo_channel


NUM_TRAJECTORIES = 200
CHANNEL_SEED_BASE = 40261001


def verify_trajectory_aggregation(aggregates: list, trajectories: list, count: int = 200) -> bool:
    """Reconstruct pooled metrics from sufficient statistics, retaining every ID."""
    if len(aggregates) != 20 or len(trajectories) != count * 20:
        return False
    keys = {(r["trajectory_id"], r["snr_db"], r["dF"]) for r in trajectories}
    expected = {(tid, snr, df) for tid in range(count) for snr in SNR_POINTS for df in DF_OPTIONS}
    if keys != expected:
        return False
    for row in aggregates:
        selected = [r for r in trajectories if (r["snr_db"], r["dF"]) == (row["snr_db"], row["dF"])]
        if len(selected) != count or row["num_trajectories"] != count:
            return False
        error = sum(r["nmse_error_energy"] for r in selected)
        truth = sum(r["nmse_truth_energy"] for r in selected)
        bits = sum(r["data_bits"] for r in selected)
        if row["total_data_bits"] != bits:
            return False
        if not np.allclose([row["nmse_error_energy"], row["nmse_truth_energy"], row["nmse_data_linear"], row["nmse_data_db"]],
                           [error, truth, error / truth, 10 * np.log10(error / truth)], rtol=1e-12, atol=1e-12):
            return False
        for prefix in ("perfect_csi", "ls_lin"):
            errors = sum(r[f"{prefix}_bit_errors"] for r in selected)
            if row[f"{prefix}_bit_errors"] != errors or row[f"{prefix}_ber"] != errors / bits:
                return False
        correct, rate = correct_bit_rate_proxy(bits / count, row["ls_lin_ber"], frame_duration_seconds())
        if not np.allclose([row["correct_bits_per_frame_proxy"], row["correct_bit_rate_proxy_bps"]], [correct, rate], rtol=1e-12):
            return False
    return True


def fixed_channel_regressions(t06b_baseline: dict, t07a_frames: list) -> dict:
    """Use accepted T06B and one saved T07A frame per condition as references."""
    from src.awgn_scan import run_scan, zero_noise_regression
    zero = zero_noise_regression(t06b_baseline)
    _, small_frames, _ = run_scan(num_frames=1)
    reference = {(int(r["dF"]), int(r["snr_db"])): r for r in t07a_frames if int(r["frame_index"]) == 0}
    checks = []
    for current in small_frames:
        expected = reference[current["dF"], current["snr_db"]]
        passed = bool(all(current[field] == int(expected[field]) for field in
                          ("data_bits", "perfect_csi_bit_errors", "ls_lin_bit_errors", "bits_seed", "bits_seed_tx1", "noise_seed"))
                      and all(np.isclose(current[field], float(expected[field]), rtol=1e-12, atol=1e-12)
                              for field in ("nmse_error_energy", "nmse_truth_energy")))
        checks.append({"dF": current["dF"], "snr_db": current["snr_db"], "passed": passed})
    return {"t06b_zero_noise": zero, "t07a_one_frame_per_condition": checks,
            "passed": bool(zero["passed"] and len(checks) == 20 and all(r["passed"] for r in checks))}


class ScanFailure(ValueError):
    """Numerical failure with reproducible trajectory/RE context; no retry/filter."""

    def __init__(self, record: dict):
        self.record = record
        super().__init__(str(record))


def trajectory_seeds(trajectory_id: int, snr_index: int) -> tuple[int, tuple[int, int], int]:
    if not isinstance(trajectory_id, (int, np.integer)) or isinstance(trajectory_id, bool) or not 0 <= trajectory_id < NUM_TRAJECTORIES:
        raise ValueError("trajectory_id must be an integer in [0,199]")
    if not isinstance(snr_index, (int, np.integer)) or isinstance(snr_index, bool) or not 0 <= snr_index < 5:
        raise ValueError("snr_index must be an integer in [0,4]")
    bits0 = BITS_SEED_BASE + 2 * trajectory_id
    return CHANNEL_SEED_BASE + trajectory_id, (bits0, bits0 + 1), NOISE_SEED_BASE + snr_index * 100000 + trajectory_id


def frame_duration_seconds(num_symbols=16, nfft=64, cp_len=16, delta_f_hz=30000) -> float:
    if num_symbols <= 0 or nfft <= 0 or not 0 <= cp_len <= nfft or delta_f_hz <= 0:
        raise ValueError("frame parameters must define a positive duration")
    return float(num_symbols * (1 / delta_f_hz + (cp_len / nfft) / delta_f_hz))


def correct_bit_rate_proxy(data_bits_per_frame: float, ber: float, duration: float) -> tuple[float, float]:
    if not np.all(np.isfinite([data_bits_per_frame, ber, duration])) or data_bits_per_frame <= 0 or duration <= 0 or not 0 <= ber <= 1:
        raise ValueError("proxy needs positive bits/duration and BER in [0,1]")
    correct_bits = float(data_bits_per_frame * (1 - ber))
    return correct_bits, correct_bits / duration


def run_random_scan(num_trajectories: int = NUM_TRAJECTORIES, progress=None) -> tuple[list, list, dict]:
    """Small runs support tests/benchmark; official entry always uses 200 trajectories."""
    if isinstance(num_trajectories, bool) or not isinstance(num_trajectories, int) or not 1 <= num_trajectories <= NUM_TRAJECTORIES:
        raise ValueError("num_trajectories must be an integer in [1,200]")
    layouts = {df: build_fdm_pilot_layout(2, 64, 16, df, 4) for df in DF_OPTIONS}
    resources = {df: count_layout_resources(layout) for df, layout in layouts.items()}
    accumulators = {(df, snr): MetricAccumulator() for df in DF_OPTIONS for snr in SNR_POINTS}
    trajectory_rows, link_powers, all_conditions = [], [], []
    noise_energy = {snr: 0.0 for snr in SNR_POINTS}
    noise_samples = {snr: 0 for snr in SNR_POINTS}
    for trajectory_id in range(num_trajectories):
        channel_seed, bits_seeds, _ = trajectory_seeds(trajectory_id, 0)
        impulse, h_true = build_random_mimo_channel(channel_seed)
        powers = np.sum(np.abs(impulse) ** 2, axis=-1)
        link_powers.append(powers)
        conditions = np.array([np.linalg.cond(h_true[:, :, k, 0]) for k in range(64)])
        all_conditions.append(conditions)
        for k in range(64):
            if not np.isfinite(conditions[k]) or np.linalg.matrix_rank(h_true[:, :, k, 0]) != 2:
                raise ScanFailure({"trajectory_id": trajectory_id, "channel_seed": channel_seed,
                                   "k": k, "condition_number": float(conditions[k]) if np.isfinite(conditions[k]) else None,
                                   "condition_number_text": str(conditions[k]), "stage": "true_channel_rank",
                                   "reason": "numerically singular true channel", "completed_evaluations": len(trajectory_rows)})
        summary = {"condition_number_min": float(np.min(conditions)),
                   "condition_number_median": float(np.median(conditions)),
                   "condition_number_max": float(np.max(conditions))}
        bits_master, s_master = generate_master_qpsk(bits_seeds=bits_seeds)
        for snr_index, snr in enumerate(SNR_POINTS):
            _, _, noise_seed = trajectory_seeds(trajectory_id, snr_index)
            variance = noise_variance_from_snr(snr, 1.0)
            noise_master = complex_gaussian_noise((2, 64, 16), variance, np.random.default_rng(noise_seed))
            noise_energy[snr] += float(np.sum(np.abs(noise_master) ** 2))
            noise_samples[snr] += noise_master.size
            for df in DF_OPTIONS:
                layout = layouts[df]
                x = assemble_pilot_data_frame(s_master, layout)
                y_noisy = add_received_noise(apply_frequency_mimo_channel(h_true, x), noise_master)
                try:
                    _, h_est, perfect_x, estimated_x = receive_perfect_and_ls_linear(y_noisy, h_true, layout)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    # detect_data_zf packs selected REs into its frequency axis.
                    match = re.search(r"k=(\d+), t=(\d+)", str(exc))
                    coordinate = np.argwhere(layout.data_mask[0])[int(match.group(1))].tolist() if match else [None, None]
                    k, t = coordinate
                    estimated_condition = None
                    if k is not None:
                        _, failed_estimate = estimate_ls_linear_channel(y_noisy, layout)
                        value = np.linalg.cond(failed_estimate[:, :, k, t])
                        estimated_condition = float(value) if np.isfinite(value) else None
                    raise ScanFailure({"trajectory_id": trajectory_id, "channel_seed": channel_seed,
                                       "bits_seed_tx0": bits_seeds[0], "bits_seed_tx1": bits_seeds[1],
                                       "noise_seed": noise_seed, "snr_db": snr, "dF": df,
                                       "k": k, "t": t, "condition_number": float(conditions[k]) if k is not None else None,
                                       "estimated_condition_number": estimated_condition, "stage": "receiver_solve",
                                       "reason": str(exc), "completed_evaluations": len(trajectory_rows)}) from exc
                reference = extract_reference_data_bits(bits_master, layout.data_mask)
                perfect_errors, data_bits = count_bit_errors(demodulate_data_bits(np.sqrt(2) * perfect_x, layout.data_mask), reference)
                ls_errors, ls_bits = count_bit_errors(demodulate_data_bits(np.sqrt(2) * estimated_x, layout.data_mask), reference)
                if data_bits != 2 * int(np.count_nonzero(layout.data_mask)) or ls_bits != data_bits:
                    raise ValueError("data-only bit accounting mismatch")
                error, truth = channel_error_energies(h_est, h_true, layout.data_mask)
                accumulators[df, snr].add(error, truth, perfect_errors, ls_errors, data_bits)
                trajectory_rows.append({
                    "trajectory_id": trajectory_id, "channel_seed": channel_seed, "snr_db": snr, "dF": df,
                    "pilot_ratio": resources[df]["union_ratio"], "bits_seed_tx0": bits_seeds[0],
                    "bits_seed_tx1": bits_seeds[1], "noise_seed": noise_seed, "data_bits": data_bits,
                    "nmse_error_energy": error, "nmse_truth_energy": truth,
                    "perfect_csi_bit_errors": perfect_errors, "ls_lin_bit_errors": ls_errors,
                    **{f"channel_power_rx{r}_tx{q}": float(powers[r, q]) for r in range(2) for q in range(2)}, **summary,
                })
        if progress is not None and (trajectory_id + 1) % 25 == 0:
            progress(f"Completed {trajectory_id + 1}/{num_trajectories} independent trajectories")

    noise_checks = []
    for snr in SNR_POINTS:
        variance = noise_variance_from_snr(snr, 1.0)
        empirical = noise_energy[snr] / noise_samples[snr]
        deviation = abs(empirical - variance) / variance
        noise_checks.append({"snr_db": snr, "target_noise_variance": variance, "empirical_noise_power": empirical,
                             "relative_error": deviation, "unique_complex_samples": noise_samples[snr],
                             "passed": bool(deviation <= 0.02)})
    rows = []
    for df in DF_OPTIONS:
        for snr in SNR_POINTS:
            metrics = accumulators[df, snr].finalize()
            actual_trajectories = metrics.pop("num_frames")
            bits_per_frame = metrics["total_data_bits"] / actual_trajectories
            correct_bits, correct_rate = correct_bit_rate_proxy(bits_per_frame, metrics["ls_lin_ber"], frame_duration_seconds())
            passed = bool(actual_trajectories == num_trajectories
                          and metrics["total_data_bits"] == num_trajectories * 2 * resources[df]["two_stream_data_symbols"]
                          and all(np.isfinite(value) for value in metrics.values())
                          and all(check["passed"] for check in noise_checks))
            rows.append({"dF": df, "pilot_ratio": resources[df]["union_ratio"], "snr_db": snr,
                         "num_trajectories": actual_trajectories, "data_bits_per_frame": int(bits_per_frame),
                         "noise_variance": noise_variance_from_snr(snr, 1.0), **metrics,
                         "correct_bits_per_frame_proxy": correct_bits, "correct_bit_rate_proxy_bps": correct_rate,
                         "all_checks_passed": passed})
    powers_array, condition_array = np.array(link_powers), np.array(all_conditions)
    stats = {f"rx{r}_tx{q}": {"mean": float(np.mean(powers_array[:, r, q])),
                              "std": float(np.std(powers_array[:, r, q], ddof=1)) if num_trajectories > 1 else 0.0,
                              "min": float(np.min(powers_array[:, r, q])), "max": float(np.max(powers_array[:, r, q]))}
             for r in range(2) for q in range(2)}
    worst = np.unravel_index(np.argmax(condition_array), condition_array.shape)
    metadata = {"channel_power_statistics": stats, "channel_power_statistic_unit": "trajectory; sample std ddof=1",
                "condition_number_statistics": {**{f"P{p}": float(np.percentile(condition_array, p)) for p in (50, 90, 99)},
                                                "max": float(np.max(condition_array)), "bin_count": int(condition_array.size),
                                                "worst_trajectory_id": int(worst[0]), "worst_k": int(worst[1]),
                                                "worst_channel_seed": CHANNEL_SEED_BASE + int(worst[0])},
                "noise_power_checks": noise_checks, "numerical_failures": [], "filtered_trajectories": 0}
    return rows, trajectory_rows, metadata
