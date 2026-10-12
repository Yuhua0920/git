"""T09B paired dynamic TDL-A baseline with ensemble-defined AWGN."""

import hashlib

import numpy as np

from src.detectors import zf_detect
from src.doppler import build_tdl_a_doppler_channel, mobility_parameters
from src.link import (
    PILOT_SYMBOLS, add_received_noise, apply_frequency_mimo_channel,
    assemble_pilot_data_frame, demodulate_data_bits, estimate_ls_linear_channel,
    extract_reference_data_bits, generate_master_qpsk,
)
from src.metrics import MetricAccumulator, channel_error_energies, count_bit_errors
from src.numerics import complex_gaussian_noise, noise_variance_from_snr
from src.pilots import build_fdm_pilot_layout, count_layout_resources
from src.stochastic_scan import correct_bit_rate_proxy, frame_duration_seconds
from src.tdl_noiseless_scan import (
    BITS_SEED_BASE, CHANNEL_SEED_BASE, NUM_TRAJECTORIES, SPACINGS,
    NoiselessLinkFailure, complex_record, condition_numbers, failure_context, safe_detect,
)


SNR_POINTS = (0, 5, 10, 15, 20)
NOISE_SEED_BASE = 30261001
# Fixed before the formal experiment; small timing runs do not redefine these.
PILOT_MSE_RELATIVE_TOLERANCE = 0.05
NOISE_POWER_RELATIVE_TOLERANCE = 0.02
REGRESSION_RTOL = 1e-12
REGRESSION_ATOL = 1e-15


def scale_unit_noise(unit_noise: np.ndarray, variance: float) -> np.ndarray:
    """Scale a fixed CN(0,1) realization; no signal/channel power input."""
    z = np.asarray(unit_noise, dtype=np.complex128)
    if z.shape != (2, 64, 16) or not np.all(np.isfinite(z)):
        raise ValueError("unit noise must be finite (2,64,16)")
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("variance must be finite and nonnegative")
    return np.sqrt(variance) * z


def manual_diagnostics(h_true, h_est, h_ls, x, y_clean, noise, y_noisy, layout) -> list[dict]:
    records = []
    for kind, k, t in (("pilot", 0, 0), ("interior_data", 1, 2), ("trailing_data", 1, 15)):
        owner = int(layout.pilot_owner[k, t])
        detections = {}
        for path, h_used in (("perfect_csi", h_true), ("ls_lin", h_est)):
            try:
                detections[path] = zf_detect(y_noisy[:, k:k+1, t:t+1], h_used[:, :, k:k+1, t:t+1])[:, 0, 0]
            except (ValueError, np.linalg.LinAlgError) as exc:
                record = failure_context(str(exc), 0, h_true, k, t, dF=4, x=x, h_used=h_used, path=path)
                record.update({"snr_db": 10, "noise_variance": 0.1, "noise_seed": NOISE_SEED_BASE,
                               "stage": "manual_trace_solve"})
                raise NoiselessLinkFailure(record) from exc
        pilot_error = h_ls[:, owner, k, t] - h_true[:, owner, k, t] if owner >= 0 else None
        records.append({
            "trajectory_id": 0, "channel_seed": CHANNEL_SEED_BASE, "noise_seed": NOISE_SEED_BASE,
            "dF": 4, "snr_db": 10, "kind": kind, "k": k, "t": t,
            "pilot_owner": owner if owner >= 0 else None,
            "H_true": complex_record(h_true[:, :, k, t]), "H_est": complex_record(h_est[:, :, k, t]),
            "X": complex_record(x[:, k, t]), "Y_clean": complex_record(y_clean[:, k, t]),
            "N": complex_record(noise[:, k, t]), "Y_noisy": complex_record(y_noisy[:, k, t]),
            "perfect_csi_X_hat": complex_record(detections["perfect_csi"]),
            "ls_lin_X_hat": complex_record(detections["ls_lin"]),
            "pilot_ls_error": complex_record(pilot_error) if owner >= 0 else None,
            "noise_divided_by_pilot": complex_record(noise[:, k, t] / PILOT_SYMBOLS[owner]) if owner >= 0 else None,
            "included_in_ber": owner < 0,
        })
    return records


def evaluate_condition(h_true, bits, s_master, noise, variance, layout, trajectory_id,
                       snr_db, dF, conditions) -> tuple[dict, list[dict]]:
    """A zero-variance regression enters this same receiver without drawing noise."""
    x = assemble_pilot_data_frame(s_master, layout)
    y_clean = apply_frequency_mimo_channel(h_true, x)
    if variance == 0:
        if noise is not None:
            raise ValueError("zero-noise regression requires noise=None")
        y_noisy = y_clean
    else:
        y_noisy = add_received_noise(y_clean, noise)
    try:
        # The estimator receives only observations, layout, and known pilots.
        h_ls, h_est = estimate_ls_linear_channel(y_noisy, layout, PILOT_SYMBOLS)
        if not np.all(np.isfinite(h_est)):
            k, t = np.argwhere(~np.all(np.isfinite(h_est), axis=(0, 1)))[0]
            raise NoiselessLinkFailure(failure_context(
                "nonfinite LS-LIN grid", trajectory_id, h_true, k, t, dF=dF, x=x, h_used=h_est,
            ))
        perfect_x = safe_detect(y_noisy, h_true, layout, h_true, x, trajectory_id, dF, "perfect_csi")
        ls_x = safe_detect(y_noisy, h_est, layout, h_true, x, trajectory_id, dF, "ls_lin")
    except NoiselessLinkFailure as exc:
        exc.record.update({"snr_db": snr_db, "noise_variance": variance,
                           "noise_seed": NOISE_SEED_BASE + trajectory_id if variance else None,
                           "bits_seed_tx0": BITS_SEED_BASE + 2 * trajectory_id,
                           "bits_seed_tx1": BITS_SEED_BASE + 2 * trajectory_id + 1})
        raise
    reference = extract_reference_data_bits(bits, layout.data_mask)
    perfect_errors, data_bits = count_bit_errors(demodulate_data_bits(np.sqrt(2) * perfect_x, layout.data_mask), reference)
    ls_errors, ls_bits = count_bit_errors(demodulate_data_bits(np.sqrt(2) * ls_x, layout.data_mask), reference)
    if data_bits != 2 * int(np.count_nonzero(layout.data_mask)) or ls_bits != data_bits:
        raise ValueError("data-only BER accounting mismatch")
    error, truth = channel_error_energies(h_est, h_true, layout.data_mask)
    observed = np.broadcast_to(layout.pilot_mask[None], h_true.shape)
    pilot_error = float(np.sum(np.abs(h_ls[observed] - h_true[observed]) ** 2))
    pilot_truth = float(np.sum(np.abs(h_true[observed]) ** 2))
    row = {
        "trajectory_id": trajectory_id, "channel_seed": CHANNEL_SEED_BASE + trajectory_id,
        "snr_db": snr_db, "noise_variance": variance,
        "noise_seed": NOISE_SEED_BASE + trajectory_id if variance else None,
        "dF": dF, "pilot_ratio": float(count_layout_resources(layout)["union_ratio"]),
        "bits_seed_tx0": BITS_SEED_BASE + 2 * trajectory_id,
        "bits_seed_tx1": BITS_SEED_BASE + 2 * trajectory_id + 1, "data_bits": data_bits,
        "pilot_ls_error_energy": pilot_error, "pilot_ls_sample_count": int(np.count_nonzero(observed)),
        "pilot_ls_truth_energy": pilot_truth, "nmse_error_energy": error, "nmse_truth_energy": truth,
        "perfect_csi_bit_errors": perfect_errors, "ls_lin_bit_errors": ls_errors,
        "perfect_csi_correct_bits": data_bits - perfect_errors, "ls_lin_correct_bits": data_bits - ls_errors,
        "condition_number_max": float(conditions.max()),
    }
    trace = manual_diagnostics(h_true, h_est, h_ls, x, y_clean, noise, y_noisy, layout) \
        if trajectory_id == 0 and dF == 4 and snr_db == 10 else []
    return row, trace


def sum_metrics(rows: list[dict]) -> dict:
    accumulator = MetricAccumulator()
    for row in rows:
        accumulator.add(row["nmse_error_energy"], row["nmse_truth_energy"],
                        row["perfect_csi_bit_errors"], row["ls_lin_bit_errors"], row["data_bits"])
    metrics = accumulator.finalize()
    metrics["num_trajectories"] = metrics.pop("num_frames")
    metrics["data_bits_per_trajectory"] = rows[0]["data_bits"]
    metrics["pilot_ls_error_energy"] = sum(row["pilot_ls_error_energy"] for row in rows)
    metrics["pilot_ls_sample_count"] = sum(row["pilot_ls_sample_count"] for row in rows)
    metrics["pilot_ls_truth_energy"] = sum(row["pilot_ls_truth_energy"] for row in rows)
    return metrics


def verify_noiseless_regression(actual: list[dict], baseline: list[dict]) -> dict:
    """Compare freshly executed zero-noise sufficient statistics to archived T09A."""
    expected = {int(row["dF"]): row for row in baseline}
    if set(expected) != set(SPACINGS) or len(actual) != 4:
        raise ValueError("T09A regression requires all four archived layouts")
    checks = []
    for row in actual:
        old = expected[row["dF"]]
        energy_fields = ("nmse_error_energy", "nmse_truth_energy", "nmse_data_linear", "nmse_data_db")
        matches = {name: bool(np.isclose(row[name], float(old[name]), rtol=REGRESSION_RTOL, atol=REGRESSION_ATOL))
                   for name in energy_fields}
        count_fields = ("num_trajectories", "data_bits_per_trajectory", "total_data_bits",
                        "perfect_csi_bit_errors", "ls_lin_bit_errors")
        matches.update({name: row[name] == int(old[name]) for name in count_fields})
        checks.append({"dF": row["dF"], "nmse_data_linear": row["nmse_data_linear"],
                       "baseline_nmse_data_linear": float(old["nmse_data_linear"]),
                       "nmse_data_db": row["nmse_data_db"], "ls_lin_bit_errors": row["ls_lin_bit_errors"],
                       "baseline_ls_lin_bit_errors": int(old["ls_lin_bit_errors"]),
                       "perfect_csi_bit_errors": row["perfect_csi_bit_errors"],
                       "field_checks": matches, "passed": all(matches.values()) and row["perfect_csi_bit_errors"] == 0})
    return {"noise_variance": 0.0, "num_trajectories": actual[0]["num_trajectories"],
            "source": "outputs/t09a_tdl_noiseless_scan.csv", "rtol": REGRESSION_RTOL, "atol": REGRESSION_ATOL,
            "same_noisy_scan_pipeline": True, "conditions": checks, "passed": all(row["passed"] for row in checks)}


def verify_pooled_results(aggregate: list[dict], trajectories: list[dict], count: int) -> bool:
    """Independently reconstruct energies, counts, MSE and both proxy formulas."""
    expected = {(tid, snr, dF) for tid in range(count) for snr in SNR_POINTS for dF in SPACINGS}
    if len(trajectories) != count * 20 or len(aggregate) != 20:
        return False
    if {(r["trajectory_id"], r["snr_db"], r["dF"]) for r in trajectories} != expected:
        return False
    for row in aggregate:
        group = [r for r in trajectories if (r["dF"], r["snr_db"]) == (row["dF"], row["snr_db"])]
        for field in ("nmse_error_energy", "nmse_truth_energy", "pilot_ls_error_energy"):
            if not np.isclose(row[field], sum(r[field] for r in group), rtol=1e-12, atol=1e-12):
                return False
        bits = sum(r["data_bits"] for r in group)
        samples = sum(r["pilot_ls_sample_count"] for r in group)
        if bits != row["total_data_bits"] or samples != row["pilot_ls_sample_count"]:
            return False
        if not np.isclose(row["nmse_data_linear"], row["nmse_error_energy"] / row["nmse_truth_energy"], rtol=1e-12):
            return False
        if not np.isclose(row["pilot_ls_mse"], row["pilot_ls_error_energy"] / samples, rtol=1e-12):
            return False
        for prefix in ("perfect_csi", "ls_lin"):
            errors = sum(r[f"{prefix}_bit_errors"] for r in group)
            if row[f"{prefix}_bit_errors"] != errors or row[f"{prefix}_ber"] != errors / bits:
                return False
            expected_proxy = (bits - errors) / (count * frame_duration_seconds())
            if not np.isclose(row[f"{prefix}_correct_bit_rate_proxy_bps"], expected_proxy, rtol=1e-12):
                return False
    return True


def run_tdl_awgn_scan(num_trajectories=NUM_TRAJECTORIES, *, include_noiseless_regression=True,
                      progress=None) -> tuple[dict, list[dict]]:
    if isinstance(num_trajectories, bool) or not isinstance(num_trajectories, int) or not 1 <= num_trajectories <= 200:
        raise ValueError("num_trajectories must be an integer in [1,200]")
    layouts = {dF: build_fdm_pilot_layout(2, 64, 16, dF, 4) for dF in SPACINGS}
    rows, floor_rows, manual, channels = [], [], [], []
    unit_energy, unit_count = 0.0, 0
    scaled_energy = {snr: 0.0 for snr in SNR_POINTS}
    for tid in range(num_trajectories):
        _, h_true = build_tdl_a_doppler_channel(CHANNEL_SEED_BASE + tid)
        conditions = condition_numbers(h_true, tid)
        channels.append({"trajectory_id": tid, "channel_seed": CHANNEL_SEED_BASE + tid,
                         "H_sha256_complex128_C_order": hashlib.sha256(h_true.tobytes(order="C")).hexdigest(),
                         "condition_number_max": float(conditions.max())})
        bits, s_master = generate_master_qpsk(bits_seeds=(BITS_SEED_BASE + 2*tid, BITS_SEED_BASE + 2*tid + 1))
        z = complex_gaussian_noise((2, 64, 16), 1.0, np.random.default_rng(NOISE_SEED_BASE + tid))
        unit_energy += float(np.sum(np.abs(z) ** 2))
        unit_count += z.size
        if include_noiseless_regression:
            for dF, layout in layouts.items():
                floor_row, _ = evaluate_condition(h_true, bits, s_master, None, 0.0, layout, tid, None, dF, conditions)
                floor_rows.append(floor_row)
        for snr in SNR_POINTS:
            variance = noise_variance_from_snr(snr, total_power=1.0)
            noise = scale_unit_noise(z, variance)
            scaled_energy[snr] += float(np.sum(np.abs(noise) ** 2))
            for dF, layout in layouts.items():
                row, trace = evaluate_condition(h_true, bits, s_master, noise, variance, layout, tid, snr, dF, conditions)
                rows.append(row)
                manual.extend(trace)
        if progress and (tid + 1) % 25 == 0:
            progress(tid + 1)
    noise_statistics = []
    for snr in SNR_POINTS:
        variance = noise_variance_from_snr(snr, 1.0)
        empirical = scaled_energy[snr] / unit_count
        noise_statistics.append({"snr_db": snr, "target_noise_variance": variance,
                                 "empirical_unit_noise_power": unit_energy / unit_count,
                                 "empirical_scaled_noise_power": empirical,
                                 "relative_error": abs(empirical - variance) / variance,
                                 "independent_noise_samples": unit_count})
    floor_aggregate = []
    if include_noiseless_regression:
        for dF in SPACINGS:
            metric = sum_metrics([row for row in floor_rows if row["dF"] == dF])
            floor_aggregate.append({"dF": dF, **metric})
    floors = {row["dF"]: row for row in floor_aggregate}
    aggregate, pilot_checks = [], []
    for dF in SPACINGS:
        for snr in SNR_POINTS:
            group = [row for row in rows if row["dF"] == dF and row["snr_db"] == snr]
            metrics = sum_metrics(group)
            variance = noise_variance_from_snr(snr, 1.0)
            mse = metrics["pilot_ls_error_energy"] / metrics["pilot_ls_sample_count"]
            mse_ratio = mse / variance
            pilot_passed = bool(abs(mse_ratio - 1) <= PILOT_MSE_RELATIVE_TOLERANCE)
            pilot_checks.append({"dF": dF, "snr_db": snr, "theoretical_mse": variance,
                                 "empirical_mse": mse, "relative_difference": abs(mse_ratio - 1),
                                 "sample_count": metrics["pilot_ls_sample_count"],
                                 "relative_tolerance": PILOT_MSE_RELATIVE_TOLERANCE, "passed": pilot_passed})
            perfect_proxy = correct_bit_rate_proxy(metrics["data_bits_per_trajectory"], metrics["perfect_csi_ber"], frame_duration_seconds())[1]
            ls_proxy = correct_bit_rate_proxy(metrics["data_bits_per_trajectory"], metrics["ls_lin_ber"], frame_duration_seconds())[1]
            floor = floors.get(dF)
            noise_passed = next(r["relative_error"] <= NOISE_POWER_RELATIVE_TOLERANCE for r in noise_statistics if r["snr_db"] == snr)
            aggregate.append({
                "dF": dF, "pilot_ratio": group[0]["pilot_ratio"], "snr_db": snr, "noise_variance": variance,
                "num_trajectories": metrics["num_trajectories"], "data_bits_per_trajectory": metrics["data_bits_per_trajectory"],
                "total_data_bits": metrics["total_data_bits"], "pilot_ls_error_energy": metrics["pilot_ls_error_energy"],
                "pilot_ls_sample_count": metrics["pilot_ls_sample_count"], "pilot_ls_mse": mse,
                "pilot_ls_mse_to_noise_ratio": mse_ratio, "nmse_error_energy": metrics["nmse_error_energy"],
                "nmse_truth_energy": metrics["nmse_truth_energy"], "nmse_data_linear": metrics["nmse_data_linear"],
                "nmse_data_db": metrics["nmse_data_db"],
                "noiseless_floor_nmse_linear": floor["nmse_data_linear"] if floor else None,
                "noiseless_floor_nmse_db": floor["nmse_data_db"] if floor else None,
                "nmse_gap_above_floor_db": metrics["nmse_data_db"] - floor["nmse_data_db"] if floor else None,
                "perfect_csi_bit_errors": metrics["perfect_csi_bit_errors"], "perfect_csi_ber": metrics["perfect_csi_ber"],
                "ls_lin_bit_errors": metrics["ls_lin_bit_errors"], "ls_lin_ber": metrics["ls_lin_ber"],
                "perfect_csi_correct_bit_rate_proxy_bps": perfect_proxy, "ls_lin_correct_bit_rate_proxy_bps": ls_proxy,
                "all_checks_passed": bool(pilot_passed and noise_passed),
            })
    aggregation_passed = verify_pooled_results(aggregate, rows, num_trajectories)
    report = {
        "task": "T09B", "experiment_type": "dynamic TDL-A AWGN paired baseline calibration",
        "channel_definition": {"profile": "TDL-A", "num_taps": 23, "rms_delay_spread_ns": 100,
                               "carrier_frequency_hz": 3500000000, "speed_kmh": 30,
                               "temporal_model": "classical/Jakes covariance", "fractional_delay_frequency_response": True,
                               "independent_antenna_pairs": True, "symbol_wise_quasi_static": True,
                               "ici_modeled": False, "H_shape": [2, 2, 64, 16], "dtype": "complex128",
                               "maximum_doppler_hz": mobility_parameters()["maximum_doppler_hz"]},
        "snr_definition": {"reference": "ensemble expected useful signal power per RX branch",
                           "expected_useful_power": 1.0, "data_stream_expected_power": 0.5,
                           "noise_variance_formula": "10**(-snr_db/10)", "complex_noise_power": "E[abs(N)**2]=sigma_n_squared",
                           "instantaneous_normalization": False, "noise_added_to": "entire RX grid, including pilots and data"},
        "num_trajectories": num_trajectories, "snr_points": list(SNR_POINTS),
        "pilot_densities": [{"dF": dF, "pilot_ratio": count_layout_resources(layout)["union_ratio"]}
                            for dF, layout in layouts.items()],
        "seed_policy": {"channel_seed": "40261001 + trajectory_id", "bits_seed_txq": "20261001 + 2*trajectory_id + q",
                        "noise_seed": "30261001 + trajectory_id", "training_seed": None,
                        "common_random_numbers": True, "noise_seed_depends_on_snr": False,
                        "unit_noise_shape": [2, 64, 16], "scaled_noise": "sqrt(sigma_n_squared) * Z",
                        "protocol_note": "The same unit-variance complex noise realization is scaled to each SNR for paired Monte Carlo comparison."},
        "common_random_numbers": True, "noise_statistics": noise_statistics,
        "noise_power_relative_tolerance": NOISE_POWER_RELATIVE_TOLERANCE,
        "pilot_ls_theoretical_mse_checks": pilot_checks, "zero_noise_aggregate_results": floor_aggregate,
        "aggregate_results": aggregate, "channel_realizations": channels, "manual_diagnostics": manual,
        "frame_duration_seconds": frame_duration_seconds(), "trajectory_rows": len(rows),
        "aggregate_rows": len(aggregate), "noise_rows": len(noise_statistics),
        "zero_noise_evaluations": len(floor_rows), "num_filtered_trajectories": 0,
        "numerical_failures": [], "ls_lin_grid_finite": True,
        "aggregation_checks_passed": aggregation_passed,
        "all_checks_passed": bool(aggregation_passed and all(row["all_checks_passed"] for row in aggregate)),
        "benchmark_scope": "baseline calibration / protocol validation; not final formal E1 benchmark",
    }
    return report, rows
