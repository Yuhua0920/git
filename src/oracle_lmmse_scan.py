"""T10B paired full-link integration of the frozen T10A Oracle estimator."""

import csv
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from src.detectors import zf_detect
from src.doppler import build_tdl_a_doppler_channel, mobility_parameters
from src.link import (PILOT_SYMBOLS, add_received_noise, apply_frequency_mimo_channel,
                      assemble_pilot_data_frame, demodulate_data_bits, estimate_ls_linear_channel,
                      extract_reference_data_bits, generate_master_qpsk)
from src.lmmse import (LMMSEPreparationError, build_frequency_covariance, build_full_grid_covariance,
                       build_time_covariance, estimate_mimo_from_pilot_ls,
                       pilot_flat_indices, prepare_oracle_lmmse)
from src.metrics import channel_error_energies, count_bit_errors
from src.numerics import complex_gaussian_noise, noise_variance_from_snr
from src.pilots import build_fdm_pilot_layout, count_layout_resources
from src.stochastic_scan import correct_bit_rate_proxy, frame_duration_seconds
from src.tdl_awgn_scan import (NOISE_SEED_BASE, REGRESSION_ATOL, REGRESSION_RTOL,
                               SNR_POINTS, scale_unit_noise)
from src.tdl_noiseless_scan import (BITS_SEED_BASE, CHANNEL_SEED_BASE, NUM_TRAJECTORIES,
                                   SPACINGS, NoiselessLinkFailure, complex_record,
                                   condition_numbers, failure_context, safe_detect)


THEORY_TOLERANCE_DB = 0.75
METHODS = ("perfect_csi", "ls_lin", "oracle_lmmse")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_theoretical_table(path: Path) -> dict:
    rows = read_csv(path)
    expected = {(d_f, snr) for d_f in SPACINGS for snr in SNR_POINTS}
    table = {(int(r["dF"]), int(r["snr_db"])): r for r in rows}
    if len(rows) != 20 or set(table) != expected:
        raise ValueError("T10A theory requires exactly 20 unique dF/SNR conditions")
    for (d_f, snr), row in table.items():
        variance = noise_variance_from_snr(snr, 1.0)
        nmse = float(row["posterior_nmse_data_combined"])
        if (float(row["noise_variance"]) != variance or not np.isfinite(nmse) or nmse <= 0
                or not np.isclose(float(row["posterior_nmse_data_db"]), 10 * np.log10(nmse), rtol=0, atol=1e-12)):
            raise ValueError(f"T10A theory metadata mismatch: dF={d_f}, SNR={snr}")
    return table


def prepare_estimators(layouts: dict, theory: dict) -> tuple[dict, dict]:
    """Build covariances once and 40 explicit factors before any trajectory."""
    started = perf_counter()
    full = build_full_grid_covariance(build_frequency_covariance(), build_time_covariance())
    cache, checks = {}, []
    for d_f, layout in layouts.items():
        indices = [pilot_flat_indices(layout.pilot_mask[q]) for q in range(2)]
        for snr in SNR_POINTS:
            variance = noise_variance_from_snr(snr, 1.0)
            data_sum, data_count = 0.0, 0
            for q in range(2):
                try:
                    obj = prepare_oracle_lmmse(full, indices[q], variance, (64, 16))
                except LMMSEPreparationError as exc:
                    exc.record.update({"dF": d_f, "snr_db": snr, "tx": q, "stage": "preparation"})
                    raise
                cache[d_f, snr, q] = obj
                selected = layout.data_mask[q]
                data_sum += float(np.sum(obj.posterior_variance()[selected]))
                data_count += int(np.count_nonzero(selected))
            actual = data_sum / data_count
            expected = float(theory[d_f, snr]["posterior_nmse_data_combined"])
            counts = count_layout_resources(layout)
            check = {"dF": d_f, "snr_db": snr, "recomputed_posterior_nmse_data": actual,
                "t10a_posterior_nmse_data": expected,
                "passed": bool(np.isclose(actual, expected, rtol=REGRESSION_RTOL, atol=REGRESSION_ATOL)
                    and float(theory[d_f, snr]["pilot_ratio"]) == counts["union_ratio"]
                    and all(int(theory[d_f, snr][f"pilot_count_tx{q}"]) == indices[q].size for q in range(2)))}
            checks.append(check)
    result = {"prepared_estimator_count": len(cache), "main_cache_factorization_count": len(cache),
              "theory_table_matching": checks, "passed": len(cache) == 40 and all(x["passed"] for x in checks),
              "runtime_seconds": perf_counter() - started}
    if not result["passed"]:
        raise NoiselessLinkFailure({"stage": "T10A theory preparation regression", "details": result})
    return cache, result


def oracle_channel_from_pilot_ls(pilot_ls, prepared_by_tx):
    """Blind receiver boundary: only real full-link Pilot-LS and fixed objects."""
    return estimate_mimo_from_pilot_ls(pilot_ls, prepared_by_tx)


def manual_diagnostics(h_true, h_ls_lin, h_oracle, x, y_clean, noise, y_noisy, layout):
    records = []
    for kind, k, t in (("pilot", 0, 0), ("interior_data", 1, 2), ("trailing_data", 1, 15)):
        detections = {}
        for method, channel in zip(METHODS, (h_true, h_ls_lin, h_oracle)):
            try:
                detections[method] = zf_detect(y_noisy[:, k:k+1, t:t+1], channel[:, :, k:k+1, t:t+1])[:, 0, 0]
            except (ValueError, np.linalg.LinAlgError) as exc:
                record = failure_context(str(exc), 0, h_true, k, t, dF=4, x=x, h_used=channel, path=method)
                record.update({"snr_db": 10, "noise_seed": NOISE_SEED_BASE, "stage": "manual diagnostic",
                               "method": method, "noise_variance": 0.1})
                raise NoiselessLinkFailure(record) from exc
        records.append({"trajectory_id": 0, "dF": 4, "snr_db": 10, "kind": kind, "k": k, "t": t,
            "H_true": complex_record(h_true[:, :, k, t]), "H_ls_lin": complex_record(h_ls_lin[:, :, k, t]),
            "H_oracle": complex_record(h_oracle[:, :, k, t]), "X_true": complex_record(x[:, k, t]),
            "Y_clean": complex_record(y_clean[:, k, t]), "N": complex_record(noise[:, k, t]),
            "Y_noisy": complex_record(y_noisy[:, k, t]),
            **{f"X_hat_{method}": complex_record(detections[method]) for method in METHODS},
            "absolute_channel_error_ls_lin": np.abs(h_ls_lin[:, :, k, t] - h_true[:, :, k, t]).tolist(),
            "absolute_channel_error_oracle": np.abs(h_oracle[:, :, k, t] - h_true[:, :, k, t]).tolist(),
            "included_in_ber": bool(layout.data_mask[0, k, t])})
    return records


def evaluate_condition(h_true, bits, s_master, noise, variance, layout, trajectory_id, snr_db, d_f, conditions, prepared_by_tx):
    """All three receivers share X/H/Yclean/N/Ynoisy; Oracle never sees truth."""
    x = assemble_pilot_data_frame(s_master, layout)
    y_clean = apply_frequency_mimo_channel(h_true, x)
    y_noisy = add_received_noise(y_clean, noise)
    try:
        h_ls, h_lin = estimate_ls_linear_channel(y_noisy, layout, PILOT_SYMBOLS)
        h_oracle = oracle_channel_from_pilot_ls(h_ls, prepared_by_tx)
        for method, channel in (("ls_lin", h_lin), ("oracle_lmmse", h_oracle)):
            if not np.all(np.isfinite(channel)):
                k, t = np.argwhere(~np.all(np.isfinite(channel), axis=(0, 1)))[0]
                raise NoiselessLinkFailure(failure_context("nonfinite channel estimate", trajectory_id,
                    h_true, k, t, dF=d_f, x=x, h_used=channel, path=method))
        detected = {method: safe_detect(y_noisy, channel, layout, h_true, x, trajectory_id, d_f, method)
                    for method, channel in zip(METHODS, (h_true, h_lin, h_oracle))}
    except NoiselessLinkFailure as exc:
        estimated_condition = float(np.linalg.cond(np.array(exc.record["H_used"]["real"])
            + 1j * np.array(exc.record["H_used"]["imag"]))) if exc.record.get("H_used") else None
        exc.record.update({"snr_db": snr_db, "noise_variance": variance,
            "noise_seed": NOISE_SEED_BASE + trajectory_id,
            "bits_seed_tx0": BITS_SEED_BASE + 2 * trajectory_id,
            "bits_seed_tx1": BITS_SEED_BASE + 2 * trajectory_id + 1,
            "method": exc.record.get("path"),
            "estimated_condition_number": estimated_condition if estimated_condition is not None and np.isfinite(estimated_condition) else None,
            "estimated_condition_number_nonfinite": estimated_condition is not None and not np.isfinite(estimated_condition)})
        raise
    reference = extract_reference_data_bits(bits, layout.data_mask)
    error_counts = {method: count_bit_errors(demodulate_data_bits(np.sqrt(2) * output, layout.data_mask), reference)
                    for method, output in detected.items()}
    data_bits = error_counts["perfect_csi"][1]
    if any(n != data_bits for _, n in error_counts.values()) or data_bits != 2 * np.count_nonzero(layout.data_mask):
        raise ValueError("data-only BER accounting mismatch")
    ls_error, truth = channel_error_energies(h_lin, h_true, layout.data_mask)
    oracle_error, oracle_truth = channel_error_energies(h_oracle, h_true, layout.data_mask)
    if truth != oracle_truth:
        raise ValueError("receiver truth-energy denominators differ")
    observed = np.broadcast_to(layout.pilot_mask[None], h_true.shape)
    row = {"trajectory_id": trajectory_id, "channel_seed": CHANNEL_SEED_BASE + trajectory_id,
        "snr_db": snr_db, "noise_variance": variance, "noise_seed": NOISE_SEED_BASE + trajectory_id,
        "dF": d_f, "pilot_ratio": float(count_layout_resources(layout)["union_ratio"]),
        "bits_seed_tx0": BITS_SEED_BASE + 2 * trajectory_id,
        "bits_seed_tx1": BITS_SEED_BASE + 2 * trajectory_id + 1, "data_bits": data_bits,
        "ls_nmse_error_energy": ls_error, "oracle_nmse_error_energy": oracle_error, "nmse_truth_energy": truth,
        **{f"{method}_bit_errors": error_counts[method][0] for method in METHODS},
        "condition_number_max": float(conditions.max()),
        "pilot_ls_error_energy": float(np.sum(np.abs(h_ls[observed] - h_true[observed]) ** 2)),
        "pilot_ls_sample_count": int(np.count_nonzero(observed))}
    trace = manual_diagnostics(h_true, h_lin, h_oracle, x, y_clean, noise, y_noisy, layout) \
        if trajectory_id == 0 and d_f == 4 and snr_db == 10 else []
    return row, trace


def aggregate_results(rows, theory):
    aggregate = []
    for d_f in SPACINGS:
        for snr in SNR_POINTS:
            group = [r for r in rows if (r["dF"], r["snr_db"]) == (d_f, snr)]
            bits = sum(r["data_bits"] for r in group)
            truth = sum(r["nmse_truth_energy"] for r in group)
            ls_error = sum(r["ls_nmse_error_energy"] for r in group)
            oracle_error = sum(r["oracle_nmse_error_energy"] for r in group)
            ls_nmse, oracle_nmse = ls_error / truth, oracle_error / truth
            ls_db, oracle_db = float(10 * np.log10(ls_nmse)), float(10 * np.log10(oracle_nmse))
            theory_linear = float(theory[d_f, snr]["posterior_nmse_data_combined"])
            theory_db = float(theory[d_f, snr]["posterior_nmse_data_db"])
            row = {"dF": d_f, "pilot_ratio": group[0]["pilot_ratio"], "snr_db": snr,
                "noise_variance": group[0]["noise_variance"], "num_trajectories": len(group), "total_data_bits": bits,
                "ls_lin_nmse_linear": ls_nmse, "ls_lin_nmse_db": ls_db,
                "oracle_lmmse_nmse_linear": oracle_nmse, "oracle_lmmse_nmse_db": oracle_db,
                "oracle_theoretical_nmse_linear": theory_linear, "oracle_theoretical_nmse_db": theory_db,
                "oracle_empirical_minus_theory_db": oracle_db - theory_db,
                "ls_lin_minus_oracle_nmse_db": ls_db - oracle_db}
            for method in METHODS:
                errors = sum(r[f"{method}_bit_errors"] for r in group)
                row[f"{method}_bit_errors"] = errors
                row[f"{method}_ber"] = errors / bits
                row[f"{method}_correct_bit_rate_proxy_bps"] = correct_bit_rate_proxy(
                    group[0]["data_bits"], errors / bits, frame_duration_seconds())[1]
            row.update({"ls_nmse_error_energy": ls_error, "oracle_nmse_error_energy": oracle_error,
                "nmse_truth_energy": truth, "data_bits_per_trajectory": group[0]["data_bits"],
                "pilot_ls_error_energy": sum(r["pilot_ls_error_energy"] for r in group),
                "pilot_ls_sample_count": sum(r["pilot_ls_sample_count"] for r in group),
                "all_checks_passed": abs(oracle_db - theory_db) <= THEORY_TOLERANCE_DB})
            aggregate.append(row)
    return aggregate


def verify_t09b_regression(actual, baseline):
    expected = {(int(r["dF"]), int(r["snr_db"])): r for r in baseline}
    if len(actual) != 20 or len(baseline) != 20 or len(expected) != 20:
        raise ValueError("T09B regression requires exactly 20 conditions")
    checks = []
    for row in actual:
        old = expected[row["dF"], row["snr_db"]]
        fields = {name: row[name] == int(old[name]) for name in (
            "num_trajectories", "data_bits_per_trajectory", "total_data_bits",
            "perfect_csi_bit_errors", "ls_lin_bit_errors", "pilot_ls_sample_count")}
        fields["noise_variance_exact"] = row["noise_variance"] == float(old["noise_variance"])
        energy_map = {"ls_nmse_error_energy": "nmse_error_energy", "nmse_truth_energy": "nmse_truth_energy",
                      "ls_lin_nmse_linear": "nmse_data_linear", "ls_lin_nmse_db": "nmse_data_db",
                      "pilot_ls_error_energy": "pilot_ls_error_energy"}
        energy_map.update({name: name for name in ("perfect_csi_ber", "ls_lin_ber",
            "perfect_csi_correct_bit_rate_proxy_bps", "ls_lin_correct_bit_rate_proxy_bps")})
        fields.update({name: bool(np.isclose(row[name], float(old[source]), rtol=REGRESSION_RTOL, atol=REGRESSION_ATOL))
                       for name, source in energy_map.items()})
        checks.append({"dF": row["dF"], "snr_db": row["snr_db"], "field_checks": fields, "passed": all(fields.values())})
    return {"source": "outputs/t09b_tdl_awgn_scan.csv", "rtol": REGRESSION_RTOL, "atol": REGRESSION_ATOL,
            "conditions": checks, "passed": all(x["passed"] for x in checks)}


def verify_pooled_results(aggregate, rows, count):
    expected = {(tid, snr, d_f) for tid in range(count) for snr in SNR_POINTS for d_f in SPACINGS}
    if len(rows) != count * 20 or len(aggregate) != 20 or {(r["trajectory_id"], r["snr_db"], r["dF"]) for r in rows} != expected:
        return False
    for row in aggregate:
        group = [r for r in rows if (r["dF"], r["snr_db"]) == (row["dF"], row["snr_db"])]
        bits = sum(r["data_bits"] for r in group)
        truth = sum(r["nmse_truth_energy"] for r in group)
        if bits != row["total_data_bits"] or truth != row["nmse_truth_energy"]:
            return False
        for method, energy in (("ls_lin", "ls_nmse_error_energy"), ("oracle_lmmse", "oracle_nmse_error_energy")):
            if row[energy] != sum(r[energy] for r in group) or row[f"{method}_nmse_linear"] != row[energy] / truth:
                return False
        for method in METHODS:
            errors = sum(r[f"{method}_bit_errors"] for r in group)
            if row[f"{method}_bit_errors"] != errors or row[f"{method}_ber"] != errors / bits:
                return False
            proxy = (bits - errors) / (count * frame_duration_seconds())
            if not np.isclose(row[f"{method}_correct_bit_rate_proxy_bps"], proxy, rtol=1e-12):
                return False
    return True


def run_oracle_lmmse_scan(theory, num_trajectories=NUM_TRAJECTORIES, progress=None):
    if isinstance(num_trajectories, bool) or not isinstance(num_trajectories, int) or not 1 <= num_trajectories <= 200:
        raise ValueError("num_trajectories must be an integer in [1,200]")
    started = perf_counter()
    layouts = {d_f: build_fdm_pilot_layout(2, 64, 16, d_f, 4) for d_f in SPACINGS}
    cache, preparation = prepare_estimators(layouts, theory)
    simulation_start = perf_counter()
    rows, manual, realizations = [], [], []
    for tid in range(num_trajectories):
        _, h_true = build_tdl_a_doppler_channel(CHANNEL_SEED_BASE + tid)
        conditions = condition_numbers(h_true, tid)
        bits, s_master = generate_master_qpsk(bits_seeds=(BITS_SEED_BASE + 2 * tid, BITS_SEED_BASE + 2 * tid + 1))
        unit_noise = complex_gaussian_noise((2, 64, 16), 1.0, np.random.default_rng(NOISE_SEED_BASE + tid))
        realizations.append({"trajectory_id": tid,
            "H_sha256_complex128_C_order": hashlib.sha256(h_true.tobytes(order="C")).hexdigest(),
            "bits_sha256_C_order": hashlib.sha256(bits.tobytes(order="C")).hexdigest(),
            "unit_noise_sha256_complex128_C_order": hashlib.sha256(unit_noise.tobytes(order="C")).hexdigest()})
        for snr in SNR_POINTS:
            variance = noise_variance_from_snr(snr, 1.0)
            noise = scale_unit_noise(unit_noise, variance)
            for d_f, layout in layouts.items():
                row, trace = evaluate_condition(h_true, bits, s_master, noise, variance, layout,
                    tid, snr, d_f, conditions, tuple(cache[d_f, snr, q] for q in range(2)))
                rows.append(row)
                manual.extend(trace)
        if progress and (tid + 1) % 25 == 0:
            progress(tid + 1)
    simulation_seconds = perf_counter() - simulation_start
    aggregate = aggregate_results(rows, theory)
    pooled = verify_pooled_results(aggregate, rows, num_trajectories)
    comparisons = [{"dF": r["dF"], "snr_db": r["snr_db"],
        "empirical_nmse_db": r["oracle_lmmse_nmse_db"], "theoretical_nmse_db": r["oracle_theoretical_nmse_db"],
        "empirical_minus_theory_db": r["oracle_empirical_minus_theory_db"], "absolute_tolerance_db": THEORY_TOLERANCE_DB,
        "passed": abs(r["oracle_empirical_minus_theory_db"]) <= THEORY_TOLERANCE_DB} for r in aggregate]
    return {"task": "T10B", "experiment_type": "Oracle LMMSE full-link AWGN baseline integration",
        "num_trajectories": num_trajectories, "common_random_numbers": True,
        "prepared_estimator_count": len(cache), "preparation": preparation,
        "oracle_covariance_source": "frozen src/lmmse.py: exact matched TDL-A PDP/fractional-delay Rf and verified T08B Jakes Rt",
        "oracle_noise_variance_source": "receiver-known fixed ensemble variance 10**(-snr_db/10)",
        "oracle_observation_path": "shared Y_noisy -> actual Pilot-LS -> frozen T10A estimate_mimo_from_pilot_ls",
        "oracle_role": "ideal matched-statistics traditional reference",
        "channel_definition": {"profile": "TDL-A", "num_taps": 23, "rms_delay_spread_ns": 100,
            "carrier_frequency_hz": 3.5e9, "speed_kmh": 30, "maximum_doppler_hz": mobility_parameters()["maximum_doppler_hz"],
            "temporal_model": "Jakes", "fractional_delay_frequency_response": True,
            "independent_antenna_pairs": True, "symbol_wise_quasi_static": True, "ICI_modeled": False,
            "H_shape": [2, 2, 64, 16], "dtype": "complex128"},
        "snr_definition": {"formula": "sigma_n_squared=10**(-snr_db/10)", "Ptot": 1.0,
            "per_stream_symbol_power": 0.5, "ensemble_reference_per_RX_branch": 1.0,
            "instantaneous_normalization": False, "frame_or_Y_normalization": False},
        "pilot_definition": {"dT": 4, "resources_from_actual_masks": True,
            "layouts": [{"dF": d_f, **count_layout_resources(layout)} for d_f, layout in layouts.items()]},
        "seed_policy": {"channel_seed": "40261001 + trajectory_id", "bits_seed_txq": "20261001 + 2*trajectory_id + q",
            "noise_seed": "30261001 + trajectory_id", "training_seed": None,
            "unit_noise_shape": [2, 64, 16], "scaled_noise": "sqrt(sigma_n_squared)*Z",
            "protocol_note": "The same unit-variance complex noise realization is scaled to each SNR for paired Monte Carlo comparison."},
        "same_y_for_all_methods": True, "oracle_full_grid_finite": True,
        "explicit_inverse_used": False, "pseudo_inverse_used": False, "jitter_used": False,
        "num_filtered_trajectories": 0, "numerical_failures": [], "aggregation_checks_passed": pooled,
        "trajectory_rows": len(rows), "aggregate_rows": len(aggregate), "frame_duration_seconds": frame_duration_seconds(),
        "t10a_theory_regression": {"table_matching_passed": preparation["passed"],
            "conditions": comparisons, "absolute_tolerance_db": THEORY_TOLERANCE_DB,
            "passed": all(x["passed"] for x in comparisons)},
        "aggregate_results": aggregate, "manual_diagnostics": manual, "realization_hashes": realizations,
        "runtime": {"preparation_seconds": preparation["runtime_seconds"], "simulation_seconds": simulation_seconds,
                    "total_scan_seconds": perf_counter() - started},
        "all_checks_passed": bool(pooled and all(x["passed"] for x in comparisons))}, rows
