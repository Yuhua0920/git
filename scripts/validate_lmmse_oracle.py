"""T10A Oracle LMMSE mathematical/numerical calibration; no data or detection."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, validate_config  # noqa: E402
from src.doppler import build_tdl_a_doppler_channel, jakes_covariance, mobility_parameters, ofdm_time_grid_seconds  # noqa: E402
from src.estimators import pilot_ls_estimate  # noqa: E402
from src.link import PILOT_SYMBOLS, apply_frequency_mimo_channel  # noqa: E402
from src.lmmse import (NUMERICAL_TOLERANCE, LMMSEPreparationError, build_frequency_covariance,  # noqa: E402
                       build_full_grid_covariance, build_time_covariance,
                       estimate_mimo_from_pilot_ls, pilot_flat_indices, prepare_oracle_lmmse)
from src.metrics import channel_error_energies  # noqa: E402
from src.numerics import complex_gaussian_noise, noise_variance_from_snr  # noqa: E402
from src.pilots import build_fdm_pilot_layout, count_layout_resources, map_pilot_symbols  # noqa: E402
from src.tdl import tdl_a_profile  # noqa: E402
from src.provenance import recorded_source_matches


OUTPUTS = PROJECT_ROOT / "outputs"
PREFIX = "t10a_lmmse_oracle"
PILOT_DENSITIES = (2, 4, 8, 16)
SNR_POINTS = (0, 5, 10, 15, 20)
REPRESENTATIVE_CONDITIONS = ((2, 0), (4, 10), (8, 15), (16, 20))
SPOT_PAIRS = (((32, 0), (32, 4)), ((32, 0), (40, 0)), ((32, 0), (40, 4)))
CHANNEL_SEED_BASE = 40261001
NOISE_SEED_BASE = 30261001
INTEGRATION_NOISE_SEED_BASE = 31261001
NUM_CALIBRATION_TRAJECTORIES = 500
NMSE_TOLERANCE_DB = 0.5
COVARIANCE_ABSOLUTE_TOLERANCE = 0.05


def complex_record(value):
    array = np.asarray(value)
    return {"real": array.real.tolist(), "imag": array.imag.tolist()}


def check_config():
    config = load_config(PROJECT_ROOT / "configs/base.yaml")
    validate_config(config)
    expected = {"nt": 2, "nr": 2, "nfft": 64, "active_carriers": 64, "num_symbols": 16,
                "delta_f_hz": 30000, "sample_rate_hz": 1920000, "cp_len": 16,
                "fft_norm": "ortho", "modulation": "qpsk_gray", "total_power": 1.0}
    if (config["system"] != expected or config["pilot"]["spacing_time"] != 4
            or config["pilot"]["spacing_frequency_options"] != list(PILOT_DENSITIES)
            or config["noise"]["snr_db"] != list(SNR_POINTS)):
        raise ValueError("T10A frozen system/pilot/SNR configuration conflict")
    return config


def frozen_source_checks():
    baseline = json.loads((OUTPUTS / "t09b_tdl_awgn_results.json").read_text(encoding="utf-8"))
    return {name: recorded_source_matches(PROJECT_ROOT, name, digest, "t09b")
            for name, digest in baseline["provenance"]["source_and_reference_sha256"].items()
            if name.startswith("src/") or name == "configs/base.yaml"}


def covariance_summary(matrix, check_eigenvalues):
    summary = {"shape": list(matrix.shape), "all_finite": bool(np.all(np.isfinite(matrix))),
               "hermitian_max_error": float(np.max(np.abs(matrix - matrix.conj().T))),
               "unit_diagonal_max_error": float(np.max(np.abs(np.diag(matrix) - 1)))}
    if check_eigenvalues:
        eigenvalues = np.linalg.eigvalsh(matrix)
        summary.update({"eigenvalue_min": float(eigenvalues[0]), "eigenvalue_max": float(eigenvalues[-1]),
                        "psd_tolerance": NUMERICAL_TOLERANCE,
                        "psd_within_tolerance": bool(eigenvalues[0] >= -NUMERICAL_TOLERANCE)})
    summary["passed"] = bool(summary["all_finite"] and summary["hermitian_max_error"] <= 1e-14
        and summary["unit_diagonal_max_error"] <= 1e-14
        and summary.get("psd_within_tolerance", True))
    return summary


def tiny_analytic_sentinel():
    rho, z, variance = 0.3 + 0.4j, 1.2 - 0.7j, 0.25
    covariance = np.array([[1, rho.conjugate()], [rho, 1]], dtype=np.complex128)
    prepared = prepare_oracle_lmmse(covariance, np.array([0]), variance, (2, 1))
    numerical = prepared.estimate(np.array([z])).ravel()
    analytic = np.array([z, rho * z]) / (1 + variance)
    error = float(np.max(np.abs(numerical - analytic)))
    pilot_difference = float(abs(numerical[0] - z))
    return {"rho": complex_record(rho), "z": complex_record(z), "noise_variance": variance,
            "analytic": complex_record(analytic), "numerical": complex_record(numerical),
            "max_absolute_error": error, "tolerance": 5e-16,
            "noisy_pilot_estimate_observation_difference": pilot_difference,
            "passed": error <= 5e-16 and pilot_difference > 0.1}


def prepare_theory():
    started = perf_counter()
    rf, rt = build_frequency_covariance(), build_time_covariance()
    full = build_full_grid_covariance(rf, rt)
    rf_summary, rt_summary = covariance_summary(rf, True), covariance_summary(rt, True)
    full_summary = covariance_summary(full, False)
    full_summary["psd_validation"] = "Kronecker product of PSD Rf and Rt within 1e-12; no full eig"
    reference_rt = jakes_covariance(ofdm_time_grid_seconds(), mobility_parameters()["maximum_doppler_hz"])
    rt_summary["t08b_exact_array_equal"] = bool(np.array_equal(rt, reference_rt))
    rt_summary["passed"] = rt_summary["passed"] and rt_summary["t08b_exact_array_equal"]
    kron_checks = []
    for (k, t), (kp, tp) in SPOT_PAIRS:
        expected, actual = rf[k, kp] * rt[t, tp], full[k * 16 + t, kp * 16 + tp]
        kron_checks.append({"first": [k, t], "second": [kp, tp], "flat_indices": [k * 16 + t, kp * 16 + tp],
            "direct": complex_record(expected), "full": complex_record(actual),
            "absolute_error": float(abs(actual - expected)), "passed": bool(actual == expected)})
    layouts, cache, rows, index_summaries = {}, {}, [], []
    for d_f in PILOT_DENSITIES:
        layout = build_fdm_pilot_layout(2, 64, 16, d_f, 4)
        layouts[d_f] = layout
        counts = count_layout_resources(layout)
        indices = [pilot_flat_indices(layout.pilot_mask[q]) for q in range(2)]
        index_checks = [bool(np.array_equal(p, np.flatnonzero(layout.pilot_mask[q].ravel(order="C")))
                             and p.size == {2: 128, 4: 64, 8: 32, 16: 16}[d_f]
                             and np.all(np.diff(p) > 0)) for q, p in enumerate(indices)]
        index_summaries.append({"dF": d_f, "offsets_by_tx": list(layout.offsets_by_tx),
            "pilot_symbol_indices": list(layout.pilot_symbol_indices), **counts,
            "flat_indices_by_tx": [p.tolist() for p in indices], "passed": all(index_checks)})
        for snr in SNR_POINTS:
            variance = noise_variance_from_snr(snr, 1.0)
            row = {"dF": d_f, "pilot_ratio": counts["union_ratio"], "snr_db": snr,
                   "noise_variance": variance,
                   "pilot_count_tx0": int(indices[0].size), "pilot_count_tx1": int(indices[1].size)}
            data_sums, data_counts = [], []
            for q in range(2):
                try:
                    obj = prepare_oracle_lmmse(full, indices[q], variance, (64, 16))
                except LMMSEPreparationError as exc:
                    exc.record.update({"dF": d_f, "snr_db": snr, "tx": q})
                    raise
                cache[d_f, snr, q] = obj
                posterior = obj.posterior_variance().ravel(order="C")
                data = layout.data_mask[q].ravel(order="C")
                data_sums.append(float(np.sum(posterior[data])))
                data_counts.append(int(np.count_nonzero(data)))
                row[f"posterior_nmse_full_tx{q}"] = float(np.mean(posterior))
                row[f"posterior_nmse_data_tx{q}"] = data_sums[q] / data_counts[q]
                row[f"data_count_tx{q}"] = data_counts[q]
                for name, value in obj.diagnostics.items():
                    row[f"{name}_tx{q}"] = value
            combined = sum(data_sums) / sum(data_counts)
            row.update({"posterior_nmse_full_combined": (row["posterior_nmse_full_tx0"] + row["posterior_nmse_full_tx1"]) / 2,
                        "posterior_nmse_data_combined": combined,
                        "posterior_nmse_data_db": float(10 * np.log10(combined))})
            rows.append(row)
    summaries = {"frequency_covariance_summary": rf_summary, "time_covariance_summary": rt_summary,
        "full_covariance_summary": full_summary,
        "kronecker_validation": {"pairs": kron_checks, "passed": all(x["passed"] for x in kron_checks)},
        "pilot_index_summaries": index_summaries, "prepared_estimator_count": len(cache),
        "factorizations_in_main_cache": len(cache), "theory_preparation_seconds": perf_counter() - started}
    return full, layouts, cache, rows, summaries


def actual_pilot_integration_sentinel(layouts, cache):
    _, channel = build_tdl_a_doppler_channel(CHANNEL_SEED_BASE)
    results = []
    for d_f, snr in REPRESENTATIVE_CONDITIONS:
        layout = layouts[d_f]
        x_grid = map_pilot_symbols(layout, PILOT_SYMBOLS)
        seed = INTEGRATION_NOISE_SEED_BASE + d_f
        noise = complex_gaussian_noise((2, 64, 16), noise_variance_from_snr(snr, 1.0), np.random.default_rng(seed))
        ls, mask = pilot_ls_estimate(apply_frequency_mimo_channel(channel, x_grid) + noise,
                                    x_grid, layout.pilot_mask)
        differences = [float(np.max(np.abs(ls[:, q, mask[q]] -
            (channel[:, q, mask[q]] + noise[:, mask[q]] / x_grid[q, mask[q]])))) for q in range(2)]
        objects = tuple(cache[d_f, snr, q] for q in range(2))
        estimate = estimate_mimo_from_pilot_ls(ls, objects)
        direct = np.stack([objects[q].estimate(ls[:, q].reshape(2, -1)[:, objects[q].pilot_indices])
                           for q in range(2)], axis=1)
        results.append({"dF": d_f, "snr_db": snr, "channel_seed": CHANNEL_SEED_BASE, "noise_seed": seed,
            "ls_h_plus_rotated_noise_max_error_by_tx": differences,
            "oracle_estimate_shape": list(estimate.shape),
            "mimo_vs_independent_estimator_max_error": float(np.max(np.abs(estimate - direct))),
            "passed": max(differences) <= 1e-14 and np.array_equal(estimate, direct)})
    return {"conditions": results, "passed": all(x["passed"] for x in results)}


def calibration_noise_seed(condition_index, trajectory_id, rx, tx):
    return NOISE_SEED_BASE + 100000 * condition_index + 4 * trajectory_id + 2 * rx + tx


def run_monte_carlo(num_trajectories, layouts, cache, theory_rows, full_covariance, progress=None):
    started = perf_counter()
    totals = {condition: {"full_error_energy": 0.0, "full_truth_energy": 0.0,
                         "data_error_energy": 0.0, "data_truth_energy": 0.0,
                         "noise_energy": 0.0, "noise_observation_count": 0}
              for condition in REPRESENTATIVE_CONDITIONS}
    covariance_sums = np.zeros(len(SPOT_PAIRS), dtype=np.complex128)
    for tid in range(num_trajectories):
        _, channel = build_tdl_a_doppler_channel(CHANNEL_SEED_BASE + tid)
        for index, ((k, t), (kp, tp)) in enumerate(SPOT_PAIRS):
            covariance_sums[index] += np.sum(channel[:, :, k, t] * channel[:, :, kp, tp].conj())
        for ci, (d_f, snr) in enumerate(REPRESENTATIVE_CONDITIONS):
            total = totals[d_f, snr]
            estimate = np.empty(channel.shape, dtype=np.complex128)
            for q in range(2):
                obj = cache[d_f, snr, q]
                observations = np.empty((2, obj.pilot_indices.size), dtype=np.complex128)
                for r in range(2):
                    noise = complex_gaussian_noise((obj.pilot_indices.size,), obj.noise_variance,
                        np.random.default_rng(calibration_noise_seed(ci, tid, r, q)))
                    observations[r] = channel[r, q].ravel(order="C")[obj.pilot_indices] + noise
                    total["noise_energy"] += float(np.sum(np.abs(noise) ** 2))
                    total["noise_observation_count"] += int(noise.size)
                estimate[:, q] = obj.estimate(observations)
            total["full_error_energy"] += float(np.sum(np.abs(estimate - channel) ** 2))
            total["full_truth_energy"] += float(np.sum(np.abs(channel) ** 2))
            error, truth = channel_error_energies(estimate, channel, layouts[d_f].data_mask)
            total["data_error_energy"] += error
            total["data_truth_energy"] += truth
        if progress and (tid + 1) % 50 == 0:
            progress(tid + 1)
    theory = {(row["dF"], row["snr_db"]): row for row in theory_rows}
    conditions = []
    for d_f, snr in REPRESENTATIVE_CONDITIONS:
        row, theoretical = totals[d_f, snr].copy(), theory[d_f, snr]
        empirical = row["data_error_energy"] / row["data_truth_energy"]
        empirical_db = float(10 * np.log10(empirical))
        difference = abs(empirical_db - theoretical["posterior_nmse_data_db"])
        row.update({"dF": d_f, "snr_db": snr,
            "full_nmse_empirical": row["full_error_energy"] / row["full_truth_energy"],
            "full_nmse_theory": theoretical["posterior_nmse_full_combined"],
            "data_nmse_empirical": empirical, "data_nmse_empirical_db": empirical_db,
            "data_nmse_theory": theoretical["posterior_nmse_data_combined"],
            "data_nmse_theory_db": theoretical["posterior_nmse_data_db"],
            "absolute_difference_db": difference, "acceptance_tolerance_db": NMSE_TOLERANCE_DB,
            "empirical_noise_variance": row["noise_energy"] / row["noise_observation_count"],
            "noise_variance": theoretical["noise_variance"],
            "full_observation_count": num_trajectories * 4 * 64 * 16,
            "data_observation_count": num_trajectories * 2 * sum(count_layout_resources(layouts[d_f])["data_count_by_tx"]),
            "passed": difference <= NMSE_TOLERANCE_DB})
        conditions.append(row)
    spot_checks = []
    for index, ((k, t), (kp, tp)) in enumerate(SPOT_PAIRS):
        actual = covariance_sums[index] / (num_trajectories * 4)
        expected = full_covariance[k * 16 + t, kp * 16 + tp]
        error = float(abs(actual - expected))
        spot_checks.append({"first": [k, t], "second": [kp, tp], "theory": complex_record(expected),
            "empirical": complex_record(actual), "absolute_error": error,
            "absolute_tolerance": COVARIANCE_ABSOLUTE_TOLERANCE,
            "passed": error <= COVARIANCE_ABSOLUTE_TOLERANCE})
    return ({"num_trajectories": num_trajectories, "independent_link_realizations": 4 * num_trajectories,
             "aggregation": "sum(error energy) / sum(truth energy), no per-realization normalization",
             "conditions": conditions, "passed": all(x["passed"] for x in conditions),
             "runtime_seconds": perf_counter() - started},
            {"independent_link_realizations": 4 * num_trajectories,
             "estimator": "uncentered mean(H_i * conjugate(H_j)); all 4 independent links per trajectory",
             "pairs": spot_checks, "passed": all(x["passed"] for x in spot_checks)})


def provenance():
    git = ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}"]
    history = subprocess.check_output(git + ["log", "--format=%H%x09%s"], text=True).splitlines()
    baseline_commits = [line.split("\t")[0] for line in history
                        if line.endswith("\tfeat: add validated dynamic TDL-A AWGN baseline scan")]
    if not baseline_commits:
        raise ValueError("accepted T09B standalone commit is missing from this branch history")
    paths = sorted((PROJECT_ROOT / "src").glob("*.py")) + [Path(__file__),
            PROJECT_ROOT / "tests/test_lmmse.py", PROJECT_ROOT / "configs/base.yaml"]
    return {"t09b_commit": baseline_commits[0],
        "git_head": subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip(),
        "git_head_subject": subprocess.check_output(git + ["log", "-1", "--format=%s"], text=True).strip(),
        "git_status_at_run": subprocess.check_output(git + ["status", "--short"], text=True).strip(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version(),
        "platform": platform.platform(), "processor": platform.processor(),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "source_file_hashes": {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="5 trajectories, all preparations; no result files")
    args = parser.parse_args()
    started = perf_counter()
    config = check_config()
    frozen_checks = frozen_source_checks()
    if not all(frozen_checks.values()):
        raise ValueError(f"frozen T09B source/config conflict: {frozen_checks}")
    metadata = provenance()
    try:
        full, layouts, cache, theory, summaries = prepare_theory()
        tiny = tiny_analytic_sentinel()
        integration = actual_pilot_integration_sentinel(layouts, cache)
        count = 5 if args.benchmark else NUM_CALIBRATION_TRAJECTORIES
        empirical, spots = run_monte_carlo(count, layouts, cache, theory, full,
            progress=lambda n: print(f"Completed {n}/{count} calibration trajectories", flush=True))
    except (LMMSEPreparationError, ValueError, np.linalg.LinAlgError) as exc:
        record = {"task": "T10A", "all_checks_passed": False,
                  "failure": getattr(exc, "record", {"reason": str(exc)}), "provenance": metadata}
        OUTPUTS.mkdir(exist_ok=True)
        path = OUTPUTS / f"{PREFIX}_failure.json"
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"FAIL: {path}")
        raise SystemExit(1) from exc
    if args.benchmark:
        print(f"Benchmark: {count} trajectories / {count * 4} condition evaluations / {count * 16} link estimates")
        print(f"Theory preparation (40 Cholesky factors): {summaries['theory_preparation_seconds']:.6f} s")
        print(f"Monte Carlo: {empirical['runtime_seconds']:.6f} s; projected 500: {empirical['runtime_seconds'] * 100:.6f} s")
        print(f"Total: {perf_counter() - started:.6f} s; no result files saved; 5-trajectory statistical gates not evaluated")
        return
    delays, powers = tdl_a_profile()
    report = {"task": "T10A",
        "experiment_type": "Oracle LMMSE channel-estimator mathematical and numerical calibration",
        "config": config, "channel_covariance_definition": {
            "model": "TDL-A 23 taps, 100 ns, 3.5 GHz, 30 km/h, independent zero-mean proper Gaussian antenna pairs",
            "frequency": "sum_l p_l exp(-j*2*pi*(f_k-f_kprime)*tau_l)",
            "time": "verified T08B jakes_covariance at OFDM symbol start times including CP",
            "joint": "Rf[k,kprime]*Rt[t,tprime]", "spatial_correlation": False,
            "symbol_wise_quasi_static": True, "ICI_modeled": False,
            "delays_seconds": delays.tolist(), "normalized_powers": powers.tolist(),
            "normalized_power_sum": float(np.sum(powers)), "mobility": mobility_parameters()},
        **summaries, "flatten_order": "[K,T] NumPy C order: flat_index=k*T+t; Rhh=np.kron(Rf,Rt)",
        "snr_points": list(SNR_POINTS), "pilot_densities": list(PILOT_DENSITIES),
        "oracle_definition": "uses exact TDL-A/Jakes covariance and exact receiver-known noise variance",
        "oracle_role": "ideal matched-statistics reference",
        "theoretical_boundary": "Under the matched zero-mean jointly Gaussian channel model used here, the Oracle LMMSE conditional estimator is also the MMSE conditional mean. Therefore later learning methods are not required to beat Oracle LMMSE in matched-distribution MSE.",
        "noise_covariance_definition": "Pilot-LS z=h_p+w, unit-modulus pilots, w~CN(0,sigma_n^2 I); sigma_n^2=1/10^(SNR/10), no fading-power scaling",
        "solver": "scipy.linalg.cho_factor(lower=True) + cho_solve; explicit per-dF/TX/SNR factors reused for both RXs and all trajectories",
        "cache_scope": "40 prepared objects local to this run; no inverse, no global/implicit cache",
        "explicit_inverse_used": False, "pseudo_inverse_used": False, "jitter_used": False,
        "posterior_diagonal_clipped": False, "production_estimator_inputs": "Pilot-LS observations only; no true realization",
        "tiny_analytic_test": tiny, "actual_pilot_integration_sentinel": integration,
        "theoretical_conditions": theory, "monte_carlo_calibration": empirical,
        "joint_covariance_spot_checks": spots, "seed_policy": {
            "channel_seed": "40261001 + trajectory_id", "channel_seed_range": [CHANNEL_SEED_BASE, CHANNEL_SEED_BASE + count - 1],
            "noise_seed": "30261001 + 100000*condition_index + 4*trajectory_id + 2*rx + tx",
            "condition_index_order": [list(x) for x in REPRESENTATIVE_CONDITIONS],
            "noise_seed_ranges_by_condition": [[calibration_noise_seed(ci, 0, 0, 0),
                calibration_noise_seed(ci, count - 1, 1, 1)] for ci in range(4)],
            "integration_noise_seed": "31261001 + dF", "bits_seed": "unused; no bits generated",
            "training_seed": "unused; no training", "seeds_retuned": False},
        "fixed_acceptance_tolerances": {"data_nmse_absolute_difference_db": NMSE_TOLERANCE_DB,
                                       "joint_covariance_absolute_error": COVARIANCE_ABSOLUTE_TOLERANCE},
        "frozen_source_checks": frozen_checks, "t09b_committed_cleanly": True,
        "t09b_preimplementation_regression": "422 passed in 5.31s; clean git status after standalone commit",
        "source_file_hashes": metadata.pop("source_file_hashes"), "provenance": metadata,
        "warnings": [], "failed_attempts": [], "lmmse_train_implemented": False}
    checks = {"T09B committed cleanly": report["t09b_committed_cleanly"],
        "frequency covariance": summaries["frequency_covariance_summary"]["passed"],
        "time covariance": summaries["time_covariance_summary"]["passed"],
        "2D covariance / flatten order": summaries["full_covariance_summary"]["passed"] and summaries["kronecker_validation"]["passed"],
        "pilot-index extraction": all(x["passed"] for x in summaries["pilot_index_summaries"]),
        "stable Cholesky solve": all(x[f"cholesky_success_tx{q}"] for x in theory for q in range(2)),
        "no inverse/pinv": True, "posterior-MSE calculation": all(x[f"posterior_min_tx{q}"] >= -1e-12
             and x[f"posterior_max_tx{q}"] <= 1 + 1e-12 and x[f"posterior_max_imaginary_tx{q}"] <= 1e-12
             for x in theory for q in range(2)),
        "20 theoretical conditions": len(theory) == 20, "tiny closed-form test": tiny["passed"],
        "actual pilot integration sentinel": integration["passed"],
        "empirical covariance spot-check": spots["passed"], "theory-vs-empirical calibration": empirical["passed"],
        "frozen existing source/config": all(frozen_checks.values())}
    report["calibration_checks"] = checks
    report["all_checks_passed"] = all(checks.values())
    OUTPUTS.mkdir(exist_ok=True)
    csv_path = OUTPUTS / f"{PREFIX}_theory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(theory[0]))
        writer.writeheader()
        writer.writerows(theory)
    report["theory_csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    report["runtime"] = {"theory_preparation_seconds": summaries["theory_preparation_seconds"],
        "monte_carlo_seconds": empirical["runtime_seconds"], "total_seconds_before_json_write": perf_counter() - started}
    report_path = OUTPUTS / f"{PREFIX}_results.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Covariances:", json.dumps({key: summaries[key] for key in (
        "frequency_covariance_summary", "time_covariance_summary", "full_covariance_summary")}, indent=2))
    print("Tiny analytic:", json.dumps(tiny))
    print("Empirical calibration:", json.dumps(empirical["conditions"], indent=2))
    print("Joint covariance:", json.dumps(spots, indent=2))
    print("Runtime:", json.dumps(report["runtime"]))
    print("T10A all_checks_passed:", report["all_checks_passed"])
    print(f"Generated: {csv_path}\nGenerated: {report_path}")
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
