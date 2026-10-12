"""T10C-PRE channel-only calibration; no formal dataset, BER or training."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import scipy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import trajectories as tr  # noqa: E402
from src.provenance import recorded_source_matches
from src.config import load_config, validate_config  # noqa: E402
from src.doppler import (  # noqa: E402
    build_tdl_a_doppler_channel, jakes_complex_gaussian,
    mobility_parameters, ofdm_time_grid_seconds,
)
from src.numerics import complex_gaussian_noise  # noqa: E402
from src.tdl import build_static_tdl_a_channel  # noqa: E402
from scripts.validate_tdl_a_doppler import summarize_link_values  # noqa: E402


NUM_CORRELATION_TRAJECTORIES = 3000
NUM_FULL_TDL_TRAJECTORIES = 1000
CORRELATION_SEED = 50261001
NEGATIVE_CONTROL_SEED = 60261001
LAGS = (1, 4, 15, 16, 17, 31, 32, 80, 159)
# Fixed before any statistical calibration run. Independent units are trajectories.
CORRELATION_TOLERANCES = {lag: 0.02 if lag <= 32 else 0.03 if lag == 80 else 0.05 for lag in LAGS}
POWER_TOLERANCE = 0.10
PDP_RELATIVE_TOLERANCE = 0.15
INDEPENDENCE_TOLERANCE = 0.15
FACTOR_RECONSTRUCTION_TOLERANCE = 5e-14
NEGATIVE_CONTROL_ZERO_TOLERANCE = 0.10


def correlations(series, covariance, lags):
    """Same uncentered valid-pair/source-power estimator as T08B."""
    rows = []
    for lag in lags:
        source, target = series[:, :-lag], series[:, lag:]
        denominator = float(np.mean(np.abs(source) ** 2))
        empirical = np.mean(target * source.conj()) / denominator
        theoretical = float(covariance[lag, 0])
        error = float(abs(empirical - theoretical))
        rows.append({"lag_symbols": lag, "theoretical": theoretical,
            "empirical_real": float(empirical.real), "empirical_imag": float(empirical.imag),
            "absolute_error": error, "absolute_tolerance": CORRELATION_TOLERANCES[lag],
            "source_sample_mean_power": denominator, "independent_trajectory_count": series.shape[0],
            "valid_pair_count": int(source.size), "passed": error <= CORRELATION_TOLERANCES[lag]})
    return rows


def cross_frame_boundary_correlation(series, covariance):
    source, target = series[:, 15], series[:, 16]
    denominator = float(np.mean(np.abs(source) ** 2))
    empirical = np.mean(target * source.conj()) / denominator
    theoretical = float(covariance[1, 0])
    error = float(abs(empirical - theoretical))
    return {"global_symbol_source": 15, "global_symbol_target": 16, "lag_symbols": 1,
        "theoretical": theoretical, "empirical_real": float(empirical.real),
        "empirical_imag": float(empirical.imag), "absolute_error": error,
        "absolute_tolerance": CORRELATION_TOLERANCES[1], "source_sample_mean_power": denominator,
        "independent_trajectory_count": series.shape[0], "valid_pair_count": series.shape[0],
        "passed": error <= CORRELATION_TOLERANCES[1]}


def benchmark(model):
    groups = []
    for count in (1, 10):
        totals = {"trajectory_generation_seconds": 0.0, "frequency_response_seconds": 0.0,
                  "frame_slicing_seconds": 0.0}
        for tid in range(count):
            start = perf_counter()
            taps = model.generate_taps(tr.CHANNEL_SEED_BASE + tid)
            totals["trajectory_generation_seconds"] += perf_counter() - start
            start = perf_counter()
            h_long = model.frequency_response(taps)
            totals["frequency_response_seconds"] += perf_counter() - start
            start = perf_counter()
            frames = tr.slice_channel_frames(h_long)
            totals["frame_slicing_seconds"] += perf_counter() - start
            assert np.shares_memory(frames, h_long)
        groups.append({"num_trajectories": count, "symbols_per_trajectory": 160,
                       **totals, "total_generation_seconds": sum(totals.values())})
    return {"preparation": model.preparation_timing, "groups": groups,
        "estimated_2000_trajectory_generation_seconds": groups[1]["total_generation_seconds"] * 200,
        "estimated_2000_total_seconds_including_one_preparation":
            groups[1]["total_generation_seconds"] * 200 + model.preparation_timing["total_preparation_seconds"],
        "planning_only": True, "formal_training_trajectories_generated": 0}


def full_tdl_statistics(model, progress=None):
    count = NUM_FULL_TDL_TRAJECTORIES
    link_energy = np.empty((count, 2, 2))
    frequency_power = np.empty_like(link_energy)
    frame_energy = np.empty((count, 10))
    dc = np.empty((count, 4, 160), dtype=np.complex128)
    standardized_t0 = np.empty((count, 92), dtype=np.complex128)
    tap_power_sum = np.zeros((2, 2, 23))
    power_by_time_sum = np.zeros((2, 2, 160))
    power_by_carrier_sum = np.zeros((2, 2, 64))
    finite, slicing_exact, views = True, True, True
    dc_error, time_variation, formula_error = 0.0, 0.0, 0.0
    raw_min, raw_max = np.inf, -np.inf
    for tid in range(count):
        draw = model.generate(tid)
        taps, h = draw.taps, draw.h_long
        p_taps, p_h = np.abs(taps) ** 2, np.abs(h) ** 2
        link_energy[tid] = np.mean(np.sum(p_taps, axis=2), axis=-1)
        frequency_power[tid] = np.mean(p_h, axis=(2, 3))
        frame_energy[tid] = np.mean(np.abs(draw.h_frames) ** 2, axis=(1, 2, 3, 4))
        tap_power_sum += np.sum(p_taps, axis=-1)
        power_by_time_sum += np.mean(p_h, axis=2)
        power_by_carrier_sum += np.mean(p_h, axis=3)
        dc[tid] = h[:, :, 32, :].reshape(4, 160)
        standardized_t0[tid] = (taps[..., 0] / np.sqrt(model.powers)).reshape(92)
        dc_error = max(dc_error, float(np.max(np.abs(h[:, :, 32, :] - taps.sum(axis=2)))))
        time_variation = max(time_variation, float(np.max(np.abs(h - h[..., :1]))))
        raw_min = min(raw_min, float(p_h.min()))
        raw_max = max(raw_max, float(p_h.max()))
        finite &= bool(np.all(np.isfinite(taps)) and np.all(np.isfinite(h)))
        slicing_exact &= bool(np.array_equal(np.concatenate(list(draw.h_frames), axis=-1), h))
        views &= bool(np.shares_memory(draw.h_frames, h))
        if tid == 0:
            z = complex_gaussian_noise((2, 2, 23, 160), 1, np.random.default_rng(draw.channel_seed))
            reference = (z @ model.factor.T) * np.sqrt(model.powers)[None, None, :, None]
            formula_error = float(np.max(np.abs(reference - taps)))
        if progress and (tid + 1) % 100 == 0:
            progress(tid + 1)
    empirical_pdp = tap_power_sum / (count * 160)
    tap_errors = np.abs(empirical_pdp / model.powers - 1)
    link_dc = dc[:, :, 0]
    covariance = link_dc.T @ link_dc.conj() / count
    independence = float(np.max(np.abs(covariance - np.diag(np.diag(covariance)))))
    tap_covariance = standardized_t0.T @ standardized_t0.conj() / count
    tap_independence = float(np.max(np.abs(tap_covariance - np.diag(np.diag(tap_covariance)))))
    channel_correlations = correlations(dc[:, 0, :], model.covariance, (1, 16, 80))
    boundary = cross_frame_boundary_correlation(dc[:, 0, :], model.covariance)
    normalized_increment = float(np.mean(np.abs(np.diff(dc[:, 0, :], axis=-1)) ** 2)
                                 / np.mean(np.abs(dc[:, 0, :-1]) ** 2))
    expected_increment = float(2 * (1 - model.covariance[1, 0]))
    increment_error = abs(normalized_increment / expected_increment - 1)
    checks = {
        "full_tdl_temporal_correlations": all(r["passed"] for r in channel_correlations),
        "cross_frame_boundary_correlation": boundary["passed"],
        "tap_pdp_preserved": bool(np.all(tap_errors <= PDP_RELATIVE_TOLERANCE)),
        "tap_ensemble_power": bool(np.all(np.abs(link_energy.mean(axis=0) - 1) <= POWER_TOLERANCE)),
        "frequency_ensemble_power": bool(np.all(np.abs(frequency_power.mean(axis=0) - 1) <= POWER_TOLERANCE)),
        "four_mimo_links_independent": independence <= INDEPENDENCE_TOLERANCE,
        "tap_link_processes_independent": tap_independence <= INDEPENDENCE_TOLERANCE,
        "dc_fractional_response": dc_error <= 1e-14,
        "frame_concatenation_exact": slicing_exact,
        "frame_views": views,
        "all_finite": finite,
        "no_random_power_normalization": formula_error == 0,
        "frame_energy_natural_variation": bool(np.all(np.ptp(frame_energy, axis=1) > 1e-8)),
        "not_static": bool(time_variation > 1e-8 and normalized_increment > 1e-8 and increment_error <= 0.15),
        "not_iid": bool(channel_correlations[0]["empirical_real"] > 0.95 and channel_correlations[0]["passed"]),
    }
    statistics = {"num_independent_trajectories": count, "coordinate": [0, 0, 32],
        "correlations": channel_correlations, "frame_boundary": boundary,
        "max_time_difference_from_t0": time_variation, "normalized_increment_power": normalized_increment,
        "theoretical_normalized_increment_power": expected_increment, "increment_relative_error": increment_error,
        "dc_max_absolute_error": dc_error, "normalization_formula_max_error": formula_error,
        "max_cross_link_covariance_at_t0": independence,
        "max_cross_tap_link_standardized_covariance_at_t0": tap_independence,
        "frame_energy_per_trajectory": frame_energy.tolist(),
        "frame_energy_range_min": float(np.ptp(frame_energy, axis=1).min()),
        "frame_energy_range_max": float(np.ptp(frame_energy, axis=1).max()),
        "checks": checks, "passed": bool(all(checks.values()))}
    power = {"num_independent_trajectories": count,
        "statistical_unit": "trajectory; all 160 symbols / 64 carriers averaged within each trajectory",
        "tap_energy_by_link": summarize_link_values(link_energy),
        "frequency_power_by_link": summarize_link_values(frequency_power),
        "frequency_power_over_all_links_carriers_symbols": float(frequency_power.mean()),
        "target_pdp": model.powers.tolist(), "empirical_pdp_by_link": empirical_pdp.tolist(),
        "max_per_link_per_tap_relative_error": float(tap_errors.max()),
        "frequency_ensemble_power_by_time": (power_by_time_sum / count).tolist(),
        "frequency_ensemble_power_by_carrier": (power_by_carrier_sum / count).tolist(),
        "raw_instantaneous_frequency_power_min": raw_min, "raw_instantaneous_frequency_power_max": raw_max}
    return statistics, power


def calibration(model):
    unit = tr.correlated_complex_gaussian(np.random.default_rng(CORRELATION_SEED), model.factor,
                                         (NUM_CORRELATION_TRAJECTORIES,))
    unit_correlations = correlations(unit, model.covariance, LAGS)
    unit_boundary = cross_frame_boundary_correlation(unit, model.covariance)
    # Deliberately WRONG channel construction used only as a negative sentinel:
    # 10 independent internally correlated 16-symbol blocks, then concatenate.
    independent_frames = jakes_complex_gaussian(np.random.default_rng(NEGATIVE_CONTROL_SEED),
        ofdm_time_grid_seconds(), model.mobility["maximum_doppler_hz"], (NUM_CORRELATION_TRAJECTORIES, 10))
    negative = cross_frame_boundary_correlation(independent_frames.reshape(NUM_CORRELATION_TRAJECTORIES, 160), model.covariance)
    negative["detected"] = bool(not negative["passed"] and abs(complex(negative["empirical_real"], negative["empirical_imag"])) <= NEGATIVE_CONTROL_ZERO_TOLERANCE)
    full, power = full_tdl_statistics(model, lambda n: print(f"Completed {n}/1000 calibration TDL trajectories", flush=True))
    return unit_correlations, {"unit_process": unit_boundary, "full_tdl": full["frame_boundary"],
        "independent_frame_negative_sentinel": negative}, full, power


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-only", action="store_true", help="1/10 trajectory timing only; no result files")
    args = parser.parse_args()
    started = perf_counter()
    config = load_config(PROJECT_ROOT / "configs/base.yaml")
    validate_config(config)
    if any(config["system"][k] != v for k, v in {"nt": 2, "nr": 2, "nfft": 64,
        "cp_len": 16, "sample_rate_hz": 1920000, "delta_f_hz": 30000, "num_symbols": 16,
        "active_carriers": 64, "fft_norm": "ortho", "total_power": 1.0, "modulation": "qpsk_gray"}.items()):
        raise ValueError("T10C-PRE requires the frozen base system")
    baseline = json.loads((PROJECT_ROOT / "outputs/t10b_oracle_lmmse_results.json").read_text())
    frozen = {p: recorded_source_matches(PROJECT_ROOT,p,h,"t10b")
              for p, h in baseline["provenance"]["source_and_reference_sha256"].items()}
    if not all(frozen.values()):
        raise ValueError("frozen T10B source/reference conflict")
    with patch.object(tr, "covariance_eigenfactor", wraps=tr.covariance_eigenfactor) as factors, \
         patch.object(tr, "jakes_covariance", wraps=tr.jakes_covariance) as covariances:
        model = tr.prepare_trajectory()
        timing = benchmark(model)
        if args.benchmark_only:
            print(json.dumps(timing, indent=2))
            return
        unit, boundary, full, power = calibration(model)
        reuse = {"main_30kmh_160_covariance_preparation_count": covariances.call_count,
                 "main_30kmh_160_factorization_count": factors.call_count}
    zero = tr.prepare_trajectory(speed_kmh=0).generate(0)
    static_taps, static_h = build_static_tdl_a_channel(tr.CHANNEL_SEED_BASE, num_symbols=160)
    zero_regression = {"maximum_doppler_hz": mobility_parameters(0)["maximum_doppler_hz"],
        "taps_equal_t08a_exactly": bool(np.array_equal(zero.taps, np.repeat(static_taps[..., None], 160, axis=-1))),
        "channel_equals_t08a_exactly": bool(np.array_equal(zero.h_long, static_h)),
        "all_10_frames_identical": bool(np.array_equal(zero.h_frames, np.repeat(zero.h_frames[:1], 10, axis=0))),
        "max_time_difference": float(np.max(np.abs(zero.h_long - zero.h_long[..., :1])))}
    zero_regression["passed"] = bool(all(zero_regression[k] for k in (
        "taps_equal_t08a_exactly", "channel_equals_t08a_exactly", "all_10_frames_identical")) and zero_regression["max_time_difference"] == 0)
    old = json.loads((PROJECT_ROOT / "outputs/t09b_tdl_awgn_results.json").read_text())
    old_hashes = {x["trajectory_id"]: x["H_sha256_complex128_C_order"] for x in old["channel_realizations"]}
    old_checks = [{"trajectory_id": tid, "passed": hashlib.sha256(build_tdl_a_doppler_channel(
        tr.CHANNEL_SEED_BASE + tid)[1].tobytes(order="C")).hexdigest() == old_hashes[tid]} for tid in (0, 1, 199)]
    draw = model.generate(17)
    metadata = draw.metadata
    slicing = {"H_long_shape": list(draw.h_long.shape), "H_frames_shape": list(draw.h_frames.shape),
        "taps_shape": list(draw.taps.shape), "shares_memory": bool(np.shares_memory(draw.h_frames, draw.h_long)),
        "concatenation_max_error": float(np.max(np.abs(np.concatenate(list(draw.h_frames), axis=-1) - draw.h_long))),
        "all_frame_samples_match_global_indices": all(np.array_equal(draw.h_frames[f], draw.h_long[..., 16*f:16*(f+1)]) for f in range(10)),
        "boundary_global_indices": [metadata[0]["global_symbol_stop"], metadata[1]["global_symbol_start"]],
        "metadata_example_trajectory17": metadata, "stop_index_inclusive": True}
    eigen = model.factorization_summary
    checks = {"continuous_time_grid": bool(np.array_equal(model.time_grid_s, ofdm_time_grid_seconds(num_symbols=160))),
        "covariance_hermitian": bool(np.array_equal(model.covariance, model.covariance.T)),
        "covariance_unit_diagonal": bool(np.array_equal(np.diag(model.covariance), np.ones(160))),
        "covariance_psd_and_factor": eigen["minimum_covariance_eigenvalue"] >= -1e-12 and eigen["factor_reconstruction_max_error"] <= FACTOR_RECONSTRUCTION_TOLERANCE,
        "factor_reused": all(v == 1 for v in reuse.values()),
        "unit_long_lag_correlations": all(x["passed"] for x in unit),
        "unit_boundary_correlation": boundary["unit_process"]["passed"],
        "independent_frame_negative_sentinel": boundary["independent_frame_negative_sentinel"]["detected"],
        "full_tdl_statistics": full["passed"], "zero_speed": zero_regression["passed"],
        "old_16_symbol_regression": all(x["passed"] for x in old_checks),
        "frame_slicing": slicing["shares_memory"] and slicing["concatenation_max_error"] == 0 and slicing["all_frame_samples_match_global_indices"],
        "frozen_sources": all(frozen.values())}
    git = ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}"]
    head = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    history = subprocess.check_output(git + ["log", "--format=%H%x09%s"], text=True).splitlines()
    commit = next(line.split("\t")[0] for line in history if line.endswith("\tfeat: add validated Oracle LMMSE full-link baseline"))
    paths = ["src/trajectories.py", "src/doppler.py", "src/tdl.py", "configs/base.yaml", "scripts/validate_multiframe_trajectory.py", "tests/test_trajectories.py", "docs/dataset_protocol.md", "pyproject.toml"]
    report = {"task": "T10C-PRE", "experiment_type": "continuous multi-frame Jakes trajectory and dataset-protocol calibration",
        "frames_per_trajectory": 10, "symbols_per_frame": 16, "symbols_per_trajectory": 160,
        "symbol_duration": model.symbol_duration, "trajectory_duration": 160 * model.symbol_duration,
        "sampled_span": float(model.time_grid_s[-1]), "time_grid_s": model.time_grid_s.tolist(),
        "speed": model.mobility["speed_kmh"], "maximum_doppler": model.mobility["maximum_doppler_hz"],
        "long_covariance_summary": {"shape": [160,160], "hermitian_max_error": float(np.max(np.abs(model.covariance-model.covariance.T))),
            "diagonal_max_error": float(np.max(np.abs(np.diag(model.covariance)-1)))},
        "factorization_summary": {**eigen, **reuse, "factor_shape": list(model.factor.shape),
            "positive_eigenvalues_truncated": 0, "rank_truncation_used": False},
        "theoretical_correlations": [{"lag_symbols": x["lag_symbols"], "correlation": x["theoretical"]} for x in unit],
        "empirical_correlations": unit, "frame_boundary_diagnostic": boundary,
        "zero_speed_regression": zero_regression, "old_16_symbol_regression": old_checks,
        "full_tdl_statistics": full, "ensemble_power_statistics": power,
        "seed_policy": {"channel_seed_base": tr.CHANNEL_SEED_BASE, "channel_seed": "40261001 + trajectory_id",
            "one_seed_per_whole_trajectory": True, "frame_seed_offset_used": False, "full_tdl_id_range": [0,999],
            "unit_correlation_seed": CORRELATION_SEED, "unit_independent_process_count": 3000,
            "unit_draw": "one seeded batch of independent 160-symbol process rows",
            "negative_control_seed": NEGATIVE_CONTROL_SEED, "bits_seed": None, "noise_seed": None, "training_seed": None},
        "correlation_estimator": "mean(target*conj(source))/mean(abs(source)**2), valid pairs, no centering",
        "statistical_unit": "independent trajectory; symbols and frames inside it are correlated",
        "frame_slicing_definition": slicing, "global_symbol_metadata_definition": "trajectory_id/frame_index/start/stop; stop inclusive",
        "dataset_split_contract": {"split_unit": "trajectory_id", "split_before_variants": True,
            "same_trajectory_all_frames_snr_pilot_noise_same_split": True, "same_trajectory_frame_variants_same_H_true": True,
            "formal_test_sealed": True, "formal_split_ids_assigned": False, "formal_datasets_generated": False,
            "planned_sizes": {"train": "2000 trajectories x 10 frames", "validation_id": "400 x 10",
                "ood_dev": "200 x 5 per OOD type", "test_id": "1000 x 4 per condition", "test_ood": "1000 x 4 per condition"}},
        "benchmark": timing, "memory_estimate": {"dtype": "complex128", "bytes_per_complex": 16,
            "single_H_long_bytes": int(draw.h_long.nbytes), "single_H_long_MiB": draw.h_long.nbytes/2**20,
            "H_frames_is_view": True, "additional_frame_data_allocation_bytes": 0,
            "2000_H_true_raw_bytes": int(draw.h_long.nbytes*2000), "2000_H_true_raw_MiB": draw.h_long.nbytes*2000/2**20,
            "2000_H_true_raw_GiB": draw.h_long.nbytes*2000/2**30, "includes_only_H_true": True},
        "per_trajectory_normalization": False, "per_frame_normalization": False, "per_time_normalization": False,
        "validation_tolerances": {"correlation_by_lag": CORRELATION_TOLERANCES, "ensemble_power_absolute": POWER_TOLERANCE,
            "pdp_relative": PDP_RELATIVE_TOLERANCE, "independence_absolute": INDEPENDENCE_TOLERANCE,
            "factor_reconstruction_absolute": FACTOR_RECONSTRUCTION_TOLERANCE, "increment_relative": 0.15},
        "checks": checks, "all_checks_passed": bool(all(checks.values())), "frozen_T10B_source_checks": frozen,
        "provenance": {"git_head": head, "t10b_commit": commit, "t10b_committed_cleanly": True,
            "source_sha256": {p: hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in paths},
            "python_version": platform.python_version(), "platform": platform.platform(),
            "numpy_version": np.__version__, "scipy_version": scipy.__version__},
        "runtime_seconds": perf_counter() - started}
    path = PROJECT_ROOT / "outputs/t10c_pre_multiframe_trajectory_results.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Covariance/factor:", json.dumps(report["long_covariance_summary"]),
          json.dumps({k:v for k,v in report["factorization_summary"].items() if k!="raw_eigenvalues"}))
    print("Long-lag correlations:", json.dumps(unit, indent=2))
    print("Frame boundary:", json.dumps(boundary, indent=2))
    print("Full TDL correlations:", json.dumps(full["correlations"], indent=2))
    print("Ensemble power:", json.dumps(power["frequency_power_by_link"], indent=2))
    print("Zero speed:", json.dumps(zero_regression))
    print("Benchmark:", json.dumps(timing, indent=2))
    print("Memory:", json.dumps(report["memory_estimate"]))
    print("Checks:", json.dumps(checks)); print("all_checks_passed:", report["all_checks_passed"])
    print("Saved:", path); print("Runtime seconds:", report["runtime_seconds"])
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
