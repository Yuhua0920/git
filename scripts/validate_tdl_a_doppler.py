"""T08B channel-only Jakes calibration; no BER/NMSE or within-symbol waveform."""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.special import j0


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, validate_config  # noqa: E402
from src.doppler import (  # noqa: E402
    CARRIER_FREQUENCY_HZ, ICI_MODELED, SPEED_KMH, SYMBOL_WISE_QUASI_STATIC,
    build_tdl_a_doppler_channel, covariance_eigenfactor, jakes_complex_gaussian,
    jakes_covariance, mobility_parameters, ofdm_time_grid_seconds,
)
from src.tdl import (  # noqa: E402
    DESIRED_RMS_DELAY_SPREAD_NS, STANDARD_REFERENCE, STANDARD_URL,
    build_static_tdl_a_channel, rms_delay_spread_seconds, tdl_a_profile,
)


NUM_CORRELATION_TRAJECTORIES = 5000
NUM_CHANNEL_TRAJECTORIES = 1000
CHANNEL_SEED_BASE = 40261001
CORRELATION_SEED = 50261001
LAGS = (1, 4, 8, 12, 15)
# Fixed before executing the calibration. Time samples are correlated replicas,
# so the independent statistical unit is a trajectory, not a symbol.
CORRELATION_ABSOLUTE_TOLERANCE = 0.02
ENSEMBLE_POWER_ABSOLUTE_TOLERANCE = 0.10
TAP_POWER_RELATIVE_TOLERANCE = 0.15
INCREMENT_POWER_RELATIVE_TOLERANCE = 0.15
INDEPENDENCE_ABSOLUTE_TOLERANCE = 0.15


def correlation_statistics(series: np.ndarray, fd: float, symbol_duration_s: float) -> list[dict]:
    """Pool valid lag pairs; divide by mean source-sample power, without centering."""
    rows = []
    for lag in LAGS:
        source = series[:, :-lag]
        target = series[:, lag:]
        denominator = float(np.mean(np.abs(source) ** 2))
        empirical = np.mean(target * source.conj()) / denominator
        theoretical = float(j0(2 * np.pi * fd * symbol_duration_s * lag))
        rows.append({
            "lag_symbols": lag, "theoretical": theoretical,
            "empirical_real": float(empirical.real), "empirical_imag": float(empirical.imag),
            "absolute_error": float(abs(empirical - theoretical)),
            "source_sample_mean_power": denominator,
            "independent_trajectory_count": int(series.shape[0]),
            "valid_pair_count": int(source.size),
        })
    return rows


def summarize_link_values(values: np.ndarray) -> dict:
    """Summarize [trajectory,RX,TX] values; sample std uses independent trajectories."""
    return {
        f"rx{r}_tx{q}": {
            "mean": float(np.mean(values[:, r, q])),
            "std": float(np.std(values[:, r, q], ddof=1)),
            "min": float(np.min(values[:, r, q])),
            "max": float(np.max(values[:, r, q])),
        }
        for r in range(2) for q in range(2)
    }


def main() -> None:
    started = perf_counter()
    config_path = PROJECT_ROOT / "configs" / "base.yaml"
    config = load_config(config_path)
    validate_config(config)
    system = config["system"]
    expected = {
        "nt": 2, "nr": 2, "nfft": 64, "active_carriers": 64,
        "num_symbols": 16, "delta_f_hz": 30000, "sample_rate_hz": 1920000,
        "cp_len": 16, "fft_norm": "ortho", "modulation": "qpsk_gray", "total_power": 1.0,
    }
    for key, value in expected.items():
        if system[key] != value:
            raise ValueError(f"T08B requires {key}={value}; got {system[key]}")
    if config["pilot"]["spacing_time"] != 4:
        raise ValueError("T08B requires pilot spacing_time=4")

    mobility = mobility_parameters(SPEED_KMH, CARRIER_FREQUENCY_HZ)
    fd = mobility["maximum_doppler_hz"]
    times = ofdm_time_grid_seconds(
        system["nfft"], system["cp_len"], system["sample_rate_hz"], system["num_symbols"],
    )
    symbol_duration = (system["nfft"] + system["cp_len"]) / system["sample_rate_hz"]
    covariance = jakes_covariance(times, fd)
    _, eigen_summary = covariance_eigenfactor(covariance)
    delays, powers = tdl_a_profile()

    # One seeded batch of 5000 mutually independent unit-power process rows.
    unit_series = jakes_complex_gaussian(
        np.random.default_rng(CORRELATION_SEED), times, fd, (NUM_CORRELATION_TRAJECTORIES,),
    )
    unit_correlations = correlation_statistics(unit_series, fd, symbol_duration)

    link_energy = np.empty((NUM_CHANNEL_TRAJECTORIES, 2, 2), dtype=np.float64)
    link_frequency_power = np.empty_like(link_energy)
    tap_power_sum = np.zeros((2, 2, 23), dtype=np.float64)
    t0_standardized_taps = np.empty((NUM_CHANNEL_TRAJECTORIES, 92), dtype=np.complex128)
    dc_series = np.empty((NUM_CHANNEL_TRAJECTORIES, 16), dtype=np.complex128)
    dc_error = 0.0
    time_variation_max = 0.0
    instantaneous_energy_min = np.inf
    instantaneous_energy_max = -np.inf
    all_finite = bool(np.all(np.isfinite(unit_series)))
    all_shapes_correct = True
    for i in range(NUM_CHANNEL_TRAJECTORIES):
        taps, channel = build_tdl_a_doppler_channel(CHANNEL_SEED_BASE + i)
        tap_power = np.abs(taps) ** 2
        instantaneous_energy = np.sum(tap_power, axis=2)
        link_energy[i] = np.mean(instantaneous_energy, axis=-1)
        link_frequency_power[i] = np.mean(np.abs(channel) ** 2, axis=(2, 3))
        tap_power_sum += np.sum(tap_power, axis=-1)
        t0_standardized_taps[i] = (taps[..., 0] / np.sqrt(powers)).reshape(92)
        dc_series[i] = channel[0, 0, 32, :]
        dc_error = max(dc_error, float(np.max(np.abs(channel[:, :, 32, :] - taps.sum(axis=2)))))
        time_variation_max = max(time_variation_max, float(np.max(np.abs(channel - channel[..., :1]))))
        instantaneous_energy_min = min(instantaneous_energy_min, float(np.min(instantaneous_energy)))
        instantaneous_energy_max = max(instantaneous_energy_max, float(np.max(instantaneous_energy)))
        all_finite &= bool(np.all(np.isfinite(taps)) and np.all(np.isfinite(channel)))
        all_shapes_correct &= taps.shape == (2, 2, 23, 16) and channel.shape == (2, 2, 64, 16)

    empirical_tap_power = tap_power_sum / (NUM_CHANNEL_TRAJECTORIES * 16)
    tap_relative_errors = np.abs(empirical_tap_power / powers - 1)
    energy_statistics = summarize_link_values(link_energy)
    frequency_statistics = summarize_link_values(link_frequency_power)
    t0_covariance = t0_standardized_taps.T @ t0_standardized_taps.conj() / NUM_CHANNEL_TRAJECTORIES
    maximum_off_diagonal_covariance = float(np.max(np.abs(t0_covariance - np.diag(np.diag(t0_covariance)))))
    channel_correlations = correlation_statistics(dc_series, fd, symbol_duration)
    increment_power = float(np.mean(np.abs(np.diff(dc_series, axis=-1)) ** 2))
    normalized_increment = increment_power / np.mean(np.abs(dc_series[:, :-1]) ** 2)
    theoretical_increment = float(2 * (1 - covariance[1, 0]))
    increment_relative_error = float(abs(normalized_increment / theoretical_increment - 1))
    nonstatic_passed = bool(
        time_variation_max > 1e-8 and normalized_increment > 1e-8
        and increment_relative_error <= INCREMENT_POWER_RELATIVE_TOLERANCE
    )
    not_iid_passed = bool(
        channel_correlations[0]["empirical_real"] > 0.95
        and channel_correlations[0]["absolute_error"] <= CORRELATION_ABSOLUTE_TOLERANCE
    )

    zero_taps, zero_channel = build_tdl_a_doppler_channel(CHANNEL_SEED_BASE, speed_kmh=0)
    static_taps, static_channel = build_static_tdl_a_channel(CHANNEL_SEED_BASE)
    zero_regression = {
        "channel_seed": CHANNEL_SEED_BASE, "maximum_doppler_hz": mobility_parameters(0)["maximum_doppler_hz"],
        "channel_equals_t08a_exactly": bool(np.array_equal(zero_channel, static_channel)),
        "taps_equal_t08a_exactly": bool(np.array_equal(zero_taps, np.repeat(static_taps[..., None], 16, axis=-1))),
        "all_time_slices_equal": bool(np.array_equal(zero_channel, np.repeat(zero_channel[..., :1], 16, axis=-1))),
        "max_time_difference": float(np.max(np.abs(zero_channel - zero_channel[..., :1]))),
    }
    zero_regression["passed"] = bool(
        zero_regression["channel_equals_t08a_exactly"] and zero_regression["taps_equal_t08a_exactly"]
        and zero_regression["all_time_slices_equal"] and zero_regression["maximum_doppler_hz"] == 0
    )
    pilot_spacing = config["pilot"]["spacing_time"]
    pilot_spacing_seconds = pilot_spacing * symbol_duration
    pilot_correlation = float(j0(2 * np.pi * fd * pilot_spacing_seconds))

    checks = {
        "mobility_formula": bool(fd == abs(mobility["speed_mps"]) / mobility["wavelength_m"]),
        "time_grid_includes_cp": bool(np.allclose(times, np.arange(16) * 80 / 1920000, rtol=0, atol=1e-18)),
        "covariance_unit_diagonal": bool(np.array_equal(np.diag(covariance), np.ones(16))),
        "covariance_hermitian": bool(np.array_equal(covariance, covariance.conj().T)),
        "covariance_psd_and_factor": bool(
            eigen_summary["minimum_covariance_eigenvalue"] >= -eigen_summary["negative_eigenvalue_tolerance"]
            and eigen_summary["factor_reconstruction_max_error"] <= 5e-14 and eigen_summary["jitter_added"] == 0
        ),
        "empirical_unit_temporal_correlation": all(
            row["absolute_error"] <= CORRELATION_ABSOLUTE_TOLERANCE
            and abs(row["empirical_imag"]) <= CORRELATION_ABSOLUTE_TOLERANCE for row in unit_correlations
        ),
        "unit_process_marginal_power": bool(np.all(np.abs(np.mean(np.abs(unit_series) ** 2, axis=0) - 1) <= 0.05)),
        "zero_speed_regression": zero_regression["passed"],
        "time_variation_nonzero": nonstatic_passed,
        "not_iid_across_symbols": not_iid_passed,
        "representative_channel_temporal_correlation": all(
            row["absolute_error"] <= CORRELATION_ABSOLUTE_TOLERANCE
            and abs(row["empirical_imag"]) <= CORRELATION_ABSOLUTE_TOLERANCE for row in channel_correlations
        ),
        "ensemble_link_tap_energy": bool(np.all(np.abs(np.mean(link_energy, axis=0) - 1) <= ENSEMBLE_POWER_ABSOLUTE_TOLERANCE)),
        "per_link_tap_power_preserves_pdp": bool(np.all(tap_relative_errors <= TAP_POWER_RELATIVE_TOLERANCE)),
        "ensemble_frequency_power": bool(np.all(np.abs(np.mean(link_frequency_power, axis=0) - 1) <= ENSEMBLE_POWER_ABSOLUTE_TOLERANCE)),
        "independent_links_and_taps": maximum_off_diagonal_covariance <= INDEPENDENCE_ABSOLUTE_TOLERANCE,
        "fractional_delays_preserved": bool(
            np.unique(delays).size == 23
            and np.count_nonzero(np.abs(delays * 1920000 - np.rint(delays * 1920000)) > 1e-6) == 22
        ),
        "dc_sentinel_all_times": dc_error <= 1e-14,
        "output_shape": all_shapes_correct, "all_outputs_finite": all_finite,
        "symbol_wise_quasi_static": SYMBOL_WISE_QUASI_STATIC,
        "no_ici": not ICI_MODELED,
    }
    baseline_commit = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    report = {
        "task": "T08B", "standard_reference": STANDARD_REFERENCE, "standard_url": STANDARD_URL,
        "doppler_standard_clause": "7.7.2", "model": "TDL-A",
        "rms_delay_spread_ns": DESIRED_RMS_DELAY_SPREAD_NS,
        "measured_rms_delay_spread_ns": rms_delay_spread_seconds(delays, powers) / 1e-9,
        **mobility,
        "ofdm_symbol_duration_s": symbol_duration, "time_grid_s": times.tolist(),
        "frame_duration_s": 16 * symbol_duration, "frame_duration_seconds": 16 * symbol_duration,
        "sampled_span_s": float(times[-1] - times[0]),
        "temporal_model": "proper CN(0,R_time), R[m,n]=J0(2*pi*fD*abs(t_m-t_n)); covariance eigendecomposition",
        "jakes_covariance": covariance.tolist(), "jakes_covariance_eigenvalue_summary": eigen_summary,
        "theoretical_temporal_correlations": [{"lag_symbols": row["lag_symbols"], "correlation": row["theoretical"]} for row in unit_correlations],
        "empirical_temporal_correlations": unit_correlations,
        "correlation_sample_count": NUM_CORRELATION_TRAJECTORIES,
        "correlation_estimator": "mean(g[:,lag:]*conj(g[:,:-lag])) / mean(abs(g[:,:-lag])**2); no mean subtraction",
        "correlation_statistical_unit": "independent process trajectory; valid lag pairs within a trajectory are correlated",
        "unit_process_power_by_time": np.mean(np.abs(unit_series) ** 2, axis=0).tolist(),
        "representative_channel_dc_temporal_correlations": channel_correlations,
        "representative_channel_coordinate": [0, 0, 32],
        "pilot_spacing_symbols": pilot_spacing, "pilot_spacing_seconds": pilot_spacing_seconds,
        "pilot_time_spacing_symbols": pilot_spacing, "pilot_time_spacing_seconds": pilot_spacing_seconds,
        "pilot_time_correlation": pilot_correlation,
        "zero_speed_regression": zero_regression,
        "time_variation_nonzero": nonstatic_passed, "time_variation_nonzero_passed": nonstatic_passed,
        "not_iid_across_symbols": not_iid_passed, "not_iid_across_symbols_passed": not_iid_passed,
        "variation_diagnostics": {
            "max_channel_time_difference_from_t0": time_variation_max,
            "representative_dc_increment_power": increment_power,
            "representative_dc_normalized_increment_power": float(normalized_increment),
            "theoretical_normalized_increment_power": theoretical_increment,
            "increment_power_relative_error": increment_relative_error,
        },
        "dc_time_series_max_error": dc_error,
        "ensemble_power_statistics": {
            "num_independent_trajectories": NUM_CHANNEL_TRAJECTORIES,
            "statistical_unit": "trajectory; time-average per trajectory, sample std ddof=1 across trajectories",
            "tap_energy_by_link": energy_statistics,
            "instantaneous_tap_energy_min": float(instantaneous_energy_min),
            "instantaneous_tap_energy_max": float(instantaneous_energy_max),
            "frequency_power_by_link": frequency_statistics,
            "frequency_power_mean_over_trajectories_links_k_t": float(np.mean(link_frequency_power)),
            "target_pdp": powers.tolist(),
            "empirical_tap_power_by_link": {f"rx{r}_tx{q}": empirical_tap_power[r, q].tolist() for r in range(2) for q in range(2)},
            "max_per_link_per_tap_power_relative_error": float(np.max(tap_relative_errors)),
            "max_cross_link_or_tap_standardized_covariance_at_t0": maximum_off_diagonal_covariance,
        },
        "seed_policy": {
            "channel_seed_base": CHANNEL_SEED_BASE, "channel_seed": "channel_seed_base + trajectory_id",
            "channel_trajectory_id_range": [0, NUM_CHANNEL_TRAJECTORIES - 1],
            "unit_correlation_seed": CORRELATION_SEED,
            "unit_correlation_draw": "one RNG batch of 5000 independent process rows",
            "bits_seed": None, "noise_seed": None, "training_seed": None,
        },
        "per_realization_normalization": False, "per_time_normalization": False,
        "symbol_wise_quasi_static": SYMBOL_WISE_QUASI_STATIC, "ici_modeled": ICI_MODELED,
        "antenna_pair_fading": "independent", "output_channel_shape": [2, 2, 64, 16],
        "validation_tolerances": {
            "temporal_correlation_absolute": CORRELATION_ABSOLUTE_TOLERANCE,
            "ensemble_power_absolute": ENSEMBLE_POWER_ABSOLUTE_TOLERANCE,
            "per_link_tap_power_relative": TAP_POWER_RELATIVE_TOLERANCE,
            "increment_power_relative": INCREMENT_POWER_RELATIVE_TOLERANCE,
            "independence_covariance_absolute": INDEPENDENCE_ABSOLUTE_TOLERANCE,
            "unit_process_power_absolute": 0.05,
            "dc_absolute": 1e-14,
        },
        "checks": checks, "all_checks_passed": bool(all(checks.values())),
        "config_path": str(config_path), "system_config": system,
        "environment": {"python_version": platform.python_version(), "numpy_version": np.__version__, "scipy_version": scipy.__version__, "platform": platform.platform()},
        "baseline_git_commit": baseline_commit,
        "source_sha256": {
            path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
            for path in ("src/doppler.py", "src/tdl.py", "scripts/validate_tdl_a_doppler.py")
        },
        "runtime_seconds": perf_counter() - started,
    }

    print("T08B TDL-A 100 ns + 30 km/h Doppler validation")
    for name, value in mobility.items():
        print(f"{name}: {value:.12g}")
    print(f"NFFT: {system['nfft']}; CP: {system['cp_len']}")
    print(f"OFDM symbol duration: {symbol_duration:.12g} s")
    print(f"frame duration: {report['frame_duration_s']:.12g} s")
    print(f"sampled t0-to-t15 span: {report['sampled_span_s']:.12g} s")
    print("Jakes covariance:")
    print(f"min eigenvalue: {eigen_summary['minimum_covariance_eigenvalue']:.16e}")
    print(f"max eigenvalue: {eigen_summary['maximum_covariance_eigenvalue']:.16e}")
    print(f"negative eigenvalues clipped: {eigen_summary['negative_eigenvalues_clipped']}; jitter added: 0")
    print(f"factor reconstruction max error: {eigen_summary['factor_reconstruction_max_error']:.16e}")
    print(f"Theoretical/empirical correlation ({NUM_CORRELATION_TRAJECTORIES} independent unit processes):")
    for row in unit_correlations:
        print(f"  lag {row['lag_symbols']:2}: theory={row['theoretical']:.12f}, real={row['empirical_real']:.12f}, imag={row['empirical_imag']:+.12f}, abs_error={row['absolute_error']:.12f}")
    print(f"pilot spacing dT: {pilot_spacing}; separation: {pilot_spacing_seconds:.12g} s; correlation: {pilot_correlation:.12f}")
    print(f"zero-speed regression: {'PASS' if zero_regression['passed'] else 'FAIL'} (max time difference={zero_regression['max_time_difference']})")
    print(f"30-km/h non-static sentinel: {'PASS' if nonstatic_passed else 'FAIL'}")
    print(f"not-IID sentinel: {'PASS' if not_iid_passed else 'FAIL'}")
    print(f"H[0,0,32,t] lag1: {channel_correlations[0]}")
    print(f"normalized increment power: {normalized_increment:.12g}; theory: {theoretical_increment:.12g}; relative error: {increment_relative_error:.6%}")
    print(f"DC time-series error: {dc_error:.16e}")
    print(f"ensemble tap-energy statistics ({NUM_CHANNEL_TRAJECTORIES} independent trajectories, time average per trajectory):")
    for name, values in energy_statistics.items():
        print(f"  {name}: " + ", ".join(f"{key}={value:.9f}" for key, value in values.items()))
    print(f"maximum per-link/per-tap power relative error: {np.max(tap_relative_errors):.6%}")
    print(f"frequency-domain mean power: {np.mean(link_frequency_power):.12f}")
    for name, values in frequency_statistics.items():
        print(f"  {name}: " + ", ".join(f"{key}={value:.9f}" for key, value in values.items()))
    print(f"symbol-wise quasi-static: {SYMBOL_WISE_QUASI_STATIC}")
    print(f"ICI modeled: {ICI_MODELED}")
    for name, passed in checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"All checks: {'PASS' if report['all_checks_passed'] else 'FAIL'}")
    output_path = PROJECT_ROOT / "outputs" / "t08b_tdl_a_doppler_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report saved: {output_path}")
    print(f"Runtime: {report['runtime_seconds']:.3f} s")
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
