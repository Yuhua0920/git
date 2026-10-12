"""Calibrate the static 100 ns TDL-A generator; no BER/NMSE scan is run."""

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

from src.config import load_config, validate_config  # noqa: E402
from src.numerics import complex_gaussian_noise  # noqa: E402
from src.random_channels import build_random_mimo_channel  # noqa: E402
from src.tdl import (  # noqa: E402
    CHANNEL_MODEL, DESIRED_RMS_DELAY_SPREAD_NS, SOURCE, STANDARD_REFERENCE,
    STANDARD_URL, TDL_A_FADING_DISTRIBUTIONS, TDL_A_NORMALIZED_DELAYS,
    TDL_A_POWERS_DB, build_static_tdl_a_channel, centered_frequency_grid_hz,
    fractional_delay_frequency_response, rms_delay_spread_seconds, tdl_a_profile,
)


NUM_TRAJECTORIES = 1000
CHANNEL_SEED_BASE = 40261001
# Declared before inspecting the ensemble, using the trajectory as sampling unit.
ENERGY_MEAN_ABSOLUTE_TOLERANCE = 0.10
TAP_POWER_RELATIVE_TOLERANCE = 0.15
FREQUENCY_POWER_ABSOLUTE_TOLERANCE = 0.10
ANALYTIC_ABSOLUTE_TOLERANCE = 1e-14


def main() -> None:
    started = perf_counter()
    config_path = PROJECT_ROOT / "configs" / "base.yaml"
    config = load_config(config_path)
    validate_config(config)
    system = config["system"]
    expected = {
        "nt": 2, "nr": 2, "nfft": 64, "active_carriers": 64,
        "num_symbols": 16, "delta_f_hz": 30000, "sample_rate_hz": 1920000,
        "cp_len": 16, "fft_norm": "ortho", "modulation": "qpsk_gray",
        "total_power": 1.0,
    }
    for key, value in expected.items():
        if system[key] != value:
            raise ValueError(f"T08A requires {key}={value}; got {system[key]}")
    delays, powers = tdl_a_profile(DESIRED_RMS_DELAY_SPREAD_NS)
    frequencies = centered_frequency_grid_hz(system["nfft"], system["delta_f_hz"])
    scaled_ns = delays / 1e-9
    measured_rms_ns = rms_delay_spread_seconds(delays, powers) / 1e-9
    cp_duration_ns = system["cp_len"] / system["sample_rate_hz"] * 1e9
    cp_sufficient = bool(cp_duration_ns > np.max(scaled_ns))

    single = fractional_delay_frequency_response([1 + 0j], [250e-9], frequencies)
    phase = 2 * np.pi * frequencies * 250e-9
    analytic = np.cos(phase) - 1j * np.sin(phase)
    single_error = float(np.max(np.abs(single - analytic)))

    taps = np.empty((NUM_TRAJECTORIES, 2, 2, 23), dtype=np.complex128)
    frequency = np.empty((NUM_TRAJECTORIES, 2, 2, 64), dtype=np.complex128)
    time_static = True
    shapes_correct = True
    for i in range(NUM_TRAJECTORIES):
        a, h = build_static_tdl_a_channel(
            CHANNEL_SEED_BASE + i, system["nfft"], system["num_symbols"],
            system["delta_f_hz"], DESIRED_RMS_DELAY_SPREAD_NS,
        )
        taps[i] = a
        frequency[i] = h[..., 0]
        time_static &= bool(np.array_equal(h, np.repeat(h[..., :1], 16, axis=-1)))
        shapes_correct &= a.shape == (2, 2, 23) and h.shape == (2, 2, 64, 16)

    dc_error = float(np.max(np.abs(frequency[..., 32] - np.sum(taps, axis=-1))))
    energies = np.sum(np.abs(taps) ** 2, axis=-1)
    empirical_by_link = np.mean(np.abs(taps) ** 2, axis=0)
    empirical_pooled = np.mean(np.abs(taps) ** 2, axis=(0, 1, 2))
    relative_errors = np.abs(empirical_by_link / powers - 1)
    frequency_power = np.abs(frequency) ** 2
    energy_statistics = {}
    tap_power_by_link = {}
    for r in range(2):
        for q in range(2):
            name = f"rx{r}_tx{q}"
            values = energies[:, r, q]
            energy_statistics[name] = {
                "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)), "max": float(np.max(values)),
            }
            tap_power_by_link[name] = empirical_by_link[r, q].tolist()
    frequency_means_by_link = np.mean(frequency_power, axis=(0, 3))
    frequency_statistics = {
        "mean_over_trajectories_links_subcarriers": float(np.mean(frequency_power)),
        "mean_by_link": {
            f"rx{r}_tx{q}": float(frequency_means_by_link[r, q])
            for r in range(2) for q in range(2)
        },
        "mean_by_subcarrier": np.mean(frequency_power, axis=(0, 1, 2)).tolist(),
        "max_subcarrier_mean_absolute_deviation_from_one": float(
            np.max(np.abs(np.mean(frequency_power, axis=(0, 1, 2)) - 1))
        ),
        "statistical_trajectory_count": NUM_TRAJECTORIES,
        "time_replicas_counted": False,
    }
    expected_first_taps = complex_gaussian_noise(
        (2, 2, 23), 1, np.random.default_rng(CHANNEL_SEED_BASE)
    ) * np.sqrt(powers)
    standardized = (taps / np.sqrt(powers)).reshape(NUM_TRAJECTORIES, 4, 23)
    max_cross_link_covariance = 0.0
    for l in range(23):
        covariance = standardized[:, :, l].T @ standardized[:, :, l].conj() / NUM_TRAJECTORIES
        max_cross_link_covariance = max(
            max_cross_link_covariance,
            float(np.max(np.abs(covariance - np.diag(np.diag(covariance))))),
        )

    # Same seed is a model diagnostic only; equality is not an acceptance gate.
    old_impulse, old_channel = build_random_mimo_channel(CHANNEL_SEED_BASE)
    old_comparison = {
        "channel_seed": CHANNEL_SEED_BASE, "old_nonzero_tap_count": 4,
        "new_tap_count": int(taps.shape[-1]),
        "old_integer_delays_samples": [0, 1, 3, 5],
        "new_continuous_delays_ns": scaled_ns.tolist(),
        "frequency_response_relative_difference": float(
            np.linalg.norm(frequency[0] - old_channel[..., 0])
            / np.linalg.norm(old_channel[..., 0])
        ),
        "old_link_tap_energies": np.sum(np.abs(old_impulse) ** 2, axis=-1).tolist(),
        "new_link_tap_energies": energies[0].tolist(),
        "equality_required": False,
    }
    checks = {
        "tap_count_23": len(TDL_A_NORMALIZED_DELAYS) == len(TDL_A_POWERS_DB) == 23,
        "all_taps_rayleigh": TDL_A_FADING_DISTRIBUTIONS == ("Rayleigh",) * 23,
        "pdp_sums_to_one": abs(float(np.sum(powers)) - 1) < 1e-12,
        "rms_delay_spread_within_0_1ns": abs(measured_rms_ns - 100) < 0.1,
        "max_delay_965_86ns": abs(float(np.max(scaled_ns)) - 965.86) < 1e-10,
        "continuous_delays_preserved": bool(
            np.unique(delays).size == 23
            and np.count_nonzero(np.abs(delays * system["sample_rate_hz"]
                - np.rint(delays * system["sample_rate_hz"])) > 1e-6) == 22
        ),
        "centered_physical_frequency_grid": bool(
            frequencies[0] == -960000 and frequencies[32] == 0
            and frequencies[-1] == 930000 and np.all(np.diff(frequencies) == 30000)
        ),
        "single_tap_analytic": single_error <= ANALYTIC_ABSOLUTE_TOLERANCE,
        "dc_sentinel": dc_error <= ANALYTIC_ABSOLUTE_TOLERANCE,
        "no_realization_scaling": bool(np.array_equal(taps[0], expected_first_taps)),
        "ensemble_link_energy": bool(np.all(np.abs(np.mean(energies, axis=0) - 1)
            <= ENERGY_MEAN_ABSOLUTE_TOLERANCE)),
        "empirical_per_link_per_tap_power": bool(np.all(relative_errors <= TAP_POWER_RELATIVE_TOLERANCE)),
        "ensemble_frequency_power": bool(
            abs(np.mean(frequency_power) - 1) <= FREQUENCY_POWER_ABSOLUTE_TOLERANCE
            and np.all(np.abs(frequency_means_by_link - 1) <= FREQUENCY_POWER_ABSOLUTE_TOLERANCE)
        ),
        "independent_link_draws": max_cross_link_covariance < 0.15,
        "cp_sufficient": cp_sufficient,
        "shape_2x2x64x16": shapes_correct,
        "time_static": time_static,
        "all_generated_values_finite": bool(np.all(np.isfinite(taps)) and np.all(np.isfinite(frequency))),
    }
    baseline_commit = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    report = {
        "task": "T08A", "standard_reference": STANDARD_REFERENCE,
        "source": SOURCE, "standard_url": STANDARD_URL,
        "source_table": "7.7.2-1", "delay_scaling_clause": "7.7.3",
        "model": CHANNEL_MODEL, "tap_count": len(TDL_A_NORMALIZED_DELAYS),
        "normalized_delays": list(TDL_A_NORMALIZED_DELAYS),
        "powers_db": list(TDL_A_POWERS_DB), "fading_distributions": list(TDL_A_FADING_DISTRIBUTIONS),
        "normalized_linear_powers": powers.tolist(),
        "desired_rms_delay_spread_ns": DESIRED_RMS_DELAY_SPREAD_NS,
        "scaled_delays_ns": scaled_ns.tolist(),
        "mean_excess_delay_ns": float(np.sum(powers * scaled_ns)),
        "measured_rms_delay_spread_ns": measured_rms_ns,
        "maximum_delay_ns": float(np.max(scaled_ns)),
        "sample_rate_hz": system["sample_rate_hz"],
        "sample_interval_ns": 1e9 / system["sample_rate_hz"],
        "cp_samples": system["cp_len"], "cp_duration_ns": cp_duration_ns,
        "cp_sufficient": cp_sufficient,
        "fractional_delay_method": "H[k] = sum_l a_l exp(-j*2*pi*f_k*tau_l); tau in seconds, f in baseband Hz",
        "integer_delay_rounding_used": False, "frequencies_hz": frequencies.tolist(),
        "single_tap_delay_ns": 250, "single_tap_analytic_error": single_error,
        "dc_sentinel_error": dc_error, "dc_sentinel_trajectory_count": NUM_TRAJECTORIES,
        "num_statistical_trajectories": NUM_TRAJECTORIES,
        "seed_policy": {
            "channel_seed_base": CHANNEL_SEED_BASE,
            "channel_seed": "channel_seed_base + trajectory_id",
            "trajectory_id_range": [0, NUM_TRAJECTORIES - 1],
            "bits_seed": None, "noise_seed": None, "training_seed": None,
        },
        "channel_energy_statistics": energy_statistics,
        "channel_energy_statistical_unit": "trajectory; sample std ddof=1",
        "empirical_tap_power_statistics": {
            "target_pdp": powers.tolist(), "empirical_power_by_link": tap_power_by_link,
            "empirical_power_pooled_over_links": empirical_pooled.tolist(),
            "relative_error_by_link": {
                f"rx{r}_tx{q}": relative_errors[r, q].tolist() for r in range(2) for q in range(2)
            },
            "max_relative_error_by_link": float(np.max(relative_errors)),
            "max_relative_error_pooled": float(np.max(np.abs(empirical_pooled / powers - 1))),
        },
        "max_standardized_cross_link_covariance": max_cross_link_covariance,
        "frequency_power_statistics": frequency_statistics,
        "per_realization_normalization": False, "time_varying": False,
        "antenna_pair_fading": "independent", "output_channel_shape": [2, 2, 64, 16],
        "old_four_tap_same_seed_diagnostic": old_comparison,
        "validation_tolerances": {
            "rms_delay_spread_absolute_ns": 0.1,
            "link_energy_mean_absolute": ENERGY_MEAN_ABSOLUTE_TOLERANCE,
            "per_link_per_tap_power_relative": TAP_POWER_RELATIVE_TOLERANCE,
            "frequency_power_absolute": FREQUENCY_POWER_ABSOLUTE_TOLERANCE,
            "analytic_absolute": ANALYTIC_ABSOLUTE_TOLERANCE,
            "standardized_cross_link_covariance": 0.15,
        },
        "checks": checks, "all_checks_passed": bool(all(checks.values())),
        "config_path": str(config_path), "system_config": system,
        "environment": {"python_version": platform.python_version(), "numpy_version": np.__version__, "platform": platform.platform()},
        "baseline_git_commit": baseline_commit,
        "source_sha256": {
            path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
            for path in ("src/tdl.py", "scripts/validate_tdl_a_static.py")
        },
        "runtime_seconds": perf_counter() - started,
    }

    print("T08A 3GPP TDL-A static validation")
    print(f"source: {SOURCE}; {STANDARD_REFERENCE}; Clause 7.7.3")
    print(f"tap count: {report['tap_count']} (all Rayleigh, original table order)")
    print(f"desired RMS delay spread: {DESIRED_RMS_DELAY_SPREAD_NS:.9f} ns")
    print(f"measured RMS delay spread: {measured_rms_ns:.9f} ns")
    print(f"min scaled delay: {np.min(scaled_ns):.9f} ns")
    print(f"max scaled delay: {np.max(scaled_ns):.9f} ns")
    print(f"Fs: {system['sample_rate_hz']} Hz")
    print(f"sample interval: {report['sample_interval_ns']:.9f} ns")
    print(f"CP samples: {system['cp_len']}")
    print(f"CP duration: {cp_duration_ns:.9f} ns")
    print(f"CP sufficient: {cp_sufficient}")
    print(f"first five normalized delays: {list(TDL_A_NORMALIZED_DELAYS[:5])}")
    print(f"first five scaled delays (ns): {scaled_ns[:5].tolist()}")
    print(f"first five powers dB: {list(TDL_A_POWERS_DB[:5])}")
    print(f"first five normalized linear powers: {powers[:5].tolist()}")
    print(f"single fractional tap analytic error: {single_error:.16e}")
    print(f"DC sentinel error (all trajectories/links): {dc_error:.16e}")
    print(f"trajectory statistics: {NUM_TRAJECTORIES}; channel seeds {CHANNEL_SEED_BASE}..{CHANNEL_SEED_BASE + NUM_TRAJECTORIES - 1}")
    print("mean tap energy by MIMO link; std/min/max:")
    for name, values in energy_statistics.items():
        print(f"  {name}: " + ", ".join(f"{key}={value:.9f}" for key, value in values.items()))
    print(f"maximum per-link per-tap empirical power relative error: {np.max(relative_errors):.9%}")
    print(f"maximum pooled per-tap empirical power relative error: {report['empirical_tap_power_statistics']['max_relative_error_pooled']:.9%}")
    print(f"empirical frequency power: {np.mean(frequency_power):.9f}")
    print(f"empirical frequency power by link: {frequency_statistics['mean_by_link']}")
    print(f"per-realization normalization: {report['per_realization_normalization']}")
    print(f"integer delay rounding: {report['integer_delay_rounding_used']}")
    print(f"time variation: {report['time_varying']}")
    for name, passed in checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"All checks: {'PASS' if report['all_checks_passed'] else 'FAIL'}")

    output_path = PROJECT_ROOT / "outputs" / "t08a_tdl_a_static_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report saved: {output_path}")
    print(f"Runtime: {report['runtime_seconds']:.3f} s")
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
