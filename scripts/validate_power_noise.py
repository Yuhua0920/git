"""Run the T01 noise and transmit-power checks from configs/base.yaml."""

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, validate_config  # noqa: E402
from src.numerics import (  # noqa: E402
    complex_gaussian_noise,
    noise_variance_from_snr,
    scale_data_streams,
)


NUM_NOISE_SAMPLES = 1_000_000
NUM_DATA_RE = 1_000_000
VALIDATION_SNR_DB = (0, 10, 20)
NOISE_RELATIVE_TOLERANCE = 0.01
MEAN_STANDARD_ERROR_MULTIPLIER = 5.0
POWER_ABSOLUTE_TOLERANCE = 1e-12


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    validate_config(config)
    system = config["system"]
    if system["nt"] != 2 or system["total_power"] != 1.0:
        raise ValueError("T01 requires nt=2 and total_power=1.0")
    if not set(VALIDATION_SNR_DB).issubset(config["noise"]["snr_db"]):
        raise ValueError("base.yaml must contain the T01 SNR points 0, 10, and 20 dB")

    seed = config["seed"]["base_seed"]
    rng = np.random.default_rng(seed)
    noise_results = []
    for snr_db in VALIDATION_SNR_DB:
        variance = noise_variance_from_snr(snr_db, system["total_power"])
        noise = complex_gaussian_noise(
            NUM_NOISE_SAMPLES, variance, rng, dtype=np.complex128
        )
        empirical_power = float(np.mean(np.abs(noise) ** 2))
        real_mean = float(np.mean(noise.real))
        imag_mean = float(np.mean(noise.imag))
        real_variance = float(np.var(noise.real))
        imag_variance = float(np.var(noise.imag))
        component_target = variance / 2.0
        mean_limit = MEAN_STANDARD_ERROR_MULTIPLIER * np.sqrt(
            component_target / NUM_NOISE_SAMPLES
        )
        relative_error = abs(empirical_power - variance) / variance
        real_variance_relative_error = abs(real_variance - component_target) / component_target
        imag_variance_relative_error = abs(imag_variance - component_target) / component_target
        noise_results.append(
            {
                "snr_db": snr_db,
                "target_variance": variance,
                "empirical_power": empirical_power,
                "relative_error": relative_error,
                "real_mean": real_mean,
                "imag_mean": imag_mean,
                "real_variance": real_variance,
                "imag_variance": imag_variance,
                "real_variance_relative_error": real_variance_relative_error,
                "imag_variance_relative_error": imag_variance_relative_error,
                "mean_limit": float(mean_limit),
                "power_passed": relative_error <= NOISE_RELATIVE_TOLERANCE,
                "components_passed": (
                    abs(real_mean) <= mean_limit
                    and abs(imag_mean) <= mean_limit
                    and real_variance_relative_error <= NOISE_RELATIVE_TOLERANCE
                    and imag_variance_relative_error <= NOISE_RELATIVE_TOLERANCE
                ),
            }
        )

    unit_symbols = np.array([1 + 0j, -1 + 0j, 1j, -1j], dtype=np.complex128)
    positions = np.arange(NUM_DATA_RE) % len(unit_symbols)
    data_symbols = np.column_stack(
        (unit_symbols[positions], unit_symbols[(3 * positions + 1) % len(unit_symbols)])
    )
    tx_data = scale_data_streams(
        data_symbols, nt=system["nt"], total_power=system["total_power"]
    )
    stream_power = np.mean(np.abs(tx_data) ** 2, axis=0)
    total_data_power = float(np.mean(np.sum(np.abs(tx_data) ** 2, axis=1)))
    data_power = {
        "tx0": float(stream_power[0]),
        "tx1": float(stream_power[1]),
        "total": total_data_power,
        "passed": bool(
            np.isclose(stream_power[0], 0.5, rtol=0, atol=POWER_ABSOLUTE_TOLERANCE)
            and np.isclose(stream_power[1], 0.5, rtol=0, atol=POWER_ABSOLUTE_TOLERANCE)
            and np.isclose(total_data_power, 1.0, rtol=0, atol=POWER_ABSOLUTE_TOLERANCE)
        ),
    }

    pilot_tx0 = np.array([1 + 0j, 0 + 0j], dtype=np.complex128)
    pilot_tx1 = np.array([0 + 0j, 1 + 0j], dtype=np.complex128)
    pilot_power = {
        "tx0_pilot_re": {
            "tx0": float(abs(pilot_tx0[0]) ** 2),
            "tx1": float(abs(pilot_tx0[1]) ** 2),
            "total": float(np.sum(np.abs(pilot_tx0) ** 2)),
        },
        "tx1_pilot_re": {
            "tx0": float(abs(pilot_tx1[0]) ** 2),
            "tx1": float(abs(pilot_tx1[1]) ** 2),
            "total": float(np.sum(np.abs(pilot_tx1) ** 2)),
        },
    }
    pilot_power["passed"] = (
        pilot_power["tx0_pilot_re"] == {"tx0": 1.0, "tx1": 0.0, "total": 1.0}
        and pilot_power["tx1_pilot_re"] == {"tx0": 0.0, "tx1": 1.0, "total": 1.0}
    )

    snr_conversion_passed = all(
        np.isclose(result["target_variance"], expected, rtol=0, atol=1e-15)
        for result, expected in zip(noise_results, (1.0, 0.1, 0.01), strict=True)
    )
    all_checks_passed = bool(
        snr_conversion_passed
        and all(result["power_passed"] and result["components_passed"] for result in noise_results)
        and data_power["passed"]
        and pilot_power["passed"]
    )
    report = {
        "task": "T01",
        "seed": seed,
        "noise_seed": seed,
        "num_noise_samples": NUM_NOISE_SAMPLES,
        "num_data_re": NUM_DATA_RE,
        "noise_relative_tolerance": NOISE_RELATIVE_TOLERANCE,
        "mean_standard_error_multiplier": MEAN_STANDARD_ERROR_MULTIPLIER,
        "noise_results": noise_results,
        "snr_conversion_passed": bool(snr_conversion_passed),
        "data_power": data_power,
        "pilot_power": pilot_power,
        "all_checks_passed": all_checks_passed,
    }

    print(f"T01 seed: {seed}; noise samples per SNR: {NUM_NOISE_SAMPLES}; data REs: {NUM_DATA_RE}")
    print("SNR(dB)  target_variance  empirical_power  relative_error")
    for result in noise_results:
        print(
            f"{result['snr_db']:>7}  {result['target_variance']:>15.8f}  "
            f"{result['empirical_power']:>15.8f}  {result['relative_error']:>14.6%}"
        )
    print("SNR(dB)  real_mean       imag_mean       real_variance   imag_variance")
    for result in noise_results:
        print(
            f"{result['snr_db']:>7}  {result['real_mean']:>+14.8f}  "
            f"{result['imag_mean']:>+14.8f}  {result['real_variance']:>13.8f}  "
            f"{result['imag_variance']:>13.8f}"
        )
    print(f"TX0 data power: {data_power['tx0']:.12f}")
    print(f"TX1 data power: {data_power['tx1']:.12f}")
    print(f"Total data RE power: {data_power['total']:.12f}")
    print(f"TX0 pilot RE power: {pilot_power['tx0_pilot_re']['total']:.12f}")
    print(f"TX1 pilot RE power: {pilot_power['tx1_pilot_re']['total']:.12f}")
    print(f"All checks: {'PASS' if all_checks_passed else 'FAIL'}")

    output_path = PROJECT_ROOT / "outputs" / "t01_power_noise_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report saved: {output_path}")
    if not all_checks_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
