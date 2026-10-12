"""Validate the single-stream Gray-QPSK AWGN BER against theory.

The horizontal axis is Eb/N0, not the main MIMO system's SNR setting.
The plot can be regenerated without simulation via ``--plot-only``.
"""

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if "MPLCONFIGDIR" not in os.environ:
    matplotlib_cache = Path(tempfile.gettempdir()) / "t02_qpsk_mpl_cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, validate_config  # noqa: E402
from src.modulation import (  # noqa: E402
    ebn0_to_esn0_db,
    ebn0_to_noise_variance,
    qpsk_awgn_theoretical_ber,
    qpsk_demodulate,
    qpsk_modulate,
)
from src.numerics import complex_gaussian_noise, noise_variance_from_snr  # noqa: E402


EBN0_DB_POINTS = (0, 2, 4, 6, 8)
NUM_BITS_PER_POINT = 2_000_000
CSV_COLUMNS = (
    "ebn0_db",
    "esn0_db",
    "noise_variance",
    "num_bits",
    "num_errors",
    "simulated_ber",
    "theoretical_ber",
    "standard_error",
    "tolerance",
    "absolute_error",
    "relative_error",
    "passed",
    "bits_seed",
    "noise_seed",
)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CSV_PATH = OUTPUT_DIR / "t02_qpsk_awgn_ber.csv"
JSON_PATH = OUTPUT_DIR / "t02_qpsk_awgn_results.json"
PLOT_PATH = OUTPUT_DIR / "t02_qpsk_awgn_ber.png"


def plot_ber_from_csv(csv_path: Path, plot_path: Path) -> None:
    """Rebuild the T02 semilog plot solely from the saved CSV statistics."""
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError("BER CSV contains no data rows")

    ebn0_db = [float(row["ebn0_db"]) for row in rows]
    theory = [float(row["theoretical_ber"]) for row in rows]
    simulation = [float(row["simulated_ber"]) for row in rows]
    figure, axes = plt.subplots(figsize=(6.5, 4.5))
    axes.semilogy(ebn0_db, theory, "-o", label="Theory QPSK")
    axes.semilogy(ebn0_db, simulation, "--s", label="Simulation QPSK")
    axes.set_xlabel("Eb/N0 (dB)")
    axes.set_ylabel("BER")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only", action="store_true", help="regenerate the PNG from the existing CSV"
    )
    args = parser.parse_args()
    if args.plot_only:
        plot_ber_from_csv(CSV_PATH, PLOT_PATH)
        print(f"Plot regenerated from CSV: {PLOT_PATH}")
        return

    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    validate_config(config)
    if config["system"]["total_power"] != 1.0:
        raise ValueError("T02 single-stream constellation requires unit symbol power")
    if config["system"]["modulation"] != "qpsk_gray":
        raise ValueError("T02 requires qpsk_gray modulation")
    # T02 is single-stream. The main system's nt=2 and SNR list are not used here.
    bits_seed = config["seed"]["base_seed"]
    bits_rng = np.random.default_rng(bits_seed)
    bits = bits_rng.integers(0, 2, size=NUM_BITS_PER_POINT, dtype=np.uint8)
    symbols = qpsk_modulate(bits)
    noiseless_recovered = qpsk_demodulate(symbols)
    noiseless_bit_errors = int(np.count_nonzero(noiseless_recovered != bits))
    noiseless_roundtrip_passed = noiseless_bit_errors == 0
    average_symbol_energy = float(np.mean(np.abs(symbols) ** 2))
    energy_passed = bool(np.isclose(average_symbol_energy, 1.0, rtol=0, atol=1e-12))

    mapping_bits = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
    mapping_symbols = np.array([1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j]) / np.sqrt(2.0)
    mapping_passed = bool(np.array_equal(qpsk_modulate(mapping_bits), mapping_symbols))
    theory_known_value_passed = bool(
        np.isclose(qpsk_awgn_theoretical_ber(0), 0.07864960352514257, rtol=0, atol=1e-14)
    )

    results = []
    for index, ebn0_db in enumerate(EBN0_DB_POINTS):
        noise_seed = bits_seed + index + 1
        esn0_db = ebn0_to_esn0_db(ebn0_db)
        noise_variance = ebn0_to_noise_variance(ebn0_db)
        via_esn0 = noise_variance_from_snr(esn0_db, total_power=1.0)
        conversion_passed = bool(np.isclose(noise_variance, via_esn0, rtol=1e-12, atol=0))

        noise_rng = np.random.default_rng(noise_seed)
        received = symbols + complex_gaussian_noise(
            symbols.shape, noise_variance, noise_rng, dtype=np.complex128
        )
        recovered = qpsk_demodulate(received)
        num_errors = int(np.count_nonzero(recovered != bits))
        simulated_ber = num_errors / NUM_BITS_PER_POINT
        theoretical_ber = qpsk_awgn_theoretical_ber(ebn0_db)
        standard_error = float(
            np.sqrt(theoretical_ber * (1.0 - theoretical_ber) / NUM_BITS_PER_POINT)
        )
        tolerance = max(
            5.0 * standard_error,
            0.05 * theoretical_ber,
            2.0 / NUM_BITS_PER_POINT,
        )
        absolute_error = abs(simulated_ber - theoretical_ber)
        relative_error = absolute_error / theoretical_ber if theoretical_ber > 0 else None
        results.append(
            {
                "ebn0_db": ebn0_db,
                "esn0_db": esn0_db,
                "noise_variance": noise_variance,
                "num_bits": NUM_BITS_PER_POINT,
                "num_errors": num_errors,
                "simulated_ber": simulated_ber,
                "theoretical_ber": theoretical_ber,
                "standard_error": standard_error,
                "tolerance": tolerance,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "passed": absolute_error <= tolerance,
                "bits_seed": bits_seed,
                "noise_seed": noise_seed,
                "conversion_passed": conversion_passed,
            }
        )

    strictly_decreasing = all(
        earlier["simulated_ber"] > later["simulated_ber"]
        for earlier, later in zip(results, results[1:], strict=False)
    )
    trend_passed = results[-1]["simulated_ber"] < results[0]["simulated_ber"]
    conversion_passed = all(result["conversion_passed"] for result in results)
    all_ber_checks_passed = all(result["passed"] for result in results)
    all_checks_passed = bool(
        mapping_passed
        and energy_passed
        and noiseless_roundtrip_passed
        and theory_known_value_passed
        and conversion_passed
        and all_ber_checks_passed
        and trend_passed
    )

    report = {
        "task": "T02",
        "mapping_definition": {
            "00": "(+1+j)/sqrt(2)",
            "01": "(+1-j)/sqrt(2)",
            "11": "(-1-j)/sqrt(2)",
            "10": "(-1+j)/sqrt(2)",
        },
        "axis_definition": "Eb/N0 (dB), single-stream Es=1, Eb=1/2",
        "num_bits_per_point": NUM_BITS_PER_POINT,
        "bits_seed": bits_seed,
        "noise_seed_rule": "bits_seed + point_index + 1",
        "results": results,
        "noiseless_bit_errors": noiseless_bit_errors,
        "noiseless_roundtrip_passed": noiseless_roundtrip_passed,
        "average_symbol_energy": average_symbol_energy,
        "mapping_passed": mapping_passed,
        "energy_passed": energy_passed,
        "theory_known_value_passed": theory_known_value_passed,
        "conversion_passed": conversion_passed,
        "strictly_decreasing": strictly_decreasing,
        "trend_passed": trend_passed,
        "all_ber_checks_passed": all_ber_checks_passed,
        "all_checks_passed": all_checks_passed,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow({column: result[column] for column in CSV_COLUMNS})
    JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_ber_from_csv(CSV_PATH, PLOT_PATH)

    print("T02 single-stream Gray-QPSK + AWGN; horizontal axis: Eb/N0 (dB)")
    print(f"Bits per point: {NUM_BITS_PER_POINT}; bits seed: {bits_seed}")
    print(
        f"Noiseless roundtrip: {'PASS' if noiseless_roundtrip_passed else 'FAIL'} "
        f"({noiseless_bit_errors} errors / {NUM_BITS_PER_POINT} bits)"
    )
    print(f"QPSK average energy: {average_symbol_energy:.12f}")
    print("Eb/N0  Errors   Bits       Sim BER       Theory BER    Difference    Tolerance   Pass")
    for result in results:
        print(
            f"{result['ebn0_db']:>5}  {result['num_errors']:>6}  "
            f"{result['num_bits']:>8}  {result['simulated_ber']:>12.8f}  "
            f"{result['theoretical_ber']:>12.8f}  {result['absolute_error']:>12.8f}  "
            f"{result['tolerance']:>11.8f}  {'PASS' if result['passed'] else 'FAIL'}"
        )
    print(
        f"Trend: {'PASS' if trend_passed else 'FAIL'} "
        f"(strictly decreasing: {strictly_decreasing})"
    )
    print(f"All BER checks: {'PASS' if all_checks_passed else 'FAIL'}")
    print(f"CSV saved: {CSV_PATH}")
    print(f"JSON saved: {JSON_PATH}")
    print(f"Plot saved from CSV: {PLOT_PATH}")
    if not all_checks_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
