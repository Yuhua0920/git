"""Validate centered OFDM transforms and CP without channel or noise."""

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, validate_config  # noqa: E402
from src.modulation import qpsk_modulate  # noqa: E402
from src.ofdm import (  # noqa: E402
    ofdm_demodulate_grid,
    ofdm_modulate_grid,
    remove_cyclic_prefix,
)


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    validate_config(config)
    system = config["system"]
    expected = {
        "nfft": 64,
        "active_carriers": 64,
        "num_symbols": 16,
        "cp_len": 16,
        "fft_norm": "ortho",
        "modulation": "qpsk_gray",
    }
    for key, value in expected.items():
        if system[key] != value:
            raise ValueError(f"T03A requires {key}={value}; got {system[key]}")

    nfft = system["nfft"]
    num_symbols = system["num_symbols"]
    cp_len = system["cp_len"]
    bits_seed = config["seed"]["base_seed"]
    if bits_seed != 20260930:
        raise ValueError(f"T03A requires bits_seed=20260930; got {bits_seed}")
    rng = np.random.default_rng(bits_seed)
    num_bits = nfft * num_symbols * 2
    bits = rng.integers(0, 2, size=num_bits, dtype=np.uint8)
    frequency_grid = qpsk_modulate(bits).reshape(nfft, num_symbols)

    time_without_cp = ofdm_modulate_grid(frequency_grid, nfft, cp_len=0)
    time_with_cp = ofdm_modulate_grid(frequency_grid, nfft, cp_len)
    useful_time = remove_cyclic_prefix(time_with_cp, nfft, cp_len)
    recovered_grid = ofdm_demodulate_grid(time_with_cp, nfft, cp_len)

    roundtrip_relative_error = float(
        np.linalg.norm(recovered_grid - frequency_grid) / np.linalg.norm(frequency_grid)
    )
    frequency_energy = np.sum(np.abs(frequency_grid) ** 2, axis=0)
    useful_time_energy = np.sum(np.abs(time_without_cp) ** 2, axis=0)
    max_parseval_relative_error = float(
        np.max(np.abs(useful_time_energy - frequency_energy) / frequency_energy)
    )
    max_cp_copy_error = float(
        np.max(np.abs(time_with_cp[:cp_len, :] - time_without_cp[-cp_len:, :]))
    )
    max_cp_removal_error = float(np.max(np.abs(useful_time - time_without_cp)))
    cp_copy_exact = bool(np.array_equal(time_with_cp[:cp_len, :], time_without_cp[-cp_len:, :]))
    cp_removal_exact = bool(np.array_equal(useful_time, time_without_cp))

    dc_grid = np.zeros((nfft, 1), dtype=np.complex128)
    dc_grid[nfft // 2, 0] = 1.0
    dc_waveform = ofdm_modulate_grid(dc_grid, nfft, cp_len=0)[:, 0]
    dc_waveform_max_error = float(np.max(np.abs(dc_waveform - 1.0 / np.sqrt(nfft))))

    lowest_grid = np.zeros((nfft, 1), dtype=np.complex128)
    lowest_grid[0, 0] = 1.0
    lowest_waveform = ofdm_modulate_grid(lowest_grid, nfft, cp_len=0)[:, 0]
    sample_indices = np.arange(nfft)
    # Centered index 0 is k=-nfft/2, so the unitary inverse DFT is exp(-j*pi*n)/sqrt(nfft).
    expected_lowest_waveform = np.exp(-1j * np.pi * sample_indices) / np.sqrt(nfft)
    lowest_bin_waveform_max_error = float(
        np.max(np.abs(lowest_waveform - expected_lowest_waveform))
    )

    block_samples = nfft + cp_len
    cp_ratio_vs_useful = cp_len / nfft
    all_checks_passed = bool(
        frequency_grid.shape == (64, 16)
        and time_without_cp.shape == (64, 16)
        and time_with_cp.shape == (80, 16)
        and roundtrip_relative_error <= 1e-10
        and max_parseval_relative_error <= 1e-12
        and cp_copy_exact
        and cp_removal_exact
        and dc_waveform_max_error <= 1e-12
        and lowest_bin_waveform_max_error <= 1e-12
        and cp_ratio_vs_useful == 0.25
        and block_samples == 80
    )
    report = {
        "task": "T03A",
        "nfft": nfft,
        "num_symbols": num_symbols,
        "cp_len": cp_len,
        "frequency_order": "centered physical frequency ascending (fftshifted)",
        "frequency_axis": 0,
        "fft_norm": system["fft_norm"],
        "input_shape": list(frequency_grid.shape),
        "time_shape_without_cp": list(time_without_cp.shape),
        "time_shape_with_cp": list(time_with_cp.shape),
        "bits_seed": bits_seed,
        "num_bits": num_bits,
        "roundtrip_relative_error": roundtrip_relative_error,
        "max_parseval_relative_error": max_parseval_relative_error,
        "max_cp_copy_error": max_cp_copy_error,
        "max_cp_removal_error": max_cp_removal_error,
        "cp_copy_exact": cp_copy_exact,
        "cp_removal_exact": cp_removal_exact,
        "dc_waveform_max_error": dc_waveform_max_error,
        "lowest_bin_waveform_max_error": lowest_bin_waveform_max_error,
        "useful_samples": nfft,
        "cp_samples": cp_len,
        "block_samples": block_samples,
        "cp_ratio_vs_useful": cp_ratio_vs_useful,
        "all_checks_passed": all_checks_passed,
    }

    print("T03A OFDM validation")
    print(f"Grid shape: {frequency_grid.shape}")
    print(f"Nfft: {nfft}")
    print(f"OFDM symbols: {num_symbols}")
    print(f"CP length: {cp_len}")
    print(f"Output time-grid shape: {time_with_cp.shape}")
    print(f"Roundtrip relative error: {roundtrip_relative_error:.16e}")
    print(f"Maximum Parseval relative error: {max_parseval_relative_error:.16e}")
    print(f"Maximum CP-copy error: {max_cp_copy_error:.16e}")
    print(f"Maximum CP-removal error: {max_cp_removal_error:.16e}")
    print(f"DC waveform maximum error: {dc_waveform_max_error:.16e}")
    print(f"Lowest-bin test error: {lowest_bin_waveform_max_error:.16e}")
    print(f"Useful samples: {nfft}")
    print(f"CP samples: {cp_len}")
    print(f"Block samples: {block_samples}")
    print(f"CP/useful ratio: {cp_ratio_vs_useful:.2f}")
    print(f"All checks: {'PASS' if all_checks_passed else 'FAIL'}")

    output_path = PROJECT_ROOT / "outputs" / "t03a_ofdm_roundtrip_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report saved: {output_path}")
    if not all_checks_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
