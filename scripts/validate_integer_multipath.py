"""Validate deterministic integer-delay multipath in CP-OFDM, without noise."""

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.channels import (  # noqa: E402
    build_integer_multipath_channel,
    channel_frequency_response,
    convolve_cp_ofdm_blocks,
)
from src.config import load_config, validate_config  # noqa: E402
from src.modulation import qpsk_modulate  # noqa: E402
from src.ofdm import ofdm_modulate_grid  # noqa: E402


DELAYS_SAMPLES = (0, 1, 3, 5)
POWERS_DB = (0.0, -3.0, -6.0, -9.0)
PHASES_RAD = (0.0, np.pi / 5.0, -np.pi / 3.0, np.pi / 2.0)
SHORT_CP_LEN = 4


def centered_unitary_fft(useful_time_grid: np.ndarray) -> np.ndarray:
    """Apply the receiver's unitary FFT on the sample/frequency axis 0."""
    unshifted = np.fft.fft(useful_time_grid, axis=0, norm="ortho")
    return np.fft.fftshift(unshifted, axes=0)


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
            raise ValueError(f"T03B requires {key}={value}; got {system[key]}")
    bits_seed = config["seed"]["base_seed"]
    if bits_seed != 20260930:
        raise ValueError(f"T03B requires bits_seed=20260930; got {bits_seed}")

    nfft = system["nfft"]
    cp_len = system["cp_len"]
    num_symbols = system["num_symbols"]
    h, normalized_path_powers = build_integer_multipath_channel(
        DELAYS_SAMPLES, POWERS_DB, PHASES_RAD, dtype=np.complex128
    )
    channel_length = len(h)
    required_min_cp = channel_length - 1
    cp_sufficient = cp_len >= required_min_cp
    normalized_path_power_sum = float(np.sum(normalized_path_powers))
    impulse_response_energy = float(np.sum(np.abs(h) ** 2))

    h_pad = np.zeros(nfft, dtype=np.complex128)
    h_pad[:channel_length] = h
    h_ordinary = np.fft.fft(h_pad, n=nfft, norm=None)
    h_scaled_unitary = np.sqrt(nfft) * np.fft.fft(h_pad, n=nfft, norm="ortho")
    ordinary_vs_scaled_unitary_relative_error = float(
        np.linalg.norm(h_ordinary - h_scaled_unitary) / np.linalg.norm(h_ordinary)
    )
    h_centered = channel_frequency_response(h, nfft)
    frequency_response_average_power = float(np.mean(np.abs(h_centered) ** 2))

    rng = np.random.default_rng(bits_seed)
    num_bits = nfft * num_symbols * 2
    bits = rng.integers(0, 2, size=num_bits, dtype=np.uint8)
    x_grid = qpsk_modulate(bits).reshape(nfft, num_symbols)
    x_with_cp = ofdm_modulate_grid(x_grid, nfft, cp_len)
    y_useful = convolve_cp_ofdm_blocks(x_with_cp, h, nfft, cp_len)
    y_td = centered_unitary_fft(y_useful)
    y_fd = h_centered[:, None] * x_grid
    time_frequency_relative_error = float(np.linalg.norm(y_td - y_fd) / np.linalg.norm(y_fd))
    time_frequency_max_absolute_error = float(np.max(np.abs(y_td - y_fd)))

    dc_grid = np.zeros((nfft, 1), dtype=np.complex128)
    dc_grid[nfft // 2, 0] = 1.0
    dc_with_cp = ofdm_modulate_grid(dc_grid, nfft, cp_len)
    dc_useful = convolve_cp_ofdm_blocks(dc_with_cp, h, nfft, cp_len)
    dc_td = centered_unitary_fft(dc_useful)
    dc_fd = h_centered[:, None] * dc_grid
    dc_diagnostic_error = float(np.max(np.abs(dc_td - dc_fd)))
    dc_active_bin_error = float(abs(dc_td[nfft // 2, 0] - h_centered[nfft // 2]))
    dc_other_bins_max = float(np.max(np.abs(np.delete(dc_td[:, 0], nfft // 2))))

    short_cp_with_cp = ofdm_modulate_grid(x_grid, nfft, SHORT_CP_LEN)
    short_cp_useful = convolve_cp_ofdm_blocks(short_cp_with_cp, h, nfft, SHORT_CP_LEN)
    short_cp_td = centered_unitary_fft(short_cp_useful)
    short_cp_relative_error = float(np.linalg.norm(short_cp_td - y_fd) / np.linalg.norm(y_fd))
    short_cp_breaks_equivalence = bool(short_cp_relative_error > 1e-9)

    h_wrong_unitary = np.fft.fftshift(np.fft.fft(h_pad, n=nfft, norm="ortho"))
    wrong_unitary_h_relative_error = float(
        np.linalg.norm(y_td - h_wrong_unitary[:, None] * x_grid) / np.linalg.norm(y_fd)
    )
    scale_sentinel_passed = bool(
        wrong_unitary_h_relative_error > 0.1
        and wrong_unitary_h_relative_error > 1000.0 * time_frequency_relative_error
    )

    all_checks_passed = bool(
        abs(normalized_path_power_sum - 1.0) <= 1e-12
        and abs(impulse_response_energy - 1.0) <= 1e-12
        and abs(frequency_response_average_power - 1.0) <= 1e-12
        and ordinary_vs_scaled_unitary_relative_error <= 1e-12
        and cp_sufficient
        and x_with_cp.shape == (80, 16)
        and y_td.shape == y_fd.shape == (64, 16)
        and time_frequency_relative_error <= 1e-9
        and dc_diagnostic_error <= 1e-12
        and dc_active_bin_error <= 1e-12
        and dc_other_bins_max <= 1e-12
        and SHORT_CP_LEN < required_min_cp
        and short_cp_breaks_equivalence
        and scale_sentinel_passed
    )
    report = {
        "task": "T03B",
        "delays_samples": list(DELAYS_SAMPLES),
        "powers_db": list(POWERS_DB),
        "powers_linear_normalized": normalized_path_powers.tolist(),
        "normalized_path_power_sum": normalized_path_power_sum,
        "phases_rad": list(PHASES_RAD),
        "impulse_response_real_imag": [
            {"real": float(tap.real), "imag": float(tap.imag)} for tap in h
        ],
        "channel_length": channel_length,
        "impulse_response_energy": impulse_response_energy,
        "nfft": nfft,
        "num_symbols": num_symbols,
        "bits_seed": bits_seed,
        "num_bits": num_bits,
        "cp_len": cp_len,
        "required_min_cp": required_min_cp,
        "cp_sufficient": cp_sufficient,
        "frequency_response_average_power": frequency_response_average_power,
        "ordinary_vs_scaled_unitary_relative_error": ordinary_vs_scaled_unitary_relative_error,
        "time_domain_y_shape": list(y_td.shape),
        "frequency_domain_y_shape": list(y_fd.shape),
        "time_frequency_relative_error": time_frequency_relative_error,
        "time_frequency_max_absolute_error": time_frequency_max_absolute_error,
        "dc_diagnostic_error": dc_diagnostic_error,
        "dc_active_bin_error": dc_active_bin_error,
        "dc_other_bins_max": dc_other_bins_max,
        "short_cp_len": SHORT_CP_LEN,
        "short_cp_relative_error": short_cp_relative_error,
        "short_cp_breaks_equivalence": short_cp_breaks_equivalence,
        "wrong_unitary_h_relative_error": wrong_unitary_h_relative_error,
        "scale_sentinel_passed": scale_sentinel_passed,
        "all_checks_passed": all_checks_passed,
    }

    print("T03B integer multipath validation")
    print(f"Delays: {list(DELAYS_SAMPLES)}")
    print(f"Input powers dB: {list(POWERS_DB)}")
    print(f"Normalized powers: {normalized_path_powers.tolist()}")
    print(f"Phases: {list(PHASES_RAD)}")
    print(f"Impulse response h: {h.tolist()}")
    print(f"Channel length: {channel_length}")
    print(f"Impulse-response energy: {impulse_response_energy:.16e}")
    print(f"Nfft: {nfft}")
    print(f"CP: {cp_len}")
    print(f"Required minimum CP: {required_min_cp}")
    print(f"CP sufficient: {cp_sufficient}")
    print(f"Mean |H|^2: {frequency_response_average_power:.16e}")
    print(f"Ordinary-vs-unitary-scaled H error: {ordinary_vs_scaled_unitary_relative_error:.16e}")
    print(f"Time-domain Y shape: {y_td.shape}")
    print(f"Frequency-domain Y shape: {y_fd.shape}")
    print(f"Time/frequency relative error: {time_frequency_relative_error:.16e}")
    print(f"Maximum absolute error: {time_frequency_max_absolute_error:.16e}")
    print(f"DC-only diagnostic error: {dc_diagnostic_error:.16e}")
    print(f"Short-CP length: {SHORT_CP_LEN}")
    print(f"Short-CP relative error: {short_cp_relative_error:.16e}")
    print(f"Short CP correctly breaks equivalence: {'PASS' if short_cp_breaks_equivalence else 'FAIL'}")
    print(f"Wrong unitary-H relative error: {wrong_unitary_h_relative_error:.16e}")
    print(f"Scale sentinel: {'PASS' if scale_sentinel_passed else 'FAIL'}")
    print(f"All checks: {'PASS' if all_checks_passed else 'FAIL'}")

    output_path = PROJECT_ROOT / "outputs" / "t03b_integer_multipath_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report saved: {output_path}")
    if not all_checks_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
