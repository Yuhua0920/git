"""T03B deterministic integer-multipath and CP-OFDM equivalence checks."""

from pathlib import Path

import numpy as np
import pytest

from src.channels import (
    build_integer_multipath_channel,
    channel_frequency_response,
    convolve_cp_ofdm_blocks,
)
from src.config import load_config
from src.modulation import qpsk_modulate
from src.ofdm import ofdm_modulate_grid


CONFIG = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
NFFT = CONFIG["system"]["nfft"]
NUM_SYMBOLS = CONFIG["system"]["num_symbols"]
CP_LEN = CONFIG["system"]["cp_len"]
DELAYS = [0, 1, 3, 5]
POWERS_DB = [0, -3, -6, -9]
PHASES = [0, np.pi / 5, -np.pi / 3, np.pi / 2]
EQUIVALENCE_TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def calibration_channel() -> tuple[np.ndarray, np.ndarray]:
    return build_integer_multipath_channel(DELAYS, POWERS_DB, PHASES)


@pytest.fixture(scope="module")
def qpsk_grid() -> np.ndarray:
    assert NFFT == CONFIG["system"]["active_carriers"] == 64
    assert NUM_SYMBOLS == 16
    assert CP_LEN == 16
    assert CONFIG["system"]["fft_norm"] == "ortho"
    rng = np.random.default_rng(20260930)
    bits = rng.integers(0, 2, size=NFFT * NUM_SYMBOLS * 2, dtype=np.uint8)
    return qpsk_modulate(bits).reshape(NFFT, NUM_SYMBOLS)


def _time_domain_channel_spectrum(x_grid: np.ndarray, h: np.ndarray, cp_len: int) -> np.ndarray:
    x_with_cp = ofdm_modulate_grid(x_grid, NFFT, cp_len)
    y_useful = convolve_cp_ofdm_blocks(x_with_cp, h, NFFT, cp_len)
    assert y_useful.shape == x_grid.shape
    return np.fft.fftshift(np.fft.fft(y_useful, axis=0, norm="ortho"), axes=0)


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def test_path_power_normalization(calibration_channel: tuple[np.ndarray, np.ndarray]) -> None:
    h, powers = calibration_channel
    expected = 10.0 ** (np.asarray(POWERS_DB, dtype=np.float64) / 10.0)
    expected /= np.sum(expected)
    assert h.shape == (6,)
    assert h.dtype == np.complex128
    np.testing.assert_allclose(powers, expected, rtol=0, atol=1e-15)
    assert abs(np.sum(powers) - 1.0) <= 1e-15


def test_impulse_response_energy(calibration_channel: tuple[np.ndarray, np.ndarray]) -> None:
    h, powers = calibration_channel
    expected_taps = np.zeros(6, dtype=np.complex128)
    expected_taps[DELAYS] = np.sqrt(powers) * np.exp(1j * np.asarray(PHASES))
    np.testing.assert_allclose(h, expected_taps, rtol=0, atol=1e-15)
    assert abs(np.sum(np.abs(h) ** 2) - 1.0) <= 1e-15


def test_frequency_response_average_power(calibration_channel: tuple[np.ndarray, np.ndarray]) -> None:
    h, _ = calibration_channel
    H = channel_frequency_response(h, NFFT)
    assert H.shape == (NFFT,)
    assert abs(np.mean(np.abs(H) ** 2) - 1.0) <= 1e-12


def test_ordinary_fft_matches_scaled_unitary_fft(
    calibration_channel: tuple[np.ndarray, np.ndarray],
) -> None:
    h, _ = calibration_channel
    h_pad = np.zeros(NFFT, dtype=np.complex128)
    h_pad[: len(h)] = h
    H = channel_frequency_response(h, NFFT)
    ordinary = np.fft.fftshift(np.fft.fft(h_pad, norm=None))
    scaled_unitary = np.fft.fftshift(np.sqrt(NFFT) * np.fft.fft(h_pad, norm="ortho"))
    np.testing.assert_allclose(H, ordinary, rtol=0, atol=1e-12)
    np.testing.assert_allclose(H, scaled_unitary, rtol=0, atol=1e-12)


def test_cp_is_sufficient_for_calibration_channel(
    calibration_channel: tuple[np.ndarray, np.ndarray],
) -> None:
    h, _ = calibration_channel
    assert len(h) - 1 == 5
    assert CP_LEN == 16
    assert CP_LEN >= len(h) - 1


def test_time_frequency_equivalence(
    calibration_channel: tuple[np.ndarray, np.ndarray], qpsk_grid: np.ndarray
) -> None:
    h, _ = calibration_channel
    assert qpsk_grid.shape == (64, 16)
    assert qpsk_grid.dtype == np.complex128
    Y_td = _time_domain_channel_spectrum(qpsk_grid, h, CP_LEN)
    Y_fd = channel_frequency_response(h, NFFT)[:, None] * qpsk_grid
    assert Y_td.shape == Y_fd.shape == (NFFT, NUM_SYMBOLS)
    assert _relative_error(Y_td, Y_fd) <= EQUIVALENCE_TOLERANCE


def test_single_dc_subcarrier_through_channel(
    calibration_channel: tuple[np.ndarray, np.ndarray],
) -> None:
    h, _ = calibration_channel
    x_dc = np.zeros((NFFT, 1), dtype=np.complex128)
    x_dc[NFFT // 2, 0] = 1.0
    H = channel_frequency_response(h, NFFT)
    Y_td = _time_domain_channel_spectrum(x_dc, h, CP_LEN)
    Y_fd = H[:, None] * x_dc
    np.testing.assert_allclose(Y_td, Y_fd, rtol=0, atol=1e-12)
    np.testing.assert_allclose(Y_td[NFFT // 2, 0], H[NFFT // 2], rtol=0, atol=1e-12)
    assert np.max(np.abs(np.delete(Y_td[:, 0], NFFT // 2))) <= 1e-12


def test_insufficient_cp_breaks_equivalence(
    calibration_channel: tuple[np.ndarray, np.ndarray], qpsk_grid: np.ndarray
) -> None:
    h, _ = calibration_channel
    short_cp = 4
    assert short_cp < len(h) - 1
    Y_td_short = _time_domain_channel_spectrum(qpsk_grid, h, short_cp)
    Y_fd = channel_frequency_response(h, NFFT)[:, None] * qpsk_grid
    assert _relative_error(Y_td_short, Y_fd) > EQUIVALENCE_TOLERANCE


def test_unscaled_unitary_channel_response_is_wrong(
    calibration_channel: tuple[np.ndarray, np.ndarray], qpsk_grid: np.ndarray
) -> None:
    h, _ = calibration_channel
    h_pad = np.zeros(NFFT, dtype=np.complex128)
    h_pad[: len(h)] = h
    H_wrong = np.fft.fftshift(np.fft.fft(h_pad, norm="ortho"))
    Y_td = _time_domain_channel_spectrum(qpsk_grid, h, CP_LEN)
    Y_fd = channel_frequency_response(h, NFFT)[:, None] * qpsk_grid
    correct_error = _relative_error(Y_td, Y_fd)
    wrong_error = float(np.linalg.norm(Y_td - H_wrong[:, None] * qpsk_grid) / np.linalg.norm(Y_fd))
    assert correct_error <= EQUIVALENCE_TOLERANCE
    assert wrong_error > 1e-3
    assert wrong_error > 1e6 * correct_error


@pytest.mark.parametrize(
    ("delays", "powers_db", "phases"),
    [
        ([0, 1], [0], [0, 0]),
        ([0, 1], [0, -3], [0]),
        ([0, -1], [0, -3], [0, 0]),
        ([0, 1.5], [0, -3], [0, 0]),
        ([], [], []),
        ([0, 0], [0, -3], [0, 0]),
        ([0, 1], [0, np.inf], [0, 0]),
        ([0, 1], [0, np.nan], [0, 0]),
    ],
)
def test_invalid_channel_parameters_rejected(
    delays: list[int | float], powers_db: list[float], phases: list[float]
) -> None:
    with pytest.raises(ValueError):
        build_integer_multipath_channel(delays, powers_db, phases)
