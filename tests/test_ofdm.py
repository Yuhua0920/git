"""T03A checks for centered OFDM grids and exact cyclic-prefix handling."""

from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.modulation import qpsk_modulate
from src.ofdm import ofdm_demodulate_grid, ofdm_modulate_grid, remove_cyclic_prefix


CONFIG = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
NFFT = CONFIG["system"]["nfft"]
NUM_SYMBOLS = CONFIG["system"]["num_symbols"]
CP_LEN = CONFIG["system"]["cp_len"]


@pytest.fixture(scope="module")
def qpsk_grid() -> np.ndarray:
    assert CONFIG["system"]["active_carriers"] == NFFT == 64
    assert NUM_SYMBOLS == 16
    assert CP_LEN == 16
    assert CONFIG["system"]["fft_norm"] == "ortho"
    rng = np.random.default_rng(20260930)
    bits = rng.integers(0, 2, size=NFFT * NUM_SYMBOLS * 2, dtype=np.uint8)
    return qpsk_modulate(bits).reshape(NFFT, NUM_SYMBOLS)


def test_ofdm_roundtrip_qpsk_grid(qpsk_grid: np.ndarray) -> None:
    with_cp = ofdm_modulate_grid(qpsk_grid, NFFT, CP_LEN)
    recovered = ofdm_demodulate_grid(with_cp, NFFT, CP_LEN)
    assert with_cp.shape == (NFFT + CP_LEN, NUM_SYMBOLS)
    assert recovered.shape == qpsk_grid.shape
    assert with_cp.dtype == recovered.dtype == np.complex128
    relative_error = np.linalg.norm(recovered - qpsk_grid) / np.linalg.norm(qpsk_grid)
    assert relative_error <= 1e-10


def test_unitary_parseval(qpsk_grid: np.ndarray) -> None:
    with_cp = ofdm_modulate_grid(qpsk_grid, NFFT, CP_LEN)
    useful_time = remove_cyclic_prefix(with_cp, NFFT, CP_LEN)
    frequency_energy = np.sum(np.abs(qpsk_grid) ** 2, axis=0)
    time_energy = np.sum(np.abs(useful_time) ** 2, axis=0)
    relative_errors = np.abs(time_energy - frequency_energy) / frequency_energy
    assert np.max(relative_errors) <= 1e-12


def test_cp_is_exact_suffix_copy(qpsk_grid: np.ndarray) -> None:
    with_cp = ofdm_modulate_grid(qpsk_grid, NFFT, CP_LEN)
    np.testing.assert_array_equal(with_cp[:CP_LEN, :], with_cp[-CP_LEN:, :])


def test_cp_remove_recovers_time_samples(qpsk_grid: np.ndarray) -> None:
    expected_useful = np.fft.ifft(
        np.fft.ifftshift(qpsk_grid, axes=0), axis=0, norm="ortho"
    )
    with_cp = ofdm_modulate_grid(qpsk_grid, NFFT, CP_LEN)
    recovered_useful = remove_cyclic_prefix(with_cp, NFFT, CP_LEN)
    np.testing.assert_array_equal(recovered_useful, expected_useful)


def test_centered_dc_becomes_constant_waveform() -> None:
    centered = np.zeros((NFFT, 1), dtype=np.complex128)
    centered[NFFT // 2, 0] = 1.0
    with_cp = ofdm_modulate_grid(centered, NFFT, CP_LEN)
    useful = remove_cyclic_prefix(with_cp, NFFT, CP_LEN)[:, 0]
    np.testing.assert_allclose(useful, np.full(NFFT, 1 / np.sqrt(NFFT)), rtol=0, atol=1e-15)


def test_centered_lowest_bin_ordering() -> None:
    centered = np.zeros((NFFT, 1), dtype=np.complex128)
    centered[0, 0] = 1.0  # Centered index 0 is the -Nfft/2 frequency bin.
    with_cp = ofdm_modulate_grid(centered, NFFT, CP_LEN)
    useful = remove_cyclic_prefix(with_cp, NFFT, CP_LEN)[:, 0]
    sample_indices = np.arange(NFFT)
    expected = np.exp(2j * np.pi * (-NFFT // 2) * sample_indices / NFFT) / np.sqrt(NFFT)
    np.testing.assert_allclose(useful, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("cp_len", [-1, NFFT + 1])
def test_invalid_cp_rejected(cp_len: int, qpsk_grid: np.ndarray) -> None:
    with pytest.raises(ValueError, match="cp_len"):
        ofdm_modulate_grid(qpsk_grid, NFFT, cp_len)
    with pytest.raises(ValueError, match="cp_len"):
        remove_cyclic_prefix(np.zeros((NFFT + CP_LEN, NUM_SYMBOLS)), NFFT, cp_len)


@pytest.mark.parametrize("shape", [(NFFT - 1, NUM_SYMBOLS), (NUM_SYMBOLS, NFFT), (NFFT,)])
def test_invalid_grid_shape_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="frequency_grid.*shape"):
        ofdm_modulate_grid(np.zeros(shape, dtype=np.complex128), NFFT, CP_LEN)
    with pytest.raises(ValueError, match="time_grid_with_cp.*shape"):
        ofdm_demodulate_grid(np.zeros(shape, dtype=np.complex128), NFFT, CP_LEN)
