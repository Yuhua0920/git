import numpy as np
import pytest

from src.modulation import (
    ebn0_to_esn0_db,
    ebn0_to_noise_variance,
    qpsk_awgn_theoretical_ber,
    qpsk_demodulate,
    qpsk_modulate,
)
from src.numerics import noise_variance_from_snr


ALL_DIBITS = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)


def test_qpsk_all_constellation_points() -> None:
    expected = np.array([1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j]) / np.sqrt(2.0)
    actual = qpsk_modulate(ALL_DIBITS)
    assert actual.dtype == np.complex128
    np.testing.assert_array_equal(actual, expected)


def test_qpsk_symbol_energy() -> None:
    symbols = qpsk_modulate(ALL_DIBITS)
    np.testing.assert_allclose(np.abs(symbols) ** 2, np.ones(4), rtol=0, atol=1e-15)


def test_qpsk_roundtrip_all_points() -> None:
    np.testing.assert_array_equal(qpsk_demodulate(qpsk_modulate(ALL_DIBITS)), ALL_DIBITS)


def test_qpsk_random_noiseless_roundtrip() -> None:
    rng = np.random.default_rng(20260930)
    bits = rng.integers(0, 2, size=100_000, dtype=np.uint8)
    recovered = qpsk_demodulate(qpsk_modulate(bits))
    assert int(np.count_nonzero(bits != recovered)) == 0


def test_invalid_bit_values_rejected() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        qpsk_modulate([0, 1, 2, 0])


def test_odd_number_of_bits_rejected() -> None:
    with pytest.raises(ValueError, match="even"):
        qpsk_modulate([0, 1, 0])


def test_qpsk_near_axis_hard_decisions() -> None:
    samples = np.array([0.001 + 1j, -0.001 + 1j, 1 - 0.001j, -1 - 0.001j])
    expected = np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype=np.uint8)
    np.testing.assert_array_equal(qpsk_demodulate(samples), expected)
    np.testing.assert_array_equal(qpsk_demodulate([0 + 1j, 1 + 0j]), [0, 0, 0, 0])


@pytest.mark.parametrize("ebn0_db,expected", [(0, 0.5), (10, 0.05)])
def test_ebn0_noise_variance(ebn0_db: float, expected: float) -> None:
    assert ebn0_to_noise_variance(ebn0_db) == pytest.approx(expected)


@pytest.mark.parametrize("ebn0_db", [0, 2, 4, 6, 8, 10])
def test_ebn0_and_esn0_conversion_consistent(ebn0_db: float) -> None:
    esn0_db = ebn0_to_esn0_db(ebn0_db)
    assert esn0_db == pytest.approx(ebn0_db + 10 * np.log10(2))
    assert noise_variance_from_snr(esn0_db, total_power=1.0) == pytest.approx(
        ebn0_to_noise_variance(ebn0_db)
    )


@pytest.mark.parametrize(
    "ebn0_db,expected",
    [(0, 0.07864960352514257), (10, 3.872108215522035e-6)],
)
def test_theoretical_ber_known_values(ebn0_db: float, expected: float) -> None:
    assert qpsk_awgn_theoretical_ber(ebn0_db) == pytest.approx(expected, rel=1e-12)
