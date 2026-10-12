import numpy as np
import pytest

from src.numerics import (
    complex_gaussian_noise,
    noise_variance_from_snr,
    scale_data_streams,
)


@pytest.fixture(scope="module")
def million_noise_samples() -> np.ndarray:
    return complex_gaussian_noise(
        1_000_000, variance=1.0, rng=np.random.default_rng(20260930)
    )


def test_complex_noise_zero_variance() -> None:
    samples = complex_gaussian_noise(
        (4, 3), variance=0.0, rng=np.random.default_rng(20260930)
    )
    assert samples.shape == (4, 3)
    assert samples.dtype == np.complex128
    assert np.all(samples == 0)


def test_complex_noise_negative_variance_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        complex_gaussian_noise(8, variance=-0.1, rng=np.random.default_rng(20260930))


def test_complex_noise_empirical_power(million_noise_samples: np.ndarray) -> None:
    empirical_power = float(np.mean(np.abs(million_noise_samples) ** 2))
    assert abs(empirical_power - 1.0) / 1.0 <= 0.01


def test_complex_noise_real_imag_variance(million_noise_samples: np.ndarray) -> None:
    for component in (million_noise_samples.real, million_noise_samples.imag):
        assert abs(float(np.var(component)) - 0.5) / 0.5 <= 0.01
        assert abs(float(np.mean(component))) <= 5 * np.sqrt(0.5 / 1_000_000)


@pytest.mark.parametrize("snr_db,expected", [(0, 1.0), (10, 0.1), (20, 0.01)])
def test_noise_variance_from_snr(snr_db: float, expected: float) -> None:
    assert noise_variance_from_snr(snr_db, total_power=1.0) == pytest.approx(expected)


def test_two_stream_total_power() -> None:
    unit_symbols = np.array([1, -1, 1j, -1j], dtype=np.complex128)
    symbols = np.column_stack((unit_symbols, unit_symbols[::-1]))
    tx = scale_data_streams(symbols, nt=2, total_power=1.0)
    stream_power = np.mean(np.abs(tx) ** 2, axis=0)
    assert stream_power[0] == pytest.approx(0.5)
    assert stream_power[1] == pytest.approx(0.5)
    assert float(np.sum(stream_power)) == pytest.approx(1.0)


def test_pilot_re_total_power() -> None:
    for pilot_re in (
        np.array([1 + 0j, 0 + 0j], dtype=np.complex128),
        np.array([0 + 0j, 1 + 0j], dtype=np.complex128),
    ):
        assert np.sum(np.abs(pilot_re) ** 2) == 1.0
