import numpy as np
import pytest
from scipy.special import j0

from src.doppler import (
    ICI_MODELED, NEGATIVE_EIGENVALUE_TOLERANCE, SYMBOL_WISE_QUASI_STATIC,
    build_tdl_a_doppler_channel, correlated_complex_gaussian,
    covariance_eigenfactor, jakes_complex_gaussian, jakes_covariance,
    mobility_parameters, ofdm_time_grid_seconds,
)
from src.numerics import complex_gaussian_noise
from src.tdl import (
    build_static_tdl_a_channel, centered_frequency_grid_hz,
    fractional_delay_frequency_response, tdl_a_profile,
)


LAGS = (1, 4, 8, 12, 15)


def empirical_correlation(series, lag):
    source = series[..., :-lag]
    target = series[..., lag:]
    return np.mean(target * source.conj()) / np.mean(np.abs(source) ** 2)


@pytest.fixture(scope="module")
def time_model():
    times = ofdm_time_grid_seconds()
    fd = mobility_parameters()["maximum_doppler_hz"]
    covariance = jakes_covariance(times, fd)
    factor, summary = covariance_eigenfactor(covariance)
    return times, fd, covariance, factor, summary


@pytest.fixture(scope="module")
def unit_processes(time_model):
    times, fd, _, _, _ = time_model
    return jakes_complex_gaussian(np.random.default_rng(50261001), times, fd, (5000,))


@pytest.fixture(scope="module")
def channel_ensemble():
    draws = [build_tdl_a_doppler_channel(40261001 + i) for i in range(1000)]
    return np.stack([taps for taps, _ in draws]), np.stack([h[0, 0, 32, :] for _, h in draws])


def test_30_kmh_speed_in_mps():
    assert mobility_parameters()["speed_mps"] == pytest.approx(30 / 3.6, rel=1e-15)


def test_carrier_wavelength_correct():
    assert mobility_parameters()["wavelength_m"] == pytest.approx(299792458 / 3.5e9, rel=1e-15)


def test_maximum_doppler_correct():
    expected = (30 / 3.6) * 3.5e9 / 299792458
    assert mobility_parameters()["maximum_doppler_hz"] == pytest.approx(expected, rel=1e-15)
    assert mobility_parameters(0)["maximum_doppler_hz"] == 0


def test_symbol_duration_includes_cp():
    grid = ofdm_time_grid_seconds()
    assert grid[1] == pytest.approx(80 / 1920000, rel=1e-15)
    assert grid[1] != pytest.approx(64 / 1920000, rel=1e-3)


def test_time_grid_and_sampled_span():
    grid = ofdm_time_grid_seconds()
    np.testing.assert_allclose(grid, np.arange(16) * 80 / 1920000, rtol=0, atol=1e-18)
    assert grid[-1] == pytest.approx(15 * 80 / 1920000, rel=1e-15)
    assert 16 * grid[1] > grid[-1]


def test_jakes_covariance_unit_diagonal(time_model):
    _, _, covariance, _, _ = time_model
    assert covariance.shape == (16, 16)
    np.testing.assert_array_equal(np.diag(covariance), np.ones(16))


def test_jakes_covariance_symmetric_and_hermitian(time_model):
    _, _, covariance, _, _ = time_model
    np.testing.assert_array_equal(covariance, covariance.conj().T)


def test_jakes_covariance_psd_and_factor_reconstruction(time_model):
    _, _, covariance, factor, summary = time_model
    assert summary["minimum_covariance_eigenvalue"] >= -NEGATIVE_EIGENVALUE_TOLERANCE
    assert summary["jitter_added"] == 0
    np.testing.assert_allclose(factor @ factor.conj().T, covariance, rtol=0, atol=5e-14)


def test_substantive_negative_eigenvalue_fails():
    with pytest.raises(ValueError, match="substantive negative eigenvalue"):
        covariance_eigenfactor([[1, 2], [2, 1]])


def test_roundoff_negative_eigenvalue_clipped_and_recorded():
    factor, summary = covariance_eigenfactor(np.diag([1.0, -1e-14]))
    assert summary["negative_eigenvalues_clipped"] == 1
    assert summary["minimum_covariance_eigenvalue"] == -1e-14
    assert summary["jitter_added"] == 0
    np.testing.assert_allclose(factor @ factor.conj().T, np.diag([1.0, 0.0]), atol=0)


def test_correlated_gaussian_reproducible_and_different_seeds(time_model):
    times, fd, _, _, _ = time_model
    first = jakes_complex_gaussian(np.random.default_rng(100), times, fd, (4,))
    repeat = jakes_complex_gaussian(np.random.default_rng(100), times, fd, (4,))
    other = jakes_complex_gaussian(np.random.default_rng(101), times, fd, (4,))
    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, other)


def test_complex_covariance_orientation_and_proper_gaussian():
    # A genuinely complex covariance catches an erroneous conjugate transpose.
    covariance = np.array([[1, 0.4 + 0.2j], [0.4 - 0.2j, 1]])
    factor, _ = covariance_eigenfactor(covariance)
    series = correlated_complex_gaussian(np.random.default_rng(70261001), factor, (5000,))
    empirical = series.T @ series.conj() / 5000
    np.testing.assert_allclose(empirical, covariance, rtol=0, atol=0.04)
    assert np.max(np.abs(series.T @ series / 5000)) < 0.04


def test_empirical_jakes_correlation_matches_theory(time_model, unit_processes):
    times, fd, _, _, _ = time_model
    for lag in LAGS:
        theory = j0(2 * np.pi * fd * times[1] * lag)
        empirical = empirical_correlation(unit_processes, lag)
        assert abs(empirical - theory) <= 0.02
        assert abs(empirical.imag) <= 0.02


def test_unit_process_power_and_gaussian_marginals(unit_processes):
    np.testing.assert_allclose(np.mean(np.abs(unit_processes) ** 2, axis=0), 1, rtol=0, atol=0.05)
    np.testing.assert_allclose(np.var(unit_processes.real, axis=0), 0.5, rtol=0, atol=0.04)
    np.testing.assert_allclose(np.var(unit_processes.imag, axis=0), 0.5, rtol=0, atol=0.04)


def test_zero_speed_matches_t08a_static_channel_exactly():
    taps, channel = build_tdl_a_doppler_channel(40261001, speed_kmh=0)
    static_taps, static_channel = build_static_tdl_a_channel(40261001)
    np.testing.assert_array_equal(channel, static_channel)
    np.testing.assert_array_equal(taps, np.repeat(static_taps[..., None], 16, axis=-1))


def test_zero_doppler_covariance_and_unit_process_are_exactly_static():
    times = ofdm_time_grid_seconds()
    np.testing.assert_array_equal(jakes_covariance(times, 0), np.ones((16, 16)))
    series = jakes_complex_gaussian(np.random.default_rng(100), times, 0, (4,))
    np.testing.assert_array_equal(series, np.repeat(series[:, :1], 16, axis=-1))


def test_30_kmh_channel_reproducible_and_seed_specific():
    first = build_tdl_a_doppler_channel(40261001)
    repeat = build_tdl_a_doppler_channel(40261001)
    other = build_tdl_a_doppler_channel(40261002)
    for a, b, c in zip(first, repeat, other):
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)


def test_30_kmh_variation_nonzero_and_matches_increment_power(time_model, channel_ensemble):
    _, _, covariance, _, _ = time_model
    _, dc_series = channel_ensemble
    assert not np.array_equal(dc_series, np.repeat(dc_series[:, :1], 16, axis=-1))
    normalized_increment = np.mean(np.abs(np.diff(dc_series, axis=-1)) ** 2) / np.mean(np.abs(dc_series[:, :-1]) ** 2)
    theory_increment = 2 * (1 - covariance[1, 0])
    assert normalized_increment > 1e-8
    assert normalized_increment == pytest.approx(theory_increment, rel=0.15)


def test_30_kmh_is_not_iid_per_symbol(time_model, channel_ensemble):
    _, _, covariance, _, _ = time_model
    _, dc_series = channel_ensemble
    empirical = empirical_correlation(dc_series, 1)
    assert empirical.real > 0.95
    assert abs(empirical - covariance[1, 0]) < 0.02
    assert abs(empirical - 1) > 1e-10


def test_iid_and_static_negative_controls_are_detected(time_model):
    _, _, covariance, _, _ = time_model
    independent = complex_gaussian_noise((5000, 16), 1, np.random.default_rng(60261001))
    assert abs(empirical_correlation(independent, 1) - covariance[1, 0]) > 0.9
    static = np.repeat(independent[:, :1], 16, axis=-1)
    assert np.mean(np.abs(np.diff(static, axis=-1)) ** 2) == 0


def test_tap_powers_preserve_t08a_pdp(channel_ensemble):
    taps, _ = channel_ensemble
    _, powers = tdl_a_profile()
    empirical = np.mean(np.abs(taps) ** 2, axis=(0, 4))
    np.testing.assert_allclose(empirical, np.broadcast_to(powers, (2, 2, 23)), rtol=0.15, atol=0)
    means = np.mean(np.sum(np.abs(taps) ** 2, axis=3), axis=(0, 3))
    np.testing.assert_allclose(means, 1, rtol=0, atol=0.10)


def test_no_per_trajectory_or_per_time_normalization(time_model, channel_ensemble):
    _, _, _, factor, _ = time_model
    taps, _ = channel_ensemble
    _, powers = tdl_a_profile()
    z = complex_gaussian_noise((2, 2, 23, 16), 1, np.random.default_rng(40261001))
    expected = (z @ factor.T) * np.sqrt(powers)[None, None, :, None]
    np.testing.assert_array_equal(taps[0], expected)
    energies = np.sum(np.abs(taps) ** 2, axis=3)
    assert energies.min() < 0.5 and energies.max() > 1.5
    assert not np.allclose(energies, 1)


def test_links_and_taps_are_independent(channel_ensemble):
    taps, _ = channel_ensemble
    _, powers = tdl_a_profile()
    standardized = (taps[..., 0] / np.sqrt(powers)).reshape(1000, 92)
    covariance = standardized.T @ standardized.conj() / 1000
    off_diagonal = covariance - np.diag(np.diag(covariance))
    assert np.max(np.abs(off_diagonal)) < 0.15


def test_fractional_response_preserved_and_dc_correct_at_every_time():
    taps, channel = build_tdl_a_doppler_channel(40261001)
    delays, _ = tdl_a_profile()
    frequencies = centered_frequency_grid_hz(64, 30000)
    assert np.unique(delays).size == 23
    assert np.count_nonzero(np.abs(delays * 1920000 - np.rint(delays * 1920000)) > 1e-6) == 22
    for t in range(16):
        reference = fractional_delay_frequency_response(taps[..., t], delays, frequencies)
        np.testing.assert_allclose(channel[..., t], reference, rtol=0, atol=2e-15)
    np.testing.assert_allclose(channel[:, :, 32, :], taps.sum(axis=2), rtol=0, atol=1e-14)


def test_carrier_only_affects_doppler_not_baseband_delay_phase():
    first = build_tdl_a_doppler_channel(40261001, speed_kmh=30, carrier_frequency_hz=3.5e9)
    same_fd = build_tdl_a_doppler_channel(40261001, speed_kmh=15, carrier_frequency_hz=7e9)
    for a, b in zip(first, same_fd):
        np.testing.assert_array_equal(a, b)


def test_output_shapes_finite_and_no_within_symbol_ici():
    taps, channel = build_tdl_a_doppler_channel(40261001)
    assert taps.shape == (2, 2, 23, 16) and channel.shape == (2, 2, 64, 16)
    assert taps.dtype == channel.dtype == np.complex128
    assert np.all(np.isfinite(taps)) and np.all(np.isfinite(channel))
    assert SYMBOL_WISE_QUASI_STATIC is True and ICI_MODELED is False


def test_pilot_lag4_and_frame_lag15_are_distinct(time_model):
    times, fd, covariance, _, _ = time_model
    assert covariance[4, 0] == pytest.approx(j0(2 * np.pi * fd * times[1] * 4), abs=1e-15)
    assert covariance[15, 0] == pytest.approx(j0(2 * np.pi * fd * times[-1]), abs=1e-15)
    assert covariance[15, 0] != pytest.approx(j0(2 * np.pi * fd * 16 * times[1]), abs=1e-5)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_channel_seed_rejected(seed):
    with pytest.raises(ValueError):
        build_tdl_a_doppler_channel(seed)


def test_invalid_inputs_and_inconsistent_sample_rate_rejected():
    with pytest.raises(ValueError):
        mobility_parameters(-30)
    with pytest.raises(ValueError):
        mobility_parameters(carrier_frequency_hz=0)
    with pytest.raises(ValueError):
        jakes_covariance([0, 0], 97)
    with pytest.raises(ValueError):
        covariance_eigenfactor([[1, 1], [0, 1]])
    with pytest.raises(ValueError):
        build_tdl_a_doppler_channel(1, sample_rate_hz=2000000)
