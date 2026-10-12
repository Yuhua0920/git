import numpy as np
import pytest

from src.numerics import complex_gaussian_noise
from src.tdl import (
    TDL_A_FADING_DISTRIBUTIONS, TDL_A_NORMALIZED_DELAYS, TDL_A_POWERS_DB,
    build_static_tdl_a_channel, centered_frequency_grid_hz,
    fractional_delay_frequency_response, rms_delay_spread_seconds, tdl_a_profile,
)


@pytest.fixture(scope="module")
def ensemble():
    # Independent channel seeds; neither symbols nor links are trajectories.
    draws = [build_static_tdl_a_channel(40261001 + i) for i in range(1000)]
    return np.stack([a for a, _ in draws]), np.stack([h[..., 0] for _, h in draws])


def test_tdl_a_has_23_taps():
    assert len(TDL_A_NORMALIZED_DELAYS) == len(TDL_A_POWERS_DB) == 23


def test_tdl_a_table_exact():
    # Independent expected values from the task and official table, not derived
    # from the module: this catches accidental changes or sorting of the table.
    expected_delays = (
        0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750,
        0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942, 2.5119, 3.0582,
        4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586,
    )
    expected_powers = (
        -13.4, 0.0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.5, -7.5, -15.9,
        -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2, -18.3,
        -18.9, -16.6, -19.9, -29.7,
    )
    assert TDL_A_NORMALIZED_DELAYS == expected_delays
    assert TDL_A_POWERS_DB == expected_powers
    assert TDL_A_NORMALIZED_DELAYS[3] > TDL_A_NORMALIZED_DELAYS[4]


def test_all_tdl_a_taps_rayleigh():
    assert TDL_A_FADING_DISTRIBUTIONS == ("Rayleigh",) * 23
    _, powers = tdl_a_profile()
    expected = complex_gaussian_noise((2, 2, 23), 1.0, np.random.default_rng(40261001))
    taps, _ = build_static_tdl_a_channel(40261001)
    np.testing.assert_array_equal(taps, expected * np.sqrt(powers))


def test_100ns_scaled_delays():
    delays, _ = tdl_a_profile()
    np.testing.assert_allclose(delays / 1e-9, np.asarray(TDL_A_NORMALIZED_DELAYS) * 100, rtol=0, atol=2e-13)


def test_max_delay_is_965_86ns():
    assert max(tdl_a_profile()[0]) / 1e-9 == pytest.approx(965.86, abs=1e-10)


def test_rms_delay_spread_about_100ns():
    delays, powers = tdl_a_profile()
    assert abs(rms_delay_spread_seconds(delays, powers) / 1e-9 - 100) < 0.1


def test_pdp_linear_power_sums_to_one():
    _, powers = tdl_a_profile()
    assert powers.sum() == pytest.approx(1, abs=1e-15)
    raw = np.power(10, np.asarray(TDL_A_POWERS_DB) / 10)
    np.testing.assert_allclose(powers / powers[1], raw / raw[1], rtol=1e-15, atol=0)


def test_no_integer_delay_rounding():
    delays, _ = tdl_a_profile()
    assert np.unique(delays).size == 23
    assert np.count_nonzero(np.abs(delays * 1920000 - np.rint(delays * 1920000)) > 1e-6) == 22
    taps = np.zeros(23, dtype=np.complex128)
    taps[1] = 1
    frequencies = centered_frequency_grid_hz(64, 30000)
    response = fractional_delay_frequency_response(taps, delays, frequencies)
    analytic = np.cos(2 * np.pi * frequencies * 38.19e-9) - 1j * np.sin(2 * np.pi * frequencies * 38.19e-9)
    np.testing.assert_allclose(response, analytic, rtol=0, atol=1e-14)
    rounded_response = np.exp(-2j * np.pi * frequencies * np.rint(delays[1] * 1920000) / 1920000)
    assert np.max(np.abs(response - rounded_response)) > 0.1


def test_fractional_single_tap_analytic_response():
    frequencies = centered_frequency_grid_hz(64, 30000)
    response = fractional_delay_frequency_response([1 + 0j], [250e-9], frequencies)
    phase = 2 * np.pi * frequencies * 250e-9
    analytic = np.cos(phase) - 1j * np.sin(phase)
    np.testing.assert_allclose(response, analytic, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.abs(response), 1, rtol=0, atol=1e-15)


def test_dc_response_equals_sum_of_taps():
    taps, channel = build_static_tdl_a_channel(40261001)
    np.testing.assert_allclose(channel[:, :, 32, 0], taps.sum(axis=-1), rtol=0, atol=1e-14)


def test_centered_frequency_order():
    frequencies = centered_frequency_grid_hz(64, 30000)
    assert frequencies[0] == -960000 and frequencies[32] == 0 and frequencies[-1] == 930000
    np.testing.assert_array_equal(np.diff(frequencies), np.full(63, 30000))
    np.testing.assert_allclose(frequencies, np.fft.fftshift(np.fft.fftfreq(64, d=1 / 1920000)), atol=1e-9)


def test_random_taps_reproducible():
    first = build_static_tdl_a_channel(40261001)
    second = build_static_tdl_a_channel(40261001)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_different_seed_changes_channel():
    first = build_static_tdl_a_channel(40261001)
    second = build_static_tdl_a_channel(40261002)
    assert all(not np.array_equal(a, b) for a, b in zip(first, second))


def test_four_mimo_links_independent(ensemble):
    taps, _ = ensemble
    assert len({a.tobytes() for a in taps[0].reshape(4, 23)}) == 4
    _, powers = tdl_a_profile()
    standardized = (taps / np.sqrt(powers)).reshape(1000, 4, 23)
    for l in range(23):
        covariance = standardized[:, :, l].T @ standardized[:, :, l].conj() / 1000
        off_diagonal = covariance - np.diag(np.diag(covariance))
        assert np.max(np.abs(off_diagonal)) < 0.15


def test_no_per_realization_normalization(ensemble):
    taps, _ = ensemble
    energies = np.sum(np.abs(taps) ** 2, axis=-1)
    assert not np.allclose(energies, 1)
    assert np.min(energies) < 0.5 and np.max(energies) > 1.5
    _, powers = tdl_a_profile()
    expected = complex_gaussian_noise((2, 2, 23), 1, np.random.default_rng(40261001)) * np.sqrt(powers)
    np.testing.assert_array_equal(taps[0], expected)


def test_ensemble_tap_energy_about_one(ensemble):
    taps, _ = ensemble
    np.testing.assert_allclose(np.mean(np.sum(np.abs(taps) ** 2, axis=-1), axis=0), 1, rtol=0, atol=0.10)


def test_empirical_per_tap_power_matches_pdp(ensemble):
    taps, _ = ensemble
    _, powers = tdl_a_profile()
    target = np.broadcast_to(powers, (2, 2, 23))
    np.testing.assert_allclose(np.mean(np.abs(taps) ** 2, axis=0), target, rtol=0.15, atol=0)
    pooled = taps.reshape(4000, 23)
    np.testing.assert_allclose(np.var(pooled.real, axis=0), powers / 2, rtol=0.10, atol=0)
    np.testing.assert_allclose(np.var(pooled.imag, axis=0), powers / 2, rtol=0.10, atol=0)


def test_cp_duration_exceeds_max_delay():
    assert 16 / 1920000 > np.max(tdl_a_profile()[0])


def test_output_shape_2x2x64x16():
    taps, channel = build_static_tdl_a_channel(40261001)
    assert taps.shape == (2, 2, 23) and channel.shape == (2, 2, 64, 16)
    assert taps.dtype == channel.dtype == np.complex128
    assert np.all(np.isfinite(channel))


def test_time_static_in_t08a():
    _, channel = build_static_tdl_a_channel(40261001)
    np.testing.assert_array_equal(channel, np.repeat(channel[..., :1], 16, axis=-1))


def test_ensemble_frequency_power_about_one(ensemble):
    _, channel = ensemble
    assert abs(np.mean(np.abs(channel) ** 2) - 1) < 0.10
    np.testing.assert_allclose(np.mean(np.abs(channel) ** 2, axis=(0, 3)), 1, rtol=0, atol=0.10)


def test_fractional_response_agrees_with_existing_integer_fft():
    # Integer-delay special case preserves the previously verified ordinary FFT.
    taps = np.array([0.7 + 0.2j, -0.1 + 0.3j, 0.05 - 0.4j])
    delays = np.array([0, 1, 3]) / 1920000
    impulse = np.zeros(4, dtype=np.complex128)
    impulse[[0, 1, 3]] = taps
    expected = np.fft.fftshift(np.fft.fft(impulse, n=64))
    response = fractional_delay_frequency_response(taps, delays, centered_frequency_grid_hz(64, 30000))
    np.testing.assert_allclose(response, expected, rtol=0, atol=2e-15)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_channel_seed_rejected(seed):
    with pytest.raises(ValueError):
        build_static_tdl_a_channel(seed)


@pytest.mark.parametrize("spread", [0, -100, np.nan])
def test_invalid_delay_spread_rejected(spread):
    with pytest.raises(ValueError):
        tdl_a_profile(spread)


def test_fractional_response_rejects_mismatched_or_invalid_units_vectors():
    with pytest.raises(ValueError):
        fractional_delay_frequency_response([1, 2], [1e-9], [0])
    with pytest.raises(ValueError):
        fractional_delay_frequency_response([1], [-1e-9], [0])
    with pytest.raises(ValueError):
        fractional_delay_frequency_response([1], [1e-9], [np.nan])
