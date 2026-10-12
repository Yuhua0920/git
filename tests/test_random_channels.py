import numpy as np
import pytest

from src.numerics import complex_gaussian_noise
from src.random_channels import DELAY_SAMPLES, NORMALIZED_PDP, build_random_mimo_channel, random_mimo_taps


def test_random_taps_reproducible_and_seed_specific():
    impulse, channel = build_random_mimo_channel(40261001)
    repeat_impulse, repeat_channel = build_random_mimo_channel(40261001)
    np.testing.assert_array_equal(impulse, repeat_impulse)
    np.testing.assert_array_equal(channel, repeat_channel)
    assert not np.array_equal(impulse, build_random_mimo_channel(40261002)[0])


def test_independent_link_draws_and_no_realization_normalization():
    impulse, _ = build_random_mimo_channel(40261001)
    rng = np.random.default_rng(40261001)
    expected = complex_gaussian_noise((2, 2, 4), 1, rng) * np.sqrt(NORMALIZED_PDP)
    np.testing.assert_array_equal(impulse[:, :, DELAY_SAMPLES], expected)
    assert len({v.tobytes() for v in expected.reshape(4, 4)}) == 4
    assert not np.allclose(np.sum(np.abs(impulse) ** 2, axis=-1), 1)


def test_long_term_tap_variance_matches_pdp():
    rng = np.random.default_rng(40261001)
    taps = np.stack([random_mimo_taps(rng) for _ in range(8000)])
    np.testing.assert_allclose(np.mean(np.abs(taps) ** 2, axis=0), np.broadcast_to(NORMALIZED_PDP, (2, 2, 4)), rtol=0.05, atol=0)
    np.testing.assert_allclose(np.var(taps.real, axis=0), np.broadcast_to(NORMALIZED_PDP / 2, (2, 2, 4)), rtol=0.05, atol=0)
    np.testing.assert_allclose(np.var(taps.imag, axis=0), np.broadcast_to(NORMALIZED_PDP / 2, (2, 2, 4)), rtol=0.05, atol=0)
    assert np.all(np.abs(np.mean(taps, axis=0)) < 0.03)
    assert np.sum(NORMALIZED_PDP) == pytest.approx(1)


def test_random_channel_frequency_order_power_and_time_static():
    impulse, channel = build_random_mimo_channel(40261001)
    assert impulse.shape == (2, 2, 6) and channel.shape == (2, 2, 64, 16)
    assert channel.dtype == np.complex128
    for r in range(2):
        for q in range(2):
            expected = np.fft.fftshift(np.fft.fft(impulse[r, q], n=64))
            np.testing.assert_array_equal(channel[r, q], np.repeat(expected[:, None], 16, axis=1))
    np.testing.assert_allclose(np.mean(np.abs(channel[..., 0]) ** 2, axis=-1), np.sum(np.abs(impulse) ** 2, axis=-1), rtol=0, atol=1e-15)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_channel_seeds_rejected(seed):
    with pytest.raises(ValueError):
        build_random_mimo_channel(seed)
