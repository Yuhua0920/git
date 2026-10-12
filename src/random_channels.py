"""Independent Gaussian tapped MIMO fading for T07B, with ensemble normalization."""

import numpy as np

from src.channels import channel_frequency_response
from src.numerics import complex_gaussian_noise


DELAY_SAMPLES = (0, 1, 3, 5)
POWERS_DB = (0, -3, -6, -9)
_linear_powers = 10.0 ** (np.asarray(POWERS_DB, dtype=float) / 10)
NORMALIZED_PDP = _linear_powers / np.sum(_linear_powers)


def random_mimo_taps(rng: np.random.Generator) -> np.ndarray:
    """Return [RX,TX,path] taps with independent CN(0,p_l) entries."""
    unit_gaussians = complex_gaussian_noise((2, 2, 4), 1.0, rng, dtype=np.complex128)
    return unit_gaussians * np.sqrt(NORMALIZED_PDP)[None, None, :]


def build_random_mimo_channel(
    channel_seed: int, nfft: int = 64, num_symbols: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Return h[RX,TX,delay] and centered H[RX,TX,K,T], static within a trajectory.

    Only the PDP is normalized. Instantaneous impulse energy is retained.
    """
    for name, value in (("channel_seed", channel_seed), ("nfft", nfft), ("num_symbols", num_symbols)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if channel_seed < 0 or nfft <= max(DELAY_SAMPLES) or num_symbols <= 0:
        raise ValueError("seed must be nonnegative, nfft > max delay, and num_symbols positive")
    taps = random_mimo_taps(np.random.default_rng(channel_seed))
    impulse = np.zeros((2, 2, max(DELAY_SAMPLES) + 1), dtype=np.complex128)
    impulse[:, :, DELAY_SAMPLES] = taps
    frequency = np.empty((2, 2, nfft), dtype=np.complex128)
    for r in range(2):
        for q in range(2):
            frequency[r, q] = channel_frequency_response(impulse[r, q], nfft)
    return impulse, np.repeat(frequency[..., None], num_symbols, axis=-1)
