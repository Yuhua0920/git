"""T08A static TDL-A PDP and continuous-delay baseband frequency responses.

Source: 3GPP TR 38.901 V17.0.0, Table 7.7.2-1 and Clause 7.7.3;
ETSI TR 138 901 V17.0.0 (2022-04), printed pages 74 and 76.
No Doppler, antenna correlation, delay quantization, or realization scaling.
"""

import numpy as np

from src.numerics import complex_gaussian_noise


CHANNEL_MODEL = "TDL-A"
SOURCE = "3GPP TR 38.901 Table 7.7.2-1"
STANDARD_REFERENCE = "3GPP TR 38.901 V17.0.0 / ETSI TR 138 901 V17.0.0 (2022-04)"
STANDARD_URL = (
    "https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/"
    "17.00.00_60/tr_138901v170000p.pdf"
)
DESIRED_RMS_DELAY_SPREAD_NS = 100.0

# Original table order is deliberate: several delays are not ascending.
TDL_A_NORMALIZED_DELAYS = (
    0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750,
    0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942, 2.5119, 3.0582,
    4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586,
)
TDL_A_POWERS_DB = (
    -13.4, 0.0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.5, -7.5, -15.9,
    -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2, -18.3,
    -18.9, -16.6, -19.9, -29.7,
)
TDL_A_FADING_DISTRIBUTIONS = ("Rayleigh",) * 23


def _positive_integer(name: str, value: int) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")


def tdl_a_profile(
    desired_rms_delay_spread_ns: float = DESIRED_RMS_DELAY_SPREAD_NS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return delays in seconds and normalized ensemble PDP, in table order.

    Clause 7.7.3: delay_ns = normalized_delay * desired_delay_spread_ns.
    Only deterministic PDP weights are normalized; random taps are never scaled.
    """
    if not np.isfinite(desired_rms_delay_spread_ns) or desired_rms_delay_spread_ns <= 0:
        raise ValueError("desired_rms_delay_spread_ns must be finite and positive")
    delays_seconds = (
        np.asarray(TDL_A_NORMALIZED_DELAYS, dtype=np.float64)
        * desired_rms_delay_spread_ns * 1e-9
    )
    linear_powers = 10.0 ** (np.asarray(TDL_A_POWERS_DB, dtype=np.float64) / 10.0)
    return delays_seconds, linear_powers / np.sum(linear_powers)


def rms_delay_spread_seconds(delays_seconds: np.ndarray, powers: np.ndarray) -> float:
    """Compute the power-weighted RMS spread; weights must already sum to one."""
    delays = np.asarray(delays_seconds, dtype=np.float64)
    weights = np.asarray(powers, dtype=np.float64)
    if (
        delays.ndim != 1 or delays.size == 0 or weights.shape != delays.shape
        or not np.all(np.isfinite(delays)) or np.any(delays < 0)
        or not np.all(np.isfinite(weights)) or np.any(weights < 0)
        or not np.isclose(np.sum(weights), 1.0, rtol=0, atol=1e-12)
    ):
        raise ValueError("finite nonnegative delays and matching unit-sum powers required")
    mean_delay = np.sum(weights * delays)
    return float(np.sqrt(np.sum(weights * (delays - mean_delay) ** 2)))


def centered_frequency_grid_hz(nfft: int, delta_f_hz: float) -> np.ndarray:
    """Return fftshift physical baseband offsets, in ascending order, in Hz."""
    _positive_integer("nfft", nfft)
    if not np.isfinite(delta_f_hz) or delta_f_hz <= 0:
        raise ValueError("delta_f_hz must be finite and positive")
    return (np.arange(nfft, dtype=np.float64) - nfft // 2) * delta_f_hz


def fractional_delay_frequency_response(
    tap_coefficients: np.ndarray, delays_seconds: np.ndarray, frequencies_hz: np.ndarray,
) -> np.ndarray:
    """Return H[...,K] = sum_l a[...,l] exp(-j 2 pi f[K] tau[l]).

    Coefficients have shape [...,L]. Delays are continuous seconds, not sampled
    impulse-response indices. Frequencies are baseband offsets; no RF carrier is
    added. The response has no FFT normalization factor.
    """
    taps = np.asarray(tap_coefficients, dtype=np.complex128)
    delays = np.asarray(delays_seconds, dtype=np.float64)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if (
        taps.ndim < 1 or delays.ndim != 1 or delays.size == 0
        or taps.shape[-1] != delays.size
        or frequencies.ndim != 1 or frequencies.size == 0
        or not np.all(np.isfinite(taps)) or not np.all(np.isfinite(delays))
        or np.any(delays < 0) or not np.all(np.isfinite(frequencies))
    ):
        raise ValueError("finite [...,L] taps, L nonnegative delays, and a frequency vector required")
    phase = np.exp(-2j * np.pi * delays[:, None] * frequencies[None, :])
    return taps @ phase


def random_tdl_a_taps(rng: np.random.Generator) -> np.ndarray:
    """Draw independent [RX,TX,tap]=[2,2,23] coefficients a_l ~ CN(0,p_l)."""
    _, powers = tdl_a_profile()
    gaussians = complex_gaussian_noise((2, 2, 23), 1.0, rng, dtype=np.complex128)
    return gaussians * np.sqrt(powers)[None, None, :]


def build_static_tdl_a_channel(
    channel_seed: int, nfft: int = 64, num_symbols: int = 16,
    delta_f_hz: float = 30000.0,
    desired_rms_delay_spread_ns: float = DESIRED_RMS_DELAY_SPREAD_NS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a[2,2,23] and H[2,2,K,T], with identical time slices in T08A."""
    if (
        isinstance(channel_seed, (bool, np.bool_))
        or not isinstance(channel_seed, (int, np.integer)) or channel_seed < 0
    ):
        raise ValueError("channel_seed must be a nonnegative integer")
    _positive_integer("num_symbols", num_symbols)
    frequencies = centered_frequency_grid_hz(nfft, delta_f_hz)
    delays, _ = tdl_a_profile(desired_rms_delay_spread_ns)
    taps = random_tdl_a_taps(np.random.default_rng(channel_seed))
    frequency = fractional_delay_frequency_response(taps, delays, frequencies)
    return taps, np.repeat(frequency[..., None], num_symbols, axis=-1)
