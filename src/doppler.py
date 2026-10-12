"""T08B Jakes-correlated Gaussian taps on a symbol-wise quasi-static grid.

3GPP TR 38.901 V17.0.0 Clause 7.7.2 specifies classical/Jakes tap Doppler.
This implementation samples its J0 covariance at OFDM symbol starts. No sample
waveform, within-symbol fading, ICI, or spatial correlation is implemented.
The T08A TDL-A profile and fractional-delay response are reused unchanged.
"""

import numpy as np
from scipy.special import j0

from src.numerics import complex_gaussian_noise
from src.tdl import (
    build_static_tdl_a_channel, centered_frequency_grid_hz,
    fractional_delay_frequency_response, tdl_a_profile,
)


SPEED_OF_LIGHT_MPS = 299792458.0
CARRIER_FREQUENCY_HZ = 3.5e9
SPEED_KMH = 30.0
NEGATIVE_EIGENVALUE_TOLERANCE = 1e-12
SYMBOL_WISE_QUASI_STATIC = True
ICI_MODELED = False


def mobility_parameters(
    speed_kmh: float = SPEED_KMH, carrier_frequency_hz: float = CARRIER_FREQUENCY_HZ,
) -> dict[str, float]:
    """Compute v, wavelength and maximum Doppler from physical constants."""
    if not np.isfinite(speed_kmh) or speed_kmh < 0:
        raise ValueError("speed_kmh must be finite and nonnegative")
    if not np.isfinite(carrier_frequency_hz) or carrier_frequency_hz <= 0:
        raise ValueError("carrier_frequency_hz must be finite and positive")
    speed_mps = float(speed_kmh / 3.6)
    wavelength = SPEED_OF_LIGHT_MPS / carrier_frequency_hz
    return {
        "speed_kmh": float(speed_kmh), "speed_mps": speed_mps,
        "carrier_frequency_hz": float(carrier_frequency_hz),
        "speed_of_light_mps": SPEED_OF_LIGHT_MPS, "wavelength_m": float(wavelength),
        "maximum_doppler_hz": float(abs(speed_mps) / wavelength),
    }


def ofdm_time_grid_seconds(
    nfft: int = 64, cp_len: int = 16, sample_rate_hz: float = 1920000.0,
    num_symbols: int = 16,
) -> np.ndarray:
    """Return symbol start times n*(nfft+CP)/Fs, including CP in the spacing."""
    for name, value in (("nfft", nfft), ("num_symbols", num_symbols)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(cp_len, (bool, np.bool_)) or not isinstance(cp_len, (int, np.integer))
        or cp_len < 0 or cp_len > nfft
    ):
        raise ValueError("cp_len must be an integer from zero through nfft")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    return np.arange(num_symbols, dtype=np.float64) * ((nfft + cp_len) / sample_rate_hz)


def jakes_covariance(time_grid_s: np.ndarray, maximum_doppler_hz: float) -> np.ndarray:
    """Return the real symmetric covariance J0(2*pi*fD*|tm-tn|)."""
    times = np.asarray(time_grid_s, dtype=np.float64)
    if (
        times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times))
        or np.any(times < 0) or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("time_grid_s must be a finite, nonnegative, increasing vector")
    if not np.isfinite(maximum_doppler_hz) or maximum_doppler_hz < 0:
        raise ValueError("maximum_doppler_hz must be finite and nonnegative")
    return j0(2 * np.pi * maximum_doppler_hz * np.abs(times[:, None] - times[None, :]))


def covariance_eigenfactor(covariance: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return F=V*sqrt(clipped eigenvalues), with F F^H approximating R.

    Only roundoff-size negative eigenvalues (>= -1e-12) are clipped. A substantive
    negative eigenvalue raises; no jitter or positive-eigenvalue truncation occurs.
    """
    matrix = np.asarray(covariance, dtype=np.complex128)
    if (
        matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix, matrix.conj().T, rtol=0, atol=1e-14)
    ):
        raise ValueError("covariance must be a finite square Hermitian matrix")
    eigenvalues, vectors = np.linalg.eigh(matrix)
    minimum = float(eigenvalues[0])
    if minimum < -NEGATIVE_EIGENVALUE_TOLERANCE:
        raise ValueError(
            f"covariance has substantive negative eigenvalue {minimum}; "
            f"allowed minimum is {-NEGATIVE_EIGENVALUE_TOLERANCE}"
        )
    clipped = np.maximum(eigenvalues, 0.0)
    factor = vectors * np.sqrt(clipped)[None, :]
    summary = {
        "minimum_covariance_eigenvalue": minimum,
        "maximum_covariance_eigenvalue": float(eigenvalues[-1]),
        "raw_eigenvalues": eigenvalues.tolist(),
        "negative_eigenvalues_clipped": int(np.count_nonzero(eigenvalues < 0)),
        "negative_eigenvalue_tolerance": NEGATIVE_EIGENVALUE_TOLERANCE,
        "jitter_added": 0.0,
        "factor_reconstruction_max_error": float(np.max(np.abs(factor @ factor.conj().T - matrix))),
    }
    return factor, summary


def correlated_complex_gaussian(
    rng: np.random.Generator, factor: np.ndarray, process_shape: tuple[int, ...] = (),
) -> np.ndarray:
    """Draw independent processes of shape [...,T] using g_column=F*z_column."""
    transform = np.asarray(factor, dtype=np.complex128)
    if (
        transform.ndim != 2 or transform.shape[0] == 0
        or transform.shape[0] != transform.shape[1] or not np.all(np.isfinite(transform))
    ):
        raise ValueError("factor must be a finite square matrix")
    if not isinstance(process_shape, tuple) or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1
        for v in process_shape
    ):
        raise ValueError("process_shape must be a tuple of positive integers")
    z = complex_gaussian_noise(process_shape + (transform.shape[1],), 1, rng, dtype=np.complex128)
    # Row storage of F*z requires transpose, not conjugate transpose.
    return z @ transform.T


def jakes_complex_gaussian(
    rng: np.random.Generator, time_grid_s: np.ndarray, maximum_doppler_hz: float,
    process_shape: tuple[int, ...] = (),
) -> np.ndarray:
    """Draw proper CN(0,R_time) processes; zero Doppler is exactly time static."""
    covariance = jakes_covariance(time_grid_s, maximum_doppler_hz)
    if maximum_doppler_hz == 0:
        # One coefficient per process, copied exactly across all times.
        unit = correlated_complex_gaussian(rng, np.ones((1, 1)), process_shape)
        return np.repeat(unit, covariance.shape[0], axis=-1)
    factor, _ = covariance_eigenfactor(covariance)
    return correlated_complex_gaussian(rng, factor, process_shape)


def build_tdl_a_doppler_channel(
    channel_seed: int, speed_kmh: float = SPEED_KMH,
    carrier_frequency_hz: float = CARRIER_FREQUENCY_HZ,
    nfft: int = 64, cp_len: int = 16, sample_rate_hz: float = 1920000.0,
    num_symbols: int = 16, delta_f_hz: float = 30000.0,
    desired_rms_delay_spread_ns: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a[RX,TX,tap,T], H[RX,TX,K,T] for symbol-start channel samples.

    The receiver's channel is held fixed during each OFDM symbol. This function
    generates channel coefficients only, with no within-symbol waveform/ICI path.
    """
    if (
        isinstance(channel_seed, (bool, np.bool_))
        or not isinstance(channel_seed, (int, np.integer)) or channel_seed < 0
    ):
        raise ValueError("channel_seed must be a nonnegative integer")
    parameters = mobility_parameters(speed_kmh, carrier_frequency_hz)
    times = ofdm_time_grid_seconds(nfft, cp_len, sample_rate_hz, num_symbols)
    frequencies = centered_frequency_grid_hz(nfft, delta_f_hz)
    if sample_rate_hz != nfft * delta_f_hz:
        raise ValueError("sample_rate_hz must equal nfft * delta_f_hz")
    delays, powers = tdl_a_profile(desired_rms_delay_spread_ns)
    if speed_kmh == 0:
        static_taps, channel = build_static_tdl_a_channel(
            channel_seed, nfft, num_symbols, delta_f_hz, desired_rms_delay_spread_ns,
        )
        return np.repeat(static_taps[..., None], num_symbols, axis=-1), channel
    processes = jakes_complex_gaussian(
        np.random.default_rng(channel_seed), times, parameters["maximum_doppler_hz"], (2, 2, 23),
    )
    taps = processes * np.sqrt(powers)[None, None, :, None]
    # Reuse the verified last-axis tap interface, then restore [RX,TX,K,T].
    response = fractional_delay_frequency_response(np.moveaxis(taps, -2, -1), delays, frequencies)
    return taps, np.moveaxis(response, -2, -1)
