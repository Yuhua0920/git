"""T01 noise and transmit-power conventions; no signal chain is implemented."""

import numpy as np


def complex_gaussian_noise(
    shape: int | tuple[int, ...],
    variance: float,
    rng: np.random.Generator,
    dtype: np.dtype | type = np.complex128,
) -> np.ndarray:
    """Draw noise with E[|n|²] = variance per complex sample."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("variance must be finite and nonnegative")
    output_dtype = np.dtype(dtype)
    if not np.issubdtype(output_dtype, np.complexfloating):
        raise TypeError("dtype must be a complex floating-point type")
    if variance == 0:
        return np.zeros(shape, dtype=output_dtype)

    scale = np.sqrt(variance / 2.0)
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return (scale * (real + 1j * imag)).astype(output_dtype, copy=False)


def noise_variance_from_snr(snr_db: float, total_power: float = 1.0) -> float:
    """Convert SNR=Ptot/variance to complex-noise variance."""
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if not np.isfinite(total_power) or total_power <= 0:
        raise ValueError("total_power must be finite and positive")
    return float(total_power / (10.0 ** (snr_db / 10.0)))


def scale_data_streams(
    symbols: np.ndarray, nt: int, total_power: float = 1.0
) -> np.ndarray:
    """Scale unit-power input streams; input shape is (RE, transmit antenna)."""
    streams = np.asarray(symbols, dtype=np.complex128)
    if nt < 1 or streams.ndim != 2 or streams.shape[1] != nt:
        raise ValueError("symbols must have shape (num_re, nt)")
    if not np.isfinite(total_power) or total_power <= 0:
        raise ValueError("total_power must be finite and positive")
    return streams * np.sqrt(total_power / nt)
