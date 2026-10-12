"""Deterministic channel models for T03B and T04B calibration."""

import numpy as np


def build_integer_multipath_channel(
    delays: np.ndarray,
    powers_db: np.ndarray,
    phases: np.ndarray,
    dtype: np.dtype | type = np.complex128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return impulse response and normalized path powers without random phases."""
    delay_array = np.asarray(delays)
    if delay_array.ndim != 1 or delay_array.size == 0:
        raise ValueError("delays must be a nonempty one-dimensional array")
    if not np.issubdtype(delay_array.dtype, np.integer) or np.any(delay_array < 0):
        raise ValueError("delays must be nonnegative integers")
    if np.unique(delay_array).size != delay_array.size:
        raise ValueError("duplicate delays are not supported")

    try:
        powers_db_array = np.asarray(powers_db, dtype=np.float64)
        phase_array = np.asarray(phases, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("powers_db and phases must contain real numbers") from exc
    if (
        powers_db_array.ndim != 1
        or phase_array.ndim != 1
        or powers_db_array.size != delay_array.size
        or phase_array.size != delay_array.size
    ):
        raise ValueError("delays, powers_db, and phases must have equal lengths")
    if not np.all(np.isfinite(powers_db_array)) or not np.all(np.isfinite(phase_array)):
        raise ValueError("powers_db and phases must be finite")
    output_dtype = np.dtype(dtype)
    if not np.issubdtype(output_dtype, np.complexfloating):
        raise TypeError("dtype must be a complex floating-point type")

    with np.errstate(over="raise", invalid="raise"):
        try:
            powers_linear = 10.0 ** (powers_db_array / 10.0)
        except FloatingPointError as exc:
            raise ValueError("linear path powers must be finite") from exc
    if np.any(powers_linear <= 0) or not np.isfinite(np.sum(powers_linear)):
        raise ValueError("linear path powers must be finite and positive")
    normalized_path_powers = powers_linear / np.sum(powers_linear)

    h = np.zeros(int(np.max(delay_array)) + 1, dtype=output_dtype)
    h[delay_array] = np.sqrt(normalized_path_powers) * np.exp(1j * phase_array)
    return h, normalized_path_powers


def channel_frequency_response(h: np.ndarray, nfft: int) -> np.ndarray:
    """Return centered H from the ordinary, unnormalized DFT of zero-padded h."""
    if isinstance(nfft, (bool, np.bool_)) or not isinstance(nfft, (int, np.integer)) or nfft <= 0:
        raise ValueError("nfft must be a positive integer")
    impulse_response = np.asarray(h, dtype=np.complex128)
    if impulse_response.ndim != 1 or not 1 <= impulse_response.size <= nfft:
        raise ValueError("h must be one-dimensional with length from 1 to nfft")
    if not np.all(np.isfinite(impulse_response)):
        raise ValueError("h must contain finite values")
    h_pad = np.zeros(nfft, dtype=np.complex128)
    h_pad[: impulse_response.size] = impulse_response
    return np.fft.fftshift(np.fft.fft(h_pad, n=nfft, norm=None))


def convolve_cp_ofdm_blocks(
    time_grid_with_cp: np.ndarray, h: np.ndarray, nfft: int, cp_len: int
) -> np.ndarray:
    """Linearly convolve each CP block and take the fixed [CP:CP+nfft] window."""
    if isinstance(nfft, (bool, np.bool_)) or not isinstance(nfft, (int, np.integer)) or nfft <= 0:
        raise ValueError("nfft must be a positive integer")
    if (
        isinstance(cp_len, (bool, np.bool_))
        or not isinstance(cp_len, (int, np.integer))
        or not 0 <= cp_len <= nfft
    ):
        raise ValueError("cp_len must be an integer between 0 and nfft")
    blocks = np.asarray(time_grid_with_cp, dtype=np.complex128)
    impulse_response = np.asarray(h, dtype=np.complex128)
    if blocks.ndim != 2 or blocks.shape[0] != nfft + cp_len or blocks.shape[1] < 1:
        raise ValueError("time_grid_with_cp must have shape (nfft + cp_len, num_symbols)")
    if impulse_response.ndim != 1 or impulse_response.size < 1:
        raise ValueError("h must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(blocks)) or not np.all(np.isfinite(impulse_response)):
        raise ValueError("channel input and h must contain finite values")

    useful = np.empty((nfft, blocks.shape[1]), dtype=np.complex128)
    for symbol_index in range(blocks.shape[1]):
        full_convolution = np.convolve(blocks[:, symbol_index], impulse_response, mode="full")
        useful[:, symbol_index] = full_convolution[cp_len : cp_len + nfft]
    return useful


def static_mimo_receive(h: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Return noiseless Y[r,k,t] = sum_q H[r,q] X[q,k,t] for a static channel."""
    channel = np.asarray(h, dtype=np.complex128)
    grid = np.asarray(x_grid, dtype=np.complex128)
    if channel.ndim != 2 or channel.shape[0] < 1 or channel.shape[1] < 1:
        raise ValueError("h must have shape (nr, nt)")
    if grid.ndim != 3 or grid.shape[0] != channel.shape[1] or min(grid.shape[1:]) < 1:
        raise ValueError("x_grid must have shape (nt, num_subcarriers, num_symbols)")
    if not np.all(np.isfinite(channel)) or not np.all(np.isfinite(grid)):
        raise ValueError("h and x_grid must contain finite values")
    # H[r, q] is the channel from transmit antenna q to receive antenna r.
    return np.einsum("rq,qkt->rkt", channel, grid)
