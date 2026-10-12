"""T03A OFDM transform and cyclic prefix for centered [frequency, symbol] grids."""

import numpy as np


def _validate_parameters(nfft: int, cp_len: int) -> None:
    if isinstance(nfft, (bool, np.bool_)) or not isinstance(nfft, (int, np.integer)) or nfft <= 0:
        raise ValueError("nfft must be a positive integer")
    if (
        isinstance(cp_len, (bool, np.bool_))
        or not isinstance(cp_len, (int, np.integer))
        or cp_len < 0
        or cp_len > nfft
    ):
        raise ValueError("cp_len must be an integer between 0 and nfft")


def ofdm_modulate_grid(frequency_grid: np.ndarray, nfft: int, cp_len: int) -> np.ndarray:
    """Convert a centered [K,T] grid to [nfft+CP,T] without CP power scaling."""
    _validate_parameters(nfft, cp_len)
    grid = np.asarray(frequency_grid, dtype=np.complex128)
    if grid.ndim != 2 or grid.shape[0] != nfft or grid.shape[1] < 1:
        raise ValueError("frequency_grid must have shape (nfft, num_symbols)")
    if not np.all(np.isfinite(grid)):
        raise ValueError("frequency_grid must contain finite values")

    unshifted = np.fft.ifftshift(grid, axes=0)
    useful_time = np.fft.ifft(unshifted, axis=0, norm="ortho")
    if cp_len == 0:
        return useful_time.copy()
    return np.concatenate((useful_time[-cp_len:, :], useful_time), axis=0)


def remove_cyclic_prefix(time_grid_with_cp: np.ndarray, nfft: int, cp_len: int) -> np.ndarray:
    """Return the exact nfft useful time samples from each OFDM symbol."""
    _validate_parameters(nfft, cp_len)
    grid = np.asarray(time_grid_with_cp, dtype=np.complex128)
    if grid.ndim != 2 or grid.shape[0] != nfft + cp_len or grid.shape[1] < 1:
        raise ValueError("time_grid_with_cp must have shape (nfft + cp_len, num_symbols)")
    if not np.all(np.isfinite(grid)):
        raise ValueError("time_grid_with_cp must contain finite values")
    return grid[cp_len:, :]


def ofdm_demodulate_grid(
    time_grid_with_cp: np.ndarray, nfft: int, cp_len: int
) -> np.ndarray:
    """Remove CP, apply unitary FFT on axis 0, and restore centered ordering."""
    useful_time = remove_cyclic_prefix(time_grid_with_cp, nfft, cp_len)
    unshifted = np.fft.fft(useful_time, axis=0, norm="ortho")
    return np.fft.fftshift(unshifted, axes=0)
