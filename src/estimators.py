"""Pilot LS and sequential frequency/time linear interpolation."""

import numpy as np


def pilot_ls_estimate(
    y_grid: np.ndarray,
    x_grid: np.ndarray,
    pilot_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate H[r,q,k,t] = Y[r,k,t]/X[q,k,t] at TX q's pilots only.

    H[r,q] denotes the channel from TX q to RX r. Nonpilot entries are
    NaN+jNaN and must never be used without the returned estimate mask.
    """
    received = np.asarray(y_grid, dtype=np.complex128)
    transmitted = np.asarray(x_grid, dtype=np.complex128)
    mask = np.asarray(pilot_mask)
    if transmitted.ndim != 3 or min(transmitted.shape) < 1:
        raise ValueError("x_grid must have shape (nt, num_subcarriers, num_symbols)")
    nt, num_subcarriers, num_symbols = transmitted.shape
    if received.ndim != 3 or received.shape[1:] != (num_subcarriers, num_symbols) or received.shape[0] < 1:
        raise ValueError("y_grid must have shape (nr, num_subcarriers, num_symbols)")
    if mask.shape != transmitted.shape or mask.dtype != np.bool_:
        raise ValueError("pilot_mask must be a bool array with x_grid's shape")
    if np.any(np.sum(mask, axis=0) > 1):
        raise ValueError("pilot_mask must have no overlapping pilots")
    if not np.all(np.isfinite(received)) or not np.all(np.isfinite(transmitted)):
        raise ValueError("y_grid and x_grid must contain finite values")
    if np.any(transmitted[mask] == 0):
        raise ValueError("pilot symbols must be nonzero")

    nr = received.shape[0]
    estimates = np.full(
        (nr, nt, num_subcarriers, num_symbols),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    for q in range(nt):
        at_q_pilots = mask[q]
        estimates[:, q, at_q_pilots] = (
            received[:, at_q_pilots] / transmitted[q, at_q_pilots][None, :]
        )
    return estimates, mask.copy()


def frequency_linear_interpolate_ls(
    h_ls: np.ndarray,
    pilot_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate each RX/TX frequency trace only at OFDM symbols with pilots.

    Real and imaginary parts are interpolated separately. Values outside the
    first and last pilot hold the nearest pilot value. Unobserved time symbols
    remain NaN+jNaN, with False entries in the returned [Nt,K,T] mask.
    """
    observations = np.asarray(h_ls)
    mask = np.asarray(pilot_mask)
    if observations.ndim != 4 or min(observations.shape) < 1:
        raise ValueError("h_ls must have shape (nr, nt, num_subcarriers, num_symbols)")
    if not np.issubdtype(observations.dtype, np.complexfloating):
        raise ValueError("h_ls must have a complex floating-point dtype")
    nr, nt, num_subcarriers, num_symbols = observations.shape
    if mask.shape != (nt, num_subcarriers, num_symbols) or mask.dtype != np.bool_:
        raise ValueError("pilot_mask must be a bool array with shape (nt, num_subcarriers, num_symbols)")
    if np.any(np.isinf(observations)):
        raise ValueError("h_ls must not contain Inf")
    valid_pilots = np.broadcast_to(mask[None, :, :, :], observations.shape)
    if not np.all(np.isfinite(observations[valid_pilots])):
        raise ValueError("h_ls must contain finite values at every marked pilot")

    pilot_times = np.flatnonzero(np.any(mask, axis=(0, 1)))
    if pilot_times.size == 0:
        raise ValueError("pilot_mask must contain pilot observations")
    for t in pilot_times:
        for q in range(nt):
            if not np.any(mask[q, :, t]):
                raise ValueError(f"TX {q} has no pilots at pilot symbol {t}")

    interpolated = np.full(
        (nr, nt, num_subcarriers, num_symbols),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    interpolated_mask = np.zeros((nt, num_subcarriers, num_symbols), dtype=bool)
    frequency_indices = np.arange(num_subcarriers)
    for t in pilot_times:
        for q in range(nt):
            pilot_indices = np.flatnonzero(mask[q, :, t])
            for r in range(nr):
                pilot_values = observations[r, q, pilot_indices, t]
                real_interp = np.interp(frequency_indices, pilot_indices, pilot_values.real)
                imag_interp = np.interp(frequency_indices, pilot_indices, pilot_values.imag)
                interpolated[r, q, :, t] = real_interp + 1j * imag_interp
            # Preserve the observed complex values exactly at their pilot REs.
            interpolated[:, q, pilot_indices, t] = observations[:, q, pilot_indices, t]
            interpolated_mask[q, :, t] = True
    return interpolated, interpolated_mask


def time_linear_interpolate_frequency_estimates(
    h_frequency: np.ndarray,
    pilot_symbol_indices: list[int] | tuple[int, ...] | np.ndarray,
) -> np.ndarray:
    """Fill time traces of frequency estimates, holding the nearest edge anchor."""
    frequency_estimates = np.asarray(h_frequency)
    if frequency_estimates.ndim != 4 or min(frequency_estimates.shape) < 1:
        raise ValueError("h_frequency must have shape (nr, nt, num_subcarriers, num_symbols)")
    if not np.issubdtype(frequency_estimates.dtype, np.complexfloating):
        raise ValueError("h_frequency must have a complex floating-point dtype")
    nr, nt, num_subcarriers, num_symbols = frequency_estimates.shape
    try:
        index_list = list(pilot_symbol_indices)
    except TypeError as exc:
        raise ValueError("pilot_symbol_indices must be a nonempty one-dimensional integer sequence") from exc
    if not index_list or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in index_list
    ):
        raise ValueError("pilot_symbol_indices must be a nonempty one-dimensional integer sequence")
    try:
        anchors = np.asarray(index_list, dtype=np.int64)
    except (OverflowError, ValueError) as exc:
        raise ValueError("pilot_symbol_indices must fit in signed 64-bit integers") from exc
    if np.any(anchors < 0) or np.any(anchors >= num_symbols):
        raise ValueError("pilot_symbol_indices must be within the time grid")
    if np.any(np.diff(anchors) <= 0):
        raise ValueError("pilot_symbol_indices must be strictly increasing")
    if np.any(np.isinf(frequency_estimates)):
        raise ValueError("h_frequency must not contain Inf")
    if not np.all(np.isfinite(frequency_estimates[..., anchors])):
        raise ValueError("h_frequency must be finite at every declared pilot time")
    nonanchor_times = np.ones(num_symbols, dtype=bool)
    nonanchor_times[anchors] = False
    missing = frequency_estimates[..., nonanchor_times]
    if not (np.all(np.isnan(missing.real)) and np.all(np.isnan(missing.imag))):
        raise ValueError("h_frequency must be NaN+jNaN outside declared pilot times")

    full_estimates = np.empty((nr, nt, num_subcarriers, num_symbols), dtype=np.complex128)
    times = np.arange(num_symbols)
    for r in range(nr):
        for q in range(nt):
            for k in range(num_subcarriers):
                pilot_values = frequency_estimates[r, q, k, anchors]
                real_interp = np.interp(times, anchors, pilot_values.real)
                imag_interp = np.interp(times, anchors, pilot_values.imag)
                full_estimates[r, q, k, :] = real_interp + 1j * imag_interp
    # Do not alter the complete frequency-interpolated anchor slices.
    full_estimates[..., anchors] = frequency_estimates[..., anchors]
    return full_estimates


def ls_linear_interpolate(h_ls: np.ndarray, pilot_mask: np.ndarray) -> np.ndarray:
    """Compose Pilot-LS frequency interpolation followed by time interpolation."""
    frequency_estimates, frequency_mask = frequency_linear_interpolate_ls(h_ls, pilot_mask)
    pilot_times = np.flatnonzero(np.any(frequency_mask, axis=(0, 1)))
    return time_linear_interpolate_frequency_estimates(frequency_estimates, pilot_times)
