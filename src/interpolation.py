"""R2: separable nearest interpolation of existing sparse Pilot-LS tensors.

Axes are [RX,TX,physical-frequency index,time]. Frequency precedes time;
ties choose the lower index, and boundaries hold without circular wrapping.
The estimator receives observations/masks only, never channel truth.
"""

import numpy as np


def _complex_grid(values: np.ndarray, name: str) -> np.ndarray:
    grid = np.asarray(values)
    if grid.ndim != 4 or min(grid.shape) < 1:
        raise ValueError(f"{name} must have shape (nr, nt, num_subcarriers, num_symbols)")
    if not np.issubdtype(grid.dtype, np.complexfloating):
        raise ValueError(f"{name} must have a complex floating-point dtype")
    if np.any(np.isinf(grid)):
        raise ValueError(f"{name} must not contain Inf")
    return grid


def _nearest_indices(anchors: np.ndarray, grid_size: int) -> np.ndarray:
    """Map a linear integer grid to sorted anchors; <= explicitly breaks ties.

    Callers provide nonempty, strictly increasing, in-grid anchors. Clipping
    both neighboring positions to the endpoint implements non-circular hold.
    """
    coordinates = np.arange(grid_size)
    right = np.searchsorted(anchors, coordinates, side="left")
    left = np.clip(right - 1, 0, anchors.size - 1)
    right = np.clip(right, 0, anchors.size - 1)
    prefer_left = np.abs(coordinates - anchors[left]) <= np.abs(anchors[right] - coordinates)
    return np.where(prefer_left, anchors[left], anchors[right])


def nearest_interpolate_frequency(
    h_ls: np.ndarray, pilot_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill frequency at pilot times, retaining NaN+jNaN at other times.

    Return complex128 [Nr,Nt,K,T] and its boolean [Nt,K,T] coverage mask.
    Only each TX's marked observations are available as complex anchors.
    """
    observations = _complex_grid(h_ls, "h_ls")
    mask = np.asarray(pilot_mask)
    nr, nt, num_subcarriers, num_symbols = observations.shape
    if mask.shape != (nt, num_subcarriers, num_symbols) or mask.dtype != np.bool_:
        raise ValueError("pilot_mask must be boolean with shape (nt, num_subcarriers, num_symbols)")
    if np.any(mask.sum(axis=0) > 1):
        raise ValueError("pilot_mask must contain TX-separated, nonoverlapping pilots")
    marked = np.broadcast_to(mask[None], observations.shape)
    if not np.all(np.isfinite(observations[marked])):
        raise ValueError("h_ls must contain finite values at every marked pilot")
    missing = observations[~marked]
    if not (np.all(np.isnan(missing.real)) and np.all(np.isnan(missing.imag))):
        raise ValueError("h_ls must be NaN+jNaN outside marked pilots")
    pilot_times = np.flatnonzero(np.any(mask, axis=(0, 1)))
    if pilot_times.size == 0:
        raise ValueError("pilot_mask must contain pilot observations")
    for t in pilot_times:
        for q in range(nt):
            if not np.any(mask[q, :, t]):
                raise ValueError(f"TX {q} has no pilots at pilot symbol {t}")

    output = np.full(observations.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    coverage = np.zeros(mask.shape, dtype=bool)
    for t in pilot_times:
        for q in range(nt):
            anchors = np.flatnonzero(mask[q, :, t])
            chosen = _nearest_indices(anchors, num_subcarriers)
            # One complex sample is copied, including its paired real/imag parts.
            output[:, q, :, t] = observations[:, q, :, t][:, chosen]
            coverage[q, :, t] = True
    return output, coverage


def nearest_interpolate_time(
    h_frequency: np.ndarray,
    pilot_symbol_indices: list[int] | tuple[int, ...] | np.ndarray,
) -> np.ndarray:
    """Copy complete frequency slices from the nearest time anchor.

    Equal distances choose the earlier time. Off-anchor input slices must
    retain the existing NaN+jNaN semantics. This is frame/block estimation:
    a symbol can use a later pilot inside the same frame.
    """
    frequency = _complex_grid(h_frequency, "h_frequency")
    try:
        index_list = list(pilot_symbol_indices)
    except TypeError as exc:
        raise ValueError("pilot_symbol_indices must be a nonempty integer sequence") from exc
    if not index_list or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
        for v in index_list
    ):
        raise ValueError("pilot_symbol_indices must be a nonempty integer sequence")
    try:
        anchors = np.asarray(index_list, dtype=np.int64)
    except (OverflowError, ValueError) as exc:
        raise ValueError("pilot_symbol_indices must fit in signed 64-bit integers") from exc
    if np.any(anchors < 0) or np.any(anchors >= frequency.shape[-1]):
        raise ValueError("pilot_symbol_indices must be within the time grid")
    if np.any(np.diff(anchors) <= 0):
        raise ValueError("pilot_symbol_indices must be strictly increasing")
    if not np.all(np.isfinite(frequency[..., anchors])):
        raise ValueError("h_frequency must be finite at every declared pilot time")
    nonanchor = np.ones(frequency.shape[-1], dtype=bool)
    nonanchor[anchors] = False
    missing = frequency[..., nonanchor]
    if not (np.all(np.isnan(missing.real)) and np.all(np.isnan(missing.imag))):
        raise ValueError("h_frequency must be NaN+jNaN outside declared pilot times")
    chosen = _nearest_indices(anchors, frequency.shape[-1])
    return np.take(frequency, chosen, axis=-1).astype(np.complex128, copy=False)


def ls_nearest_interpolate(h_ls: np.ndarray, pilot_mask: np.ndarray) -> np.ndarray:
    """Pilot-LS -> frequency nearest -> time nearest, preserving all pilots."""
    frequency, coverage = nearest_interpolate_frequency(h_ls, pilot_mask)
    times = np.flatnonzero(np.any(coverage, axis=(0, 1)))
    return nearest_interpolate_time(frequency, times)
