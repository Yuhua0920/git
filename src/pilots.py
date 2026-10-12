"""T04A transmit-resource masks for orthogonal FDM pilots."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PilotLayout:
    """Masks use [transmit antenna, centered frequency, OFDM symbol] axes."""

    pilot_mask: np.ndarray
    pilot_union: np.ndarray
    data_mask: np.ndarray
    mute_mask: np.ndarray
    pilot_owner: np.ndarray
    offsets_by_tx: tuple[int, ...]
    pilot_symbol_indices: tuple[int, ...]


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def build_fdm_pilot_layout(
    nt: int,
    num_subcarriers: int,
    num_symbols: int,
    spacing_frequency: int,
    spacing_time: int,
) -> PilotLayout:
    """Build disjoint per-TX pilots and the complementary data/mute masks."""
    for name, value in (
        ("nt", nt),
        ("num_subcarriers", num_subcarriers),
        ("num_symbols", num_symbols),
        ("spacing_frequency", spacing_frequency),
        ("spacing_time", spacing_time),
    ):
        _positive_integer(value, name)
    if spacing_frequency % nt != 0:
        raise ValueError("spacing_frequency must be divisible by nt")

    offset_step = spacing_frequency // nt
    offsets_by_tx = tuple(q * offset_step for q in range(nt))
    if any(offset >= spacing_frequency or offset >= num_subcarriers for offset in offsets_by_tx):
        raise ValueError("pilot offset is outside the frequency grid")
    if len(set(offsets_by_tx)) != nt:
        raise ValueError("pilot offsets must be unique")
    pilot_symbol_indices = tuple(range(0, num_symbols, spacing_time))

    pilot_mask = np.zeros((nt, num_subcarriers, num_symbols), dtype=bool)
    for q, offset in enumerate(offsets_by_tx):
        frequency_indices = np.arange(offset, num_subcarriers, spacing_frequency)
        pilot_mask[q][np.ix_(frequency_indices, pilot_symbol_indices)] = True

    pilot_union = np.any(pilot_mask, axis=0)
    if np.any(np.sum(pilot_mask, axis=0) > 1):
        raise ValueError("pilot overlap between transmit antennas")
    data_mask = np.broadcast_to(~pilot_union, pilot_mask.shape).copy()
    mute_mask = np.broadcast_to(pilot_union, pilot_mask.shape) & ~pilot_mask
    pilot_owner = np.full((num_subcarriers, num_symbols), -1, dtype=np.int32)
    for q in range(nt):
        pilot_owner[pilot_mask[q]] = q

    if not np.all(pilot_mask.astype(np.int8) + data_mask.astype(np.int8) + mute_mask.astype(np.int8) == 1):
        raise RuntimeError("pilot, data, and mute masks must partition every TX resource")
    return PilotLayout(
        pilot_mask=pilot_mask,
        pilot_union=pilot_union,
        data_mask=data_mask,
        mute_mask=mute_mask,
        pilot_owner=pilot_owner,
        offsets_by_tx=offsets_by_tx,
        pilot_symbol_indices=pilot_symbol_indices,
    )


def count_layout_resources(layout: PilotLayout) -> dict[str, int | float | list[int]]:
    """Count actual mask entries; no theoretical pilot count is assumed."""
    _, num_subcarriers, num_symbols = layout.pilot_mask.shape
    total_re = num_subcarriers * num_symbols
    pilot_count_by_tx = np.count_nonzero(layout.pilot_mask, axis=(1, 2)).astype(int).tolist()
    data_count_by_tx = np.count_nonzero(layout.data_mask, axis=(1, 2)).astype(int).tolist()
    mute_count_by_tx = np.count_nonzero(layout.mute_mask, axis=(1, 2)).astype(int).tolist()
    union_count = int(np.count_nonzero(layout.pilot_union))
    return {
        "pilot_count_by_tx": pilot_count_by_tx,
        "union_count": union_count,
        "union_ratio": union_count / total_re,
        "data_count_by_tx": data_count_by_tx,
        "two_stream_data_symbols": int(sum(data_count_by_tx)),
        "mute_count_by_tx": mute_count_by_tx,
        "total_re": total_re,
    }


def map_pilot_symbols(layout: PilotLayout, pilot_symbols: np.ndarray) -> np.ndarray:
    """Map one unit-energy complex pilot per TX; leave every other grid entry zero."""
    nt = layout.pilot_mask.shape[0]
    try:
        symbols = np.asarray(pilot_symbols, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError("pilot_symbols must contain complex numbers") from exc
    if symbols.shape != (nt,) or not np.all(np.isfinite(symbols)):
        raise ValueError("pilot_symbols must have shape (nt,) and be finite")
    if np.any(symbols == 0):
        raise ValueError("pilot symbols must be nonzero")
    if not np.allclose(np.abs(symbols) ** 2, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("pilot symbols must have unit energy")

    grid = np.zeros(layout.pilot_mask.shape, dtype=np.complex128)
    for q, symbol in enumerate(symbols):
        grid[q, layout.pilot_mask[q]] = symbol
    return grid
