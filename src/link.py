"""Deterministic MIMO integration helpers; axes remain [TX,K,T]."""

import numpy as np

from src.channels import build_integer_multipath_channel, channel_frequency_response
from src.detectors import zf_detect
from src.estimators import ls_linear_interpolate, pilot_ls_estimate
from src.modulation import qpsk_demodulate, qpsk_modulate
from src.numerics import scale_data_streams
from src.pilots import PilotLayout, map_pilot_symbols


DELAYS = (0, 1, 3, 5)
POWERS_DB = (0, -3, -6, -9)
# First two axes are RX, TX, respectively; each link has fixed path phases.
LINK_PHASES = np.array([
    [[0, np.pi / 5, -np.pi / 3, np.pi / 2],
     [np.pi / 4, -np.pi / 6, np.pi / 3, -np.pi / 2]],
    [[-np.pi / 5, np.pi / 2, np.pi / 7, -np.pi / 4],
     [np.pi / 3, np.pi / 8, -np.pi / 2, np.pi / 6]],
], dtype=np.float64)
PILOT_SYMBOLS = np.array([1 + 1j, 1 - 1j], dtype=np.complex128) / np.sqrt(2)


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def generate_master_qpsk(
    num_subcarriers: int = 64,
    num_symbols: int = 16,
    bits_seeds: tuple[int, int] = (20260930, 20260931),
) -> tuple[np.ndarray, np.ndarray]:
    """Return bits[TX,K,T,dibit] and unit-energy S[TX,K,T] from separate RNGs."""
    _positive_integer(num_subcarriers, "num_subcarriers")
    _positive_integer(num_symbols, "num_symbols")
    if len(bits_seeds) != 2 or bits_seeds[0] == bits_seeds[1]:
        raise ValueError("T06B requires two distinct bit seeds")
    bits = np.stack([
        np.random.default_rng(seed).integers(
            0, 2, 2 * num_subcarriers * num_symbols, dtype=np.uint8
        ).reshape(num_subcarriers, num_symbols, 2)
        for seed in bits_seeds
    ])
    symbols = np.stack([
        qpsk_modulate(stream.ravel()).reshape(num_subcarriers, num_symbols)
        for stream in bits
    ])
    return bits, symbols


def build_calibration_mimo_channel(
    nfft: int = 64, num_symbols: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Return independently normalized h[RX,TX,delay] and centered H[RX,TX,K,T]."""
    _positive_integer(nfft, "nfft")
    _positive_integer(num_symbols, "num_symbols")
    impulses = np.empty((2, 2, max(DELAYS) + 1), dtype=np.complex128)
    frequency = np.empty((2, 2, nfft), dtype=np.complex128)
    for r in range(2):
        for q in range(2):
            impulses[r, q], _ = build_integer_multipath_channel(
                np.array(DELAYS), np.array(POWERS_DB), LINK_PHASES[r, q]
            )
            frequency[r, q] = channel_frequency_response(impulses[r, q], nfft)
    return impulses, np.repeat(frequency[..., None], num_symbols, axis=-1)


def assemble_pilot_data_frame(
    s_master: np.ndarray, layout: PilotLayout, pilot_symbols: np.ndarray = PILOT_SYMBOLS,
) -> np.ndarray:
    """Map scaled data, unit-modulus owner pilots, and zero non-owner mute."""
    symbols = np.asarray(s_master, dtype=np.complex128)
    if symbols.shape != layout.pilot_mask.shape or symbols.shape[0] != 2:
        raise ValueError("s_master must match the two-TX layout shape")
    if not np.all(np.isfinite(symbols)) or not np.allclose(np.abs(symbols) ** 2, 1, rtol=0, atol=1e-12):
        raise ValueError("s_master must contain finite unit-energy symbols")
    data = scale_data_streams(symbols.reshape(2, -1).T, 2).T.reshape(symbols.shape)
    frame = map_pilot_symbols(layout, pilot_symbols)
    frame[layout.data_mask] = data[layout.data_mask]
    return frame


def apply_frequency_mimo_channel(h_grid: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Apply noiseless Y[r,k,t] = sum_q H[r,q,k,t] X[q,k,t]."""
    channel = np.asarray(h_grid, dtype=np.complex128)
    transmitted = np.asarray(x_grid, dtype=np.complex128)
    if channel.ndim != 4 or min(channel.shape) < 1:
        raise ValueError("h_grid must have shape (nr, nt, K, T)")
    if transmitted.shape != channel.shape[1:]:
        raise ValueError("x_grid must match h_grid's (nt, K, T) axes")
    if not np.all(np.isfinite(channel)) or not np.all(np.isfinite(transmitted)):
        raise ValueError("h_grid and x_grid must be finite")
    return np.einsum("rqkt,qkt->rkt", channel, transmitted)


def add_received_noise(y_grid: np.ndarray, noise_master: np.ndarray) -> np.ndarray:
    """Add supplied full-grid noise to every RX pilot/data RE without rescaling."""
    received = np.asarray(y_grid, dtype=np.complex128)
    noise = np.asarray(noise_master, dtype=np.complex128)
    if received.ndim != 3 or noise.shape != received.shape:
        raise ValueError("Y and noise must share shape (nr,K,T)")
    if not np.all(np.isfinite(received)) or not np.all(np.isfinite(noise)):
        raise ValueError("Y and noise must be finite")
    return received + noise


def receive_perfect_and_ls_linear(
    y_noisy: np.ndarray, h_reference: np.ndarray, layout: PilotLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use the same received object for both paths; truth is only the reference."""
    h_ls, h_est = estimate_ls_linear_channel(y_noisy, layout)
    perfect_x = detect_data_zf(y_noisy, h_reference, layout.data_mask)
    estimated_x = detect_data_zf(y_noisy, h_est, layout.data_mask)
    return h_ls, h_est, perfect_x, estimated_x


def estimate_ls_linear_channel(
    y_grid: np.ndarray, layout: PilotLayout, pilot_symbols: np.ndarray = PILOT_SYMBOLS,
) -> tuple[np.ndarray, np.ndarray]:
    """Receiver uses only Y and known pilots/layout; no truth or data input."""
    known_pilot_grid = map_pilot_symbols(layout, pilot_symbols)
    h_ls, observed_mask = pilot_ls_estimate(y_grid, known_pilot_grid, layout.pilot_mask)
    return h_ls, ls_linear_interpolate(h_ls, observed_mask)


def _data_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    selected = np.asarray(mask)
    if selected.dtype != np.bool_ or selected.shape != shape:
        raise ValueError("data_mask must be boolean and match (nt, K, T)")
    if not np.any(selected):
        raise ValueError("data_mask must contain data resources")
    return selected


def detect_data_zf(y_grid: np.ndarray, h_grid: np.ndarray, data_mask: np.ndarray) -> np.ndarray:
    """Solve only common data REs; returned nondata symbols are NaN+jNaN."""
    channel = np.asarray(h_grid, dtype=np.complex128)
    received = np.asarray(y_grid, dtype=np.complex128)
    if channel.ndim != 4 or received.shape != (channel.shape[0], *channel.shape[2:]):
        raise ValueError("Y/H must have matching (nr,K,T)/(nr,nt,K,T) shapes")
    mask = _data_mask(data_mask, channel.shape[1:])
    if not np.all(mask == mask[0]):
        raise ValueError("ZF integration requires both TX streams on each data RE")
    selected = mask[0]
    detected = zf_detect(received[:, selected, None], channel[:, :, selected, None])
    output = np.full(channel.shape[1:], np.nan + 1j * np.nan, dtype=np.complex128)
    output[:, selected] = detected[:, :, 0]
    return output


def extract_reference_data_bits(bits_grid: np.ndarray, data_mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Select master dibits using the same TX,K,T order as symbol selection."""
    bits = np.asarray(bits_grid)
    if bits.ndim != 4 or bits.shape[-1] != 2 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits_grid must contain binary dibits with shape (nt,K,T,2)")
    mask = _data_mask(data_mask, bits.shape[:-1])
    return tuple(bits[q][mask[q]].reshape(-1) for q in range(bits.shape[0]))


def demodulate_data_bits(s_grid: np.ndarray, data_mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Demap only selected data symbols, already restored to unit constellation scale."""
    symbols = np.asarray(s_grid, dtype=np.complex128)
    if symbols.ndim != 3:
        raise ValueError("s_grid must have shape (nt,K,T)")
    mask = _data_mask(data_mask, symbols.shape)
    return tuple(qpsk_demodulate(symbols[q][mask[q]]) for q in range(symbols.shape[0]))


def data_channel_nmse(h_est: np.ndarray, h_true: np.ndarray, data_mask: np.ndarray) -> float:
    """Evaluation only: aggregate squared errors over RX, TX and selected data REs."""
    estimate = np.asarray(h_est, dtype=np.complex128)
    truth = np.asarray(h_true, dtype=np.complex128)
    if estimate.ndim != 4 or truth.shape != estimate.shape:
        raise ValueError("h_est and h_true must share shape (nr,nt,K,T)")
    mask = _data_mask(data_mask, truth.shape[1:])
    selected = np.broadcast_to(mask[None], truth.shape)
    if not np.all(np.isfinite(estimate[selected])) or not np.all(np.isfinite(truth[selected])):
        raise ValueError("channel values on data REs must be finite")
    denominator = float(np.sum(np.abs(truth[selected]) ** 2))
    if denominator <= 0:
        raise ValueError("true channel energy on data REs must be positive")
    return float(np.sum(np.abs(estimate[selected] - truth[selected]) ** 2) / denominator)


def detect_data_linear_mmse(
    y_grid: np.ndarray, h_est: np.ndarray, data_mask: np.ndarray,
    noise_variance: float, stream_symbol_power: float, *, return_diagnostics: bool = False,
):
    """D1 wrapper: common data REs and nondata NaNs, exactly as D0's wrapper.

    Return the actual scaled transmit symbols; the caller uses the existing
    constellation restoration and hard slicer. h_est is supplied by the caller.
    """
    from src.detectors import linear_mmse_detect

    channel = np.asarray(h_est, dtype=np.complex128)
    received = np.asarray(y_grid, dtype=np.complex128)
    if channel.ndim != 4 or received.shape != (channel.shape[0], *channel.shape[2:]):
        raise ValueError("Y/H must have matching (nr,K,T)/(nr,nt,K,T) shapes")
    mask = _data_mask(data_mask, channel.shape[1:])
    if not np.all(mask == mask[0]):
        raise ValueError("Linear MMSE integration requires both TX streams on each data RE")
    selected = mask[0]
    result = linear_mmse_detect(received[:, selected, None], channel[:, :, selected, None],
        noise_variance, stream_symbol_power, return_diagnostics=return_diagnostics)
    detected, diagnostics = result if return_diagnostics else (result, None)
    output = np.full(channel.shape[1:], np.nan + 1j * np.nan, dtype=np.complex128)
    output[:, selected] = detected[:, :, 0]
    return (output, diagnostics) if return_diagnostics else output
