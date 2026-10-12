"""Single-stream Gray-QPSK primitives for the T02 AWGN reference link."""

import numpy as np
from scipy.special import erfc


def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """Map dibits 00, 01, 11, 10 to QPSK quadrants I, IV, III, II."""
    bit_array = np.asarray(bits)
    if bit_array.ndim != 1:
        raise ValueError("bits must be a one-dimensional sequence")
    if bit_array.size % 2 != 0:
        raise ValueError("the number of bits must be even")
    if not np.all((bit_array == 0) | (bit_array == 1)):
        raise ValueError("bits must contain only 0 and 1")

    dibits = bit_array.astype(np.float64, copy=False).reshape(-1, 2)
    real = 1.0 - 2.0 * dibits[:, 0]
    imag = 1.0 - 2.0 * dibits[:, 1]
    return np.asarray((real + 1j * imag) / np.sqrt(2.0), dtype=np.complex128)


def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """Return hard-decision bits; an exactly zero component chooses bit 0."""
    symbol_array = np.asarray(symbols, dtype=np.complex128)
    if symbol_array.ndim != 1:
        raise ValueError("symbols must be a one-dimensional sequence")
    if not np.all(np.isfinite(symbol_array)):
        raise ValueError("symbols must be finite")

    bits = np.empty(2 * symbol_array.size, dtype=np.uint8)
    bits[0::2] = symbol_array.real < 0
    bits[1::2] = symbol_array.imag < 0
    return bits


def ebn0_to_esn0_db(ebn0_db: float) -> float:
    """For unit-energy QPSK, Es/N0 is 2 times Eb/N0 in linear units."""
    if not np.isfinite(ebn0_db):
        raise ValueError("ebn0_db must be finite")
    return float(ebn0_db + 10.0 * np.log10(2.0))


def ebn0_to_noise_variance(ebn0_db: float) -> float:
    """Return N0=E[|n|²]=1/(2*10**(Eb/N0[dB]/10)) for Es=1."""
    if not np.isfinite(ebn0_db):
        raise ValueError("ebn0_db must be finite")
    return float(0.5 / (10.0 ** (ebn0_db / 10.0)))


def qpsk_awgn_theoretical_ber(ebn0_db: float) -> float:
    """Gray-QPSK AWGN bit error probability, 0.5*erfc(sqrt(Eb/N0))."""
    if not np.isfinite(ebn0_db):
        raise ValueError("ebn0_db must be finite")
    gamma_b = 10.0 ** (ebn0_db / 10.0)
    return float(0.5 * erfc(np.sqrt(gamma_b)))
