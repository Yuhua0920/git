"""Data-only error energies and count-weighted Monte Carlo aggregation."""

from dataclasses import dataclass

import numpy as np


def channel_error_energies(
    h_est: np.ndarray, h_true: np.ndarray, data_mask: np.ndarray,
) -> tuple[float, float]:
    """Return unnormalized error/truth energies over RX,TX and data REs only."""
    estimate = np.asarray(h_est, dtype=np.complex128)
    truth = np.asarray(h_true, dtype=np.complex128)
    mask = np.asarray(data_mask)
    if estimate.ndim != 4 or estimate.shape != truth.shape:
        raise ValueError("H estimate/truth must share shape (nr,nt,K,T)")
    if mask.dtype != np.bool_ or mask.shape != truth.shape[1:] or not np.any(mask):
        raise ValueError("data_mask must be nonempty boolean (nt,K,T)")
    selected = np.broadcast_to(mask[None], truth.shape)
    if not np.all(np.isfinite(estimate[selected])) or not np.all(np.isfinite(truth[selected])):
        raise ValueError("selected channel entries must be finite")
    error_energy = float(np.sum(np.abs(estimate[selected] - truth[selected]) ** 2))
    truth_energy = float(np.sum(np.abs(truth[selected]) ** 2))
    if truth_energy <= 0:
        raise ValueError("truth energy must be positive")
    return error_energy, truth_energy


def count_bit_errors(recovered: tuple, reference: tuple) -> tuple[int, int]:
    """Return total errors and actual bits from already selected data streams."""
    if len(recovered) != len(reference) or not recovered:
        raise ValueError("recovered/reference streams must have matching nonempty length")
    errors, bits = 0, 0
    for actual, expected in zip(recovered, reference):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.ndim != 1 or actual.shape != expected.shape:
            raise ValueError("bit streams must be matching one-dimensional arrays")
        if not np.all((actual == 0) | (actual == 1)) or not np.all((expected == 0) | (expected == 1)):
            raise ValueError("bit streams must contain only binary values")
        errors += int(np.count_nonzero(actual != expected))
        bits += expected.size
    if bits == 0:
        raise ValueError("data bit count must be positive")
    return errors, int(bits)


@dataclass
class MetricAccumulator:
    """Sum energies and bit counts; ratios and dB are computed only at finalization."""

    num_frames: int = 0
    nmse_error_energy: float = 0.0
    nmse_truth_energy: float = 0.0
    total_data_bits: int = 0
    perfect_csi_bit_errors: int = 0
    ls_lin_bit_errors: int = 0

    def add(self, error_energy: float, truth_energy: float, perfect_errors: int,
            ls_errors: int, data_bits: int) -> None:
        if not np.isfinite(error_energy) or error_energy < 0 or not np.isfinite(truth_energy) or truth_energy <= 0:
            raise ValueError("error/truth energies must be finite, nonnegative/positive")
        for name, value in (("perfect_errors", perfect_errors), ("ls_errors", ls_errors), ("data_bits", data_bits)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if data_bits <= 0 or not 0 <= perfect_errors <= data_bits or not 0 <= ls_errors <= data_bits:
            raise ValueError("bit counts must satisfy 0 <= errors <= positive data_bits")
        self.num_frames += 1
        self.nmse_error_energy += float(error_energy)
        self.nmse_truth_energy += float(truth_energy)
        self.total_data_bits += int(data_bits)
        self.perfect_csi_bit_errors += int(perfect_errors)
        self.ls_lin_bit_errors += int(ls_errors)

    def finalize(self) -> dict:
        if self.num_frames == 0:
            raise ValueError("cannot finalize an empty accumulator")
        nmse = self.nmse_error_energy / self.nmse_truth_energy
        return {
            "num_frames": self.num_frames, "total_data_bits": self.total_data_bits,
            "nmse_error_energy": self.nmse_error_energy, "nmse_truth_energy": self.nmse_truth_energy,
            "nmse_data_linear": nmse,
            "nmse_data_db": float(10 * np.log10(nmse)) if nmse > 0 else float("-inf"),
            "perfect_csi_bit_errors": self.perfect_csi_bit_errors,
            "perfect_csi_ber": self.perfect_csi_bit_errors / self.total_data_bits,
            "ls_lin_bit_errors": self.ls_lin_bit_errors,
            "ls_lin_ber": self.ls_lin_bit_errors / self.total_data_bits,
        }
