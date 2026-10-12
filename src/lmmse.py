"""T10A matched-statistics Oracle LMMSE, independently for each antenna pair.

Grid axes are [K,T], flattened in C order: index = k*T+t. Observations are
Pilot-LS values, not raw received Y. No channel realization enters this module.
Prepared objects retain Cholesky factors, never an inverse; no jitter or fallback.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from src.doppler import jakes_covariance, mobility_parameters, ofdm_time_grid_seconds
from src.tdl import centered_frequency_grid_hz, tdl_a_profile


NUMERICAL_TOLERANCE = 1e-12


def build_frequency_covariance(
    nfft: int = 64, delta_f_hz: float = 30000.0,
    desired_rms_delay_spread_ns: float = 100.0,
) -> np.ndarray:
    """Rf[k,k'] = sum_l p_l exp(-j*2*pi*(f_k-f_k')*tau_l)."""
    frequencies = centered_frequency_grid_hz(nfft, delta_f_hz)
    delays, powers = tdl_a_profile(desired_rms_delay_spread_ns)
    difference = frequencies[:, None] - frequencies[None, :]
    return np.sum(powers * np.exp(-2j * np.pi * difference[..., None] * delays), axis=-1)


def build_time_covariance(
    nfft: int = 64, cp_len: int = 16, sample_rate_hz: float = 1920000.0,
    num_symbols: int = 16, speed_kmh: float = 30.0,
    carrier_frequency_hz: float = 3.5e9,
) -> np.ndarray:
    """Reuse the verified T08B time grid, mobility parameters and Jakes API."""
    times = ofdm_time_grid_seconds(nfft, cp_len, sample_rate_hz, num_symbols)
    doppler = mobility_parameters(speed_kmh, carrier_frequency_hz)["maximum_doppler_hz"]
    return jakes_covariance(times, doppler)


def _hermitian_matrix(covariance: np.ndarray) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.complex128)
    if (matrix.ndim != 2 or not matrix.size or matrix.shape[0] != matrix.shape[1]
            or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix, matrix.conj().T, rtol=0, atol=NUMERICAL_TOLERANCE)):
        raise ValueError("covariance must be a finite square Hermitian matrix")
    return matrix


def build_full_grid_covariance(r_frequency: np.ndarray, r_time: np.ndarray) -> np.ndarray:
    """Rf tensor Rt for [K,T] C order; no full-grid eigenvalue decomposition."""
    return np.kron(_hermitian_matrix(r_frequency), _hermitian_matrix(r_time))


def pilot_flat_indices(pilot_mask: np.ndarray) -> np.ndarray:
    """Extract sorted indices exclusively from the actual per-TX [K,T] mask."""
    mask = np.asarray(pilot_mask)
    if mask.ndim != 2 or mask.dtype != np.bool_ or not np.any(mask):
        raise ValueError("pilot_mask must be a nonempty boolean [K,T] mask")
    return np.flatnonzero(mask.ravel(order="C"))


class LMMSEPreparationError(ValueError):
    """Numerical preparation failure with diagnostics, without any fallback."""

    def __init__(self, record: dict):
        self.record = record
        super().__init__(record["reason"])


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PreparedOracleLMMSE:
    """Explicit per-layout/TX/SNR cache, shared by RXs and trajectories.

    estimate() accepts only pilot observations. Covariance/noise/grid are fixed
    at preparation; changing parameters requires preparing a new object.
    """

    pilot_indices: np.ndarray
    r_hp: np.ndarray
    r_pp: np.ndarray
    observation_covariance: np.ndarray
    cholesky_factor: tuple[np.ndarray, bool]
    noise_variance: float
    grid_shape: tuple[int, int]
    prior_diagonal: np.ndarray
    posterior_diagonal: np.ndarray
    diagnostics: dict

    def estimate(self, pilot_observations: np.ndarray) -> np.ndarray:
        """Map [...,Np] Pilot-LS to [...,K,T], without mixing leading axes."""
        observations = np.asarray(pilot_observations, dtype=np.complex128)
        if (observations.ndim < 1 or observations.shape[-1] != self.pilot_indices.size
                or not np.all(np.isfinite(observations))):
            raise ValueError("finite pilot_observations[...,Np] are required")
        rhs = observations.reshape(-1, self.pilot_indices.size).T
        weights = cho_solve(self.cholesky_factor, rhs, check_finite=True)
        estimate = (self.r_hp @ weights).T
        if not np.all(np.isfinite(estimate)):
            raise ValueError("LMMSE solve produced nonfinite estimates")
        return estimate.reshape(observations.shape[:-1] + self.grid_shape, order="C")

    def posterior_variance(self) -> np.ndarray:
        """Return an un-clipped, precomputed real posterior diagonal [K,T]."""
        return self.posterior_diagonal.reshape(self.grid_shape, order="C")


def prepare_oracle_lmmse(
    covariance: np.ndarray, pilot_indices: np.ndarray, noise_variance: float,
    grid_shape: tuple[int, int],
) -> PreparedOracleLMMSE:
    """Generic Cholesky implementation used by both the tiny sentinel and TDL-A.

    The caller establishes PSD of the prior (Rf/Rt factors for the main model).
    Positive observation noise is mandatory. Diagonal-only posterior calculation
    uses a solve against Rph; it does not allocate a full error covariance.
    """
    matrix = _hermitian_matrix(covariance)
    indices = np.asarray(pilot_indices)
    if (indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices >= matrix.shape[0])
            or np.any(np.diff(indices.astype(np.int64)) <= 0)):
        raise ValueError("pilot_indices must be sorted unique integer covariance indices")
    if (not isinstance(grid_shape, tuple) or len(grid_shape) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                   or v <= 0 for v in grid_shape) or np.prod(grid_shape) != matrix.shape[0]):
        raise ValueError("grid_shape must be a positive (K,T) tuple matching covariance")
    if not np.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError("noise_variance must be finite and strictly positive")
    prior = np.diag(matrix).real.copy()
    if np.any(prior < 0):
        raise ValueError("prior variances must be nonnegative")
    indices = indices.astype(np.int64, copy=True)
    r_hp = matrix[:, indices].copy()
    r_pp = matrix[np.ix_(indices, indices)].copy()
    a_matrix = r_pp + float(noise_variance) * np.eye(indices.size)
    condition = float(np.linalg.cond(a_matrix))
    try:
        factor = cho_factor(a_matrix, lower=True, check_finite=True)
    except np.linalg.LinAlgError as exc:
        raise LMMSEPreparationError({"reason": f"Cholesky failed: {exc}",
            "noise_variance": float(noise_variance), "pilot_count": int(indices.size),
            "condition_number": condition, "jitter_used": False,
            "pseudo_inverse_used": False, "cholesky_success": False}) from exc
    solved_cross = cho_solve(factor, r_hp.conj().T, check_finite=True)
    posterior_complex = np.diag(matrix) - np.sum(r_hp * solved_cross.T, axis=1)
    imaginary_error = float(np.max(np.abs(posterior_complex.imag)))
    posterior = posterior_complex.real.copy()
    if (not np.all(np.isfinite(posterior_complex)) or imaginary_error > NUMERICAL_TOLERANCE
            or np.any(posterior < -NUMERICAL_TOLERANCE)
            or np.any(posterior > prior + NUMERICAL_TOLERANCE)):
        raise LMMSEPreparationError({"reason": "posterior diagonal failed numerical bounds",
            "posterior_min": float(np.min(posterior)), "posterior_max": float(np.max(posterior)),
            "imaginary_error": imaginary_error, "jitter_used": False})
    cholesky_diagonal = np.abs(np.diag(factor[0]))
    diagnostics = {
        "cholesky_success": True, "condition_number": condition,
        "cholesky_diagonal_min": float(np.min(cholesky_diagonal)),
        "cholesky_diagonal_max": float(np.max(cholesky_diagonal)),
        "posterior_min": float(np.min(posterior)), "posterior_max": float(np.max(posterior)),
        "posterior_max_imaginary": imaginary_error,
        "posterior_negative_roundoff_count": int(np.count_nonzero(posterior < 0)),
        "posterior_clipped": False, "jitter_used": False,
    }
    return PreparedOracleLMMSE(_readonly(indices), _readonly(r_hp), _readonly(r_pp),
        _readonly(a_matrix), (_readonly(factor[0]), factor[1]), float(noise_variance), grid_shape,
        _readonly(prior), _readonly(posterior), diagnostics)


def estimate_mimo_from_pilot_ls(
    pilot_ls: np.ndarray, prepared_by_tx: tuple[PreparedOracleLMMSE, ...],
) -> np.ndarray:
    """Return [Nr,Nt,K,T] using only each TX's observed Pilot-LS entries."""
    observations = np.asarray(pilot_ls, dtype=np.complex128)
    if (not prepared_by_tx or observations.ndim != 4 or observations.shape[0] < 1
            or observations.shape[1] != len(prepared_by_tx)
            or any(obj.grid_shape != observations.shape[2:] for obj in prepared_by_tx)):
        raise ValueError("pilot_ls [Nr,Nt,K,T] must match the prepared per-TX estimators")
    output = np.empty(observations.shape, dtype=np.complex128)
    for q, estimator in enumerate(prepared_by_tx):
        z = observations[:, q].reshape(observations.shape[0], -1, order="C")[:, estimator.pilot_indices]
        output[:, q] = estimator.estimate(z)
    return output
