"""Oracle covariance-weighted filter correction; no learned/online adapter."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cholesky, solve_triangular, subspace_angles

RANKS = (0, 1, 2, 4, 8, 16, 32, 48, 64)
SOURCE_MEAN_RANKS = (0, 4, 8, 16, 64)
T15C_COMMIT = "43c09b43b0e8226efacc71a46881dc9208163db6"
EXPERIMENT_TYPE = "oracle representation gate for low-rank source-to-target LMMSE filter correction under realistic CDL distribution shifts"


def readonly(a):
    a = np.asarray(a, dtype=np.complex128).copy()
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class WeightedCorrection:
    """SVD of (W_target-W_source)L, Cz=LL^H, in complex128.

    Q L=V_r^H is solved by transposing the right solve, not conjugating
    an ordinary transpose. Production inference stores A,Q, not dense B.
    """
    source: np.ndarray
    target: np.ndarray
    covariance: np.ndarray
    lower: np.ndarray
    left: np.ndarray
    singular: np.ndarray
    right_h: np.ndarray

    @classmethod
    def build(cls, source, target, covariance):
        source, target, covariance = map(readonly, (source, target, covariance))
        if source.ndim != 2 or source.shape != target.shape:
            raise ValueError("source/target filter shape mismatch")
        if covariance.shape != (source.shape[1], source.shape[1]):
            raise ValueError("pilot covariance shape mismatch")
        if not all(np.isfinite(a).all() for a in (source, target, covariance)):
            raise ValueError("nonfinite filter/covariance")
        if not np.allclose(covariance, covariance.conj().T, rtol=1e-13, atol=1e-13):
            raise ValueError("Cz must be Hermitian; no silent symmetrization")
        lower = cholesky(covariance, lower=True)
        u, singular, vh = np.linalg.svd((target-source) @ lower, full_matrices=False)
        singular.flags.writeable = False
        return cls(source, target, covariance, readonly(lower), readonly(u), singular, readonly(vh))

    def factors(self, rank):
        if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or not 0 <= rank <= len(self.singular):
            raise ValueError("invalid rank")
        a = self.left[:, :rank] * self.singular[:rank]
        q = solve_triangular(self.lower.T, self.right_h[:rank].T, lower=False).T
        return readonly(a), readonly(q)

    def estimate(self, pilot, mean_p, mean_d, factors):
        x = np.asarray(pilot, dtype=np.complex128) - mean_p
        a, q = factors
        base = mean_d + x @ self.source.T
        # The rank-zero path retains the exact source multiply/affine order.
        return base if q.shape[0] == 0 else base + (x @ q.T) @ a.T

    def spectrum(self):
        energy = self.singular**2
        total = float(energy.sum())
        if total == 0:
            return np.zeros_like(energy), 0.
        p = energy/total
        cumulative = np.cumsum(energy)/total
        return cumulative, float(np.exp(-np.sum(p[p > 0]*np.log(p[p > 0]))))

    def diagnostics(self):
        a, q = self.factors(len(self.singular))
        delta = self.target-self.source
        correction = a @ q
        return dict(
            covariance_hermitian_relative_residual=float(np.linalg.norm(self.covariance-self.covariance.conj().T)/np.linalg.norm(self.covariance)),
            covariance_min_eigenvalue=float(np.linalg.eigvalsh(self.covariance)[0]),
            cholesky_relative_residual=float(np.linalg.norm(self.lower@self.lower.conj().T-self.covariance)/np.linalg.norm(self.covariance)),
            full_rank_correction_relative_residual=float(np.linalg.norm(correction-delta)/max(np.linalg.norm(delta), np.finfo(float).tiny)),
            full_rank_filter_relative_residual=float(np.linalg.norm(self.source+correction-self.target)/max(np.linalg.norm(self.target), np.finfo(float).tiny)),
            triangular_right_solve_relative_residual=float(np.linalg.norm(q@self.lower-self.right_h)/np.linalg.norm(self.right_h)))


def closure(source_db, matched_db, rank_db):
    gap = source_db-matched_db
    if not np.isfinite([source_db, matched_db, rank_db]).all() or gap <= 0:
        raise ValueError("closure requires finite, positive frozen-to-matched gap")
    recovery = source_db-rank_db
    return dict(full_mismatch_gap_db=float(gap), rank_recovery_db=float(recovery), closure_r=float(recovery/gap))


def required_rank(rows, threshold):
    candidates = [int(r["rank"]) for r in rows if r["closure_r"] >= threshold]
    return min(candidates) if candidates else None


def design_gate(rows):
    focus = [r for r in rows if r["snr_db"] in (10, 20)]
    if len(focus) != 4*len(RANKS):
        raise ValueError("gate requires two shifts x two primary SNRs x declared ranks")
    viable = any(all(r["closure_r"] >= .8 for r in focus if r["rank"] == k) for k in RANKS if k <= 8)
    weak = not viable and any(r["closure_r"] < .5 for r in focus if r["rank"] == 16)
    return dict(low_rank_correction_representation_viable=bool(viable),
                low_rank_correction_representation_weak=bool(weak),
                low_rank_correction_representation_partial=bool(not viable and not weak))


def complexity(rank, data_coefficients=3968, pilot_coefficients=64):
    extra = rank*(pilot_coefficients+data_coefficients)
    base = data_coefficients*pilot_coefficients
    return dict(rank=rank, A_shape=[data_coefficients, rank], Q_shape=[rank, pilot_coefficients],
                extra_complex_MAC_per_frame=extra, source_complex_MAC_per_frame=base,
                overhead_ratio=extra/base, coefficient_adaptation_cost_included=False)


def angles(first, second, rank, space):
    if space == "left_data":
        a, b = first.left[:, :rank], second.left[:, :rank]
    elif space == "right_weighted_pilot":
        a, b = first.right_h.conj().T[:, :rank], second.right_h.conj().T[:, :rank]
    else:
        raise ValueError("unknown singular subspace")
    return np.degrees(subspace_angles(a, b))
