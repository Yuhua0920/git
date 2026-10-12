"""T16G development-weighted correction factors and frozen-coordinate inference.

This module has no file, channel, reporting-label, or coefficient-fitting API.
Factor construction receives only one development correction and its covariance.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import cholesky, solve_triangular

RANKS = (1, 2, 4, 8, 16)
SENTINEL_RANK = 64
PRIMARY_CONDITIONS = frozenset(
    (shift, snr) for shift in ("CDL-C-75", "CDL-B-120") for snr in (10, 20)
)


def _readonly(value, dtype=np.complex128):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _matrix(value, name):
    result = np.asarray(value, dtype=np.complex128)
    if result.ndim != 2 or 0 in result.shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite, nonempty matrix")
    return result


def _covariance(value, size):
    result = _matrix(value, "covariance")
    if result.shape != (size, size):
        raise ValueError("pilot covariance dimensions must match the correction")
    scale = max(float(np.max(np.abs(result))), np.finfo(float).tiny)
    if np.max(np.abs(result - result.conj().T)) > 32 * np.finfo(float).eps * scale:
        raise ValueError("covariance must be Hermitian; no silent symmetrization")
    return result


def _rank(rank, maximum):
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)):
        raise ValueError("rank must be an integer")
    if not 1 <= rank <= maximum:
        raise ValueError("rank exceeds the available singular dimensions")
    return int(rank)


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.complex128)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with the required length")
    return result


@dataclass(frozen=True)
class RankFactors:
    """Immutable A (data x rank), Q (rank x pilot); no dense reconstruction API."""
    A: np.ndarray
    Q: np.ndarray

    def __post_init__(self):
        a, q = _matrix(self.A, "A"), _matrix(self.Q, "Q")
        if a.shape[1] != q.shape[0]:
            raise ValueError("A/Q rank dimensions must match")
        object.__setattr__(self, "A", _readonly(a))
        object.__setattr__(self, "Q", _readonly(q))

    @property
    def rank(self):
        return self.Q.shape[0]

    def apply(self, centered_pilot):
        x = _vector(centered_pilot, self.Q.shape[1], "centered pilot")
        return self.A @ (self.Q @ x)


@dataclass(frozen=True)
class WeightedModule:
    """SVD of DeltaW L with Cz=LL^H, retaining complex128 factor arrays.

    Q L=V^H is a right solve. Its ordinary-transpose form is
    L.T Q.T=Vh.T; a conjugate transpose here would produce the wrong operator.
    Full factors are retained so every declared rank uses the same SVD prefix.
    """
    delta: np.ndarray
    covariance: np.ndarray
    lower: np.ndarray
    singular_values: np.ndarray
    A_full: np.ndarray
    Q_full: np.ndarray
    right_h: np.ndarray
    construction_seconds: float
    svd_seconds: float

    @classmethod
    def build(cls, delta, covariance):
        started = perf_counter()
        delta = _matrix(delta, "correction")
        covariance = _covariance(covariance, delta.shape[1])
        lower = cholesky(covariance, lower=True, check_finite=True)
        weighted = delta @ lower
        svd_started = perf_counter()
        u, singular, vh = np.linalg.svd(weighted, full_matrices=False)
        svd_seconds = perf_counter() - svd_started
        a = u * singular
        q = solve_triangular(lower.T, vh.T, lower=False, check_finite=True).T
        return cls(_readonly(delta), _readonly(covariance), _readonly(lower),
            _readonly(singular, np.float64), _readonly(a), _readonly(q),
            _readonly(vh),
            float(perf_counter() - started), float(svd_seconds))

    def factors(self, rank):
        rank = _rank(rank, len(self.singular_values))
        return RankFactors(self.A_full[:, :rank], self.Q_full[:rank, :])

    def capture(self, rank):
        rank = _rank(rank, len(self.singular_values))
        energy = self.singular_values ** 2
        total = float(np.sum(energy))
        if total == 0:
            raise ValueError("weighted capture is undefined for a zero-energy module")
        return float(np.sum(energy[:rank]) / total)

    def heldout_residual(self, rank, covariance):
        """Evaluation-only residual; supplied covariance never refits the factors."""
        factor = self.factors(rank)
        covariance = _covariance(covariance, self.delta.shape[1])
        lower = cholesky(covariance, lower=True, check_finite=True)
        weighted = self.delta @ lower
        residual = weighted - factor.A @ (factor.Q @ lower)
        denominator = float(np.linalg.norm(weighted) ** 2)
        if denominator == 0:
            raise ValueError("relative residual is undefined for a zero-energy module")
        return float(np.linalg.norm(residual) ** 2 / denominator)

    def diagnostics(self):
        """Construction-only full-rank sentinel, never an inference path."""
        def relative(residual, reference):
            return float(np.linalg.norm(residual) /
                max(float(np.linalg.norm(reference)), np.finfo(float).tiny))
        restored = self.A_full @ self.Q_full
        weighted = self.delta @ self.lower
        return dict(covariance_hermitian_relative_residual=relative(
                self.covariance - self.covariance.conj().T, self.covariance),
            covariance_min_eigenvalue=float(np.linalg.eigvalsh(self.covariance)[0]),
            cholesky_relative_residual=relative(
                self.lower @ self.lower.conj().T - self.covariance, self.covariance),
            triangular_right_solve_relative_residual=relative(
                self.Q_full @ self.lower - self.right_h, self.right_h),
            full_rank_correction_relative_residual=relative(restored - self.delta, self.delta),
            weighted_svd_relative_residual=relative(
                self.A_full @ self.right_h - weighted, weighted),
            construction_seconds=self.construction_seconds, svd_seconds=self.svd_seconds)


def factored_affine_predict(pilot, source_mean_p, source_mean_d, source_filter,
                            coefficients, profile, mobility):
    """Use frozen real coordinates and source means, with two factorized products."""
    source = _matrix(source_filter, "source filter")
    ndata, npilot = source.shape
    pilot = _vector(pilot, npilot, "pilot")
    mean_p = _vector(source_mean_p, npilot, "source pilot mean")
    mean_d = _vector(source_mean_d, ndata, "source data mean")
    raw_coefficients = np.asarray(coefficients)
    if np.iscomplexobj(raw_coefficients):
        raise ValueError("frozen affine coefficients must be real")
    coefficients = np.asarray(raw_coefficients, dtype=np.float64)
    if coefficients.shape != (2,) or not np.isfinite(coefficients).all():
        raise ValueError("two finite real frozen coefficients required")
    if not isinstance(profile, RankFactors) or not isinstance(mobility, RankFactors):
        raise ValueError("profile and mobility must be frozen RankFactors")
    if any(f.A.shape[0] != ndata or f.Q.shape[1] != npilot for f in (profile, mobility)):
        raise ValueError("module factors must match source filter dimensions")
    x = pilot - mean_p
    # Preserve the frozen dense source multiplication/affine evaluation order.
    base = mean_d + x @ source.T
    return base + coefficients[0] * profile.apply(x) + coefficients[1] * mobility.apply(x)


def compression_metrics(source_db, dense_db, convex_db, lowrank_db):
    values = np.asarray([source_db, dense_db, convex_db, lowrank_db], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("finite actual NMSE values required")
    dense_recovery = float(values[0] - values[1])
    if dense_recovery == 0:
        raise ValueError("recovery preservation is undefined when dense recovery is zero")
    lowrank_recovery = float(values[0] - values[3])
    return dict(compression_penalty_db=float(values[3] - values[1]),
        dense_recovery_db=dense_recovery, lowrank_recovery_db=lowrank_recovery,
        recovery_preservation=float(lowrank_recovery / dense_recovery),
        lowrank_affine_vs_convex_gain_db=float(values[2] - values[3]))


def minimum_practical_rank(rows):
    """Search only the task's frozen five-rank grid, without monotonic assumptions."""
    if hasattr(rows, "items"):
        penalties = dict(rows)
    else:
        penalties = {}
        for row in rows:
            rank = row["rank"]
            if rank in penalties:
                raise ValueError("each rank must occur once per condition")
            penalties[rank] = row["compression_penalty_db"]
    if set(penalties) != set(RANKS):
        raise ValueError("all five frozen ranks are required")
    if not np.isfinite(list(penalties.values())).all():
        raise ValueError("finite compression penalties required")
    def first(threshold):
        accepted = [rank for rank in RANKS if penalties[rank] <= threshold]
        return min(accepted) if accepted else None
    return dict(r_025db=first(.25), r_05db=first(.50))


def complexity(rank):
    rank = _rank(rank, 16)
    if rank not in RANKS:
        raise ValueError("only the frozen validation rank grid is allowed")
    source = 3968 * 64
    pilot_cv = 46080
    one_module = rank * (3968 + 64)
    correction = 2 * one_module
    dense_correction = 2 * source
    total = source + pilot_cv + correction
    dense_total = source + pilot_cv + dense_correction
    return dict(rank=rank, A_shape=[3968, rank], Q_shape=[rank, 64],
        single_module_complex_MAC_per_frame=one_module,
        correction_complex_MAC_per_frame=correction,
        source_complex_MAC_per_frame=source,
        pilot_cv_complex_MAC_per_current_window=pilot_cv,
        correction_over_source_ratio=correction / source,
        correction_over_source_percent=100 * correction / source,
        total_arithmetic_proxy=total, total_over_source_ratio=total / source,
        dense_correction_complex_MAC_per_frame=dense_correction,
        dense_affine_total_arithmetic_proxy=dense_total,
        saving_vs_dense_affine_percent=100 * (dense_total - total) / dense_total,
        factor_complex_coefficient_count=correction,
        factor_complex64_bytes=8 * correction, factor_complex128_bytes=16 * correction,
        dense_module_complex_coefficient_count=dense_correction,
        dense_module_complex64_bytes=8 * dense_correction,
        dense_module_complex128_bytes=16 * dense_correction,
        scalar_coefficient_products_in_MAC_proxy=False,
        dense_pilot_coordinate_inference_unchanged=True)


def scientific_gate(primary_rows):
    """Literal pre-registered predicates; weak is independent of strong/moderate."""
    if hasattr(primary_rows, "items"):
        primary_by_rank = primary_rows
    else:
        primary_by_rank = {}
        for row in primary_rows:
            condition = (row["shift"], row["snr_db"])
            rank = row["rank"]
            bucket = primary_by_rank.setdefault(rank, {})
            if condition in bucket:
                raise ValueError("each primary rank/condition must occur once")
            bucket[condition] = row
    if set(primary_by_rank) != set(RANKS):
        raise ValueError("all frozen ranks are required for the primary gate")
    for conditions in primary_by_rank.values():
        if set(conditions) != PRIMARY_CONDITIONS:
            raise ValueError("all four primary five-frame conditions required")
        for value in conditions.values():
            if not np.isfinite([value["compression_penalty_db"],
                                value["lowrank_affine_vs_convex_gain_db"]]).all():
                raise ValueError("finite compression penalties and gains required")
    def passes(rank):
        rows = primary_by_rank[rank]
        return (all(value["compression_penalty_db"] <= .50 for value in rows.values())
            and all(rows[("CDL-B-120", snr)]["lowrank_affine_vs_convex_gain_db"] >= 1.
                    for snr in (10, 20)))
    strong = passes(4)
    moderate = not strong and passes(8)
    weak = any(value["compression_penalty_db"] > 1.
        for value in primary_by_rank[16].values())
    return dict(lowrank_affine_modules_strong=bool(strong),
        lowrank_affine_modules_moderate=bool(moderate),
        lowrank_affine_modules_weak=bool(weak),
        lowrank_affine_modules_partial=bool(not strong and not moderate and not weak))
