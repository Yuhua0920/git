"""MR05 independent compression controls; no file IO, fitting or selection.

All controls use supplied, unchanged real coordinates and source centering.
The existing WeightedModule implements SVD and the ordinary-transpose right
solve. Whole-bank ranks are caller-fixed controls, not a selection API.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from src.lowrank_modules import RankFactors, WeightedModule

RTOL = 1e-10
ATOL = 1e-10
CORRECTION_METHODS = (
    "CORR_ORIGIN_R4", "CORR_I_R4", "CORR_ORIGIN_M2_R4", "CORR_POOL_M2_R4"
)
WHOLE_METHODS = ("WHOLE_POOL_M2_R23", "WHOLE_POOL_M2_R24")
METHODS = ("SOURCE", "POOLED_NATURAL", "DENSE_TW5") + CORRECTION_METHODS + WHOLE_METHODS


def _complex_array(value, name, ndim):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biufc":
        raise ValueError(name + "_NUMERIC_REQUIRED")
    result = np.asarray(raw, dtype=np.complex128)
    if result.ndim != ndim or 0 in result.shape or not np.isfinite(result).all():
        raise ValueError(name + "_FINITE_SHAPE_REQUIRED")
    return result


def _check(error, scale, label):
    threshold = ATOL + RTOL * scale
    if not np.isfinite([error, scale]).all() or error > threshold:
        raise ValueError(label + "_RESIDUAL")
    return float(threshold)


def weighted_factor(matrix, C, rank, module=None):
    """Return an original RankFactors and component-specific numerical audit.

    An optional existing module lets ranks 23/24 share one immutable SVD. Its
    matrix and covariance must match exactly; no approximate rebinding occurs.
    Tail checks scale to the total weighted energy, including full-rank/zero
    matrix cases where relative error divided by the tail is inappropriate.
    """
    matrix = _complex_array(matrix, "MATRIX", 2)
    C = _complex_array(C, "WEIGHT", 2)
    maximum = min(matrix.shape)
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)) or not 1 <= rank <= maximum:
        raise ValueError("RANK_REQUIRED")
    rank = int(rank)
    if C.shape != (matrix.shape[1], matrix.shape[1]):
        raise ValueError("WEIGHT_SHAPE")
    started = perf_counter()
    reused = module is not None
    if reused:
        if not isinstance(module, WeightedModule):
            raise ValueError("WEIGHTED_MODULE_REQUIRED")
        if not np.array_equal(module.delta, matrix) or not np.array_equal(module.covariance, C):
            raise ValueError("MODULE_MATRIX_WEIGHT_BINDING")
    else:
        module = WeightedModule.build(matrix, C)
    factor = module.factors(rank)
    weighted = matrix @ module.lower
    right_reference = module.right_h[:rank]
    right_residual = factor.Q @ module.lower - right_reference
    restored = factor.A @ factor.Q
    residual = weighted - factor.A @ (factor.Q @ module.lower)
    residual_energy = float(np.linalg.norm(residual, "fro") ** 2)
    tail_energy = float(np.sum(module.singular_values[rank:] ** 2))
    weighted_energy = float(np.linalg.norm(weighted, "fro") ** 2)
    tail_error = abs(residual_energy - tail_energy)
    right_error = float(np.linalg.norm(right_residual, "fro"))
    right_scale = float(np.linalg.norm(right_reference, "fro"))
    _check(right_error, right_scale, "RIGHT_SOLVE")
    tail_threshold = _check(tail_error, max(weighted_energy, tail_energy, residual_energy), "WEIGHTED_TAIL")
    full = module.A_full @ module.Q_full
    full_error = float(np.linalg.norm(full - matrix, "fro"))
    _check(full_error, float(np.linalg.norm(matrix, "fro")), "FULL_RANK")
    chol_error = float(np.linalg.norm(module.lower @ module.lower.conj().T - C, "fro"))
    _check(chol_error, float(np.linalg.norm(C, "fro")), "CHOLESKY")
    eigen = np.linalg.eigvalsh(C)
    if eigen[0] <= 0:
        raise ValueError("WEIGHT_NOT_POSITIVE_DEFINITE")
    gap = None if rank == len(module.singular_values) else float(module.singular_values[rank - 1] - module.singular_values[rank])
    maximum_singular = float(module.singular_values[0])
    audit = dict(
        matrix_shape=list(matrix.shape), weight_shape=list(C.shape), rank=rank,
        factor_A_shape=list(factor.A.shape), factor_Q_shape=list(factor.Q.shape),
        dtype=str(factor.A.dtype), hermitian_relative_residual=float(np.linalg.norm(C - C.conj().T, "fro") / np.linalg.norm(C, "fro")),
        minimum_weight_eigenvalue=float(eigen[0]),
        cholesky_absolute_residual=chol_error,
        cholesky_relative_residual=float(chol_error / np.linalg.norm(C, "fro")),
        right_solve_absolute_residual=right_error,
        right_solve_relative_residual=float(right_error / max(right_scale, np.finfo(float).tiny)),
        full_rank_absolute_residual=full_error,
        full_rank_relative_residual=float(full_error / max(float(np.linalg.norm(matrix, "fro")), np.finfo(float).tiny)),
        weighted_residual_energy=residual_energy, weighted_tail_energy=tail_energy,
        weighted_total_energy=weighted_energy, weighted_tail_absolute_error=tail_error,
        weighted_tail_error_over_total=float(tail_error / max(weighted_energy, np.finfo(float).tiny)),
        weighted_tail_acceptance_threshold=tail_threshold,
        truncated_matrix_frobenius_norm=float(np.linalg.norm(restored, "fro")),
        boundary_singular_gap=gap,
        boundary_singular_gap_relative=None if gap is None else float(gap / max(maximum_singular, np.finfo(float).tiny)),
        factorization_reused=reused,
        construction_seconds=0.0 if reused else float(module.construction_seconds),
        original_module_construction_seconds=float(module.construction_seconds),
        original_module_svd_seconds=float(module.svd_seconds),
        audit_and_optional_construction_seconds=float(perf_counter() - started),
        atol=ATOL, rtol=RTOL,
        right_solve="solve_triangular(L.T, Vh_r.T, lower=False).T",
        tail_rule="absolute_error <= atol + rtol * max(total_weighted_energy, tail_energy, residual_energy)",
    )
    return factor, audit


def _coordinates(value):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
        raise ValueError("REAL_COORDINATES_REQUIRED")
    result = np.asarray(raw, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError("TWO_FINITE_REAL_COORDINATES_REQUIRED")
    return result


def _factor_tuple(bank, method, count, ndata, npilot):
    factors = tuple(bank[method])
    if len(factors) != count or any(not isinstance(f, RankFactors) for f in factors):
        raise ValueError("FACTOR_BANK_TYPE_OR_COUNT")
    if any(f.A.shape[0] != ndata or f.Q.shape[1] != npilot for f in factors):
        raise ValueError("FACTOR_BANK_SHAPE")
    return factors


def deployed(method, u, coeff, op, bank, snr):
    """Apply one declared control to a current source-centered observation.

    The input u is already p - source_mean_p. No covariance/kernel/coordinate
    update occurs here. Only POOLED_NATURAL changes to its own prepared mean.
    Dimensions derive from the frozen source operator (3968 by 64 in MR05),
    permitting small mathematical unit tests of exactly the same interface.
    """
    if method not in METHODS:
        raise ValueError("UNKNOWN_CONTROL_METHOD")
    suffix = str(snr)
    Ws = _complex_array(op["Wsource_" + suffix], "SOURCE_OPERATOR", 2)
    ndata, npilot = Ws.shape
    u = _complex_array(u, "CENTERED_PILOT", 1)
    md = _complex_array(op["source_mean_d"], "SOURCE_DATA_MEAN", 1)
    if u.shape != (npilot,) or md.shape != (ndata,):
        raise ValueError("SOURCE_INPUT_SHAPE")
    if method == "SOURCE":
        result = md + u @ Ws.T
    elif method == "POOLED_NATURAL":
        mp_s = _complex_array(op["source_mean_p"], "SOURCE_PILOT_MEAN", 1)
        mp_pool = _complex_array(op["pool_mean_p"], "POOLED_PILOT_MEAN", 1)
        md_pool = _complex_array(op["pool_mean_d"], "POOLED_DATA_MEAN", 1)
        Wpool = _complex_array(op["Wpool_" + suffix], "POOLED_OPERATOR", 2)
        if mp_s.shape != (npilot,) or mp_pool.shape != (npilot,) or md_pool.shape != (ndata,) or Wpool.shape != Ws.shape:
            raise ValueError("POOLED_INPUT_SHAPE")
        result = md_pool + (u + mp_s - mp_pool) @ Wpool.T
    elif method == "DENSE_TW5":
        c = _coordinates(coeff)
        BP = _complex_array(op["DeltaWP_" + suffix], "PROFILE_CORRECTION", 2)
        BM = _complex_array(op["DeltaWM_" + suffix], "MOBILITY_CORRECTION", 2)
        if BP.shape != Ws.shape or BM.shape != Ws.shape:
            raise ValueError("DENSE_CORRECTION_SHAPE")
        base = md + u @ Ws.T
        result = base + c[0] * (u @ BP.T) + c[1] * (u @ BM.T)
    elif method in CORRECTION_METHODS:
        c = _coordinates(coeff)
        P, M = _factor_tuple(bank, method, 2, ndata, npilot)
        base = md + u @ Ws.T
        result = base + c[0] * P.apply(u) + c[1] * M.apply(u)
    else:
        c = _coordinates(coeff)
        S, P, M = _factor_tuple(bank, method, 3, ndata, npilot)
        result = md + (1.0 - c[0] - c[1]) * S.apply(u) + c[0] * P.apply(u) + c[1] * M.apply(u)
    if result.shape != (ndata,) or not np.isfinite(result).all():
        raise ValueError("NONFINITE_CONTROL_OUTPUT")
    return result


def prevalidated_deployment(method, op, bank, snr):
    """Certify deployment arrays once, return a pure-math timing kernel.

    Returned signature is kernel(u, coeff). The benchmark caller must certify
    finite complex128 u and finite real two-coordinate coeff outside timing,
    keep all bound assets unchanged, and check outputs outside timing. Source
    and Pooled ignore coeff. Construction/validation is excluded from the
    deployment endpoint; no per-call full-matrix scan or dictionary lookup is
    performed. Scientific inference can retain the validated deployed() API.
    """
    if method not in METHODS:
        raise ValueError("UNKNOWN_CONTROL_METHOD")
    suffix = str(snr)
    Ws = _complex_array(op["Wsource_" + suffix], "SOURCE_OPERATOR", 2)
    ndata, npilot = Ws.shape
    md = _complex_array(op["source_mean_d"], "SOURCE_DATA_MEAN", 1)
    if md.shape != (ndata,):
        raise ValueError("SOURCE_INPUT_SHAPE")
    if method == "SOURCE":
        def kernel(u, coeff):
            return md + u @ Ws.T
        return kernel
    if method == "POOLED_NATURAL":
        mp_s = _complex_array(op["source_mean_p"], "SOURCE_PILOT_MEAN", 1)
        mp_pool = _complex_array(op["pool_mean_p"], "POOLED_PILOT_MEAN", 1)
        md_pool = _complex_array(op["pool_mean_d"], "POOLED_DATA_MEAN", 1)
        Wpool = _complex_array(op["Wpool_" + suffix], "POOLED_OPERATOR", 2)
        if mp_s.shape != (npilot,) or mp_pool.shape != (npilot,) or md_pool.shape != (ndata,) or Wpool.shape != Ws.shape:
            raise ValueError("POOLED_INPUT_SHAPE")
        def kernel(u, coeff):
            return md_pool + (u + mp_s - mp_pool) @ Wpool.T
        return kernel
    if method == "DENSE_TW5":
        BP = _complex_array(op["DeltaWP_" + suffix], "PROFILE_CORRECTION", 2)
        BM = _complex_array(op["DeltaWM_" + suffix], "MOBILITY_CORRECTION", 2)
        if BP.shape != Ws.shape or BM.shape != Ws.shape:
            raise ValueError("DENSE_CORRECTION_SHAPE")
        def kernel(u, coeff):
            base = md + u @ Ws.T
            return base + coeff[0] * (u @ BP.T) + coeff[1] * (u @ BM.T)
        return kernel
    if method in CORRECTION_METHODS:
        P, M = _factor_tuple(bank, method, 2, ndata, npilot)
        for f in (P, M):
            if not np.isfinite(f.A).all() or not np.isfinite(f.Q).all():
                raise ValueError("NONFINITE_CONTROL_FACTOR")
        AP, QP, AM, QM = P.A, P.Q, M.A, M.Q
        def kernel(u, coeff):
            base = md + u @ Ws.T
            return base + coeff[0] * (AP @ (QP @ u)) + coeff[1] * (AM @ (QM @ u))
        return kernel
    S, P, M = _factor_tuple(bank, method, 3, ndata, npilot)
    for f in (S, P, M):
        if not np.isfinite(f.A).all() or not np.isfinite(f.Q).all():
            raise ValueError("NONFINITE_CONTROL_FACTOR")
    AS, QS, AP, QP, AM, QM = S.A, S.Q, P.A, P.Q, M.A, M.Q
    def kernel(u, coeff):
        return md + (1.0 - coeff[0] - coeff[1]) * (AS @ (QS @ u)) + coeff[0] * (AP @ (QP @ u)) + coeff[1] * (AM @ (QM @ u))
    return kernel
