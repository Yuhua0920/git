"""Independent proper-complex K=1 PPCA controls; no IO or scientific generator.

Rows are ordinary transposes of complex column observations. The spectral
solution optimizes the feasible covariance family nu >= residual_floor; it
does not reproduce the frozen author's clip-each-dimension-then-average EM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, subspace_angles

RTOL = 1e-10
ATOL = 1e-10
EIGEN_RTOL = 1e-9
RESIDUAL_FLOOR = 1e-6  # clean complex-channel residual variance; not sigma^2


def _complex(value, label, ndim):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biufc":
        raise ValueError(label + "_NUMERIC_REQUIRED")
    a = np.asarray(raw, dtype=np.complex128)
    if a.ndim != ndim or 0 in a.shape or not np.isfinite(a).all():
        raise ValueError(label + "_FINITE_SHAPE_REQUIRED")
    return a


def _positive(value, label):
    if np.iscomplexobj(value) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(label + "_POSITIVE_REAL_REQUIRED")
    return float(value)


def _readonly(a):
    a = np.array(a, copy=True)
    a.flags.writeable = False
    return a


def _check(error, scale, label, rtol=RTOL):
    if error > ATOL + rtol * scale:
        raise ValueError(label + "_NUMERICAL_RESIDUAL")


def _nonnegative(value, scale, label):
    if value < -ATOL - RTOL * scale:
        raise ValueError(label + "_NEGATIVE_BEYOND_ROUNDOFF")
    return max(float(value), 0.0), bool(value < 0)


def training_moments(rows, block_size=256):
    """Two-pass blocked free sample mean and centered population covariance.

    No N-by-N Gram is created; ndarray/memmap inputs remain unmodified. The
    input is read in bounded blocks, so finite checks never copy the full P.
    """
    if not isinstance(block_size, (int, np.integer)) or isinstance(block_size, (bool, np.bool_)) or block_size < 1:
        raise ValueError("BLOCK_SIZE_REQUIRED")
    if getattr(rows, "ndim", None) != 2 or 0 in rows.shape or rows.dtype.kind not in "biufc":
        raise ValueError("TRAINING_ROWS_REQUIRED")
    started = perf_counter()
    N, d = rows.shape
    mean = np.zeros(d, np.complex128)
    for start in range(0, N, block_size):
        block = _complex(rows[start:start + block_size], "TRAINING_BLOCK", 2)
        mean += block.sum(axis=0, dtype=np.complex128)
    mean /= N
    S = np.zeros((d, d), np.complex128)
    centered_sum = np.zeros(d, np.complex128)
    for start in range(0, N, block_size):
        Z = _complex(rows[start:start + block_size], "TRAINING_BLOCK", 2) - mean
        S += Z.T @ Z.conj()
        centered_sum += Z.sum(axis=0)
    S /= N
    hermitian = float(np.linalg.norm(S - S.conj().T, "fro"))
    scale = float(np.linalg.norm(S, "fro"))
    _check(hermitian, scale, "TRAINING_HERMITIAN")
    trace = float(math.fsum(float(v) for v in S.diagonal().real))
    audit = dict(samples=N, dimension=d, dtype=str(S.dtype), denominator=N,
                 row_convention="ordinary transpose of column h_i; S=Z.T@Z.conj()/N",
                 block_size=int(block_size), free_mean=True,
                 covariance_Hermitian_absolute_residual=hermitian,
                 covariance_Hermitian_relative_residual=hermitian / max(scale, np.finfo(float).tiny),
                 trace=trace, centered_mean_norm=float(np.linalg.norm(centered_sum / N)),
                 seconds=float(perf_counter() - started), atol=ATOL, rtol=RTOL,
                 training_independence="frame marginal weighting; N is not independent trajectory count")
    return mean, S, audit


@dataclass(frozen=True)
class Spectrum:
    eigenvalues: np.ndarray  # descending top eigenvalues
    eigenvectors: np.ndarray
    trace: float
    dimension: int
    audit: dict


def leading_eigenpairs(S, max_rank=64):
    """Exact top eigenpairs shared by all ranks, plus one minimum-eigenvalue check.

    The additional bottom-one subset certifies that an externally supplied S
    is PSD to stated precision. Both checks happen once, not per rank/SNR.
    """
    S = _complex(S, "COVARIANCE", 2)
    d = S.shape[0]
    if S.shape != (d, d) or isinstance(max_rank, (bool, np.bool_)) or not isinstance(max_rank, (int, np.integer)) or not 1 <= max_rank < d:
        raise ValueError("SPECTRUM_SHAPE_OR_RANK")
    norm = float(np.linalg.norm(S, "fro"))
    herror = float(np.linalg.norm(S - S.conj().T, "fro"))
    _check(herror, norm, "SPECTRUM_HERMITIAN")
    started = perf_counter()
    values, vectors = eigh(S, subset_by_index=(d - int(max_rank), d - 1), driver="evr", check_finite=False)
    top_seconds = perf_counter() - started
    values, vectors = values[::-1].copy(), vectors[:, ::-1].copy()
    minimum_started = perf_counter()
    minimum = float(eigh(S, subset_by_index=(0, 0), eigvals_only=True, driver="evr", check_finite=False)[0])
    minimum_seconds = perf_counter() - minimum_started
    if minimum < -ATOL - RTOL * norm:
        raise ValueError("COVARIANCE_NOT_PSD")
    residual = float(np.linalg.norm(S @ vectors - vectors * values, "fro"))
    orthogonal = float(np.linalg.norm(vectors.conj().T @ vectors - np.eye(max_rank), "fro"))
    _check(residual, norm, "EIGENPAIR", EIGEN_RTOL)
    _check(orthogonal, math.sqrt(max_rank), "EIGENVECTOR_ORTHOGONAL", EIGEN_RTOL)
    if np.any(np.diff(values) > ATOL + RTOL * max(abs(values[0]), 1.0)):
        raise ValueError("EIGENVALUE_ORDER")
    trace = float(math.fsum(float(v) for v in S.diagonal().real))
    audit = dict(driver="scipy.linalg.eigh evr exact Hermitian subsets", max_rank=int(max_rank),
                 top_subset=[d - int(max_rank), d - 1], minimum_subset=[0, 0],
                 minimum_eigenvalue=minimum, Hermitian_absolute_residual=herror,
                 eigenpair_absolute_residual=residual,
                 eigenpair_relative_residual=residual / max(norm, np.finfo(float).tiny),
                 orthogonality_frobenius_residual=orthogonal,
                 top_eigh_seconds=float(top_seconds), minimum_eigh_seconds=float(minimum_seconds),
                 total_eigh_seconds=float(top_seconds + minimum_seconds),
                 trace=trace, atol=ATOL, rtol=RTOL, eigen_rtol=EIGEN_RTOL)
    return Spectrum(_readonly(values), _readonly(vectors), trace, d, audit)


@dataclass(frozen=True)
class PPCAModel:
    mean: np.ndarray
    loading: np.ndarray
    nu: float
    audit: dict = field(default_factory=dict)

    def __post_init__(self):
        mean = _complex(self.mean, "MODEL_MEAN", 1)
        F = _complex(self.loading, "MODEL_LOADING", 2)
        if F.shape[0] != mean.size or F.shape[1] >= mean.size:
            raise ValueError("MODEL_DIMENSIONS")
        object.__setattr__(self, "mean", _readonly(mean))
        object.__setattr__(self, "loading", _readonly(F))
        object.__setattr__(self, "nu", _positive(self.nu, "RESIDUAL_VARIANCE"))

    @property
    def rank(self):
        return self.loading.shape[1]

    @property
    def dimension(self):
        return self.mean.size


def direct_ppca(mean, spectrum, rank, residual_floor=RESIDUAL_FLOOR):
    """Spectral MLE in the independent feasible family nu >= a known floor."""
    mean = _complex(mean, "SAMPLE_MEAN", 1)
    floor = _positive(residual_floor, "RESIDUAL_FLOOR")
    if not isinstance(spectrum, Spectrum) or mean.size != spectrum.dimension:
        raise ValueError("SPECTRUM_MEAN_BINDING")
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)) or not 1 <= rank <= len(spectrum.eigenvalues) or rank >= spectrum.dimension:
        raise ValueError("PPCA_RANK")
    rank = int(rank)
    values = spectrum.eigenvalues[:rank]
    tail_raw = float(math.fsum([spectrum.trace] + [-float(v) for v in values]))
    tail, rounded = _nonnegative(tail_raw, abs(spectrum.trace), "SPECTRAL_TAIL")
    raw_nu = tail_raw / (spectrum.dimension - rank)
    nu = max(tail / (spectrum.dimension - rank), floor)
    # This max is part of the constrained optimizer, not a PSD-error fallback.
    strengths = np.maximum(values - nu, 0.0)
    F = spectrum.eigenvectors[:, :rank] * np.sqrt(strengths)
    audit = dict(declared_rank=rank, effective_loading_rank=int(np.count_nonzero(strengths > 0)),
                 raw_tail_trace=tail_raw, accepted_tail_trace=tail,
                 raw_residual_variance=raw_nu, residual_variance=nu,
                 residual_floor=floor, floor_active=bool(raw_nu < floor),
                 roundoff_negative_tail_projected=rounded,
                 loading_strengths=strengths.tolist(), selected_eigenvalues=values.tolist(),
                 floor_scope="independent MLE over nu>=floor; not author clip-per-coordinate EM update",
                 standard_nondegenerate=bool(raw_nu > 0 and np.all(values > raw_nu)),
                 atol=ATOL, rtol=RTOL)
    return PPCAModel(mean, F, nu, audit)


def from_author_arrays(means, lambdas, psis, amps):
    """Parse actual K=1 arrays; never synthesizes absent original parameters."""
    means = _complex(means, "AUTHOR_MEANS", 2)
    F = _complex(lambdas, "AUTHOR_LOADINGS", 3)
    psi = np.asarray(psis)
    weights = np.asarray(amps)
    if means.shape[0] != 1 or F.shape[:2] != means.shape or psi.shape != means.shape or weights.shape != (1,):
        raise ValueError("AUTHOR_K1_SHAPES")
    if np.iscomplexobj(psi) or np.iscomplexobj(weights) or not np.isfinite(psi).all() or not np.isfinite(weights).all():
        raise ValueError("AUTHOR_REAL_VARIANCES_WEIGHTS")
    _check(abs(float(weights[0]) - 1.0), 1.0, "AUTHOR_K1_WEIGHT")
    nu = _positive(float(psi[0, 0]), "AUTHOR_RESIDUAL_VARIANCE")
    _check(float(np.max(np.abs(psi - nu))), nu, "AUTHOR_ISOTROPIC_VARIANCE")
    if nu < RESIDUAL_FLOOR - ATOL:
        raise ValueError("AUTHOR_RESIDUAL_BELOW_FROZEN_FLOOR")
    return PPCAModel(means[0], F[0], nu, dict(origin="actual supplied author arrays", variance_field="psis"))


def _indices(model, pilot_positions, data_positions):
    ids = []
    for raw, label in ((pilot_positions, "PILOT"), (data_positions, "DATA")):
        a = np.asarray(raw)
        if a.ndim != 1 or len(a) == 0 or a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a >= model.dimension) or len(np.unique(a)) != len(a):
            raise ValueError(label + "_INDICES")
        ids.append(a.astype(np.int64, copy=False))
    if np.intersect1d(*ids).size:
        raise ValueError("PILOT_DATA_MUST_BE_DISJOINT")
    return ids


@dataclass(frozen=True)
class ConditionalOperator:
    W: np.ndarray
    mean_p: np.ndarray
    mean_d: np.ndarray
    sigma2: float
    audit: dict

    def estimate(self, pilots):
        p = _complex(pilots, "PILOTS", np.asarray(pilots).ndim)
        if p.ndim not in (1, 2) or p.shape[-1] != self.mean_p.size:
            raise ValueError("PILOT_SHAPE")
        return self.mean_d + (p - self.mean_p) @ self.W.T


def conditional_operator(model, pilot_positions, data_positions, sigma2):
    """Cache the natural-mean conditional Gaussian map; noise is added once."""
    if not isinstance(model, PPCAModel):
        raise ValueError("PPCA_MODEL_REQUIRED")
    sigma2 = _positive(sigma2, "OBSERVATION_NOISE_VARIANCE")
    pidx, didx = _indices(model, pilot_positions, data_positions)
    Fp, Fd = model.loading[pidx], model.loading[didx]
    Cpp = Fp @ Fp.conj().T + (model.nu + sigma2) * np.eye(len(pidx))
    Cdp = Fd @ Fp.conj().T
    started = perf_counter()
    # Solve Cpp W^H = Cdp^H and take the adjoint back: W=Cdp Cpp^-1.
    factor = cho_factor(Cpp, lower=True, check_finite=True)
    W = cho_solve(factor, Cdp.conj().T, check_finite=True).conj().T
    solve_residual = float(np.linalg.norm(W @ Cpp - Cdp, "fro"))
    _check(solve_residual, float(np.linalg.norm(Cdp, "fro")), "CONDITIONAL_SOLVE")
    singular = np.linalg.svd(W, compute_uv=False)
    cutoff = (1e-10 * singular[0]) if len(singular) else 0.0
    audit = dict(noise_added_once=True, prior_residual_variance=model.nu,
                 receiver_noise_variance=sigma2, observed_isotropic_variance=model.nu + sigma2,
                 W_shape=list(W.shape), solve_absolute_residual=solve_residual,
                 solve_relative_residual=solve_residual / max(float(np.linalg.norm(Cdp, "fro")), np.finfo(float).tiny),
                 numerical_operator_rank=int(np.count_nonzero(singular > cutoff)),
                 operator_singular_values=singular.tolist(), operator_rank_rtol=1e-10,
                 seconds=float(perf_counter() - started), natural_model_means=True)
    return ConditionalOperator(_readonly(W), _readonly(model.mean[pidx]), _readonly(model.mean[didx]), sigma2, audit)


def training_logscore(model, empirical_mean, S):
    """Average frame marginal proper-complex score, stable orthogonal formula.

    Thin loading SVD gives C's non-isotropic basis. Residual energy is evaluated
    separately from top-direction energies, avoiding subtracting two large
    Woodbury inverse terms. Mean mismatch is retained explicitly.
    """
    mu = _complex(empirical_mean, "EMPIRICAL_MEAN", 1)
    S = _complex(S, "EMPIRICAL_COVARIANCE", 2)
    d = model.dimension
    if mu.shape != (d,) or S.shape != (d, d):
        raise ValueError("SCORE_MOMENT_SHAPE")
    _check(float(np.linalg.norm(S - S.conj().T, "fro")), float(np.linalg.norm(S, "fro")), "SCORE_HERMITIAN")
    U, strengths_sqrt, _ = np.linalg.svd(model.loading, full_matrices=False)
    theta = strengths_sqrt ** 2
    projections = np.real(np.sum(U.conj() * (S @ U), axis=0))
    trace = float(math.fsum(float(v) for v in S.diagonal().real))
    if np.any(projections < -ATOL - RTOL * abs(trace)):
        raise ValueError("SCORE_NEGATIVE_PROJECTED_COVARIANCE")
    covariance_tail_raw = float(math.fsum([trace] + [-float(v) for v in projections]))
    covariance_tail, rounded_cov = _nonnegative(covariance_tail_raw, abs(trace), "SCORE_COVARIANCE_TAIL")
    delta = mu - model.mean
    projected_delta = U.conj().T @ delta
    delta_energy = float(np.vdot(delta, delta).real)
    delta_top = np.abs(projected_delta) ** 2
    mean_tail_raw = float(math.fsum([delta_energy] + [-float(v) for v in delta_top]))
    mean_tail, rounded_mean = _nonnegative(mean_tail_raw, delta_energy, "SCORE_MEAN_TAIL")
    covariance_quadratic = covariance_tail / model.nu + float(np.sum(projections / (model.nu + theta)))
    mean_quadratic = mean_tail / model.nu + float(np.sum(delta_top / (model.nu + theta)))
    logdet = (d - len(theta)) * math.log(model.nu) + float(np.sum(np.log(model.nu + theta)))
    score = -d * math.log(math.pi) - logdet - covariance_quadratic - mean_quadratic
    if not np.isfinite(score):
        raise ValueError("NONFINITE_TRAINING_LOGSCORE")
    return dict(mean_frame_logscore=float(score), mean_complex_dimension_logscore=float(score / d),
                logdet=float(logdet), covariance_quadratic=float(covariance_quadratic),
                mean_mismatch_quadratic=float(mean_quadratic), mean_mismatch_norm=math.sqrt(delta_energy),
                raw_covariance_tail=covariance_tail_raw, raw_mean_tail=mean_tail_raw,
                roundoff_tail_projection=bool(rounded_cov or rounded_mean),
                formula="-d log(pi)-logdet(C)-tr(C^-1 S)-delta^H C^-1 delta",
                independence_scope="mean frame marginal log score, not ten-frame joint likelihood")


def blocked_residual_energy(rows, mean, eigenvectors, block_size=256):
    """Independent row-space residual norm to audit spectral tail subtraction."""
    mu = _complex(mean, "RESIDUAL_MEAN", 1)
    U = _complex(eigenvectors, "RESIDUAL_BASIS", 2)
    if rows.ndim != 2 or rows.shape[1] != mu.size or U.shape[0] != mu.size:
        raise ValueError("RESIDUAL_DIMENSIONS")
    _check(float(np.linalg.norm(U.conj().T @ U - np.eye(U.shape[1]), "fro")), math.sqrt(U.shape[1]), "RESIDUAL_BASIS_ORTHOGONAL", EIGEN_RTOL)
    terms = []
    for start in range(0, len(rows), block_size):
        Z = _complex(rows[start:start + block_size], "RESIDUAL_ROWS", 2) - mu
        residual = Z - (Z @ U.conj()) @ U.T
        terms.append(float(np.vdot(residual, residual).real))
    return float(math.fsum(terms) / len(rows))


def model_comparison(first, second, pilot_positions, data_positions, sigma2):
    """Low-rank covariance distance; never constructs a d-by-d inverse."""
    if first.dimension != second.dimension:
        raise ValueError("MODEL_COMPARISON_DIMENSIONS")
    F, G = first.loading, second.loading
    ff = float(np.linalg.norm(F.conj().T @ F, "fro") ** 2)
    gg = float(np.linalg.norm(G.conj().T @ G, "fro") ** 2)
    fg = float(np.linalg.norm(F.conj().T @ G, "fro") ** 2)
    tf, tg = float(np.vdot(F, F).real), float(np.vdot(G, G).real)
    dv, d = first.nu - second.nu, first.dimension
    terms = [ff, gg, -2 * fg, 2 * dv * (tf - tg), d * dv ** 2]
    # A common thin QR basis avoids square-root amplification of cancellation
    # in the trace-norm identity when the models differ only by unitary phase.
    Q, R = np.linalg.qr(np.concatenate((F, G), axis=1), mode="reduced")
    RF, RG = R[:, :F.shape[1]], R[:, F.shape[1]:]
    reduced_difference = RF @ RF.conj().T - RG @ RG.conj().T + dv * np.eye(Q.shape[1])
    distance2 = float(np.linalg.norm(reduced_difference, "fro") ** 2 + (d - Q.shape[1]) * dv ** 2)
    raw_identity_distance2 = float(math.fsum(terms))
    _check(abs(raw_identity_distance2 - distance2), sum(abs(v) for v in terms), "COVARIANCE_DISTANCE_IDENTITY")
    norm_first2 = ff + 2 * first.nu * tf + d * first.nu ** 2
    norm_second2 = gg + 2 * second.nu * tg + d * second.nu ** 2
    Ua, sa, _ = np.linalg.svd(F, full_matrices=False)
    Ub, sb, _ = np.linalg.svd(G, full_matrices=False)
    Ua = Ua[:, sa > 1e-10 * (sa[0] if len(sa) else 0)]
    Ub = Ub[:, sb > 1e-10 * (sb[0] if len(sb) else 0)]
    angles = subspace_angles(Ua, Ub) if Ua.shape[1] and Ub.shape[1] else np.array([], dtype=float)
    pidx, _ = _indices(first, pilot_positions, data_positions)
    Ca = F[pidx] @ F[pidx].conj().T + first.nu * np.eye(len(pidx))
    Cb = G[pidx] @ G[pidx].conj().T + second.nu * np.eye(len(pidx))
    wa = conditional_operator(first, pilot_positions, data_positions, sigma2)
    wb = conditional_operator(second, pilot_positions, data_positions, sigma2)
    mdiff = float(np.linalg.norm(first.mean - second.mean))
    wdiff = float(np.linalg.norm(wa.W - wb.W, "fro"))
    return dict(mean_absolute_difference=mdiff,
                covariance_frobenius_difference=math.sqrt(distance2),
                covariance_relative_difference=math.sqrt(distance2) / max(math.sqrt(norm_first2), math.sqrt(norm_second2), np.finfo(float).tiny),
                covariance_distance_raw_trace_identity=raw_identity_distance2,
                covariance_distance_stable_squared=distance2,
                effective_subspace_ranks=[Ua.shape[1], Ub.shape[1]], principal_angles_radians=angles.tolist(),
                pilot_covariance_absolute_difference=float(np.linalg.norm(Ca - Cb, "fro")),
                pilot_covariance_relative_difference=float(np.linalg.norm(Ca - Cb, "fro") / max(np.linalg.norm(Ca, "fro"), np.linalg.norm(Cb, "fro"), np.finfo(float).tiny)),
                conditional_operator_absolute_difference=wdiff,
                conditional_operator_relative_difference=wdiff / max(np.linalg.norm(wa.W, "fro"), np.linalg.norm(wb.W, "fro"), np.finfo(float).tiny),
                sigma2=float(sigma2))
