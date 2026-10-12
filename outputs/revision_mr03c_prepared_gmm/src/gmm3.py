"""Prepared-origin proper-complex Gaussian posterior mixture.

Only the received current pilot p enters predict(). Origin means and centered
covariances are supplied, not estimated here. No labels, truth, source centering,
cross-pilot fitting, covariance regularization or explicit inverse is used.

Column convention: R_dp = E[(h_d-mu_d)(h_p-mu_p)^H]. A batch contains ordinary
row transposes of these complex columns, so output rows use delta @ W.T.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular, eigvalsh
from scipy.special import logsumexp, xlogy

ATOL = 1e-10
RTOL = 1e-10
POSTERIOR_ATOL = 1e-12


def _complex(value, label, ndim):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biufc":
        raise ValueError(label + "_NUMERIC_REQUIRED")
    array = np.asarray(raw, dtype=np.complex128)
    if array.ndim != ndim or 0 in array.shape or not np.isfinite(array).all():
        raise ValueError(label + "_FINITE_SHAPE_REQUIRED")
    return array


def _positive_real(value, label):
    if (not np.isscalar(value) or isinstance(value, (bool, np.bool_))
            or np.iscomplexobj(value)):
        raise ValueError(label + "_POSITIVE_REAL_REQUIRED")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(label + "_POSITIVE_REAL_REQUIRED") from None
    if not np.isfinite(result) or result <= 0:
        raise ValueError(label + "_POSITIVE_REAL_REQUIRED")
    return result


def _readonly(array):
    result = np.array(array, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class GaussianComponent:
    name: str
    mu_p: np.ndarray
    mu_d: np.ndarray
    C: np.ndarray
    chol_lower: np.ndarray
    W: np.ndarray
    logdet: float
    audit: dict


@dataclass(frozen=True)
class MixturePrediction:
    estimate: np.ndarray
    posterior_weights: np.ndarray
    posterior_entropy_nats: np.ndarray
    max_posterior_weight: np.ndarray
    log_scores: np.ndarray
    quadratic_forms: np.ndarray
    component_estimates: np.ndarray | None = None


@dataclass(frozen=True)
class PreparedGaussianMixture:
    components: tuple[GaussianComponent, ...]
    priors: np.ndarray
    log_priors: np.ndarray
    sigma2: float
    setup_audit: dict

    @property
    def n_p(self):
        return self.components[0].mu_p.size

    @property
    def n_d(self):
        return self.components[0].mu_d.size

    @property
    def n_components(self):
        return len(self.components)

    def predict(self, pilots, *, return_components=False):
        """Current-frame posterior conditional mean; a batch is only vectorization.

        A 1D p gives estimate (n_d,), weights (Q,), scalar entropy and maximum.
        A 2D batch of ordinary row transposes gives estimate (N,n_d), weights
        (N,Q) and entropy/max (N,). No temporal operation occurs across rows.
        Scores exclude the shared proper-complex constant -n_p log(pi).
        """
        raw = np.asarray(pilots)
        scalar = raw.ndim == 1
        p = _complex(raw, "PILOTS", 1 if scalar else 2)
        if scalar:
            p = p[None, :]
        if p.shape[1] != self.n_p:
            raise ValueError("PILOT_DIMENSION_MISMATCH")
        count = p.shape[0]
        Q = self.n_components
        scores = np.empty((count, Q), np.float64)
        quadratic = np.empty_like(scores)
        outputs = []
        for q, component in enumerate(self.components):
            delta = p - component.mu_p
            whitened = solve_triangular(component.chol_lower, delta.T,
                                        lower=True, check_finite=False)
            quad = np.sum(whitened.real ** 2 + whitened.imag ** 2,
                          axis=0, dtype=np.float64)
            if not np.isfinite(quad).all() or np.any(quad < 0):
                raise ValueError("QUADRATIC_FORM_NONFINITE_OR_NEGATIVE")
            quadratic[:, q] = quad
            scores[:, q] = self.log_priors[q] - component.logdet - quad
            current = component.mu_d + delta @ component.W.T
            if not np.isfinite(current).all():
                raise ValueError("COMPONENT_PREDICTION_NONFINITE")
            outputs.append(current)
        # Center BEFORE logsumexp, including when scores share a huge offset.
        # Zero synthetic priors have log=-inf and produce posterior zero.
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        weights = np.exp(shifted - logsumexp(shifted, axis=1, keepdims=True))
        if (not np.isfinite(weights).all() or np.any(weights < 0)
                or not np.allclose(weights.sum(axis=1), 1.0, rtol=0,
                                   atol=POSTERIOR_ATOL)):
            raise ValueError("POSTERIOR_WEIGHTS_INVALID")
        estimate = np.zeros((count, self.n_d), np.complex128)
        for q, current in enumerate(outputs):
            estimate += weights[:, q, None] * current
        if not np.isfinite(estimate).all():
            raise ValueError("MIXTURE_PREDICTION_NONFINITE")
        entropy = -np.sum(xlogy(weights, weights), axis=1)
        maximum = np.max(weights, axis=1)
        if (not np.isfinite(entropy).all() or np.any(entropy < -POSTERIOR_ATOL)
                or np.any(entropy > np.log(Q) + POSTERIOR_ATOL)):
            raise ValueError("POSTERIOR_ENTROPY_INVALID")
        all_components = np.stack(outputs, axis=1) if return_components else None
        if scalar:
            return MixturePrediction(estimate[0], weights[0], entropy[0],
                                     maximum[0], scores[0], quadratic[0],
                                     None if all_components is None else all_components[0])
        return MixturePrediction(estimate, weights, entropy, maximum, scores,
                                 quadratic, all_components)


def prepare_gaussian_mixture(origins: Sequence[Mapping], sigma2, priors=None):
    """Instantiate noise-dependent maps from supplied clean statistics.

    Each origin has keys name, mu_p, mu_d, R_pp and R_dp. Default priors are
    exactly 1/Q; the production Q=3 pipeline fixes 1/3 for each prepared origin.
    Nonuniform/zero priors exist solely to test mathematical limiting cases.
    """
    sigma2 = _positive_real(sigma2, "SIGMA2")
    if not isinstance(origins, (list, tuple)) or len(origins) == 0:
        raise ValueError("ORIGIN_SEQUENCE_REQUIRED")
    Q = len(origins)
    if priors is None:
        prior = np.full(Q, 1.0 / Q, dtype=np.float64)
    else:
        raw = np.asarray(priors)
        if raw.dtype.kind not in "biuf" or raw.ndim != 1 or raw.shape != (Q,):
            raise ValueError("PRIOR_REAL_VECTOR_REQUIRED")
        prior = np.array(raw, dtype=np.float64, copy=True)
        if (not np.isfinite(prior).all() or np.any(prior < 0)
                or not np.isclose(prior.sum(), 1, rtol=0, atol=POSTERIOR_ATOL)):
            raise ValueError("PRIOR_PROBABILITY_VECTOR_REQUIRED")
    log_prior = np.full(Q, -np.inf, np.float64)
    np.log(prior, out=log_prior, where=prior > 0)
    started = perf_counter()
    components = []
    expected_shape = None
    for q, origin in enumerate(origins):
        if not isinstance(origin, Mapping):
            raise ValueError("ORIGIN_MAPPING_REQUIRED")
        label = "ORIGIN_" + str(q)
        try:
            mu_p = _complex(origin["mu_p"], label + "_MU_P", 1)
            mu_d = _complex(origin["mu_d"], label + "_MU_D", 1)
            rpp = _complex(origin["R_pp"], label + "_R_PP", 2)
            rdp = _complex(origin["R_dp"], label + "_R_DP", 2)
        except KeyError as exc:
            raise ValueError("ORIGIN_STATISTIC_MISSING_" + str(exc.args[0])) from None
        npilot, ndata = mu_p.size, mu_d.size
        if rpp.shape != (npilot, npilot) or rdp.shape != (ndata, npilot):
            raise ValueError("ORIGIN_STATISTIC_DIMENSION_MISMATCH")
        if expected_shape is None:
            expected_shape = (npilot, ndata)
        elif expected_shape != (npilot, ndata):
            raise ValueError("ORIGIN_COMMON_DIMENSION_REQUIRED")
        C = rpp + sigma2 * np.eye(npilot, dtype=np.complex128)
        normC = float(np.linalg.norm(C, "fro"))
        hermitian_error = float(np.linalg.norm(C - C.conj().T, "fro"))
        if hermitian_error > ATOL + RTOL * normC:
            raise ValueError("OBSERVATION_COVARIANCE_NOT_HERMITIAN")
        chol_started = perf_counter()
        try:
            fact, lower = cho_factor(C, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            raise ValueError("OBSERVATION_COVARIANCE_NOT_POSITIVE_DEFINITE") from None
        L = np.tril(fact)
        logdet = float(2 * np.log(L.diagonal().real).sum(dtype=np.float64))
        chol_seconds = float(perf_counter() - chol_started)
        solve_started = perf_counter()
        W = cho_solve((fact, lower), rdp.conj().T,
                      check_finite=False).conj().T
        solve_seconds = float(perf_counter() - solve_started)
        residual = float(np.linalg.norm(W @ C - rdp, "fro"))
        residual_scale = float(np.linalg.norm(W, "fro") * normC
                               + np.linalg.norm(rdp, "fro"))
        if residual > ATOL + RTOL * residual_scale:
            raise ValueError("CONDITIONAL_OPERATOR_SOLVE_RESIDUAL")
        eig_started = perf_counter()
        eigenvalues = eigvalsh(C, check_finite=False)
        if eigenvalues[0] <= 0:
            raise ValueError("OBSERVATION_COVARIANCE_NOT_POSITIVE_DEFINITE")
        audit_seconds = float(perf_counter() - eig_started)
        audit = dict(name=str(origin.get("name", label)), n_p=npilot, n_d=ndata,
                     dtype="complex128", sigma2=sigma2,
                     noise_additions=1, covariance_Hermitian_absolute_residual=hermitian_error,
                     covariance_Hermitian_relative_residual=hermitian_error / normC,
                     min_eigenvalue=float(eigenvalues[0]),
                     max_eigenvalue=float(eigenvalues[-1]),
                     condition_number=float(eigenvalues[-1] / eigenvalues[0]),
                     conditional_solve_absolute_residual=residual,
                     conditional_solve_backward_residual=residual / max(residual_scale, np.finfo(float).tiny),
                     logdet=logdet, cholesky_logdet_seconds=chol_seconds,
                     conditional_map_seconds=solve_seconds,
                     diagnostic_eigenvalue_seconds=audit_seconds,
                     persistent_array_bytes=int(mu_p.nbytes + mu_d.nbytes
                                                + C.nbytes + L.nbytes + W.nbytes),
                     atol=ATOL, rtol=RTOL)
        components.append(GaussianComponent(audit["name"], _readonly(mu_p),
                          _readonly(mu_d), _readonly(C), _readonly(L),
                          _readonly(W), logdet, audit))
    npilot, ndata = expected_shape
    setup_audit = dict(n_components=Q, n_p=npilot, n_d=ndata, sigma2=sigma2,
                       priors=prior.tolist(), total_setup_seconds=float(perf_counter() - started),
                       online_information="current received pilot only",
                       covariance_definition="centered clean R_pp plus sigma2 I exactly once",
                       row_convention="ordinary transpose of complex columns",
                       complex_Gaussian_constant_omitted="-n_p log(pi), common to all components",
                       covariance_regularization="none",
                       posterior_atol=POSTERIOR_ATOL,
                       means_complex_coefficients=Q * (npilot + ndata),
                       conditional_maps_complex_coefficients=Q * ndata * npilot,
                       Cholesky_full_array_complex_coefficients=Q * npilot * npilot,
                       Cholesky_mathematical_nonzero_coefficients=Q * npilot * (npilot + 1) // 2,
                       online_dense_conditional_complex_MAC_proxy=Q * ndata * npilot,
                       online_triangular_solve_complex_MAC_proxy=Q * npilot * (npilot - 1) // 2,
                       online_weighted_output_complex_by_real_multiplications=Q * ndata,
                       components=[c.audit for c in components])
    return PreparedGaussianMixture(tuple(components), _readonly(prior),
                                   _readonly(log_prior), sigma2, setup_audit)
