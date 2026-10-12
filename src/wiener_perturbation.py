"""T17A operator-only Wiener covariance sensitivities; complex128, no channels."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular, svd


def matrix(value, shape=None):
    value = np.asarray(value, dtype=np.complex128)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError('finite matrix required')
    if shape is not None and value.shape != shape:
        raise ValueError('matrix shape mismatch')
    return value


def hermitian(value):
    value = matrix(value)
    if value.shape[0] != value.shape[1]:
        raise ValueError('square covariance required')
    if np.linalg.norm(value-value.conj().T) > 1e-11 * max(np.linalg.norm(value), 1.):
        raise ValueError('Hermitian covariance required; no silent repair')
    return value


def frozen(value):
    value = np.array(value, dtype=np.complex128, copy=True)
    value.flags.writeable = False
    return value


@dataclass(frozen=True)
class WienerSensitivity:
    d0: np.ndarray
    p0: np.ndarray
    sigma2: float
    w0: np.ndarray
    factor: tuple

    @classmethod
    def build(cls, d0, p0, sigma2, *, historical_w0=None):
        p0 = hermitian(p0)
        d0 = matrix(d0)
        if d0.shape[1] != p0.shape[0] or not np.isfinite(sigma2) or sigma2 <= 0:
            raise ValueError('invalid dimensions or noise variance')
        factor = cho_factor(p0+sigma2*np.eye(p0.shape[0]), lower=True)
        w0 = cho_solve(factor, d0.conj().T).conj().T
        if historical_w0 is not None:
            historical_w0 = matrix(historical_w0, d0.shape)
            if not np.array_equal(w0, historical_w0):
                raise ValueError('historical W0 must match exactly')
            w0 = historical_w0
        return cls(frozen(d0), frozen(p0), float(sigma2), frozen(w0),
                   (frozen(factor[0]), factor[1]))

    def right_solve(self, value):
        value = matrix(value)
        if value.shape[1] != self.p0.shape[0]:
            raise ValueError('right solve dimensions')
        return cho_solve(self.factor, value.conj().T).conj().T

    def perturbations(self, f, e):
        return matrix(f, self.d0.shape), hermitian(matrix(e, self.p0.shape))

    def exact(self, f, e):
        f, e = self.perturbations(f, e)
        factor = cho_factor(self.p0+self.sigma2*np.eye(e.shape[0])+e, lower=True)
        return cho_solve(factor, (self.d0+f).conj().T).conj().T

    def first(self, f, e):
        f, e = self.perturbations(f, e)
        return self.right_solve(f)-self.right_solve(self.w0@e)

    def second(self, f, e):
        f, e = self.perturbations(f, e)
        fg, eg = self.right_solve(f), self.right_solve(e)
        return -fg@eg+(self.w0@eg)@eg

    def cross(self, fp, ep, fm, em):
        fp, ep = self.perturbations(fp, ep)
        fm, em = self.perturbations(fm, em)
        fpg, fmg = self.right_solve(fp), self.right_solve(fm)
        epg, emg = self.right_solve(ep), self.right_solve(em)
        return -fpg@emg-fmg@epg+self.w0@(epg@emg+emg@epg)

    def neumann_radius(self, e):
        e = hermitian(matrix(e, self.p0.shape))
        lower = np.tril(self.factor[0])
        left = solve_triangular(lower, e, lower=True)
        whitened = solve_triangular(lower, left.conj().T, lower=True).conj().T
        return float(np.max(np.abs(np.linalg.eigvalsh(whitened))))


def energy(value):
    value = matrix(value)
    return float(np.vdot(value, value).real)


def alignment(actual, prediction):
    actual, prediction = matrix(actual), matrix(prediction)
    if prediction.shape != actual.shape:
        raise ValueError('alignment dimensions')
    denominator = np.sqrt(energy(actual)*energy(prediction))
    if denominator == 0:
        return dict(cosine_real=None, normalized_inner_real=None,
                    normalized_inner_imag=None, normalized_inner_abs=None)
    inner = np.vdot(actual, prediction)/denominator
    return dict(cosine_real=float(inner.real), normalized_inner_real=float(inner.real),
                normalized_inner_imag=float(inner.imag), normalized_inner_abs=float(abs(inner)))


def compare(actual, prediction, lower):
    actual = matrix(actual)
    prediction = matrix(prediction, actual.shape)
    lower = matrix(lower, (actual.shape[1], actual.shape[1]))
    residual = actual-prediction
    weighted_actual, weighted_prediction = actual@lower, prediction@lower
    total, unexplained = energy(weighted_actual), energy(residual@lower)
    fraction = unexplained/total if total else None
    raw_total = energy(actual)
    result = dict(reference_weighted_energy=total, residual_weighted_energy=unexplained,
                  weighted_residual_fraction=fraction,
                  relative_weighted_residual=None if fraction is None else float(np.sqrt(fraction)),
                  closure=None if fraction is None else 1.-fraction,
                  frobenius_residual_norm=float(np.sqrt(energy(residual))),
                  relative_frobenius_residual=None if not raw_total else float(np.sqrt(energy(residual)/raw_total)))
    result.update({f'weighted_{k}': v for k,v in alignment(weighted_actual,weighted_prediction).items()})
    result.update({f'frobenius_{k}': v for k,v in alignment(actual,prediction).items()})
    return result


def real_scalar(actual, direction):
    actual = matrix(actual)
    direction = matrix(direction, actual.shape)
    denominator = energy(direction)
    if denominator == 0 or energy(actual) == 0:
        raise ValueError('nonzero scalar direction/reference required')
    gamma = float(np.vdot(direction, actual).real/denominator)
    return dict(gamma=gamma, relative_residual=float(np.sqrt(energy(actual-gamma*direction)/energy(actual))),
                normal_equation_relative_residual=float(abs(np.vdot(direction,actual-gamma*direction).real)/max(abs(np.vdot(direction,actual).real),denominator,1e-300)))


def spectrum(value):
    value = matrix(value)
    u, singular, _ = svd(value, full_matrices=False)
    total = float(np.sum(singular**2))
    numerical_rank = int(np.sum(singular > max(value.shape)*np.finfo(float).eps*singular[0])) if total else 0
    if total:
        p = singular[singular > 0]/np.sum(singular)
        entropy_rank = float(np.exp(-np.sum(p*np.log(p))))
        participation = float(total**2/np.sum(singular**4))
        capture = np.cumsum(singular**2)/total
    else:
        entropy_rank = participation = 0.
        capture = np.zeros_like(singular)
    return u, singular, capture, dict(numerical_rank=numerical_rank,
        entropy_singular_value_effective_rank=entropy_rank, energy_participation_rank=participation,
        squared_frobenius_energy=total)


def principal_angles(left, right, q):
    left, right = matrix(left), matrix(right)
    if left.shape[0] != right.shape[0] or q not in (4,8):
        raise ValueError('same output domain and top4/top8 required')
    if left.shape[1] < q or right.shape[1] < q:
        raise ValueError('insufficient singular vectors')
    c = svd(left[:,:q].conj().T@right[:,:q], compute_uv=False)
    # Singular-value roundoff only; no clipping of scientific closure/energy.
    return list(map(float,np.degrees(np.arccos(np.clip(c,0.,1.)))))


def mechanism_gate(rows):
    selected = {int(x['snr_db']): x for x in rows if int(x['snr_db']) in (10,20)}
    if set(selected) != {10,20}:
        raise ValueError('two pre-registered primary SNRs required')
    first = [float(selected[s]['first_order_additive_closure']) for s in (10,20)]
    corrected = [float(selected[s]['interaction_corrected_closure']) for s in (10,20)]
    if not np.isfinite(first+corrected).all():
        raise ValueError('finite gate values required')
    supported = all(x>=.70 for x in first) and all(x>=.90 for x in corrected)
    contradicted = all(x<.30 for x in first) and all(x<.50 for x in corrected)
    return dict(wiener_perturbation_mechanism_supported=supported,
                wiener_perturbation_mechanism_contradicted=contradicted,
                wiener_perturbation_mechanism_partial=not supported and not contradicted)
