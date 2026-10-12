"""T19B exact complex operator diagnostics. No receiver or fitted statistics."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from src.delay_factor_challenge import ENVIRONMENTS, PURE, SNRS

EXPERIMENT_TYPE = 'exact operator-space, mean-bias, compression, and residual-direction audit for delay-factor mismatch'
DEVELOPMENT = 'CDL-C-30-200'
HELDOUT = tuple(n for n in ENVIRONMENTS if n != DEVELOPMENT)
PRIMARY = ('CDL-C-30-50', 'CDL-C-90-200', 'CDL-B-30-200', 'CDL-B-90-200')
C10 = 'A delay-specific residual direction outside the frozen P/M span is reusable across held-out delay-factor contexts.'
CALIBRATION_RTOL = 1e-11  # numerical agreement only; not a scientific threshold


def complex_array(x):
    a = np.asarray(x, dtype=np.complex128)
    if not np.isfinite(a).all():
        raise ValueError('finite complex128 arrays required')
    return a


def energy(x):
    x = complex_array(x)
    return float(np.sum(x.real*x.real + x.imag*x.imag, dtype=np.float64))


def ratio(numerator, denominator):
    if not np.isfinite([numerator, denominator]).all() or denominator <= 0:
        return None
    return float(numerator/denominator)


def operator_excess(w, target, lower):
    return energy((complex_array(w)-complex_array(target))@complex_array(lower))


def mean_bias(w, delta_p, delta_d):
    return complex_array(delta_d)-complex_array(w)@complex_array(delta_p)


def mismatch(w, target, lower, delta_p, delta_d):
    op = operator_excess(w, target, lower)
    mean = energy(mean_bias(w, delta_p, delta_d))
    return dict(E_op=op, E_mean=mean, E_total=op+mean)


def real_ls(response, columns):
    """Independent real normal equations from complex128 stacked arrays.

    Cholesky solve, no inverse, SVD, pseudoinverse or iterative optimizer.
    Ill-conditioned engineering sentinel uses the existing 1e-12 rank rule.
    """
    response = complex_array(response).reshape(-1)
    design = np.stack([complex_array(c).reshape(-1) for c in columns], axis=1)
    if design.shape[0] != response.size or design.shape[1] not in (2, 3):
        raise ValueError('two or three equal-length complex columns required')
    real = np.concatenate((design.real, design.imag), axis=0)
    rhs_vector = np.concatenate((response.real, response.imag))
    gram = real.T@real
    rhs = real.T@rhs_vector
    eig = np.linalg.eigvalsh(gram)
    if eig[0] <= 1e-12*eig[-1]:
        raise ValueError('ill-conditioned real Gram; do not change basis or regularize')
    coefficients = cho_solve(cho_factor(gram, lower=True), rhs)
    residual = response-design@coefficients
    return dict(coefficients=coefficients, gram=gram, rhs=rhs,
                residual=residual, excess=energy(residual),
                normal_relative_residual=float(np.linalg.norm(gram@coefficients-rhs)/max(np.linalg.norm(rhs), 1e-300)))


def operator_fit(source, target, lower, modules):
    return real_ls((target-source)@lower, [m@lower for m in modules])


def mean_aware_fit(source, target, lower, modules, delta_p, delta_d):
    response = np.concatenate((((target-source)@lower).reshape(-1), mean_bias(source, delta_p, delta_d)))
    columns = [np.concatenate(((m@lower).reshape(-1), m@delta_p)) for m in modules]
    return real_ls(response, columns)


def deployed_excess(coefficients, optimum, gram, optimum_excess):
    """Real-LS orthogonality avoids cancellation across 9,600 saved windows."""
    delta = np.asarray(coefficients, dtype=np.float64)-np.asarray(optimum, dtype=np.float64)
    return float(optimum_excess+delta@gram@delta)


def deployed_efficiency(source_excess, fitted_excess, oracle_excess):
    return ratio(source_excess-fitted_excess, source_excess-oracle_excess)


def signed_cosine(candidate, residual, source_lower):
    a = complex_array(candidate)@complex_array(source_lower)
    b = complex_array(residual)@complex_array(source_lower)
    denominator = np.sqrt(energy(a))*np.sqrt(energy(b))
    return ratio(float(np.vdot(a, b).real), float(denominator))


def centered_quadratic(w, cz, rdp, rdd):
    return float(np.trace(rdd).real-2*np.trace(w@rdp.conj().T).real+np.trace(w@cz@w.conj().T).real)


def dense_gate(rows):
    index = {(r['environment'], int(r['snr_db'])):r for r in rows}
    expected = {(n, s) for n in ENVIRONMENTS for s in SNRS}
    if len(rows) != 16 or set(index) != expected:
        raise ValueError('sixteen unique conditions required')
    pure = [index[n, s] for n in PURE for s in SNRS]
    valid = all(r['operator_span_capture_dense'] is not None for r in pure)
    deficient = [dict(environment=r['environment'], snr_db=r['snr_db']) for r in pure
                 if r['operator_span_capture_dense'] is not None and r['operator_span_capture_dense'] < .50]
    sufficient = valid and all(r['operator_span_capture_dense'] >= .70 for r in pure)
    deficiency = valid and len(deficient) >= 3
    return dict(dense_pm_operator_space_delay_sufficient=sufficient,
                dense_pm_operator_space_delay_deficient=deficiency,
                dense_pm_operator_space_delay_partial=not sufficient and not deficiency,
                condition_specific_operator_deficiency=bool(deficient), deficient_conditions=deficient)


def reuse_gate(rows):
    index = {(r['environment'], int(r['snr_db'])):r for r in rows}
    if len(rows) != 14 or set(index) != {(n, s) for n in HELDOUT for s in SNRS}:
        raise ValueError('exactly fourteen held-out conditions; self-condition forbidden')
    primary = [index[n, s] for n in PRIMARY for s in SNRS]
    valid = all(r['operator_span_capture_3d'] is not None and r['delay_direction_incremental_capture'] is not None for r in primary)
    promising = sum(r['operator_span_capture_3d'] is not None and r['delay_direction_incremental_capture'] is not None
                    and r['operator_span_capture_3d'] >= .70 and r['delay_direction_incremental_capture'] >= .10 for r in primary)
    weak = sum(r['delay_direction_incremental_capture'] is not None and r['delay_direction_incremental_capture'] < .05 for r in primary)
    reusable = valid and promising >= 6
    is_weak = valid and weak >= 6
    return dict(delay_residual_direction_reusable=reusable, delay_direction_reuse_weak=is_weak,
                delay_direction_reuse_partial=not reusable and not is_weak,
                promising_count=promising, weak_count=weak, primary_conditions=8,
                c10_claim_status=('SUPPORTED_AS_ORACLE_DIRECTION_DIAGNOSTIC' if reusable else
                                 'CONTRADICTED_BY_DIRECTION_REUSE_TEST' if is_weak else 'INCONCLUSIVE'))


def calibration():
    """Small deterministic complete joint covariance and symmetric finite support.

    No channel generator/random realization; dimensions differ from production.
    Rdd exists only here. The +/- support has exactly zero centered first moment.
    """
    k = np.arange(1, 26).reshape(5, 5)
    a = np.sin(k)+1j*np.cos(k*.7)
    joint = a@a.conj().T+np.eye(5)*.7
    lower_joint = np.linalg.cholesky(joint)
    cz, rdp, rdd = joint[:2, :2], joint[2:, :2], joint[2:, 2:]
    lower = np.linalg.cholesky(cz)
    target = cho_solve(cho_factor(cz, lower=True), rdp.conj().T).conj().T
    source = (np.arange(6).reshape(3, 2)+1j*np.arange(6, 12).reshape(3, 2))*.04
    modules = [np.sin(np.arange(6).reshape(3, 2)*j+.3)+1j*np.cos(np.arange(6).reshape(3, 2)*(.5+j)) for j in (1, 2, 4)]
    dp = np.array([.3+.2j, -.2+.1j]);dd = np.array([.1+.2j, .4-.3j, -.2+.5j])
    error = {}
    q0 = centered_quadratic(target, cz, rdp, rdd)
    for label, w in (('source', source), ('arbitrary', source+modules[0]*.17), ('matched', target)):
        budget = mismatch(w, target, lower, dp, dd)
        error[label+'_quadratic'] = abs(centered_quadratic(w, cz, rdp, rdd)-q0-budget['E_op'])
        centered = np.concatenate((np.sqrt(5)*lower_joint.T, -np.sqrt(5)*lower_joint.T), axis=0)
        ec = centered[:, 2:]-centered[:, :2]@w.T
        bias = mean_bias(w, dp, dd)
        full_error = ec+bias
        direct = float(np.mean(np.sum(np.abs(full_error)**2, axis=1)))
        error[label+'_mean_decomposition'] = abs(direct-q0-budget['E_total'])
        error[label+'_cross_term'] = abs(float(np.mean(2*np.real(ec.conj()@bias))))
    fit = operator_fit(source, target, lower, modules[:2])
    total = mean_aware_fit(source, target, lower, modules[:2], dp, dd)
    fitted = source+sum(c*m for c,m in zip(fit['coefficients'], modules))
    total_w = source+sum(c*m for c,m in zip(total['coefficients'], modules))
    error['operator_fit_residual'] = abs(operator_excess(fitted, target, lower)-fit['excess'])
    error['mean_aware_fit_residual'] = abs(mismatch(total_w, target, lower, dp, dd)['E_total']-total['excess'])
    co = fit['coefficients']+np.array([.7, -.3])
    dw = source+sum(c*m for c,m in zip(co, modules))
    error['deployed_quadratic'] = abs(operator_excess(dw, target, lower)-deployed_excess(co, fit['coefficients'], fit['gram'], fit['excess']))
    fit3 = operator_fit(source, target, lower, modules)
    error['joint_3d_normal'] = fit3['normal_relative_residual']
    if any(v > 1e-11 for v in error.values()):
        raise ValueError('independent complete covariance calibration failed')
    return dict(passed=True, maximum_absolute_error=max(error.values()), errors=error,
                full_Rdd_available_in_synthetic_sentinel_only=True,
                centered_support='deterministic symmetric +/- columns of full joint Cholesky',
                random_sampling=False, machine_precision_criterion=1e-11)
