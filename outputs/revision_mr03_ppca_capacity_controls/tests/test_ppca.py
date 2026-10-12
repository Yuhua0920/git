"""Synthetic tests only; no preparation/candidate/target files are read."""
import math
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ppca import (ATOL, RTOL, PPCAModel, training_moments, leading_eigenpairs,
                  direct_ppca, conditional_operator, training_logscore,
                  blocked_residual_energy, model_comparison, from_author_arrays)


def toy(seed=630301):
    rng = np.random.default_rng(seed)
    rows = rng.standard_normal((37, 7)) + 1j * rng.standard_normal((37, 7))
    rows[:, 1] += (0.8 + 0.7j) * rows[:, 0]
    rows += np.arange(7) * (0.3 - 0.2j)
    return rows


def model(seed=630302):
    rng = np.random.default_rng(seed)
    F = (rng.standard_normal((7, 3)) + 1j * rng.standard_normal((7, 3))) / 2
    mu = rng.standard_normal(7) + 1j * rng.standard_normal(7)
    return PPCAModel(mu, F, 0.27)


def full_score(m, rows):
    C = m.loading @ m.loading.conj().T + m.nu * np.eye(m.dimension)
    Z = rows - m.mean
    # Explicit column-wise dense inverse reference, synthetic dimensions only.
    invC = np.linalg.solve(C, np.eye(m.dimension))
    q = np.einsum('ni,ij,nj->n', Z.conj(), invC, Z).real
    logdet = np.linalg.slogdet(C)[1]
    return float(np.mean(-m.dimension * math.log(math.pi) - logdet - q))


def test_complex_covariance_convention_and_population_denominator():
    rows = toy()
    before = rows.copy()
    mu, S, audit = training_moments(rows, block_size=8)
    Z = rows - rows.mean(axis=0)
    expected = Z.T @ Z.conj() / len(rows)
    assert abs(expected[0, 1].imag) > 0.3
    assert_allclose(mu, rows.mean(axis=0), atol=ATOL, rtol=RTOL)
    assert_allclose(S, expected, atol=ATOL, rtol=RTOL)
    assert np.linalg.norm(S - Z.conj().T @ Z / len(rows)) > 1.0
    assert np.linalg.norm(S - Z.T @ Z.conj() / (len(rows) - 1)) > 0.1
    assert audit['denominator'] == len(rows)
    assert np.array_equal(before, rows)


@pytest.mark.parametrize('block', [1, 3, 37, 100])
def test_blocked_moments_consistent(block):
    mu, S, _ = training_moments(toy(), block_size=block)
    refmu, refS, _ = training_moments(toy(), block_size=8)
    assert_allclose(mu, refmu, atol=ATOL, rtol=RTOL)
    assert_allclose(S, refS, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize('block', [0, -1, True, 1.5])
def test_invalid_block_size_rejected(block):
    with pytest.raises(ValueError):
        training_moments(toy(), block)


def test_nonfinite_training_rejected():
    rows = toy()
    rows[20, 3] = np.nan
    with pytest.raises(ValueError):
        training_moments(rows, 8)


def test_exact_spectrum_matches_dense_eigh():
    _, S, _ = training_moments(toy(), 8)
    s = leading_eigenpairs(S, 4)
    values = np.linalg.eigvalsh(S)[::-1][:4]
    assert_allclose(s.eigenvalues, values, atol=ATOL, rtol=RTOL)
    assert_allclose(S @ s.eigenvectors, s.eigenvectors * s.eigenvalues, atol=ATOL, rtol=RTOL)
    assert_allclose(s.eigenvectors.conj().T @ s.eigenvectors, np.eye(4), atol=ATOL, rtol=RTOL)
    assert not s.eigenvectors.flags.writeable


def test_bad_covariance_rejected():
    with pytest.raises(ValueError, match='NOT_PSD'):
        leading_eigenpairs(np.diag([3., 2., 1., -0.01]).astype(complex), 2)
    S = np.eye(4, dtype=complex)
    S[0, 1] = 0.1j
    with pytest.raises(ValueError, match='HERMITIAN'):
        leading_eigenpairs(S, 2)


def test_direct_ppca_floor_and_effective_rank():
    S = np.diag([9., 4., .01, .005, .001]).astype(complex)
    s = leading_eigenpairs(S, 3)
    m = direct_ppca(np.zeros(5), s, 3, residual_floor=0.1)
    assert_allclose(m.audit['raw_residual_variance'], .003, atol=ATOL, rtol=RTOL)
    assert m.nu == .1
    assert m.audit['floor_active']
    assert m.rank == 3
    assert m.audit['effective_loading_rank'] == 2
    assert_allclose(m.loading @ m.loading.conj().T + m.nu*np.eye(5),
                    np.diag([9.,4.,.1,.1,.1]), atol=ATOL, rtol=RTOL)
    assert 'not author' in m.audit['floor_scope']


def test_direct_ppca_mean_and_spectral_mle_against_perturbations():
    rows = toy()
    mu, S, _ = training_moments(rows, 8)
    s = leading_eigenpairs(S, 4)
    m = direct_ppca(mu, s, 3)
    reference = full_score(m, rows)
    assert not m.audit['floor_active']
    assert_allclose(training_logscore(m, mu, S)['mean_frame_logscore'], reference,
                    atol=ATOL, rtol=RTOL)
    variants = [PPCAModel(mu+.07+.04j, m.loading, m.nu),
                PPCAModel(mu, m.loading*1.1, m.nu),
                PPCAModel(mu, m.loading, m.nu*1.2),
                PPCAModel(mu, np.roll(m.loading, 1, axis=0), m.nu)]
    for alternative in variants:
        assert reference > full_score(alternative, rows)


def test_spectral_loading_rank_is_not_whole_conditional_rank():
    m = model()
    op = conditional_operator(m, [3,0], [6,2,1,5,4], .13)
    assert m.rank == 3
    assert op.audit['numerical_operator_rank'] <= 2


def test_noise_once_full_gaussian_noncontiguous_indices_and_natural_means():
    m = model()
    pidx = np.array([3, 0]); didx = np.array([6, 2, 1])
    sigma2 = .13
    op = conditional_operator(m, pidx, didx, sigma2)
    C = m.loading @ m.loading.conj().T + m.nu * np.eye(7)
    Cpp = C[np.ix_(pidx, pidx)] + sigma2 * np.eye(2)
    Cdp = C[np.ix_(didx, pidx)]
    W = np.linalg.solve(Cpp, Cdp.conj().T).conj().T
    p = toy()[:5, pidx]
    expected = m.mean[didx] + (p-m.mean[pidx]) @ W.T
    assert abs(Cdp[0,0].imag) > 0.05
    assert_allclose(op.W, W, atol=ATOL, rtol=RTOL)
    assert_allclose(op.estimate(p), expected, atol=ATOL, rtol=RTOL)
    assert_allclose(op.estimate(p[0]), expected[0], atol=ATOL, rtol=RTOL)
    assert_allclose(op.estimate(m.mean[pidx]), m.mean[didx], atol=ATOL, rtol=RTOL)
    W_twice = np.linalg.solve(Cpp + sigma2*np.eye(2), Cdp.conj().T).conj().T
    assert np.linalg.norm(op.W - W_twice) > 0.001
    assert op.audit['noise_added_once']
    assert_allclose(op.audit['observed_isotropic_variance'], m.nu+sigma2,
                    atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize('pidx,didx', [([0,0],[2]), ([0],[0]), ([-1],[2]), ([0.0],[2]), ([0],[7])])
def test_bad_joint_positions_rejected(pidx,didx):
    with pytest.raises(ValueError):
        conditional_operator(model(), pidx, didx, .1)


def test_full_gaussian_logscore_with_mean_mismatch():
    rows = toy()
    mu, S, _ = training_moments(rows)
    m = model()
    scored = training_logscore(m, mu, S)
    assert_allclose(scored['mean_frame_logscore'], full_score(m, rows), atol=ATOL, rtol=RTOL)
    assert scored['mean_mismatch_quadratic'] > 0
    matched_mean = PPCAModel(mu, m.loading, m.nu)
    assert_allclose(training_logscore(matched_mean, mu, S)['mean_frame_logscore'] - scored['mean_frame_logscore'],
                    scored['mean_mismatch_quadratic'], atol=ATOL, rtol=RTOL)


def test_unitary_loading_invariance_and_stable_model_distances():
    m = model()
    rng = np.random.default_rng(630303)
    Q, _ = np.linalg.qr(rng.standard_normal((3,3))+1j*rng.standard_normal((3,3)))
    rotated = PPCAModel(m.mean, m.loading@Q, m.nu)
    pidx=[3,0];didx=[6,2,1]
    a=conditional_operator(m,pidx,didx,.13);b=conditional_operator(rotated,pidx,didx,.13)
    assert_allclose(a.W,b.W,atol=ATOL,rtol=RTOL)
    mu,S,_=training_moments(toy())
    assert_allclose(training_logscore(m,mu,S)['mean_frame_logscore'],
                    training_logscore(rotated,mu,S)['mean_frame_logscore'],atol=ATOL,rtol=RTOL)
    comparison=model_comparison(m,rotated,pidx,didx,.13)
    assert comparison['covariance_frobenius_difference'] < ATOL
    assert max(comparison['principal_angles_radians']) < ATOL
    assert comparison['conditional_operator_absolute_difference'] < ATOL


def test_model_comparison_covariance_distance_matches_dense():
    a,b=model(),model(630304)
    comparison=model_comparison(a,b,[3,0],[6,2,1],.13)
    Ca=a.loading@a.loading.conj().T+a.nu*np.eye(7)
    Cb=b.loading@b.loading.conj().T+b.nu*np.eye(7)
    assert_allclose(comparison['covariance_frobenius_difference'],np.linalg.norm(Ca-Cb,'fro'),
                    atol=ATOL,rtol=RTOL)


def test_blocked_residual_energy_audits_trace_tail():
    rows=toy();mu,S,_=training_moments(rows)
    s=leading_eigenpairs(S,4)
    tail=s.trace-math.fsum(s.eigenvalues)
    explicit=blocked_residual_energy(rows,mu,s.eigenvectors,8)
    assert_allclose(explicit,tail,atol=ATOL,rtol=RTOL)


def test_author_arrays_use_variance_and_reject_invalid_shape():
    m=model(); parsed=from_author_arrays(m.mean[None,:],m.loading[None,:,:],
                                         np.full((1,7),m.nu),np.ones(1))
    assert_allclose(parsed.loading,m.loading,atol=ATOL,rtol=RTOL)
    assert parsed.nu==m.nu
    with pytest.raises(ValueError,match='K1_SHAPES'):
        from_author_arrays(np.stack([m.mean]*2),np.stack([m.loading]*2),np.full((2,7),m.nu),np.full(2,.5))
    psi=np.full((1,7),m.nu);psi[0,3]+=.1
    with pytest.raises(ValueError,match='ISOTROPIC'):
        from_author_arrays(m.mean[None,:],m.loading[None,:,:],psi,np.ones(1))


def test_synthetic_author_adapter_full_float_allcomponents_equivalence():
    # No fitting: parameters below are explicitly synthetic, not the absent final model.
    from src.baselines.mfa_reference_adapter import make_reference_model, MFAReferenceAdapter
    m=model();C=m.loading@m.loading.conj().T+m.nu*np.eye(7)
    original=make_reference_model(1,3)
    original.D=7
    original.means=m.mean[None,:].copy()
    original.lambdas=m.loading[None,:,:].copy()
    original.psis=np.full((1,7),m.nu)
    original.covs=C[None,:,:]
    original.inv_covs=np.linalg.solve(C,np.eye(7))[None,:,:]
    original.amps=np.ones(1)
    pidx=np.array([3,0]);didx=np.array([6,2,1])
    A=np.zeros((2,7),complex);A[np.arange(2),pidx]=1
    adapter=MFAReferenceAdapter(original,A)
    p=toy()[:5,pidx]
    actual=adapter.estimate(p,.13*np.eye(2))
    expected=conditional_operator(m,pidx,didx,.13).estimate(p)
    assert_allclose(actual[:,didx],expected,atol=ATOL,rtol=RTOL)
