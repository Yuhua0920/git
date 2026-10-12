"""Small numerical oracles for the new prepared-origin estimator, not outcomes."""
import inspect
import ast

import numpy as np
import pytest

from gmm3 import prepare_gaussian_mixture, POSTERIOR_ATOL


def origins(seed=938):
    rng = np.random.default_rng(seed)
    result = []
    # Noncontiguous pilot indices in a joint complex vector detect mapping bugs.
    pidx = np.array([4, 1, 6])
    didx = np.array([0, 2, 3, 5])
    for q in range(3):
        B = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
        C = B @ B.conj().T + .3 * np.eye(7)
        mean = rng.normal(size=7) + 1j * rng.normal(size=7)
        result.append(dict(name=str(q), mu_p=mean[pidx], mu_d=mean[didx],
                           R_pp=C[np.ix_(pidx, pidx)],
                           R_dp=C[np.ix_(didx, pidx)]))
    return result


def oracle(origins_, p, sigma2, priors=None):
    """Independent np.linalg.solve and manually centered scalar softmax."""
    if priors is None:
        priors = np.full(len(origins_), 1 / len(origins_))
    estimates, scores, quadratics = [], [], []
    for prior, stat in zip(priors, origins_):
        C = stat["R_pp"] + sigma2 * np.eye(len(p))
        delta = p - stat["mu_p"]
        solved = np.linalg.solve(C, delta)
        quad = np.vdot(delta, solved)
        assert abs(quad.imag) < 1e-12
        sign, logdet = np.linalg.slogdet(C)
        assert abs(sign - 1) < 1e-12
        scores.append(-np.inf if prior == 0 else np.log(prior) - logdet - quad.real)
        quadratics.append(quad.real)
        estimates.append(stat["mu_d"] + stat["R_dp"] @ solved)
    scores = np.array(scores)
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    estimates = np.array(estimates)
    return weights @ estimates, weights, estimates, np.array(quadratics), scores


def test_complex_conditional_noncontiguous_oracle_and_centers():
    stats = origins()
    p = np.array([1.2 - .7j, -.3 + .4j, .7 + 2j], np.complex128)
    sigma2 = .72
    model = prepare_gaussian_mixture(stats, sigma2)
    actual = model.predict(p, return_components=True)
    expected = oracle(stats, p, sigma2)
    np.testing.assert_allclose(actual.estimate, expected[0], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual.posterior_weights, expected[1], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual.component_estimates, expected[2], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual.quadratic_forms, expected[3], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual.log_scores, expected[4], rtol=2e-13, atol=2e-13)
    assert actual.estimate.dtype == np.complex128
    assert np.isrealobj(actual.quadratic_forms)
    assert (actual.quadratic_forms >= 0).all()
    assert actual.component_estimates.shape == (3, 4)


def test_noise_added_once_not_twice():
    stats = origins()
    model = prepare_gaussian_mixture(stats, .72)
    p = np.array([1 + 2j, 3 + 4j, -1 - 2j])
    got = model.predict(p)
    expected = oracle(stats, p, .72)
    wrong = oracle(stats, p, 1.44)
    np.testing.assert_allclose(got.estimate, expected[0], rtol=2e-13, atol=2e-13)
    assert np.linalg.norm(got.estimate - wrong[0]) > 1e-3
    for component, stat in zip(model.components, stats):
        np.testing.assert_array_equal(component.C, stat["R_pp"] + .72 * np.eye(3))
        assert component.audit["noise_additions"] == 1


def test_identical_components_reduce_to_lmmse():
    stat = origins()[0]
    model = prepare_gaussian_mixture([stat, stat, stat], .1)
    p = np.array([.4j, 1 + .2j, 5 - .1j])
    got = model.predict(p)
    ref = oracle([stat], p, .1)
    np.testing.assert_allclose(got.estimate, ref[0], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(got.posterior_weights, [1/3, 1/3, 1/3], atol=POSTERIOR_ATOL, rtol=0)
    assert got.posterior_entropy_nats == pytest.approx(np.log(3), abs=1e-14)


def test_single_prior_limit_and_zero_entropy():
    stats = origins()
    model = prepare_gaussian_mixture(stats, .3, priors=[0, 1, 0])
    p = np.array([.2 + .8j, .7j, -4j])
    got = model.predict(p)
    expected = oracle([stats[1]], p, .3)
    np.testing.assert_allclose(got.estimate, expected[0], rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(got.posterior_weights, [0, 1, 0])
    assert got.posterior_entropy_nats == 0
    assert got.max_posterior_weight == 1


def test_shared_large_log_score_offset_is_stable():
    stat = dict(mu_p=np.zeros(2, complex), mu_d=np.zeros(1, complex),
                R_pp=np.eye(2, dtype=complex), R_dp=np.ones((1, 2), complex))
    got = prepare_gaussian_mixture([stat, stat, stat], .1).predict(
        np.array([1e7 + 1e7j, -1e7j]))
    assert np.isfinite(got.estimate).all()
    np.testing.assert_allclose(got.posterior_weights, [1/3] * 3, atol=1e-14, rtol=0)
    assert got.posterior_weights.sum() == pytest.approx(1, abs=1e-14)


def test_batch_is_current_frame_vectorization_and_immutable_inputs():
    stats = origins()
    originals = [{k: v.copy() for k, v in s.items() if isinstance(v, np.ndarray)} for s in stats]
    pilots = np.array([[1j, -1 + 2j, 3], [2j, 2 + .1j, 3 - 1j]], np.complex128)
    before = pilots.copy()
    model = prepare_gaussian_mixture(stats, .2)
    batch = model.predict(pilots)
    assert batch.estimate.shape == (2, 4)
    assert batch.posterior_weights.shape == (2, 3)
    assert batch.posterior_entropy_nats.shape == (2,)
    assert batch.component_estimates is None
    for i in range(2):
        single = model.predict(pilots[i])
        np.testing.assert_allclose(batch.estimate[i], single.estimate, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(batch.posterior_weights[i], single.posterior_weights, atol=1e-13, rtol=1e-13)
    np.testing.assert_array_equal(pilots, before)
    for stat, old in zip(stats, originals):
        for k, value in old.items():
            np.testing.assert_array_equal(stat[k], value)
    assert not model.components[0].W.flags.writeable


def test_setup_hpd_solve_and_cost_audit():
    model = prepare_gaussian_mixture(origins(), .123)
    assert model.n_p == 3 and model.n_d == 4 and model.n_components == 3
    assert model.setup_audit["online_dense_conditional_complex_MAC_proxy"] == 36
    for c in model.components:
        np.testing.assert_allclose(c.chol_lower @ c.chol_lower.conj().T, c.C, rtol=1e-13, atol=1e-13)
        assert c.audit["min_eigenvalue"] > 0
        assert c.audit["conditional_solve_backward_residual"] < 1e-13
        assert c.audit["cholesky_logdet_seconds"] >= 0
        assert c.audit["conditional_map_seconds"] >= 0


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, 1 + 0j, True, "not_numeric"])
def test_invalid_noise_rejected(value):
    with pytest.raises(ValueError, match="SIGMA2_POSITIVE_REAL_REQUIRED"):
        prepare_gaussian_mixture(origins(), value)


@pytest.mark.parametrize("value", [[0, 0, 0], [-.1, .5, .6], [1, 1, 1], [np.nan, 0, 1], [1, 0], ["x", "y", "z"], [1j, 0, 0]])
def test_invalid_prior_rejected(value):
    with pytest.raises(ValueError, match="PRIOR_"):
        prepare_gaussian_mixture(origins(), .1, value)


def test_nonhermitian_and_indefinite_rejected():
    stats = origins()
    stats[0]["R_pp"][0, 1] += 1j
    with pytest.raises(ValueError, match="NOT_HERMITIAN"):
        prepare_gaussian_mixture(stats, .1)
    stats = origins()
    stats[0]["R_pp"] = -2 * np.eye(3)
    with pytest.raises(ValueError, match="NOT_POSITIVE_DEFINITE"):
        prepare_gaussian_mixture(stats, .1)


@pytest.mark.parametrize("key,bad", [("mu_p", np.zeros(2)), ("mu_d", np.zeros(3)),
                                    ("R_pp", np.ones((2, 2))), ("R_dp", np.ones((3, 3))),
                                    ("mu_p", ["x", "y", "z"]), ("R_pp", np.full((3,3), np.nan))])
def test_invalid_statistic_dimensions_or_data_rejected(key, bad):
    stats = origins()
    stats[0][key] = bad
    with pytest.raises(ValueError):
        prepare_gaussian_mixture(stats, .1)


@pytest.mark.parametrize("bad", [np.ones(4), np.zeros((2, 4)), np.array([np.nan, 0, 0]), ["x", "y", "z"], np.zeros((2, 2, 3))])
def test_invalid_pilot_rejected(bad):
    model = prepare_gaussian_mixture(origins(), .1)
    with pytest.raises(ValueError):
        model.predict(bad)


def test_no_explicit_inverse_or_truth_or_temporal_argument():
    import gmm3
    tree = ast.parse(inspect.getsource(gmm3))
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in {"inv", "pinv"} for node in ast.walk(tree))
    parameters = list(inspect.signature(gmm3.PreparedGaussianMixture.predict).parameters)
    assert parameters == ["self", "pilots", "return_components"]
