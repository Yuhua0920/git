"""Small deterministic theory calibrations; never import CDL or formal inputs."""
import numpy as np
import pytest

from theory_toys import (
    NUMERICAL_TOLERANCE, alignment_check, failure_toy, fixed_toy_library,
    heldout_noise_monte_carlo, joint_covariance, kernels, proper_complex,
    real_operator_gram, success_toy, weighted_truncation,
)


def test_exact_prepared_target_selects_same_identifiable_dense_endpoint():
    result = success_toy()
    assert result["PASS"]
    np.testing.assert_allclose(result["pilot_coordinate"], [1., 0.], atol=NUMERICAL_TOLERANCE, rtol=0)
    np.testing.assert_allclose(result["dense_data_coordinate"], [1., 0.], atol=NUMERICAL_TOLERANCE, rtol=0)
    assert min(result["pilot_gram_eigenvalues"]) > 0
    assert min(result["dense_data_gram_eigenvalues"]) > 0


def test_compressed_toy_endpoint_does_not_inherit_dense_optimum():
    result = success_toy()
    assert result["compressed_coordinate_difference_from_dense"] > 1e-3


def test_same_pilot_marginal_distinct_legal_data_optima():
    result = failure_toy()
    assert result["same_noisy_pilot_covariance"]
    assert result["same_prepared_receiver_library"]
    assert min(result["joint_covariance_min_eigenvalues"]) > 0
    np.testing.assert_allclose(result["same_unique_population_pilot_coordinate"], [1., 0.], atol=NUMERICAL_TOLERANCE)
    np.testing.assert_allclose(result["first_dense_data_coordinate"], [1., 0.], atol=NUMERICAL_TOLERANCE)
    np.testing.assert_allclose(result["second_dense_data_coordinate"], [0., 1.], atol=NUMERICAL_TOLERANCE)
    assert result["cross_covariance_difference"] > 0


def test_fixed_population_risk_and_nonzero_empirical_gradient():
    result = alignment_check()
    assert result["maximum_identity_error"] < NUMERICAL_TOLERANCE
    assert result["population_risk_difference"] > 0
    assert abs(result["empirical_evaluation_gradient_cross_term"]) > 0.01
    assert abs(result["empirical_evaluation_risk_difference"] - result["empirical_evaluation_gram_distance"]) > 0.01


def test_noise_expectation_has_320_complex_not_640_complex_variances():
    result = heldout_noise_monte_carlo(windows=64, sigma2=.3, seed=919)
    assert result["expected_constant"] == 96.0
    assert result["complex_heldout_equations"] == 320
    assert result["real_components"] == 640
    assert result["real_component_variance"] == .15
    assert result["noisy_fold_inputs_retained"]
    assert result["fold_reuse_preserved"]
    assert result["monte_carlo_standard_error"] > 0
    # Test the reported sampling error, not an unjustified floating-point gate.
    np.testing.assert_allclose(result["mean_minus_expectation"] / result["monte_carlo_standard_error"],
                               result["standardized_monte_carlo_error"], atol=NUMERICAL_TOLERANCE)


@pytest.mark.parametrize("windows,sigma2", [(1, .4), (0, .4), (2, 0.), (2, -.1)])
def test_noise_check_rejects_invalid_mc_settings(windows, sigma2):
    with pytest.raises(ValueError):
        heldout_noise_monte_carlo(windows=windows, sigma2=sigma2)


def test_joint_covariance_matches_constructive_linear_model():
    R = np.array([[2.0, .3j], [-.3j, 1.0]])
    A = np.array([[1.0, .2+.1j]])
    joint, cross, data = joint_covariance(R, A, .5)
    np.testing.assert_allclose(cross, A @ R)
    np.testing.assert_allclose(data, A @ R @ A.conj().T + .5)
    assert np.linalg.eigvalsh(joint)[0] > 0


def test_weighted_rank_one_tail_and_proper_complex_moment():
    lib = fixed_toy_library()
    B = lib["W"]["P"]-lib["W"]["Source"]
    C = lib["rpp"]["P"]+lib["sigma2"]*np.eye(4)
    lower = np.linalg.cholesky(C)
    compressed = weighted_truncation(B, C, 1)
    singular = np.linalg.svd(B @ lower, compute_uv=False)
    np.testing.assert_allclose(np.linalg.norm((B-compressed) @ lower)**2,
                               np.sum(singular[1:]**2), atol=NUMERICAL_TOLERANCE, rtol=NUMERICAL_TOLERANCE)
    assert np.linalg.matrix_rank(compressed, tol=1e-11) == 1
    z = proper_complex(np.random.default_rng(17), (100000,))
    assert abs(np.mean(abs(z)**2)-1.) < .015
    assert abs(np.mean(z*z)) < .015


def test_population_gram_equals_explicit_real_stacked_design():
    rng = np.random.default_rng(29)
    X = [proper_complex(rng, (3, 4)), proper_complex(rng, (3, 4))]
    C = np.array([[1., .1j, 0., 0.], [-.1j, 1.5, .1, 0.], [0., .1, .8, .05], [0., 0., .05, 1.2]])
    lower = np.linalg.cholesky(C)
    complex_columns = np.column_stack([(a @ lower).ravel() for a in X])
    G = np.concatenate((complex_columns.real, complex_columns.imag), axis=0)
    np.testing.assert_allclose(real_operator_gram(X, C), G.T @ G, atol=NUMERICAL_TOLERANCE, rtol=NUMERICAL_TOLERANCE)


def test_fold_kernel_is_exact_conditional_lmmse_without_target_data():
    lib = fixed_toy_library()
    values = kernels(lib["rpp"]["P"], lib["sigma2"], lib["folds"])
    for K, (observed, held) in zip(values, lib["folds"]):
        C = lib["rpp"]["P"][np.ix_(observed, observed)] + lib["sigma2"]*np.eye(len(observed))
        np.testing.assert_allclose(K @ C, lib["rpp"]["P"][np.ix_(held, observed)],
                                   atol=NUMERICAL_TOLERANCE, rtol=NUMERICAL_TOLERANCE)
