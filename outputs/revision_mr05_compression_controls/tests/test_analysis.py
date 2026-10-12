"""Small synthetic checks only; no project data, inference or new draws."""
import numpy as np
import pytest

from analyze_mr05 import (paired_bootstrap, normalized_decomposition,
                         contrast_definitions, validate_draws, assert_close,
                         oriented_old_contrast)


def test_paired_energy_pooling_uses_same_trajectory_indices_and_dB_difference():
    # The two trajectories deliberately have unequal truth energies; averaging
    # trajectory NMSEs gives the wrong answer. Both SNRs repeat each draw.
    truth = np.array([1., 9.])
    energy = np.array([[[1., 2.], [2., 4.]], [[9., 27.], [18., 54.]]])
    indices = np.array([[0, 1], [1, 1], [0, 0]], dtype=np.int64)
    before = indices.copy()
    point, db, bootstrap_linear, bootstrap_db = paired_bootstrap(energy, truth, indices)
    np.testing.assert_allclose(point, [[1., 2.9], [2., 5.8]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(bootstrap_linear[1], [[1., 3.], [2., 6.]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(bootstrap_linear[2], [[1., 2.], [2., 4.]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(bootstrap_db[:, 0, 1] - bootstrap_db[:, 0, 0],
                               10 * np.log10([2.9, 3., 2.]), rtol=0, atol=1e-14)
    assert np.array_equal(indices, before)
    assert len(contrast_definitions()) == 19
    original = dict(estimate_db=-1.3, CI_lower=-1.5, CI_upper=-1.1)
    assert oriented_old_contrast(original, reverse=True) == (1.3, 1.1, 1.5)
    with pytest.raises(ValueError, match="INDEX_RANGE"):
        validate_draws(np.array([[0, 2]], dtype=np.int64), n=2, resamples=1)
    with pytest.raises(ValueError, match="INDEX_AXES"):
        validate_draws(np.array([[0., 1.]]), n=2, resamples=1)


def test_signed_cross_term_survives_normalization_and_corruption_is_rejected():
    # Dense error is the third method. A is positive but B can make the
    # compressed method's risk lower. The identity is linear, never in dB.
    E = np.array([[[11., 8., 10., 9.]], [[23., 17., 20., 18.]]])
    Q = np.array([5., 15.])
    A = np.array([[[2., 1., 0., 3.]], [[4., 2., 0., 6.]]])
    B = E - E[:, :, 2:3] - A
    av, bv, summed, difference, check, aggregate = normalized_decomposition(E, Q, A, B)
    assert av[0, 3] == .45
    assert bv[0, 3] == -.6
    np.testing.assert_allclose(difference[0, 3], -.15, rtol=0, atol=1e-14)
    np.testing.assert_allclose(summed, difference, rtol=0, atol=1e-14)
    assert check["max_abs"] == aggregate["max_abs"] == 0
    corrupted = B.copy(); corrupted[0, 0, 3] += 1e-3
    with pytest.raises(ValueError, match="TRAJECTORY_SIGNED_IDENTITY"):
        normalized_decomposition(E, Q, A, corrupted)
    with pytest.raises(ValueError, match="NONFINITE"):
        assert_close(np.array([np.nan]), np.array([0.]), "synthetic")
