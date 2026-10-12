"""Independent synthetic numeric verification of population covariance pooling."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pooled import FrozenStatistics, pool_statistics


def statistics_from_rows(p, d, ids=()):
    """Independent reference: explicit sample outer-product sums, denominator N."""
    p, d = np.asarray(p, dtype=np.complex128), np.asarray(d, dtype=np.complex128)
    assert p.ndim == d.ndim == 2 and len(p) == len(d)
    n = len(p)
    mp, md = np.mean(p, axis=0), np.mean(d, axis=0)
    pp = sum(np.outer(a - mp, (a - mp).conj()) for a in p) / n
    dp = sum(np.outer(b - md, (a - mp).conj()) for a, b in zip(p, d)) / n
    return FrozenStatistics(mp, md, pp, dp, tuple(ids), n)


def example_origins():
    rng = np.random.default_rng(170102)
    origins, pilots, data = [], [], []
    for origin, n in enumerate((3, 7, 5)):
        p = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3)) + (origin + 1j * origin) * np.array([1, -2, .5])
        d = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2)) + (2 * origin - 1j * origin) * np.array([.3, -1])
        origins.append(statistics_from_rows(p, d, ids=(origin + 10,)))
        pilots.append(p)
        data.append(d)
    return origins, pilots, data


def assert_statistics_close(actual, expected, atol=2e-13):
    for key in ("mean_p", "mean_d", "rpp", "rdp"):
        np.testing.assert_allclose(getattr(actual, key), getattr(expected, key), rtol=2e-14, atol=atol)
    assert actual.samples == expected.samples


def test_concatenated_complex_rows_with_unequal_sample_counts():
    origins, p, d = example_origins()
    expected = statistics_from_rows(np.concatenate(p), np.concatenate(d))
    actual = pool_statistics(origins)
    assert_statistics_close(actual, expected)
    assert actual.samples == 15


def test_nonzero_between_means_matches_hand_calculated_complex_values():
    source = FrozenStatistics(np.array([1 + 1j]), np.array([2 - 2j]), np.zeros((1, 1)), np.zeros((1, 1)), (0,), 1)
    target = FrozenStatistics(np.array([3 - 1j]), np.array([-2 + 4j]), np.zeros((1, 1)), np.zeros((1, 1)), (0,), 3)
    result = pool_statistics([source, target])
    np.testing.assert_allclose(result.mean_p, [2.5 - .5j], rtol=0, atol=1e-15)
    np.testing.assert_allclose(result.mean_d, [-1 + 2.5j], rtol=0, atol=1e-15)
    np.testing.assert_allclose(result.rpp, [[1.5]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(result.rdp, [[-3.75 + .75j]], rtol=0, atol=1e-15)
    assert result.rpp[0, 0].real > 0  # A within-only average would incorrectly return 0.


def test_pooled_pilot_covariance_is_hermitian_positive_semidefinite():
    origins, _, _ = example_origins()
    result = pool_statistics(origins)
    np.testing.assert_allclose(result.rpp, result.rpp.conj().T, rtol=0, atol=1e-13)
    assert np.linalg.eigvalsh(result.rpp).min() >= -1e-13


def test_existing_cholesky_preparation_has_small_independent_residual():
    origins, _, _ = example_origins()
    result = pool_statistics(origins)
    prepared = result.prepare(0.7)
    a = result.rpp + 0.7 * np.eye(3)
    independent = np.linalg.solve(a.T, result.rdp.T).T
    np.testing.assert_allclose(prepared.weight, independent, rtol=2e-13, atol=2e-13)
    residual = np.linalg.norm(prepared.weight @ a - result.rdp) / np.linalg.norm(result.rdp)
    assert residual < 2e-13
    assert prepared.solve_relative_residual < 2e-13


def test_output_shape_complex128_and_readonly_without_mutating_inputs():
    origins, _, _ = example_origins()
    before = [(st.mean_p.copy(), st.mean_d.copy(), st.rpp.copy(), st.rdp.copy()) for st in origins]
    result = pool_statistics(origins)
    assert isinstance(result, FrozenStatistics)
    for key, shape in (("mean_p", (3,)), ("mean_d", (2,)), ("rpp", (3, 3)), ("rdp", (2, 3))):
        a = getattr(result, key)
        assert a.shape == shape and a.dtype == np.complex128
        assert not a.flags.writeable
        assert all(not np.shares_memory(a, getattr(st, key)) for st in origins)
    for st, saved in zip(origins, before):
        for key, original in zip(("mean_p", "mean_d", "rpp", "rdp"), saved):
            np.testing.assert_array_equal(getattr(st, key), original)


def test_single_origin_is_numerically_identical():
    origins, _, _ = example_origins()
    result = pool_statistics([origins[1]])
    assert_statistics_close(result, origins[1], atol=0)
    assert result.train_trajectory_ids == ((0, 11),)


def test_equal_means_reduce_to_weighted_within_covariances():
    mp = np.array([1 + 2j, 3 - 1j])
    md = np.array([2 - 3j])
    a = FrozenStatistics(mp, md, np.array([[2, 1j], [-1j, 3]]), np.array([[1 + 2j, 2 - 1j]]), (1,), 2)
    b = FrozenStatistics(mp.copy(), md.copy(), np.array([[4, -.5j], [.5j, 5]]), np.array([[-3 + 1j, 4 + 2j]]), (1,), 6)
    result = pool_statistics([a, b])
    np.testing.assert_array_equal(result.mean_p, mp)
    np.testing.assert_array_equal(result.mean_d, md)
    np.testing.assert_allclose(result.rpp, [[3.5, -.125j], [.125j, 4.5]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(result.rdp, [[-2 + 1.25j, 3.5 + 1.25j]], rtol=0, atol=1e-15)


def test_small_complex_manual_lmmse_estimate_with_existing_prepare():
    a = FrozenStatistics(np.array([1 + 1j]), np.array([2 - 2j]), np.zeros((1, 1)), np.zeros((1, 1)), (), 1)
    b = FrozenStatistics(np.array([3 - 1j]), np.array([-2 + 4j]), np.zeros((1, 1)), np.zeros((1, 1)), (), 3)
    pooled = pool_statistics([a, b])
    prepared = pooled.prepare(.5)
    # Hand values: Rpp+.5=2; W=(-3.75+.75j)/2.
    expected_w = -1.875 + .375j
    np.testing.assert_allclose(prepared.weight, [[expected_w]], rtol=0, atol=1e-14)
    # p=3.5+.5j; delta_p=1+1j; d_hat=-1+2.5j+W*(1+1j)=-3.25+1j.
    np.testing.assert_allclose(prepared.estimate(np.array([3.5 + .5j])), [-3.25 + 1j], rtol=0, atol=1e-14)


def test_imaginary_means_and_cross_covariance_are_not_truncated():
    origins, _, _ = example_origins()
    promoted = [replace(st, mean_p=st.mean_p.astype(np.complex64), mean_d=st.mean_d.astype(np.complex64),
                        rpp=st.rpp.astype(np.complex64), rdp=st.rdp.astype(np.complex64)) for st in origins]
    result = pool_statistics(promoted)
    assert result.mean_p.dtype == result.rdp.dtype == np.complex128
    assert np.max(np.abs(result.mean_p.imag)) > .1
    assert np.max(np.abs(result.rdp.imag)) > .1
    assert np.isfinite(result.rdp).all()


def test_natural_minus_source_centered_predictions_are_constant():
    origins, _, _ = example_origins()
    pooled = pool_statistics(origins)
    source = origins[0]
    prepared = pooled.prepare(.8)
    pilots = np.array([[1 + 2j, -1 + 3j, .5 - 1j],
                       [-2 + 4j, 1 - 1j, 3 + 2j],
                       [4 - 2j, 5 + 1j, -2 - 2j]], dtype=np.complex128)
    natural = prepared.estimate(pilots)
    sc = source.mean_d + (pilots - source.mean_p) @ prepared.weight.T
    offset = pooled.mean_d - source.mean_d + (source.mean_p - pooled.mean_p) @ prepared.weight.T
    np.testing.assert_allclose(natural - sc, np.broadcast_to(offset, natural.shape), rtol=2e-13, atol=2e-13)
    assert np.linalg.norm(offset) > .1


def test_sample_count_weights_ignore_metadata_length_and_duplicate_ids():
    origins, _, _ = example_origins()
    changed = [replace(st, train_trajectory_ids=(99,) * (1 + i * 17)) for i, st in enumerate(origins)]
    result = pool_statistics(changed)
    expected = pool_statistics(origins)
    assert_statistics_close(result, expected, atol=0)
    assert result.samples == 15
    assert len(result.train_trajectory_ids) == 54
    assert result.train_trajectory_ids[0] == (0, 99)
    assert result.train_trajectory_ids[-1] == (2, 99)


def test_compatible_small_dataclass_interface_is_accepted():
    origins, _, _ = example_origins()
    objects = [SimpleNamespace(**{key: getattr(st, key) for key in (
        "mean_p", "mean_d", "rpp", "rdp", "samples", "train_trajectory_ids")}) for st in origins]
    assert_statistics_close(pool_statistics(objects), pool_statistics(origins), atol=0)


def test_reordering_origins_preserves_numeric_statistics():
    origins, _, _ = example_origins()
    assert_statistics_close(pool_statistics(reversed(origins)), pool_statistics(origins))


@pytest.mark.parametrize("field,value", [
    ("mean_p", np.zeros((1, 3))), ("mean_d", np.zeros((2, 1))),
    ("mean_p", np.empty(0)), ("rpp", np.zeros((2, 3))), ("rdp", np.zeros((3, 3))),
])
def test_invalid_dimensions_are_rejected(field, value):
    origins, _, _ = example_origins()
    origins[0] = replace(origins[0], **{field: value})
    with pytest.raises(ValueError, match="dimensions|vectors"):
        pool_statistics(origins)


def test_cross_origin_dimension_mismatch_is_rejected():
    origins, _, _ = example_origins()
    different = FrozenStatistics(np.ones(4), np.ones(2), np.eye(4), np.ones((2, 4)), (), 3)
    with pytest.raises(ValueError, match="same pilot and data dimensions"):
        pool_statistics([origins[0], different])


@pytest.mark.parametrize("field", ["mean_p", "mean_d", "rpp", "rdp"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, complex(0, np.inf)])
def test_nonfinite_statistics_are_rejected(field, nonfinite):
    origins, _, _ = example_origins()
    a = getattr(origins[0], field).copy()
    a.flat[0] = nonfinite
    origins[0] = replace(origins[0], **{field: a})
    with pytest.raises(ValueError, match="finite"):
        pool_statistics(origins)


@pytest.mark.parametrize("count", [0, -1, True, 2.5, np.nan, np.inf])
def test_invalid_sample_count_is_rejected(count):
    origins, _, _ = example_origins()
    with pytest.raises(ValueError, match="positive integer"):
        pool_statistics([replace(origins[0], samples=count)])


def test_empty_pool_is_rejected():
    with pytest.raises(ValueError, match="at least one origin"):
        pool_statistics([])
