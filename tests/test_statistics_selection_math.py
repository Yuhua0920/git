"""Independent T16F complex residual and dense-simplex mathematical sentinels."""
import inspect
import numpy as np
import pytest
from src.statistics_mixing import (objective_moments, objective_value,
    hard_selection, convex_mixing, geometry, gain, scientific_gate)
from src.pilot_cv import PilotCV, frozen_folds
from src.cdl_benchmark import observation_geometry


def family():
    folds, _ = frozen_folds(observation_geometry().selected)
    n = np.arange(16 * 48).reshape(16, 48)
    base = tuple(.003 * np.cos(n + f) + .002j * np.sin(n + f) for f in range(4))
    dp = tuple(.002 * np.sin(n / 3 + f) + .001j * np.cos(n / 4 + f) for f in range(4))
    dm = tuple(.004 * np.cos(n / 5 + f) + .003j * np.sin(n / 7 + f) for f in range(4))
    return PilotCV(folds, np.zeros(64, complex), base, dp, dm)


def moments(design, response):
    return (design.conj().T @ design).real, (design.conj().T @ response).real, float(np.vdot(response, response).real)


@pytest.mark.parametrize("window", [1, 5])
def test_objective_moments_preserve_frozen_affine_reduction_exact(window):
    cv = family()
    z = (np.arange(window * 64).reshape(window, 64) + 1j * .37) / 90
    g, b, c = objective_moments(cv, z)
    a, info = cv.fit(z, modules=("P", "M"))
    np.testing.assert_array_equal(g, info["gram"])
    np.testing.assert_array_equal(b, info["rhs"])
    assert c == info["source_cv_loss"]
    assert objective_value(a, g, b, c) == info["fitted_cv_loss"]


@pytest.mark.parametrize("chosen,target", [("W0", [0., 0.]), ("WP", [1., 0.]), ("WM", [0., 1.])])
def test_hard_sentinel_direct_complex_cross_prediction_loss(chosen, target):
    a = np.array([[1 + 2j, .3 - 1j], [-.5 + .2j, 2 + .5j], [1j, -1 + 1j]])
    y = a @ np.asarray(target)
    g, b, c = moments(a, y)
    coeff, info = hard_selection(g, b, c)
    np.testing.assert_array_equal(coeff, target)
    assert info["selected_candidate"] == chosen
    for key, point in (("W0", [0, 0]), ("WP", [1, 0]), ("WM", [0, 1])):
        assert info["candidate_losses"][key] == pytest.approx(float(np.sum(abs(y - a @ point)**2)), abs=1e-14)


def test_hard_and_convex_exact_ties_prefer_declared_vertices():
    g = np.zeros((2, 2)); b = np.zeros(2)
    for solver in (hard_selection, convex_mixing):
        coeff, info = solver(g, b, 7.)
        np.testing.assert_array_equal(coeff, [0, 0])
        assert info["fitted_cv_loss"] == 7.
    g = np.eye(2); b = np.array([.5, .5])
    coeff, info = hard_selection(g, b, 3.)
    np.testing.assert_array_equal(coeff, [0, 0])
    assert list(info["candidate_losses"].values()) == [3., 3., 3.]


@pytest.mark.parametrize("target", [[.2, .3], [-.3, .4], [.4, -.3], [.8, .9], [-2., -3.], [2., -.7]])
def test_convex_active_set_beats_independent_dense_simplex_residual(target, monkeypatch):
    monkeypatch.setattr(np.linalg, "inv", lambda *a: pytest.fail("inverse forbidden"))
    monkeypatch.setattr(np.linalg, "pinv", lambda *a: pytest.fail("pinv forbidden"))
    design = np.array([[1 + 2j, .3 - 1j], [-.5 + .2j, 2 + .5j], [1j, -1 + 1j]])
    response = design @ target + np.array([.02j, .01, -.02])
    coeff, info = convex_mixing(*moments(design, response))
    # Independent oracle evaluates the actual complex residual, not Gram formulas.
    grid = np.linspace(0, 1, 401)
    points = np.asarray([(a, b) for a in grid for b in grid if a + b <= 1])
    losses = np.sum(abs(response[:, None] - design @ points.T)**2, axis=0)
    actual = float(np.sum(abs(response - design @ coeff)**2))
    assert actual <= losses.min() + 1e-12
    assert actual == pytest.approx(info["fitted_cv_loss"], abs=1e-12)
    assert not geometry(coeff)["outside_any"]
    assert not info["regularized"] and not info["black_box_optimizer_used"]


@pytest.mark.parametrize("target,expected,kind", [
    ([.2, .3], [.2, .3], "interior"),
    ([-.3, .4], [0., .4], "edge"),
    ([.4, -.3], [.4, 0.], "edge"),
    ([.8, .9], [.45, .55], "edge"),
    ([-2., -3.], [0., 0.], "vertex"),
    ([2., -.7], [1., 0.], "vertex"),
    ([-.7, 2.], [0., 1.], "vertex"),
])
def test_euclidean_projection_covers_interior_three_edges_three_vertices(target, expected, kind):
    # Independent identity-design LS is Euclidean projection onto a triangle.
    response = np.asarray(target)
    coeff, info = convex_mixing(np.eye(2), response, float(response @ response))
    np.testing.assert_allclose(coeff, expected, atol=2e-16)
    assert info["active_kind"] == kind


@pytest.mark.parametrize("gram,rhs,expected", [
    (np.diag([1., 0.]), [.4, 0.], [.4, 0.]),
    (np.ones((2, 2)), [.7, .7], [0., .7]),
    (np.zeros((2, 2)), [0., 0.], [0., 0.]),
])
def test_rank_deficient_convex_global_optimum_on_boundary(gram, rhs, expected):
    coeff, info = convex_mixing(gram, rhs, 1.)
    np.testing.assert_allclose(coeff, expected, atol=1e-15)
    assert info["active_kind"] in ("edge", "vertex")


@pytest.mark.parametrize("coeff,kind,violations", [
    ([0., 0.], "vertex", (False, False, False)),
    ([1., 0.], "vertex", (False, False, False)),
    ([.2, 0.], "edge", (False, False, False)),
    ([.4, .6], "edge", (False, False, False)),
    ([.2, .3], "interior", (False, False, False)),
    ([-np.finfo(float).tiny, .3], "outside", (True, False, False)),
    ([.1, -np.finfo(float).tiny], "outside", (False, True, False)),
    ([1., 1.], "outside", (False, False, True)),
])
def test_weight_geometry_uses_exact_signs_without_affine_clipping(coeff, kind, violations):
    original = np.asarray(coeff).copy(); info = geometry(original)
    np.testing.assert_array_equal(original, coeff)
    assert info["active_kind"] == kind
    assert (info["outside_alpha"], info["outside_beta"], info["outside_lambda0"]) == violations
    assert info["outside_any"] == any(violations)
    assert info["lambda0"] == 1. - coeff[0] - coeff[1]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_objectives_weights_and_gains_fail(bad):
    with pytest.raises(ValueError): hard_selection(np.eye(2), [bad, 0], 1)
    with pytest.raises(ValueError): convex_mixing(np.eye(2), [0, 0], bad)
    with pytest.raises(ValueError): geometry([bad, 0])
    with pytest.raises(ValueError): gain(-10, bad, -12)


def test_indefinite_gram_rejected_and_no_estimator_label_truth_inputs():
    with pytest.raises(ValueError, match="semidefinite"):
        convex_mixing([[1, 2], [2, 1]], [0, 0], 1)
    for function in (objective_moments, hard_selection, convex_mixing):
        assert not any(token in name for name in inspect.signature(function).parameters
            for token in ("label", "environment", "target", "truth", "oracle"))


def gate_rows(b120_convex=(1., 1.), c75_convex=(-.25, 0.), hard=(1., 1.)):
    rows = {}
    for name, gains in (("CDL-C-75", c75_convex), ("CDL-B-120", b120_convex)):
        for j, snr in enumerate((10, 20)):
            rows[(name, snr)] = dict(affine_vs_hard_gain_db=hard[j], affine_vs_convex_gain_db=gains[j])
    return rows


@pytest.mark.parametrize("b120,c75,expected", [
    ((1., 1.), (-.25, 0.), "material"),
    ((.249, .249), (-1., -1.), "weak"),
    ((.25, .25), (0., 0.), "partial"),
    ((1., 1.), (-.25001, 0.), "partial"),
])
def test_fixed_scientific_thresholds_and_strict_geometry_gate(b120, c75, expected):
    outside = {("CDL-B-120", 10): .75001, ("CDL-B-120", 20): 1.}
    result = scientific_gate(gate_rows(b120, c75), outside)
    assert result[f"affine_composition_value_{expected}"]
    assert sum(result[k] for k in ("affine_composition_value_material", "affine_composition_value_weak", "affine_composition_value_partial")) == 1
    assert result["continuous_composition_beats_discrete_selection"]
    assert result["composed_shift_requires_extrapolation_empirically"]
    outside[("CDL-B-120", 10)] = .75
    assert not scientific_gate(gate_rows(b120, c75, hard=(.999, 1.)), outside)["composed_shift_requires_extrapolation_empirically"]
    assert not scientific_gate(gate_rows(b120, c75, hard=(.999, 1.)), outside)["continuous_composition_beats_discrete_selection"]


def test_gain_formula_and_gate_reject_missing_or_nonfinite_primary_results():
    assert gain(-10, -11, -12) == dict(affine_vs_hard_gain_db=2., affine_vs_convex_gain_db=1.)
    outside = {("CDL-B-120", 10): .9, ("CDL-B-120", 20): .9}
    rows = gate_rows(); rows.pop(("CDL-C-75", 10))
    with pytest.raises(ValueError): scientific_gate(rows, outside)
    rows = gate_rows(); rows[("CDL-C-75", 10)]["affine_vs_convex_gain_db"] = np.nan
    with pytest.raises(ValueError): scientific_gate(rows, outside)
    with pytest.raises(ValueError): scientific_gate(gate_rows(), {("CDL-B-120", 10): .9})
    with pytest.raises(ValueError): scientific_gate(gate_rows(), {k: np.inf for k in outside})
