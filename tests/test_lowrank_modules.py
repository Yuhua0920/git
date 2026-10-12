"""Independent complex128 T16G mathematical sentinels; no channel generation."""
import inspect

import numpy as np
import pytest
from scipy.linalg import solve_triangular

from src.lowrank_modules import (
    PRIMARY_CONDITIONS, RANKS, SENTINEL_RANK, RankFactors, WeightedModule,
    complexity, compression_metrics, factored_affine_predict,
    minimum_practical_rank, scientific_gate,
)

CALIBRATION_SEED = 20261607


def random_complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def fixture(ndata=11, npilot=6):
    calibration_rng = np.random.default_rng(CALIBRATION_SEED)
    delta = random_complex(calibration_rng, (ndata, npilot))
    matrix = random_complex(calibration_rng, (npilot, npilot))
    covariance = matrix @ matrix.conj().T + .7 * np.eye(npilot)
    return WeightedModule.build(delta, covariance)


@pytest.mark.parametrize("rank", [1, 2, 4, 6])
def test_independent_weighted_trace_and_singular_tail_sentinel(rank):
    module = fixture()
    factors = module.factors(rank)
    residual = module.delta - factors.A @ factors.Q
    energy = np.linalg.norm(residual @ module.lower) ** 2
    trace = np.trace(residual @ module.covariance @ residual.conj().T).real
    tail = np.sum(module.singular_values[rank:] ** 2)
    np.testing.assert_allclose(energy, trace, rtol=2e-12, atol=1e-24)
    np.testing.assert_allclose(energy, tail, rtol=2e-12, atol=1e-24)
    assert factors.A.dtype == factors.Q.dtype == np.complex128


def test_complex_covariance_requires_ordinary_transpose_right_solve():
    module = fixture()
    assert np.linalg.norm(module.lower.imag) > 1
    np.testing.assert_allclose(module.Q_full @ module.lower, module.right_h,
        rtol=2e-12, atol=2e-12)
    wrong = solve_triangular(module.lower.conj().T, module.right_h.T, lower=False).T
    assert np.linalg.norm(wrong @ module.lower - module.right_h) > .1
    assert abs(np.linalg.norm(module.delta @ module.lower) ** 2 -
        np.linalg.norm(module.delta @ module.lower.conj().T) ** 2) > 1


@pytest.mark.parametrize("rank", [1, 2, 4, 6])
def test_factorized_application_equals_independent_dense_test_only(rank):
    module = fixture()
    factors = module.factors(rank)
    pilot = np.arange(6) + 1j * np.arange(6)[::-1]
    np.testing.assert_allclose(factors.apply(pilot), (factors.A @ factors.Q) @ pilot,
        rtol=2e-12, atol=2e-12)


def test_rank64_math_sentinel_restores_original_delta_without_validation():
    module = fixture(ndata=71, npilot=SENTINEL_RANK)
    factors = module.factors(SENTINEL_RANK)
    restored = factors.A @ factors.Q
    assert np.linalg.norm(restored - module.delta) / np.linalg.norm(module.delta) < 1e-12
    diagnostics = module.diagnostics()
    assert diagnostics["covariance_min_eigenvalue"] > 0
    for key, value in diagnostics.items():
        if "residual" in key:
            assert value < 1e-12
    assert 0 <= module.svd_seconds <= module.construction_seconds


def test_capture_is_monotone_and_matches_independent_weighted_energy():
    module = fixture()
    capture = [module.capture(rank) for rank in range(1, 7)]
    assert np.all(np.diff(capture) >= 0)
    assert capture[-1] == 1
    denominator = np.linalg.norm(module.delta @ module.lower) ** 2
    for rank, fraction in enumerate(capture, 1):
        factor = module.factors(rank)
        energy = np.linalg.norm((factor.A @ factor.Q) @ module.lower) ** 2
        np.testing.assert_allclose(fraction, energy / denominator, rtol=2e-12)


def test_cholesky_requires_pd_and_does_not_silently_repair_covariance():
    delta = np.ones((5, 3), dtype=np.complex128)
    for covariance in (-np.eye(3), np.diag([1., 1., 0.])):
        with pytest.raises(np.linalg.LinAlgError):
            WeightedModule.build(delta, covariance)
    nonhermitian = np.eye(3, dtype=np.complex128)
    nonhermitian[0, 1] = .2j
    with pytest.raises(ValueError, match="Hermitian"):
        WeightedModule.build(delta, nonhermitian)
    assert nonhermitian[0, 1] == .2j and nonhermitian[1, 0] == 0


@pytest.mark.parametrize("delta,covariance", [
    (np.ones(3), np.eye(3)),
    (np.ones((0, 3)), np.eye(3)),
    (np.ones((4, 3)), np.eye(4)),
    (np.ones((4, 3)), np.ones(3)),
    (np.full((4, 3), np.nan), np.eye(3)),
    (np.ones((4, 3)), np.full((3, 3), np.inf)),
])
def test_invalid_construction_inputs_fail(delta, covariance):
    with pytest.raises(ValueError):
        WeightedModule.build(delta, covariance)


@pytest.mark.parametrize("rank", [-1, 0, 7, True, np.bool_(False), 1.5])
def test_invalid_factor_rank_fails(rank):
    with pytest.raises(ValueError):
        fixture().factors(rank)


def test_all_saved_factorization_arrays_are_readonly_owned_copies():
    delta = np.ones((5, 3), dtype=np.complex128)
    covariance = np.eye(3, dtype=np.complex128)
    module = WeightedModule.build(delta, covariance)
    delta[:] = 100
    covariance[:] = 200
    assert np.all(module.delta == 1) and np.array_equal(module.covariance, np.eye(3))
    factors = module.factors(2)
    for array in (module.delta, module.covariance, module.lower, module.singular_values,
                  module.A_full, module.Q_full, module.right_h, factors.A, factors.Q):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0


def test_constructor_svd_uses_only_supplied_development_delta_and_covariance(monkeypatch):
    module = fixture()
    development_delta = module.delta.copy()
    development_covariance = module.covariance.copy()
    unavailable_target_covariance = development_covariance * 17
    observed = []
    original_svd = np.linalg.svd
    def record(matrix, *args, **kwargs):
        observed.append(matrix.copy())
        return original_svd(matrix, *args, **kwargs)
    monkeypatch.setattr(np.linalg, "svd", record)
    built = WeightedModule.build(development_delta, development_covariance)
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], development_delta @ built.lower)
    assert not np.array_equal(observed[0], development_delta @
        np.linalg.cholesky(unavailable_target_covariance))
    assert list(inspect.signature(WeightedModule.build).parameters) == ["delta", "covariance"]
    source = inspect.getsource(WeightedModule.build)
    assert all(word not in source for word in ("target", "condition", "open(", "load(", "Path("))


def test_build_never_uses_inverse_or_pseudoinverse(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit inverse or pseudoinverse forbidden")
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    monkeypatch.setattr(np.linalg, "pinv", forbidden)
    module = fixture()
    assert module.diagnostics()["triangular_right_solve_relative_residual"] < 1e-12


def test_heldout_residual_is_evaluation_only_and_cannot_refit(monkeypatch):
    module = fixture()
    original_a, original_q = module.A_full.copy(), module.Q_full.copy()
    target_covariance = module.covariance + np.diag(np.arange(1, 7))
    lower = np.linalg.cholesky(target_covariance)
    factors = module.factors(2)
    reference = (np.linalg.norm((module.delta - factors.A @ factors.Q) @ lower) ** 2 /
        np.linalg.norm(module.delta @ lower) ** 2)
    def forbidden(*args, **kwargs):
        raise AssertionError("held-out residual must not call SVD or build")
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    monkeypatch.setattr(WeightedModule, "build", forbidden)
    assert module.heldout_residual(2, target_covariance) == pytest.approx(reference, rel=2e-12)
    np.testing.assert_array_equal(module.A_full, original_a)
    np.testing.assert_array_equal(module.Q_full, original_q)


def test_zero_module_prediction_defined_but_relative_energies_are_undefined():
    module = WeightedModule.build(np.zeros((5, 3)), np.eye(3))
    np.testing.assert_array_equal(module.factors(2).apply(np.ones(3)), np.zeros(5))
    with pytest.raises(ValueError, match="undefined"):
        module.capture(2)
    with pytest.raises(ValueError, match="undefined"):
        module.heldout_residual(2, np.eye(3))


def test_factored_estimator_uses_only_source_means_and_frozen_real_coordinates():
    module = fixture()
    rng = np.random.default_rng(CALIBRATION_SEED + 1)
    source = random_complex(rng, (11, 6))
    mean_p, mean_d = random_complex(rng, (6,)), random_complex(rng, (11,))
    coefficients = np.array([-.37, 1.61])
    factors = module.factors(2)
    np.testing.assert_array_equal(factored_affine_predict(mean_p, mean_p, mean_d,
        source, coefficients, factors, factors), mean_d)
    pilot = random_complex(rng, (6,))
    x = pilot - mean_p
    expected = (mean_d + x @ source.T + coefficients[0] * ((factors.A @ factors.Q) @ x)
        + coefficients[1] * ((factors.A @ factors.Q) @ x))
    result = factored_affine_predict(pilot, mean_p, mean_d, source, coefficients, factors, factors)
    np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(coefficients, [-.37, 1.61])
    np.testing.assert_array_equal(factored_affine_predict(pilot, mean_p, mean_d,
        source, [0., 0.], factors, factors), mean_d + x @ source.T)
    assert "target" not in inspect.signature(factored_affine_predict).parameters


def test_actual_prediction_forbids_dense_AQ_reconstruction():
    class GuardArray(np.ndarray):
        def __matmul__(self, other):
            if self.ndim == 2 and np.asarray(other).ndim == 2:
                raise AssertionError("dense matrix reconstruction inside inference")
            return super().__matmul__(other)
    module = fixture()
    factors = module.factors(2)
    object.__setattr__(factors, "A", factors.A.view(GuardArray))
    object.__setattr__(factors, "Q", factors.Q.view(GuardArray))
    pilot = np.arange(6) + .1j
    result = factored_affine_predict(pilot, np.zeros(6), np.zeros(11), np.zeros((11, 6)),
        [1., -.2], factors, factors)
    assert result.shape == (11,) and np.isfinite(result).all()
    with pytest.raises(AssertionError, match="dense"):
        factors.A @ factors.Q


@pytest.mark.parametrize("coefficients", [[1.], [1., np.nan], [1.+0j, 2.+0j]])
def test_invalid_prediction_coefficients_fail(coefficients):
    factors = fixture().factors(2)
    with pytest.raises(ValueError):
        factored_affine_predict(np.ones(6), np.zeros(6), np.zeros(11), np.zeros((11, 6)),
            coefficients, factors, factors)


def test_invalid_prediction_dimensions_and_factor_inputs_fail():
    factors = fixture().factors(2)
    with pytest.raises(ValueError):
        factors.apply(np.ones((1, 6)))
    with pytest.raises(ValueError):
        RankFactors(np.ones((3, 2)), np.ones((3, 4)))
    with pytest.raises(ValueError):
        factored_affine_predict(np.ones(6), np.zeros(6), np.zeros(11), np.zeros((11, 6)),
            [1., 2.], fixture(ndata=10).factors(2), factors)


def test_compression_penalty_recovery_and_gain_follow_unclipped_definitions():
    result = compression_metrics(-9, -21, -16, -20.5)
    assert result == dict(compression_penalty_db=.5, dense_recovery_db=12.,
        lowrank_recovery_db=11.5, recovery_preservation=11.5 / 12,
        lowrank_affine_vs_convex_gain_db=4.5)
    assert compression_metrics(-9, -21, -16, -22)["recovery_preservation"] > 1
    assert compression_metrics(-9, -21, -16, -8)["recovery_preservation"] < 0
    assert compression_metrics(-9, -21, -16, -22)["compression_penalty_db"] == -1
    with pytest.raises(ValueError, match="undefined"):
        compression_metrics(-9, -9, -16, -21)
    with pytest.raises(ValueError, match="finite"):
        compression_metrics(-9, -21, -16, np.nan)


def test_minimum_practical_ranks_use_frozen_grid_and_nonmonotone_penalties():
    penalties = {1: .4, 2: .5, 4: .24, 8: .8, 16: .2}
    assert minimum_practical_rank(penalties) == dict(r_025db=4, r_05db=1)
    rows = [dict(rank=r, compression_penalty_db=penalties[r]) for r in reversed(RANKS)]
    assert minimum_practical_rank(rows) == dict(r_025db=4, r_05db=1)
    assert minimum_practical_rank({r: 1. for r in RANKS}) == dict(r_025db=None, r_05db=None)
    with pytest.raises(ValueError):
        minimum_practical_rank({1: .1, 3: .2})
    with pytest.raises(ValueError):
        minimum_practical_rank(rows + rows[:1])


@pytest.mark.parametrize("rank,correction", [(1, 8064), (2, 16128), (4, 32256), (8, 64512), (16, 129024)])
def test_cost_and_storage_formulas_include_source_and_pilot_cv(rank, correction):
    result = complexity(rank)
    assert result["single_module_complex_MAC_per_frame"] == 4032 * rank
    assert result["correction_complex_MAC_per_frame"] == correction
    assert result["source_complex_MAC_per_frame"] == 253952
    assert result["pilot_cv_complex_MAC_per_current_window"] == 46080
    assert result["total_arithmetic_proxy"] == 300032 + correction
    assert result["dense_affine_total_arithmetic_proxy"] == 807936
    assert result["correction_over_source_ratio"] == correction / 253952
    assert result["total_over_source_ratio"] == (300032 + correction) / 253952
    assert result["saving_vs_dense_affine_percent"] == 100 * (507904 - correction) / 807936
    assert result["factor_complex_coefficient_count"] == correction
    assert result["factor_complex64_bytes"] == 8 * correction
    assert result["factor_complex128_bytes"] == 16 * correction
    assert result["dense_module_complex_coefficient_count"] == 507904
    assert result["dense_module_complex64_bytes"] == 4063232
    assert result["dense_module_complex128_bytes"] == 8126464


def gate_rows(penalty=.4, gain=1.1):
    return [dict(shift=shift, snr_db=snr, rank=rank, compression_penalty_db=penalty,
        lowrank_affine_vs_convex_gain_db=gain) for rank in RANKS
        for shift, snr in sorted(PRIMARY_CONDITIONS)]


@pytest.mark.parametrize("expected", ["strong", "moderate", "weak", "partial"])
def test_preregistered_gate_categories_and_exact_thresholds(expected):
    rows = gate_rows(penalty=.5, gain=1.)
    if expected in ("moderate", "weak", "partial"):
        for row in rows:
            if row["rank"] == 4:
                row["compression_penalty_db"] = np.nextafter(.5, np.inf)
    if expected in ("weak", "partial"):
        for row in rows:
            if row["rank"] == 8:
                row["lowrank_affine_vs_convex_gain_db"] = np.nextafter(1., -np.inf)
    if expected == "weak":
        next(row for row in rows if row["rank"] == 16)["compression_penalty_db"] = np.nextafter(1., np.inf)
    if expected == "partial":
        for row in rows:
            if row["rank"] == 16:
                row["compression_penalty_db"] = 1.
    result = scientific_gate(rows)
    assert result[f"lowrank_affine_modules_{expected}"]
    assert sum(result.values()) == 1


def test_gate_does_not_add_unregistered_strong_guard_to_weak():
    rows = gate_rows()
    next(row for row in rows if row["rank"] == 16)["compression_penalty_db"] = 1.01
    result = scientific_gate(rows)
    assert result["lowrank_affine_modules_strong"]
    assert result["lowrank_affine_modules_weak"]
    assert not result["lowrank_affine_modules_partial"]


def test_gate_only_requires_b120_gain_threshold_c75_uses_penalty():
    rows = gate_rows()
    for row in rows:
        if row["shift"] == "CDL-C-75":
            row["lowrank_affine_vs_convex_gain_db"] = -10.
    assert scientific_gate(rows)["lowrank_affine_modules_strong"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nan", "foreign_rank"])
def test_gate_input_validation(mutation):
    rows = gate_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "nan":
        rows[0]["compression_penalty_db"] = np.nan
    else:
        rows[0]["rank"] = 3
    with pytest.raises(ValueError):
        scientific_gate(rows)
