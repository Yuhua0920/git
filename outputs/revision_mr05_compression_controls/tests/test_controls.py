"""Small deterministic mathematical controls tests; no scientific corpus IO."""
import numpy as np
import pytest

from controls import ATOL, RTOL, CORRECTION_METHODS, WHOLE_METHODS, deployed, weighted_factor
from src.lowrank_modules import RankFactors, WeightedModule


def complex_toy():
    rng = np.random.default_rng(1705001)
    B = rng.normal(size=(7, 4)) + 1j * rng.normal(size=(7, 4))
    Z = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    C = Z @ Z.conj().T + 0.7 * np.eye(4)
    u = rng.normal(size=4) + 1j * rng.normal(size=4)
    return B, C, u


def assert_equal(a, b):
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


def toy_operators():
    rng = np.random.default_rng(1705002)
    make = lambda shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
    Ws, BP, BM, Wpool = (make((7, 4)) for _ in range(4))
    return dict(Wsource_10=Ws, DeltaWP_10=BP, DeltaWM_10=BM, Wpool_10=Wpool,
                source_mean_p=make(4), source_mean_d=make(7), pool_mean_p=make(4), pool_mean_d=make(7)), make(4)


def test_complex_cholesky_and_right_solve_detect_wrong_conjugation():
    B, C, u = complex_toy()
    module = WeightedModule.build(B, C)
    f, a = weighted_factor(B, C, 2, module=module)
    assert np.max(np.abs(module.lower.imag)) > 0.01
    assert_equal(module.lower @ module.lower.conj().T, C)
    assert_equal(f.Q @ module.lower, module.right_h[:2])
    assert_equal(f.apply(u), (f.A @ f.Q) @ u)
    wrong_Q = np.linalg.solve(module.lower.conj().T, module.right_h[:2].T).T
    assert np.linalg.norm(wrong_Q @ module.lower - module.right_h[:2]) > 0.1
    assert a["right_solve_relative_residual"] < RTOL
    assert a["factorization_reused"] and a["construction_seconds"] == 0.0


def test_identity_weight_matches_ordinary_truncated_svd():
    B, _, u = complex_toy()
    f, _ = weighted_factor(B, np.eye(4), 2)
    U, s, Vh = np.linalg.svd(B, full_matrices=False)
    expected = (U[:, :2] * s[:2]) @ Vh[:2]
    assert_equal(f.A @ f.Q, expected)
    assert_equal(f.apply(u), expected @ u)


def test_source_centered_second_moment_includes_between_origin_means_and_noise_once():
    rng = np.random.default_rng(1705003)
    samples = [rng.normal(size=(n, 4)) + 1j * rng.normal(size=(n, 4)) + (i + 1j * i)
               for i, n in enumerate((2, 3, 5))]
    means = [p.mean(axis=0) for p in samples]
    covariances = [(p - mp).T @ (p - mp).conj() / len(p) for p, mp in zip(samples, means)]
    weights = np.array([len(p) for p in samples], dtype=float)
    weights /= weights.sum()
    pool_mean = sum(w * mp for w, mp in zip(weights, means))
    pool_rpp = sum(w * (R + np.outer(mp - pool_mean, (mp - pool_mean).conj()))
                   for w, mp, R in zip(weights, means, covariances))
    source_mean, sigma2 = means[0], 0.07
    mean_shift = pool_mean - source_mean
    C_pool_m2 = pool_rpp + sigma2 * np.eye(4) + np.outer(mean_shift, mean_shift.conj())
    all_frames = np.concatenate(samples)
    centered = all_frames - source_mean
    direct = centered.T @ centered.conj() / len(centered) + sigma2 * np.eye(4)
    assert_equal(C_pool_m2, direct)
    missing_between = sum(w * R for w, R in zip(weights, covariances)) + sigma2 * np.eye(4) + np.outer(mean_shift, mean_shift.conj())
    assert np.linalg.norm(missing_between - direct) > 0.1
    assert np.linalg.norm((C_pool_m2 + sigma2 * np.eye(4)) - direct) > 0.01
    for p, mp, R in zip(samples, means, covariances):
        d = mp - source_mean
        source_centered = R + sigma2 * np.eye(4) + np.outer(d, d.conj())
        pp = p - source_mean
        assert_equal(source_centered, pp.T @ pp.conj() / len(pp) + sigma2 * np.eye(4))


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_weighted_tail_and_full_rank(rank):
    B, C, _ = complex_toy()
    module = WeightedModule.build(B, C)
    f, audit = weighted_factor(B, C, rank, module=module)
    exact_tail = np.sum(module.singular_values[rank:] ** 2)
    measured = np.linalg.norm((B - f.A @ f.Q) @ module.lower) ** 2
    assert abs(measured - exact_tail) <= ATOL + RTOL * audit["weighted_total_energy"]
    assert audit["weighted_tail_absolute_error"] <= audit["weighted_tail_acceptance_threshold"]
    if rank == 4:
        assert_equal(f.A @ f.Q, B)
        assert audit["boundary_singular_gap"] is None


def test_reuse_same_svd_for_both_rank_prefixes_and_reject_changed_binding():
    B, C, _ = complex_toy()
    module = WeightedModule.build(B, C)
    f2, a2 = weighted_factor(B, C, 2, module=module)
    f3, a3 = weighted_factor(B.copy(), C.copy(), 3, module=module)
    assert_equal(f2.A, f3.A[:, :2])
    assert_equal(f2.Q, f3.Q[:2])
    assert a2["original_module_construction_seconds"] == a3["original_module_construction_seconds"]
    altered = B.copy()
    altered[0, 0] += 1e-12
    with pytest.raises(ValueError, match="MODULE_MATRIX_WEIGHT_BINDING"):
        weighted_factor(altered, C, 2, module=module)
    with pytest.raises(ValueError, match="MODULE_MATRIX_WEIGHT_BINDING"):
        weighted_factor(B, C + 1e-12 * np.eye(4), 2, module=module)


def test_reconstruction_is_phase_invariant_not_factorwise_identity():
    B, C, u = complex_toy()
    f, _ = weighted_factor(B, C, 2)
    phases = np.exp(1j * np.array([0.3, 1.2]))
    rotated = RankFactors(f.A * phases, phases.conj()[:, None] * f.Q)
    assert not np.array_equal(f.A, rotated.A)
    assert_equal(f.A @ f.Q, rotated.A @ rotated.Q)
    assert_equal(f.apply(u), rotated.apply(u))


def test_cutoff_degeneracy_reported_without_claiming_unique_operator():
    f, audit = weighted_factor(np.diag([3.0, 1.0, 1.0, 0.2]).astype(np.complex128), np.eye(4), 2)
    assert audit["boundary_singular_gap"] == 0.0
    alternatives = [np.diag([3.0, 1.0, 0.0, 0.0]), np.diag([3.0, 0.0, 1.0, 0.0])]
    assert not np.array_equal(alternatives[0], alternatives[1])
    for alternative in alternatives:
        assert_equal(np.linalg.norm(np.diag([3.0, 1.0, 1.0, 0.2]) - alternative) ** 2, audit["weighted_tail_energy"])
    # Either prefix basis, or a rotated one, is a valid optimizer at this tie.
    assert audit["weighted_tail_absolute_error"] <= audit["weighted_tail_acceptance_threshold"]


def test_zero_matrix_tail_audit_is_well_defined():
    f, a = weighted_factor(np.zeros((7, 4), np.complex128), np.eye(4), 2)
    assert_equal(f.A @ f.Q, np.zeros((7, 4)))
    assert a["weighted_tail_energy"] == 0.0 and a["weighted_residual_energy"] == 0.0
    assert a["weighted_tail_error_over_total"] == 0.0


@pytest.mark.parametrize("bad_rank", [0, 5, True, 2.0])
def test_invalid_rank_rejected(bad_rank):
    B, C, _ = complex_toy()
    with pytest.raises(ValueError, match="RANK_REQUIRED"):
        weighted_factor(B, C, bad_rank)


def test_invalid_weights_and_nonfinite_values_rejected():
    B, C, _ = complex_toy()
    wrong = C.copy()
    wrong[0, 1] += 0.01j
    with pytest.raises(ValueError, match="Hermitian"):
        weighted_factor(B, wrong, 2)
    with pytest.raises(np.linalg.LinAlgError):
        weighted_factor(B, -np.eye(4), 2)
    with pytest.raises(ValueError, match="WEIGHT_SHAPE"):
        weighted_factor(B, np.eye(3), 2)
    bad = B.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="MATRIX_FINITE"):
        weighted_factor(bad, C, 2)


def test_source_and_pooled_natural_mean_contract():
    op, u = toy_operators()
    before = {name: value.copy() for name, value in op.items()}
    saved_u = u.copy()
    actual_source = deployed("SOURCE", u, None, op, {}, 10)
    actual_pool = deployed("POOLED_NATURAL", u, None, op, {}, 10)
    assert_equal(actual_source, op["source_mean_d"] + op["Wsource_10"] @ u)
    p = u + op["source_mean_p"]
    expected_pool = op["pool_mean_d"] + op["Wpool_10"] @ (p - op["pool_mean_p"])
    assert_equal(actual_pool, expected_pool)
    wrong_pool = op["pool_mean_d"] + op["Wpool_10"] @ u
    assert np.linalg.norm(actual_pool - wrong_pool) > 0.1
    for name in op:
        assert np.array_equal(op[name], before[name])
    assert np.array_equal(u, saved_u)


def test_whole_full_rank_restores_dense_with_unrestricted_coordinates():
    op, u = toy_operators()
    _, C, _ = complex_toy()
    factors = tuple(weighted_factor(W, C, 4)[0] for W in
                    (op["Wsource_10"], op["Wsource_10"] + op["DeltaWP_10"], op["Wsource_10"] + op["DeltaWM_10"]))
    coeff = np.array([-0.7, 2.1])
    saved_c = coeff.copy()
    dense = deployed("DENSE_TW5", u, coeff, op, {}, 10)
    for name in WHOLE_METHODS:
        assert_equal(deployed(name, u, coeff, op, {name: factors}, 10), dense)
    assert np.array_equal(coeff, saved_c)


def test_zero_coordinates_corr_keeps_source_whole_keeps_compressed_source():
    op, u = toy_operators()
    _, C, _ = complex_toy()
    correction = tuple(weighted_factor(op[name], C, 2)[0] for name in ("DeltaWP_10", "DeltaWM_10"))
    whole = tuple(weighted_factor(W, C, 2)[0] for W in
                  (op["Wsource_10"], op["Wsource_10"] + op["DeltaWP_10"], op["Wsource_10"] + op["DeltaWM_10"]))
    source = deployed("SOURCE", u, None, op, {}, 10)
    for name in CORRECTION_METHODS:
        assert_equal(deployed(name, u, np.zeros(2), op, {name: correction}, 10), source)
    expected_whole = op["source_mean_d"] + whole[0].apply(u)
    for name in WHOLE_METHODS:
        actual = deployed(name, u, np.zeros(2), op, {name: whole}, 10)
        assert_equal(actual, expected_whole)
        assert np.linalg.norm(actual - source) > 0.1


@pytest.mark.parametrize("truth", [np.array([2.0, 0.0]), np.array([2.0, 1.0])])
def test_shared_coordinate_signed_A_plus_B_and_nonmonotone_target_risk(truth):
    B = np.diag([2.0, 1.0]).astype(np.complex128)
    zero = np.zeros((2, 2), np.complex128)
    op = dict(Wsource_10=zero, DeltaWP_10=B, DeltaWM_10=zero, source_mean_d=np.zeros(2))
    factors = (weighted_factor(B, np.eye(2), 1)[0], weighted_factor(zero, np.eye(2), 1)[0])
    u, coeff = np.ones(2), np.array([1.0, 0.0])
    hD = deployed("DENSE_TW5", u, coeff, op, {}, 10)
    hC = deployed("CORR_I_R4", u, coeff, op, {"CORR_I_R4": factors}, 10)
    eD, delta = hD - truth, hC - hD
    A, cross = np.vdot(delta, delta).real, 2 * np.vdot(eD, delta).real
    direct = np.vdot(hC - truth, hC - truth).real - np.vdot(eD, eD).real
    assert_equal(direct, A + cross)
    if truth[1] == 0.0:
        assert cross < 0.0 and direct < 0.0
        assert np.linalg.norm(hC - truth) < np.linalg.norm(hD - truth)
    else:
        assert direct > 0.0


def test_deployment_rejects_complex_coords_unknown_method_and_wrong_factor_shape():
    op, u = toy_operators()
    with pytest.raises(ValueError, match="REAL_COORDINATES_REQUIRED"):
        deployed("DENSE_TW5", u, np.array([1.0 + 0j, 0.0]), op, {}, 10)
    with pytest.raises(ValueError, match="UNKNOWN_CONTROL_METHOD"):
        deployed("NOT_A_METHOD", u, np.zeros(2), op, {}, 10)
    wrong = RankFactors(np.ones((6, 2)), np.ones((2, 4)))
    with pytest.raises(ValueError, match="FACTOR_BANK_SHAPE"):
        deployed("CORR_I_R4", u, np.ones(2), op, {"CORR_I_R4": (wrong, wrong)}, 10)
