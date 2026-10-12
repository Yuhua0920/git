"""T10A mathematical sentinels, reuse/leakage checks and archived calibration."""

import ast
import csv
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import src.lmmse as lmmse
from src.doppler import (build_tdl_a_doppler_channel, jakes_covariance,
                         mobility_parameters, ofdm_time_grid_seconds)
from src.estimators import pilot_ls_estimate
from src.link import PILOT_SYMBOLS, apply_frequency_mimo_channel
from src.numerics import complex_gaussian_noise, noise_variance_from_snr
from src.pilots import build_fdm_pilot_layout, map_pilot_symbols
from src.tdl import centered_frequency_grid_hz, tdl_a_profile


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = [(d_f, snr, q) for d_f in (2, 4, 8, 16) for snr in (0, 5, 10, 15, 20) for q in (0, 1)]


@pytest.fixture(scope="module")
def covariances():
    rf = lmmse.build_frequency_covariance()
    rt = lmmse.build_time_covariance()
    return rf, rt, lmmse.build_full_grid_covariance(rf, rt)


@pytest.fixture(scope="module")
def prepared(covariances):
    return {(d_f, snr, q): lmmse.prepare_oracle_lmmse(covariances[2],
            lmmse.pilot_flat_indices(build_fdm_pilot_layout(2, 64, 16, d_f, 4).pilot_mask[q]),
            noise_variance_from_snr(snr, 1.0), (64, 16)) for d_f, snr, q in CONDITIONS}


def test_frequency_covariance(covariances):
    rf = covariances[0]
    assert rf.shape == (64, 64) and rf.dtype == np.complex128
    np.testing.assert_allclose(rf, rf.conj().T, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.diag(rf), 1, rtol=0, atol=1e-14)
    assert np.linalg.eigvalsh(rf)[0] >= -1e-12
    # Scalar calculation with an independent conjugation-sensitive difference.
    delays, powers = tdl_a_profile()
    f = centered_frequency_grid_hz(64, 30000)
    expected = sum(p * np.exp(-2j * np.pi * (f[4] - f[53]) * delay)
                   for p, delay in zip(powers, delays))
    assert abs(rf[4, 53] - expected) <= 1e-14


def test_time_covariance_matches_t08b(covariances):
    expected = jakes_covariance(ofdm_time_grid_seconds(), mobility_parameters()["maximum_doppler_hz"])
    np.testing.assert_array_equal(covariances[1], expected)
    assert covariances[1].shape == (16, 16)
    assert np.linalg.eigvalsh(expected)[0] >= -1e-12


def test_full_covariance_basic_checks(covariances):
    full = covariances[2]
    assert full.shape == (1024, 1024) and np.all(np.isfinite(full))
    np.testing.assert_allclose(full, full.conj().T, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.diag(full), 1, rtol=0, atol=1e-14)


@pytest.mark.parametrize("pair", [((32, 0), (32, 4)), ((32, 0), (40, 0)), ((32, 0), (40, 4))])
def test_kronecker_order(covariances, pair):
    rf, rt, full = covariances
    (k, t), (kp, tp) = pair
    assert full[k * 16 + t, kp * 16 + tp] == rf[k, kp] * rt[t, tp]
    grid = np.arange(1024).reshape(64, 16)
    assert grid[k, t] == k * 16 + t
    assert abs(np.kron(rt, rf)[k * 16 + t, kp * 16 + tp] - full[k * 16 + t, kp * 16 + tp]) > 1e-5


@pytest.mark.parametrize("d_f,count,offset", [(2, 128, 1), (4, 64, 2), (8, 32, 4), (16, 16, 8)])
def test_actual_pilot_indices(d_f, count, offset):
    layout = build_fdm_pilot_layout(2, 64, 16, d_f, 4)
    for q in range(2):
        indices = lmmse.pilot_flat_indices(layout.pilot_mask[q])
        np.testing.assert_array_equal(indices, np.flatnonzero(layout.pilot_mask[q].ravel(order="C")))
        assert indices.size == count and np.all(np.diff(indices) > 0)
        k, t = np.unravel_index(indices, (64, 16), order="C")
        assert np.all(k % d_f == q * offset)
        assert set(t) == {0, 4, 8, 12}


def test_tiny_analytic_and_noisy_pilot_shrinkage():
    rho, z, sigma2 = 0.3 + 0.4j, 1.2 - 0.7j, 0.25
    covariance = np.array([[1, rho.conjugate()], [rho, 1]])
    obj = lmmse.prepare_oracle_lmmse(covariance, np.array([0]), sigma2, (2, 1))
    actual = obj.estimate(np.array([z])).ravel()
    expected = np.array([z, rho * z]) / (1 + sigma2)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-16)
    np.testing.assert_allclose(obj.posterior_variance().ravel(),
                               [sigma2 / (1 + sigma2), 1 - abs(rho) ** 2 / (1 + sigma2)], atol=5e-16)
    assert abs(actual[0] - z) > 0.1


@pytest.mark.parametrize("condition", CONDITIONS)
def test_cholesky_and_posterior_all_conditions(prepared, condition):
    obj = prepared[condition]
    posterior = obj.posterior_variance()
    assert obj.diagnostics["cholesky_success"]
    assert obj.diagnostics["cholesky_diagonal_min"] > 0
    assert np.all(np.isfinite(posterior))
    assert np.min(posterior) >= -1e-12 and np.max(posterior) <= 1 + 1e-12
    assert not obj.diagnostics["posterior_clipped"]
    lower = np.tril(obj.cholesky_factor[0])
    np.testing.assert_allclose(lower @ lower.conj().T, obj.observation_covariance, atol=1e-13)


def test_prepared_reused_without_inverse_or_new_factorization(prepared, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected inverse or new factorization")
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    monkeypatch.setattr(np.linalg, "pinv", forbidden)
    monkeypatch.setattr(lmmse, "cho_factor", forbidden)
    obj = prepared[4, 10, 0]
    for seed in (30261001, 30261002):
        observations = complex_gaussian_noise((2, obj.pilot_indices.size), 1, np.random.default_rng(seed))
        batch = obj.estimate(observations)
        assert batch.shape == (2, 64, 16)
        for r in range(2):
            np.testing.assert_allclose(batch[r], obj.estimate(observations[r]), atol=1e-13)
    assert not obj.pilot_indices.flags.writeable and not obj.cholesky_factor[0].flags.writeable


def test_production_no_inverse_ast_and_no_truth_argument():
    tree = ast.parse(inspect.getsource(lmmse))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            assert called not in {"inv", "pinv", "inverse"}
    assert list(inspect.signature(lmmse.PreparedOracleLMMSE.estimate).parameters) == ["self", "pilot_observations"]
    assert "h_true" not in inspect.getsource(lmmse).lower()


@pytest.mark.parametrize("d_f,snr", [(2, 0), (4, 10), (8, 15), (16, 20)])
def test_actual_pilot_ls_integration_and_nonpilot_independence(prepared, d_f, snr):
    layout = build_fdm_pilot_layout(2, 64, 16, d_f, 4)
    _, channel = build_tdl_a_doppler_channel(40261001)
    x_grid = map_pilot_symbols(layout, PILOT_SYMBOLS)
    variance = noise_variance_from_snr(snr, 1.0)
    noise = complex_gaussian_noise((2, 64, 16), variance, np.random.default_rng(31261001 + d_f))
    ls, mask = pilot_ls_estimate(apply_frequency_mimo_channel(channel, x_grid) + noise, x_grid, layout.pilot_mask)
    for q in range(2):
        expected = channel[:, q, mask[q]] + noise[:, mask[q]] / x_grid[q, mask[q]]
        np.testing.assert_allclose(ls[:, q, mask[q]], expected, rtol=0, atol=1e-14)
    objects = tuple(prepared[d_f, snr, q] for q in range(2))
    estimate = lmmse.estimate_mimo_from_pilot_ls(ls, objects)
    corrupted = ls.copy()
    corrupted[np.broadcast_to(~mask, ls.shape)] = 1e9 + 2e9j
    np.testing.assert_array_equal(estimate, lmmse.estimate_mimo_from_pilot_ls(corrupted, objects))
    assert estimate.shape == (2, 2, 64, 16) and np.all(np.isfinite(estimate))


def test_fixed_channel_and_noise_seeds():
    _, a = build_tdl_a_doppler_channel(40261001)
    _, b = build_tdl_a_doppler_channel(40261001)
    _, c = build_tdl_a_doppler_channel(40261002)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    first = complex_gaussian_noise((2, 16), 0.1, np.random.default_rng(30261001))
    repeat = complex_gaussian_noise((2, 16), 0.1, np.random.default_rng(30261001))
    changed = complex_gaussian_noise((2, 16), 0.1, np.random.default_rng(30261002))
    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, changed)


def test_posterior_diagonal_against_independent_small_full_solve():
    transform = np.array([[1, 0.2j, 0.3], [0.3j, 1, -0.2j], [0.4, -0.1j, 0.8]])
    covariance = transform @ transform.conj().T
    indices = np.array([0, 2])
    variance = 0.2
    obj = lmmse.prepare_oracle_lmmse(covariance, indices, variance, (3, 1))
    a = covariance[np.ix_(indices, indices)] + variance * np.eye(2)
    full_error = covariance - covariance[:, indices] @ np.linalg.solve(a, covariance[indices, :])
    np.testing.assert_allclose(obj.posterior_variance().ravel(), np.diag(full_error).real, atol=5e-16)


def test_preparation_factorizes_once_and_does_not_use_inverse(monkeypatch):
    calls = []
    original = lmmse.cho_factor
    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected inverse")
    monkeypatch.setattr(lmmse, "cho_factor", tracked)
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    monkeypatch.setattr(np.linalg, "pinv", forbidden)
    obj = lmmse.prepare_oracle_lmmse(np.array([[1, 0.3j], [-0.3j, 1]]), np.array([0]), 0.1, (2, 1))
    for seed in range(3):
        obj.estimate(complex_gaussian_noise((2, 1), 1, np.random.default_rng(seed)))
        obj.posterior_variance()
    assert len(calls) == 1


def test_calibration_noise_seed_domains_are_disjoint():
    from scripts.validate_lmmse_oracle import calibration_noise_seed
    seeds = [calibration_noise_seed(ci, tid, r, q)
             for ci in range(4) for tid in range(500) for r in range(2) for q in range(2)]
    assert len(seeds) == len(set(seeds)) == 8000
    assert set(seeds).isdisjoint(range(40261001, 40261501))
    assert set(seeds).isdisjoint(31261001 + d_f for d_f in (2, 4, 8, 16))


@pytest.mark.parametrize("variance", [0, -1, np.nan, np.inf])
def test_invalid_noise_raises(variance):
    with pytest.raises(ValueError, match="strictly positive"):
        lmmse.prepare_oracle_lmmse(np.eye(2), np.array([0]), variance, (2, 1))


def test_failed_cholesky_has_diagnostics_without_fallback():
    with pytest.raises(lmmse.LMMSEPreparationError) as caught:
        lmmse.prepare_oracle_lmmse(np.array([[1, 2], [2, 1]]), np.array([0, 1]), 0.1, (2, 1))
    assert not caught.value.record["cholesky_success"]
    assert not caught.value.record["jitter_used"] and not caught.value.record["pseudo_inverse_used"]


@pytest.mark.parametrize("indices", [np.array([]), np.array([1, 0]), np.array([0, 0]), np.array([2]), np.array([0.0])])
def test_invalid_indices_raise(indices):
    with pytest.raises(ValueError, match="pilot_indices"):
        lmmse.prepare_oracle_lmmse(np.eye(2), indices, 0.1, (2, 1))


def test_saved_mathematical_calibration():
    report = json.loads((ROOT / "outputs/t10a_lmmse_oracle_results.json").read_text(encoding="utf-8"))
    assert report["all_checks_passed"] and report["task"] == "T10A"
    assert len(report["theoretical_conditions"]) == 20
    assert report["prepared_estimator_count"] == 40
    assert report["monte_carlo_calibration"]["num_trajectories"] == 500
    assert report["monte_carlo_calibration"]["independent_link_realizations"] == 2000
    conditions = report["monte_carlo_calibration"]["conditions"]
    assert [(r["dF"], r["snr_db"]) for r in conditions] == [(2, 0), (4, 10), (8, 15), (16, 20)]
    for row in conditions:
        assert row["absolute_difference_db"] <= 0.5
        assert row["data_nmse_empirical"] == row["data_error_energy"] / row["data_truth_energy"]
        assert row["full_nmse_empirical"] == row["full_error_energy"] / row["full_truth_energy"]
    for row in report["joint_covariance_spot_checks"]["pairs"]:
        assert row["absolute_error"] <= 0.05
    assert all(report["frozen_source_checks"].values())
    assert not any(report[key] for key in ("explicit_inverse_used", "pseudo_inverse_used", "jitter_used"))


def test_saved_csv_and_source_hashes():
    report = json.loads((ROOT / "outputs/t10a_lmmse_oracle_results.json").read_text(encoding="utf-8"))
    path = ROOT / "outputs/t10a_lmmse_oracle_theory.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["theory_csv_sha256"]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    for actual, expected in zip(rows, report["theoretical_conditions"]):
        assert int(actual["dF"]) == expected["dF"]
        assert int(actual["snr_db"]) == expected["snr_db"]
        for name, value in expected.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert float(actual[name]) == value
    for name, expected in report["source_file_hashes"].items():
        from src.provenance import recorded_source_matches
        assert recorded_source_matches(ROOT,name,expected,"t10a")
