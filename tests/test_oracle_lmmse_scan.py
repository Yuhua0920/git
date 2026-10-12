"""T10B small full-link tests first; independent audits of formal artifacts."""

import hashlib
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import src.lmmse as lmmse
import src.oracle_lmmse_scan as scan
import src.tdl_awgn_scan as baseline
from src.pilots import build_fdm_pilot_layout
from src.tdl_noiseless_scan import NoiselessLinkFailure


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def theory():
    return scan.load_theoretical_table(ROOT / "outputs/t10a_lmmse_oracle_theory.csv")


@pytest.fixture(scope="module")
def preparation(theory):
    layouts = {d_f: build_fdm_pilot_layout(2, 64, 16, d_f, 4) for d_f in scan.SPACINGS}
    with patch.object(lmmse, "cho_factor", wraps=lmmse.cho_factor) as factor, \
         patch.object(scan, "build_frequency_covariance", wraps=scan.build_frequency_covariance) as rf, \
         patch.object(scan, "build_time_covariance", wraps=scan.build_time_covariance) as rt, \
         patch.object(scan, "build_full_grid_covariance", wraps=scan.build_full_grid_covariance) as full:
        cache, report = scan.prepare_estimators(layouts, theory)
    assert factor.call_count == 40 and rf.call_count == rt.call_count == full.call_count == 1
    return layouts, cache, report


@pytest.mark.parametrize("tid", [0, 1, 199])
def test_exact_t09b_channel_bits_and_unit_noise(tid):
    _, channel = scan.build_tdl_a_doppler_channel(scan.CHANNEL_SEED_BASE + tid)
    _, expected = baseline.build_tdl_a_doppler_channel(baseline.CHANNEL_SEED_BASE + tid)
    np.testing.assert_array_equal(channel, expected)
    seeds = (scan.BITS_SEED_BASE + 2 * tid, scan.BITS_SEED_BASE + 2 * tid + 1)
    for a, b in zip(scan.generate_master_qpsk(bits_seeds=seeds), baseline.generate_master_qpsk(bits_seeds=seeds)):
        np.testing.assert_array_equal(a, b)
    z = scan.complex_gaussian_noise((2, 64, 16), 1.0, np.random.default_rng(scan.NOISE_SEED_BASE + tid))
    expected = baseline.complex_gaussian_noise((2, 64, 16), 1.0, np.random.default_rng(baseline.NOISE_SEED_BASE + tid))
    np.testing.assert_array_equal(z, expected)
    if tid == 0:
        archived = json.loads((ROOT / "outputs/t09b_tdl_awgn_results.json").read_text())
        assert hashlib.sha256(channel.tobytes(order="C")).hexdigest() == archived["channel_realizations"][0]["H_sha256_complex128_C_order"]


def test_40_prepared_objects_and_theory_table(preparation):
    layouts, cache, report = preparation
    assert len(cache) == report["main_cache_factorization_count"] == 40 and report["passed"]
    assert len(report["theory_table_matching"]) == 20
    for (d_f, snr, q), obj in cache.items():
        np.testing.assert_array_equal(obj.pilot_indices, np.flatnonzero(layouts[d_f].pilot_mask[q].ravel(order="C")))
        assert obj.noise_variance == scan.noise_variance_from_snr(snr, 1.0)


def test_shared_real_pilot_path_no_leakage_and_t09b_small_regression(preparation, monkeypatch):
    layouts, cache, _ = preparation
    _, channel = scan.build_tdl_a_doppler_channel(scan.CHANNEL_SEED_BASE)
    bits, symbols = scan.generate_master_qpsk(bits_seeds=(scan.BITS_SEED_BASE, scan.BITS_SEED_BASE + 1))
    z = scan.complex_gaussian_noise((2, 64, 16), 1, np.random.default_rng(scan.NOISE_SEED_BASE))
    observations, received, detects = [], [], []
    original_estimate, original_oracle, original_detect = scan.estimate_ls_linear_channel, scan.oracle_channel_from_pilot_ls, scan.safe_detect
    def estimate(y, layout, pilots):
        received.append(y)
        result = original_estimate(y, layout, pilots)
        observations.append(result[0])
        return result
    def oracle(ls, objects):
        assert ls is observations[-1]
        assert np.isnan(ls[~np.broadcast_to(layouts[8].pilot_mask[None], ls.shape)]).all()
        for q in range(2):
            mask = layouts[8].pilot_mask[q]
            np.testing.assert_allclose(ls[:, q, mask], channel[:, q, mask] + noise[:, mask] / scan.PILOT_SYMBOLS[q], atol=1e-14)
        return original_oracle(ls, objects)
    def detect(y, *args):
        detects.append(y)
        return original_detect(y, *args)
    monkeypatch.setattr(scan, "estimate_ls_linear_channel", estimate)
    monkeypatch.setattr(scan, "oracle_channel_from_pilot_ls", oracle)
    monkeypatch.setattr(scan, "safe_detect", detect)
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected inverse/pinv/factorization")
    monkeypatch.setattr(lmmse, "cho_factor", forbidden)
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    monkeypatch.setattr(np.linalg, "pinv", forbidden)
    variance = scan.noise_variance_from_snr(15, 1)
    noise = scan.scale_unit_noise(z, variance)
    cond = scan.condition_numbers(channel, 0)
    row, _ = scan.evaluate_condition(channel, bits, symbols, noise, variance, layouts[8], 0, 15, 8, cond,
                                    tuple(cache[8, 15, q] for q in range(2)))
    old, _ = baseline.evaluate_condition(channel, bits, symbols, noise, variance, layouts[8], 0, 15, 8, cond)
    assert len(detects) == 3 and all(y is received[0] for y in detects)
    for field in ("nmse_truth_energy", "data_bits", "perfect_csi_bit_errors", "ls_lin_bit_errors", "noise_variance"):
        assert row[field] == old[field]
    assert row["ls_nmse_error_energy"] == old["nmse_error_energy"]
    assert list(inspect.signature(original_oracle).parameters) == ["pilot_ls", "prepared_by_tx"]
    assert list(inspect.signature(lmmse.PreparedOracleLMMSE.estimate).parameters) == ["self", "pilot_observations"]


def test_generated_once_and_cache_reused_outside_loop(theory, preparation, monkeypatch):
    layouts, cache, report = preparation
    calls = {"channel": 0, "bits": 0, "noise": 0, "preparation": 0}
    original_channel, original_bits, original_noise, original_evaluate = scan.build_tdl_a_doppler_channel, scan.generate_master_qpsk, scan.complex_gaussian_noise, scan.evaluate_condition
    seen, scales = [], []
    def channel(*args, **kwargs):
        calls["channel"] += 1
        return original_channel(*args, **kwargs)
    def bits(*args, **kwargs):
        calls["bits"] += 1
        return original_bits(*args, **kwargs)
    def noise(*args, **kwargs):
        calls["noise"] += 1
        return original_noise(*args, **kwargs)
    def prepare(*args):
        calls["preparation"] += 1
        return cache, report
    def evaluate(*args):
        seen.append(args)
        assert all(args[-1][q] is cache[args[8], args[7], q] for q in range(2))
        return original_evaluate(*args)
    def forbidden(*args, **kwargs):
        raise AssertionError("covariance/factorization inside simulation")
    monkeypatch.setattr(scan, "build_tdl_a_doppler_channel", channel)
    monkeypatch.setattr(scan, "generate_master_qpsk", bits)
    monkeypatch.setattr(scan, "complex_gaussian_noise", noise)
    monkeypatch.setattr(scan, "prepare_estimators", prepare)
    monkeypatch.setattr(scan, "evaluate_condition", evaluate)
    for name in ("build_frequency_covariance", "build_time_covariance", "build_full_grid_covariance", "prepare_oracle_lmmse"):
        monkeypatch.setattr(scan, name, forbidden)
    monkeypatch.setattr(lmmse, "cho_factor", forbidden)
    result, rows = scan.run_oracle_lmmse_scan(theory, 2)
    assert calls == {"channel": 2, "bits": 2, "noise": 2, "preparation": 1}
    assert len(rows) == len(seen) == 40
    for tid in range(2):
        group = [args for args in seen if args[6] == tid]
        assert all(all(args[i] is group[0][i] for i in (0, 1, 2)) for args in group)
        for snr in scan.SNR_POINTS:
            at_snr = [args for args in group if args[7] == snr]
            assert all(args[3] is at_snr[0][3] for args in at_snr)
    assert scan.verify_pooled_results(result["aggregate_results"], rows, 2)


def test_solve_failure_stops_and_keeps_context(preparation, monkeypatch):
    layouts, cache, _ = preparation
    _, channel = scan.build_tdl_a_doppler_channel(scan.CHANNEL_SEED_BASE)
    bits, symbols = scan.generate_master_qpsk(bits_seeds=(scan.BITS_SEED_BASE, scan.BITS_SEED_BASE + 1))
    noise = scan.complex_gaussian_noise((2, 64, 16), 0.1, np.random.default_rng(scan.NOISE_SEED_BASE))
    monkeypatch.setattr(scan, "oracle_channel_from_pilot_ls", lambda *args: np.ones((2, 2, 64, 16), dtype=np.complex128))
    with pytest.raises(NoiselessLinkFailure) as caught:
        scan.evaluate_condition(channel, bits, symbols, noise, 0.1, layouts[4], 0, 10, 4,
                                scan.condition_numbers(channel, 0), tuple(cache[4, 10, q] for q in range(2)))
    record = caught.value.record
    assert record["trajectory_id"] == 0 and record["dF"] == 4 and record["snr_db"] == 10
    assert record["method"] == "oracle_lmmse" and record["k"] == 0 and record["t"] == 1
    assert "estimated_condition_number" in record
    json.dumps(record, allow_nan=False)


def test_saved_full_regressions_and_counts():
    report = json.loads((ROOT / "outputs/t10b_oracle_lmmse_results.json").read_text())
    rows = scan.read_csv(ROOT / "outputs/t10b_oracle_lmmse_trajectory_metrics.csv")
    aggregate = scan.read_csv(ROOT / "outputs/t10b_oracle_lmmse_scan.csv")
    assert len(rows) == report["trajectory_rows"] == 4000
    assert len(aggregate) == report["aggregate_rows"] == 20
    assert report["all_checks_passed"] and report["t09b_regression"]["passed"]
    assert report["t10a_theory_regression"]["passed"]
    assert len(report["t10a_theory_regression"]["conditions"]) == 20
    assert report["num_filtered_trajectories"] == 0 and report["numerical_failures"] == []
    assert report["prepared_estimator_count"] == 40
    assert not any(report[name] for name in ("explicit_inverse_used", "pseudo_inverse_used", "jitter_used"))
    expected = {(tid, snr, d_f) for tid in range(200) for snr in scan.SNR_POINTS for d_f in scan.SPACINGS}
    assert {(int(r["trajectory_id"]), int(r["snr_db"]), int(r["dF"])) for r in rows} == expected
    assert all(report["frozen_source_checks"].values())
    for name, digest in report["provenance"]["source_and_reference_sha256"].items():
        from src.provenance import recorded_source_matches
        assert recorded_source_matches(ROOT,name,digest,"t10b")


def test_regression_rejects_changed_integer_count():
    report = json.loads((ROOT / "outputs/t10b_oracle_lmmse_results.json").read_text())
    actual = [r.copy() for r in report["aggregate_results"]]
    actual[0]["perfect_csi_bit_errors"] += 1
    regression = scan.verify_t09b_regression(actual, scan.read_csv(ROOT / "outputs/t09b_tdl_awgn_scan.csv"))
    assert not regression["passed"]


def test_plot_only_rebuilds_from_csv_with_no_existing_images(tmp_path, monkeypatch):
    from scripts import run_oracle_lmmse_scan as entry
    csv_path = tmp_path / "t10b_oracle_lmmse_scan.csv"
    csv_path.write_bytes((ROOT / "outputs/t10b_oracle_lmmse_scan.csv").read_bytes())
    monkeypatch.setattr(entry, "OUTPUTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_oracle_lmmse_scan.py", "--plot-only"])
    def forbidden(*args, **kwargs):
        raise AssertionError("plot-only unexpectedly used simulation or theory data")
    monkeypatch.setattr(entry, "run_oracle_lmmse_scan", forbidden)
    monkeypatch.setattr(entry, "load_theoretical_table", forbidden)
    entry.main()
    assert len(list(tmp_path.glob("*.png"))) == 4
    assert len(list(tmp_path.glob("*.pdf"))) == 4
    assert len(list(tmp_path.glob("*.svg"))) == 4


@pytest.mark.parametrize("d_f,snr", [(d_f, snr) for d_f in scan.SPACINGS for snr in scan.SNR_POINTS])
def test_saved_energy_ber_proxy_and_theory(d_f, snr):
    rows = scan.read_csv(ROOT / "outputs/t10b_oracle_lmmse_trajectory_metrics.csv")
    group = [r for r in rows if int(r["dF"]) == d_f and int(r["snr_db"]) == snr]
    aggregate = next(r for r in scan.read_csv(ROOT / "outputs/t10b_oracle_lmmse_scan.csv")
                     if int(r["dF"]) == d_f and int(r["snr_db"]) == snr)
    bits = sum(int(r["data_bits"]) for r in group)
    truth = sum(float(r["nmse_truth_energy"]) for r in group)
    for method, energy in (("ls_lin", "ls_nmse_error_energy"), ("oracle_lmmse", "oracle_nmse_error_energy")):
        error = sum(float(r[energy]) for r in group)
        assert float(aggregate[f"{method}_nmse_linear"]) == error / truth
    for method in scan.METHODS:
        errors = sum(int(r[f"{method}_bit_errors"]) for r in group)
        assert int(aggregate[f"{method}_bit_errors"]) == errors
        assert float(aggregate[f"{method}_ber"]) == errors / bits
        assert float(aggregate[f"{method}_correct_bit_rate_proxy_bps"]) == pytest.approx(
            (bits - errors) / (200 * scan.frame_duration_seconds()), rel=1e-12)
    assert abs(float(aggregate["oracle_empirical_minus_theory_db"])) <= 0.75
