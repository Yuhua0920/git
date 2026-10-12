"""T09B protocol checks before integration plus independent full-artifact audit."""

import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import src.tdl_awgn_scan as scan
from scripts.run_tdl_awgn_scan import frozen_source_checks, verify_t09a_channels
from src.doppler import build_tdl_a_doppler_channel
from src.link import apply_frequency_mimo_channel, assemble_pilot_data_frame, generate_master_qpsk
from src.pilots import build_fdm_pilot_layout
from src.tdl_noiseless_scan import NoiselessLinkFailure, run_noiseless_scan


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def small_result():
    return scan.run_tdl_awgn_scan(2)


def test_frozen_t09a_sources_and_exact_archived_channel_values():
    baseline = json.loads((ROOT / "outputs/t09a_tdl_noiseless_results.json").read_text(encoding="utf-8"))
    assert all(frozen_source_checks(baseline).values())
    _, h = scan.build_tdl_a_doppler_channel(scan.CHANNEL_SEED_BASE)
    for trace in baseline["manual_diagnostics"]:
        expected = np.array(trace["H_true"]["real"]) + 1j * np.array(trace["H_true"]["imag"])
        np.testing.assert_array_equal(h[:, :, trace["k"], trace["t"]], expected)


@pytest.mark.parametrize("snr,variance", [(0, 1), (5, 0.31622776601683794), (10, 0.1),
                                         (15, 0.03162277660168379), (20, 0.01)])
def test_fixed_variance_and_noise_scaling(snr, variance):
    z = scan.complex_gaussian_noise((2, 64, 16), 1, np.random.default_rng(scan.NOISE_SEED_BASE))
    actual_variance = scan.noise_variance_from_snr(snr, 1.0)
    assert actual_variance == pytest.approx(variance, rel=1e-15)
    np.testing.assert_array_equal(scan.scale_unit_noise(z, actual_variance), np.sqrt(actual_variance) * z)


def test_noise_marginal_power_and_unique_sample_count(small_result):
    report = small_result[0]
    samples = 2 * 2 * 64 * 16
    for row in report["noise_statistics"]:
        assert row["independent_noise_samples"] == samples
        assert row["empirical_scaled_noise_power"] == pytest.approx(
            row["target_noise_variance"] * row["empirical_unit_noise_power"], rel=1e-14)
    # An independent statistical check uses 409600 complex samples without reusing them.
    z = scan.complex_gaussian_noise((200, 2, 64, 16), 1, np.random.default_rng(scan.NOISE_SEED_BASE))
    assert abs(np.mean(np.abs(z)**2) - 1) <= 0.02
    assert np.var(z.real) == pytest.approx(0.5, rel=0.02)
    assert np.var(z.imag) == pytest.approx(0.5, rel=0.02)


def test_channel_bits_and_unit_noise_generated_once_and_shared(monkeypatch):
    calls = {"channel": 0, "bits": 0, "unit_noise": 0}
    original_channel, original_bits, original_noise = scan.build_tdl_a_doppler_channel, scan.generate_master_qpsk, scan.complex_gaussian_noise
    original_scale, original_evaluate = scan.scale_unit_noise, scan.evaluate_condition
    observed, scales = [], []
    def channel(*args, **kwargs):
        calls["channel"] += 1
        return original_channel(*args, **kwargs)
    def bits(*args, **kwargs):
        calls["bits"] += 1
        return original_bits(*args, **kwargs)
    def noise(*args, **kwargs):
        calls["unit_noise"] += 1
        assert args[0] == (2, 64, 16) and args[1] == 1.0
        return original_noise(*args, **kwargs)
    def scale(z, variance):
        scales.append((z, variance))
        return original_scale(z, variance)
    def evaluate(h, b, s, n, v, layout, tid, snr, dF, cond):
        observed.append((h, b, s, n, v, layout, tid, snr, dF))
        return original_evaluate(h, b, s, n, v, layout, tid, snr, dF, cond)
    monkeypatch.setattr(scan, "build_tdl_a_doppler_channel", channel)
    monkeypatch.setattr(scan, "generate_master_qpsk", bits)
    monkeypatch.setattr(scan, "complex_gaussian_noise", noise)
    monkeypatch.setattr(scan, "scale_unit_noise", scale)
    monkeypatch.setattr(scan, "evaluate_condition", evaluate)
    report, rows = scan.run_tdl_awgn_scan(2, include_noiseless_regression=False)
    assert calls == {"channel": 2, "bits": 2, "unit_noise": 2}
    assert len(observed) == 40 and len(scales) == 10
    for tid in range(2):
        group = [row for row in observed if row[6] == tid]
        for row in group:
            assert all(row[index] is group[0][index] for index in (0, 1, 2))
        for snr in scan.SNR_POINTS:
            subset = [row for row in group if row[7] == snr]
            assert all(row[3] is subset[0][3] for row in subset)
        assert all(scales[5*tid+i][0] is scales[5*tid][0] for i in range(5))
        # Common data positions transmit exactly the same dibits/symbols across dF.
        first = group[0]
        for other in group[1:]:
            common = first[5].data_mask & other[5].data_mask
            np.testing.assert_array_equal(first[1][common], other[1][common])
            np.testing.assert_array_equal(assemble_pilot_data_frame(first[2], first[5])[common],
                                          assemble_pilot_data_frame(other[2], other[5])[common])
    assert len(rows) == 40 and report["common_random_numbers"]


def test_identical_noisy_y_for_both_paths_and_blind_estimator(monkeypatch):
    original_detect, original_estimate, original_add = scan.safe_detect, scan.estimate_ls_linear_channel, scan.add_received_noise
    detects, estimates, additions = [], [], []
    def detect(y, *args):
        detects.append(y)
        return original_detect(y, *args)
    def estimate(y, layout, pilots):
        estimates.append(y)
        assert pilots.shape == (2,) and y.shape == (2, 64, 16)
        return original_estimate(y, layout, pilots)
    def add(clean, noise):
        noisy = original_add(clean, noise)
        np.testing.assert_array_equal(noisy, clean + noise)
        assert np.all(noise != 0)
        additions.append(noisy)
        return noisy
    monkeypatch.setattr(scan, "safe_detect", detect)
    monkeypatch.setattr(scan, "estimate_ls_linear_channel", estimate)
    monkeypatch.setattr(scan, "add_received_noise", add)
    scan.run_tdl_awgn_scan(1, include_noiseless_regression=False)
    assert len(detects) == 40 and len(estimates) == len(additions) == 20
    for i in range(20):
        assert detects[2*i] is detects[2*i+1] is estimates[i] is additions[i]
    assert set(inspect.signature(original_estimate).parameters) == {"y_grid", "layout", "pilot_symbols"}


@pytest.mark.parametrize("channel_scale", [0.1, 10.0])
def test_no_instantaneous_normalization_and_pilots_data_both_noisy(channel_scale, monkeypatch):
    _, h = build_tdl_a_doppler_channel(scan.CHANNEL_SEED_BASE)
    h *= channel_scale
    before = h.copy()
    bits, s = generate_master_qpsk()
    layout = build_fdm_pilot_layout(2, 64, 16, 4, 4)
    z = scan.complex_gaussian_noise((2, 64, 16), 1, np.random.default_rng(scan.NOISE_SEED_BASE))
    n = scan.scale_unit_noise(z, 0.1)
    original_estimate = scan.estimate_ls_linear_channel
    def estimate(y, supplied_layout, pilots):
        clean = apply_frequency_mimo_channel(h, assemble_pilot_data_frame(s, layout))
        np.testing.assert_allclose(y - clean, n, rtol=0, atol=1e-14)
        for mask in (layout.pilot_union, layout.data_mask[0]):
            assert np.all(np.abs((y-clean)[:, mask]) > 0)
        return original_estimate(y, supplied_layout, pilots)
    monkeypatch.setattr(scan, "estimate_ls_linear_channel", estimate)
    row, _ = scan.evaluate_condition(h, bits, s, n, 0.1, layout, 1, 10, 4, scan.condition_numbers(h, 1))
    assert row["noise_variance"] == 0.1
    np.testing.assert_array_equal(h, before)


def test_pilot_ls_samplewise_noise_identity_and_mse(small_result):
    report, rows = small_result
    trace = report["manual_diagnostics"][0]
    for part in ("real", "imag"):
        np.testing.assert_allclose(trace["pilot_ls_error"][part], trace["noise_divided_by_pilot"][part], rtol=0, atol=1e-14)
    for row in report["aggregate_results"]:
        assert row["pilot_ls_sample_count"] == 2 * 2 * (128 // row["dF"]) * 4
        group = [item for item in rows if item["dF"] == row["dF"] and item["snr_db"] == row["snr_db"]]
        assert row["pilot_ls_mse"] == sum(item["pilot_ls_error_energy"] for item in group) / row["pilot_ls_sample_count"]
        assert row["pilot_ls_mse_to_noise_ratio"] == row["pilot_ls_mse"] / row["noise_variance"]


def test_full_grid_finite_and_data_only_pooled_metrics_proxy(small_result):
    report, rows = small_result
    assert report["ls_lin_grid_finite"]
    assert scan.verify_pooled_results(report["aggregate_results"], rows, 2)
    for aggregate in report["aggregate_results"]:
        assert aggregate["data_bits_per_trajectory"] == {2: 3072, 4: 3584, 8: 3840, 16: 3968}[aggregate["dF"]]
        for prefix in ("perfect_csi", "ls_lin"):
            expected = aggregate["data_bits_per_trajectory"] * (1-aggregate[f"{prefix}_ber"]) / (16*(64+16)/1920000)
            assert aggregate[f"{prefix}_correct_bit_rate_proxy_bps"] == pytest.approx(expected, rel=1e-14)
        assert aggregate["nmse_gap_above_floor_db"] == aggregate["nmse_data_db"] - aggregate["noiseless_floor_nmse_db"]
    damaged = [dict(row) for row in report["aggregate_results"]]
    damaged[0]["perfect_csi_bit_errors"] += 1
    assert not scan.verify_pooled_results(damaged, rows, 2)


def test_zero_noise_pipeline_matches_actual_t09a_small_run(small_result):
    old, _ = run_noiseless_scan(2)
    regression = scan.verify_noiseless_regression(small_result[0]["zero_noise_aggregate_results"], old["aggregate_results"])
    assert regression["passed"]
    assert small_result[0]["zero_noise_evaluations"] == 8
    for current, previous in zip(small_result[0]["zero_noise_aggregate_results"], old["aggregate_results"]):
        assert current["nmse_error_energy"] == previous["nmse_error_energy"]
        assert current["nmse_truth_energy"] == previous["nmse_truth_energy"]
        assert current["ls_lin_bit_errors"] == previous["ls_lin_bit_errors"]


def test_noise_seed_and_condition_are_invariant_and_no_filtering(small_result):
    report, rows = small_result
    assert report["num_filtered_trajectories"] == 0 and report["numerical_failures"] == []
    for tid in range(2):
        group = [row for row in rows if row["trajectory_id"] == tid]
        assert len(group) == 20
        assert {row["noise_seed"] for row in group} == {scan.NOISE_SEED_BASE + tid}
        assert len({row["condition_number_max"] for row in group}) == 1
        assert {row["bits_seed_tx0"] for row in group} == {scan.BITS_SEED_BASE + 2*tid}


def test_solver_failure_keeps_snr_df_and_physical_coordinate(monkeypatch):
    _, h = build_tdl_a_doppler_channel(scan.CHANNEL_SEED_BASE)
    bits, s = generate_master_qpsk()
    layout = build_fdm_pilot_layout(2, 64, 16, 4, 4)
    import src.tdl_noiseless_scan as original_scan
    def fail(*args):
        raise ValueError("singular channel at RE (k=10, t=0)")
    monkeypatch.setattr(original_scan, "detect_data_zf", fail)
    n = scan.scale_unit_noise(np.ones((2, 64, 16), dtype=np.complex128), 0.1)
    with pytest.raises(NoiselessLinkFailure) as error:
        scan.evaluate_condition(h, bits, s, n, 0.1, layout, 0, 10, 4, scan.condition_numbers(h, 0))
    record = error.value.record
    k, t = np.argwhere(layout.data_mask[0])[10]
    assert (record["k"], record["t"], record["snr_db"], record["dF"]) == (k, t, 10, 4)
    assert record["channel_seed"] == scan.CHANNEL_SEED_BASE and record["noise_seed"] == scan.NOISE_SEED_BASE
    assert record["symbol"] is not None


def test_saved_full_run_counts_statistics_and_regression():
    output = ROOT / "outputs"
    report = json.loads((output / "t09b_tdl_awgn_results.json").read_text(encoding="utf-8"))
    rows = read_csv(output / "t09b_tdl_awgn_trajectory_metrics.csv")
    aggregates = read_csv(output / "t09b_tdl_awgn_scan.csv")
    noise = read_csv(output / "t09b_noise_statistics.csv")
    assert len(rows) == 4000 and len(aggregates) == 20 and len(noise) == 5
    assert report["num_trajectories"] == 200 and report["all_checks_passed"]
    assert report["num_filtered_trajectories"] == 0 and report["numerical_failures"] == []
    assert report["t09a_noiseless_regression"]["passed"] and report["zero_noise_evaluations"] == 800
    assert report["t09a_channel_checks"]["passed"]
    assert all(check["passed"] for check in report["pilot_ls_theoretical_mse_checks"])
    assert {(int(row["trajectory_id"]), int(row["snr_db"]), int(row["dF"])) for row in rows} == {
        (tid, snr, dF) for tid in range(200) for snr in scan.SNR_POINTS for dF in scan.SPACINGS}
    for row in noise:
        assert int(row["independent_noise_samples"]) == 409600
        assert float(row["relative_error"]) <= 0.02
    for aggregate in aggregates:
        group = [row for row in rows if row["dF"] == aggregate["dF"] and row["snr_db"] == aggregate["snr_db"]]
        assert len(group) == 200 and [int(row["trajectory_id"]) for row in group] == list(range(200))
        for name in ("nmse_error_energy", "nmse_truth_energy", "pilot_ls_error_energy"):
            assert float(aggregate[name]) == pytest.approx(sum(float(row[name]) for row in group), rel=1e-12)
        for name in ("pilot_ls_sample_count", "perfect_csi_bit_errors", "ls_lin_bit_errors"):
            assert int(aggregate[name]) == sum(int(row[name]) for row in group)
        assert int(aggregate["total_data_bits"]) == sum(int(row["data_bits"]) for row in group)
        assert float(aggregate["pilot_ls_mse_to_noise_ratio"]) == pytest.approx(
            float(aggregate["pilot_ls_error_energy"]) / int(aggregate["pilot_ls_sample_count"]) / float(aggregate["noise_variance"]), rel=1e-14)
        assert abs(float(aggregate["pilot_ls_mse_to_noise_ratio"])-1) <= 0.05
        for prefix in ("perfect_csi", "ls_lin"):
            errors = int(aggregate[f"{prefix}_bit_errors"])
            total_bits = int(aggregate["total_data_bits"])
            assert float(aggregate[f"{prefix}_ber"]) == errors / total_bits
            expected_proxy = (total_bits - errors) / (200 * 16*(64+16)/1920000)
            assert float(aggregate[f"{prefix}_correct_bit_rate_proxy_bps"]) == pytest.approx(expected_proxy, rel=1e-14)
            assert sum(int(row[f"{prefix}_correct_bits"]) for row in group) == total_bits - errors
    for tid in range(200):
        group = [row for row in rows if int(row["trajectory_id"]) == tid]
        assert {int(row["noise_seed"]) for row in group} == {scan.NOISE_SEED_BASE + tid}
        assert len({row["condition_number_max"] for row in group}) == 1
    for name in ("nmse", "ber", "correct_bit_rate"):
        assert (output / f"t09b_{name}_vs_snr.png").stat().st_size > 0


@pytest.mark.parametrize("count", [0, 201, True, 1.5])
def test_invalid_trajectory_count_rejected(count):
    with pytest.raises(ValueError):
        scan.run_tdl_awgn_scan(count)
