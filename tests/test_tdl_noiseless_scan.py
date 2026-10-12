"""T09A small numerical checks and independently audited saved full-run counts."""

import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import src.tdl_noiseless_scan as scan
from src.doppler import build_tdl_a_doppler_channel
from src.link import (apply_frequency_mimo_channel, assemble_pilot_data_frame, estimate_ls_linear_channel,
                      extract_reference_data_bits, generate_master_qpsk)
from src.pilots import build_fdm_pilot_layout


@pytest.fixture(scope="module")
def small_result():
    return scan.run_noiseless_scan(2)


def test_small_scan_shapes_counts_and_exact_paths(small_result):
    report, rows = small_result
    assert len(rows) == 8
    assert len(report["aggregate_results"]) == 4
    assert len(report["per_symbol_results"]) == 64
    assert report["condition_number_statistics"]["count"] == 2 * 64 * 16
    assert report["num_filtered_trajectories"] == 0
    assert report["all_checks_passed"]
    assert report["ls_lin_grid_finite"]
    for row in report["aggregate_results"]:
        assert row["perfect_csi_bit_errors"] == 0
        assert row["pilot_ls_relative_error"] <= 1e-12
        assert row["data_bits_per_trajectory"] == {2: 3072, 4: 3584, 8: 3840, 16: 3968}[row["dF"]]
        assert row["pilot_ratio"] == {2: 1/4, 4: 1/8, 8: 1/16, 16: 1/32}[row["dF"]]


def test_paired_channel_master_bits_and_receiver_inputs(monkeypatch):
    original_evaluate = scan.evaluate_layout
    original_estimate = scan.estimate_ls_linear_channel
    references, receiver_calls = [], []
    def observe(h_true, bits, s_master, *args):
        references.append((h_true, bits, s_master))
        return original_evaluate(h_true, bits, s_master, *args)
    def receiver(y, layout, pilots):
        assert y.shape == (2, 64, 16)
        assert pilots.shape == (2,)
        receiver_calls.append((y, layout, pilots))
        return original_estimate(y, layout, pilots)
    monkeypatch.setattr(scan, "evaluate_layout", observe)
    monkeypatch.setattr(scan, "estimate_ls_linear_channel", receiver)
    scan.run_noiseless_scan(1)
    assert len(references) == len(receiver_calls) == 4
    for reference in references[1:]:
        assert all(reference[i] is references[0][i] for i in range(3))
    assert set(inspect.signature(original_estimate).parameters) == {"y_grid", "layout", "pilot_symbols"}


def test_dynamic_shapes_and_explicit_noiseless_path(monkeypatch):
    _, h = build_tdl_a_doppler_channel(scan.CHANNEL_SEED_BASE)
    bits, s = generate_master_qpsk()
    layout = build_fdm_pilot_layout(2, 64, 16, 4, 4)
    x = assemble_pilot_data_frame(s, layout)
    y = apply_frequency_mimo_channel(h, x)
    expected = np.stack([h[r, 0] * x[0] + h[r, 1] * x[1] for r in range(2)])
    np.testing.assert_array_equal(y, np.einsum("rqkt,qkt->rkt", h, x))
    np.testing.assert_allclose(y, expected, rtol=0, atol=1e-14)
    assert h.shape == (2, 2, 64, 16) and s.shape == (2, 64, 16) and bits.shape == (2, 64, 16, 2)
    assert np.max(np.abs(h[..., 15] - h[..., 0])) > 0
    def forbidden(*args, **kwargs):
        raise AssertionError("AWGN was called")
    import src.link as link
    monkeypatch.setattr(link, "add_received_noise", forbidden)
    # Channel latent CN draws remain authorized; only receiver AWGN is prohibited.
    assert scan.run_noiseless_scan(1)[0]["all_checks_passed"]
    source = inspect.getsource(scan)
    assert "add_received_noise" not in source and "noise_master" not in source


def test_data_only_and_per_symbol_accounting():
    h = np.ones((2, 2, 64, 16), dtype=np.complex128)
    for dF in scan.SPACINGS:
        layout = build_fdm_pilot_layout(2, 64, 16, dF, 4)
        estimate = h.copy()
        estimate[:, ~layout.data_mask] = 1000  # Errors on pilot/mute must be excluded.
        estimate[:, layout.data_mask] += 2
        errors, truth, counts = scan.per_symbol_energies(estimate, h, layout.data_mask)
        np.testing.assert_array_equal(errors, 16 * counts)
        np.testing.assert_array_equal(truth, 4 * counts)
        for t in range(16):
            assert counts[t] == (64 - 128 // dF if t in scan.PILOT_TIMES else 64)
        rows = scan.make_symbol_rows(dF, 1/(2*dF), errors, truth, counts)
        for t, row in enumerate(rows):
            assert row["is_pilot_time"] == (t in scan.PILOT_TIMES)
            assert row["nmse_linear"] == (4 if counts[t] else None)
            assert row["nmse_db"] == (10 * np.log10(4) if counts[t] else None)
        pooled = scan.channel_error_energies(estimate, h, layout.data_mask)
        np.testing.assert_allclose([errors.sum(), truth.sum()], pooled, rtol=1e-14)


def test_actual_channel_pilot_exact_and_right_edge_hold():
    _, h = build_tdl_a_doppler_channel(scan.CHANNEL_SEED_BASE)
    _, s = generate_master_qpsk()
    for dF in scan.SPACINGS:
        layout = build_fdm_pilot_layout(2, 64, 16, dF, 4)
        y = apply_frequency_mimo_channel(h, assemble_pilot_data_frame(s, layout))
        ls, estimate = estimate_ls_linear_channel(y, layout)
        observed = np.broadcast_to(layout.pilot_mask[None], h.shape)
        np.testing.assert_allclose(ls[observed], h[observed], rtol=0, atol=1e-12)
        assert np.all(np.isfinite(estimate))
        for t in (13, 14, 15):
            np.testing.assert_array_equal(estimate[..., t], estimate[..., 12])


def test_receiver_ignores_data_observations_and_ber_excludes_pilot_bits():
    _, h = build_tdl_a_doppler_channel(scan.CHANNEL_SEED_BASE)
    bits, s = generate_master_qpsk()
    layout = build_fdm_pilot_layout(2, 64, 16, 4, 4)
    y = apply_frequency_mimo_channel(h, assemble_pilot_data_frame(s, layout))
    original = estimate_ls_linear_channel(y, layout)[1]
    modified_y = y.copy()
    modified_y[:, layout.data_mask[0]] = 12345 + 9876j
    np.testing.assert_array_equal(estimate_ls_linear_channel(modified_y, layout)[1], original)
    changed_bits = bits.copy()
    changed_bits[~layout.data_mask] ^= 1
    for actual, expected in zip(extract_reference_data_bits(changed_bits, layout.data_mask),
                                extract_reference_data_bits(bits, layout.data_mask)):
        np.testing.assert_array_equal(actual, expected)


def test_pooled_ratios_and_boundary_diagnostics(small_result):
    report, rows = small_result
    for aggregate in report["aggregate_results"]:
        group = [row for row in rows if row["dF"] == aggregate["dF"]]
        e = sum(row["nmse_error_energy"] for row in group)
        truth = sum(row["nmse_truth_energy"] for row in group)
        bits = sum(row["data_bits"] for row in group)
        assert aggregate["nmse_data_linear"] == e / truth
        assert aggregate["ls_lin_ber"] == sum(row["ls_lin_bit_errors"] for row in group) / bits
        symbol_rows = [row for row in report["per_symbol_results"] if row["dF"] == aggregate["dF"]]
        np.testing.assert_allclose(sum(row["error_energy"] for row in symbol_rows), e, rtol=1e-14)
        np.testing.assert_allclose(sum(row["truth_energy"] for row in symbol_rows), truth, rtol=1e-14)
    assert {tuple(row["symbols"]) for row in report["boundary_diagnostics"]} == {(1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15)}
    assert len(report["pilot_time_whole_symbol_diagnostics"]) == 16


def test_large_condition_is_retained_and_singular_stops():
    h = np.zeros((2, 2, 64, 16), dtype=np.complex128)
    h[0, 0] = 1
    h[1, 1] = 1e-14
    conditions = scan.condition_numbers(h, 7)
    assert conditions.size == 64 * 16
    np.testing.assert_allclose(conditions, 1e14)
    h[1, 1, 3, 15] = 0
    with pytest.raises(scan.NoiselessLinkFailure) as error:
        scan.condition_numbers(h, 7)
    assert error.value.record["k"] == 3 and error.value.record["t"] == 15
    assert error.value.record["channel_seed"] == scan.CHANNEL_SEED_BASE + 7


def test_nonfinite_failure_context_is_valid_json():
    _, h = build_tdl_a_doppler_channel(scan.CHANNEL_SEED_BASE)
    h[1, 0, 5, 14] = np.nan
    with pytest.raises(scan.NoiselessLinkFailure) as error:
        scan.condition_numbers(h, 0)
    record = error.value.record
    assert (record["k"], record["t"]) == (5, 14)
    assert record["H_true"]["real"][1][0] is None
    json.dumps(record, allow_nan=False)


def test_failure_maps_data_coordinate_and_keeps_symbols(monkeypatch):
    layout = build_fdm_pilot_layout(2, 64, 16, 4, 4)
    _, h = build_tdl_a_doppler_channel(scan.CHANNEL_SEED_BASE)
    bits, s = generate_master_qpsk()
    original = scan.detect_data_zf
    def fail_solve(*args):
        raise ValueError("singular channel at RE (k=10, t=0)")
    monkeypatch.setattr(scan, "detect_data_zf", fail_solve)
    with pytest.raises(scan.NoiselessLinkFailure) as error:
        scan.evaluate_layout(h, bits, s, layout, 0, 4, scan.condition_numbers(h, 0))
    expected_k, expected_t = np.argwhere(layout.data_mask[0])[10]
    assert (error.value.record["k"], error.value.record["t"]) == (expected_k, expected_t)
    assert error.value.record["symbol"] is not None
    def wrong_symbols(*args):
        return -original(*args)
    monkeypatch.setattr(scan, "detect_data_zf", wrong_symbols)
    with pytest.raises(scan.NoiselessLinkFailure, match="Perfect-CSI bit error") as error:
        scan.evaluate_layout(h, bits, s, layout, 0, 4, scan.condition_numbers(h, 0))
    assert error.value.record["detected_symbol"] is not None
    assert error.value.record["expected_bit"] != error.value.record["actual_bit"]


def test_manual_trace_distinguishes_owner_ls_from_interpolated_column(small_result):
    records = small_result[0]["manual_diagnostics"]
    assert [(row["kind"], row["k"], row["t"]) for row in records] == [("pilot", 0, 0), ("interior_data", 1, 2), ("trailing_data", 1, 15)]
    assert records[0]["pilot_owner_ls_relative_error"] <= 1e-12
    assert records[0]["relative_channel_error"] > 1e-12
    assert not records[0]["S_master_used_as_data"]


def test_saved_full_run_counts_and_reaggregation():
    root = Path(__file__).resolve().parents[1] / "outputs"
    def read(name):
        with (root / f"t09a_tdl_noiseless_{name}.csv").open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    trajectories, aggregate, symbols = read("trajectory_metrics"), read("scan"), read("symbol_nmse")
    report = json.loads((root / "t09a_tdl_noiseless_results.json").read_text(encoding="utf-8"))
    assert len(trajectories) == 800 and len(aggregate) == 4 and len(symbols) == 64
    assert report["num_trajectories"] == 200 and report["all_checks_passed"]
    assert report["condition_number_statistics"]["count"] == 200 * 64 * 16
    assert report["num_filtered_trajectories"] == 0
    for dF in scan.SPACINGS:
        group = [row for row in trajectories if int(row["dF"]) == dF]
        assert [int(row["trajectory_id"]) for row in group] == list(range(200))
        assert [int(row["channel_seed"]) for row in group] == list(range(scan.CHANNEL_SEED_BASE, scan.CHANNEL_SEED_BASE + 200))
        row = next(row for row in aggregate if int(row["dF"]) == dF)
        assert sum(int(item["perfect_csi_bit_errors"]) for item in group) == 0
        assert sum(int(item["data_bits"]) for item in group) == int(row["total_data_bits"])
        assert sum(int(item["ls_lin_bit_errors"]) for item in group) == int(row["ls_lin_bit_errors"])
        for name in ("nmse_error_energy", "nmse_truth_energy"):
            np.testing.assert_allclose(sum(float(item[name]) for item in group), float(row[name]), rtol=1e-14)
        for symbol in (item for item in symbols if int(item["dF"]) == dF):
            t = int(symbol["symbol_index"])
            expected_count = 200 * (64 - 128 // dF if t in scan.PILOT_TIMES else 64)
            assert int(symbol["data_re_count_total"]) == expected_count
            if expected_count == 0:
                assert symbol["nmse_linear"] == symbol["nmse_db"] == ""
            else:
                assert float(symbol["nmse_linear"]) == float(symbol["error_energy"]) / float(symbol["truth_energy"])
    assert (root / "t09a_symbol_nmse.png").stat().st_size > 0
