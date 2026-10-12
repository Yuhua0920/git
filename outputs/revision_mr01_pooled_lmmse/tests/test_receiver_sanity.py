"""Meaningful analytic chain, scale/order and synthetic noise checks for MR-01."""

import numpy as np
import pytest

from receiver_sanity import (
    ANALYTIC_H, EXPECTED_GRAY, FLOAT_ATOL, GRAY_DIBITS,
    analytic_chain_check, data_wrapper_check, make_analytic_fixture,
    near_singular_check, noise_convention_check, run_sanity, system_check,
)


def test_latest_v4_2_dimensions_and_sample_rate(apis):
    result = system_check(apis)
    assert result["passed"], result


def test_known_gray_constellation_all_points(apis):
    actual = apis["modulation"].qpsk_modulate(GRAY_DIBITS.reshape(-1))
    np.testing.assert_allclose(actual, EXPECTED_GRAY, rtol=0, atol=FLOAT_ATOL)
    np.testing.assert_array_equal(apis["modulation"].qpsk_demodulate(actual), GRAY_DIBITS.reshape(-1))


def test_manual_independent_rx_tx_equation(apis):
    x = np.array([0.5 + 0.5j, -0.5 + 0.5j])
    y = apis["link"].apply_frequency_mimo_channel(ANALYTIC_H[:, :, None, None], x[:, None, None])
    np.testing.assert_allclose(y[:, 0, 0], [0.0 + 0.25j, -0.2 + 0.4j], rtol=0, atol=FLOAT_ATOL)
    actual = apis["detectors"].zf_detect(y, ANALYTIC_H[:, :, None, None])
    np.testing.assert_allclose(actual[:, 0, 0], x, rtol=0, atol=FLOAT_ATOL)


def test_full_rank_noiseless_chain_scale_and_recovery(apis):
    result = analytic_chain_check(apis)
    assert result["matrix_rank"] == 2
    assert result["passed"], result
    assert result["bit_errors_by_stream"] == [0, 0]


def test_fixture_covers_all_two_stream_dibits_and_varies_tx_k_t(apis):
    fixture = make_analytic_fixture(apis)
    pairs = fixture["bits"].transpose(1, 2, 0, 3).reshape(-1, 4)
    assert len(np.unique(pairs, axis=0)) == 16
    assert not np.array_equal(fixture["bits"][0], fixture["bits"][1])
    assert not np.array_equal(fixture["bits"][:, 0], fixture["bits"][:, 1])
    assert not np.array_equal(fixture["bits"][:, :, 0], fixture["bits"][:, :, 1])


def test_masked_existing_perfect_csi_chain_and_bit_order(apis):
    result = data_wrapper_check(apis)
    assert result["passed"], result
    assert result["data_REs_by_stream"] == [992, 992]
    assert result["bits_by_stream"] == [1984, 1984]


def test_current_batched_adapter_matches_existing_individual_re_solves(apis):
    result = data_wrapper_check(apis)
    assert result["batched_bit_errors_by_stream"] == [0, 0]
    assert result["max_abs_batched_X_error"] <= FLOAT_ATOL
    assert result["max_abs_batched_vs_individual_RE_error"] <= FLOAT_ATOL


def test_near_singular_is_numeric_diagnostic_without_forced_zero_errors(apis):
    result = near_singular_check(apis)
    assert result["passed"], result
    for row in result["near_singular_rows"]:
        assert row["condition_number_2norm"] > 1e6
        assert row["behavior"] in ("finite_solution", "existing_solver_rejected")
        if row["behavior"] == "finite_solution":
            assert np.isfinite(row["max_abs_X_error"])
            assert np.isfinite(row["relative_normwise_backward_residual"])
        else:
            assert row["exception"]
    # Deliberately no assertion of zero bit errors or small forward error here.


def test_proper_complex_noise_synthetic_convention(apis):
    result = noise_convention_check(apis)
    assert result["passed"], result


def test_mismatched_tx_data_masks_are_rejected(apis):
    fixture = make_analytic_fixture(apis)
    mask = np.ones((2, 64, 16), dtype=bool)
    mask[0, 0, 0] = False
    with pytest.raises(ValueError, match="both TX streams"):
        apis["link"].detect_data_zf(fixture["y"], fixture["h"], mask)


def test_wrong_tx_rx_transpose_does_not_satisfy_analytic_symbols(apis):
    fixture = make_analytic_fixture(apis)
    wrong = apis["detectors"].zf_detect(fixture["y"], fixture["h"].swapaxes(0, 1))
    assert np.linalg.norm(wrong - fixture["x"]) / np.linalg.norm(fixture["x"]) > 1e-3


def test_existing_source_hashes_unchanged_by_receiver_checks(project_root):
    report = run_sanity(project_root)
    assert report["status"] == "PASS", report["failed_checks"]
    assert report["existing_source_preserved"]
    assert report["source_sha256_before"] == report["source_sha256_after"]
