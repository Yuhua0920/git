"""T05B time interpolation and complete noiseless Pilot-LS-LIN checks."""

import numpy as np
import pytest

from src.channels import static_mimo_receive
from src.estimators import (
    ls_linear_interpolate,
    pilot_ls_estimate,
    time_linear_interpolate_frequency_estimates,
)
from src.pilots import build_fdm_pilot_layout, map_pilot_symbols


ANCHORS = (0, 4, 8, 12)
TRUE_H = np.array(
    [[0.8 + 0.2j, -0.3 + 0.5j], [0.4 - 0.1j, 0.6 + 0.7j]],
    dtype=np.complex128,
)
PILOT_SYMBOLS = np.array(
    [(1 + 1j) / np.sqrt(2), (1 - 1j) / np.sqrt(2)], dtype=np.complex128
)


def _observed_on_anchors(truth: np.ndarray) -> np.ndarray:
    observations = np.full(truth.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    observations[..., list(ANCHORS)] = truth[..., list(ANCHORS)]
    return observations


def _constant_time_truth() -> np.ndarray:
    r = np.arange(2)[:, None, None, None]
    q = np.arange(2)[None, :, None, None]
    k = np.arange(64)[None, None, :, None]
    frequency_profile = (
        (0.3 + 2.0 * r + 7.0 * q + 0.07 * k)
        + 1j * (-0.4 + 3.0 * r - 5.0 * q - 0.11 * k)
    )
    return np.broadcast_to(frequency_profile, (2, 2, 64, 16)).copy()


def _affine_time_truth() -> np.ndarray:
    r = np.arange(2)[:, None, None, None]
    q = np.arange(2)[None, :, None, None]
    k = np.arange(64)[None, None, :, None]
    t = np.arange(16)[None, None, None, :]
    intercept = (
        (0.2 + 3.0 * r + 20.0 * q + 0.07 * k)
        + 1j * (-0.4 + 5.0 * r + 30.0 * q - 0.11 * k)
    )
    slope = (
        (0.01 + 1.2 * r + 3.4 * q + 0.006 * k)
        + 1j * (0.02 - 1.7 * r + 2.1 * q - 0.004 * k)
    )
    return np.asarray(intercept + slope * t, dtype=np.complex128)


def _manual_observations() -> np.ndarray:
    observations = np.full((2, 2, 64, 16), np.nan + 1j * np.nan, dtype=np.complex128)
    observations[..., list(ANCHORS)] = 1 + 2j
    observations[0, 0, 10, 0] = 2 + 5j
    observations[0, 0, 10, 4] = 10 - 3j
    return observations


def _noiseless_ls(spacing_frequency: int) -> tuple[np.ndarray, np.ndarray]:
    layout = build_fdm_pilot_layout(2, 64, 16, spacing_frequency, 4)
    x_grid = map_pilot_symbols(layout, PILOT_SYMBOLS)
    y_grid = static_mimo_receive(TRUE_H, x_grid)
    h_ls, estimate_mask = pilot_ls_estimate(y_grid, x_grid, layout.pilot_mask)
    np.testing.assert_array_equal(estimate_mask, layout.pilot_mask)
    return h_ls, layout.pilot_mask


def test_constant_time_channel_exact() -> None:
    truth = _constant_time_truth()
    h_full = time_linear_interpolate_frequency_estimates(_observed_on_anchors(truth), ANCHORS)
    assert h_full.shape == (2, 2, 64, 16)
    assert h_full.dtype == np.complex128
    relative_error = np.linalg.norm(h_full - truth) / np.linalg.norm(truth)
    assert relative_error <= 1e-12


def test_complex_affine_time_interior_exact() -> None:
    truth = _affine_time_truth()
    h_full = time_linear_interpolate_frequency_estimates(_observed_on_anchors(truth), ANCHORS)
    interior = slice(0, 13)
    relative_error = np.linalg.norm(h_full[..., interior] - truth[..., interior]) / np.linalg.norm(
        truth[..., interior]
    )
    assert relative_error <= 1e-12


def test_time_midpoint() -> None:
    h_full = time_linear_interpolate_frequency_estimates(_manual_observations(), ANCHORS)
    np.testing.assert_allclose(h_full[0, 0, 10, 2], 6 + 1j, rtol=0, atol=1e-15)


def test_time_quarterpoint() -> None:
    h_full = time_linear_interpolate_frequency_estimates(_manual_observations(), ANCHORS)
    np.testing.assert_allclose(h_full[0, 0, 10, 1], 4 + 3j, rtol=0, atol=1e-15)
    np.testing.assert_allclose(h_full[0, 0, 10, 3], 8 - 1j, rtol=0, atol=1e-15)


def test_right_time_edge_hold() -> None:
    h_full = time_linear_interpolate_frequency_estimates(
        _observed_on_anchors(_affine_time_truth()), ANCHORS
    )
    for t in (13, 14, 15):
        np.testing.assert_array_equal(h_full[..., t], h_full[..., 12])


def test_anchor_times_unchanged() -> None:
    observations = _observed_on_anchors(_affine_time_truth())
    h_full = time_linear_interpolate_frequency_estimates(observations, ANCHORS)
    for t in ANCHORS:
        np.testing.assert_array_equal(h_full[..., t], observations[..., t])


def test_all_values_finite_after_time_interpolation() -> None:
    h_full = time_linear_interpolate_frequency_estimates(
        _observed_on_anchors(_affine_time_truth()), ANCHORS
    )
    assert h_full.size == 4096
    assert np.count_nonzero(np.isfinite(h_full.real) & np.isfinite(h_full.imag)) == 4096


def test_rx_tx_subcarrier_axes_never_mixed() -> None:
    observations = np.full((2, 2, 64, 16), np.nan + 1j * np.nan, dtype=np.complex128)
    observations[..., list(ANCHORS)] = 1 + 0j
    # Distinct trends at neighboring RX, TX, and subcarrier coordinates.
    anchor_pairs = {
        (0, 0, 10): (2 + 5j, 10 - 3j),
        (1, 0, 10): (40 - 20j, 80 + 12j),
        (0, 1, 10): (-100 + 30j, -60 + 70j),
        (0, 0, 11): (200 - 400j, 100 + 200j),
    }
    for (r, q, k), (start, end) in anchor_pairs.items():
        observations[r, q, k, 0] = start
        observations[r, q, k, 4] = end
    h_full = time_linear_interpolate_frequency_estimates(observations, ANCHORS)
    for coordinate, expected in {
        (0, 0, 10): 6 + 1j,
        (1, 0, 10): 60 - 4j,
        (0, 1, 10): -80 + 50j,
        (0, 0, 11): 150 - 100j,
    }.items():
        np.testing.assert_allclose(h_full[*coordinate, 2], expected, rtol=0, atol=1e-14)


def test_full_ls_linear_constant_channel() -> None:
    h_ls, pilot_mask = _noiseless_ls(4)
    h_base = ls_linear_interpolate(h_ls, pilot_mask)
    expected = np.broadcast_to(TRUE_H[:, :, None, None], (2, 2, 64, 16))
    assert h_base.shape == (2, 2, 64, 16)
    assert h_base.dtype == np.complex128
    relative_error = np.linalg.norm(h_base - expected) / np.linalg.norm(expected)
    assert relative_error <= 1e-12
    assert np.all(np.isfinite(h_base.real) & np.isfinite(h_base.imag))


@pytest.mark.parametrize("spacing_frequency", [2, 4, 8, 16])
def test_all_four_df_full_ls_linear(spacing_frequency: int) -> None:
    h_ls, pilot_mask = _noiseless_ls(spacing_frequency)
    h_base = ls_linear_interpolate(h_ls, pilot_mask)
    expected = np.broadcast_to(TRUE_H[:, :, None, None], (2, 2, 64, 16))
    assert h_base.shape == expected.shape
    relative_error = np.linalg.norm(h_base - expected) / np.linalg.norm(expected)
    assert relative_error <= 1e-12


@pytest.mark.parametrize(
    "case",
    [
        "wrong_rank",
        "empty_axis",
        "real_input",
        "empty_indices",
        "negative_index",
        "out_of_range_index",
        "duplicate_indices",
        "descending_indices",
        "noninteger_index",
        "bool_index",
        "all_nan_anchor",
        "partial_nan_anchor",
        "imag_nan_anchor",
        "inf_anchor",
        "imag_inf_anchor",
        "inf_nonanchor",
    ],
)
def test_invalid_time_input_rejected(case: str) -> None:
    observations = _observed_on_anchors(_constant_time_truth())
    anchors = ANCHORS
    if case == "wrong_rank":
        observations = observations[0]
    elif case == "empty_axis":
        observations = observations[:, :, :0, :]
    elif case == "real_input":
        observations = observations.real.copy()
    elif case == "empty_indices":
        anchors = ()
    elif case == "negative_index":
        anchors = (-1, 0, 4, 8, 12)
    elif case == "out_of_range_index":
        anchors = (0, 4, 8, 16)
    elif case == "duplicate_indices":
        anchors = (0, 4, 4, 12)
    elif case == "descending_indices":
        anchors = (0, 8, 4, 12)
    elif case == "noninteger_index":
        anchors = (0, 4.0, 8, 12)
    elif case == "bool_index":
        anchors = (0, 4, True, 12)
    elif case == "all_nan_anchor":
        observations[..., 4] = np.nan + 1j * np.nan
    elif case == "partial_nan_anchor":
        observations[0, 0, 10, 4] = np.nan + 1j * np.nan
    elif case == "imag_nan_anchor":
        observations[0, 0, 10, 4] = 1 + 1j * np.nan
    elif case == "inf_anchor":
        observations[0, 0, 10, 4] = np.inf + 1j
    elif case == "imag_inf_anchor":
        observations[0, 0, 10, 4] = 1 + 1j * np.inf
    elif case == "inf_nonanchor":
        observations[0, 0, 10, 1] = np.inf + 0j
    with pytest.raises(ValueError):
        time_linear_interpolate_frequency_estimates(observations, anchors)
