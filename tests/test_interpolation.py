"""T05A frequency-only interpolation of sparse Pilot-LS observations."""

import numpy as np
import pytest

from src.estimators import frequency_linear_interpolate_ls
from src.pilots import build_fdm_pilot_layout


CONSTANT_H = np.array(
    [[0.8 + 0.2j, -0.3 + 0.5j], [0.4 - 0.1j, 0.6 + 0.7j]],
    dtype=np.complex128,
)
AFFINE_INTERCEPT = np.array(
    [[0.2 + 0.1j, 10.0 - 4.0j], [-0.7 + 0.8j, -20.0 + 30.0j]],
    dtype=np.complex128,
)
AFFINE_SLOPE = np.array(
    [[0.01 - 0.003j, -0.08 + 0.04j], [0.015 + 0.007j, 0.3 - 0.11j]],
    dtype=np.complex128,
)


@pytest.fixture
def layout():
    return build_fdm_pilot_layout(2, 64, 16, 4, 4)


def _empty_ls(layout) -> np.ndarray:
    return np.full((2, 2, 64, 16), np.nan + 1j * np.nan, dtype=np.complex128)


def _constant_pilots(layout) -> np.ndarray:
    h_ls = _empty_ls(layout)
    for q in range(2):
        h_ls[:, q, layout.pilot_mask[q]] = CONSTANT_H[:, q, None]
    return h_ls


def _affine_pilots(layout, intercept=AFFINE_INTERCEPT, slope=AFFINE_SLOPE) -> np.ndarray:
    h_ls = _empty_ls(layout)
    for t in layout.pilot_symbol_indices:
        time_shift = (t // 4) * (0.07 - 0.09j)
        for q in range(2):
            pilot_k = np.flatnonzero(layout.pilot_mask[q, :, t])
            h_ls[:, q, pilot_k, t] = (
                intercept[:, q, None] + time_shift + slope[:, q, None] * pilot_k[None, :]
            )
    return h_ls


def test_constant_channel_frequency_interpolation(layout) -> None:
    h_freq, output_mask = frequency_linear_interpolate_ls(
        _constant_pilots(layout), layout.pilot_mask
    )
    assert h_freq.shape == (2, 2, 64, 16)
    assert h_freq.dtype == np.complex128
    assert output_mask.shape == (2, 64, 16)
    assert output_mask.dtype == np.bool_
    valid_time = np.array(layout.pilot_symbol_indices)
    expected = np.broadcast_to(CONSTANT_H[:, :, None, None], (2, 2, 64, valid_time.size))
    actual = h_freq[:, :, :, valid_time]
    relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_error <= 1e-12
    assert np.all(output_mask[:, :, valid_time])


def test_complex_affine_channel_interior_exact(layout) -> None:
    h_freq, _ = frequency_linear_interpolate_ls(_affine_pilots(layout), layout.pilot_mask)
    for t in layout.pilot_symbol_indices:
        time_shift = (t // 4) * (0.07 - 0.09j)
        for q, first_k, last_k in ((0, 0, 60), (1, 2, 62)):
            interior_k = np.arange(first_k, last_k + 1)
            expected = (
                AFFINE_INTERCEPT[:, q, None]
                + time_shift
                + AFFINE_SLOPE[:, q, None] * interior_k[None, :]
            )
            actual = h_freq[:, q, interior_k, t]
            relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            assert relative_error <= 1e-12


def test_tx0_right_boundary_edge_hold(layout) -> None:
    h_ls = _affine_pilots(layout)
    h_freq, _ = frequency_linear_interpolate_ls(h_ls, layout.pilot_mask)
    for t in layout.pilot_symbol_indices:
        np.testing.assert_array_equal(
            h_freq[:, 0, 61:64, t],
            np.broadcast_to(h_ls[:, 0, 60, t, None], (2, 3)),
        )


def test_tx1_left_boundary_edge_hold(layout) -> None:
    h_ls = _affine_pilots(layout)
    h_freq, _ = frequency_linear_interpolate_ls(h_ls, layout.pilot_mask)
    for t in layout.pilot_symbol_indices:
        np.testing.assert_array_equal(
            h_freq[:, 1, 0:2, t],
            np.broadcast_to(h_ls[:, 1, 2, t, None], (2, 2)),
        )


def test_tx1_right_boundary_edge_hold(layout) -> None:
    h_ls = _affine_pilots(layout)
    h_freq, _ = frequency_linear_interpolate_ls(h_ls, layout.pilot_mask)
    for t in layout.pilot_symbol_indices:
        np.testing.assert_array_equal(h_freq[:, 1, 63, t], h_ls[:, 1, 62, t])


def test_midpoint_interpolation(layout) -> None:
    h_ls = _affine_pilots(layout)
    h_freq, _ = frequency_linear_interpolate_ls(h_ls, layout.pilot_mask)
    np.testing.assert_allclose(h_ls[0, 0, 0, 0], 0.2 + 0.1j, rtol=0, atol=1e-15)
    np.testing.assert_allclose(h_ls[0, 0, 4, 0], 0.24 + 0.088j, rtol=0, atol=1e-15)
    np.testing.assert_allclose(h_freq[0, 0, 1, 0], 0.21 + 0.097j, rtol=0, atol=1e-15)
    np.testing.assert_allclose(h_freq[0, 0, 2, 0], 0.22 + 0.094j, rtol=0, atol=1e-15)


def test_pilot_values_unchanged(layout) -> None:
    h_ls = _affine_pilots(layout)
    h_freq, _ = frequency_linear_interpolate_ls(h_ls, layout.pilot_mask)
    pilot_entries = np.broadcast_to(layout.pilot_mask[None, :, :, :], h_ls.shape)
    np.testing.assert_array_equal(h_freq[pilot_entries], h_ls[pilot_entries])


def test_nonpilot_time_symbols_remain_nan(layout) -> None:
    h_freq, output_mask = frequency_linear_interpolate_ls(
        _affine_pilots(layout), layout.pilot_mask
    )
    nonpilot_times = sorted(set(range(16)) - set(layout.pilot_symbol_indices))
    assert len(nonpilot_times) == 12
    assert np.all(np.isnan(h_freq[:, :, :, nonpilot_times].real))
    assert np.all(np.isnan(h_freq[:, :, :, nonpilot_times].imag))
    assert not np.any(output_mask[:, :, nonpilot_times])
    assert np.all(output_mask[:, :, list(layout.pilot_symbol_indices)])


def test_tx_channels_never_mixed(layout) -> None:
    intercept = np.array(
        [[0.2 + 0.1j, 100.0 + 300.0j], [-0.7 + 0.8j, -200.0 + 400.0j]],
        dtype=np.complex128,
    )
    slope = np.array(
        [[0.01 - 0.003j, -3.0 + 2.0j], [0.015 + 0.007j, 4.0 - 5.0j]],
        dtype=np.complex128,
    )
    h_freq, _ = frequency_linear_interpolate_ls(
        _affine_pilots(layout, intercept, slope), layout.pilot_mask
    )
    # These off-pilot coordinates lie between pilots belonging to different TXs.
    np.testing.assert_allclose(h_freq[0, 0, 2, 0], 0.22 + 0.094j, rtol=0, atol=1e-14)
    np.testing.assert_allclose(h_freq[0, 1, 4, 0], 88.0 + 308.0j, rtol=0, atol=1e-12)
    assert abs(h_freq[0, 0, 2, 0] - h_freq[0, 1, 2, 0]) > 100


@pytest.mark.parametrize("spacing_frequency", [2, 4, 8, 16])
def test_all_four_df_supported(spacing_frequency: int) -> None:
    layout = build_fdm_pilot_layout(2, 64, 16, spacing_frequency, 4)
    h_ls = _affine_pilots(layout)
    h_freq, output_mask = frequency_linear_interpolate_ls(h_ls, layout.pilot_mask)
    assert h_freq.shape == h_ls.shape
    assert output_mask.shape == layout.pilot_mask.shape
    assert output_mask.dtype == np.bool_
    pilot_entries = np.broadcast_to(layout.pilot_mask[None, :, :, :], h_ls.shape)
    np.testing.assert_array_equal(h_freq[pilot_entries], h_ls[pilot_entries])
    nonpilot_times = sorted(set(range(16)) - set(layout.pilot_symbol_indices))
    assert np.all(np.isnan(h_freq[:, :, :, nonpilot_times].real))
    assert np.all(np.isnan(h_freq[:, :, :, nonpilot_times].imag))
    assert not np.any(output_mask[:, :, nonpilot_times])


@pytest.mark.parametrize(
    "case, message",
    [
        ("h_shape", "pilot_mask must be a bool array"),
        ("h_ndim", "h_ls must have shape"),
        ("mask_shape", "pilot_mask must be a bool array"),
        ("mask_ndim", "pilot_mask must be a bool array"),
        ("mask_dtype", "pilot_mask must be a bool array"),
        ("no_pilot", "pilot observations"),
        ("missing_tx", "TX 1 has no pilots"),
        ("nan_pilot", "finite values at every marked pilot"),
        ("inf_nonpilot", "must not contain Inf"),
        ("real_h", "complex floating-point dtype"),
    ],
)
def test_invalid_input_rejected(layout, case: str, message: str) -> None:
    h_ls = _constant_pilots(layout)
    mask = layout.pilot_mask.copy()
    if case == "h_shape":
        h_ls = h_ls[:, :, :-1, :]
    elif case == "h_ndim":
        h_ls = h_ls[:, 0, :, :]
    elif case == "mask_shape":
        mask = mask[:, :-1, :]
    elif case == "mask_ndim":
        mask = mask[0]
    elif case == "mask_dtype":
        mask = mask.astype(np.int8)
    elif case == "no_pilot":
        mask[:] = False
    elif case == "missing_tx":
        mask[1, :, 4] = False
    elif case == "nan_pilot":
        h_ls[0, 0, 0, 0] = np.nan + 1j * np.nan
    elif case == "inf_nonpilot":
        h_ls[0, 0, 1, 1] = np.inf + 0j
    elif case == "real_h":
        h_ls = np.zeros(h_ls.shape, dtype=np.float64)
    with pytest.raises(ValueError, match=message):
        frequency_linear_interpolate_ls(h_ls, mask)
