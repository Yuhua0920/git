"""T04B pilot-only 2x2 MIMO mapping and noiseless Pilot-LS checks."""

import numpy as np
import pytest

from src.channels import static_mimo_receive
from src.estimators import pilot_ls_estimate
from src.pilots import build_fdm_pilot_layout, map_pilot_symbols


PILOT_SYMBOLS = np.array(
    [(1 + 1j) / np.sqrt(2), (1 - 1j) / np.sqrt(2)], dtype=np.complex128
)
# H[r, q] is the channel from transmit antenna q to receive antenna r.
TRUE_H = np.array(
    [[0.8 + 0.2j, -0.3 + 0.5j], [0.4 - 0.1j, 0.6 + 0.7j]],
    dtype=np.complex128,
)


@pytest.fixture
def layout():
    return build_fdm_pilot_layout(2, 64, 16, 4, 4)


@pytest.fixture
def pilot_grid(layout):
    return map_pilot_symbols(layout, PILOT_SYMBOLS)


@pytest.fixture
def received_grid(pilot_grid):
    return static_mimo_receive(TRUE_H, pilot_grid)


@pytest.fixture
def ls_result(layout, pilot_grid, received_grid):
    return pilot_ls_estimate(received_grid, pilot_grid, layout.pilot_mask)


def test_pilot_symbols_unit_energy(layout, pilot_grid) -> None:
    assert PILOT_SYMBOLS[0] != PILOT_SYMBOLS[1]
    np.testing.assert_allclose(np.abs(pilot_grid[layout.pilot_mask]) ** 2, 1.0, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(
        pilot_grid[0, layout.pilot_mask[0]],
        np.full(64, PILOT_SYMBOLS[0], dtype=np.complex128),
    )
    np.testing.assert_array_equal(
        pilot_grid[1, layout.pilot_mask[1]],
        np.full(64, PILOT_SYMBOLS[1], dtype=np.complex128),
    )


def test_pilot_grid_nonzero_only_on_pilot_mask(layout, pilot_grid) -> None:
    assert pilot_grid.shape == (2, 64, 16)
    assert pilot_grid.dtype == np.complex128
    assert np.count_nonzero(pilot_grid) == 128
    assert np.all(pilot_grid[layout.pilot_mask] != 0)
    assert np.all(pilot_grid[~layout.pilot_mask] == 0)
    assert np.all(pilot_grid[:, ~layout.pilot_union] == 0)


def test_non_owner_is_zero(layout, pilot_grid) -> None:
    assert np.all(pilot_grid[layout.mute_mask] == 0)
    np.testing.assert_array_equal(
        np.count_nonzero(pilot_grid[:, layout.pilot_union], axis=0),
        np.ones(128, dtype=int),
    )


def test_union_pilot_total_power_is_one(layout, pilot_grid) -> None:
    total_power = np.sum(np.abs(pilot_grid) ** 2, axis=0)
    np.testing.assert_allclose(total_power[layout.pilot_union], 1.0, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(total_power[~layout.pilot_union], 0.0)


def test_static_mimo_received_shape(received_grid) -> None:
    assert received_grid.shape == (2, 64, 16)
    assert received_grid.dtype == np.complex128
    assert np.all(np.isfinite(received_grid))


def test_tx0_manual_coordinate(layout, pilot_grid, received_grid, ls_result) -> None:
    h_ls, estimate_mask = ls_result
    assert layout.pilot_owner[0, 0] == 0
    np.testing.assert_array_equal(pilot_grid[:, 0, 0], [PILOT_SYMBOLS[0], 0])
    np.testing.assert_allclose(received_grid[:, 0, 0], TRUE_H[:, 0] * PILOT_SYMBOLS[0], rtol=0, atol=1e-15)
    assert estimate_mask[0, 0, 0]
    np.testing.assert_allclose(h_ls[:, 0, 0, 0], TRUE_H[:, 0], rtol=0, atol=1e-15)


def test_tx1_manual_coordinate(layout, pilot_grid, received_grid, ls_result) -> None:
    h_ls, estimate_mask = ls_result
    assert layout.pilot_owner[2, 0] == 1
    np.testing.assert_array_equal(pilot_grid[:, 2, 0], [0, PILOT_SYMBOLS[1]])
    np.testing.assert_allclose(received_grid[:, 2, 0], TRUE_H[:, 1] * PILOT_SYMBOLS[1], rtol=0, atol=1e-15)
    assert estimate_mask[1, 2, 0]
    np.testing.assert_allclose(h_ls[:, 1, 2, 0], TRUE_H[:, 1], rtol=0, atol=1e-15)


def test_noiseless_pilot_ls_tx0(layout, ls_result) -> None:
    h_ls, estimate_mask = ls_result
    assert np.count_nonzero(estimate_mask[0]) == 64
    expected = np.repeat(TRUE_H[:, 0, None], 64, axis=1)
    np.testing.assert_allclose(h_ls[:, 0][:, layout.pilot_mask[0]], expected, rtol=0, atol=1e-15)


def test_noiseless_pilot_ls_tx1(layout, ls_result) -> None:
    h_ls, estimate_mask = ls_result
    assert np.count_nonzero(estimate_mask[1]) == 64
    expected = np.repeat(TRUE_H[:, 1, None], 64, axis=1)
    np.testing.assert_allclose(h_ls[:, 1][:, layout.pilot_mask[1]], expected, rtol=0, atol=1e-15)


def test_noiseless_pilot_ls_all(layout, ls_result) -> None:
    h_ls, estimate_mask = ls_result
    assert h_ls.shape == (2, 2, 64, 16)
    assert h_ls.dtype == np.complex128
    np.testing.assert_array_equal(estimate_mask, layout.pilot_mask)
    assert np.count_nonzero(estimate_mask, axis=(1, 2)).tolist() == [64, 64]
    valid = np.broadcast_to(estimate_mask[None, :, :, :], h_ls.shape)
    assert np.count_nonzero(valid) == 256
    truth = np.broadcast_to(TRUE_H[:, :, None, None], h_ls.shape)
    relative_error = np.linalg.norm(h_ls[valid] - truth[valid]) / np.linalg.norm(truth[valid])
    assert relative_error <= 1e-12


def test_ls_does_not_estimate_nonpilot_locations(layout, ls_result) -> None:
    h_ls, estimate_mask = ls_result
    np.testing.assert_array_equal(estimate_mask, layout.pilot_mask)
    nonpilot = np.broadcast_to(~estimate_mask[None, :, :, :], h_ls.shape)
    assert np.all(np.isnan(h_ls.real[nonpilot]))
    assert np.all(np.isnan(h_ls.imag[nonpilot]))
    assert not estimate_mask[:, 1, 0].any()
    assert not estimate_mask[:, 0, 1].any()


def test_zero_pilot_symbol_rejected(layout, pilot_grid, received_grid) -> None:
    zero_symbol = PILOT_SYMBOLS.copy()
    zero_symbol[0] = 0
    with pytest.raises(ValueError, match="nonzero"):
        map_pilot_symbols(layout, zero_symbol)

    zero_grid = pilot_grid.copy()
    zero_grid[0, 0, 0] = 0
    with pytest.raises(ValueError, match="nonzero"):
        pilot_ls_estimate(received_grid, zero_grid, layout.pilot_mask)


def test_contaminated_pilot_breaks_tx0_ls(layout, pilot_grid, received_grid, ls_result) -> None:
    clean_h_ls, _ = ls_result
    contaminated_grid = pilot_grid.copy()
    contaminated_grid[1, 0, 0] = PILOT_SYMBOLS[1]
    contaminated_y = static_mimo_receive(TRUE_H, contaminated_grid)
    np.testing.assert_allclose(
        contaminated_y[:, 0, 0],
        TRUE_H[:, 0] * PILOT_SYMBOLS[0] + TRUE_H[:, 1] * PILOT_SYMBOLS[1],
        rtol=0,
        atol=1e-15,
    )
    contaminated_h_ls, _ = pilot_ls_estimate(contaminated_y, contaminated_grid, layout.pilot_mask)
    clean_error = np.linalg.norm(clean_h_ls[:, 0, 0, 0] - TRUE_H[:, 0])
    contaminated_error = np.linalg.norm(contaminated_h_ls[:, 0, 0, 0] - TRUE_H[:, 0])
    assert contaminated_error > 1_000 * max(clean_error, np.finfo(np.float64).eps)


@pytest.mark.parametrize("spacing_frequency", [2, 4, 8, 16])
def test_pilot_ls_reuses_all_t04a_layouts(spacing_frequency: int) -> None:
    layout = build_fdm_pilot_layout(2, 64, 16, spacing_frequency, 4)
    x_grid = map_pilot_symbols(layout, PILOT_SYMBOLS)
    y_grid = static_mimo_receive(TRUE_H, x_grid)
    h_ls, estimate_mask = pilot_ls_estimate(y_grid, x_grid, layout.pilot_mask)
    np.testing.assert_array_equal(estimate_mask, layout.pilot_mask)
    for q in range(2):
        num_pilots = np.count_nonzero(layout.pilot_mask[q])
        expected = np.repeat(TRUE_H[:, q, None], num_pilots, axis=1)
        np.testing.assert_allclose(h_ls[:, q][:, layout.pilot_mask[q]], expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("bad_symbols", [[1 + 0j], [1 + 0j, 1 + 0j, 1 + 0j], [1 + 0j, 2 + 0j]])
def test_pilot_mapping_rejects_wrong_shape_or_energy(layout, bad_symbols) -> None:
    with pytest.raises(ValueError):
        map_pilot_symbols(layout, bad_symbols)


@pytest.mark.parametrize(
    "bad_h,bad_x",
    [
        (np.zeros((2, 2, 1)), np.zeros((2, 64, 16))),
        (np.zeros((2, 2)), np.zeros((3, 64, 16))),
        (np.zeros((2, 2)), np.zeros((2, 64))),
    ],
)
def test_static_mimo_receive_rejects_invalid_shapes(bad_h, bad_x) -> None:
    with pytest.raises(ValueError):
        static_mimo_receive(bad_h, bad_x)


@pytest.mark.parametrize("bad_argument", ["y_shape", "x_shape", "mask_shape", "mask_dtype"])
def test_pilot_ls_rejects_invalid_shapes(layout, pilot_grid, received_grid, bad_argument) -> None:
    y_grid = received_grid
    x_grid = pilot_grid
    mask = layout.pilot_mask
    if bad_argument == "y_shape":
        y_grid = received_grid[:, :-1, :]
    elif bad_argument == "x_shape":
        x_grid = pilot_grid[:, :, :-1]
    elif bad_argument == "mask_shape":
        mask = layout.pilot_mask[:, :-1, :]
    else:
        mask = layout.pilot_mask.astype(np.int8)
    with pytest.raises(ValueError):
        pilot_ls_estimate(y_grid, x_grid, mask)
