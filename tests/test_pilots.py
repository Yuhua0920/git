"""T04A checks for 2x2 orthogonal FDM pilot resource masks."""

from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.pilots import build_fdm_pilot_layout, count_layout_resources


CONFIG = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
NT = CONFIG["system"]["nt"]
K = CONFIG["system"]["active_carriers"]
T = CONFIG["system"]["num_symbols"]
DT = CONFIG["pilot"]["spacing_time"]
PILOT_CASES = [
    (2, (0, 1), 128, 256, 1 / 4, 1536),
    (4, (0, 2), 64, 128, 1 / 8, 1792),
    (8, (0, 4), 32, 64, 1 / 16, 1920),
    (16, (0, 8), 16, 32, 1 / 32, 1984),
]


def _layout(spacing_frequency: int):
    return build_fdm_pilot_layout(NT, K, T, spacing_frequency, DT)


@pytest.mark.parametrize("df,offsets,per_tx,union,ratio,data", PILOT_CASES)
def test_four_expected_pilot_ratios(df, offsets, per_tx, union, ratio, data) -> None:
    layout = _layout(df)
    counts = count_layout_resources(layout)
    assert counts["total_re"] == 1024
    assert counts["union_ratio"] == ratio
    assert np.count_nonzero(layout.pilot_union) / (K * T) == ratio


@pytest.mark.parametrize("df,offsets,per_tx,union,ratio,data", PILOT_CASES)
def test_expected_pilot_counts_per_tx(df, offsets, per_tx, union, ratio, data) -> None:
    layout = _layout(df)
    assert count_layout_resources(layout)["pilot_count_by_tx"] == [per_tx, per_tx]
    assert np.count_nonzero(layout.pilot_mask, axis=(1, 2)).tolist() == [per_tx, per_tx]


@pytest.mark.parametrize("df,offsets,per_tx,union,ratio,data", PILOT_CASES)
def test_expected_union_counts(df, offsets, per_tx, union, ratio, data) -> None:
    layout = _layout(df)
    assert count_layout_resources(layout)["union_count"] == union
    np.testing.assert_array_equal(layout.pilot_union, np.any(layout.pilot_mask, axis=0))


@pytest.mark.parametrize("df,offsets,per_tx,union,ratio,data", PILOT_CASES)
def test_expected_two_stream_data_counts(df, offsets, per_tx, union, ratio, data) -> None:
    counts = count_layout_resources(_layout(df))
    assert counts["two_stream_data_symbols"] == data
    assert counts["data_count_by_tx"] == [data // NT] * NT
    assert counts["mute_count_by_tx"] == [per_tx, per_tx]


@pytest.mark.parametrize("df", [2, 4, 8, 16])
def test_no_pilot_overlap_between_tx(df: int) -> None:
    layout = _layout(df)
    assert layout.pilot_mask.shape == (NT, K, T)
    assert layout.pilot_union.shape == (K, T)
    assert layout.pilot_mask.dtype == np.bool_
    assert layout.pilot_union.dtype == np.bool_
    assert np.all(np.sum(layout.pilot_mask, axis=0) <= 1)


@pytest.mark.parametrize("df", [2, 4, 8, 16])
def test_non_owner_tx_muted_on_pilot(df: int) -> None:
    layout = _layout(df)
    assert layout.mute_mask.shape == (NT, K, T)
    assert layout.mute_mask.dtype == np.bool_
    assert layout.pilot_owner.shape == (K, T)
    for k, t in np.argwhere(layout.pilot_union):
        owner = int(layout.pilot_owner[k, t])
        assert owner in (0, 1)
        assert layout.pilot_mask[owner, k, t]
        assert not layout.mute_mask[owner, k, t]
        assert not layout.pilot_mask[1 - owner, k, t]
        assert layout.mute_mask[1 - owner, k, t]


@pytest.mark.parametrize("df", [2, 4, 8, 16])
def test_no_data_on_union_pilot_re(df: int) -> None:
    layout = _layout(df)
    assert layout.data_mask.shape == (NT, K, T)
    assert layout.data_mask.dtype == np.bool_
    assert not np.any(layout.data_mask[:, layout.pilot_union])


@pytest.mark.parametrize("df", [2, 4, 8, 16])
def test_both_tx_data_outside_pilot_union(df: int) -> None:
    layout = _layout(df)
    assert np.all(layout.data_mask[:, ~layout.pilot_union])
    assert not np.any(layout.pilot_mask[:, ~layout.pilot_union])
    assert not np.any(layout.mute_mask[:, ~layout.pilot_union])
    assert np.all(layout.pilot_owner[~layout.pilot_union] == -1)


@pytest.mark.parametrize("df", [2, 4, 8, 16])
def test_masks_partition_tx_resources(df: int) -> None:
    layout = _layout(df)
    resource_count = (
        layout.pilot_mask.astype(np.int8)
        + layout.data_mask.astype(np.int8)
        + layout.mute_mask.astype(np.int8)
    )
    assert np.all(resource_count == 1)
    np.testing.assert_array_equal(layout.data_mask[0], ~layout.pilot_union)
    np.testing.assert_array_equal(layout.data_mask[1], ~layout.pilot_union)


@pytest.mark.parametrize("df", [2, 4, 8, 16])
def test_expected_pilot_symbol_indices(df: int) -> None:
    layout = _layout(df)
    assert list(layout.pilot_symbol_indices) == [0, 4, 8, 12]
    assert np.array_equal(np.flatnonzero(np.any(layout.pilot_union, axis=0)), [0, 4, 8, 12])


@pytest.mark.parametrize("df,offsets,per_tx,union,ratio,data", PILOT_CASES)
def test_expected_offsets(df, offsets, per_tx, union, ratio, data) -> None:
    layout = _layout(df)
    assert list(layout.offsets_by_tx) == list(offsets)
    for q, offset in enumerate(offsets):
        expected_frequency = np.arange(K) % df == offset
        for t in layout.pilot_symbol_indices:
            np.testing.assert_array_equal(layout.pilot_mask[q, :, t], expected_frequency)


def test_invalid_df_not_divisible_by_nt() -> None:
    with pytest.raises(ValueError, match="divisible"):
        build_fdm_pilot_layout(2, 64, 16, 3, 4)


@pytest.mark.parametrize("field", ["nt", "num_subcarriers", "num_symbols", "spacing_frequency", "spacing_time"])
@pytest.mark.parametrize("bad_value", [0, -1, 1.5, True])
def test_invalid_nonpositive_or_noninteger_parameters(field: str, bad_value) -> None:
    parameters = dict(nt=2, num_subcarriers=64, num_symbols=16, spacing_frequency=4, spacing_time=4)
    parameters[field] = bad_value
    with pytest.raises(ValueError, match=field):
        build_fdm_pilot_layout(**parameters)


def test_offset_outside_frequency_grid_rejected() -> None:
    with pytest.raises(ValueError, match="offset"):
        build_fdm_pilot_layout(2, 1, 16, 2, 4)


@pytest.mark.parametrize(
    "k,t,pilot,mute,data,owner",
    [
        (0, 0, [True, False], [False, True], [False, False], 0),
        (2, 0, [False, True], [True, False], [False, False], 1),
        (4, 0, [True, False], [False, True], [False, False], 0),
        (6, 0, [False, True], [True, False], [False, False], 1),
        (1, 0, [False, False], [False, False], [True, True], -1),
        (3, 0, [False, False], [False, False], [True, True], -1),
        (5, 0, [False, False], [False, False], [True, True], -1),
        (0, 1, [False, False], [False, False], [True, True], -1),
    ],
)
def test_df4_manual_coordinates(k, t, pilot, mute, data, owner) -> None:
    layout = _layout(4)
    assert layout.pilot_mask[:, k, t].tolist() == pilot
    assert layout.mute_mask[:, k, t].tolist() == mute
    assert layout.data_mask[:, k, t].tolist() == data
    assert layout.pilot_owner[k, t] == owner
