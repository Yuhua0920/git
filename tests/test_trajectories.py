import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.special import j0

from src import trajectories as tr
from src.doppler import build_tdl_a_doppler_channel, ofdm_time_grid_seconds
from src.numerics import complex_gaussian_noise
from src.tdl import build_static_tdl_a_channel, fractional_delay_frequency_response, tdl_a_profile
from scripts.validate_multiframe_trajectory import (
    CORRELATION_TOLERANCES, LAGS, cross_frame_boundary_correlation,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    return tr.prepare_trajectory()


@pytest.fixture(scope="module")
def draw(model):
    return model.generate(17)


@pytest.fixture(scope="module")
def saved():
    return json.loads((ROOT / "outputs/t10c_pre_multiframe_trajectory_results.json").read_text())


def test_160_symbol_time_grid_and_duration(model):
    np.testing.assert_array_equal(model.time_grid_s, np.arange(160) * (80 / 1920000))
    assert model.symbol_duration == 80 / 1920000
    assert model.time_grid_s[-1] == 159 * model.symbol_duration
    assert model.time_grid_s[-1] < 160 * model.symbol_duration


def test_R160_shape_hermitian_diagonal_and_psd(model):
    r = model.covariance
    assert r.shape == model.factor.shape == (160, 160)
    np.testing.assert_array_equal(r, r.conj().T)
    np.testing.assert_array_equal(np.diag(r), np.ones(160))
    summary = model.factorization_summary
    assert summary["minimum_covariance_eigenvalue"] >= -1e-12
    assert summary["factor_reconstruction_max_error"] <= 5e-14
    assert summary["jitter_added"] == 0


def test_positive_eigenvalues_kept_without_rank_truncation(model):
    eigenvalues = np.array(model.factorization_summary["raw_eigenvalues"])
    column_powers = np.sum(np.abs(model.factor) ** 2, axis=0)
    positive = eigenvalues > 0
    assert np.all(column_powers[positive] > 0)
    np.testing.assert_allclose(column_powers[positive], eigenvalues[positive], rtol=2e-14, atol=0)
    np.testing.assert_array_equal(column_powers[eigenvalues < 0], 0)


@pytest.mark.parametrize("lag", LAGS)
def test_theoretical_long_lag_formula(model, lag):
    expected = j0(2 * np.pi * model.mobility["maximum_doppler_hz"] * model.symbol_duration * lag)
    # The independent scalar formula associates time/fD products differently.
    assert model.covariance[lag, 0] == pytest.approx(expected, rel=0, abs=1e-15)


def test_old_16_symbol_covariance_and_channel_regression_preserved():
    old = tr.prepare_trajectory(frames_per_trajectory=1)
    np.testing.assert_array_equal(old.time_grid_s, ofdm_time_grid_seconds())
    np.testing.assert_array_equal(old.covariance,
        j0(2*np.pi*old.mobility["maximum_doppler_hz"] * np.abs(old.time_grid_s[:,None]-old.time_grid_s[None,:])))
    archive = json.loads((ROOT / "outputs/t09b_tdl_awgn_results.json").read_text())
    h = build_tdl_a_doppler_channel(tr.CHANNEL_SEED_BASE)[1]
    assert hashlib.sha256(h.tobytes(order="C")).hexdigest() == archive["channel_realizations"][0]["H_sha256_complex128_C_order"]


def test_covariance_factor_once_and_one_draw_per_whole_trajectory():
    with patch.object(tr, "covariance_eigenfactor", wraps=tr.covariance_eigenfactor) as factors, \
         patch.object(tr, "jakes_covariance", wraps=tr.jakes_covariance) as covariances, \
         patch.object(tr, "correlated_complex_gaussian", wraps=tr.correlated_complex_gaussian) as gaussians:
        prepared = tr.prepare_trajectory()
        for tid in range(3):
            prepared.generate(tid)
        assert covariances.call_count == factors.call_count == 1
        assert gaussians.call_count == 3
        assert all(call.args[1] is prepared.factor for call in gaussians.call_args_list)
        assert all(call.args[2] == (2,2,23) for call in gaussians.call_args_list)


def test_long_process_reproducibility_and_trajectory_seed(model, draw):
    repeat = model.generate(17)
    other = model.generate(18)
    assert draw.channel_seed == 40261001 + 17
    np.testing.assert_array_equal(draw.taps, repeat.taps)
    np.testing.assert_array_equal(draw.h_long, repeat.h_long)
    assert not np.array_equal(draw.h_long, other.h_long)


def test_frame_slicing_exact_arrays_shared_memory_and_global_indices(draw):
    assert draw.taps.shape == (2,2,23,160)
    assert draw.h_long.shape == (2,2,64,160)
    assert draw.h_frames.shape == (10,2,2,64,16)
    assert np.shares_memory(draw.h_frames, draw.h_long)
    assert draw.h_frames.flags.writeable is False
    for record in draw.metadata:
        f = record["frame_index"]
        assert record == {"trajectory_id":17,"frame_index":f,
            "global_symbol_start":16*f,"global_symbol_stop":16*f+15}
        np.testing.assert_array_equal(draw.h_frames[f], draw.h_long[...,16*f:16*f+16])
    np.testing.assert_array_equal(np.concatenate(list(draw.h_frames), axis=-1), draw.h_long)


def test_frame0_symbol15_frame1_symbol0_is_global_lag1(model, draw):
    np.testing.assert_array_equal(draw.h_frames[0,...,15], draw.h_long[...,15])
    np.testing.assert_array_equal(draw.h_frames[1,...,0], draw.h_long[...,16])
    assert draw.metadata[1]["global_symbol_start"] - draw.metadata[0]["global_symbol_stop"] == 1
    assert model.covariance[16,15] == pytest.approx(model.covariance[1,0], abs=1e-15)


def test_no_trajectory_frame_time_normalization_and_reused_fractional_delay(model, draw):
    z = complex_gaussian_noise((2,2,23,160), 1, np.random.default_rng(draw.channel_seed))
    expected_taps = (z @ model.factor.T) * np.sqrt(tdl_a_profile()[1])[None,None,:,None]
    np.testing.assert_array_equal(draw.taps, expected_taps)
    direct = fractional_delay_frequency_response(np.moveaxis(draw.taps,-2,-1), model.delays_s, model.frequencies_hz)
    np.testing.assert_array_equal(draw.h_long, np.moveaxis(direct,-2,-1))
    frame_power = np.mean(np.abs(draw.h_frames)**2, axis=(1,2,3,4))
    assert np.ptp(frame_power) > 1e-8 and not np.allclose(frame_power, 1)
    assert not np.array_equal(draw.h_long, np.repeat(draw.h_long[...,:1],160,axis=-1))


def test_zero_speed_identical_ten_frames_and_exact_static_regression():
    draw = tr.prepare_trajectory(speed_kmh=0).generate(0)
    taps, h = build_static_tdl_a_channel(tr.CHANNEL_SEED_BASE, num_symbols=160)
    np.testing.assert_array_equal(draw.taps, np.repeat(taps[...,None],160,axis=-1))
    np.testing.assert_array_equal(draw.h_long, h)
    np.testing.assert_array_equal(draw.h_frames, np.repeat(draw.h_frames[:1],10,axis=0))


def test_trajectory_manifest_split_unit_and_variants_cannot_cross_split():
    manifest = tr.validate_trajectory_splits({"train":[17], "validation_id":[18]})
    assert manifest == {17:"train",18:"validation_id"}
    variants = [{"trajectory_id":17,"frame_index":f,"snr_db":snr,
        "dF":d_f,"noise_id":noise,"split":manifest[17]}
        for f in (0,9) for snr in (0,20) for d_f in (2,16) for noise in (0,1)]
    assert tr.validate_variant_splits(variants, manifest)
    with pytest.raises(ValueError, match="appears more than once"):
        tr.validate_trajectory_splits({"train":[17],"test_id":[17]})
    with pytest.raises(ValueError, match="violates trajectory-level split"):
        tr.validate_variant_splits([{**variants[-1],"split":"test_id"}],manifest)


@pytest.mark.parametrize("tid", [-1,True,1.5])
def test_invalid_trajectory_id_rejected(model, tid):
    with pytest.raises(ValueError):
        model.generate(tid)


def test_invalid_frame_extent_and_split_rejected():
    with pytest.raises(ValueError):
        tr.prepare_trajectory(frames_per_trajectory=0)
    with pytest.raises(ValueError):
        tr.slice_channel_frames(np.zeros((2,2,64,159),dtype=np.complex128))
    with pytest.raises(ValueError):
        tr.validate_trajectory_splits({"arbitrary":[17]})


@pytest.mark.parametrize("lag", LAGS)
def test_saved_empirical_long_lag_correlation(saved, lag):
    row = next(x for x in saved["empirical_correlations"] if x["lag_symbols"] == lag)
    assert row["independent_trajectory_count"] == 3000
    assert row["valid_pair_count"] == 3000*(160-lag)
    assert row["absolute_tolerance"] == CORRELATION_TOLERANCES[lag]
    assert row["absolute_error"] <= CORRELATION_TOLERANCES[lag] and row["passed"]


def test_saved_cross_frame_boundary_correlation_and_independent_frame_negative_sentinel(saved):
    diagnostic = saved["frame_boundary_diagnostic"]
    for source in ("unit_process","full_tdl"):
        assert diagnostic[source]["lag_symbols"] == 1 and diagnostic[source]["passed"]
        assert diagnostic[source]["empirical_real"] > 0.95
    wrong = diagnostic["independent_frame_negative_sentinel"]
    assert wrong["detected"] and not wrong["passed"]
    assert abs(complex(wrong["empirical_real"], wrong["empirical_imag"])) <= 0.10


@pytest.mark.parametrize("link", ["rx0_tx0","rx0_tx1","rx1_tx0","rx1_tx1"])
def test_saved_full_tdl_pdp_and_frequency_power(saved, link):
    power = saved["ensemble_power_statistics"]
    assert power["num_independent_trajectories"] == 1000
    for name in ("tap_energy_by_link","frequency_power_by_link"):
        assert abs(power[name][link]["mean"]-1) <= 0.10
    assert power["max_per_link_per_tap_relative_error"] <= 0.15
    assert all(saved["full_tdl_statistics"]["checks"].values())


def test_saved_protocol_counters_memory_and_sources(saved):
    assert saved["all_checks_passed"] and all(saved["checks"].values())
    factor = saved["factorization_summary"]
    assert factor["main_30kmh_160_covariance_preparation_count"] == factor["main_30kmh_160_factorization_count"] == 1
    assert factor["positive_eigenvalues_truncated"] == 0 and not factor["rank_truncation_used"]
    assert saved["zero_speed_regression"]["passed"]
    assert not any(saved[k] for k in ("per_trajectory_normalization","per_frame_normalization","per_time_normalization"))
    assert saved["memory_estimate"]["single_H_long_bytes"] == 2*2*64*160*16
    assert saved["memory_estimate"]["H_frames_is_view"]
    assert saved["dataset_split_contract"]["split_unit"] == "trajectory_id"
    assert saved["dataset_split_contract"]["formal_test_sealed"]
    assert not saved["dataset_split_contract"]["formal_datasets_generated"]
    assert all(saved["frozen_T10B_source_checks"].values())
    for path, digest in saved["provenance"]["source_sha256"].items():
        if path in {"docs/dataset_protocol.md", "tests/test_trajectories.py"}:
            # Researcher-approved T10C migration: these are historical artifacts.
            # Runtime generator/config sources still check the current bytes.
            data = subprocess.run(["git", "show",
                f"c0a1ac3525cde5d90a9fe26b75b00170d2416caf:{path}"],
                cwd=ROOT, check=True, capture_output=True).stdout
        elif path == "scripts/validate_multiframe_trajectory.py":
            from src.provenance import recorded_source_matches
            assert recorded_source_matches(ROOT,path,digest,"t10c_pre")
            continue
        else:
            data = (ROOT/path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
