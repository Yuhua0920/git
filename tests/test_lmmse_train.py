import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import cho_solve

from src import lmmse_train as lt
from src import lmmse
from src.trajectories import prepare_trajectory
from src.pilots import build_fdm_pilot_layout
from scripts.validate_lmmse_train import (mismatch_error_diagonal, validation_noise_seed,
    CONDITIONS, NMSE_TOLERANCE_DB, formal_trajectory)

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = np.array([[1+2j,3-1j],[-2+1j,1+3j],[2-1j,-1+2j]],dtype=np.complex128)
# Offdiagonal products: 1+7j, 1+7j, -4-3j; sum = -2+11j.
HAND = np.array([[5,(-2+11j)/3],[(-2-11j)/3,25/3]],dtype=np.complex128)


def test_complex_second_moment_independent_hand_sentinel_and_wrong_orientation():
    a = lt.SecondMomentAccumulator(2)
    a.add(SENTINEL)
    r,mean,stats = a.finish()
    np.testing.assert_allclose(r,HAND,rtol=0,atol=2e-15)
    np.testing.assert_allclose(mean,np.array([(1+2j)/3,(3+4j)/3]),rtol=0,atol=1e-15)
    wrong = SENTINEL.conj().T @ SENTINEL / 3
    assert np.max(np.abs(wrong-HAND)) > 7
    assert stats["denominator"] == 3 and not stats["sample_mean_subtracted"]
    assert not np.allclose(r,np.cov(SENTINEL,rowvar=False))
    assert stats["pre_symmetry_max_error"] <= 1e-14
    assert stats["max_symmetrization_correction"] <= 1e-14
    assert stats["post_symmetry_max_error"] == 0
    # Nonunit sample diagonal survives; no normalization, shrinkage or blending.
    np.testing.assert_allclose(np.diag(r),[5,25/3],rtol=0,atol=2e-15)


@pytest.mark.parametrize("sizes",[(3,),(1,1,1),(1,2),(2,1)])
def test_streaming_batch_partitions_equal_direct_calculation(sizes):
    a = lt.SecondMomentAccumulator(2)
    start = 0
    for size in sizes:
        a.add(SENTINEL[start:start+size]); start += size
    r,_,stats = a.finish()
    np.testing.assert_allclose(r,HAND,rtol=0,atol=2e-15)
    assert stats["sample_count"] == 3


def test_formal_manifest_disjoint_namespace_and_corruption_rejection(tmp_path):
    path = tmp_path/"manifest.csv"
    rows = lt.ensure_manifest(path)
    before = lt.sha256_file(path)
    assert len(rows) == 2400
    assert [r["trajectory_id"] for r in rows if r["split"]=="train"] == list(range(2000))
    assert [r["trajectory_id"] for r in rows if r["split"]=="validation_id"] == list(range(2000,2400))
    assert rows[0]["channel_seed"] == 60261001 and rows[-1]["channel_seed"] == 60263400
    assert {r["channel_seed"] for r in rows}.isdisjoint({40261001+i for i in range(2400)})
    assert lt.ensure_manifest(path) == rows and lt.sha256_file(path) == before
    path.write_text(path.read_text(encoding="utf-8").replace("60261001","40261001",1),encoding="utf-8")
    assert lt.sha256_file(path) != before
    with pytest.raises(ValueError,match="manifest differs"):
        lt.read_manifest(path)


@pytest.fixture(scope="module")
def model():
    return prepare_trajectory()


def test_continuous_160_generator_formal_seed_and_exact_C_flatten(model):
    row = lt.formal_manifest_rows()[1]
    # Calling the baseline generate(tid) or old 16-symbol path is forbidden.
    with patch.object(type(model),"generate",side_effect=AssertionError("baseline seed path forbidden")), \
            patch("src.doppler.build_tdl_a_doppler_channel",side_effect=AssertionError("independent 16-symbol generation forbidden")), \
            patch.object(type(model),"generate_taps",autospec=True,side_effect=type(model).generate_taps) as taps_spy:
        draw = formal_trajectory(model,row)
    assert taps_spy.call_count == 1 and taps_spy.call_args.args[1] == 60261002
    assert draw.h_long.shape == (2,2,64,160) and draw.taps.shape == (2,2,23,160)
    assert np.shares_memory(draw.h_long,draw.h_frames)
    packed = lt.pack_link_frames(draw.h_frames)
    for f,r,q,k,t in [(0,0,0,32,4),(9,1,1,63,15),(1,0,1,40,0)]:
        assert packed[(f*2+r)*2+q,k*16+t] == draw.h_long[r,q,k,f*16+t]
    np.testing.assert_array_equal(formal_trajectory(model,row).h_long,draw.h_long)


def test_train_only_generator_spy_batch_invariance_and_validation_rejected(model):
    rows = lt.formal_manifest_rows()
    with patch("src.lmmse_train.formal_trajectory",wraps=formal_trajectory) as spy:
        r1,m1,s1 = lt.estimate_training_covariance(rows[:2],model,batch_size=1,generator=spy)
        assert [call.args[1]["trajectory_id"] for call in spy.call_args_list] == [0,1]
    r2,m2,s2 = lt.estimate_training_covariance(rows[:2],model,batch_size=2)
    np.testing.assert_allclose(r1,r2,rtol=0,atol=2e-14)
    np.testing.assert_allclose(m1,m2,rtol=0,atol=1e-15)
    assert s1["sample_count"] == s2["sample_count"] == 80
    assert s1["validation_trajectories_accessed"] == 0
    spy = patch("src.lmmse_train.formal_trajectory").start()
    try:
        with pytest.raises(ValueError,match="only unique train"):
            lt.estimate_training_covariance([rows[0],rows[2000]],model,generator=spy)
        spy.assert_not_called()
    finally:
        patch.stopall()


def test_validation_leak_negative_changes_count_and_covariance_hash():
    a = lt.SecondMomentAccumulator(2); a.add(SENTINEL)
    before,_,s0 = a.finish()
    a.add(np.array([[9+2j,-3+5j]],dtype=np.complex128))
    leaked,_,s1 = a.finish()
    assert s0["sample_count"] == 3 and s1["sample_count"] == 4
    assert lt.raw_array_sha256(before) != lt.raw_array_sha256(leaked)


def test_frozen_npz_bitwise_stable_metadata_and_manifest_tamper(tmp_path):
    # A synthetic artifact exercises serialization, not formal estimation.
    r = np.eye(1024,dtype=np.complex128)*2
    mean = np.zeros(1024,dtype=np.complex128)
    a,b = tmp_path/"a.npz",tmp_path/"b.npz"
    frozen = lt.save_frozen_covariance(a,r,mean,"synthetic-test-manifest")
    second = lt.save_frozen_covariance(b,r,mean,"synthetic-test-manifest")
    np.testing.assert_array_equal(r,frozen.R_train)
    assert frozen.R_train.tobytes() == r.tobytes()
    assert frozen.file_sha256 == second.file_sha256
    assert frozen.raw_array_sha256 == lt.raw_array_sha256(r)
    assert not frozen.R_train.flags.writeable
    with pytest.raises(ValueError): frozen.R_train[0,0] = 1
    with pytest.raises(ValueError,match="metadata/hash mismatch"):
        lt.load_frozen_covariance(a,"changed-manifest")
    with patch("numpy.linalg.inv",side_effect=AssertionError("inverse forbidden")), \
            patch("numpy.linalg.pinv",side_effect=AssertionError("pinv forbidden")), \
            patch("src.lmmse.cho_factor",wraps=lmmse.cho_factor) as factor_spy:
        layout = build_fdm_pilot_layout(2,64,16,16,4)
        core = lt.prepare_train_lmmse(frozen,layout.pilot_mask[0],0.1)
        prediction = core.estimate(np.ones(core.pilot_indices.size,dtype=np.complex128))
    assert factor_spy.call_count == 1 and np.all(np.isfinite(prediction))
    assert core.diagnostics["cholesky_success"] and not core.diagnostics["jitter_used"]
    assert lt.sha256_file(a) == frozen.file_sha256


def test_tiny_mismatch_formula_independent_full_error_covariance():
    prior = np.array([[1.6,0.2+0.4j],[0.2-0.4j,0.8]],dtype=np.complex128)
    truth = np.array([[1,0.1-0.3j],[0.1+0.3j,1.2]],dtype=np.complex128)
    est = lmmse.prepare_oracle_lmmse(prior,np.array([0]),0.25,(2,1))
    diagonal,imaginary = mismatch_error_diagonal(est,truth)
    w = prior[:,0:1]/(prior[0,0]+0.25)  # independent scalar solution
    p = truth[:,0:1]
    full = truth-w@p.conj().T-p@w.conj().T+w@np.array([[truth[0,0]+0.25]])@w.conj().T
    np.testing.assert_allclose(diagonal,np.diag(full).real,rtol=0,atol=2e-15)
    assert imaginary <= 1e-15
    wrong = truth-w@p.T-p@w.T+w@np.array([[truth[0,0]+0.25]])@w.T
    assert np.max(np.abs(np.diag(wrong).real-diagonal)) > 0.01


def test_noise_seed_distinct_for_every_heldout_condition_and_link():
    seeds = [validation_noise_seed(c,tid,r,q) for c in range(4) for tid in range(2000,2400) for r in range(2) for q in range(2)]
    assert len(set(seeds)) == 6400
    assert set(seeds).isdisjoint({60261001+tid for tid in range(2400)})
    assert CONDITIONS == ((2,0),(4,10),(8,15),(16,20)) and NMSE_TOLERANCE_DB == 0.5


@pytest.fixture(scope="module")
def saved():
    return json.loads((ROOT/"outputs/t10c_lmmse_train_results.json").read_text(encoding="utf-8"))


def test_saved_formal_counts_freeze_sources_and_no_forbidden_corrections(saved):
    assert saved["all_checks_passed"] and all(saved["checks"].values())
    assert saved["num_independent_train_trajectory_units"] == 2000
    assert saved["num_train_link_frame_observations"] == 80000
    assert saved["accumulation"]["visited_train_trajectory_ids"] == list(range(2000))
    assert saved["validation_visited_trajectory_ids"] == list(range(2000,2400))
    assert saved["accumulation"]["batch_observation_counts"] == [1000]*80
    assert saved["accumulation"]["validation_trajectories_accessed"] == 0
    order = saved["freeze_event_order"]
    assert order.index("compute_SHA256") < order.index("mark_covariance_frozen") < order.index("oracle_offline_audit") < order.index("evaluate_validation")
    assert saved["covariance_frozen_before_validation"] and saved["validation_did_not_mutate_frozen_covariance"]
    assert saved["hashes_before_validation"] == saved["hashes_after_validation"]
    for key in ("explicit_inverse_used","pseudo_inverse_used","jitter_used","diagonal_renormalization_used",
                "shrinkage_used","oracle_blending_used","eigenvalue_clipping_used","rank_truncation_used",
                "sample_mean_subtracted","full_link_BER_run","test_id_generated","test_ood_generated","ood_dev_generated"):
        assert saved[key] is False
    assert saved["memory"]["batch_matrix_MiB"] == 15.625
    assert not saved["memory"]["full_observation_matrix_materialized"]
    assert saved["train_estimators_prepared"] == len(saved["train_cholesky_diagnostics"]) == 40
    assert all(d["cholesky_success"] and not d["jitter_used"] for d in saved["train_cholesky_diagnostics"])
    assert saved["covariance_statistics"]["min_eigenvalue"] >= -1e-12
    for path,digest in saved["provenance"]["source_sha256"].items():
        from src.provenance import recorded_source_matches
        assert recorded_source_matches(ROOT,path,digest,"t10c",current_hash=lt.sha256_file)


def test_saved_artifact_and_manifest_match_frozen_hashes(saved):
    manifest_path = ROOT/"outputs/t10c_train_validation_manifest.csv"
    rows = lt.read_manifest(manifest_path)
    assert len(rows) == 2400 and lt.sha256_file(manifest_path) == saved["manifest"]["sha256"]
    frozen = lt.load_frozen_covariance(ROOT/"outputs/t10c_lmmse_train_covariance.npz",saved["manifest"]["sha256"])
    assert frozen.file_sha256 == saved["covariance_sha256"]
    assert frozen.raw_array_sha256 == saved["R_train_raw_array_sha256"]


@pytest.mark.parametrize("index",range(20))
def test_saved_mismatch_theory_train_ge_oracle_and_actual_masks(saved,index):
    row = saved["train_theoretical_conditions"][index]
    assert row["train_predicted_nmse_data_linear"] >= row["oracle_nmse_data_linear"]-1e-10
    assert row["train_cholesky_success_tx0"] and row["train_cholesky_success_tx1"]
    assert row["pilot_ratio"] == 1/(2*row["dF"])
    assert row["noise_variance"] == 1/(10**(row["snr_db"]/10))


@pytest.mark.parametrize("index",range(4))
def test_saved_heldout_mismatch_calibration_and_Pilot_LS(saved,index):
    row = saved["heldout_validation_conditions"][index]
    assert (row["dF"],row["snr_db"]) == CONDITIONS[index]
    assert row["num_independent_validation_trajectories"] == 400
    assert row["num_link_frame_observations"] == 16000
    assert row["absolute_difference_db"] <= 0.5 and row["passed"]
    assert saved["actual_pilot_ls_integration"]["passed"]
    assert saved["actual_pilot_ls_integration"]["actual_pilot_ls_observation_max_error"] <= 1e-14
