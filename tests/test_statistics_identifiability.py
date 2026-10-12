"""T13C receiver math, frozen replay and result reaggregation gates."""
import inspect
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src import statistics_selection as rx,statistics_identifiability as scan
from src.b0_development import replay_link
from src.lmmse_train import raw_array_sha256,sha256_file
from src.pilots import count_layout_resources,map_pilot_symbols
from src.trajectories import prepare_trajectory

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"
torch.set_num_threads(4)


def test_T13B_committed_and_history_accepted():
    assert subprocess.check_output(["git","show","-s","--format=%s",scan.T13B_COMMIT],cwd=ROOT,text=True).strip()=="feat: add heuristic robust LMMSE literature baseline gate"
    blob=subprocess.check_output(["git","show",scan.T13B_COMMIT+":outputs/t13b_robust_lmmse_results.json"],cwd=ROOT)
    assert blob==(OUT/"t13b_robust_lmmse_results.json").read_bytes().replace(b"\r\n",b"\n")
    r=json.loads(blob)
    assert r["all_checks_passed"] and r["formal_test_sealed"] and not r["formal_test_used"]
    context=scan.aliasing_context()
    assert context["delay_ambiguity_period_s"]==1/480000
    assert context["HR_uncertainty_support_s"]==pytest.approx(2.89758e-6)
    assert context["interpretation"]=="the tested wide-envelope HR prior is insufficient"


@pytest.fixture(scope="module")
def bank():
    return rx.build_candidate_bank()


@pytest.fixture(scope="module")
def pools(bank):
    with patch.object(rx,"cho_factor",wraps=rx.cho_factor) as a,patch.object(rx.lmmse,"cho_factor",wraps=rx.lmmse.cho_factor) as b:
        result=rx.prepare_pools(bank[0])
    assert a.call_count==36 and b.call_count==18
    return result


def test_exact_three_candidates_frozen_R5_theory_and_resources(bank):
    assert rx.CANDIDATES==(("ID",30.,100.),("speed120",120.,100.),("DS300",30.,300.))
    assert len(bank[0])==3
    old=json.loads((OUT/"t13a_ood_mismatch_results.json").read_text(encoding="utf8"))
    for c in bank[0]:
        assert c["full_frame"].dtype==np.complex128 and not c["full_frame"].flags.writeable
        np.testing.assert_array_equal(c["full_frame"],np.kron(c["rf"],c["rt"][1]))
        for factor in (c["rf"],*c["rt"].values()):
            assert np.max(np.abs(factor-factor.conj().T))<1e-12
            assert np.linalg.eigvalsh(factor).min()>-1e-12
        if c["name"]!="ID":
            assert raw_array_sha256(c["full_frame"])==old["r5_ood_covariance_provenance"][c["name"]]["full_grid_raw_sha256"]
    assert count_layout_resources(rx.LAYOUT)["union_ratio"]==1/32
    np.testing.assert_array_equal(rx.PILOT_VALUES,map_pilot_symbols(rx.LAYOUT,scan.link.PILOT_SYMBOLS))


@pytest.mark.parametrize("window",[1,5])
@pytest.mark.parametrize("tx",[0,1])
def test_global_C_order_packing_and_covariance_cross_frame(bank,window,tx):
    coords=rx.pilot_coordinates(window,tx)
    assert len(coords)==16*window
    frames=np.empty((window,2,2,16),complex)
    for frame in range(window):
        for q in range(2):
            xy=rx.pilot_coordinates(1,q)
            frames[frame,:,q]=xy[:,0]*1000+xy[:,1]+16*frame
    packed=rx.pack_joint_pilots(frames)
    np.testing.assert_array_equal(packed[0,tx],coords[:,0]*1000+coords[:,1])
    for c in bank[0]:
        k=np.unique(coords[:,0]); t=np.unique(coords[:,1])
        direct=np.kron(c["rf"][np.ix_(k,k)],c["rt"][window][np.ix_(t,t)])
        np.testing.assert_array_equal(c["pilot_prior"][window,tx],direct)
        if window==5:
            # Same carrier, anchor0 in frame0 and frame1: retained Jakes lag16.
            assert abs(direct[0,4])>1e-12
            assert direct[0,4]==pytest.approx(c["rt"][5][0,16])


def test_Cholesky_scores_and_logdet_against_independent_direct_solve(bank,pools):
    records=scan.validate_likelihood(bank[0],pools[0])
    assert len(records)==6 and all(r["all_checks_passed"] for r in records)
    assert max(r["score_direct_max_absolute_error"] for r in records)<1e-9


def test_four_independent_links_sum_and_stable_tie(pools):
    z=np.arange(64).reshape(1,2,2,16)/64+1j/7
    scores=rx.select_candidate(z,pools[0][1,10])
    np.testing.assert_array_equal(scores["scores"],scores["scores_by_link"].sum((1,2)))
    same=(pools[0][1,10][0],)*3
    tied=rx.select_candidate(z,same)
    assert tied["candidate_index"]==0 and tied["tie"] and tied["margin"]==0


@pytest.mark.parametrize("bad",[0,np.zeros((2,2,16)),np.zeros((2,2,2,16)),np.full((5,2,2,16),np.nan)])
def test_malformed_packed_observations_rejected(bad):
    with pytest.raises(ValueError):
        rx.pack_joint_pilots(bad)


@pytest.mark.parametrize("noise",[0,-1,np.nan])
def test_invalid_noise_rejected(noise):
    with pytest.raises(ValueError):
        rx.prepare_likelihood(np.eye(16),noise)


def test_receiver_input_isolation_and_no_inverse_or_data_grid(pools):
    assert list(inspect.signature(rx.select_candidate).parameters)==["pilot_frames","models"]
    assert list(inspect.signature(rx.estimate_pilot_frames).parameters)==["pilot_frames","candidate_index","estimators"]
    source=inspect.getsource(rx)
    assert not any(s in source for s in ("np.linalg.inv(","np.linalg.pinv(","data_mask","H_true","bits_seed","trajectory_id"))
    rng=np.random.default_rng(972); y=rng.normal(size=(2,64,16))+1j*rng.normal(size=(2,64,16))
    def observations(y):
        return rx.pilot_ls_from_received_pilots(np.stack([y[:,rx.LAYOUT.pilot_mask[q]] for q in range(2)],axis=1))
    a=observations(y); y[:,~rx.LAYOUT.pilot_union]=1e99+1e99j
    np.testing.assert_array_equal(a,observations(y))
    s=rx.select_candidate(a[None],pools[0][1,10]); other=rx.select_candidate(observations(y)[None],pools[0][1,10])
    np.testing.assert_array_equal(s["scores"],other["scores"])


@pytest.fixture(scope="module")
def refs():
    return scan.references(OUT)


@pytest.fixture(scope="module")
def r4():
    return scan.prepare_R4(OUT)[0]


@pytest.mark.parametrize("condition,tid",[("ID",2000),("speed120",0),("DS300",0)])
def test_small_exact_replay_OracleBank_R5_and_no_B0(condition,tid,pools,r4,refs):
    model=prepare_trajectory(frames_per_trajectory=10) if condition=="ID" else scan.ood.prepare_channel(condition)
    timing={k:0. for k in scan.TIMING_KEYS}
    with patch.object(scan.ood,"load_frozen_estimators",side_effect=AssertionError("B0 load")),patch.object(scan.ood,"evaluate_trajectory",side_effect=AssertionError("old scan")):
        selected,performance,r=scan.evaluate_trajectory(condition,tid,model,pools[0],pools[1],r4,refs,replay_link(ROOT),timing)
    assert len(selected)==6 and len(performance)==3 and len(r["frame_realizations"])==5
    assert r["historical_X_Y_checks"]==15
    assert all(a["total_bits"]==19840 and a["OracleBank_R5_max_channel_difference"]<1e-12 for a in performance)
    assert all(a["R7_OracleBank_bit_errors"]==a["R5_bit_errors"] for a in performance)
    assert all(np.isfinite(a["margin"]) and a["margin"]>=0 for a in selected)


@pytest.mark.parametrize("macro,recalls,key",[(.9,[.85,.95,.9],"pilot_statistics_readily_identifiable"),(.9,[.84,.96,.9],"partial_identifiability"),(.699,[.699]*3,"sparse_pilot_statistics_identifiability_weak"),(.7,[.7]*3,"partial_identifiability")])
def test_identification_thresholds(macro,recalls,key):
    flags=scan.identification_flags(macro,recalls)
    assert sum(flags.values())==1 and flags[key]


@pytest.mark.parametrize("macro,closures,viable,bottleneck",[(.9,[.7,.7],True,False),(.7,[.5,.5],False,False),(.699,[1.,1.],False,True),(.95,[.49,1.],False,True),(.8,[-.1,1.1],False,True)])
def test_total_gate_without_clipping(macro,closures,viable,bottleneck):
    performance=[{"condition":c,"snr_db":20,"gap_closure":v} for c,v in zip(scan.CONDITIONS[1:],closures)]
    confusion=[{"window_frames":5,"snr_db":s,"macro_accuracy":macro} for s in scan.SNRS]
    r=scan.total_gate(performance,confusion)
    assert r["simple_covariance_selection_viable"]==viable and r["sparse_pilot_adaptation_bottleneck"]==bottleneck


def test_balanced_confusion_counts_and_missing_sample_rejected():
    rows=[{"window_frames":w,"snr_db":s,"true_condition":c,"trajectory_id":tid,"selected_condition":c,"correct":True}
        for w in rx.WINDOWS for s in rx.SNRS for c in scan.CONDITIONS for tid in (range(2000,2200) if c=="ID" else range(200))]
    assert len(rows)==3600
    results=scan.confusion_results(rows)
    assert len(results)==6 and all(r["confusion_matrix"]==np.diag([200]*3).tolist() for r in results)
    with pytest.raises(ValueError,match="600 balanced"):
        scan.confusion_results(rows[:-1])


@pytest.fixture(scope="module")
def report():
    path=OUT/f"{scan.PREFIX}_results.json"
    if not path.exists():
        pytest.skip("full T13C run pending; no result claim")
    return json.loads(path.read_text(encoding="utf8"))


def test_final_counts_and_sealed_scope(report):
    assert report["selector_rows"]==3600 and report["performance_trajectory_rows"]==1800
    assert len(report["realization_hashes"])==600 and len(report["bank_lmmse_results"])==9
    assert report["formal_test_sealed"] and not report["formal_test_used"]
    assert not report["proposed_method"] and not report["new_model_trained"] and not report["old_model_loaded"]
    assert not report["MMSE_detector_used"] and not report["numerical_failures"] and report["num_filtered_trajectories"]==0
    assert sum(r["historical_X_Y_checks"] for r in report["realization_hashes"])==9000


def test_artifact_provenance_unchanged_and_csv_regeneration(report):
    from scripts.run_statistics_identifiability import accepted_files,read_selection,read_performance
    assert report["provenance"]["accepted_before"]==report["provenance"]["accepted_after"]==accepted_files()
    assert all(sha256_file(ROOT/p)==h for p,h in report["provenance"]["source_sha256"].items())
    assert scan.confusion_results(read_selection(OUT/f"{scan.PREFIX}.csv"))==report["confusion_matrices"]
    expected=scan.aggregate_performance(read_performance(OUT/f"{scan.PERFORMANCE_PREFIX}_trajectory_metrics.csv"))
    assert expected==report["bank_lmmse_results"]
    assert all(report[k]==v for k,v in scan.total_gate(expected,report["confusion_matrices"],report["decision_scope"]).items())
    for name,h in report["provenance"]["CSV_sha256"].items():
        assert sha256_file(OUT/name)==h


def test_CSV_only_figure_preflight_before_main_scan(tmp_path):
    from scripts.run_statistics_identifiability import render_and_audit
    from scripts.run_lmmse_train_validation import save_csv
    selected=[{"window_frames":w,"snr_db":s,"true_condition":c,"trajectory_id":tid,"selected_condition":c,"correct":True}
        for w in rx.WINDOWS for s in rx.SNRS for c in scan.CONDITIONS for tid in (range(2000,2200) if c=="ID" else range(200))]
    performance=[{"condition":c,"snr_db":s,"R4_nmse_db":-s,"R7_BankML_nmse_db":-s-1,"R5_nmse_db":-s-2} for c in scan.CONDITIONS for s in scan.SNRS]
    save_csv(tmp_path/f"{scan.PREFIX}.csv",selected); save_csv(tmp_path/f"{scan.PERFORMANCE_PREFIX}.csv",performance)
    with patch.object(scan,"evaluate_trajectory",side_effect=AssertionError("numerical scan")):
        records=render_and_audit(tmp_path,tmp_path)
    assert len(records)==2 and all(r["alignment_audit"]["verdict"]=="PASS" for r in records)
    assert all(r["collision_audit"]["verdict"]=="PASS" for r in records)
