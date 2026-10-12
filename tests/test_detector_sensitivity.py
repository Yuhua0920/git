"""T11C-B: bounded paired replay, fixed bins and descriptive trajectory statistics."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from src import detector_sensitivity as ds,link
from src.lmmse_train import raw_array_sha256,sha256_file
from src.trajectories import prepare_trajectory
from scripts import run_detector_sensitivity as entry

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs():
    refs=ds.load_inputs(ROOT)
    layouts,train,diagnostics=ds.prepare_train_pool(refs)
    return refs,layouts,train,diagnostics,prepare_trajectory()


@pytest.fixture(scope="module")
def small(inputs):
    refs,layouts,train,_,model=inputs
    with patch("src.interpolation.ls_nearest_interpolate",side_effect=AssertionError("no R2")), \
         patch("src.lmmse_train_validation.estimated_receivers",side_effect=AssertionError("no R5")), \
         patch("src.metrics.channel_error_energies",side_effect=AssertionError("no NMSE")):
        return ds.run_scan(refs,refs["manifest"][:1],model,layouts,train)


def test_standalone_t11c_a_commit_and_locked_accepted_sources(inputs):
    refs=inputs[0]
    assert subprocess.check_output(["git","show","-s","--format=%s",ds.T11C_A_COMMIT],cwd=ROOT,text=True).strip()=="feat: add calibrated linear-MMSE detector sensitivity path"
    assert all(sha256_file(ROOT/p)==h for p,h in refs["accepted_sources"].items())
    assert ds.verify_unchanged(refs)==refs["before_all"]


def test_six_conditions_three_estimators_twelve_frozen_train_preparations(inputs):
    assert ds.CONDITIONS==((2,0),(2,10),(2,20),(16,0),(16,10),(16,20))
    assert ds.ESTIMATORS==("R0","R3","R4")
    assert inputs[3]["prepared_train_estimators"]==12 and inputs[3]["prepared_oracle_estimators"]==0
    assert all(r["cholesky_success"] and not r["jitter_used"] for r in inputs[3]["diagnostics"])


def test_small_exact_replay_and_frozen_ZF_counts(small,inputs):
    report,rows=small
    assert len(rows)==18 and len(report["aggregate_results"])==18
    assert report["realization_hashes"]==[inputs[0]["realizations"][2000]]
    assert report["replay_verification"]["num_frame_h_hash_matches"]==10
    assert report["execution_counters"]["frame_condition_evaluations"]==60
    assert report["execution_counters"]["paired_estimator_evaluations"]==180
    assert report["execution_counters"]["master_bits_calls"]==report["execution_counters"]["unit_noise_calls"]==10
    assert all(r["frozen_T10D_ZF_exact_match"] and r["frames_aggregated"]==10 for r in rows)


def test_historical_calibration_exact_six_integer_errors(inputs):
    r=ds.historical_calibration_regression(inputs[0])
    assert r["all_checks_passed"] and r["trajectory_id"]==0
    assert [v["actual_bit_errors"] for v in r["records"]]==[546,421,677,532,583,477]
    assert all(v["exact_match"] for v in r["records"])


def test_validation_sentinel_stays_in_its_own_namespace(small):
    r=small[0]["validation_t10d_replay_regression"]
    assert r["trajectory_id"]==2000 and r["frame_index"]==0 and r["dF"]==16 and r["snr_db"]==10
    assert r["all_checks_passed"] and r["Y_exact_accepted_link_replay"]
    assert not r["compared_to_calibration_trajectory0"]
    assert not r["original_archive_single_frame_dF16_errors_available"]
    assert all(v["ZF_integer_exact_match"] for v in r["records"])


def test_actual_detector_calls_share_H_Y_bits_and_mask(inputs):
    h=inputs[4].generate(0).h_frames[0]
    bits,master=link.generate_master_qpsk(bits_seeds=(20261001,20261002))
    layout=inputs[1][16]
    y=link.apply_frequency_mimo_channel(h,link.assemble_pilot_data_frame(master,layout))
    calls=[]; a=ds.batched_data_zf; b=link.detect_data_linear_mmse
    def zf(y,h,mask): calls.append((y,h,mask)); return a(y,h,mask)
    def mmse(y,h,mask,*args,**kwargs): calls.append((y,h,mask)); return b(y,h,mask,*args,**kwargs)
    seen=[]
    with patch.object(ds,"batched_data_zf",side_effect=zf),patch.object(link,"detect_data_linear_mmse",side_effect=mmse):
        ds.detect_pair(y,h,bits,layout.data_mask,0.1,0.5,lambda name,*args:seen.append(args))
    assert all(a is b for a,b in zip(calls[0],calls[1]))
    assert all(a is b for a,b in zip(seen[0],seen[1]))


def test_mutation_between_detectors_is_rejected(inputs):
    h=inputs[4].generate(0).h_frames[0]; layout=inputs[1][16]
    bits,master=link.generate_master_qpsk(bits_seeds=(20261001,20261002))
    y=link.apply_frequency_mimo_channel(h,link.assemble_pilot_data_frame(master,layout))
    old=ds.batched_data_zf
    def mutate(y,h,mask):
        result=old(y,h,mask); y[0,0,0]+=1; return result
    with patch.object(ds,"batched_data_zf",side_effect=mutate):
        with pytest.raises(ValueError,match="mutated"):
            ds.detect_pair(y,h,bits,layout.data_mask,0.1,0.5)


@pytest.mark.parametrize("tid,split",[(1999,"train"),(2400,"test_id"),(2000,"test_id")])
def test_formal_or_nonvalidation_generation_is_rejected(inputs,tid,split):
    refs,layouts,train,_,model=inputs
    row={**refs["manifest"][0],"trajectory_id":tid,"split":split}
    with patch.object(ds,"formal_trajectory",side_effect=AssertionError("must reject before generation")):
        with pytest.raises(ValueError,match="validation-only"):
            ds.run_scan(refs,[row],model,layouts,train)


def test_fixed_bins_boundaries_and_infinite_extreme_are_retained():
    values=np.array([1,1.99,2,3.99,4,7.99,8,15.99,16,1e10,np.inf])
    assert np.array_equal(ds.condition_bin_indices(values),[0,0,1,1,2,2,3,3,4,4,4])
    bins=ds.ConditionBins(); bins.add(values,np.ones(11,dtype=int),np.zeros(11,dtype=int))
    rows=bins.finalize()
    assert sum(r["num_RE"] for r in rows)==11 and sum(r["num_bits"] for r in rows)==44
    assert rows[-1]["num_RE"]==3 and rows[-1]["mean_condition_number"]=="inf"
    assert rows[-1]["median_condition_number"]==1e10


@pytest.mark.parametrize("values",[[0.99],[np.nan],[-np.inf],[[1,2]]])
def test_invalid_condition_numbers_rejected(values):
    with pytest.raises(ValueError): ds.condition_bin_indices(values)


def test_every_RE_and_error_exactly_one_bin(small):
    report=small[0]; bins=report["condition_number_bins"]
    a=next(r for r in report["aggregate_results"] if (r["dF"],r["snr_db"],r["estimator"])==(16,10,"R0"))
    assert sum(r["num_RE"] for r in bins)==992*10
    assert sum(r["num_bits"] for r in bins)==a["total_bits"]
    assert sum(r["zf_bit_errors"] for r in bins)==a["zf_bit_errors"]
    assert sum(r["mmse_bit_errors"] for r in bins)==a["mmse_bit_errors"]


def test_paired_pooled_aggregation_with_negative_detector_changes(small):
    rows=[]
    for tid,zerr,merr in ((2000,20,10),(2001,10,30)):
        for old in small[1]:
            r=copy.deepcopy(old); bits=r["total_bits"]
            r.update(trajectory_id=tid,zf_bit_errors=zerr,mmse_bit_errors=merr,zf_ber=zerr/bits,mmse_ber=merr/bits)
            rows.append(r)
    aggregate,paired=ds.aggregate_trajectories(rows)
    assert len(aggregate)==len(paired)==18
    for a,p in zip(aggregate,paired):
        assert a["zf_bit_errors"]==30 and a["mmse_bit_errors"]==40
        assert a["absolute_ber_reduction"]<0 and p["fraction_MMSE_errors_less"]==0.5
        assert np.isclose(p["mean"],a["absolute_ber_reduction"])
        assert not p["p_value_computed"] and not p["bootstrap_CI_computed"]


@pytest.mark.parametrize("damage",["missing_row","duplicate_row","bad_frames","unpaired","bad_ber"])
def test_incomplete_or_unpaired_rows_rejected(small,damage):
    rows=copy.deepcopy(small[1])
    if damage=="missing_row": rows.pop()
    if damage=="duplicate_row": rows[-1]=rows[0]
    if damage=="bad_frames": rows[0]["frames_aggregated"]=9
    if damage=="unpaired": rows[0]["same_H_est"]=False
    if damage=="bad_ber": rows[0]["zf_ber"]+=0.1
    with pytest.raises(ValueError): ds.aggregate_trajectories(rows)


def test_two_figures_rebuild_from_only_two_CSVs(tmp_path,small):
    entry.save_csv(tmp_path/entry.AGGREGATE,small[0]["aggregate_results"])
    entry.save_csv(tmp_path/entry.BINS,small[0]["condition_number_bins"])
    with patch.object(entry,"load_inputs",side_effect=AssertionError("no covariance/artifact input")), \
         patch.object(entry,"run_scan",side_effect=AssertionError("no numerical replay")), \
         patch.object(entry,"prepare_trajectory",side_effect=AssertionError("no channel")):
        entry.main(["--plot-only","--output-dir",str(tmp_path)])
    assert len(list(tmp_path.glob("*.png")))==len(list(tmp_path.glob("*.pdf")))==len(list(tmp_path.glob("*.svg")))==2


def test_saved_full_results_7200_rows_18_aggregates_and_400_replays(inputs):
    report=json.loads((ROOT/"outputs"/entry.RESULTS).read_text(encoding="utf8"))
    assert report["all_checks_passed"] and not report["figures_pending"]
    assert report["num_independent_trajectories"]==400 and report["num_correlated_frames"]==4000
    assert report["visited_validation_trajectory_ids"]==list(range(2000,2400))
    assert report["realization_hashes"]==inputs[0]["old"]["realization_hashes"]
    replay=report["replay_verification"]
    assert replay["num_h_hash_matches"]==400
    assert replay["num_frame_h_hash_matches"]==replay["num_bits_hash_matches"]==replay["num_noise_hash_matches"]==4000
    assert replay["exact_match_fraction"]==1 and replay["all_checks_passed"]
    rows=entry.plot_results.__globals__["read_rows"](ROOT/"outputs"/entry.TRAJECTORY)
    assert len(rows)==7200 and len(report["aggregate_results"])==18
    for r in rows:
        old=inputs[0]["trajectory_index"][int(r["trajectory_id"]),int(r["dF"]),int(r["snr_db"])]
        assert int(r["zf_bit_errors"])==int(old[f"{ds.OLD_METHOD[r['estimator']]}_bit_errors"])
        assert all(r[k]=="True" for k in ("same_H_est","same_Y","same_bits","same_data_mask"))
    for path,digest in report["output_csv_sha256"].items(): assert sha256_file(ROOT/"outputs"/path)==digest
    assert report["frozen_hash_before_after"]["before"]==report["frozen_hash_before_after"]["after"]==ds.verify_unchanged(inputs[0])
    assert report["main_detector"]=="ZF" and report["formal_test_sealed"] and not report["formal_test_used"]
    assert not any(report[k] for k in ("main_detector_changed","OOD_generated","covariance_reestimated"))
    assert report["num_filtered_trajectories"]==0 and report["numerical_failures"]==[]
    assert report["historical_t11c_a_calibration_regression"]["all_checks_passed"]
    assert report["validation_t10d_replay_regression"]["all_checks_passed"]
    assert report["csv_only_rebuild"]["all_checks_passed"] and len(report["figures"])==2
    for p,h in report["provenance"]["source_sha256"].items(): assert sha256_file(ROOT/p)==h
