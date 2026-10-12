"""T11B gates: actual replay, incremental-only computation, exact frozen merge."""

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src import nearest_validation as nv
from src.lmmse_train import raw_array_sha256
from src.pilots import build_fdm_pilot_layout
from src.stochastic_scan import frame_duration_seconds
from src.trajectories import prepare_trajectory

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def references():
    return nv.load_references(ROOT/"outputs")


@pytest.fixture(scope="module")
def model():
    return prepare_trajectory()


@pytest.fixture(scope="module")
def layouts():
    return {df:build_fdm_pilot_layout(2,64,16,df,4) for df in nv.DF_OPTIONS}


def test_t11a_standalone_commit_and_accepted_numerical_sources():
    assert subprocess.check_output(["git","show","-s","--format=%s",nv.T11A_COMMIT],cwd=ROOT,text=True).strip()=="feat: add calibrated LS-nearest baseline"
    assert subprocess.check_output(["git","diff",f"{nv.T11A_COMMIT}^",nv.T11A_COMMIT,"--name-only"],cwd=ROOT,text=True).splitlines()==[
        "README.md","docs/baseline_registry.md","docs/stage_status.md","outputs/t11a_nearest_interpolation_results.json",
        "scripts/validate_nearest_interpolation.py","src/interpolation.py","tests/test_nearest_interpolation.py"]
    assert all(nv.verify_accepted_sources(ROOT).values())


def test_frozen_references_and_exact_validation_manifest(references):
    assert nv.verify_frozen(ROOT/"outputs")==nv.FROZEN_HASHES==references["before"]
    assert [r["trajectory_id"] for r in references["manifest"]]==list(range(2000,2400))
    assert all(r["split"]=="validation_id" and r["frames_per_trajectory"]==10 for r in references["manifest"])
    assert len(references["trajectory_index"])==8000


def test_changed_frozen_reference_is_rejected():
    with patch("src.nearest_validation.sha256_file",return_value="changed"):
        with pytest.raises(ValueError,match="frozen"):
            nv.verify_frozen(ROOT/"outputs")


@pytest.mark.parametrize("tid,frame,tx", [(2000,0,0),(2000,0,1),(2037,6,0),(2399,9,1)])
def test_seed_policy_exactly_matches_t10d(tid,frame,tx):
    assert nv.bits_seed(tid,frame,tx)==80261001+100*tid+2*frame+tx
    assert nv.noise_seed(tid,frame)==90261001+100*tid+frame


@pytest.mark.parametrize("tid,split", [(0,"train"),(1999,"train"),(2400,"test"),(2000,"test")])
def test_invalid_manifest_rejected_before_generation(references,model,layouts,tid,split):
    row = {**references["manifest"][0],"trajectory_id":tid,"split":split,"channel_seed":60261001+tid}
    with patch("src.nearest_validation.formal_trajectory") as generate:
        with pytest.raises(ValueError,match="validation-only"):
            nv.run_replay([row],model,layouts,references,generator=generate)
        generate.assert_not_called()


@pytest.fixture(scope="module")
def small_replay(references,model,layouts):
    with patch("src.lmmse_train_validation.run_validation",side_effect=AssertionError("old scan forbidden")), \
         patch("src.lmmse_train_validation.prepare_pools",side_effect=AssertionError("no LMMSE preparation")), \
         patch("src.lmmse_train_validation.estimated_receivers",side_effect=AssertionError("no old receivers")), \
         patch("src.link.estimate_ls_linear_channel",side_effect=AssertionError("no R3 scan")), \
         patch("numpy.linalg.inv",side_effect=AssertionError("no inverse")), \
         patch("numpy.linalg.pinv",side_effect=AssertionError("no pinv")), \
         patch("src.nearest_validation.pilot_ls_estimate",wraps=nv.pilot_ls_estimate) as pilot, \
         patch("src.nearest_validation.ls_nearest_interpolate",wraps=nv.ls_nearest_interpolate) as nearest, \
         patch("src.nearest_validation.batched_data_zf",wraps=nv.batched_data_zf) as zf:
        report,rows = nv.run_replay(references["manifest"][:1],model,layouts,references)
    return report,rows,pilot.call_args_list,nearest.call_args_list,zf.call_args_list


def test_small_actual_noisy_pilot_path_and_no_old_performance(small_replay):
    report,rows,pilot,nearest,zf = small_replay
    assert len(pilot)==len(nearest)==len(zf)==200 and len(rows)==20
    for p,n,z in zip(pilot,nearest,zf):
        assert z.args[0] is p.args[0]
        observed = n.args[0]; mask = n.args[1]
        assert np.isnan(observed[:,~mask]).all()
        for tx in range(2):
            assert np.array_equal(observed[:,tx,mask[tx]],p.args[0][:,mask[tx]]/p.args[1][tx,mask[tx]])
    counters = report["execution_counters"]
    assert counters["trajectory_generator_calls"]==1
    assert counters["master_bits_calls"]==counters["unit_noise_calls"]==counters["shared_frame_mutation_checks"]==10
    assert counters["old_method_performance_evaluations"]==counters["LMMSE_preparations"]==0
    assert report["num_filtered_trajectories"]==0 and report["numerical_failures"]==[]


def test_small_exact_hash_replay_and_saved_sentinel(small_replay,references):
    report,*_ = small_replay
    assert report["realization_hashes"]==[references["realizations"][2000]]
    replay = report["replay_verification"]
    assert replay["num_h_hash_matches"]==1 and replay["num_frame_h_hash_matches"]==10
    assert replay["num_bits_hash_matches"]==replay["num_noise_hash_matches"]==10
    assert replay["all_stored_fields_checked"] and replay["exact_match_fraction"]==1
    assert [(r["k"],r["t"]) for r in report["manual_diagnostic"]]==[(1,2),(1,15)]
    assert all(r["saved_H_X_Y_exact_match"] for r in report["manual_diagnostic"])


def test_continuous_ten_frame_views_and_hashes(references,model):
    trajectory = nv.formal_trajectory(model,references["manifest"][0])
    assert np.shares_memory(trajectory.h_frames,trajectory.h_long)
    assert trajectory.h_long.shape==(2,2,64,160) and trajectory.h_frames.shape==(10,2,2,64,16)
    assert raw_array_sha256(trajectory.h_long)==references["realizations"][2000]["H_long_sha256"]
    for frame,h in enumerate(trajectory.h_frames):
        assert np.array_equal(h,trajectory.h_long[...,16*frame:16*(frame+1)])
        assert raw_array_sha256(h)==references["realizations"][2000]["frame_realizations"][frame]["H_sha256"]


@pytest.mark.parametrize("field", ["H_long_sha256","H_sha256","bits_sha256","unit_noise_sha256"])
def test_replay_mismatch_stops_before_estimation(references,model,layouts,field):
    changed = {**references,"realizations":copy.deepcopy(references["realizations"])}
    if field=="H_long_sha256":
        changed["realizations"][2000][field] = "bad"
    else:
        changed["realizations"][2000]["frame_realizations"][0][field] = "bad"
    with patch("src.nearest_validation.evaluate_frame",side_effect=AssertionError("mismatch must stop before estimation")) as evaluate:
        with pytest.raises(ValueError,match="replay mismatch"):
            nv.run_replay(references["manifest"][:1],model,layouts,changed)
        evaluate.assert_not_called()


def test_trajectory_accumulator_requires_ten_frames_and_order(references):
    a = nv.TrajectoryAccumulator(references["manifest"][0],2,0,0.25)
    with pytest.raises(ValueError,match="ten continuous"):
        a.finalize({})
    with pytest.raises(ValueError,match="once in order"):
        a.add(1,{},1)
    for frame in range(10):
        a.add(frame,{"nearest_nmse_error_energy":frame+1,"nmse_truth_energy":10*(frame+1),
            "nearest_bit_errors":1,"total_data_bits":100},frame+1)
    old = {k:str(a.row[k]) for k in ("channel_seed","frames_aggregated","total_data_bits","nmse_truth_energy",
        "condition_number_max_over_10_frames","noise_variance","pilot_ratio")}
    row = a.finalize(old)
    assert row["nearest_nmse_error_energy"]==55 and row["total_data_bits"]==1000
    assert row["condition_number_max_over_10_frames"]==10
    with pytest.raises(ValueError):
        a.add(10,{},1)


def test_merge_rejects_partial_scan(small_replay,references):
    with pytest.raises(ValueError,match="full-validation"):
        nv.merge_baselines(small_replay[0]["r2_aggregate_results"],references)


def test_pooled_ratios_are_energy_and_integer_count_weighted(small_replay):
    rows = []
    for template in small_replay[1]:
        for tid,error,truth,bits,errors in ((2000,1.0,10.0,1000,1),(2001,2.0,100.0,3000,9)):
            rows.append({**template,"trajectory_id":tid,"nearest_nmse_error_energy":error,
                "nmse_truth_energy":truth,"total_data_bits":bits,"nearest_bit_errors":errors})
    for a in nv.aggregate_r2(rows):
        assert a["nearest_nmse_linear"]==3/110
        assert a["nearest_nmse_linear"]!=(1/10+2/100)/2
        assert a["nearest_ber"]==10/4000
        assert a["nearest_proxy_bps"]==200*(1-10/4000)/frame_duration_seconds()


def test_zf_failure_is_retained_and_raised_not_filtered(references,model,layouts):
    with patch("src.nearest_validation.batched_data_zf",side_effect=ValueError("singular channel at k=1,t=2")):
        with pytest.raises(nv.NoiselessLinkFailure) as failure:
            nv.run_replay(references["manifest"][:1],model,layouts,references)
    context = failure.value.record
    assert context["trajectory_id"]==2000 and context["channel_seed"]==60263001
    assert context["frame_index"]==0 and context["snr_db"]==0 and context["dF"]==2
    assert context["path"]=="R2 LS-nearest"


@pytest.fixture(scope="module")
def saved():
    return json.loads((ROOT/"outputs/t11b_nearest_validation_results.json").read_text(encoding="utf-8"))


def test_saved_full_exact_hash_replay_and_semantics(saved,references):
    assert saved["all_checks_passed"] and all(saved["checks"].values())
    assert saved["provenance"]["T11A_committed_cleanly_at_task_start"]
    assert saved["realization_hashes"]==references["old"]["realization_hashes"]
    assert saved["visited_validation_trajectory_ids"]==list(range(2000,2400))
    replay = saved["replay_verification"]
    assert replay["num_h_hash_matches"]==saved["num_h_hash_matches"]==400
    assert replay["num_frame_h_hash_matches"]==replay["num_complete_frame_provenance_matches"]==4000
    assert replay["num_bits_hash_matches"]==saved["num_bits_hash_matches"]==4000
    assert replay["num_noise_hash_matches"]==saved["num_noise_hash_matches"]==4000
    assert replay["all_stored_fields_checked"] and replay["exact_match_fraction"]==1
    assert saved["seed_policy"]==references["old"]["seed_policy"]
    assert saved["frozen_hash_before_after"]["before"]==saved["frozen_hash_before_after"]["after"]==nv.FROZEN_HASHES
    assert saved["formal_test_sealed"] and not saved["formal_test_used"] and not saved["OOD_generated"]
    assert saved["num_filtered_trajectories"]==0 and saved["numerical_failures"]==[]
    assert not saved["method_ordering_required"] and not saved["covariance_updated"]
    assert saved["execution_counters"]["R2_frame_condition_evaluations"]==80000
    assert saved["execution_counters"]["old_method_performance_evaluations"]==saved["execution_counters"]["LMMSE_preparations"]==0
    for name,digest in saved["output_csv_sha256"].items():
        assert nv.sha256_file(ROOT/"outputs"/name)==digest
    for path,digest in saved["provenance"]["source_sha256"].items():
        from src.provenance import recorded_source_matches
        assert recorded_source_matches(ROOT,path,digest,"t11b",current_hash=nv.sha256_file)


def _typed_trajectory_rows():
    rows = nv.read_csv(ROOT/"outputs/t11b_nearest_validation_trajectory_metrics.csv")
    integers = {"trajectory_id","channel_seed","snr_db","dF","frames_aggregated","total_data_bits","nearest_bit_errors"}
    return [{k:(int(v) if k in integers else v=="True" if k.startswith("replay_") else float(v)) for k,v in r.items()} for r in rows]


def test_saved_8000_trajectory_units_and_full_aggregation(saved,references):
    rows = _typed_trajectory_rows()
    assert len(rows)==8000
    assert nv.aggregate_r2(rows)==saved["r2_aggregate_results"]
    for r in rows:
        old = references["trajectory_index"][r["trajectory_id"],r["dF"],r["snr_db"]]
        assert r["frames_aggregated"]==10
        assert r["nmse_truth_energy"]==float(old["nmse_truth_energy"])
        assert r["total_data_bits"]==int(old["total_data_bits"])
        assert r["condition_number_max_over_10_frames"]==float(old["condition_number_max_over_10_frames"])
    for a in saved["r2_aggregate_results"]:
        assert a["nearest_nmse_linear"]==a["nearest_nmse_error_energy"]/a["nmse_truth_energy"]
        assert a["nearest_ber"]==a["nearest_bit_errors"]/a["total_data_bits"]
        assert np.isclose(a["nearest_proxy_bps"],(a["total_data_bits"]-a["nearest_bit_errors"])/(4000*frame_duration_seconds()),rtol=1e-12)


def test_saved_merge_preserves_every_old_value_verbatim(saved,references):
    merged = nv.read_csv(ROOT/"outputs/t11b_r0_r5_validation_baselines.csv")
    assert len(merged)==20
    expected = nv.merge_baselines(saved["r2_aggregate_results"],references)
    for actual,row,old in zip(merged,expected,references["scan"]):
        assert actual=={k:str(v) for k,v in row.items()}
        for target,source in nv.OLD_TO_MERGED.items():
            assert actual[target]==old[source]
        assert actual["r1_data_ber"]=="N/A"
    assert saved["r1_semantics"]["r1_data_ber_supported"] is False
    assert not saved["r1_semantics"]["new_R1_experiment_performed"]


def test_saved_three_figures_csv_only_regeneration(tmp_path,saved):
    from scripts.run_nearest_validation import MERGED_CSV, main
    from scripts.plot_nearest_validation import FIGURE_NAMES
    source = ROOT/"outputs"/MERGED_CSV
    shutil.copyfile(source,tmp_path/MERGED_CSV)
    for name in FIGURE_NAMES:
        copy = tmp_path/f"{name}.png"
        shutil.copyfile(ROOT/"outputs"/copy.name,copy)
        copy.unlink()
    assert [p.name for p in tmp_path.iterdir()]==[MERGED_CSV]
    with patch("scripts.run_nearest_validation.load_references",side_effect=AssertionError("CSV-only")), \
         patch("scripts.run_nearest_validation.prepare_trajectory",side_effect=AssertionError("no generation")), \
         patch("scripts.run_nearest_validation.run_replay",side_effect=AssertionError("no receivers")):
        main(["--plot-only","--output-dir",str(tmp_path)])
    for name in FIGURE_NAMES:
        assert nv.sha256_file(tmp_path/f"{name}.png")==nv.sha256_file(ROOT/"outputs"/f"{name}.png")
    assert all(saved["CSV_only_redraw"]["PNG_hash_matches"].values())
