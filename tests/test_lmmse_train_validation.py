import inspect
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src import lmmse_train_validation as v
from src.link import detect_data_zf,generate_master_qpsk,assemble_pilot_data_frame,apply_frequency_mimo_channel
from src.numerics import complex_gaussian_noise
from src.pilots import build_fdm_pilot_layout
from src.trajectories import prepare_trajectory
from src.stochastic_scan import frame_duration_seconds
from src.lmmse_train import formal_manifest_rows,formal_trajectory,sha256_file
from scripts.run_lmmse_train_validation import main,source_checks
from scripts.plot_lmmse_train_validation import FIGURE_NAMES

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup():
    frozen,rows,theory,before = v.load_frozen_inputs(ROOT/"outputs")
    layouts = {df:build_fdm_pilot_layout(2,64,16,df,4) for df in v.DF_OPTIONS}
    model = prepare_trajectory()
    train,oracle,counters = v.prepare_pools(frozen,layouts,theory)
    return frozen,rows,theory,before,layouts,model,train,oracle,counters


def test_accepted_T10C_commit_sources_and_frozen_hashes(setup):
    frozen,rows,_,before,*_ = setup
    assert all(source_checks().values())
    assert len(rows)==400 and [r["trajectory_id"] for r in rows]==list(range(2000,2400))
    assert frozen.R_train.flags.writeable is False
    assert v.frozen_hashes(ROOT/"outputs",frozen)==before


def test_every_Train_core_uses_frozen_prior_and_blind_receiver_boundary(setup):
    frozen,_,_,_,_,_,train,oracle,_ = setup
    for key,obj in train.items():
        np.testing.assert_array_equal(obj.r_hp,frozen.R_train[:,obj.pilot_indices])
        np.testing.assert_array_equal(obj.r_pp,frozen.R_train[np.ix_(obj.pilot_indices,obj.pilot_indices)])
        np.testing.assert_array_equal(obj.prior_diagonal,np.diag(frozen.R_train).real)
        assert not np.array_equal(obj.r_hp,oracle[key].r_hp)
    assert set(inspect.signature(v.estimated_receivers).parameters)=={"y","layout","train_pair","oracle_pair","timing"}


def test_tampered_artifact_file_is_rejected_without_reestimation(tmp_path):
    for name in ("t10c_lmmse_train_results.json","t10c_train_validation_manifest.csv",
                 "t10c_lmmse_train_covariance.npz","t10c_lmmse_train_theory.csv"):
        shutil.copyfile(ROOT/"outputs"/name,tmp_path/name)
    with (tmp_path/"t10c_lmmse_train_covariance.npz").open("ab") as handle:
        handle.write(b"tampered-file")
    with pytest.raises(ValueError,match="hashes changed"):
        v.load_frozen_inputs(tmp_path)


def test_bits_and_full_link_noise_seed_namespace_unique_and_deterministic():
    bits = {v.bits_seed(tid,f,q) for tid in range(2000,2400) for f in range(10) for q in range(2)}
    noise = {v.noise_seed(tid,f) for tid in range(2000,2400) for f in range(10)}
    channels = {60261001+tid for tid in range(2400)}
    assert len(bits)==8000 and len(noise)==4000
    assert bits.isdisjoint(noise|channels) and noise.isdisjoint(channels)
    assert v.bits_seed(2000,0,0)==80461001 and v.noise_seed(2000,0)==90461001
    with pytest.raises(ValueError): v.bits_seed(0,0,0)
    with pytest.raises(ValueError): v.noise_seed(2400,0)
    with pytest.raises(ValueError): v.noise_seed(2000,10)


@pytest.mark.parametrize("df",(2,4,8,16))
def test_batched_ZF_matches_frozen_per_RE_solver_and_decisions(setup,df):
    _,rows,_,_,layouts,model,*_ = setup
    h = formal_trajectory(model,rows[0]).h_frames[0]
    _,symbols = generate_master_qpsk(bits_seeds=(80461001,80461002))
    layout = layouts[df]
    y = apply_frequency_mimo_channel(h,assemble_pilot_data_frame(symbols,layout))
    y += complex_gaussian_noise(y.shape,0.1,np.random.default_rng(90461001))
    with patch("numpy.linalg.inv",side_effect=AssertionError("inverse forbidden")), \
            patch("numpy.linalg.pinv",side_effect=AssertionError("pinv forbidden")):
        expected = detect_data_zf(y,h,layout.data_mask)
        actual = v.batched_data_zf(y,h,layout.data_mask)
    np.testing.assert_allclose(actual[:,layout.data_mask[0]],expected[:,layout.data_mask[0]],rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(actual[:,layout.data_mask[0]].real<0,expected[:,layout.data_mask[0]].real<0)
    np.testing.assert_array_equal(actual[:,layout.data_mask[0]].imag<0,expected[:,layout.data_mask[0]].imag<0)
    assert np.all(np.isnan(actual[:,~layout.data_mask[0]]))


def test_batched_ZF_singular_case_stops_without_filter_or_fallback():
    layout = build_fdm_pilot_layout(2,64,16,4,4)
    h = np.broadcast_to(np.eye(2,dtype=np.complex128)[:,:,None,None],(2,2,64,16)).copy()
    h[:,:,1,2] = 1
    y = np.ones((2,64,16),dtype=np.complex128)
    with pytest.raises(ValueError,match="k=1, t=2"):
        v.batched_data_zf(y,h,layout.data_mask)
    with pytest.raises(ValueError,match="singular"):
        detect_data_zf(y,h,layout.data_mask)


def test_real_Pilot_LS_shared_by_both_LMMSE_and_same_Y_for_four_methods(setup):
    _,rows,_,_,layouts,model,train,oracle,_ = setup
    h = formal_trajectory(model,rows[0]).h_frames[0]
    bits,symbols = generate_master_qpsk(bits_seeds=(80461001,80461002))
    noise = complex_gaussian_noise((2,64,16),0.1,np.random.default_rng(90461001))
    timing = {k:0.0 for k in ("LS_seconds","Train_seconds","Oracle_seconds","ZF_seconds")}
    with patch("src.lmmse_train_validation.estimated_receivers",wraps=v.estimated_receivers) as boundary, \
            patch("src.lmmse.estimate_mimo_from_pilot_ls",wraps=v.lmmse.estimate_mimo_from_pilot_ls) as ls_spy, \
            patch("src.lmmse_train_validation.batched_data_zf",wraps=v.batched_data_zf) as detection:
        metrics,diagnostic = v.evaluate_frame(h,bits,symbols,noise,layouts[4],
            tuple(train[4,10,q] for q in range(2)),tuple(oracle[4,10,q] for q in range(2)),2000,0,10,timing)
    assert ls_spy.call_count==2 and ls_spy.call_args_list[0].args[0] is ls_spy.call_args_list[1].args[0]
    assert len({id(c.args[0]) for c in detection.call_args_list})==1 and detection.call_count==4
    assert boundary.call_args.args[0] is detection.call_args_list[0].args[0]
    assert "h_true" not in inspect.signature(v.estimated_receivers).parameters
    pilot_ls = ls_spy.call_args_list[0].args[0]
    x = assemble_pilot_data_frame(symbols,layouts[4])
    for q in range(2):
        mask = layouts[4].pilot_mask[q]
        np.testing.assert_allclose(pilot_ls[:,q,mask],h[:,q,mask]+noise[:,mask]/x[q,mask],rtol=0,atol=1e-14)
    assert metrics["same_y_for_all_methods"] and len(diagnostic["fixed_case"])==2


def test_trajectory_accumulator_requires_all_ten_frames_and_sums():
    row = {"data_bits":100,"nmse_truth_energy":2.0,"ls_nmse_error_energy":0.4,
        "train_nmse_error_energy":0.2,"oracle_nmse_error_energy":0.1,
        "perfect_bit_errors":0,"ls_bit_errors":4,"train_bit_errors":2,"oracle_bit_errors":1,
        "same_y_for_all_methods":True}
    a = v.TrajectoryAccumulator(formal_manifest_rows()[2000],4,10,0.125)
    with pytest.raises(ValueError,match="all ten"): a.finalize()
    for f in range(10): a.add(f,row,f+1)
    actual = a.finalize()
    assert actual["total_data_bits"]==1000 and actual["frames_aggregated"]==10
    assert actual["train_nmse_error_energy"]==pytest.approx(2)
    assert actual["nmse_truth_energy"]==20 and actual["condition_number_max_over_10_frames"]==10
    with pytest.raises(ValueError): a.add(9,row,0)


def test_aggregation_uses_pooled_energy_integer_BER_and_per_frame_proxy():
    rows,theory = [],{}
    for df in v.DF_OPTIONS:
        for snr in v.SNR_OPTIONS:
            for tid,error,truth in ((2000,1.0,10.0),(2001,9.0,30.0)):
                rows.append({"trajectory_id":tid,"dF":df,"snr_db":snr,"frames_aggregated":10,
                    "pilot_ratio":1/(2*df),"total_data_bits":1000,"nmse_truth_energy":truth,
                    **{f"{m}_nmse_error_energy":error for m in v.ESTIMATED_METHODS},
                    **{f"{m}_bit_errors":10 if tid==2000 else 30 for m in v.METHODS},"same_y_for_all_methods":True})
            theory[df,snr] = {"train_predicted_nmse_data_db":10*np.log10(0.25),"oracle_nmse_data_db":10*np.log10(0.25)}
    aggregate = v.aggregate_results(rows,theory)
    a = aggregate[0]
    assert a["train_nmse_linear"]==0.25 and a["train_nmse_linear"]!=0.2
    assert a["train_ber"]==40/2000 and a["data_bits_per_frame"]==100
    assert a["train_proxy_bps"]==pytest.approx(100*0.98/frame_duration_seconds())
    assert a["num_trajectories"]==2 and a["num_correlated_frames"]==20
    assert v.verify_aggregation(aggregate,rows,[2000,2001])


def test_validation_only_rejects_train_before_generation(setup):
    _,_,theory,_,layouts,model,train,oracle,_ = setup
    with patch("src.lmmse_train_validation.formal_trajectory") as generator:
        with pytest.raises(ValueError,match="validation manifest"):
            v.run_validation(formal_manifest_rows()[:1],model,layouts,train,oracle,theory,generator=generator)
        generator.assert_not_called()


@pytest.fixture(scope="module")
def small_run(setup):
    _,rows,theory,_,layouts,model,train,oracle,_ = setup
    with patch("src.lmmse.cho_factor",side_effect=AssertionError("no Cholesky inside loops")), \
            patch("src.lmmse_train.estimate_training_covariance",side_effect=AssertionError("no covariance re-estimation")), \
            patch.object(type(model),"generate",side_effect=AssertionError("no old seed wrapper")), \
            patch("src.doppler.build_tdl_a_doppler_channel",side_effect=AssertionError("no independent 16-symbol generator")), \
            patch("src.lmmse_train_validation.evaluate_frame",wraps=v.evaluate_frame) as frames, \
            patch("src.lmmse_train_validation.scale_unit_noise",wraps=v.scale_unit_noise) as scaled:
        report,result = v.run_validation(rows[:1],model,layouts,train,oracle,theory)
    return report,result,frames.call_args_list,scaled.call_args_list


def test_continuous_CRN_same_H_bits_and_noise_across_conditions_no_preparation(small_run,setup):
    report,rows,calls,scaled = small_run
    assert len(rows)==20 and all(r["frames_aggregated"]==10 for r in rows)
    assert report["execution_counters"]=={"trajectory_generator_calls":1,"master_bits_calls":10,"unit_noise_calls":10,"frame_condition_evaluations":200}
    for frame in range(10):
        group = calls[20*frame:20*(frame+1)]
        assert len({id(c.args[0]) for c in group})==1
        assert len({id(c.args[1]) for c in group})==1 and len({id(c.args[2]) for c in group})==1
        for snr_index in range(5):
            assert len({id(c.args[3]) for c in group[4*snr_index:4*(snr_index+1)]})==1
        assert len({id(c.args[0]) for c in scaled[5*frame:5*(frame+1)]})==1
    assert v.frozen_hashes(ROOT/"outputs",setup[0])==setup[3]
    assert setup[-1]["train_Cholesky_calls"]==setup[-1]["oracle_Cholesky_calls"]==40


@pytest.fixture(scope="module")
def saved():
    return json.loads((ROOT/"outputs/t10d_lmmse_train_validation_results.json").read_text(encoding="utf-8"))


def test_saved_full_counts_hashes_preparation_and_sources(saved,setup):
    assert saved["all_checks_passed"] and all(saved["checks"].values())
    assert saved["visited_validation_trajectory_ids"]==list(range(2000,2400))
    assert saved["num_independent_trajectories"]==400 and saved["num_correlated_frames"]==4000
    assert saved["covariance_hash_before_after"]["before"]==saved["covariance_hash_before_after"]["after"]==setup[3]
    assert saved["prepared_train_estimators"]==saved["prepared_oracle_estimators"]==40
    assert saved["execution_counters"]["frame_condition_evaluations"]==80000
    assert saved["execution_counters"]["master_bits_calls"]==saved["execution_counters"]["unit_noise_calls"]==4000
    assert saved["num_filtered_trajectories"]==0 and saved["numerical_failures"]==[]
    for key in ("covariance_reestimated","covariance_updated","pseudo_inverse_used","jitter_used","formal_test_used","OOD_generated","bootstrap_CI_computed","p_value_computed"):
        assert saved[key] is False
    for path,digest in saved["provenance"]["source_sha256"].items():
        from src.provenance import recorded_source_matches
        assert recorded_source_matches(ROOT,path,digest,"t10d",current_hash=sha256_file)


@pytest.mark.parametrize("method",("train","oracle"))
@pytest.mark.parametrize("index",range(20))
def test_saved_all_40_empirical_theory_comparisons(saved,method,index):
    row = saved["theory_comparisons"][method][index]
    assert row["tolerance_db"]==0.30 and abs(row["delta_db"])<=0.30 and row["passed"]


def test_saved_8000_rows_reconstruct_aggregate(saved):
    import csv
    with (ROOT/"outputs/t10d_lmmse_train_validation_trajectory_metrics.csv").open(encoding="utf-8",newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows)==8000
    assert len({(r["trajectory_id"],r["dF"],r["snr_db"]) for r in rows})==8000
    for r in rows:
        for key in ("trajectory_id","channel_seed","snr_db","dF","frames_aggregated","total_data_bits","perfect_bit_errors","ls_bit_errors","train_bit_errors","oracle_bit_errors"):
            r[key] = int(r[key])
        for key in ("ls_nmse_error_energy","train_nmse_error_energy","oracle_nmse_error_energy","nmse_truth_energy"):
            r[key] = float(r[key])
        r["same_y_for_all_methods"] = r["same_y_for_all_methods"]=="True"
    assert v.verify_aggregation(saved["aggregate_results"],rows,list(range(2000,2400)))


def test_saved_CSV_only_redraw_without_frozen_inputs_or_old_images(saved,tmp_path):
    name = "t10d_lmmse_train_validation_scan.csv"
    shutil.copyfile(ROOT/"outputs"/name,tmp_path/name)
    with patch("scripts.run_lmmse_train_validation.load_frozen_inputs",side_effect=AssertionError("CSV only")), \
            patch("scripts.run_lmmse_train_validation.prepare_pools",side_effect=AssertionError("no preparation")), \
            patch("scripts.run_lmmse_train_validation.run_validation",side_effect=AssertionError("no simulation")):
        main(["--plot-only","--output-dir",str(tmp_path)])
    for figure in FIGURE_NAMES:
        assert sha256_file(tmp_path/f"{figure}.png")==sha256_file(ROOT/"outputs"/f"{figure}.png")
        assert (tmp_path/f"{figure}.pdf").is_file() and (tmp_path/f"{figure}.svg").is_file()
