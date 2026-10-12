"""Small actual-link/gate tests before T13A scan; final artifacts regressions."""
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src import ood_mismatch as scan
from src.b0_development import replay_link, state_hash
from src.lmmse_train import raw_array_sha256, sha256_file
from src.pilots import count_layout_resources
from src.tdl import rms_delay_spread_seconds, tdl_a_profile
from src.trajectories import prepare_trajectory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs"


def test_T12E_committed_at_required_message_with_accepted_artifacts():
    assert subprocess.check_output(["git", "show", "-s", "--format=%s", scan.T12E_COMMIT], cwd=ROOT, text=True).strip() == "feat: add high-SNR capacity probe for B0"
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", scan.T12E_COMMIT], cwd=ROOT, text=True).splitlines()
    assert "outputs/t12e_b0_df16_20db_results.json" in files
    report = json.loads((OUT/"t12e_b0_df16_20db_results.json").read_text(encoding="utf8"))
    assert report["all_checks_passed"]
    for p, h in report["provenance"]["source_sha256"].items():
        assert sha256_file(ROOT/p) == h


def test_dedicated_roles_purposes_shifts_and_no_old_seed_collision():
    from src.lmmse_train_validation import bits_seed, noise_seed
    seeds = set()
    for purpose in (0, 1):
        for shift in scan.SHIFTS:
            for tid in range(200):
                key = scan.seed(13001, shift, tid, purpose=purpose)
                assert key not in seeds; seeds.add(key)
                assert not 40261001 <= key < 40263401 and not 60261001 <= key < 60263401
                if purpose == 1:
                    for frame in range(5):
                        for role, q in [(13002, 0), (13002, 1), (13003, 0)]:
                            key = scan.seed(role, shift, tid, frame, q)
                            assert key not in seeds; seeds.add(key)
                            assert key not in {bits_seed(2000, 0, 0), bits_seed(2000, 0, 1), noise_seed(2000, 0)}
    assert scan.seed_words(13003, "speed120", 0) == [13003, 20261301, 1, 1, 0, 0, 0]


@pytest.mark.parametrize("args", [(0,"speed120",0), (13001,"bad",0), (13001,"speed120",200), (13001,"DS300",-1)])
def test_bad_seed_coordinates_rejected(args):
    with pytest.raises(ValueError):
        scan.seed(*args)


@pytest.mark.parametrize("shift", list(scan.SHIFTS))
def test_single_factor_channel_and_continuous_frames(shift):
    base = prepare_trajectory(frames_per_trajectory=5)
    changed = scan.prepare_channel(shift)
    assert np.array_equal(base.time_grid_s, changed.time_grid_s)
    assert changed.time_grid_s.size == 80 and changed.symbol_duration == 80/1920000
    if shift == "speed120":
        assert changed.mobility["speed_kmh"] == 120
        assert np.array_equal(changed.delays_s, base.delays_s)
        assert np.array_equal(changed.phase_matrix, base.phase_matrix)
        assert not np.array_equal(changed.covariance, base.covariance)
    else:
        assert changed.mobility == base.mobility
        assert np.array_equal(changed.covariance, base.covariance)
        np.testing.assert_allclose(changed.delays_s, 3*base.delays_s, rtol=1e-15, atol=0)
    assert np.array_equal(changed.powers, base.powers)
    taps, h, frames = scan.draw_channel(changed, shift, 0)
    assert h.dtype == taps.dtype == np.complex128
    assert h.shape == (2,2,64,80) and frames.shape == (5,2,2,64,16)
    assert np.shares_memory(h,frames) and not h.flags.writeable
    assert np.array_equal(frames[1], h[...,16:32])
    assert not np.array_equal(frames[0],frames[1])
    assert np.array_equal(h, scan.draw_channel(changed,shift,0)[1])
    assert not np.array_equal(h,scan.draw_channel(changed,shift,0,purpose=0)[1])


def test_actual_pilot_masks_only_df16_three_SNRs_and_manifest():
    assert scan.SNRS == (5,10,20)
    assert scan.LAYOUT.offsets_by_tx == (0,8)
    assert count_layout_resources(scan.LAYOUT)["union_ratio"] == 1/32
    assert np.array_equal(np.flatnonzero(scan.LAYOUT.pilot_mask[0,:,0]),np.arange(0,64,16))
    rows = scan.manifest()
    assert len(rows) == 400 and {r["split"] for r in rows} == {"ood_dev"}
    assert all(r["frames_per_trajectory"]==5 and r["symbols_per_trajectory"]==80 for r in rows)


def test_calibration_records_independent_small_pool_and_existing_RMS(monkeypatch):
    monkeypatch.setattr(scan,"NUM_TRAJECTORIES",2)
    model=scan.prepare_channel("DS300"); record=scan.calibrate_channel(model,"DS300")
    delays,powers=tdl_a_profile(300)
    assert record["deterministic_RMS_delay_ns"] == rms_delay_spread_seconds(delays,powers)*1e9
    assert record["seed_purpose"]==0 and record["num_independent_trajectories"]==2
    assert [r["lag"] for r in record["temporal_correlations"]]==[1,4]
    assert [r["distance"] for r in record["frequency_correlations"]]==[1,4,8,16,32]
    assert np.isfinite(record["channel_power"]) and np.isfinite(record["empirical_RMS_delay_ns"])


def test_failed_calibration_blocks_any_estimator_or_scan(tmp_path,monkeypatch):
    from scripts import run_ood_mismatch as script
    monkeypatch.setattr(script,"OUT",tmp_path)
    monkeypatch.setattr(script,"accepted_files",lambda:{})
    monkeypatch.setattr(script,"verify_frozen",lambda _: {})
    monkeypatch.setattr(script,"load_config",lambda _: {})
    monkeypatch.setattr(script,"configure_reproducibility",lambda *a: {})
    monkeypatch.setattr(scan,"load_frozen_estimators",lambda *a:(None,None,{},{}))
    monkeypatch.setattr(scan,"id_references",lambda _: {})
    monkeypatch.setattr(script,"replay_link",lambda _: {})
    monkeypatch.setattr(scan,"prepare_channel",lambda _: None)
    monkeypatch.setattr(scan,"calibrate_channel",lambda model,shift:{"channel_power":2.,"all_checks_passed":False})
    with patch.object(script.subprocess,"check_output",return_value=scan.T12E_COMMIT),patch.object(scan,"prepare_estimators",side_effect=AssertionError("scan started")):
        with pytest.raises(ValueError,match="calibration failed"):
            script.execute()
    record=json.loads((tmp_path/f"{scan.PREFIX}_results.json").read_text(encoding="utf8"))
    assert not record["all_checks_passed"]
    assert not (tmp_path/f"{scan.PREFIX}.csv").exists()


@pytest.fixture(scope="module")
def frozen_pair():
    with patch("torch.optim.Adam",side_effect=AssertionError("new optimizer")):
        return scan.load_frozen_estimators(OUT,torch.device("cpu"))


def test_frozen_R4_B0_hash_identity_and_inference_only(frozen_pair):
    frozen,model,r4,b0=frozen_pair
    assert r4["file_sha256"]=="b41ec0128243d07efb7577785e798994e5d8bc00905ab40d174ff1ee0da8dc3e"
    assert r4["raw_array_sha256"]=="8a0d13fafee0ca20d0eac195cc58910d4465f646047b51075d774883e474f597"
    assert b0["file_sha256"]=="999d5cb1318cd5be8fc02c3abbdfcf2cd8bb9821f5d65f1d761b58b27028a50e"
    assert b0["model_state_sha256"]=="82a373b1a2e37bb049031cf1d51c0e25675067e67573a9f2b695876523ac9aec"
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert not frozen.R_train.flags.writeable and b0["epoch"]==57


@pytest.mark.parametrize("shift", list(scan.SHIFTS))
def test_R5_covariance_is_OOD_theory_and_only_other_factor_fixed(shift,frozen_pair):
    from src import lmmse
    pools,record=scan.prepare_estimators(frozen_pair[0],shift)
    spec=scan.SHIFTS[shift]
    rf=lmmse.build_frequency_covariance(desired_rms_delay_spread_ns=spec["delay_spread_ns"])
    rt=lmmse.build_time_covariance(speed_kmh=spec["speed_kmh"])
    full=np.kron(rf,rt)
    assert record["full_grid_raw_sha256"]==raw_array_sha256(full) and not record["empirical_data_used"]
    for snr,pairs in pools.items():
        for q in range(2):
            obj=pairs["R5_ood_oracle"][q]
            assert np.array_equal(obj.r_hp,full[:,obj.pilot_indices])
            assert np.array_equal(pairs["R4_frozen"][q].r_hp,frozen_pair[0].R_train[:,obj.pilot_indices])
            assert obj.noise_variance==10**(-snr/10)


@pytest.mark.parametrize("shift", list(scan.SHIFTS))
def test_small_actual_OOD_full_link_shared_observations_and_no_training(shift,frozen_pair):
    frozen,model,r4,b0=frozen_pair
    pools,_=scan.prepare_estimators(frozen,shift)
    timing={k:0. for k in ("channel_generation_seconds","R3_seconds","R4_seconds","R5_seconds","B0_inference_seconds","ZF_seconds")}
    observed=[]
    original=scan.batched_data_zf
    def capture(y,h,mask):
        observed.append((id(y),raw_array_sha256(y)))
        return original(y,h,mask)
    with patch.object(scan,"batched_data_zf",side_effect=capture),patch("src.lmmse_train.formal_trajectory",side_effect=AssertionError("formal pool accessed")),patch("src.b0_model.training_step",side_effect=AssertionError("training")):
        rows,realization=scan.evaluate_trajectory(scan.prepare_channel(shift),shift,0,model,pools,replay_link(ROOT),timing)
    assert len(rows)==3 and len(realization["frame_realizations"])==5
    assert all(r["frames_aggregated"]==5 and r["same_H_bits_noise_Y"] for r in rows)
    assert len(observed)==75
    for i in range(0,len(observed),5):
        assert len(set(observed[i:i+5]))==1
    assert all(r["total_bits"]==5*3968 for r in rows)
    assert state_hash(model)==b0["model_state_sha256"] and raw_array_sha256(frozen.R_train)==r4["raw_array_sha256"]
    assert all(np.isfinite(r[f"{m}_error_energy"]) for r in rows for m in scan.METHODS)


def toy_rows():
    return [{"shift":s,"snr_db":n,"trajectory_id":i,"frames_aggregated":5,"same_H_bits_noise_Y":True,
        "nmse_truth_energy":10.,"total_bits":100,
        **{f"{m}_error_energy":float((j+1)*(i+1)) for j,m in enumerate(scan.METHODS)},
        **{f"{m}_bit_errors":j+i for j,m in enumerate((*scan.METHODS,"Perfect"))}}
        for s in scan.SHIFTS for n in scan.SNRS for i in range(2)]


def test_trajectory_first_pooled_energy_BER_aggregation_and_incomplete_rejection():
    refs={s:{m:-10. for m in ("R3","R4_frozen","B0")} for s in scan.SNRS}
    rows=toy_rows(); aggregates=scan.aggregate(rows,refs,expected_units=2)
    assert len(aggregates)==6
    assert aggregates[0]["R3_error_energy"]==3 and aggregates[0]["nmse_truth_energy"]==20
    assert aggregates[0]["R3_nmse_db"]==10*np.log10(3/20)
    assert aggregates[0]["R3_ber"]==1/200
    with pytest.raises(ValueError):
        scan.aggregate(rows[:-1],refs,expected_units=2)
    paired=scan.paired_diagnostics(rows)
    assert len(paired)==36
    assert paired[0]["NMSE_difference_db"]["median"]==pytest.approx(10*np.log10(.5))
    assert paired[0]["NMSE_difference_db"]["fraction_lower_error"]==1
    assert all(p["inferential_test"] is None for p in paired)


@pytest.mark.parametrize("penalty,margin,weak,material,outperforms",[(.49,1.,True,False,False),(.5,1.,False,False,False),(1.,0.,False,True,False),(2.,-.1,False,True,True)])
def test_preregistered_direction_boundaries(penalty,margin,weak,material,outperforms):
    assert scan.direction_flags([{"R4_mismatch_penalty_db":penalty,"B0_minus_R4_db":margin}])=={
        "frozen_statistics_mismatch_weak":weak,"statistics_mismatch_material":material,"learned_baseline_outperforms_frozen_lmmse":outperforms}


def test_two_CSV_only_figures_and_alignment_before_scan(tmp_path):
    from scripts.run_ood_mismatch import render_and_audit
    from scripts.run_lmmse_train_validation import save_csv
    refs={s:{m:-10. for m in ("R3","R4_frozen","B0")} for s in scan.SNRS}
    save_csv(tmp_path/f"{scan.PREFIX}.csv",scan.aggregate(toy_rows(),refs,expected_units=2))
    with patch.object(scan,"evaluate_trajectory",side_effect=AssertionError("numerical scan")):
        figures=render_and_audit(tmp_path,tmp_path)
    assert len(figures)==2
    assert figures[0]["alignment_audit"]["verdict"]=="PASS"
    assert all(f["collision_audit"]["verdict"]=="PASS" for f in figures)


@pytest.fixture(scope="module")
def report():
    path=OUT/f"{scan.PREFIX}_results.json"
    if not path.exists():
        pytest.skip("T13A full OOD scan has not executed")
    return json.loads(path.read_text(encoding="utf8"))


def test_complete_channel_gate_provenance_and_formal_seal(report):
    assert all(c["all_checks_passed"] for c in report["calibration"].values())
    assert report["git"]["T12E_committed_cleanly"] and report["git"]["clean_after_commit"]
    assert report["snr_db"]==[5,10,20] and report["dF"]==16
    assert not report["new_model_trained"] and not report["formal_training_performed"]
    assert report["formal_test_sealed"] and not report["formal_test_used"] and not report["MMSE_detector_used"]
    assert report["provenance"]["accepted_before"]==report["provenance"]["accepted_after"]
    assert all(sha256_file(ROOT/p)==h for p,h in report["provenance"]["accepted_after"].items())
    assert report["b0_checkpoint_provenance"]["epoch"]==57 and not report["b0_checkpoint_provenance"]["T12E_model_used"]


def test_complete_1200_wide_rows_reaggregate_and_paired_descriptive_only(report):
    rows=scan.read_csv(OUT/f"{scan.PREFIX}_trajectory_metrics.csv")
    assert len(rows)==1200 and len(report["aggregate_results"])==6
    for a in report["aggregate_results"]:
        group=[r for r in rows if r["shift"]==a["shift"] and int(r["snr_db"])==a["snr_db"]]
        assert len(group)==200 and all(int(r["frames_aggregated"])==5 for r in group)
        assert sum(int(r["total_bits"]) for r in group)==a["total_bits"]
        assert sum(float(r["R4_frozen_error_energy"]) for r in group)==a["R4_frozen_error_energy"]
    assert len(report["paired_diagnostics"])==36 and report["no_inferential_statistics"]
    assert report["common_random_numbers_verified"] and report["exact_T10D_link_checks"]==6000
    assert report["ID_comparison_paired"] is False and report["ID_scan_rerun"] is False
    assert all(report[k]==v for k,v in scan.direction_flags(report["aggregate_results"]).items())
