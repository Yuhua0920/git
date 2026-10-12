"""T14A standard-table, independent namespace, projection and frozen-link gates."""
import inspect
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src import profile_shape_gate as gate,tdl_profiles as profiles,trajectories,ood_mismatch as old,lmmse
from src.b0_development import replay_link,state_hash
from src.lmmse_train import raw_array_sha256,sha256_file

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"
torch.set_num_threads(4)


def test_T13D_committed_and_formal_sealed():
    assert subprocess.check_output(["git","show","-s","--format=%s",gate.T13D_COMMIT],cwd=ROOT,text=True).strip()=="feat: add sparse-pilot channel-statistics information gate"
    p="outputs/t13d_statistics_information_results.json"
    blob=subprocess.check_output(["git","show",gate.T13D_COMMIT+":"+p],cwd=ROOT)
    assert blob==(ROOT/p).read_bytes().replace(b"\r\n",b"\n")
    r=json.loads(blob)
    assert r["all_checks_passed"] and r["formal_test_sealed"] and not r["formal_test_used"]


@pytest.mark.parametrize("profile,count,digest",[("A",23,"5334211f87ec82677a271593b3edcc8e7427a3a290b51737d5b753f6d4dfb5fe"),
    ("B",23,"6d73688d15c57483a42955c7363d1df09f4ab552e733868c005ad6855a583cb6"),
    ("C",24,"faf4e52a341c6bc96623ecf9375b0702c99d54d543909fa2ca51886654119c32")])
def test_official_standard_table_exact_provenance(profile,count,digest):
    from scripts.verify_profile_standard import audit_standard,PDF
    r=audit_standard()
    assert r["PDF_sha256"]==sha256_file(PDF)=="31eb6aec7090bfea17558aad09db41ccd53a764ca1f5754f8be2023268d63c74"
    assert r["profiles"][profile]["tap_count"]==count and r["profiles"][profile]["canonical_table_sha256"]==digest
    assert all(v["fading"]=="Rayleigh" for v in r["profiles"][profile]["rows"])


@pytest.mark.parametrize("profile",["A","B","C"])
def test_normalized_ensemble_power_and_original_delay_scaling(profile):
    from src.tdl import rms_delay_spread_seconds
    delays,power=profiles.profile_parameters(profile,100.)
    assert power.sum()==pytest.approx(1,abs=1e-15)
    np.testing.assert_array_equal(delays,np.asarray(profiles.NORMALIZED_DELAYS[profile])*100*1e-9)
    assert abs(rms_delay_spread_seconds(delays,power)*1e9-100)<.01
    longer,p2=profiles.profile_parameters(profile,300.)
    np.testing.assert_allclose(longer,3*delays,atol=1e-21,rtol=0)
    np.testing.assert_array_equal(power,p2)
    rf=profiles.frequency_covariance(profile)
    f=(np.arange(64)-32)*30000
    direct=sum(p*np.outer(np.exp(-2j*np.pi*f*d),np.exp(-2j*np.pi*f*d).conj()) for p,d in zip(power,delays))
    np.testing.assert_allclose(rf,direct,atol=1e-14,rtol=0)
    np.testing.assert_allclose(np.diag(rf),1,atol=1e-15,rtol=0)
    assert np.linalg.eigvalsh(rf).min()>-1e-12


def test_historical_A_generator_source_and_seeded_arrays_unchanged():
    for p in ("src/tdl.py","src/doppler.py","src/trajectories.py"):
        blob=subprocess.check_output(["git","show",gate.T13D_COMMIT+":"+p],cwd=ROOT)
        assert blob==(ROOT/p).read_bytes().replace(b"\r\n",b"\n")
    a=profiles.prepare_profile_trajectory("A",10)
    assert isinstance(a,trajectories.PreparedTrajectory)
    x=profiles.draw_profile_channel(a,40261018)
    y=trajectories.prepare_trajectory().generate(17)
    np.testing.assert_array_equal(x[0],y.taps); np.testing.assert_array_equal(x[1],y.h_long); np.testing.assert_array_equal(x[2],y.h_frames)
    np.testing.assert_array_equal(profiles.frequency_covariance("A"),lmmse.build_frequency_covariance())


def test_only_PDP_changes_same_Jakes_time_and_continuous_frames():
    a=profiles.prepare_profile_trajectory("A")
    for p in gate.PROFILES:
        model=profiles.prepare_profile_trajectory(p)
        np.testing.assert_array_equal(model.covariance,a.covariance)
        np.testing.assert_array_equal(model.template.factor,a.factor)
        assert model.mobility==a.mobility and model.template.symbol_duration==a.symbol_duration
        taps,h,frames=profiles.draw_profile_channel(model,gate.seed(14001,p,0))
        assert taps.shape==(2,2,len(model.powers),80) and h.shape==(2,2,64,80) and frames.shape==(5,2,2,64,16)
        assert np.shares_memory(h,frames) and not h.flags.writeable
        assert model.covariance[0,16]>.9
    for p in ("D","E","CDL"):
        with pytest.raises(ValueError):
            profiles.prepare_profile_trajectory(p)


def test_new_seed_namespaces_and_calibration_performance_disjoint():
    new=[]; historical=[]
    for p in gate.PROFILES:
        for tid in range(200):
            new.extend(gate.seed(14001,p,tid,purpose=k) for k in (0,1))
            for f in range(5):
                new.append(gate.seed(14003,p,tid,f))
                new.extend(gate.seed(14002,p,tid,f,q) for q in (0,1))
    for p in old.SHIFTS:
        for tid in range(200):
            historical.extend(old.seed(13001,p,tid,purpose=k) for k in (0,1))
            for f in range(5):
                historical.append(old.seed(13003,p,tid,f))
                historical.extend(old.seed(13002,p,tid,f,q) for q in (0,1))
    assert len(set(new))==len(new) and set(new).isdisjoint(historical)
    assert gate.BASE_SEED!=old.BASE_SEED and gate.seed_words(14001,"B",0)[0]==14001
    assert len(gate.manifest())==400 and all(r["split"]=="profile_shape_dev" for r in gate.manifest())
    assert not any(40261001<=s<=40265000 for s in new)
    with pytest.raises(ValueError):
        gate.seed(13001,"B",0)


@pytest.mark.parametrize("p",gate.PROFILES)
def test_BC_empirical_calibration_before_scan(p):
    r=gate.calibrate_channel(profiles.prepare_profile_trajectory(p),p)
    assert r["all_checks_passed"] and len(r["frequency_covariance"])==5 and len(r["temporal_covariance"])==2
    with pytest.raises(ValueError):
        gate.require_calibration({"B":r,"C":{**r,"all_checks_passed":False}})


@pytest.fixture(scope="module")
def projections():
    return {(p,s):gate.project_to_A_family(p,s) for p in ("A","B","C") for s in gate.SNRS}


@pytest.mark.parametrize("snr",gate.SNRS)
def test_A100_self_projection_recovers100_and_KL_identity(projections,snr):
    row,grid,rf=projections["A",snr]
    assert abs(row["tau_eff_ns"]-100)<1e-3 and abs(row["projection_KL"])<4e-9
    assert not row["projection_boundary_hit"]
    np.testing.assert_array_equal(rf,lmmse.build_frequency_covariance())
    coarse=[r for r in grid if r["phase"]=="coarse"]
    assert [r["tau_ns"] for r in coarse]==list(gate.COARSE_TAUS)
    assert not row["true_channel_used"] and not row["empirical_data_used"]


def test_BC_projection_four_actual_link_KL_no_half(projections):
    for p in gate.PROFILES:
        for s in gate.SNRS:
            row,grid,rf=projections[p,s]
            assert row["family"]=="TDL-A" and 20<=row["tau_eff_ns"]<=600
            np.testing.assert_array_equal(rf,profiles.frequency_covariance("A",row["tau_eff_ns"]))
            assert row["projection_KL"]<=row["A100_KL"]+1e-9
            for r in grid:
                assert r["KL_total"]==sum(r[f"KL_H{nr}{tx}"] for nr in range(2) for tx in range(2))
            assert min(r["KL_total"] for r in grid)>-4e-9
    from src import statistics_information as i
    a=i.prepare_covariance(2*np.eye(3)); b=i.prepare_covariance(3*np.eye(3))
    assert i.kl_four_links((a,)*4,(b,)*4)[0]==pytest.approx(12*(np.log(1.5)+2/3-1),abs=1e-14)


@pytest.fixture(scope="module")
def frozen():
    return old.load_frozen_estimators(OUT,torch.device("cpu"))


def test_R4_B0_frozen_and_R5_exact_profile_theory(frozen,projections):
    covariance,b0,r4,b0_record=frozen
    assert b0_record["epoch"]==57 and b0_record["mixed_SNR_model"] and not b0_record["adapted"]
    assert not any(p.requires_grad for p in b0.parameters()) and not b0.training
    assert covariance.file_sha256==r4["file_sha256"] and state_hash(b0)==b0_record["model_state_sha256"]
    for p in gate.PROFILES:
        pools,records=gate.prepare_estimators(covariance,p,{k:v[0] for k,v in projections.items()})
        true=lmmse.build_full_grid_covariance(profiles.frequency_covariance(p),lmmse.build_time_covariance())
        assert all(r["R5_full_grid_sha256"]==raw_array_sha256(true) for r in records)
        for snr,pool in pools.items():
            assert pool["R8"].family=="TDL-A"
            for q,e in enumerate(pool["R5"]):
                indices=lmmse.pilot_flat_indices(gate.LAYOUT.pilot_mask[q])
                np.testing.assert_array_equal(e.r_pp,true[np.ix_(indices,indices)])


def test_R8_receiver_only_ignores_nonpilot_entries(frozen,projections):
    assert list(inspect.signature(gate.estimate_A_family).parameters)==["pilot_ls","prepared"]
    assert list(inspect.signature(gate.project_to_A_family).parameters)==["true_profile","snr"]
    pools,_=gate.prepare_estimators(frozen[0],"B",{k:v[0] for k,v in projections.items()})
    ls=np.full((2,2,64,16),np.nan+1j*np.nan)
    rng=np.random.default_rng(44)
    for q in range(2):
        ls[:,q,gate.LAYOUT.pilot_mask[q]]=rng.normal(size=(2,16))+1j*rng.normal(size=(2,16))
    original=gate.estimate_A_family(ls,pools[10]["R8"])
    for q in range(2):
        ls[:,q,~gate.LAYOUT.pilot_mask[q]]=1e20+1j*1e20
    np.testing.assert_array_equal(original,gate.estimate_A_family(ls,pools[10]["R8"]))


@pytest.mark.parametrize("p",gate.PROFILES)
def test_small_five_frame_full_link_paired_inputs(frozen,projections,p):
    pools,_=gate.prepare_estimators(frozen[0],p,{k:v[0] for k,v in projections.items()})
    timing={k:0. for k in gate.TIMING_KEYS}; state=state_hash(frozen[1])
    rows,r=gate.evaluate_trajectory(profiles.prepare_profile_trajectory(p),p,0,frozen[1],pools,replay_link(ROOT),timing)
    assert len(rows)==3 and all(row["frames_aggregated"]==5 and row["total_bits"]==19840 for row in rows)
    assert all(row["same_H_bits_noise_Y"] and row["truth_energy"]>0 for row in rows)
    assert r["historical_link_exact_checks"]==15 and len(r["frame_realizations"])==5
    assert state_hash(frozen[1])==state


def test_scientific_thresholds_are_fixed_at20dB():
    def rows(a,b):
        return [{"profile":"B","snr_db":20,"in_family_residual_db":a},{"profile":"C","snr_db":20,"in_family_residual_db":b}]
    assert gate.scientific_flags(rows(.5,.5))["tdla_parametric_family_adequate"]
    assert gate.scientific_flags(rows(.5001,1.4999))["partial_profile_model_mismatch"]
    assert gate.scientific_flags(rows(.5,1.5))["out_of_family_shape_mismatch_material"]


@pytest.fixture(scope="module")
def report():
    path=OUT/(gate.PREFIX+"_results.json")
    if not path.exists():
        pytest.skip("full profile gate not yet executed")
    return json.loads(path.read_text(encoding="utf8"))


def test_full_artifact_counts_frozen_provenance_and_formal_sealed(report):
    from scripts.run_profile_shape_gate import accepted_files
    assert report["trajectory_rows"]==1200 and len(report["performance_results"])==6
    assert len(report["realization_hashes"])==400
    assert report["formal_test_sealed"] and not report["formal_test_used"] and not report["new_model_trained"]
    assert report["provenance"]["accepted_before"]==report["provenance"]["accepted_after"]==accepted_files()
    assert all(sha256_file(ROOT/p)==h for p,h in report["provenance"]["source_sha256"].items())
    assert all(sha256_file(OUT/p)==h for p,h in report["provenance"]["output_sha256"].items())


def test_raw_CSV_reaggregation_and_scientific_gate(report):
    from scripts.run_profile_shape_gate import read_trajectory
    rows=read_trajectory(OUT/(gate.PREFIX+"_trajectory_metrics.csv"))
    assert gate.aggregate(rows)==report["performance_results"]
    assert gate.paired_diagnostics(rows)==report["paired_diagnostics"]
    assert all(report[k]==v for k,v in gate.scientific_flags(report["performance_results"]).items())


def test_CSV_only_plot_preflight(tmp_path):
    from scripts.run_profile_shape_gate import render_and_audit
    from scripts.run_lmmse_train_validation import save_csv
    rows=[{"profile":p,"snr_db":s,**{m+"_nmse_db":-s*.8+j*1.5 for j,m in enumerate(gate.METHODS)},"in_family_residual_db":1.5}
        for p in gate.PROFILES for s in gate.SNRS]
    save_csv(tmp_path/(gate.PREFIX+".csv"),rows)
    with patch.object(gate,"evaluate_trajectory",side_effect=AssertionError("simulation")):
        record=render_and_audit(tmp_path,tmp_path)
    assert len(record)==2 and all(r["collision_audit"]["verdict"]=="PASS" for r in record)
