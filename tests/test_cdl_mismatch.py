"""T15C small correctness, namespace, historical and paired regression gates."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest
from src.cdl_mismatch import (SHIFTS,METHODS,SNRS,T15B_COMMIT,shift_definition,seeds,
    manifest_rows,frozen_variances,gaussian_kl,covariance_diagnostics,scientific_flags,paired_summary)
from src.cdl_benchmark import ARRAY,SYSTEM,FrozenStatistics,StreamingStatistics,seeds as old_seeds
from src.spatial_information import observation_geometry
ROOT=Path(__file__).resolve().parents[1]


def load(path):return json.loads((ROOT/path).read_text(encoding="utf8"))


def test_T15B_historical_snapshot_committed_and_saved_hashes_unchanged():
    assert subprocess.check_output(["git","show","-s","--format=%s",T15B_COMMIT],text=True).strip()=="feat: add full-covariance complex-GMM nonlinear-prior value gate"
    subprocess.check_call(["git","merge-base","--is-ancestor",T15B_COMMIT,"HEAD"])
    r=load("outputs/t15b_gmm_results.json")
    for p,h in r["source_sha256"].items():
        assert hashlib.sha256(subprocess.check_output(["git","show",f"{T15B_COMMIT}:{p}"])).hexdigest()==h
    for p,h in r["numeric_artifact_sha256"].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h


def test_source_R10_hashes_exact_and_readonly():
    from scripts.run_cdl_mismatch import source_statistics
    stats,p=source_statistics()
    assert p["sha256"]=="ce9f2d0f567e20ff98fa88e098a270bec47dafb5839e14356a98282a514b4b69"
    assert p["exact_T15A_FIX_hash"] and not p["source_statistics_refitted"]
    assert stats.samples==5000 and stats.train_trajectory_ids==tuple(range(500))
    assert all(not a.flags.writeable for a in (stats.mean_p,stats.mean_d,stats.rpp,stats.rdp))


@pytest.mark.parametrize("name",SHIFTS)
def test_only_declared_shift_and_no_normalization(name):
    d=shift_definition(name)
    assert d["ofdm_channel_normalize"] is False and d["dF"]==16 and d["spacing_time"]==4
    system=dict(d["system"]);system["speed_kmh"]=30.
    array=dict(d["array"]);array["ut_velocity_mps"]=[30/3.6,0.,0.]
    assert system==SYSTEM and array==ARRAY
    assert d["profile"]==("B" if name=="CDL-B-30" else "C")
    assert d["system"]["speed_kmh"]==(30. if name=="CDL-B-30" else 120.)


def test_namespaces_splits_channel_frame_scope_and_no_old_seed_reuse():
    rows=manifest_rows();assert len(rows)==12000
    roles=list(old_seeds(0));new={r:set() for r in roles};old={r:set() for r in roles}
    for row in rows:
        assert row["namespace"].startswith("T15C_")
        for r in roles:new[r].add(row[r])
        if row["frame_id"]==9:
            initial=seeds(row["shift"],row["split"],row["trajectory_id"],0)
            assert all(row[r]==initial[r] for r in roles[:3])
            assert row["noise_seed"]!=initial["noise_seed"]
    for i in range(600):
        for j in range(10):
            for r,v in old_seeds(i,j).items():old[r].add(v)
    assert all(new[r].isdisjoint(old[r]) for r in roles)
    for name in SHIFTS:
        train={r["trajectory_id"] for r in rows if r["shift"]==name and r["split"]=="train_dev"}
        val={r["trajectory_id"] for r in rows if r["shift"]==name and r["split"]=="validation_dev"}
        assert train==set(range(500)) and val==set(range(500,600)) and train.isdisjoint(val)


@pytest.mark.parametrize("split,i",[("train_dev",500),("validation_dev",0),("formal_test",600)])
def test_invalid_split_rejected(split,i):
    with pytest.raises(ValueError):seeds("CDL-B-30",split,i)


def test_source_noise_reference_frozen_and_no_target_argument():
    p,v=frozen_variances(load("outputs/t15a_cdl_results.json"))
    assert p==9.941134822523972
    assert v=={5:3.1436628565989504,10:.9941134822523973,20:.09941134822523973}
    assert METHODS==("R3","R10-C-frozen","R10-matched","Perfect-CSI") and SNRS==(5,10,20)


def test_same_actual_pilot_layout_and_counts():
    g=observation_geometry()
    assert len(g.selected)==64 and np.count_nonzero(g.data)==3968
    assert g.layout.pilot_union.mean()==1/32
    assert np.flatnonzero(g.layout.pilot_mask[0].any(axis=0)).tolist()==[0,4,8,12]


def test_matched_statistics_train_only_population_and_validation_rejected():
    stream=StreamingStatistics(np.array([0,1]),np.array([False,False,True]),dimension=3)
    x=np.array([[1+1j,2,3],[4,5+2j,6]],complex);stream.update(x,[0]);f=stream.freeze()
    xc=x-x.mean(axis=0)
    np.testing.assert_allclose(f.rpp,xc[:,:2].T@xc[:,:2].conj()/2)
    np.testing.assert_allclose(f.rdp,xc[:,2:].T@xc[:,:2].conj()/2)
    with pytest.raises(ValueError):stream.update(x,[500])
    assert f.train_trajectory_ids==(0,)


def test_complex_Gaussian_KL_scalar_known_value_no_half():
    m=np.zeros(1,complex);a=np.array([[2]],complex);b=np.array([[3]],complex)
    assert gaussian_kl(m,a,np.array([1j]),b)==pytest.approx(np.log(3/2)+2/3-1+1/3,abs=1e-14)
    assert gaussian_kl(m,a,m,a)==pytest.approx(0.,abs=1e-14)


def test_covariance_diagnostics_identity_and_top_r_angles():
    rng=np.random.default_rng(3);q=np.linalg.qr(rng.normal(size=(64,64))+1j*rng.normal(size=(64,64)))[0]
    r=q@np.diag(np.arange(1,65))@q.conj().T
    s=FrozenStatistics(np.zeros(64,complex),np.zeros(5,complex),r,np.ones((5,64),complex),(0,),10)
    d=covariance_diagnostics(s,s,{10:1.})
    assert d["rpp_relative_frobenius"]==d["rdp_relative_frobenius"]==0
    assert max(d["subspaces"]["16"]["angles_degrees"])<1e-10
    assert abs(d["pilot_gaussian_kl"]["10"]["source_to_target_nats"])<1e-10


@pytest.mark.parametrize("values,expected",[([.1]*6,"weak"),([.5]*6,"partial"),([2.,.1,.1,.1,.1,.1],"partial"),([.1,2.,.1,.1,.1,.1],"material")])
def test_preregistered_scientific_gate(values,expected):
    rows=[dict(snr_db=s,mismatch_penalty_db=v) for s,v in zip((5,10,20,5,10,20),values)]
    flags=scientific_flags(rows)
    assert sum(flags.values())==1 and flags[f"realistic_cdl_adaptation_gap_{expected}"]


def test_paired_trajectory_summary():
    f=[dict(error_energy=2.,nmse_linear=.2,ber=.1),dict(error_energy=1.,nmse_linear=.1,ber=.1)]
    m=[dict(error_energy=1.,nmse_linear=.1,ber=.05),dict(error_energy=2.,nmse_linear=.2,ber=.2)]
    d=paired_summary(f,m)
    assert d["independent_unit"]=="trajectory" and d["fraction_matched_lower_error"]==.5
    assert d["error_energy_difference_q25_median_q75"][1]==0.


def test_small_paired_link_shares_inputs_no_mutation():
    from scripts.run_cdl_mismatch import evaluate_trajectory
    g=observation_geometry();h=np.ones((2,2,64,160),complex);h[0,0]*=2;h[1,1]*=3
    class FakeEstimator:
        def estimate(self,z):
            assert not z.flags.writeable
            return h[:,:,:,:16].ravel()[g.data].copy()
    estimates={(m,s):FakeEstimator() for m in ("R10-C-frozen","R10-matched") for s in SNRS}
    before=hashlib.sha256(h.tobytes()).hexdigest()
    rows,_,proof,_,_,_=evaluate_trajectory("CDL-B-30",500,h,g,estimates,{s:.01 for s in SNRS})
    assert len(rows)==12 and proof["shared_H_bits_Z_Y_verified"]
    assert before==hashlib.sha256(h.tobytes()).hexdigest()
    for s in SNRS:
        r={x["method"]:x for x in rows if x["snr_db"]==s}
        assert r["R10-C-frozen"]["bit_errors"]==r["R10-matched"]["bit_errors"]==r["Perfect-CSI"]["bit_errors"]


def test_actual_preflight_source_and_target_if_present():
    if not (ROOT/"outputs/t15c_cdl_preflight.json").exists():return
    r=load("outputs/t15c_cdl_preflight.json")
    assert r["all_checks_passed"] and r["source_H_hash_matched"]
    assert len(r["exact_source_link_regression"])==9 and all(x["exact"] for x in r["exact_source_link_regression"])
    assert not r["complete_raw_H_saved"]


def test_csv_only_figure_preflight_scoped_style_and_byte_rebuild(tmp_path):
    from scripts.plot_cdl_mismatch import audit
    import matplotlib
    before=dict(matplotlib.rcParams)
    with (tmp_path/"t15c_cdl_mismatch.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["shift","method","snr_db","nmse_db"]);w.writeheader()
        for name in SHIFTS:
            for s in SNRS:
                for m,offset in (("R3",4),("R10-C-frozen",2),("R10-matched",0)):
                    w.writerow(dict(shift=name,method=m,snr_db=s,nmse_db=-s+offset))
    audit(tmp_path,tmp_path/"figures")
    assert dict(matplotlib.rcParams)==before
    r=json.loads((tmp_path/"figures/t15c_cdl_figure_audit.json").read_text(encoding="utf8"))
    assert all(r["CSV_only_rebuild"].values())


def test_generated_outputs_frozen_noise_train_only_paired_and_sealed_if_present():
    if not (ROOT/"outputs/t15c_cdl_mismatch_results.json").exists():return
    r=load("outputs/t15c_cdl_mismatch_results.json")
    assert r["formal_test_sealed"] and not r["formal_test_used"] and not r["new_model_trained"]
    assert not r["source_statistics_refitted"] and r["snr_policy"]["no_per_shift_H_normalization"]
    assert len(r["paired_evaluation_evidence"])==200 and all(x["shared_H_bits_Z_Y_verified"] for x in r["paired_evaluation_evidence"])
    for info in r["matched_statistics_provenance"].values():
        assert info["training_only"] and not info["validation_used_for_fitting"] and info["trajectories"]==500
        assert hashlib.sha256((ROOT/info["path"]).read_bytes()).hexdigest()==info["sha256"]
    for audit in r["snr_policy"]["audit"]:
        assert audit["production_sigma2"]==r["snr_policy"]["production_variances"][str(audit["nominal_source_snr_db"])]
        assert not audit["validation_recalibrated_noise"] and not audit["target_power_used_for_production_noise"]
    with (ROOT/"outputs/t15c_cdl_mismatch_trajectory_metrics.csv").open(encoding="utf8") as f:rows=list(csv.DictReader(f))
    assert len(rows)==2400
    for a in r["nmse_results"]:
        group=[x for x in rows if x["shift"]==a["shift"] and x["method"]==a["method"] and int(x["snr_db"])==a["snr_db"]]
        assert len(group)==100 and {int(x["trajectory_id"]) for x in group}==set(range(500,600))
        assert sum(int(x["bit_errors"]) for x in group)==a["bit_errors"]
        assert sum(int(x["total_bits"]) for x in group)==a["total_bits"]
        np.testing.assert_allclose(a["nmse_linear"],sum(float(x["error_energy"]) for x in group)/sum(float(x["truth_energy"]) for x in group),rtol=1e-14)
    for p,h in r["source_sha256"].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    for p,h in r["frozen_history_sha256"].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
