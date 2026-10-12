"""T16C load-only provenance, namespace, complex oracle and measured gates."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from src.heldout_basis import (T16B_COMMIT,HELDOUT,RANKS,definition,seeds,manifest_rows,load_bases,
    FrozenOutputBasis,gate,qthreshold,gap_closure)
from src.cdl_benchmark import FrozenStatistics,StreamingStatistics,array_hash
from src.lowrank_filter import WeightedCorrection
from src.shared_output_basis import covariance_mse
ROOT=Path(__file__).resolve().parents[1]


def test_T16B_historical_snapshot():
    from scripts.run_heldout_basis import snapshot
    r=snapshot();assert r["all_checks_passed"]
    subprocess.check_call(["git","merge-base","--is-ancestor",T16B_COMMIT,"HEAD"],cwd=ROOT)


def test_frozen_U32_load_never_calls_SVD_or_historical_construction(monkeypatch):
    from src.shared_output_basis import SharedOutputBasis
    from scripts import run_shared_output_basis
    def forbidden(*args,**kwargs):raise AssertionError("formal basis refit/SVD forbidden")
    monkeypatch.setattr(np.linalg,"svd",forbidden)
    monkeypatch.setattr(SharedOutputBasis,"build",forbidden)
    monkeypatch.setattr(SharedOutputBasis,"for_shift",forbidden)
    monkeypatch.setattr(run_shared_output_basis,"construct",forbidden)
    bases,proof=load_bases(ROOT)
    assert proof["historical_u32_hash_exact_match"] and proof["t16c_basis_loaded_from_frozen_artifact"]
    assert not proof["T16C_data_accessed"] and proof["formal_basis_construction_SVD_calls"]==0
    for scope,b in bases.items():
        assert b.u32.dtype==np.complex128 and not b.u32.flags.writeable
        for q in RANKS:
            assert np.shares_memory(b.output(q),b.u32)
            assert array_hash(b.output(q))==proof["basis_artifacts"][scope]["Uq_sha256"][str(q)]
            np.testing.assert_allclose(b.output(q).conj().T@b.output(q),np.eye(q),rtol=1e-11,atol=1e-11)
        assert all(n not in HELDOUT for n,_ in proof["basis_artifacts"][scope]["condition_ids"])
        assert (ROOT/proof["basis_artifacts"][scope]["path"]).stat().st_mode & 0o222 == 0
        with pytest.raises(ValueError):b.u32[0,0]=0


def test_frozen_basis_tamper_is_rejected(tmp_path):
    out=tmp_path/"outputs";out.mkdir()
    for name in ("t16b_basis_provenance_recovery.json","t16b_shared_basis_results.json"):
        shutil.copyfile(ROOT/"outputs"/name,out/name)
    r=json.loads((out/"t16b_basis_provenance_recovery.json").read_text(encoding="utf8"))
    first=next(iter(r["basis_artifacts"].values()))
    (tmp_path/first["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError,match="file hash"):load_bases(tmp_path)


def test_only_two_fixed_heldout_definitions_and_original_geometry():
    assert [(v["profile"],v["speed_kmh"]) for v in HELDOUT.values()]==[("C",75.),("B",120.)]
    for n in HELDOUT:
        d=definition(n)
        assert d["system"]["nominal_delay_spread_seconds"]==100e-9
        assert d["system"]["sample_rate_hz"]==64*30000 and d["system"]["carrier_frequency_hz"]==3.5e9
        assert d["pilot_ratio"]==1/32 and not d["ofdm_channel_normalize"]
        assert d["array"]["ut_orientation_radians"]==[float(np.pi),0.,0.]


def test_new_namespace_actual_scope_and_train_validation_separation():
    from scripts.run_heldout_basis import namespace_audit
    a=namespace_audit();assert a["channel_seed_tuples"]==1200 and a["frame_communication_tuples"]==12000
    rows=manifest_rows();assert len(rows)==12000
    for n in HELDOUT:
        a=seeds(n,"train_dev",0,0);b=seeds(n,"train_dev",0,1);c=seeds(n,"validation_dev",500,0)
        assert all(a[k]==b[k] for k in ("sionna_seed","torch_seed","numpy_seed"))
        assert all(a[k]!=b[k] for k in ("bits_seed_tx0","bits_seed_tx1","noise_seed"))
        assert a!=c
    for args in (("CDL-A","train_dev",0,0),("CDL-C-75","formal_test",1000,0),
        ("CDL-C-75","train_dev",500,0),("CDL-C-75","validation_dev",0,0),("CDL-C-75","train_dev",0,10)):
        with pytest.raises(ValueError):seeds(*args)


def test_source_noise_and_source_statistics_exact():
    from scripts.run_heldout_basis import policy
    source,prov,power,var=policy()
    assert power==9.941134822523972
    assert var=={5:3.1436628565989504,10:.9941134822523973,20:.09941134822523973}
    assert hashlib.sha256((ROOT/prov["path"]).read_bytes()).hexdigest()==prov["sha256"]
    assert source.train_trajectory_ids==tuple(range(500)) and source.samples==5000


def test_historical_H_table_actual_schemas():
    from scripts.run_heldout_basis import historical_channel_hashes
    hashes=historical_channel_hashes()
    assert len(hashes)==1800 and all(len(h)==64 for h in hashes)


def test_target_statistics_boundary_rejects_validation_before_mutation():
    stream=StreamingStatistics(np.array([0,1]),np.array([False,False,True,True]),dimension=4)
    with pytest.raises(ValueError):stream.update(np.ones((10,4)),[500])
    assert stream.n==0 and stream.ids==[]
    x=np.array([[1+1j,2,3-2j,1],[2-1j,3,2j,2],[1,1j,2,3]],complex)
    stream.update(x,[0]);r=stream.freeze();centered=x-x.mean(axis=0)
    np.testing.assert_allclose(r.rpp,centered[:,:2].T@centered[:,:2].conj()/3,atol=1e-12)
    np.testing.assert_allclose(r.rdp,centered[:,2:].T@centered[:,:2].conj()/3,atol=1e-12)
    assert r.train_trajectory_ids==(0,)


@pytest.mark.parametrize("q",RANKS)
def test_complex_shared_projection_and_oracle_T16A_math(q):
    rng=np.random.default_rng(20261631)
    a=rng.normal(size=(52,52))+1j*rng.normal(size=(52,52));joint=a@a.conj().T+np.eye(52)
    pp=joint[:12,:12];dp=joint[12:,:12];dd=joint[12:,12:];sigma=.7
    target=FrozenStatistics(np.zeros(12),np.zeros(40),pp,dp,(),1).prepare(sigma).weight
    source=rng.normal(size=(40,12))+1j*rng.normal(size=(40,12))
    u,_=np.linalg.qr(rng.normal(size=(40,32))+1j*rng.normal(size=(40,32)))
    u.flags.writeable=False;b=FrozenOutputBasis(u,"synthetic")
    c=WeightedCorrection.build(source,target,pp+sigma*np.eye(12));delta=target-source
    g=b.coefficient_map(delta,q);m=delta@c.lower;approx=source+b.output(q)@g
    np.testing.assert_allclose(covariance_mse(approx,pp,dp,np.trace(dd).real,sigma),
        covariance_mse(target,pp,dp,np.trace(dd).real,sigma)+b.energy(m,q)["weighted_excess_MSE_per_frame"],rtol=1e-11,atol=1e-11)
    x=rng.normal(size=12)+1j*rng.normal(size=12)
    np.testing.assert_allclose(b.estimate(x,source,np.zeros(12),np.zeros(40),q,g),x@approx.T,rtol=1e-11,atol=1e-11)
    for rank in (4,8):
        aa,qq=c.factors(rank)
        np.testing.assert_allclose(qq@c.lower,c.right_h[:rank],rtol=1e-11,atol=1e-11)
        assert float(np.sum(c.singular[rank:]**2))==pytest.approx(np.linalg.norm((delta-aa@qq)@c.lower)**2,rel=1e-11)


def test_closure_keeps_small_nonpositive_gap_and_zero_undefined():
    assert gap_closure(-10.,-20.,-18.)["closure"]==.8
    assert gap_closure(-10.,-9.,-8.)["closure"]==2.
    assert gap_closure(-10.,-10.,-11.)["closure"] is None
    assert abs(gap_closure(-10.,-10.-1e-8,-11.)["closure"])>1e7


@pytest.mark.parametrize("values,expected",[((.7,.81,.9,.95),"strong"),((.5,.7,.81,.95),"moderate"),((.4,.5,.6,.79),"weak"),((.4,.5,.79,.85),"none")])
def test_preregistered_scientific_gates(values,expected):
    rows=[dict(shift=n,snr_db=s,rank=q,closure=v,metric_type="actual_validation") for n in HELDOUT for s in (10,20) for q,v in zip(RANKS,values)]
    oracle=[dict(shift=n,snr_db=s,rank=4,closure=.91) for n in HELDOUT for s in (10,20)]
    r=gate(rows,oracle);assert r["heldout_correction_intrinsically_lowrank"]
    if expected=="none":assert not any(r[k] for k in r if k.startswith("heldout_shared_basis_"))
    else:assert r[f"heldout_shared_basis_{expected}"]
    group=[x for x in rows if x["shift"]==next(iter(HELDOUT)) and x["snr_db"]==10]
    assert qthreshold(group,.8)==next((q for q,v in zip(RANKS,values) if v>=.8),None)
    with pytest.raises(ValueError):qthreshold([dict(rank=12,closure=.99,metric_type="actual_validation")],.8)


def test_formal_script_no_detector_or_historical_basis_constructor():
    from src.heldout_basis import HeldoutCDL
    import inspect
    source=(ROOT/"scripts/run_heldout_basis.py").read_text(encoding="utf8")
    assert all(s not in source for s in ("detect_data_zf","demodulate_data_bits","run_shared_output_basis","SharedOutputBasis.build"))
    assert "normalize=False" in inspect.getsource(HeldoutCDL.generate)
    assert "normalize_delays=True" in inspect.getsource(HeldoutCDL.__init__)
    tree=ast.parse(source)
    assert not any(isinstance(n,ast.Attribute) and n.attr=="svd" for n in ast.walk(tree))


def test_figure_preflight_all_QA_and_CSV_rebuild(tmp_path):
    import matplotlib
    from scripts.plot_heldout_basis import audit,FILES
    from scripts.run_cdl_mismatch import csvsave
    before=dict(matplotlib.rcParams)
    shared=[dict(shift=n,snr_db=s,rank=q,closure=.3+q/40) for n in HELDOUT for s in (5,10,20) for q in RANKS]
    oracle=[dict(shift=n,snr_db=s,rank=r,closure=.95) for n in HELDOUT for s in (5,10,20) for r in (4,8)]
    for filename,rows in zip(FILES,(shared,oracle)):csvsave(tmp_path/filename,rows)
    audit(tmp_path,tmp_path/"images")
    assert dict(matplotlib.rcParams)==before


def test_actual_results_if_present():
    p=ROOT/"outputs/t16c_heldout_results.json"
    if not p.exists():return
    r=json.loads(p.read_text(encoding="utf8"))
    assert r["formal_test_sealed"] and not r["formal_test_used"] and not r["new_model_trained"]
    assert r["BER_evaluations"]==r["detector_calls"]==r["formal_basis_construction_SVD_calls"]==0
    assert not r["shared_basis_refitted"] and not r["absolute_theoretical_expected_nmse_available"]
    assert len(r["shared_validation_nmse"])==24 and len(r["condition_specific_lowrank"])==12 and len(r["shared_capture"])==60
    for info in r["target_statistics_provenance"].values():
        assert info["train_only"] and not info["validation_used_for_fitting"] and info["samples"]==5000
        with np.load(ROOT/info["path"]) as f:
            assert np.array_equal(f["train_trajectory_ids"],np.arange(500)) and "rdd" not in f.files
            for k in ("mean_p","mean_d","rpp","rdp"):assert array_hash(f[k])==info["component_sha256"][k]
    for p,h in {**r["source_sha256"],**r["frozen_history_sha256"]}.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    load_bases(ROOT)
    with (ROOT/"outputs/t16c_heldout_trajectory_metrics.csv").open(encoding="utf8") as f:rows=list(csv.DictReader(f))
    assert len(rows)==5600 and "ber" not in rows[0] and "bit_errors" not in rows[0]
    for a in r["shared_validation_nmse"]:
        v=[x for x in rows if x["shift"]==a["shift"] and int(x["snr_db"])==a["snr_db"] and x["method"]==a["method"]]
        assert len(v)==100 and {int(x["trajectory_id"]) for x in v}==set(range(500,600))
        assert a["nmse_linear"]==pytest.approx(sum(float(x["error_energy"]) for x in v)/sum(float(x["truth_energy"]) for x in v),rel=1e-14)
    for n in HELDOUT:
        for s in (5,10,20):
            group=[x for x in r["shared_validation_nmse"] if x["shift"]==n and x["snr_db"]==s]
            assert r["q80"][n][str(s)]==qthreshold(group,.8) and r["q90"][n][str(s)]==qthreshold(group,.9)
    assert gate(r["shared_validation_nmse"],r["condition_specific_lowrank"])=={k:r[k] for k in gate(r["shared_validation_nmse"],r["condition_specific_lowrank"])}
