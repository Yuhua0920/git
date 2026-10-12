"""T15A small mathematics, fresh official mapping, and artifact/provenance tests."""
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from src.cdl_benchmark import (StreamingStatistics,marginal_kronecker_audit,
    seeds,manifest_rows,frames,T14B_COMMIT,observation_geometry,PREFIXES,SYSTEM)

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def official_probe():
    p=subprocess.run([str(ROOT/".venv-cdl/Scripts/python.exe"),str(ROOT/"scripts/check_cdl_adapter.py")],capture_output=True,text=True,encoding="utf8",errors="replace")
    assert p.returncode==0,p.stdout+p.stderr
    return json.loads((ROOT/"outputs/t15a_cdl_adapter_calibration.json").read_text(encoding="utf8"))


def test_official_mapping_repeat_and_continuity(official_probe):
    assert official_probe["all_checks_passed"]
    assert official_probe["frequency_exact"] and official_probe["slicing_exact"]
    assert official_probe["shape"]==[2,2,64,160]
    assert official_probe["dtype"]=="complex128"
    assert official_probe["provenance"]["profile"]=="CDL-C"
    assert official_probe["provenance"]["spec_version"]=="19.2"
    assert official_probe["array_separation_wavelengths"]==pytest.approx(.5,abs=1e-12)


def test_streaming_orientation_and_population_mean():
    rng=np.random.default_rng(42); x=rng.normal(size=(17,6))+1j*rng.normal(size=(17,6))
    x+=np.arange(6)*(1+2j); selected=np.array([1,4]); data=np.ones(6,bool); data[selected]=False
    s=StreamingStatistics(selected,data,6,full_audit=True)
    s.update(x[:5],[0]); s.update(x[5:],[1]); frozen=s.freeze()
    centered=x-x.mean(0); r=centered.T@centered.conj()/len(x)
    np.testing.assert_allclose(frozen.rpp,r[np.ix_(selected,selected)],atol=2e-15)
    np.testing.assert_allclose(frozen.rdp,r[np.ix_(np.flatnonzero(data),selected)],atol=2e-15)
    np.testing.assert_allclose(s.mfull/s.n,r,atol=2e-15)
    np.testing.assert_allclose(frozen.mean_d,x.mean(0)[data],atol=2e-15)
    assert not frozen.rpp.flags.writeable
    before=frozen.rpp.copy(); s.update(x[:1],[2]); assert np.array_equal(before,frozen.rpp)


def test_cholesky_matches_independent_linear_solve():
    rng=np.random.default_rng(5); x=rng.normal(size=(20,6))+1j*rng.normal(size=(20,6))
    selected=np.array([0,3]); data=np.ones(6,bool); data[selected]=False
    s=StreamingStatistics(selected,data,6); s.update(x,[0]); f=s.freeze()
    estimator=f.prepare(.1); z=np.array([2+1j,-3j])
    ref=f.mean_d+f.rdp@np.linalg.solve(f.rpp+.1*np.eye(2),z-f.mean_p)
    np.testing.assert_allclose(estimator.estimate(z),ref,atol=1e-14)
    assert estimator.solve_relative_residual<1e-14
    np.testing.assert_allclose(estimator.estimate(np.tile(z,(3,1))),np.tile(ref,(3,1)),atol=1e-14)


@pytest.mark.parametrize("sigma",[0,-1,np.nan,np.inf])
def test_cholesky_invalid_noise(sigma):
    s=StreamingStatistics([0],np.array([False,True]),2); s.update(np.ones((2,2)),[0])
    with pytest.raises(ValueError): s.freeze().prepare(sigma)


@pytest.mark.parametrize("ids",[[500],[-1],[599]])
def test_validation_cannot_fit(ids):
    s=StreamingStatistics([0],np.array([False,True]),2)
    with pytest.raises(ValueError): s.update(np.ones((2,2)),ids)


def test_duplicate_train_trajectory_rejected():
    s=StreamingStatistics([0],np.array([False,True]),2); s.update(np.ones((2,2)),[0])
    with pytest.raises(ValueError): s.update(np.ones((2,2)),[0])


def test_separable_metric_exact_and_nonseparable_control():
    rng=np.random.default_rng(7)
    def psd(n):
        a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); return a@a.conj().T
    r=np.kron(np.kron(psd(2),psd(3)),psd(2))
    assert marginal_kronecker_audit(r,(2,3,2))["relative_Frobenius_residual"]<1e-14
    z=psd(12)
    assert marginal_kronecker_audit(z,(2,3,2))["relative_Frobenius_residual"]>.1


def test_manifest_split_and_seeds():
    rows=manifest_rows(); assert len(rows)==6000
    train=[r for r in rows if r["split"]=="train_dev"]; val=[r for r in rows if r["split"]=="validation_dev"]
    assert len(train)==5000 and len(val)==1000
    assert set(r["trajectory_id"] for r in train).isdisjoint(r["trajectory_id"] for r in val)
    assert all(len({r[k] for k in seeds(0)})==6 for r in rows)
    for trajectory in (0,499,500,599):
        same=[r for r in rows if r["trajectory_id"]==trajectory]
        for role in ("sionna_seed","torch_seed","numpy_seed"):
            assert {r[role] for r in same}=={seeds(trajectory)[role]}
        assert len({r["noise_seed"] for r in same})==10
    assert seeds(0)==seeds(0) and seeds(0)!=seeds(500)
    assert PREFIXES==(100,250,500)
    assert SYSTEM["sample_rate_hz"]==SYSTEM["nfft"]*SYSTEM["delta_f_hz"]
    assert SYSTEM["symbol_duration_seconds"]==80/1920000


def test_pilot_geometry_exact():
    g=observation_geometry(); assert g.selected.size==64 and np.count_nonzero(g.data)==3968
    assert g.layout.offsets_by_tx==(0,8)
    assert np.count_nonzero(g.layout.pilot_union)/1024==1/32
    assert g.layout.pilot_symbol_indices==(0,4,8,12)


def test_isolated_pinned_environment_report():
    r=json.loads((ROOT/"outputs/t15a_cdl_environment.json").read_text(encoding="utf8"))
    assert r["all_checks_passed"] and r["sionna_version"]=="2.2.0"
    assert r["main_environment"]["all_checks_passed"]
    assert not r["forbidden_dependencies"]
    assert r["T14B_commit"]==T14B_COMMIT
    assert subprocess.check_output(["git","show","-s","--format=%s",T14B_COMMIT],text=True).strip()=="feat: add MIMO spatial-correlation information gate"


def test_results_if_generated():
    # This preflight is also useful before the full run. Final artifact checks
    # execute once results exist, with no skip markers or claimed absent data.
    p=ROOT/"outputs/t15a_cdl_results.json"
    if not p.exists(): return
    r=json.loads(p.read_text(encoding="utf8"))
    assert r["empirical_lmmse_stable"] is None
    assert r["empirical_lmmse_stability_assessment"]=="descriptive_only"
    assert r["formal_test_used"] is False and r["formal_test_sealed"] is True
    assert r["new_model_trained"] is False and r["proposed_method"] is False
    assert r["manifest"]["train_trajectories"]==500 and r["manifest"]["validation_trajectories"]==100
    assert r["validation_rows"]==1500
    assert len(r["validation_results"])==15
    assert r["historical_files_unchanged"]
    assert max(r["cholesky_solve_relative_residuals"])<1e-10


def test_frozen_snapshots_and_validation_aggregates_if_generated():
    p=ROOT/"outputs/t15a_cdl_results.json"
    if not p.exists(): return
    import csv
    r=json.loads(p.read_text(encoding="utf8"))
    for n in PREFIXES:
        with np.load(ROOT/f"outputs/t15a_cdl_statistics_{n}.npz") as f:
            assert np.array_equal(f["train_trajectory_ids"],np.arange(n))
            assert int(f["samples"])==10*n
            assert f["rpp"].shape==(64,64) and f["rdp"].shape==(3968,64)
            assert np.max(abs(f["rpp"]-f["rpp"].conj().T))<1e-10
    with (ROOT/"outputs/t15a_cdl_trajectory_metrics.csv").open(encoding="utf8") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==1500 and {int(a["trajectory_id"]) for a in rows}==set(range(500,600))
    for aggregate in r["validation_results"]:
        subset=[a for a in rows if a["method"]==aggregate["method"] and int(a["snr_db"])==aggregate["snr_db"]]
        assert len(subset)==100
        assert sum(int(a["total_bits"]) for a in subset)==3968000==aggregate["total_bits"]
        assert sum(int(a["bit_errors"]) for a in subset)==aggregate["bit_errors"]
        assert aggregate["ber"]==aggregate["bit_errors"]/aggregate["total_bits"]
        if aggregate["method"]!="R0-Perfect-CSI":
            assert aggregate["nmse_linear"]==sum(float(a["error_energy"]) for a in subset)/sum(float(a["truth_energy"]) for a in subset)


def test_actual_seed_manifest_and_frozen_history_if_generated():
    p=ROOT/"outputs/t15a_cdl_results.json"
    if not p.exists(): return
    import csv
    import hashlib
    r=json.loads(p.read_text(encoding="utf8"))
    with (ROOT/"outputs/t15a_cdl_manifest.csv").open(encoding="utf8") as f:
        rows=list(csv.DictReader(f))
    for actual,expected in zip(rows,manifest_rows(),strict=True):
        assert actual=={k:str(v) for k,v in expected.items()}
    for path,value in r["provenance"]["historical_before"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value
    assert r["calibration"]["nominal_RMS_delay_seconds"]==pytest.approx(100e-9,rel=1e-5)
