"""T13D analytic derivatives, complex information constants, physical units."""
import inspect
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src import statistics_information as info,statistics_information_scan as scan
from src import lmmse
from src.lmmse_train import raw_array_sha256,sha256_file

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"
torch.set_num_threads(4)


def test_T13C_committed_and_formal_sealed():
    subject=subprocess.check_output(["git","show","-s","--format=%s",scan.T13C_COMMIT],cwd=ROOT,text=True).strip()
    assert subject=="feat: add pilot-domain statistics identifiability gate"
    blob=subprocess.check_output(["git","show",scan.T13C_COMMIT+":outputs/t13c_statistics_identifiability_results.json"],cwd=ROOT)
    assert blob==(OUT/"t13c_statistics_identifiability_results.json").read_bytes().replace(b"\r\n",b"\n")
    r=json.loads(blob)
    assert r["all_checks_passed"] and r["formal_test_sealed"] and not r["formal_test_used"]


@pytest.mark.parametrize("dF",[16,8])
@pytest.mark.parametrize("window",[1,5])
def test_actual_TX_coordinates_and_joint_Kronecker_order(dF,window):
    a,b=info.geometry(dF,window,0),info.geometry(dF,window,1)
    assert not np.array_equal(a.flat_indices,b.flat_indices)
    assert a.pilot_ratio==(1/32 if dF==16 else 1/16)
    assert len(a.coordinates)==64//dF*4*window
    np.testing.assert_array_equal(a.flat_indices,a.coordinates[:,0]*16*window+a.coordinates[:,1])
    theta=info.theta_from_physical(30,100); c,dc=info.observation_covariance(theta,a,.1)
    rf,rt,_,_=info.factors(*theta,window)
    k=np.unique(a.coordinates[:,0]); t=np.unique(a.coordinates[:,1])
    direct=np.kron(rf[np.ix_(k,k)],rt[np.ix_(t,t)])+.1*np.eye(len(a.coordinates))
    np.testing.assert_array_equal(c,direct)
    np.testing.assert_array_equal(c,info.observation_covariance(theta,b,.1,derivatives=False))
    assert dc.shape==(2,*c.shape)
    if window==5:
        assert c[0,4]==pytest.approx(rt[0,16]) and abs(c[0,4])>.1


@pytest.mark.parametrize("point,speed,ds",info.POINTS)
def test_exact_old_channel_family_and_analytic_derivatives(point,speed,ds):
    fd,tau=info.theta_from_physical(speed,ds)
    rf,rt,drf,drt=info.factors(fd,tau,5)
    np.testing.assert_allclose(rf,lmmse.build_frequency_covariance(desired_rms_delay_spread_ns=ds),atol=1e-14,rtol=0)
    np.testing.assert_allclose(rt,lmmse.build_time_covariance(num_symbols=80,speed_kmh=speed),atol=1e-14,rtol=0)
    for a in (drf,drt):
        np.testing.assert_allclose(a,a.conj().T,atol=1e-10,rtol=1e-14)
        np.testing.assert_array_equal(np.diag(a),np.zeros(a.shape[0]))
    # Signed frequency phase derivative and J1 sign independent sentinels.
    assert drf[1,0].imag<0 and drt[1,0]<0
    assert info.RMS_SCALE_FACTOR==pytest.approx(1.0000579391604876,abs=1e-15)


def test_central_FD_multiple_steps_converge():
    r=scan.derivative_validation()
    assert len(r["records"])==48 and all(v["all_checks_passed"] for v in r["convergence"])
    assert max(v["relative_Frobenius_error"] for v in r["records"] if v["relative_step"]==1e-4)<2e-6


def test_proper_complex_FIM_no_half_and_real_augmented_equivalence():
    model=info.prepare_covariance(np.diag([2.,3.]))
    dc=np.array([np.diag([1.,0.]),np.diag([0.,1.])],dtype=complex)
    np.testing.assert_allclose(info.fisher_link(model,dc),np.diag([1/4,1/9]),atol=1e-15,rtol=0)
    theta=info.theta_from_physical(30,100); c,d=info.observation_covariance(theta,info.geometry(16,1,0),.1)
    d=d*theta[:,None,None]
    def augmented(a):
        return .5*np.block([[a.real,-a.imag],[a.imag,a.real]])
    solved=[np.linalg.solve(augmented(c),augmented(a)) for a in d]
    real=np.array([[.5*np.trace(a@b) for b in solved] for a in solved])
    np.testing.assert_allclose(info.fisher_link(info.prepare_covariance(c),d),real,atol=1e-11,rtol=1e-12)


def test_FIM_whitened_against_direct_trace_four_links_PSD():
    theta=info.theta_from_physical(30,100); values=[]
    for q in (0,1,0,1):
        c,d=info.observation_covariance(theta,info.geometry(16,5,q),.1); d*=theta[:,None,None]
        solved=[np.linalg.solve(c,a) for a in d]
        direct=np.array([[np.trace(a@b).real for b in solved] for a in solved])
        j=info.fisher_link(info.prepare_covariance(c),d)
        np.testing.assert_allclose(j,direct,atol=1e-9,rtol=1e-11); values.append(j)
    row,r=info.information_point("P0",30,100,16,5,10)
    np.testing.assert_allclose(r["four_link_FIM_sum"],np.sum(values,0),atol=1e-10,rtol=1e-12)
    assert row["four_link_FIM_equality_max_error"]<1e-10
    assert row["dimensionless_eigenvalue_min"]>0
    assert np.linalg.eigvalsh(r["four_link_FIM_sum"]).min()>0


def test_five_frames_FIM_not_five_independent_frame_shortcut():
    a,r1=info.information_point("P0",30,100,16,1,10)
    b,r5=info.information_point("P0",30,100,16,5,10)
    assert not np.allclose(np.array(r5["four_link_FIM_sum"]),5*np.array(r1["four_link_FIM_sum"]),rtol=.01,atol=0)
    assert b["speed_std_kmh"]<a["speed_std_kmh"] and b["delay_std_ns"]<a["delay_std_ns"]


def test_physical_unit_conversion_and_log_transform_solve():
    theta=info.theta_from_physical(30,100); j=np.array([[40.,3.],[3.,20.]])
    b,b_log,diag=info.crlb_from_log_fim(j,theta)
    np.testing.assert_allclose(b_log,np.linalg.solve(j,np.eye(2)),atol=1e-15,rtol=0)
    np.testing.assert_allclose(b,b_log*np.outer(theta,theta),atol=1e-20,rtol=1e-15)
    assert info.SPEED_PER_FD*theta[0]==pytest.approx(30,abs=1e-13)
    assert diag["parameter_error_correlation"]==-3/np.sqrt(40*20)
    with pytest.raises(info.SingularInformationError) as e:
        info.crlb_from_log_fim(np.diag([1.,1e-14]),theta)
    assert e.value.record["dimensionless_eigenvalues"]==[1e-14,1.] and not e.value.record["pinv_used"]


def test_KL_proper_complex_factor_nonnegative_identity():
    a=info.prepare_covariance(2*np.eye(3)); b=info.prepare_covariance(3*np.eye(3))
    expected=3*(np.log(1.5)+2/3-1)
    assert info.kl_link(a,b)==pytest.approx(expected,abs=1e-14)
    assert info.kl_link(a,a)==pytest.approx(0,abs=1e-14)
    total,links=info.kl_four_links((a,)*4,(b,)*4)
    assert total==pytest.approx(4*expected,abs=1e-14) and links==[info.kl_link(a,b)]*4


def test_MC_score_zero_mean_covariance_and_no_channel_simulation():
    source=inspect.getsource(info)+inspect.getsource(scan)
    assert "np.linalg.inv(" not in source and "pinv(" not in source
    assert not any(x in source for x in ("generate_master_qpsk","formal_trajectory","draw_channel","torch.nn"))
    record=scan.score_monte_carlo_audit()
    assert record["draws"]==10000 and record["all_checks_passed"]
    assert max(abs(v) for v in record["score_mean_in_analytic_standard_errors"])<5
    assert record["FIM_whitened_covariance_relative_error"]<.08


def test_prescribed_scientific_thresholds_and_distinct_reference_count():
    def rows(value=9.):
        return [{"point":p,"dF":16,"window_frames":5,"snr_db":s,
            "relative_speed_std_percent":value,"relative_delay_std_percent":value}
            for p,_,_ in info.POINTS for s in (10,20)]
    cases=rows(10.)
    assert scan.scientific_flags(cases,False)["continuous_statistics_locally_well_informed"]
    cases[0]["relative_delay_std_percent"]=10.0001
    assert scan.scientific_flags(cases,False)["partial_information_limit"]
    cases=rows(); cases[0]["relative_speed_std_percent"]=25; cases[1]["relative_speed_std_percent"]=25
    assert scan.scientific_flags(cases,False)["partial_information_limit"]
    cases[2]["relative_speed_std_percent"]=25
    assert scan.scientific_flags(cases,False)["continuous_statistics_information_limited"]
    assert scan.scientific_flags(rows(),True)["local_information_strong_but_global_ambiguity_present"]
    assert scan.scientific_flags(rows(),None)["scientific_decision_pending_global_review"]


def test_result_writer_serializes_numpy_diagnostic_scalars(tmp_path):
    from scripts.run_statistics_information import json_save
    path=tmp_path/"diagnostic.json"
    json_save(path,{"nonlocal_flag":np.bool_(True),"basin_count":np.int64(2),"KL":np.float64(.25)})
    assert json.loads(path.read_text(encoding="utf8"))=={"nonlocal_flag":True,"basin_count":2,"KL":.25}
    with pytest.raises(ValueError):
        json_save(path,{"KL":np.nan})


def test_T13C_covariance_provenance_and_directed_separation():
    from scripts.run_statistics_information import t13c_consistency
    grid=[]
    for p,v,ds in info.POINTS[:3]:
        for s in info.SNRS:
            g=info.geometry(16,5,0)
            a=info.prepare_covariance(info.observation_covariance(info.theta_from_physical(v,ds),g,10**(-s/10),derivatives=False))
            for q,v1,ds1 in info.POINTS[:3]:
                if q==p:
                    continue
                b=info.prepare_covariance(info.observation_covariance(info.theta_from_physical(v1,ds1),g,10**(-s/10),derivatives=False))
                grid.append({"reference_point":p,"window_frames":5,"snr_db":s,"candidate_speed":int(v1),"candidate_ds":int(ds1),
                    "KL_total":info.kl_four_links((a,)*4,(b,)*4)[0]})
    r=t13c_consistency(grid)
    assert r["all_checks_passed"] and len(r["covariance_regression"])==12
    assert r["minimum_directed_pairwise_KL"]>1e-8 and not r["closed_set_experiment_rerun"]


@pytest.fixture(scope="module")
def report():
    p=OUT/f"{scan.PREFIX}_results.json"
    if not p.exists():
        pytest.skip("full information scan not yet executed")
    return json.loads(p.read_text(encoding="utf8"))


def test_full_information_cases_counts_and_historical_immutability(report):
    from scripts.run_statistics_information import accepted_files
    assert len(report["crlb_results"])==48 and report["KL_grid_rows"]==21576
    assert report["formal_test_sealed"] and not report["formal_test_used"] and not report["new_model_trained"]
    assert not report["proposed_method"] and not report["parameter_estimator_implemented"] and not report["new_channel_simulation"]
    assert report["provenance"]["accepted_before"]==report["provenance"]["accepted_after"]==accepted_files()
    assert all(sha256_file(ROOT/p)==h for p,h in report["provenance"]["source_sha256"].items())
    assert all(sha256_file(OUT/p)==h for p,h in report["provenance"]["CSV_sha256"].items())


def test_results_reaggregate_from_raw_csv(report):
    from scripts.run_statistics_information import read_crlb,read_kl
    crlb=read_crlb(OUT/"t13d_statistics_crlb.csv"); kl=read_kl(OUT/"t13d_statistics_kl_grid.csv")
    assert crlb==report["crlb_results"]
    density,window=scan.comparisons(crlb)
    assert density==report["pilot_density_comparison"] and window==report["window_length_comparison"]
    summaries,minima=scan.global_summaries(kl)
    assert summaries==report["nonlocal_ambiguity"]["summaries"] and minima==report["nonlocal_ambiguity"]["grid_local_minima"]


def test_CSV_only_figure_preflight(tmp_path):
    from scripts.run_statistics_information import render_and_audit
    from scripts.run_lmmse_train_validation import save_csv
    crlb=[{"point":p,"dF":d,"window_frames":w,"snr_db":s,"relative_speed_std_percent":(30-s)/(w*d/8),"relative_delay_std_percent":(35-s)/(w*d/8)}
        for p,_,_ in info.POINTS for d in info.DFS for w in info.WINDOWS for s in info.SNRS]
    kl=[{"reference_point":p,"window_frames":w,"snr_db":s,"candidate_speed":v,"candidate_ds":ds,
        "KL_total":((v-v0)/20)**2+((ds-ds0)/50)**2,"nonlocal_flag":abs(v-v0)>=20 or abs(ds-ds0)>=50}
        for p,v0,ds0 in info.POINTS for w in info.WINDOWS for s in info.SNRS for v in scan.SPEED_GRID for ds in scan.DS_GRID]
    save_csv(tmp_path/"t13d_statistics_crlb.csv",crlb); save_csv(tmp_path/"t13d_statistics_kl_grid.csv",kl)
    with patch.object(scan,"kl_grid",side_effect=AssertionError("numeric scan")):
        records=render_and_audit(tmp_path,tmp_path)
    assert len(records)<=3 and all(r["collision_audit"]["verdict"]=="PASS" for r in records)
