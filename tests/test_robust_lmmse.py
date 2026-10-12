"""Independent HR math gates, tiny exact-link replay, frozen artifacts."""
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src import robust_lmmse as hr
from src.b0_development import replay_link
from src.lmmse_train import raw_array_sha256,sha256_file
from src.tdl import tdl_a_profile
from src.trajectories import prepare_trajectory

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"


def test_T13A_committed_and_historical_report_accepted():
    subject=subprocess.check_output(["git","show","-s","--format=%s",hr.T13A_COMMIT],cwd=ROOT,text=True).strip()
    assert subject=="feat: add low-pilot OOD statistics-mismatch scan"
    blob=subprocess.check_output(["git","show",hr.T13A_COMMIT+":outputs/t13a_ood_mismatch_results.json"],cwd=ROOT)
    assert json.loads(blob)["all_checks_passed"]
    # Git autocrlf stores LF; separately freeze raw working bytes before/after scan.
    assert blob==(OUT/"t13a_ood_mismatch_results.json").read_bytes().replace(b"\r\n",b"\n")


def test_tau_envelope_uses_maximum_physical_tap_not_RMS():
    e=hr.envelope(); delays,_=tdl_a_profile(300)
    assert e["normalized_max_delay"]==9.6586
    assert e["tau_max_seconds"]==max(delays)
    assert e["tau_max_ns"]==pytest.approx(2897.58,abs=1e-12)
    assert hr.TAU_BOUND>300e-9 and hr.FD_BOUND==389.158111
    assert e["bound_rounding_difference_hz"]>0 and not e["condition_label_used"]


@pytest.fixture(scope="module")
def prior():
    return hr.build_prior()


@pytest.mark.parametrize("key",["rf_validation","rt_validation"])
def test_analytic_quadrature_Hermitian_PSD_unit_diagonal(prior,key):
    _,r=prior; v=r[key]
    assert v["all_checks_passed"] and v["quadrature_max_error"]<1e-12
    assert v["minimum_eigenvalue"]>=-1e-12
    assert v["Hermitian_max_error"]<=1e-12 and v["diagonal_max_error"]<=1e-12
    assert not v["eigenvalues_clipped"] and v["distinct_lags_checked"]>=31
    assert all(p["absolute_error"]<1e-12 for p in v["lag_correlations"])


def test_frequency_phase_sign_independent_delay_ramp_and_physical_order(prior):
    f=hr.centered_frequency_grid_hz(64,30000)
    assert f[0]==-960000 and f[-1]==930000 and np.all(np.diff(f)==30000)
    rf=hr.frequency_covariance(f)
    # Composite midpoint integration, independent of adaptive quadrature and sinc.
    u=(np.arange(20000)+.5)/20000
    expected=np.mean(np.exp(-2j*np.pi*30000*hr.TAU_BOUND*u))
    np.testing.assert_allclose(rf[1,0],expected,atol=1e-10,rtol=0)
    assert rf[1,0].imag<0 and rf[0,1].imag>0


def test_time_is_rectangular_Doppler_not_Jakes(prior):
    from scipy.special import j0
    times=np.arange(16)*(80/1920000)
    rt=hr.time_covariance(times)
    lag=times[15]-times[0]
    assert rt[15,0]==pytest.approx(np.sinc(2*hr.FD_BOUND*lag),abs=1e-15)
    assert abs(rt[15,0]-j0(2*np.pi*hr.FD_BOUND*lag))>.05


@pytest.mark.parametrize("func,coordinates,bound",[(hr.frequency_covariance,[0.,1.],0),(hr.frequency_covariance,[0.,1.],np.nan),
    (hr.time_covariance,[0.,1.],-1),(hr.time_covariance,[0.,np.nan],1)])
def test_invalid_support_and_coordinates_raise(func,coordinates,bound):
    with pytest.raises(ValueError):
        func(coordinates,bound)


def test_Kronecker_ordering_via_independent_tensor_contraction(prior):
    full,r=prior
    rf=hr.frequency_covariance(hr.centered_frequency_grid_hz(64,30000))
    rt=hr.time_covariance(np.arange(16)*(80/1920000))
    # A deterministic nonseparable index sentinel makes swapped axes visible.
    v=(np.arange(1024).reshape(64,16)+1j*np.arange(1024,0,-1).reshape(64,16))/1024
    tensor=np.einsum("kl,tu,lu->kt",rf,rt,v)
    np.testing.assert_allclose((full@v.ravel(order="C")).reshape(64,16),tensor,atol=1e-12,rtol=1e-12)
    assert r["ordering_sentinel_max_error"]==0
    assert full.dtype==np.complex128 and not full.flags.writeable


@pytest.fixture(scope="module")
def pool(prior):
    with patch.object(hr.lmmse,"cho_factor",wraps=hr.lmmse.cho_factor) as factor:
        result=hr.prepare_pool(prior[0])
    assert factor.call_count==6
    return result


def test_condition_blind_same_covariance_hash_and_six_reused_Cholesky(prior,pool):
    import inspect
    assert list(inspect.signature(hr.prepare_pool).parameters)==["full"]
    assert len(pool)==3 and sum(len(pair) for pair in pool.values())==6
    assert len({raw_array_sha256(prior[0]) for _ in hr.CONDITIONS})==1
    for snr,pair in pool.items():
        for q,p in enumerate(pair):
            assert p.pilot_indices.size==16 and p.noise_variance==10**(-snr/10)
            np.testing.assert_array_equal(p.r_hp,prior[0][:,p.pilot_indices])
            np.testing.assert_allclose(p.observation_covariance,p.r_pp+p.noise_variance*np.eye(16),rtol=0,atol=1e-15)
            assert p.diagnostics["cholesky_success"] and not p.diagnostics["jitter_used"]
            assert not p.diagnostics["posterior_clipped"]
    assert hr.count_layout_resources(hr.LAYOUT)["union_ratio"]==1/32


@pytest.fixture(scope="module")
def refs():
    return hr.load_replay_references(OUT)


@pytest.mark.parametrize("condition,tid",[("ID",2000),("speed120",0),("DS300",0)])
def test_small_actual_exact_replay_only_R6_and_no_old_estimators(condition,tid,pool,refs):
    model=prepare_trajectory(frames_per_trajectory=10) if condition=="ID" else hr.ood.prepare_channel(condition)
    timing={k:0. for k in ("channel_replay_seconds","Pilot_LS_seconds","R6_inference_seconds","ZF_seconds")}
    with patch("src.link.estimate_ls_linear_channel",side_effect=AssertionError("old R3 run")), \
         patch("src.ood_mismatch.load_frozen_estimators",side_effect=AssertionError("R4/B0 loaded")), \
         patch("src.ood_mismatch.evaluate_trajectory",side_effect=AssertionError("old OOD estimators")):
        rows,r=hr.evaluate_trajectory(condition,tid,model,pool,refs,replay_link(ROOT),timing)
    frames=10 if condition=="ID" else 5
    assert len(rows)==3 and len(r["frame_realizations"])==frames
    assert all(a["total_bits"]==3968*frames and a["exact_replay_verified"] for a in rows)
    assert r["historical_link_X_Y_exact_checks"]==frames*3
    assert all(np.isfinite(a["R6_HR_error_energy"]) and a["R6_HR_bit_errors"]>=0 for a in rows)
    assert all(set(f["Y_by_snr"])=={"5","10","20"} for f in r["frame_realizations"])
    if condition!="ID":
        expected=refs["OOD"][condition,tid]
        assert all(f["Y_by_snr"]==e["Y_by_snr"] and f["shared_pilot_LS_by_snr"]==e["shared_pilot_LS_by_snr"] for f,e in zip(r["frame_realizations"],expected["frame_realizations"]))


def test_replay_rejects_saved_hash_mismatch_before_inference(pool,refs):
    import copy
    wrong=copy.deepcopy(refs)
    wrong["OOD"]["speed120",0]["frame_realizations"][0]["unit_noise_sha256"]="bad"
    timing={k:0. for k in ("channel_replay_seconds","Pilot_LS_seconds","R6_inference_seconds","ZF_seconds")}
    with patch.object(hr.lmmse,"estimate_mimo_from_pilot_ls",side_effect=AssertionError("inference before replay gate")):
        with pytest.raises(ValueError,match="exact replay mismatch"):
            hr.evaluate_trajectory("speed120",0,hr.ood.prepare_channel("speed120"),pool,wrong,replay_link(ROOT),timing)


@pytest.mark.parametrize("r4,r6,r5,expected",[(-10,-12,-14,.5),(-10,-9,-14,-.25),(-10,-16,-14,1.5)])
def test_gap_formula_without_clipping(r4,r6,r5,expected):
    r=hr.closure(r4,r6,r5)
    assert r["robust_closure"]==expected and r["R6_recovery_db"]==r4-r6
    assert r["R4_R5_gap_db"]==r4-r5


@pytest.mark.parametrize("values,key",[((.5,.5),"fixed_robust_baseline_materially_closes_gap"),
    ((-.2,.249),"fixed_robust_prior_insufficient"),((.25,.25),"mixed_result"),((.8,.1),"mixed_result")])
def test_preregistered_decisions(values,key):
    rows=[{"condition":c,"snr_db":20,"robust_closure":v} for c,v in zip(hr.ood.SHIFTS,values)]
    flags=hr.decision_flags(rows)
    assert sum(flags.values())==1 and flags[key]


def test_feasibility_readonly_no_implementation_and_alias_disclosure():
    records=hr.feasibility()
    mr=records["mr_sdp_feasibility"]; ccf=records["ccf_selection_feasibility"]
    assert mr["regular_per_TX_grid_applicable"] and mr["delay_support_exceeds_pilot_alias_period"]
    assert not mr["dependencies_installed"] and not mr["numerical_solver_run"]
    assert ccf["literature_itself_uses_leave_one_out"] and not ccf["direct_1D_block_pilot_reproduction"]
    assert not ccf["selection_implemented"] and not ccf["extension_validated"]


def test_CSV_only_figure_preflight_before_full_scan(tmp_path):
    from scripts.run_robust_lmmse import render_and_audit
    from scripts.run_lmmse_train_validation import save_csv
    rows=[{"condition":c,"snr_db":s,"R4_nmse_db":-s,"R6_HR_nmse_db":-s+1,
        "R5_nmse_db":-s-1,"robust_closure":-.5} for c in hr.CONDITIONS for s in hr.SNRS]
    save_csv(tmp_path/f"{hr.PREFIX}.csv",rows)
    with patch.object(hr,"evaluate_trajectory",side_effect=AssertionError("numerical scan")):
        records=render_and_audit(tmp_path,tmp_path)
    assert len(records)==2 and records[0]["alignment_audit"]["verdict"]=="PASS"
    assert all(r["collision_audit"]["verdict"]=="PASS" for r in records)


@pytest.fixture(scope="module")
def report():
    p=OUT/f"{hr.PREFIX}_results.json"
    if not p.exists():
        pytest.skip("full R6-only scan not yet run")
    return json.loads(p.read_text(encoding="utf8"))


def test_all_2400_trajectories_and_exact_replay_provenance(report):
    assert report["trajectory_rows"]==2400 and len(report["realization_hashes"])==800
    records=report["realization_hashes"]
    assert {r["trajectory_id"] for r in records if r["condition"]=="ID"}==set(range(2000,2400))
    for c in hr.ood.SHIFTS:
        assert {r["trajectory_id"] for r in records if r["condition"]==c}==set(range(200))
    assert report["replay_verification"]["all_checks_passed"]
    assert sum(r["historical_link_X_Y_exact_checks"] for r in records)==18000
    assert len(set(report["covariance_hash_by_condition"].values()))==1
    assert report["factorization_count"]==6 and not report["old_estimators_rerun"]


def test_historical_files_untouched_and_formal_test_sealed(report):
    from scripts.run_robust_lmmse import accepted_files
    assert report["provenance"]["accepted_before"]==report["provenance"]["accepted_after"]==accepted_files()
    assert all(sha256_file(ROOT/p)==h for p,h in report["provenance"]["source_sha256"].items())
    assert report["formal_test_sealed"] and not report["formal_test_used"] and not report["new_model_trained"]
    assert not report["MMSE_detector_used"] and not report["old_model_loaded"]
    assert not report["numerical_failures"] and report["num_filtered_trajectories"]==0


def test_all_numeric_results_reaggregate_from_trajectory_CSV_and_frozen_refs(report):
    rows=hr.ood.read_csv(OUT/f"{hr.PREFIX}_trajectory_metrics.csv")
    assert len(rows)==2400
    types={"trajectory_id":int,"snr_db":int,"frames_aggregated":int,"nmse_truth_energy":float,
        "total_bits":int,"R6_HR_error_energy":float,"R6_HR_bit_errors":int}
    for r in rows:
        for k,convert in types.items():
            r[k]=convert(r[k])
        r["exact_replay_verified"]=r["exact_replay_verified"]=="True"
    expected=hr.aggregate(rows,hr.frozen_performance(OUT))
    assert expected==report["id_results"]+report["ood_results"]
    assert all(report[k]==v for k,v in hr.decision_flags(expected).items())
    for suffix,h in report["provenance"]["CSV_sha256"].items():
        assert sha256_file(OUT/f"{hr.PREFIX}{suffix}.csv")==h
    with pytest.raises(ValueError,match="complete frozen"):
        hr.aggregate(rows[:-1],hr.frozen_performance(OUT))
