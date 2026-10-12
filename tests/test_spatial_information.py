"""Independent mathematical sentinels before the T14B theory scan."""
import ast
import csv
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import cho_solve

from src import lmmse, spatial_information as info
from src.lmmse_train import sha256_file
from src.numerics import noise_variance_from_snr

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/"outputs"


@pytest.fixture(scope="module")
def geometry():
    return info.observation_geometry()


@pytest.fixture(scope="module")
def covariances():
    return lmmse.build_frequency_covariance(), lmmse.build_time_covariance()


@pytest.fixture(scope="module")
def moderate(geometry, covariances):
    rf, rt = covariances
    joint = info.JointCovariance(info.spatial_covariances(.6,.6)[3],rf,rt)
    return joint, info.prepare_posterior(joint,geometry.selected,.1)


def test_T14A_committed_exact_subject_and_accepted_sources():
    subject = subprocess.check_output(["git","show","-s","--format=%s",info.T14A_COMMIT],cwd=ROOT,text=True).strip()
    assert subject == "feat: add out-of-family PDP-shape mismatch gate"
    for p in ("src/tdl.py","src/doppler.py","src/lmmse.py","src/pilots.py","src/detectors.py","src/link.py"):
        historical = subprocess.check_output(["git","show",info.T14A_COMMIT+":"+p],cwd=ROOT)
        assert historical.replace(b"\r\n",b"\n") == (ROOT/p).read_bytes().replace(b"\r\n",b"\n")


@pytest.mark.parametrize("rho",info.RHOS)
def test_spatial_Hermitian_PSD_condition_and_zero_identity(rho):
    tx, rx, column, project = info.spatial_covariances(rho,rho)
    for m in (tx,rx,column,project):
        np.testing.assert_array_equal(m,m.conj().T)
        assert np.linalg.eigvalsh(m).min()>0
    np.testing.assert_allclose(np.linalg.eigvalsh(column),np.sort(np.outer([1-rho,1+rho],[1-rho,1+rho]).ravel()),atol=1e-15)
    if rho==0: np.testing.assert_array_equal(project,np.eye(4))
    if rho==.9: assert np.linalg.cond(column)==pytest.approx(361)


@pytest.mark.parametrize("bad",[-.1,1.,np.inf,.2j])
def test_invalid_or_singular_correlation_rejected(bad):
    with pytest.raises(ValueError): info.correlation(bad)


def test_complex_ordering_and_explicit_observation_sentinel(geometry):
    result = info.ordering_audit(geometry)
    assert result["all_checks_passed"] and result["complex_nonsymmetric_vec_covariance_max_error"]<1e-12
    # Real unequal correlations distinguish cross-RX from cross-TX.
    _,_,column,project = info.spatial_covariances(.3,.6)
    assert column[0,1]==.6 and column[0,2]==.3
    assert project[0,1]==.3 and project[0,2]==.6


def test_actual_pilot_offsets_selection_and_data_mask_exact(geometry):
    g = geometry
    assert g.layout.offsets_by_tx==(0,8) and g.layout.pilot_symbol_indices==(0,4,8,12)
    assert g.selector.shape==(64,4096) and g.selector.nnz==64
    np.testing.assert_array_equal((g.selector@g.selector.T).toarray(),np.eye(64))
    for rx in range(2):
        for tx,offset in enumerate((0,8)):
            selected = g.selected[g.selected//1024==2*rx+tx] % 1024
            expected = [k*16+t for k in range(offset,64,16) for t in (0,4,8,12)]
            np.testing.assert_array_equal(selected,expected)
            np.testing.assert_array_equal(g.data.reshape(2,2,64,16)[rx,tx],~g.layout.pilot_union)
    assert g.counts["union_ratio"]==1/32 and g.data.sum()==3968


def test_joint_index_mapping_against_explicit_pair_product(geometry,covariances):
    rf, rt = covariances
    tx,rx,_,sp = info.spatial_covariances(.3,.6)
    joint = info.JointCovariance(sp,rf,rt)
    coords = [(0,0,3,7),(1,0,6,9),(0,1,17,1),(1,1,28,12)]
    indices = np.array([((r*2+q)*64+k)*16+t for r,q,k,t in coords])
    actual = joint.entries(indices,indices)
    expected = np.array([[tx[bq,aq]*rx[ar,br]*rf[ak,bk]*rt[at,bt] for br,bq,bk,bt in coords] for ar,aq,ak,at in coords])
    np.testing.assert_array_equal(actual,expected)
    assert joint.psd_diagnostics()["Hermitian_PSD_passed"]
    np.testing.assert_allclose(joint.diagonal(),1,atol=1e-15)


def test_small_full_posterior_PSD_Loewner_and_independent_solve():
    sp = info.spatial_covariances(.3,.6)[3]
    rf = np.array([[1,.25+.15j],[.25-.15j,1]])
    rt = np.array([[1,.7],[.7,1]])
    joint = info.JointCovariance(sp,rf,rt); indices = np.array([0,3,5,10,13])
    prior = np.kron(sp,np.kron(rf,rt)); p = info.prepare_posterior(joint,indices,.1)
    np.testing.assert_array_equal(joint.entries(np.arange(16),np.arange(16)),prior)
    cross = prior[:,indices]; c = prior[np.ix_(indices,indices)]+.1*np.eye(5)
    full = prior-cross@np.linalg.solve(c,cross.conj().T)
    np.testing.assert_allclose(p.diagonal,np.diag(full).real,atol=1e-15)
    assert np.linalg.eigvalsh(full).min()>-1e-12 and np.linalg.eigvalsh(prior-full).min()>-1e-12
    assert p.diagnostics["latent_posterior_PSD_lower_bound"]>0
    z = np.array([[1+2j,3-1j,2j,4.,-2j]])
    np.testing.assert_allclose(p.estimate(z),(cross@np.linalg.solve(c,z.T)).T,atol=1e-15)


@pytest.mark.parametrize("snr",info.SNRS)
def test_rho0_R9_equals_four_R5_and_frozen_ID_theory(geometry,covariances,snr):
    rf,rt = covariances
    ref,objects = info.per_link_reference(rf,rt,geometry,snr)
    p = info.prepare_posterior(info.JointCovariance(np.eye(4),rf,rt),geometry.selected,noise_variance_from_snr(snr))
    m = info.metric(p,geometry.data)
    assert abs(m["nmse_db"]-ref["nmse_db"])<=1e-10
    for rx in range(2):
        for tx in range(2):
            np.testing.assert_allclose(p.diagonal[(2*rx+tx)*1024:(2*rx+tx+1)*1024],objects[tx].posterior_diagonal,atol=1e-15)
    h = json.loads((OUT/"t10a_lmmse_oracle_results.json").read_text(encoding="utf8"))
    old = next(r for r in h["theoretical_conditions"] if r["dF"]==16 and r["snr_db"]==snr)
    assert abs(ref["nmse_linear"]-old["posterior_nmse_data_combined"])<=1e-12


def test_single_point_spatial_generation_covariance_audit():
    result = info.spatial_generation_audit()
    assert result["samples"]==200000 and result["all_checks_passed"]
    assert result["relative_error_limit"]==.015


def test_joint_gaussian_MSE_agrees_with_posterior_theory(geometry,moderate):
    result = info.monte_carlo_audit(moderate[0],geometry,moderate[1])
    assert result["samples"]==10000 and result["all_checks_passed"]
    assert result["channel_seed"]!=result["noise_seed"] and result["relative_error_limit"]==.02
    assert not result["communication_experiment"]


def test_stable_solver_uses_no_inverse_pinv_and_estimator_only_observations(moderate):
    tree = ast.parse(inspect.getsource(info))
    prohibited = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ("inv","pinv")]
    assert not prohibited
    assert list(inspect.signature(info.Posterior.estimate).parameters)==["self","z"]
    assert all(not moderate[1].diagnostics[k] for k in ("inverse_used","pinv_used","jitter_used","posterior_clipped"))
    with pytest.raises(ValueError): moderate[1].estimate(np.ones((3,63)))


def test_metrics_trace_and_dB_gain_are_consistent(geometry,moderate):
    p = moderate[1]; m = info.metric(p,geometry.data)
    assert m["nmse_linear"]==m["posterior_trace"]/m["prior_trace"]
    assert m["nmse_db"]==10*np.log10(m["nmse_linear"])
    assert m["prior_trace"]==pytest.approx(3968)
    assert np.min(p.diagonal)>=-1e-12 and np.max(p.diagonal-p.prior)<=1e-12


def test_fixed_scientific_decision_boundaries():
    def rows(m,h):
        return [{"mode":"BOTH","rho_tx":rho,"snr_db":s,"spatial_gain_db":v} for rho,v in ((.6,m),(.9,h)) for s in info.SNRS]
    a = info.scientific_flags(rows(1,1.2)); assert a["moderate_spatial_correlation_material"]
    a = info.scientific_flags(rows(.49,.99)); assert a["spatial_information_gain_weak"]
    a = info.scientific_flags(rows(.5,1)); assert a["spatial_gain_material_only_at_high_correlation"] and a["final_paper_should_include_spatially_correlated_evaluation"]


def test_CSV_only_figure_preflight_and_JSON_metadata(tmp_path):
    from scripts.run_spatial_information import save_csv,render_and_audit,json_save
    rows = [{"mode":m,"rho":r,"snr_db":s,"spatial_gain_db":r*r*(s/20)} for m in info.MODES for r in info.RHOS for s in info.SNRS]
    save_csv(tmp_path/(info.PREFIX+".csv"),rows)
    figures = render_and_audit(tmp_path,tmp_path)
    assert len(figures)==1 and figures[0]["collision_audit"]["verdict"]=="PASS"
    json_save(tmp_path/"metadata.json",{"bool":np.bool_(True),"int":np.int64(3),"float":np.float64(.1),"figures":figures})
    assert json.loads((tmp_path/"metadata.json").read_text(encoding="utf8"))["bool"] is True


def test_script_preserves_historical_BLAS_execution_environment():
    environment = dict(os.environ); environment.pop("OPENBLAS_NUM_THREADS", None)
    output = subprocess.check_output([sys.executable,"-c",
        "import os; import scripts.run_spatial_information; print(repr(os.environ.get('OPENBLAS_NUM_THREADS')))"],
        cwd=ROOT,env=environment,text=True)
    assert output.strip()=="None"


@pytest.fixture(scope="module")
def report():
    p = OUT/(info.PREFIX+"_results.json")
    if not p.exists(): pytest.skip("theory scan artifacts not yet generated")
    return json.loads(p.read_text(encoding="utf8"))


def test_main_artifact_count_frozen_provenance_and_formal_sealed(report):
    from scripts.run_spatial_information import accepted_files
    assert len(report["theoretical_results"])==36 and len(report["spatial_eigenvalue_diagnostics"])==12
    assert report["formal_test_sealed"] and not report["formal_test_used"] and not report["new_model_trained"]
    assert not report["B0_loaded"] and not report["BER_computed"] and not report["production_correlated_generator_implemented"]
    assert report["provenance"]["accepted_before"]==report["provenance"]["accepted_after"]==accepted_files()
    assert all(sha256_file(ROOT/p)==h for p,h in report["provenance"]["source_sha256"].items())
    assert all(sha256_file(OUT/p)==h for p,h in report["provenance"]["output_sha256"].items())
    assert not report["numerical_failures"]


def test_saved_CSV_trace_dB_flags_and_hard_gates(report):
    with (OUT/(info.PREFIX+".csv")).open(encoding="utf8",newline="") as f: rows = list(csv.DictReader(f))
    assert len(rows)==36
    for row,expected in zip(rows,report["theoretical_results"]):
        assert row["mode"]==expected["mode"]
        for k in ("R5_nmse_linear","R9_nmse_linear","R5_nmse_db","R9_nmse_db","posterior_trace","prior_trace","spatial_gain_db"):
            assert float(row[k])==expected[k]
        assert float(row["R9_nmse_linear"])==float(row["posterior_trace"])/float(row["prior_trace"])
        assert float(row["spatial_gain_db"])==float(row["R5_nmse_db"])-float(row["R9_nmse_db"])
    assert all(report[k]==v for k,v in info.scientific_flags(report["theoretical_results"]).items())
    assert all(report[k]["all_checks_passed"] for k in ("monte_carlo_audit","spatial_generation_audit","rho0_regression","R5_historical_regression","ordering_validation"))
