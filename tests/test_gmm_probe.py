"""Small mathematical/provenance gates before the T15B integration run."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from src.gmm_probe import (probe_indices,selection_matrix,pilot_unit_noise,gaussian_statistics,
                           gaussian_reference,choose_training_seed,paired_diagnostics,scientific_flags,
                           EM_SEEDS,EM_PARAMETERS,KS)
from src.spatial_information import observation_geometry
ROOT=Path(__file__).resolve().parents[1]


def test_accepted_fix_historical_snapshot_provenance():
    commit="d923830a4f90a9b0b85d2f4f11edbb495811168e"
    assert subprocess.check_output(["git","show","-s","--format=%s",commit],text=True).strip()=="feat: integrate CDL-C benchmark with corrected receive-SNR semantics"
    assert "outputs/t15a_cdl_results.json" in subprocess.check_output(["git","ls-tree","-r","--name-only",commit],text=True)
    subprocess.check_call(["git","merge-base","--is-ancestor",commit,"HEAD"])


def test_probe_exact_pilots_stratified_data_and_dimensions():
    p,d=probe_indices();g=observation_geometry()
    np.testing.assert_array_equal(p,g.selected)
    assert len(p)==len(d)==64 and set(p).isdisjoint(d) and g.data[d].all()
    coords=np.array(np.unravel_index(d,(2,2,64,16))).T
    assert set(coords[:,2])=={7,23,39,55} and set(coords[:,3])=={1,5,9,13}
    assert np.all(np.bincount(coords[:,0]*2+coords[:,1])==16)


def test_observation_selects_only_first_64():
    h=np.arange(128)+1j*np.arange(128)[::-1]
    np.testing.assert_array_equal(selection_matrix()@h,h[:64])


def test_unit_noise_uses_original_pilot_phase_and_order():
    from src.link import PILOT_SYMBOLS
    p,_=probe_indices();rx,tx,k,t=np.unravel_index(p,(2,2,64,16))
    n=np.arange(2048).reshape(2,64,16)+1j
    np.testing.assert_array_equal(pilot_unit_noise(n),n[rx,k,t]/PILOT_SYMBOLS[tx])
    np.testing.assert_allclose(abs(pilot_unit_noise(n))**2,abs(n[rx,k,t])**2,rtol=1e-15)


@pytest.mark.parametrize("bad",[500,599,-1])
def test_validation_cannot_fit_probe_statistics(bad):
    with pytest.raises(ValueError,match="train_dev"):
        gaussian_statistics(np.ones((2,128),complex),[0,bad])


def test_population_complex_covariance_and_gaussian_reference():
    rng=np.random.default_rng(9);x=rng.normal(size=(140,128))+1j*rng.normal(size=(140,128))
    mean,cov=gaussian_statistics(x,np.arange(140))
    xc=x-x.mean(0)
    np.testing.assert_allclose(cov,xc.T@xc.conj()/len(x)+1e-6*np.eye(128),atol=1e-14)
    z=x[:3,:64]
    expected=np.array([mean+cov[:,:64]@np.linalg.solve(cov[:64,:64]+.5*np.eye(64),row-mean[:64]) for row in z])
    np.testing.assert_allclose(gaussian_reference(z,mean,cov,.5),expected,atol=1e-13)


def test_declared_capacity_seeds_and_package_defaults():
    assert KS==(1,2,4,8) and EM_SEEDS==(20261631,20261632,20261633)
    assert EM_PARAMETERS==dict(covariance_type="full",zero_mean=False,n_init=1,init_params="kmeans",max_iter=100,tol=.001,reg_covar=1e-6)


def test_seed_selection_uses_training_only_and_not_validation_nmse():
    records=[dict(seed=1,fit_succeeded=True,selection_split="train_dev",train_log_likelihood=2,validation_nmse=10),
             dict(seed=2,fit_succeeded=True,selection_split="train_dev",train_log_likelihood=1,validation_nmse=.001)]
    assert choose_training_seed(records)==1
    records[0]["selection_split"]="validation_dev"
    with pytest.raises(ValueError):choose_training_seed(records)


def test_paired_statistics_use_trajectory_energies():
    d=paired_diagnostics([1,4,2,3],[2,3,3,3])
    assert d["fraction_trajectories_improved"]==.5
    assert d["median_error_energy_difference"]==-.5


@pytest.mark.parametrize("gain,fraction,flags",[(.24,.9,(True,False,False)),(1.,.7,(False,True,False)),(.25,.9,(False,False,True)),(1.,.69,(False,False,True))])
def test_preregistered_interpretation(gain,fraction,flags):
    rows=[dict(K=k,snr_db=s,nonlinear_gain_db=gain,fraction_trajectories_improved=fraction) for k in (4,8) for s in (10,20)]
    actual=scientific_flags(rows)
    assert tuple(actual.values())==flags


def test_package_provenance_and_actual_formula_gate():
    r=json.loads((ROOT/"outputs/t15b_gmm_environment.json").read_text(encoding="utf8"))
    assert r["all_checks_passed"] and r["isolated_environment"]
    assert r["package_provenance"]["cplx-gmm"]["version"]=="0.2.0"
    assert r["package_provenance"]["gmm-estimator"]["version"]=="0.1.0"
    for package in r["package_provenance"].values():assert package["license"]=="BSD-3-Clause"
    p=subprocess.run([str(ROOT/".venv-gmm/Scripts/python.exe"),"-c","from scripts.check_gmm_environment import package_formula_gate; assert package_formula_gate()['all_checks_passed']"],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr


def test_generated_probe_and_corrected_noise_if_present():
    path=ROOT/"outputs/t15b_gmm_probe_dataset_manifest.json"
    if not path.exists():return
    r=json.loads(path.read_text(encoding="utf8"));assert r["all_checks_passed"]
    assert r["historical_six_h_hashes_matched"] and r["T15A_FIX_600_h_hashes_matched"]
    assert r["T15A_FIX_validation_1000_unit_noise_hashes_matched"]
    assert r["formal_test_sealed"] and not r["formal_test_used"]
    target=ROOT/r["dataset_path"]
    assert hashlib.sha256(target.read_bytes()).hexdigest()==r["dataset_sha256"]
    with np.load(target) as f:
        assert f["train"].shape==(5000,128) and f["validation"].shape==(1000,128)
        assert f["train"].dtype==np.complex128
        assert set(f["train_trajectory_ids"]).isdisjoint(f["validation_trajectory_ids"])
        fix=json.loads((ROOT/"outputs/t15a_cdl_results.json").read_text(encoding="utf8"))
        np.testing.assert_array_equal(f["sigma2"],[a["new_sigma2"] for a in fix["noise_variances"]])
        p,d=probe_indices();np.testing.assert_array_equal(f["pilot_indices"],p);np.testing.assert_array_equal(f["data_indices"],d)


def test_small_fitted_complex_k1_and_memory_likelihood_preflight():
    code='''
import numpy as np
from threadpoolctl import threadpool_limits
from gmm_estimator import GmmEstimator
from cplx_gmm._covariances import compute_precision_cholesky
from scripts.run_gmm_probe import k1_gate,MemorySampler,ram
from src.gmm_probe import EM_PARAMETERS,pilot_log_likelihood
rng=np.random.default_rng(20261601)
x=(rng.normal(size=(180,128))+1j*rng.normal(size=(180,128)))/np.sqrt(2)
with threadpool_limits(limits=1),MemorySampler() as memory:
 model=GmmEstimator(n_components=1,random_state=20261631,**EM_PARAMETERS).fit(x)
 mean,cov,gate=k1_gate(model,x,np.arange(180),[3.1436628565989504,.9941134822523973,.09941134822523973])
 assert gate['all_checks_passed'] and model.converged_
 z=x[:3,:64]
 marginal=GmmEstimator(n_components=1,covariance_type='full')
 marginal.n_features_in_=64;marginal._em_covariance_type='full'
 marginal.weights_=model.weights_;marginal.means_=model.means_[:,:64]
 marginal.covariances_=model.covariances_[:,:64,:64]+.7*np.eye(64)[None]
 marginal.precisions_cholesky_=compute_precision_cholesky(marginal.covariances_,'full')
 np.testing.assert_allclose(pilot_log_likelihood(z,model.weights_,model.means_,model.covariances_,.7),marginal.score_samples(z),atol=1e-12)
assert memory.peak>0 and ram()[1]>=ram()[0]
print(gate)
'''
    p=subprocess.run([str(ROOT/".venv-gmm/Scripts/python.exe"),"-c",code],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr


def test_generated_results_seed_selection_metrics_and_provenance_if_present():
    path=ROOT/"outputs/t15b_gmm_results.json"
    if not path.exists():return
    r=json.loads(path.read_text(encoding="utf8"))
    if not r["validation_nmse"]:return
    assert r["literature_baseline"] and not r["proposed_method"]
    assert r["formal_test_sealed"] and not r["formal_test_used"] and not r["new_neural_model_trained"]
    assert r["k1_regression"]["all_checks_passed"] and all(g["all_checks_passed"] for g in r["k1_regression"]["validation_prediction_regression"])
    assert not r["CDL_generated_in_GMM_environment"] and r["gmm_environment_unchanged"]
    for value in r["gmm_results_by_K"].values():
        assert len(value["runs"])==3
        if value["status"]=="fitted":assert value["selected_seed"]==choose_training_seed(value["runs"])
    import csv
    with (ROOT/"outputs/t15b_gmm_trajectory_metrics.csv").open(encoding="utf8") as f:rows=list(csv.DictReader(f))
    for aggregate in r["validation_nmse"]:
        subset=[x for x in rows if int(x["K"])==aggregate["K"] and int(x["snr_db"])==aggregate["snr_db"]]
        assert len(subset)==100 and set(int(x["trajectory_id"]) for x in subset)==set(range(500,600))
        np.testing.assert_allclose(aggregate["nmse_linear"],sum(float(x["error_energy"]) for x in subset)/sum(float(x["truth_energy"]) for x in subset),rtol=1e-14)
    # Accepted T15B result snapshot; HEAD can advance. Preserve every saved hash.
    commit="448092c93b6bad39c9862929c8c4820c178cf83e"
    for file,value in r["source_sha256"].items():
        historical=subprocess.check_output(["git","show",f"{commit}:{file}"])
        assert hashlib.sha256(historical).hexdigest()==value
        if file!="tests/test_gmm_probe.py":
            assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==value


def test_csv_only_figures_preflight_without_gmm_or_channel(tmp_path):
    import csv
    from scripts.plot_gmm_probe import audit
    input_dir=tmp_path/"input";input_dir.mkdir()
    with (input_dir/"t15b_gmm_probe_validation.csv").open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["K","snr_db","nmse_db","nonlinear_gain_db"]);writer.writeheader()
        for k in KS:
            for s in (5,10,20):writer.writerow(dict(K=k,snr_db=s,nmse_db=-s-.1*k,nonlinear_gain_db=.1*(k-1)))
    audit(input_dir,tmp_path/"figures")
    r=json.loads((tmp_path/"figures/t15b_gmm_figure_audit.json").read_text(encoding="utf8"))
    assert all(r["CSV_only_rebuild"].values())
    assert len(list((tmp_path/"figures").glob("*.png")))==2
