"""T15B fixed author-package EM scan and complex conditional-mean evaluation.

The GMM process never calls CDL. It reads the frozen probe NPZ. No package
implementation is modified or replaced, and no validation quantity selects EM.
"""
import argparse
import csv
import ctypes
from ctypes import wintypes
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from gmm_estimator import GmmEstimator
from src.gmm_probe import (KS,EM_SEEDS,EM_PARAMETERS,REGRESSION_ATOL,REGRESSION_RTOL,
                           gaussian_statistics,gaussian_reference,selection_matrix,
                           choose_training_seed,paired_diagnostics,scientific_flags,pilot_log_likelihood)
from scripts.check_cdl_environment import environment_fingerprint

OUT=ROOT/"outputs"
EXPERIMENT="full-covariance complex-GMM nonlinear-prior value probe on a fixed sparse-pilot CDL-C channel subproblem"
SOURCES=("src/gmm_probe.py","scripts/check_gmm_environment.py","scripts/build_gmm_probe_dataset.py",
         "scripts/run_gmm_probe.py","scripts/plot_gmm_probe.py","tests/test_gmm_probe.py","docs/gmm_probe_protocol.md")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding="utf8"))
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf8")


def csv_save(path,rows):
    if not rows:return
    with Path(path).open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class MemoryCounters(ctypes.Structure):
    _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
        ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage",
         "QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]


def ram():
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL("psapi",use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
    result=MemoryCounters();result.cb=ctypes.sizeof(result)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(result),result.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(result.WorkingSetSize),int(result.PeakWorkingSetSize)


class MemorySampler:
    def __enter__(self):
        self.peak=ram()[0];self.stop=threading.Event()
        def sample():
            while not self.stop.wait(.02):self.peak=max(self.peak,ram()[0])
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.peak=max(self.peak,ram()[0]);self.stop.set();self.thread.join()


def check_package_bytes():
    report=load(OUT/"t15b_gmm_environment.json")
    for name,value in report["package_provenance"].items():
        d=md.distribution(name)
        if d.version!=value["version"]:raise ValueError("package version drift")
        for p,digest in value["source_and_metadata_sha256"].items():
            if sha(d.locate_file(p))!=digest:raise ValueError("author package source changed")
    return True


def fit_trial(train,ids,k,seed):
    if set(map(int,ids))!=set(range(500)) or train.shape!=(5000,128):
        raise ValueError("only all 500 original train trajectories may fit")
    model=GmmEstimator(n_components=k,random_state=seed,**EM_PARAMETERS)
    record=dict(K=k,seed=seed,selection_split="train_dev",fit_succeeded=False,selected=False,
                converged=False,iterations=None,lower_bound=None,train_log_likelihood=None,
                component_weights="",effective_sample_counts="",fit_seconds=0.,
                peak_sampled_ram_bytes=0,warning_messages="",error="",model_path="",model_sha256="")
    start=time.perf_counter()
    with MemorySampler() as memory,warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model.fit(train)
            score=float(model.score(train));resp=model.predict_proba(train)
            if not np.isfinite(score) or not np.isfinite(model.covariances_).all() or not np.iscomplexobj(model.covariances_):
                raise ValueError("non-finite/non-complex package model")
            np.linalg.cholesky(model.covariances_)
            target=OUT/f"t15b_gmm_model_K{k}_seed{seed}.npz"
            np.savez(target,weights=model.weights_,means=model.means_,covariances=model.covariances_,
                     precisions=model.precisions_,precisions_cholesky=model.precisions_cholesky_)
            record.update(fit_succeeded=True,converged=bool(model.converged_),iterations=int(model.n_iter_),
                          lower_bound=float(model.lower_bound_),train_log_likelihood=score,
                          component_weights=json.dumps(model.weights_.tolist()),
                          effective_sample_counts=json.dumps(resp.sum(0).tolist()),
                          model_path=str(target.relative_to(ROOT)).replace("\\","/"),model_sha256=sha(target))
        except Exception as exc:
            record["error"]=f"{type(exc).__name__}: {exc}";model=None
        record["warning_messages"]=json.dumps([f"{type(w.message).__name__}: {w.message}" for w in caught])
    record["fit_seconds"]=time.perf_counter()-start;record["peak_sampled_ram_bytes"]=memory.peak
    return model,record


def k1_gate(model,train,ids,sigmas):
    mean,cov=gaussian_statistics(train,ids,reg_covar=EM_PARAMETERS["reg_covar"])
    mean_error=float(np.max(abs(mean-model.means_[0])));cov_error=float(np.max(abs(cov-model.covariances_[0])))
    np.testing.assert_allclose(model.means_[0],mean,atol=REGRESSION_ATOL,rtol=REGRESSION_RTOL)
    np.testing.assert_allclose(model.covariances_[0],cov,atol=REGRESSION_ATOL,rtol=REGRESSION_RTOL)
    errors=[]
    for sigma in sigmas:
        z=train[:32,:64]
        actual=model.estimate(z,sigma*np.eye(64),selection_matrix(),n_components_or_probability=1.0)
        expected=gaussian_reference(z,mean,cov,sigma)
        np.testing.assert_allclose(actual,expected,atol=REGRESSION_ATOL,rtol=REGRESSION_RTOL)
        errors.append(float(np.max(abs(actual-expected))))
    return mean,cov,dict(mean_max_abs_error=mean_error,covariance_max_abs_error=cov_error,
                         train_sentinel_prediction_max_abs_errors=errors,
                         atol=REGRESSION_ATOL,rtol=REGRESSION_RTOL,all_checks_passed=True,
                         reference="independent 128D R10-probe, population covariance plus identical reg_covar")


def evaluate(models,validation,unit,snrs,sigmas,mean,cov):
    validation_rows=[];trajectory_rows=[];timings={};k1_validation=[]
    for k,model in models.items():
        for snr,sigma in zip(snrs,sigmas,strict=True):
            z=validation[:,:64]+np.sqrt(sigma)*unit
            with MemorySampler() as memory:
                start=time.perf_counter()
                prediction=model.estimate(z,sigma*np.eye(64),selection_matrix(),n_components_or_probability=1.0)
                seconds=time.perf_counter()-start
            if k==1:
                expected=gaussian_reference(z,mean,cov,sigma)
                np.testing.assert_allclose(prediction,expected,atol=REGRESSION_ATOL,rtol=REGRESSION_RTOL)
                k1_validation.append(dict(snr_db=int(snr),max_abs_prediction_error=float(np.max(abs(prediction-expected))),all_checks_passed=True))
            err=(abs(prediction[:,64:]-validation[:,64:])**2).reshape(100,10,64).sum(axis=(1,2))
            truth=(abs(validation[:,64:])**2).reshape(100,10,64).sum(axis=(1,2))
            likelihood=pilot_log_likelihood(z,model.weights_,model.means_,model.covariances_,sigma)
            entry=dict(K=k,snr_db=int(snr),trajectories=100,frames_per_trajectory=10,data_coefficients_per_frame=64,
                       error_energy=float(err.sum()),truth_energy=float(truth.sum()),nmse_linear=float(err.sum()/truth.sum()),
                       nmse_db=float(10*np.log10(err.sum()/truth.sum())),nonlinear_gain_db=0.,
                       fraction_trajectories_improved=0.,median_error_energy_difference=0.,q25_error_energy_difference=0.,q75_error_energy_difference=0.,
                       validation_pilot_log_likelihood=float(likelihood.mean()),inference_seconds=seconds,
                       inference_seconds_per_sample=seconds/1000,peak_inference_sampled_ram_bytes=memory.peak,
                       online_ratio_to_K1=1.)
            validation_rows.append(entry)
            for i in range(100):
                trajectory_rows.append(dict(K=k,snr_db=int(snr),trajectory_id=500+i,frames=10,coefficients=640,
                                            error_energy=float(err[i]),truth_energy=float(truth[i]),nmse_linear=float(err[i]/truth[i])))
            timings[k,int(snr)]=seconds
            print(f"Validation K={k}, SNR={snr}: NMSE={entry['nmse_db']:.6f}dB",flush=True)
    for entry in validation_rows:
        snr=entry["snr_db"];k=entry["K"]
        ref=next(r for r in validation_rows if r["K"]==1 and r["snr_db"]==snr)
        entry["nonlinear_gain_db"]=ref["nmse_db"]-entry["nmse_db"]
        base=[r["error_energy"] for r in trajectory_rows if r["K"]==1 and r["snr_db"]==snr]
        candidate=[r["error_energy"] for r in trajectory_rows if r["K"]==k and r["snr_db"]==snr]
        entry.update({p:v for p,v in paired_diagnostics(candidate,base).items() if p!="difference_definition"})
        entry["online_ratio_to_K1"]=timings[k,snr]/timings[1,snr]
    return validation_rows,trajectory_rows,k1_validation


def main():
    if Path(sys.prefix).resolve()!=(ROOT/".venv-gmm").resolve():raise ValueError("isolated GMM environment required")
    if (OUT/"t15b_gmm_training.csv").exists():raise ValueError("existing EM scan protected; no implicit retries")
    dataset=load(OUT/"t15b_gmm_probe_dataset_manifest.json")
    env=load(OUT/"t15b_gmm_environment.json")
    if not dataset["all_checks_passed"] or not env["all_checks_passed"]:raise ValueError("preflight failed")
    if sha(ROOT/dataset["dataset_path"])!=dataset["dataset_sha256"]:raise ValueError("dataset drift")
    if sha(ROOT/"docs/gmm_probe_protocol.md")!=dataset["protocol_sha256"]:raise ValueError("registered protocol drift")
    check_package_bytes()
    with np.load(ROOT/dataset["dataset_path"]) as f:
        train=f["train"].copy();ids=f["train_trajectory_ids"].copy()
        sigmas=f["sigma2"].copy();snrs=f["snr_db"].copy()
    train.flags.writeable=False
    gmm_before=environment_fingerprint(ROOT/".venv-gmm")
    save(OUT/"t15b_gmm_environment_payload.json",gmm_before)
    result=dict(task="T15B",experiment_type=EXPERIMENT,literature_baseline=True,proposed_method=False,
                package_provenance=env["package_provenance"],package_formula_gate=env["package_formula_gate"],
                accepted_T15A_FIX_commit=dataset["accepted_T15A_FIX_commit"],dataset_manifest_sha256=sha(OUT/"t15b_gmm_probe_dataset_manifest.json"),
                probe_definition=load(OUT/"t15b_gmm_probe_index_manifest.json"),probe_index_hash=dataset["probe_index_hash"],
                training_protocol=dict(K=list(KS),EM_initialization_seeds=list(EM_SEEDS),parameters=EM_PARAMETERS,
                                       selection="highest final train log-likelihood; smaller seed breaks exact ties",selection_split="train_dev",
                                       train_trajectories=500,train_frames=5000,BLAS_threads=1),
                global_mean_magnitude=dict(mean_abs=float(np.mean(abs(train.mean(0)))),l2_norm=float(np.linalg.norm(train.mean(0))),max_abs=float(np.max(abs(train.mean(0))))),
                gmm_results_by_K={},k1_regression=None,validation_nmse=[],nonlinear_gain=[],paired_diagnostics=[],complexity={},
                formal_test_used=False,formal_test_sealed=True,new_neural_model_trained=False,
                all_checks_passed=False,completion_state="running",warnings=[],source_sha256={p:sha(ROOT/p) for p in SOURCES})
    save(OUT/"t15b_gmm_results.json",result)
    start=time.perf_counter();records=[];models={};mean=cov=None
    try:
        with threadpool_limits(limits=1):
            result["threadpools_during_run"]=threadpool_info()
            for k in KS:
                trials={};group=[]
                for seed in EM_SEEDS:
                    print(f"Author EM K={k}, seed={seed}",flush=True)
                    model,record=fit_trial(train,ids,k,seed);records.append(record);group.append(record)
                    if model is not None:trials[seed]=model
                    csv_save(OUT/"t15b_gmm_training.csv",records)
                    print(f"  success={record['fit_succeeded']} converged={record['converged']} iterations={record['iterations']} LL={record['train_log_likelihood']} time={record['fit_seconds']:.2f}s",flush=True)
                    if not record["fit_succeeded"] and k!=8:raise ValueError("STOP: package default stability failed; no regularization search")
                if not trials:
                    result["gmm_results_by_K"][str(k)]=dict(status="fit_failed",runs=group)
                    result["warnings"].append("K8 full covariance failed for all three declared seeds; preserved, no diagonal substitution")
                    continue
                selected=choose_training_seed(group);models[k]=trials[selected]
                for record in group:record["selected"]=record["seed"]==selected
                csv_save(OUT/"t15b_gmm_training.csv",records)
                model=models[k]
                result["gmm_results_by_K"][str(k)]=dict(status="fitted",selected_seed=selected,runs=group,
                        convergence=bool(model.converged_),iterations=int(model.n_iter_),train_log_likelihood=float(model.score(train)),
                        component_weights=model.weights_.tolist(),effective_sample_counts=model.predict_proba(train).sum(0).tolist())
                best=next(r for r in group if r["selected"])
                result["complexity"][str(k)]=dict(covariance_bytes=int(model.covariances_.nbytes),
                        parameter_arrays_bytes=int(sum(getattr(model,a).nbytes for a in ("weights_","means_","covariances_","precisions_","precisions_cholesky_"))),
                        selected_model_file_bytes=(ROOT/best["model_path"]).stat().st_size,
                        three_seed_fit_seconds=float(sum(r["fit_seconds"] for r in group)),selected_fit_seconds=best["fit_seconds"],
                        peak_fit_sampled_ram_bytes=max(r["peak_sampled_ram_bytes"] for r in group))
                if k==1:
                    mean,cov,result["k1_regression"]=k1_gate(model,train,ids,sigmas)
                    np.savez(OUT/"t15b_r10_probe_statistics.npz",mean=mean,covariance=cov,train_trajectory_ids=ids)
                    print("K1 independent train-statistics/prediction gate PASS before K>1",flush=True)
                save(OUT/"t15b_gmm_results.json",result)
            # Validation is read for evaluation only after all training-only selections.
            with np.load(ROOT/dataset["dataset_path"]) as f:validation=f["validation"].copy();unit=f["validation_unit_pilot_noise"].copy()
            values,trajectories,k1_val=evaluate(models,validation,unit,snrs,sigmas,mean,cov)
            result["k1_regression"]["validation_prediction_regression"]=k1_val
            result["validation_nmse"]=values
            result["nonlinear_gain"]=[dict(K=r["K"],snr_db=r["snr_db"],nonlinear_gain_db=r["nonlinear_gain_db"]) for r in values]
            result["paired_diagnostics"]=[{p:r[p] for p in ("K","snr_db","fraction_trajectories_improved","median_error_energy_difference","q25_error_energy_difference","q75_error_energy_difference")} for r in values if r["K"]!=1]
            result.update(scientific_flags(values))
            for k in models:
                rows=[r for r in values if r["K"]==k]
                result["complexity"][str(k)].update(inference_by_snr=[{p:r[p] for p in ("snr_db","inference_seconds","inference_seconds_per_sample","online_ratio_to_K1")} for r in rows],
                    peak_sampled_ram_bytes=max(result["complexity"][str(k)]["peak_fit_sampled_ram_bytes"],max(r["peak_inference_sampled_ram_bytes"] for r in rows)),
                    RAM_measurement="20ms process working-set sampling; all loaded data/prior models included")
            csv_save(OUT/"t15b_gmm_probe_validation.csv",values)
            csv_save(OUT/"t15b_gmm_trajectory_metrics.csv",trajectories)
        check_package_bytes()
        gmm_after=environment_fingerprint(ROOT/".venv-gmm")
        if gmm_before["files"]!=gmm_after["files"]:raise ValueError("GMM environment payload changed")
        result["gmm_environment_unchanged"]=True
        if any(p.startswith(("torch","sionna")) for p in sys.modules):raise ValueError("GMM process imported channel runtime")
        result["CDL_generated_in_GMM_environment"]=False
        result["runtime_seconds"]=time.perf_counter()-start
        result["process_lifetime_peak_ram_bytes"]=ram()[1]
        result["completion_state"]="numeric scan complete; final regression/figure gates pending"
        result["warnings"].extend(r["warning_messages"] for r in records if r["warning_messages"]!="[]")
        save(OUT/"t15b_gmm_results.json",result)
        print("T15B numerical scan saved; no BER; final regression pending",flush=True)
    except Exception as exc:
        result["completion_state"]="failed; STOP";result["failure"]=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        save(OUT/"t15b_gmm_results.json",result);raise


if __name__=="__main__":main()
