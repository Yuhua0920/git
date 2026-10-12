"""T15C bounded streaming CDL OOD experiment; no historical outputs modified."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".pytest_cache/matplotlib-t15c"))
import numpy as np
from src.cdl_mismatch import (ShiftCDL,SHIFTS,METHODS,SNRS,T15B_COMMIT,EXPERIMENT_TYPE,
    shift_definition,seeds,manifest_rows,frozen_variances,covariance_diagnostics,scientific_flags,paired_summary)
from src.cdl_benchmark import (OfficialCDL,FrozenStatistics,StreamingStatistics,frames,array_hash,
                               observation_geometry,seeds as source_seeds)
from src.cdl_snr import TrainReceivePowerAccumulator
from src.link import (generate_master_qpsk,assemble_pilot_data_frame,apply_frequency_mimo_channel,
    estimate_ls_linear_channel,detect_data_zf,extract_reference_data_bits,demodulate_data_bits)
from src.numerics import complex_gaussian_noise
from scripts.build_gmm_probe_dataset import frozen_environments
from scripts.check_cdl_environment import environment_fingerprint
OUT=ROOT/"outputs"
SOURCES=("src/cdl_mismatch.py","scripts/run_cdl_mismatch.py","scripts/plot_cdl_mismatch.py",
         "scripts/audit_cdl_mismatch.py","tests/test_cdl_mismatch.py","docs/cdl_mismatch_protocol.md")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding="utf8"))
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf8")
def csvsave(path,rows):
    with Path(path).open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def environments():
    result=frozen_environments()
    old=load(OUT/"t15b_gmm_environment_payload.json");new=environment_fingerprint(ROOT/".venv-gmm")
    if old["files"]!=new["files"]:raise ValueError("GMM environment payload changed")
    result["gmm"]=dict(payload_unchanged=True,sha256=new["sha256"])
    return result


def history():
    paths=subprocess.check_output(["git","ls-files"],cwd=ROOT,text=True).splitlines()
    return {p:sha(ROOT/p) for p in paths if p.startswith(("src/","scripts/","tests/","configs/","outputs/"))}


def require_commit():
    subprocess.check_call(["git","merge-base","--is-ancestor",T15B_COMMIT,"HEAD"],cwd=ROOT)
    subject=subprocess.check_output(["git","show","-s","--format=%s",T15B_COMMIT],text=True).strip()
    if subject!="feat: add full-covariance complex-GMM nonlinear-prior value gate":raise ValueError("accepted T15B snapshot required")
    return T15B_COMMIT


def source_statistics():
    result=load(OUT/"t15a_cdl_results.json");info=result["empirical_lmmse_definition"]["snapshots"]["500"]
    path=ROOT/info["path"]
    if sha(path)!=info["sha256"]:raise ValueError("source frozen statistics changed")
    archived=OUT/"t15a_attempt01_nominal_tx_snr/outputs/t15a_cdl_statistics_500.npz"
    if sha(path)!=sha(archived):raise ValueError("source clean statistics archive changed")
    with np.load(path) as f:
        values=[f[k].copy() for k in ("mean_p","mean_d","rpp","rdp")]
        ids=tuple(map(int,f["train_trajectory_ids"]));n=int(f["samples"])
    if ids!=tuple(range(500)) or n!=5000:raise ValueError("source train protocol drift")
    for a in values:a.flags.writeable=False
    stats=FrozenStatistics(*values,ids,n)
    return stats,dict(path=info["path"],sha256=sha(path),exact_T15A_FIX_hash=True,
        component_sha256={k:array_hash(v) for k,v in zip(("mean_p","mean_d","rpp","rdp"),values)},
        source_commit="d923830a4f90a9b0b85d2f4f11edbb495811168e",source_statistics_refitted=False)


def communication(seed):
    bits,symbols=generate_master_qpsk(bits_seeds=(seed["bits_seed_tx0"],seed["bits_seed_tx1"]))
    z=complex_gaussian_noise((2,64,16),1.,np.random.default_rng(seed["noise_seed"]))
    for a in (bits,symbols,z):a.flags.writeable=False
    return bits,symbols,z


def evaluate_trajectory(name,i,h,g,estimators,variances,*,seed_function=None):
    h.flags.writeable=False
    accum={(s,m):dict(error_energy=0.,truth_energy=0.,bit_errors=0,total_bits=0) for s in SNRS for m in METHODS}
    timing={"R3_seconds":0.,"R10_C_frozen_seconds":0.,"R10_matched_seconds":0.,"ZF_seconds":0.}
    signal=np.zeros(2);noise=np.zeros(2);samples=0;digest=hashlib.sha256()
    for j,truth in enumerate(frames(h)):
        seed=seed_function(i,j) if seed_function else seeds(name,"validation_dev",i,j)
        bits,symbols,z=communication(seed)
        x=assemble_pilot_data_frame(symbols,g.layout);clean=apply_frequency_mimo_channel(truth,x)
        reference=extract_reference_data_bits(bits,g.layout.data_mask)
        data=g.layout.data_mask[0];signal+=np.sum(abs(clean[:,data])**2,axis=1)
        noise+=np.sum(abs(z[:,data])**2,axis=1);samples+=int(data.sum())
        energy=float(np.sum(abs(truth.ravel()[g.data])**2))
        input_hash=[array_hash(a) for a in (truth,bits,symbols,z)]
        for s in SNRS:
            y=clean+np.sqrt(variances[s])*z;y.flags.writeable=False;yh=array_hash(y)
            start=time.perf_counter();sparse,linear=estimate_ls_linear_channel(y,g.layout)
            timing["R3_seconds"]+=time.perf_counter()-start
            pilot=sparse.ravel()[g.selected];pilot.flags.writeable=False
            estimates={"R3":linear,"Perfect-CSI":truth}
            for method,key in (("R10-C-frozen","R10_C_frozen_seconds"),("R10-matched","R10_matched_seconds")):
                start=time.perf_counter();estimate=np.full(4096,np.nan+1j*np.nan,np.complex128)
                estimate[g.data]=estimators[method,s].estimate(pilot)
                estimates[method]=estimate.reshape(2,2,64,16);timing[key]+=time.perf_counter()-start
            for method in METHODS:
                estimate=estimates[method];estimate.flags.writeable=False
                start=time.perf_counter();detected=detect_data_zf(y,estimate,g.layout.data_mask)*np.sqrt(2)
                timing["ZF_seconds"]+=time.perf_counter()-start
                predicted=demodulate_data_bits(detected,g.layout.data_mask)
                entry=accum[s,method];entry["bit_errors"]+=sum(int(np.count_nonzero(a!=b)) for a,b in zip(predicted,reference))
                entry["total_bits"]+=sum(a.size for a in reference);entry["truth_energy"]+=energy
                if method!="Perfect-CSI":entry["error_energy"]+=float(np.sum(abs(estimate.ravel()[g.data]-truth.ravel()[g.data])**2))
                if array_hash(y)!=yh:raise ValueError("paired Y mutated")
            if input_hash!=[array_hash(a) for a in (truth,bits,symbols,z)]:raise ValueError("paired H/bits/Z mutated")
            for v in (*input_hash,yh):digest.update(v.encode("ascii"))
    rows=[dict(shift=name,trajectory_id=i,split="validation_dev",frames=10,dF=16,snr_db=s,method=m,**e,
               nmse_linear=e["error_energy"]/e["truth_energy"],ber=e["bit_errors"]/e["total_bits"])
          for (s,m),e in accum.items()]
    return rows,timing,dict(H_sha256=array_hash(h),paired_inputs_sha256=digest.hexdigest(),
        shared_H_bits_Z_Y_verified=True),signal,noise,samples


def aggregate(rows,g):
    result=[]
    for name in SHIFTS:
        for s in SNRS:
            for m in METHODS:
                v=[r for r in rows if r["shift"]==name and r["snr_db"]==s and r["method"]==m]
                e=sum(r["error_energy"] for r in v);d=sum(r["truth_energy"] for r in v)
                n=sum(r["bit_errors"] for r in v);b=sum(r["total_bits"] for r in v)
                result.append(dict(shift=name,method=m,dF=16,pilot_ratio=float(g.layout.pilot_union.mean()),snr_db=s,
                    trajectories=len(v),frames_per_trajectory=10,error_energy=e,truth_energy=d,nmse_linear=e/d,
                    nmse_db=float(10*np.log10(e/d)) if e else None,bit_errors=n,total_bits=b,ber=n/b,
                    trajectory_nmse_mean=float(np.mean([r["nmse_linear"] for r in v])),
                    trajectory_nmse_std=float(np.std([r["nmse_linear"] for r in v],ddof=1)),
                    trajectory_ber_std=float(np.std([r["ber"] for r in v],ddof=1))))
    return result


def preflight():
    if Path(sys.prefix).resolve()!=(ROOT/".venv-cdl").resolve():raise ValueError("frozen CDL environment required")
    require_commit();env=environments();source,prov=source_statistics()
    power,variances=frozen_variances(load(OUT/"t15a_cdl_results.json"));g=observation_geometry()
    estimators={(m,s):source.prepare(variances[s]) for m in ("R10-C-frozen","R10-matched") for s in SNRS}
    original=OfficialCDL();h=original.generate(500)
    with (OUT/"t15a_fix_replay_h_hashes.csv").open(encoding="utf8") as f:
        expected=next(r for r in csv.DictReader(f) if int(r["trajectory_id"])==500)
    # The original saved column is H_sha256.
    expected_hash=next(v for k,v in expected.items() if "sha256" in k or "hash" in k)
    if array_hash(h)!=expected_hash:raise ValueError("frozen source H replay drift")
    actual=evaluate_trajectory("source-sentinel",500,h,g,estimators,variances,seed_function=source_seeds)[0]
    with (OUT/"t15a_cdl_trajectory_metrics.csv").open(encoding="utf8") as f:
        expected_rows=[r for r in csv.DictReader(f) if int(r["trajectory_id"])==500]
    checks=[]
    for r in actual:
        if r["method"]=="R10-matched":continue
        mapping={"R3":"R3-LS-LIN","R10-C-frozen":"R10-500","Perfect-CSI":"R0-Perfect-CSI"}
        old=next(x for x in expected_rows if x["method"]==mapping[r["method"]] and int(x["snr_db"])==r["snr_db"])
        for field in ("error_energy","truth_energy","bit_errors","total_bits","ber"):
            if str(r[field])!=old[field]:raise ValueError(f"source full-link sentinel drift {r['method']} {field}")
        checks.append(dict(method=r["method"],snr_db=r["snr_db"],exact=True,bit_errors=r["bit_errors"]))
    target_hashes={};target_prov={}
    for name in SHIFTS:
        adapter=ShiftCDL(name);target_prov[name]=adapter.provenance();target_hashes[name]={}
        for split,i in (("train_dev",0),("validation_dev",500)):
            h=adapter.generate(i,split=split);repeat=adapter.generate(i,split=split)
            if array_hash(h)!=array_hash(repeat):raise ValueError("target deterministic small replay failed")
            target_hashes[name][split]=array_hash(h)
        if target_hashes[name]["train_dev"]==target_hashes[name]["validation_dev"]:raise ValueError("split collision")
    result=dict(all_checks_passed=True,environments=env,source_statistics=prov,source_H_hash_matched=True,
        exact_source_link_regression=checks,target_selected_hashes=target_hashes,target_provenance=target_prov,
        source_power=power,variances=variances,complete_raw_H_saved=False,
        source_sha256={p:sha(ROOT/p) for p in SOURCES})
    save(OUT/"t15c_cdl_preflight.json",result);print("T15C source exact H/link sentinel and both target replay smoke PASS",flush=True)


def run():
    if Path(sys.prefix).resolve()!=(ROOT/".venv-cdl").resolve():raise ValueError("frozen CDL environment required")
    if (OUT/"t15c_cdl_mismatch.csv").exists():raise ValueError("existing numerical scan protected")
    started=time.perf_counter();require_commit();env=environments();before=history()
    gate=load(OUT/"t15c_cdl_preflight.json")
    if not gate["all_checks_passed"] or any(sha(ROOT/p)!=v for p,v in gate["source_sha256"].items()):raise ValueError("small preflight or source gate failed")
    source,prov=source_statistics();fix=load(OUT/"t15a_cdl_results.json");power,variances=frozen_variances(fix)
    csvsave(OUT/"t15c_cdl_manifest.csv",manifest_rows());g=observation_geometry()
    runtime={};matched_prov={};diagnostics={};rows=[];snr_audit=[];pairing=[];channel_hashes=[]
    target_prov={};storage=[]
    for name in SHIFTS:
        adapter=ShiftCDL(name);target_prov[name]=adapter.provenance()
        stream=StreamingStatistics(g.selected,g.data);powers=TrainReceivePowerAccumulator()
        batch=[];ids=[];generation=0.;trainstart=time.perf_counter()
        for i in range(500):
            start=time.perf_counter();h=adapter.generate(i,split="train_dev");generation+=time.perf_counter()-start
            if i==0 and array_hash(h)!=gate["target_selected_hashes"][name]["train_dev"]:raise ValueError("target train replay drift")
            powers.update(i,h);batch.append(frames(h).reshape(10,4096));ids.append(i)
            channel_hashes.append(dict(shift=name,split="train_dev",trajectory_id=i,H_sha256=array_hash(h)))
            if len(batch)==10:stream.update(np.concatenate(batch),ids);batch=[];ids=[]
            if (i+1)%50==0:print(f"{name} train statistics {i+1}/500",flush=True)
        target=stream.freeze();trainref=powers.freeze();trainseconds=time.perf_counter()-trainstart
        path=OUT/f"t15c_cdl_statistics_{name}.npz"
        np.savez_compressed(path,mean_p=target.mean_p,mean_d=target.mean_d,rpp=target.rpp,rdp=target.rdp,
            train_trajectory_ids=target.train_trajectory_ids,samples=target.samples)
        matched_prov[name]=dict(path=str(path.relative_to(ROOT)).replace("\\","/"),sha256=sha(path),
            component_sha256={k:array_hash(getattr(target,k)) for k in ("mean_p","mean_d","rpp","rdp")},
            namespace=f"T15C_{name}_OOD_v1",split="train_dev",trajectories=500,frames=5000,
            train_trajectory_ids=list(target.train_trajectory_ids),training_only=True,
            mean_and_covariance_centered=True,covariance_divisor="N population",batch_frames=100,
            train_power_per_rx=list(trainref.per_rx_signal_power),train_ensemble_power=trainref.reference,
            matched_reference_oracle_style=True,validation_used_for_fitting=False)
        diagnostics[name]=covariance_diagnostics(source,target,variances)
        prep={};estimators={}
        for method,stats in (("R10-C-frozen",source),("R10-matched",target)):
            start=time.perf_counter()
            for s in SNRS:estimators[method,s]=stats.prepare(variances[s])
            prep[method]=time.perf_counter()-start
        runtime[name]=dict(target_train_generation_seconds=generation,target_train_statistics_pipeline_seconds=trainseconds,
            target_train_non_generation_seconds=trainseconds-generation,filter_preparation_seconds=prep,
            validation_generation_seconds=0.,R3_seconds=0.,R10_C_frozen_seconds=0.,R10_matched_seconds=0.,ZF_seconds=0.)
        signal=np.zeros(2);noise=np.zeros(2);expected=np.zeros(2);samples=0;valstart=time.perf_counter()
        for i in range(500,600):
            start=time.perf_counter();h=adapter.generate(i,split="validation_dev")
            runtime[name]["validation_generation_seconds"]+=time.perf_counter()-start
            if i==500 and array_hash(h)!=gate["target_selected_hashes"][name]["validation_dev"]:raise ValueError("target validation replay drift")
            channel_hashes.append(dict(shift=name,split="validation_dev",trajectory_id=i,H_sha256=array_hash(h)))
            expected+=.5*np.mean(abs(h)**2,axis=(2,3)).sum(axis=1)
            values,timing,evidence,s,n,count=evaluate_trajectory(name,i,h,g,estimators,variances)
            rows.extend(values);signal+=s;noise+=n;samples+=count
            for k,v in timing.items():runtime[name][k]+=v
            pairing.append(dict(shift=name,trajectory_id=i,**evidence))
            if (i-499)%10==0:print(f"{name} paired validation {i-499}/100",flush=True)
        runtime[name]["validation_pipeline_seconds"]=time.perf_counter()-valstart
        expected/=100
        for s in SNRS:
            sigma=variances[s]
            snr_audit.append(dict(shift=name,nominal_source_snr_db=s,production_sigma2=sigma,
                source_train_reference_power=power,target_train_ensemble_power=trainref.reference,
                target_train_power_per_rx=list(trainref.per_rx_signal_power),
                target_ensemble_snr_offset_db=float(10*np.log10(trainref.reference/power)),
                target_train_referenced_effective_snr_db=float(10*np.log10(trainref.reference/sigma)),
                validation_expected_power_per_rx=expected.tolist(),
                validation_expected_receive_snr_per_rx_db=(10*np.log10(expected/sigma)).tolist(),
                validation_expected_mean_receive_snr_db=float(10*np.log10(expected.mean()/sigma)),
                validation_realised_signal_power_per_rx=(signal/samples).tolist(),
                validation_realised_noise_power_per_rx=(sigma*noise/samples).tolist(),
                validation_realised_receive_snr_per_rx_db=(10*np.log10(signal/(sigma*noise))).tolist(),
                validation_realised_mean_receive_snr_db=float(10*np.log10(signal.sum()/(sigma*noise.sum()))),
                validation_recalibrated_noise=False,target_power_used_for_production_noise=False))
        storage.append(dict(path=str(path.relative_to(ROOT)).replace("\\","/"),bytes=path.stat().st_size))
    aggregate_rows=aggregate(rows,g)
    penalties=[];ber=[];paired=[];context=[]
    for name in SHIFTS:
        for s in SNRS:
            lookup={r["method"]:r for r in aggregate_rows if r["shift"]==name and r["snr_db"]==s}
            frozen,matched=lookup["R10-C-frozen"],lookup["R10-matched"]
            penalties.append(dict(shift=name,snr_db=s,mismatch_penalty_db=frozen["nmse_db"]-matched["nmse_db"]))
            ber.append(dict(shift=name,snr_db=s,frozen_ber=frozen["ber"],matched_ber=matched["ber"],
                absolute_ber_gap=frozen["ber"]-matched["ber"],relative_ber_degradation=frozen["ber"]/matched["ber"]-1 if matched["ber"] else None))
            context.append(dict(shift=name,snr_db=s,frozen_minus_R3_nmse_db=frozen["nmse_db"]-lookup["R3"]["nmse_db"],
                                frozen_minus_R3_ber=frozen["ber"]-lookup["R3"]["ber"]))
            f=[r for r in rows if r["shift"]==name and r["snr_db"]==s and r["method"]=="R10-C-frozen"]
            m=[r for r in rows if r["shift"]==name and r["snr_db"]==s and r["method"]=="R10-matched"]
            paired.append(dict(shift=name,snr_db=s,**paired_summary(f,m)))
    csvsave(OUT/"t15c_cdl_mismatch.csv",aggregate_rows)
    csvsave(OUT/"t15c_cdl_mismatch_trajectory_metrics.csv",rows)
    csvsave(OUT/"t15c_cdl_channel_hashes.csv",channel_hashes)
    diagrows=[]
    for name,d in diagnostics.items():
        for s in SNRS:
            kl=d["pilot_gaussian_kl"][str(s)]
            diagrows.append(dict(shift=name,snr_db=s,rpp_relative_frobenius=d["rpp_relative_frobenius"],
                rdp_relative_frobenius=d["rdp_relative_frobenius"],mean_p_distance=d["mean_p_distance"],mean_d_distance=d["mean_d_distance"],
                top8_angles_degrees=json.dumps(d["subspaces"]["8"]["angles_degrees"]),
                top16_angles_degrees=json.dumps(d["subspaces"]["16"]["angles_degrees"]),**kl))
    csvsave(OUT/"t15c_cdl_covariance_diagnostics.csv",diagrows)
    after=history()
    if before!=after:raise ValueError("frozen historical files changed")
    if len({r["H_sha256"] for r in channel_hashes})!=1200:raise ValueError("channel duplicate across namespaces")
    runtime["total_numeric_wall_seconds"]=time.perf_counter()-started
    result=dict(task="T15C",experiment_type=EXPERIMENT_TYPE,literature_gate_passed=True,proposed_method=False,
        accepted_T15B_commit=T15B_COMMIT,source_model_provenance=prov,shift_definitions={n:shift_definition(n) for n in SHIFTS},
        sionna_provenance=target_prov,preflight=gate,
        snr_policy=dict(definition=fix["snr_definition"],production="deployment-frozen source train ensemble receive-power reference",
            source_reference_power=power,production_variances={str(k):v for k,v in variances.items()},
            target_train_power_audit_only=True,no_validation_recalibration=True,no_per_shift_H_normalization=True,audit=snr_audit),
        matched_statistics_provenance=matched_prov,covariance_diagnostics=diagnostics,
        nmse_results=aggregate_rows,ber_results=ber,mismatch_penalties=penalties,R3_context=context,
        trajectory_diagnostics=paired,independent_unit="trajectory",paired_evaluation_evidence=pairing,
        **scientific_flags(penalties),formal_test_used=False,formal_test_sealed=True,new_model_trained=False,
        source_statistics_refitted=False,all_checks_passed=False,completion_state="numeric scan complete; final audit pending",
        source_sha256={p:sha(ROOT/p) for p in SOURCES},frozen_history_sha256=before,
        frozen_history_unchanged=True,frozen_history_files_verified=len(before),environment_before=env,
        runtime=runtime,storage=dict(statistics=storage,complete_raw_H_saved=False,max_train_batch_trajectories=10,
            max_training_H_scratch_bytes=100*4096*16,validation_H_residency_trajectories=1),
        warnings=["Development validation only; final formal test remains sealed.",
            "Ten continuous frames are correlated; independent replication unit is trajectory.",
            "Target train ensemble-power SNR offset is finite-sample diagnostic, not a theoretical exact power.",
            "Matched empirical reference has oracle-style access to 500 independent clean target training trajectories; no adaptation algorithm.",
            "Pilot Gaussian KL uses noisy LS marginals and does not establish CDL joint Gaussianity.",
            "Small boundary eigengaps can make principal-subspace angles sensitive; eigengaps are reported.",
            "Original calibrated per-OFDM-symbol frequency-domain channel abstraction retained; no within-symbol ICI model added.",
            "Optional Triton flop counter may be unavailable in frozen Windows CDL environment."])
    save(OUT/"t15c_cdl_mismatch_results.json",result)
    print(json.dumps(dict(penalties=penalties,scientific_flags=scientific_flags(penalties),runtime=runtime),indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--preflight",action="store_true")
    if p.parse_args().preflight:preflight()
    else:run()
