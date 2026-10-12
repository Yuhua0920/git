"""T16C load-only basis, dedicated train statistics and NMSE-only heldouts."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.linalg import cholesky
from src.heldout_basis import (T16B_COMMIT,HELDOUT,RANKS,EXPERIMENT_TYPE,definition,seeds,manifest_rows,
    HeldoutCDL,load_bases,gate,qthreshold,gap_closure)
from src.cdl_benchmark import StreamingStatistics,frames,array_hash,observation_geometry
from src.cdl_snr import TrainReceivePowerAccumulator
from src.cdl_mismatch import SNRS,frozen_variances,covariance_diagnostics
from src.lowrank_filter import WeightedCorrection
from src.shared_output_basis import complexity_memory
from src.link import assemble_pilot_data_frame,apply_frequency_mimo_channel,estimate_ls_linear_channel
from scripts.run_cdl_mismatch import load,save,sha,csvsave,history,environments,source_statistics,communication
OUT=ROOT/"outputs"; RESULT=OUT/"t16c_heldout_results.json"
SOURCES=("src/heldout_basis.py","scripts/run_heldout_basis.py","scripts/plot_heldout_basis.py",
    "scripts/audit_heldout_basis.py","tests/test_heldout_basis.py","docs/heldout_basis_protocol.md","scripts/recover_t16b_basis.py")


def snapshot():
    subprocess.check_call(["git","merge-base","--is-ancestor",T16B_COMMIT,"HEAD"],cwd=ROOT)
    subject=subprocess.check_output(["git","show","-s","--format=%s",T16B_COMMIT],cwd=ROOT,text=True).strip()
    if subject!="feat: add shared low-rank correction output-subspace gate": raise ValueError("accepted T16B snapshot required")
    r=load(OUT/"t16b_shared_basis_results.json")
    if not r["all_checks_passed"]: raise ValueError("accepted T16B required")
    for p,h in {**r["source_sha256"],**r["numeric_artifact_sha256"],**r["frozen_history_sha256"]}.items():
        if sha(ROOT/p)!=h: raise ValueError(f"frozen history drift: {p}")
    for p in r["source_sha256"]:
        blob=subprocess.check_output(["git","show",f"{T16B_COMMIT}:{p}"],cwd=ROOT)
        if blob.replace(b"\r\n",b"\n")!=(ROOT/p).read_bytes().replace(b"\r\n",b"\n"): raise ValueError("historical snapshot drift")
    return r


def policy():
    snapshot(); source,prov=source_statistics(); power,var=frozen_variances(load(OUT/"t15a_cdl_results.json"))
    if power!=9.941134822523972 or {str(k):v for k,v in var.items()}!=load(OUT/"t16b_shared_basis_results.json")["snr_policy"]["production_variances"]:
        raise ValueError("source deployment noise policy drift")
    return source,prov,power,var


def namespace_audit():
    import csv
    new=manifest_rows()
    old=[]
    for filename in ("t15a_cdl_manifest.csv","t15c_cdl_manifest.csv"):
        with (OUT/filename).open(encoding="utf8",newline="") as f: old.extend(csv.DictReader(f))
    channel_keys=("sionna_seed","torch_seed","numpy_seed")
    communication_keys=("bits_seed_tx0","bits_seed_tx1","noise_seed")
    historical_channels={tuple(int(r[k]) for k in channel_keys) for r in old}
    historical_frames={tuple(int(r[k]) for k in communication_keys) for r in old}
    new_channels={tuple(r[k] for k in channel_keys) for r in new}
    new_frames={tuple(r[k] for k in communication_keys) for r in new}
    if new_channels&historical_channels or new_frames&historical_frames or len(new_channels)!=1200 or len(new_frames)!=12000:
        raise ValueError("dedicated random stream tuple collision")
    return dict(namespace_inputs_disjoint=True,channel_seed_tuples_disjoint=True,communication_seed_tuples_disjoint=True,
        channel_seed_tuples=1200,frame_communication_tuples=12000,consumed_validation_communication_tuples=2000,
        train_communication_seeds_declared_but_not_consumed=True,historical_namespaces=["T15A","T15C","T16A/B reuse T15C"],
        formal_test_namespace_accessed=False)


def historical_channel_hashes():
    """Use each archived table's verified schema, never assume a shared column."""
    import csv
    hashes=set()
    for filename,column in (("t15a_fix_replay_h_hashes.csv","replayed_h_sha256"),("t15c_cdl_channel_hashes.csv","H_sha256")):
        with (OUT/filename).open(encoding="utf8",newline="") as f:
            reader=csv.DictReader(f)
            if column not in reader.fieldnames:raise ValueError(f"historical H table schema drift: {filename}")
            hashes.update(r[column] for r in reader)
    return hashes


def filters(source,target,var,bases):
    start=time.perf_counter(); estimates={}; controls={}; maps={}; capture=[]; oracle=[]; svd_seconds=0.; projection_seconds=0.
    for s in SNRS:
        estimates["source",s]=source.prepare(var[s]); estimates["matched",s]=target.prepare(var[s])
    filter_seconds=time.perf_counter()-start
    for s in SNRS:
        ws=estimates["source",s].weight; wt=estimates["matched",s].weight
        c=target.rpp+var[s]*np.eye(64); tick=time.perf_counter()
        # SVD only for NEW per-condition oracle controls, never for frozen shared U.
        control=WeightedCorrection.build(ws,wt,c); svd_seconds+=time.perf_counter()-tick
        delta=control.target-control.source; lower=cholesky(c,lower=True); m=delta@lower
        cumulative,_=control.spectrum(); controls[s]=(control,{r:control.factors(r) for r in (4,8)})
        for r in (4,8):
            oracle.append(dict(snr_db=s,rank=r,capture=float(cumulative[r-1]),
                weighted_excess_MSE_per_frame=float(np.sum(control.singular[r:]**2)),
                full_rank_filter_relative_residual=control.diagnostics()["full_rank_filter_relative_residual"]))
        tick=time.perf_counter()
        for scope,b in bases.items():
            for q in RANKS:
                maps[scope,s,q]=b.coefficient_map(delta,q)
                e=b.energy(m,q)
                direct=float(np.linalg.norm((delta-b.output(q)@maps[scope,s,q])@lower)**2)
                np.testing.assert_allclose(direct,e["weighted_excess_MSE_per_frame"],rtol=1e-11,atol=1e-11)
                capture.append(dict(snr_db=s,rank=q,basis_scope=scope,**e,independent_excess_MSE_per_frame=direct))
        projection_seconds+=time.perf_counter()-tick
    return estimates,controls,maps,capture,oracle,dict(filter_construction_seconds=filter_seconds,
        SVD_controls_seconds=svd_seconds,shared_projection_seconds=projection_seconds)


def evaluate(name,i,h,g,var,source,target,estimates,controls,maps,bases):
    h.flags.writeable=False
    methods=[("source",None,"source"),("matched",None,"target")]+[(f"oracle-r{r}",r,"target") for r in (4,8)]
    methods += [(f"shared-q{q}",q,"target") for q in RANKS]
    extra=[(f"shared-q{q}-SourceMeans",q,"source") for q in (8,16)]
    accum={(s,m):dict(error_energy=0.,truth_energy=0.) for s in SNRS for m,_,_ in methods+(extra if s in (10,20) else [])}
    signal=np.zeros(2); noise=np.zeros(2); samples=0; digest=hashlib.sha256(); inference=0.
    for j,truth in enumerate(frames(h)):
        bits,symbols,z=communication(seeds(name,"validation_dev",i,j))
        clean=apply_frequency_mimo_channel(truth,assemble_pilot_data_frame(symbols,g.layout)); clean.flags.writeable=False
        data=g.layout.data_mask[0]; signal+=np.sum(abs(clean[:,data])**2,axis=1); noise+=np.sum(abs(z[:,data])**2,axis=1); samples+=int(data.sum())
        energy=float(np.sum(abs(truth.ravel()[g.data])**2)); input_hash=[array_hash(a) for a in (truth,bits,symbols,z)]
        for s in SNRS:
            y=clean+np.sqrt(var[s])*z; y.flags.writeable=False; yh=array_hash(y)
            sparse,_=estimate_ls_linear_channel(y,g.layout); pilot=sparse.ravel()[g.selected]; pilot.flags.writeable=False
            for method,rank,mean_kind in methods+(extra if s in (10,20) else []):
                tick=time.perf_counter()
                if rank is None: coeff=estimates[method,s].estimate(pilot)
                elif method.startswith("oracle"):
                    c,f=controls[s]; coeff=c.estimate(pilot,target.mean_p,target.mean_d,f[rank])
                else:
                    mean=target if mean_kind=="target" else source
                    coeff=bases["all_conditions"].estimate(pilot,estimates["source",s].weight,mean.mean_p,mean.mean_d,
                        rank,maps["all_conditions",s,rank])
                inference+=time.perf_counter()-tick
                e=accum[s,method]; e["truth_energy"]+=energy; e["error_energy"]+=float(np.sum(abs(coeff-truth.ravel()[g.data])**2))
                if array_hash(y)!=yh: raise ValueError("shared Y changed")
            if input_hash!=[array_hash(a) for a in (truth,bits,symbols,z)]: raise ValueError("shared input changed")
            for v in (*input_hash,yh): digest.update(v.encode("ascii"))
    rows=[]
    for s in SNRS:
        for m,r,means in methods+(extra if s in (10,20) else []):
            e=accum[s,m]
            rows.append(dict(shift=name,namespace=definition(name)["namespace"],trajectory_id=i,split="validation_dev",frames=10,
                dF=16,pilot_ratio=float(g.layout.pilot_union.mean()),snr_db=s,method=m,rank=r,means=means,
                **e,nmse_linear=e["error_energy"]/e["truth_energy"]))
    return rows,dict(H_sha256=array_hash(h),paired_inputs_sha256=digest.hexdigest()),signal,noise,samples,inference


def aggregate(rows):
    pooled=[]
    for name in HELDOUT:
        for s in SNRS:
            for method in dict.fromkeys(r["method"] for r in rows if r["shift"]==name and r["snr_db"]==s):
                v=[r for r in rows if r["shift"]==name and r["snr_db"]==s and r["method"]==method]
                e=sum(r["error_energy"] for r in v); d=sum(r["truth_energy"] for r in v)
                pooled.append(dict(shift=name,snr_db=s,method=method,rank=v[0]["rank"],means=v[0]["means"],
                    trajectories=len(v),frames_per_trajectory=10,error_energy=e,truth_energy=d,nmse_linear=e/d,nmse_db=float(10*np.log10(e/d)),
                    trajectory_nmse_mean=float(np.mean([r["nmse_linear"] for r in v])),
                    trajectory_nmse_std=float(np.std([r["nmse_linear"] for r in v],ddof=1)),metric_type="actual_validation"))
    for r in pooled:
        a=next(x for x in pooled if x["shift"]==r["shift"] and x["snr_db"]==r["snr_db"] and x["method"]=="source")
        b=next(x for x in pooled if x["shift"]==r["shift"] and x["snr_db"]==r["snr_db"] and x["method"]=="matched")
        r.update(gap_closure(a["nmse_db"],b["nmse_db"],r["nmse_db"]))
    return pooled


def environment():
    if Path(sys.prefix).resolve()!=(ROOT/".venv-cdl").resolve(): raise ValueError("frozen CDL environment required")


def preflight():
    environment(); source,prov,power,var=policy(); bases,proof=load_bases(ROOT); g=observation_geometry(); selected={}
    namespace_audit()
    historical_channel_hashes()
    for name in HELDOUT:
        adapter=HeldoutCDL(name); selected[name]={}
        for split,i in (("train_dev",0),("validation_dev",500)):
            h=adapter.generate(i,split=split); repeat=adapter.generate(i,split=split)
            if array_hash(h)!=array_hash(repeat): raise ValueError("small deterministic generator mismatch")
            selected[name][split]=array_hash(h)
            if split=="train_dev":
                stream=StreamingStatistics(g.selected,g.data); stream.update(frames(h).reshape(10,4096),[0]); target=stream.freeze()
                est,controls,maps,capture,oracle,_=filters(source,target,var,bases)
            else:
                rows,evidence,*_=evaluate(name,i,h,g,var,source,target,est,controls,maps,bases)
                if len(rows)!=28 or any(not np.isfinite(r["nmse_linear"]) for r in rows): raise ValueError("small inference count/finiteness mismatch")
        if selected[name]["train_dev"]==selected[name]["validation_dev"]: raise ValueError("small split collision")
    load_bases(ROOT)
    save(OUT/"t16c_heldout_preflight.json",dict(all_checks_passed=True,selected_H_sha256=selected,
        source_statistics_provenance=prov,frozen_basis_provenance=proof,source_reference_power=power,
        source_sha256={p:sha(ROOT/p) for p in SOURCES},BER_evaluations=0,detector_calls=0))
    print("T16C load-only frozen basis / dedicated split / generator / train-only statistics / NMSE inference preflight PASS",flush=True)


def run():
    environment()
    if RESULT.exists() or (OUT/"t16c_heldout_shared_basis.csv").exists(): raise ValueError("existing numerical outputs protected")
    small=load(OUT/"t16c_heldout_preflight.json")
    if not small["all_checks_passed"] or any(sha(ROOT/p)!=h for p,h in small["source_sha256"].items()): raise ValueError("small preflight absent/drift")
    start=time.perf_counter(); before=history(); env=environments()
    source,source_prov,power,var=policy(); bases,basis_prov=load_bases(ROOT); namespace=namespace_audit()
    g=observation_geometry()
    if len(g.selected)!=64 or int(g.data.sum())!=3968 or g.layout.pilot_union.mean()!=1/32: raise ValueError("actual pilot geometry drift")
    csvsave(OUT/"t16c_heldout_manifest.csv",manifest_rows())
    rows=[]; hashes=[]; pairing=[]; captures=[]; oracle_energy=[]; statistics_prov={}; audits=[]; diagnostics={}; runtimes={}; adapters={}
    for name in HELDOUT:
        adapter=HeldoutCDL(name); adapters[name]=adapter.provenance()
        stream=StreamingStatistics(g.selected,g.data); powers=TrainReceivePowerAccumulator(); batch=[]; ids=[]
        generation=0.; stats_seconds=0.; train_start=time.perf_counter()
        for i in range(500):
            tick=time.perf_counter(); h=adapter.generate(i,split="train_dev"); generation+=time.perf_counter()-tick
            if i==0 and array_hash(h)!=small["selected_H_sha256"][name]["train_dev"]: raise ValueError("train preflight hash drift")
            powers.update(i,h); batch.append(frames(h).reshape(10,4096)); ids.append(i)
            hashes.append(dict(shift=name,split="train_dev",trajectory_id=i,H_sha256=array_hash(h)))
            if len(batch)==10:
                tick=time.perf_counter(); stream.update(np.concatenate(batch),ids); stats_seconds+=time.perf_counter()-tick; batch=[]; ids=[]
            if (i+1)%50==0: print(f"{name} target train {i+1}/500",flush=True)
        target=stream.freeze(); trainref=powers.freeze(); path=OUT/f"t16c_heldout_statistics_{name}.npz"
        np.savez_compressed(path,mean_p=target.mean_p,mean_d=target.mean_d,rpp=target.rpp,rdp=target.rdp,
            train_trajectory_ids=target.train_trajectory_ids,samples=target.samples)
        statistics_prov[name]=dict(path=str(path.relative_to(ROOT)).replace("\\","/"),sha256=sha(path),
            component_sha256={k:array_hash(getattr(target,k)) for k in ("mean_p","mean_d","rpp","rdp")},
            namespace=definition(name)["namespace"],train_only=True,validation_used_for_fitting=False,
            train_trajectory_ids=list(target.train_trajectory_ids),samples=5000,centered=True,covariance_divisor="N population",
            train_ensemble_power_audit_only=trainref.reference)
        diagnostics[name]=covariance_diagnostics(source,target,var)
        est,controls,maps,capture,control_energy,timing=filters(source,target,var,bases)
        allowed=("all_conditions","CDL-C-120") if name=="CDL-C-75" else tuple(bases)
        captures.extend(dict(shift=name,**r) for r in capture if r["basis_scope"] in allowed)
        oracle_energy.extend(dict(shift=name,**r) for r in control_energy)
        runtimes[name]=dict(target_train_generation_seconds=generation,statistics_construction_seconds=stats_seconds,
            target_train_pipeline_seconds=time.perf_counter()-train_start,**timing,validation_generation_seconds=0.,validation_inference_seconds=0.)
        signal=np.zeros(2); noise=np.zeros(2); samples=0; val_start=time.perf_counter()
        for i in range(500,600):
            tick=time.perf_counter(); h=adapter.generate(i,split="validation_dev"); runtimes[name]["validation_generation_seconds"]+=time.perf_counter()-tick
            if i==500 and array_hash(h)!=small["selected_H_sha256"][name]["validation_dev"]: raise ValueError("validation preflight hash drift")
            hashes.append(dict(shift=name,split="validation_dev",trajectory_id=i,H_sha256=array_hash(h)))
            actual,evidence,s,n,count,seconds=evaluate(name,i,h,g,var,source,target,est,controls,maps,bases)
            rows.extend(actual); pairing.append(dict(shift=name,trajectory_id=i,**evidence)); signal+=s; noise+=n; samples+=count
            runtimes[name]["validation_inference_seconds"]+=seconds
            if (i-499)%25==0: print(f"{name} heldout validation {i-499}/100",flush=True)
        runtimes[name]["validation_pipeline_seconds"]=time.perf_counter()-val_start
        for s in SNRS:
            audits.append(dict(shift=name,snr_db=s,source_reference_power=power,production_sigma2=var[s],
                target_train_ensemble_power_audit_only=trainref.reference,target_train_power_per_rx=list(trainref.per_rx_signal_power),
                target_train_referenced_effective_snr_db=float(10*np.log10(trainref.reference/var[s])),
                validation_realised_signal_power_per_rx=(signal/samples).tolist(),
                validation_realised_noise_power_per_rx=(var[s]*noise/samples).tolist(),
                validation_realised_receive_snr_per_rx_db=(10*np.log10(signal/(var[s]*noise))).tolist(),
                validation_realised_mean_receive_snr_db=float(10*np.log10(signal.sum()/(var[s]*noise.sum()))),
                target_power_used_for_production_noise=False,validation_recalibrated_noise=False,channel_normalized=False))
    pooled=aggregate(rows); shared=[r for r in pooled if r["method"].startswith("shared") and r["means"]=="target"]
    oracle=[dict(**r,**next({k:v for k,v in e.items() if k not in ("shift","snr_db","rank")} for e in oracle_energy
        if e["shift"]==r["shift"] and e["snr_db"]==r["snr_db"] and e["rank"]==r["rank"])) for r in pooled if r["method"].startswith("oracle")]
    sensitivity=[r for r in pooled if r["means"]=="source" and r["rank"] is not None]
    for r in sensitivity:
        ref=next(x for x in shared if x["shift"]==r["shift"] and x["snr_db"]==r["snr_db"] and x["rank"]==r["rank"])
        r["source_minus_target_means_nmse_db"]=r["nmse_db"]-ref["nmse_db"]
    mismatches=[r for r in pooled if r["method"] in ("source","matched")]
    q80={n:{str(s):qthreshold([r for r in shared if r["shift"]==n and r["snr_db"]==s],.8) for s in SNRS} for n in HELDOUT}
    q90={n:{str(s):qthreshold([r for r in shared if r["shift"]==n and r["snr_db"]==s],.9) for s in SNRS} for n in HELDOUT}
    csvsave(OUT/"t16c_heldout_shared_basis.csv",shared); csvsave(OUT/"t16c_heldout_oracle_lowrank.csv",oracle)
    csvsave(OUT/"t16c_heldout_capture.csv",captures); csvsave(OUT/"t16c_heldout_trajectory_metrics.csv",rows)
    csvsave(OUT/"t16c_heldout_baselines.csv",mismatches); csvsave(OUT/"t16c_heldout_mean_sensitivity.csv",sensitivity)
    csvsave(OUT/"t16c_heldout_channel_hashes.csv",hashes)
    if len({r["H_sha256"] for r in hashes})!=1200: raise ValueError("new heldout channel duplicate")
    historical_H=historical_channel_hashes()
    if {r["H_sha256"] for r in hashes}&historical_H: raise ValueError("historical/heldout H collision")
    load_bases(ROOT)
    if any(sha(ROOT/p)!=h for p,h in before.items()): raise ValueError("historical files changed")
    attempt=OUT/"t16c_attempt01/attempt01_numeric_sha256.json"
    recovery=dict(prior_numeric_attempt_failed=False)
    if attempt.exists():
        frozen=load(attempt)
        exact={p:sha(OUT/p)==h for p,h in frozen.items()}
        if not all(exact.values()):raise ValueError("STOP: deterministic numeric retry differs from archived failed attempt")
        recovery=dict(prior_numeric_attempt_failed=True,failure="KeyError reading historical H table column after numeric CSV save",
            archive="outputs/t16c_attempt01",same_manifest_seeds_environment=True,all_archived_numeric_artifacts_exact=True,
            exact_artifact_comparisons=exact,retry_scope="schema-only fix; unchanged estimators/channel/noise/statistics arithmetic")
    runtimes["total_numeric_wall_seconds"]=time.perf_counter()-start
    result=dict(task="T16C",experiment_type=EXPERIMENT_TYPE,literature_gate_passed=True,proposed_method=False,heldout_representation_gate=True,
        accepted_T16B_commit=T16B_COMMIT,frozen_basis_provenance=basis_prov,heldout_definitions={n:definition(n) for n in HELDOUT},
        source_statistics_provenance=source_prov,target_statistics_provenance=statistics_prov,
        source_target_mismatch=mismatches,condition_specific_lowrank=oracle,shared_capture=captures,shared_validation_nmse=shared,
        shared_closure=shared,q80=q80,q90=q90,interpolation_diagnostic=[r for r in captures if r["shift"]=="CDL-C-75"],
        composition_diagnostic=[r for r in captures if r["shift"]=="CDL-B-120"],mean_sensitivity=sensitivity,**gate(shared,oracle),
        covariance_diagnostics=diagnostics,snr_audit=audits,namespace_audit=namespace,source_noise_variances={str(k):v for k,v in var.items()},
        source_reference_power=power,sionna_provenance=adapters,paired_input_evidence=pairing,
        exact_excess_mse_available=True,absolute_theoretical_expected_nmse_available=False,expected_nmse=None,
        anchored_nmse_used_for_decision_gate=False,anchor_used=False,shared_basis_refitted=False,
        formal_basis_construction_SVD_calls=0,per_condition_oracle_control_SVD_calls=6,
        complexity=[complexity_memory(q) for q in (8,16)],new_model_trained=False,formal_test_used=False,formal_test_sealed=True,
        BER_evaluations=0,detector_calls=0,runtime=runtimes,numeric_attempt_recovery=recovery,source_sha256={p:sha(ROOT/p) for p in SOURCES},
        frozen_history_sha256=before,environment_before=env,all_checks_passed=False,completion_state="numeric complete; final audit pending",
        warnings=["Historical U arrays missing in T16B; authorized exact provenance recovery, not basis refit.",
            "Oracle target means/statistics/maps and per-condition controls; no deployable predictor.",
            "Rdd absent; no absolute theoretical NMSE or anchor-based gates.",
            "Development heldout environments; formal test sealed; correlated frames, independent trajectory.",
            "Source deployment noise reference retained; target realised SNR may differ from labels.",
            "Unclipped closure; small/nonpositive gaps retained; exactly zero gap undefined.",
            "Optional Triton flop counter unavailable; existing per-symbol abstraction omits within-symbol ICI."])
    save(RESULT,result); print(dict(scientific_gate=gate(shared,oracle),q80=q80,q90=q90,runtime=runtimes),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--preflight",action="store_true")
    if p.parse_args().preflight: preflight()
    else: run()
