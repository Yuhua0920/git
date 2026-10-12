"""T16D frozen development modules; mathematical preflight without channels."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.linalg import cholesky
from src.modular_correction import (T16C_COMMIT,BUDGETS,EXPERIMENT_TYPE,RANK_TOLERANCE,
    frozen_module_unions,weighted_projection,ScalarComposition,interaction_spectrum,scientific_gate)
from src.heldout_basis import load_bases,HELDOUT,gap_closure
from src.cdl_benchmark import FrozenStatistics,array_hash
from src.lowrank_filter import readonly
from scripts.run_cdl_mismatch import load,save,sha,csvsave,history,environments
from scripts.run_lowrank_filter import statistics,prepare,BASELINES
OUT=ROOT/"outputs";RESULT=OUT/"t16d_modular_results.json"
VALIDATION_REPLAY_AUTHORIZED=True
SOURCES=("src/modular_correction.py","scripts/run_modular_correction.py","scripts/plot_modular_correction.py",
         "scripts/audit_modular_correction.py","tests/test_modular_correction.py","docs/modular_correction_protocol.md")


def accepted_snapshot():
    subprocess.check_call(["git","merge-base","--is-ancestor",T16C_COMMIT,"HEAD"],cwd=ROOT)
    subject=subprocess.check_output(["git","show","-s","--format=%s",T16C_COMMIT],cwd=ROOT,text=True).strip()
    if subject!="feat: add held-out shared correction-subspace generalization gate":raise ValueError("accepted T16C snapshot required")
    old=load(OUT/"t16c_heldout_results.json")
    if not old["all_checks_passed"] or old["completion_state"]!="completed":raise ValueError("T16C final acceptance absent")
    for p,h in {**old["source_sha256"],**old["frozen_history_sha256"],**old["numeric_artifact_sha256"]}.items():
        if sha(ROOT/p)!=h:raise ValueError(f"historical file drift: {p}")
    for p in [*old["source_sha256"],"outputs/t16c_heldout_results.json"]:
        blob=subprocess.check_output(["git","show",f"{T16C_COMMIT}:{p}"],cwd=ROOT)
        if blob.replace(b"\r\n",b"\n")!=(ROOT/p).read_bytes().replace(b"\r\n",b"\n"):raise ValueError("historical Git snapshot drift")
    return old


def frozen_validation_tables():
    """Read only accepted evidence; never create seeds or call a generator."""
    import csv
    old=accepted_snapshot()
    tables={}
    for name in ("manifest","channel_hashes","trajectory_metrics","baselines","shared_basis"):
        with (OUT/f"t16c_heldout_{name}.csv").open(encoding="utf8",newline="") as f:
            tables[name]=list(csv.DictReader(f))
    validation=[r for r in tables["manifest"] if r["split"]=="validation_dev"]
    expected={(name,i,j) for name in HELDOUT for i in range(500,600) for j in range(10)}
    if len(validation)!=2000 or {(r["shift"],int(r["trajectory_id"]),int(r["frame_id"])) for r in validation}!=expected:
        raise ValueError("frozen validation manifest identity/count drift")
    hashes=[r for r in tables["channel_hashes"] if r["split"]=="validation_dev"]
    if len(hashes)!=200 or len({r["H_sha256"] for r in hashes})!=200:raise ValueError("frozen validation H evidence drift")
    paired={(r["shift"],int(r["trajectory_id"])):r for r in old["paired_input_evidence"]}
    if len(paired)!=200 or any(paired[r["shift"],int(r["trajectory_id"])]["H_sha256"]!=r["H_sha256"] for r in hashes):
        raise ValueError("historical H evidence groups disagree")
    for r in [*tables["baselines"],*tables["shared_basis"]]:
        method=r["method"];name=r["shift"];s=int(r["snr_db"])
        rows=[x for x in tables["trajectory_metrics"] if x["method"]==method and x["shift"]==name
              and int(x["snr_db"])==s and x["means"]==r["means"]]
        if len(rows)!=100:raise ValueError("baseline trajectory grouping drift")
        error=sum(float(x["error_energy"]) for x in rows);truth=sum(float(x["truth_energy"]) for x in rows)
        if error!=float(r["error_energy"]) or truth!=float(r["truth_energy"]):raise ValueError("historical pooled baseline evidence drift")
    return old,tables


def target_statistics(old):
    result={}
    for name,info in old["target_statistics_provenance"].items():
        path=ROOT/info["path"]
        if sha(path)!=info["sha256"]:raise ValueError("frozen target statistics drift")
        blob=subprocess.check_output(["git","show",f"{T16C_COMMIT}:{info['path']}"],cwd=ROOT)
        if hashlib.sha256(blob).hexdigest()!=info["sha256"]:raise ValueError("target statistics snapshot drift")
        with np.load(path,allow_pickle=False) as f:
            a=[f[k].copy() for k in ("mean_p","mean_d","rpp","rdp")]
            ids=tuple(map(int,f["train_trajectory_ids"]));count=int(f["samples"])
            if "rdd" in f.files:raise ValueError("unexpected Rdd")
        if count!=5000 or ids!=tuple(range(500)):raise ValueError("target train-only protocol drift")
        for k,v in zip(("mean_p","mean_d","rpp","rdp"),a):
            if array_hash(v)!=info["component_sha256"][k]:raise ValueError("target component hash drift")
            v.flags.writeable=False
        result[name]=FrozenStatistics(*a,ids,count)
    return result


def construct():
    start=time.perf_counter();old=accepted_snapshot()
    _,source,development,var=statistics();estimates,_,filter_seconds=prepare(source,development,var)
    bases,basis_proof=load_bases(ROOT);regression=load(OUT/"t16b_shared_basis_results.json")["matrix_regression"]
    modules={};checks=[]
    for name in ("CDL-B-30","CDL-C-120"):
        for s in (5,10,20):
            ws=readonly(estimates[name,BASELINES[0],s].weight);wt=readonly(estimates[name,BASELINES[1],s].weight)
            delta=readonly(wt-ws);lower=cholesky(development[name].rpp+var[s]*np.eye(64),lower=True)
            saved=next(r for r in regression if r["shift"]==name and r["snr_db"]==s)
            if any(array_hash(a)!=saved[k] for a,k in ((ws,"source_W_sha256"),(wt,"target_W_sha256"),(delta@lower,"current_M_sha256"))):
                raise ValueError("STOP: development correction operator historical hash mismatch")
            modules[name,s]=delta
            checks.append(dict(shift=name,snr_db=s,source_W_sha256=array_hash(ws),target_W_sha256=array_hash(wt),
                               DeltaW_sha256=array_hash(delta),weighted_DeltaW_historical_sha256=saved["current_M_sha256"],historical_exact=True))
    tick=time.perf_counter();unions=frozen_module_unions(bases);union_seconds=time.perf_counter()-tick
    targets=target_statistics(old);operators={};capture=[];scalar=[];spectra=[];shared_maps={};new_estimates={}
    projection_seconds=0.;scalar_seconds=0.;interaction_seconds=0.
    for name,target in targets.items():
        for s in (5,10,20):
            ws=source.prepare(var[s]);wt=target.prepare(var[s]);new_estimates[name,s]=(ws,wt)
            delta=readonly(readonly(wt.weight)-readonly(ws.weight));lower=cholesky(target.rpp+var[s]*np.eye(64),lower=True)
            tick=time.perf_counter()
            for budget,(q,definition) in unions.items():
                g,energy=weighted_projection(delta,lower,q)
                operators[name,s,f"modular-Q{budget}"]=(q,g)
                capture.append(dict(shift=name,snr_db=s,dimension_budget=budget,numerical_rank=definition["numerical_rank"],**energy))
                shared_maps[name,s,budget]=bases["all_conditions"].coefficient_map(delta,budget)
            projection_seconds+=time.perf_counter()-tick
            tick=time.perf_counter()
            mode_names=("CDL-C-120",) if name=="CDL-C-75" else ("CDL-B-30","CDL-C-120")
            fit=ScalarComposition.fit(delta,[modules[n,s] for n in mode_names],lower)
            operators[name,s,"scalar-real"]=fit.correction
            scalar.append(dict(shift=name,snr_db=s,alpha=None if name=="CDL-C-75" else float(fit.coefficients[0]),
                beta=float(fit.coefficients[-1]),real_coordinate_count=len(mode_names),**fit.diagnostics))
            scalar_seconds+=time.perf_counter()-tick
            if name=="CDL-B-120":
                tick=time.perf_counter();spec=interaction_spectrum(fit.residual,lower);interaction_seconds+=time.perf_counter()-tick
                spectra.append(dict(shift=name,snr_db=s,weighted_energy_fraction=fit.diagnostics["weighted_residual_fraction"],**spec))
    provenance=dict(accepted_T16C_commit=T16C_COMMIT,frozen_basis_provenance=basis_proof,development_operator_regression=checks,
        modules={"P":"C30->B30","M":"C30->C120"},module_statistics_refitted=False,target_statistics_refitted=False,
        heldout_used_for_module_construction=False,construction_conditions=[[n,s] for n in ("CDL-B-30","CDL-C-120") for s in (5,10,20)])
    runtime=dict(development_filter_construction_seconds=filter_seconds,union_orthonormalization_seconds=union_seconds,
        heldout_projection_seconds=projection_seconds,scalar_fit_seconds=scalar_seconds,
        interaction_SVD_seconds=interaction_seconds,math_construction_wall_seconds=time.perf_counter()-start)
    return dict(old=old,source=source,targets=targets,var=var,bases=bases,unions=unions,modules=modules,
        operators=operators,estimates=new_estimates,shared_maps=shared_maps,capture=capture,scalar=scalar,
        interaction=spectra,provenance=provenance,runtime=runtime)


def environment():
    if Path(sys.prefix).resolve()!=(ROOT/".venv-cdl").resolve():raise ValueError("frozen CDL environment required")


def math_preflight():
    environment();c=construct()
    r=dict(all_checks_passed=True,stage="math_only_no_channel_calls",module_provenance=c["provenance"],
        union_basis_definition={str(q):info for q,(_,info) in c["unions"].items()},capture=c["capture"],
        scalar_fits=c["scalar"],runtime=c["runtime"],protocol_sha256=sha(ROOT/"docs/modular_correction_protocol.md"),
        source_sha256={p:sha(ROOT/p) for p in SOURCES if (ROOT/p).exists()},
        channel_calls=0,new_channel_realizations=0,BER_evaluations=0)
    save(OUT/"t16d_modular_math_preflight.json",r)
    print("T16D development hashes / union / weighted projection / real scalar math preflight PASS",flush=True)


def evaluate(name,i,h,c,*,baseline_only,input_cache=None):
    """Same input digest and scalar accumulation as historical T16C; no detector."""
    from src.heldout_basis import seeds
    from src.cdl_benchmark import observation_geometry,frames
    from scripts.run_cdl_mismatch import communication
    from src.link import assemble_pilot_data_frame,apply_frequency_mimo_channel,estimate_ls_linear_channel
    if not baseline_only and input_cache is None:raise ValueError("formal methods require the same historical replay input cache")
    g=observation_geometry();target=c["targets"][name]
    methods=["source","matched",*[f"shared-q{q}" for q in BUDGETS]]
    if not baseline_only:methods += [*[f"modular-Q{q}" for q in BUDGETS],"scalar-real"]
    accum={(s,m):dict(error_energy=0.,truth_energy=0.) for s in (5,10,20) for m in methods}
    digest=hashlib.sha256();inference=0.;h.flags.writeable=False
    for j,truth in enumerate(frames(h)):
        if input_cache is None:
            bits,symbols,z=communication(seeds(name,"validation_dev",i,j))
            clean=apply_frequency_mimo_channel(truth,assemble_pilot_data_frame(symbols,g.layout));clean.flags.writeable=False
        else:bits,symbols,z=(input_cache[k][j] for k in ("bits","symbols","z"))
        input_hash=[array_hash(a) for a in (truth,bits,symbols,z)]
        energy=float(np.sum(abs(truth.ravel()[g.data])**2))
        for si,s in enumerate((5,10,20)):
            y=clean+np.sqrt(c["var"][s])*z if input_cache is None else input_cache["y"][j,si]
            y.flags.writeable=False;yh=array_hash(y)
            sparse,_=estimate_ls_linear_channel(y,g.layout);pilot=sparse.ravel()[g.selected];pilot.flags.writeable=False
            ws,wt=c["estimates"][name,s]
            for method in methods:
                tick=time.perf_counter()
                if method=="source":coeff=ws.estimate(pilot)
                elif method=="matched":coeff=wt.estimate(pilot)
                elif method.startswith("shared"):
                    q=int(method.split("q")[-1]);b=c["bases"]["all_conditions"]
                    coeff=b.estimate(pilot,ws.weight,target.mean_p,target.mean_d,q,c["shared_maps"][name,s,q])
                else:
                    x=pilot-target.mean_p;base=target.mean_d+x@ws.weight.T
                    operator=c["operators"][name,s,method]
                    coeff=base+x@operator.T if method=="scalar-real" else base+(x@operator[1].T)@operator[0].T
                inference+=time.perf_counter()-tick
                accum[s,method]["truth_energy"]+=energy
                accum[s,method]["error_energy"]+=float(np.sum(abs(coeff-truth.ravel()[g.data])**2))
                if array_hash(y)!=yh:raise ValueError("shared Y changed")
            if input_hash!=[array_hash(a) for a in (truth,bits,symbols,z)]:raise ValueError("shared inputs changed")
            for v in (*input_hash,yh):digest.update(v.encode("ascii"))
    rows=[]
    for s in (5,10,20):
        for method in methods:
            e=accum[s,method]
            rows.append(dict(shift=name,trajectory_id=i,split="validation_dev",frames=10,dF=16,
                pilot_ratio=float(g.layout.pilot_union.mean()),snr_db=s,method=method,
                means="source" if method=="source" else "target",**e,nmse_linear=e["error_energy"]/e["truth_energy"]))
    return rows,digest.hexdigest(),inference


def pooled(rows):
    result=[]
    for name in HELDOUT:
        for s in (5,10,20):
            for method in dict.fromkeys(r["method"] for r in rows if r["shift"]==name and r["snr_db"]==s):
                v=[r for r in rows if r["shift"]==name and r["snr_db"]==s and r["method"]==method]
                error=sum(r["error_energy"] for r in v);truth=sum(r["truth_energy"] for r in v)
                result.append(dict(shift=name,snr_db=s,method=method,means="source" if method=="source" else "target",
                    trajectories=len(v),frames=10*len(v),error_energy=error,truth_energy=truth,nmse_linear=error/truth,
                    nmse_db=float(10*np.log10(error/truth)),metric_type="actual_validation",
                    trajectory_nmse_mean=float(np.mean([r["nmse_linear"] for r in v])),
                    trajectory_nmse_std=float(np.std([r["nmse_linear"] for r in v],ddof=1))))
    for r in result:
        select=lambda m:next(x for x in result if x["shift"]==r["shift"] and x["snr_db"]==r["snr_db"] and x["method"]==m)
        r.update(gap_closure(select("source")["nmse_db"],select("matched")["nmse_db"],r["nmse_db"]))
    return result


def check_baselines(rows,aggregates,tables):
    """Exact float fields, not approximate NMSE agreement."""
    index={(r["shift"],int(r["trajectory_id"]),int(r["snr_db"]),r["method"]):r for r in tables["trajectory_metrics"]}
    base_methods={"source","matched",*[f"shared-q{q}" for q in BUDGETS]}
    for r in rows:
        if r["method"] not in base_methods:continue
        old=index[r["shift"],r["trajectory_id"],r["snr_db"],r["method"]]
        if any(r[k]!=float(old[k]) for k in ("error_energy","truth_energy","nmse_linear")):
            raise ValueError("STOP: T16C baseline trajectory numerical fields not exact")
    originals={(r["shift"],int(r["snr_db"]),r["method"]):r for r in [*tables["baselines"],*tables["shared_basis"]]}
    for r in aggregates:
        if r["method"] not in base_methods:continue
        old=originals[r["shift"],r["snr_db"],r["method"]]
        if any(r[k]!=float(old[k]) for k in ("error_energy","truth_energy","nmse_linear","nmse_db","closure")):
            raise ValueError("STOP: T16C pooled baseline numerical fields not exact")


def capture_historical_trajectory(name,i,h,c,estimates,controls,maps):
    """Observe original T16C function outputs without changing its arithmetic."""
    from unittest.mock import patch
    from src.cdl_benchmark import observation_geometry
    from scripts import run_heldout_basis as historical
    recorded=[];original_communication=historical.communication;original_ls=historical.estimate_ls_linear_channel
    def capture_communication(seed):
        bits,symbols,z=original_communication(seed)
        recorded.append(dict(bits=bits,symbols=symbols,z=z,y=[]))
        return bits,symbols,z
    def capture_ls(y,layout):
        recorded[-1]["y"].append(y)
        return original_ls(y,layout)
    with patch.object(historical,"communication",capture_communication),patch.object(historical,"estimate_ls_linear_channel",capture_ls):
        result=historical.evaluate(name,i,h,observation_geometry(),c["var"],c["source"],c["targets"][name],estimates,controls,maps,c["bases"])
    if len(recorded)!=10 or any(len(r["y"])!=3 for r in recorded):raise ValueError("historical input capture count drift")
    arrays={k:np.stack([r[k] for r in recorded]) for k in ("bits","symbols","z","y")}
    return result,arrays


def check_historical_rows(actual,tables):
    index={(r["shift"],int(r["trajectory_id"]),int(r["snr_db"]),r["method"]):r for r in tables["trajectory_metrics"]}
    for row in actual:
        old=index[row["shift"],row["trajectory_id"],row["snr_db"],row["method"]]
        for k,value in row.items():
            exact=old[k]=="" if value is None else float(old[k])==value if isinstance(value,(float,int)) else old[k]==value
            if not exact:raise ValueError(f"STOP: historical trajectory field not exact: {k}")


def replay_gate(c):
    """Authorized historical replay only; all original T16C rows checked first."""
    if not VALIDATION_REPLAY_AUTHORIZED:
        raise ValueError("STOP: original T16C deterministic validation replay authorization pending")
    from src.heldout_basis import HeldoutCDL,manifest_rows
    from scripts import run_heldout_basis as historical
    old,tables=frozen_validation_tables()
    expected=[{k:str(v) for k,v in r.items()} for r in manifest_rows()]
    if tables["manifest"]!=expected:raise ValueError("STOP: historical manifest/seed mapping not exact")
    cache=ROOT/".pytest_cache/t16d_validation_replay";cache.mkdir(exist_ok=True)
    if any(cache.glob("*.npy")):raise ValueError("existing replay cache protected")
    evidence={(r["shift"],int(r["trajectory_id"])):r for r in old["paired_input_evidence"]}
    cache_info={};rows=[];replay_seconds=0.;baseline_seconds=0.;start=time.perf_counter();control_seconds=0.
    for name in HELDOUT:
        adapter=HeldoutCDL(name);saved=None;info=None;signal=np.zeros(2);noise=np.zeros(2);samples=0
        tick=time.perf_counter()
        estimates,controls,maps,*_=historical.filters(c["source"],c["targets"][name],c["var"],c["bases"])
        control_seconds+=time.perf_counter()-tick
        # These are original per-condition oracle regression controls, never a shared/union basis refit.
        for i in range(500,600):
            tick=time.perf_counter();h=adapter.generate(i,split="validation_dev");replay_seconds+=time.perf_counter()-tick
            ref=evidence[name,i]
            if array_hash(h)!=ref["H_sha256"]:raise ValueError("STOP: historical validation H hash mismatch")
            if saved is None:
                path=cache/f"{name}.npy"
                saved=np.lib.format.open_memmap(path,mode="w+",dtype=np.complex128,shape=(100,h.size))
                info=dict(path=str(path.relative_to(ROOT)).replace("\\","/"),shape=list(h.shape),strides=list(h.strides),input_artifacts={})
            if list(h.shape)!=info["shape"] or list(h.strides)!=info["strides"]:raise ValueError("historical H layout changed")
            saved[i-500]=h.ravel(order="K")
            result,inputs=capture_historical_trajectory(name,i,h,c,estimates,controls,maps)
            current,actual_evidence,s,n,count,seconds=result;baseline_seconds+=seconds
            signal+=s;noise+=n;samples+=count
            if actual_evidence!=dict(H_sha256=ref["H_sha256"],paired_inputs_sha256=ref["paired_inputs_sha256"]):
                raise ValueError("STOP: historical H/bits/unit-noise/Y digest mismatch")
            check_historical_rows(current,tables);rows.extend(current)
            input_path=cache/f"{name}_{i}_inputs.npz"
            np.savez_compressed(input_path,**inputs);input_path.chmod(0o444)
            info["input_artifacts"][str(i)]=dict(path=str(input_path.relative_to(ROOT)).replace("\\","/"),file_sha256=sha(input_path),
                replay_tensor_sha256={k:array_hash(a) for k,a in inputs.items()},evidence_scope="new replay cache hashes, not separately saved historical tensor hashes")
            if (i-499)%25==0:print(f"{name} original T16C replay gate {i-499}/100",flush=True)
        saved.flush();del saved
        path=ROOT/info["path"];info["file_sha256"]=sha(path);path.chmod(0o444);cache_info[name]=info
        for s in (5,10,20):
            prior=next(r for r in old["snr_audit"] if r["shift"]==name and r["snr_db"]==s)
            current=dict(validation_realised_signal_power_per_rx=(signal/samples).tolist(),
                validation_realised_noise_power_per_rx=(c["var"][s]*noise/samples).tolist(),
                validation_realised_receive_snr_per_rx_db=(10*np.log10(signal/(c["var"][s]*noise))).tolist(),
                validation_realised_mean_receive_snr_db=float(10*np.log10(signal.sum()/(c["var"][s]*noise.sum()))))
            if any(current[k]!=prior[k] for k in current):raise ValueError("STOP: historical validation SNR summary not exact")
    aggregates=historical.aggregate(rows);check_baselines(rows,aggregates,tables)
    frozen=[*old["source_target_mismatch"],*old["shared_validation_nmse"],*old["condition_specific_lowrank"],*old["mean_sensitivity"]]
    index={(r["shift"],r["snr_db"],r["method"]):r for r in frozen}
    for r in aggregates:
        ref=index[r["shift"],r["snr_db"],r["method"]]
        if any(r[k]!=ref[k] for k in r):raise ValueError("STOP: complete historical pooled summaries not exact")
    proof=dict(all_checks_passed=True,accepted_T16C_commit=T16C_COMMIT,manifest_seed_mapping_exact=True,
        all_200_H_hashes_exact=True,all_200_paired_input_digests_exact=True,
        all_5600_historical_trajectory_rows_exact=True,all_56_historical_pooled_rows_exact=True,validation_SNR_audit_exact=True,
        baseline_trajectory_rows_exact=3000,baseline_aggregate_rows_exact=30,new_channel_realizations=0,
        original_validation_generator_calls=200,train_generator_calls=0,formal_test_used=False,
        historical_full_tensor_hashes_available=False,
        historical_full_tensor_hashes_scope="complete separately saved set of H/bits/unit-noise/Y tensor hashes",
        historical_H_trajectory_tensor_hashes_available=True,historical_combined_paired_input_digests_available=True,
        historical_separate_bits_tensor_hashes_available=False,historical_separate_unit_noise_tensor_hashes_available=False,
        historical_separate_Y_tensor_hashes_available=False,historical_T16C_oracle_control_SVD_replays=6,
        frozen_shared_basis_refitted=False,
        cache=cache_info,runtime=dict(original_validation_replay_generation_seconds=replay_seconds,
            historical_baseline_inference_seconds=baseline_seconds,historical_control_reconstruction_seconds=control_seconds,
            replay_gate_wall_seconds=time.perf_counter()-start))
    save(OUT/"t16d_t16c_validation_replay_gate.json",proof)
    print("All 200 H / paired inputs / 5600 original trajectory rows / 56 pooled summaries / SNR summaries EXACT; gate PASS",flush=True)
    return proof,tables


def cached_H(info,index):
    path=ROOT/info["path"]
    data=np.load(path,mmap_mode="r",allow_pickle=False)
    h=np.ndarray(tuple(info["shape"]),dtype=np.complex128,buffer=data[index],strides=tuple(info["strides"]))
    h.flags.writeable=False
    return h


def cached_inputs(info,trajectory):
    artifact=info["input_artifacts"][str(trajectory)];path=ROOT/artifact["path"]
    if sha(path)!=artifact["file_sha256"]:raise ValueError("replay input cache changed")
    with np.load(path,allow_pickle=False) as f:inputs={k:f[k].copy() for k in ("bits","symbols","z","y")}
    if any(array_hash(a)!=artifact["replay_tensor_sha256"][k] for k,a in inputs.items()):raise ValueError("replay cached tensor changed")
    for a in inputs.values():a.flags.writeable=False
    return inputs


def run():
    environment()
    if not VALIDATION_REPLAY_AUTHORIZED:raise ValueError("STOP: original T16C replay clarification pending")
    if RESULT.exists() or (OUT/"t16d_modular_union.csv").exists():raise ValueError("existing formal numerical results protected")
    preflight=load(OUT/"t16d_modular_math_preflight.json")
    if not preflight["all_checks_passed"] or set(preflight["source_sha256"])!=set(SOURCES):raise ValueError("complete mathematical preflight required")
    if any(sha(ROOT/p)!=h for p,h in preflight["source_sha256"].items()):raise ValueError("preflight source drift")
    start=time.perf_counter();before=history();env=environments();c=construct()
    proof,tables=replay_gate(c)
    if not proof["all_checks_passed"]:raise ValueError("STOP: historical replay gate failed")
    # New actual inference begins only after the complete historical replay gate.
    basis_path=OUT/"t16d_modular_union_bases.npz"
    np.savez_compressed(basis_path,**{f"Q{q}":value[0] for q,value in c["unions"].items()},
        **{f"D{q}_singular":np.array(value[1]["singular_values"]) for q,value in c["unions"].items()},
        relative_rank_tolerance=RANK_TOLERANCE,source_commit=T16C_COMMIT,
        module_provenance_json=json.dumps(c["provenance"],sort_keys=True))
    basis_path.chmod(0o444)
    basis_provenance=dict(path=str(basis_path.relative_to(ROOT)).replace("\\","/"),file_sha256=sha(basis_path),readonly=True,
        Q_sha256={str(q):array_hash(value[0]) for q,value in c["unions"].items()},
        construction_used_only_frozen_development_U=True,heldout_used_for_construction=False)
    paired={(r["shift"],int(r["trajectory_id"])):r for r in c["old"]["paired_input_evidence"]}
    rows=[];inference={};phase=time.perf_counter();small={};small_evidence=[]
    for name,info in proof["cache"].items():
        if sha(ROOT/info["path"])!=info["file_sha256"]:raise ValueError("verified historical replay cache changed")
        h=cached_H(info,0);inputs=cached_inputs(info,500)
        actual,digest,seconds=evaluate(name,500,h,c,baseline_only=False,input_cache=inputs)
        if len(actual)!=27 or any(not np.isfinite(r["nmse_linear"]) for r in actual):raise ValueError("small actual inference count/finiteness failed")
        if array_hash(h)!=paired[name,500]["H_sha256"] or digest!=paired[name,500]["paired_inputs_sha256"]:
            raise ValueError("STOP: small actual inference input drift")
        check_baselines(actual,[],tables);small[name]=(actual,digest,seconds)
        small_evidence.append(dict(shift=name,trajectory_id=500,rows=27,paired_digest_exact=True,baselines_exact=True))
    save(OUT/"t16d_actual_inference_preflight.json",dict(all_checks_passed=True,historical_replay_gate_passed=True,
        selected_original_trajectories=small_evidence,new_generator_calls=0,new_communication_calls=0))
    for name,info in proof["cache"].items():
        if sha(ROOT/info["path"])!=info["file_sha256"]:raise ValueError("verified historical replay cache changed")
        inference[name]=0.
        for i in range(500,600):
            h=cached_H(info,i-500)
            if array_hash(h)!=paired[name,i]["H_sha256"]:raise ValueError("cache changed historical H")
            if i==500:current,digest,seconds=small[name]
            else:
                inputs=cached_inputs(info,i)
                current,digest,seconds=evaluate(name,i,h,c,baseline_only=False,input_cache=inputs)
            inference[name]+=seconds
            if digest!=paired[name,i]["paired_inputs_sha256"]:raise ValueError("STOP: formal paired input digest changed")
            check_baselines(current,[],tables);rows.extend(current)
            if (i-499)%25==0:print(f"{name} modular/scalar frozen validation {i-499}/100",flush=True)
    aggregates=pooled(rows);check_baselines(rows,aggregates,tables)
    union=[];scalar=[]
    for row in aggregates:
        if row["method"].startswith("modular"):
            budget=int(row["method"].split("Q")[-1]);name=row["shift"];s=row["snr_db"]
            energy=next(r for r in c["capture"] if r["shift"]==name and r["snr_db"]==s and r["dimension_budget"]==budget)
            generic=next(r for r in aggregates if r["shift"]==name and r["snr_db"]==s and r["method"]==f"shared-q{budget}")
            union.append(dict(**row,**{k:v for k,v in energy.items() if k not in ("shift","snr_db")},
                generic_nmse_db=generic["nmse_db"],generic_closure=generic["closure"],
                delta_closure=row["closure"]-generic["closure"],modular_minus_generic_nmse_db=row["nmse_db"]-generic["nmse_db"]))
        elif row["method"]=="scalar-real":
            fit=next(r for r in c["scalar"] if r["shift"]==row["shift"] and r["snr_db"]==row["snr_db"])
            select=lambda m:next(r for r in aggregates if r["shift"]==row["shift"] and r["snr_db"]==row["snr_db"] and r["method"]==m)
            # Keep array-valued normal-equation diagnostics in JSON; numeric CSV is scalar-only.
            values={k:v for k,v in fit.items() if k not in ("shift","snr_db","normal_equation_gram","normal_equation_rhs")}
            scalar.append(dict(**row,**values,source_nmse_db=select("source")["nmse_db"],matched_nmse_db=select("matched")["nmse_db"]))
    spectrum=[dict(shift=r["shift"],snr_db=r["snr_db"],singular_index=i+1,singular_value=v,
        weighted_energy_fraction=r["weighted_energy_fraction"],effective_rank=r["effective_rank"],
        numerical_rank=r["numerical_rank"],relative_rank_tolerance=RANK_TOLERANCE)
        for r in c["interaction"] for i,v in enumerate(r["singular_values"])]
    csvsave(OUT/"t16d_modular_union.csv",union);csvsave(OUT/"t16d_scalar_composition.csv",scalar)
    csvsave(OUT/"t16d_interaction_spectrum.csv",spectrum);csvsave(OUT/"t16d_trajectory_metrics.csv",rows)
    load_bases(ROOT)
    if any(sha(ROOT/p)!=h for p,h in before.items()):raise ValueError("historical files changed")
    runtime=dict(**c["runtime"],**proof["runtime"],validation_inference_seconds=inference,
        new_inference_pipeline_seconds=time.perf_counter()-phase,total_numeric_wall_seconds=time.perf_counter()-start,
        target_train_generation_seconds=0.,statistics_fitting_seconds=0.)
    result=dict(task="T16D",experiment_type=EXPERIMENT_TYPE,literature_gate_passed=True,proposed_method=False,
        modular_representation_gate=True,module_provenance=c["provenance"],union_basis_artifact=basis_provenance,
        union_basis_definition={str(q):v[1] for q,v in c["unions"].items()},
        union_numerical_rank={str(q):v[1]["numerical_rank"] for q,v in c["unions"].items()},
        capture=c["capture"],validation_nmse=union,closure=union,generic_vs_modular=union,
        scalar_mode_results=scalar,alpha_beta=c["scalar"],interaction_residual=c["interaction"],**scientific_gate(union,scalar),
        validation_replay_gate=proof,formal_validation_pairing_exact=True,shared_T16C_baselines_exact=True,
        deterministic_validation_replay_used=True,new_validation_realizations_generated=False,
        t16c_historical_replay_gate_passed=True,historical_full_tensor_hashes_available=False,
        historical_hash_evidence_definition=proof["historical_full_tensor_hashes_scope"],
        t16d_methods_evaluated_on_same_replayed_realizations=True,
        formal_communication_regeneration_calls=0,formal_noise_rescaling_calls=0,
        source_noise_reference_power=c["old"]["source_reference_power"],source_noise_variances=c["old"]["source_noise_variances"],
        target_statistics_provenance=c["old"]["target_statistics_provenance"],
        complexity=[dict(dimension_budget=q,actual_numerical_rank=c["unions"][q][1]["numerical_rank"],
            reconstruction_complex_MAC_per_frame=q*3968,actual_rank_reconstruction_MAC=c["unions"][q][1]["numerical_rank"]*3968,
            reconstruction_overhead_ratio=q/64,predictor_cost_estimated=False) for q in BUDGETS],
        scalar_representation_dimension={"CDL-C-75":1,"CDL-B-120":2},complex_scalar_control_run=False,
        exact_excess_mse_available=True,absolute_theoretical_expected_nmse_available=False,
        new_channel_realizations=0,train_channel_generator_calls=0,formal_inference_generator_calls=0,
        deterministic_original_T16C_validation_replay=True,BER_evaluations=0,detector_calls=0,
        new_model_trained=False,formal_test_used=False,formal_test_sealed=True,runtime=runtime,
        source_sha256={p:sha(ROOT/p) for p in SOURCES},frozen_history_sha256=before,environment_before=env,
        numeric_artifact_sha256={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in OUT.glob("t16d_*.csv")},
        all_checks_passed=False,completion_state="numeric complete; final audit pending",
        warnings=["Oracle target operators/means/coefficients; no deployable adaptive estimator.",
            "Original T16C validation replay allowed by researcher; no new realizations, no target-train generation.",
            "Rdd absent; no absolute theoretical NMSE or anchor gate.",
            "Correlated frames within independent trajectories; pooled ratios, descriptive trajectory dispersion, no CI.",
            "Original source deployment SNR retained; no target normalization; per-symbol abstraction omits within-symbol ICI.",
            "Optional complex scalar control not run; primary real coefficients unchanged."])
    save(RESULT,result)
    print(dict(scientific_gate=scientific_gate(union,scalar),runtime=runtime),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--math-preflight",action="store_true")
    if p.parse_args().math_preflight:math_preflight()
    else:run()
