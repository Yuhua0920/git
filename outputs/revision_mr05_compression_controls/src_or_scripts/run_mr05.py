"""MR05 disclosed-development controls: no channel generation, no method tuning."""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import argparse,csv,hashlib,importlib.metadata,json,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"outputs/revision_mr05_compression_controls"
MR1=ROOT/"outputs/revision_mr01_pooled_lmmse"
MR2=ROOT/"outputs/revision_mr02_coordinate_mechanism"
sys.path[:0]=[str(ROOT),str(Path(__file__).parent),str(MR2/"src_or_scripts")]
from scripts import sp_u12b2a_common as common
from src.cdl_benchmark import frames,observation_geometry,array_hash
from src.lowrank_modules import RankFactors,WeightedModule
from scripts.sp_u12b4a_adapter import blas_threads,rss
import run_mr02 as prior
from controls import METHODS,weighted_factor,deployed

SNRS=(10,20,30)
CONDITIONS=("ID","Profile","Mobility","Composed","Boundary")
NEW=METHODS[4:]
CONTROL=METHODS[3:]
RTOL=1e-10
ATOL=1e-10
OLD_INDEX=(0,1,4,5)
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf8") as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def archive(p,**a):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("xb") as f:np.savez_compressed(f,**a)
    return dict(path=p.relative_to(OUT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
def csvout(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows or any(set(r)!=set(rows[0]) for r in rows):raise ValueError("CSV_SCHEMA")
    with p.open("x",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def close(a,b,label,scale=None):
    a=np.asarray(a);b=np.asarray(b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError("NONFINITE:"+label)
    err=float(np.max(np.abs(a-b)))
    den=max(float(np.max(np.abs(a))),float(np.max(np.abs(b))),0 if scale is None else float(scale))
    if err>ATOL+RTOL*den:raise ValueError(label+":"+str((err,den)))
    return dict(max_abs=err,scaled=err/(1+den))
def source_input(condition,i):
    prior.ROOT=ROOT;prior.MR=MR1;prior.ah=array_hash;prior.c=common
    raw,binding=prior.source_input("development",condition,i)
    if raw["H"].shape!=(2,2,64,160) or raw["H"].dtype!=np.complex128:raise ValueError("H_SHAPE_DTYPE")
    if raw["pilots"].shape!=(3,10,64) or raw["coordinates"].shape!=(3,6,2):raise ValueError("INPUT_AXES")
    if not np.array_equal(raw["method_names"],np.array(("Source","Pooled","Hard","Convex","Dense","LR4","Pooled-source-centered"))):raise ValueError("ORIGINAL_METHOD_AXIS")
    if raw["frame_error"].shape!=(3,7,6) or raw["frame_truth"].shape!=(6,):raise ValueError("ENERGY_AXES")
    if any(not np.isfinite(raw[k]).all() for k in ("H","pilots","unit_noise","coordinates","frame_error","frame_truth","normal_residual")):raise ValueError("NONFINITE_ORIGINAL")
    if np.any(raw["frame_truth"]<=0) or np.max(raw["normal_residual"])>common.SOLVE_TOL:raise ValueError("ORIGINAL_ENERGY_SOLVE")
    return raw,binding
def verify_sources():
    path=OUT/"reports/execution_freeze_after_test_fixture_fix.json"
    if not path.exists():path=OUT/"reports/execution_freeze_after_timing_interface_fix.json"
    if not path.exists():path=OUT/"reports/execution_freeze_after_serialization_fix.json"
    f=read(path if path.exists() else OUT/"reports/execution_freeze.json")
    for p,h in f["hashes"].items():
        if sha(OUT/p)!=h:raise ValueError("NEW_CODE_CHANGED:"+p)
    for p,h in read(OUT/"reports/preservation_before.json")["hashes"].items():
        if sha(ROOT/p)!=h:raise ValueError("PRESERVED_INPUT_CHANGED:"+p)
    return dict(PASS=True,files=len(read(OUT/"reports/preservation_before.json")["hashes"]),modified=0)
def prepare():
    if tuple(common.CONDITIONS)!=CONDITIONS:raise ValueError("CONDITIONS_CHANGED")
    config=dict(identity="MR05 FIRST-STAGE DISCLOSED DEVELOPMENT",
        methods=list(METHODS),conditions=list(CONDITIONS),snrs=list(SNRS),
        physical_trajectories=500,per_condition=100,frame_count=10,scored_frames=list(range(4,10)),
        same_saved_coordinates=True,new_coordinate_fits_for_inference=0,
        weights_from_preparation_only=True,no_final_design_selection=True,
        source_power=common.POWER,sigma2={str(s):common.variance(s) for s in SNRS},
        rank_correction=4,whole_ranks=[23,24],rtol=RTOL,atol=ATOL,
        identity_scale="abs(A)+abs(B)+E_control+E_Dense, no clipping",
        bootstrap=dict(resamples=2000,source="MR01 saved indices",new_draws=0,
          confidence=.95,adjusted=False,quantile_method="linear",unit="physical trajectory"),
        timing=dict(warmup_calls=30,windows_per_snr=100,total_distinct_windows=300,blocks=10,order_seed=170505,
          threads=1,endpoint_names=["deployment-only","full-estimator-core"],
          windows="all conditions x trajectories0..19 x SNR10/20/30, frame4",
          exclude="generation, disk, logging, hashing, plotting"))
    write(OUT/"config.json",config)
    oldbind=read(MR2/"reports/preservation_before.json")["hashes"]
    oldbind.update(read(MR1/"reports/execution_freeze.json")["scientific_sources"])
    paths=[MR1/"data"/n for n in ("operators.npz","pooled_statistics.npz",
           "seed_registry.json","exploratory_bootstrap_indices.npz","verified_trajectory_energies.npz")]
    paths += [MR1/"reports"/n for n in ("protocol.json","development_summary.json","pooled_construction.json")]
    paths += [MR1/"tables"/n for n in ("core_results.csv","paired_contrasts.csv","nmse_all_methods.csv")]
    paths += [MR2/"data/eval_summary.npz",MR2/"reports/evaluation_receipt.json",MR2/"reports/protocol.json"]
    paths += [ROOT/"outputs"/n for n in common.STATS.values()]
    paths += list((ROOT/"src").glob("*.py"))
    paths += [ROOT/"scripts/sp_u12b2a_common.py",ROOT/"scripts/sp_u12b4a_adapter.py",
              ROOT/"scripts/run_sp_u12b2a.py",MR2/"src_or_scripts/run_mr02.py"]
    paths += list((ROOT/"outputs/paper_v4_2_candidate").rglob("*.tex"))
    paths += [ROOT/"outputs/paper_v4_2_candidate/main.pdf",ROOT/"outputs/paper_v4_2_candidate/supplement.pdf"]
    preserved={}
    for p in sorted(set(paths)):
        rel=p.relative_to(ROOT).as_posix();h=sha(p)
        if rel in oldbind and h!=oldbind[rel]:raise ValueError("OLD_HASH_BINDING:"+rel)
        preserved[rel]=h
    summary=read(MR1/"reports/development_summary.json")
    for row in summary["receipts"]:
        _,_,condition,idx=row["identity"].split(":")
        for suffix,key in ((".npz","npz_sha256"),(".json","json_sha256")):
            p=MR1/"data/development"/f"{condition}_{int(idx):04d}{suffix}"
            preserved[p.relative_to(ROOT).as_posix()]=row[key]
    if len(summary["receipts"])!=500:raise ValueError("OLD_GRID")
    write(OUT/"reports/preservation_before.json",dict(hashes=preserved))
    with np.load(MR1/"data/exploratory_bootstrap_indices.npz",allow_pickle=False) as z:
        draws={k:z[k] for k in z.files}
    if set(draws)!=set(CONDITIONS) or any(v.shape!=(2000,100) or v.dtype!=np.int64 or v.min()<0 or v.max()>99 for v in draws.values()):raise ValueError("OLD_BOOTSTRAP_AXES")
    archive(OUT/"data/bootstrap_indices.npz",**draws)
    env=dict(python=sys.version,executable=sys.executable,
        packages={k:importlib.metadata.version(k) for k in ("numpy","scipy","pytest","matplotlib")},
        active_blas=blas_threads(),resource=rss(),
        manuscript_baseline="outputs/paper_v4_2_candidate",
        review_named_pdf_aliases="main(1).pdf/supplement(1).pdf not found; current V4.2 canonical PDFs/source chain used; no manuscript edits")
    write(OUT/"reports/environment.json",env)
    hashes={p.relative_to(OUT).as_posix():sha(p) for p in (OUT/"src_or_scripts").glob("*.py") if p.name not in ("plot_mr05.py","finalize_mr05.py")}
    hashes.update({p.relative_to(OUT).as_posix():sha(p) for p in (OUT/"tests").glob("*.py")})
    hashes["config.json"]=sha(OUT/"config.json")
    write(OUT/"reports/execution_freeze.json",dict(hashes=hashes,first_control_inference=False))
    print(json.dumps(dict(action="prepare",PASS=True,protected_files=len(preserved))),flush=True)
def load_bank():
    verify_sources()
    receipt=read(OUT/"reports/factor_construction.json")
    if sha(OUT/receipt["cache"]["path"])!=receipt["cache"]["sha256"]:raise ValueError("FACTOR_CACHE_CHANGED")
    op=dict(np.load(MR1/"data/operators.npz",allow_pickle=False))
    cache=dict(np.load(OUT/"data/control_factors.npz",allow_pickle=False))
    bank={}
    for s in SNRS:
        bank[s]={}
        for method in CONTROL:
            count=3 if method.startswith("WHOLE") else 2
            bank[s][method]=tuple(RankFactors(cache[f"{method}_{s}_{j}_A"],cache[f"{method}_{s}_{j}_Q"]) for j in range(count))
            rank=int(method.rsplit("R",1)[1])
            if any(f.A.shape!=(3968,rank) or f.Q.shape!=(rank,64) or f.A.dtype!=np.complex128 or f.Q.dtype!=np.complex128 for f in bank[s][method]):raise ValueError("FROZEN_FACTOR_DIMENSION")
            if method in ("CORR_POOL_M2_R4","WHOLE_POOL_M2_R23","WHOLE_POOL_M2_R24"):
                for j in range(count):
                    if not np.array_equal(cache[f"{method}_{s}_{j}_C"],cache[f"CORR_POOL_M2_R4_{s}_0_C"]):raise ValueError("COMMON_POOL_WEIGHT")
    for v in op.values():v.flags.writeable=False
    return op,bank
def construct():
    verify_sources()
    op=dict(np.load(MR1/"data/operators.npz",allow_pickle=False))
    pool=dict(np.load(MR1/"data/pooled_statistics.npz",allow_pickle=False))
    stats={k:common.statistics(ROOT/"outputs"/v) for k,v in common.STATS.items()}
    recomposed=sum((st.rpp+np.outer(st.mean_p-pool["mean_p"],(st.mean_p-pool["mean_p"]).conj()))/3 for st in stats.values())
    pool_check=close(recomposed,pool["rpp"],"POOL_BETWEEN_ORIGIN_MOMENTS")
    close(pool["mean_p"],op["pool_mean_p"],"POOL_MEAN")
    arrays={};audits=[];offline=[]
    for s in SNRS:
        sig=common.variance(s);ident=np.eye(64,dtype=np.complex128)
        dp=pool["mean_p"]-op["source_mean_p"]
        Cp=pool["rpp"]+sig*ident+np.outer(dp,dp.conj())
        for m in ("P","M"):
            delta=op[f"DeltaW{m}_{s}"];st=stats[m];shift=st.mean_p-op["source_mean_p"]
            weights={"CORR_ORIGIN_R4":st.rpp+sig*ident,
                     "CORR_I_R4":ident,
                     "CORR_ORIGIN_M2_R4":st.rpp+sig*ident+np.outer(shift,shift.conj()),
                     "CORR_POOL_M2_R4":Cp}
            for method,C in weights.items():
                factor,audit=weighted_factor(delta,C,4)
                audit.update(method=method,snr_db=s,operator=m,frozen_operator_max_abs=None,frozen_operator_scaled=None)
                if method=="CORR_ORIGIN_R4":
                    frozen=RankFactors(op[f"A_{m}_{s}"],op[f"Q_{m}_{s}"])
                    comparison=close(factor.A@factor.Q,frozen.A@frozen.Q,"ORIGIN_FROZEN_RECONSTRUCTION")
                    audit["frozen_operator_max_abs"]=comparison["max_abs"]
                    audit["frozen_operator_scaled"]=comparison["scaled"]
                    factor=frozen
                audits.append(audit)
                j=("P","M").index(m)
                arrays[f"{method}_{s}_{j}_A"]=factor.A;arrays[f"{method}_{s}_{j}_Q"]=factor.Q
                arrays[f"{method}_{s}_{j}_C"]=C
        for j,q in enumerate(("source","P","M")):
            matrix=op[f"W{q}_{s}"];module=WeightedModule.build(matrix,Cp)
            for r in (23,24):
                method=f"WHOLE_POOL_M2_R{r}"
                factor,audit=weighted_factor(matrix,Cp,r,module=module)
                audit.update(method=method,snr_db=s,operator=q,frozen_operator_max_abs=None,frozen_operator_scaled=None)
                audits.append(audit)
                arrays[f"{method}_{s}_{j}_A"]=factor.A;arrays[f"{method}_{s}_{j}_Q"]=factor.Q
                arrays[f"{method}_{s}_{j}_C"]=Cp
    cache_receipt=archive(OUT/"data/control_factors.npz",**arrays)
    csvout(OUT/"tables/factor_audit.csv",audits)
    write(OUT/"reports/factor_construction.json",dict(PASS=True,pool_recomposition=pool_check,
       weights_preparation_only=True,clean_eval_weight_access=0,
       original_frozen_factor_reused=True,cache=cache_receipt,audits=audits,resource=rss()))
    print(json.dumps(dict(action="construct",PASS=True,audits=len(audits))),flush=True)
def evaluate_one(condition,i,op,bank):
    raw,binding=source_input(condition,i);hf=frames(raw["H"]);g=observation_geometry()
    error=np.empty((3,9,6));truth=raw["frame_truth"].copy()
    A=np.zeros((3,9,6));B=np.zeros((3,9,6))
    checks=dict(replay_max_abs=0.,replay_scaled=0.,identity_max_abs=0.,identity_scaled=0.,original_energy_bitwise=True)
    example=np.empty((3,9,16),dtype=np.complex128) if i==0 else None
    example_hash=np.empty((3,9),dtype="U64") if i==0 else None
    for mi,old in enumerate(OLD_INDEX):error[:,mi]=raw["frame_error"][:,old]
    for si,s in enumerate(SNRS):
        for ji,j in enumerate(range(4,10)):
            target=hf[j].ravel()[g.data];u=raw["pilots"][si,j]-op["source_mean_p"];cc=raw["coordinates"][si,ji]
            before=(array_hash(u),array_hash(cc));pred={}
            for method in METHODS:pred[method]=deployed(method,u,cc,op,bank[s],s)
            if i==0 and j==4:
                for mi,method in enumerate(METHODS):
                    example[si,mi]=pred[method][:16];example_hash[si,mi]=array_hash(pred[method])
            eD=pred["DENSE_TW5"]-target;ED=float(np.vdot(eD,eD).real)
            truth_check=close(float(np.vdot(target,target).real),truth[ji],"TRUTH_REPLAY")
            for mi,method in enumerate(METHODS):
                e=pred[method]-target;direct=float(np.vdot(e,e).real)
                if mi<4:
                    chk=close(direct,error[si,mi,ji],"ORIGINAL_ENERGY_REPLAY")
                    checks["replay_max_abs"]=max(checks["replay_max_abs"],chk["max_abs"],truth_check["max_abs"])
                    checks["replay_scaled"]=max(checks["replay_scaled"],chk["scaled"],truth_check["scaled"])
                    checks["original_energy_bitwise"] &= bool(direct==error[si,mi,ji])
                else:error[si,mi,ji]=direct
                delta=pred[method]-pred["DENSE_TW5"]
                A[si,mi,ji]=float(np.vdot(delta,delta).real)
                B[si,mi,ji]=float(2*np.vdot(eD,delta).real)
                scale=A[si,mi,ji]+abs(B[si,mi,ji])+direct+ED
                chk=close(direct-ED,A[si,mi,ji]+B[si,mi,ji],"SIGNED_RISK_IDENTITY",scale)
                checks["identity_max_abs"]=max(checks["identity_max_abs"],chk["max_abs"])
                checks["identity_scaled"]=max(checks["identity_scaled"],chk["scaled"])
            if before!=(array_hash(u),array_hash(cc)):raise ValueError("SHARED_INPUT_MUTATION")
    if not np.isfinite(error).all() or np.any(error<0):raise ValueError("CONTROL_ENERGY")
    a=dict(frame_error=error,frame_truth=truth,A=A,B=B,
        coordinates=raw["coordinates"].copy(),snr_db=np.array(SNRS),frame_ids=np.arange(4,10),
        method_names=np.array(METHODS),input_pilot_sha256=np.array(array_hash(raw["pilots"])),
        coordinate_sha256=np.array(array_hash(raw["coordinates"])))
    if i==0:a.update(vector_example=example,vector_full_sha256=example_hash,vector_example_frame=np.array(4),vector_example_data_indices=np.arange(16))
    return a,binding,checks
def run(action,smoke_tag="smoke"):
    op,bank=load_bank();n=1 if action=="smoke" else 100;t=time.monotonic();receipts=[];checks={}
    namespace=smoke_tag if action=="smoke" else action
    for condition in CONDITIONS:
        for i in range(n):
            arrays,binding,audit=evaluate_one(condition,i,op,bank)
            p=OUT/"data"/namespace/f"{condition}_{i:04d}.npz"
            payload=archive(p,**arrays)
            write(p.with_suffix(".json"),dict(identity=f"MR05:{action}:{condition}:{i}",condition=condition,
                trajectory_id=i,input=binding,raw=payload,audit=audit,
                original_coordinates_reused=True,method_axis=list(METHODS),complete=True))
            receipts.append(dict(condition=condition,trajectory_id=i,**payload,json_sha256=sha(p.with_suffix(".json"))))
            for key,value in audit.items():
                checks[key]=(checks.get(key,True) and value) if isinstance(value,bool) else max(checks.get(key,0.),value)
            if i%20==0:print(json.dumps(dict(action=action,condition=condition,completed=i+1,n=n)),flush=True)
    write(OUT/f"reports/{namespace}_receipt.json",dict(PASS=True,receipts=receipts,
        physical_trajectories=5*n,scored_positions_per_method=5*n*3*6,
        reused_energy_positions=4*5*n*3*6,new_control_positions=5*5*n*3*6,
        check_replay_positions=4*5*n*3*6,elapsed_seconds=time.monotonic()-t,
        audit=checks,resource=rss(),all_raw_saved_before_aggregate=True))
    print(json.dumps(dict(action=action,PASS=True,seconds=time.monotonic()-t,audit=checks)),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=("prepare","construct","smoke","development"))
    p.add_argument("--smoke-tag",choices=("smoke","smoke_after_json_fix"),default="smoke")
    args=p.parse_args()
    for d in ("src_or_scripts","tests","reports","tables","data","figures"):(OUT/d).mkdir(parents=True,exist_ok=True)
    try:globals()[args.action]() if args.action in ("prepare","construct") else run(args.action,args.smoke_tag)
    except Exception as e:
        write(OUT/f"reports/{args.action}_failure_{time.time_ns()}.json",dict(action=args.action,
           status="STOP",error=repr(e),traceback=traceback.format_exc(),no_old_files_written=True))
        raise
if __name__=="__main__":main()
