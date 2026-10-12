"""One frozen independent confirmation: generate, score and persist in blocks."""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,argparse,json,csv,hashlib,time,traceback,zipfile,subprocess,datetime
import numpy as np
ROOT=Path(r'D:\Users\yuhua\Desktop\Project');OUT=ROOT/'outputs/revision_mr08a_independent_confirmation'
sys.path[:0]=[str(OUT/'src'),str(ROOT)]
CONDS=('ID','Profile','Mobility','Composed','Boundary');SNRS=(0,5,10,15,20,25,30)
METHODS=('SOURCE','POOLED_NATURAL','PPCA_DIRECT_R32','GMM_PREPARED_3','HARD_TW5','CONVEX_TW5','LR4_TW1','LR4_TW5','DENSE_TW5','WHOLE_POOL_M2_R24')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def archive(p,**x):
    with Path(p).open('xb') as f:np.savez_compressed(f,**x)
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def verify():
    f=read(OUT/'reports/execution_freeze.json')
    for p,h in f['new_hashes'].items():
        if sha(OUT/p)!=h:raise ValueError('NEW_EXECUTION_BYTES_CHANGED:'+p)
    for p,h in f['old_hashes'].items():
        if sha(ROOT/p)!=h:raise ValueError('FROZEN_ASSET_CHANGED:'+p)
    return dict(PASS=True,new_frozen_files=len(f['new_hashes']),old_frozen_files=len(f['old_hashes']),modified=0)

def prepare():
    from models import prepare_assets
    from historical_identity import prepare_seeds
    for name in ('src','tests','data','reports','tables','figures'):(OUT/name).mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('PROTOCOL_ALREADY_FROZEN')
    models=read(OUT/'reports/model_assets.json') if (OUT/'reports/model_assets.json').exists() else prepare_assets(ROOT,OUT)
    seeds=prepare_seeds(ROOT,OUT,smoke_per_condition=2)
    suite=read(OUT/'reports/pytest_preflight.json')
    if not suite['PASS']:raise ValueError('PYTEST_PREFLIGHT_REQUIRED')
    config=dict(task='SP-U17/MR-08A independent trajectory confirmation',methods=list(METHODS),conditions={'ID':'CDL-C-30-100','Profile':'CDL-B-30-100','Mobility':'CDL-C-120-100','Composed':'CDL-B-120-100','Boundary':'CDL-C-30-200'},snrs=list(SNRS),per_condition=1000,physical_trajectories=5000,frames=10,scored_frames=list(range(4,10)),method_scored_positions=210000,total_method_scored_positions=2100000,system=dict(Nt=2,Nr=2,K=64,T=16,pilot_dim=64,data_dim=3968,dF=16,dT=4),source_power=9.941134822523972,noise='original Source-referenced variance; same standardized noise across SNR; no condition normalization',method_history={'SOURCE':0,'POOLED_NATURAL':0,'PPCA_DIRECT_R32':0,'GMM_PREPARED_3':0,'LR4_TW1':0,'HARD_TW5':4,'CONVEX_TW5':4,'LR4_TW5':4,'DENSE_TW5':4,'WHOLE_POOL_M2_R24':4},fixed_models=dict(GMM_prior=[1/3]*3,PPCA_rank=32,PPCA_residual_floor=1e-6,correction_rank=4,whole_rank=24,whole_weight='original common Pooled M2',whole_center='original Source mean'),NMSE='sum same-trajectory six-frame error energy/sum same truth energy; dB=10log10 after pooling',primary_families={'A':dict(first='POOLED_NATURAL',second='LR4_TW5',positive='LR4_TW5 better',comparisons=35),'B':dict(first='GMM_PREPARED_3',second='LR4_TW1',positive='LR4_TW1 better',comparisons=35),'C':dict(first='LR4_TW5',second='WHOLE_POOL_M2_R24',positive='Whole24 better',comparisons=35)},bootstrap=dict(replicates=100000,unit='physical trajectory',shared='all6frames,7SNR,10methods per condition',algorithm='default_rng(SeedSequence([new master,condition ordinal])); int32 original draws cast losslessly to uint16 persistent indices',quantile_method='linear',two_sided=True,each_family_size=35,each_family_confidence=.95,tail_probability=.05/(2*35),no_unified_105_comparison_guarantee=True,old_draws_reused=False),smoke=dict(per_condition=2,separate_domain=True,formal_reuse=False,scored_positions_per_method=420,statistics_replicates=100,uncertainty_inference=False),failure_policy='Record exact condition/trajectory/SNR/frame/method and STOP on scientific failure; no dropping/replacement/fallback/retry; technical interruption preserves verified blocks for minimal recovery',models=models,seed_identity=seeds,git_HEAD=subprocess.check_output(['git','-C',str(ROOT),'--no-optional-locks','rev-parse','HEAD'],text=True).strip(),git_uncommitted_source_truth='exact SHA256 of loaded sources and assets; HEAD alone is not code identity',existing_manuscript='unchanged',confirmation_scope='new independent trajectories from existing5 known distributions, not unseen-distribution/OOD proof')
    write(OUT/'protocol.json',config)
    old={}
    for p in (ROOT/'src').glob('*.py'):old[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in (ROOT/'scripts').glob('*.py'):
        if p.name in ('run_sp_u12b2a.py','sp_u12b2a_common.py','run_formal_test.py','resume_formal_test.py','sp_u12b4a_adapter.py','sp_u7a_metadata.py'):old[p.relative_to(ROOT).as_posix()]=sha(p)
    # Modules/assets actually used in the source-disclosed fixed baseline bank.
    for field in ('source_assets','source_code'):
        source=models.get(field,{})
        if isinstance(source,dict):
            for p,h in source.items():
                if isinstance(h,str) and len(h)==64:old[p]=h
        else:
            for r in source:old[r['path']]=r.get('SHA256',r.get('sha256'))
    for p in (ROOT/'paper').rglob('*.tex'):old[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in (ROOT/'paper/bibliography').glob('*.bib'):old[p.relative_to(ROOT).as_posix()]=sha(p)
    fs=list((OUT/'src').glob('*.py'))+list((OUT/'tests').glob('*.py'))+[OUT/'protocol.json',OUT/'reports/pytest_preflight.json',OUT/'reports/seed_independence.json']
    fs+=list((OUT/'data').glob('*seed_registry.json'))+[OUT/'data/historical_H_hashes.json']
    # All instantiated noise-level asset arrays bind before any scientific H.
    fs+=list((OUT/'data').glob('*assets*.npz'))+list((OUT/'data').glob('*operators*.npz'))+list((OUT/'data').glob('*estimators*.npz'))+[OUT/'reports/model_assets.json']
    write(OUT/'reports/execution_freeze.json',dict(new_hashes={p.relative_to(OUT).as_posix():sha(p) for p in fs},old_hashes=old,first_scientific_trajectory_generated=False,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(stage='prepare',PASS=True,methods=10,formal_trajectories=5000,formal_generated=0,preservation=verify())),flush=True)

def make_pilots(h,communication):
    from scripts.sp_u12b2a_common import variance
    from src.cdl_benchmark import frames,observation_geometry,array_hash
    from src.link import assemble_pilot_data_frame,apply_frequency_mimo_channel,estimate_ls_linear_channel
    if h.shape!=(2,2,64,160) or h.dtype!=np.complex128 or not np.isfinite(h).all() or len(communication)!=10:raise ValueError('H_INPUT_SHAPE_DTYPE_FINITE')
    g=observation_geometry();hf=frames(h);p=np.empty((7,10,64),np.complex128);ids=[]
    for j,(bits,symbols,z) in enumerate(communication):
        if z.shape!=(2,64,16) or z.dtype!=np.complex128 or not np.isfinite(z).all():raise ValueError('NOISE_INPUT_SHAPE_DTYPE_FINITE')
        clean=apply_frequency_mimo_channel(hf[j],assemble_pilot_data_frame(symbols,g.layout))
        for si,s in enumerate(SNRS):
            y=clean+np.sqrt(variance(s))*z;sparse,_=estimate_ls_linear_channel(y,g.layout);p[si,j]=sparse.ravel(order='C')[g.selected]
            ids.append(dict(frame_id=j,snr_db=s,truth_frame_SHA256=array_hash(hf[j]),unit_noise_SHA256=array_hash(z),transmitted_symbols_SHA256=array_hash(symbols),bits_SHA256=array_hash(bits),received_SHA256=array_hash(y),pilot_SHA256=array_hash(p[si,j])))
    truth=np.stack([hf[j].ravel(order='C')[g.data] for j in range(4,10)])
    if not np.isfinite(p).all() or truth.shape!=(6,3968):raise ValueError('PILOT_TRUTH_SHAPE_FINITE')
    return p,truth,ids

def run(mode):
    verify();from models import load_assets
    from scripts.run_sp_u12b2a import generator,seed_index,consumed
    from scripts.sp_u12b2a_common import communicate
    from scripts.sp_u12b4a_adapter import rss,blas_threads
    from src.cdl_benchmark import array_hash
    protocol=read(OUT/'protocol.json');n=2 if mode=='smoke' else 1000
    if mode=='formal' and not read(OUT/'reports/smoke_chain.json')['PASS']:raise ValueError('FULL_SMOKE_CHAIN_REQUIRED')
    if list(OUT.glob('reports/*STOP*.json')):raise ValueError('STOP_REQUIRES_EXPLICIT_TECHNICAL_REVIEW')
    bank=load_assets(ROOT,OUT);reg=read(OUT/'data'/f'{mode}_seed_registry.json');idx=seed_index(reg)
    forbidden=set(read(OUT/'data/historical_H_hashes.json')['H_sha256']);seen=set()
    if mode=='formal':forbidden.update(r['H_sha256'] for r in read(OUT/'reports/smoke_receipt.json')['receipts'])
    folder=OUT/'data'/mode;folder.mkdir(exist_ok=True);recs=[];begin=time.monotonic();determinism=False
    for c in CONDS:
        gen=generator(c,idx,'EVAL');gp=OUT/'reports'/f'{mode}_{c}_generator.json'
        if not gp.exists():write(gp,gen.provenance())
        for i in range(n):
            path=folder/f'{c}_{i:04d}.npz';jp=path.with_suffix('.json');started=time.monotonic();identity=f'{reg["namespace"]}:{c}:{i}'
            if path.exists() or jp.exists():
                if not(path.exists() and jp.exists()):raise ValueError('INCOMPLETE_PERSISTED_BLOCK:'+identity)
                meta=read(jp)
                if not meta['complete'] or meta['identity']!=identity or meta['protocol_SHA256']!=sha(OUT/'protocol.json') or meta['raw_SHA256']!=sha(path):raise ValueError('EXISTING_BLOCK_IDENTITY_OR_HASH:'+identity)
                with zipfile.ZipFile(path) as z:
                    if z.testzip():raise ValueError('PERSISTED_CRC:'+identity)
                hh=meta['input']['H_sha256']
                if hh in forbidden or hh in seen:raise ValueError('EXISTING_PHYSICAL_OVERLAP:'+identity)
                seen.add(hh);recs.append(dict(condition=c,trajectory_id=i,path=path.relative_to(OUT).as_posix(),SHA256=sha(path),JSON_SHA256=sha(jp),H_sha256=hh,technical_resume_verified=True));continue
            try:
                h=gen.generate(i);hh=array_hash(h)
                if hh in forbidden or hh in seen:raise ValueError('NEW_PHYSICAL_IDENTITY_OVERLAP')
                seen.add(hh)
                comm=[communicate(consumed(idx,c,'EVAL',i,j)) for j in range(10)]
                p,truth,input_hashes=make_pilots(h,comm);ph=array_hash(p);th=array_hash(truth)
                result=bank.predict(p)
                pred=result.pop('predictions')
                if pred.shape!=(7,10,6,3968) or pred.dtype!=np.complex128 or not np.isfinite(pred).all():raise ValueError('SCIENTIFIC_PREDICTION_SHAPE_DTYPE_FINITE')
                if array_hash(p)!=ph or array_hash(truth)!=th:raise ValueError('SHARED_INPUT_MUTATED')
                if mode=='smoke' and not determinism:
                    check=bank.predict(p)
                    if not np.array_equal(pred,check.pop('predictions')) or set(check)!=set(result):raise ValueError('DETERMINISTIC_PREDICTION_FAILED')
                    for k in result:
                        if not np.array_equal(np.asarray(result[k]),np.asarray(check[k])):raise ValueError('DETERMINISTIC_DIAGNOSTICS_FAILED:'+k)
                    determinism=True
                for key,expected in (('methods',METHODS),('snrs',SNRS),('frame_ids',tuple(range(4,10)))):
                    if tuple(result.pop(key))!=expected:raise ValueError('SCIENTIFIC_METHOD_OR_SCORING_AXIS_DRIFT:'+key)
                error=np.sum(abs(pred-truth[None,None,:,:])**2,axis=-1,dtype=np.float64);q=np.sum(abs(truth)**2,axis=-1,dtype=np.float64)
                if error.shape!=(7,10,6) or q.shape!=(6,) or not np.isfinite(error).all() or np.any(error<0) or np.any(q<=0):raise ValueError('SCIENTIFIC_ENERGY_INVALID')
                arrays={k:np.asarray(v) for k,v in result.items()}
                if any(a.dtype.kind=='O' for a in arrays.values()):raise ValueError('OBJECT_DIAGNOSTIC_SERIALIZATION')
                archive(path,frame_error=error,frame_truth=q,snr_db=np.array(SNRS),frame_ids=np.arange(4,10),method_names=np.array(METHODS),failure_flags=np.zeros((7,10,6),bool),**arrays)
                write(jp,dict(mode=mode,condition=c,trajectory_id=i,identity=identity,input=dict(H_sha256=hh,pilots_SHA256=ph,truth_scored_SHA256=th,channel_seeds=consumed(idx,c,'EVAL',i),actual_channel_roles=['sionna_seed','torch_seed','numpy_seed'],frame_seeds=[consumed(idx,c,'EVAL',i,j) for j in range(10)],paired_input_hashes=input_hashes),raw_SHA256=sha(path),protocol_SHA256=sha(OUT/'protocol.json'),seed_registry_SHA256=sha(OUT/'data'/f'{mode}_seed_registry.json'),methods=list(METHODS),common_truth_for_all_methods_and_SNR=True,complete=True,failures=0,nonfinite=0,elapsed_seconds=time.monotonic()-started))
                recs.append(dict(condition=c,trajectory_id=i,path=path.relative_to(OUT).as_posix(),SHA256=sha(path),JSON_SHA256=sha(jp),H_sha256=hh,technical_resume_verified=False))
            except BaseException as e:
                write(OUT/'reports'/f'{mode}_{c}_{i:04d}_STOP_{time.time_ns()}.json',dict(status='STOP',mode=mode,condition=c,trajectory_id=i,identity=identity,channel_seeds=consumed(idx,c,'EVAL',i,-1),error=repr(e),traceback=traceback.format_exc(),dropped=False,replaced=False,method_changed=False,completed_blocks=len(recs),statistics_allowed=False));raise
            if i%50==0 or i==n-1:print(json.dumps(dict(stage=mode,condition=c,completed_in_condition=i+1,trajectories_total=len(recs),total_planned=5*n,elapsed_seconds=time.monotonic()-begin,resource=rss())),flush=True)
            del h,comm,p,truth,result,pred,error,arrays
    if len(seen)!=5*n or len(recs)!=5*n:raise ValueError('CANONICAL_POPULATION_INCOMPLETE')
    verify();write(OUT/'reports'/f'{mode}_receipt.json',dict(PASS=True,identity_PASS=True,mode=mode,physical_trajectories=5*n,unique_H_identities=len(seen),per_condition=n,method_count=10,scored_positions_per_method=5*n*7*6,total_method_scored_positions=5*n*7*6*10,failures=0,nonfinite=0,missing=0,substituted=0,smoke_mixed_into_formal=False,deterministic_prediction_check=determinism if mode=='smoke' else 'previous independent smoke PASS',protocol_SHA256=sha(OUT/'protocol.json'),receipts=recs,elapsed_seconds=time.monotonic()-begin,resource=rss(),BLAS=blas_threads(),raw_saved_before_statistics=True))
    print(json.dumps(dict(stage=mode,PASS=True,physical_trajectories=len(recs))),flush=True)

def analyze(mode):
    verify();from statistics_mr08a import analyze_dataset
    reg=read(OUT/'data'/f'{mode}_seed_registry.json');masters=[r['seed'] for r in reg['rows'] if r['role']=='bootstrap_master_seed']
    if len(masters)!=1:raise ValueError('BOOTSTRAP_MASTER_IDENTITY')
    result=analyze_dataset(OUT,mode,masters[0],replicates=100 if mode=='smoke' else 100000,n=2 if mode=='smoke' else 1000)
    if mode=='smoke':
        # Full plotting schema is checked before formal data generation. No
        # smoke intervals or performance sign is promoted to confirmation.
        receipt=read(OUT/'reports/smoke_receipt.json')
        with (OUT/'tables/smoke_nmse.csv').open(encoding='utf-8-sig',newline='') as f:points=list(csv.DictReader(f))
        with (OUT/'tables/smoke_primary_contrasts.csv').open(encoding='utf-8-sig',newline='') as f:contrasts=list(csv.DictReader(f))
        if len(points)!=len(CONDS)*len(SNRS)*len(METHODS) or len(contrasts)!=3*len(CONDS)*len(SNRS):raise ValueError('SMOKE_PLOT_GRID_INCOMPLETE')
        if {(r['condition'],int(r['snr_db']),r['method']) for r in points}!={(c,s,m) for c in CONDS for s in SNRS for m in METHODS}:raise ValueError('SMOKE_PLOT_IDENTITY_GRID')
        if not all(np.isfinite(float(r['nmse_db'])) for r in points):raise ValueError('SMOKE_PLOT_NONFINITE')
        write(OUT/'reports/smoke_chain.json',dict(PASS=True,raw=receipt['PASS'],identity=receipt['identity_PASS'],determinism=receipt['deterministic_prediction_check'],statistics_readback=True,all_methods_conditions_SNR=True,plot_input_schema_checked=True,formal_scientific_positions=0,smoke_never_reused=True,statistics=result))
    print(json.dumps(dict(stage=mode+'_analysis',PASS=True,result=result),ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','formal','smoke-analysis','analysis']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='smoke-analysis':analyze('smoke')
    elif a.action=='analysis':analyze('formal')
    else:run(a.action)
if __name__=='__main__':main()
