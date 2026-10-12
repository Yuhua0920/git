"""MR03 staged PPCA controls; all writes confined to this new output directory."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,csv,hashlib,sys,time,argparse,traceback,zipfile,ast
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/revision_mr03_ppca_capacity_controls'
MR1=ROOT/'outputs/revision_mr01_pooled_lmmse';MR2=ROOT/'outputs/revision_mr02_coordinate_mechanism';MR5=ROOT/'outputs/revision_mr05_compression_controls'
sys.path[:0]=[str(ROOT),str(Path(__file__).parent),str(MR5/'src_or_scripts'),str(MR1/'src_or_scripts')]
from scripts import sp_u12b2a_common as common
from src.cdl_benchmark import array_hash,frames,observation_geometry
from scripts.sp_u12b4a_adapter import rss,blas_threads
from capacity_controls import FrameControls,make_received_pilots,BACKGROUND_METHODS,V_NAMESPACE,SMOKE_NAMESPACE
CONDITIONS=('ID','Profile','Mobility','Composed','Boundary');SNRS=(10,20,30)
METHODS=list(BACKGROUND_METHODS)+['DENSE_TW1','LR4_TW1','PPCA_DIRECT_R16','PPCA_DIRECT_R32','PPCA_DIRECT_R64']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def archive(p,**x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.savez_compressed(f,**x)
    return dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
def csvout(p,rows,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def close(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('NONFINITE:'+label)
    err=float(np.max(np.abs(a-b)));scale=max(float(np.max(np.abs(a))),float(np.max(np.abs(b))))
    if err>1e-10+1e-10*scale:raise ValueError(label+str((err,scale)))
    return dict(max_abs=err,scaled=err/(1+scale))
def verify():
    f=read(OUT/'reports/execution_freeze.json')
    for p,h in f['hashes'].items():
        if sha(OUT/p)!=h:raise ValueError('NEW_SOURCE_CHANGED:'+p)
    for p,h in read(OUT/'reports/preservation_before.json')['hashes'].items():
        if sha(ROOT/p)!=h:raise ValueError('OLD_SOURCE_CHANGED:'+p)
    return dict(PASS=True,modified=0,files=len(read(OUT/'reports/preservation_before.json')['hashes']))

def prepare():
    from seed_union import build_forbidden_seed_union,serializable_report
    from scripts.run_sp_u12b2a import allocate
    import run_mr05 as old
    old.verify_sources()
    source=read(OUT/'reports/input_provenance.json')
    if not source['training']['manifest_binding_matches'] or source['training']['sha256']!='2553aab058511770ca95036d6e0d0e7fee9248080b43e826f672ea156099aed0':raise ValueError('P_TRAINING_BINDING')
    # Both recommendation scopes and all methods are fixed before V data/results.
    config=dict(task='MR03 Stage1 capacity and current-frame controls',methods=METHODS,planned_but_missing=['MFA_K1_R16_EM_FROZEN'],conditions=list(CONDITIONS),canonical_definitions=common.CONDITIONS,snrs=list(SNRS),origin_conditions=['ID','Profile','Mobility'],validation_scopes=['origin-only','expanded'],rank_grid=[16,32,64],K=1,nu_floor=1e-6,covariance='proper complex column covariance; row storage Z.T @ Z.conj()/N',sample_weight='equal frame marginal observations; N=15000, not 15000 independent trajectories',joint_layout='authenticated ascending union of pilot/data full indices, not contiguous pilot first',source_noise_power=common.POWER,sigma2={str(s):common.variance(s) for s in SNRS},frames=list(range(10)),scored_frames=list(range(4,10)),per_condition=100,per_dataset_physical_trajectories=500,atol=1e-10,rtol=1e-10,eigen_relative_tolerance=1e-9,resource=dict(RSS_limit_bytes=8*1024**3,system_headroom_bytes=3*1024**3,spectral_timeout_seconds=1800,construct_BLAS_threads=6,inference_BLAS_threads=1),bootstrap=dict(resamples=2000,confidence=.95,adjusted=False,method='linear percentile',unit='physical trajectory',D_eval='exact original MR01 indices',V_new='independent frozen condition substreams'),validation_recommendation='minimum macro linear pooled NMSE; exact ties lower rank; no final deployment selection',no_T_new=True,no_MFA_training=True,no_original_model_modification=True,EM_entity_status='PENDING_ORIGINAL_MODEL',prediction_interface='pilot observations and fixed operators/noise only; no truth/labels',Tw1_real_rows=128,Tw5_real_rows=640)
    write(OUT/'config.json',config)
    oldhash=read(MR5/'reports/preservation_before.json')['hashes']
    # Include MR05 factors/raw small results and sources; do not reopen formal outcomes.
    add=[MR5/'data/control_factors.npz',MR5/'reports/factor_construction.json',MR5/'reports/development_receipt.json',MR5/'tables/nmse.csv',MR5/'tables/contrasts.csv',MR5/'data/bootstrap_indices.npz',ROOT/'FINAL_MODEL_MANIFEST.json',ROOT/'FINAL_MODEL_MANIFEST.sha256',ROOT/'SP_U17_MR03_Stage1_PPCA_Capacity_Controls.md']
    add+=list((MR5/'src_or_scripts').glob('*.py'))
    for r in read(MR5/'reports/development_receipt.json')['receipts']:
        for p,h in [(MR5/r['path'],r['sha256']),((MR5/r['path']).with_suffix('.json'),r['json_sha256'])]:
            if sha(p)!=h:raise ValueError('MR05_RAW_CHANGED')
            oldhash[p.relative_to(ROOT).as_posix()]=h
    for p in add:oldhash[p.relative_to(ROOT).as_posix()]=sha(p)
    write(OUT/'reports/preservation_before.json',dict(hashes=oldhash))
    ex=build_forbidden_seed_union(ROOT,with_report=True);forbidden=ex['forbidden_seeds'];later=set()
    for split in read(MR1/'data/seed_registry.json').values():later.update(r['seed'] for r in split['rows'])
    later.update(r['seed'] for r in read(MR2/'data/reference_seed_registry.json')['rows'])
    # Reserve all positive integer literals in directly relevant synthetic tests;
    # conservative superset is a pre-data collision exclusion, not seed tuning.
    literal_files=list((MR5/'tests').glob('*.py'))+list((OUT/'tests').glob('*.py'))
    for p in literal_files:
        later.update(n.value for n in ast.walk(ast.parse(p.read_text(encoding='utf8'))) if isinstance(n,ast.Constant) and type(n.value) is int and 0<=n.value<2**32)
    later.add(170505)
    forbidden |= later | {v+d for v in later for d in (-1,1) if 0<=v+d<2**32}
    used=set();smoke=allocate(SMOKE_NAMESPACE,1,0,forbidden,used,False);V=allocate(V_NAMESPACE,100,0,forbidden,used,True)
    write(OUT/'data/validation_seed_registry.json',V);write(OUT/'data/smoke_seed_registry.json',smoke)
    audit=serializable_report(ex)
    write(OUT/'reports/seed_independence.json',dict(PASS=True,historical_scientific_scope=audit,added_MR01_MR02_and_engineering_exclusions=len(later),new_seed_count=len(used),new_anyrole_overlap=0,SDK_neighbor_overlap=0,namespaces=[SMOKE_NAMESPACE,V_NAMESPACE],engineering_scope='named historical registries and literals in directly relevant old/new tests; no claim of exhaustive undocumented external random states',performance_read=False))
    with np.load(MR1/'data/exploratory_bootstrap_indices.npz',allow_pickle=False) as z:old_draws={k:z[k] for k in z.files}
    archive(OUT/'data/bootstrap_D_eval.npz',**old_draws)
    master=next(r['seed'] for r in V['rows'] if r['role']=='bootstrap_master_seed')
    draws={c:np.random.default_rng(np.random.SeedSequence([master,j])).integers(0,100,size=(2000,100),dtype=np.int64) for j,c in enumerate(CONDITIONS)}
    archive(OUT/'data/bootstrap_V_new.npz',**draws)
    historical=set(read(MR1/'reports/b2_H_identity_readonly.json')['H_sha256'])
    for p in (MR1/'data/development').glob('*.json'):historical.add(read(p)['H_sha256'])
    for p in (MR1/'data/smoke').glob('*.json'):historical.add(read(p)['H_sha256'])
    for p in (MR2/'data/reference').glob('*.json'):historical.add(read(p)['input']['H_sha256'])
    # Full preparation hashes were retained by the authenticated original corpus.
    with (ROOT/'outputs/sp_u4a_clean_hashes.csv').open(encoding='utf8',newline='') as f:
        for r in csv.DictReader(f):historical.update(v for k,v in r.items() if ('hash' in k.lower() or 'sha256' in k.lower()) and len(v)==64)
    write(OUT/'data/historical_H_hashes.json',dict(H_sha256=sorted(historical),scope='preparation hash CSV, B2 metadata-only H identity set, MR01 smoke/D_eval, MR02 D_ref; B3/B5C reuse B2; no outcomes read'))
    frozen={p.relative_to(OUT).as_posix():sha(p) for folder in ('src_or_scripts','tests') for p in (OUT/folder).glob('*.py') if not p.name.startswith(('plot_','finalize_'))}
    for name in ('config.json','data/validation_seed_registry.json','data/smoke_seed_registry.json','data/bootstrap_D_eval.npz','data/bootstrap_V_new.npz'):frozen[name]=sha(OUT/name)
    write(OUT/'reports/execution_freeze.json',dict(hashes=frozen,first_new_V_generated=False,EM_status='PENDING_ORIGINAL_MODEL'))
    print(json.dumps(dict(action='prepare',PASS=True,methods=len(METHODS),original_EM_pending=True,source_files=len(oldhash),resource=rss(),blas=blas_threads())),flush=True)

def load_assets():
    import run_mr05 as old
    op,bank=old.load_bank();cvs,_=common.family(op,SNRS)
    ctr=FrameControls(op,cvs,bank)
    receipt=read(OUT/'reports/spectral_construction.json')
    if not receipt['PASS']:raise ValueError('SPECTRAL_CONSTRUCTION_NOT_VALID')
    with np.load(OUT/'data/conditional_operators.npz',allow_pickle=False) as z:operators={k:z[k] for k in z.files}
    if sha(OUT/'data/conditional_operators.npz')!=receipt['operators_sha256']:raise ValueError('PPCA_OPERATOR_HASH')
    return old,ctr,operators

def evaluate_vectors(h,pilots,ctr,operators,*,saved=None,reused=None,replay=False):
    g=observation_geometry();hf=frames(h);q=np.array([np.vdot(hf[j].ravel()[g.data],hf[j].ravel()[g.data]).real for j in range(4,10)])
    e=np.empty((3,len(METHODS),6));c1=np.empty((3,6,2));c5=np.empty_like(c1);S1=np.empty((3,6,2,2));b1=np.empty((3,6,2));normal1=np.empty((3,6));normal5=np.empty_like(normal1);S5=np.empty_like(S1);b5=np.empty_like(b1)
    audit=dict(background_replay_max_abs=0.,background_replay_scaled=0.,tw1_max_normal_residual=0.,tw5_max_normal_residual=0.)
    if reused is not None:
        if np.asarray(reused).shape!=(3,6,6):raise ValueError('REUSED_BACKGROUND_SHAPE')
        e[:,:6]=reused
        if e[:,:6].shape!=(3,6,6):raise ValueError('REUSED_BACKGROUND_SHAPE')
    example=np.zeros((3,len(METHODS),16),np.complex128)
    for si,s in enumerate(SNRS):
        for ri,j in enumerate(range(4,10)):
            mapping=None if saved is None else dict(coordinates=saved['coordinates'][si,ri],gram=saved['normal_gram'][si,ri],rhs=saved['normal_rhs'][si,ri],normal_equation_relative_residual=float(saved['normal_residual'][si,ri]))
            z=ctr.predict_frame(pilots[si,j-4:j+1],s,saved_tw5=mapping,include_background=reused is None or replay)
            target=hf[j].ravel()[g.data]
            c1[si,ri]=z['tw1_coordinates'];c5[si,ri]=z['tw5_coordinates'];S1[si,ri]=z['tw1_gram'];b1[si,ri]=z['tw1_rhs'];S5[si,ri]=z['tw5_gram'];b5[si,ri]=z['tw5_rhs'];normal1[si,ri]=z['tw1_normal_residual'];normal5[si,ri]=z['tw5_normal_residual']
            predictions=z['predictions']
            for r in (16,32,64):
                key=f'PPCA_DIRECT_R{r}_{s}';predictions[f'PPCA_DIRECT_R{r}']=operators[key+'_mean_d']+operators[key+'_W']@(pilots[si,j]-operators[key+'_mean_p'])
            for method,v in predictions.items():
                mi=METHODS.index(method)
                if v.shape!=(3968,) or v.dtype!=np.complex128 or not np.isfinite(v).all():raise ValueError('NONFINITE_PREDICTION:'+method)
                err=v-target;energy=float(np.vdot(err,err).real)
                if reused is not None and mi<6:
                    ck=close(energy,e[si,mi,ri],'BACKGROUND_REPLAY');audit['background_replay_max_abs']=max(audit['background_replay_max_abs'],ck['max_abs']);audit['background_replay_scaled']=max(audit['background_replay_scaled'],ck['scaled'])
                else:e[si,mi,ri]=energy
                if j==4:example[si,mi]=v[:16]
    if not np.isfinite(e).all() or not np.isfinite(q).all() or np.any(e<0) or np.any(q<=0):raise ValueError('ENERGY_COVERAGE')
    audit['tw1_max_normal_residual']=float(normal1.max());audit['tw5_max_normal_residual']=float(normal5.max())
    arrays=dict(frame_error=e,frame_truth=q,pilots=pilots,coordinates_tw1=c1,coordinates_tw5=c5,gram_tw1=S1,rhs_tw1=b1,gram_tw5=S5,rhs_tw5=b5,normal_residual_tw1=normal1,normal_residual_tw5=normal5,fit_failed=np.zeros((3,6,2),dtype=bool),snr_db=np.array(SNRS),frame_ids=np.arange(4,10),method_names=np.array(METHODS))
    if replay or saved is None:arrays['vector_example_frame4_first16']=example
    return arrays,audit

def run(dataset):
    verify();old,ctr,operators=load_assets();t=time.monotonic();receipts=[];checks={};seen=set();n=1 if dataset=='smoke' else 100
    historical=set(read(OUT/'data/historical_H_hashes.json')['H_sha256'])
    if dataset=='V_new':
        smoke=read(OUT/'reports/smoke_receipt.json')
        if not smoke['PASS']:raise ValueError('SMOKE_REQUIRED')
        historical.update(r['H_sha256'] for r in smoke['receipts'])
    if dataset=='D_eval':
        recommendation=OUT/'reports/validation_recommendations.json'
        if not read(recommendation)['PASS']:raise ValueError('VALIDATION_RECOMMENDATION_BEFORE_D_REQUIRED')
        write(OUT/'reports/D_eval_validation_binding.json',dict(validation_recommendations_sha256=sha(recommendation),all_predeclared_ranks_evaluated=True,no_D_eval_selection=True))
    else:
        from scripts.run_sp_u12b2a import generator,seed_index,consumed
        reg=read(OUT/('data/smoke_seed_registry.json' if dataset=='smoke' else 'data/validation_seed_registry.json'));idx=seed_index(reg)
    folder=OUT/'data'/dataset
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir();old_receipts={(r['condition'],r['trajectory_id']):r for r in read(MR5/'reports/development_receipt.json')['receipts']}
    for c in CONDITIONS:
        if dataset!='D_eval':
            gen=generator(c,idx,'EVAL')
            if not (OUT/f'reports/{dataset}_generator_provenance.json').exists():write(OUT/f'reports/{dataset}_generator_provenance.json',gen.provenance())
        for i in range(n):
            start=time.monotonic();saved=None;reuse=None;binding={}
            if dataset=='D_eval':
                raw,binding=old.source_input(c,i);h=raw['H'];pilots=raw['pilots'];saved=raw
                rr=old_receipts[c,i];pp=MR5/rr['path'];jp=pp.with_suffix('.json')
                if sha(pp)!=rr['sha256'] or sha(jp)!=rr['json_sha256']:raise ValueError('MR05_REUSE_HASH')
                with np.load(pp,allow_pickle=False) as z:
                    if not np.array_equal(z['coordinates'],raw['coordinates']):raise ValueError('COORDINATE_REUSE')
                    reuse=z['frame_error'][:,[0,1,2,3,7,8],:]
                    close(z['frame_truth'],raw['frame_truth'],'TRUTH_REUSE')
                meta0=read(jp)['input'];input_meta=dict(H_sha256=array_hash(h),pilot_sha256=array_hash(pilots),original_MR01=binding,original_MR05_raw_sha256=rr['sha256'],channel_seeds=meta0.get('channel_seeds',binding.get('channel_seeds')))
            else:
                h=gen.generate(i);hh=array_hash(h)
                if hh in historical or hh in seen:raise ValueError('VALIDATION_H_IDENTITY_OVERLAP')
                seen.add(hh)
                comm=[common.communicate(consumed(idx,c,'EVAL',i,j)) for j in range(10)]
                pilots,identities=make_received_pilots(h,comm)
                input_meta=dict(H_sha256=hh,pilot_sha256=array_hash(pilots),channel_seeds=consumed(idx,c,'EVAL',i,-1),frame_seeds=[consumed(idx,c,'EVAL',i,j) for j in range(10)],H_noise_pilot_bindings=identities,seed_registry_sha256=sha(OUT/('data/smoke_seed_registry.json' if dataset=='smoke' else 'data/validation_seed_registry.json')))
            replay=dataset=='D_eval' and i==0
            arrays,audit=evaluate_vectors(h,pilots,ctr,operators,saved=saved,reused=reuse,replay=replay)
            if dataset=='D_eval':
                if not np.array_equal(arrays['coordinates_tw5'],raw['coordinates']) or not np.array_equal(arrays['frame_error'][:,:6],reuse):raise ValueError('REUSED_BYTES_CHANGED')
            p=folder/f'{c}_{i:04d}.npz';ar=archive(p,**arrays)
            write(p.with_suffix('.json'),dict(dataset=dataset,condition=c,trajectory_id=i,identity=f'MR03:{dataset}:{c}:{i}',input=input_meta,raw=ar,audit=audit,methods=METHODS,complete=True,seconds=time.monotonic()-start,operator_manifest_sha256=sha(OUT/'reports/spectral_construction.json'),new_control_outputs=5*18 if dataset=='D_eval' else len(METHODS)*18,old_reused_outputs=6*18 if dataset=='D_eval' else 0,background_check_replays=6*18 if replay else 0))
            receipts.append(dict(condition=c,trajectory_id=i,H_sha256=input_meta['H_sha256'],**ar,json_sha256=sha(p.with_suffix('.json'))))
            for k,v in audit.items():checks[k]=max(checks.get(k,0.),v)
            if i%20==0:print(json.dumps(dict(dataset=dataset,condition=c,completed=i+1,n=n,elapsed_seconds=time.monotonic()-t)),flush=True)
    verify();answer=dict(PASS=True,dataset=dataset,physical_trajectories=5*n,scored_positions_per_method=5*n*18,method_count=len(METHODS),raw_saved_before_statistics=True,receipts=receipts,new_outputs=5*n*18*(5 if dataset=='D_eval' else len(METHODS)),reused_outputs=5*n*18*6 if dataset=='D_eval' else 0,background_check_replays=540 if dataset=='D_eval' else 0,fit_failure_count=0,nonfinite_count=0,audit=checks,elapsed_seconds=time.monotonic()-t,resource=rss(),blas=blas_threads(),original_EM_status='PENDING_ORIGINAL_MODEL')
    write(OUT/f'reports/{dataset}_receipt.json',answer)
    print(json.dumps({k:v for k,v in answer.items() if k!='receipts'}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','V_new','D_eval']);a=p.parse_args()
    try:prepare() if a.action=='prepare' else run(a.action)
    except Exception as e:
        write(OUT/f'reports/{a.action}_failure_{time.time_ns()}.json',dict(status='STOP',error=repr(e),traceback=traceback.format_exc(),no_old_files_written=True));raise
if __name__=='__main__':main()
