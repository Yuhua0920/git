"""MR-01 development only; new data, frozen estimators, immutable raw blocks.

Run prepare, smoke, development, analyze in that order. No historical outcomes
are opened and no formal experiment or manuscript is written.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,csv,hashlib,json,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np

SNRS=(10,20,30)
METHODS=('Source','Pooled','Hard','Convex','Dense','LR4','Pooled-source-centered')
CONTRASTS=(('Source-Pooled',0,1),('Pooled-LR4',1,5),('Source-LR4',0,5),
           ('Pooled-Dense',1,4),('Dense-LR4',4,5),('Natural-SourceCentered',1,6))
TOL=1e-10
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def archive(p,**a):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    with p.open('xb') as f:np.savez_compressed(f,**a)
    return dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size)
def csvout(p,rows):
    with Path(p).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),np.finfo(float).tiny))
def init(root,out):
    global ROOT,OUT,c,ah
    ROOT=Path(root).resolve();OUT=Path(out).resolve()
    if OUT!=ROOT/'outputs/revision_mr01_pooled_lmmse':raise ValueError('MR01 output root required')
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT/'src_or_scripts'))
    from scripts import sp_u12b2a_common
    from src.cdl_benchmark import array_hash
    c=sp_u12b2a_common;ah=array_hash
    for d in ('data','tables','figures','reports'): (OUT/d).mkdir(parents=True,exist_ok=True)
def preserved_paths():
    paths=[ROOT/'outputs'/n for n in c.STATS.values()]
    paths += [ROOT/'outputs/sp_u4a_clean.npy',ROOT/'outputs/sp_u12b2a/operators.npz']
    paths += list((ROOT/'src').glob('*.py'))
    paths += [ROOT/'scripts'/n for n in ('sp_u12b2a_common.py','run_sp_u12b2a.py','run_formal_test.py')]
    paths += list((ROOT/'outputs/paper_v4_2_candidate').rglob('*.tex'))
    return sorted(set(paths))
def check_preservation():
    baseline=read(OUT/'reports/preservation_before.json')
    bad=[p for p,h in baseline['hashes'].items() if sha(ROOT/p)!=h]
    if bad:raise ValueError('HISTORICAL_BYTES_CHANGED:'+str(bad))
    return dict(PASS=True,files=len(baseline['hashes']),modified=0)
def verify_freeze():
    freeze=read(OUT/'reports/execution_freeze.json')
    for p,h in freeze['hashes'].items():
        if sha(OUT/p)!=h:raise ValueError('NEW_RUNNER_BYTES_DRIFT:'+p)
    for p,h in freeze['scientific_sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('FROZEN_SOURCE_BYTES_DRIFT:'+p)
    for p,key in (('reports/protocol.json','protocol_sha256'),('data/seed_registry.json','seed_registry_sha256'),('data/operators.npz','operator_sha256')):
        if sha(OUT/p)!=freeze[key]:raise ValueError('PROTOCOL_SEED_OPERATOR_DRIFT:'+p)
    return freeze
def direct_moments(raw,geometry,mean_p=None,mean_d=None):
    rows=raw.reshape(-1,4096);n=len(rows)
    if mean_p is None:
        sp=np.zeros(64,np.complex128);sd=np.zeros(3968,np.complex128)
        for k in range(0,n,256):
            x=np.asarray(rows[k:k+256]);sp+=x[:,geometry.selected].sum(0);sd+=x[:,geometry.data].sum(0)
        mean_p=sp/n;mean_d=sd/n
    rpp=np.zeros((64,64),np.complex128);rdp=np.zeros((3968,64),np.complex128)
    for k in range(0,n,256):
        x=np.asarray(rows[k:k+256]);p=x[:,geometry.selected]-mean_p;d=x[:,geometry.data]-mean_d
        rpp+=p.T@p.conj();rdp+=d.T@p.conj()
    return (mean_p,mean_d,rpp/n,rdp/n)
def prepare():
    from pooled import pool_statistics
    from seed_union import build_forbidden_seed_union,serializable_report
    from src.cdl_benchmark import observation_geometry
    from scripts.run_sp_u12b2a import allocate
    t=time.monotonic()
    write(OUT/'reports/preservation_before.json',dict(hashes={p.relative_to(ROOT).as_posix():sha(p) for p in preserved_paths()}))
    lineage=read(OUT/'reports/preparation_readonly_lineage.json')
    # Agent's independent lineage report and original frozen source hashes are
    # checked explicitly; no performance table is a construction input.
    original=read(ROOT/'outputs/sp_u12b2a_formal_manifest.json')
    frozen=original['preparation_statistics_hashes']
    stats={m:c.statistics(ROOT/'outputs'/n) for m,n in c.STATS.items()}
    for m,n in c.STATS.items():
        h=sha(ROOT/'outputs'/n)
        if h!=frozen[m]:raise ValueError('PREPARATION_STATISTICS_HASH_MISMATCH:'+n)
        if stats[m].samples!=5000 or stats[m].train_trajectory_ids!=tuple(range(500)):raise ValueError('P_CORPUS_ID')
    pool=pool_statistics(tuple(stats.values()))
    g=observation_geometry();raw=np.load(ROOT/'outputs/sp_u4a_clean.npy',mmap_mode='r',allow_pickle=False)
    if sha(ROOT/'outputs/sp_u4a_clean.npy')!=lineage['files']['outputs/sp_u4a_clean.npy']['sha256'] or not lineage['corpus_sha256_exact']:raise ValueError('P_RAW_LINEAGE')
    if raw.shape!=(3,500,10,4096) or raw.dtype!=np.complex128:raise ValueError('RAW_P_SHAPE_DTYPE')
    rows=[]
    for i,(m,st) in enumerate(stats.items()):
        direct=direct_moments(raw[i],g)
        for key,a,b in zip(('mean_p','mean_d','rpp','rdp'),direct,(st.mean_p,st.mean_d,st.rpp,st.rdp)):
            e=rel(a,b);rows.append(dict(origin=m,object=key,relative_error=e,tolerance=TOL,PASS=e<=TOL))
    direct=direct_moments(raw,g)
    for key,a,b in zip(('mean_p','mean_d','rpp','rdp'),direct,(pool.mean_p,pool.mean_d,pool.rpp,pool.rdp)):
        e=rel(a,b);rows.append(dict(origin='Pooled',object=key,relative_error=e,tolerance=TOL,PASS=e<=TOL))
    if not all(r['PASS'] for r in rows):raise ValueError('DIRECT_FULL_SAMPLE_MOMENT_MISMATCH')
    csvout(OUT/'tables/preparation_moment_comparison.csv',rows)
    archive(OUT/'data/pooled_statistics.npz',mean_p=pool.mean_p,mean_d=pool.mean_d,rpp=pool.rpp,rdp=pool.rdp,samples=np.array(pool.samples))
    op={};orows=[]
    with np.load(ROOT/'outputs/sp_u12b2a/operators.npz',allow_pickle=False) as z:
        for k in z.files:
            if k in ('source_mean_p','source_mean_d') or k.rsplit('_',1)[-1] in tuple(map(str,SNRS)):op[k]=z[k].copy()
    anchor=read(ROOT/'outputs/sp_u12b2a/operator_anchor.json')['operator_archive']
    if sha(ROOT/'outputs/sp_u12b2a/operators.npz')!=anchor['sha256']:raise ValueError('OPERATOR_CACHE_HASH')
    for s in SNRS:
        for m,st in stats.items():
            e=rel(st.prepare(c.variance(s)).weight,op[f'W{"source" if m=="Source" else m}_{s}'])
            if e>TOL:raise ValueError('PREPARED_OPERATOR_ANCHOR')
        est=pool.prepare(c.variance(s));op[f'Wpool_{s}']=est.weight
        C=pool.rpp+c.variance(s)*np.eye(64)
        orows.append(dict(snr_db=s,sigma2=c.variance(s),hermitian_residual=rel(pool.rpp,pool.rpp.conj().T),minimum_eigenvalue=float(np.linalg.eigvalsh(pool.rpp)[0]),minimum_noisy_eigenvalue=float(np.linalg.eigvalsh(C)[0]),solve_residual=est.solve_relative_residual,mean_shift_intercept_norm=float(np.linalg.norm(pool.mean_d-stats['Source'].mean_d-est.weight@(pool.mean_p-stats['Source'].mean_p)))))
    op['pool_mean_p']=pool.mean_p;op['pool_mean_d']=pool.mean_d
    if any(not np.isfinite(v).all() for v in op.values()):raise ValueError('NONFINITE_OPERATOR')
    archive(OUT/'data/operators.npz',**op);csvout(OUT/'tables/pooled_operator_audit.csv',orows)
    between_pp=pool.rpp-sum(st.rpp/3 for st in stats.values());between_dp=pool.rdp-sum(st.rdp/3 for st in stats.values())
    write(OUT/'reports/pooled_construction.json',dict(PASS=True,origins=list(stats),samples_per_origin=5000,total_samples=15000,independent_preparation_trajectories=1500,shapes={k:list(getattr(pool,k).shape) for k in ('mean_p','mean_d','rpp','rdp')},dtype='complex128',population_normalization='1/N frames',between_mean_pp_frobenius=float(np.linalg.norm(between_pp)),between_mean_dp_frobenius=float(np.linalg.norm(between_dp)),maximum_direct_moment_relative_error=max(r['relative_error'] for r in rows),tolerance=TOL,source_cache_sha256=sha(ROOT/'outputs/sp_u12b2a/operators.npz'),seconds=time.monotonic()-t))
    exclusion_raw=build_forbidden_seed_union(ROOT,with_report=True);forbidden=exclusion_raw['forbidden_seeds'];exclusion=serializable_report(exclusion_raw)
    used=set();smoke=allocate('SP_U17_MR01_SMOKE_V1',3,0,forbidden,used)
    development=allocate('SP_U17_MR01_DEVELOPMENT_V1',100,0,forbidden,used,True)
    write(OUT/'data/seed_registry.json',dict(smoke=smoke,development=development))
    write(OUT/'reports/seed_independence.json',dict(PASS=True,exclusion=exclusion,new_seeds=len(used),old_overlap=len(used&forbidden),individual_roles_unique=True,SDK_adjacent_state_protection=True,P_D_old_V_T_disjoint=True,smoke_development_disjoint=True,future_selection_validation_generated=False,future_confirmation_generated=False))
    protocol=dict(identity='MR-01 DEVELOPMENT',conditions=c.CONDITIONS,snrs=list(SNRS),smoke_per_condition=3,development_per_condition=100,frames=10,scored_frames=list(range(4,10)),methods=list(METHODS),source_referenced_power=c.POWER,sigma2={str(s):c.variance(s) for s in SNRS},rank=4,window=5,folds=4,temporal_information='Source/Pooled use current pilot; Hard/Convex/Dense/LR4 use current-inclusive five-frame window. Not equal-temporal-information ablation.',independent_unit='physical trajectory; frames, methods and SNR repeat the same unit',bootstrap=dict(resamples=2000,confidence=.95,adjusted=False,quantile_method='linear',seed=development['rows'][-1]['seed'],substreams='SeedSequence([master,condition ordinal]); same trajectory indices across methods and SNR'),contrasts=[dict(name=n,sign='first minus second in pooled dB',first=METHODS[i],second=METHODS[j]) for n,i,j in CONTRASTS],practical_reference_db=.5,practical_reference_is_not_significance_threshold=True,smoke_excluded_from_science=True,resource_gate=dict(maximum_estimated_development_hours=1,maximum_smoke_process_RSS_GiB=8,reason='Personal-computer development engineering limits fixed before smoke; not a scientific acceptance criterion.'),method_selection_using_D=False)
    write(OUT/'reports/protocol.json',protocol)
    sourcepaths=[p for p in preserved_paths() if p.suffix=='.py']
    write(OUT/'reports/execution_freeze.json',dict(hashes={p.relative_to(OUT).as_posix():sha(p) for d in ('src_or_scripts','tests') for p in (OUT/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts},scientific_sources={p.relative_to(ROOT).as_posix():sha(p) for p in sourcepaths},protocol_sha256=sha(OUT/'reports/protocol.json'),seed_registry_sha256=sha(OUT/'data/seed_registry.json'),operator_sha256=sha(OUT/'data/operators.npz'),new_H_generated=0))
    print(json.dumps(dict(prepare='PASS',elapsed_seconds=time.monotonic()-t,seed_count=len(used))),flush=True)
def evaluate(h,comm,op,cvs,factors):
    from src.cdl_benchmark import observation_geometry,frames
    from src.link import assemble_pilot_data_frame,apply_frequency_mimo_channel,estimate_ls_linear_channel
    g=observation_geometry();hf=frames(h);ps=np.zeros((3,10,64),np.complex128);inputs=[]
    for j,(bits,symbols,z) in enumerate(comm):
        clean=apply_frequency_mimo_channel(hf[j],assemble_pilot_data_frame(symbols,g.layout))
        for si,s in enumerate(SNRS):
            y=clean+np.sqrt(c.variance(s))*z
            sparse,_=estimate_ls_linear_channel(y,g.layout);p=sparse.ravel()[g.selected]
            ps[si,j]=p;inputs.append(dict(snr_db=s,frame_id=j,H_frame_sha256=ah(hf[j]),unit_noise_sha256=ah(z),pilot_sha256=ah(p),received_sha256=ah(y),shared_by_methods=list(METHODS)))
    ps.flags.writeable=False;before=ah(ps)
    errors=np.empty((3,7,6));truth=np.array([np.vdot(hf[j].ravel()[g.data],hf[j].ravel()[g.data]).real for j in range(4,10)])
    coordinates=np.empty((3,6,2));center_identity=[];normal=np.empty((3,6));grams=np.empty((3,6,2,2));rhs=np.empty((3,6,2))
    mapping={'source':0,'hard':2,'convex':3,'dense-affine':4,'lr4-affine':5}
    for si,s in enumerate(SNRS):
        records=c.infer(ps[si],op,s,cvs,factors)
        if [r['frame_id'] for r in records]!=list(range(4,10)):raise ValueError('SCORING_FRAME_IDENTITY')
        for ri,row in enumerate(records):
            j=row['frame_id'];p=ps[si,j];target=hf[j].ravel()[g.data]
            if row['window_start']!=j-4 or row['dense_coefficients'] is not row['lr4_coefficients']:raise ValueError('WINDOW_SHARED_DENSE_LR4_COORDINATES')
            diag=row['diagnostics']['affine'];cc=row['coefficients']['affine']
            if diag['clipped'] or diag['regularized'] or not np.isfinite(np.r_[cc,np.asarray(diag['gram']).ravel(),diag['rhs'],diag['normal_equation_relative_residual']]).all():raise ValueError('COORDINATE_SOLVE_GATE')
            if diag['normal_equation_relative_residual']>c.SOLVE_TOL:raise ValueError('NORMAL_EQUATION_RESIDUAL')
            normal[si,ri]=diag['normal_equation_relative_residual'];grams[si,ri]=diag['gram'];rhs[si,ri]=diag['rhs']
            coordinates[si,ri]=row['dense_coefficients']
            predictions={mapping[k]:v for k,v in row['predictions'].items()}
            predictions[1]=op['pool_mean_d']+op[f'Wpool_{s}']@(p-op['pool_mean_p'])
            predictions[6]=op['source_mean_d']+op[f'Wpool_{s}']@(p-op['source_mean_p'])
            offset=op['pool_mean_d']-op['source_mean_d']-op[f'Wpool_{s}']@(op['pool_mean_p']-op['source_mean_p'])
            center_identity.append(float(np.linalg.norm(predictions[1]-predictions[6]-offset)))
            for m,pred in predictions.items():
                if pred.shape!=(3968,) or pred.dtype!=np.complex128 or not np.isfinite(pred).all():raise ValueError('PREDICTION_SHAPE_DTYPE_FINITE')
                e=pred-target;errors[si,m,ri]=float(np.vdot(e,e).real)
    if ah(ps)!=before:raise ValueError('SHARED_PILOTS_MUTATED')
    if not np.isfinite(errors).all() or np.any(errors<0) or not np.isfinite(truth).all() or np.any(truth<=0):raise ValueError('ENERGY_GATE')
    if max(center_identity)>1e-10:raise ValueError('CENTERING_CONSTANT_IDENTITY')
    return errors,truth,coordinates,ps,inputs,max(center_identity),normal,grams,rhs
def run(split):
    import torch
    from scripts.sp_u4a_common import peak_memory
    from scripts.run_sp_u12b2a import seed_index,generator,consumed
    verify_freeze();reg=read(OUT/'data/seed_registry.json')[split];index=seed_index(reg)
    if split=='development':
        if not read(OUT/'reports/smoke_summary.json')['resource_gate_PASS']:raise ValueError('SMOKE_RESOURCE_GATE_STOP')
    n=3 if split=='smoke' else 100;folder=OUT/'data'/split
    if folder.exists():raise FileExistsError('Independent run already started: '+str(folder))
    folder.mkdir();op=dict(np.load(OUT/'data/operators.npz',allow_pickle=False))
    for v in op.values():v.flags.writeable=False
    cvs,factors=c.family(op,SNRS);rsspeak=peak_memory()['peak_working_set_bytes'];times=[];receipts=[];seen=set()
    historical=set(read(OUT/'reports/b2_H_identity_readonly.json')['H_sha256'])
    # Preparation lineage includes full trajectory hashes; retain exact output check.
    prep=read(OUT/'reports/preparation_readonly_lineage.json')
    for k in ('H_hashes','trajectory_H_hashes'):
        if k in prep:historical.update(prep[k])
    with (ROOT/'outputs/sp_u4a_clean_hashes.csv').open(encoding='utf8',newline='') as f:
        for r in csv.DictReader(f):
            for k,v in r.items():
                if 'hash' in k.lower() or 'sha256' in k.lower():historical.add(v)
    if split=='development':
        historical.update(read(p)['H_sha256'] for p in (OUT/'data/smoke').glob('*.json'))
    start=time.monotonic();device_info=None
    for condition in c.CONDITIONS:
        gen=generator(condition,index,'EVAL')
        if device_info is None:
            device_info=gen.provenance()
            write(OUT/'reports'/f'{split}_runtime_provenance.json',device_info)
        for i in range(n):
            t=time.monotonic();h=gen.generate(i);H_hash=ah(h)
            if H_hash in historical or H_hash in seen:raise ValueError('H_TRAJECTORY_IDENTITY_OVERLAP')
            seen.add(H_hash)
            comm=[c.communicate(consumed(index,condition,'EVAL',i,j)) for j in range(10)]
            e,q,coords,pilots,inputs,centering,normal,grams,rhs=evaluate(h,comm,op,cvs,factors)
            stem=f'{condition}_{i:04d}';path=folder/(stem+'.npz')
            payload=archive(path,H=h,unit_noise=np.stack([v[2] for v in comm]),bits=np.stack([v[0] for v in comm]),pilots=pilots,frame_error=e,frame_truth=q,coordinates=coords,normal_residual=normal,normal_gram=grams,normal_rhs=rhs,snr_db=np.array(SNRS),frame_ids=np.arange(4,10),method_names=np.array(METHODS))
            elapsed=time.monotonic()-t;times.append(elapsed);rsspeak=max(rsspeak,peak_memory()['peak_working_set_bytes'])
            receipt=dict(identity=f'MR01:{split}:{condition}:{i}',condition=condition,split=split,trajectory_id=i,seeds=consumed(index,condition,'EVAL',i),communication_seeds=[consumed(index,condition,'EVAL',i,j) for j in range(10)],H_sha256=H_hash,inputs=inputs,archive=payload,centering_identity_max_norm=centering,normal_equation_max_residual=float(normal.max()),seconds=elapsed,process_memory=peak_memory())
            write(folder/(stem+'.json'),receipt);receipts.append(dict(identity=receipt['identity'],json_sha256=sha(folder/(stem+'.json')),npz_sha256=payload['sha256']))
            if i%10==0 or i==n-1:print(json.dumps(dict(split=split,condition=condition,completed=len(receipts),expected=5*n,elapsed_seconds=time.monotonic()-start,RSS_GiB=rsspeak/2**30)),flush=True)
    total=time.monotonic()-start;expected_hours=np.mean(times)*500/3600
    summary=dict(PASS=True,split=split,physical_trajectories=5*n,conditions=5,snr_points=3,scored_frames=6,scored_positions_per_method=5*n*3*6,methods=7,method_predictions=5*n*3*6*7,wall_seconds=total,per_trajectory_seconds=dict(mean=float(np.mean(times)),max=float(np.max(times))),peak_sampled_RSS_bytes=rsspeak,peak_torch_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,expected_development_hours=float(expected_hours),resource_gate_PASS=bool(expected_hours<=1 and rsspeak<8*2**30),CPU_threads=1,actual_device=str(gen.device),all_raw_blocks_persisted_before_aggregate=True,receipts=receipts)
    write(OUT/'reports'/f'{split}_summary.json',summary)
    verified_energies(split)
    write(OUT/'reports'/f'{split}_raw_gate.json',dict(PASS=True,all_receipts_hashes_CRC_finite_axes_paired_pilots=True,trajectories=5*n,science_aggregate_released=False))
    print(json.dumps({k:v for k,v in summary.items() if k!='receipts'}),flush=True)
def verified_energies(split='development'):
    from scripts.run_sp_u12b2a import seed_index,consumed
    idx=seed_index(read(OUT/'data/seed_registry.json')[split])
    summary=read(OUT/'reports'/f'{split}_summary.json')
    if not summary['PASS']:raise ValueError('RAW_COMPLETENESS')
    n=100 if split=='development' else 3;E=np.empty((5,n,3,7));Q=np.empty((5,n));raw=[]
    expected={f'MR01:{split}:{condition}:{i}' for condition in c.CONDITIONS for i in range(n)}
    if {r['identity'] for r in summary['receipts']}!=expected:raise ValueError('IDENTITY_GRID')
    for ci,condition in enumerate(c.CONDITIONS):
        for i in range(n):
            stem=f'{condition}_{i:04d}';jp=OUT/'data'/split/(stem+'.json');p=jp.with_suffix('.npz');r=read(jp)
            if (r['identity'],r['condition'],r['split'],r['trajectory_id'])!=(f'MR01:{split}:{condition}:{i}',condition,split,i):raise ValueError('RECEIPT_FILENAME_GRID_IDENTITY')
            if r['seeds']!=consumed(idx,condition,'EVAL',i) or r['communication_seeds']!=[consumed(idx,condition,'EVAL',i,j) for j in range(10)]:raise ValueError('RAW_FROZEN_SEED_BINDING')
            expected_receipt=next(v for v in summary['receipts'] if v['identity']==r['identity'])
            if sha(jp)!=expected_receipt['json_sha256'] or sha(p)!=expected_receipt['npz_sha256']:raise ValueError('RAW_BLOCK_HASH')
            with zipfile.ZipFile(p) as z:
                if z.testzip() is not None:raise ValueError('NPZ_CRC')
            with np.load(p,allow_pickle=False) as z:
                e=z['frame_error'];q=z['frame_truth']
                if e.shape!=(3,7,6) or q.shape!=(6,) or not np.isfinite(e).all() or not np.isfinite(q).all() or np.any(e<0) or np.any(q<=0):raise ValueError('RAW_ENERGY_SHAPE_FINITE')
                if tuple(z['snr_db'])!=SNRS or tuple(z['frame_ids'])!=tuple(range(4,10)) or tuple(z['method_names'])!=METHODS:raise ValueError('RAW_AXES')
                if ah(z['H'])!=r['H_sha256']:raise ValueError('RAW_H_HASH')
                if z['coordinates'].shape!=(3,6,2) or z['normal_residual'].shape!=(3,6) or not np.isfinite(z['coordinates']).all() or not np.isfinite(z['normal_residual']).all() or np.max(z['normal_residual'])>c.SOLVE_TOL:raise ValueError('RAW_SOLVE_AUDIT')
                if len(r['inputs'])!=30 or len({(x['snr_db'],x['frame_id']) for x in r['inputs']})!=30:raise ValueError('PAIRED_PILOT_GRID')
                for x in r['inputs']:
                    si=SNRS.index(x['snr_db']);j=x['frame_id']
                    if ah(z['pilots'][si,j])!=x['pilot_sha256'] or ah(z['unit_noise'][j])!=x['unit_noise_sha256'] or x['shared_by_methods']!=list(METHODS):raise ValueError('PAIRED_INPUT_HASH')
                E[ci,i]=e.sum(-1);Q[ci,i]=q.sum()
                for si,s in enumerate(SNRS):
                    for m,name in enumerate(METHODS):raw.append(dict(identity=r['identity'],condition=condition,snr_db=s,method=name,trajectory_id=i,error_energy=E[ci,i,si,m],truth_energy=Q[ci,i],scored_frames=6))
    return E,Q,raw
def pooled_bootstrap(error,truth,draw):
    """One condition: trajectory errors[N,S,M], truths[N], indices[B,N]."""
    if error.ndim!=3 or truth.shape!=(error.shape[0],) or draw.ndim!=2 or draw.shape[1]!=len(truth):raise ValueError('STATISTICAL_UNIT_AXES')
    if draw.dtype.kind not in 'iu' or np.any(draw<0) or np.any(draw>=len(truth)):raise ValueError('BOOTSTRAP_IDENTITY_RANGE')
    if not np.isfinite(error).all() or not np.isfinite(truth).all() or np.any(error<=0) or np.any(truth<=0):raise ValueError('METRIC_ENERGIES')
    point=error.sum(0)/truth.sum()
    boot=error[draw].sum(1)/truth[draw].sum(1)[:,None,None]
    return point,10*np.log10(point),10*np.log10(boot)
def analyze():
    verify_freeze();E,Q,raw=verified_energies();protocol=read(OUT/'reports/protocol.json');seed=protocol['bootstrap']['seed']
    csvout(OUT/'tables/trajectory_energies.csv',raw);aggregate=[];contrasts=[];core=[];draws={};absolute_ci={};difference_ci={}
    for ci,condition in enumerate(c.CONDITIONS):
        draw=np.random.default_rng(np.random.SeedSequence([seed,ci])).integers(0,100,size=(2000,100),dtype=np.int64);draws[condition]=draw
        point,db,bdb=pooled_bootstrap(E[ci],Q[ci],draw)
        if not np.isfinite(bdb).all():raise ValueError('BOOTSTRAP_NONFINITE')
        absolute_ci[condition]=np.quantile(bdb,[.025,.975],axis=0,method='linear')
        for si,s in enumerate(SNRS):
            for m,name in enumerate(METHODS):aggregate.append(dict(condition=condition,snr_db=s,method=name,error_energy=E[ci,:,si,m].sum(),truth_energy=Q[ci].sum(),nmse_linear=point[si,m],nmse_db=db[si,m],trajectory_count=100,scored_frames=600,interval_identity='exploratory development, unadjusted 95% paired trajectory percentile',CI_lower=absolute_ci[condition][0,si,m],CI_upper=absolute_ci[condition][1,si,m]))
            cr=dict(condition=condition,snr_db=s,Pooled_nmse_db=db[si,1],LR4_nmse_db=db[si,5])
            for name,i,j in CONTRASTS:
                bd=bdb[:,si,i]-bdb[:,si,j];lower,upper=np.quantile(bd,[.025,.975],method='linear');delta=db[si,i]-db[si,j]
                contrasts.append(dict(condition=condition,snr_db=s,contrast=name,first_method=METHODS[i],second_method=METHODS[j],estimate_db=delta,CI_lower=lower,CI_upper=upper,trajectory_count=100,bootstrap_resamples=2000,interval_identity='Exploratory development interval; unadjusted 95%; not confirmatory',first_method_worse_trajectory_fraction=float(np.mean(E[ci,:,si,i]>E[ci,:,si,j]))))
                cr[name]=delta
                if name=='Pooled-LR4':cr.update(CI_lower=lower,CI_upper=upper)
            core.append(cr)
    csvout(OUT/'tables/nmse_all_methods.csv',aggregate);csvout(OUT/'tables/paired_contrasts.csv',contrasts);csvout(OUT/'tables/core_results.csv',core)
    archive(OUT/'data/exploratory_bootstrap_indices.npz',**draws)
    archive(OUT/'data/verified_trajectory_energies.npz',error=E,truth=Q,snr_db=np.array(SNRS),method_names=np.array(METHODS),condition_names=np.array(tuple(c.CONDITIONS)))
    preservation=check_preservation()
    write(OUT/'reports/analysis_integrity.json',dict(PASS=True,physical_trajectories=500,unique_identities=500,condition_SNR_cells=15,pilot_input_records=15000,scored_positions_per_method=9000,methods=7,method_predictions=63000,raw_CRC_and_hash_gate='PASS',centering_control='same Wpool, changed intercept only',all_methods_same_truth=True,preserved=preservation,bootstrap=protocol['bootstrap'],historical_outcome_reads=0,method_tuning=False,scientific_interpretation='Development only, conditional on one frozen preparation corpus; no new confirmation',core_results=core))
    print(json.dumps(dict(analysis='PASS',cells=15,rows=105,preservation=preservation)),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','smoke','development','analyze'));p.add_argument('--root',required=True);p.add_argument('--out',required=True);args=p.parse_args();init(args.root,args.out)
    try:
        if args.action=='prepare':prepare()
        elif args.action=='analyze':analyze()
        else:run(args.action)
    except Exception as e:
        failure=OUT/'reports'/f'{args.action}_failure_{time.time_ns()}.json'
        write(failure,dict(status='STOP',action=args.action,error=repr(e),traceback=traceback.format_exc(),no_original_files_modified=True));raise
if __name__=='__main__':main()
