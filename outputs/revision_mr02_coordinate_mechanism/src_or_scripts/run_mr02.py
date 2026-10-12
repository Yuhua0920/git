"""MR02 development diagnostics, frozen original estimators, no formal tests.

Ordered actions: prepare, smoke, reference, fit, evaluate, analyze. Exclusive
blocks precede aggregation. Truth-assisted references never enter infer().
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,csv,hashlib,json,sys,time,traceback,zipfile,functools
from pathlib import Path
import numpy as np

SNRS=(10,20,30);CONDITIONS=('ID','Profile','Mobility','Composed','Boundary')
METHODS=('Source','Pooled','Hard','Convex','Dense-online','LR4-online',
         'Dense-cp','Dense-cd','Dense-oracle','LR4-cp','LR4-cd','LR4-oracle','Matched')
SPACE_NAMES=('Dense','LR4');RCOND=1e-12;TOL=1e-10;ATOL=1e-10
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(8*1024*1024),b''):h.update(v)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def archive(p,**arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.savez_compressed(f,**arrays)
    return dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
def csvout(p,rows):
    with Path(p).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def assert_close(a,b,label,scale=None):
    a=np.asarray(a);b=np.asarray(b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('NONFINITE:'+label)
    err=float(np.max(np.abs(a-b)))
    den=float(max(np.max(np.abs(a)),np.max(np.abs(b)),0 if scale is None else scale))
    if err>ATOL+TOL*den:raise ValueError(label+':'+str((err,den)))
    return err/(1+den)
def init(root,out):
    global ROOT,OUT,MR,c,m,old,g,ah,op,cvs,factors
    ROOT=Path(root).resolve();OUT=Path(out).resolve();MR=ROOT/'outputs/revision_mr01_pooled_lmmse'
    if OUT!=ROOT/'outputs/revision_mr02_coordinate_mechanism':raise ValueError('MR02 output namespace')
    sys.path[:0]=[str(ROOT),str(OUT/'src_or_scripts'),str(MR/'src_or_scripts')]
    from scripts import sp_u12b2a_common as c
    from src.cdl_benchmark import observation_geometry,array_hash
    import mechanism as m
    import run_mr01 as old
    old.c=c;old.ah=array_hash;old.ROOT=ROOT;old.OUT=MR
    g=observation_geometry();ah=array_hash
    op=dict(np.load(MR/'data/operators.npz',allow_pickle=False))
    for v in op.values():v.flags.writeable=False
    cvs,factors=c.family(op,SNRS)
    for d in ('data','tables','figures','reports'):(OUT/d).mkdir(parents=True,exist_ok=True)
def preserve():
    before=read(OUT/'reports/preservation_before.json')
    bad=[p for p,h in before['hashes'].items() if sha(ROOT/p)!=h]
    if bad:raise ValueError('PRESERVED_INPUT_CHANGED:'+str(bad))
    return dict(files=len(before['hashes']),modified=0,PASS=True)
def frozen():
    f=read(OUT/'reports/execution_freeze.json')
    for p,h in f['hashes'].items():
        if sha(OUT/p)!=h:raise ValueError('EXECUTION_BYTES_CHANGED:'+p)
    for p,h in f['dependencies'].items():
        if sha(ROOT/p)!=h:raise ValueError('DEPENDENCY_BYTES_CHANGED:'+p)
    if sha(OUT/'data/reference_seed_registry.json')!=f['reference_seed_sha256'] or sha(OUT/'reports/protocol.json')!=f['protocol_sha256']:raise ValueError('FROZEN_SEED_PROTOCOL_CHANGED')
    p='outputs/revision_mr01_pooled_lmmse/data/operators.npz'
    if sha(ROOT/p)!=read(OUT/'reports/preservation_before.json')['hashes'][p]:raise ValueError('FROZEN_PREPARATION_OPERATORS_CHANGED')
    return f
def prepare():
    from seed_union import build_forbidden_seed_union,serializable_report
    from scripts.run_sp_u12b2a import allocate
    old.verify_freeze();old.check_preservation()
    hashes=dict(read(MR/'reports/preservation_before.json')['hashes'])
    names=['data/operators.npz','data/seed_registry.json','data/exploratory_bootstrap_indices.npz',
           'data/verified_trajectory_energies.npz','reports/protocol.json','reports/pooled_construction.json',
           'tables/preparation_moment_comparison.csv','tables/pooled_operator_audit.csv',
           'tables/core_results.csv','tables/nmse_all_methods.csv','tables/paired_contrasts.csv','tables/trajectory_energies.csv']
    for p in names:hashes[(MR/ p).relative_to(ROOT).as_posix()]=sha(MR/p)
    for split in ('smoke','development'):
        for r in read(MR/'reports'/f'{split}_summary.json')['receipts']:
            _,_,condition,i=r['identity'].split(':')
            base=MR/'data'/split/f'{condition}_{int(i):04d}'
            hashes[base.with_suffix('.npz').relative_to(ROOT).as_posix()]=r['npz_sha256']
            hashes[base.with_suffix('.json').relative_to(ROOT).as_posix()]=r['json_sha256']
    for condition in CONDITIONS:
        for suffix in ('.npz','.json'):
            p=ROOT/f'outputs/sp_u12b2b_sealed/reference/{condition}/statistics{suffix}'
            hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    task=ROOT/'SP_U17_MR02_Mechanism_Taskbook.md';hashes[task.relative_to(ROOT).as_posix()]=sha(task)
    write(OUT/'reports/preservation_before.json',dict(hashes=hashes))
    ex=build_forbidden_seed_union(ROOT,with_report=True)
    mrseeds=read(MR/'data/seed_registry.json');prior=set()
    for split in mrseeds.values():prior.update(r['seed'] for r in split['rows'])
    forbidden=ex['forbidden_seeds']|prior|{v+d for v in prior for d in (-1,1) if 0<=v+d<=2**32-1}
    used=set();registry=allocate('SP_U17_MR02_REFERENCE_V1',100,0,forbidden,used,True)
    write(OUT/'data/reference_seed_registry.json',registry)
    report=serializable_report(ex)
    write(OUT/'reports/reference_independence.json',dict(PASS=True,namespace=registry.get('namespace','SP_U17_MR02_REFERENCE_V1'),old_scientific_union=report,MR01_any_role_seeds=len(prior),new_seeds=len(used),overlap=0,SDK_adjacent_states_excluded=True,D_ref_role='condition-labelled clean-truth-assisted diagnostic reference fit only',D_eval_role='previously disclosed MR01 development; not final confirmation',future_V_T_generated=False))
    protocol=dict(identity='MR02 MECHANISM DEVELOPMENT',conditions=c.CONDITIONS,snrs=SNRS,
      D_ref_per_condition=100,D_eval_per_condition=100,scored_frames=list(range(4,10)),window=5,folds=4,
      pilot_complex_rows=320,pilot_real_rows=640,data_real_rows=7936,coordinate_dtype='float64',signal_dtype='complex128',
      fixed_references=45,reference_labels_and_truth='diagnostic additional information only',one_shared_pilot_objective=True,
      window_frame_multiplicity=[sum(j-4<=ell<=j for j in range(4,10)) for ell in range(10)],
      current_data_frame_multiplicity=[int(j>=4) for j in range(10)],oracle_rcond=RCOND,
      normal_matrix_rank_floor='max(rcond squared,64 machine epsilon), numerical diagnostic only',
      vector_identity_rtol=TOL,vector_identity_atol=ATOL,oracle_inequality_tolerance='1e-10 absolute +1e-10 contribution scale; no clipping',
      bootstrap_eval='exact saved MR01 2000 trajectory draws, unadjusted 95% percentile',
      reference_bootstrap_resamples=500,reference_bootstrap_seed=registry['rows'][-1]['seed'],
      independent_unit='trajectory; repeated frames/windows/methods/SNR',resource_limit_hours=2,
      no_online_method_changes=True,no_model_selection=True,no_formal_test=True,matched='Check existing operator cache only; no new matched training or historical performance curves')
    write(OUT/'reports/protocol.json',protocol)
    dep={p:hashes[p] for p in hashes if p.endswith('.py')}
    write(OUT/'reports/execution_freeze.json',dict(hashes={p.relative_to(OUT).as_posix():sha(p) for d in ('src_or_scripts','tests') for p in (OUT/d).rglob('*.py')},dependencies=dep,reference_seed_sha256=sha(OUT/'data/reference_seed_registry.json'),protocol_sha256=sha(OUT/'reports/protocol.json'),first_new_H_generated=False))
    print(json.dumps(dict(prepare='PASS',new_seeds=len(used))),flush=True)
@functools.lru_cache(maxsize=2)
def source_metadata(split):
    from scripts.run_sp_u12b2a import seed_index
    receipts={r['identity']:r for r in read(MR/'reports'/f'{split}_summary.json')['receipts']}
    return receipts,seed_index(read(MR/'data/seed_registry.json')[split])
def source_input(split,condition,i):
    identity=f'MR01:{split}:{condition}:{i}'
    receipts,idx=source_metadata(split);expected=receipts[identity]
    jp=MR/'data'/split/f'{condition}_{i:04d}.json';p=jp.with_suffix('.npz')
    if sha(jp)!=expected['json_sha256'] or sha(p)!=expected['npz_sha256']:raise ValueError('MR01_RAW_SHA_IDENTITY')
    with zipfile.ZipFile(p) as z:
        if z.testzip():raise ValueError('MR01_RAW_CRC')
    rec=read(jp);z=dict(np.load(p,allow_pickle=False))
    if rec['identity']!=identity or ah(z['H'])!=rec['H_sha256'] or tuple(z['snr_db'])!=SNRS or tuple(z['frame_ids'])!=tuple(range(4,10)):raise ValueError('MR01_PHYSICAL_AXES')
    from scripts.run_sp_u12b2a import consumed
    if rec['seeds']!=consumed(idx,condition,'EVAL',i):raise ValueError('MR01_SEED_BINDING')
    for x in rec['inputs']:
        si=SNRS.index(x['snr_db']);j=x['frame_id']
        if ah(z['pilots'][si,j])!=x['pilot_sha256'] or ah(z['unit_noise'][j])!=x['unit_noise_sha256']:raise ValueError('MR01_PAIRED_PILOT_NOISE')
    return z,dict(original_identity=identity,original_json_sha256=expected['json_sha256'],original_npz_sha256=expected['npz_sha256'],original_path=p.relative_to(ROOT).as_posix())
def pilot_design(cv,values):
    D=[];z=[]
    for p in values:
        u=p-cv.source_mean_p
        for f,(obs,hold) in enumerate(cv.folds):
            x=u[obs]
            D.append(np.column_stack((cv.profile_delta_k[f]@x,cv.mobility_delta_k[f]@x)))
            z.append(u[hold]-cv.source_k[f]@x)
    D=np.concatenate(D);z=np.concatenate(z)
    return D,z
def diagnostics(H,pilots,original=None,refs=None,matched=None):
    from src.cdl_benchmark import frames
    hf=frames(H)
    a=dict(pilot_S=np.empty((3,6,2,2)),pilot_b=np.empty((3,6,2)),pilot_k=np.empty((3,6)),
      online=np.empty((3,6,2)),online_normal_residual=np.empty((3,6)),data_H=np.empty((3,6,2,2,2)),data_b=np.empty((3,6,2,2)),data_k=np.empty((3,6)),
      cross_gram=np.empty((3,6,2,2)),oracle=np.empty((3,6,2,2)),oracle_rank=np.empty((3,6,2),np.int64),
      oracle_sv=np.empty((3,6,2,2)),oracle_error=np.empty((3,6,2)),oracle_normal=np.empty((3,6,2)),
      truth=np.empty((6)),compression_A=np.empty((3,6)),compression_B=np.empty((3,6)),compression_difference=np.empty((3,6)),
      pilot_min_eigen_normalized=np.empty((3,6)),pilot_condition=np.empty((3,6)),data_min_eigen_normalized=np.empty((3,6,2)),data_condition=np.empty((3,6,2)),
      pilot_loss=np.empty((3,6,9)),baseline_error=np.empty((3,6,6)))
    if refs is not None:a['error']=np.empty((3,6,len(METHODS) if matched is not None else len(METHODS)-1))
    audit=dict(max_vector_quadratic_scaled_error=0.,max_compression_scaled_error=0.,max_pilot_Gram_scaled_error=0.,replayed_baseline_bitwise=True,replayed_coordinates_bitwise=True,oracle_inequalities_checked=0,numerically_rank_truncated_frames=0)
    for si,s in enumerate(SNRS):
        records=c.infer(pilots[si],op,s,cvs,factors)
        for ji,row in enumerate(records):
            j=ji+4;target=hf[j].ravel()[g.data];u=pilots[si,j]-op['source_mean_p'];base=row['predictions']['source'];r=target-base
            XD=np.column_stack([u@op[f'DeltaW{q}_{s}'].T for q in ('P','M')])
            XL=np.column_stack([factors[q,s].apply(u) for q in ('P','M')])
            cc=np.asarray(row['dense_coefficients']);diag=row['diagnostics']['affine'];S=np.array(diag['gram']);b=np.array(diag['rhs']);k=diag['source_cv_loss']
            if row['dense_coefficients'] is not row['lr4_coefficients'] or diag['regularized'] or diag['clipped']:raise ValueError('ORIGINAL_ONLINE_METHOD_DRIFT')
            D,z=pilot_design(cvs[s],pilots[si,j-4:j+1]);SG,bG,kG=m.real_sufficient(D,z)
            audit['max_pilot_Gram_scaled_error']=max(audit['max_pilot_Gram_scaled_error'],assert_close(S,SG,'PILOT_G_REAL'),assert_close(b,bG,'PILOT_b_REAL'),assert_close(k,kG,'PILOT_k_REAL'))
            a['pilot_S'][si,ji]=S;a['pilot_b'][si,ji]=b;a['pilot_k'][si,ji]=k;a['online'][si,ji]=cc
            a['online_normal_residual'][si,ji]=diag['normal_equation_relative_residual']
            if diag['normal_equation_relative_residual']>c.SOLVE_TOL:raise ValueError('ORIGINAL_DIRECT_SOLVE_RESIDUAL')
            peig=np.linalg.eigvalsh(S);a['pilot_min_eigen_normalized'][si,ji]=peig[0]/np.trace(S);a['pilot_condition'][si,ji]=peig[-1]/peig[0]
            a['truth'][ji]=float(np.vdot(target,target).real);a['data_k'][si,ji]=float(np.vdot(r,r).real);a['cross_gram'][si,ji]=(XD.conj().T@XL).real
            bc=[row['predictions'][name] for name in ('source','hard','convex','dense-affine','lr4-affine')]
            bc.insert(1,op['pool_mean_d']+op[f'Wpool_{s}']@(pilots[si,j]-op['pool_mean_p']))
            for mi,pred in enumerate(bc):a['baseline_error'][si,ji,mi]=float(np.vdot(pred-target,pred-target).real)
            if original is not None:
                for x,y,label in ((cc,original['coordinates'][si,ji],'COORDINATES'),(S,original['normal_gram'][si,ji],'GRAM'),(b,original['normal_rhs'][si,ji],'RHS'),(a['truth'][ji],original['frame_truth'][ji],'TRUTH'),(a['baseline_error'][si,ji],original['frame_error'][si,:6,ji],'ENERGY')):
                    assert_close(x,y,'MR01_REPLAY_'+label)
                    same=np.array_equal(x,y)
                    if label=='COORDINATES':audit['replayed_coordinates_bitwise'] &= same
                    else:audit['replayed_baseline_bitwise'] &= same
            for ai,X in enumerate((XD,XL)):
                Ha,ba,ka=m.real_sufficient(X,r);o=m.real_oracle(X,r,rcond=RCOND)
                a['data_H'][si,ji,ai]=Ha;a['data_b'][si,ji,ai]=ba;a['oracle'][si,ji,ai]=o['coeff'];a['oracle_rank'][si,ji,ai]=o['rank'];a['oracle_sv'][si,ji,ai]=o['singular_values'];a['oracle_error'][si,ji,ai]=o['error'];a['oracle_normal'][si,ji,ai]=o['normal_residual']
                eig=np.linalg.eigvalsh(Ha);a['data_min_eigen_normalized'][si,ji,ai]=eig[0]/np.trace(Ha);a['data_condition'][si,ji,ai]=eig[-1]/eig[0]
                audit['numerically_rank_truncated_frames']+=int(o['rank']<2)
                coords=[cc,np.zeros(2),np.array([1.,0.]),np.array([0.,1.]),o['coeff']]
                if refs is not None:coords += [refs['cp'][si],refs['cd'][si,ai]]
                for co in coords:
                    e=X@co-r;direct=float(np.vdot(e,e).real);quad=m.quadratic_risk(Ha,ba,ka,co)
                    scale=ka+2*abs(float(co@ba))+abs(float(co@Ha@co))
                    audit['max_vector_quadratic_scaled_error']=max(audit['max_vector_quadratic_scaled_error'],assert_close(direct,quad,'DIRECT_DATA_QUADRATIC',scale))
                    if o['error']>direct+ATOL+TOL*max(scale,direct):raise ValueError('ORACLE_INEQUALITY')
                    audit['oracle_inequalities_checked']+=1
                if refs is not None:
                    start=6+3*ai
                    for qi,co in enumerate((refs['cp'][si],refs['cd'][si,ai],o['coeff'])):
                        e=X@co-r;a['error'][si,ji,start+qi]=float(np.vdot(e,e).real)
            if refs is not None:
                a['error'][si,ji,:6]=a['baseline_error'][si,ji]
                if matched is not None:
                    e=matched[s].estimate(pilots[si,j])-target;a['error'][si,ji,-1]=float(np.vdot(e,e).real)
            eD=row['predictions']['dense-affine']-target;delta=row['predictions']['lr4-affine']-row['predictions']['dense-affine'];ct=m.compression_terms(eD,delta)
            difference=a['baseline_error'][si,ji,5]-a['baseline_error'][si,ji,4]
            a['compression_A'][si,ji]=ct['A'];a['compression_B'][si,ji]=ct['B'];a['compression_difference'][si,ji]=difference
            audit['max_compression_scaled_error']=max(audit['max_compression_scaled_error'],assert_close(difference,ct['total'],'COMPRESSION_IDENTITY',ct['A']+abs(ct['B'])))
            losscoords=[cc]+([refs['cp'][si],refs['cd'][si,0],refs['cd'][si,1]] if refs is not None else [np.zeros(2)]*3)+[a['oracle'][si,ji,0],a['oracle'][si,ji,1],np.zeros(2),np.array([1.,0.]),np.array([0.,1.])]
            for li,co in enumerate(losscoords):
                e=D@co-z;direct=float(np.vdot(e,e).real);a['pilot_loss'][si,ji,li]=direct
                assert_close(direct,m.quadratic_risk(S,b,k,co),'PILOT_LOSS_QUADRATIC',k+2*abs(float(co@b))+abs(float(co@S@co)))
    if any(not np.isfinite(v).all() for v in a.values()):raise ValueError('NONFINITE_DIAGNOSTIC_BLOCK')
    return a,audit
def save_block(folder,condition,i,arrays,identity,audit):
    path=OUT/'data'/folder/f'{condition}_{i:04d}.npz';payload=archive(path,**arrays)
    write(path.with_suffix('.json'),dict(identity=f'MR02:{folder}:{condition}:{i}',condition=condition,trajectory_id=i,role=folder,raw=payload,input=identity,audit=audit,complete=True))
    return dict(condition=condition,trajectory_id=i,path=payload['path'],sha256=payload['sha256'],json_sha256=sha(path.with_suffix('.json')))
def load_blocks(folder,n):
    receipts=read(OUT/'reports'/f'{folder}_receipt.json')['blocks'];by={(r['condition'],r['trajectory_id']):r for r in receipts}
    if len(by)!=5*n:raise ValueError('BLOCK_COUNT')
    arrays={}
    for ci,condition in enumerate(CONDITIONS):
        for i in range(n):
            r=by[condition,i];p=OUT/r['path']
            if sha(p)!=r['sha256'] or sha(p.with_suffix('.json'))!=r['json_sha256']:raise ValueError('SMALL_BLOCK_HASH')
            with zipfile.ZipFile(p) as z:
                if z.testzip():raise ValueError('SMALL_BLOCK_CRC')
            z=dict(np.load(p,allow_pickle=False))
            for k,v in z.items():
                if k not in arrays:arrays[k]=np.empty((5,n,*v.shape),dtype=v.dtype)
                arrays[k][ci,i]=v
    return arrays
def fit_arrays(a):
    cp=np.empty((5,3,2));cd=np.empty((5,3,2,2));ranks=[]
    moments=dict(Hp=a['pilot_S'].mean((1,3)),bp=a['pilot_b'].mean((1,3)),kp=a['pilot_k'].mean((1,3)),Hd=a['data_H'].mean((1,3)),bd=a['data_b'].mean((1,3)),kd=a['data_k'].mean((1,3)))
    for ci,condition in enumerate(CONDITIONS):
        for si,s in enumerate(SNRS):
            for ri in range(3):
                H,b=(moments['Hp'][ci,si],moments['bp'][ci,si]) if ri==0 else (moments['Hd'][ci,si,ri-1],moments['bd'][ci,si,ri-1])
                result=m.fixed_reference(H,b,rcond=RCOND)
                if ri==0:cp[ci,si]=result['coeff']
                else:cd[ci,si,ri-1]=result['coeff']
                eigen=np.linalg.eigvalsh(H)
                ranks.append(dict(condition=condition,snr_db=s,reference=('pilot','data-Dense','data-LR4')[ri],alpha=result['coeff'][0],beta=result['coeff'][1],rank=result['rank'],lambda_min=eigen[0],lambda_max=eigen[-1],condition_number=eigen[-1]/eigen[0] if eigen[0]>0 else 'infinite',normal_residual=result['normal_residual'],rcond=RCOND,diagnostic_only=True))
    return dict(cp=cp,cd=cd,**moments),ranks
def smoke():
    frozen();t=time.monotonic();rblocks=[]
    for condition in CONDITIONS:
        for i in range(2):
            raw,identity=source_input('smoke',condition,i);a,audit=diagnostics(raw['H'],raw['pilots'],original=raw);rblocks.append(save_block('smoke_ref',condition,i,a,identity,audit))
    write(OUT/'reports/smoke_ref_receipt.json',dict(blocks=rblocks,real_science=False))
    a=load_blocks('smoke_ref',2);refs,ranks=fit_arrays(a);archive(OUT/'data/smoke_references.npz',**refs)
    eblocks=[]
    for ci,condition in enumerate(CONDITIONS):
        raw,identity=source_input('smoke',condition,2);e,audit=diagnostics(raw['H'],raw['pilots'],original=raw,refs={k:v[ci] for k,v in refs.items()});eblocks.append(save_block('smoke_eval',condition,0,e,identity,audit))
    write(OUT/'reports/smoke_eval_receipt.json',dict(blocks=eblocks,real_science=False));e=load_blocks('smoke_eval',1)
    # This exercises the actual same trajectory-energy summary path before D_ref.
    point=e['error'].sum((1,3))/e['truth'].sum((1,2))[:,None,None]
    archive(OUT/'data/smoke_summary.npz',nmse=point,cp=refs['cp'],cd=refs['cd'])
    seconds=time.monotonic()-t;previous=read(MR/'reports/development_summary.json')['wall_seconds']
    estimated=previous+seconds*(1000/15)+90
    if estimated>7200:raise ValueError('RESOURCE_ESTIMATE_EXCEEDS_TWO_HOURS')
    write(OUT/'reports/smoke_report.json',dict(PASS=True,MR01_smoke_inputs=15,ref_fitting_trajectories=10,evaluation_trajectories=5,real_pipeline='input->diagnostic blocks->fixed references->heldout smoke evaluation->raw save->summary',seconds=seconds,estimated_full_seconds=estimated,science_tables_contain_smoke=False,reference_rank_counts={str(k):sum(r['rank']==k for r in ranks) for k in (0,1,2)}))
    print(json.dumps(dict(smoke='PASS',seconds=seconds,estimated_full_seconds=estimated)),flush=True)
def reference():
    import torch
    from scripts.sp_u4a_common import peak_memory
    from scripts.run_sp_u12b2a import generator,seed_index,consumed
    frozen()
    if not read(OUT/'reports/smoke_report.json')['PASS']:raise ValueError('SMOKE_NOT_PASS')
    registry=read(OUT/'data/reference_seed_registry.json');index=seed_index(registry)
    known=set(read(MR/'reports/b2_H_identity_readonly.json')['H_sha256'])
    known.update(read(p)['H_sha256'] for split in ('smoke','development') for p in (MR/'data'/split).glob('*.json'))
    with (ROOT/'outputs/sp_u4a_clean_hashes.csv').open(encoding='utf8',newline='') as f:
        for row in csv.DictReader(f):known.update(v for k,v in row.items() if 'hash' in k.lower() or 'sha' in k.lower())
    receipts=[];seen=set();peak=0;t=time.monotonic()
    for condition in CONDITIONS:
        gen=generator(condition,index,'EVAL')
        for i in range(100):
            h=gen.generate(i);hh=ah(h)
            if hh in known or hh in seen:raise ValueError('REFERENCE_H_IDENTITY_COLLISION')
            seen.add(hh);comm=[c.communicate(consumed(index,condition,'EVAL',i,j)) for j in range(10)]
            original,q,coords,pilots,inputs,_,normal,grams,rhs=old.evaluate(h,comm,op,cvs,factors)
            raw=dict(H=h,pilots=pilots,frame_error=original,frame_truth=q,coordinates=coords,normal_gram=grams,normal_rhs=rhs)
            inp=archive(OUT/'data/reference_inputs'/f'{condition}_{i:04d}.npz',H=h,pilots=pilots,unit_noise=np.stack([x[2] for x in comm]),bits=np.stack([x[0] for x in comm]))
            a,audit=diagnostics(h,pilots,original=raw)
            identity=dict(namespace='SP_U17_MR02_REFERENCE_V1',H_sha256=hh,seeds=consumed(index,condition,'EVAL',i),communication_seeds=[consumed(index,condition,'EVAL',i,j) for j in range(10)],input_archive=inp,pilot_frame_bindings=inputs)
            receipts.append(save_block('reference',condition,i,a,identity,audit));peak=max(peak,peak_memory()['peak_working_set_bytes'])
            if i%20==0 or i==99:print(json.dumps(dict(stage='reference',condition=condition,completed=len(receipts),elapsed_seconds=time.monotonic()-t,working_set_GiB=peak/2**30)),flush=True)
    write(OUT/'reports/reference_receipt.json',dict(PASS=True,blocks=receipts,physical_trajectories=500,scored_positions=9000,seconds=time.monotonic()-t,peak_working_set_bytes=peak,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),all_blocks_before_fit=True))
    preserve()
def fit():
    frozen();a=load_blocks('reference',100);refs,rows=fit_arrays(a)
    archive(OUT/'data/fixed_references.npz',**refs);archive(OUT/'data/reference_summary.npz',**a)
    csvout(OUT/'tables/fixed_references.csv',rows)
    write(OUT/'reports/reference_fit.json',dict(PASS=True,total_references=45,ranks={str(k):sum(r['rank']==k for r in rows) for k in (0,1,2)},fitted_only_D_ref=True,all_cells_covered=True,reference_archive_sha256=sha(OUT/'data/fixed_references.npz'),unique_identified=sum(r['rank']==2 for r in rows),window_frame_multiplicity=read(OUT/'reports/protocol.json')['window_frame_multiplicity']))
    print(json.dumps(dict(fit='PASS',identified=sum(r['rank']==2 for r in rows))),flush=True)
def matched_operators():
    # Exact existing B2 MATCHED_REF sufficient statistics, not its test outcomes.
    available={};details=[]
    for condition in CONDITIONS:
        path=ROOT/f'outputs/sp_u12b2b_sealed/reference/{condition}/statistics.npz'
        record=read(path.with_suffix('.json'));entry=record['artifacts'][0]
        if record['condition']!=condition or record['clean_trajectories']!=500 or record['clean_frames']!=5000 or sha(path)!=entry['sha256']:raise ValueError('EXISTING_MATCHED_STATISTICS_IDENTITY')
        st=c.statistics(path)
        available[condition]={s:st.prepare(c.variance(s)) for s in SNRS}
        details.append(dict(condition=condition,status='available',statistics=path.relative_to(ROOT).as_posix(),sha256=sha(path),extra_information='condition-specific independent finite B2 MATCHED_REF corpus; 500 trajectories/5000frames; no new fitting or historical evaluation curve'))
    write(OUT/'reports/matched_availability.json',dict(conditions=details))
    return available
def evaluate():
    from scripts.sp_u4a_common import peak_memory
    frozen();t=time.monotonic();refs=dict(np.load(OUT/'data/fixed_references.npz',allow_pickle=False));receipts=[];maxima={};peak=0
    available=matched_operators()
    for ci,condition in enumerate(CONDITIONS):
        for i in range(100):
            raw,identity=source_input('development',condition,i)
            a,audit=diagnostics(raw['H'],raw['pilots'],original=raw,refs={k:v[ci] for k,v in refs.items()},matched=available[condition])
            receipts.append(save_block('evaluation',condition,i,a,identity,audit));peak=max(peak,peak_memory()['peak_working_set_bytes'])
            for k,v in audit.items():
                if k.startswith('max_'):maxima[k]=max(maxima.get(k,0.),v)
                elif k.startswith('replayed'):maxima[k]=maxima.get(k,True) and v
                else:maxima[k]=maxima.get(k,0)+v
            if i%20==0 or i==99:print(json.dumps(dict(stage='evaluation',condition=condition,completed=len(receipts),elapsed_seconds=time.monotonic()-t)),flush=True)
    write(OUT/'reports/evaluation_receipt.json',dict(PASS=True,blocks=receipts,physical_trajectories=500,scored_positions=9000,seconds=time.monotonic()-t,peak_working_set_bytes=peak,replay=maxima,all_raw_blocks_saved_before_aggregate=True,matched_added=True))
    preserve()
def ref_bootstrap(ref,ev,refs):
    master=read(OUT/'reports/protocol.json')['reference_bootstrap_seed'];draws={};boot=np.empty((5,500,3,3,2));rank=np.empty((5,500,3,3),np.int64);rows=[];risk_boot=np.empty((5,500,3,2,2));distance=np.empty((5,500,3,2,2))
    for ci,condition in enumerate(CONDITIONS):
        idx=np.random.default_rng(np.random.SeedSequence([master,ci])).integers(0,100,size=(500,100),dtype=np.int64);draws[condition]=idx
        hp=ref['pilot_S'][ci].sum(2);bp=ref['pilot_b'][ci].sum(2);hd=ref['data_H'][ci].sum(2);bd=ref['data_b'][ci].sum(2)
        he=ev['data_H'][ci].mean((0,2));be=ev['data_b'][ci].mean((0,2));ke=ev['data_k'][ci].mean((0,2));qe=ev['truth'][ci].mean()
        for bi in range(500):
            for si in range(3):
                for ri in range(3):
                    H,b=(hp[idx[bi],si].sum(0),bp[idx[bi],si].sum(0)) if ri==0 else (hd[idx[bi],si,ri-1].sum(0),bd[idx[bi],si,ri-1].sum(0))
                    result=m.fixed_reference(H,b,rcond=RCOND);boot[ci,bi,si,ri]=result['coeff'];rank[ci,bi,si,ri]=result['rank']
                for ai in range(2):
                    for qi,ri in enumerate((0,ai+1)):
                        co=boot[ci,bi,si,ri];fixed=refs['cp'][ci,si] if qi==0 else refs['cd'][ci,si,ai];d=co-fixed
                        distance[ci,bi,si,ai,qi]=d@he[si,ai]@d/qe
                        risk_boot[ci,bi,si,ai,qi]=10*np.log10(m.quadratic_risk(he[si,ai],be[si,ai],ke[si],co)/qe)
        for si,s in enumerate(SNRS):
            for ri,name in enumerate(('pilot','data-Dense','data-LR4')):
                q=np.quantile(boot[ci,:,si,ri],[.025,.5,.975],axis=0)
                rows.append(dict(condition=condition,snr_db=s,reference=name,alpha_q025=q[0,0],alpha_median=q[1,0],alpha_q975=q[2,0],beta_q025=q[0,1],beta_median=q[1,1],beta_q975=q[2,1],bootstrap=500,full_rank_fraction=float(np.mean(rank[ci,:,si,ri]==2)),identity='D_ref trajectory bootstrap, conditional on fixed P, diagnostic only'))
    archive(OUT/'data/reference_bootstrap.npz',coordinates=boot,rank=rank,eval_nmse_db=risk_boot,eval_Gram_distance_linear=distance,**{f'indices_{k}':v for k,v in draws.items()})
    csvout(OUT/'tables/reference_stability.csv',rows)
    rr=[]
    for ci,condition in enumerate(CONDITIONS):
        for si,s in enumerate(SNRS):
            for ai,space in enumerate(SPACE_NAMES):
                for qi,name in enumerate(('cp','cd')):
                    q=np.quantile(risk_boot[ci,:,si,ai,qi],[.025,.5,.975]);d=np.quantile(distance[ci,:,si,ai,qi],[.025,.5,.975])
                    rr.append(dict(condition=condition,snr_db=s,space=space,reference=name,eval_nmse_q025=q[0],eval_nmse_median=q[1],eval_nmse_q975=q[2],eval_Gram_distance_q025=d[0],eval_Gram_distance_median=d[1],eval_Gram_distance_q975=d[2],identity='reference-fit uncertainty diagnostic, not joint evaluation/preparation CI'))
    csvout(OUT/'tables/reference_risk_stability.csv',rr)
def analyze():
    from scipy.stats import spearmanr
    frozen();t=time.monotonic();ref=load_blocks('reference',100);ev=load_blocks('evaluation',100);refs=dict(np.load(OUT/'data/fixed_references.npz',allow_pickle=False))
    archive(OUT/'data/eval_summary.npz',**ev)
    E=ev['error'].sum(3);Q=ev['truth'].sum(2);absolute=[];contrasts=[];compression=[];transfer=[];tails=[];rawrows=[];observed_old=[]
    with np.load(MR/'data/exploratory_bootstrap_indices.npz',allow_pickle=False) as draws:
        for ci,condition in enumerate(CONDITIONS):
            idx=draws[condition]
            if idx.shape!=(2000,100) or np.any(idx<0) or np.any(idx>=100):raise ValueError('MR01_BOOTSTRAP_DRAW_IDENTITIES')
            point=E[ci].sum(0)/Q[ci].sum();db=10*np.log10(point);bdb=10*np.log10(E[ci][idx].sum(1)/Q[ci][idx].sum(1)[:,None,None]);lo,hi=np.quantile(bdb,[.025,.975],axis=0,method='linear')
            for si,s in enumerate(SNRS):
                for mi,name in enumerate(METHODS):
                    absolute.append(dict(condition=condition,snr_db=s,method=name,error_energy=E[ci,:,si,mi].sum(),truth_energy=Q[ci].sum(),nmse_linear=point[si,mi],nmse_db=db[si,mi],CI_lower=lo[si,mi],CI_upper=hi[si,mi],trajectory_count=100,interval='Exploratory, unadjusted 95%, fixed D_ref; 2000 exact MR01 paired draws'))
                    for i in range(100):rawrows.append(dict(condition=condition,snr_db=s,method=name,trajectory_id=i,error_energy=E[ci,i,si,mi],truth_energy=Q[ci,i],scored_frames=6))
                for ai,space in enumerate(SPACE_NAMES):
                    online=4+ai;cp=6+3*ai;cd=cp+1;oracle=cp+2
                    pairs=[('online-cp',online,cp),('online-cd',online,cd),('cp-cd',cp,cd),('online-oracle',online,oracle),('cp-oracle',cp,oracle),('cd-oracle',cd,oracle),('online-Pooled',online,1),('cp-Pooled',cp,1),('cd-Pooled',cd,1),('oracle-Pooled',oracle,1)]
                    for name,i,j in pairs:
                        l,h=np.quantile(bdb[:,si,i]-bdb[:,si,j],[.025,.975],method='linear')
                        contrasts.append(dict(condition=condition,snr_db=s,space=space,contrast=name,first=METHODS[i],second=METHODS[j],estimate_db=db[si,i]-db[si,j],CI_lower=l,CI_upper=h,trajectory_count=100,bootstrap=2000,sign='first minus second; positive favors second',interval='unadjusted exploratory development'))
                    H=ev['data_H'][ci,:,si,:,ai].mean((0,1));b=ev['data_b'][ci,:,si,:,ai].mean((0,1));k=ev['data_k'][ci,:,si].mean();cpv=refs['cp'][ci,si];cdv=refs['cd'][ci,si,ai];d=cpv-cdv
                    exact=m.quadratic_risk(H,b,k,cpv)-m.quadratic_risk(H,b,k,cdv);dist=float(d@H@d);gradient=float(2*d@(H@cdv-b));assert_close(exact,dist+gradient,'EVAL_TRANSFER_GRADIENT',abs(dist)+abs(gradient));meanQ=Q[ci].sum()/600
                    transfer.append(dict(condition=condition,snr_db=s,space=space,eval_cp_minus_cd_linear=exact/meanQ,Gram_distance_linear=dist/meanQ,gradient_cross_term_linear=gradient/meanQ,identity_error=exact-dist-gradient,empirical_reference_gradient_norm=float(np.linalg.norm(H@cdv-b)),pilot_ref_loss=(refs['kp'][ci,si]-2*cpv@refs['bp'][ci,si]+cpv@refs['Hp'][ci,si]@cpv),data_ref_loss=(refs['kd'][ci,si]-2*cdv@refs['bd'][ci,si,ai]+cdv@refs['Hd'][ci,si,ai]@cdv)))
                A=ev['compression_A'][ci,:,si].sum();B=ev['compression_B'][ci,:,si].sum();difference=E[ci,:,si,5].sum()-E[ci,:,si,4].sum();assert_close(difference,A+B,'AGGREGATE_COMPRESSION',A+abs(B))
                compression.append(dict(condition=condition,snr_db=s,A_energy=A,B_energy=B,truth_energy=Q[ci].sum(),A_over_truth=A/Q[ci].sum(),B_over_truth=B/Q[ci].sum(),sum_over_truth=(A+B)/Q[ci].sum(),LR4_minus_Dense_linear=difference/Q[ci].sum(),LR4_minus_Dense_db=db[si,5]-db[si,4],identity_error=difference-A-B,negative_cross_term_fraction=float(np.mean(ev['compression_B'][ci,:,si]<0))))
                co=ev['online'][ci,:,si].reshape(-1,2);cond=ev['pilot_condition'][ci,:,si].ravel();norm=np.linalg.norm(co,axis=1);loss=(ev['error'][ci,:,si,:,4]-ev['oracle_error'][ci,:,si,:,0]).ravel()
                tails.append(dict(condition=condition,snr_db=s,alpha_min=co[:,0].min(),alpha_median=np.median(co[:,0]),alpha_max=co[:,0].max(),beta_min=co[:,1].min(),beta_median=np.median(co[:,1]),beta_max=co[:,1].max(),coordinate_norm_q95=np.quantile(norm,.95),coordinate_norm_max=norm.max(),pilot_condition_median=np.median(cond),pilot_condition_q95=np.quantile(cond,.95),pilot_min_eig_normalized_min=ev['pilot_min_eigen_normalized'][ci,:,si].min(),outside_simplex_fraction=float(np.mean((co[:,0]<0)|(co[:,1]<0)|(co.sum(1)>1))),Spearman_condition_coordinate_norm=spearmanr(cond,norm).statistic,Spearman_condition_Dense_oracle_excess=spearmanr(cond,loss).statistic,independence_warning='600 repeated frames from100 trajectories; descriptive associations, no causal or pvalue inference'))
    csvout(OUT/'tables/nmse.csv',absolute);csvout(OUT/'tables/contrasts.csv',contrasts);csvout(OUT/'tables/compression.csv',compression);csvout(OUT/'tables/risk_transfer.csv',transfer);csvout(OUT/'tables/coordinate_diagnostics.csv',tails);csvout(OUT/'tables/trajectory_energies.csv',rawrows)
    # Recover every saved MR01 numerical point and interval from actual new replay,
    # not from its PASS report. No new draws and no old result file overwritten.
    with (MR/'tables/nmse_all_methods.csv').open(encoding='utf8',newline='') as f:oldrows=list(csv.DictReader(f))
    current={(r['condition'],str(r['snr_db']),r['method']):r for r in absolute};mx=0.
    for row in oldrows:
        if row['method']=='Pooled-source-centered':continue
        name={'Dense':'Dense-online','LR4':'LR4-online'}.get(row['method'],row['method']);r=current[row['condition'],row['snr_db'],name]
        for key in ('error_energy','truth_energy','nmse_linear','nmse_db','CI_lower','CI_upper'):mx=max(mx,assert_close(float(r[key]),float(row[key]),'MR01_SAVED_TABLE_RECHECK'))
    write(OUT/'reports/MR01_numerical_recheck.json',dict(PASS=True,replayed_rows=90,max_scaled_difference=mx,saved_draws_reused=True,actual_raw_block_replay=True,not_inferred_from_PASS_flag=True))
    ref_bootstrap(ref,ev,refs)
    write(OUT/'reports/analysis_report.json',dict(PASS=True,absolute_rows=len(absolute),comparison_rows=len(contrasts),compression_cells=15,transfer_rows=30,D_ref=500,D_eval=500,method_scored_positions=9000,total_fixed_references=45,total_data_oracles=18000,bootstrap_eval=2000,bootstrap_reference=500,bootstrap_eval_indices_sha256=sha(MR/'data/exploratory_bootstrap_indices.npz'),seconds=time.monotonic()-t,preservation=preserve(),scientific_interpretation='Exploratory mechanisms with truth/labels assisted diagnostics; not a new estimator or confirmation',future_tasks_started=False))
    print(json.dumps(dict(analysis='PASS',seconds=time.monotonic()-t,rows=len(absolute))),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','smoke','reference','fit','evaluate','analyze'));p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();init(a.root,a.out)
    try:globals()[a.action]()
    except Exception as e:
        write(OUT/'reports'/f'{a.action}_failure_{time.time_ns()}.json',dict(status='STOP',action=a.action,error=repr(e),traceback=traceback.format_exc(),original_files_unchanged=True));raise
if __name__=='__main__':main()
