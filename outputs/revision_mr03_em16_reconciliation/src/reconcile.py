"""MR03B existing-entity reconciliation; no fit, rank selection or formal replay."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='6';os.environ['MKL_NUM_THREADS']='6';os.environ['OMP_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,csv,hashlib,zipfile,shutil,time,argparse,traceback,importlib.util,math
import numpy as np
from scipy.linalg import cho_factor,cho_solve
ROOT=Path(r'D:\Users\yuhua\Desktop\Project');MR=ROOT/'outputs/revision_mr03_ppca_capacity_controls';OUT=ROOT/'outputs/revision_mr03_em16_reconciliation'
sys.path[:0]=[str(ROOT),str(MR/'src_or_scripts')]
import ppca
CONDITIONS=('ID','Profile','Mobility','Composed','Boundary');SNRS=(10,20,30)
ZIP_SHA='74d2238736b4047d0dff254775e4b6cc952c2074358cccb4907ed8d5fbc6c170'
INNER_SHA='22056806f377641cc7c36e9ce7e5decefdaf2c87d4b2e8d37b66b6ae5d2d6c54'
P_SHA='2553aab058511770ca95036d6e0d0e7fee9248080b43e826f672ea156099aed0'
PUBLIC=('means','lambdas','psis','amps','covs','inv_covs')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ah(a):return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def objsha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def archive(p,**v):
    with Path(p).open('xb') as f:np.savez_compressed(f,**v)
def csvout(p,rs):
    with Path(p).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def check(a,b,label,atol=1e-10,rtol=1e-9):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('FINITE_SHAPE:'+label)
    error=float(np.max(abs(a-b),initial=0));scale=max(float(np.max(abs(a),initial=0)),float(np.max(abs(b),initial=0)))
    if error>atol+rtol*scale:raise ValueError('NUMERICAL:'+label+':'+str((error,scale)))
    return dict(max_absolute=error,scaled=error/(1+scale),atol=atol,rtol=rtol)
def conditional_from_covariance(C,mean,pidx,didx,sigma2):
    pidx,didx=np.asarray(pidx),np.asarray(didx)
    if sigma2<=0 or set(pidx)&set(didx):raise ValueError('CONDITIONAL_DOMAIN')
    cpp=C[np.ix_(pidx,pidx)]+sigma2*np.eye(len(pidx));cdp=C[np.ix_(didx,pidx)]
    W=cho_solve(cho_factor(cpp,lower=True),cdp.conj().T).conj().T
    return W,mean[pidx],mean[didx]
def predict(W,mp,md,p):return md+(p-mp)@W.T
def verify():
    for p,h in read(OUT/'reports/preservation_before.json')['hashes'].items():
        if sha(ROOT/p)!=h:raise ValueError('ORIGINAL_CHANGED:'+p)
    if (OUT/'reports/execution_freeze.json').exists():
        for p,h in read(OUT/'reports/execution_freeze.json')['hashes'].items():
            if sha(OUT/p)!=h:raise ValueError('NEW_SCIENCE_CHANGED:'+p)
    return dict(PASS=True,protected_files=len(read(OUT/'reports/preservation_before.json')['hashes']),changed=0)
def arrays():return {n:np.load(OUT/'original_model'/(n+'.npy'),mmap_mode='r',allow_pickle=False) for n in PUBLIC}
def prepare():
    if OUT.exists():raise FileExistsError('NEW_OUTPUT_ALREADY_EXISTS')
    for d in ('reports','src','tests','tables','data','figures','original_model'):(OUT/d).mkdir(parents=True,exist_ok=True)
    p=ROOT/'EM16_final_model_transfer.zip';actual=sha(p)
    if actual!=ZIP_SHA:raise ValueError('LAB_TRANSFER_ZIP_SHA')
    top=read(ROOT/'FINAL_MODEL_MANIFEST.json')
    if sha(ROOT/'FINAL_MODEL_MANIFEST.json')!=(ROOT/'FINAL_MODEL_MANIFEST.sha256').read_text(encoding='ascii').strip():raise ValueError('OUTER_SHA')
    with zipfile.ZipFile(p) as z:
        expected={n+'.npy' for n in PUBLIC}|{'training_history.npy','model_manifest.json'}
        if set(z.namelist())!=expected or len(z.namelist())!=8:raise ValueError('EXACT_EIGHT_MEMBERS')
        b=z.read('model_manifest.json')
        if hashlib.sha256(b).hexdigest()!=INNER_SHA or top['final_model_manifest_sha256']!=INNER_SHA:raise ValueError('INNER_SHA')
        m=json.loads(b.decode('utf-8-sig'));ctx=m['context']
        if (m['K'],m['rank'],m['D'],m['N'],m['iterations'],m['convergence'])!=(1,16,4032,15000,22,'CONVERGED'):raise ValueError('IDENTITY_COUNTS')
        if ctx['role']!='final' or ctx['seed']!=4143049057 or not ctx['fresh_fit'] or ctx['candidate_checkpoint_used'] or ctx['data_sha256']!=P_SHA:raise ValueError('FINAL_TRAINING_IDENTITY')
        if top['full_data_sha256']!=P_SHA or top['final_seed']!=4143049057 or top['selected']['K']!=1 or top['selected']['rank']!=16:raise ValueError('OUTER_IDENTITY')
        if ctx['SELECTION_MANIFEST_SHA256']!=top['SELECTION_MANIFEST_SHA256']:raise ValueError('SELECTION_BINDING')
        if set(m['file_hashes'])!=expected-{'model_manifest.json'}:raise ValueError('NPY_HASH_INVENTORY')
        for name in sorted(expected):
            target=OUT/'original_model'/name
            with z.open(name) as f,target.open('xb') as out:shutil.copyfileobj(f,out)
            if name!='model_manifest.json' and sha(target)!=m['file_hashes'][name]:raise ValueError('NPY_HASH:'+name)
    a=arrays();facts=[]
    for n in PUBLIC:
        v=a[n];meta=m['arrays'][n]
        if list(v.shape)!=meta['shape'] or v.dtype.str!=meta['dtype'] or ah(v)!=meta['sha256']:raise ValueError('PARAMETER_ARRAY:'+n)
        if not np.isfinite(v).all():raise ValueError('PARAMETER_NONFINITE:'+n)
        facts.append(dict(file=n+'.npy',shape=json.dumps(list(v.shape)),dtype=v.dtype.str,file_SHA256=sha(OUT/'original_model'/(n+'.npy')),payload_SHA256=ah(v)))
    if objsha(m['arrays'])!=m['parameter_hash'] or objsha(m['initialization'])!=m['initialization_hash']:raise ValueError('OBJECT_HASH')
    model=ppca.from_author_arrays(a['means'],a['lambdas'],a['psis'],a['amps'])
    worst=0
    for s in range(0,4032,64):
        ex=model.loading[s:s+64]@model.loading.conj().T
        ex[np.arange(ex.shape[0]),np.arange(s,s+ex.shape[0])]+=model.nu
        ck=check(ex,a['covs'][0,s:s+64],'SAVED_PRIOR_COVARIANCE');worst=max(worst,ck['max_absolute'])
    history=np.load(OUT/'original_model/training_history.npy',allow_pickle=False)
    ratio=abs((history[-1]-history[-2])/history[-1])
    if history.shape!=(22,) or history.dtype!=np.float64 or not np.isfinite(history).all() or ratio>=1e-4:raise ValueError('ACTUAL_HISTORY_STOP')
    needed=('reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py','reference/sp_u12b5a/paper_period/cplx-mfa/utils.py','reference/sp_u12b5a/paper_period/MFA_estimator/MFA_estimator.py','src/baselines/mfa_reference_adapter.py','outputs/sp_u12b5b/full_retraining.npy','outputs/sp_u12b5b/input_manifest.json','configs/sp_u12b5b_mfa_selection.yaml')
    for n in needed:
        if sha(ROOT/n)!=ctx['author_and_adapter_hashes'][n]:raise ValueError('TRAINING_SOURCE:'+n)
    # Historic protected source/data bindings plus every old MR03 file are immutable.
    protected=read(MR/'reports/preservation_before.json')['hashes'].copy()
    for q in MR.rglob('*'):
        if q.is_file():protected[q.relative_to(ROOT).as_posix()]=sha(q)
    for n in needed+('FINAL_MODEL_MANIFEST.json','FINAL_MODEL_MANIFEST.sha256','EM16_final_model_transfer.zip','deployment/b5c_paired_r2_v1/science.py','deployment/r6_formal_v1/model_ops.py'):
        protected[n]=sha(ROOT/n)
    write(OUT/'reports/preservation_before.json',dict(hashes=protected))
    csvout(OUT/'tables/original_parameter_inventory.csv',facts)
    write(OUT/'reports/model_identity.json',dict(PASS=True,ZIP_SHA256=actual,ZIP_expected_source='Human scientific operator reported lab PowerShell Get-FileHash output; no original sidecar existed',lab_ZIP_path=r'D:\Research\Project_MFA_Lab\EM16_final_model_transfer.zip',inner_SHA256=INNER_SHA,outer_SHA256=sha(ROOT/'FINAL_MODEL_MANIFEST.json'),seven_file_hashes=m['file_hashes'],all_payload_hashes_PASS=True,training_shape=[15000,4032],physical_preparation_trajectories=1500,seed=4143049057,K=1,rank=16,iterations=22,convergence='CONVERGED',last_history_relative_change=float(ratio),history_negative_increment_count=int(np.count_nonzero(np.diff(history)<0)),history_timing='scores pre-update parameters; final saved parameters follow last update',isotropic_nu=model.nu,weights=a['amps'].tolist(),covariance_identity_max_absolute=worst,trained_sources_checked=list(needed),candidate_checkpoint_used=False,original_model_modified=False))
    write(OUT/'config.json',dict(conditions=CONDITIONS,snrs=SNRS,frames=list(range(4,10)),datasets=['V_new','D_eval'],methods=['ORIGINAL_EM16']+read(MR/'config.json')['methods'],bootstrap_replicates=2000,interval='unadjusted exploratory 95% paired trajectory percentile',reuse_original_draws=True,no_training=True,no_new_dataset=True,rtol=1e-9,atol=1e-10,fidelity_checks_before_evaluation=True))
    print(json.dumps(dict(stage='identity',PASS=True,nu=model.nu,history_relative=float(ratio),files=len(protected))),flush=True)
def author_model(a,m):
    from src.baselines.mfa_reference_adapter import make_reference_model,reference_modules
    # Reuse the already validated source-only importer; no reference bytecode is used.
    lp=ROOT/'deployment/r7_evidence_audit_r1_v1/source_only_loader.py'
    spec=importlib.util.spec_from_file_location('_mr03b_original_source_loader',lp);loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    expected={p:{'sha256':h} for p,h in read(OUT/'reports/preservation_before.json')['hashes'].items() if p.endswith('.py')}
    expected.update({p.relative_to(ROOT).as_posix():{'sha256':sha(p)} for p in (OUT/'src').glob('*.py')})
    loader.install(ROOT,expected)
    ref=make_reference_model(1,16);ref.D=4032;ref.N=15000
    for n,v in a.items():setattr(ref,n,v)
    ref.L_all=np.load(OUT/'original_model/training_history.npy',allow_pickle=False).tolist()
    return ref,reference_modules(),loader
def compare():
    verify();start=time.monotonic();a=arrays();em=ppca.from_author_arrays(a['means'],a['lambdas'],a['psis'],a['amps'])
    with np.load(MR/'data/PPCA_DIRECT_R16.npz',allow_pickle=False) as z:direct=ppca.PPCAModel(z['mean'],z['loading'],float(z['nu']))
    from src.baselines.mfa_reference_adapter import ProjectPilotInterface,MFAReferenceAdapter
    from scripts.sp_u12b4a_adapter import rss
    interface=ProjectPilotInterface();pidx,didx=interface.pilot_positions,interface.data_positions
    with np.load(MR/'data/shared_spectrum.npz',allow_pickle=False) as z:
        check(z['pilot_positions'],pidx,'PILOT_ORDER');check(z['data_positions'],didx,'DATA_ORDER')
    ref,mods,loader=author_model(a,read(OUT/'original_model/model_manifest.json'));adapter=MFAReferenceAdapter(ref,interface.A)
    sigma_rule=read(MR/'config.json')['source_noise_power'];ops={};fid=[];comparison=[]
    saved_public={n:ah(v) for n,v in a.items()}
    for si,s in enumerate(SNRS):
        sigma=sigma_rule/10**(s/10)
        W,mp,md=conditional_from_covariance(a['covs'][0],em.mean,pidx,didx,sigma)
        co=ppca.conditional_operator(em,pidx,didx,sigma);kernel=check(W,co.W,'COVARIANCE_VS_LOADING')
        input_rows=[]
        for c in CONDITIONS:
            with np.load(MR/'data/V_new'/f'{c}_0000.npz',allow_pickle=False) as z:input_rows.extend([z['pilots'][si,4],z['pilots'][si,9]])
        p=np.asarray(input_rows);calls=[];terms=[];estfile=mods[1].__file__;previous=sys.gettrace()
        def trace(frame,event,arg):
            if event=='call' and frame.f_code.co_filename==estfile:
                if frame.f_code.co_name=='estimate':calls.append([type(frame.f_locals['n_summands_or_proba']).__name__,frame.f_locals['n_summands_or_proba']])
                if frame.f_code.co_name=='_lmmse':terms.append(1)
            return trace
        try:sys.settrace(trace);out=adapter.estimate(p,sigma*np.eye(64))
        finally:sys.settrace(previous)
        expected=predict(W,mp,md,p);result=check(out[:,didx],expected,'AUTHOR_VS_INDEPENDENT')
        single=adapter.estimate(p[:1],sigma*np.eye(64));singlecheck=check(single[0],out[0],'AUTHOR_SINGLE_BATCH')
        responsibilities=ref.predict_proba(p)
        if calls!=[['float',1.0]] or len(terms)!=10 or not np.array_equal(responsibilities,np.ones((10,1))):raise ValueError('FLOAT1_K1_RESPONSIBILITY')
        ops[f'EM16_{s}_W']=W;ops[f'EM16_{s}_mean_p']=mp;ops[f'EM16_{s}_mean_d']=md
        archive(OUT/'data'/f'fidelity_{s}.npz',pilots=p,author_joint=out,independent_data=expected,responsibility=responsibilities)
        fid.append(dict(SNR_dB=s,sigma2=sigma,author_vs_covariance=result,loading_vs_covariance=kernel,single_batch=singlecheck,all_component_API_calls=calls,conditional_calls=len(terms),K1_responsibilities_exactly_one=True))
        comparison.append(dict(snr_db=s,**ppca.model_comparison(em,direct,pidx,didx,sigma)))
    if {n:ah(v) for n,v in a.items()}!=saved_public:raise ValueError('PUBLIC_MODEL_CHANGED_BY_INFERENCE')
    archive(OUT/'data/em16_conditional_operators.npz',**ops)
    write(OUT/'reports/conditional_fidelity.json',dict(PASS=True,checks=fid,source_loader_observations=loader.observations,public_parameters_unchanged=True,history_B5C_inference_definition='same frozen author + adapter; independently checked real model conditional mean',author_pinverse_not_added_to_new_independent_map=True,original_arrays_readonly=True))
    # Recompute empirical moments only, never redo spectral construction or fit.
    P=np.load(ROOT/'outputs/sp_u12b5b/full_retraining.npy',mmap_mode='r',allow_pickle=False)
    mean,S,moments=ppca.training_moments(P,block_size=256);P._mmap.close();del P
    scores={};score_rows=[]
    for name,model in [('ORIGINAL_EM16',em),('PPCA_DIRECT_R16',direct)]:
        score=ppca.training_logscore(model,mean,S);scores[name]=score
        score_rows.append(dict(method=name,nu=model.nu,**{k:v for k,v in score.items() if isinstance(v,(int,float,bool))}))
    del S
    csvout(OUT/'tables/training_scores.csv',score_rows)
    csvout(OUT/'tables/em16_vs_direct16.csv',[{k:(json.dumps(v) if isinstance(v,list) else v) for k,v in row.items()} for row in comparison])
    write(OUT/'reports/model_comparison.json',dict(PASS=True,same_statistical_family=True,same_preparation=True,same_free_mean_and_isotropic_residual_family=True,optimizer_update_not_identical=True,EM_floor_active=bool(em.nu<=1e-6),Direct16_floor_active=bool(direct.nu<=1e-6),scores=scores,comparisons=comparison,moments=moments,elapsed_seconds=time.monotonic()-start,resource=rss(),original_EM_fit_executed=False))
    print(json.dumps(dict(stage='compare',PASS=True,score_EM=scores['ORIGINAL_EM16']['mean_complex_dimension_logscore'],score_direct=scores['PPCA_DIRECT_R16']['mean_complex_dimension_logscore'],fidelity=fid,resource=rss())),flush=True)
def evaluate(dataset):
    verify()
    if not read(OUT/'reports/conditional_fidelity.json')['PASS']:raise ValueError('FIDELITY_REQUIRED')
    from scripts import sp_u12b2a_common as common
    from scripts.run_sp_u12b2a import generator,seed_index
    from src.cdl_benchmark import array_hash,frames,observation_geometry
    sys.path.insert(0,str(ROOT/'outputs/revision_mr05_compression_controls/src_or_scripts'))
    import run_mr05 as old
    receipts={(r['condition'],r['trajectory_id']):r for r in read(MR/'reports'/f'{dataset}_receipt.json')['receipts']}
    with np.load(OUT/'data/em16_conditional_operators.npz',allow_pickle=False) as z:ops=dict(z)
    folder=OUT/'data'/dataset;folder.mkdir(exist_ok=False);g=observation_geometry();start=time.monotonic();written=[];worst=0
    methods=['ORIGINAL_EM16']+read(MR/'config.json')['methods']
    if dataset=='V_new':idx=seed_index(read(MR/'data/validation_seed_registry.json'))
    for c in CONDITIONS:
        if dataset=='V_new':gen=generator(c,idx,'EVAL')
        for i in range(100):
            rec=receipts[c,i];p=MR/rec['path'];j=p.with_suffix('.json')
            if sha(p)!=rec['sha256'] or sha(j)!=rec['json_sha256']:raise ValueError('ORIGINAL_RAW_HASH')
            meta=read(j)
            with np.load(p,allow_pickle=False) as z:raw=dict(z)
            if dataset=='V_new':h=gen.generate(i)
            else:
                original,binding=old.source_input(c,i);h=original['H']
                if not np.array_equal(original['pilots'],raw['pilots']):raise ValueError('D_PILOT_BYTES')
            if array_hash(h)!=meta['input']['H_sha256'] or array_hash(raw['pilots'])!=meta['input']['pilot_sha256']:raise ValueError('PAIRED_H_PILOT_IDENTITY')
            hf=frames(h);target=np.stack([hf[t].ravel()[g.data] for t in range(4,10)])
            q=np.real(np.einsum('ij,ij->i',target.conj(),target));check(q,raw['frame_truth'],'TRUTH_ENERGY')
            energy=np.empty((3,1,6),np.float64);samples=[]
            for si,s in enumerate(SNRS):
                pred=predict(ops[f'EM16_{s}_W'],ops[f'EM16_{s}_mean_p'],ops[f'EM16_{s}_mean_d'],raw['pilots'][si,4:10])
                if not np.isfinite(pred).all():raise ValueError('EM_NONFINITE')
                err=pred-target;energy[si,0]=np.real(np.einsum('ij,ij->i',err.conj(),err));samples.append(pred[0,:16])
            # Existing Direct16/32/64/Pooled/LR4/Whole predictions are not rerun.
            combined=np.concatenate([energy,raw['frame_error']],axis=1)
            path=folder/f'{c}_{i:04d}.npz'
            archive(path,frame_error=combined,frame_truth=raw['frame_truth'],methods=np.array(methods),snr_db=np.array(SNRS),frame_ids=np.arange(4,10),EM_vector_example_frame4=np.array(samples))
            info=dict(dataset=dataset,condition=c,trajectory_id=i,identity=meta['identity'],H_sha256=meta['input']['H_sha256'],pilot_sha256=meta['input']['pilot_sha256'],original_MR03_npz_sha256=sha(p),original_MR03_json_sha256=sha(j),raw_path=path.relative_to(OUT).as_posix(),raw_sha256=sha(path),all_existing_method_energies_exactly_reused=True,truth_energy_exactly_reused=True,new_EM_positions=18,complete=True)
            write(path.with_suffix('.json'),info);written.append(dict(**info,JSON_sha256=sha(path.with_suffix('.json'))))
            if i%25==0:print(json.dumps(dict(dataset=dataset,condition=c,completed=i+1,elapsed=time.monotonic()-start)),flush=True)
    write(OUT/'reports'/f'{dataset}_receipt.json',dict(PASS=True,dataset=dataset,physical_trajectories=500,condition_SNR_cells=15,scored_EM_positions=9000,existing_methods_reused=11,receipts=written,nonfinite=0,input_H_hash_matches=500,pilot_hash_matches=500,new_noise_realizations=0,new_dataset=False,elapsed_seconds=time.monotonic()-start))
    print(json.dumps(dict(stage=dataset,PASS=True,elapsed=time.monotonic()-start)),flush=True)
def analyze():
    verify();sys.path.insert(0,str(MR/'src_or_scripts'))
    from independent_review_capacity import cell_statistics,linear_percentiles
    methods=read(OUT/'config.json')['methods'];nmse=[];contrasts=[];trajectory=[]
    pairs=[('ORIGINAL_EM16',m) for m in ('PPCA_DIRECT_R16','PPCA_DIRECT_R32','PPCA_DIRECT_R64','POOLED_NATURAL','LR4_TW1','LR4_TW5','WHOLE_POOL_M2_R23','WHOLE_POOL_M2_R24')]
    arrays_out={}
    for ds in ('V_new','D_eval'):
        receipt=read(OUT/'reports'/f'{ds}_receipt.json')
        if not receipt['PASS'] or len(receipt['receipts'])!=500:raise ValueError('DATASET_INCOMPLETE')
        with np.load(MR/'data'/f'bootstrap_{ds}.npz',allow_pickle=False) as z:draws=dict(z)
        for c in CONDITIONS:
            es=[];qs=[]
            for r in receipt['receipts']:
                if r['condition']!=c:continue
                p=OUT/r['raw_path']
                if sha(p)!=r['raw_sha256'] or sha(p.with_suffix('.json'))!=r['JSON_sha256']:raise ValueError('NEW_RAW_CHANGED')
                with np.load(p,allow_pickle=False) as z:es.append(z['frame_error']);qs.append(z['frame_truth'])
            stats=cell_statistics(np.array(es),np.array(qs),draws[c]);ci=linear_percentiles(stats['bootstrap_db'])
            for si,s in enumerate(SNRS):
                for mi,m in enumerate(methods):
                    nmse.append(dict(dataset=ds,condition=c,snr_db=s,method=m,nmse_linear=float(stats['nmse_linear'][si,mi]),nmse_db=float(stats['nmse_db'][si,mi]),CI_lower=float(ci[0,si,mi]),CI_upper=float(ci[1,si,mi]),trajectory_count=100,scored_frames=600))
                for first,second in pairs:
                    a,b=methods.index(first),methods.index(second);boot=stats['bootstrap_db'][:,si,a]-stats['bootstrap_db'][:,si,b];bounds=linear_percentiles(boot)
                    contrasts.append(dict(dataset=ds,condition=c,snr_db=s,first=first,second=second,estimate_db=float(stats['nmse_db'][si,a]-stats['nmse_db'][si,b]),CI_lower=float(bounds[0]),CI_upper=float(bounds[1]),trajectory_count=100,bootstrap=2000,interval='Unadjusted exploratory 95% trajectory paired percentile; both sets already disclosed',sign='first minus second; positive favors second',EM_worse_trajectory_fraction=float(np.mean(stats['error'][:,si,a]>stats['error'][:,si,b]))))
                for t in range(100):
                    for mi,m in enumerate(methods):trajectory.append(dict(dataset=ds,condition=c,trajectory_id=t,snr_db=s,method=m,error_energy=float(stats['error'][t,si,mi]),truth_energy=float(stats['truth'][t])))
            arrays_out[ds+'_'+c+'_bootstrap_linear']=stats['bootstrap_linear']
    csvout(OUT/'tables/capacity_comparison.csv',nmse);csvout(OUT/'tables/paired_contrasts.csv',contrasts);csvout(OUT/'tables/trajectory_energies.csv',trajectory)
    archive(OUT/'data/paired_bootstrap_statistics.npz',**arrays_out)
    write(OUT/'reports/analysis.json',dict(PASS=True,NMSE_rows=len(nmse),contrast_rows=len(contrasts),new_random_draws=0,original_draws_SHA256={ds:sha(MR/'data'/f'bootstrap_{ds}.npz') for ds in ('V_new','D_eval')},statistical_unit='physical trajectory',independent_confirmation=False,formal_B5C_not_recomputed=True))
    print(json.dumps(dict(stage='analysis',PASS=True,rows=len(nmse),contrasts=len(contrasts))))
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','compare','V_new','D_eval','analyze']);args=p.parse_args()
    try:
        if args.action=='prepare':prepare()
        elif args.action=='compare':compare()
        elif args.action=='analyze':analyze()
        else:evaluate(args.action)
    except BaseException as e:
        if (OUT/'reports').exists():write(OUT/'reports'/f'{args.action}_STOP_{time.time_ns()}.json',dict(status='STOP',error=repr(e),traceback=traceback.format_exc(),no_original_files_changed=True))
        raise
if __name__=='__main__':main()
