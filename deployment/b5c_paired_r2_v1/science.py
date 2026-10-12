"""Identity transport and original MFA inference; target truth stays in scorer."""
from pathlib import Path
import functools,importlib.util,sys,time
import numpy as np
import base as b
ROLES=('sionna_seed','torch_seed','numpy_seed','bits_seed_tx0','bits_seed_tx1','noise_seed')
@functools.lru_cache(maxsize=1)
def modules():
    b.verify();p=b.contract();loaderpath=b.ROOT/'deployment/r7_evidence_audit_r1_v1/source_only_loader.py'
    spec=importlib.util.spec_from_file_location('_b5c_source_loader',loaderpath)
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    sources={n:v for n,v in p['source_dependencies'].items() if n.endswith('.py')}
    sources.update({n:v for n,v in b.read(b.HERE/'package_manifest.json')['files'].items() if n.endswith('.py')})
    loader.install(b.ROOT,sources)
    sys.path.insert(0,str(b.ROOT/'deployment/r6_formal_v1'));sys.path.insert(0,str(b.ROOT))
    import common,model_ops,supervisor,windows_resources
    return common,model_ops,supervisor,windows_resources
def array_hash(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def final_identity():
    p=b.contract();top=b.ROOT/'lab_runs/r6_formal/FINAL_MODEL_MANIFEST.json'
    b.require(b.sha(top)==p['final_manifest_sha256']==top.with_suffix('.sha256').read_text(encoding='ascii').strip(),'LAB_FINAL_TOP_HASH_DRIFT')
    v=b.read(top);s=v['selected']
    b.require(v['PASS'] is True and (s['K'],s['rank'],v['final_seed'])==(1,16,4143049057),'LAB_FINAL_IDENTITY_DRIFT')
    b.require(v['full_data_sha256']==p['full_training_data_sha256'] and v['SELECTION_MANIFEST_SHA256']==p['selection_sha256'],'LAB_FINAL_PROVENANCE_DRIFT')
    for name,key in [('model/model_manifest.json','final_model_manifest_sha256'),('training_result.json','training_result_sha256')]:
        b.require(b.sha(b.FINAL/name)==p[key]==v[key],'LAB_SUBORDINATE_HASH_DRIFT:'+name)
    manifest=b.read(b.FINAL/'model/model_manifest.json');ctx=manifest['context']
    b.require((manifest['K'],manifest['rank'],manifest['D'],manifest['N'],manifest['iterations'],manifest['convergence'])==(1,16,4032,15000,22,'CONVERGED'),'LAB_MODEL_COUNTS_OR_CONVERGENCE_DRIFT')
    b.require(ctx['role']=='final' and ctx['seed']==4143049057 and ctx['data_sha256']==p['full_training_data_sha256'] and ctx['fresh_fit'] is True and ctx['candidate_checkpoint_used'] is False,'LAB_FINAL_FRESH_FIT_DRIFT')
    frozen=b.read(b.ROOT/'deployment/r6_formal_v1/input_contract.json')['frozen_hashes']
    b.require(ctx['author_and_adapter_hashes']==frozen,'LAB_TRAINING_SOURCE_BINDING_DRIFT')
    b.require(set(manifest['file_hashes'])==set(p['model_file_names']),'SEVEN_MODEL_FILES_REQUIRED')
    for n,h in manifest['file_hashes'].items():b.require(b.sha(b.FINAL/'model'/n)==h,'LAB_MODEL_NPY_HASH_DRIFT:'+n)
    result=b.read(b.FINAL/'training_result.json')
    b.require(result['PASS'] and result['serialization_roundtrip'] and result['all_components_smoke']['PASS'] and result['seed']==4143049057 and result['target_access']==0,'FINAL_TRAINING_RECEIPT_NOT_PASS')
    return {'PASS':True,'top_manifest_sha256':b.sha(top),'seven_model_file_hashes':manifest['file_hashes'],'iterations':22,'convergence':'CONVERGED','fresh_fit':True}
def load_final():
    identity=final_identity();c,m,_,_=modules();model,manifest=m.load_model(b.FINAL/'model')
    legal=m.parameter_validity(model,full=True);smoke=m.inference_smoke(model)
    from src.baselines.mfa_reference_adapter import ProjectPilotInterface,MFAReferenceAdapter
    from src.cdl_benchmark import observation_geometry
    interface=ProjectPilotInterface();g=observation_geometry()
    b.require(np.array_equal(interface.selected_full,g.selected) and np.array_equal(interface.data_full,np.flatnonzero(g.data)),'MFA_LR4_GEOMETRY_NOT_IDENTICAL')
    return model,interface,MFAReferenceAdapter(model,interface.A),{'identity':identity,'legal':legal,'all_components':smoke}
def seed_index():
    v=b.read(b.DATA/'outputs/sp_u12b2a_seed_registry.json')
    rows=[r for r in v['rows'] if r['group']=='EVAL'];ix={}
    for r in rows:
        key=(r['condition'],r['trajectory_id'],r['frame_id'],r['role'])
        b.require(key not in ix,'DUPLICATE_B2_SEED_IDENTITY');ix[key]=r['seed']
    b.require(len(ix)==165000,'B2_EVAL_SEED_CARDINALITY')
    return ix
def proof_check(proof,condition,i,cfg,index):
    p=b.contract()
    b.require(proof['condition']==condition and proof['trajectory_id']==i and proof['group']=='EVAL','B2_RAW_IDENTITY_DRIFT')
    b.require(proof['snr_db']==p['snrs'] and proof['estimators']==p['B2_methods'] and proof['scored_frames']==p['score_frames'],'B2_RAW_AXIS_DRIFT')
    b.require(proof['rank']==4 and proof['window_frames']==5 and proof['fold_hash']==cfg['fold_hash'] and proof['pilot_geometry_hash']==cfg['pilot_geometry_hash'] and proof['operator_sha256']==cfg['operator_cache_sha256'],'LR4_METHOD_FREEZE_DRIFT')
    b.require(all(proof[k] is True for k in ('method_pairing','CRN_pairing','dense_lr4_shared_coordinates','finite_estimates','valid_coordinate_solve')),'B2_PAIRING_OR_SOLVE_NOT_PASS')
    expected={r:index[condition,i,-1 if r in ROLES[:3] else 0,r] for r in ROLES}
    b.require(proof['channel_seeds']==expected,'CHANNEL_SEED_IDENTITY_DRIFT')
    expected_frames=[{r:index[condition,i,-1 if r in ROLES[:3] else j,r] for r in ROLES} for j in range(10)]
    b.require(proof['communication_seeds']==expected_frames,'COMM_SEED_IDENTITY_DRIFT')
    ins={(v['snr_db'],v['frame_id']):v for v in proof['inputs']}
    b.require(len(proof['inputs'])==len(ins)==70 and set(ins)=={(s,j) for s in p['snrs'] for j in range(10)},'EXACT_INPUT_KEY_GRID_REQUIRED')
    return ins
def checked_input(condition,i,cfg,index):
    # Called only inside the LAB's one-time frozen evaluation worker.
    prefix='outputs/sp_u12b2b_sealed/';cache=b.DATA/(prefix+'cache')/condition
    raw=b.DATA/(prefix+'raw_evaluation')/condition
    proof=b.read(raw/f'{i:04d}.json');ins=proof_check(proof,condition,i,cfg,index)
    hi=b.read(cache/f'H_{i:04d}.json');ci=b.read(cache/f'comm_{i:04d}.json')
    b.require((hi['condition'],hi['trajectory_id'],hi['group'])==(condition,i,'EVAL') and (ci['condition'],ci['trajectory_id'])==(condition,i),'CACHE_IDENTITY_DRIFT')
    for info in (hi,ci,proof):
        for rec in info['artifacts']:
            q=b.safe(b.DATA,rec['path']);b.require(b.sha(q)==rec['sha256'],'COMMITTED_INPUT_HASH_DRIFT')
    b.require(proof['H_artifact_sha256']==b.sha(cache/f'H_{i:04d}.npz') and proof['communication_sha256']==b.sha(cache/f'comm_{i:04d}.npz'),'RAW_CACHE_LINK_DRIFT')
    with np.load(cache/f'H_{i:04d}.npz',allow_pickle=False) as z:
        b.require(z.files==['H'],'H_ARCHIVE_SCHEMA');h=z['H']
    b.require(h.shape==(2,2,64,160) and h.dtype==np.complex128 and np.isfinite(h).all(),'H_INPUT_INVALID')
    # Exact existing load_native_h storage restoration, without importing old orchestrators.
    h=np.ascontiguousarray(h.transpose(1,0,3,2)).transpose(1,0,3,2)
    b.require(list(h.strides)==hi['export_strides_bytes'] and array_hash(h)==hi['H_sha256']==proof['H_sha256'],'H_ARRAY_OR_STORAGE_IDENTITY_DRIFT')
    with np.load(cache/f'comm_{i:04d}.npz',allow_pickle=False) as z:
        b.require(set(z.files)=={'bits','symbols','unit_noise'},'COMM_ARCHIVE_SCHEMA');comm={n:z[n] for n in z.files}
    b.require(comm['bits'].shape==(10,2,64,16,2) and comm['bits'].dtype==np.uint8,'BITS_SCHEMA')
    for n in ('symbols','unit_noise'):b.require(comm[n].shape==(10,2,64,16) and comm[n].dtype==np.complex128 and np.isfinite(comm[n]).all(),'COMM_ARRAY_INVALID')
    b.require(ci['shared_across_SNR'] is True and ci['frame_seeds']==proof['communication_seeds'] and hi['channel_seeds']==proof['channel_seeds'],'CACHE_SEED_OR_CRN_DRIFT')
    for j in range(10):
        for n in ('bits','symbols','unit_noise'):b.require(array_hash(comm[n][j])==ci[n+'_sha256'][j],'COMM_FRAME_HASH_DRIFT')
    with np.load(raw/f'{i:04d}.npz',allow_pickle=False) as z:
        f=z['frame_error'];t=z['frame_truth'];te=z['trajectory_error'];tt=z['trajectory_truth'];status=z['solve_status']
    b.require(f.shape==(7,7,6) and t.shape==(7,6) and f.dtype==t.dtype==np.float64,'LR4_RAW_SCHEMA')
    b.require(np.isfinite(f).all() and np.isfinite(t).all() and np.all(f>0) and np.all(t>0),'LR4_NONFINITE_OR_INVALID_ENERGY')
    b.require(np.array_equal(te,f.sum(axis=2)) and np.array_equal(tt,t.sum(axis=1)),'LR4_REDUCTION_DRIFT')
    b.require(status.shape==(7,7,6) and np.all(status[:,5,:]==1),'LR4_SOLVE_FAILED')
    b.require(all(np.array_equal(t[0],t[s]) for s in range(7)),'CROSS_SNR_TRUTH_DRIFT')
    return h,comm,ins,f[:,5,:],t,{'H_file_sha256':b.sha(cache/f'H_{i:04d}.npz'),'comm_file_sha256':b.sha(cache/f'comm_{i:04d}.npz'),'LR4_file_sha256':b.sha(raw/f'{i:04d}.npz'),'proof_sha256':b.sha(raw/f'{i:04d}.json')},te[:,5].copy()
def paired_reduction(frame_error):
    # Preserve B2's [SNR,method,6 contiguous frames] reduction, not a strided
    # middle-axis sum after interleaving methods. This matters at floating ULPs.
    b.require(frame_error.shape==(7,6,2),'PAIRED_FRAME_ENERGY_SHAPE')
    return np.ascontiguousarray(frame_error.transpose(0,2,1)).sum(axis=2)
def pilot_inputs(h,comm,ins):
    # Exact B2 power/noise/owner-pilot extraction, no RNG or channel generation.
    from src.cdl_benchmark import observation_geometry,frames
    from src.link import assemble_pilot_data_frame,apply_frequency_mimo_channel,estimate_ls_linear_channel
    p=b.contract();g=observation_geometry();truths=frames(h);pilots=np.empty((7,10,64),np.complex128)
    target=np.empty((6,3968),np.complex128);energy=np.empty(6,np.float64)
    for j in range(10):
        z=comm['unit_noise'][j];clean=apply_frequency_mimo_channel(truths[j],assemble_pilot_data_frame(comm['symbols'][j],g.layout))
        for si,s in enumerate(p['snrs']):
            sigma2=float(p['power']/10**(s/10));y=clean+np.sqrt(sigma2)*z
            sparse,_=estimate_ls_linear_channel(y,g.layout);pilot=sparse.ravel()[g.selected]
            actual={'snr_db':s,'frame_id':j,'truth_sha256':array_hash(truths[j]),'unit_noise_sha256':array_hash(z),'pilot_sha256':array_hash(pilot),'Y_sha256':array_hash(y)}
            b.require(actual==ins[s,j],'EXACT_PILOT_TRUTH_NOISE_PAIRING_FAILED:'+str((s,j)))
            pilots[si,j]=pilot
        if j>=4:target[j-4]=truths[j].ravel()[g.data];energy[j-4]=float(np.sum(abs(target[j-4])**2))
    pilots.flags.writeable=False;target.flags.writeable=False
    return pilots,target,energy
class Timing:
    """Observe the original posterior setup; do not replace its implementation."""
    def __init__(self):
        from src.baselines.mfa_reference_adapter import reference_modules
        self.filename=reference_modules()[1].__file__;self.calls=[];self.setup=0.;self.starts={};self.components=0
    def trace(self,frame,event,arg):
        if frame.f_code.co_filename==self.filename:
            name=frame.f_code.co_name
            if event=='call' and name=='estimate':self.calls.append([type(frame.f_locals['n_summands_or_proba']).__name__,frame.f_locals['n_summands_or_proba']])
            if name=='_prepare_for_prediction':
                if event=='call':self.starts[id(frame)]=time.perf_counter()
                if event=='return':self.setup+=time.perf_counter()-self.starts.pop(id(frame))
            if event=='call' and name=='_lmmse':self.components+=1
        return None
    def __enter__(self):self.previous=sys.getprofile();sys.setprofile(self.trace);return self
    def __exit__(self,*args):sys.setprofile(self.previous)
def infer(adapter,pilots,sigma2):
    # Signature intentionally excludes target truth, labels, seeds and future history.
    b.require(pilots.ndim==2 and pilots.shape[1]==adapter.A.shape[0] and np.isfinite(pilots).all(),'PILOT_INFERENCE_SHAPE_OR_FINITE')
    begin=time.perf_counter()
    with Timing() as observed:full=adapter.estimate(pilots,sigma2*np.eye(adapter.A.shape[0],dtype=np.complex128))
    wall=time.perf_counter()-begin
    b.require(observed.calls==[['float',1.0]] and observed.components==len(pilots)*adapter.model.n_components,'FLOAT1_ALL_COMPONENTS_INTERFACE_DRIFT')
    b.require(full.shape==(len(pilots),adapter.A.shape[1]) and np.isfinite(full).all(),'MFA_NONFINITE_OR_SHAPE')
    return full,{'adapter_wall_seconds':wall,'posterior_setup_seconds':observed.setup,'conditional_calls':observed.components,'rows':len(pilots),'observer_profile_included':True}
def evaluate(job):
    c,m,_,_=modules();p=b.contract();begin=time.monotonic();model,interface,adapter,model_gate=load_final()
    b.write(job/'model_gate.json',model_gate,True);initial=m.parameter_hashes(model)
    cfg=b.read(b.DATA/'configs/sp_u12b2b_sealed.json');registry=seed_index()
    predictions=0;receipts={};runtime=[]
    for condition in p['conditions']:
        for i in range(p['N']):
            h,comm,inputs,lr4,original_truth,links,old_lr4_trajectory=checked_input(condition,i,cfg,registry)
            pilots,target,truth=pilot_inputs(h,comm,inputs)
            b.require(np.array_equal(original_truth,np.broadcast_to(truth,(7,6))),'EXACT_SCORING_TRUTH_ENERGY_MISMATCH')
            errors=np.empty((7,6,2),np.float64);errors[:,:,0]=lr4
            for si,s in enumerate(p['snrs']):
                predicted,timing=infer(adapter,pilots[si,4:10],float(p['power']/10**(s/10)))
                errors[si,:,1]=np.sum(abs(predicted[:,interface.data_positions]-target)**2,axis=1,dtype=np.float64)
                b.require(np.isfinite(errors[si]).all() and np.all(errors[si]>0),'SCORING_ENERGY_INVALID')
                runtime.append({'condition':condition,'trajectory_id':i,'snr_db':s,**timing});predictions+=6
            path=job/'raw'/condition/f'{i:04d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            trajectory=paired_reduction(errors)
            b.require(np.array_equal(trajectory[:,0],old_lr4_trajectory),'EXACT_ORIGINAL_LR4_TRAJECTORY_SUM_DRIFT')
            trajectory[:,0]=old_lr4_trajectory # Store the original sufficient statistic verbatim.
            with path.open('xb') as f:np.savez(f,frame_error=errors,frame_truth=original_truth,trajectory_error=trajectory,trajectory_truth=original_truth.sum(axis=1))
            receipt={'condition':condition,'trajectory_id':i,'snrs':p['snrs'],'frames':p['score_frames'],'methods':['LR4','MFA'],
                'original_inputs':links,'scored_pilot_and_truth_identities':[inputs[s,j] for s in p['snrs'] for j in p['score_frames']],
                'model_manifest_sha256':p['final_model_manifest_sha256'],'raw_sha256':b.sha(path),'pairing_verified':True}
            b.write(path.with_suffix('.json'),receipt,True);receipts[path.relative_to(job).as_posix()]=b.sha(path)
            b.write(job/'phase.json',{'phase':'FORMAL_EVALUATION','condition':condition,'trajectory_id':i,'MFA_predictions_complete':predictions,'expected':210000,'elapsed_seconds':time.monotonic()-begin})
            with (job/'inference_runtime.jsonl').open('a',encoding='utf8') as f:
                import json
                for row in runtime[-7:]:f.write(json.dumps(row,allow_nan=False)+'\n')
    b.require(predictions==210000 and len(receipts)==5000,'FULL_GRID_CARDINALITY_REQUIRED')
    b.require(initial==m.parameter_hashes(model),'MODEL_PARAMETER_MUTATION_AFTER_INFERENCE')
    final_identity();b.verify();b.verify_data()
    result={'PASS':True,'paired_frame_predictions_per_method':predictions,'trajectory_cells':35000,'raw_trajectories':5000,
       'raw_hashes':receipts,'all_pairs_exact':True,'model_unchanged':True,'data_unchanged':True,'runtime_calls':35000,
       'inference_adapter_seconds':sum(v['adapter_wall_seconds'] for v in runtime),'posterior_setup_seconds':sum(v['posterior_setup_seconds'] for v in runtime),
       'evaluation_wall_seconds':time.monotonic()-begin,'formal_training_executed':False,'validation_recomputed':False}
    b.write(job/'integrity_gate.json',result,True);return result
