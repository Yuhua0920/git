"""Full-size synthetic inference/IO rehearsal. No training or formal inputs.

Uses the original worker, original adapter/author inference, exact six-field
input checks, original LR4 kernels and original pooled-energy reduction.
The untrained synthetic model and all observations are generated in this job.
"""
from pathlib import Path
import gc,json,types
import numpy as np
import base as b
import science as s

def observers(m):
    # Relocate only R6 bookkeeping IO globals to the synthetic namespace.
    # Every original function retains its identical bytecode/math functions.
    g=dict(vars(m));g.update(OUT=b.OUT,sha=b.sha,read=b.read,write=b.write)
    for name,fn in vars(m).items():
        if isinstance(fn,types.FunctionType) and fn.__globals__ is vars(m):
            g[name]=types.FunctionType(fn.__code__,g,fn.__name__,fn.__defaults__,fn.__closure__)
    return types.SimpleNamespace(**g)

def stage(job,name):b.write(job/'phase.json',{'phase':'SYNTHETIC_'+name,'formal_predictions':0})

def run(job):
    b.require(job.is_relative_to(b.OUT/'synthetic'),'REHEARSAL_SYNTHETIC_ONLY')
    c,m,_,_=s.modules();b.verify()
    from control import scope
    scope(job,False) # Denies every formal B2/B3 array and final model NPY.
    steps=['ACTUAL_WORKER_INITIALIZED'];stage(job,'PREPARING_FIXTURE')
    from src.baselines.mfa_reference_adapter import make_reference_model,MFAReferenceAdapter,ProjectPilotInterface
    from src.cdl_benchmark import observation_geometry,frames
    from src.link import assemble_pilot_data_frame,apply_frequency_mimo_channel,estimate_ls_linear_channel
    from src.pilot_cv import frozen_folds,PilotCV
    from src.lowrank_modules import RankFactors,factored_affine_predict
    from paired_stats import pooled
    rng=np.random.default_rng(807211)
    def cn(shape,scale=1.):return np.asarray(scale*(rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2),np.complex128)
    interface=ProjectPilotInterface();g=observation_geometry();folds,definition=frozen_folds(g.selected)
    b.require(np.array_equal(interface.selected_full,g.selected) and np.array_equal(interface.data_full,np.flatnonzero(g.data)),'SYNTHETIC_GEOMETRY_MISMATCH')
    dims=4032;model=make_reference_model(1,16);model.D=dims;model.N=10
    model.means=np.zeros((1,dims),np.complex128);model.lambdas=np.zeros((1,dims,16),np.complex128)
    model.psis=np.ones((1,dims),np.float64);model.amps=np.ones(1,np.float64)
    model.covs=np.eye(dims,dtype=np.complex128)[None];model.inv_covs=model.covs.copy()
    # Synthetic valid serialization history only: no fit or EM update is called.
    model.L_all=[-10.,-8.,-7.,-6.,-5.001,-5.,-5.]
    mo=observers(m);mo.parameter_validity(model,full=True)
    manifest=mo.save_model(model,job/'fixture_model',{'role':'synthetic_untrained_rehearsal','no_training_executed':True},mo.parameter_hashes(model))
    del model;gc.collect()
    model,reloaded=mo.load_model(job/'fixture_model');mo.parameter_validity(model,full=True)
    b.require(manifest['parameter_hash']==reloaded['parameter_hash'],'SYNTHETIC_MODEL_ROUNDTRIP_FAILED')
    initial=mo.parameter_hashes(model);adapter=MFAReferenceAdapter(model,interface.A)
    steps+=['MODEL_FILES_HASH_CHECKED','MODEL_VALIDITY_CHECKED','MODEL_SERIALIZATION_ROUNDTRIP']
    stage(job,'INPUT_BUILD')
    p=b.contract();condition=next(iter(p['conditions']));i=0
    h=cn((2,2,64,160));h=np.ascontiguousarray(h.transpose(1,0,3,2)).transpose(1,0,3,2)
    bits=rng.integers(0,2,size=(10,2,64,16,2),dtype=np.uint8)
    symbols=((1-2*bits[...,0].astype(float))+1j*(1-2*bits[...,1].astype(float)))/np.sqrt(2)
    comm={'bits':bits,'symbols':symbols.astype(np.complex128),'unit_noise':cn((10,2,64,16))}
    index={(condition,0,-1,r):1000+k for k,r in enumerate(s.ROLES[:3])}
    index.update({(condition,0,j,r):2000+j*3+k for j in range(10) for k,r in enumerate(s.ROLES[3:])})
    seeds={r:index[condition,0,-1 if r in s.ROLES[:3] else 0,r] for r in s.ROLES}
    fseeds=[{r:index[condition,0,-1 if r in s.ROLES[:3] else j,r] for r in s.ROLES} for j in range(10)]
    truths=frames(h);pilots=np.empty((7,10,64),np.complex128);inputs=[]
    for j in range(10):
        z=comm['unit_noise'][j];clean=apply_frequency_mimo_channel(truths[j],assemble_pilot_data_frame(comm['symbols'][j],g.layout))
        for si,snr in enumerate(p['snrs']):
            y=clean+np.sqrt(float(p['power']/10**(snr/10)))*z
            sparse,_=estimate_ls_linear_channel(y,g.layout);pilot=sparse.ravel()[g.selected];pilots[si,j]=pilot
            inputs.append({'snr_db':snr,'frame_id':j,'truth_sha256':s.array_hash(truths[j]),'unit_noise_sha256':s.array_hash(z),'pilot_sha256':s.array_hash(pilot),'Y_sha256':s.array_hash(y)})
    mp=np.zeros(64,np.complex128);md=np.zeros(3968,np.complex128);ws=cn((3968,64),.01)
    profile=RankFactors(cn((3968,4),.02),cn((4,64),.04));mobility=RankFactors(cn((3968,4),.02),cn((4,64),.04))
    cv=PilotCV(folds,mp,tuple(cn((16,48),.03) for _ in folds),tuple(cn((16,48),.03) for _ in folds),tuple(cn((16,48),.03) for _ in folds))
    truth=np.array([np.sum(abs(truths[j].ravel()[g.data])**2,dtype=np.float64) for j in p['score_frames']])
    lr4=np.empty((7,6));coef=np.empty((7,6,2));max_residual=0.
    for si in range(7):
        for jj,j in enumerate(p['score_frames']):
            coef[si,jj],diagnostic=cv.fit(pilots[si,j-4:j+1],modules=('P','M'))
            prediction=factored_affine_predict(pilots[si,j],mp,md,ws,coef[si,jj],profile,mobility)
            lr4[si,jj]=np.sum(abs(prediction-truths[j].ravel()[g.data])**2,dtype=np.float64)
            max_residual=max(max_residual,diagnostic['normal_equation_relative_residual'])
    steps+=['ORIGINAL_LR4_FIVE_FRAME_FOUR_FOLD_SOLVE','ORIGINAL_RANK4_FACTORED_PREDICTION']
    data=job/'fixture_inputs';prefix='outputs/sp_u12b2b_sealed/'
    cache=data/(prefix+'cache')/condition;raw=data/(prefix+'raw_evaluation')/condition
    cache.mkdir(parents=True);raw.mkdir(parents=True)
    hp=cache/'H_0000.npz';cp=cache/'comm_0000.npz';rp=raw/'0000.npz'
    with hp.open('xb') as f:np.savez(f,H=h)
    with cp.open('xb') as f:np.savez(f,**comm)
    # Seven-method shape is a synthetic schema fixture; only LR4 is actually
    # computed here. It is never represented as a scientific seven-method run.
    fe=np.repeat(lr4[:,None,:],7,axis=1);ft=np.broadcast_to(truth,(7,6)).copy()
    with rp.open('xb') as f:np.savez(f,frame_error=fe,frame_truth=ft,trajectory_error=fe.sum(axis=2),trajectory_truth=ft.sum(axis=1),solve_status=np.ones((7,7,6),np.int8))
    def artifact(path):return {'path':path.relative_to(data).as_posix(),'sha256':b.sha(path)}
    hi={'condition':condition,'trajectory_id':0,'group':'EVAL','artifacts':[artifact(hp)],'H_sha256':s.array_hash(h),'export_strides_bytes':list(h.strides),'channel_seeds':seeds}
    ci={'condition':condition,'trajectory_id':0,'artifacts':[artifact(cp)],'shared_across_SNR':True,'frame_seeds':fseeds,
        **{n+'_sha256':[s.array_hash(comm[n][j]) for j in range(10)] for n in comm}}
    cfg={'fold_hash':definition['sha256'],'pilot_geometry_hash':'SYNTHETIC_GEOMETRY','operator_cache_sha256':'SYNTHETIC_OPERATOR_FIXTURE'}
    proof={'condition':condition,'trajectory_id':0,'group':'EVAL','snr_db':p['snrs'],'estimators':p['B2_methods'],'scored_frames':p['score_frames'],
     'rank':4,'window_frames':5,'fold_hash':cfg['fold_hash'],'pilot_geometry_hash':cfg['pilot_geometry_hash'],'operator_sha256':cfg['operator_cache_sha256'],
     **{n:True for n in ('method_pairing','CRN_pairing','dense_lr4_shared_coordinates','finite_estimates','valid_coordinate_solve')},
     'channel_seeds':seeds,'communication_seeds':fseeds,'inputs':inputs,'artifacts':[artifact(hp),artifact(cp),artifact(rp)],
     'H_artifact_sha256':b.sha(hp),'communication_sha256':b.sha(cp),'H_sha256':s.array_hash(h)}
    b.write(hp.with_suffix('.json'),hi,True);b.write(cp.with_suffix('.json'),ci,True);b.write(raw/'0000.json',proof,True)
    previous=b.DATA;b.DATA=data
    try:
        h2,comm2,ins,old_lr4,old_truth,links,old_sum=s.checked_input(condition,0,cfg,index)
        paired_pilots,target,energies=s.pilot_inputs(h2,comm2,ins)
    finally:b.DATA=previous
    b.require(np.array_equal(pilots,paired_pilots) and np.array_equal(old_truth,np.broadcast_to(energies,(7,6))),'SYNTHETIC_PAIRING_DRIFT')
    steps+=['INPUT_ARTIFACT_HASH_CHECKED','EXACT_SIX_FIELD_SEED_IDENTITY','H_NOISE_PILOT_TRUTH_70_IDENTITIES']
    stage(job,'AUTHOR_MFA_INFERENCE');errors=np.empty((7,6,2),np.float64);errors[:,:,0]=old_lr4;timings=[]
    with c.threads(6,1):
        for si,snr in enumerate(p['snrs']):
            stage(job,'AUTHOR_MFA_INFERENCE_'+str(snr))
            full,timing=s.infer(adapter,paired_pilots[si,4:10],float(p['power']/10**(snr/10)))
            errors[si,:,1]=np.sum(abs(full[:,interface.data_positions]-target)**2,axis=1,dtype=np.float64);timings.append(timing)
    b.require(np.isfinite(errors).all() and np.all(errors>0),'SYNTHETIC_NONFINITE_ENERGY')
    reduction=s.paired_reduction(errors);b.require(np.array_equal(reduction[:,0],old_sum),'SYNTHETIC_LR4_REDUCTION_DRIFT')
    nmse=pooled(reduction,np.repeat(energies.sum()[None,None],14).reshape(7,2))
    with (job/'synthetic_paired_raw.npz').open('xb') as f:
        np.savez(f,frame_error=errors,frame_truth=old_truth,trajectory_error=reduction,trajectory_truth=old_truth.sum(axis=1),coordinates=coef)
    with np.load(job/'synthetic_paired_raw.npz',allow_pickle=False) as saved:
        b.require(np.array_equal(saved['frame_error'],errors) and np.array_equal(saved['trajectory_error'],reduction),'SYNTHETIC_RESULT_ROUNDTRIP')
    b.require(initial==mo.parameter_hashes(model),'SYNTHETIC_MODEL_CHANGED')
    steps+=['ORIGINAL_AUTHOR_FLOAT1_ALL_COMPONENTS_INFERENCE','ORIGINAL_PAIRED_ENERGY_REDUCTION','ORIGINAL_LINEAR_POOLED_NMSE','RESULT_SAVED_AND_RELOADED','MODEL_UNCHANGED']
    result={'PASS':True,'synthetic_only':True,'untrained_synthetic_model':True,'MFA_dimensions':4032,'MFA_K':1,'MFA_rank':16,
      'synthetic_trajectories':1,'synthetic_scored_positions':42,'pilot_identity_positions':70,'steps':steps,'timings':timings,
      'normal_equation_max_relative_residual':max_residual,'synthetic_pooled_NMSE':np.asarray(nmse).tolist(),'input_sha256':links,
      'result_sha256':b.sha(job/'synthetic_paired_raw.npz'),'formal_predictions':0,'formal_training_executed':False,
      'formal_inputs_or_final_NPY_accessed':False,'bootstrap_executed':False,'scientific_claims':None}
    b.write(job/'rehearsal_result.json',result,True);return result
