"""Deployment observers/serialization around the unchanged original adapter."""
from pathlib import Path
import sys, hashlib, json, csv, importlib.util
import numpy as np
from common import ROOT, OUT, require, read, write, sha, object_sha, science, registry, candidate
from src.baselines.mfa_reference_adapter import make_reference_model, fit_reference, MFAReferenceAdapter, ProjectPilotInterface, FROZEN_POLICY, reference_modules

PUBLIC=('means','lambdas','psis','amps','covs','inv_covs')
INITIAL={'means':'_means','lambdas':'_lambdas','psis':'_psis','amps':'amps','covs':'_covs','inv_covs':'_inv_covs'}

def array_hash(a):
    a=np.ascontiguousarray(a); h=hashlib.sha256(); h.update(memoryview(a).cast('B')); return h.hexdigest()

def parameter_hashes(model,initial=False):
    mapping=INITIAL if initial else {n:n for n in PUBLIC}
    return {n:{'shape':list(a.shape),'dtype':a.dtype.str,'sha256':array_hash(a)} for n,attr in mapping.items() for a in [getattr(model,attr)]}

def parameter_validity(model,initial=False,full=True):
    mapping=INITIAL if initial else {n:n for n in PUBLIC}
    a={n:getattr(model,attr) for n,attr in mapping.items()}
    finite={n:bool(np.isfinite(v).all()) for n,v in a.items()}
    K,D,r=model.n_components,model.D,model.M
    shapes={'means':(K,D),'lambdas':(K,D,r),'psis':(K,D),'amps':(K,), 'covs':(K,D,D),'inv_covs':(K,D,D)}
    require(all(v.shape==shapes[n] for n,v in a.items()),'PARAMETER_SHAPE_INVALID')
    pi,psi,W,C=a['amps'],a['psis'],a['lambdas'],a['covs']
    weights=bool(np.all(pi>=0) and np.all(pi<=1) and abs(float(pi.sum())-1)<=1e-10)
    residual=bool(np.all(psi>=1e-6) and np.all(psi==psi[:,:1]))
    worst=0.
    if full:
        for k in range(K):
            for start in range(0,D,64):
                end=min(start+64,D); expected=W[k,start:end]@W[k].conj().T
                expected[np.arange(end-start),np.arange(start,end)]+=psi[k,start:end]
                worst=max(worst,float(np.max(abs(expected-C[k,start:end]),initial=0)))
    covariance=not full or worst<=1e-10+1e-9*float(np.max(abs(C),initial=0))
    positive=True
    if full and all(finite.values()) and covariance:
        from scipy.linalg import cholesky
        for k in range(K):
            try:cholesky(C[k],lower=True,overwrite_a=False,check_finite=False)
            except np.linalg.LinAlgError:positive=False;break
    result={'PASS':all(finite.values()) and weights and residual and covariance and positive,'finite':finite,
            'weights_valid':weights,'isotropic_residual_valid':residual,'min_residual':float(psi.min()),
            'covariance_valid':bool(covariance),'full_covariance_check':full,'covariance_loading_residual_max_absolute':worst,
            'positive_definite':positive,'positive_definiteness_certificate':'C=W W^H+psi I, psi>=1e-6; full saved-matrix Cholesky also checked when full=True'}
    require(result['PASS'],'ILLEGAL_MODEL_PARAMETERS:'+str(result)); return result

class Observe:
    """R5 call tracing mechanism: observes, never replaces, author methods."""
    def __init__(self,job):
        core,_,_=reference_modules(); self.filename=core.__file__; self.job=job
        self.previous=None; self.steps=0; self.initial=None; self.kmeans=[]
    def tracer(self,frame,event,arg):
        if frame.f_code.co_filename!=self.filename: return None
        name=frame.f_code.co_name
        if event=='call' and name in ('_initialize','_EM_per_component'):
            write(self.job/'phase.json',{'phase':name,'completed_iterations':self.steps})
        if event=='return' and arg is None and name=='_initialize':
            m=frame.f_locals['self']; self.initial=parameter_hashes(m,True)
            require(all(np.isfinite(getattr(m,n)).all() for n in INITIAL.values()),'INITIALIZATION_NONFINITE')
            write(self.job/'initialization.json',{'arrays':self.initial,'combined_parameter_hash':object_sha(self.initial)})
        if event=='return' and arg is None and name=='_EM_per_component':
            m=frame.f_locals['self']; value=float(m.logLs.sum()); require(np.isfinite(value),'EM_OBJECTIVE_NONFINITE')
            parameter_validity(m,initial=True,full=False); self.steps+=1
            with (self.job/'iteration_history.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps({'iteration':self.steps,'author_log_likelihood':value},allow_nan=False)+'\n')
            write(self.job/'phase.json',{'phase':'EM_update_complete','completed_iterations':self.steps,'objective':value})
        return self.tracer
    def __enter__(self): self.previous=sys.gettrace(); sys.settrace(self.tracer); return self
    def __exit__(self,*args): sys.settrace(self.previous)

def convergence(history):
    h=np.asarray(history,dtype=np.float64)
    require(h.ndim==1 and 7<=len(h)<=100 and np.isfinite(h).all(),'TRAINING_HISTORY_INVALID')
    if len(h)==100: return 'MAX_ITER_REACHED'
    require(abs((h[-1]-h[-2])/h[-1])<1e-4,'AUTHOR_STOPPING_CRITERION_INCONSISTENT')
    return 'CONVERGED'

def save_model(model,directory,context,initial):
    directory.mkdir(parents=True,exist_ok=False); hashes=parameter_hashes(model)
    for name in PUBLIC:
        with (directory/f'{name}.npy').open('xb') as f: np.save(f,getattr(model,name),allow_pickle=False)
    with (directory/'training_history.npy').open('xb') as f: np.save(f,np.asarray(model.L_all,dtype=np.float64),allow_pickle=False)
    artifact={'format':'R6_ORIGINAL_MFA_NPY_V1','K':model.n_components,'rank':model.M,'D':model.D,'N':model.N,
              'policy':FROZEN_POLICY,'context':context,'initialization':initial,'initialization_hash':object_sha(initial),
              'arrays':hashes,'parameter_hash':object_sha(hashes),'convergence':convergence(model.L_all),
              'iterations':len(model.L_all),'file_hashes':{p.name:sha(p) for p in directory.glob('*.npy')}}
    write(directory/'model_manifest.json',artifact,exclusive=True); return artifact

def check_model_files(directory):
    v=read(directory/'model_manifest.json')
    require(v['format']=='R6_ORIGINAL_MFA_NPY_V1' and v['policy']==FROZEN_POLICY,'MODEL_POLICY_DRIFT')
    require(set(v['file_hashes'])=={n+'.npy' for n in PUBLIC}|{'training_history.npy'},'MODEL_FILE_INVENTORY_INVALID')
    for name,h in v['file_hashes'].items():
        require(Path(name).name==name,'MODEL_PATH_TRAVERSAL'); require(sha(directory/name)==h,'SERIALIZATION_FILE_CORRUPTION')
    require(object_sha(v['arrays'])==v['parameter_hash'] and object_sha(v['initialization'])==v['initialization_hash'],'PARAMETER_MANIFEST_CORRUPTION')
    return v

def load_model(directory):
    artifact=check_model_files(directory)
    m=make_reference_model(artifact['K'],artifact['rank']); m.D=artifact['D']; m.N=artifact['N']
    for n in PUBLIC:
        a=np.load(directory/(n+'.npy'),mmap_mode='r',allow_pickle=False)
        require({'shape':list(a.shape),'dtype':a.dtype.str,'sha256':array_hash(a)}==artifact['arrays'][n],'SERIALIZATION_PARAMETER_CORRUPTION')
        setattr(m,n,a)
    m.L_all=np.load(directory/'training_history.npy',allow_pickle=False).tolist()
    require(convergence(m.L_all)==artifact['convergence'] and len(m.L_all)==artifact['iterations'],'HISTORY_MANIFEST_DRIFT')
    return m,artifact

def inference_smoke(m):
    # Scientific adapter call; trace float 1.0 and every conditional estimator
    # without monkeypatching scientific functions.
    D=m.D; no=min(64,D); A=ProjectPilotInterface().A if D==4032 else np.eye(D,dtype=np.complex128)[:no]
    y=np.zeros((1,no),dtype=np.complex128); Cn=.9941134822523973*np.eye(no,dtype=np.complex128)
    calls=[]; terms=[]; previous=sys.gettrace(); _,est,_=reference_modules(); filename=est.__file__
    def tracer(frame,event,arg):
        if event=='call' and frame.f_code.co_filename==filename:
            if frame.f_code.co_name=='estimate':
                v=frame.f_locals['n_summands_or_proba']; calls.append([type(v).__name__,v])
            if frame.f_code.co_name=='_lmmse': terms.append(1)
        return tracer
    try:
        sys.settrace(tracer); result=MFAReferenceAdapter(m,A).estimate(y,Cn)
    finally: sys.settrace(previous)
    require(np.isfinite(result).all() and result.shape==(1,D),'INFERENCE_SMOKE_FAILED')
    require(calls==[['float',1.0]] and len(terms)==m.n_components,'ALL_COMPONENTS_FLOAT1_REGRESSION')
    return {'PASS':True,'API_calls':calls,'component_calls':len(terms),'synthetic_zero_observation_only':True}

def pooled_ratio(error_energy,truth_energy):
    e=np.asarray(error_energy,dtype=np.float64); t=np.asarray(truth_energy,dtype=np.float64)
    require(e.shape==t.shape and not np.any(e<0) and not np.any(t<=0) and np.isfinite(e).all() and np.isfinite(t).all(),'INVALID_POOLED_ENERGY_INPUTS')
    return float(e.sum(dtype=np.float64)/t.sum(dtype=np.float64))

def macro_score(cells):
    require(len(cells)==6 and all(np.isfinite(x) and x>=0 for x in cells),'SIX_VALID_LINEAR_CELLS_REQUIRED')
    return float(np.asarray(cells,dtype=np.float64).sum(dtype=np.float64)/6)

def automatic_select(rows):
    require(len(rows)==9 and {(v['K'],v['rank']) for v in rows}=={(K,r) for K in (1,2,4) for r in (4,8,16)},'COMPLETE_NINE_CANDIDATE_GRID_REQUIRED')
    require(all(np.isfinite(v['macro_selection_score']) and v['macro_selection_score']>=0 for v in rows),'INVALID_SELECTION_SCORES')
    return min(rows,key=lambda v:(float(v['macro_selection_score']),int(v['K']),int(v['rank'])))

def canonical():
    path=ROOT/'experiments/mfa_low_memory_fidelity_r4/canonical_validation_contract.py'
    spec=importlib.util.spec_from_file_location('_r6_canonical_metadata_only',path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def tokens_from_rows(rows):
    for v in rows: yield ((int(v['K']),int(v['rank'])),v['origin'],int(v['snr_db']),int(v['trajectory_id']),int(v['frame_id']))

def candidate_cardinality(rows,K,r,cfg=None):
    expected={t for t in canonical().metadata_contract(cfg or science()) if t[0]==(K,r)}
    seen=set(); count=0
    for t in tokens_from_rows(rows):
        require(t in expected and t not in seen,'UNEXPECTED_OR_DUPLICATE_VALIDATION_IDENTITY'); seen.add(t); count+=1
    require(count==3600 and seen==expected,'INCOMPLETE_CANDIDATE_VALIDATION'); return count

def total_cardinality(rows,cfg=None):
    # This is the unmodified R4 assertion, now called by R6 selection.
    return canonical().assert_exact_identity_coverage(tokens_from_rows(rows),cfg or science())
