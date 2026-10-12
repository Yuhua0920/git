"""Frozen author adapter plus parameter-level serialization and validation."""
import os,sys,json,hashlib
for _n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_n]='1'
import numpy as np
import scipy
from scripts.sp_u12b5b_common import ROOT,read,write,sha,tree
sys.path.append(str(ROOT/'.venv-gmm/Lib/site-packages'))
from threadpoolctl import threadpool_limits,threadpool_info
from src.baselines.mfa_reference_adapter import make_reference_model,fit_reference,MFAReferenceAdapter,ProjectPilotInterface,FROZEN_POLICY
PUBLIC=('means','lambdas','psis','amps','covs','inv_covs')
INITIAL={'means':'_means','lambdas':'_lambdas','psis':'_psis','amps':'amps','covs':'_covs','inv_covs':'_inv_covs'}
def array_hash(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(memoryview(a).cast('B'));return h.hexdigest()
def parameter_hashes(model,initial=False):
 return {name:dict(shape=list(a.shape),dtype=a.dtype.str,sha256=array_hash(a)) for name,attr in (INITIAL.items() if initial else [(k,k) for k in PUBLIC]) for a in [getattr(model,attr)]}
def parameter_validity(model):
 arrays={n:getattr(model,n) for n in PUBLIC}
 finite={n:bool(np.isfinite(a).all()) for n,a in arrays.items()}
 pi,psi,W,C=model.amps,model.psis,model.lambdas,model.covs
 weight_valid=bool(np.all(pi>=0) and np.all(pi<=1) and abs(float(pi.sum())-1)<=1e-10)
 residual_valid=bool(np.all(psi>=1e-6) and np.all(psi==psi[:,:1]))
 worst=0.0
 for k in range(model.n_components):
  for start in range(0,model.D,64):
   end=min(start+64,model.D);expected=W[k,start:end]@W[k].conj().T
   expected[np.arange(end-start),np.arange(start,end)]+=psi[k,start:end]
   worst=max(worst,float(np.max(abs(expected-C[k,start:end]),initial=0)))
 covariance_valid=worst<=1e-10+1e-9*float(np.max(abs(C),initial=0))
 return dict(PASS=all(finite.values()) and weight_valid and residual_valid and covariance_valid,finite=finite,weights_valid=weight_valid,weights_sum=float(pi.sum()),isotropic_residual_valid=residual_valid,min_residual=float(psi.min()),covariance_loading_residual_max_absolute=worst,positive_definiteness_certificate='C=W W^H+psi I, psi>=1e-6',covariance_valid=bool(covariance_valid))
def save_model(model,directory,seed,training_hash,initial_hashes):
 directory.mkdir(parents=True,exist_ok=True);hashes=parameter_hashes(model)
 for name in PUBLIC:
  with (directory/f'{name}.npy').open('xb') as f:np.save(f,getattr(model,name),allow_pickle=False)
 history=np.asarray(model.L_all,dtype=np.float64)
 with (directory/'training_history.npy').open('xb') as f:np.save(f,history,allow_pickle=False)
 write(directory/'parameter_hashes.json',dict(arrays=hashes,combined_parameter_hash=tree(hashes)))
 write(directory/'model_parameters.json',dict(format='separate NPY arrays indexed here; no pickle',arrays={n:dict(path=f'{n}.npy',**hashes[n]) for n in PUBLIC},combined_parameter_hash=tree(hashes)))
 artifact=dict(K=model.n_components,loading_rank=model.M,dimension=model.D,training_samples=model.N,seed=int(seed),training_data_hash=training_hash,initialization_hash=tree(initial_hashes),initial_parameter_hashes=initial_hashes,parameter_hash=tree(hashes),policy=FROZEN_POLICY,parameter_files={n:dict(path=f'{n}.npy',file_sha256=sha(directory/f'{n}.npy')) for n in PUBLIC},history_file_sha256=sha(directory/'training_history.npy'),serialization='NPY core arrays + JSON metadata; no pickle',reference='2023 pinned B5A author-code adapter')
 write(directory/'model_artifact.json',artifact);return artifact
def load_model(directory):
 artifact=read(directory/'model_artifact.json');hashes=read(directory/'parameter_hashes.json')['arrays']
 model=make_reference_model(artifact['K'],artifact['loading_rank']);model.D=artifact['dimension'];model.N=artifact['training_samples']
 for name in PUBLIC:
  path=directory/f'{name}.npy';assert sha(path)==artifact['parameter_files'][name]['file_sha256'],'SERIALIZATION_FILE_CORRUPTION'
  a=np.load(path,allow_pickle=False);assert dict(shape=list(a.shape),dtype=a.dtype.str,sha256=array_hash(a))==hashes[name],'SERIALIZATION_PARAMETER_CORRUPTION'
  setattr(model,name,a)
 assert sha(directory/'training_history.npy')==artifact['history_file_sha256'],'SERIALIZATION_HISTORY_CORRUPTION'
 model.L_all=np.load(directory/'training_history.npy',allow_pickle=False).tolist()
 assert tree(parameter_hashes(model))==artifact['parameter_hash']
 return model,artifact
def pooled_ratio(error_energy,truth_energy):
 e=np.asarray(error_energy,dtype=np.float64);t=np.asarray(truth_energy,dtype=np.float64)
 if e.shape!=t.shape or np.any(e<0) or np.any(t<=0) or not np.all(np.isfinite(e)) or not np.all(np.isfinite(t)):raise ValueError('invalid pooled-energy inputs')
 return float(e.sum(dtype=np.float64)/t.sum(dtype=np.float64))
def macro_score(cells):
 if len(cells)!=6 or not all(np.isfinite(x) and x>=0 for x in cells):raise ValueError('six valid linear cells required')
 return float(np.asarray(cells,dtype=np.float64).sum(dtype=np.float64)/6)
def automatic_select(rows):
 if len(rows)!=9 or {(int(r['K']),int(r['rank'])) for r in rows}!={(K,L) for K in (1,2,4) for L in (4,8,16)}:raise ValueError('complete nine-candidate grid required')
 if not all(np.isfinite(r['macro_selection_score']) and r['macro_selection_score']>=0 for r in rows):raise ValueError('nine finite nonnegative selection scores required')
 return min(rows,key=lambda r:(float(r['macro_selection_score']),int(r['K']),int(r['rank'])))
