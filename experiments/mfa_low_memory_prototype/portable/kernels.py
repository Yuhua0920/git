"""Identical synthetic fixtures, original reference math, experimental alternatives."""
from pathlib import Path
import sys,importlib.util,hashlib,json,copy,types
import numpy as np
from low_memory import loading_numerator,latent_moment,real_quadratic,residual_moment,em_step,log_gaussian_no_det
PORT=Path(__file__).resolve().parent
_adapter=None
def adapter():
 global _adapter
 if _adapter is None:
  expected=json.loads((PORT/'portable_reference_hashes.json').read_text(encoding='utf8'))
  for path,digest in expected['files'].items():assert hashlib.sha256((PORT/path).read_bytes()).hexdigest()==digest,'PORTABLE_REFERENCE_DRIFT'
  spec=importlib.util.spec_from_file_location('_r3_byte_identical_adapter',PORT/'src/baselines/mfa_reference_adapter.py')
  _adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(_adapter)
 return _adapter
def cnormal(rng,shape):return np.asarray((rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2),dtype=np.complex128)
def hash_array(a):return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def specifications():
 return [
 {'kind':'slogdet','D':4032,'N':0,'K':1,'rank':16},
 {'kind':'likelihood_gemm','D':4032,'N':512,'K':1,'rank':16},
 {'kind':'loading_direct','D':512,'N':1024,'K':1,'rank':16},
 {'kind':'loading_low','D':512,'N':1024,'K':1,'rank':16},
 {'kind':'covariance_init','D':4032,'N':0,'K':1,'rank':16},
 {'kind':'woodbury','D':4032,'N':0,'K':1,'rank':16},
 {'kind':'beta','D':4032,'N':0,'K':1,'rank':16},
 {'kind':'latent_direct','D':0,'N':2048,'K':1,'rank':16},
 {'kind':'latent_low','D':0,'N':2048,'K':1,'rank':16},
 {'kind':'quadratic_direct','D':4032,'N':512,'K':1,'rank':0},
 {'kind':'quadratic_low','D':4032,'N':512,'K':1,'rank':0},
 {'kind':'residual_direct','D':4032,'N':512,'K':1,'rank':0},
 {'kind':'residual_low','D':4032,'N':512,'K':1,'rank':0},
 {'kind':'initialization','D':256,'N':2048,'K':4,'rank':16},
 {'kind':'em_original','D':64,'N':256,'K':2,'rank':8},
 {'kind':'em_low','D':64,'N':256,'K':2,'rank':8},
 {'kind':'inference_setup_K1','D':4032,'N':6,'K':1,'rank':16},
 {'kind':'inference_batch_K1','D':4032,'N':6,'K':1,'rank':16},
 {'kind':'inference_setup_K4','D':4032,'N':6,'K':4,'rank':16},
 {'kind':'inference_batch_K4','D':4032,'N':6,'K':4,'rank':16}]
def initialize(data,K,r,seed):
 m=adapter().make_reference_model(K,r);m.N,m.D=data.shape;m.rs=np.zeros((K,m.N));m._covs=np.zeros((K,m.D,m.D),complex);m._inv_covs=np.zeros_like(m._covs)
 with adapter().recorded_initialization(seed):m._initialize(data)
 return m
def attach_low(model):
 model._log_multi_gauss_nodet=types.MethodType(log_gaussian_no_det,model)
 model._EM_per_component=types.MethodType(em_step,model)
 return model
def state(model):return {name:np.asarray(getattr(model,name)).copy() for name in ('_means','_lambdas','_psis','_covs','_inv_covs','amps')}
def restored(model,initial):
 for name,value in initial.items():setattr(model,name,value.copy())
 model.L_all=[]
def fixture(case,seed,init_seed):
 """Must be called under1-thread fixture-generation context for equal inputs."""
 d,n,k,r=case['D'],case['N'],case['K'],case['rank'];kind=case['kind']
 assert d<=4032 and n<=2048 and (d,n)!=(4032,12000)
 if kind=='loading_direct':assert 16*d*r*n<=128*2**20
 # Paired alternatives use identical seeds independent of their case index.
 family=kind.replace('_direct','').replace('_low','').replace('em_original','em').replace('inference_setup','inference').replace('inference_batch','inference')
 offsets={'loading':0,'latent':1,'quadratic':2,'residual':3,'em':4}
 rng=np.random.default_rng(seed+100+offsets.get(family,5));meta={'kind':kind,'shape':case,'layout':'see strides','formal_N_difference':12000-n}
 arrays=[];reset=lambda:None
 if kind in ('slogdet','likelihood_gemm','covariance_init','woodbury','beta'):
  W=cnormal(rng,(d,r))*.01;psi=np.ones(d);arrays=[W,psi]
  if kind in ('slogdet','likelihood_gemm','beta'):
   C=W@W.conj().T;C[np.diag_indices(d)]+=psi;arrays.append(C)
  if kind=='slogdet':fn=lambda:np.linalg.slogdet(C)
  elif kind=='likelihood_gemm':
   data=cnormal(rng,(n,d));X=(data-np.zeros(d)).T;arrays.append(X);fn=lambda:C@X
  elif kind=='covariance_init':
   m=adapter().make_reference_model(1,r);m.D=d;m._lambdas=W[None];m._psis=psi[None]
   def fn():
    m._update_covs();return m._covs,m._inv_covs
  elif kind=='woodbury':
   m=adapter().make_reference_model(1,r);m.D=d;m._lambdas=W[None];m._psis=psi[None]
   fn=lambda:m._invert_cov_all()
  else:fn=lambda:W.conj().T@C
 elif kind.startswith('loading_'):
  Z=cnormal(rng,(n,d)).T;H=cnormal(rng,(r,n));weights=rng.random(n);arrays=[Z,H,weights]
  fn=(lambda:np.dot(Z[:,None,:]*H[None,:,:].conj(),weights)) if kind.endswith('direct') else lambda:loading_numerator(Z,H,weights)
 elif kind.startswith('latent_'):
  H=cnormal(rng,(r,n));weights=rng.random(n);base=np.eye(r)*.7;arrays=[H,weights,base]
  fn=(lambda:np.dot(base[:,:,None]+H[:,None,:]*H[None,:,:].conj(),weights)) if kind.endswith('direct') else lambda:latent_moment(H,weights,base)
 elif kind.startswith('quadratic_'):
  X=cnormal(rng,(n,d)).T;Y=cnormal(rng,(d,n));arrays=[X,Y]
  fn=(lambda:np.real(np.sum(X.conj()*Y,axis=0))) if kind.endswith('direct') else lambda:real_quadratic(X,Y)
 elif kind.startswith('residual_'):
  Z=cnormal(rng,(n,d)).T;old=cnormal(rng,(d,n));weights=rng.random(n);arrays=[Z,old,weights]
  fn=(lambda:np.real(np.dot((Z-old)*Z.conj(),weights)/weights.sum())) if kind.endswith('direct') else lambda:residual_moment(Z,old,weights)
 elif kind=='initialization':
  data=cnormal(rng,(n,d));m=adapter().make_reference_model(k,r);m.N=n;m.D=d;arrays=[data]
  def fn():
   with adapter().recorded_initialization(init_seed):m._initialize(data)
   return tuple(getattr(m,name) for name in ('_means','_lambdas','_psis','_covs','_inv_covs','amps'))
 elif kind.startswith('em_'):
  data=cnormal(rng,(n,d));m=initialize(data,k,r,init_seed);original=state(m);arrays=[data]+list(original.values())
  if kind=='em_low':attach_low(m)
  reset=lambda:restored(m,original)
  def fn():
   m._EM_per_component(data,True)
   return tuple(getattr(m,name) for name in ('_means','_lambdas','_psis','_covs','_inv_covs','amps','rs','logLs'))
 else:
  m=adapter().make_reference_model(k,r);m.D=d;m.N=6
  m.means=cnormal(rng,(k,d))*.1;m.lambdas=cnormal(rng,(k,d,r))*.01;m.psis=np.ones((k,d));m.amps=np.ones(k)/k
  m.covs=m.lambdas@m.lambdas.conj().transpose(0,2,1)
  for component in range(k):m.covs[component,np.arange(d),np.arange(d)]+=m.psis[component]
  A=np.zeros((64,d),complex)
  A[np.arange(64),np.arange(64)*63]=1
  Cn=.37*np.eye(64,dtype=complex);y=cnormal(rng,(6,64));arrays=[m.covs,m.means,A,Cn,y]
  fn=(lambda:m._prepare_for_prediction(A,Cn)) if 'setup' in kind else lambda:adapter().MFAReferenceAdapter(m,A).estimate(y,Cn)
 meta['input_hashes']=[hash_array(a) for a in arrays];meta['array_metadata']=[{'shape':list(a.shape),'dtype':str(a.dtype),'strides':list(a.strides),'bytes':int(a.nbytes)} for a in arrays]
 return fn,reset,meta
