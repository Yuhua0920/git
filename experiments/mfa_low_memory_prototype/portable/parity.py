"""Synthetic B5A-case, initialization, stop, posterior and full-trace comparisons."""
from pathlib import Path
import numpy as np,copy,types,io,contextlib,json
from kernels import adapter,cnormal,initialize,attach_low,state
from low_memory import loading_numerator,latent_moment,real_quadratic,residual_moment
ATOL=1e-10;RTOL=1e-9
def error(a,b):
 a,b=np.asarray(a),np.asarray(b);difference=abs(a-b)
 return {'PASS':bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.allclose(a,b,atol=ATOL,rtol=RTOL)),
 'max_absolute':float(np.max(difference,initial=0)),'max_relative':float(np.max(difference/np.maximum(abs(b),ATOL),initial=0))}
def expression_tests():
 rng=np.random.default_rng(4237659001);Z=cnormal(rng,(127,29)).T;H=cnormal(rng,(8,127));w=rng.random(127);old=cnormal(rng,Z.shape);Y=cnormal(rng,Z.shape);base=np.eye(8)*.7
 original={'loading':np.dot(Z[:,None,:]*H[None,:,:].conj(),w),'latent':np.dot(base[:,:,None]+H[:,None,:]*H[None,:,:].conj(),w),
 'quadratic':np.real(np.sum(Z.conj()*Y,axis=0)),'residual':np.real(np.dot((Z-old)*Z.conj(),w)/w.sum())}
 new={'loading':loading_numerator(Z,H,w),'latent':latent_moment(H,w,base),'quadratic':real_quadratic(Z,Y),'residual':residual_moment(Z,old,w)}
 return {name:error(new[name],value) for name,value in original.items()}
def analytical_case():
 from toy_case_reference import known_model,dense_em_step,DATA_SEED
 m=known_model(2,8,2);data=cnormal(np.random.default_rng(DATA_SEED),(160,8));m.N=len(data)
 expected=dense_em_step(m,data);m._EM_per_component(data,True)
 names={'means':'_means','lambdas':'_lambdas','psis':'_psis','amps':'amps','covs':'_covs','inv_covs':'_inv_covs','responsibilities':'rs','log_likelihood':'logLs'}
 return {name:error(getattr(m,attr),expected[name]) for name,attr in names.items()}
def traces(variant):
 configs=[(1,20,4,160),(2,8,2,160),(2,20,4,160),(2,20,8,160),(2,20,16,160),(4,32,8,256)]
 result={};status=[]
 for index,(k,d,r,n) in enumerate(configs):
  rng=np.random.default_rng(4237659002+index);data=cnormal(rng,(n,d));model=initialize(data,k,r,4237659011+index)
  initial=state(model)
  for name,value in initial.items():result[f'case{index}_initial_{name}']=value
  if variant=='low':attach_low(model)
  # Exercise a nontrivial known-model trajectory as well as exact initialization.
  from toy_case_reference import known_model
  warm=known_model(k,d,r);model._means=warm._means.copy();model._lambdas=warm._lambdas.copy();model._psis=warm._psis.copy();model.amps=warm.amps.copy();model._covs=warm._covs.copy();model._inv_covs=warm._inv_covs.copy()
  step=model._EM_per_component;iteration=0
  def capture(values,PPCA):
   nonlocal iteration
   step(values,PPCA)
   for name,value in state(model).items():result[f'case{index}_step{iteration}_{name}']=value
   result[f'case{index}_step{iteration}_responsibilities']=model.rs.copy();result[f'case{index}_step{iteration}_logLs']=model.logLs.copy()
   result[f'case{index}_step{iteration}_objective']=np.asarray(model.logLs.sum());iteration+=1
  model._EM_per_component=capture
  with contextlib.redirect_stdout(io.StringIO()):model.run_em(data)
  history=np.asarray(model.L_all);result[f'case{index}_history']=history
  # Preserve the actual author stopping rule/status, not an artificial iteration cap.
  stop=bool(len(history)>=7 and abs((history[-1]-history[-2])/history[-1])<1e-4)
  status.append({'case':index,'K':k,'D':d,'rank':r,'N':n,'iterations':len(history),'stopping_criterion_satisfied':stop,'finite':bool(np.isfinite(history).all())})
  for public,private in (('means','_means'),('lambdas','_lambdas'),('psis','_psis'),('covs','_covs'),('inv_covs','_inv_covs')):setattr(model,public,getattr(model,private).copy())
  A=np.eye(d,dtype=complex)[[0,d//2,d-1]];y=cnormal(rng,(11,3));Cn=.37*np.eye(3,dtype=complex)
  posterior=adapter().MFAReferenceAdapter(model,A).estimate(y,Cn)
  result[f'case{index}_posterior']=posterior;result[f'case{index}_posterior_responsibility']=model.predict_proba(y)
  from toy_case_reference import dense_posterior
  expected=dense_posterior(model,y,Cn,A)[2]
  result[f'case{index}_posterior_analytic_difference']=np.asarray(float(np.max(abs(posterior-expected))))
 return result,status
def run(threads,out,reference_path,reference_status):
 out=Path(out);expressions=expression_tests();analytical=analytical_case();allchecks={};statuses={}
 for variant in ('original','low'):
  values,states=traces(variant);statuses[variant]=states
  # These archives are SMALL SYNTHETIC PARAMETER TRACES, not a fitted baseline.
  np.savez(out/f'synthetic_{variant}_trace.npz',**values)
  if reference_path is None and variant=='original':reference=values;refstates=states
  else:
   if reference_path is None:reference=original_values;refstates=statuses['original']
   else:
    with np.load(reference_path,allow_pickle=False) as f:reference={key:f[key] for key in f.files}
    refstates=reference_status
  same_stops=states==refstates
  keys=set(values)==set(reference)
  checks={key:error(value,reference[key]) for key,value in values.items() if key in reference and value.shape==reference[key].shape}
  analytic_ok=all(values[f'case{i}_posterior_analytic_difference']<=ATOL+RTOL for i in range(6))
  allchecks[variant]={'PASS':same_stops and keys and len(checks)==len(reference) and all(v['PASS'] for v in checks.values()) and analytic_ok,
   'same_stop_decisions':same_stops,'same_trace_keys':keys,'checks':checks,'analytic_posterior_PASS':analytic_ok,'largest_absolute_difference':max(v['max_absolute'] for v in checks.values())}
  if variant=='original':original_values=values
 result={'threads':threads,'tolerance':{'atol':ATOL,'rtol':RTOL},'B5A_exact_analytical_case':analytical,'expression_parity':expressions,
 'variants':allchecks,'stopping':statuses,'formal_fit':False,'toy_only':True,'mathematical_equivalence':'exact-arithmetic identities; floating summation order differs',
 'PASS':all(v['PASS'] for v in list(expressions.values())+list(analytical.values())) and all(v['PASS'] for v in allchecks.values())}
 return result
