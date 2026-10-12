"""Full-shape synthetic resource probe, original batch code, no model outputs."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from scripts.sp_u12b5b_r1_common import *
firewall()
import numpy as np,scipy
sys.path.append(str(ROOT/'.venv-gmm/Lib/site-packages'))
from threadpoolctl import threadpool_limits,threadpool_info
from src.baselines.mfa_reference_adapter import make_reference_model,fit_reference
import traceback
def probe():
 cfg=read(CONFIG);structure=cfg['structures'][0];name=structure['name'];directory=OUT/name;directory.mkdir(exist_ok=True)
 write(directory/'PID.json',dict(actual_PID=os.getpid(),created_UTC=utc(),role='RESOURCE_ONLY'))
 started=time.monotonic()
 while not (directory/'MONITOR_READY.json').exists():
  if time.monotonic()-started>30:raise TimeoutError('MONITOR_NOT_ATTACHED')
  time.sleep(.05)
 handle=process_handle(os.getpid());eventfile=directory/'stages.jsonl';previous=None
 def event(stage,frame=None):
  arrays=[]
  if frame is not None:
   seen=set()
   for key,value in frame.f_locals.items():
    if isinstance(value,np.ndarray) and id(value) not in seen:
     seen.add(id(value));arrays.append(dict(name=key,shape=list(value.shape),dtype=str(value.dtype),nbytes=int(value.nbytes)))
   instance=frame.f_locals.get('self')
   if instance is not None:
    for key in ('_means','_lambdas','_psis','amps','_covs','_inv_covs','rs'):
     value=getattr(instance,key,None)
     if isinstance(value,np.ndarray) and id(value) not in seen:
      seen.add(id(value));arrays.append(dict(name='self.'+key,shape=list(value.shape),dtype=str(value.dtype),nbytes=int(value.nbytes)))
  record=dict(UTC=utc(),elapsed_seconds=time.monotonic()-started,stage=stage,source_function=frame.f_code.co_name if frame is not None else None,source_line=frame.f_lineno if frame is not None else None,arrays_metadata_only=arrays,process=process_memory(handle),system=system_memory())
  with eventfile.open('a',encoding='utf8') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
 def trace(frame,kind,arg):
  nonlocal previous
  filename=Path(frame.f_code.co_filename)
  if kind=='line' and (filename==CORE or filename.name=='_kmeans.py' and frame.f_code.co_name in ('fit','_tolerance','_kmeans_single_lloyd')):
   stage=f'{filename.name}:{frame.f_code.co_name}:{frame.f_lineno}'
   if stage!=previous:event(stage,frame);previous=stage
   return trace
  return trace if filename==CORE or filename.name=='_kmeans.py' else None
 try:
  event('SYNTHETIC_INPUT_ALLOCATION')
  rng=np.random.default_rng(cfg['seeds']['synthetic_data_seed']);data=np.empty((cfg['N'],cfg['D']),dtype=np.complex128)
  for first in range(0,cfg['N'],256):
   last=min(first+256,cfg['N']);data.real[first:last]=rng.standard_normal((last-first,cfg['D']))/np.sqrt(2);data.imag[first:last]=rng.standard_normal((last-first,cfg['D']))/np.sqrt(2)
  assert data.shape==(12000,4032) and data.dtype==np.complex128 and data.flags.c_contiguous
  event('SYNTHETIC_INPUT_READY')
  model=make_reference_model(structure['K'],structure['rank']);model.maxiter=cfg['maxiter']
  write(directory/'input_metadata.json',dict(shape=list(data.shape),dtype=str(data.dtype),nbytes=int(data.nbytes),synthetic=True,seeds=cfg['seeds'],preparation_IDs=[],target_IDs=[],formal_seed_used=False,author_iteration_cap_only=cfg['maxiter']))
  write(directory/'access.json',dict(reads=list(ACCESS),denied=list(DENIED),preparation_payload_reads=0,target_scientific_value_reads=0,parameters_saved=False,objective_values_saved=False,snapshot_stage='before_original_fit',enforced_allowlist=True))
  event('ORIGINAL_AUTHOR_FIT_ENTRY');sys.settrace(trace)
  with threadpool_limits(1):
   write(directory/'environment.json',dict(Python=sys.version,NumPy=np.__version__,SciPy=scipy.__version__,threadpools=threadpool_info(),BLAS_threads=1))
   fit_reference(model,data,cfg['seeds']['resource_initialization_seed'])
  sys.settrace(None);event('ORIGINAL_FIT_RETURN_NO_MODEL_SERIALIZED')
  write(directory/'completion.json',dict(resource_allocation_path_completed=True,formal_candidate=False,validation=False,selection=False,parameter_values_exported=False,objective_values_exported=False,threadpools=threadpool_info()))
 except BaseException as error:
  sys.settrace(None);write(directory/'technical_failure.json',dict(exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),scientific_values_exported=False));raise
 finally:
  sys.settrace(None);close_handle(handle);write(directory/'access.json',dict(reads=ACCESS,denied=DENIED,preparation_payload_reads=0,target_scientific_value_reads=0,parameters_saved=False,objective_values_saved=False))
if __name__=='__main__':probe()
