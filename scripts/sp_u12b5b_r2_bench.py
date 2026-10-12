"""Bounded independent synthetic kernels; never fits a formal MFA candidate."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sp_u12b5b_r2_common import *
import gc, traceback, platform
CFG=ROOT/'outputs/sp_u12b5b_r2_microbenchmark_config.json'
def cases():
 return ([{'kind':'slogdet','D':d,'N':0,'rank':16} for d in (1024,2048,4032)]+
 [{'kind':'likelihood_matmul','D':4032,'N':n,'rank':0} for n in (128,512)]+
 [{'kind':kind,'D':4032,'N':0,'rank':r} for kind in ('covariance','woodbury','beta') for r in (4,16)]+
 [{'kind':'loading_tensor','D':512,'N':1024,'rank':r} for r in (4,8,16)]+
 [{'kind':'latent_covariance','D':0,'N':1024,'rank':16},
  {'kind':'residual','D':4032,'N':512,'rank':0},
  {'kind':'latent_mean','D':4032,'N':512,'rank':16},
  {'kind':'kmeans','D':256,'N':2048,'rank':0},
  {'kind':'kmeans','D':512,'N':4096,'rank':0},
  {'kind':'array_io','D':1024,'N':1024,'rank':0}])
def validate_case(case):
 assert case['kind'] in {'slogdet','likelihood_matmul','covariance','woodbury','beta','loading_tensor','latent_covariance','residual','latent_mean','kmeans','array_io'}
 d,n,r=case['D'],case['N'],case['rank']
 assert 0<=d<=4032 and 0<=n<=4096 and r in (0,4,8,16)
 assert (n,d)!=(12000,4032) and (n,d)!=(15000,4032)
 if case['kind']=='loading_tensor':assert 16*d*r*n<=128*2**20
 assert 16*d*d+16*d*n+16*d*r*n<2*2**30
def prepare():
 firewall();OUT.mkdir(exist_ok=True)
 registry=read(ROOT/'outputs/sp_u12b5b_candidate_seed_registry.json')
 cfgold=read(ROOT/'configs/sp_u12b5b_mfa_selection.yaml')
 banned={row[v] for row in cfgold['candidate_order'] for v in ('seed','final_seed')}|{4197123401,4197123402,4197123403,4190123501,4190123500,4190123499}
 master=4218352701;seeds=[master+i for i in range(len(cases()))]
 assert not set(seeds)&banned
 for case in cases():validate_case(case)
 write(CFG,{'task':'SP-U12B5B-R2','role':'RESOURCE_TIMING_ONLY','RESOURCE_TIMING_SEED':master,'case_seeds':seeds,'cases':cases(),
 'repeats':3,'warmups':1,'dtype':'complex128','BLAS_threads':1,'concurrency':1,
 'current_limits_UNCHANGED':LIMITS,'monitor_interval_seconds':0.1,'max_worker_seconds':1800,
 'max_materialized_loading_tensor_bytes':128*2**20,'formal_seed_collision':False,
 'no_preparation_validation_target_payloads':True,'no_model_export':True,'no_formal_fit':True,
 'case_inputs':'independent Gaussian arrays or low-rank-plus-positive-diagonal synthetic covariance; no signal/channel generation'})
 print(json.dumps({'stage':'R2_synthetic_schedule_prepared','cases':len(cases()),'seeds_disjoint':True}),flush=True)
def worker():
 firewall();cfg=read(CFG)
 write(OUT/'actual_pid.json',{'actual_PID':os.getpid(),'role':'RESOURCE_TIMING_ONLY'})
 while not (OUT/'monitor_ready.json').exists():
  time.sleep(.05)
 for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
 import numpy as np, scipy
 sys.path.append(str(ROOT/'.venv-gmm/Lib/site-packages'))
 import sklearn
 from sklearn.cluster import KMeans
 from threadpoolctl import threadpool_limits,threadpool_info
 from src.baselines.mfa_reference_adapter import recorded_initialization
 def random_complex(rng,shape):
  return np.asarray((rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2),dtype=np.complex128)
 with threadpool_limits(1):
  write(OUT/'timing_environment.json',{'Python':sys.version,'NumPy':np.__version__,'SciPy':scipy.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),'processor':platform.processor(),'threadpools':threadpool_info()})
  assert all(v['num_threads']==1 for v in threadpool_info())
  proof_done=False
  for index,case in enumerate(cfg['cases']):
   validate_case(case);gc.collect();rng=np.random.default_rng(cfg['case_seeds'][index]);d,n,r=case['D'],case['N'],case['rank'];kind=case['kind'];meta={}
   write(OUT/'current_kernel.json',{'index':index,'case':case,'stage':'allocate'})
   if kind in ('slogdet','covariance','woodbury','beta'):
    W=random_complex(rng,(d,r))*.01;psi=np.full(d,1.0);WH=W.conj().T
    if kind in ('slogdet','beta'):
     C=W@WH;C[np.diag_indices(d)]+=psi
    if kind=='slogdet':fn=lambda:np.linalg.slogdet(C)
    elif kind=='covariance':fn=lambda:W@W.conj().T+np.diag(psi)
    elif kind=='beta':fn=lambda:W.conj().T@C
    else:
     def fn():
      psiI=1/psi
      inner=np.linalg.pinv(np.eye(r)+(W.conj().T*psiI[None,:])@W)
      step=psiI[:,None]*(W@inner@W.conj().T)*psiI[None,:]
      step-=np.diag(psiI)
      return -step
   elif kind=='likelihood_matmul':
    C=random_complex(rng,(d,d));data=random_complex(rng,(n,d));X1=(data-np.zeros(d)).T
    fn=lambda:C@X1
    meta={'input_strides':list(X1.strides),'input_C':bool(X1.flags.c_contiguous),'input_F':bool(X1.flags.f_contiguous)}
   elif kind=='loading_tensor':
    zeroed=random_complex(rng,(n,d)).T;latents=random_complex(rng,(r,n));rs=rng.random(n)
    if not proof_done:
     product=zeroed[:,None,:]*latents[None,:,:].conj();reduced=np.dot(product,rs)
     meta={'materialization_proof':{'shape':list(product.shape),'dtype':str(product.dtype),'nbytes':int(product.nbytes),
      'owns_data':bool(product.flags.owndata),'base_is_none':product.base is None,'shares_memory_zeroed':bool(np.shares_memory(product,zeroed)),
      'shares_memory_latents':bool(np.shares_memory(product,latents)),'strides':list(product.strides),'all_entries_finite_checked':bool(np.isfinite(product).all()),
      'all_entries_consumed_by_dot':True,'reduction_shape':list(reduced.shape),'reduction_finite':bool(np.isfinite(reduced).all()),
      'full_scale_tensor_ALLOCATED':False,'semantics':'identical NumPy eager binary multiply followed by np.dot; broadcast operand indexing is view, resulting product owns full storage'}}
     assert product.flags.owndata and product.dtype==np.complex128 and not np.shares_memory(product,zeroed)
     del product,reduced;proof_done=True
    fn=lambda:np.dot(zeroed[:,None,:]*latents[None,:,:].conj(),rs)
   elif kind=='latent_covariance':
    lat=random_complex(rng,(r,n));constant=np.eye(r)
    fn=lambda:constant[:,:,None]+lat[:,None,:]*lat[None,:,:].conj()
   elif kind=='residual':
    z=random_complex(rng,(n,d)).T;l=random_complex(rng,(d,n));rs=rng.random(n)
    fn=lambda:np.real(np.dot((z-l)*z.conj(),rs)/rs.sum())
   elif kind=='latent_mean':
    beta=random_complex(rng,(r,d));data=random_complex(rng,(n,d));mu=random_complex(rng,(d,))
    fn=lambda:beta@(data.T-mu[:,None])
   elif kind=='kmeans':
    X=rng.standard_normal((n,2*d))
    def fn():
     with recorded_initialization(cfg['case_seeds'][index]):return KMeans(n_clusters=4,n_init=1).fit(X)
   else:
    array=random_complex(rng,(n,d));scratch=OUT/'synthetic_io.npy'
    def fn():
     with scratch.open('wb') as f:np.save(f,array,allow_pickle=False)
     copy=array.copy();digest=__import__('hashlib').sha256(memoryview(copy).cast('B')).hexdigest()
     return digest
    meta={'bytes_written_per_call':int(array.nbytes),'io_caveat':'OS cached writes, not forced durable throughput; includes one array copy + SHA256'}
   write(OUT/'current_kernel.json',{'index':index,'case':case,'stage':'timing'})
   result=None
   for _ in range(cfg['warmups']):result=fn();del result
   samples=[]
   for _ in range(cfg['repeats']):
    tick=time.perf_counter();result=fn();samples.append(time.perf_counter()-tick)
    if kind=='kmeans':meta['last_kmeans_iteration_count']=int(result.n_iter_)
    del result
   row={'index':index,'case':case,'seed':cfg['case_seeds'][index],'seconds':samples,'median_seconds':float(np.median(samples)),
    'status':'MEASURED_SYNTHETIC_KERNEL','formal_fit':False,'scientific_model_exported':False,**meta}
   with (OUT/'kernel_timings.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps({'stage':'R2_kernel_complete','kind':kind,'D':d,'N':n,'rank':r,'median_seconds':row['median_seconds']}),flush=True)
   # Release case state; allocator retention remains captured by external monitor.
   del fn
   W=WH=psi=C=data=X1=zeroed=latents=rs=lat=constant=z=l=beta=mu=X=array=None
   gc.collect()
 write(OUT/'worker_access.json',{'allowed_metadata_reads':ACCESS,'denied_reads':DENIED,'scientific_value_reads':0,'formal_fit_calls':0,'model_exports':0})
def monitor():
 firewall();cfg=read(CFG);freeze=read(OUT/'execution_freeze.json')
 for n,h in freeze['sources'].items():assert sha(ROOT/n)==h,'EXECUTION_BYTE_DRIFT:'+n
 assert freeze['preflight_passed']
 assert not (OUT/'kernel_timings.jsonl').exists(),'NO_RERUN_OF_COMPLETED_TIMING'
 hs=host_snapshot();assert hs['available_physical_bytes']>4*2**30,'CURRENT_HOST_TOO_BUSY_FOR_SAFE_KERNELS'
 env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
 log=(OUT/'worker.log').open('w',encoding='utf8');proc=subprocess.Popen([sys.executable,'-B','-X','utf8',str(ROOT/'scripts/sp_u12b5b_r2_bench.py'),'worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
 start=time.monotonic();pid=None;handle=None;reason=None;samples=0;peak=0;commit=0;minfree=hs['available_physical_bytes'];records=[]
 try:
  while proc.poll() is None:
   if handle is None and (OUT/'actual_pid.json').exists():
    pid=read(OUT/'actual_pid.json')['actual_PID'];handle=process_handle(pid);write(OUT/'monitor_ready.json',{'actual_PID':pid,'launcher_PID':proc.pid})
   if handle is not None:
    mem=process_memory(handle);host=host_snapshot();samples+=1;peak=max(peak,mem['peak_working_set_bytes']);commit=max(commit,mem['peak_commit_bytes']);minfree=min(minfree,host['available_physical_bytes'])
    record={'elapsed_seconds':time.monotonic()-start,'actual_PID':pid,'process':mem,'system':host};records.append(record)
    reason=threshold_reason(mem,host,LIMITS)
    if time.monotonic()-start>cfg['max_worker_seconds']:reason='TIMING_WALL_LIMIT'
    if reason:
     # Terminate the actual interpreter, then launcher; preserve all output bytes.
     import ctypes
     fn=ctypes.windll.kernel32.TerminateProcess;fn.argtypes=[ctypes.c_void_p,ctypes.c_uint];fn.restype=ctypes.c_int
     assert fn(handle,77);proc.wait(timeout=30);break
   elif time.monotonic()-start>60:reason='PID_HANDSHAKE_TIMEOUT';proc.terminate();break
   time.sleep(cfg['monitor_interval_seconds'])
  code=proc.wait(timeout=30)
 finally:
  if handle is not None:close_handle(handle)
  log.close()
  with (OUT/'resource.jsonl').open('w',encoding='utf8') as f:
   for record in records:f.write(json.dumps(record)+'\n')
 result={'completed':code==0 and reason is None,'exit_code':code,'stop_reason':reason,'actual_PID':pid,'launcher_PID':proc.pid,'actual_PID_monitored':bool(pid),
 'samples':samples,'peak_RSS_bytes':peak,'peak_commit_bytes':commit,'minimum_available_physical_bytes':minfree,'wall_seconds':time.monotonic()-start,
 'thresholds_UNCHANGED':LIMITS,'synthetic_only':True,'formal_fit':False}
 write(OUT/'resource_summary.json',result)
 assert result['completed'],'R2_MICROBENCHMARK_STOP; preserve failed attempt; no automatic retry'
 timings=[json.loads(v) for v in (OUT/'kernel_timings.jsonl').read_text(encoding='utf8').splitlines()]
 assert len(timings)==len(cfg['cases'])
 write(ROOT/'outputs/sp_u12b5b_r2_timing_microbenchmark.json',{'resource_summary':result,'configuration':cfg,'environment':read(OUT/'timing_environment.json'),'kernels':timings,'access':read(OUT/'worker_access.json'),'full_scale_timings':'NOT_MEASURED; any full-training totals are EXTRAPOLATED'})
 print(json.dumps({'stage':'R2_microbenchmark_closed',**result}),flush=True)
if __name__=='__main__':
 try:{'prepare':prepare,'worker':worker,'monitor':monitor}[sys.argv[1]]()
 except Exception:
  OUT.mkdir(exist_ok=True);(OUT/f'exception_{sys.argv[1]}.txt').write_text(traceback.format_exc(),encoding='utf8');raise
