"""Portable bounded synthetic runner; NO formal data, training, or remote access."""
from pathlib import Path
import os,sys,json,time,argparse,subprocess,ctypes,traceback,random,gc
sys.dont_write_bytecode=True
PORT=Path(__file__).resolve().parent;sys.path.insert(0,str(PORT))
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
 for attempt in range(101):
  try:os.replace(temporary,path);break
  except PermissionError as exc:
   if getattr(exc,'winerror',None) not in (5,32,33) or attempt==100:raise
   time.sleep(.01)
def get(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def optional_marker(path):
 try:return get(path)
 except (PermissionError,FileNotFoundError):return None
def clean_exit_race(proc,final_path):
 # Windows Python venv launcher can outlive its actual interpreter briefly.
 # Accept a lost PID only after launcher exit0 AND a complete worker final record.
 try:code=proc.wait(timeout=2)
 except subprocess.TimeoutExpired:return False
 return code==0 and Path(final_path).is_file()
def terminate(pid):
 if os.name!='nt':os.kill(pid,15);return
 op=ctypes.windll.kernel32.OpenProcess;op.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];op.restype=ctypes.c_void_p
 handle=op(1,False,pid);assert handle
 fn=ctypes.windll.kernel32.TerminateProcess;fn.argtypes=[ctypes.c_void_p,ctypes.c_uint];fn.restype=ctypes.c_int
 try:assert fn(handle,77)
 finally:
  close=ctypes.windll.kernel32.CloseHandle;close.argtypes=[ctypes.c_void_p];close(handle)
def install_scope(root):
 if root:
  sys.path.insert(0,str(Path(root)));from scripts.sp_u12b5b_r3_common import firewall
  firewall()
def worker(args):
 install_scope(args.project_root);cfg=get(args.config);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 write(out/'actual_pid.json',{'actual_PID':os.getpid(),'role':args.role})
 while not (out/'monitor_ready.json').exists():time.sleep(.02)
 requested=args.threads
 for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]=str(requested)
 import numpy as np,scipy
 if args.project_root:sys.path.append(str(Path(args.project_root)/'.venv-gmm/Lib/site-packages'))
 import sklearn
 from threadpoolctl import threadpool_limits,threadpool_info
 from host_inventory import process_memory,frequencies,topology
 from kernels import fixture
 assert requested<=topology()['physical_cores'],'NO_NONEXISTENT_CORES'
 with threadpool_limits(requested):
  pools=threadpool_info();assert pools and all(v['num_threads']==requested for v in pools),'THREAD_CONTROL_NOT_EFFECTIVE'
  write(out/'environment.json',{'requested_threads':requested,'actual_pools':pools,'Python':sys.version,'NumPy':np.__version__,'SciPy':scipy.__version__,'sklearn':sklearn.__version__,'thread_env':{n:os.environ[n] for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},'frequency':frequencies()})
  if args.role=='parity':
   from parity import run
   ref=None if requested==1 else Path(args.reference)/'synthetic_original_trace.npz'
   refstatus=None if requested==1 else get(Path(args.reference)/'parity.json')['stopping']['original']
   value=run(requested,out,ref,refstatus);value['actual_pools']=threadpool_info();write(out/'parity.json',value)
   print(json.dumps({'role':'parity','threads':requested,'PASS':value['PASS']}),flush=True)
  else:
   job=cfg['schedule'][args.job];assert job['threads']==requested
   for index in job['case_order']:
    case=cfg['cases'][index];write(out/'current_case.json',{'case_index':index,'kind':case['kind'],'stage':'fixture'})
    # All configurations receive bitwise-identical fixtures generated with1 thread.
    with threadpool_limits(1):fn,reset,meta=fixture(case,cfg['DATA_SEED'],cfg['INITIALIZATION_SEED'])
    actual=threadpool_info();assert all(v['num_threads']==requested for v in actual)
    reset();fn();gc.collect()
    reset();write(out/'current_case.json',{'case_index':index,'kind':case['kind'],'stage':'measurement'})
    before=process_memory();freq_before=frequencies();cpu_start=time.process_time_ns();start=time.perf_counter_ns();result=fn();wall=time.perf_counter_ns()-start;cpu=time.process_time_ns()-cpu_start
    after=process_memory();freq_after=frequencies()
    write(out/'current_case.json',{'case_index':index,'kind':case['kind'],'stage':'1-thread numerical reference, NOT timed'})
    with threadpool_limits(1):reset();reference=fn()
    left=result if isinstance(result,tuple) else (result,);right=reference if isinstance(reference,tuple) else (reference,)
    checks=[]
    for a,b in zip(left,right):
     a,b=np.asarray(a),np.asarray(b);difference=abs(a-b)
     checks.append({'PASS':bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.allclose(a,b,atol=cfg['parity_tolerance']['atol'],rtol=cfg['parity_tolerance']['rtol'])),
      'max_absolute':float(np.max(difference,initial=0)),'max_relative':float(np.max(difference/np.maximum(abs(b),1e-10),initial=0)),'bitwise':bool(np.array_equal(a,b))})
    row={'job':args.job,'block':job['block'],'threads':requested,'case_index':index,'kind':case['kind'],'wall_ns':wall,'process_CPU_ns':cpu,
    'effective_CPU_cores':cpu/wall,'CPU_utilization_percent_of_logical_host':100*cpu/wall/topology()['logical_threads'],
    'RSS_before':before['working_set_bytes'],'RSS_after':after['working_set_bytes'],'OS_peak_RSS_so_far':after['peak_working_set_bytes'],
    'private_commit_after':after['private_bytes'],'OS_peak_commit_so_far':after['peak_commit_bytes'],'actual_pools':actual,
    'frequency_before':freq_before,'frequency_after':freq_after,'metadata':meta,'numerical_parity':checks,'numerical_parity_PASS':all(v['PASS'] for v in checks),'timing_scope':'kernel only; fixture/model reset/hash/warmup/parity check not timed','full_scale':False}
    with (out/'timings.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps({'role':'kernel','block':job['block'],'threads':requested,'kind':case['kind'],'seconds':wall/1e9}),flush=True)
    del result,reference,left,right,checks,fn,reset,meta,a,b,difference;gc.collect()
  write(out/'worker_final.json',{'memory':process_memory(),'actual_pools':threadpool_info(),'formal_fit_calls':0,'validation_scores':0,'target_data_reads':0})
def monitored(config,out,role,threads,job=None,reference=None,project_root=None):
 from host_inventory import process_memory,memory
 cfg=get(config);out=Path(out);assert not out.exists(),'FRESH_JOB_DIRECTORY_REQUIRED';out.mkdir(parents=True)
 log=(out/'worker.log').open('w',encoding='utf8');command=[sys.executable,'-B','-X','utf8',str(PORT/'runner.py'),'--role',role,'--config',str(config),'--out',str(out),'--threads',str(threads)]
 if job is not None:command+=['--job',str(job)]
 if reference is not None:command+=['--reference',str(reference)]
 if project_root is not None:command+=['--project-root',str(project_root)]
 proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);start=time.monotonic();pid=None;reason=None;records=[];peak=commit=0;minfree=memory()['available_physical_bytes']
 assert minfree>4*2**30,'LAUNCH_AVAILABLE_MEMORY_TOO_LOW'
 while proc.poll() is None:
  if pid is None and (out/'actual_pid.json').exists():
   pid=get(out/'actual_pid.json')['actual_PID'];write(out/'monitor_ready.json',{'actual_PID':pid,'launcher_PID':proc.pid})
  if pid is not None:
   try:m=process_memory(pid);system=memory()
   except OSError:
    if not clean_exit_race(proc,out/'worker_final.json'):raise
    break
   peak=max(peak,m['peak_working_set_bytes']);commit=max(commit,m['peak_commit_bytes']);minfree=min(minfree,system['available_physical_bytes'])
   marker=optional_marker(out/'current_case.json')
   records.append({'elapsed':time.monotonic()-start,'actual_PID':pid,'process':m,'system':system,'marker':marker})
   if peak>=cfg['resource_limits']['max_process_bytes']:reason='PROCESS_RSS_LIMIT'
   if minfree<cfg['resource_limits']['min_system_available_physical_bytes']:reason='PHYSICAL_HEADROOM'
   if time.monotonic()-start>cfg['worker_timeout_seconds']:reason='WORKER_TIMEOUT'
   if reason:terminate(pid);break
  elif time.monotonic()-start>60:reason='PID_HANDSHAKE_TIMEOUT';proc.terminate();break
  time.sleep(cfg['sampling_seconds'])
 code=proc.wait(timeout=30);log.close()
 with (out/'resource.jsonl').open('w',encoding='utf8') as f:
  for item in records:f.write(json.dumps(item)+'\n')
 final=get(out/'worker_final.json') if (out/'worker_final.json').exists() else {}
 if final:peak=max(peak,final['memory']['peak_working_set_bytes']);commit=max(commit,final['memory']['peak_commit_bytes'])
 if peak>=cfg['resource_limits']['max_process_bytes']:reason='FINAL_PROCESS_RSS_LIMIT'
 value={'role':role,'threads':threads,'job':job,'actual_PID':pid,'launcher_PID':proc.pid,'completed':code==0 and reason is None,'exit_code':code,'stop_reason':reason,'samples':len(records),
 'peak_RSS':peak,'peak_private_commit':commit,'minimum_available_physical_bytes':minfree,'wall_seconds':time.monotonic()-start,'limits_UNCHANGED':cfg['resource_limits']}
 write(out/'resource_summary.json',value);assert value['completed'],'RESOURCE_OR_WORKER_FAILURE; retain this directory; no automatic retry'
 return value
def portable_run(args):
 # This command creates a new host-specific safe synthetic config, never a fit.
 from host_inventory import inventory
 import numpy,scipy,sklearn
 host=inventory();physical=host['topology']['physical_cores'];assert physical
 out=Path(args.out);assert not out.exists();out.mkdir(parents=True)
 cfg=get(PORT/'current_host_config.json');threads=sorted({v for v in (1,2,4,8,12,physical) if v<=physical});cfg['thread_candidates']=threads;cfg['physical_core_count']=physical;cfg['logical_thread_count']=host['topology']['logical_threads'];cfg['schedule']=[]
 rng=random.Random(cfg['THREAD_ORDER_SEED'])
 for block in range(cfg['measurement_blocks']):
  order=threads.copy();rng.shuffle(order);caseorder=list(range(len(cfg['cases'])));rng.shuffle(caseorder)
  for t in order:cfg['schedule'].append({'job':len(cfg['schedule']),'block':block,'threads':t,'case_order':caseorder})
 write(out/'host_inventory.json',host);write(out/'host_specific_config.json',cfg)
 for t in threads:monitored(out/'host_specific_config.json',out/f'parity_{t}', 'parity',t,reference=None if t==1 else out/'parity_1')
 for job in cfg['schedule']:monitored(out/'host_specific_config.json',out/f"job_{job['job']:03d}",'kernels',job['threads'],job=job['job'])
 print('SAFE_SYNTHETIC_BENCHMARK_COMPLETE; NO FULL-SCALE TRAINING AUTHORIZED')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--role',choices=('kernels','parity','inventory','portable-run'),required=True);parser.add_argument('--config');parser.add_argument('--out');parser.add_argument('--threads',type=int,default=1);parser.add_argument('--job',type=int);parser.add_argument('--reference');parser.add_argument('--project-root');args=parser.parse_args()
 if args.role=='inventory':
  from host_inventory import inventory
  result=inventory();write(args.out,result) if args.out else print(json.dumps(result,indent=2));return
 if args.role=='portable-run':portable_run(args)
 else:worker(args)
if __name__=='__main__':
 try:main()
 except Exception:
  # Preserve errors beside the requested output; no rescue or threshold changes.
  traceback.print_exc();raise
