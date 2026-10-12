"""Fair CPU estimator-core adapter; no scientific method changes or result access."""
import os
THREAD_KEYS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','GOTO_NUM_THREADS','BLIS_NUM_THREADS')
for key in THREAD_KEYS:os.environ[key]='1'
from pathlib import Path
import sys,ctypes,struct,platform,subprocess,json,time,gc,itertools,threading,winreg
import numpy as np
import scipy,scipy.linalg
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u12b4a_preserve import sha,read,write,OUT,firewall,tree
OPERATOR_PATH=ROOT/'outputs/sp_u12b2a/operators.npz'
OPERATOR_SHA='6df61f6b547c4f1f553212a90c197eb78d838786af3498548d3062e733668134'
CONFIG=ROOT/'configs/sp_u12b4a_runtime_protocol.yaml'
B4B=ROOT/'outputs/sp_u12b4b_runtime'
STREAM=OUT/'benchmark_safe/pilot_streams.npz'
METHODS=('Source','Dense','LR4');SNRS=(10,20);ENDPOINTS=('estimator_core','correction_only')
STREAM_SEED=4290123401;ORDER_SEED=4290123402;BOOTSTRAP_SEED=4290123403

def need(v,code):
 if not v:raise RuntimeError(code)

def keys(method,snr):
 names=['source_mean_p','source_mean_d',f'Wsource_{snr}']
 if method!='Source':names += [f'KSource_{snr}',f'DeltaKP_{snr}',f'DeltaKM_{snr}']
 if method=='Dense':names += [f'DeltaWP_{snr}',f'DeltaWM_{snr}']
 if method=='LR4':names += [f'A_P_{snr}',f'Q_P_{snr}',f'A_M_{snr}',f'Q_M_{snr}']
 need(method in METHODS,'UNDECLARED_METHOD');return names

def load(method,snr):
 """Only method-required arrays; no other-SNR/origin arrays retained."""
 need(snr in SNRS,'UNDECLARED_SNR')
 with np.load(OPERATOR_PATH,allow_pickle=False) as archive:
  a={k:archive[k].copy() for k in keys(method,snr)}
 for v in a.values():
  need(v.dtype==np.complex128 and v.flags.c_contiguous and np.isfinite(v).all(),'PRODUCTION_DTYPE_LAYOUT')
  v.flags.writeable=False
 return Estimator(method,snr,a)

class Estimator:
 """Common fit and Source application. Only two product representations differ."""
 def __init__(self,method,snr,a):
  self.method,self.snr,self.a=method,snr,a
  self.mean_p=a['source_mean_p'];self.mean_d=a['source_mean_d'];self.W=a[f'Wsource_{snr}']
  need(self.mean_p.shape==(64,) and self.mean_d.shape==(3968,) and self.W.shape==(3968,64),'DEPLOYMENT_SHAPE')
  if method!='Source':
   from src.pilot_cv import frozen_folds
   from src.cdl_benchmark import observation_geometry
   self.folds,_=frozen_folds(observation_geometry().selected)
   self.K=a[f'KSource_{snr}'];self.KP=a[f'DeltaKP_{snr}'];self.KM=a[f'DeltaKM_{snr}']
   self.buffer=np.zeros((5,64),np.complex128)
   self.S=np.zeros((2,2),np.float64);self.b=np.zeros(2,np.float64);self.c=np.zeros(2,np.float64)
   self.fill=0
  if method=='Dense':
   self.BP=a[f'DeltaWP_{snr}'];self.BM=a[f'DeltaWM_{snr}']
   need(self.BP.shape==self.BM.shape==(3968,64),'DENSE_SHAPE')
  if method=='LR4':
   self.AP=a[f'A_P_{snr}'];self.QP=a[f'Q_P_{snr}'];self.AM=a[f'A_M_{snr}'];self.QM=a[f'Q_M_{snr}']
   need(self.AP.shape==self.AM.shape==(3968,4) and self.QP.shape==self.QM.shape==(4,64),'FACTOR_SHAPE')
 def reset(self):
  if self.method!='Source':self.buffer.fill(0);self.fill=0
 def update(self,p):
  if self.fill<5:self.buffer[self.fill]=p;self.fill+=1
  else:self.buffer[:-1]=self.buffer[1:];self.buffer[-1]=p
 def fit(self):
  # The production frame_objective multiplication and sum order are preserved.
  # Objective-value/condition-number diagnostics are omitted for both paths.
  moments=[]
  for p in self.buffer:
   u=p-self.mean_p;columns=[];responses=[]
   for f,(observed,held) in enumerate(self.folds):
    x=u[observed];base=self.K[f]@x
    columns.append(np.column_stack((self.KP[f]@x,self.KM[f]@x)))
    responses.append(u[held]-base)
   D=np.concatenate(columns);e=np.concatenate(responses)
   moments.append((D.conj().T@D,D.conj().T@e))
  self.S[:]=sum(v[0] for v in moments).real[np.ix_([0,1],[0,1])]
  self.b[:]=sum(v[1] for v in moments).real[[0,1]]
  self.c[:]=np.linalg.solve(self.S,self.b)
  return self.c
 def product_P(self,u):
  return u@self.BP.T if self.method=='Dense' else self.AP@(self.QP@u)
 def product_M(self,u):
  return u@self.BM.T if self.method=='Dense' else self.AM@(self.QM@u)
 def correction(self,u,c):
  return c[0]*self.product_P(u)+c[1]*self.product_M(u)
 def source(self,u):return self.mean_d+u@self.W.T
 def step(self,p):
  if self.method!='Source':
   self.update(p)
   if self.fill<5:return None
   c=self.fit()
  u=p-self.mean_p;base=self.source(u)
  if self.method=='Source':return base
  # Preserve base + alpha*P + beta*M order, rather than base+(alpha*P+beta*M).
  return base+c[0]*self.product_P(u)+c[1]*self.product_M(u)
 def persistent_arrays(self):
  rows=[(k,'CORRECTION_SPECIFIC' if k.startswith(('DeltaW','A_','Q_')) else 'COMMON' if self.method!='Source' else 'SOURCE_ONLY',v) for k,v in self.a.items()]
  if self.method!='Source':
   rows += [('five_frame_buffer','COMMON',self.buffer),('normal_S','COMMON',self.S),('normal_b','COMMON',self.b),('coordinates_c','COMMON',self.c)]
   rows += [(f'fold_{f}_{role}','COMMON',v) for f,pair in enumerate(self.folds) for role,v in zip(('observed','heldout'),pair)]
  # Aliases (self.W, self.K, etc.) are not charged a second time.
  return rows

def stream_arrays(snr):
 with np.load(STREAM,allow_pickle=False) as f:answer={k:f[f'{k}_{snr}'].copy() for k in ('p','u','c')}
 for v in answer.values():v.flags.writeable=False
 return answer

def warmup(obj,data,endpoint,passes):
 for _ in range(passes):
  if endpoint=='estimator_core':
   obj.reset()
   for p in data['p']:obj.step(p)
  else:
   for j in range(4,len(data['p'])):obj.correction(data['u'][j],data['c'][j])

def measure(obj,data,endpoint,passes):
 """Only step/correction between adjacent clocks; all storage/I/O is outside."""
 n=len(data['p'])-4;latencies=np.empty(passes*n,np.int64);index=0
 for _ in range(passes):
  if endpoint=='estimator_core':
   obj.reset()
   for p in data['p'][:4]:obj.step(p)
  for j in range(4,len(data['p'])):
   if endpoint=='estimator_core':
    p=data['p'][j];start=time.perf_counter_ns();output=obj.step(p);stop=time.perf_counter_ns()
   else:
    u,c=data['u'][j],data['c'][j];start=time.perf_counter_ns();output=obj.correction(u,c);stop=time.perf_counter_ns()
   latencies[index]=stop-start;index+=1
 need(np.all(latencies>0),'TIMER_MONOTONICITY_FAILURE')
 return latencies

def orders():
 rng=np.random.default_rng(ORDER_SEED);answer={}
 for endpoint,names in [('estimator_core',METHODS),('correction_only',METHODS[1:])]:
  perms=list(itertools.permutations(names));rows=(perms*(12//len(perms)))
  rows=[rows[i] for i in rng.permutation(12)]
  for snr in SNRS:answer[f'{endpoint}_{snr}']=[list(v) for v in rows]
 return answer

def blas_threads():
 answer=[]
 for package in (np,scipy):
  for p in (Path(package.__file__).parent.parent/(package.__name__+'.libs')).glob('*openblas*.dll'):
   lib=ctypes.CDLL(str(p))
   names=('scipy_openblas_get_num_threads64_','scipy_openblas_get_num_threads','openblas_get_num_threads64_','openblas_get_num_threads')
   symbol=next((n for n in names if hasattr(lib,n)),None);need(symbol is not None,'BLAS_THREAD_QUERY_UNAVAILABLE')
   f=getattr(lib,symbol);f.restype=ctypes.c_int
   info_symbol=next((n for n in ('scipy_openblas_get_config64_','scipy_openblas_get_config','openblas_get_config') if hasattr(lib,n)),None)
   details='UNAVAILABLE'
   if info_symbol:
    info=getattr(lib,info_symbol);info.restype=ctypes.c_char_p;details=info().decode('ascii',errors='replace')
   answer.append(dict(library=str(p),symbol=symbol,threads=f(),runtime_configuration=details))
 need(len(answer)>=2 and all(v['threads']==1 for v in answer),'ACTIVE_BLAS_THREADS_NOT_ONE')
 return answer

class PMC(ctypes.Structure):
 _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
def rss():
 k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p
 ps=ctypes.WinDLL('psapi',use_last_error=True);ps.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
 p=PMC();p.cb=ctypes.sizeof(p)
 if not ps.GetProcessMemoryInfo(k.GetCurrentProcess(),ctypes.byref(p),p.cb):raise ctypes.WinError(ctypes.get_last_error())
 return dict(PID=os.getpid(),RSS_bytes=int(p.WorkingSetSize),lifetime_peak_RSS_bytes=int(p.PeakWorkingSetSize))
def suspend_clock():
 """Windows interrupt counter includes suspend; unbiased counter excludes it."""
 k=ctypes.WinDLL('kernelbase',use_last_error=True)
 included=k.QueryInterruptTimePrecise;excluded=k.QueryUnbiasedInterruptTimePrecise
 included.argtypes=excluded.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
 included.restype=excluded.restype=None
 inc=ctypes.c_ulonglong();exc=ctypes.c_ulonglong();included(ctypes.byref(inc));excluded(ctypes.byref(exc))
 return dict(including_sleep_100ns=int(inc.value),excluding_sleep_100ns=int(exc.value))
def suspended(start,end):
 inc=end['including_sleep_100ns']-start['including_sleep_100ns'];exc=end['excluding_sleep_100ns']-start['excluding_sleep_100ns']
 need(inc>=0 and exc>=0,'POWER_COUNTER_MONOTONICITY_FAILURE')
 # 100ms tolerance accounts for sequential counter reads/OS scheduling.
 return inc-exc>1_000_000
def system_context():
 """Optional OS load/frequency context, never an outlier-exclusion rule."""
 k=ctypes.WinDLL('kernel32',use_last_error=True);idle=ctypes.c_ulonglong();kernel=ctypes.c_ulonglong();user=ctypes.c_ulonglong()
 call=k.GetSystemTimes;call.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
 ok=call(ctypes.byref(idle),ctypes.byref(kernel),ctypes.byref(user))
 answer=dict(system_CPU_times_available=bool(ok),idle_100ns=int(idle.value),kernel_100ns=int(kernel.value),user_100ns=int(user.value),CPU_total_includes_benchmark=True,thermal_telemetry='UNAVAILABLE',telemetry_unavailable_is_not_latency_filter=True)
 try:
  class PowerInfo(ctypes.Structure):
   _fields_=[(name,ctypes.c_ulong) for name in ('Number','MaxMhz','CurrentMhz','MhzLimit','MaxIdleState','CurrentIdleState')]
  buffer=(PowerInfo*os.cpu_count())();f=ctypes.WinDLL('powrprof').CallNtPowerInformation
  f.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_ulong,ctypes.c_void_p,ctypes.c_ulong];f.restype=ctypes.c_ulong
  status=f(11,None,0,ctypes.byref(buffer),ctypes.sizeof(buffer))
  answer['frequency_query_available']=status==0
  answer['frequency_MHz_OS_reported']=[int(v.CurrentMhz) for v in buffer] if status==0 else None
 except (AttributeError,OSError):answer['frequency_query_available']=False;answer['frequency_MHz_OS_reported']=None
 return answer
def physical_cores():
 k=ctypes.WinDLL('kernel32',use_last_error=True);f=k.GetLogicalProcessorInformationEx
 f.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)];size=ctypes.c_ulong()
 f(0,None,ctypes.byref(size));need(size.value>0,'CPU_TOPOLOGY_UNAVAILABLE');buffer=ctypes.create_string_buffer(size.value)
 need(f(0,buffer,ctypes.byref(size)),'CPU_TOPOLOGY_QUERY_FAILED');offset=0;count=0
 while offset<size.value:
  relationship,n=struct.unpack_from('II',buffer.raw,offset);need(n>=8,'CPU_TOPOLOGY_SCHEMA')
  if relationship==0:count+=1
  offset+=n
 return count
def environment():
 with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:model=winreg.QueryValueEx(k,'ProcessorNameString')[0].strip()
 power=subprocess.check_output(['powercfg','/getactivescheme'],text=True,encoding='utf8',errors='replace').strip()
 timer=time.get_clock_info('perf_counter')
 return dict(OS=platform.platform(),CPU_model=model,physical_cores=physical_cores(),logical_cores=os.cpu_count(),process_eligible_cores=os.process_cpu_count(),Python=sys.version,executable=sys.executable,NumPy=np.__version__,SciPy=scipy.__version__,BLAS_build=np.__config__.show(mode='dicts')['Build Dependencies']['blas'],active_BLAS=blas_threads(),thread_environment={k:os.environ[k] for k in THREAD_KEYS},power_scheme=power,timer=dict(name='perf_counter_ns',monotonic=timer.monotonic,adjustable=timer.adjustable,resolution_seconds=timer.resolution),GPU_used=False,benchmark_processes_concurrent=1,CPU_affinity='OS default;no pinning/no priority change')
