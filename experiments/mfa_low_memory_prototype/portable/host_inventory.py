"""Read-only hardware inventory; portable Windows/Linux, no remote access."""
from pathlib import Path
import os,sys,platform,ctypes,struct,shutil,json
def topology():
 if os.name!='nt':
  try:
   import psutil
   return {'physical_cores':psutil.cpu_count(logical=False),'logical_threads':psutil.cpu_count(logical=True),'method':'psutil','hybrid_core_details':'UNKNOWN'}
  except ImportError:return {'physical_cores':None,'logical_threads':os.cpu_count(),'method':'os.cpu_count; physical UNKNOWN'}
 fn=ctypes.windll.kernel32.GetLogicalProcessorInformationEx
 fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)];fn.restype=ctypes.c_int
 length=ctypes.c_ulong();fn(0,None,ctypes.byref(length));buf=ctypes.create_string_buffer(length.value)
 if not fn(0,buf,ctypes.byref(length)):raise ctypes.WinError()
 data=buf.raw[:length.value];offset=0;cores=[]
 while offset<len(data):
  relationship,size=struct.unpack_from('<II',data,offset);assert size>=32
  if relationship==0:
   flags,eff=struct.unpack_from('<BB',data,offset+8);groups=struct.unpack_from('<H',data,offset+30)[0]
   masks=[struct.unpack_from('<QH',data,offset+32+16*i) for i in range(groups)]
   cores.append({'efficiency_class':eff,'SMT':bool(flags&1),'groups':[{'mask':m,'group':g,'logical_threads':m.bit_count()} for m,g in masks]})
  offset+=size
 return {'physical_cores':len(cores),'logical_threads':sum(v['logical_threads'] for c in cores for v in c['groups']),
 'method':'Windows GetLogicalProcessorInformationEx(RelationProcessorCore)','cores':cores,'efficiency_class_counts':{str(k):sum(c['efficiency_class']==k for c in cores) for k in sorted({c['efficiency_class'] for c in cores})}}
def frequencies():
 if os.name!='nt':
  try:
   import psutil
   return {'reported_cpu_freq':None if psutil.cpu_freq() is None else psutil.cpu_freq()._asdict(),'thermal':'NOT_AVAILABLE','sustained_frequency_verified':False}
  except Exception as e:return {'status':'UNAVAILABLE','reason':repr(e)}
 class CPU(ctypes.Structure):_fields_=[(n,ctypes.c_uint32) for n in ('Number','MaxMhz','CurrentMhz','MhzLimit','MaxIdleState','CurrentIdleState')]
 array=(CPU*(os.cpu_count() or 1))();fn=ctypes.windll.powrprof.CallNtPowerInformation
 fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_ulong,ctypes.c_void_p,ctypes.c_ulong];fn.restype=ctypes.c_ulong
 status=fn(11,None,0,ctypes.byref(array),ctypes.sizeof(array))
 return {'method':'CallNtPowerInformation ProcessorInformation','status_code':status,'logical_CPU_reports':[{'index':v.Number,'reported_max_MHz':v.MaxMhz,'reported_current_MHz':v.CurrentMhz,'reported_limit_MHz':v.MhzLimit} for v in array] if status==0 else [],'thermal_sensor':'NOT_AVAILABLE','actual_sustained_clock_or_throttle_state':'NOT_VERIFIED; OS-reported MHz is not a hardware clock/thermal measurement'}
def process_memory(pid=None):
 if os.name!='nt':
  import psutil
  v=psutil.Process(pid).memory_info();return {'working_set_bytes':v.rss,'peak_working_set_bytes':v.rss,'private_bytes':getattr(v,'data',v.vms),'peak_commit_bytes':getattr(v,'data',v.vms),'peak_status':'current RSS only on non-Windows'}
 class MEM(ctypes.Structure):_fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
 x=MEM();x.cb=ctypes.sizeof(x);fn=ctypes.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(MEM),ctypes.c_ulong];fn.restype=ctypes.c_int
 if pid is None:
  current=ctypes.windll.kernel32.GetCurrentProcess;current.restype=ctypes.c_void_p;handle=current();owned=False
 else:
  op=ctypes.windll.kernel32.OpenProcess;op.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];op.restype=ctypes.c_void_p;handle=op(0x0411,False,pid);owned=True
  if not handle:raise ctypes.WinError()
 try:
  if not fn(handle,ctypes.byref(x),x.cb):raise ctypes.WinError()
 finally:
  if owned:
   close=ctypes.windll.kernel32.CloseHandle;close.argtypes=[ctypes.c_void_p];close.restype=ctypes.c_int;close(handle)
 return {'working_set_bytes':int(x.WorkingSetSize),'peak_working_set_bytes':int(x.PeakWorkingSetSize),'private_bytes':int(x.PrivateUsage),'peak_commit_bytes':int(x.PeakPagefileUsage)}
def memory():
 if os.name!='nt':
  import psutil
  m=psutil.virtual_memory();s=psutil.swap_memory()
  return {'total_physical_bytes':m.total,'available_physical_bytes':m.available,'commit_limit_bytes':m.total+s.total,'commit_available_bytes':m.available+s.free,'commit_semantics':'Linux RAM+swap approximation; not Windows commit accounting'}
 class MEM(ctypes.Structure):_fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total','available','totalpagefile','availablepagefile','totalvirtual','availablevirtual','availableextended')]
 x=MEM();x.length=ctypes.sizeof(x);fn=ctypes.windll.kernel32.GlobalMemoryStatusEx;fn.argtypes=[ctypes.POINTER(MEM)];fn.restype=ctypes.c_int
 if not fn(ctypes.byref(x)):raise ctypes.WinError()
 class PI(ctypes.Structure):_fields_=[('cb',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[(n,ctypes.c_ulong) for n in ('HandleCount','ProcessCount','ThreadCount')]
 pi=PI();pi.cb=ctypes.sizeof(pi);f=ctypes.windll.psapi.GetPerformanceInfo;f.argtypes=[ctypes.POINTER(PI),ctypes.c_ulong];f.restype=ctypes.c_int
 if not f(ctypes.byref(pi),pi.cb):raise ctypes.WinError()
 return {'total_physical_bytes':int(x.total),'available_physical_bytes':int(x.available),'commit_limit_bytes':int(pi.CommitLimit*pi.PageSize),'commit_used_bytes':int(pi.CommitTotal*pi.PageSize),'commit_available_bytes':int((pi.CommitLimit-pi.CommitTotal)*pi.PageSize)}
def inventory(with_software=True):
 model=platform.processor();regMHz=None
 if os.name=='nt':
  import winreg
  with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
   model=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
   try:regMHz=winreg.QueryValueEx(key,'~MHz')[0]
   except OSError:pass
 answer={'CPU_exact_model':model,'topology':topology(),'frequency_reports':frequencies(),'registry_MHz':regMHz,
 'base_frequency':'UNKNOWN unless source verified; registry MHz is not a base-frequency specification','max_boost_specification':'UNKNOWN unless source verified',
 'memory':memory(),'OS':platform.platform(),'Python_executable':sys.executable,'Python_version':sys.version,'free_disk_bytes':shutil.disk_usage(Path.cwd()).free,
 'environment_threads':{n:os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},'laboratory_host_tested':False,'scope':'THIS_ACTUAL_EXECUTION_HOST_ONLY'}
 if with_software:
  import numpy,scipy,sklearn
  from threadpoolctl import threadpool_info
  answer['software']={'NumPy':numpy.__version__,'SciPy':scipy.__version__,'scikit_learn':sklearn.__version__,'actual_threadpools':threadpool_info()}
 return answer
if __name__=='__main__':
 result=inventory();print(json.dumps(result,indent=2,ensure_ascii=False))
