"""Monitor the actual interpreter PID, not a Windows venv launcher PID."""
from scripts.sp_u12b5b_common import *
import shutil

def run_worker(kind,name,calibration=False,budget_seconds=None):
 cfg=read(CONFIG);limits=cfg['resource_contract'];handshake=OUT/f'worker_{name}_{kind}_PID.json'
 assert not handshake.exists(),'WORKER_RESTART_FORBIDDEN'
 logfile=OUT/f'worker_{name}_{kind}.log';tick=time.monotonic();samples=0;handle=None;pid=None;reason=None;observed_peak=0;commit_peak=0;min_available=None
 with logfile.open('x',encoding='utf8') as log:
  proc=subprocess.Popen([sys.executable,'-X','utf8','-m','scripts.sp_u12b5b_worker',kind,name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  try:
   while proc.poll() is None:
    if pid is None and handshake.exists():
     try:info=read(handshake)
     except json.JSONDecodeError:info=None
     if info is not None:
      pid=info['actual_PID'];handle=process_handle(pid)
      write(OUT/f'worker_{name}_{kind}_MONITOR_READY.json',dict(actual_PID=pid,attached=True,created_UTC=utc()))
    if handle is not None:
     try:mem=process_memory(handle)
     except OSError:
      if proc.poll() is not None:break
      time.sleep(.05);continue
     system=system_memory();samples+=1;observed_peak=max(observed_peak,mem['peak_working_set_bytes']);commit_peak=max(commit_peak,mem['peak_commit_bytes'])
     available=system['available_physical_bytes'];min_available=available if min_available is None else min(min_available,available)
     with (OUT/f'resource_{name}_{kind}.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(elapsed_seconds=time.monotonic()-tick,actual_PID=pid,launcher_PID=proc.pid,process=mem,system=system))+'\n')
     if not calibration:
      if mem['peak_working_set_bytes']>=limits['max_process_bytes']:reason='PROCESS_RSS_CAP'
      elif available<limits['min_system_available_physical_bytes']:reason='SYSTEM_PHYSICAL_HEADROOM'
      elif time.monotonic()-tick>=(limits['max_candidate_seconds'] if budget_seconds is None else budget_seconds):reason='CANDIDATE_TIME_CAP'
    if pid is None and time.monotonic()-tick>120:reason='WORKER_PID_HANDSHAKE_TIMEOUT'
    if reason:
     if pid is None:proc.terminate()
     else:
      fn=ctypes.windll.kernel32.OpenProcess;fn.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];fn.restype=ctypes.c_void_p
      termination=fn(1,False,pid);assert termination,'CANNOT_OPEN_OWN_WORKER_FOR_STOP'
      stop=ctypes.windll.kernel32.TerminateProcess;stop.argtypes=[ctypes.c_void_p,ctypes.c_uint];stop.restype=ctypes.c_int
      assert stop(termination,77),'CANNOT_STOP_OWN_WORKER';close_handle(termination)
     break
    time.sleep(.25)
   try:code=proc.wait(timeout=30)
   except subprocess.TimeoutExpired:proc.terminate();code=proc.wait(timeout=10)
  finally:
   if handle is not None:close_handle(handle)
 result=dict(kind=kind,candidate=name,completed=code==0 and reason is None,exit_code=code,stop_reason=reason,actual_interpreter_PID=pid,launcher_PID=proc.pid,actual_PID_handshake_used=pid is not None,wall_seconds=time.monotonic()-tick,peak_RSS_bytes=observed_peak,peak_commit_bytes=commit_peak,min_system_available_physical_bytes=min_available,monitor_samples=samples,resource_metrics_used_for_selection=False,calibration_only=calibration)
 write(OUT/f'monitor_{name}_{kind}.json',result)
 print(json.dumps(dict(stage='worker',candidate=name,kind=kind,completed=result['completed'],stop_reason=reason,peak_RSS_bytes=observed_peak)),flush=True)
 return result
