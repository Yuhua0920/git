"""Monitor actual worker PID and enforce unchanged resource thresholds."""
from scripts.sp_u12b5b_r1_common import *
def run_probe():
 cfg=read(CONFIG);name=cfg['structures'][0]['name'];directory=OUT/name;directory.mkdir(exist_ok=True)
 assert not (OUT/'PROBE_STARTED.json').exists(),'PROBE_RESTART_FORBIDDEN'
 write(OUT/'PROBE_STARTED.json',dict(UTC=utc(),role='RESOURCE_ONLY',structures=cfg['structures'],seeds=cfg['seeds']))
 pid=None;handle=None;reason=None;peak=commit=0;minimum=None;samples=0;tick=time.monotonic()
 with (directory/'stdout.log').open('x',encoding='utf8') as log:
  proc=subprocess.Popen([sys.executable,'-X','utf8','-m','scripts.sp_u12b5b_r1_worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  try:
   while proc.poll() is None:
    if pid is None and (directory/'PID.json').exists():
     try:pid=read(directory/'PID.json')['actual_PID']
     except json.JSONDecodeError:pid=None
     if pid is not None:
      handle=process_handle(pid);write(directory/'MONITOR_READY.json',dict(actual_PID=pid,UTC=utc(),attached=True))
    if handle:
     mem=process_memory(handle);host=host_snapshot();samples+=1;peak=max(peak,mem['peak_working_set_bytes']);commit=max(commit,mem['peak_commit_bytes']);minimum=host['available_physical_bytes'] if minimum is None else min(minimum,host['available_physical_bytes'])
     with (directory/'resource.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(elapsed_seconds=time.monotonic()-tick,actual_PID=pid,launcher_PID=proc.pid,process=mem,host=host))+'\n')
     reason=threshold_reason(mem,host,cfg['limits'])
    if pid is None and time.monotonic()-tick>120:reason='PID_HANDSHAKE_TIMEOUT'
    if time.monotonic()-tick>1800:reason='RESOURCE_PROBE_TECHNICAL_TIME_CAP'
    if reason:
     if pid is None:proc.terminate()
     else:
      fn=ctypes.windll.kernel32.OpenProcess;fn.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];fn.restype=ctypes.c_void_p;termination=fn(1,False,pid);assert termination
      stop=ctypes.windll.kernel32.TerminateProcess;stop.argtypes=[ctypes.c_void_p,ctypes.c_uint];stop.restype=ctypes.c_int
      assert stop(termination,77);close_handle(termination)
     break
    time.sleep(.1)
   code=proc.wait(timeout=30)
  finally:
   if handle:close_handle(handle)
 stages=[json.loads(v) for v in (directory/'stages.jsonl').read_text(encoding='utf8').splitlines()] if (directory/'stages.jsonl').exists() else []
 result=dict(role='RESOURCE_ONLY',K=cfg['structures'][0]['K'],rank=cfg['structures'][0]['rank'],seeds=cfg['seeds'],shape=[cfg['N'],cfg['D']],dtype=cfg['dtype'],actual_PID=pid,launcher_PID=proc.pid,actual_PID_monitored=pid is not None,completed=code==0 and reason is None,exit_code=code,stop_reason=reason,peak_RSS_bytes=peak,peak_commit_bytes=commit,min_available_physical_bytes=minimum,monitor_samples=samples,wall_seconds=time.monotonic()-tick,last_entered_stage=stages[-1]['stage'] if stages else 'UNKNOWN',last_stage_process=stages[-1]['process'] if stages else None,last_stage_arrays_metadata=stages[-1]['arrays_metadata_only'] if stages else [],parameters_exported=False,objectives_exported=False,preparation_data_used=False,formal_data_used=False,validation=False,selection=False,formal_candidate_completed=False)
 write(ROOT/'outputs/sp_u12b5b_r1_resource_probe_results.json',dict(executed=True,records=[result],no_further_probes=True,all_scientific_uses_prohibited=True))
 print(json.dumps(dict(stage='R1_probe_closed',K=result['K'],rank=result['rank'],peak_RSS_bytes=peak,peak_commit_bytes=commit,min_available_physical_bytes=minimum,stop_reason=reason,last_stage=result['last_entered_stage'])),flush=True)
