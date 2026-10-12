"""Actual synthetic failed future exercises the exact Run2 dispatcher, no science."""
from pathlib import Path
import sys,os,json,time,errno,multiprocessing
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b3d_common as b
PROBE=ROOT/'outputs/sp_u12b3d/failstop_probe'
def init():
 b.TECH=PROBE;parent=multiprocessing.parent_process()
 b.write(PROBE/'actual_worker_pids'/f'{os.getpid()}.json',dict(PID=os.getpid(),parent_PID=os.getppid(),controller_PID=parent.pid))
def fatal_job(condition,i):
 b.technical_context(condition=condition,trajectory_id=i,SNR=[10,20]);b.write(PROBE/f'started_{condition}_{i}.json',dict(condition=condition,trajectory_id=i))
 if i==0:
  path=PROBE/'CONTROLLED_DENIAL';b.technical_record_operation('OPEN_READ',path)
  try:raise PermissionError(errno.EACCES,'synthetic fail-stop injection',str(path))
  except Exception as exc:b.write(PROBE/f'worker_exception_{os.getpid()}.json',b.technical_exception_record(exc));raise
 time.sleep(10)
 b.write(PROBE/f'SHOULD_NOT_COMPLETE_{i}.json',dict(unexpected=True))
 return dict(condition=condition,trajectory_id=i,resumed=False,worker_pid=os.getpid(),peak={})
def main():
 from scripts import run_sp_u12b3d as runner
 b.TECH=PROBE;b.CONDITIONS=('SYNTHETIC_FIRST','FORBIDDEN_SECOND');b.init_worker=init;b.worker=fatal_job
 tick=time.monotonic();caught=False
 try:runner.evaluation()
 except PermissionError:caught=True
 elapsed=time.monotonic()-tick
 started=list(PROBE.glob('started_*.json'));ended=list(PROBE.glob('SHOULD_NOT_COMPLETE*'));exceptions=list(PROBE.glob('worker_exception*'))
 assert caught and (PROBE/'PERMANENT_STOP.json').exists() and len(started)<=2 and not ended and not list(PROBE.glob('*FORBIDDEN_SECOND*')) and exceptions
 assert all(b.read(p)['controller_PID']==os.getpid() for p in (PROBE/'actual_worker_pids').glob('*.json'))
 b.write(ROOT/'outputs/sp_u12b3d/failstop_gate.json',dict(PASS=True,synthetic_only=True,exact_dispatcher=True,first_exception_stops=True,actual_worker_termination=True,pending_limit=2,started_tasks=len(started),later_cell_started=False,retries=0,elapsed_seconds=elapsed))
 print(json.dumps(dict(stage='failstop_gate',PASS=True,trajectory_count=len(started))),flush=True)
if __name__=='__main__':main()
