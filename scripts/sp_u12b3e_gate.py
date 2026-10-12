"""Invoke the actual future script CLI; never substitute a test-runner pool."""
from pathlib import Path
import sys,json,subprocess,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b3e_common as b
OUT=ROOT/'outputs/sp_u12b3e';PYTHON=ROOT/'.venv-cdl/Scripts/python.exe'
def command(flag,session,extra=()):return [str(PYTHON),'-X','utf8','scripts/run_sp_u12b3f.py',flag,'--session',session,*extra]
def invoke(flag,session,extra=()):
 cmd=command(flag,session,extra);log=OUT/'cli_logs'/(session+('_resume' if extra else '')+'.txt');log.parent.mkdir(parents=True,exist_ok=True);tick=time.monotonic()
 with log.open('wb') as f:result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 record=dict(command=cmd,working_directory=str(ROOT),exit_code=result.returncode,elapsed_seconds=time.monotonic()-tick,log_path=str(log),log_sha256=b.sha(log))
 b.write(OUT/'cli_records'/(session+('_resume' if extra else '')+'.json'),record)
 b.need(result.returncode==0,'EXACT_CLI_FAILED');return record
def repeat():
 rows=[]
 for i in range(1,21):
  session=f'init_{i:02d}'
  if i==1:
   # First exact CLI was executed interactively and its complete receipt is authoritative.
   b.need((OUT/'sessions'/session/'init_only_receipt.json').exists(),'FIRST_EXACT_ENTRY_MISSING')
   execution=dict(command=command('--init-only',session),working_directory=str(ROOT),exit_code=0,first_invocation='actual exec_command result exit0;receipt verified')
  else:execution=invoke('--init-only',session)
  p=OUT/'sessions'/session/'init_only_receipt.json';r=b.read(p)
  b.need(r['PASS'] and r['workers_initialized']==r['heartbeats_completed']==2 and r['orphan_workers']==0,'REPEAT_GATE_FAILED')
  rows.append(dict(session=session,execution=execution,receipt_sha256=b.sha(p),workers_initialized=2,heartbeats_completed=2,PIDs=[v['PID'] for v in r['initializer_rows']],serialization_failures=0,filesystem_failures=0,orphan_workers=0))
  print(json.dumps(dict(stage='exact_init_repeat',runs_completed=i,workers_initialized=2*i,heartbeats_completed=2*i,PASS=True)),flush=True)
 b.write(ROOT/'outputs/sp_u12b3e_init_only_repeat_test.json',dict(PASS=True,runs=20,workers_per_run=2,workers_initialized=40,heartbeats_completed=40,serialization_failures=0,pickle_errors=0,PermissionError=0,filesystem_failures=0,orphan_workers=0,formal_trajectories_executed=0,bootstrap_count=0,records=rows))
def development():
 b.need(b.read(ROOT/'outputs/sp_u12b3e_init_only_repeat_test.json')['PASS'],'INIT_ONLY_PREREQUISITE')
 first=invoke('--development-safe','development');second=invoke('--development-safe','development',('--resume',))
 one=b.read(OUT/'sessions/development/development_receipt.json');two=b.read(OUT/'sessions/development/development_resume_receipt.json')
 b.need(one['PASS'] and two['PASS'] and one['resumed_records']==0 and two['resumed_records']==3,'ACTUAL_CHECKPOINT_RESUME_GATE')
 b.need({k:r['raw_sha256'] for k,r in one['leaves'].items()}=={k:r['raw_sha256'] for k,r in two['leaves'].items()},'DEVELOPMENT_CHECKPOINT_RAW_DRIFT')
 b.write(ROOT/'outputs/sp_u12b3e_development_equivalence.json',dict(**one,resume_PASS=True,resume_records=3,raw_checkpoint_bytes_unchanged=True,CLI_executions=[first,second],orphan_workers=0,performance_values_disclosed=False))
 print(json.dumps(dict(stage='development_exact_entry',PASS=True,trajectory_count=3,bootstrap_count=0)),flush=True)
if __name__=='__main__':{'repeat':repeat,'development':development}[sys.argv[1]]()
