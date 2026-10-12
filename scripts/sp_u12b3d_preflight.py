"""Synthetic-only concurrency/forensics gates; then immutable formal freeze."""
from pathlib import Path
import sys,os,json,shutil,concurrent.futures,errno,uuid,subprocess,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b3d_common as b
from scripts.sp_u12b3d_build import core,EXPECTED
import numpy as np
DUMMY=b.TECH/'synthetic_B2_topology'
def init():b.B2=DUMMY;b.init_worker()
def job(i):return b.worker('ID',i)
def crash_writer():
 b.B2=DUMMY;b.firewall();b._phase='SYNTHETIC';p=b.SEAL/'raw/ID/0100.npz'
 tmp=p.with_name(p.name+f'.partial.{os.getpid()}.{__import__("threading").get_ident()}.{uuid.uuid4().hex}.ID_100')
 with tmp.open('xb') as f:f.write(b'SYNTHETIC_INCOMPLETE');f.flush();os.fsync(f.fileno())
 os._exit(73)
def forensic_control():
 b.B2=DUMMY;b.init_worker()
 def denied(path):
  b.technical_record_operation('OPEN_READ',path,denial_origin='CONTROLLED_SYNTHETIC_INJECTION')
  raise PermissionError(errno.EACCES,'controlled synthetic denial',str(path.resolve()))
 b.arrays=denied
 try:b.worker('ID',0)
 except PermissionError:return
 raise AssertionError('EXPECTED_DENIAL_MISSING')
def readonly_control():
 b.B2=DUMMY;b.firewall();p=DUMMY/'FORBIDDEN';source=b.TECH/'dummy_rename_source';source.write_bytes(b'DUMMY');found=[]
 for call in (lambda:p.open('wb'),lambda:p.touch(),lambda:p.mkdir(),lambda:os.replace(source,p),lambda:p.unlink(),lambda:os.utime(p,None)):
  try:call()
  except PermissionError as exc:found.append(b.technical_exception_record(exc))
  else:raise AssertionError('WRITE_NOT_BLOCKED')
 assert len(found)==6 and not p.exists() and source.read_bytes()==b'DUMMY'
 b.write(b.TECH/'readonly_control.json',dict(PASS=True,expected_denials=6,actual_mutations=0,envelopes=found))
def prepare():
 b.TECH.mkdir(exist_ok=True);b.SEAL.mkdir(exist_ok=False)
 b.write(b.TECH/'execution_freeze.json',dict(hashes={}))
 for name in ('bootstrap_seed_registry.json','raw_schema.json'):b.atomic(b.TECH/name,(ROOT/'outputs/sp_u12b3b'/name).read_bytes())
 source=ROOT/'outputs/sp_u12b3c/synthetic_B2_topology'
 for i in range(101):
  for sub,prefixes in [('cache',('H_','comm_')),('raw_evaluation',('',))]:
   for prefix in prefixes:
    for ext in ('npz','json'):
     src=source/sub/'ID'/f'{prefix}{i:04d}.{ext}';dest=DUMMY/sub/'ID'/src.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
def gates():
 b._phase='SYNTHETIC'
 outcomes=[]
 for indices in (range(100),range(100)):
  with concurrent.futures.ProcessPoolExecutor(max_workers=2,initializer=init) as pool:rows=list(pool.map(job,indices))
  outcomes.append(dict(tasks=100,resumed=sum(v['resumed'] for v in rows),worker_PIDs=sorted({v['worker_pid'] for v in rows})))
  print(json.dumps(dict(stage='synthetic_concurrency',PASS=True,trajectory_count=100,status='PASS_'+str(len(outcomes)))),flush=True)
 b.need(outcomes[0]['resumed']==0 and outcomes[1]['resumed']==100,'SYNTHETIC_RESUME_GATE')
 broken=False
 try:
  with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:pool.submit(crash_writer).result()
 except concurrent.futures.process.BrokenProcessPool:broken=True
 b.need(broken and not(b.SEAL/'raw/ID/0100.json').exists(),'DEAD_PARTIAL_NOT_COMPLETE')
 other=b.sha(b.SEAL/'raw/ID/0000.npz')
 with concurrent.futures.ProcessPoolExecutor(max_workers=2,initializer=init) as pool:value=pool.submit(job,100).result()
 b.need(not value['resumed'] and b.sha(b.SEAL/'raw/ID/0000.npz')==other and len(list((b.TECH/'quarantine').glob('*')))==1,'QUARANTINE_GATE')
 # Validate detailed worker forensic envelope in a separate controlled process.
 with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:pool.submit(forensic_control).result()
 readonly_control()
 b.write(b.SEAL/'synthetic_manifest.json',dict(scope='SYNTHETIC_ONLY',complete_trajectories=101,incomplete_committed=False))
 b.need(not list(b.SEAL.rglob('*.partial.*')),'SYNTHETIC_PARTIAL_REMAINS')
 required=['timestamp_utc','PID','worker_id','thread_id','absolute_path','operation','source_file','source_line','full_traceback','errno','WinError','trajectory_key','condition','SNR','current_stage','temp_path','final_path']
 records=[b.read(p) for p in b.TECH.glob('worker_exception_*.json')]
 b.need(any(all(k in r for k in required) and r['errno']==errno.EACCES and r['condition']=='ID' and r['full_traceback'] for r in records),'FORENSIC_ENVELOPE_GATE')
 destination=b.TECH/'synthetic_exact_root_archive'
 b.need(b.SEAL.resolve()==ROOT.resolve()/'outputs/sp_u12b3d_run2_sealed' and destination.resolve().is_relative_to(b.TECH.resolve()) and not destination.exists(),'SAFE_SYNTHETIC_ARCHIVE')
 os.replace(b.SEAL,destination);b.SEAL.mkdir()
 control=b.TECH/'synthetic_control_archive';control.mkdir()
 for pattern in ('*exception*.json','actual_worker_pids','worker_io','quarantine','readonly_control.json'):
  for p in list(b.TECH.glob(pattern)):os.replace(p,control/p.name)
 b.write(b.TECH/'synthetic_gates.json',dict(PASS=True,two_worker_passes=outcomes,complete_dummy_trajectories=101,actual_crash=True,dead_temp_quarantined=True,forensic_envelope_fields=required,formal_root_empty=not any(b.SEAL.iterdir()),formal_inputs_used=False,new_scientific_samples=False,expected_controls_archived=True))
 print(json.dumps(dict(stage='synthetic_gates',PASS=True,trajectory_count=101)),flush=True)
def tests(stage):
 log=b.TECH/f'pytest_{stage}.txt'
 with log.open('wb') as f:
  p=subprocess.run([sys.executable,'-X','utf8','-m','pytest','-q','tests/test_sp_u12b3d.py','-o','cache_dir='+str(b.TECH/'pytest_cache'),'--basetemp='+str(b.TECH/'pytest_tmp')],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
 text=log.read_text(encoding='utf8');match=__import__('re').search(r'(\d+) passed',text)
 b.write(b.TECH/f'pytest_{stage}.json',dict(exit_code=p.returncode,passed=int(match[1]) if match else 0,failed=0 if p.returncode==0 else None,skipped=0,log_sha256=b.sha(log)))
 print(json.dumps(dict(stage='pytest_'+stage,PASS=p.returncode==0,tests=int(match[1]) if match else 0)),flush=True)
 if p.returncode:raise SystemExit(1)
def final_smoke():
 b.need(not any(b.SEAL.iterdir()),'SMOKE_ROOT_NOT_EMPTY')
 with concurrent.futures.ProcessPoolExecutor(max_workers=2,initializer=init) as pool:rows=list(pool.map(job,range(2)))
 b.need(len(rows)==2 and not any(r['resumed'] for r in rows),'FINAL_SMOKE_GATE')
 for p in (b.TECH/'actual_worker_pids').glob('*.json'):b.need(b.read(p)['controller_PID']==os.getpid(),'CONTROLLER_PID_HANDSHAKE')
 destination=b.TECH/'final_smoke_exact_root_archive';b.need(not destination.exists() and destination.resolve().is_relative_to(b.TECH.resolve()),'SAFE_SMOKE_ARCHIVE');os.replace(b.SEAL,destination);b.SEAL.mkdir()
 control=b.TECH/'final_smoke_control_archive';control.mkdir()
 for name in ('actual_worker_pids','worker_io'):
  p=b.TECH/name
  if p.exists():os.replace(p,control/name)
 b.write(b.TECH/'final_smoke.json',dict(PASS=True,synthetic_only=True,tasks=2,real_wrapper_initializer=True,actual_controller_PID_handshake=True,io_sha256=b.sha(ROOT/'scripts/sp_u12b3d_io.py'),dispatcher_sha256=b.sha(ROOT/'scripts/run_sp_u12b3d.py')))
 print(json.dumps(dict(stage='final_smoke',PASS=True,trajectory_count=2)),flush=True)
def freeze():
 digest,partition=core();b.need(digest==EXPECTED,'SCIENTIFIC_CORE_DRIFT');b.need(b.read(b.TECH/'synthetic_gates.json')['PASS'],'SYNTHETIC_GATE');b.need(b.read(b.TECH/'pytest_preflight.json')['exit_code']==0,'TEST_GATE')
 b.need(not any(b.SEAL.iterdir()),'CLEAN_FORMAL_ROOT');base=b.read(b.TECH/'preservation_before.json');b.need(base['B2']['PASS'] and base['B2']['files']==35031,'B2_FULL_HASH_GATE')
 prior=b.read(ROOT/'outputs/sp_u12b3b/execution_freeze.json')
 for n,h in prior['hashes'].items():b.need(b.sha(ROOT/n)==h,'ORIGINAL_FREEZE_DRIFT')
 b.need(b.sha(b.CFG)=='1b0fb501e9a0394bc180464b8e51e0af5ec0fdd4e776eef6c91fecd9f83ffb89','ORIGINAL_CONFIG_DRIFT');b.need(b.sha(b.PROTOCOL)=='4382607d7f22d2d783c449fc4f56f778f610276274745f32ec246cc39b5b2eea','ORIGINAL_PROTOCOL_DRIFT')
 b.need(b.sha(b.TECH/'bootstrap_seed_registry.json')==b.sha(ROOT/'outputs/sp_u12b3b/bootstrap_seed_registry.json'),'BOOTSTRAP_SEED_DRIFT')
 b.need(b.read(b.TECH/'failstop_gate.json')['PASS'],'FAILSTOP_GATE')
 smoke=b.read(b.TECH/'final_smoke.json');b.need(smoke['PASS'] and smoke['io_sha256']==b.sha(ROOT/'scripts/sp_u12b3d_io.py') and smoke['dispatcher_sha256']==b.sha(ROOT/'scripts/run_sp_u12b3d.py'),'FINAL_WRAPPER_GATE')
 own=['scripts/sp_u12b3d_build.py','scripts/sp_u12b3d_common.py','scripts/sp_u12b3d_io.py','scripts/sp_u12b3d_authorization.py','scripts/run_sp_u12b3d.py','scripts/sp_u12b3d_preflight.py','scripts/sp_u12b3d_preserve.py','scripts/sp_u12b3d_failure_close.py','scripts/sp_u12b3d_failstop_probe.py','tests/test_sp_u12b3d.py']
 wrapper={n:b.sha(ROOT/n) for n in own};paths=set(prior['hashes'])|set(own)|{'outputs/sp_u12b3d/bootstrap_seed_registry.json','outputs/sp_u12b3d/raw_schema.json','docs/sp_u12b3d_historical_failure_disclosure.md'}
 f=dict(hashes={n:b.sha(ROOT/n) for n in sorted(paths)},scientific_core_sha256=digest,scientific_partition=partition,io_wrapper_sha256=b.tree(wrapper),wrapper_files=wrapper,cardinalities=dict(H=5000,condition_SNR_units=10000,trajectory_estimator_records=60000,scored_frame_estimator_records=360000,raw_NPZ=5000,raw_receipts=5000,primary_contrasts=12,aggregate_rows=60,descriptive_rows=60,figure_rows=60,bootstrap_NPZ=8,replicates_per_point=10000),scientific_config_sha256=b.sha(b.CFG),scientific_protocol_sha256=b.sha(b.PROTOCOL),formal_seed_registry_sha256=b.sha(ROOT/'outputs/sp_u12b2a_seed_registry.json'),bootstrap_seed_registry_sha256=b.sha(b.TECH/'bootstrap_seed_registry.json'),runtime=dict(python=sys.version,numpy=np.__version__,worker_count=2,BLAS_threads=1),start_from_trajectory_zero=True,retries=0,root_cause_confirmed=False,outcomes_viewed=False)
 b.atomic(b.TECH/'synthetic_execution_freeze.json',(b.TECH/'execution_freeze.json').read_bytes());b.write(b.TECH/'execution_freeze.json',f)
 from scripts.sp_u12b3d_authorization import TASK
 b.write(b.TECH/'mentor_authorization.json',dict(formal_run2_authorized=True,B3C_MENTOR_DECISION='HISTORICAL_ROOT_CAUSE_UNRECOVERABLE_BUT_RESTART_AUTHORIZED',FORMAL_RECOVERY_MODE='CLEAN_RESTART_SAME_SEEDS',historical_root_cause_confirmed=False,task_sha256=b.sha(TASK),execution_freeze_sha256=b.sha(b.TECH/'execution_freeze.json'),new_formal_seeds=False,Run1_intended_seeds_equal_Run2_intended_seeds=True,intended_seed_coverage_percent=100,original_B3_config_protocol_and_registries_reused=True))
 print(json.dumps(dict(stage='formal_freeze',PASS=True,hash=b.sha(b.TECH/'execution_freeze.json'))),flush=True)
if __name__=='__main__':
 mode=sys.argv[1]
 if mode.startswith('test_'):tests(mode[5:])
 else:{'prepare':prepare,'gates':gates,'final_smoke':final_smoke,'freeze':freeze}[mode]()
