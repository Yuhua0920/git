"""Single future formal CLI with exact-entrypoint technical validation modes."""
from pathlib import Path
import sys,os,json,time,argparse,multiprocessing,concurrent.futures,ctypes
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b3e_common as b
from scripts.sp_u12b3e_build import core,EXPECTED
from scripts import sp_u12b3g_schema as contract
import numpy as np
_run_config=None
b.TECH=contract.CONTROL
def create_pool(mode,session):
 config=dict(_run_config) if mode=='formal' and _run_config is not None else dict(mode=mode,session=session)
 return concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=b.init_worker,initargs=(config,))
def setup_session(mode,session,resume=False):
 b.need(__import__('re').fullmatch(r'[A-Za-z0-9_-]+',session) is not None,'INVALID_SESSION')
 b.TECH=ROOT/'outputs/sp_u12b3e/sessions'/session
 b.SEAL=ROOT/'outputs/sp_u12b3e/development_runs'/session/'sealed' if mode=='development-safe' else ROOT/'outputs/sp_u12b3f_run3_sealed'
 b.B2=ROOT/'outputs/sp_u12b3e/development_inputs' if mode=='development-safe' else ROOT/'outputs/sp_u12b2b_sealed'
 if not resume:b.need(not b.TECH.exists(),'SESSION_ALREADY_EXISTS')
 b.TECH.mkdir(parents=True,exist_ok=True)
 if not resume:
  b.atomic(b.TECH/'execution_freeze.json',(ROOT/'outputs/sp_u12b3e/candidate_validation_freeze.json').read_bytes())
 b.firewall();b.check_freeze()
def process_alive(pid):
 k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.restype=ctypes.c_void_p;k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];k.CloseHandle.argtypes=[ctypes.c_void_p];k.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
 h=k.OpenProcess(0x1000,False,pid)
 if not h:
  b.need(ctypes.get_last_error()==87,'WORKER_EXIT_STATE_UNKNOWN');return False
 code=ctypes.c_ulong();ok=k.GetExitCodeProcess(h,ctypes.byref(code));k.CloseHandle(h);b.need(ok,'WORKER_EXIT_QUERY_FAILED');return code.value==259
def launch_heartbeats(pool):
 futures=[pool.submit(b.worker,'__HEARTBEAT__',i) for i in range(2)]
 return [f.result(timeout=60) for f in futures]
def startup_receipt(heartbeats,require_exit=True):
 rows=[b.read(p) for p in sorted((b.TECH/'initializers').glob('*.json'))]
 pids={r['PID'] for r in rows};b.need(len(rows)==len(pids)==2 and {r['PID'] for r in heartbeats}==pids,'EXACT_TWO_WORKERS')
 b.need(all(r['initializer_entered'] and r['initializer_completed'] and r['B2_readonly_guard_installed'] and r['technical_writer_initialized'] and r['start_method']=='spawn' and all(v=='1' for v in r['BLAS_threads'].values()) for r in rows),'INITIALIZER_CONTRACT')
 if require_exit:b.need(all(not process_alive(pid) for pid in pids),'ORPHAN_WORKER')
 return dict(PASS=True,workers_requested=2,workers_initialized=2,heartbeats_completed=2,initializer_rows=rows,heartbeat_rows=heartbeats,Windows_spawn=True,orphan_workers=0,serialization_failures=0,filesystem_failures=0,formal_trajectories_executed=0,bootstrap_count=0,sys_version=sys.version,sys_executable=sys.executable,working_directory=str(ROOT),entrypoint=str(Path(__file__).resolve()),pool_factory='scripts.run_sp_u12b3f.create_pool',initializer_qualified_name=b.init_worker.__module__+'.'+b.init_worker.__qualname__,worker_qualified_name=b.worker.__module__+'.'+b.worker.__qualname__)
def init_only(session):
 setup_session('init-only',session)
 with create_pool('init-only',session) as pool:heartbeats=launch_heartbeats(pool)
 report=startup_receipt(heartbeats);b.write(b.TECH/'init_only_receipt.json',report)
 b.emit(stage='exact_init_only',PASS=True,worker_count=2,trajectory_count=0,outcomes_viewed=False)
def development(session,resume):
 setup_session('development-safe',session,resume);records=[]
 with create_pool('development-safe',session) as pool:
  # Same target as formal; only the registered development input root differs.
  records=[f.result(timeout=120) for f in [pool.submit(b.worker,c,0) for c in ('Profile','Mobility','Composed')]]
 pids={r['PID'] for r in [b.read(p) for p in (b.TECH/'initializers').glob('*.json')]}
 # Resume can add a new pair of initialized workers; none may remain alive.
 b.need(all(not process_alive(pid) for pid in pids),'DEVELOPMENT_ORPHAN_WORKER')
 maximum_abs=0.;maximum_rel=0.;leaves={}
 with b.phase('INTEGRITY'):
  for condition in ('Profile','Mobility','Composed'):
   path=b.SEAL/'raw'/condition/'0000.npz';proof=b.read(path.with_suffix('.json'));b.need(b.sha(path)==proof['artifact']['sha256'],'DEVELOPMENT_CHECKPOINT_HASH')
   got=b.arrays(path);reference=b.arrays(ROOT/'outputs/sp_u12b3e/development_reference'/condition/'0000.npz');b.validate(got)
   for key in got:
    b.need(np.array_equal(got[key],reference[key]),'DEVELOPMENT_NUMERICAL_DRIFT')
    if got[key].dtype!=np.bool_:
     maximum_abs=max(maximum_abs,float(np.max(np.abs(got[key]-reference[key]))));maximum_rel=max(maximum_rel,b.c.rel(got[key],reference[key]))
   leaves[condition]=dict(raw_sha256=b.sha(path),receipt_sha256=b.sha(path.with_suffix('.json')))
 b.need(not list(b.SEAL.rglob('*.partial.*')),'INCOMPLETE_DEVELOPMENT_ARTIFACT')
 report=dict(PASS=True,scope='B3A_CERTIFIED_DEVELOPMENT_ONLY',development_trajectories=3,trajectory_SNR_cases=6,actual_spawn=True,actual_initializer=True,actual_worker=True,actual_writer=True,actual_checkpoint=True,actual_finalization=True,resumed_records=sum(r['resumed'] for r in records),max_absolute_difference=maximum_abs,max_relative_difference=maximum_rel,tolerance=b.c.TOL,formal_inputs_decoded=False,formal_seeds_used=False,formal_trajectories_executed=0,bootstrap_count=0,leaves=leaves,reference_provenance=b.read(ROOT/'outputs/sp_u12b3e/development_reference_provenance.json'))
 b.write(b.TECH/('development_resume_receipt.json' if resume else 'development_receipt.json'),report)
 b.emit(stage='exact_development_resume' if resume else 'exact_development',PASS=True,worker_count=2,trajectory_count=3,outcomes_viewed=False)
def main():
 parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--init-only',action='store_true');mode.add_argument('--development-safe',action='store_true');mode.add_argument('--run',action='store_true');parser.add_argument('--session',default='formal');parser.add_argument('--resume',action='store_true');parser.add_argument('--pre-scientific-gate',action='store_true');args=parser.parse_args()
 if args.pre_scientific_gate and not args.run:parser.error('--pre-scientific-gate requires --run')
 try:
  if args.init_only:init_only(args.session)
  elif args.development_safe:development(args.session,args.resume)
  else:
   run(pre_scientific_gate=args.pre_scientific_gate,session=args.session)
 except Exception as exc:
  try:b.write(b.TECH/f'entry_exception_{os.getpid()}.json',b.technical_exception_record(exc))
  except OSError:pass
  b.emit(stage='exact_entrypoint',PASS=False,error_type=type(exc).__name__,failure_code=str(exc) if isinstance(exc,contract.ExecutionFreezeSchemaError) else type(exc).__name__,outcomes_viewed=False);raise SystemExit(1)

def compute():
 seeds=b.read(b.TECH/'bootstrap_seed_registry.json');aggregate=[];descriptive=[];contrasts=[];bootproof=[]
 for ordinal,condition in enumerate(b.CONDITIONS):
  E=[];T=[]
  for i in range(1000):
   raw=b.arrays(b.SEAL/'raw'/condition/f'{i:04d}.npz');E.append(raw['trajectory_error']);T.append(raw['trajectory_truth'])
  E=np.stack(E);T=np.stack(T);point=10*np.log10(E.sum(axis=0)/T.sum(axis=0)[:,None,None])
  db,digest=b.bootstrap(E,T,seeds['descriptive_master_seed'],ordinal);rec=b.archive(b.SEAL/'bootstrap'/f'descriptive_{condition}.npz',draws=db);bootproof.append(dict(kind='DESCRIPTIVE_ONLY',condition=condition,replicates=10000,points=12,index_hash=digest,artifact=rec))
  for si,s in enumerate(b.SNRS):
   for ai,arm in enumerate(b.ARMS):
    for di,d in enumerate(b.DEPLOY):
     row=dict(condition=condition,snr_db=s,arm=arm,deployment=d,independent_unit='trajectory',N=1000,pooled_nmse_db=float(point[si,ai,di]));aggregate.append(row)
     lo,hi=np.quantile(db[:,si,ai,di],[.025,.975],method='linear');descriptive.append(dict(**row,lower=float(lo),upper=float(hi),interval_role='DESCRIPTIVE_ONLY',replicates=10000))
  if condition in ('Profile','Mobility','Composed'):
   primary,pdigest=b.bootstrap(E,T,seeds['primary_master_seed'],ordinal);g=np.stack([primary[:,si,ai,1]-primary[:,si,2,1] for si in range(2) for ai in range(2)],axis=1);rec=b.archive(b.SEAL/'bootstrap'/f'primary_{condition}.npz',draws=g);bootproof.append(dict(kind='PRIMARY',condition=condition,replicates=10000,contrasts=4,index_hash=pdigest,artifact=rec))
   for si,s in enumerate(b.SNRS):
    for ai,name in enumerate(('Both_LR4_vs_Ponly_LR4','Both_LR4_vs_Monly_LR4')):
     lo,hi=np.quantile(g[:,2*si+ai],[.05/24,1-.05/24],method='linear');contrasts.append(dict(condition=condition,snr_db=s,comparison=name,estimate_db=float(point[si,ai,1]-point[si,2,1]),lower=float(lo),upper=float(hi),classification=b.classify(lo,hi),family_size=12,replicates=10000,independent_unit='trajectory'))
  b.emit(stage='bootstrap',condition=condition,replicates_completed=10000,replicates_expected=10000,status='COMPLETE')
 b.need(len(aggregate)==len(descriptive)==60 and len(contrasts)==12,'TABLE_CARDINALITY')
 for rows in (aggregate,descriptive,contrasts):
  b.need(all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float))),'TABLE_NONFINITE')
 for name,rows in [('aggregate_results',aggregate),('descriptive_results',descriptive),('primary_contrasts',contrasts),('figure_data',descriptive)]:b.csvwrite(b.SEAL/(name+'.csv'),rows)
 b.write(ROOT/'outputs/sp_u12b3b_bootstrap_audit.json',dict(PASS=True,primary_contrasts=12,primary_replicates_per_contrast=10000,primary_draws=120000,descriptive_points=60,descriptive_replicates_per_point=10000,primary_q=.05/24,descriptive_quantiles=[.025,.975],quantile_method='linear',paired_unit='trajectory',paired_across='arms/deployments/SNR withincondition',seed_registry_sha256=b.sha(b.TECH/'bootstrap_seed_registry.json'),bootstrap_receipts=bootproof,global_hypothesis=False,scientific_values_disclosed=False))

def write_report(name,value):return b.write(ROOT/'outputs'/('sp_u12b3g_next_'+name+'.json'),value)

def check_core():
 digest,partition=core();b.need(digest==EXPECTED,'B3G_SCIENTIFIC_CODE_CHANGE_REQUIRED');return digest

def evaluation(pool):
 tick=time.monotonic();last=tick;done=0;peaks={};pending={}
 try:
  for condition in b.CONDITIONS:
   next_id=0
   while next_id<1000 or pending:
    while next_id<1000 and len(pending)<2:
     pending[pool.submit(b.worker,condition,next_id)]=(condition,next_id);next_id+=1
    completed,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
    # Examine every completed exception before submitting any further task.
    for f in completed:
     if f.exception() is not None:raise f.exception()
    for f in completed:
     cond,i=pending.pop(f);v=f.result();b.need(v['condition']==cond and v['trajectory_id']==i and not v['resumed'],'CLEAN_RESTART_ID_OR_RESUME');done+=1;peaks[str(v['worker_pid'])]=v['peak']
    if time.monotonic()-last>25:b.emit(stage='evaluation',trajectory_count=done,elapsed_seconds=time.monotonic()-tick,worker_count=2);last=time.monotonic()
   b.emit(stage='evaluation',condition=condition,trajectory_count=done,cells_completed=2*(b.CONDITIONS.index(condition)+1),status='COMPLETE')
 except Exception as exc:
  info=b.technical_exception_record(exc)
  b.write(b.TECH/f'parent_exception_{os.getpid()}.json',info)
  b.write(b.TECH/'PERMANENT_STOP.json',dict(verdict='B3_NEXT_TECHNICAL_FAILURE_WITH_FORENSICS',stage='evaluation',timestamp_utc=info['timestamp_utc'],exception_type=type(exc).__name__,completed_H=done,outcomes_viewed=False,retries=0))
  for f in pending:f.cancel()
  # Terminate actual interpreter PIDs from worker handshakes, never venv launcher guesses.
  kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.restype=ctypes.c_void_p;kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint];kernel.CloseHandle.argtypes=[ctypes.c_void_p]
  for p in (b.TECH/'actual_worker_pids').glob('*.json'):
   rec=b.read(p)
   if rec['controller_PID']!=os.getpid():continue
   handle=kernel.OpenProcess(1,False,rec['PID'])
   if handle:kernel.TerminateProcess(handle,74);kernel.CloseHandle(handle)
  pool.shutdown(wait=True,cancel_futures=True);raise
 else:pool.shutdown(wait=True)
 b.need(done==5000,'TRAJECTORY_COVERAGE');b.write(b.TECH/'evaluation_receipt.json',dict(PASS=True,completed_H_reused=done,cells_completed=10,new_H_generated=0,resumes=0,retries=0,worker_peaks=peaks,elapsed_seconds=time.monotonic()-tick))

def integrity():
 keys=set();tick=time.monotonic();maximum=0.;count=0
 for condition in b.CONDITIONS:
  for i in range(1000):
   p=b.SEAL/'raw'/condition/f'{i:04d}.npz';proof=b.read(p.with_suffix('.json'));b.need(b.sha(p)==proof['artifact']['sha256'],'RAW_HASH');raw=b.arrays(p);b.validate(raw)
   b.need(proof['condition']==condition and proof['trajectory_id']==i,'RECEIPT_ID');b.need(proof['execution_freeze_sha256']==b.sha(b.TECH/'execution_freeze.json') and proof['operator_sha256']==b.read(b.CFG)['operator_sha256'],'FROZEN_BINDING')
   b.need(proof['Dense_LR4_coordinates_shared'] and proof['Both_current_predictions_bitwise'] and proof['Both_B2_coordinates_energies_bitwise'],'REPLAY_FLAGS');maximum=max(maximum,proof['normal_residual_max'])
   b.need(proof['unexpected_solve_failures']==0 and proof['nonfinite_outputs']==0,'TECHNICAL_FAILURE')
   for s in b.SNRS:
    for arm in b.ARMS:
     for d in b.DEPLOY:
      key=(condition,s,i,arm,d);b.need(key not in keys,'DUPLICATE_PRIMARY_KEY');keys.add(key)
   count+=1
   if time.monotonic()-tick>25:b.emit(stage='integrity',trajectory_count=count);tick=time.monotonic()
 b.need(len(keys)==60000 and len(list((b.SEAL/'raw').rglob('*.npz')))==5000 and len(list((b.SEAL/'raw').rglob('*.json')))==5000,'RAW_POPULATION')
 b.need(not list(b.SEAL.rglob('*.partial.*')),'INCOMPLETE_SEALED_ARTIFACT')
 write_report('integrity_audit',dict(PASS=True,condition_SNR_cells=10,trajectories_per_cell=1000,distinct_H_reused=5000,new_H_generated=0,units=10000,trajectory_rows=len(keys),frame_rows=6*len(keys),arms=list(b.ARMS),deployments=list(b.DEPLOY),inactive_coordinates_NA=True,coordinate_sharing=True,no_duplicate_primary_keys=True,finite_energies=True,unexpected_solve_failures=0,nonfinite_outputs=0,normal_residual_max=maximum,normal_residual_tolerance=b.c.SOLVE_TOL,independent_unit='trajectory',outcomes_viewed=False))
 write_report('both_replay_audit',dict(PASS=True,units=10000,scored_frames=60000,current_full_predictions_bitwise=True,B2_coordinates_frame_error_truth_bitwise=True,input_hashes_equal_B2=True,max_absolute_difference=0.,max_relative_difference=0.,tolerance=0.,B2_values_disclosed=False,B3_values_disclosed=False))

def seal():
 leaves={p.relative_to(b.SEAL).as_posix():dict(sha256=b.sha(p),bytes=p.stat().st_size) for p in sorted(b.SEAL.rglob('*')) if p.is_file()}
 f=b.read(b.TECH/'execution_freeze.json')
 m=dict(status='B3_NEXT_SEALED_COMPLETE_UNVIEWED',run_id='SP_U12B3_NEXT_CANDIDATE',mentor_authorization=b.read(b.TECH/'mentor_authorization.json'),historical_failure_disclosure_sha256=b.sha(ROOT/'docs/sp_u12b3e_spawn_failure_analysis.md'),scientific_core_sha256=check_core(),technical_wrapper_sha256=f['technical_wrapper_sha256'],execution_freeze_sha256=b.sha(b.TECH/'execution_freeze.json'),protocol_sha256=b.sha(b.PROTOCOL),config_sha256=b.sha(b.CFG),bootstrap_seed_registry_sha256=b.sha(b.TECH/'bootstrap_seed_registry.json'),formal_seed_registry_sha256=b.sha(ROOT/'outputs/sp_u12b2a_seed_registry.json'),B2_seal_before=b.read(b.TECH/'b2_seal_before.json'),run1_archive_manifest_sha256=b.sha(ROOT/'outputs/sp_u12b3c_failed_run1_archive.json'),raw_hash_tree=b.tree({k:v for k,v in leaves.items() if k.startswith('raw/')}),bootstrap_hash_tree=b.tree({k:v for k,v in leaves.items() if k.startswith('bootstrap/')}),all_artifact_hash_tree=b.tree(leaves),aggregate_sha256=leaves['aggregate_results.csv']['sha256'],primary_contrasts_sha256=leaves['primary_contrasts.csv']['sha256'],descriptive_sha256=leaves['descriptive_results.csv']['sha256'],figure_data_sha256=leaves['figure_data.csv']['sha256'],files=leaves,outcomes_viewed=False,scientific_figures_viewed=False)
 write_report('manifest',m)

def formal_preflight(pre_scientific_gate,session):
 global _run_config
 cfg=b.read(b.CFG)
 f=contract.load(contract.FREEZE_PATH)
 validation=contract.validate_execution_freeze(f)
 contract.require_authorization(pre_scientific_gate,f)
 _run_config=contract.runtime_config(pre_scientific_gate,session)
 from scripts.sp_u12b3e_io import activate_execution_contract
 activate_execution_contract(_run_config)
 b.need(not b.TECH.exists() if pre_scientific_gate else not(b.TECH/'PERMANENT_STOP.json').exists(),'FRESH_TECHNICAL_CONTROL_REQUIRED')
 b.TECH.mkdir(parents=True,exist_ok=True)
 b.atomic(b.TECH/'execution_freeze.json',contract.FREEZE_PATH.read_bytes())
 b.atomic(b.TECH/'bootstrap_seed_registry.json',(ROOT/'outputs/sp_u12b3b/bootstrap_seed_registry.json').read_bytes())
 b.write(b.TECH/'preflight_validation.json',dict(**validation,config_loaded=True,config_sha256=b.sha(b.CFG),formal_seed_registry_validated=True,mode='formal',pre_scientific_gate=pre_scientific_gate))
 b.firewall();b.check_freeze();check_core()
 return f,validation
def run(pre_scientific_gate=False,session='formal'):
 f,validation=formal_preflight(pre_scientific_gate,session)
 tick=time.monotonic();b.emit(stage='run',status='PRE_SCIENTIFIC',worker_count=2,outcomes_viewed=False)
 pool=create_pool('formal',session)
 try:
  heartbeats=launch_heartbeats(pool)
  if pre_scientific_gate:
   pool.shutdown(wait=True)
   receipt=startup_receipt(heartbeats)
   b.need(all(r.get('technical_checkpoint_roundtrip_PASS') for r in heartbeats),'WRITER_CHECKPOINT_NOT_READY')
   contract.validate_execution_freeze(f,verify_leaves=False)
   b.write(b.TECH/'pre_scientific_receipt.json',dict(**receipt,validation=validation,config_loaded=True,pre_scientific_gate_reached=True,trajectory_zero_not_started=True,technical_checkpoint_ready=True,elapsed_seconds=time.monotonic()-tick,exact_command=sys.argv,formal_base_command=contract.COMMAND,environment_variables={k:os.environ[k] for k in contract.THREAD_KEYS},stages=['CLI_PARSE','CONFIG_LOAD','FREEZE_LOAD','SCHEMA_VALIDATION','HASH_VALIDATION','B2_PROVENANCE','SEED_REGISTRY','OUTPUT_ROOT','SPAWN_POOL','TWO_INITIALIZERS','TWO_HEARTBEATS','WRITER_CHECKPOINT_READY'],formal_scientific_outputs=0,primary_bootstrap=0,descriptive_bootstrap=0,outcomes_viewed=False))
   b.emit(stage='exact_run_pre_scientific_gate',PASS=True,worker_count=2,trajectory_count=0,outcomes_viewed=False);return
  startup_receipt(heartbeats,require_exit=False)
  with b.phase('RAW'):evaluation(pool)
 except Exception:
  pool.shutdown(wait=True,cancel_futures=True);raise
 b.check_freeze();check_core()
 with b.phase('INTEGRITY'):integrity()
 b.need(b.read(ROOT/'outputs/sp_u12b3g_next_integrity_audit.json')['PASS'],'RAW_ALL_CELLS_REQUIRED')
 b.check_freeze();check_core()
 with b.phase('COMPUTE'):compute()
 with b.phase('SEAL_CHECK'):seal()
 e=b.read(b.TECH/'evaluation_receipt.json');write_report('resource_log',dict(elapsed_seconds=time.monotonic()-tick,parent_pid=os.getpid(),parent_peak=b.peak(),worker_peaks=e['worker_peaks'],CPU_workers=2,BLAS_threads=1,GPU_used=False,new_H_generated=0,retries=0,resumes=0,sealed_disk_bytes=sum(p.stat().st_size for p in b.SEAL.rglob('*') if p.is_file()),runtime_claim=False,scope='execution resources only;not B4',outcomes_viewed=False))
 b.emit(stage='run',status='SEALED_COMPUTATION_COMPLETE',cells_completed=10,trajectory_count=5000,outcomes_viewed=False)

def finalize():
 b.check_freeze();check_core();b.firewall()
 p=b.read(ROOT/'outputs/sp_u12b3g_next_preservation.json');b.need(p['PASS'],'PRESERVATION_GATE')
 b.need(b.read(ROOT/'outputs/sp_u12b3g_next_b2_seal_audit.json')['PASS'],'B2_SEAL_GATE');b.need(b.read(ROOT/'outputs/sp_u12b3g_next_run1_preservation.json')['PASS'],'RUN1_PRESERVATION_GATE')
 tests=b.read(b.TECH/'pytest_final.json');b.need(tests['exit_code']==0,'FINAL_TEST_GATE')
 m=b.read(ROOT/'outputs/sp_u12b3g_next_manifest.json')
 for n,r in m['files'].items():b.need(b.sha(b.SEAL/n)==r['sha256'],'FINAL_SEAL_HASH')
 failures=list(b.TECH.glob('*exception*.json'));b.need(not failures,'FILESYSTEM_EXCEPTION_GATE')
 operations=b.technical_operation_counts()
 for f in (b.TECH/'worker_io').glob('*.json'):
  for k,v in b.read(f)['operations'].items():operations[k]=operations.get(k,0)+v
 write_report('io_audit',dict(PASS=True,formal_filesystem_exceptions=0,B2_write_attempts=0,retries=0,partial_files_in_sealed_tree=0,operations=operations,required_operation_wrappers=['OPEN_READ','OPEN_WRITE','MKDIR','TEMP_CREATE','FLUSH','FSYNC','RENAME','REPLACE','DELETE','CHECKPOINT_READ','CHECKPOINT_WRITE','MANIFEST_WRITE'],RENAME_DELETE='audit policy records actual calls;formal runner did not need these mutations',crash_gate_provenance='SP-U12B3C passed actual dead-worker quarantine;SP-U12B3D synthetic tests retained',technical_wrapper_sha256=m['technical_wrapper_sha256']))
 m['B2_seal_after']=b.read(ROOT/'outputs/sp_u12b3g_next_b2_seal_audit.json')['after'];m['preservation_PASS']=True;write_report('manifest',m)
 cert='# B3 next-candidate seal certificate\n\nVerdict: B3_NEXT_SEALED_COMPLETE\n\n'+json.dumps({k:v for k,v in m.items() if k!='files'},indent=2)+'\n\nManifest SHA256: '+b.sha(ROOT/'outputs/sp_u12b3g_next_manifest.json')+'\n\nAll outcomes and scientific figures remain unviewed. Failed runs excluded; no new formal samples; no retry; no B4. STOP for mentor review.\n'
 b.atomic(ROOT/'docs/sp_u12b3g_next_seal_certificate.md',cert.encode())
 b.atomic(ROOT/'docs/sp_u12b3g_next_execution_log.md',('# B3 next-candidate execution log\n\n'+cert+'\nActual tests: '+json.dumps(tests)+'\n').encode())
 b.write(b.TECH/'decision.json',dict(verdict='B3_NEXT_SEALED_COMPLETE',outcomes_viewed=False,STOP=True));b.emit(stage='finalize',verdict='B3_NEXT_SEALED_COMPLETE',PASS=True,outcomes_viewed=False)

if __name__=='__main__':main()
