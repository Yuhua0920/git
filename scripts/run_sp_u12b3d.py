"""Mentor-authorized clean restart. Scientific compute appended byte-for-byte by builder."""
from pathlib import Path
import sys,os,json,time,concurrent.futures,subprocess,ctypes
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b3d_common as b
from scripts.sp_u12b3d_build import core,EXPECTED
import numpy as np
def write_report(name,value):return b.write(ROOT/'outputs'/('sp_u12b3d_run2_'+name+'.json'),value)
def check_core():
 digest,partition=core();b.need(digest==EXPECTED,'B3_RUN2_ABORT_SCIENTIFIC_CORE_DRIFT');return digest
def evaluation():
 tick=time.monotonic();last=tick;done=0;peaks={};pending={}
 pool=concurrent.futures.ProcessPoolExecutor(max_workers=2,initializer=b.init_worker)
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
  b.write(b.TECH/'PERMANENT_STOP.json',dict(verdict='B3_RUN2_TECHNICAL_FAILURE_WITH_FORENSICS',stage='evaluation',timestamp_utc=info['timestamp_utc'],exception_type=type(exc).__name__,completed_H=done,outcomes_viewed=False,retries=0))
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
 m=dict(status='B3_RUN2_SEALED_COMPLETE_UNVIEWED',run_id='SP_U12B3D_RUN2',mentor_authorization=b.read(b.TECH/'mentor_authorization.json'),historical_failure_disclosure_sha256=b.sha(ROOT/'docs/sp_u12b3d_historical_failure_disclosure.md'),scientific_core_sha256=check_core(),io_wrapper_sha256=f['io_wrapper_sha256'],execution_freeze_sha256=b.sha(b.TECH/'execution_freeze.json'),protocol_sha256=b.sha(b.PROTOCOL),config_sha256=b.sha(b.CFG),bootstrap_seed_registry_sha256=b.sha(b.TECH/'bootstrap_seed_registry.json'),formal_seed_registry_sha256=b.sha(ROOT/'outputs/sp_u12b2a_seed_registry.json'),B2_seal_before=b.read(b.TECH/'b2_seal_before.json'),run1_archive_manifest_sha256=b.sha(ROOT/'outputs/sp_u12b3c_failed_run1_archive.json'),raw_hash_tree=b.tree({k:v for k,v in leaves.items() if k.startswith('raw/')}),bootstrap_hash_tree=b.tree({k:v for k,v in leaves.items() if k.startswith('bootstrap/')}),all_artifact_hash_tree=b.tree(leaves),aggregate_sha256=leaves['aggregate_results.csv']['sha256'],primary_contrasts_sha256=leaves['primary_contrasts.csv']['sha256'],descriptive_sha256=leaves['descriptive_results.csv']['sha256'],figure_data_sha256=leaves['figure_data.csv']['sha256'],files=leaves,outcomes_viewed=False,scientific_figures_viewed=False)
 write_report('manifest',m)
def run():
 from scripts.sp_u12b3d_authorization import require_formal_authorization
 require_formal_authorization();b.check_freeze();check_core();b.need(not any(b.SEAL.iterdir()),'FRESH_RUN2_ROOT_REQUIRED');b.firewall();tick=time.monotonic();b.emit(stage='run',status='START',worker_count=2,outcomes_viewed=False)
 with b.phase('RAW'):evaluation()
 b.check_freeze();check_core()
 with b.phase('INTEGRITY'):integrity()
 b.need(b.read(ROOT/'outputs/sp_u12b3d_run2_integrity_audit.json')['PASS'],'RAW_ALL_CELLS_REQUIRED')
 b.check_freeze();check_core()
 with b.phase('COMPUTE'):compute()
 with b.phase('SEAL_CHECK'):seal()
 e=b.read(b.TECH/'evaluation_receipt.json');write_report('resource_log',dict(elapsed_seconds=time.monotonic()-tick,parent_pid=os.getpid(),parent_peak=b.peak(),worker_peaks=e['worker_peaks'],CPU_workers=2,BLAS_threads=1,GPU_used=False,new_H_generated=0,retries=0,resumes=0,sealed_disk_bytes=sum(p.stat().st_size for p in b.SEAL.rglob('*') if p.is_file()),runtime_claim=False,scope='execution resources only;not B4',outcomes_viewed=False))
 b.emit(stage='run',status='SEALED_COMPUTATION_COMPLETE',cells_completed=10,trajectory_count=5000,outcomes_viewed=False)
def finalize():
 b.check_freeze();check_core();b.firewall()
 p=b.read(ROOT/'outputs/sp_u12b3d_run2_preservation.json');b.need(p['PASS'],'PRESERVATION_GATE')
 b.need(b.read(ROOT/'outputs/sp_u12b3d_run2_b2_seal_audit.json')['PASS'],'B2_SEAL_GATE');b.need(b.read(ROOT/'outputs/sp_u12b3d_run2_run1_preservation.json')['PASS'],'RUN1_PRESERVATION_GATE')
 tests=b.read(b.TECH/'pytest_final.json');b.need(tests['exit_code']==0,'FINAL_TEST_GATE')
 m=b.read(ROOT/'outputs/sp_u12b3d_run2_manifest.json')
 for n,r in m['files'].items():b.need(b.sha(b.SEAL/n)==r['sha256'],'FINAL_SEAL_HASH')
 failures=list(b.TECH.glob('*exception*.json'));b.need(not failures,'FILESYSTEM_EXCEPTION_GATE')
 operations=b.technical_operation_counts()
 for f in (b.TECH/'worker_io').glob('*.json'):
  for k,v in b.read(f)['operations'].items():operations[k]=operations.get(k,0)+v
 write_report('io_audit',dict(PASS=True,formal_filesystem_exceptions=0,B2_write_attempts=0,retries=0,partial_files_in_sealed_tree=0,operations=operations,required_operation_wrappers=['OPEN_READ','OPEN_WRITE','MKDIR','TEMP_CREATE','FLUSH','FSYNC','RENAME','REPLACE','DELETE','CHECKPOINT_READ','CHECKPOINT_WRITE','MANIFEST_WRITE'],RENAME_DELETE='audit policy records actual calls;formal runner did not need these mutations',crash_gate_provenance='SP-U12B3C passed actual dead-worker quarantine;SP-U12B3D synthetic tests retained',io_wrapper_sha256=m['io_wrapper_sha256']))
 m['B2_seal_after']=b.read(ROOT/'outputs/sp_u12b3d_run2_b2_seal_audit.json')['after'];m['preservation_PASS']=True;write_report('manifest',m)
 cert='# SP-U12B3D Run2 seal certificate\n\nVerdict: B3_RUN2_SEALED_COMPLETE\n\n'+json.dumps({k:v for k,v in m.items() if k!='files'},indent=2)+'\n\nManifest SHA256: '+b.sha(ROOT/'outputs/sp_u12b3d_run2_manifest.json')+'\n\nAll outcomes and scientific figures remain unviewed. Run1 excluded; no new formal samples; no retry; no B4. STOP for mentor review.\n'
 b.atomic(ROOT/'docs/sp_u12b3d_run2_seal_certificate.md',cert.encode())
 b.atomic(ROOT/'docs/sp_u12b3d_run2_execution_log.md',('# SP-U12B3D Run2 execution log\n\n'+cert+'\nActual tests: '+json.dumps(tests)+'\n').encode())
 b.write(b.TECH/'decision.json',dict(verdict='B3_RUN2_SEALED_COMPLETE',outcomes_viewed=False,STOP=True));b.emit(stage='finalize',verdict='B3_RUN2_SEALED_COMPLETE',PASS=True,outcomes_viewed=False)
def main():
 mode=sys.argv[1]
 try:{'run':run,'finalize':finalize}[mode]()
 except Exception as exc:
  info=b.technical_exception_record(exc)
  try:
   b.write(b.TECH/f'parent_exception_{mode}_{os.getpid()}.json',info);b.write(b.TECH/'PERMANENT_STOP.json',dict(verdict='B3_RUN2_TECHNICAL_FAILURE_WITH_FORENSICS',stage=mode,exception_type=type(exc).__name__,outcomes_viewed=False,retries=0))
  except OSError:pass
  b.emit(stage=mode,verdict='B3_RUN2_TECHNICAL_FAILURE_WITH_FORENSICS',error_type=type(exc).__name__,outcomes_viewed=False);raise SystemExit(1)

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

if __name__=='__main__':main()
