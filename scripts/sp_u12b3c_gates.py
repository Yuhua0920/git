"""Dummy/development-only I/O gates. Never run B3 formal trajectory processing."""
from pathlib import Path
import os,sys,json,uuid,shutil,concurrent.futures,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b3b_common as old
from scripts import sp_u12b3c_run2_common as b
import numpy as np
OUT=ROOT/'outputs/sp_u12b3c';DUMMY=OUT/'synthetic_B2_topology';DRY=ROOT/'outputs/sp_u12b3b_sealed_run2'
def safe_archive_directory(name):
 source=DRY.resolve();dest=(OUT/name).resolve();workspace=ROOT.resolve()
 if source!=workspace/'outputs/sp_u12b3b_sealed_run2' or not dest.is_relative_to(OUT.resolve()) or dest.exists() or not source.is_relative_to(workspace):raise RuntimeError('UNSAFE_DUMMY_ARCHIVE_TARGET')
 if source.exists():os.replace(source,dest)
 source.mkdir(parents=True,exist_ok=False)
def prepare():
 report=old.read(OUT/'old_wrapper_technical/exception_forensics.json')
 old.write(ROOT/'outputs/sp_u12b3c_exception_forensics.json',report)
 safe_archive_directory('old_wrapper_formal_path_dry_archive')
 b.TECH.mkdir(exist_ok=True);b.atomic(b.TECH/'execution_freeze.json',(ROOT/'outputs/sp_u12b3b/execution_freeze.json').read_bytes())
 for i in range(100,201):
  for sub,prefixes in [('cache',('H_','comm_')),('raw_evaluation',('',))]:
   for prefix in prefixes:
    for ext in ('npz','json'):
     src=DUMMY/sub/'ID'/f'{prefix}0000.{ext}';dest=DUMMY/sub/'ID'/f'{prefix}{i:04d}.{ext}';shutil.copyfile(src,dest)
 print(json.dumps(dict(stage='gate_prepare',PASS=True,formal_inputs_decoded=False)))
def init():
 b.B2=DUMMY;b.init_worker()
def job(i):
 try:return dict(PASS=True,result=b.worker('ID',i))
 except Exception as exc:return dict(PASS=False,exception=b.technical_exception_record(exc))
def concurrency():
 allresults=[];passes=[]
 for indices in (range(100),range(100),range(100,200)):
  with concurrent.futures.ProcessPoolExecutor(max_workers=2,initializer=init) as pool:
   records=list(pool.map(job,indices))
  allresults+=records;passed=all(r['PASS'] for r in records);passes.append(dict(tasks=len(records),PASS=passed,resumed=sum(r['result']['resumed'] for r in records if r['PASS'])));print(json.dumps(dict(stage='candidate_concurrency',pass_number=len(passes),tasks=len(records),PASS=passed)),flush=True)
 errors=[r['exception'] for r in allresults if not r['PASS']];raw=list((DRY/'raw/ID').glob('*.npz'));receipts=list((DRY/'raw/ID').glob('*.json'));partial=list(DRY.rglob('*.partial*'))
 expected=old.arrays(OUT/'synthetic_expected.npz');identical=True
 for p in raw:
  got=b.arrays(p);identical &= all(np.array_equal(got[k],v) for k,v in expected.items())
 report=dict(PASS=not errors and len(raw)==len(receipts)==200 and not partial and identical,workers=2,passes=passes,total_tasks=300,distinct_write_cycles=200,checkpoint_resume_cycles=100,PermissionError=sum(r['exception_type']=='PermissionError' for r in errors),collision=0 if len(raw)==200 and len(receipts)==200 else None,partial_rename=len(partial),numerical_fixture_identity=bool(identical),data_scope='SYNTHETIC_ONLY',errors=errors)
 b.write(ROOT/'outputs/sp_u12b3c_concurrency_test.json',report)
 if not report['PASS']:raise RuntimeError('CONCURRENCY_GATE_FAILED')
def interrupted_writer():
 b.B2=DUMMY;b.firewall()
 with b.phase('SYNTHETIC'):
  path=DRY/'raw/ID/0200.npz';tmp=path.with_name(path.name+f'.partial.{os.getpid()}.1.{uuid.uuid4().hex}')
  with tmp.open('xb') as f:f.write(b'INCOMPLETE_SYNTHETIC_TEST_BYTES');f.flush();os.fsync(f.fileno())
  os._exit(73)
def crash():
 other=DRY/'raw/ID/0000.npz';before=b.sha(other);interrupted=False
 try:
  with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:pool.submit(interrupted_writer).result()
 except concurrent.futures.process.BrokenProcessPool:interrupted=True
 partial=list((DRY/'raw/ID').glob('0200.npz.partial.*'));incomplete=not(DRY/'raw/ID/0200.json').exists() and not(DRY/'raw/ID/0200.npz').exists()
 with concurrent.futures.ProcessPoolExecutor(max_workers=2,initializer=init) as pool:
  recovery=pool.submit(job,200).result();resume=pool.submit(job,200).result()
 remaining=list((DRY/'raw/ID').glob('0200.npz.partial.*'));quarantine=list((b.TECH/'orphan_temp_quarantine').glob('*'));other_unchanged=b.sha(other)==before
 report=dict(PASS=interrupted and len(partial)==1 and incomplete and recovery['PASS'] and not recovery['result']['resumed'] and resume['PASS'] and resume['result']['resumed'] and not remaining and len(quarantine)==1 and other_unchanged,interruption='actual worker os._exit(73)',partial_not_complete=incomplete,interrupted_temp_count=len(partial),restart_fresh_worker_pool=True,recovered_missing_key_only=True,complete_checkpoint_recognized=resume.get('PASS',False) and resume.get('result',{}).get('resumed',False),other_trajectory_byte_unchanged=other_unchanged,dead_worker_temp_quarantined=len(quarantine),partial_temp_in_sealed_tree_after_recovery=len(remaining),data_scope='DUMMY_ONLY')
 b.write(ROOT/'outputs/sp_u12b3c_crash_recovery_test.json',report);print(json.dumps(dict(stage='crash_recovery',PASS=report['PASS'])),flush=True)
 if not report['PASS']:raise RuntimeError('CRASH_GATE_FAILED')
def readonly_child():
 b.firewall();operations=[]
 # Header bytes only: no arrays, pilots or scientific values are decoded.
 for phase,path in [('RAW',b.B2/'cache/ID/H_0000.npz'),('RAW',b.B2/'cache/ID/comm_0000.npz')]:
  with b.phase(phase):
   with path.open('rb') as f:assert f.read(4)==b'PK\x03\x04'
  operations.append(dict(operation='READ',path=str(path.resolve()),result='PASS_ZIP_HEADER_ONLY',worker_pid=os.getpid()))
 candidate=b.B2/'_B3C_FORBIDDEN_TEST_FILE';sources=b.TECH/'rename_source_dummy.txt';sources.parent.mkdir(exist_ok=True);sources.write_bytes(b'DUMMY')
 # The audit policy rejects before the OS operation; no B2 file is created/touched.
 tests=[('WRITE',lambda:candidate.open('wb')),('CREATE',lambda:candidate.touch()),('MKDIR',lambda:candidate.mkdir()),('RENAME',lambda:os.replace(sources,candidate)),('DELETE',lambda:candidate.unlink()),('OTHER',lambda:os.utime(candidate,None))]
 exceptions=[]
 for op,call in tests:
  try:call()
  except PermissionError as exc:
   info=b.technical_exception_record(exc);exceptions.append(info);operations.append(dict(operation=op,path=str(candidate.resolve()),result='EXPECTED_POLICY_REJECTION',worker_pid=os.getpid()))
  else:raise RuntimeError('B2_READONLY_BOUNDARY_FAILED')
 assert not candidate.exists() and sources.read_bytes()==b'DUMMY'
 b.write(b.TECH/'readonly_integration.json',dict(PASS=True,read_real_B2_header_only=True,B2_attempted_mutations_blocked=len(exceptions),B2_actual_mutations=0,ACL_changed=False,operations=operations,expected_negative_control_tracebacks=exceptions,historical_failure_relation='controls only;not a reconstruction of original failure'))
 print(json.dumps(dict(stage='B2_readonly',PASS=True,expected_mutation_denials=len(exceptions))))
def manifest_dry():
 # Exactly the same atomic JSON writer, against actual future output root.
 records=[]
 for p in sorted((DRY/'raw/ID').glob('*.json')):
  r=b.read(p);artifact=r['artifact'];assert b.sha(ROOT/artifact['path'])==artifact['sha256'];records.append(artifact)
 assert len(records)==201
 b.write(DRY/'dummy_dry_run_manifest.json',dict(scope='SYNTHETIC_DRY_RUN_NOT_FORMAL_EVIDENCE',complete_artifacts=records,partial_files_registered=0,complete_trajectory_keys=201))
 m=b.read(DRY/'dummy_dry_run_manifest.json');assert len(m['complete_artifacts'])==201
 report=dict(PASS=True,exact_future_absolute_root=str(DRY.resolve()),fixture='SYNTHETIC_ONLY',formal_data_processed=False,steps=dict(open=True,mkdir=True,worker_write=True,checkpoint=True,flush=True,fsync=True,rename_replace=True,manifest=True,close=True),complete_only_manifest=True,interrupted_partial_registered=False,root_to_be_archived_before_freeze=True)
 b.write(ROOT/'outputs/sp_u12b3c_formal_path_dry_run.json',report)
 safe_archive_directory('candidate_formal_path_dry_archive')
 b.write(OUT/'future_output_empty.json',dict(PASS=True,root=str(DRY.resolve()),empty=not any(DRY.iterdir()),dummy_test_archive='outputs/sp_u12b3c/candidate_formal_path_dry_archive',formal_run2_started=False))
 print(json.dumps(dict(stage='formal_path_dry_run',PASS=True,output_root_empty=True)))
def numerical_development():
 from scripts import run_sp_u12b2a as a
 from src.cdl_benchmark import SYSTEM,array_hash
 assert SYSTEM['sample_rate_hz']==SYSTEM['nfft']*SYSTEM['delta_f_hz']
 registry=old.read(old.c.OUT/'development_seed_registry.json');assert registry['namespace']=='SP_U12B2A_DEVELOPMENT_ONLY_V1';index=a.seed_index(registry);operators=old.c.load_operators();cvs,factors=old.c.family(operators,old.SNRS)
 metadata=old.read(ROOT/'outputs/sp_u12b3a_development_smoke.json');hashes={(r['condition'],r['trajectory_id']):r['H_sha256'] for r in metadata['records']};max_abs=0.;max_rel=0.;cases=0
 for condition in ('Profile','Mobility','Composed'):
  adapter=a.generator(condition,index,'EVAL');H=adapter.generate(0);assert array_hash(H)==hashes[condition,0]
  comm=[old.c.communicate(a.consumed(index,condition,'EVAL',0,j)) for j in range(10)]
  before,inputs,r=old.evaluate(H,comm,operators,cvs,factors);after,inputs2,r2=b.evaluate(H,comm,operators,cvs,factors);assert inputs==inputs2
  for k,v in before.items():
   x=after[k];assert np.array_equal(v,x)
   if v.dtype!=np.bool_:max_abs=max(max_abs,float(np.max(abs(v-x))));max_rel=max(max_rel,old.c.rel(x,v))
  cases+=2;print(json.dumps(dict(stage='development_equivalence',condition=condition,PASS=True)),flush=True)
 # Exact unchanged bootstrap mathematics checked on dummy sufficient statistics.
 E=np.arange(60,dtype=float).reshape(5,2,3,2)+1;T=np.ones((5,2));seed=old.read(ROOT/'outputs/sp_u12b3b/bootstrap_seed_registry.json')['primary_master_seed'];x,h1=old.bootstrap(E,T,seed,0);y,h2=b.bootstrap(E,T,seed,0);assert np.array_equal(x,y) and h1==h2
 b.write(OUT/'development_numerical_equivalence.json',dict(PASS=True,namespace=registry['namespace'],conditions=['Profile','Mobility','Composed'],development_trajectories=3,trajectory_SNR_cases=cases,original_development_H_hashes_match=True,max_absolute_difference=max_abs,max_relative_difference=max_rel,raw_outputs_bitwise_equal=True,bootstrap_dummy_bitwise_equal=True,formal_inputs_used=False,failed_partial_data_used=False,scientific_performance_disclosed=False))
def envelope():
 with concurrent.futures.ProcessPoolExecutor(max_workers=2,initializer=init) as pool:rows=list(pool.map(job,range(100)))
 assert all(r['PASS'] for r in rows)
 report=old.read(ROOT/'outputs/sp_u12b3c_concurrency_test.json');report['final_envelope_regression']=dict(PASS=True,tasks=100,workers=2,distinct_write_keys=100);old.write(ROOT/'outputs/sp_u12b3c_concurrency_test.json',report)
 safe_archive_directory('final_envelope_formal_path_archive')
 print(json.dumps(dict(stage='final_exception_envelope_regression',tasks=100,PASS=True,output_root_empty=True)))
if __name__=='__main__':{'prepare':prepare,'concurrency':concurrency,'crash':crash,'readonly':readonly_child,'manifest':manifest_dry,'development':numerical_development,'envelope':envelope}[sys.argv[1]]()
