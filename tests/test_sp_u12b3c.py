"""Technical gates only; never decode failed formal scientific rows."""
from pathlib import Path
import json,subprocess,sys,hashlib
from scripts import sp_u12b3c_build as builder
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u12b3c';TECH=ROOT/'outputs/sp_u12b3c_run2'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def test_failed_manifest_and_archive_opaque_bytes():
 assert hashlib.sha256((ROOT/'outputs/sp_u12b3b_manifest.json').read_bytes()).hexdigest()=='530cc5bc124bfc945a381fb6a3d30ad2ec70666c2887a48aa9b69f9847d8a250'
 for ledger in ('sp_u12b3c_failed_run1_archive.json','sp_u12b3c_failed_run1_archive_supplement.json'):
  for n,r in read(ROOT/'outputs'/ledger)['files'].items():
   assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==r['sha256'];assert hashlib.sha256((ROOT/r['archive_path']).read_bytes()).hexdigest()==r['sha256']
def test_no_historical_root_cause_invented():
 f=read(ROOT/'outputs/sp_u12b3c_exception_forensics.json');assert not f['original_traceback_saved'] and not f['original_formal_root_cause_confirmed'] and f['original_denied_path'] is None and f['original_denied_operation'] is None
 assert f['synthetic_cycles']==f['passed']==100 and not f['errors'] and f['decision']=='B3_IO_REPAIR_NEEDS_REVISION'
def test_scientific_function_and_dependency_hashes_unchanged():
 a=read(ROOT/'outputs/sp_u12b3c_scientific_hash_audit.json');assert a['PASS'] and a['scientific_core_sha256_before']==a['scientific_core_sha256_after'] and not a['scientific_core_modified']
 got=builder.functions(ROOT/'scripts/sp_u12b3c_run2_common.py',builder.CORE);got['compute']=builder.functions(ROOT/'scripts/sp_u12b3c_run2.py',('compute',))['compute'];assert got==a['before_function_source_hashes']
 for n,h in a['dependencies'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h
def test_three_two_worker_passes_and_checkpoint():
 a=read(ROOT/'outputs/sp_u12b3c_concurrency_test.json');assert a['PASS'] and a['workers']==2 and len(a['passes'])==3 and a['total_tasks']==300 and a['distinct_write_cycles']==200 and a['checkpoint_resume_cycles']==100
 assert a['PermissionError']==a['collision']==a['partial_rename']==0
def test_actual_worker_crash_restart_isolated():
 a=read(ROOT/'outputs/sp_u12b3c_crash_recovery_test.json');assert a['PASS'] and a['partial_not_complete'] and a['complete_checkpoint_recognized'] and a['other_trajectory_byte_unchanged'] and a['dead_worker_temp_quarantined']==1 and a['partial_temp_in_sealed_tree_after_recovery']==0
def test_exact_formal_path_writes_and_future_root_empty():
 a=read(ROOT/'outputs/sp_u12b3c_formal_path_dry_run.json');assert a['PASS'] and all(a['steps'].values()) and a['complete_only_manifest'] and not a['formal_data_processed']
 assert not any((ROOT/'outputs/sp_u12b3b_sealed_run2').iterdir())
def test_readonly_controls_full_trace_context_never_original_trace():
 a=read(TECH/'readonly_integration.json');assert a['PASS'] and a['B2_actual_mutations']==0 and a['B2_attempted_mutations_blocked']==6 and not a['ACL_changed']
 for r in a['expected_negative_control_tracebacks']:
  assert r['exception_type']=='PermissionError' and r['errno'] is None and r['winerror'] is None and r['full_python_traceback'] and r['call_stack'] and r['worker_pid'] and r['timestamp_utc'] and Path(r['filesystem_operation']['exact_absolute_path']).is_absolute()
def test_original_development_numerical_equivalence():
 a=read(OUT/'development_numerical_equivalence.json');assert a['PASS'] and a['development_trajectories']==3 and a['trajectory_SNR_cases']==6 and a['original_development_H_hashes_match'] and a['max_absolute_difference']==a['max_relative_difference']==0 and a['bootstrap_dummy_bitwise_equal'] and not a['formal_inputs_used'] and not a['failed_partial_data_used']
def test_real_B2_worker_initialization_and_cli_forbidden_without_future_authorization():
 for code in ("from scripts import sp_u12b3c_run2_common as b; b.init_worker()", "from scripts.sp_u12b3c_authorization import require_formal_authorization; require_formal_authorization('run')"):
  p=subprocess.run([sys.executable,'-X','utf8','-c',code],capture_output=True,text=True);assert p.returncode!=0 and 'SEPARATE_FUTURE_MENTOR_AUTHORIZATION_REQUIRED' in p.stderr
def test_registered_dummy_worker_initialization_after_authorization_hardening():
 code="from scripts import sp_u12b3c_run2_common as b; b.B2=b.ROOT/'outputs/sp_u12b3c/synthetic_B2_topology'; b.init_worker()"
 p=subprocess.run([sys.executable,'-X','utf8','-c',code],capture_output=True,text=True);assert p.returncode==0
def test_worker_envelope_persists_complete_injected_READ_permission_context():
 code="""from scripts import sp_u12b3c_run2_common as b
import json,errno
b.B2=b.ROOT/'outputs/sp_u12b3c/synthetic_B2_topology';b.init_worker()
def blocked(path):
 b.technical_record_operation('READ',path,denial_origin='CONTROLLED_INJECTION_NOT_HISTORICAL')
 raise PermissionError(errno.EACCES,'controlled synthetic read denial',str(path.resolve()))
b.arrays=blocked
try:b.worker('ID',0)
except PermissionError:pass
else:raise AssertionError('EXPECTED_DENIAL_MISSING')
paths=list(b.TECH.glob('worker_exception_*.json'))
found=False
for path in paths:
 r=json.loads(path.read_text(encoding='utf8'))
 if r['filesystem_operation'].get('denial_origin')=='CONTROLLED_INJECTION_NOT_HISTORICAL':
  assert r['errno']==errno.EACCES and r['exception_type']=='PermissionError'
  assert r['filesystem_operation']['operation']=='READ' and r['call_stack'] and r['full_python_traceback']
  found=True
assert found
"""
 p=subprocess.run([sys.executable,'-X','utf8','-c',code],capture_output=True,text=True);assert p.returncode==0
