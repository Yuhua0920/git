"""Canonical schema and actual --run gate regressions; no scientific outcomes."""
import ast,copy,pickle,subprocess,sys,os
from pathlib import Path
import pytest
from scripts import sp_u12b3g_schema as s
from scripts.sp_u12b3g_preserve import read,sha,OUT,ROOT
@pytest.fixture
def freeze():
 for k in s.THREAD_KEYS:os.environ[k]='1'
 return read(s.FREEZE_PATH)
def test_canonical_only_passes_same_formal_validation(freeze):
 from scripts.run_sp_u12b3f import contract
 assert 'technical_wrapper_sha256' in freeze and 'io_wrapper_sha256' not in freeze
 assert contract.validate_execution_freeze(freeze,verify_leaves=False)['PASS']
@pytest.mark.parametrize('field',list(s.REQUIRED))
def test_missing_required_field_is_explicit_not_keyerror(freeze,field):
 del freeze[field]
 with pytest.raises(s.ExecutionFreezeSchemaError,match='EXECUTION_FREEZE_SCHEMA_ERROR: '+field):s.validate_execution_freeze(freeze,verify_leaves=False)
@pytest.mark.parametrize('field',list(s.REQUIRED))
def test_null_required_field_is_rejected(freeze,field):
 freeze[field]=None
 with pytest.raises(s.ExecutionFreezeSchemaError,match='EXECUTION_FREEZE_SCHEMA_ERROR: '+field):s.validate_structure(freeze)
def test_obsolete_alias_only_rejected(freeze):
 freeze['io_wrapper_sha256']=freeze.pop('technical_wrapper_sha256')
 with pytest.raises(s.ExecutionFreezeSchemaError,match='technical_wrapper_sha256'):s.validate_execution_freeze(freeze,verify_leaves=False)
def test_obsolete_alias_plus_canonical_rejected(freeze):
 freeze['io_wrapper_sha256']=freeze['technical_wrapper_sha256']
 with pytest.raises(s.ExecutionFreezeSchemaError,match='obsolete alias rejected'):s.validate_structure(freeze)
@pytest.mark.parametrize('field,value',[('CPU_workers',3),('BLAS_threads_per_process',2),('Windows_spawn_context','fork'),('formal_output_root',str(ROOT/'outputs/sp_u12b3f_run3_sealed')),('technical_wrapper_sha256','0'*64),('formal_seed_registry_sha256','0'*64)])
def test_actual_contract_mismatch_names_field(freeze,field,value):
 freeze[field]=value
 with pytest.raises(s.ExecutionFreezeSchemaError,match='EXECUTION_FREEZE_SCHEMA_ERROR: '+field):s.validate_execution_freeze(freeze,verify_leaves=False)
def test_nested_b2_missing_field_rejected(freeze):
 del freeze['B2_provenance_before']['manifest_sha256']
 with pytest.raises(s.ExecutionFreezeSchemaError,match='B2_provenance_before.manifest_sha256'):s.validate_structure(freeze)
def test_no_formal_authorization_created(freeze):
 assert not (s.CONTROL/'formal_authorization.json').exists()
 with pytest.raises(PermissionError,match='NEXT_FORMAL_MENTOR_AUTHORIZATION_PENDING'):s.require_authorization(False,freeze)
def test_initializer_and_worker_qualified_pickle_roundtrip():
 from scripts import sp_u12b3e_common as b
 assert b.init_worker.__module__+'.'+b.init_worker.__qualname__=='scripts.sp_u12b3e_io.init_worker'
 assert b.worker.__module__+'.'+b.worker.__qualname__=='scripts.sp_u12b3e_io.guarded_worker'
 for f in (b.init_worker,b.worker):assert pickle.loads(pickle.dumps(f)) is f
def test_same_formal_path_and_same_pool_for_science_or_technical_exit():
 from scripts import run_sp_u12b3f as r
 source=(ROOT/'scripts/run_sp_u12b3f.py').read_text(encoding='utf8');tree=ast.parse(source)
 run=ast.get_source_segment(source,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run'))
 assert run.index('formal_preflight(')<run.index("create_pool('formal',session)")<run.index('launch_heartbeats(pool)')<run.index('if pre_scientific_gate:')<run.index('evaluation(pool)')
 main=ast.get_source_segment(source,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'))
 assert 'run(pre_scientific_gate=args.pre_scientific_gate,session=args.session)' in main
 assert "get_context('spawn')" in source and 'max_workers=2' in source
def test_shared_guard_blocks_scientific_target_in_gate():
 import scripts.sp_u12b3e_io as io
 prior=io._entry_config;io._entry_config=dict(mode='formal',execution_freeze_path=str(s.FREEZE_PATH),pre_scientific_gate=True)
 try:
  with pytest.raises(PermissionError,match='B3G_FORMAL_EXECUTION_VIOLATION'):io.guarded_worker('ID',0)
 finally:io._entry_config=prior
def test_B2_readonly_guard_blocks_before_OS():
 cmd=[str(ROOT/'.venv-cdl/Scripts/python.exe'),'-X','utf8','scripts/sp_u12b3g_audit.py','readonly-probe']
 env=dict(os.environ);env.update({k:'1' for k in s.THREAD_KEYS})
 r=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf8')
 assert r.returncode==0,r.stderr
 assert '"B2_write_blocked_before_OS": true' in r.stdout
 assert not (ROOT/'outputs/sp_u12b2b_sealed/B3G_FORBIDDEN_WRITE_PROBE').exists()
def test_original_failed_manifests_preserved():
 h=read(OUT/'preservation_before.json')['history']['hashes']
 for n,v in h.items():assert sha(ROOT/n)==v
def test_scientific_core_and_frozen_compute_unchanged():
 from scripts.sp_u12b3e_build import core
 from scripts.sp_u12b3g_audit import source_functions
 assert core()[0]==s.CORE
 assert source_functions(OUT/'original_bytes/scripts/run_sp_u12b3f.py',('compute',))==source_functions(ROOT/'scripts/run_sp_u12b3f.py',('compute',))
def test_no_scientific_artifact_generated():
 assert not any(p.is_file() for p in s.FORMAL_ROOT.rglob('*'))
 assert not any((ROOT/'outputs/sp_u12b3f_run3_sealed').iterdir())
def test_20_actual_exact_run_receipts_if_available():
 p=ROOT/'outputs/sp_u12b3g_repeat_preflight.json'
 if not p.exists():pytest.skip('Actual20-gate receipts not generated yet;final pytest requires them')
 a=read(p);assert a['PASS'] and a['runs']==a['passed']==20 and a['worker_initializations']==a['heartbeats']==40
 for r in a['run_receipts']:
  d=read(OUT/'gates'/r['session']/'pre_scientific_receipt.json')
  assert d['workers_initialized']==d['heartbeats_completed']==2
  assert d['pre_scientific_gate_reached'] and d['technical_checkpoint_ready']
  assert d['formal_trajectories_executed']==d['formal_scientific_outputs']==d['primary_bootstrap']==d['descriptive_bootstrap']==0
  assert d['orphan_workers']==d['serialization_failures']==d['filesystem_failures']==0
  assert all(v['technical_checkpoint_roundtrip_PASS'] for v in d['heartbeat_rows'])
