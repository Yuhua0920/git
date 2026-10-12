"""Technical contract tests only. Never opens scientific values."""
from pathlib import Path
import json, ast, pickle
from scripts import sp_u12b3f_bookkeeping as t
def test_frozen_core_wrapper_and_all_leaves():
 assert t.frozen()['scientific_core_sha256']==t.CORE
def test_original_failures_excluded():
 assert t.sha(t.ROOT/'outputs/sp_u12b3b_manifest.json')==t.RUN1
 assert t.sha(t.ROOT/'outputs/sp_u12b3d_run2_manifest.json')==t.RUN2
def test_bound_mentor_authorization():
 from scripts.sp_u12b3e_authorization import require_formal_authorization
 require_formal_authorization()
 a=t.read(t.ROOT/'configs/sp_u12b3f_run3_authorization.json')
 assert a['mentor_authorization']=='B3_RUN3_FORMAL_EXECUTION_APPROVED'
 assert a['execution_freeze_sha256']==t.FREEZE and not a['new_formal_samples']
def test_seed_reuse_complete():
 a=t.read(t.OUT/'seed_reuse_audit.json')
 assert a['PASS'] and a['B2_cache_trajectories_matched']==5000
 assert a['new_seeds']==a['new_samples']==0
def test_expected_cardinality_and_coordinate_null_schema():
 a=t.read(t.OUT/'preflight.json')
 assert a['expected_units']==5*2*1000
 assert a['expected_trajectory_rows']==a['expected_units']*6
 assert a['expected_frame_rows']==a['expected_trajectory_rows']*6
 assert a['expected_coordinate_rows']==a['expected_units']*3*6
 assert a['scientific_files_before']==a['bootstrap_before']==a['aggregate_before']==0
def test_spawn_targets_importable_and_roundtrip():
 from scripts import sp_u12b3e_common as b
 for f in (b.init_worker,b.worker):
  assert '<locals>' not in f.__qualname__
  assert pickle.loads(pickle.dumps(f)) is f
def test_actual_cli_spawn_workers_and_primitive_initargs():
 text=(t.ROOT/'scripts/run_sp_u12b3f.py').read_text(encoding='utf8')
 tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='create_pool')
 source=ast.get_source_segment(text,node)
 assert "get_context('spawn')" in source and 'max_workers=2' in source
 assert 'initializer=b.init_worker,initargs=(config,)' in source
 assert "pool=create_pool('formal','formal')" in text
def test_frozen_run_integrity_before_bootstrap():
 text=(t.ROOT/'scripts/run_sp_u12b3f.py').read_text(encoding='utf8')
 node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='run');s=ast.get_source_segment(text,node)
 assert s.index('evaluation()')<s.index('integrity()')<s.index('compute()')<s.index('seal()')
 assert "['PASS'],'RAW_ALL_CELLS_REQUIRED'" in s
def test_preservation_exclusions_are_scoped():
 assert t.allowed_control_addition('outputs/sp_u12b3e/run3_control/worker_io/123.json')
 assert not t.allowed_control_addition('outputs/sp_u12b3e/run3_control/execution_freeze.json')
 assert not t.allowed_control_addition('outputs/sp_u12b3e/development_inputs/Profile/H_0000.json')
 assert not t.owned(Path('scripts/run_sp_u12b3f.py'))
 assert t.owned(Path('scripts/sp_u12b3f_bookkeeping.py'))
def test_before_snapshot_complete_and_b2_intact():
 a=t.read(t.OUT/'preservation_before.json')
 assert a['B2']['PASS'] and a['B2']['files']==35031 and a['history']['PASS']
 assert 'scripts/run_sp_u12b3f.py' in a['hashes']
 assert 'outputs/sp_u12b3e/run3_control/execution_freeze.json' in a['hashes']
def test_final_if_present():
 # Before formal execution this explicitly tests the frozen technical preflight.
 p=t.OUT/'decision.json'
 if not p.exists():
  assert t.read(t.OUT/'preflight.json')['PASS'];return
 a=t.read(p);assert a['STOP'] and not a['outcomes_viewed']
 for n in ('preservation','b2_seal_audit','history_preservation'):
  assert t.read(t.ROOT/'outputs'/('sp_u12b3f_run3_'+n+'.json'))['PASS']
 if a['verdict']=='B3_RUN3_SEALED_COMPLETE':
  for n in ('integrity_audit','both_replay_audit','bootstrap_audit','io_audit'):
   assert t.read(t.ROOT/'outputs'/('sp_u12b3f_run3_'+n+'.json'))['PASS']
  m=t.read(t.ROOT/'outputs/sp_u12b3f_run3_manifest.json')
  assert m['status']=='B3_RUN3_SEALED_COMPLETE_UNVIEWED' and len(m['files'])==10013
def test_bookkeeping_frozen_if_present():
 if (t.OUT/'bookkeeping_freeze.json').exists():t.helpercheck()
 else:assert (t.ROOT/'tests/test_sp_u12b3f.py').exists()
def test_frozen_seal_missing_key_failure_reproduced():
 import pytest
 f=t.read(t.TECH/'execution_freeze.json')
 assert 'technical_wrapper_sha256' in f and 'io_wrapper_sha256' not in f
 with pytest.raises(KeyError,match='io_wrapper_sha256'):f['io_wrapper_sha256']
 text=(t.ROOT/'scripts/run_sp_u12b3f.py').read_text(encoding='utf8')
 node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='seal')
 assert "f['io_wrapper_sha256']" in ast.get_source_segment(text,node)
def test_no_formal_data_and_permanent_stop():
 assert not any(t.SCI.iterdir())
 p=t.OUT/'compatibility_audit.json'
 if p.exists():
  a=t.read(p);assert not a['PASS'] and not a['formal_command_started']
  assert t.read(t.TECH/'PERMANENT_STOP.json')['completed_H']==0
