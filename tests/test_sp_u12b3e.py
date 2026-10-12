"""Actual completed gate receipts plus import/pickle/unchanged-source assertions."""
from pathlib import Path
import ast,json,inspect,pickle,importlib,pytest
from scripts import sp_u12b3e_common as b
from scripts.sp_u12b3e_build import core,EXPECTED,sources,CORE
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u12b3e'
@pytest.mark.parametrize('role',['init_worker','worker'])
def test_actual_callable_import_and_pickle(role):
 obj=getattr(b,role);assert '<locals>' not in obj.__qualname__ and obj.__qualname__==obj.__name__
 assert getattr(importlib.import_module(obj.__module__),obj.__name__) is obj and pickle.loads(pickle.dumps(obj)) is obj
def test_scientific_core_exact_and_original_worker_unchanged():
 assert core()[0]==EXPECTED
 assert sources(ROOT/'scripts/sp_u12b3e_common.py',CORE+('worker',))==sources(ROOT/'scripts/sp_u12b3d_common.py',CORE+('worker',))
 assert sources(ROOT/'scripts/run_sp_u12b3f.py',('compute',))==sources(ROOT/'scripts/run_sp_u12b3d.py',('compute',))
@pytest.mark.parametrize('name',['record','exception_record','save_failure','deny','atomic','read'])
def test_original_io_protection_function_bytes_retained(name):
 def nested(path):
  s=path.read_text(encoding='utf8');lines=s.splitlines(keepends=True);tree=ast.parse(s)
  install=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install_io')
  n=next(n for n in install.body if isinstance(n,ast.FunctionDef) and n.name==name)
  return ''.join(lines[n.lineno-1:n.end_lineno])
 assert nested(ROOT/'scripts/sp_u12b3e_io.py')==nested(ROOT/'scripts/sp_u12b3d_io.py')
def test_exact_shared_pool_factory_explicit_spawn_and_two_workers():
 from scripts import run_sp_u12b3f as runner
 s=inspect.getsource(runner.create_pool);assert "get_context('spawn')" in s and 'max_workers=2' in s and 'initializer=b.init_worker' in s and 'initargs=(config,)' in s
 for function in (runner.init_only,runner.development,runner.evaluation):assert 'create_pool(' in inspect.getsource(function)
 assert b.read(OUT/'nested_callable_audit.json')['PASS']
def test_twenty_real_cli_runs_forty_distinct_initializations_and_heartbeats():
 r=b.read(ROOT/'outputs/sp_u12b3e_init_only_repeat_test.json');assert r['PASS'] and r['runs']==20 and r['workers_initialized']==r['heartbeats_completed']==40 and r['orphan_workers']==r['pickle_errors']==r['PermissionError']==0
 for item in r['records']:
  assert len(set(item['PIDs']))==2
  p=OUT/'sessions'/item['session']/'init_only_receipt.json';assert b.sha(p)==item['receipt_sha256']
  receipt=b.read(p);assert receipt['Windows_spawn'] and receipt['working_directory']==str(ROOT)
  assert all(v['initializer_completed'] and v['B2_readonly_guard_installed'] and v['technical_writer_initialized'] for v in receipt['initializer_rows'])
def test_actual_development_and_checkpoint_equivalence():
 r=b.read(ROOT/'outputs/sp_u12b3e_development_equivalence.json');assert r['PASS'] and r['resume_PASS'] and r['development_trajectories']==3 and r['trajectory_SNR_cases']==6 and r['resume_records']==3
 assert r['max_absolute_difference']==r['max_relative_difference']==0 and r['raw_checkpoint_bytes_unchanged']
 assert not r['formal_inputs_decoded'] and not r['formal_seeds_used'] and r['bootstrap_count']==0
def test_run2_manifest_and_traceback_original_bytes():
 assert b.sha(ROOT/'outputs/sp_u12b3d_run2_manifest.json')=='a22d244473336acda5ab8b75d9a848e44f8b1fcab19fea2391b93649d361e7bb'
 r=b.read(ROOT/'outputs/sp_u12b3e_root_cause_freeze.json');assert r['ROOT_CAUSE']=='WINDOWS_SPAWN_COULD_NOT_SERIALIZE_LOCAL_INITIALIZER' and r['Run1_root_cause']=='UNRECOVERABLE_UNKNOWN'
 for n,h in r['original_traceback_files'].items():assert b.sha(ROOT/n)==h
def test_original_formal_seeds_hash_only_and_future_empty():
 original=b.read(ROOT/'outputs/sp_u12b3d/execution_freeze.json');assert b.sha(ROOT/'outputs/sp_u12b2a_seed_registry.json')==original['formal_seed_registry_sha256']
 assert not any((ROOT/'outputs/sp_u12b3f_run3_sealed').rglob('*'))
def test_run3_requires_new_authorization_without_invoking_formal_cli():
 from scripts.sp_u12b3e_authorization import require_formal_authorization
 with pytest.raises(PermissionError,match='RUN3_MENTOR_AUTHORIZATION_PENDING'):require_formal_authorization()
