"""H technical regression receipts only; no scientific generation or disclosure."""
import ast,pickle
from pathlib import Path
import pytest
from scripts import sp_u12b3h_audit as h
from scripts.sp_u12b3h_preserve import ROOT,OUT,SCI,CONTROL,read,sha

def test_actual_frozen_bytes():
    assert h.frozen()['technical_wrapper_sha256']==h.WRAP
    h.owncheck()

def test_authorization_bound_to_exact_formal_contract():
    a=read(CONTROL/'formal_authorization.json')
    assert a['formal_run_authorized'] and a['execution_freeze_sha256']==h.FREEZE
    assert a['scientific_core_sha256']==h.CORE and a['technical_wrapper_sha256']==h.WRAP
    assert a['exact_command']==h.s.COMMAND
    assert not a['pre_scientific_gate'] and not a['new_formal_samples'] and not a['new_formal_seeds']
    assert a['formal_retries']==0 and sha(Path(a['task_path']))==a['task_sha256']
    h.s.require_authorization(False)

def test_fresh_start_and_exact_expected_schema():
    p=read(OUT/'preflight.json')
    assert p['scientific_files_before']==p['bootstrap_before']==p['aggregate_before']==0
    assert p['started_from_trajectory_zero'] and p['seed_metadata_trajectories_checked']==5000
    assert p['expected_units']==10000 and p['expected_trajectory_records']==60000
    assert p['expected_frame_records']==360000 and p['expected_coordinate_rows']==180000
    assert p['expected_final_scientific_files']==10014

def test_original_full_B2_baseline_and_failed_history():
    before=read(OUT/'preservation_before.json')
    assert before['B2']['PASS'] and before['B2']['files']==35031
    assert before['scientific_files_before']==0
    for n,v in before['history']['hashes'].items():assert sha(ROOT/n)==v

def test_no_method_or_bootstrap_compute_change():
    from scripts.sp_u12b3g_audit import source_functions
    names=('compute',)
    assert source_functions(ROOT/'outputs/sp_u12b3g/original_bytes/scripts/run_sp_u12b3f.py',names)==source_functions(ROOT/'scripts/run_sp_u12b3f.py',names)

def test_actual_twenty_previous_exact_entrypoint_gates():
    a=read(ROOT/'outputs/sp_u12b3g_repeat_preflight.json')
    assert a['PASS'] and a['runs']==a['passed']==20
    assert a['worker_initializations']==a['heartbeats']==40
    for r in a['run_receipts']:
        d=read(ROOT/'outputs/sp_u12b3g/gates'/r['session']/'pre_scientific_receipt.json')
        assert d['workers_initialized']==d['heartbeats_completed']==2
        assert d['formal_trajectories_executed']==d['primary_bootstrap']==d['descriptive_bootstrap']==0
        assert d['orphan_workers']==d['serialization_failures']==d['filesystem_failures']==0

def test_spawn_targets_pickle_to_qualified_functions():
    from scripts import sp_u12b3e_common as b
    assert b.init_worker.__module__+'.'+b.init_worker.__qualname__=='scripts.sp_u12b3e_io.init_worker'
    assert b.worker.__module__+'.'+b.worker.__qualname__=='scripts.sp_u12b3e_io.guarded_worker'
    for v in (b.init_worker,b.worker):assert pickle.loads(pickle.dumps(v)) is v
    config=h.s.runtime_config(False,'formal')
    assert pickle.loads(pickle.dumps(config))==config
    assert all(type(v) in (str,bool) for v in config.values())

def test_same_pool_complete_raw_integrity_before_bootstrap():
    source=(ROOT/'scripts/run_sp_u12b3f.py').read_text(encoding='utf8')
    tree=ast.parse(source)
    run=ast.get_source_segment(source,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run'))
    assert run.index('formal_preflight(')<run.index("create_pool('formal',session)")<run.index('launch_heartbeats(pool)')<run.index('evaluation(pool)')<run.index('integrity()')<run.index('RAW_ALL_CELLS_REQUIRED')<run.index('compute()')
    assert "get_context('spawn')" in source and 'max_workers=2' in source

@pytest.mark.parametrize('condition',h.COND)
def test_registered_conditions_have_complete_input_metadata(condition):
    base=ROOT/'outputs/sp_u12b2b_sealed/cache'/condition
    assert len(list(base.glob('H_*.json')))==1000
    assert len(list(base.glob('comm_*.json')))==1000
    # prepare() compared all 5000 identities against the original registry.
    assert read(OUT/'preflight.json')['formal_seed_registry_identical']

def test_primary_and_descriptive_family_frozen():
    source=(ROOT/'scripts/run_sp_u12b3f.py').read_text(encoding='utf8')
    assert "if condition in ('Profile','Mobility','Composed')" in source
    assert '[.05/24,1-.05/24]' in source and '[.025,.975]' in source
    assert "quantile_method='linear'" in source and 'global_hypothesis=False' in source
    assert "interval_role='DESCRIPTIVE_ONLY'" in source

def test_formal_lifecycle_receipts():
    end=OUT/'formal_exit.json'
    if not end.exists():
        assert not any(p.is_file() for p in SCI.rglob('*'))
        return
    assert read(end)['exit_code']==0
    assert not h.exception_files() and not(CONTROL/'PERMANENT_STOP.json').exists()
    assert read(CONTROL/'evaluation_receipt.json')['completed_H_reused']==5000
    assert len(list((CONTROL/'initializers').glob('*.json')))==2
    assert len(list((CONTROL/'heartbeats').glob('*.json')))==2

@pytest.mark.parametrize('name',('integrity_audit','both_replay_audit','bootstrap_audit'))
def test_postformal_machine_audit_or_preformal_clean_root(name):
    if not(OUT/'formal_exit.json').exists():
        assert not any(p.is_file() for p in SCI.rglob('*'))
        return
    a=read(ROOT/'outputs'/('sp_u12b3h_'+name+'.json'))
    assert a['PASS']
    if name=='integrity_audit':
        assert a['units']==10000 and a['trajectory_rows']==60000 and a['frame_rows']==360000
        assert a['coordinate_sharing'] and a['inactive_coordinates_NA']
        assert a['unexpected_solve_failures']==a['nonfinite_outputs']==0
    if name=='both_replay_audit':
        assert a['max_absolute_difference']==a['max_relative_difference']==0
    if name=='bootstrap_audit':
        assert a['primary_contrasts']==12 and a['descriptive_points']==60
        assert a['primary_replicates_per_contrast']==a['descriptive_replicates_per_point']==10000
        assert len(a['shape_hash_finiteness_checks'])==8

def test_postformal_exports_or_preformal_absence():
    if not(OUT/'formal_exit.json').exists():
        assert not(SCI/'coordinate_data.csv').exists()
        return
    e=read(OUT/'export_audit.json')
    assert e['PASS'] and e['trajectory_rows']==60000 and e['coordinate_rows']==180000
    assert not e['new_scientific_analysis'] and not e['outcomes_disclosed']
    assert len(list((SCI/'raw').rglob('*.npz')))==5000
    assert len(list((SCI/'raw').rglob('*.json')))==5000
    assert len(list((SCI/'bootstrap').glob('*.npz')))==8
    assert len([p for p in SCI.rglob('*') if p.is_file()])==10014
