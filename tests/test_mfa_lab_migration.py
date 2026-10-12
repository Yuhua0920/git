"""Migration unit checks; no formal data numeric reads or R5 smoke execution."""
from pathlib import Path
import os,sys,json,importlib.util,subprocess
import pytest
ROOT=Path(os.environ['MFA_MIGRATION_TEST_ROOT']).resolve()
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'deployment/mfa_lab_v1'))
import runtime

def test_actual_relative_root():assert runtime.ROOT==ROOT

def test_complete_payload_integrity():
    r=runtime.integrity();assert r['PASS']and r['payload_files_verified']>100

def test_exact_environment():assert runtime.environment()['PASS']

def test_complete_original_author_trees():
    import hashlib
    metadata=runtime.read(ROOT/'outputs/sp_u12b5a/historical_reference.json')
    combined={}
    for repo,entry in metadata.items():
        actual={name:runtime.sha(ROOT/'reference/sp_u12b5a/paper_period'/repo/name)for name in entry['files']}
        assert actual==entry['files']
        assert hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(',',':')).encode()).hexdigest()==entry['tree_sha256']
        combined.update({repo+'/'+name:value for name,value in actual.items()})
    assert len(combined)==15
    assert hashlib.sha256(json.dumps(combined,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='e6c27180ba2abb07c699e943ba581290371a4dea04583b19b63037c552c3db5f'

def test_no_sealed_outcomes_or_personal_envs():
    for n in('.git','.venv-cdl','.venv-gmm','outputs/sp_u12b2b_sealed','outputs/sp_u12b3g_next_sealed','paper'):
        assert not(ROOT/n).exists()

def test_mfa_policy_and_original_adapter():
    from src.baselines.mfa_reference_adapter import make_reference_model,FROZEN_POLICY
    for k in(1,2,4):
        for r in(4,8,16):
            m=make_reference_model(k,r);assert m.PPCA and not m.lock_psis and m.rs_clip==0 and m.maxiter==100 and m.tol==1e-4
    assert FROZEN_POLICY['max_condition_number']==1e6

def test_equivalence_and_old_loading_sentinel():
    from experiments.mfa_low_memory_fidelity_r4.fidelity import independent_identities,boundary_stop_test
    assert independent_identities()['PASS']and boundary_stop_test()['PASS']

def test_one_safe_synthetic_step_and_author_inference():
    from threadpoolctl import threadpool_limits
    from experiments.mfa_low_memory_fidelity_r4.fidelity import model,initialize,synthetic,capture_step,discrepancy
    with threadpool_limits(1):
        x=synthetic(64,64,2,4,42495000);a=initialize(model(2,4),x,42495001);b=initialize(model(2,4,True),x,42495001)
        left,right=capture_step(a,x),capture_step(b,x)
        assert all(discrepancy(right[n],v)['PASS']for n,v in left.items())

def test_preparation_numeric_read_is_blocked_before_header():
    code="import runtime; runtime.install_scope(); p=runtime.ROOT/'outputs/sp_u12b5b/selection_train.npy'; p.open('rb')"
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=ROOT/'deployment/mfa_lab_v1',capture_output=True,text=True,encoding='utf8')
    assert p.returncode!=0 and'PREPARATION_VALUE_READ_NOT_AUTHORIZED'in p.stderr

@pytest.mark.parametrize('action',['train','validate','final','target-evaluation'])
def test_formal_actions_are_not_exposed(action):
    p=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'deployment/mfa_lab_v1/lab.py'),action],capture_output=True,text=True,encoding='utf8')
    assert p.returncode!=0 and'invalid choice'in p.stderr

def test_validation_count_and_identity_no_change():
    from experiments.mfa_low_memory_fidelity_r4.canonical_validation_contract import metadata_contract,assert_exact_identity_coverage
    cfg=runtime.read(ROOT/'configs/sp_u12b5b_mfa_selection.yaml')
    assert assert_exact_identity_coverage(metadata_contract(cfg),cfg)['scored_frame_predictions']==32400

def test_installer_and_wheels_match_official_record():
    d=runtime.read(ROOT/'deployment/mfa_lab_v1/environment_source.json')
    assert len(d['wheels'])==14
    for row in d['wheels']:assert runtime.sha(ROOT/'deployment/mfa_lab_v1/wheelhouse'/row['file'])==row['sha256']
    assert runtime.sha(ROOT/'deployment/mfa_lab_v1/installers'/d['installer']['file'])==d['installer']['sha256']
