"""Cache/source-authentication/firewall regressions, isolated analytic children."""
from pathlib import Path
import ast,hashlib,json,subprocess,sys
import pytest
import audit_core as a

def probe(tmp_path, mode, case='valid'):
    result = subprocess.run([sys.executable, '-B', '-X', 'utf8',
        str(a.HERE/'cache_probe.py'), str(tmp_path/'probe'), mode, case],
        capture_output=True, text=True, encoding='utf8')
    assert result.returncode == 0, result.stdout+result.stderr
    state = json.loads(result.stdout.strip().splitlines()[-1])
    cache=Path(state['cache_path'])
    if case=='absent':assert not cache.exists()
    else:
        assert hashlib.sha256(cache.read_bytes()).hexdigest()==state['cache_before']['sha256']
        assert cache.stat().st_mtime_ns==state['cache_before']['mtime_ns']
    assert state['real_lab_models_verified']==0
    return state

@pytest.mark.parametrize('case',['valid','absent'])
def test_legacy_failure_exact_cache_path_reproduced_even_with_B(tmp_path,case):
    v=probe(tmp_path,'legacy',case)
    assert v['dont_write_bytecode'] and v['complete_synthetic_models']==0
    assert v['error']['message']=='READONLY_AUDIT_READ_DENIED:'+v['cache_path']
    assert 'get_code' in v['error']['traceback'] and 'reference_modules' in v['error']['traceback']

@pytest.mark.parametrize('case',['valid','poisoned','corrupt','absent'])
def test_existing_untrusted_cache_never_read_or_executed_and_guarded_check_passes(tmp_path,case):
    v=probe(tmp_path,'fixed',case)
    assert v['error'] is None and v['complete_synthetic_models']==1
    assert not v['cache_read_attempts']
    assert len(v['model_check']['files'])==7 and v['model_check']['smoke']['component_calls']==2
    assert all(x['rejected'] for x in v['firewall'].values())
    refs=[x for x in v['source_loader'] if 'paper_period' in x['path']]
    assert len(refs)==3 and all(x['compiled_filename']==x['path'] for x in refs)
    assert all(not x['cache_read'] and not x['cache_write'] for x in refs)

def test_wrong_source_rejected_even_with_existing_cache(tmp_path):
    v=probe(tmp_path,'fixed','wrong_source')
    assert v['complete_synthetic_models']==0
    assert 'SOURCE_SHA256_MISMATCH' in v['error']['message']
    assert not v['cache_read_attempts']

def function_ast(path,name):
    tree=ast.parse(path.read_text(encoding='utf8'))
    return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name),include_attributes=False)

@pytest.mark.parametrize('name',['model_check','install_guard','first_stop','array_sha','resource_history','dependencies','archive_leaves','legacy_history','source_population','evidence_binding','export_job'])
def test_original_scientific_checker_and_permission_logic_unchanged(name):
    old=a.ROOT/'deployment/r7_evidence_audit_v1/audit_core.py'
    assert function_ast(old,name)==function_ast(a.HERE/'audit_core.py',name)

def test_original_42_tests_and_frozen_scientific_source_bytes_unchanged():
    old=a.ROOT/'deployment/r7_evidence_audit_v1'
    assert (old/'test_audit.py').read_bytes()==(a.HERE/'test_audit.py').read_bytes()
    for n in ['src/baselines/mfa_reference_adapter.py',
              'deployment/r6_formal_v1/model_ops.py',
              'reference/sp_u12b5a/paper_period/cplx-mfa/utils.py',
              'reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py',
              'reference/sp_u12b5a/paper_period/MFA_estimator/MFA_estimator.py']:
        assert hashlib.sha256((a.ROOT/n).read_bytes()).hexdigest()==a.inputs()['dependencies'][n]['sha256']

def test_failed_attempt_requires_all_raw_files_and_snapshot_never_writes(tmp_path):
    import preservation as p
    d=tmp_path/'lab_runs/r7_evidence_audit/runs'/p.FAILED_ATTEMPT;d.mkdir(parents=True)
    with pytest.raises(RuntimeError,match='ORIGINAL_FAILED_ATTEMPT_MISSING'):p.snapshot(tmp_path)
    for n in p.REQUIRED:(d/n).write_bytes(b'SYNTHETIC failure preservation fixture only')
    (d/'extra_original.log').write_bytes(b'SYNTHETIC')
    before={f.name:f.read_bytes()for f in d.iterdir()}
    v=p.snapshot(tmp_path);assert len(v['files'])==6
    assert before=={f.name:f.read_bytes()for f in d.iterdir()}
    (d/'status.json').write_bytes(b'SYNTHETIC changed')
    assert p.snapshot(tmp_path)!=v
