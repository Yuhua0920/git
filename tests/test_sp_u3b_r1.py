"""Exact scope, equivalence evidence, reliable resource control and actual gates."""
import ast
import os
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u3b_r1_common as c
from scripts import run_sp_u3b_r1 as run

def test_original_hashes_and_prior_fail_evidence():
    assert c.verify_prior()['protected_sources']==539
    old=c.load(c.OUT/'sp_u3b_resource_gate.json')
    assert old['gate']=='FAIL' and old['scientific_fits_executed']==0

def test_patch_scope_whitelist_reverse_identity():
    audit=c.load(c.path('patch_audit.json'))
    original=(ROOT/audit['original_path']).read_text(encoding='utf8')
    patched=c.path('patched_mfa.py').read_text(encoding='utf8')
    assert patched.replace(run.REPLACEMENT,run.ORIGINAL)==original
    assert audit['changed_functions']==['ComplexMFA._em_step']
    assert c.sha(c.path('patched_mfa.py'))==audit['patched_sha256']
    a=ast.parse(original);b=ast.parse(patched)
    def functions(tree):
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        return {n.name:ast.dump(n) for n in cls.body if isinstance(n,ast.FunctionDef)}
    aa=functions(a);bb=functions(b)
    assert {k for k in aa if aa[k]!=bb[k]}=={'_em_step'}

@pytest.mark.parametrize('field',['original_loop','original_patched','loop_patched'])
def test_contraction_direct_loop_strict_identity(field):
    q=c.load(c.path('equivalence.json'))
    assert q['rtol']==1e-11 and q['atol']==1e-12
    assert {r['rank'] for r in q['contraction']}=={1,4,16}
    assert all(r['errors'][field]['allclose'] for r in q['contraction'])

def test_conjugation_weighting_negative_controls():
    rows=c.load(c.path('equivalence.json'))['contraction']
    assert all(r['wrong_conjugation_error']>1e-3 for r in rows)
    assert all(r['wrong_weight_error']>1e-3 for r in rows if r['K']>1)

def test_original_patched_EM_step_complete():
    rows=c.load(c.path('equivalence.json'))['EM_steps']
    required={'means_','loadings_','noise_variances_','weights_','covariances_','precisions_',
              '_responsibilities','_log_likelihoods','updated_likelihood'}
    for r in rows:
        assert r['initial_bit_exact'] and r['RNG_state_exact']
        assert set(r['errors'])==required and all(v['allclose'] for v in r['errors'].values())

def test_five_iteration_trace_parameters_estimator():
    rows=c.load(c.path('equivalence.json'))['fits']
    for r in rows:
        assert r['iterations']>=5 and len(r['original_trace'])==len(r['patched_trace'])
        assert r['warnings'][0]==r['warnings'][1]
        assert all(v['allclose'] for v in r['errors'].values())
        assert 'estimator' in r['errors'] and 'posterior_responsibilities' in r['errors']

def test_no_three_dimensional_cross_product_AST():
    t=ast.parse(c.path('patched_mfa.py').read_text(encoding='utf8'))
    cls=next(n for n in t.body if isinstance(n,ast.ClassDef))
    step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_em_step')
    text=ast.unparse(step)
    assert 'zeroed_data[:, None, :]' not in text
    assert 'weighted_cross_covariance = zeroed_data @ weighted_latents.T' in text

def test_static_all_major_arrays_and_remaining_bottlenecks():
    rows=c.load(c.path('static_audit.json'))['rows']
    for r in rows:
        assert r['arrays']['training']==12000*4096*16
        assert r['patched_max_named_live_array_estimate']<c.cfg()['max_process_bytes']
        assert r['arrays']['latent_covariances']==r['rank']**2*12000*16
        assert r['arrays']['observation_buffers']==0
    assert rows[1]['arrays']['original_weighted_cross_tensor']==12582912000

def test_actual_windows_memory_monitor():
    m=c.memory()
    assert m['peak_working_set_bytes']>=m['working_set_bytes']>0
    assert m['peak_commit_bytes']>=m['commit_bytes']>0
    assert m['private_bytes']>0
    h=c.open_process(os.getpid())
    try:assert c.memory(h)['peak_commit_bytes']>0
    finally:c.close_process(h)

def test_monitor_actual_redirected_child_PID_and_terminate(tmp_path):
    import subprocess,time,json,signal
    marker=tmp_path/'child_pid.json'
    code="import os,time,json; from pathlib import Path; Path("+repr(str(marker))+\
      ").write_text(json.dumps({'pid':os.getpid()}),encoding='utf8'); x=bytearray(32*1024*1024); time.sleep(20)"
    proc=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    actual_pid=None;handle=None
    try:
        deadline=time.perf_counter()+10
        while not marker.exists() and time.perf_counter()<deadline:time.sleep(.02)
        assert marker.exists()
        actual_pid=json.loads(marker.read_text(encoding='utf8'))['pid'];handle=c.open_process(actual_pid)
        deadline=time.perf_counter()+5
        while time.perf_counter()<deadline:
            memory=c.memory(handle)
            if memory['peak_commit_bytes']>32*1024*1024:break
            time.sleep(.02)
        assert memory['peak_commit_bytes']>32*1024*1024
        os.kill(actual_pid,signal.SIGTERM);proc.wait(timeout=5)
        assert proc.returncode!=0
    finally:
        if proc.poll() is None:
            if actual_pid:
                try:os.kill(actual_pid,signal.SIGTERM)
                except OSError:pass
            proc.terminate();proc.wait()
        if handle:c.close_process(handle)

@pytest.mark.parametrize('cause',['working_set','commit','time','within'])
def test_resource_stop_boundary(cause):
    q=c.cfg();mem={'peak_working_set_bytes':1,'peak_commit_bytes':1};secs=0
    if cause=='working_set':mem['peak_working_set_bytes']=q['max_process_bytes']
    if cause=='commit':mem['peak_commit_bytes']=q['max_process_bytes']
    if cause=='time':secs=q['max_fit_seconds']
    expected={'working_set':'MEMORY_CAP','commit':'MEMORY_CAP','time':'FIT_TIME_CAP','within':None}[cause]
    assert c.stop_reason(mem,secs)==expected

@pytest.mark.parametrize('file',['t20b_formal_outcomes.csv','sp_u2d_age_rows.csv','t16c_target_statistics_B120.npz'])
def test_formal_U2_B120_denied_before_open(file):
    with pytest.raises(PermissionError,match='firewall'):(c.OUT/file).open('rb')

def test_original_outputs_immutable():
    with pytest.raises(PermissionError,match='firewall'):(c.OUT/'sp_u3b_resource_gate.json').open('a',encoding='utf8')

def test_original_online_estimator_same_method():
    from scripts import sp_u3b_r1_mfa as p
    assert p.MemorySafeMfaEstimator.estimate is p.MfaEstimator.estimate
    assert p.MemorySafeMfaEstimator._initialize is p.MfaEstimator._initialize
    assert p.MemorySafeMfaEstimator._run_em is p.MfaEstimator._run_em
    assert p.MemorySafeMfaEstimator._calculate_probabilities is p.MfaEstimator._calculate_probabilities

def test_corpus_entry_preserves_frozen_default_device():
    import inspect
    source=inspect.getsource(run.corpus)
    assert "OfficialCDL()" in source and "ShiftCDL(env)" in source
    assert "device='cpu'" not in source

def actual(name):
    file=c.path(name)
    if not file.exists():pytest.skip('Actual later gate not executed: '+name)
    return c.load(file)

def test_actual_medium_4096D_smoke():
    r=actual('medium_gate.json');assert r['passed']
    w=r['worker_result'];assert w['samples']==120 and w['dimension']==4096
    assert w['K']==4 and w['rank']==16 and w['finite_parameters']
    assert not w['forbidden_tensor_seen'] and w['observed_EM_array_shapes']['weighted_latents']==[16,120]
    assert max(r['observed_peaks'].values())<c.cfg()['max_process_bytes']

def test_actual_full_corpus_identity_order():
    r=actual('full_corpus.json')
    assert r['samples']==12000 and r['all_frozen_H_exact'] and r['all_actual_seeds_exact']
    assert len(r['channel_identity'])==1200 and r['trajectories_per_environment']==400
    assert r['environments']==['CDL-C-30','CDL-B-30','CDL-C-120']
    assert c.sha(c.path('full_clean.npy'))==r['data_sha256']
    import csv
    with c.path('full_order.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    expected=[(env,i,j) for env in r['environments'] for i in range(400) for j in range(10)]
    assert [(a['environment'],int(a['trajectory_id']),int(a['frame_id'])) for a in rows]==expected
    assert c.sha(c.path('full_order.csv'))==r['order_sha256']

@pytest.mark.parametrize('kind',['A','B'])
def test_actual_full_sentinel_config_resource_and_freeze(kind):
    r=actual(kind+'_gate.json')
    c.frozen_sources()
    if r['worker_result'] is None:
        assert not r['passed'] and r['reason'] in ('MEMORY_CAP','FIT_TIME_CAP','INVALID_FIT')
        return
    w=r['worker_result'];spec=c.cfg()['sentinels'][kind]
    assert (w['samples'],w['dimension'],w['seed'],w['K'],w['rank'])==(12000,4096,202603101,spec['K'],spec['rank'])
    if r['passed']:
        assert w['native_converged'] and w['finite_parameters']
        assert max(r['observed_peaks'].values())<8589934592 and w['fit_seconds']<3600
        assert not w['forbidden_tensor_seen']

def test_no_mutation_after_full_fit_start():
    if not c.path('execution_freeze.json').exists():pytest.skip('Pre-execution freeze created after pre tests')
    c.frozen_sources()
