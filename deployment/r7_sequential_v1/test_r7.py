"""Technical acceptance: synthetic queue, failures, Windows jobs, no science."""
from pathlib import Path
import sys,os,json,time,contextlib,subprocess,threading,ast,inspect,copy
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import pytest
import r7_common as q
from queue_engine import run_queue
from toy_queue import SyntheticBackend

BASE=Path(os.environ['R7_TECHNICAL_TEST_ROOT'])
ROWS=q.contract()['candidate_order']
def folder(name):p=BASE/name;p.mkdir(parents=True,exist_ok=False);return p

def test_exact_frozen_bytes_order_seed_resource_contract():
    q.verify()
    c,r,s,w=q.r6_imports()
    assert ROWS==c.registry()==c.science()['candidate_order']
    assert [r['name'] for r in ROWS]==['K1_r4','K1_r8','K1_r16','K2_r4','K2_r8','K2_r16','K4_r4','K4_r8','K4_r16']
    assert [r['seed'] for r in ROWS]==[375404078,2201718927,1485464306,1443235906,1625978320,3827983190,1714107369,80346791,1483984871]
    assert q.contract()['resource_contract']==c.protocol() and c.protocol()['status']=='MENTOR_APPROVAL_PENDING'
    assert q.contract()['queue_total_timeout_seconds'] is None
    assert q.contract()['R6_stricter_combined_candidate_timeout_seconds']==86400

def test_nine_actual_author_toy_models_sequential_monitored():
    p=folder('nine_actual_original');b=SyntheticBackend(p)
    result=run_queue(p,ROWS,b)
    assert result['completion']=='QUEUE_COMPLETE_9_OF_9' and len(result['completed_candidates'])==9
    assert b.started==[(r['name'],stage) for r in ROWS for stage in ('train','validate')]
    for r in ROWS:
        for stage in ('train','validate'):
            v=q.read(p/'jobs'/r['name']/stage/'status.json')
            assert v['state']=='COMPLETE' and v['PASS'] and v['all_descendants_kernel_contained'] and v['resource_samples']>0
    assert not (q.R6OUT/'candidates').exists() and not (q.R6OUT/'MENTOR_APPROVAL.json').exists()

@pytest.mark.parametrize('fault',['third_train','missing_frame','resource','timeout','duplicate_frame'])
def test_actual_faults_stop_before_fourth_candidate(fault):
    p=folder('fault_'+fault);b=SyntheticBackend(p,fault)
    with pytest.raises(RuntimeError):run_queue(p,ROWS,b)
    state=q.read(p/'queue.json');assert state['completion']=='QUEUE_STOPPED_AT_CANDIDATE' and state['state'] in ('QUEUE_STOPPED','QUEUE_FAILED')
    assert state['current_candidate']=='K1_r16' and len(state['completed_candidates'])==2
    assert all(name not in {v['name'] for v in ROWS[3:]} for name,stage in b.started)
    assert not (p/'jobs/K2_r4').exists() and (p/'failure.json').is_file()
    original=q.digest(p/'queue.json')
    with pytest.raises(RuntimeError,match='DUPLICATE'):run_queue(p,ROWS,b)
    assert q.digest(p/'queue.json')==original

class FastBackend:
    def __init__(self,p,fault=None):self.p=p;self.fault=fault;self.started=[]
    def guard(self):return contextlib.nullcontext()
    def prepare(self):pass
    def admit(self,row,stage):
        if self.fault=='capacity' and row==ROWS[2]:raise RuntimeError('NEXT_STAGE_LAUNCH_CAPACITY_FAILED')
    def stage(self,row,stage):
        self.started.append((row['name'],stage))
        if self.fault=='request_stop' and row==ROWS[2]:q.write(self.p/'STOP_REQUESTED.json',{'reason':'USER_STOP'})
    def verify_stage(self,row,stage):
        if self.fault in ('seed','manifest','weights','history','nonfinite','exit','resource_log') and row==ROWS[2]:raise RuntimeError(self.fault.upper()+'_INVALID')
    def verify_candidate(self,row):return {'PASS':True}
    def finish(self):pass
    def abort(self,reason):pass

@pytest.mark.parametrize('fault',['capacity','request_stop','seed','manifest','weights','history','nonfinite','exit','resource_log'])
def test_state_machine_integrity_failure_propagation(fault):
    p=folder('state_'+fault);b=FastBackend(p,fault)
    with pytest.raises(RuntimeError):run_queue(p,ROWS,b)
    assert not any(n in {r['name'] for r in ROWS[3:]} for n,s in b.started)
    assert q.read(p/'queue.json')['state']=='QUEUE_STOPPED'

def test_duplicate_live_queue_refused_without_mutating_owner():
    p=folder('duplicate_live')
    with q.lease(p):
        with pytest.raises(FileExistsError):run_queue(p,ROWS,FastBackend(p))
        assert (p/'execution.lock').is_file() and not (p/'queue.json').exists()

def test_maxiter_100_legally_preserved():
    q.r6_imports();from model_ops import convergence
    import numpy as np
    assert convergence(np.linspace(-1000,-200,100))=='MAX_ITER_REACHED'

def test_two_scope_bound_missing_approval_start_rejected_before_real_data():
    p=subprocess.run([sys.executable,'-B','-X','utf8',str(HERE/'r7.py'),'start'],capture_output=True,text=True,encoding='utf8')
    assert p.returncode!=0 and 'MENTOR_APPROVAL_PENDING' in p.stderr
    assert not (q.R6OUT/'candidates').exists() and not (q.OUT/'queue.json').exists()

def toy_record():
    # Labelled synthetic validation of parser only. Never written/imported.
    return {'status':'MENTOR_APPROVED','scope':'R7_NINE_CANDIDATE_TRAIN_VALIDATE_ONLY','allowed_actions':['train','validate'],
            'candidate_order':ROWS,'mentor_instruction_reference':'SYNTHETIC_PARSER_FIXTURE_NOT_AUTHORIZATION','mentor_instruction_SHA256':'a'*64,
            'approved_UTC':'2020-01-01T00:00:00+00:00',**q.bindings(),'resource_contract':q.contract()['resource_contract'],
            'no_automatic_retry':True,'selection_authorized':False,'final_training_authorized':False,'target_access_authorized':False}

@pytest.mark.parametrize('field',['status','scope','allowed_actions','candidate_order','mentor_instruction_SHA256','approved_UTC','R7_package_manifest_SHA256','input_contract_SHA256','resource_contract','selection_authorized','final_training_authorized','target_access_authorized'])
def test_in_memory_parser_rejects_invalid_authorization(field):
    a=toy_record();a[field]=None
    with pytest.raises(RuntimeError):q.approval_values(a,'a'*64)
    assert not (q.OUT/'MENTOR_APPROVAL.json').exists()

def test_no_selection_final_target_dispatch_or_algorithm_replacement():
    tree=ast.parse((HERE/'r7.py').read_text(encoding='utf8'))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('complete_selection','selection_audit','automatic_select','selection_frozen') for n in ast.walk(tree))
    body=(HERE/'formal_backend.py').read_text(encoding='utf8')
    assert 'automatic_select(' not in body and 'low_memory import' not in body and 'fit_reference(' not in body
    bridge=(HERE/'worker_bridge.py').read_text(encoding='utf8')
    assert 'worker.main()' in bridge and 'TimingObserver' in bridge
    assert not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr in ('fit_reference','train','validate','fit') for t in n.targets) for n in ast.walk(ast.parse(bridge)))

def test_r6_entire_queue_global_lock_blocks_old_stage_start():
    # Isolated lease fixture; the frozen original function is unchanged.
    q.r6_imports();import runner,common
    original_out=runner.OUT
    p=folder('r6_lock_fixture');runner.OUT=p
    try:
        with runner.lock():
            with pytest.raises(RuntimeError,match='SINGLE_JOB_LOCK_PRESENT'):
                with runner.lock():pass
            assert (p/'pipeline.lock').is_file()
    finally:runner.OUT=original_out

def test_toy_numerical_firewall_in_source():
    assert 'TOY_REAL_DATA_READ_DENIED' in (HERE/'toy_worker.py').read_text(encoding='utf8')
    assert not (q.R6OUT/'SELECTION_MANIFEST.json').exists() and not (q.R6OUT/'final').exists()

def test_resource_audit_rejects_missing_series(tmp_path):
    from formal_backend import monitor_complete
    p=folder('missing_series');cfg=q.contract()['resource_contract']
    state={'state':'COMPLETE','PASS':True,'exit_code':0,'stop_reason':None,'all_descendants_kernel_contained':True,'actual_interpreter_PID':123,
           'limits':cfg['limits'],'blas_threads':6,'openmp_threads':1,'peak_RSS_bytes':1,'peak_private_commit_bytes':1,
           'minimum_available_physical_bytes':100*2**30,'minimum_remaining_commit_bytes':100*2**30,'resource_samples':1}
    q.write(p/'status.json',state);(p/'resources.jsonl').write_text('',encoding='utf8')
    with pytest.raises(RuntimeError,match='INCOMPLETE'):monitor_complete(p)

@pytest.mark.parametrize('K,rank',[(1,4),(4,16)])
def test_timing_observer_bitwise_math_parity_and_disjoint_accounting(K,rank):
    c,_,_,_=q.r6_imports();import model_ops as m
    import numpy as np,math
    from timing_observer import TimingObserver
    rng=np.random.default_rng(928716);x=(rng.normal(size=(96,32))+1j*rng.normal(size=(96,32)))/np.sqrt(2);x[:48]+=1.1
    row=next(v for v in ROWS if v['K']==K and v['rank']==rank)
    a=m.make_reference_model(K,rank);b=m.make_reference_model(K,rank)
    with c.threads(6,1):
        m.fit_reference(a,x,row['seed'])
        with TimingObserver() as observer:m.fit_reference(b,x,row['seed'])
    assert m.parameter_hashes(a)==m.parameter_hashes(b) and np.array_equal(a.L_all,b.L_all)
    t=observer.result
    assert t['exclusive_seconds']['initialization']>0 and t['exclusive_seconds']['training']>0
    assert math.isclose(sum(t['exclusive_seconds'].values()),t['wall_seconds_in_bridge'],abs_tol=1e-9)

def test_parent_firewall_allows_lease_but_denies_real_array_decoding():
    # Separate interpreter: irreversible audit hook never affects pytest state.
    p=folder('parent_firewall');script=p/'check.py'
    script.write_text('''from pathlib import Path
import sys,json
sys.path.insert(0,sys.argv[1])
import r7_common as q
c,r,s,w=q.r6_imports()
mock=Path(sys.argv[2]);mock.mkdir()
q.R6OUT=c.OUT=r.OUT=mock
from formal_backend import install_parent_firewall
install_parent_firewall()
with r.lock():
    assert (mock/'pipeline.lock').exists()
    for name in ('outputs/sp_u12b5b/selection_train.npy','outputs/sp_u12b2b_sealed_manifest.json'):
        try:
            with (q.ROOT/name).open('rb') as f:f.read(1)
        except (PermissionError,RuntimeError):pass
        else:raise AssertionError('Protected read unexpectedly admitted')
assert not (mock/'pipeline.lock').exists()
print('parent firewall + lease PASS; real values read 0')
''',encoding='utf8')
    result=subprocess.run([sys.executable,'-B','-X','utf8',str(script),str(HERE),str(p/'mock_R6')],capture_output=True,text=True,encoding='utf8')
    assert result.returncode==0,(result.stdout,result.stderr)

def test_toy_parameter_validity_rejects_nonfinite_and_negative_weights():
    c,_,_,_=q.r6_imports();import model_ops as m
    import numpy as np
    model=m.make_reference_model(1,1);model.D=3;model.N=12
    model.means=np.zeros((1,3),complex);model.lambdas=np.zeros((1,3,1),complex);model.psis=np.ones((1,3));model.amps=np.ones(1)
    model.covs=np.eye(3,dtype=complex)[None];model.inv_covs=model.covs.copy()
    assert m.parameter_validity(model)['PASS']
    model.amps[0]=-1
    with pytest.raises(RuntimeError,match='ILLEGAL_MODEL'):m.parameter_validity(model)
    model.amps[0]=1;model.means[0,0]=np.nan
    with pytest.raises(RuntimeError,match='ILLEGAL_MODEL'):m.parameter_validity(model)
