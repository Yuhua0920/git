"""Synthetic-only rule, authorization, raw-record, freeze and firewall tests."""
from pathlib import Path
import sys,json,subprocess,copy,hashlib,ast
import pytest
import selection_core as a
from authorization import check,GRANTS,ACTION

def record():
    instruction=b'SYNTHETIC_TEST_ONLY'
    return {'format':'R8_SELECTION_ONLY_HUMAN_ATTESTATION_V1','status':'SYNTHETIC_TEST_APPROVED','scope':ACTION,'allowed_actions':[ACTION],
      'grants':GRANTS.copy(),'bindings':{'synthetic_binding':'a'*64},'approval_id':'SYNTHETIC_ONLY','approved_by':'SYNTHETIC_ONLY','approved_UTC':'2000-01-01T00:00:00+00:00',
      'mentor_instruction_reference':'SYNTHETIC_ONLY','mentor_instruction_SHA256':hashlib.sha256(instruction).hexdigest(),'no_automatic_retry':True,'maximum_executions':1},instruction
def test_specific_synthetic_authorization_accepted_only_in_test_validator():
    v,i=record();assert check(v,v['bindings'],i,True)['PASS']
    with pytest.raises(RuntimeError,match='NOT_AUTHORIZED'):check(v,v['bindings'],i,False)
@pytest.mark.parametrize('field',list(GRANTS))
def test_every_excess_or_missing_grant_rejected(field):
    v,i=record();v['grants'][field]=not v['grants'][field]
    with pytest.raises(RuntimeError,match='GRANTS'):check(v,v['bindings'],i,True)
@pytest.mark.parametrize('action',['train','validate','final','target','unseal','CompleteSelection'])
def test_no_action_alias_can_expand_selection_authority(action):
    v,i=record();v['allowed_actions']=[ACTION,action]
    with pytest.raises(RuntimeError,match='ACTION_SCOPE'):check(v,v['bindings'],i,True)
@pytest.mark.parametrize('fault',['pending','timestamp','future','binding','instruction','twice','schema','integer_grant'])
def test_human_record_bindings_and_once_policy(fault):
    v,i=record();expected=v['bindings'].copy()
    if fault=='pending':v['status']='PENDING_NOT_APPROVED'
    elif fault=='timestamp':v['approved_UTC']=None
    elif fault=='future':v['approved_UTC']='2999-01-01T00:00:00Z'
    elif fault=='binding':v['bindings']={'synthetic_binding':'b'*64}
    elif fault=='instruction':i=b'changed'
    elif fault=='twice':v['maximum_executions']=2
    elif fault=='schema':v['extra_grant']=True
    else:v['grants']['selection_authorized']=1
    with pytest.raises(RuntimeError):check(v,expected,i,True)
def test_old_R7_train_validate_authority_rejected():
    old={'status':'MENTOR_APPROVED','scope':'R7_NINE_CANDIDATE_TRAIN_VALIDATE_ONLY','allowed_actions':['train','validate'],
         'selection_authorized':False,'final_training_authorized':False,'target_access_authorized':False}
    with pytest.raises(RuntimeError,match='SCHEMA'):check(old,{},b'SYNTHETIC',False)
def test_pending_template_contains_no_approved_identity_time_or_grant():
    v=a.read(a.HERE/'authorization_PENDING_TEMPLATE.json');assert v['status']=='PENDING_NOT_APPROVED'
    assert v['approved_by']is None and v['approved_UTC']is None and not any(v['grants'].values())
    with pytest.raises(RuntimeError):check(v,{},b'',False)

@pytest.mark.parametrize('case',['success','tie','repeat','missing_candidate','corrupt_hash','bad_model_state','wrong_seed',
                              'duplicate_identity','missing_cell','wrong_cell_metric','stop','concurrent','existing_sha','unauthorized','legacy_r7','firewall'])
def test_raw_synthetic_selection_and_negative_gates(tmp_path,case):
    p=tmp_path/'f';r=subprocess.run([sys.executable,'-B','-X','utf8',str(a.HERE/'toy_case.py'),str(p),case],capture_output=True,text=True,encoding='utf8')
    assert r.returncode==0,r.stdout+r.stderr
    v=json.loads(r.stdout.strip().splitlines()[-1]);assert v['PASS']and v['synthetic_only']and not v['real_selection_executed']and not v['final_model_trained']and v['target_access']==0
    if case in('success','tie','repeat'):assert v['R6_selection_frozen_compatibility']
    elif case!='firewall':
        reasons={'missing_candidate':'FROZEN_INPUT_OR_RESULT_HASH_DRIFT','corrupt_hash':'FROZEN_INPUT_OR_RESULT_HASH_DRIFT',
          'bad_model_state':'ILLEGAL_MODEL_STATE','wrong_seed':'TRAINED_MODEL_SEED_OR_DIMENSION_DRIFT',
          'duplicate_identity':'UNEXPECTED_OR_DUPLICATE_VALIDATION_IDENTITY','missing_cell':'VALIDATION_CELL_INVENTORY_INVALID',
          'wrong_cell_metric':'POOLED_CELL_RECONSTRUCTION_DRIFT','stop':'STOP_PRESENT_NO_RETRY','concurrent':'CONCURRENT_OR_STALE_LOCK',
          'existing_sha':'SELECTION_OR_FINAL_ALREADY_EXISTS_NO_OVERWRITE','unauthorized':'R8_SELECTION_NOT_AUTHORIZED','legacy_r7':'R8_AUTHORIZATION_SCHEMA_INVALID'}
        assert reasons[case]in v['expected_rejection']['message'],v['expected_rejection']
    else:assert all(v['firewall'].values())
def test_frozen_rule_functions_sources_have_not_changed():
    n='deployment/r6_formal_v1/model_ops.py';assert hashlib.sha256((a.ROOT/n).read_bytes()).hexdigest()==a.cfg()['dependencies'][n]['sha256']
    tree=ast.parse((a.HERE/'selection_core.py').read_text(encoding='utf8'))
    labels=[node.func.attr for node in ast.walk(tree)if isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute)]
    assert 'selection_audit'in labels and 'complete_selection'not in labels
    assert not any(x in labels for x in('fit','fit_reference','train','validate','load_model','inference_smoke'))
def test_production_entry_exposes_no_synthetic_approval_or_training_action():
    import cli
    assert cli.ACTIONS==('verify','toy-tests','authorization-bindings','selection-preflight','complete-selection-once','monitor-toy','status','export','stop')
    assert not any('synthetic' in x for x in cli.ACTIONS)

def test_frozen_pooled_ratio_is_ratio_of_sums_not_mean_of_frame_ratios():
    _,ops,*_=a.modules()
    assert ops.pooled_ratio([1.,9.],[1.,3.])==2.5
    assert ops.pooled_ratio([1.,9.],[1.,3.])!=(1.+3.)/2

def test_frozen_macro_linear_metric_not_db_and_no_complexity_penalty():
    _,ops,*_=a.modules();rows=[]
    for row in a.cfg()['candidate_order']:
        cells=[100.]*6
        if row['name']=='K1_r4':cells=[.01,100.,.01,.01,.01,.01]
        if row['name']=='K4_r16':cells=[1.]*6
        rows.append(row|{'macro_selection_score':ops.macro_score(cells)})
    assert ops.automatic_select(rows)['name']=='K4_r16'

def test_frozen_exact_equality_uses_K_then_rank_and_no_tolerance():
    _,ops,*_=a.modules();rows=[v|{'macro_selection_score':2.}for v in a.cfg()['candidate_order']]
    rows[2]['macro_selection_score']=1.;rows[3]['macro_selection_score']=1.
    assert ops.automatic_select(rows)['name']=='K1_r16'
    rows[3]['macro_selection_score']=__import__('numpy').nextafter(1.,0.)
    assert ops.automatic_select(rows)['name']=='K2_r4'

def test_windows_status_publication_waits_for_short_read_handle():
    import ctypes,ctypes.wintypes as w,threading,time,uuid
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,ctypes.c_void_p,w.DWORD,w.DWORD,w.HANDLE];kernel.CreateFileW.restype=w.HANDLE
    kernel.CloseHandle.argtypes=[w.HANDLE];kernel.CloseHandle.restype=w.BOOL
    path=a.OUT/'synthetic_status_tests'/uuid.uuid4().hex/'status.json';a.write(path,{'state':'FIRST','synthetic_only':True})
    handle=kernel.CreateFileW(str(path),0x80000000,1,None,3,0,None)
    assert handle not in(None,ctypes.c_void_p(-1).value)
    released=[]
    def close_later():time.sleep(.1);released.append(bool(kernel.CloseHandle(handle)))
    thread=threading.Thread(target=close_later);thread.start()
    try:a.write(path,{'state':'SECOND','synthetic_only':True})
    finally:thread.join(timeout=5)
    assert released==[True]and a.read(path)['state']=='SECOND'and not path.with_suffix('.json.tmp').exists()

def test_permanent_status_denial_is_bounded_not_a_job_retry(monkeypatch):
    import uuid
    path=a.OUT/'synthetic_status_tests'/uuid.uuid4().hex/'status.json';a.write(path,{'state':'ORIGINAL'})
    original=a.sha(path);calls=[]
    def denied(*args):
        calls.append(args);error=PermissionError('SYNTHETIC_STATUS_SHARE_DENIAL');error.winerror=5;raise error
    monkeypatch.setattr(a.os,'replace',denied)
    with pytest.raises(PermissionError):a.write(path,{'state':'SHOULD_NOT_PUBLISH'})
    assert len(calls)==101 and a.sha(path)==original

def test_exclusive_freeze_metadata_creation_never_retries_or_overwrites(monkeypatch):
    import uuid
    path=a.OUT/'synthetic_status_tests'/uuid.uuid4().hex/'exclusive_marker.json';a.write(path,{'original':True},True);digest=a.sha(path)
    def never(*args):raise AssertionError('exclusive creation must never use replace/retry')
    monkeypatch.setattr(a.os,'replace',never)
    with pytest.raises(FileExistsError):a.write(path,{'original':False},True)
    assert a.sha(path)==digest
