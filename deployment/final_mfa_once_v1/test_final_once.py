"""New launch/permission seam tests only; no prior scientific suites rerun."""
from pathlib import Path
import copy,json,sys,subprocess,types,hashlib
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import final_once as a

def synthetic(p):
    return {'PASS':True,'target_access':0,'selected':copy.deepcopy(p['selected']),
            'candidates':[{'name':f'K{k}_r{r}'}for k in(1,2,4)for r in(4,8,16)],
            'coverage':{'scored_frame_predictions':32400},'execution_package_sha256':p['R6_package_sha256'],
            'input_contract_sha256':p['R6_input_contract_sha256'],'R8_bindings':p['R8_bindings'],
            'R8_selection_authorization':{'PASS':True,'final_training_authorized':False}}

def test_selection_read_only_rule(monkeypatch):
    p=a.contract();v=synthetic(p);s=a.selection_check(v,p['selection_sha256'],p['selection_sha256']+'\n')
    assert s['K']==1 and s['rank']==16 and s['final_seed']==4143049057

@pytest.mark.parametrize('key,value',[('K',2),('rank',8),('final_seed',1)])
def test_wrong_selection_refused(monkeypatch,key,value):
    p=a.contract();v=synthetic(p);v['selected'][key]=value
    with pytest.raises(RuntimeError,match='STRUCTURE_OR_SEED'):a.selection_check(v,p['selection_sha256'],p['selection_sha256'])

def test_bad_selection_hash_refused():
    p=a.contract()
    with pytest.raises(RuntimeError,match='SHA256'):a.selection_check(synthetic(p),'0'*64,p['selection_sha256'])

@pytest.mark.parametrize('old',[{'status':'MENTOR_APPROVED','train_authorized':True,'final_training_authorized':False},
                              {'scope':'FROZEN_STRUCTURE_SELECTION_ONCE','final_training_authorized':False}])
def test_old_approval_cannot_authorize_final(old):
    with pytest.raises(RuntimeError,match='FINAL_ONLY'):a.check_confirmation(old)

@pytest.mark.parametrize('confirm,operator',[(False,'operator'),(True,''),(None,'operator')])
def test_actual_operator_confirmation_required(confirm,operator):
    with pytest.raises(RuntimeError,match='EXPLICIT_OPERATOR'):a.confirmation(operator,confirm,'synthetic-sha')

@pytest.mark.parametrize('role',['train','validate','selection','target'])
def test_nonfinal_role_refused(role):
    with pytest.raises(RuntimeError,match='ONLY_FINAL'):
        with a.final_numeric(role):pass

def test_once_and_existing_model_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a,'OUT',tmp_path/'new');monkeypatch.setattr(a,'R6',tmp_path/'r6');monkeypatch.setattr(a,'FINAL',tmp_path/'r6/final')
    a.OUT.mkdir();(a.OUT/'FINAL_START_CONSUMED.json').write_text('{}')
    with pytest.raises(RuntimeError,match='CONSUMED'):a.absent()
    (a.OUT/'FINAL_START_CONSUMED.json').unlink();a.FINAL.mkdir(parents=True)
    with pytest.raises(RuntimeError,match='FINAL_ALREADY'):a.absent()

def test_r6_fit_and_monitor_code_objects_unchanged():
    c,w,s,_,_=a.modules();f=a.fit_callable(w.train);m=a.monitor_callable(s.run_worker)
    assert f.__code__ is w.train.__code__ and m.__code__ is s.run_worker.__code__
    assert {k for k in w.train.__globals__ if f.__globals__[k]is not w.train.__globals__[k]}=={'numeric_inputs'}
    assert {k for k in s.run_worker.__globals__ if m.__globals__[k]is not s.run_worker.__globals__[k]}=={'HERE','write'}
    assert c.approval.__module__=='common'

def test_io_firewall_subprocess(tmp_path):
    # A disposable process is necessary: CPython audit hooks cannot be removed.
    root=tmp_path/'root';root.mkdir();files=['outputs/sp_u12b5b/full_retraining.npy','outputs/sp_u12b5b/validation_truth.npy','outputs/sp_u12b2b/secret.json','outputs/sp_u12b3h/secret.npy','reference/frozen.py','lab_runs/r7_sequential/MENTOR_APPROVAL.json']
    for n in files:
        p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'synthetic')
    code=r'''
import sys,types,pathlib
sys.path.insert(0,sys.argv[1]);import final_once as a
r=pathlib.Path(sys.argv[2]);a.ROOT=r;a.OUT=r/'lab_runs/final_mfa_once';a.R6=r/'lab_runs/r6_formal';a.FINAL=a.R6/'final';a.HERE=r/'deployment/final_mfa_once_v1'
a.FINAL.mkdir(parents=True);a.OUT.mkdir(parents=True);(a.FINAL/'model').mkdir()
sys.modules['common']=types.SimpleNamespace(_numeric=set())
a.contract=lambda:{'code_hashes':{'reference/frozen.py':'synthetic'}}
a.guard(a.FINAL)
denied=[]
def bad(n,mode):
 try:
  with(r/n).open(mode)as f:f.read()if mode=='rb'else f.write(b'x')
 except RuntimeError:denied.append(n);return
 raise AssertionError('UNAUTHORIZED_ACCESS_SUCCEEDED:'+n)
for n in ['outputs/sp_u12b5b/validation_truth.npy','outputs/sp_u12b2b/secret.json','outputs/sp_u12b3h/secret.npy','lab_runs/r7_sequential/MENTOR_APPROVAL.json']:bad(n,'rb')
bad('outputs/sp_u12b5b/full_retraining.npy','rb')
a._numeric=True;assert(r/a.FULL).read_bytes()==b'synthetic';a._numeric=False
bad('reference/frozen.py','wb')
p=a.FINAL/'model/means.npy'
with p.open('xb')as f:f.write(b'synthetic')
bad('lab_runs/r6_formal/final/model/means.npy','wb')
print('PASS: 7 denied accesses; authorized final lease and exclusive new model write only')
'''
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,str(a.HERE),str(root)],capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    assert (root/'reference/frozen.py').read_bytes()==b'synthetic'
    assert '7 denied' in p.stdout

def test_synthetic_preflight_does_not_validate_or_select(tmp_path,monkeypatch):
    import numpy as np
    r=tmp_path/'lab';r.mkdir();p=copy.deepcopy(a.contract());v=synthetic(p)
    data=r/a.FULL;data.parent.mkdir(parents=True);np.save(data,np.zeros((3,4),dtype=np.complex128),allow_pickle=False)
    p['full_data']={'sha256':a.sha(data),'shape':[3,4],'dtype':'<c16','fortran_order':False}
    r6=r/'lab_runs/r6_formal';r6.mkdir(parents=True)
    raw=(json.dumps(v)+'\n').encode();p['selection_sha256']=hashlib.sha256(raw).hexdigest();(r6/'SELECTION_MANIFEST.json').write_bytes(raw);(r6/'SELECTION_MANIFEST.sha256').write_text(p['selection_sha256'])
    rows=[{'name':f'K{k}_r{rank}'}for k in(1,2,4)for rank in(4,8,16)]
    for x in rows:
        for role in('train','validate'):
            q=r6/'candidates'/x['name']/role;q.mkdir(parents=True);(q/'status.json').write_text(json.dumps({'state':'COMPLETE','PASS':True,'actual_interpreter_PID':1}))
            (q/'resources.jsonl').write_text(json.dumps({'process_tree':{'members':[{'PID':1,'creation_time_100ns':1}]}})+'\n')
    q=r/'lab_runs/r7_sequential';q.mkdir(parents=True);(q/'queue.json').write_text(json.dumps({'state':'QUEUE_COMPLETE','completion':'QUEUE_COMPLETE_9_OF_9','completed_candidates':list(range(9)),'controller_identity':{'PID':1,'creation_time_100ns':1}}))
    q=r/'deployment/r6_formal_v1';q.mkdir(parents=True);(q/'input_contract.json').write_text(json.dumps({'candidate_order':rows}))
    (r6/'preflight.json').write_text(json.dumps({'PASS':True,'integrity':{'package_manifest_SHA256':p['R6_package_sha256']}}))
    q=r6/'preparation_personal';q.mkdir();(q/'toy_test_report.json').write_text(json.dumps({'PASS':True,'tested_package_manifest_sha256':p['R6_package_sha256']}))
    monkeypatch.setattr(a,'ROOT',r);monkeypatch.setattr(a,'R6',r6);monkeypatch.setattr(a,'FINAL',r6/'final');monkeypatch.setattr(a,'OUT',r/'lab_runs/final_mfa_once');monkeypatch.setattr(a,'contract',lambda:p)
    monkeypatch.setattr(a,'verify',lambda:{'PASS':True});monkeypatch.setattr(a,'live_identity',lambda*x:False)
    c=types.SimpleNamespace(npy_header=lambda x:{'shape':[3,4],'dtype':'<c16','fortran_order':False},environment=lambda:{'PASS':True})
    wr=types.SimpleNamespace(hardware=lambda:{},capacity=lambda x:{'PASS':True,'reasons':[]})
    monkeypatch.setattr(a,'modules',lambda:(c,None,None,None,wr))
    result=a.preflight();assert result['PASS']and result['validation_recomputed']is False
    assert not data.with_name('validation_truth.npy').exists()

def test_no_formal_dispatch_without_operator(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'OUT',tmp_path/'new')
    with pytest.raises(RuntimeError,match='EXPLICIT_OPERATOR'):a.start(None,False)
    assert not a.OUT.exists()

def test_synthetic_freeze_under_final_only_guard(tmp_path):
    # Only mocked bookkeeping outputs in a disposable synthetic project.
    # No MFA initialization/fit, real selection or validation is performed.
    code=r'''
import sys,pathlib,json,types
sys.path.insert(0,sys.argv[1]);import final_once as a
r=pathlib.Path(sys.argv[2]);r.mkdir();p=a.contract();a.ROOT=r;a.OUT=r/'lab_runs/final_mfa_once';a.R6=r/'lab_runs/r6_formal';a.FINAL=a.R6/'final';a.HERE=r/'deployment/final_mfa_once_v1'
a.FINAL.mkdir(parents=True);a.OUT.mkdir(parents=True);a.HERE.mkdir(parents=True);(a.FINAL/'model').mkdir();(a.HERE/'package_manifest.json').write_text('{}')
data=r/a.FULL;data.parent.mkdir(parents=True);data.write_bytes(b'SYNTHETIC_ONLY_NO_TRAINING_ARRAY');p['full_data']['sha256']=a.sha(data)
p['code_hashes']={};a.contract=lambda:p;a.verify=lambda:{'PASS':True};sys.modules['common']=types.SimpleNamespace()
selected=dict(p['selected']);v={'PASS':True,'target_access':0,'selected':selected,'candidates':[{}]*9,'coverage':{'scored_frame_predictions':32400},'execution_package_sha256':p['R6_package_sha256'],'input_contract_sha256':p['R6_input_contract_sha256'],'R8_bindings':p['R8_bindings'],'R8_selection_authorization':{'PASS':True}}
(a.R6/'SELECTION_MANIFEST.json').write_text(json.dumps(v));h=a.sha(a.R6/'SELECTION_MANIFEST.json');p['selection_sha256']=h;(a.R6/'SELECTION_MANIFEST.sha256').write_text(h)
receipt={'SELECTION_MANIFEST_SHA256':h,'synthetic_only':True};(a.FINAL/'operator_confirmation.json').write_text(json.dumps(receipt))
m={'N':15000,'D':4032,'K':1,'rank':16,'context':{'fresh_fit':True,'candidate_checkpoint_used':False},'synthetic_only':True}
(a.FINAL/'model/model_manifest.json').write_text(json.dumps(m));(a.FINAL/'training_result.json').write_text('{}')
result={'PASS':True,'seed':4143049057,'serialization_roundtrip':True,'all_components_smoke':{'PASS':True},'model_manifest_sha256':a.sha(a.FINAL/'model/model_manifest.json')}
a.guard(a.FINAL);a.freeze(result,receipt)
assert a.status()['final_frozen']
digest=a.sha(a.R6/'FINAL_MODEL_MANIFEST.json');assert(a.R6/'FINAL_MODEL_MANIFEST.sha256').read_text().strip()==digest
try:a.freeze(result,receipt)
except FileExistsError:pass
else:raise AssertionError('DUPLICATE_FREEZE_ALLOWED')
assert a.sha(a.R6/'FINAL_MODEL_MANIFEST.json')==digest
print('PASS: synthetic exclusive freeze, post-publication SHA read and duplicate refusal')
'''
    result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,str(a.HERE),str(tmp_path/'synthetic_root')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
