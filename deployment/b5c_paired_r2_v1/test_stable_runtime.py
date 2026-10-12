"""Focused final runtime guards; all task evidence below is synthetic."""
import ast,json,os
from datetime import datetime,timezone
from pathlib import Path
import pytest
import base as b
import control as c
import historical_concurrency as h
from test_concurrency import history,check,fingerprints

def closed(pid=None):
    return {'state':'COMPLETE','PASS':True,'completed_UTC':'2000-01-01T00:00:00+00:00',
            **({'supervisor_PID':pid} if pid is not None else {})}

def old_lock(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    os.utime(path,(946684700,946684700)) # New synthetic fixture only; never edits history.

def test_final_cross_namespace_reference_is_valid_and_unchanged(history):
    root,_,put=history;folder=root/'lab_runs/final_mfa_once';target=root/'lab_runs/r6_formal/final'
    put(target/'status.json',closed());put(folder/'status.json',{'state':'COMPLETE','PASS':True,'UTC':'2000-01-01T00:00:00+00:00'})
    put(folder/'active_worker.json',{'job':'lab_runs/r6_formal/final'})
    before=fingerprints(root/'lab_runs');report=check();assert report['PASS'],report
    assert any(r.get('resolved_job')=='lab_runs/r6_formal/final' for r in report['checks'])
    assert fingerprints(root/'lab_runs')==before

def test_absolute_registered_final_reference_allowed(history):
    root,_,put=history;put(root/'lab_runs/r6_formal/final/status.json',closed())
    put(root/'lab_runs/final_mfa_once/active_worker.json',{'job':str(root/'lab_runs/r6_formal/final')})
    assert check()['PASS']

@pytest.mark.parametrize('name',h.ROOTS[1:])
@pytest.mark.parametrize('lock',['execution.lock','pipeline.lock','audit_execution.lock'])
def test_retained_dead_owner_lock_with_matching_terminal_evidence_allowed(history,name,lock):
    root,_,put=history;folder=root/'lab_runs'/name;pid=os.getpid()
    put(folder/'status.json',closed(pid));old_lock(folder/lock,str(pid).encode())
    before=fingerprints(folder);report=check();assert report['PASS'],report
    assert fingerprints(folder)==before

@pytest.mark.parametrize('role,owner',[('FINAL_ONLY','final_mfa_once'),('R8_SELECTION_ONLY','r8_selection')])
def test_R6_cross_owner_lease_uses_correct_terminal_evidence(history,role,owner):
    root,_,put=history;pid=os.getpid();put(root/'lab_runs'/owner/'status.json',closed(pid))
    old_lock(root/'lab_runs/r6_formal/pipeline.lock',(role+' '+str(pid)).encode())
    assert check()['PASS']

def test_live_owner_is_blocked_despite_terminal_label(history):
    root,_,put=history;folder=root/'lab_runs/r6_formal';d=closed(os.getpid());d['completed_UTC']=datetime.now(timezone.utc).isoformat()
    put(folder/'status.json',d);old_lock(folder/'pipeline.lock',str(os.getpid()).encode())
    report=check();assert not report['PASS'];assert any(v['classification']=='ACTIVE' for v in report['blockers'])

def test_PID_reuse_cannot_validate_lock_written_after_old_completion(history):
    root,_,put=history;folder=root/'lab_runs/r6_formal';put(folder/'status.json',closed(os.getpid()))
    (folder/'pipeline.lock').write_text(str(os.getpid()),encoding='ascii')
    report=check();assert not report['PASS'];assert any('LOCK_MODIFIED_AFTER' in v['reason'] for v in report['blockers'])

def test_unknown_owner_or_missing_status_blocks(history):
    root,_,_=history;old_lock(root/'lab_runs/r6_formal/pipeline.lock',b'1234567')
    report=check();assert not report['PASS'];assert any(v['classification']=='UNKNOWN' for v in report['blockers'])

def test_R8_legacy_controller_without_recorded_PID_has_trusted_terminal_witness(history):
    root,_,put=history;folder=root/'lab_runs/r8_selection';put(folder/'job/status.json',closed())
    put(folder/'latest.json',{'job':str(folder/'job')})
    old_lock(folder/'execution.lock',str(os.getpid()).encode())
    assert check()['PASS']

def test_rehearsal_uses_real_condition_key_shape():
    assert isinstance(b.contract()['conditions'],dict)
    assert next(iter(b.contract()['conditions'])) in ('ID','Profile','Mobility','Composed','Boundary')

def test_dependency_initialization_precedes_guard_without_removing_guard():
    worker=(b.HERE/'worker.py').read_text(encoding='utf8')
    assert 'initialize_dependencies()' in worker and 'reference_modules()' in worker
    assert worker.index('c,_,_,wr=initialize_dependencies()')<worker.index('scope(job,True)')
    assert "'subprocess.Popen','os.system','os.exec'" in (b.HERE/'control.py').read_text(encoding='utf8')

def test_terminal_record_with_exact_live_creation_identity_blocks(history):
    root,_,put=history;from ctypes import wintypes as w
    import ctypes
    k=h.kernel();proc=k.OpenProcess(0x1000,False,os.getpid());values=[w.FILETIME() for _ in range(4)]
    try:assert k.GetProcessTimes(proc,*[ctypes.byref(v) for v in values])
    finally:k.CloseHandle(proc)
    ticks=(values[0].dwHighDateTime<<32)|values[0].dwLowDateTime
    put(root/'lab_runs/r7_sequential/queue.json',{'state':'QUEUE_COMPLETE','controller_PID':os.getpid(),
        'controller_identity':{'PID':os.getpid(),'creation_time_100ns':ticks},'completed_UTC':'2000-01-01T00:00:00+00:00'})
    assert not check()['PASS']

def test_all_known_namespace_outcomes_are_returned_together(history):
    root,_,put=history
    put(root/'lab_runs/r6_formal/status.json',{'state':'RUNNING'})
    put(root/'lab_runs/r8_selection/latest.json',{'job':'lab_runs/r8_selection/missing'})
    put(root/'lab_runs/final_mfa_once/active_worker.json',{'job':'lab_runs/r6_formal/final'})
    report=check();assert not report['PASS'];assert len(report['namespaces'])==9
    assert len(report['blockers'])>=3

def test_unknown_cross_namespace_reference_still_refused(history):
    root,_,put=history;put(root/'lab_runs/r6_formal/candidates/K1_r16/train/status.json',closed())
    put(root/'lab_runs/final_mfa_once/active_worker.json',{'job':'lab_runs/r6_formal/candidates/K1_r16/train'})
    assert not check()['PASS']

def test_R7_documented_candidate_reference_allowed(history):
    root,_,put=history;target='lab_runs/r6_formal/candidates/K1_r16/train'
    put(root/target/'status.json',closed());put(root/'lab_runs/r7_sequential/active_worker.json',{'job':target})
    assert check()['PASS']

def test_no_lab_rehearsal_means_no_formal_consumption(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'OUT',tmp_path/'new_R2')
    monkeypatch.setattr(c,'preflight',lambda *a:pytest.fail('FORMAL_PREFLIGHT_REACHED'))
    with pytest.raises(FileNotFoundError):c.start()
    assert not b.OUT.exists()

def test_personal_switch_never_authorizes_Run(monkeypatch):
    monkeypatch.setattr(__import__('sys'),'argv',['control.py','Run','--formal','--personal-selftest'])
    with pytest.raises(RuntimeError,match='SYNTHETIC_ONLY'):c.main()

def test_stop_targets_actual_rehearsal_without_consuming_formal(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'OUT',tmp_path/'R2');p=b.OUT/'synthetic/rehearsal_test'
    p.mkdir(parents=True);b.write(p/'status.json',{'state':'RUNNING'},True)
    assert c.stop()['STOP_requested'];assert (p/'STOP_REQUESTED.json').exists()
    assert not(b.OUT/'FORMAL_START_CONSUMED.json').exists()

def test_failed_history_check_can_export_all_blockers(tmp_path,monkeypatch):
    # Unknown history must remain exportable after a failed LabCheck,
    # without implying launch readiness or changing historical records.
    monkeypatch.setattr(b,'OUT',tmp_path/'R2');b.OUT.mkdir()
    b.write(b.OUT/'technical/test.json',{'PASS':False,'blockers':['one','two']},True)
    result=c.export();assert result['files']==2;assert not(b.OUT/'FORMAL_START_CONSUMED.json').exists()

def test_worker_formal_branch_byte_semantics_unchanged():
    baseline=b.ROOT/'deployment/b5c_paired_r1_v1/worker.py'
    # R1 and original R2 have different formal gate names; compare the scientific
    # formal branch after initialization against original R2 preserved separately.
    tree=ast.parse((b.HERE/'worker.py').read_text(encoding='utf8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    attempt=next(n for n in main.body if isinstance(n,ast.Try))
    branch=attempt.body[0]
    formal=ast.unparse(ast.Module(body=branch.orelse,type_ignores=[]))
    assert 'scope(job, True)' in formal and 'result = evaluate(job)' in formal and "with c.threads(6, 1)" in formal
    assert 'rehearsal' not in formal
