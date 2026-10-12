"""Synthetic receipts and native Windows leases only; never calls evaluation."""
import ctypes,json,msvcrt,os,shutil,subprocess,sys
from ctypes import wintypes as w
from datetime import datetime,timezone
from pathlib import Path
import pytest
import base as b
import historical_concurrency as h

@pytest.fixture
def history(tmp_path,monkeypatch):
    original=b.ROOT;root=tmp_path/'synthetic lab with spaces';root.mkdir()
    here=root/'deployment/r5_128gb_v1';here.mkdir(parents=True)
    for name in ('package_manifest.json','protocol.json','driver.py','supervisor.py','Start-R5.ps1','common.py'):
        shutil.copyfile(original/'deployment/r5_128gb_v1'/name,here/name)
    monkeypatch.setattr(b,'ROOT',root)
    folder=root/'lab_runs/r5_128gb';folder.mkdir(parents=True)
    def put(path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data),encoding='utf8')
    now=datetime.now(timezone.utc).isoformat()
    for stage in h.STAGES:
        job=folder/('synthetic_'+stage);job.mkdir();(job/'stage_result.json').write_text('SYNTHETIC_OPAQUE_RESULT_NO_SCIENTIFIC_DATA')
        record={'state':'COMPLETE','PASS':True,'stage':stage,'job':job.relative_to(root).as_posix(),'completed_UTC':now,
                'package_manifest_SHA256':h.R5_MANIFEST_SHA,'protocol_SHA256':b.sha(here/'protocol.json'),'result_sha256':b.sha(job/'stage_result.json')}
        put(folder/(stage+'_stage.json'),record);put(job/'status.json',record)
    (folder/'execution.lock').write_bytes(b'0')
    put(folder/'latest_launch.json',{'state':'RUNNING','stage':'smoke15000','job':'lab_runs/r5_128gb/synthetic_smoke15000'})
    put(folder/'latest.json',{'job':'lab_runs/r5_128gb/synthetic_smoke15000','stage':'smoke15000'})
    return root,folder,put

def check():
    result=h.inspect()
    try:return result.report
    finally:result.close()

def fingerprints(folder):
    return {p.relative_to(folder).as_posix():(b.sha(p),p.stat().st_size,p.stat().st_mtime_ns)
            for p in folder.rglob('*') if p.is_file()}

def test_original_false_positive_reproduced_and_new_history_allowed(history):
    _,folder,_=history;before=fingerprints(folder)
    with pytest.raises(RuntimeError,match='CONCURRENT_SCIENTIFIC_TASK'):
        b.require(not(folder/'execution.lock').exists(),'CONCURRENT_SCIENTIFIC_TASK:lab_runs/r5_128gb/execution.lock')
    report=check();assert report['PASS'],report
    assert any(v.get('verdict')=='COMPLETED_RETAINED_LOCK_IDLE' for v in report['checks'])
    assert fingerprints(folder)==before

def test_real_msvcrt_exclusive_lock_is_detected_despite_complete_records(history):
    _,folder,_=history;before=fingerprints(folder)
    with (folder/'execution.lock').open('r+b') as f:
        msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:
            report=check();assert not report['PASS']
            assert any('BYTE_LOCK_ACTIVE' in x['reason'] for x in report['blockers'])
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
    assert fingerprints(folder)==before

def test_readonly_lease_blocks_future_R5_exclusive_start_until_release(history):
    _,folder,_=history;before=fingerprints(folder);result=h.inspect();assert result.report['PASS']
    try:
        with (folder/'execution.lock').open('r+b') as f:
            with pytest.raises(OSError):msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
    finally:result.close()
    with (folder/'execution.lock').open('r+b') as f:
        msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
    assert fingerprints(folder)==before

@pytest.mark.parametrize('stage',h.STAGES)
def test_missing_completion_is_not_waived(history,stage):
    _,folder,_=history;(folder/(stage+'_stage.json')).unlink();assert not check()['PASS']

@pytest.mark.parametrize('field,value',[('state','RUNNING'),('PASS',False),('protocol_SHA256','0'*64),('package_manifest_SHA256','0'*64),('result_sha256','0'*64),('completed_UTC','')])
def test_unproven_stage_status_hash_time_or_freeze_rejected(history,field,value):
    _,folder,put=history;p=folder/'cpu_stage.json';d=b.read(p);d[field]=value;put(p,d);assert not check()['PASS']

def test_changed_result_bytes_rejected_without_decoding(history):
    _,folder,_=history;(folder/'synthetic_cpu/stage_result.json').write_bytes(b'CHANGED');assert not check()['PASS']

def test_unknown_lock_payload_not_treated_as_PID_zero(history):
    _,folder,_=history;(folder/'execution.lock').write_bytes(b'123');assert not check()['PASS']

def test_authenticated_lock_source_required(history):
    root,_,_=history;(root/'deployment/r5_128gb_v1/driver.py').write_text('wrong');assert not check()['PASS']

@pytest.mark.parametrize('root',h.ROOTS[1:])
@pytest.mark.parametrize('lock',['execution.lock','pipeline.lock','audit_execution.lock'])
def test_all_other_presence_locks_remain_strict(history,root,lock):
    project,_,_=history;p=project/'lab_runs'/root/lock;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'0')
    report=check();assert not report['PASS'];assert any(str(p)==x['item'] for x in report['blockers'])

def test_all_failures_reported_in_one_scan(history):
    root,_,_=history
    for name in ('r6_formal','r7_sequential','b5c_paired_r1'):
        p=root/'lab_runs'/name/'execution.lock';p.parent.mkdir();p.write_bytes(b'0')
    report=check();assert not report['PASS'];assert len(report['blockers'])==3

@pytest.mark.parametrize('state',['RUNNING','UNKNOWN',None,'ANALYZING','QUEUE_RUNNING'])
def test_worker_ambiguity_rejected(history,state):
    root,_,put=history;folder=root/'lab_runs/r6_formal';put(folder/'active_worker.json',{'job':'lab_runs/r6_formal/job'})
    put(folder/'job/status.json',{'state':state});assert not check()['PASS']

@pytest.mark.parametrize('job',['../escape','C:/unrelated','lab_runs/r6_formal/missing','lab_runs/r5_128gb/synthetic_cpu'])
def test_foreign_or_missing_job_pointer_rejected(history,job):
    root,_,put=history;put(root/'lab_runs/r6_formal/active_worker.json',{'job':job});assert not check()['PASS']

def test_running_launcher_before_lock_acquisition_rejected(history):
    _,folder,put=history;put(folder/'new_launch/status.json',{'state':'RUNNING'})
    assert not check()['PASS']

def test_live_PID_rejected(history):
    _,folder,put=history;put(folder/'old_worker/status.json',{'state':'COMPLETE','supervisor_PID':os.getpid(),'completed_UTC':datetime.now(timezone.utc).isoformat()})
    assert not check()['PASS']

def test_native_PID_reuse_is_proven_by_creation_time():
    assert h.process_state(os.getpid(),'2000-01-01T00:00:00+00:00')=='REUSED_AFTER_COMPLETION'
    assert h.process_state(os.getpid(),datetime.now(timezone.utc).isoformat())=='ACTIVE'

def test_permission_uncertainty_never_waived(history,monkeypatch):
    _,folder,put=history;put(folder/'old_worker/status.json',{'state':'COMPLETE','supervisor_PID':1,'completed_UTC':'2026-10-08T00:00:00+00:00'})
    def deny(*a):raise RuntimeError('PROCESS_ACTIVITY_UNCONFIRMED:5')
    monkeypatch.setattr(h,'process_state',deny);assert not check()['PASS']

def test_persistent_worker_marker_to_closed_worker_allowed(history):
    _,folder,put=history;put(folder/'old_worker/status.json',{'state':'COMPLETE','PASS':True,'completed_UTC':'2026-10-08T00:00:00+00:00'})
    put(folder/'active_worker.json',{'job':'lab_runs/r5_128gb/old_worker'})
    assert check()['PASS']

def test_queue_running_without_presence_lock_rejected(history):
    root,_,put=history;put(root/'lab_runs/r7_sequential/queue.json',{'state':'QUEUE_RUNNING'})
    assert not check()['PASS']

def test_old_failed_B5C_STOP_and_consumed_preserved(history):
    root,_,put=history;old=root/'lab_runs/b5c_paired';put(old/'STOP.json',{'state':'STOP'});put(old/'FORMAL_START_CONSUMED.json',{'old':True})
    before=fingerprints(old);assert check()['PASS'];assert fingerprints(old)==before

def test_record_change_during_scan_rejected(history,monkeypatch):
    _,folder,_=history;original=b.sha;p=folder/'latest_launch.json';calls=0
    def changing(q):
        nonlocal calls
        if Path(q)==p:
            calls+=1
            if calls>1:return '0'*64
        return original(q)
    monkeypatch.setattr(b,'sha',changing);assert not check()['PASS']

def test_readonly_action_does_not_create_R2_output(history,monkeypatch):
    root,_,_=history;import control
    monkeypatch.setattr(b,'OUT',root/'lab_runs/b5c_paired_r2');monkeypatch.setattr(b,'verify',lambda:True)
    assert control.concurrency_audit()['PASS'];assert not b.OUT.exists()

def test_native_job_members_still_block_completed_marker(history):
    _,folder,put=history;k=h.kernel();name='Local\\SP_U12B5_SYNTHETIC_'+os.urandom(8).hex()
    k.CreateJobObjectW.argtypes=[w.LPVOID,w.LPCWSTR];k.CreateJobObjectW.restype=w.HANDLE
    k.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];k.AssignProcessToJobObject.restype=w.BOOL
    job=k.CreateJobObjectW(None,name);assert job
    proc=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        assert k.AssignProcessToJobObject(job,int(proc._handle)),ctypes.get_last_error()
        assert h.job_active(name)
        put(folder/'old_worker/status.json',{'state':'COMPLETE','JobObject_name':name,'completed_UTC':'2026-10-08T00:00:00+00:00'})
        assert not check()['PASS']
    finally:proc.terminate();proc.wait(timeout=10);k.CloseHandle(job)
