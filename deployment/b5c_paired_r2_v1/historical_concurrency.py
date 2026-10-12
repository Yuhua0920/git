"""Native activity and trusted terminal records; no historical-file mutation."""
import ctypes,json,os,re
from ctypes import wintypes as w
from datetime import datetime
from pathlib import Path
import base as b

ROOTS=('r5_128gb','r6_formal','r7_sequential','r8_selection','final_mfa_once',
       'b5c_paired','b5c_paired_r1','r7_evidence_audit','r7_evidence_audit_r1')
R5_MANIFEST_SHA='3886a6a112175798e291a2ccb18d728198c552149fb898ee675a93849097906d'
STAGES=('preflight','cpu','smoke12000','smoke15000')
TERMINAL={'COMPLETE','STOP','FAILED','QUEUE_COMPLETE','QUEUE_STOPPED','QUEUE_FAILED'}

class OVERLAPPED(ctypes.Structure):
    _fields_=[('Internal',ctypes.c_size_t),('InternalHigh',ctypes.c_size_t),
              ('Offset',w.DWORD),('OffsetHigh',w.DWORD),('hEvent',w.HANDLE)]

def kernel():
    b.require(os.name=='nt','WINDOWS_LOCK_INSPECTION_REQUIRED')
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    signatures={
      'CreateFileW':([w.LPCWSTR,w.DWORD,w.DWORD,w.LPVOID,w.DWORD,w.DWORD,w.HANDLE],w.HANDLE),
      'CloseHandle':([w.HANDLE],w.BOOL),
      'LockFileEx':([w.HANDLE,w.DWORD,w.DWORD,w.DWORD,w.DWORD,ctypes.POINTER(OVERLAPPED)],w.BOOL),
      'UnlockFileEx':([w.HANDLE,w.DWORD,w.DWORD,w.DWORD,ctypes.POINTER(OVERLAPPED)],w.BOOL),
      'OpenProcess':([w.DWORD,w.BOOL,w.DWORD],w.HANDLE),
      'GetExitCodeProcess':([w.HANDLE,ctypes.POINTER(w.DWORD)],w.BOOL),
      'GetProcessTimes':([w.HANDLE,ctypes.POINTER(w.FILETIME),ctypes.POINTER(w.FILETIME),ctypes.POINTER(w.FILETIME),ctypes.POINTER(w.FILETIME)],w.BOOL),
      'OpenJobObjectW':([w.DWORD,w.BOOL,w.LPCWSTR],w.HANDLE),
      'QueryInformationJobObject':([w.HANDLE,ctypes.c_int,w.LPVOID,w.DWORD,ctypes.POINTER(w.DWORD)],w.BOOL),
    }
    for name,(args,restype) in signatures.items():
        fn=getattr(k,name);fn.argtypes=args;fn.restype=restype
    return k

class ReadLease:
    """Shared, nonblocking lock on byte zero; handle grants READ only, no creation."""
    def __init__(self,path):
        self.k=kernel();self.handle=None;self.overlap=OVERLAPPED()
        h=self.k.CreateFileW(str(path),0x80000000,3,None,3,0,None)
        b.require(h not in (None,ctypes.c_void_p(-1).value),'R5_LOCK_OPEN_UNCONFIRMED:'+str(ctypes.get_last_error()))
        if not self.k.LockFileEx(h,1,0,1,0,ctypes.byref(self.overlap)):
            error=ctypes.get_last_error();self.k.CloseHandle(h)
            raise RuntimeError('R5_BYTE_LOCK_ACTIVE_OR_UNCONFIRMED:'+str(error))
        self.handle=h
    def close(self):
        if self.handle is not None:
            try:self.k.UnlockFileEx(self.handle,0,1,0,ctypes.byref(self.overlap))
            finally:self.k.CloseHandle(self.handle);self.handle=None

def process_state(pid,completed,identity=None):
    b.require(type(pid) is int and 0<pid<2**32,'INVALID_PROCESS_IDENTITY')
    k=kernel();h=k.OpenProcess(0x1000,False,pid)
    if not h:
        error=ctypes.get_last_error()
        b.require(error==87,'PROCESS_ACTIVITY_UNCONFIRMED:'+str(error))
        return 'GONE'
    try:
        exit_code=w.DWORD()
        b.require(k.GetExitCodeProcess(h,ctypes.byref(exit_code)),'PROCESS_QUERY_FAILED')
        if exit_code.value!=259:return 'GONE'
        values=[w.FILETIME() for _ in range(4)]
        b.require(k.GetProcessTimes(h,*[ctypes.byref(v) for v in values]),'PROCESS_CREATION_QUERY_FAILED')
        created=(values[0].dwHighDateTime<<32)|values[0].dwLowDateTime
        if identity is not None:
            b.require(type(identity) is int and identity>0,'INVALID_PROCESS_CREATION_IDENTITY')
            return 'ACTIVE' if created==identity else 'REUSED_CREATION_IDENTITY'
        finished=datetime.fromisoformat(completed.replace('Z','+00:00'))
        b.require(finished.tzinfo is not None,'COMPLETION_TIMEZONE_REQUIRED')
        finished_ticks=int(finished.timestamp()*10**7)+116444736000000000
        # A recycled PID can be dismissed only with proof it was born after completion.
        return 'REUSED_AFTER_COMPLETION' if created>finished_ticks else 'ACTIVE'
    finally:k.CloseHandle(h)

def job_active(name):
    b.require(isinstance(name,str) and name.startswith('Local\\SP_U12B5_'),'INVALID_JOB_IDENTITY')
    k=kernel();h=k.OpenJobObjectW(4,False,name)
    if not h:
        error=ctypes.get_last_error();b.require(error==2,'JOB_ACTIVITY_UNCONFIRMED:'+str(error));return False
    try:
        # JOBOBJECT_BASIC_ACCOUNTING_INFORMATION ends with Total/Active/TerminatedProcesses.
        class Accounting(ctypes.Structure):
            _fields_=[('times',ctypes.c_longlong*4),('faults',w.DWORD),('total',w.DWORD),('active',w.DWORD),('terminated',w.DWORD)]
        info=Accounting()
        b.require(k.QueryInformationJobObject(h,1,ctypes.byref(info),ctypes.sizeof(info),None),'JOB_QUERY_FAILED')
        return bool(info.active)
    finally:k.CloseHandle(h)


# Exact cross-namespace relationships established by the frozen launchers.
# All other references must remain in their own namespace, including absolute paths.
def allowed_job(namespace,relative):
    prefix='lab_runs/'+namespace+'/'
    if relative.startswith(prefix):return True
    if namespace=='final_mfa_once':return relative=='lab_runs/r6_formal/final'
    if namespace=='r7_sequential':return bool(re.fullmatch(r'lab_runs/r6_formal/candidates/K[124]_r(?:4|8|16)/(?:train|validate)',relative))
    return False

class Inspection:
    def __init__(self):
        self.leases=[];self.snapshots={};self.rows=[];self.errors=[];self.records={};self.process_cache={}
    def close(self):
        for lease in reversed(self.leases):lease.close()
        self.leases=[]
    def safe(self,path,scope):
        path=Path(path)
        b.require(path.resolve().is_relative_to(scope.resolve()),'UNSAFE_HISTORY_PATH:'+str(path))
        p=path
        while p!=b.ROOT:
            if p.exists():b.require(not(p.lstat().st_file_attributes&0x400),'HISTORY_REPARSE_POINT:'+str(p))
            b.require(p.parent!=p,'OUTSIDE_PROJECT_HISTORY');p=p.parent
        return path
    def snapshot(self,path):
        self.safe(path,b.ROOT);value=b.sha(path)
        prior=self.snapshots.setdefault(path,value);b.require(value==prior,'HISTORY_CHANGED_DURING_INSPECTION:'+str(path))
        return value
    def read(self,path):
        self.snapshot(path);return b.read(path)
    def pid_state(self,pid,finished,identity=None):
        key=(pid,finished,identity)
        if key not in self.process_cache:self.process_cache[key]=process_state(pid,finished,identity)
        return self.process_cache[key]
    def closed_status(self,path,namespace=None):
        d=self.read(path);finished=d.get('completed_UTC') or d.get('UTC','')
        identity=d.get('controller_identity',{})
        for key in ('supervisor_PID','actual_interpreter_PID','venv_launcher_PID','controller_PID'):
            if key in d and d[key] is not None:
                created=identity.get('creation_time_100ns') if identity.get('PID')==d[key] else None
                b.require(self.pid_state(d[key],finished,created)!='ACTIVE','LIVE_HISTORICAL_PROCESS:'+str(path)+':'+key)
        if d.get('JobObject_name'):b.require(not job_active(d['JobObject_name']),'LIVE_HISTORICAL_JOB:'+str(path))
        b.require(d.get('state') in TERMINAL,'CONCURRENT_ACTIVE_WORKER_OR_UNKNOWN:'+str(path)+':'+str(d.get('state')))
        if namespace:self.records.setdefault(namespace,[]).append((path,d))
        self.rows.append({'path':str(path),'verdict':'TERMINAL_CONFIRMED','state':d['state']})
        return d
    def pointer(self,path,folder):
        d=self.read(path);job=d.get('job')
        b.require(isinstance(job,str) and '..' not in Path(job).parts,'INVALID_HISTORY_JOB_POINTER')
        target=Path(job) if Path(job).is_absolute() else b.ROOT/job
        target=self.safe(target,b.ROOT/'lab_runs').resolve()
        relative=target.relative_to(b.ROOT.resolve()).as_posix()
        b.require(allowed_job(folder.name,relative),'UNREGISTERED_HISTORY_REFERENCE:'+str(path)+':'+relative)
        status=self.closed_status(target/'status.json',folder.name)
        if d.get('JobObject_name'):
            b.require(d['JobObject_name']==status.get('JobObject_name'),'HISTORY_JOB_IDENTITY_DRIFT')
            b.require(not job_active(d['JobObject_name']),'LIVE_HISTORICAL_JOB:'+str(path))
        self.rows.append({'path':str(path),'resolved_job':relative,'verdict':'TRUSTED_TERMINAL_REFERENCE'})
        return status
    def r5_complete(self,folder):
        here=b.ROOT/'deployment/r5_128gb_v1';manifest=here/'package_manifest.json'
        b.require(self.snapshot(manifest)==R5_MANIFEST_SHA,'R5_FROZEN_LOCK_IMPLEMENTATION_DRIFT')
        m=self.read(manifest)
        for name in ('driver.py','supervisor.py','Start-R5.ps1','common.py','protocol.json'):
            key='deployment/r5_128gb_v1/'+name
            b.require(self.snapshot(here/name)==m['files'][key]['sha256'],'R5_FROZEN_LOCK_IMPLEMENTATION_DRIFT:'+name)
        protocol_sha=self.snapshot(here/'protocol.json')
        for stage in STAGES:
            record=self.read(folder/(stage+'_stage.json'))
            b.require(record.get('state')=='COMPLETE' and record.get('PASS') is True and record.get('stage')==stage,'R5_COMPLETION_NOT_PROVEN:'+stage)
            b.require(record.get('package_manifest_SHA256')==R5_MANIFEST_SHA and record.get('protocol_SHA256')==protocol_sha,'R5_COMPLETION_FREEZE_DRIFT:'+stage)
            status=self.pointer(folder/(stage+'_stage.json'),folder)
            for key in ('state','PASS','stage','job','completed_UTC','result_sha256','package_manifest_SHA256','protocol_SHA256'):
                b.require(record.get(key)==status.get(key),'R5_STAGE_STATUS_DRIFT:'+stage+':'+key)
            b.require(bool(record.get('completed_UTC')),'R5_COMPLETION_TIME_MISSING')
            result=self.safe(b.ROOT/record['job']/'stage_result.json',folder)
            b.require(self.snapshot(result)==record.get('result_sha256'),'R5_COMPLETION_RESULT_HASH_DRIFT:'+stage)
    def classify_lock(self,path,root):
        self.safe(path,b.ROOT/'lab_runs');lease=ReadLease(path);self.leases.append(lease);self.snapshot(path)
        payload=path.read_bytes()
        if root=='r5_128gb' and path.name=='execution.lock':
            b.require(payload==b'0','R5_SENTINEL_NOT_RECOGNIZED');self.r5_complete(path.parent)
        else:
            text=payload.decode('ascii').strip()
            b.require(bool(re.fullmatch(r'(?:(?:FINAL_ONLY|R8_SELECTION_ONLY|R7_QUEUE) )?[1-9][0-9]*',text)),'LOCK_OWNER_IDENTITY_UNCONFIRMED:'+str(path))
            pid=int(text.split()[-1]);witnesses=list(self.records.get(root,[]))
            if root=='r6_formal':
                if text.startswith('FINAL_ONLY '):witnesses=list(self.records.get('final_mfa_once',[]))
                elif text.startswith('R8_SELECTION_ONLY '):witnesses=list(self.records.get('r8_selection',[]))
                else:witnesses+=self.records.get('r7_sequential',[])
            matched=[]
            for status,record in witnesses:
                pids=[record.get(k) for k in ('supervisor_PID','actual_interpreter_PID','venv_launcher_PID','controller_PID')]
                if pid in pids:matched.append((status,record))
            # Some frozen controllers (notably R8) never recorded their own PID.
            # Their current terminal pointer plus an older, unchanged lock and
            # a provably absent/recycled owner is also a usable closure witness.
            if not matched:
                matched=[(status,record) for status,record in witnesses
                         if record.get('PASS') is True and (record.get('completed_UTC') or record.get('UTC'))]
            b.require(matched,'LOCK_HAS_NO_MATCHING_TERMINAL_OWNER_RECORD:'+str(path))
            matched=[(status,record) for status,record in matched
                     if path.stat().st_mtime<=datetime.fromisoformat((record.get('completed_UTC') or record.get('UTC','')).replace('Z','+00:00')).timestamp()]
            b.require(matched,'LOCK_MODIFIED_AFTER_TERMINAL_RECORD:'+str(path))
            for status,record in matched:
                finished=record.get('completed_UTC') or record.get('UTC','')
                end=datetime.fromisoformat(finished.replace('Z','+00:00'))
                b.require(end.tzinfo is not None and path.stat().st_mtime<=end.timestamp(),
                          'LOCK_MODIFIED_AFTER_TERMINAL_RECORD:'+str(path))
                identity=record.get('controller_identity',{})
                created=identity.get('creation_time_100ns') if identity.get('PID')==pid else None
                b.require(self.pid_state(pid,finished,created)!='ACTIVE','LIVE_LOCK_OWNER:'+str(path))
        self.rows.append({'path':str(path),'verdict':'COMPLETED_RETAINED_LOCK_IDLE','read_only_shared_lease':True})
    def attempt(self,label,fn):
        try:fn()
        except Exception as error:
            reason=str(error);active=reason.startswith(('LIVE_','R5_BYTE_LOCK_ACTIVE_OR_UNCONFIRMED:33'))
            self.errors.append({'item':str(label),'classification':'ACTIVE' if active else 'UNKNOWN','reason':reason})
    def inspect(self):
        # First resolve ALL status/pointer records, then ALL locks. Cross-owner
        # R6 leases can consequently use final/R7/R8 evidence without scan-order bias.
        for root in ROOTS:
            folder=b.ROOT/'lab_runs'/root
            self.attempt(root,lambda folder=folder:self.safe(folder,b.ROOT/'lab_runs'))
            for rel in ('status.json','queue.json','formal/status.json'):
                path=folder/rel
                if path.exists():self.attempt(path,lambda path=path,root=root:self.closed_status(path,root))
            names=('active_worker.json','latest.json','latest_launch.json','latest_toy.json','latest_lab.json','background_probe_latest.json')
            for name in names:
                path=folder/name
                if path.exists():self.attempt(path,lambda path=path,folder=folder:self.pointer(path,folder))
            extra=[]
            if root=='r6_formal' and folder.exists():
                extra=list((folder/'candidates').glob('*/train/status.json'))+list((folder/'candidates').glob('*/validate/status.json'))
                if (folder/'final/status.json').exists():extra.append(folder/'final/status.json')
            if root=='r5_128gb' and folder.exists():
                # Scan operational job statuses, excluding archived/personal/toy workspaces.
                def walk(directory):
                    self.safe(directory,folder)
                    for child in directory.iterdir():
                        if child.name in ('exports','preparation_personal','temporary','__pycache__','_scratch'):continue
                        self.safe(child,folder)
                        if child.is_dir():yield from walk(child)
                        elif child.name=='status.json':yield child
                try:extra=list(walk(folder))
                except Exception as error:self.attempt('r5_status_tree',lambda:(_ for _ in ()).throw(error))
            for path in extra:self.attempt(path,lambda path=path,root=root:self.closed_status(path,root))
        for root in ROOTS:
            folder=b.ROOT/'lab_runs'/root
            for name in ('execution.lock','pipeline.lock','audit_execution.lock'):
                path=folder/name
                if path.exists():self.attempt(path,lambda path=path,root=root:self.classify_lock(path,root))
        # R2 local monitor jobs have per-job status instead of the historical
        # global marker. Do not let another rehearsal overlap a formal launch.
        own=b.ROOT/'lab_runs/b5c_paired_r2/synthetic'
        if own.exists():
            self.attempt('R2_SYNTHETIC_SCOPE',lambda:self.safe(own,b.ROOT/'lab_runs'))
            for path in own.glob('*/status.json'):
                self.attempt(path,lambda path=path:self.closed_status(path))
        for path,value in list(self.snapshots.items()):
            self.attempt(path,lambda path=path,value=value:b.require(b.sha(path)==value,'HISTORY_CHANGED_DURING_INSPECTION:'+str(path)))
        namespaces=[]
        for root in ROOTS:
            prefix=str(b.ROOT/'lab_runs'/root)
            issues=[e for e in self.errors if e['item']==root or e['item'].startswith(prefix)]
            state='ACTIVE' if any(e['classification']=='ACTIVE' for e in issues) else 'UNKNOWN' if issues else 'COMPLETED' if self.records.get(root) else 'NO_ACTIVITY_EVIDENCE'
            namespaces.append({'namespace':root,'classification':state,'records':len(self.records.get(root,[])),'blockers':issues})
        self.report={'PASS':not self.errors,'namespaces':namespaces,'checks':self.rows,'blockers':self.errors,
                     'history_writes':0,'formal_evaluation_executed':False}
        return self

def inspect():
    result=Inspection()
    try:return result.inspect()
    except BaseException:result.close();raise
