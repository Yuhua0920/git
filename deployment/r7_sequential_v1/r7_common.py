"""R7 prospective engineering contract; no approval or science generated here."""
from pathlib import Path
import sys, os, json, hashlib, time, contextlib
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'lab_runs/r7_sequential'
R6 = ROOT / 'deployment/r6_formal_v1'
R6OUT = ROOT / 'lab_runs/r6_formal'

def require(ok, reason):
    if not ok: raise RuntimeError(reason)

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def write(p, value, exclusive=False):
    p = Path(p).resolve()
    require(p.is_relative_to(OUT.resolve()), 'R7_OUTPUT_ONLY')
    p.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with p.open('x', encoding='utf8') as f: f.write(content); f.flush(); os.fsync(f.fileno())
        return
    tmp = p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('x', encoding='utf8') as f: f.write(content); f.flush(); os.fsync(f.fileno())
    for attempt in range(101):
        try: tmp.replace(p); return
        except PermissionError as e:
            if getattr(e,'winerror',None) not in (5,32,33) or attempt==100: raise
            time.sleep(.01)

def contract(): return read(HERE/'queue_contract.json')
def bindings():
    return {'package_manifest_SHA256':digest(R6/'package_manifest.json'),
            'resource_amendment_SHA256':digest(R6/'resource_amendment.json'),
            'input_contract_SHA256':digest(R6/'input_contract.json'),
            'R7_package_manifest_SHA256':digest(HERE/'package_manifest.json'),
            'R7_queue_contract_SHA256':digest(HERE/'queue_contract.json')}

def verify():
    manifest=read(HERE/'package_manifest.json'); cfg=contract()
    known={Path(n).name for n in manifest['files']}|{'package_manifest.json','prepare_personal.py','deployment_check.py'}
    require(all(p.name in known for p in HERE.iterdir() if p.is_file()),'UNREGISTERED_R7_PAYLOAD_FILE')
    for name,row in manifest['files'].items():
        p=(ROOT/name).resolve()
        require(p.is_relative_to(HERE) and p.is_file() and digest(p)==row['sha256'] and p.stat().st_size==row['bytes'],'R7_PACKAGE_DRIFT:'+name)
    for name,h in cfg['R6_dependencies'].items():
        require(digest(ROOT/name)==h,'FROZEN_R6_DRIFT:'+name)
    original=read(R6/'input_contract.json')
    rows=read(ROOT/'outputs/sp_u12b5b_candidate_seed_registry.json')['candidates']
    require(rows==original['candidate_order']==cfg['candidate_order'],'FROZEN_CANDIDATE_OR_SEED_DRIFT')
    require(read(R6/'resource_amendment.json')==cfg['resource_contract'],'R6_RESOURCE_CONTRACT_DRIFT')
    return {'PASS':True,**bindings(),'candidate_order':rows,'real_training_authorized_by_verify':False}

def approval_values(a, evidence_hash):
    from datetime import datetime, timezone
    require(a.get('status')=='MENTOR_APPROVED' and a.get('scope')=='R7_NINE_CANDIDATE_TRAIN_VALIDATE_ONLY','MENTOR_APPROVAL_PENDING_OR_SCOPE_INVALID')
    require(a.get('allowed_actions')==['train','validate'] and a.get('candidate_order')==contract()['candidate_order'],'APPROVAL_STAGE_OR_SEED_DRIFT')
    require(a.get('mentor_instruction_reference') and a.get('mentor_instruction_SHA256')==evidence_hash,'MENTOR_INSTRUCTION_BINDING_INVALID')
    require(all(a.get(k)==h for k,h in bindings().items()),'APPROVAL_PACKAGE_HASH_MISMATCH')
    require(isinstance(a.get('approved_UTC'),str),'HUMAN_APPROVAL_TIMESTAMP_INVALID')
    try: stamp=datetime.fromisoformat(a['approved_UTC'].replace('Z','+00:00'))
    except (KeyError,ValueError,TypeError): raise RuntimeError('HUMAN_APPROVAL_TIMESTAMP_INVALID')
    require(stamp.tzinfo is not None and stamp<=datetime.now(timezone.utc),'HUMAN_APPROVAL_TIMESTAMP_INVALID')
    require(a.get('resource_contract')==contract()['resource_contract'] and a.get('no_automatic_retry') is True,'APPROVAL_RESOURCE_CONTRACT_DRIFT')
    require(a.get('selection_authorized') is False and a.get('final_training_authorized') is False and a.get('target_access_authorized') is False,'APPROVAL_EXCEEDS_QUEUE_SCOPE')
    return a

def approval():
    require((OUT/'MENTOR_APPROVAL.json').is_file() and (OUT/'mentor_instruction.txt').is_file(),'MENTOR_APPROVAL_PENDING: NO FORMAL EXECUTION')
    a=approval_values(read(OUT/'MENTOR_APPROVAL.json'),digest(OUT/'mentor_instruction.txt'))
    require((R6OUT/'MENTOR_APPROVAL.json').is_file() and digest(R6OUT/'MENTOR_APPROVAL.json')==digest(OUT/'MENTOR_APPROVAL.json'),'R6_R7_APPROVAL_BYTES_DIFFER')
    require(not any((p/n).exists() for p in (OUT,R6OUT) for n in ('STOP.json','STOP_REQUESTED.json')),'PIPELINE_STOPPED_NO_RETRY')
    return a

def import_approval(record, instruction):
    # Import exact, independently supplied human bytes. No template/default,
    # signature creation, invented author, timestamp, or pending upgrade.
    verify(); record=Path(record).resolve(); instruction=Path(instruction).resolve()
    a=approval_values(read(record),digest(instruction))
    require(not (OUT/'queue.json').exists() and not (OUT/'execution.lock').exists() and not (R6OUT/'pipeline.lock').exists(),'AUTHORIZATION_IMPORT_WHILE_EXECUTING_FORBIDDEN')
    require(not (R6OUT/'candidates').exists() and not (R6OUT/'final').exists() and not (R6OUT/'SELECTION_MANIFEST.json').exists(),'FORMAL_ARTIFACT_ALREADY_PRESENT')
    targets=(OUT/'MENTOR_APPROVAL.json',OUT/'mentor_instruction.txt',R6OUT/'MENTOR_APPROVAL.json')
    require(all(not p.exists() for p in targets),'APPROVAL_EXISTS_NO_OVERWRITE')
    require(not any((p/n).exists() for p in (OUT,R6OUT) for n in ('STOP.json','STOP_REQUESTED.json')),'STOP_PRESENT_NO_AUTHORIZATION_REPAIR')
    for dst,src in zip(targets,(record,instruction,record)):
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as f: f.write(src.read_bytes()); f.flush(); os.fsync(f.fileno())
    approval()
    write(OUT/'approval_import_receipt.json',{'PASS':True,'record_SHA256':digest(record),'instruction_SHA256':digest(instruction),'imported_UTC':utc(),'record_timestamp_generated_by_program':False},exclusive=True)
    return {'PASS':True,'authority':'IMPORTED_HUMAN_ATTESTATION_NOT_CRYPTOGRAPHIC_SIGNATURE','formal_execution_started':False}

def r6_imports():
    sys.path.insert(0,str(R6))
    import common, runner, supervisor, windows_resources
    return common,runner,supervisor,windows_resources

def process_identity(pid):
    import ctypes
    _,_,_,w=r6_imports();h=w.K.OpenProcess(0x1000,False,int(pid))
    if not h:return None
    try:
        code=w.w.DWORD();created,ended,kernel,user=w.FILETIME(),w.FILETIME(),w.FILETIME(),w.FILETIME()
        if not w.K.GetExitCodeProcess(h,ctypes.byref(code)) or code.value!=259:return None
        require(w.K.GetProcessTimes(h,ctypes.byref(created),ctypes.byref(ended),ctypes.byref(kernel),ctypes.byref(user)),'CONTROLLER_IDENTITY_QUERY_FAILED')
        return {'PID':int(pid),'creation_time_100ns':(created.high<<32)|created.low}
    finally:w.K.CloseHandle(h)

def lab_receipts():
    p=read(R6OUT/'preflight.json'); t=read(R6OUT/'preparation_personal/toy_test_report.json'); b=read(R6OUT/'preparation_personal/background_probe_result.json')
    require(p['PASS'] and p['capacity']['PASS'] and p['integrity']['package_manifest_SHA256']==bindings()['package_manifest_SHA256'],'ACTUAL_LAB_PREFLIGHT_REQUIRED')
    require(t['PASS'] and t['tested_package_manifest_sha256']==bindings()['package_manifest_SHA256'],'ACTUAL_LAB_R6_TOY_TEST_REQUIRED')
    require(b['PASS'] and not b.get('test_receipt',False),'ACTUAL_LAB_BACKGROUND_TEST_REQUIRED')
    require((OUT/'technical_tests.json').is_file() and read(OUT/'technical_tests.json')['PASS'] and read(OUT/'technical_tests.json')['R7_package_manifest_SHA256']==bindings()['R7_package_manifest_SHA256'],'LAB_R7_TECHNICAL_TESTS_REQUIRED')
    require(read(OUT/'technical_tests.json')['host']==os.environ.get('COMPUTERNAME'),'R7_TEST_RECEIPT_FROM_DIFFERENT_HOST')
    return {'PASS':True,'preflight_SHA256':digest(R6OUT/'preflight.json'),'toy_SHA256':digest(R6OUT/'preparation_personal/toy_test_report.json'),'background_SHA256':digest(R6OUT/'preparation_personal/background_probe_result.json')}

def formal_preflight():
    verify(); approval(); receipts=lab_receipts()
    c,r,s,w=r6_imports(); c.approval();c.integrity(); inputs=r.input_check();env=c.environment();cap=w.capacity(w.hardware())
    require(cap['PASS'],'LAUNCH_RAM_COMMIT_CAPACITY_FAILED:'+str(cap['reasons']))
    require(not (R6OUT/'pipeline.lock').exists(),'EXISTING_R6_JOB_LOCK')
    require(not any((R6OUT/n).exists() for n in ('candidates','final','SELECTION_MANIFEST.json','FINAL_MODEL_MANIFEST.json')),'EXISTING_FORMAL_ATTEMPT_NO_RETRY')
    require(not (OUT/'queue.json').exists() and not (OUT/'execution.lock').exists(),'DUPLICATE_QUEUE_OR_PRIOR_ATTEMPT')
    return {'PASS':True,'receipts':receipts,'inputs':inputs,'environment':env,'capacity':cap,'numeric_preparation_values_read':False}

@contextlib.contextmanager
def lease(directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    lock=directory/'execution.lock'
    with lock.open('x',encoding='utf8') as f: f.write(str(os.getpid()));f.flush();os.fsync(f.fileno())
    try: yield
    finally: lock.unlink(missing_ok=True)
