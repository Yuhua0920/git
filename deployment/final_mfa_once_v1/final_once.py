"""One operator-confirmed final fit, using unchanged R6 fit/monitor code objects.

No candidate fit, validation, selection, target evaluation or retry entry point.
The new permission context replaces the generic R6 approval only at the call
boundary of worker.train; common.approval and all historical files stay intact.
"""
from pathlib import Path
import argparse, contextlib, getpass, hashlib, importlib.util, json, os
import subprocess, sys, time, traceback, types, uuid, zipfile, functools
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'lab_runs/final_mfa_once'
R6 = ROOT/'lab_runs/r6_formal'
FINAL = R6/'final'
FULL = 'outputs/sp_u12b5b/full_retraining.npy'
PERMISSION = 'FINAL_MFA_RETRAIN_ONCE_K1_R16_SEED4143049057'
_numeric = False
_publishing = False
_opaque = False

def require(x, msg):
    if not x: raise RuntimeError(msg)

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
def sha(p):
    global _opaque
    previous=_opaque;_opaque=Path(p).resolve()==(ROOT/FULL).resolve()
    try:
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
        return h.hexdigest()
    finally:_opaque=previous
def write(p, value, exclusive=False):
    p=Path(p)
    if not p.parent.exists():p.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if exclusive:
        with p.open('x',encoding='utf8') as f:f.write(data);f.flush();os.fsync(f.fileno())
    else:
        tmp=p.with_name(p.name+'.tmp');tmp.write_text(data,encoding='utf8')
        for attempt in range(101):
            try:tmp.replace(p);break
            except PermissionError as e:
                if getattr(e,'winerror',None)not in(5,32,33)or attempt==100:raise
                time.sleep(.01) # Mutable status publication only; never rerun a job.

def contract(): return read(HERE/'contract.json')
def verify():
    v=read(HERE/'package_manifest.json');p=contract()
    for n,x in v['files'].items():
        q=(ROOT/n).resolve();require(q.is_relative_to(HERE)and q.is_file()and sha(q)==x['sha256']and q.stat().st_size==x['bytes'],'FINAL_TOOL_HASH_DRIFT:'+n)
    for n,h in p['code_hashes'].items():require(sha(ROOT/n)==h,'FROZEN_DEPENDENCY_DRIFT:'+n)
    return {'PASS':True,'package_sha256':sha(HERE/'package_manifest.json'),'training_executed':False}

@functools.lru_cache(maxsize=1)
def modules():
    verify();p=contract()
    loaderpath=ROOT/'deployment/r7_evidence_audit_r1_v1/source_only_loader.py'
    spec=importlib.util.spec_from_file_location('_final_source_loader',loaderpath)
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    sources={n:{'sha256':h}for n,h in p['code_hashes'].items()if n.endswith('.py')}
    sources.update({n:x for n,x in read(HERE/'package_manifest.json')['files'].items()if n.endswith('.py')})
    loader.install(ROOT,sources)
    sys.path.insert(0,str(ROOT/'deployment/r6_formal_v1'));sys.path.insert(0,str(ROOT))
    import common, worker, supervisor, model_ops, windows_resources
    return common,worker,supervisor,model_ops,windows_resources

def selection_check(v, digest, sidecar):
    p=contract();s=v.get('selected',{})
    require(digest==sidecar.strip()==p['selection_sha256'],'SELECTION_SHA256_MISMATCH')
    require(v.get('PASS')is True and v.get('target_access')==0,'SELECTION_NOT_PASS')
    require(all(s.get(k)==p['selected'][k]for k in('name','K','rank','seed','final_seed')),'FROZEN_SELECTED_STRUCTURE_OR_SEED_MISMATCH')
    require(v.get('execution_package_sha256')==p['R6_package_sha256']and v.get('input_contract_sha256')==p['R6_input_contract_sha256'],'SELECTION_SCIENTIFIC_BINDING_DRIFT')
    require(v.get('coverage',{}).get('scored_frame_predictions')==32400 and len(v.get('candidates',[]))==9,'FROZEN_SELECTION_COVERAGE_MISSING')
    require(v.get('R8_bindings')==p['R8_bindings'] and v.get('R8_selection_authorization',{}).get('PASS')is True,'R8_SELECTION_FREEZE_EVIDENCE_MISSING')
    return s # Read frozen result only: no automatic_select, no score recomputation.

def load_selection():
    path=R6/'SELECTION_MANIFEST.json'
    require(path.is_file()and (R6/'SELECTION_MANIFEST.sha256').is_file(),'REAL_LAB_SELECTION_MANIFEST_REQUIRED')
    v=read(path);h=sha(path)
    s=selection_check(v,h,(R6/'SELECTION_MANIFEST.sha256').read_text(encoding='ascii'))
    return v,s,h

def absent(owned=False):
    require(not FINAL.exists()and not(R6/'FINAL_MODEL_MANIFEST.json').exists()and not(R6/'FINAL_MODEL_MANIFEST.sha256').exists(),'FINAL_ALREADY_EXISTS_NO_RETRY')
    require(not(OUT/'FINAL_START_CONSUMED.json').exists(),'FINAL_START_ALREADY_CONSUMED_NO_RETRY')
    for ns in ('r6_formal','r7_sequential','r8_selection','final_mfa_once'):
        for name in ('STOP.json','STOP_REQUESTED.json'):
            require(not(ROOT/'lab_runs'/ns/name).exists(),'STOP_PRESENT:'+ns)
    for n in ('lab_runs/r6_formal/pipeline.lock','lab_runs/r7_sequential/execution.lock','lab_runs/r7_evidence_audit/audit_execution.lock','lab_runs/r7_evidence_audit_r1/audit_execution.lock'):
        require(owned and n=='lab_runs/r6_formal/pipeline.lock' or not(ROOT/n).exists(),'CONCURRENT_TASK_LOCK:'+n)
    for n in contract()['target_denied']:require(not(ROOT/n).exists(),'TARGET_ARTIFACT_PRESENT_NO_READ:'+n)

def live_identity(x,wr):
    import ctypes as c,ctypes.wintypes as w
    h=wr.K.OpenProcess(0x1000,False,int(x['PID']))
    if not h:return False
    try:
        code=w.DWORD();wr.K.GetExitCodeProcess(h,c.byref(code))
        if code.value!=259:return False
        a,b,d,e=wr.FILETIME(),wr.FILETIME(),wr.FILETIME(),wr.FILETIME()
        require(wr.K.GetProcessTimes(h,c.byref(a),c.byref(b),c.byref(d),c.byref(e)),'PROCESS_IDENTITY_QUERY_FAILED')
        return int(x['creation_time_100ns'])==((a.high<<32)|a.low)
    finally:wr.K.CloseHandle(h)

def preflight(owned=False):
    verify();absent(owned);v,s,h=load_selection();p=contract();c,_,_,_,wr=modules()
    data=ROOT/FULL;require(sha(data)==p['full_data']['sha256'],'FULL_RETRAINING_HASH_MISMATCH')
    header=c.npy_header(data);require(all(header[k]==p['full_data'][k]for k in('shape','dtype','fortran_order')),'FULL_RETRAINING_HEADER_MISMATCH')
    queue=read(ROOT/'lab_runs/r7_sequential/queue.json')
    require(queue['state']=='QUEUE_COMPLETE'and queue.get('completion')=='QUEUE_COMPLETE_9_OF_9'and len(queue['completed_candidates'])==9,'R7_QUEUE_NOT_COMPLETE')
    require(not live_identity(queue['controller_identity'],wr),'R7_CONTROLLER_STILL_RUNNING')
    for row in read(ROOT/'deployment/r6_formal_v1/input_contract.json')['candidate_order']:
        for role in ('train','validate'):
            stage=R6/'candidates'/row['name']/role;st=read(stage/'status.json')
            require(st['state']=='COMPLETE'and st.get('PASS')is True,'CANDIDATE_NOT_COMPLETE')
            identity=None
            with(stage/'resources.jsonl').open(encoding='utf8')as stream:
                for line in stream:
                    for rec in json.loads(line)['process_tree']['members']:
                        if rec['PID']==st['actual_interpreter_PID']:
                            identity={k:rec[k]for k in('PID','creation_time_100ns')};break
                    if identity:break
            require(identity is not None and not live_identity(identity,wr),'SCIENTIFIC_WORKER_RUNNING_OR_IDENTITY_MISSING')
    # Previously passed technical receipts are checked, not rerun.
    pf=read(R6/'preflight.json');toy=read(R6/'preparation_personal/toy_test_report.json')
    require(pf['PASS']and pf['integrity']['package_manifest_SHA256']==p['R6_package_sha256']and toy['PASS']and toy['tested_package_manifest_sha256']==p['R6_package_sha256'],'R6_VERIFIED_RECEIPTS_REQUIRED')
    env=c.environment();cap=wr.capacity(wr.hardware());require(cap['PASS'],'LAB_CAPACITY_STOP:'+str(cap['reasons']))
    return {'PASS':True,'selected':s,'selection_sha256':h,'data_sha256':p['full_data']['sha256'],'header':header,'environment':env,'capacity':cap,'validation_recomputed':False,'target_access':0}

def confirmation(operator,confirm,selection_sha):
    require(confirm is True and isinstance(operator,str)and operator.strip(),'EXPLICIT_OPERATOR_FINAL_CONFIRMATION_REQUIRED')
    return {'scope':PERMISSION,'final_training_authorized':True,'maximum_starts':1,'operator_self_identification':operator.strip(),
            'Windows_account':getpass.getuser(),'confirmed_UTC':utc(),'confirmation_source':'explicit -ConfirmFinalTraining switch executed by operator',
            'mentor_identity_claimed':False,'selected':contract()['selected'],'SELECTION_MANIFEST_SHA256':selection_sha,
            'full_data_sha256':contract()['full_data']['sha256'],'deployment_sha256':sha(HERE/'package_manifest.json'),
            'candidate_training_authorized':False,'validation_authorized':False,'selection_authorized':False,
            'target_access_authorized':False,'B2_B3_unseal_authorized':False,'automatic_retry':False}

def check_confirmation(v):
    require(v.get('scope')==PERMISSION and v.get('final_training_authorized')is True and v.get('maximum_starts')==1,'FINAL_ONLY_PERMISSION_REQUIRED')
    require(v.get('selected')==contract()['selected']and v.get('full_data_sha256')==contract()['full_data']['sha256']and v.get('deployment_sha256')==sha(HERE/'package_manifest.json'),'FINAL_CONFIRMATION_BINDING_DRIFT')
    require(v.get('confirmed_UTC')and v.get('operator_self_identification')and not v.get('mentor_identity_claimed'),'REAL_OPERATOR_CONFIRMATION_REQUIRED')
    require(all(v.get(k)is False for k in('candidate_training_authorized','validation_authorized','selection_authorized','target_access_authorized','B2_B3_unseal_authorized','automatic_retry')),'FINAL_PERMISSION_SCOPE_EXPANDED')
    require(v['SELECTION_MANIFEST_SHA256']==load_selection()[2],'SELECTION_CHANGED_AFTER_CONFIRMATION')
    require(read(OUT/'FINAL_START_CONSUMED.json')['confirmation_sha256']==sha(FINAL/'operator_confirmation.json'),'FINAL_CONSUMPTION_RECEIPT_MISMATCH')
    require(not(OUT/'STOP.json').exists(),'FINAL_STOPPED')
    return v

@contextlib.contextmanager
def final_numeric(role):
    global _numeric
    require(role=='final','ONLY_FINAL_NUMERIC_ROLE_ALLOWED')
    check_confirmation(read(FINAL/'operator_confirmation.json'))
    c=sys.modules['common'];require(not c._numeric and not _numeric,'NESTED_NUMERIC_ROLE_DENIED')
    _numeric=True;c._numeric.add(ROOT/FULL)
    try:yield
    finally:c._numeric.clear();_numeric=False

def guard(job):
    p=contract();c=sys.modules['common'];runtime={Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()}
    exact={(ROOT/n).resolve()for n in p['code_hashes']}
    exact|={(R6/n).resolve()for n in('SELECTION_MANIFEST.json','SELECTION_MANIFEST.sha256')}
    numeric={(FINAL/'model'/n).resolve()for n in ('means.npy','lambdas.npy','psis.npy','amps.npy','covs.npy','inv_covs.npy','training_history.npy')}
    finals={(R6/n).resolve()for n in('FINAL_MODEL_MANIFEST.json','FINAL_MODEL_MANIFEST.sha256')}
    lock=(R6/'pipeline.lock').resolve();full=(ROOT/FULL).resolve();job=job.resolve()
    def check(raw,writing):
        q=Path(os.fsdecode(raw)).resolve()
        if writing:
            require(q.is_relative_to(OUT)or q.is_relative_to(job)or q==lock or _publishing and q in finals,'FINAL_HISTORICAL_WRITE_DENIED:'+str(q))
            if q.suffix=='.npy':require(q in numeric and not q.exists(),'MODEL_OVERWRITE_OR_OTHER_NPY_WRITE_DENIED')
            return
        require(q in exact or q in finals or q==full and (_numeric or _opaque)or q in numeric or q.is_relative_to(HERE)or q.is_relative_to(OUT)or q.is_relative_to(job)or any(q.is_relative_to(x)for x in runtime),'FINAL_READ_DENIED:'+str(q))
        if q.suffix.lower()in('.npy','.npz','.pkl','.pickle','.h5','.mat'):
            require(q==full and (_numeric or _opaque)or q in numeric,'FINAL_NUMERIC_FIREWALL_DENIED')
    def hook(event,args):
        if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
            if os.fsdecode(args[0]).upper()in('NUL',r'\\.\NUL'):return
            mode,flags=args[1:3];writing=isinstance(mode,str)and any(x in mode for x in'wax+')or isinstance(flags,int)and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            check(args[0],writing)
        elif event in('os.remove','os.rename','os.rmdir','os.mkdir','os.utime','os.link'):
            for raw in args[:2]if event in('os.rename','os.link')else args[:1]:
                if isinstance(raw,(str,bytes,os.PathLike)):check(raw,True)
    sys.addaudithook(hook)

def fit_callable(old):
    require(old.__name__=='train','ONLY_EXISTING_R6_TRAIN_BODY')
    globals_=dict(old.__globals__);globals_['numeric_inputs']=final_numeric
    return types.FunctionType(old.__code__,globals_,old.__name__,old.__defaults__,old.__closure__)

def monitor_callable(old):
    # Same verified loop, limits, PID handshake, JobObject and timing logic.
    # Select new worker script; route mutable active pointer into new namespace.
    def routed_write(path,value,exclusive=False):
        path=Path(path)
        if path==R6/'active_worker.json':path=OUT/'active_worker.json'
        return write(path,value,exclusive)
    globals_=dict(old.__globals__);globals_.update(HERE=HERE,write=routed_write)
    return types.FunctionType(old.__code__,globals_,old.__name__,old.__defaults__,old.__closure__)

def freeze(result,receipt):
    global _publishing
    c=sys.modules['common'];p=contract();_,s,h=load_selection();require(h==receipt['SELECTION_MANIFEST_SHA256'],'SELECTION_DRIFT_AT_FREEZE')
    require(sha(ROOT/FULL)==p['full_data']['sha256'],'FULL_DATA_DRIFT_AT_FREEZE');verify()
    require(result['PASS']and result['seed']==4143049057 and result['serialization_roundtrip']and result['all_components_smoke']['PASS'],'FINAL_MODEL_NOT_LEGAL')
    m=read(FINAL/'model/model_manifest.json');require(m['N']==15000 and m['D']==4032 and m['K']==1 and m['rank']==16 and m['context']['fresh_fit']and not m['context']['candidate_checkpoint_used'],'FINAL_MODEL_IDENTITY_DRIFT')
    manifest={'PASS':True,'artifact_id':'FINAL_MFA_MODEL','final_model_manifest_sha256':result['model_manifest_sha256'],'selected':s,
              'final_seed':4143049057,'full_data_sha256':p['full_data']['sha256'],'training_result_sha256':sha(FINAL/'training_result.json'),
              'SELECTION_MANIFEST_SHA256':h,'package_manifest_sha256':p['R6_package_sha256'],
              'inference_config':{'joint_dimension':4032,'data_dimension':3968,'pilot_dimension':64,'geometry_order_sha256':p['pilot_order_sha256'],
                'all_components_float':1.0,'validation_noise_variances':p['noise_variances'],
                'operating_noise_rule':'known positive sigma2 times I64; original author noise-dependent posterior is instantiated per call',
                'online_inputs':'received owner pilots, known pilot symbols/geometry, known sigma2; no target label/covariance/CSI'},
              'target_access':0,'frozen_before_B2_unsealing':True,'fresh_fit':True,'UTC':utc(),
              'operator_confirmation_sha256':sha(FINAL/'operator_confirmation.json'),'final_only_package_sha256':sha(HERE/'package_manifest.json')}
    _publishing=True
    try:
        write(R6/'FINAL_MODEL_MANIFEST.json',manifest,True)
        with(R6/'FINAL_MODEL_MANIFEST.sha256').open('x',encoding='ascii')as f:f.write(sha(R6/'FINAL_MODEL_MANIFEST.json')+'\n');f.flush();os.fsync(f.fileno())
    finally:_publishing=False
    require(sha(R6/'FINAL_MODEL_MANIFEST.json')==(R6/'FINAL_MODEL_MANIFEST.sha256').read_text(encoding='ascii').strip(),'FINAL_FREEZE_CORRUPT')

def start(operator,confirm):
    require(confirm and operator,'EXPLICIT_OPERATOR_FINAL_CONFIRMATION_REQUIRED');verify();absent()
    with(R6/'pipeline.lock').open('x',encoding='utf8')as f:f.write('FINAL_ONLY '+str(os.getpid()));f.flush();os.fsync(f.fileno())
    consumed=False
    try:
        pf=preflight(True);receipt=confirmation(operator,confirm,pf['selection_sha256']);FINAL.mkdir(exist_ok=False)
        write(FINAL/'operator_confirmation.json',receipt,True)
        write(OUT/'FINAL_START_CONSUMED.json',{'confirmation_sha256':sha(FINAL/'operator_confirmation.json'),'UTC':utc(),'maximum_starts':1},True);consumed=True
        write(FINAL/'preflight.json',pf,True);write(OUT/'status.json',{'state':'RUNNING','PASS':False,'UTC':utc(),'automatic_retry':False})
        c,w,s,_,_=modules();guard(FINAL)
        state,result=monitor_callable(s.run_worker)('final',FINAL,blas=6,openmp=1,extra=['--candidate','K1_r16'],timeout=86400)
        state.update(worker_result_sha256=sha(FINAL/'worker_result.json'),scientific_result_sha256=sha(FINAL/'training_result.json'))
        write(FINAL/'status.json',state);freeze(result,receipt)
        write(OUT/'status.json',{'state':'COMPLETE','PASS':True,'final_model_manifest_sha256':sha(R6/'FINAL_MODEL_MANIFEST.json'),'iterations':result['iterations'],'convergence':result['convergence'],'UTC':utc(),'automatic_retry':False,'target_access':0})
        return status()
    except BaseException:
        write(OUT/'exception.json',{'UTC':utc(),'traceback':traceback.format_exc(),'start_consumed':consumed,'automatic_retry':False})
        write(OUT/'STOP.json',{'UTC':utc(),'reason':'FINAL_ENTRY_FAILED_NO_RETRY'})
        write(OUT/'status.json',{'state':'FAILED','PASS':False,'UTC':utc(),'automatic_retry':False});raise
    finally:(R6/'pipeline.lock').unlink(missing_ok=True)

def status():
    frozen=False
    if(R6/'FINAL_MODEL_MANIFEST.json').is_file()and(R6/'FINAL_MODEL_MANIFEST.sha256').is_file():
        frozen=sha(R6/'FINAL_MODEL_MANIFEST.json')==(R6/'FINAL_MODEL_MANIFEST.sha256').read_text(encoding='ascii').strip()
    return {'entry':read(OUT/'status.json')if(OUT/'status.json').exists()else{'state':'NOT_STARTED'},
            'worker':read(FINAL/'status.json')if(FINAL/'status.json').exists()else None,
            'start_consumed':(OUT/'FINAL_START_CONSUMED.json').exists(),'final_frozen':frozen,
            'STOP':read(OUT/'STOP.json')if(OUT/'STOP.json').exists()else None,'automatic_retry':False}
def stop():
    write(OUT/'STOP.json',{'UTC':utc(),'reason':'OPERATOR_STOP_NO_RETRY'},True)
    if(OUT/'active_worker.json').exists():
        *_,wr=modules();wr.stop_named_job(read(OUT/'active_worker.json')['JobObject_name'])
    return status()
def export(include_model=False):
    require(not(R6/'pipeline.lock').exists(),'WAIT_FOR_JOB_COMPLETION')
    if include_model:
        require((R6/'FINAL_MODEL_MANIFEST.sha256').is_file(),'FINAL_NOT_FROZEN')
        require(sha(R6/'FINAL_MODEL_MANIFEST.json')==(R6/'FINAL_MODEL_MANIFEST.sha256').read_text(encoding='ascii').strip(),'FINAL_MANIFEST_HASH_DRIFT')
        _,_,_,ops,_=modules();ops.check_model_files(FINAL/'model')
        require(sha(FINAL/'model/model_manifest.json')==read(R6/'FINAL_MODEL_MANIFEST.json')['final_model_manifest_sha256'],'FINAL_MODEL_BINDING_DRIFT')
    dest=OUT/'exports';dest.mkdir(parents=True,exist_ok=True);zip_path=dest/('SP_U12B5_FINAL_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.zip')
    files=[p for p in FINAL.rglob('*')if p.is_file()and 'temporary'not in p.relative_to(FINAL).parts and(include_model or p.suffix!='.npy')]
    files += [p for p in OUT.glob('*.json')if p.is_file()]+[p for p in (R6/'FINAL_MODEL_MANIFEST.json',R6/'FINAL_MODEL_MANIFEST.sha256')if p.is_file()]
    files += list(HERE.glob('*.*'))
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1)as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
    h=sha(zip_path);zip_path.with_suffix('.zip.sha256').write_text(h+'  '+zip_path.name+'\n',encoding='ascii')
    return {'PASS':True,'ZIP':str(zip_path),'sha256':h,'bytes':zip_path.stat().st_size,'files':len(files),'includes_model':include_model}

def toy_monitor():
    c,_,s,_,_=modules();job=c.LOCAL/'final_once_synthetic'/uuid.uuid4().hex[:8];job.mkdir(parents=True,exist_ok=False)
    os.environ['R6_PERSONAL_SELFTEST']='1'
    limits={'RSS_bytes':4*2**30,'private_commit_bytes':4*2**30,'min_available_physical_bytes':0,'min_remaining_commit_bytes':0}
    state,result=monitor_callable(s.run_worker)('unit-probe',job,blas=6,openmp=1,local=True,local_limits=limits,timeout=30)
    write(OUT/'toy_monitor.json',{'PASS':state['PASS'],'state':state,'result':result,'job':str(job),'scientific_fit_executed':False},True)
    return read(OUT/'toy_monitor.json')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('verify','preflight','toy-tests','monitor-toy','start-final','status','stop','export'))
    p.add_argument('--operator');p.add_argument('--confirm-final-training',action='store_true');p.add_argument('--include-model',action='store_true');a=p.parse_args()
    if a.action=='verify':v=verify()
    elif a.action=='preflight':v=preflight()
    elif a.action=='toy-tests':
        dest=OUT/'synthetic'/uuid.uuid4().hex[:8];dest.mkdir(parents=True)
        with(dest/'pytest.log').open('x',encoding='utf8')as f:
            code=subprocess.call([sys.executable,'-B','-X','utf8','-m','pytest','-q','-p','no:cacheprovider',str(HERE/'test_final_once.py'),'--basetemp',str(dest/'tmp')],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
        require(code==0,'FINAL_ENTRY_SYNTHETIC_TEST_FAILED');v={'PASS':True,'log':str(dest/'pytest.log'),'synthetic_only':True}
    elif a.action=='monitor-toy':v=toy_monitor()
    elif a.action=='start-final':v=start(a.operator,a.confirm_final_training)
    elif a.action=='status':v=status()
    elif a.action=='stop':v=stop()
    else:v=export(a.include_model)
    print(json.dumps(v,ensure_ascii=False,indent=2))
if __name__=='__main__':
    try:main()
    except BaseException:
        dest=OUT/'errors'/uuid.uuid4().hex;dest.mkdir(parents=True,exist_ok=False)
        write(dest/'cli_exception.json',{'UTC':utc(),'traceback':traceback.format_exc(),'automatic_retry':False},True)
        raise
