"""One frozen LAB evaluation, followed by gated joint analysis."""
import argparse,json,os,subprocess,sys,time,traceback,types,uuid,zipfile
from pathlib import Path
import base as b
def environment():
    import importlib.metadata
    wanted={'numpy':'2.5.3','scipy':'1.18.1','scikit-learn':'1.9.1','threadpoolctl':'3.7.0','pytest':'9.1.1'}
    got={n:importlib.metadata.version(n) for n in wanted}
    b.require(sys.version_info[:3]==(3,13,15) and sys.maxsize>2**32 and got==wanted,'PINNED_CPU_ENVIRONMENT_REQUIRED')
    return {'Python':sys.version,'packages':got,'GPU_used':False}
def scope(job,formal):
    c=b.contract();exact={b.safe(b.ROOT,n) for n in c['source_dependencies']}
    exact|={b.safe(b.ROOT,n) for n in b.read(b.HERE/'package_manifest.json')['files']}
    exact|={b.ROOT/'lab_runs/r6_formal'/n for n in ('FINAL_MODEL_MANIFEST.json','FINAL_MODEL_MANIFEST.sha256')}
    exact|={b.FINAL/'training_result.json',b.FINAL/'model/model_manifest.json'}
    model={b.FINAL/'model'/n for n in c['model_file_names']}
    transferred={b.safe(b.DATA,n) for n in b.data_inventory()} if formal else set()
    runtimes={Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()};job=job.resolve()
    def check(raw,write=False):
        p=Path(os.fsdecode(raw)).resolve()
        if write:
            b.require(p.is_relative_to(b.OUT.resolve()),'HISTORICAL_MODEL_SOURCE_INPUT_WRITE_DENIED:'+str(p));return
        allowed=p in exact or p in model and formal or p in transferred and formal or p.is_relative_to(b.OUT.resolve()) or any(p.is_relative_to(q) for q in runtimes)
        b.require(allowed,'B5C_READ_FIREWALL_DENIED:'+str(p))
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            if os.fsdecode(args[0]).upper() in ('NUL',r'\\.\NUL'):return
            mode,flags=args[1:3];writing=isinstance(mode,str) and any(x in mode for x in 'wax+') or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            check(args[0],writing)
        elif event in ('os.remove','os.rename','os.rmdir','os.mkdir','os.utime','os.link'):
            for p in args[:2] if event in ('os.rename','os.link') else args[:1]:
                if isinstance(p,(str,bytes,os.PathLike)):check(p,True)
        elif event in ('subprocess.Popen','os.system','os.exec'):raise RuntimeError('WORKER_CHILD_EXECUTION_FORBIDDEN')
    sys.addaudithook(hook)
def absent():
    for n in ('FORMAL_START_CONSUMED.json','STOP.json','STOP_REQUESTED.json','execution.lock'):
        b.require(not(b.OUT/n).exists(),'ONE_TIME_STOP_OR_START_PRESENT:'+n)
    b.require(not(b.OUT/'formal').exists(),'FORMAL_OUTPUT_ALREADY_EXISTS')
    concurrent()

def concurrent(hold=False):
    from historical_concurrency import inspect
    result=inspect()
    if not result.report['PASS']:
        result.close()
        raise RuntimeError('CONCURRENT_SCIENTIFIC_TASK_OR_UNCONFIRMED:'+json.dumps(result.report['blockers'],ensure_ascii=False))
    if hold:return result
    result.close();return result.report

def concurrency_audit():
    b.verify()
    from historical_concurrency import inspect
    result=inspect()
    try:
        print(json.dumps(result.report,ensure_ascii=False,indent=2))
        b.require(result.report['PASS'],'CONCURRENT_SCIENTIFIC_TASK_OR_UNCONFIRMED')
        return result.report
    finally:result.close()

def rehearsal(personal=False):
    b.verify();concurrent()
    b.require(not(b.OUT/'execution.lock').exists(),'R2_ALREADY_RUNNING')
    if not personal:environment()
    job=b.OUT/'synthetic'/('rehearsal_'+uuid.uuid4().hex[:12]);job.mkdir(parents=True,exist_ok=False)
    state,result=monitor(job,False,'pipeline')
    b.require(state['PASS'] and result['PASS'] and result['synthetic_only'] and result['formal_predictions']==0,'FULL_REHEARSAL_FAILED')
    return {'PASS':True,'job':job.relative_to(b.ROOT).as_posix(),'status_sha256':b.sha(job/'status.json'),
            'worker_result_sha256':b.sha(job/'worker_result.json'),'personal_synthetic_only':personal,'result':result}

def require_rehearsal():
    receipt=b.read(b.OUT/'LAB_READINESS.json')
    b.require(receipt['PASS'] is True and receipt['lab_preflight_verified'] is True,'LAB_FULL_REHEARSAL_REQUIRED')
    for key,path in [('tool_sha256',b.HERE/'package_manifest.json'),('data_manifest_sha256',b.ROOT/'deployment/b5c_paired_v1/data_manifest.json')]:
        b.require(receipt[key]==b.sha(path),'REHEARSAL_BINDING_CHANGED:'+key)
    b.require(receipt['final_manifest_sha256']==b.contract()['final_manifest_sha256'],'REHEARSAL_MODEL_BINDING_CHANGED')
    dry=receipt['rehearsal'];job=b.safe(b.ROOT,dry['job'])
    b.require(job.is_relative_to(b.OUT/'synthetic') and job.name.startswith('rehearsal_') and dry['personal_synthetic_only'] is False,'LAB_REHEARSAL_JOB_SCOPE')
    for name,key in [('status.json','status_sha256'),('worker_result.json','worker_result_sha256')]:
        b.require(b.sha(job/name)==dry[key],'LAB_REHEARSAL_RECEIPT_CHANGED:'+name)
    state=b.read(job/'status.json');result=b.read(job/'worker_result.json')
    b.require(state['state']=='COMPLETE' and state['PASS'] and state['actual_interpreter_PID']>0 and state['all_descendants_kernel_contained'],'LAB_REHEARSAL_MONITOR_NOT_COMPLETE')
    b.require(result['PASS'] and result['synthetic_only'] and result['formal_predictions']==0 and result['synthetic_scored_positions']==42 and result['pilot_identity_positions']==70,'LAB_REHEARSAL_NOT_FULL_PATH')
    return receipt

def lab_check(r1_job=None):
    # One consolidated LAB command; no new approval record and no formal lease consumed.
    b.verify();environment()
    from historical_concurrency import inspect
    inspection=inspect()
    try:history=inspection.report
    finally:inspection.close()
    b.OUT.mkdir(parents=True,exist_ok=True)
    report={'UTC':b.utc(),'history':history,'PASS':False,'formal_start_consumed':False,'scientific_evaluation_executed':False}
    path=b.OUT/'technical'/('labcheck_'+uuid.uuid4().hex[:12]+'.json')
    print(json.dumps(history,ensure_ascii=False,indent=2))
    if not history['PASS']:
        b.write(path,report,True);raise RuntimeError('LAB_HISTORY_ACTIVE_OR_UNKNOWN: '+str(path))
    try:
        pf=preflight(r1_job);dry=rehearsal(False)
        receipt={'PASS':True,'UTC':b.utc(),'lab_preflight_verified':True,'tool_sha256':b.sha(b.HERE/'package_manifest.json'),
                 'data_manifest_sha256':b.sha(b.ROOT/'deployment/b5c_paired_v1/data_manifest.json'),
                 'final_manifest_sha256':pf['model_identity']['top_manifest_sha256'],'rehearsal':dry}
        if not(b.OUT/'LAB_READINESS.json').exists():b.write(b.OUT/'LAB_READINESS.json',receipt,True)
        require_rehearsal();report.update(PASS=True,preflight=pf,rehearsal=dry)
        b.write(path,report,True);return {'PASS':True,'report':str(path),'formal_start_consumed':False,'rehearsal':dry}
    except BaseException:
        report['exception']=traceback.format_exc();b.write(path,report,True);raise

def preflight(r1_job=None):
    b.verify();absent()
    from r2_gate import check
    gate=check(r1_job);env=environment()
    from science import modules,final_identity
    _,_,_,wr=modules();cap=wr.capacity(wr.hardware());b.require(cap['PASS'],'LAB_CAPACITY_STOP:'+str(cap['reasons']))
    identity=final_identity();data=b.verify_data()
    result={'PASS':True,'R1_identity_audit':gate,'model_identity':identity,'transferred_original_inputs':data,'capacity':cap,'environment':env,
            'tool_freeze_sha256':b.sha(b.HERE/'package_manifest.json'),'formal_predictions':0,'UTC':b.utc()}
    b.write(b.OUT/'preflight.json',result);return result
def monitor(job,formal,probe='success'):
    from science import modules
    _,_,old,wr=modules()
    # Same R6 run_worker code object and kernel monitor; only new namespace/worker.
    g=dict(old.run_worker.__globals__);g.update(HERE=b.HERE,ROOT=b.ROOT,OUT=b.OUT,write=b.write,
        scientific_counters=lambda:{'automatic_retry':False,'formal_training_executed':False,'validation_recomputed':False})
    fn=types.FunctionType(old.run_worker.__code__,g,old.run_worker.__name__,old.run_worker.__defaults__,old.run_worker.__closure__)
    if formal:
        return fn('evaluate',job,blas=6,openmp=1,timeout=b.contract()['resource']['timeout_seconds'])
    limits={'RSS_bytes':2*2**30,'private_commit_bytes':3*2**30,'min_available_physical_bytes':0,'min_remaining_commit_bytes':0}
    os.environ['R6_PERSONAL_SELFTEST']='1'
    return fn('unit-probe',job,blas=6,openmp=1,local=True,local_limits=limits,timeout=900 if probe=='pipeline' else 30,extra=['--probe',probe])
def start(r1_job=None):
    require_rehearsal()
    pf=preflight(r1_job);b.OUT.mkdir(parents=True,exist_ok=True);consumed=False;history_lease=None
    with (b.OUT/'execution.lock').open('x',encoding='utf8') as f:f.write(str(os.getpid()))
    try:
        # Recheck immediately under the exclusive new lock before consumption.
        from r2_gate import receipt_unchanged
        history_lease=concurrent(hold=True)
        receipt_unchanged(pf['R1_identity_audit'])
        for name in ('STOP.json','STOP_REQUESTED.json','FORMAL_START_CONSUMED.json'):
            b.require(not(b.OUT/name).exists(),'R2_STOP_OR_CONSUMPTION_AFTER_PREFLIGHT:'+name)
        b.require(not(b.OUT/'formal').exists(),'FORMAL_OUTPUT_ALREADY_EXISTS')
        job=b.OUT/'formal';job.mkdir(exist_ok=False)
        freeze={'UTC':b.utc(),'model_manifest_sha256':b.contract()['final_manifest_sha256'],'tool_manifest_sha256':b.sha(b.HERE/'package_manifest.json'),
            'data_manifest_sha256':b.sha(b.ROOT/'deployment/b5c_paired_v1/data_manifest.json'),'contract_sha256':b.sha(b.ROOT/'deployment/b5c_paired_v1/contract.json'),
            'scope':'one independent R2 B5C evaluation and analysis; no fit/selection/validation','automatic_retry':False,'output':'lab_runs/b5c_paired_r2/formal','preflight':pf,'explicit_operator_launch':True}
        b.write(b.OUT/'FORMAL_START_CONSUMED.json',freeze,True);consumed=True
        b.write(job/'execution_freeze.json',freeze,True)
        state,result=monitor(job,True)
        b.require(state['PASS'] and result['PASS'],'FORMAL_INTEGRITY_NOT_PASS')
        receipt_unchanged(pf['R1_identity_audit'])
        # The supervisor confirms resource gates before any aggregate release.
        from report import analyze
        from science import modules
        with modules()[0].threads(6,1):analyze(job)
        b.write(b.OUT/'COMPLETE.json',{'PASS':True,'UTC':b.utc(),'formal_result_sha256':b.sha(job/'worker_result.json'),'analysis_complete':True},True)
        return status()
    except BaseException:
        b.write(b.OUT/'STOP.json',{'UTC':b.utc(),'traceback':traceback.format_exc(),'start_consumed':consumed,'automatic_retry':False},True);raise
    finally:
        if history_lease is not None:history_lease.close()
        (b.OUT/'execution.lock').unlink()
def status():
    if (b.OUT/'STOP.json').exists():return {'state':'STOP','record':b.read(b.OUT/'STOP.json')}
    if (b.OUT/'COMPLETE.json').exists():return {'state':'COMPLETE','record':b.read(b.OUT/'COMPLETE.json')}
    p=b.OUT/'formal/status.json'
    if p.exists():
        answer=b.read(p)
        if answer['state']=='COMPLETE':answer.update(state='ANALYZING',PASS=False)
        return answer
    toys=sorted((b.OUT/'synthetic').glob('*/status.json')) if (b.OUT/'synthetic').exists() else []
    return b.read(max(toys,key=lambda q:q.stat().st_mtime_ns)) if toys else {'state':'NOT_STARTED','formal_executed':False}
def stop():
    monitors=list((b.OUT/'synthetic').glob('*/status.json')) if (b.OUT/'synthetic').exists() else []
    running=[p for p in monitors if b.read(p)['state']=='RUNNING']
    target=max(running,key=lambda q:q.stat().st_mtime_ns).parent/'STOP_REQUESTED.json' if running and not(b.OUT/'formal').exists() else b.OUT/'STOP_REQUESTED.json'
    b.write(target,{'UTC':b.utc(),'automatic_retry':False},True)
    return {'STOP_requested':True,'monitor_will_terminate_owned_JobObject':True}
def export():
    b.require(b.OUT.exists(),'NO_EXECUTION_RECORDS_TO_EXPORT');dest=b.OUT/('export_'+uuid.uuid4().hex[:12]+'.zip')
    b.require(not(b.OUT/'execution.lock').exists(),'WAIT_FOR_STOP_OR_COMPLETION_BEFORE_EXPORT')
    for p in (b.OUT/'synthetic').glob('*/status.json'):
        b.require(b.read(p)['state']!='RUNNING','WAIT_FOR_SYNTHETIC_COMPLETION_BEFORE_EXPORT')
    paths=[p for p in b.OUT.rglob('*') if p.is_file() and 'temporary' not in p.parts and p.suffix not in ('.zip','.sha256')]
    hashes={p.relative_to(b.OUT).as_posix():b.sha(p) for p in paths}
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for p in paths:z.write(p,p.relative_to(b.OUT).as_posix())
        z.writestr('EXPORT_FILE_HASHES.json',json.dumps(hashes,sort_keys=True,indent=2))
    dest.with_suffix('.zip.sha256').write_text(b.sha(dest)+'\n',encoding='ascii')
    return {'path':str(dest),'bytes':dest.stat().st_size,'files':len(paths)+1,'sha256':b.sha(dest),'state':status()['state']}
def toy_tests():
    b.verify();environment();dest=b.OUT/'synthetic'/('pytest_'+uuid.uuid4().hex[:12]);dest.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['TEMP']=env['TMP']=str(dest);env['B5C_TOY_ROOT']=str(dest)
    cmd=[sys.executable,'-B','-m','pytest',str(b.HERE/'test_r2.py'),str(b.HERE/'test_concurrency.py'),str(b.HERE/'test_stable_runtime.py'),'-q','-p','no:cacheprovider','--basetemp',str(dest/'tmp'),'--junitxml',str(dest/'pytest.xml')]
    with (dest/'pytest.log').open('w',encoding='utf8') as f:r=subprocess.run(cmd,cwd=b.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    b.write(dest/'status.json',{'PASS':r.returncode==0,'state':'COMPLETE' if r.returncode==0 else 'FAILED','exit_code':r.returncode,'synthetic_only':True,'formal_predictions':0},True)
    b.require(r.returncode==0,'TOY_TESTS_FAILED');return b.read(dest/'status.json')
def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=('Verify','ConcurrencyAudit','LabCheck','Rehearsal','Preflight','ToyTests','MonitorToy','Status','Stop','Export','Run'));a.add_argument('--probe',choices=('success','exception','long'),default='success');a.add_argument('--formal',action='store_true');a.add_argument('--r1-audit-job');a.add_argument('--personal-selftest',action='store_true');x=a.parse_args()
    b.require(not x.personal_selftest or x.action=='Rehearsal','PERSONAL_SWITCH_SYNTHETIC_ONLY')
    if x.action=='LabCheck':answer=lab_check(x.r1_audit_job)
    elif x.action=='Rehearsal':answer=rehearsal(x.personal_selftest)
    elif x.action=='Run':
        b.require(x.formal,'EXPLICIT_FORMAL_LAB_RUN_SWITCH_REQUIRED');answer=start(x.r1_audit_job)
    elif x.action=='Preflight':answer=preflight(x.r1_audit_job)
    elif x.action=='MonitorToy':
        b.verify();job=b.OUT/'synthetic'/('monitor_'+uuid.uuid4().hex[:12]);job.mkdir(parents=True,exist_ok=False);answer=monitor(job,False,x.probe)[0]
    else:answer={'Verify':b.verify,'ConcurrencyAudit':concurrency_audit,'Preflight':preflight,'ToyTests':toy_tests,'Status':status,'Stop':stop,'Export':export}[x.action]()
    print(json.dumps(answer,ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        print(json.dumps({'state':'STOP','type':type(e).__name__,'message':str(e),'automatic_retry':False},ensure_ascii=False));raise
