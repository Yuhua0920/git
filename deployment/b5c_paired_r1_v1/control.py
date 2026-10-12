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
    for n in ('lab_runs/r6_formal/pipeline.lock','lab_runs/r7_sequential/execution.lock','lab_runs/r8_selection/execution.lock','lab_runs/final_mfa_once/execution.lock'):
        b.require(not(b.ROOT/n).exists(),'CONCURRENT_SCIENTIFIC_TASK:'+n)
def preflight():
    b.verify();absent();env=environment()
    from science import modules,final_identity
    _,_,_,wr=modules();cap=wr.capacity(wr.hardware());b.require(cap['PASS'],'LAB_CAPACITY_STOP:'+str(cap['reasons']))
    identity=final_identity();data=b.verify_data()
    result={'PASS':True,'model_identity':identity,'transferred_original_inputs':data,'capacity':cap,'environment':env,
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
    return fn('unit-probe',job,blas=6,openmp=1,local=True,local_limits=limits,timeout=30,extra=['--probe',probe])
def start():
    b.require(False,'MENTOR_NEW_ATTEMPT_REQUIRED_R1_TECHNICAL_ONLY')
    pf=preflight();b.OUT.mkdir(parents=True,exist_ok=True);consumed=False
    with (b.OUT/'execution.lock').open('x',encoding='utf8') as f:f.write(str(os.getpid()))
    try:
        job=b.OUT/'formal';job.mkdir(exist_ok=False)
        freeze={'UTC':b.utc(),'model_manifest_sha256':b.contract()['final_manifest_sha256'],'tool_manifest_sha256':b.sha(b.HERE/'package_manifest.json'),
            'data_manifest_sha256':b.sha(b.HERE/'data_manifest.json'),'contract_sha256':b.sha(b.HERE/'contract.json'),
            'scope':'one complete B5C evaluation and analysis; no fit/selection/validation','automatic_retry':False,'output':'lab_runs/b5c_paired/formal','preflight':pf}
        b.write(b.OUT/'FORMAL_START_CONSUMED.json',freeze,True);consumed=True
        b.write(job/'execution_freeze.json',freeze,True)
        state,result=monitor(job,True)
        b.require(state['PASS'] and result['PASS'],'FORMAL_INTEGRITY_NOT_PASS')
        # The supervisor confirms resource gates before any aggregate release.
        from report import analyze
        from science import modules
        with modules()[0].threads(6,1):analyze(job)
        b.write(b.OUT/'COMPLETE.json',{'PASS':True,'UTC':b.utc(),'formal_result_sha256':b.sha(job/'worker_result.json'),'analysis_complete':True},True)
        return status()
    except BaseException:
        b.write(b.OUT/'STOP.json',{'UTC':b.utc(),'traceback':traceback.format_exc(),'start_consumed':consumed,'automatic_retry':False},True);raise
    finally:
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
    monitors=list((b.OUT/'synthetic').glob('monitor_*/status.json')) if (b.OUT/'synthetic').exists() else []
    running=[p for p in monitors if b.read(p)['state']=='RUNNING']
    target=max(running,key=lambda q:q.stat().st_mtime_ns).parent/'STOP_REQUESTED.json' if running and not(b.OUT/'formal').exists() else b.OUT/'STOP_REQUESTED.json'
    b.write(target,{'UTC':b.utc(),'automatic_retry':False},True)
    return {'STOP_requested':True,'monitor_will_terminate_owned_JobObject':True}
def export():
    root=b.OUT/'formal' if (b.OUT/'formal').exists() else b.OUT/'synthetic'
    b.require(root.exists(),'NO_EXECUTION_RECORDS_TO_EXPORT');dest=b.OUT/('export_'+uuid.uuid4().hex[:12]+'.zip')
    b.require(not(b.OUT/'execution.lock').exists(),'WAIT_FOR_STOP_OR_COMPLETION_BEFORE_EXPORT')
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
    cmd=[sys.executable,'-B','-m','pytest',str(b.HERE/'test_b5c.py'),'-q','-p','no:cacheprovider','--basetemp',str(dest/'tmp'),'--junitxml',str(dest/'pytest.xml')]
    with (dest/'pytest.log').open('w',encoding='utf8') as f:r=subprocess.run(cmd,cwd=b.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    b.write(dest/'status.json',{'PASS':r.returncode==0,'state':'COMPLETE' if r.returncode==0 else 'FAILED','exit_code':r.returncode,'synthetic_only':True,'formal_predictions':0},True)
    b.require(r.returncode==0,'TOY_TESTS_FAILED');return b.read(dest/'status.json')
def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=('Verify','Preflight','ToyTests','MonitorToy','Status','Stop','Export','Run'));a.add_argument('--probe',choices=('success','exception','long'),default='success');a.add_argument('--formal',action='store_true');x=a.parse_args()
    if x.action=='Run':
        b.require(x.formal,'EXPLICIT_FORMAL_LAB_RUN_SWITCH_REQUIRED');answer=start()
    elif x.action=='MonitorToy':
        b.verify();job=b.OUT/'synthetic'/('monitor_'+uuid.uuid4().hex[:12]);job.mkdir(parents=True,exist_ok=False);answer=monitor(job,False,x.probe)[0]
    else:answer={'Verify':b.verify,'Preflight':preflight,'ToyTests':toy_tests,'Status':status,'Stop':stop,'Export':export}[x.action]()
    print(json.dumps(answer,ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        print(json.dumps({'state':'STOP','type':type(e).__name__,'message':str(e),'automatic_retry':False},ensure_ascii=False));raise
