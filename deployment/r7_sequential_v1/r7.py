"""R7 public technical / queue entry. No selection, final, or target action."""
from pathlib import Path
import os,sys,json,argparse,subprocess,time,zipfile,hashlib,traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import r7_common as c

def status():
    state=c.read(c.OUT/'queue.json') if (c.OUT/'queue.json').exists() else {'state':'NOT_STARTED','completed_candidates':[],'remaining_candidates':[r['name'] for r in c.contract()['candidate_order']]}
    if state['state']=='QUEUE_RUNNING':
        live=c.process_identity(state['controller_PID'])==state['controller_identity']
        if not live:state={**state,'state':'QUEUE_FAILED','STOP_reason':'CONTROLLER_EXITED_WITHOUT_TERMINAL_RECEIPT; NO_RETRY','orphan_status_detected':True}
        else:
            from datetime import datetime,timezone
            state['elapsed_seconds_live']=(datetime.now(timezone.utc)-datetime.fromisoformat(state['started_UTC'])).total_seconds()
    active=c.read(c.R6OUT/'active_worker.json') if (c.R6OUT/'active_worker.json').exists() else None
    if active:
        p=c.ROOT/active['job'];state['active_stage_job']=str(p)
        if (p/'status.json').is_file():state['stage_monitor']=c.read(p/'status.json')
        if (p/'phase.json').is_file():state['phase_progress']=c.read(p/'phase.json')
        if (p/'r7_phase.json').is_file():state['timing_phase_progress']=c.read(p/'r7_phase.json')
    return {'authorization':'MENTOR_APPROVAL_PRESENT' if (c.OUT/'MENTOR_APPROVAL.json').exists() else 'MENTOR_APPROVAL_PENDING',
            'queue':state,'selection_automated':False,'final_fit_automated':False}

def stop():
    # STOP is sticky; a later Start refuses even before any scientific input.
    c.write(c.OUT/'STOP_REQUESTED.json',{'UTC':c.utc(),'reason':'USER_STOP_REQUESTED'})
    killed=False
    if (c.R6OUT/'active_worker.json').exists():
        _,_,_,w=c.r6_imports();killed=w.stop_named_job(c.read(c.R6OUT/'active_worker.json')['JobObject_name'])
    return {'PASS':True,'kernel_job_terminated':killed,'automatic_retry':False,'restart_requires_new_mentor_task':True}

def export():
    v=status();c.require(v['queue']['state'] in ('QUEUE_COMPLETE','QUEUE_STOPPED','QUEUE_FAILED') or v['queue']['state']=='NOT_STARTED','EXPORT_WAIT_FOR_QUEUE_END')
    c.require(not (c.OUT/'execution.lock').exists() and not (c.R6OUT/'pipeline.lock').exists(),'EXPORT_BLOCKED_BY_EXECUTION_LOCK')
    folder=c.OUT/'exports';folder.mkdir(parents=True,exist_ok=True)
    path=folder/('SP_U12B5_R7_Results_'+time.strftime('%Y%m%d_%H%M%S')+'.zip')
    files=[]
    for base,prefix in ((c.OUT,'r7_sequential'),(c.R6OUT/'candidates','r6_candidates')):
        for p in base.rglob('*'):
            if p.is_file() and not p.is_relative_to(folder) and not p.is_relative_to(c.OUT/'technical') and 'temporary' not in p.parts and p.suffix in ('.json','.jsonl','.csv','.log','.txt'):
                files.append((p,prefix+'/'+p.relative_to(base).as_posix()))
    manifest={name:{'sha256':c.digest(p),'bytes':p.stat().st_size} for p,name in files}
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p,name in files:z.write(p,name)
        z.writestr('result_manifest.json',json.dumps({'files':manifest,'bindings':c.bindings(),'candidate_NPY_models_retained_on_lab':True,'preparation_arrays_included':False},indent=2))
    with zipfile.ZipFile(path) as z:
        for name,row in manifest.items():
            h=hashlib.sha256()
            with z.open(name) as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            c.require(h.hexdigest()==row['sha256'],'EXPORT_CAPTURE_DRIFT')
    h=c.digest(path);path.with_suffix('.zip.sha256').write_text(h+'  '+path.name+'\n',encoding='ascii')
    return {'PASS':True,'path':str(path),'sha256':h,'bytes':path.stat().st_size,'files':len(files)+1}

def technical_tests():
    c.require(not (c.R6OUT/'pipeline.lock').exists() and not (c.R6OUT/'candidates').exists() and not (c.OUT/'queue.json').exists(),'TOY_TEST_AFTER_FORMAL_START_FORBIDDEN')
    c.verify();folder=c.OUT/'technical'/('attempt_'+time.strftime('%Y%m%d_%H%M%S'));folder.mkdir(parents=True,exist_ok=False)
    tested_manifest=c.bindings()['R7_package_manifest_SHA256']
    source_hashes={p.name:c.digest(p) for p in HERE.iterdir() if p.is_file()}
    snapshot=folder/'source_snapshot';snapshot.mkdir()
    for source in HERE.iterdir():
        if source.is_file(): (snapshot/source.name).write_bytes(source.read_bytes())
    env=os.environ.copy();env['R7_TECHNICAL_TEST_ROOT']=str(folder);env['PYTHONDONTWRITEBYTECODE']='1';env['TEMP']=env['TMP']=str(folder/'tmp');(folder/'tmp').mkdir()
    result=subprocess.run([sys.executable,'-B','-X','utf8','-m','pytest',str(HERE/'test_r7.py'),'-q','-p','no:cacheprovider','--basetemp',str(folder/'pytest_tmp')],env=env,cwd=c.ROOT,capture_output=True,text=True,encoding='utf8',errors='replace')
    (folder/'pytest.log').write_text(result.stdout+'\nSTDERR\n'+result.stderr,encoding='utf8')
    unchanged=source_hashes=={p.name:c.digest(p) for p in HERE.iterdir() if p.is_file()} and tested_manifest==c.bindings()['R7_package_manifest_SHA256']
    receipt={'PASS':result.returncode==0 and unchanged,'source_unchanged_during_tests':unchanged,'exit_code':result.returncode,'pytest_stdout':result.stdout,'pytest_log':str(folder/'pytest.log'),
             'R7_package_manifest_SHA256':tested_manifest,'host':os.environ.get('COMPUTERNAME'),
             'real_candidate_training':0,'real_validation_scores':0,'synthetic_only':True,'UTC':c.utc()}
    c.write(folder/'receipt.json',receipt,exclusive=True)
    c.write(c.OUT/'technical_tests.json',receipt)
    c.require(receipt['PASS'],'R7_TECHNICAL_TESTS_FAILED_NO_FORMAL_START');return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('verify','technical-tests','authorization-bindings','import-approval','preflight','start','status','stop','export'))
    p.add_argument('--approval',type=Path);p.add_argument('--instruction',type=Path);a=p.parse_args()
    try:
        if a.action=='verify':v=c.verify()
        elif a.action=='technical-tests':v=technical_tests()
        elif a.action=='authorization-bindings':v={'status':'MENTOR_APPROVAL_PENDING','expected_bindings':c.bindings(),'candidate_order':c.contract()['candidate_order'],'resource_contract':c.contract()['resource_contract'],'record_generated':False}
        elif a.action=='import-approval':v=c.import_approval(a.approval,a.instruction)
        elif a.action=='preflight':v=c.formal_preflight();c.write(c.OUT/'preflight.json',v)
        elif a.action=='start':
            # Authorization before data or worker imports; no bypass / force switch.
            c.approval();c.formal_preflight()
            from formal_backend import FormalBackend,install_parent_firewall
            from queue_engine import run_queue
            backend=FormalBackend();install_parent_firewall()
            with backend.r.lock():v=run_queue(c.OUT,c.contract()['candidate_order'],backend)
        elif a.action=='status':v=status()
        elif a.action=='stop':v=stop()
        else:v=export()
        print(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),flush=True)
    except BaseException as e:
        # A denied entry is never a fake scientific attempt or a successful queue.
        c.write(c.OUT/'refusals'/('refusal_'+str(time.time_ns())+'.json'),{'UTC':c.utc(),'action':a.action,'reason':str(e),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
