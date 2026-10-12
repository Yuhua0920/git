"""PowerShell deployment and new endpoint QA, synthetic runs only."""
from pathlib import Path
import argparse,json,shutil,subprocess,sys,time,uuid
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u12b5c_r2'
sys.path.insert(0,str(ROOT/'scripts'));from prepare_sp_u12b5c_r2 import read,dump,sha,need
def run(python):
    qa=OUT/('windows_qa_'+uuid.uuid4().hex[:8]);qa.mkdir();root=qa/'Windows搬迁 Root';root.mkdir();incoming=qa/'Incoming';incoming.mkdir()
    records=dict(read(ROOT/'deployment/b5c_paired_r1_v1/package_manifest.json')['files'])
    records['deployment/b5c_paired_r1_v1/package_manifest.json']={}
    for n in records:
        q=root/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,q)
    old=root/'lab_runs/b5c_paired';(old/'formal').mkdir(parents=True)
    for n in ('STOP.json','FORMAL_START_CONSUMED.json','formal/exception.json','formal/stderr.log'):
        (old/n).write_text('SYNTHETIC_PRESERVED_FAILURE_ONLY\n',encoding='utf8')
    protected={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
    z=OUT/'SP_U12B5C_R2_Execution.zip';installer=incoming/'Install-R2.ps1';shutil.copy2(OUT/'Install-R2.ps1',installer)
    ps=['powershell','-NoProfile','-ExecutionPolicy','Bypass'];tests=[]
    def call(label,command,success=True):
        r=subprocess.run(command,cwd=root,capture_output=True,encoding='utf8',errors='replace')
        (qa/(label+'.stdout.log')).write_text(r.stdout,encoding='utf8');(qa/(label+'.stderr.log')).write_text(r.stderr,encoding='utf8')
        ok=r.returncode==0 if success else r.returncode!=0
        tests.append({'test':label,'PASS':ok,'exit_code':r.returncode});need(ok,'WINDOWS_QA_FAILED:'+label)
        for n,h in protected.items():need(sha(root/n)==h,'HISTORICAL_FILE_CHANGED:'+n)
        print(label,'PASS',flush=True);return r
    install=ps+['-File',str(installer),'-ProjectRoot',str(root),'-ZipPath',str(z),'-ExpectedSHA256',sha(z),'-Python',python]
    call('wrong_ZIP_SHA_refused',install[:-3]+['0'*64,'-Python',python],False)
    call('add_only_install',install)
    call('duplicate_install_refused',install,False)
    wrapper=root/'deployment/b5c_paired_r2_v1/R2.ps1'
    prefix=ps+['-File',str(wrapper),'-Python',python]
    for action in ('Verify','ToyTests'):call(action,prefix+['-Action',action])
    # No actual R1 PASS/model/data is copied to this relocation QA root.
    pf=call('missing_R1_preflight_refused',prefix+['-Action','Preflight'],False)
    need('LAB_R1_IDENTITY_AUDIT_COMPLETE_PASS_REQUIRED' in pf.stderr,'WRONG_PREREQUISITE_REFUSAL')
    call('missing_operator_switch_refused',prefix+['-Action','Run'],False)
    launch=call('missing_R1_formal_entry_refused',prefix+['-Action','Run','-ConfirmFormalEvaluation'],False)
    need('LAB_R1_IDENTITY_AUDIT_COMPLETE_PASS_REQUIRED' in launch.stderr,'FORMAL_ENTRY_NOT_GATED')
    call('MonitorToy_success',prefix+['-Action','MonitorToy'])
    call('Status',prefix+['-Action','Status']);call('Export',prefix+['-Action','Export'])
    call('background_MonitorToy',prefix+['-Action','MonitorToy','-Background'])
    start=time.monotonic()
    while time.monotonic()-start<30:
        paths=list((root/'lab_runs/b5c_paired_r2/synthetic').glob('monitor_*/status.json'))
        if len(paths)>=2 and all(read(q)['state']=='COMPLETE' and read(q)['PASS'] for q in paths):break
        time.sleep(.15)
    else:raise RuntimeError('BACKGROUND_SYNTHETIC_NOT_COMPLETE')
    call('background_Status',prefix+['-Action','Status'])
    controller=root/'deployment/b5c_paired_r2_v1/control.py'
    log=(qa/'long_monitor.log').open('x',encoding='utf8')
    child=subprocess.Popen([python,'-B','-X','utf8',str(controller),'MonitorToy','--probe','long'],cwd=root,stdout=log,stderr=log)
    start=time.monotonic();job=None
    while time.monotonic()-start<25:
        for q in (root/'lab_runs/b5c_paired_r2/synthetic').glob('monitor_*/status.json'):
            if read(q)['state']=='RUNNING' and (q.parent/'monitor_ready.json').exists():job=q.parent;break
        if job:break
        time.sleep(.1)
    need(job is not None,'STOP_FIXTURE_NOT_RUNNING');call('Stop',prefix+['-Action','Stop'])
    code=child.wait(timeout=25);log.close();s=read(job/'status.json')
    ok=code!=0 and s['state']=='STOP' and s['stop_reason']=='USER_STOP_REQUESTED'
    tests.append({'test':'owned_process_tree_stopped','PASS':ok,'exit_code':code});need(ok,'STOP_DID_NOT_TERMINATE_R2_JOB')
    call('stopped_Status',prefix+['-Action','Status']);call('stopped_Export',prefix+['-Action','Export'])
    need(not(root/'lab_runs/b5c_paired_r2/formal').exists() and not(root/'lab_runs/b5c_paired_r2/FORMAL_START_CONSUMED.json').exists(),'FORMAL_SCIENCE_OCCURRED')
    for n,h in protected.items():need(sha(root/n)==h,'HISTORICAL_FILE_CHANGED')
    result={'PASS':True,'tests':tests,'QA_root':str(root),'protected_files':len(protected),'historical_changes':[],
        'formal_predictions':0,'real_R1_identity_audit_pass_claimed':False,'real_model_or_data_transferred':False,'actual_LAB_failure_checked':False,
        'scope':'Actual relocated Windows PowerShell; old failure markers synthetic; missing real R1 PASS blocks formal launch; only bounded monitor toys execute'}
    dump(OUT/'windows_qa.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--python',default=sys.executable);run(p.parse_args().python)
