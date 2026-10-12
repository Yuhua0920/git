"""Technical preservation/transport tests only, never scientific evaluation."""
import argparse,csv,json,os,subprocess,sys,time,uuid,zipfile
from pathlib import Path
import base as b
P=b.ROOT/'outputs/sp_u12b5c_preparation'
def dump(q,v):q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def snapshot():
    paths={b.ROOT/n for n in b.contract()['source_dependencies']}
    paths|={q for q in (b.ROOT/'paper').rglob('*') if q.is_file() and q.suffix not in ('.aux','.log','.toc','.out','.synctex','.gz')}
    paths|={b.ROOT/n for n in ('FINAL_MODEL_MANIFEST.json','FINAL_MODEL_MANIFEST.sha256','outputs/sp_u12b2b_sealed_manifest.json','outputs/sp_u12b3h_manifest.json','outputs/sp_u12b3h_preservation.json','outputs/sp_u12b3h_history_preservation.json')}
    return {'files':{q.relative_to(b.ROOT).as_posix():b.sha(q) for q in sorted(paths)},
      'git_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.ROOT,text=True).strip(),
      'tracked_git_diff':subprocess.check_output(['git','diff','--binary'],cwd=b.ROOT,text=True),
      'staged_git_diff':subprocess.check_output(['git','diff','--cached','--binary'],cwd=b.ROOT,text=True)}
def before():dump(P/'preservation_before.json',snapshot())
def after():
    a=b.read(P/'preservation_before.json');z=snapshot();changed=[n for n,h in a['files'].items() if z['files'].get(n)!=h]
    report={'PASS':not changed and a['git_HEAD']==z['git_HEAD'] and a['tracked_git_diff']==z['tracked_git_diff'] and a['staged_git_diff']==z['staged_git_diff'],
      'protected_files_checked':len(a['files']),'changed':changed,'Git_commit_created':False,'manuscript_modifications':0 if not any(n.startswith('paper/') for n in changed) else None,
      'source_dependency_modifications':0 if not any(n in b.contract()['source_dependencies'] for n in changed) else None,
      'B2_original_minimum_files_authenticated_during_opaque_packaging':30008,'B3_seal_metadata_hashes_unchanged':not any('sp_u12b3h' in n for n in changed),
      'B3_formal_performance_decoded':False,'B2_formal_performance_decoded':False,'formal_predictions_on_personal':0,
      'scope':'Byte hashes of source/manuscript/final top/B2-B3 technical metadata; data-package pass separately authenticates actual 30008 original B2 files. No claim of full B3 numeric re-audit.'}
    dump(P/'preservation_after.json',report);b.require(report['PASS'],'PRESERVATION_DRIFT');print(json.dumps(report,indent=2))
def relocation(python):
    qa=P/('windows_qa_'+uuid.uuid4().hex[:8]);qa.mkdir(exist_ok=False);incoming=qa/'Incoming';incoming.mkdir();root=qa/'搬迁 Root';root.mkdir()
    # A synthetic stand-in data ZIP tests transport. Never label it formal inputs.
    tiny=incoming/'synthetic_data.zip'
    with zipfile.ZipFile(tiny,'x') as z:z.writestr('SYNTHETIC_ONLY.txt','No scientific records.\n')
    code=P/'SP_U12B5C_Execution.zip';install=b.HERE/'Install-B5C.ps1'
    args=['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(install),'-ProjectRoot',str(root),'-CodeZip',str(code),'-ExpectedCodeSHA256',b.sha(code),'-DataZip',str(tiny),'-ExpectedDataSHA256',b.sha(tiny)]
    # Installer expects .venv-lab. Create a path junction to the existing pinned runtime, not in delivery ZIP.
    venv=Path(python).resolve().parents[1]
    r=subprocess.run(['cmd','/c','mklink','/J',str(root/'.venv-lab'),str(venv)],capture_output=True,text=True)
    b.require(r.returncode==0,'QA_RUNTIME_JUNCTION_FAILED')
    reports=[]
    def call(label,command,expected=0):
        r=subprocess.run(command,capture_output=True,text=True,encoding='utf8',cwd=root)
        (qa/(label+'.stdout.log')).write_text(r.stdout,encoding='utf8');(qa/(label+'.stderr.log')).write_text(r.stderr,encoding='utf8')
        ok=r.returncode==expected if expected==0 else r.returncode!=0
        reports.append({'test':label,'PASS':ok,'exit_code':r.returncode});b.require(ok,'WINDOWS_QA_FAILED:'+label);return r
    call('install_add_only',args)
    # Duplicate namespaces must reject without changing original code.
    call('repeat_install_refused',args,1)
    wrapper=root/'deployment/b5c_paired_v1/B5C.ps1'
    prefix=['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(wrapper)]
    for action in ('Verify','ToyTests','MonitorToy','Status','Export'):call(action,prefix+['-Action',action])
    call('formal_switch_refused',prefix+['-Action','Run'],1)
    call('personal_host_formal_refused',prefix+['-Action','Run','-ConfirmFormalEvaluation'],1)
    b.require(not(root/'lab_runs/b5c_paired/FORMAL_START_CONSUMED.json').exists(),'PERSONAL_FORMAL_START_CONSUMED')
    call('background_launch',prefix+['-Action','MonitorToy','-Background'])
    time.sleep(4);call('background_status',prefix+['-Action','Status'])
    statuses=list((root/'lab_runs/b5c_paired/synthetic').glob('monitor_*/status.json'))
    b.require(all(b.read(q)['state']=='COMPLETE' and b.read(q)['PASS'] for q in statuses),'BACKGROUND_DID_NOT_COMPLETE')
    # STOP a separate bounded long monitor; preserve its independent failure.
    controller=root/'deployment/b5c_paired_v1/control.py'
    log=(qa/'stop_monitor.log').open('w',encoding='utf8')
    child=subprocess.Popen([python,'-B',str(controller),'MonitorToy','--probe','long'],cwd=root,stdout=log,stderr=log)
    start=time.monotonic();newjob=None
    while time.monotonic()-start<20:
        for q in (root/'lab_runs/b5c_paired/synthetic').glob('monitor_*/status.json'):
            if b.read(q)['state']=='RUNNING':newjob=q.parent;break
        if newjob:break
        time.sleep(.1)
    b.require(newjob is not None,'STOP_FIXTURE_NOT_RUNNING');call('Stop',prefix+['-Action','Stop'])
    exitcode=child.wait(timeout=20);log.close();status=b.read(newjob/'status.json')
    reports.append({'test':'stopped_owned_process_tree','PASS':exitcode!=0 and status['state']=='STOP' and status['stop_reason']=='USER_STOP_REQUESTED','exit_code':exitcode})
    b.require(reports[-1]['PASS'],'STOP_NOT_ENFORCED');call('stop_evidence_export',prefix+['-Action','Export'])
    report={'PASS':all(r['PASS'] for r in reports),'tests':reports,'root':str(root),'real_data_transferred':False,'real_model_verified':False,'formal_predictions':0,'no_automatic_retry':True}
    dump(P/'windows_qa.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
def figure_fixture():
    from report import result_draft
    from paired_stats import CONDITIONS,SNRS
    rows=[]
    for ci,c in enumerate(CONDITIONS):
        for si,s in enumerate(SNRS):
            gain=(si-3)*.2;rows.append({'condition':c,'snr_db':s,'N_trajectories':1000,'LR4_NMSE_dB':-2-si*1.4+ci*.5,'MFA_NMSE_dB':-2-si*1.4+ci*.5+gain,
                'MFA_minus_LR4_dB':gain,'simultaneous_CI_lower_dB':gain-.3,'simultaneous_CI_upper_dB':gain+.3,
                'classification':'LR4_SUPPORTED' if gain>.3 else 'MFA_SUPPORTED' if gain<-.3 else 'INDETERMINATE'})
    q=P/'synthetic_figure_fixture.csv'
    with q.open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Synthetic figure QA fixture only; not scientific evidence:',q)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=('before','after','relocation','figure-fixture'));a.add_argument('--python',default=sys.executable);x=a.parse_args()
    {'before':before,'after':after,'relocation':lambda:relocation(x.python),'figure-fixture':figure_fixture}[x.action]()
