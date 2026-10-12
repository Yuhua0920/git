"""Personal packaging/Windows synthetic QA; NEVER shipped to the laboratory."""
from pathlib import Path
import sys,os,json,hashlib,csv,zipfile,shutil,subprocess,time,uuid,re
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'lab_runs/r8_selection/preparation_personal'
EXCLUDE={'package_manifest.json','preserve_personal.py','prepare_personal.py'}

def h(p):
    v=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):v.update(b)
    return v.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def package():
    folder=OUT/('package_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True)
    files={p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':h(p)}for p in sorted(HERE.iterdir())if p.is_file()and p.name not in EXCLUDE}
    manifest={'format':'R8_SELECTION_ONLY_INCREMENTAL_V1','namespace':'deployment/r8_selection_v1','files':files,
      'real_selection_executed':False,'approved_human_record_included':False,'contains_models':False,'contains_preparation_payload':False,'contains_B2_B3_outcomes':False,
      'frozen_rule_reused':'deployment/r6_formal_v1/runner.py::selection_audit -> model_ops.automatic_select',
      'original_sources_modified':False}
    (HERE/'package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    members=files|{'deployment/r8_selection_v1/package_manifest.json':{'bytes':(HERE/'package_manifest.json').stat().st_size,'sha256':h(HERE/'package_manifest.json')}}
    zpath=folder/'SP_U12B5_R8_Selection_Only.zip'
    with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED)as z:
        for n in members:z.write(ROOT/n,n)
    with zipfile.ZipFile(zpath)as z:
        assert z.testzip()is None and set(z.namelist())==set(members)
        for n,v in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
    (folder/(zpath.name+'.sha256')).write_text(h(zpath)+'  '+zpath.name+'\n',encoding='ascii')
    with(folder/'SP_U12B5_R8_File_Manifest.csv').open('x',encoding='utf8',newline='')as f:
        w=csv.DictWriter(f,fieldnames=['relative_path','bytes','sha256']);w.writeheader();w.writerows({'relative_path':n,**v}for n,v in members.items())
    shutil.copyfile(HERE/'Install-R8Selection.ps1',folder/'Install-R8Selection.ps1')
    result={'PASS':True,'ZIP':str(zpath),'SHA256':h(zpath),'bytes':zpath.stat().st_size,'files':len(members),'members':members,
      'manifest_SHA256':h(HERE/'package_manifest.json'),'real_selection_executed':False}
    put(folder/'package_report.json',result);(OUT/'latest_package.json').write_text(json.dumps({'folder':str(folder)})+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items()if k!='members'},ensure_ascii=False),flush=True)

def technical_root(folder):
    # This relocated personal interpreter loses sys.path[0] for long script paths;
    # the actual laboratory D:\Research\Project_MFA_Lab is shorter still.
    root=OUT.parent/('win_'+uuid.uuid4().hex[:8])/'迁移 Root';root.mkdir(parents=True)
    settings=read(HERE/'trusted_inputs.json');binding=read(HERE/'selection_contract.json')['code_bindings']
    selected=[]
    for n in settings['dependencies']:
        if n in binding or n.endswith('.py')or n.startswith(('deployment/r6_formal_v1/','deployment/r7_sequential_v1/'))and n.endswith('.json')or n in('configs/sp_u12b5b_mfa_selection.yaml','outputs/sp_u12b5b_candidate_seed_registry.json','deployment/mfa_lab_v1/environment_source.json'):
            source=ROOT/n
            if source.is_file():target=root/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);selected.append(n)
    # These are original R1 deployment bytes, not real evidence/results/models.
    old=ROOT/'deployment/r7_evidence_audit_r1_v1'
    for n in list(read(old/'package_manifest.json')['files'])+['deployment/r7_evidence_audit_r1_v1/package_manifest.json']:
        target=root/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,target);selected.append(n)
    n='deployment/r5_128gb_v1/package_manifest.json';target=root/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,target);selected.append(n)
    (root/'lab_runs/r6_formal').mkdir(parents=True,exist_ok=True)
    assert not list(root.rglob('*.npy')) and not list(root.rglob('frame_audit.csv'))
    put(root/'SYNTHETIC_TECHNICAL_RELOCATION_ONLY.json',{'models_copied':0,'preparation_payload_copied':0,'real_validation_copied':0,'B2_B3_copied':0,'real_selection_authorized':False})
    return root,{n:h(root/n)for n in set(selected)}

def windows_qa():
    pkg=read(Path(read(OUT/'latest_package.json')['folder'])/'package_report.json')
    folder=OUT/('windows_qa_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);folder.mkdir()
    root,old=technical_root(folder);shell=shutil.which('powershell');env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6',OMP_NUM_THREADS='1')
    checks=[]
    def ps(label,args,expected=0):
        log=folder/(label+'.log')
        with log.open('x',encoding='utf8')as f:
            r=subprocess.run([shell,'-NoProfile','-ExecutionPolicy','Bypass','-File',*map(str,args)],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
        checks.append({'test':label,'exit_code':r.returncode,'PASS':r.returncode==expected,'log':str(log)})
        print(json.dumps(checks[-1],ensure_ascii=False),flush=True);assert r.returncode==expected,log
    def action(label,name,*args,expected=0):
        ps(label,[root/'deployment/r8_selection_v1/R8Selection.ps1','-Action',name,'-PythonPath',sys.executable,*args],expected)
    installer=Path(pkg['ZIP']).parent/'Install-R8Selection.ps1'
    ps('wrong_zip_hash',[installer,'-ProjectRoot',root,'-ZipPath',pkg['ZIP'],'-ExpectedSha256','0'*64],1)
    assert not(root/'deployment/r8_selection_v1').exists()
    ps('install',[installer,'-ProjectRoot',root,'-ZipPath',pkg['ZIP'],'-ExpectedSha256',pkg['SHA256']])
    ps('refuse_existing_namespace',[installer,'-ProjectRoot',root,'-ZipPath',pkg['ZIP'],'-ExpectedSha256',pkg['SHA256']],1)
    action('verify','Verify')
    # Independent synthetic fixtures: run installed pytest concurrently with
    # the kernel-contained toy/status/export checks. No scientific job exists.
    test_log=(folder/'toy_tests.log').open('x',encoding='utf8')
    test_process=subprocess.Popen([shell,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(root/'deployment/r8_selection_v1/R8Selection.ps1'),'-Action','ToyTests','-PythonPath',sys.executable],cwd=root,env=env,stdout=test_log,stderr=subprocess.STDOUT)
    action('bindings_no_approval_created','AuthorizationBindings')
    action('missing_independent_authorization','CompleteSelectionOnce','-AuditEvidenceZip',root/'synthetic_no_evidence.zip','-R7ArchiveZip',root/'synthetic_no_archive.zip',expected=1)
    assert not(root/'lab_runs/r6_formal/SELECTION_MANIFEST.json').exists()
    action('start_background_toy','MonitorToy','-Background')
    def wait_job(stop=False):
        started=time.monotonic();out=root/'lab_runs/r8_selection';marker=out/'latest_toy.json';sent=False
        while time.monotonic()-started<600:
            if marker.exists():
                p=Path(read(marker)['job']);s=p/'status.json'
                if s.exists():
                    state=read(s)
                    if stop and not sent and state['state']=='RUNNING'and state.get('actual_interpreter_PID'):
                        action('stop_background_toy','Stop','-JobPath',p);sent=True
                    if state['state']in('COMPLETE','FAILED','STOP'):return p,state
            time.sleep(.2)
        raise RuntimeError('WINDOWS_SYNTHETIC_MONITOR_TIMEOUT_PRESERVE_NO_RETRY')
    positive,state=wait_job();assert state['state']=='COMPLETE'and state['PASS']and state['actual_interpreter_PID']and state['resource_samples']>0
    action('completed_status','Status','-JobPath',positive);action('completed_export','Export','-JobPath',positive)
    export=positive/'R8_evidence_export.zip';assert h(export)==Path(str(export)+'.sha256').read_text().split()[0]
    # A second independent technical attempt tests STOP; preserve its persistent STOP marker.
    action('start_second_background_toy','MonitorToy','-Background')
    # Ensure latest_toy was changed by its asynchronous controller before inspecting it.
    started=time.monotonic()
    while Path(read(root/'lab_runs/r8_selection/latest_toy.json')['job'])==positive:
        assert time.monotonic()-started<60;time.sleep(.2)
    stopped,stop_state=wait_job(True);assert stop_state['state']=='STOP'and stop_state['stop_reason']=='USER_STOP'
    action('stopped_status','Status','-JobPath',stopped);action('stopped_export','Export','-JobPath',stopped)
    test_exit=test_process.wait();test_log.close();checks.append({'test':'toy_tests','exit_code':test_exit,'PASS':test_exit==0,'log':str(folder/'toy_tests.log')});assert test_exit==0
    assert all(h(root/n)==digest for n,digest in old.items())
    assert not(root/'lab_runs/r6_formal/SELECTION_MANIFEST.json').exists()
    assert not(root/'lab_runs/r8_selection/SELECTION_EXECUTION_CONSUMED.json').exists()
    result={'PASS':True,'scope':'PERSONAL_WINDOWS_SYNTHETIC_RELOCATION_ONLY','technical_root':str(root),'checks':checks,
      'frozen_files_checked':len(old),'frozen_files_changed':0,'background_complete':state,'background_stop':stop_state,
      'selection_manifest_in_canonical_technical_root':False,'approved_record_created':False,'real_selection_executed':False,'final_model_trained':False,'target_access':0,
      'ZIP_SHA256':pkg['SHA256'],'toy_export_SHA256':h(export),'independent_synthetic_checks_run_concurrently':True}
    put(folder/'windows_qa.json',result);(OUT/'latest_windows_qa.json').write_text(json.dumps({'folder':str(folder)})+'\n',encoding='utf8')
    print(json.dumps({'PASS':True,'report':str(folder/'windows_qa.json'),'checks':len(checks)},ensure_ascii=False),flush=True)

def report():
    p=read(Path(read(OUT/'latest_package.json')['folder'])/'package_report.json')
    w=read(Path(read(OUT/'latest_windows_qa.json')['folder'])/'windows_qa.json')
    preserved=read(OUT/'historical_delivery_preservation.json')
    original=read(ROOT/'lab_runs/r8_selection/latest_tests.json');old_job=Path(original['job'])
    installed=read(Path(w['technical_root'])/'lab_runs/r8_selection/latest_tests.json');new_job=Path(installed['job'])
    tests=[]
    for label,job in [('personal_development',old_job),('installed_final_package',new_job)]:
        s=read(job/'status.json');log=(job/'pytest.log').read_text(encoding='utf8');m=re.search(r'(\d+) passed in ([\d.]+)s',log)
        assert s['PASS']and m and int(m[1])in(44,47)
        tests.append({'scope':label,'passed':int(m[1]),'seconds':float(m[2]),'job':str(job),'log_SHA256':h(job/'pytest.log'),'status':s})
    evidence=read(OUT/'authorization_bindings_final.json')
    import selection_core as a
    assert a.verify()['PASS']and h(Path(p['ZIP']))==p['SHA256']and evidence['bindings']['R8_package_manifest_SHA256']==p['manifest_SHA256']
    assert preserved['PASS']and w['PASS']
    assert not(ROOT/'lab_runs/r6_formal/SELECTION_MANIFEST.json').exists()
    assert not(ROOT/'lab_runs/r8_selection/SELECTION_EXECUTION_CONSUMED.json').exists()
    assert not(ROOT/'lab_runs/r8_selection/authorization').exists()
    mapping=[
      ('pooled_ratio','deployment/r6_formal_v1/model_ops.py','sum error/sum truth in each origin-SNR cell'),
      ('macro_score','deployment/r6_formal_v1/model_ops.py','six-cell equal arithmetic average in linear domain'),
      ('automatic_select','deployment/r6_formal_v1/model_ops.py','min (exact float macro score,K,rank); all nine retained'),
      ('selection_audit','deployment/r6_formal_v1/runner.py','unchanged raw Validation reconstruction and automatic selection'),
      ('selection_frozen','deployment/r6_formal_v1/runner.py','unchanged reader actually passed against synthetic R8 manifest')]
    with(OUT/'frozen_rule_code_audit.csv').open('x',encoding='utf8',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=['function','file','source_sha256','unchanged','role']);writer.writeheader()
        for function,n,role in mapping:
            digest=h(ROOT/n);assert digest==a.cfg()['dependencies'][n]['sha256'];writer.writerow({'function':function,'file':n,'source_sha256':digest,'unchanged':True,'role':role})
    result={'verdict':'R8_SELECTION_ONLY_DEPLOYMENT_READY','package':p,'pytest':tests,'Windows_qa':w,'personal_preservation':preserved,
      'authorization_bindings':evidence,'real_selection_executed':False,'final_model_trained':False,'target_access':0,
      'approved_production_record_created':False,'selection_reader_compatibility':'PASSED_WITH_UNCHANGED_FROZEN_READER_ON_SYNTHETIC_RAW_RECORDS',
      'lab_live_model_physical_verification_performed_here':False,
      'scope_limitations':['Only personally executed Windows synthetic deployment, not laboratory execution.','Human attestation with independent expected approval hash, not cryptographic signer authentication.','This relocated personal interpreter loses sys.path[0] at a 275-character script path; explicit FileFinder still finds the source. Use the specified short lab root; no universal long-path claim.','Two exclusive fsynced files are not atomic as a pair; any partial publication is terminal and preserved, no retry.'],
      'retained_development_failures':{'first_toy':'missing technical environment_source fixture metadata','windows_qa_first':'missing synthetic R5 manifest prerequisite; installer correctly refused','windows_qa_second':'long-script startup loses sys.path[0] in this relocated runtime; 16 failed/28 passed retained, no scientific files involved','windows_qa_third':'all 44 tests passed; toy monitor FAILED due WinError5 status.json rename/read-handle sharing. Preserved old ZIP and logs; repaired only mutable R8 status/log publication.'},
      'next_step':'STOP for mentor review; no formal selection authorization issued'}
    put(OUT/'r8_delivery_report.json',result)
    print(json.dumps({'verdict':result['verdict'],'report':str(OUT/'r8_delivery_report.json'),'ZIP':p['ZIP'],'SHA256':p['SHA256'],'bytes':p['bytes'],'files':p['files'],'pytest':[{k:t[k]for k in('scope','passed','seconds')}for t in tests],'old_files_changed':len(preserved['changed']),'real_selection_executed':False,'final_model_trained':False,'target_access':0},ensure_ascii=False),flush=True)

if __name__=='__main__':
    {'package':package,'windows-qa':windows_qa,'report':report}[sys.argv[1]]()
