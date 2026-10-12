"""Personal synthetic relocation tests. This helper is excluded from the lab ZIP."""
from pathlib import Path
import sys,json,subprocess,time,hashlib,shutil,zipfile
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'lab_runs/r7_evidence_audit_r1/preparation_personal'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def tests():
    job=OUT/('test_attempt_'+str(sum(1 for p in OUT.glob('test_attempt_*'))+1));job.mkdir(exist_ok=False)
    files=[p for p in HERE.iterdir()if p.is_file()];before={p.name:sha(p)for p in files};(job/'source_snapshot').mkdir()
    for p in files:
        (job/'source_snapshot'/p.name).write_bytes(p.read_bytes())
        if p.suffix=='.py':compile(p.read_text(encoding='utf8'),str(p),'exec')
    dump(job/'source_freeze.json',before)
    with(job/'pytest.log').open('x',encoding='utf8')as f:code=subprocess.call([sys.executable,'-B','-X','utf8','-m','pytest','-q',str(HERE/'test_audit.py'),str(HERE/'test_cache_regression.py'),'-p','no:cacheprovider','--basetemp',str(job/'pytest_tmp')],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
    result={'exit_code':code,'source_unchanged':before=={p.name:sha(p)for p in files},'log':str(job/'pytest.log')};dump(job/'execution.json',result)
    print((job/'pytest.log').read_text(encoding='utf8')[-7000:]);print(json.dumps(result));assert code==0 and result['source_unchanged']
def deployment():
    job=OUT/('deployment_qa_'+time.strftime('%Y%m%d_%H%M%S'));job.mkdir(exist_ok=False);root=job/'实验室移植 Project Root';root.mkdir();incoming=job/'incoming';incoming.mkdir()
    cfg=json.loads((HERE/'trusted_inputs.json').read_text(encoding='utf8'))
    # Only frozen source/metadata for synthetic deployment. No preparation payloads,
    # scientific candidate arrays or B2/B3 files are copied to this technical fixture.
    for n in cfg['dependencies']:
        if n.endswith(('.npy','.exe','.whl'))or n.startswith('lab_runs/')or n in('migration_manifest.json','necessary_files.csv','migration_provenance.json'):continue
        src=ROOT/n
        if src.is_file():dst=root/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes())
    # Original v1 tool copies are immutable compatibility inputs, no scientific data.
    for src in (ROOT/'deployment/r7_evidence_audit_v1').iterdir():
        if src.is_file() and src.name not in ('prepare_personal.py','personal_qa.py','delivery_personal.py'):
            dst=root/'deployment/r7_evidence_audit_v1'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes())
    from preservation import snapshot,FAILED_ATTEMPT,REQUIRED
    failure=root/'lab_runs/r7_evidence_audit/runs'/FAILED_ATTEMPT;failure.mkdir(parents=True)
    for n in REQUIRED:(failure/n).write_bytes(b'SYNTHETIC ONLY: first-failure-preservation technical fixture')
    protected_before=snapshot(root)
    old_files={p.relative_to(root).as_posix():sha(p)for p in(root/'deployment/r7_evidence_audit_v1').iterdir()if p.is_file()}
    zip_path=OUT/'package/SP_U12B5_R7_Audit_R1_Fix.zip';h=sha(zip_path);localzip=incoming/zip_path.name;shutil.copy2(zip_path,localzip)
    installer=incoming/'Install-R7AuditR1.ps1';shutil.copy2(HERE/'Install-R7AuditR1.ps1',installer)
    def command(name,args,success=True):
        r=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass',*args],capture_output=True,text=True,encoding='utf8',errors='replace',cwd=ROOT)
        (job/(name+'.log')).write_text(r.stdout+'\n'+r.stderr,encoding='utf8');assert(r.returncode==0)==success,(name,r.returncode,r.stdout,r.stderr);return r
    command('install',['-File',str(installer),'-ProjectRoot',str(root),'-ZipPath',str(localzip),'-ExpectedSha256',h])
    command('overwrite_refused',['-File',str(installer),'-ProjectRoot',str(root),'-ZipPath',str(localzip),'-ExpectedSha256',h],False)
    ps=root/'deployment/r7_evidence_audit_r1_v1/R7AuditR1.ps1'
    def action(label,**kwargs):
        argv=['-File',str(ps),'-Action',label,'-PythonPath',sys.executable]
        for k,v in kwargs.items():argv.append('-'+k);argv.extend([]if v is True else[str(v)])
        return command(label,argv)
    action('Verify',CheckFailedAttempt=True);action('ToyTests');action('MonitorToy',Background=True)
    deadline=time.monotonic()+600;state=None
    while time.monotonic()<deadline:
        p=root/'lab_runs/r7_evidence_audit_r1/latest_toy.json'
        if p.exists():
            folder=Path(json.loads(p.read_text(encoding='utf8'))['job']);statepath=folder/'status.json'
            if statepath.exists():
                state=json.loads(statepath.read_text(encoding='utf8'))
                if state['state']in('COMPLETE','FAILED','STOP'):break
        time.sleep(.25)
    assert state and state['PASS']and state['state']=='COMPLETE',state
    action('Status');action('Export')
    source={p.name:sha(p)for p in HERE.iterdir()if p.is_file()and p.name not in('prepare_personal.py','personal_qa.py','delivery_personal.py')}
    installed={p.name:sha(p)for p in(root/'deployment/r7_evidence_audit_r1_v1').iterdir()if p.is_file()};assert source==installed
    assert snapshot(root)==protected_before
    assert old_files=={n:sha(root/n)for n in old_files}
    report={'PASS':True,'synthetic_failure_fixture_preserved':True,'old_deployment_unchanged':True,'installation':True,'overwrite_refusal':True,'unicode_root':str(root),'actual_task_pytest':True,'background_survived_launch_shell_exit':True,
      'monitor_actual_PID':state['actual_interpreter_PID'],'venv_launcher_PID':state['venv_launcher_PID'],'JobObject_name':state['JobObject_name'],'resource_samples':state['resource_samples'],
      'monitor_lab_models_verified':state['lab_models_verified'],'real_Audit_action_executed':False,'installed_hashes_unchanged':True,'models_fitted':0,'real_validation':0,'selection':False}
    dump(job/'deployment_result.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':tests()if sys.argv[1]=='tests'else deployment()
