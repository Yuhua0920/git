"""Personal-only PowerShell ZIP relocation and detached toy-queue acceptance."""
from pathlib import Path
import sys,os,json,time,zipfile,subprocess,hashlib,traceback,ast
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'lab_runs/r7_sequential/preparation_personal'
ATTEMPT=None
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def call(args,cwd=ROOT):
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    r=subprocess.run(args,cwd=cwd,env=env,capture_output=True,text=True,encoding='utf8',errors='replace')
    v={'command':args,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    with (ATTEMPT/'commands.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(v,ensure_ascii=False)+'\n')
    return v
def main():
    global ATTEMPT
    ATTEMPT=OUT/('deployment_qa_'+time.strftime('%Y%m%d_%H%M%S'));ATTEMPT.mkdir(parents=True,exist_ok=False)
    report=read(OUT/'package_report.json');archive=Path(report['path']);assert sha(archive)==report['sha256']
    parser=ATTEMPT/'parse.ps1';parser.write_text("param([string]$Path)\n$bad=@(); Get-ChildItem -LiteralPath $Path -Filter '*.ps1' | ForEach-Object { $tokens=$null;$errors=$null;[void][System.Management.Automation.Language.Parser]::ParseFile($_.FullName,[ref]$tokens,[ref]$errors);if($errors){$bad+=$errors} };if($bad){$bad|Out-String|Write-Output;exit 1};Write-Output 'PowerShell parse PASS'\n",encoding='utf-8-sig')
    syntax=call(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(parser),'-Path',str(HERE)]);assert syntax['exit_code']==0,syntax
    for n in report['files_manifest']:
        if n.endswith('.py'):ast.parse((ROOT/n).read_text(encoding='utf8'))
    fixture=ATTEMPT/'实验室移植 Project Root';fixture.mkdir()
    # Clone code, identity metadata and runtime provenance only; never arrays.
    contract=read(ROOT/'deployment/r6_formal_v1/input_contract.json')
    needed=set(contract['frozen_hashes'])|set(read(HERE/'queue_contract.json')['R6_dependencies'])
    excluded=[]
    for name in needed:
        if Path(name).suffix.lower() in ('.npy','.npz','.whl','.exe'):
            excluded.append(name);continue
        target=fixture/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    old={p.relative_to(fixture).as_posix():sha(p) for p in fixture.rglob('*') if p.is_file()}
    install=['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(HERE/'Install-R7.ps1'),'-ProjectRoot',str(fixture),'-ZipPath',str(archive),'-ExpectedSha256',report['sha256']]
    installed=call(install);assert installed['exit_code']==0,installed
    conflict=call(install);assert conflict['exit_code']!=0 and 'namespace already exists' in conflict['stderr'],conflict
    wrong=call(install[:-1]+['0'*64]);assert wrong['exit_code']!=0 and 'SHA256 mismatch' in wrong['stderr'],wrong
    ps=fixture/'deployment/r7_sequential_v1/R7.ps1'
    def control(action,*extras):return call(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ps),'-Action',action,'-PythonPath',sys.executable,*extras],fixture)
    verified=control('Verify');assert verified['exit_code']==0,verified
    refused=control('Start','-Background');assert refused['exit_code']!=0 and 'MENTOR_APPROVAL_PENDING' in refused['stderr'],refused
    assert not (fixture/'lab_runs/r6_formal/candidates').exists()
    toy=fixture/'lab_runs/r7_sequential/technical/detached_nine_candidate'
    launched=control('ToyQueue','-Background','-ToyDirectory',str(toy));assert launched['exit_code']==0,launched
    wait_started=time.monotonic()+30
    while not (toy/'queue.json').exists() and time.monotonic()<wait_started:time.sleep(.1)
    assert (toy/'queue.json').exists(),'DETACHED_QUEUE_START_MARKER_MISSING'
    duplicate=control('ToyQueue','-ToyDirectory',str(toy))
    assert duplicate['exit_code']!=0 and ('execution.lock' in duplicate['stderr'] or 'DUPLICATE_QUEUE' in duplicate['stderr']),duplicate
    # The launch PowerShell process has already exited. Poll independent lab-local receipt.
    end=time.monotonic()+240;last=time.monotonic()
    while time.monotonic()<end:
        if (toy/'failure.json').exists():raise RuntimeError('DETACHED_TOY_QUEUE_FAILED:'+str(read(toy/'failure.json')))
        if (toy/'completion.json').exists():break
        if time.monotonic()-last>25:
            print(json.dumps({'stage':'R7_detached_PowerShell_toy','progress':read(toy/'queue.json') if (toy/'queue.json').exists() else 'starting'},ensure_ascii=False),flush=True);last=time.monotonic()
        time.sleep(.1)
    assert (toy/'completion.json').exists(),'DETACHED_TOY_TIMEOUT_NO_RETRY'
    complete=read(toy/'completion.json');assert complete['PASS'] and complete['completion']=='QUEUE_COMPLETE_9_OF_9' and len(complete['completed_candidates'])==9
    stages=[read(p) for p in (toy/'jobs').rglob('status.json')];assert len(stages)==18 and all(s['PASS'] and s['all_descendants_kernel_contained'] for s in stages)
    status=control('Status');assert status['exit_code']==0,status
    exported=control('Export');assert exported['exit_code']==0,exported
    exports=list((fixture/'lab_runs/r7_sequential/exports').glob('*.zip'));assert len(exports)==1
    with zipfile.ZipFile(exports[0]) as z:assert all(Path(n).suffix.lower() not in ('.npy','.npz','.whl','.exe') for n in z.namelist())
    assert old=={n:sha(fixture/n) for n in old},'CLONED_R6_ORIGINAL_BYTES_DRIFT'
    value={'PASS':True,'package_sha256':report['sha256'],'package_manifest_sha256':report['manifest_sha256'],'syntax':syntax,
           'installed':installed,'overwrite_refusal':conflict,'wrong_sha_refusal':wrong,'relocated_verify':verified,
           'pending_authorization_start_refusal':refused,'actual_hidden_background_PowerShell_launch':launched,
           'actual_second_process_duplicate_launch_refusal':duplicate,
           'launching_PowerShell_exited_before_queue_completion':True,'detached_queue_completion':complete,'monitored_sequential_stages':18,
           'status':status,'export':exported,'fixture_root':str(fixture),'existing_frozen_files_preserved':len(old),
           'real_scientific_arrays_copied':0,'real_training_started':False,'real_validation_scores':0,'excluded_original_binary_files':excluded}
    (ATTEMPT/'report.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (OUT/'deployment_qa.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'PASS':True,'stage':'R7_deployment_QA','fixture':str(fixture),'monitored_sequential_stages':18,'real_scientific_arrays_copied':0},ensure_ascii=False),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if ATTEMPT:(ATTEMPT/'failure.json').write_text(json.dumps({'PASS':False,'reason':str(e),'traceback':traceback.format_exc()},ensure_ascii=False,indent=2),encoding='utf8')
        raise
