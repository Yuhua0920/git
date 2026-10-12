"""Personal technical QA: PS parse/runtime + fresh Unicode ZIP relocation."""
from pathlib import Path
import sys,json,subprocess,time,shutil,hashlib,zipfile,os
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'lab_runs/r6_formal/preparation_personal'
ATTEMPT=None
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def call(args,cwd=ROOT):
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['TEMP']=env['TMP']=str(OUT/'temporary');(OUT/'temporary').mkdir(exist_ok=True)
    r=subprocess.run(args,cwd=cwd,env=env,capture_output=True,text=True,encoding='utf8',errors='replace')
    result={'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    if ATTEMPT:
        with(ATTEMPT/'commands.jsonl').open('a',encoding='utf8')as f:f.write(json.dumps({'command':args,'result':result},ensure_ascii=False)+'\n')
    return result
def main():
    global ATTEMPT
    attempt=OUT/('deployment_qa_'+time.strftime('%Y%m%d_%H%M%S'));attempt.mkdir(parents=True)
    ATTEMPT=attempt
    # -Command args differ between Windows PowerShell hosts. Use a saved
    # UTF8 script with explicit param to avoid shell argument reinterpretation.
    parser=attempt/'parse.ps1';parser.write_text("param([string]$Path)\n$bad=@(); Get-ChildItem -LiteralPath $Path -Filter '*.ps1' | ForEach-Object { $tokens=$null; $errors=$null; [void][System.Management.Automation.Language.Parser]::ParseFile($_.FullName,[ref]$tokens,[ref]$errors); if($errors) {$bad+=$errors} }; if($bad){$bad | Out-String | Write-Output; exit 1}; Write-Output 'PowerShell parse PASS'\n",encoding='utf-8-sig')
    syntax=call(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(parser),'-Path',str(HERE)])
    fixture=attempt/'移植测试 Project Root';fixture.mkdir()
    base=ROOT/'outputs/mfa_lab_migration_v1/package/Project_MFA_Lab/migration_manifest.json'
    (fixture/'migration_manifest.json').write_bytes(base.read_bytes())
    contract=read(HERE/'input_contract.json')
    # Existing-code fixture only; no preparation numerical arrays are copied.
    for name in contract['frozen_hashes']:
        if name in contract['data'] or Path(name).suffix in('.npy','.npz','.whl'):continue
        dest=fixture/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    for name in ('necessary_files.csv','migration_provenance.json'):(fixture/name).write_bytes((base.parent/name).read_bytes())
    report=read(OUT/'package_report.json');archive=Path(report['path']);side=Path(str(archive)+'.sha256')
    command=['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(HERE/'Install-R6.ps1'),'-ZipPath',str(archive),'-Sha256Path',str(side),'-ProjectRoot',str(fixture)]
    installed=call(command);assert installed['exit_code']==0,installed
    hashes={n:sha(fixture/n) for n in read(HERE/'package_manifest.json')['files']}
    conflict=call(command);assert conflict['exit_code']!=0 and 'Overwrite conflict' in conflict['stderr'],conflict
    assert hashes=={n:sha(fixture/n) for n in hashes}
    r6=fixture/'deployment/r6_formal_v1';py=Path(sys.executable)
    status=call(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(r6/'R6.ps1'),'-Action','Status','-PythonPath',str(py)],fixture)
    assert status['exit_code']==0,status
    refusal=call(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(r6/'R6.ps1'),'-Action','StartCandidate','-Candidate','K1_r4','-Background','-PythonPath',str(py)],fixture)
    assert refusal['exit_code']!=0 and 'MENTOR_APPROVAL_PENDING' in refusal['stderr'],refusal
    assert not (fixture/'lab_runs/r6_formal/candidates').exists()
    background=call(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(r6/'R6.ps1'),'-Action','MonitorToy','-Background','-PythonPath',str(py)],fixture)
    assert background['exit_code']==0,background
    finish=fixture/'lab_runs/r6_formal/preparation_personal/background_probe_result.json'
    end=time.monotonic()+20
    while not finish.exists() and time.monotonic()<end:time.sleep(.1)
    assert finish.exists(),'BACKGROUND_TOY_COMPLETION_MISSING'
    bg=read(finish);assert bg['PASS'] and bg['state']['actual_process_tree_max_members']>=3 and bg['state']['actual_interpreter_PID']!=bg['state']['venv_launcher_PID'],bg
    missing=call([str(py),'-B','-X','utf8',str(r6/'runner.py'),'preflight','--personal-metadata-only'],fixture)
    assert missing['exit_code']!=0 and 'INTEGRITY_STOP' in missing['stderr'],missing
    source_preflight=call([str(py),'-B','-X','utf8',str(HERE/'runner.py'),'preflight','--personal-metadata-only'])
    assert source_preflight['exit_code']==0,source_preflight
    value={'PASS':syntax['exit_code']==0,'syntax':syntax,'installed':installed,'conflict_refusal':conflict,
           'unicode_root_status':status,'background_pending_approval_refusal':refusal,'missing_data_preflight_refusal':missing,
           'actual_hidden_background_toy_launch':background,'actual_background_monitor':bg,
           'source_header_and_metadata_preflight':source_preflight,'package_sha256':report['sha256'],
           'package_manifest_sha256':report['manifest_sha256'],'fixture_root':str(fixture),
           'real_scientific_arrays_copied':0,'formal_science_started':False}
    (OUT/'deployment_qa.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in value.items()if not isinstance(v,dict)},ensure_ascii=False,indent=2))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        if ATTEMPT:(ATTEMPT/'failure.json').write_text(json.dumps({'PASS':False,'reason':str(e),'traceback':traceback.format_exc()},ensure_ascii=False,indent=2),encoding='utf8')
        raise
