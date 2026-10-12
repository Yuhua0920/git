"""Real relocated-root, fresh offline environment and deployment plumbing QA.

Only B5A unit tests, tiny unit-fixture parity, host and logging/exception probes.
Never calls fidelity full replay or R5 smoke, never formal candidate training.
"""
from pathlib import Path
import sys,os,json,shutil,subprocess,time,traceback
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.mfa_lab_migration_common import OUT,DEPLOY,read,write,sha
from scripts.mfa_lab_migration_prepare import STAGE
QA=OUT/'validation_attempt_3'

def call(cmd,cwd,env=None):
    r=subprocess.run([str(c)for c in cmd],cwd=cwd,env=env,capture_output=True,text=True,encoding='utf8',errors='replace')
    return {'command':[str(c)for c in cmd],'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}

def validate():
    assert read(OUT/'stage_ready.json')['PASS']
    assert not QA.exists();QA.mkdir()
    moved=QA/'relocation_check'/'搬迁项目 Root';assert not moved.exists();shutil.copytree(STAGE,moved)
    envdir=QA/'fresh_environment';assert not envdir.exists()
    environment=os.environ.copy();environment.pop('PYTHONPATH',None);environment.pop('PYTHONHOME',None)
    temp=QA/'temporary';temp.mkdir()
    environment['TEMP']=environment['TMP']=str(temp)
    first=call([sys.executable,'-B','-X','utf8','-m','venv',envdir],ROOT,environment);write(QA/'venv_creation.json',first);assert first['exit_code']==0
    py=envdir/'Scripts/python.exe'
    install=call([py,'-m','pip','--isolated','install','--no-index','--find-links',moved/'deployment/mfa_lab_v1/wheelhouse',
        '--require-hashes','-r',moved/'deployment/mfa_lab_v1/requirements-win-cpu.lock'],moved,environment)
    write(QA/'offline_install_test.json',install);assert install['exit_code']==0
    deps=call([py,'-m','pip','check'],moved,environment);write(QA/'pip_check.json',deps);assert deps['exit_code']==0
    environment['MFA_MIGRATION_TEST_ROOT']=str(moved)
    for key in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):environment[key]='1'
    # Historical B5A tests install their own process-wide audit hook. Keep the
    # historical and deployment scopes in independent pytest processes.
    test_results={}
    for label,test_file in (('author_reference',moved/'tests/test_mfa_reference.py'),
                            ('migration',ROOT/'tests/test_mfa_lab_migration.py')):
        tests=call([py,'-B','-X','utf8','-m','pytest',test_file,
            '-q','-p','no:cacheprovider','--basetemp',QA/('pytest_tmp_'+label)],moved,environment)
        write(QA/('pytest_relocated_'+label+'.json'),tests);assert tests['exit_code']==0
        test_results[label]=tests['stdout'].strip()
    verify=call([py,'-B','-X','utf8',moved/'deployment/mfa_lab_v1/lab.py','verify'],moved,environment);write(QA/'verify_relocated.json',verify);assert verify['exit_code']==0
    for action in('host','self-test'):
        result=call([py,'-B','-X','utf8',moved/'deployment/mfa_lab_v1/lab.py',action],moved,environment)
        write(QA/(action+'_entry_test.json'),result);assert result['exit_code']==0
    # Intentional failure proves exception and STOP bookkeeping without an MFA step.
    code="import sys;from pathlib import Path;sys.path.insert(0,str(Path('deployment/mfa_lab_v1').resolve()));import lab;lab.supervise('plumbing-fail',1,lab.RUNS/'intentional_exception_probe')"
    failed=call([py,'-B','-X','utf8','-c',code],moved,environment);write(QA/'intentional_exception_entry_test.json',failed)
    assert failed['exit_code']!=0
    failure=read(moved/'lab_runs/intentional_exception_probe/status.json');assert failure['state']=='STOPPED'
    assert(moved/'lab_runs/intentional_exception_probe/exception.json').exists()
    assert(moved/'lab_runs/intentional_exception_probe/resources.jsonl').stat().st_size>0
    manifest=read(moved/'migration_manifest.json');assert all(sha(moved/n)==v['sha256']for n,v in manifest['files'].items())
    # Parse PowerShell scripts with the actual host PowerShell parser, not a regex.
    parse_code="$ErrorActionPreference='Stop'; $files=Get-ChildItem -LiteralPath 'deployment/mfa_lab_v1' -Filter '*.ps1'; foreach($file in $files){$tokens=$null;$errors=$null;[void][System.Management.Automation.Language.Parser]::ParseFile($file.FullName,[ref]$tokens,[ref]$errors);if($errors.Count -gt 0){throw ($errors | Out-String)}}; Write-Output 'POWERSHELL_PARSE_PASS'"
    parsed=call(['powershell','-NoProfile','-Command',parse_code],moved);write(QA/'powershell_parse_test.json',parsed);assert parsed['exit_code']==0
    psverify=call(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',moved/'deployment/mfa_lab_v1/Verify-Package.ps1'],moved)
    write(QA/'powershell_integrity_test.json',psverify);assert psverify['exit_code']==0
    result={'PASS':True,'relocated_root':str(moved),'space_and_unicode_root_tested':True,
        'fresh_offline_environment':True,'original_personal_venvs_used_for_test_imports':False,
        'actual_pytest':test_results,'pip_check':deps['stdout'].strip(),'PowerShell_parser_PASS':True,
        'PowerShell_integrity_PASS':True,'actual_PID_monitor_tested':True,'host_entry_PASS':True,'self_test_PASS':True,
        'exception_logging_and_STOP_PASS':True,'prepared_R5_smoke_executed':False,'full_R4_replay_executed':False,
        'formal_candidate_fits':0,'validation_scores':0,'final_retraining':0,'target_evaluation':0,
        'laboratory_host_tested':False,'package_payload_bytes_after_tests_unchanged':True,
        'attempt':3,'earlier_environment_temp_and_PowerShell_module_failures_preserved':True,'validation_logs':str(QA)}
    write(OUT/'relocation_validation.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    try:validate()
    except Exception:write(QA/'validation_failure.json',{'traceback':traceback.format_exc(),'formal_candidate_fits':0});raise
