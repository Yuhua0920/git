"""Personal-side safe test orchestration. Never a full-scale or scientific run."""
from pathlib import Path
import os,sys,subprocess,ast,json,shutil,zipfile,hashlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import ROOT,OUT,LOCAL,read,write,sha,protocol,environment,scientific_counters
import build

def call(command,folder,name,cwd=ROOT,timeout=180):
    env=os.environ.copy();env['R5_PERSONAL_SELFTEST']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    env['TEMP']=env['TMP']=str(folder/'temporary');(folder/'temporary').mkdir(exist_ok=True)
    env['OPENBLAS_NUM_THREADS']=env['OMP_NUM_THREADS']='1'
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    v=subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=timeout)
    (folder/(name+'.stdout.log')).write_text(v.stdout,encoding='utf8');(folder/(name+'.stderr.log')).write_text(v.stderr,encoding='utf8')
    result={'exit_code':v.returncode,'stdout':v.stdout,'stderr':v.stderr}
    write(folder/(name+'.json'),result);return result

def run():
    os.environ['R5_PERSONAL_SELFTEST']='1';LOCAL.mkdir(parents=True,exist_ok=True)
    index=1
    while(LOCAL/f'precheck_attempt_{index}').exists():index+=1
    folder=LOCAL/f'precheck_attempt_{index}';folder.mkdir();report={'PASS':False,'attempt':index}
    try:
        env=environment();write(folder/'environment.json',env)
        frozen=build.frozen_sources();write(folder/'source_hashes_before.json',frozen)
        build.manifest()
        for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
        pytest=call([sys.executable,'-B','-X','utf8','-m','pytest',str(HERE/'test_r5_package.py'),str(ROOT/'tests/test_mfa_reference.py'),
                    '-q','-p','no:cacheprovider','--basetemp',str(folder/'pytest_tmp')],folder,'pytest',timeout=180)
        assert pytest['exit_code']==0,'SAFE_PYTEST_FAILED'
        parser="$ErrorActionPreference='Stop'; $rows=@(); Get-ChildItem -LiteralPath 'deployment/r5_128gb_v1' -Filter '*.ps1' | ForEach-Object { $tokens=$null; $errors=$null; [void][Management.Automation.Language.Parser]::ParseFile($_.FullName,[ref]$tokens,[ref]$errors); if($errors.Count -gt 0){throw ($errors | Out-String)}; $rows += @{file=$_.Name;PASS=$true} }; ConvertTo-Json -InputObject $rows"
        ps=call(['powershell','-NoProfile','-NonInteractive','-Command',parser],folder,'PowerShell_parser')
        assert ps['exit_code']==0
        # Minimal source-only relocation, deliberately WITHOUT datasets or a cloned environment.
        relocated=folder/'relocation root 空格';dest=relocated/'deployment/r5_128gb_v1';dest.mkdir(parents=True)
        for p in build.payload()+[HERE/'package_manifest.json']:
            q=dest/p.relative_to(HERE);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        for name in frozen:
            p=relocated/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
        # Reference tests import the frozen B5A helpers; no scientific arrays are copied.
        for name in('scripts/sp_u12b5a_cases.py','scripts/sp_u12b5a_common.py'):
            if(ROOT/name).is_file():
                p=relocated/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
        code="from common import integrity,ROOT,threads;from mfa_ops import fidelity_state,compare;from pathlib import Path;assert ROOT==Path.cwd().parents[1];assert integrity(False)['PASS'];\nwith threads(1,1):\n x,state,checks=fidelity_state(2,4)\n assert all(v['PASS']for v in checks)\nprint('RELOCATED_ROOT_TINY_FIDELITY_PASS; full base payload deliberately not copied')"
        relocate=call([sys.executable,'-B','-X','utf8','-c',code],folder,'relocation',cwd=dest)
        assert relocate['exit_code']==0,'RELOCATION_TINY_FIDELITY_FAILED'
        # Actual PowerShell status and export are safe operations, not laboratory stages.
        status=call(['powershell','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(dest/'Get-R5Status.ps1')],folder,'PS_status_missing_env',cwd=relocated)
        assert status['exit_code']!=0,'MISSING_EXISTING_ENV_SHOULD_STOP'
        # Source-only helper export tested using the current verified interpreter.
        mini=relocated/'lab_runs/r5_128gb';mini.mkdir(parents=True)
        write(mini/'technical.json',{'probe':'LOCAL_EXPORT_ONLY','PASS':True})
        (mini/'excluded.npy').write_bytes(b'not a scientific array')
        scratch=mini/'_scratch';scratch.mkdir();(scratch/'excluded.json').write_text('{}',encoding='utf8')
        exp=call([sys.executable,'-B','-X','utf8',str(dest/'driver.py'),'export'],folder,'relocation_export',cwd=relocated)
        assert exp['exit_code']==0
        exports=list((mini/'exports').glob('*.zip'));assert len(exports)==1
        with zipfile.ZipFile(exports[0])as z:
            assert all(not n.endswith('.npy')and'_scratch'not in n for n in z.namelist())
            mf=json.loads(z.read('result_manifest.json'))
            for n,v in mf['files'].items():assert hashlib.sha256(z.read('r5_128gb/'+n)).hexdigest()==v['sha256']
        assert frozen==build.frozen_sources(),'IMMUTABLE_SOURCE_DRIFT'
        report.update(PASS=True,pytest={'exit_code':0,'stdout':pytest['stdout']},PowerShell_parser={'exit_code':0,'scripts':json.loads(ps['stdout'])},
            tiny_cross_thread_and_seed_checks='PASS',relocation={'PASS':True,'scope':'source-only relocated root including Unicode and spaces; no full base/data/environment copy',
                'tiny_fidelity_PASS':True,'export_manifest_and_exclusions_PASS':True,'missing_existing_env_STOPS':True},
            no_formal_cli=True,personal_full_scale_runs=0,actual_laboratory_tests_performed=False,
            original_sources_sha256_unchanged=True,source_payload_hashes={p.relative_to(ROOT).as_posix():sha(p)for p in build.payload()},**scientific_counters())
        write(LOCAL/'local_precheck.json',report)
    except Exception:
        import traceback
        report['traceback']=traceback.format_exc();write(folder/'failed_precheck.json',report);raise
    write(folder/'precheck_report.json',report);print(json.dumps({k:v for k,v in report.items()if k!='source_payload_hashes'},indent=2))
if __name__=='__main__':run()
