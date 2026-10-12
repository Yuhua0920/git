"""Run project-specific and complete author test suites, preserving every log."""
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u3a_common as c

def main():
    c.verify();env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1'
    env['PYTHONPATH']=os.pathsep.join([str(ROOT),str(c.OUT/'sp_u3a_sources/cplx-mfa/src'),
      str(c.OUT/'sp_u3a_sources/MFA_estimator/src'),str(ROOT/'.venv-gmm/Lib/site-packages')])
    results=[]
    for label,paths in [('project',['tests/test_sp_u3a_mfa.py']),
      ('author',['outputs/sp_u3a_sources/cplx-mfa/tests','outputs/sp_u3a_sources/MFA_estimator/tests'])]:
        cmd=[sys.executable,'-m','pytest',*paths,'-q','--import-mode=importlib','-p','no:cacheprovider',
         '--basetemp=outputs/sp_u3a_pytest_tmp_'+label]
        result=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
        log=c.OUT/f'sp_u3a_pytest_{label}.txt';assert not log.exists();log.write_text(result.stdout+result.stderr,encoding='utf8')
        results.append(dict(scope=label,command=cmd,exit_code=result.returncode,log_sha256=c.sha(log),tail=result.stdout.splitlines()[-5:]))
        print(label,result.stdout+result.stderr,flush=True)
        if result.returncode:break
    c.save(c.OUT/'sp_u3a_tests_gate.json',dict(passed=len(results)==2 and all(a['exit_code']==0 for a in results),results=results,access=c.ACCESS))
    if len(results)!=2 or any(a['exit_code'] for a in results):raise RuntimeError('tests failed; no CDLsmoke authorization gate')

if __name__=='__main__':main()
