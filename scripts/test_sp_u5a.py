"""Run actual pytest and freeze core execution hashes before scientific evaluation."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u5a_common import *
stage=sys.argv[1]
cmd=[sys.executable,'-m','pytest','tests/test_sp_u5a.py','-q','--basetemp',str(OUT/f'sp_u5a_pytest_tmp_{stage}')]
p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace')
(OUT/f'sp_u5a_pytest_{stage}.txt').write_text(p.stdout+'\n'+p.stderr,encoding='utf8')
save(OUT/f'sp_u5a_tests_{stage}.json',dict(passed=p.returncode==0,exit_code=p.returncode,command=cmd,summary=p.stdout.strip().splitlines()[-1],
    source_sha256={n:sha(ROOT/n) for n in ('scripts/sp_u5a_common.py','scripts/run_sp_u5a.py','scripts/test_sp_u5a.py','tests/test_sp_u5a.py','configs/sp_u5a_protocol.json')}))
print(p.stdout);print(p.stderr);sys.exit(p.returncode)
