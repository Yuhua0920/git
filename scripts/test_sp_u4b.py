import sys
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u4b_common import *
stage=sys.argv[1]
p=subprocess.run([sys.executable,'-m','pytest','tests/test_sp_u4b.py','-q','--basetemp',str(OUT/f'sp_u4b_pytest_tmp_{stage}')],cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace')
(OUT/f'sp_u4b_pytest_{stage}.txt').write_text(p.stdout+'\n'+p.stderr,encoding='utf8')
save(OUT/f'sp_u4b_tests_{stage}.json',dict(passed=p.returncode==0,exit_code=p.returncode,summary=p.stdout.strip().splitlines()[-1],
    source_sha256={k:sha(ROOT/k) for k in ('scripts/sp_u4b_common.py','scripts/run_sp_u4b.py','scripts/test_sp_u4b.py','tests/test_sp_u4b.py')}))
print(p.stdout);print(p.stderr);sys.exit(p.returncode)
