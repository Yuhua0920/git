"""Actual scoped pytest; core sources frozen before the measurement replay."""
import sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u5b_common import *
stage=sys.argv[1]
cmd=[sys.executable,'-m','pytest','tests/test_sp_u5b.py','-q','--basetemp',str(OUT/f'sp_u5b_pytest_tmp_{stage}')]
p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace')
(OUT/f'sp_u5b_pytest_{stage}.txt').write_text(p.stdout+'\n'+p.stderr,encoding='utf8')
names=('scripts/sp_u5b_common.py','scripts/run_sp_u5b.py','scripts/test_sp_u5b.py','tests/test_sp_u5b.py','configs/sp_u5b_protocol.json')
save(OUT/f'sp_u5b_tests_{stage}.json',dict(passed=p.returncode==0,exit_code=p.returncode,command=cmd,summary=p.stdout.strip().splitlines()[-1],source_sha256={n:sha(ROOT/n) for n in names}))
print(p.stdout);print(p.stderr);sys.exit(p.returncode)
