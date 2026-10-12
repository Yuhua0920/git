"""Actual pytest subprocess and prerequisite source freeze; never infer completion."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u4a_common import *


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('preparation','pre','post'));args=p.parse_args()
    cmd=[sys.executable,'-m','pytest','tests/test_sp_u4a.py','-q','--basetemp',str(OUT/f'sp_u4a_pytest_tmp_{args.stage}')]
    process=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace')
    (OUT/f'sp_u4a_pytest_{args.stage}.txt').write_text(process.stdout+'\n'+process.stderr,encoding='utf8')
    save(OUT/f'sp_u4a_tests_{args.stage}.json',dict(passed=process.returncode==0,exit_code=process.returncode,command=cmd,summary=process.stdout.strip().splitlines()[-1],
        source_sha256={p:sha(ROOT/p) for p in ('scripts/sp_u4a_common.py','scripts/run_sp_u4a.py','scripts/test_sp_u4a.py','tests/test_sp_u4a.py','configs/sp_u4a_protocol.json')}))
    print(process.stdout,flush=True);print(process.stderr,flush=True);sys.exit(process.returncode)
