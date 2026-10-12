"""Run actual pytest with a notice-first, script-generated audit log."""
import os
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import sp_u8a_common as c
from contextlib import redirect_stdout,redirect_stderr
import pytest
phase=sys.argv[1]
assert phase in ('pre','post')
with (c.OUT/f'pytest_{phase}.txt').open('w',encoding='utf8') as f:
    f.write(c.NOTICE+'\n');f.flush()
    with redirect_stdout(f),redirect_stderr(f):
        exitcode=pytest.main(['-q','tests/test_sp_u8a.py','--basetemp',str(c.OUT/f'pytest_tmp_{phase}')])
c.write(c.OUT/f'pytest_{phase}.json',dict(PASS=exitcode==0,exit_code=int(exitcode),log=f'pytest_{phase}.txt'))
print((c.OUT/f'pytest_{phase}.txt').read_text(),flush=True)
raise SystemExit(exitcode)
