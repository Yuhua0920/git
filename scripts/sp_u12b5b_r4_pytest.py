"""Capture actual scope-limited pytest; never runs historical experiment tests."""
from pathlib import Path
import sys,subprocess
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u12b5b_r4_common import firewall,write,OUT
firewall()
cmd=[sys.executable,'-B','-X','utf8','-m','pytest','tests/test_sp_u12b5b_r4_fidelity.py',
    'tests/test_sp_u12b5b_r4_canonical_contract.py','tests/test_sp_u12b5b_r4_delivery.py','-q','-p','no:cacheprovider']
completed=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8')
value={'command':cmd,'exit_code':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr,
    'scope':'R4 synthetic/math/canonical-count/delivery tests only; no scientific experiments or historical tests'}
write(OUT/'pytest_final.json',value);(OUT/'pytest_final.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf8')
print(completed.stdout);print(completed.stderr);assert completed.returncode==0
