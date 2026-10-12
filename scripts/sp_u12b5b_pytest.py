"""Run only B5B tests under the target-information firewall."""
import os,sys
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
from scripts.sp_u12b5b_common import firewall,OUT,write,DENIED
firewall()
from scripts.sp_u12b5b_model import threadpool_limits
import pytest
if __name__=='__main__':
 with threadpool_limits(1):
  code=pytest.main(['tests/test_sp_u12b5b.py','-q','-p','no:cacheprovider','--basetemp='+str(OUT/'pytest_tmp_attempt2'),'--junitxml='+str(OUT/'pytest_preflight.xml')])
 write(OUT/'pytest_firewall_expected_denials.json',dict(expected_blocked_probe_reads=DENIED,actual_target_scientific_values_read=0))
 raise SystemExit(code)
