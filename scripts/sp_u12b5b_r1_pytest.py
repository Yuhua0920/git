import os,sys
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
from scripts.sp_u12b5b_r1_common import *
firewall()
import pytest
phase=sys.argv[1]
code=pytest.main(['tests/test_sp_u12b5b_r1.py','-q','-p','no:cacheprovider','--basetemp='+str(OUT/f'pytest_tmp_{phase}'),'--junitxml='+str(OUT/f'pytest_{phase}.xml')])
write(OUT/f'pytest_expected_firewall_denials_{phase}.json',dict(blocked_before_open=DENIED,target_scientific_value_reads=0))
raise SystemExit(code)
