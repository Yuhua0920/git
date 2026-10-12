"""Run actual B5A pytest under the sealed-outcome firewall."""
import os
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
from scripts.sp_u12b5a_common import firewall
firewall()
from scripts import sp_u12b5a_cases as cases
import pytest
if __name__=='__main__':
    with cases.threadpoolctl.threadpool_limits(1):
        raise SystemExit(pytest.main(['tests/test_mfa_reference.py','-q','-p','no:cacheprovider']))
