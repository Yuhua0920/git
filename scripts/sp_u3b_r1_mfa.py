"""One training-method override; original online estimator and all other EM code."""
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u3b_r1_common as c
for name,commit in c.cfg()['author_commits'].items():
    root=c.OUT/'sp_u3a_sources'/name;pin=c.load(root/'repository_pin.json')
    assert pin['commit']==commit
    assert all(c.sha(root/p)==h for p,h in pin['files'].items())
    sys.path.insert(0,str(root/'src'))
sys.path.append(str(ROOT/'.venv-gmm/Lib/site-packages'))
import numpy as np
from mfa_estimator import MfaEstimator
from threadpoolctl import threadpool_limits,threadpool_info

execution=c.path('patched_mfa.py')
audit=c.load(c.path('patch_audit.json'))
assert c.sha(execution)==audit['patched_sha256']
spec=importlib.util.spec_from_file_location('cplx_mfa._sp_u3b_r1_execution',execution)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class MemorySafeMfaEstimator(MfaEstimator):
    _em_step=module.ComplexMFA._em_step

def make(K,rank,seed,max_iter=None,patched=True):
    q=c.cfg(); cls=MemorySafeMfaEstimator if patched else MfaEstimator
    return cls(K,rank,ppca=q['ppca'],lock_psis=q['lock_psis'],zero_mean=q['zero_mean'],
      rs_clip=q['rs_clip'],max_condition_number=q['max_condition_number'],
      max_iter=q['max_iter'] if max_iter is None else max_iter,tol=q['tol'],
      random_state=seed,verbose=False)

def patched_contraction(Z,L,w):
    return Z@(L.conj()*w[None,:]).T
