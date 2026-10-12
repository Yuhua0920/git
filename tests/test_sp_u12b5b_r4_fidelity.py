"""Synthetic-only R4 independent invariants and contract rejection tests."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'.venv-gmm/Lib/site-packages'))
from scripts.sp_u12b5b_r4_common import firewall
firewall()
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from experiments.mfa_low_memory_fidelity_r4.fidelity import (independent_identities,boundary_stop_test,
    model,initialize,synthetic,capture_step,parameters,discrepancy,inference_contract)
from experiments.mfa_low_memory_fidelity_r4.validation_contract import expected_tokens,assert_coverage,PER_CANDIDATE,ALL_CANDIDATES

def test_independent_index_formulas_and_old_loading_sentinel():
    result=independent_identities();assert result['PASS']
    assert result['checks']['old_loading_residual_regression']['wrong_new_loading_max_absolute']>1

def test_exact_author_strict_stop_boundary():assert boundary_stop_test()['PASS']

def test_valid_count():
    assert PER_CANDIDATE==3600 and ALL_CANDIDATES==32400
    assert assert_coverage(expected_tokens())['predictions']==32400

@pytest.mark.parametrize('defect',['duplicate','missing','half','wrong_frame','wrong_snr'])
def test_cardinality_rejects_identity_defects(defect):
    tokens=list(expected_tokens())
    if defect=='duplicate':tokens.append(tokens[0])
    if defect=='missing':tokens.pop()
    if defect=='half':tokens=tokens[:16200]
    if defect=='wrong_frame':tokens[0]=tokens[0][:-1]+(3,)
    if defect=='wrong_snr':tokens[0]=tokens[0][:2]+(15,)+tokens[0][3:]
    with pytest.raises(AssertionError):assert_coverage(tokens)

def test_absolute_tolerance_handles_zero_and_shape_mismatch():
    assert discrepancy(np.array([5e-11]),np.array([0.]))['PASS']
    assert not discrepancy(np.array([2e-10]),np.array([0.]))['PASS']
    with pytest.raises(AssertionError):discrepancy(np.zeros(2),np.zeros(3))

def test_observer_state_parity_and_inference_roundtrip():
    # Small safe preflight only. Full preregistered nine-grid is a separate run.
    from scripts.sp_u12b5b_r4_common import OUT
    out=OUT/'pytest_preflight';out.mkdir(parents=True,exist_ok=True)
    from scripts.sp_u12b5b_r1_common import host_snapshot
    assert host_snapshot()['available_physical_bytes']>2*2**30
    with threadpool_limits(1):
        x=synthetic(64,64,2,4,42490031)
        a=initialize(model(2,4),x,42490032);b=initialize(model(2,4,True),x,42490032)
        for n,v in parameters(a).items():assert np.array_equal(v,parameters(b)[n])
        left,right=capture_step(a,x),capture_step(b,x)
        assert left.keys()==right.keys()
        assert all(discrepancy(right[n],v)['PASS'] for n,v in left.items())
        path=out/'interface';path.mkdir(exist_ok=True)
        # The test uses a fresh directory each pytest invocation.
        child=path/str(len(list(path.iterdir())));child.mkdir()
        result=inference_contract(b,x,child)
        assert result['PASS'] and result['integer1_is_distinct_when_K_gt1']
