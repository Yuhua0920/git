import numpy as np
import pytest
from controls import METHODS,CORRECTION_METHODS,WHOLE_METHODS,deployed,prevalidated_deployment,weighted_factor
from test_controls import toy_operators,complex_toy
from benchmark_mr05 import costs

@pytest.mark.parametrize('coeff', [[0.,0.],[-.7,2.1],[.1,.2],[5.,-2.]])
def test_fast_kernel_same_numerics_and_no_validation_scan(coeff,monkeypatch):
    op,u=toy_operators();_,C,_=complex_toy()
    correction=tuple(weighted_factor(op[name],C,2)[0] for name in ('DeltaWP_10','DeltaWM_10'))
    whole=tuple(weighted_factor(W,C,2)[0] for W in (op['Wsource_10'],op['Wsource_10']+op['DeltaWP_10'],op['Wsource_10']+op['DeltaWM_10']))
    bank={m:correction for m in CORRECTION_METHODS};bank.update({m:whole for m in WHOLE_METHODS})
    cc=np.array(coeff);kernels={m:prevalidated_deployment(m,op,bank,10) for m in METHODS}
    expected={m:deployed(m,u,cc,op,bank,10) for m in METHODS}
    import controls
    def forbidden(*args,**kwargs):raise AssertionError('timed full-array validation')
    monkeypatch.setattr(controls,'_complex_array',forbidden)
    monkeypatch.setattr(controls.np,'isfinite',forbidden)
    for m in METHODS:assert np.array_equal(kernels[m](u,cc),expected[m])

def test_complexity_fixed_budget_and_storage():
    c={r['method']:r for r in costs()}
    assert c['SOURCE']['total_dominant_complex_MAC']==253952
    assert c['DENSE_TW5']['total_dominant_complex_MAC']==807936
    assert c['CORR_POOL_M2_R4']['total_dominant_complex_MAC']==332288
    assert c['WHOLE_POOL_M2_R23']['total_dominant_complex_MAC']==324288
    assert c['WHOLE_POOL_M2_R24']['total_dominant_complex_MAC']==336384
    assert c['CORR_POOL_M2_R4']['minimal_deployment_bytes']==4643840
    assert c['WHOLE_POOL_M2_R23']['minimal_deployment_bytes']==4515840
    assert c['WHOLE_POOL_M2_R24']['minimal_deployment_bytes']==4709376
