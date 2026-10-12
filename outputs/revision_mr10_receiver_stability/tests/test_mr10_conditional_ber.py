import numpy as np
from mr10_detectors import conditional_mismatched_zf,zf_risk

def test_conditional_mismatched_formula_reduces_to_exact_perfect_zf():
    h=np.stack((np.eye(2),np.array([[1+.4j,.3-.8j],[-.4+.6j,1.2+.2j]]),np.diag([.1+.2j,1+2j])))
    ber,gain,mismatch=conditional_mismatched_zf(h,h,.4)
    _,_,gg,_,bb=zf_risk(h,.4)
    np.testing.assert_allclose(ber,bb.mean(axis=-1),rtol=1e-13,atol=1e-15)
    np.testing.assert_allclose(gain,gg,rtol=1e-13)
    assert np.max(mismatch)<1e-28

def test_positive_rescaling_changes_gain_but_cannot_change_hard_zf_ber():
    h=np.array([[[1+.4j,.3-.8j],[-.4+.6j,1.2+.2j]]])
    b0,g0,m0=conditional_mismatched_zf(h,h,.4)
    b1,g1,m1=conditional_mismatched_zf(h,3*h,.4)
    np.testing.assert_allclose(b0,b1,rtol=1e-13,atol=1e-15)
    np.testing.assert_allclose(g1,g0/9,rtol=1e-13)
    assert m1[0]>0
