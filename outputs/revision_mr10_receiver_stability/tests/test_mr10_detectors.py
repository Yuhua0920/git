import numpy as np
import pytest
from scipy.special import erfc
from mr10_detectors import BITS,SYMBOLS,linear,bit_map,zf_risk
from src.modulation import qpsk_modulate
from src.detectors import zf_detect,linear_mmse_detect

H=np.array([[1+.4j,.3-.8j],[-.4+.6j,1.2+.2j]])
def test_all16_constellations_gray_mapping_and_half_power():
    assert len(np.unique(BITS,axis=0))==16
    for i,b in enumerate(BITS):
        expected=np.array([qpsk_modulate(b[:2])[0],qpsk_modulate(b[2:])[0]])/np.sqrt(2)
        np.testing.assert_allclose(SYMBOLS[i],expected,rtol=0,atol=2e-16)
    np.testing.assert_allclose(abs(SYMBOLS)**2,.5,rtol=0,atol=2e-16)
def test_noiseless_full_complex_all16_bit_map_and_zf():
    h=np.broadcast_to(H,(16,2,2));y=np.einsum('nrt,nt->nr',h,SYMBOLS)
    for actual in [linear(h,y,.1)[0],bit_map(h,y,1e-5)[0]]:assert np.array_equal(actual,BITS)
    np.testing.assert_allclose(linear(h,y,.1)[1],SYMBOLS,rtol=0,atol=1e-15)
def test_bit_map_independent_probability_enumeration():
    y=np.array([[.2+.3j,-.4+.1j]]);sigma=.7
    bit,odds,_=bit_map(H[None],y,sigma)
    scores=np.array([-sum(abs(y[0]-H@x)**2)/sigma for x in SYMBOLS])
    p=np.exp(scores-scores.max());p/=sum(p)
    p1=np.array([sum(p[BITS[:,k]==1]) for k in range(4)])
    np.testing.assert_allclose(odds[0],np.log(p1/(1-p1)),rtol=1e-13,atol=1e-13)
    assert np.array_equal(bit[0],p1>.5)
def test_map_is_not_vector_ml_on_deterministic_grid():
    y=np.array([[a+1j*b,c+1j*d] for a in (-.75,0,.75) for b in (-.5,.5) for c in (-.4,.4) for d in (-.3,.3)])
    h=np.broadcast_to(np.array([[1,.7+.3j],[.2j,.1]]),(len(y),2,2))
    bm,_,ml=bit_map(h,y,.5)
    assert np.any(bm!=ml)
def test_batch_linear_matches_original_complex_orientation():
    h=np.stack([H,np.diag([1+2j,.7-.3j]),np.eye(2)]);y=np.array([[.2+.3j,-.4+.1j]]*3)
    for mmse in (False,True):
        _,x=linear(h,y,.4,mmse)
        hg=h.transpose(1,2,0)[:,:,:,None];yg=y.T[:,:,None]
        original=linear_mmse_detect(yg,hg,.4,.5) if mmse else zf_detect(yg,hg)
        np.testing.assert_allclose(x,original[:,:,0].T,rtol=1e-12,atol=1e-12)
def test_identity_mmse_factor_two():
    y=np.array([[.2+.3j,-.4+.1j]]);_,x=linear(np.eye(2)[None],y,.4,True)
    np.testing.assert_allclose(x,y/(1+.4/.5),rtol=0,atol=2e-16)
def test_ill_conditioned_behavior_no_fallback():
    h=np.diag([1.,1e-8])[None];y=np.array([[.5+.5j,.1j]])
    _,zf=linear(h,y,.1);_,mm=linear(h,y,.1,True)
    assert abs(zf[0,1])>1e6 and np.isfinite(mm).all()
    with pytest.raises(np.linalg.LinAlgError):linear(np.array([[[1,2],[2,4]]]),y,.1)
def test_zf_covariance_and_qpsk_bit_formula():
    h=np.diag([1+2j,.7-.3j])[None];sv,k,g,snr,ber=zf_risk(h,.4)
    expected=np.array([[1/5,1/.58]])
    np.testing.assert_allclose(g,expected,rtol=1e-14)
    np.testing.assert_allclose(ber,.5*erfc(np.sqrt(.5/(2*.4*expected))),rtol=1e-14)
def test_original_receiver_sanity_is_reused():
    from scripts.sp_u8a_common import fixtures
    result=fixtures();assert result['PASS']
    assert all(r['bit_errors']==0 for r in result['algebra'])
    assert all(v>0 for v in result['ordering_mutation_bit_errors'].values())
