import numpy as np
import pytest
from analyze_capacity import paired,definitions

def test_pairing_preserves_scaled_methods_and_repeated_snrs():
    q=np.array([2.,4.,8.,16.]);e=np.empty((4,3,2));e[:,:,0]=q[:,None]*.2;e[:,:,1]=e[:,:,0]*2
    draws=np.tile(np.array([3,1,2,0],np.int64),(2000,1))
    p,d,b,bd=paired(e,q,draws)
    np.testing.assert_allclose(p[:,0],.2);np.testing.assert_allclose(bd[:,:,1]-bd[:,:,0],10*np.log10(2))

def test_invalid_counts_and_nonfinite_refused():
    E=np.ones((4,3,2));q=np.ones(4)
    with pytest.raises(ValueError):paired(E,q,np.zeros((1999,4),np.int64))
    E[0,0,0]=np.nan
    with pytest.raises(ValueError):paired(E,q,np.zeros((2000,4),np.int64))

def test_contrasts_never_fabricate_missing_EM():
    ms=['SOURCE','POOLED_NATURAL','DENSE_TW5','LR4_TW5','WHOLE_POOL_M2_R23','WHOLE_POOL_M2_R24','DENSE_TW1','LR4_TW1','PPCA_DIRECT_R16','PPCA_DIRECT_R32','PPCA_DIRECT_R64']
    d=definitions(ms);assert len(d)==22;assert not any('EM16' in x[0] for x in d)
    assert len(definitions(ms+['MFA_K1_R16_EM_FROZEN']))==23

def test_nonfinite_truth_is_rejected():
    E=np.ones((4,3,2));draws=np.zeros((2000,4),np.int64)
    for value in (np.nan,np.inf):
        q=np.ones(4);q[0]=value
        with pytest.raises(ValueError,match='ENERGY_OR_DRAW'):paired(E,q,draws)
