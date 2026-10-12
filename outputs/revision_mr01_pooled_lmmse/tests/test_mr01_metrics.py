import numpy as np
import pytest
from run_mr01 import pooled_bootstrap,CONTRASTS,METHODS

def toy():
    # Unequal truth energies deliberately distinguish pooling from averaging.
    e=np.array([[[1.,2.],[2.,3.]],[[4.,5.],[7.,11.]],[[10.,7.],[9.,6.]]])
    q=np.array([2.,7.,19.]);draw=np.array([[0,1,2],[0,0,1],[2,2,2],[1,0,2]])
    return e,q,draw
def test_pooled_ratio_and_log_not_average_db():
    e,q,d=toy();p,db,_=pooled_bootstrap(e,q,d)
    np.testing.assert_allclose(p,[[15/28,14/28],[18/28,20/28]],rtol=1e-14)
    np.testing.assert_allclose(db,10*np.log10(p),rtol=1e-14)
    assert not np.allclose(db,np.mean(10*np.log10(e/q[:,None,None]),axis=0))
def test_manual_trajectory_bootstrap_and_shared_pairing():
    e,q,d=toy();_,_,b=pooled_bootstrap(e,q,d)
    oracle=np.empty((4,2,2))
    for k,ids in enumerate(d):
        denominator=sum(q[i] for i in ids)
        for s in range(2):
            for m in range(2):oracle[k,s,m]=10*np.log10(sum(e[i,s,m] for i in ids)/denominator)
    np.testing.assert_allclose(b,oracle,rtol=1e-14)
    np.testing.assert_allclose(b[0],b[3],rtol=1e-14)
def test_difference_sign():
    e,q,d=toy();e[:,:,1]=2*e[:,:,0];_,db,b=pooled_bootstrap(e,q,d)
    np.testing.assert_allclose(db[:,1]-db[:,0],10*np.log10(2))
    np.testing.assert_allclose(b[:,:,1]-b[:,:,0],10*np.log10(2))
def test_primary_contrast_axis():
    _,i,j=next(x for x in CONTRASTS if x[0]=='Pooled-LR4')
    assert (METHODS[i],METHODS[j])==('Pooled','LR4')
@pytest.mark.parametrize('failure',['nan','zero_truth','negative_error','float_indices','high_index','negative_index','frame_axis','wrong_N'])
def test_invalid_statistical_inputs_stop(failure):
    e,q,d=toy()
    if failure=='nan':e[0,0,0]=np.nan
    elif failure=='zero_truth':q[0]=0
    elif failure=='negative_error':e[0,0,0]=-1
    elif failure=='float_indices':d=d.astype(float)
    elif failure=='high_index':d[0,0]=3
    elif failure=='negative_index':d[0,0]=-1
    elif failure=='frame_axis':e=e[:,:,:,None]
    elif failure=='wrong_N':q=q[:-1]
    with pytest.raises(ValueError):pooled_bootstrap(e,q,d)
