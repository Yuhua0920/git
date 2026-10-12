"""Independent real-stack and vector checks for MR-02 diagnostics only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from mechanism import (real_sufficient,real_oracle,fixed_reference,quadratic_risk,
                       gram_diagnostics,risk_transfer,compression_terms,
                       assert_oracle_bound,joint_gram,streaming_one_frame)


def fixture():
    X=np.array([[1+2j,2-1j],[-1+.5j,3+1j],[2-2j,-2+.2j],[.4+.6j,.1-2j]],np.complex128)
    r=np.array([3+.2j,1+2j,-.3+1j,2-.5j],np.complex128)
    return X,r


def test_real_stack_moments_and_direct_risk():
    X,r=fixture();A=np.concatenate((X.real,X.imag));y=np.r_[r.real,r.imag]
    H,b,k=real_sufficient(X,r)
    np.testing.assert_allclose(H,A.T@A,rtol=1e-15,atol=1e-14)
    np.testing.assert_allclose(b,A.T@y,rtol=1e-15,atol=1e-14)
    assert k==pytest.approx(y@y,rel=1e-15)
    for c in ([1.2,-2.3],[-.4,.8],[0.,0.]):
        err=X@c-r
        assert quadratic_risk(H,b,k,c)==pytest.approx(np.vdot(err,err).real,rel=1e-14,abs=1e-14)


def test_direct_svd_oracle_vs_independent_real_lstsq():
    X,r=fixture();A=np.concatenate((X.real,X.imag));y=np.r_[r.real,r.imag]
    c,_,rank,s=np.linalg.lstsq(A,y,rcond=1e-12)
    v=real_oracle(X,r)
    np.testing.assert_allclose(v['coeff'],c,rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(v['singular_values'],s,rtol=1e-14)
    assert v['rank']==rank==2
    assert v['normal_residual']<1e-14
    assert_oracle_bound(v,X,r,[[0,0],[1,0],[0,1],[10,-15]])


@pytest.mark.parametrize('c',[np.array([1+0j,0+0j]),np.array([1+2j,0]),np.array([np.nan,0])])
def test_nonreal_or_nonfinite_coordinate_refused(c):
    X,r=fixture();H,b,k=real_sufficient(X,r)
    with pytest.raises(ValueError):quadratic_risk(H,b,k,c)


def test_real_and_complex_ls_are_different_problems():
    X=np.array([[1+0j]]);r=np.array([1j]);v=real_oracle(X,r)
    assert v['coeff'][0]==pytest.approx(0.)
    assert v['error']==pytest.approx(1.)
    cc=np.linalg.lstsq(X,r,rcond=1e-12)[0]
    assert abs(X@cc-r).max()<1e-15


def test_rank_deficient_direct_oracle_has_minimum_norm():
    v=np.array([1+1j,2-1j,3+.2j]);X=np.c_[v,2*v];r=v.copy()
    z=real_oracle(X,r)
    assert z['rank']==1 and z['retained_subspace_only']
    np.testing.assert_allclose(z['coeff'],[.2,.4],atol=2e-15)
    assert z['error']<1e-28
    assert_oracle_bound(z,X,r,[[1,0],[0,.5],[3,-1]])


def test_numerical_rank_truncation_disclosed_not_clipped():
    X=np.array([[1,0],[0,1e-14]],np.complex128);r=np.array([0,1],np.complex128)
    z=real_oracle(X,r)
    assert z['rank']==1 and z['retained_subspace_only']
    assert z['error']==pytest.approx(1.)
    # In the discarded subspace a huge coefficient can improve the full
    # problem. The bound must fail instead of pretending truncation is exact.
    with pytest.raises(ValueError,match='ORACLE_INEQUALITY_FAILURE'):
        assert_oracle_bound(z,X,r,[[0,1e14]])


def test_fixed_reference_independent_accumulated_real_lstsq():
    X,r=fixture();Y=X[[2,0,3,1]]*.7;z=r[::-1]
    H1,b1,_=real_sufficient(X,r);H2,b2,_=real_sufficient(Y,z)
    v=fixed_reference(H1+H2,b1+b2)
    A=np.concatenate((X.real,X.imag,Y.real,Y.imag));t=np.r_[r.real,r.imag,z.real,z.imag]
    c=np.linalg.lstsq(A,t,rcond=1e-12)[0]
    assert v['rank']==2 and v['solve_flag']=='IDENTIFIED_DIRECT_SOLVE'
    np.testing.assert_allclose(v['coeff'],c,rtol=1e-14,atol=1e-14)


def test_fixed_reference_rank_deficient_minimum_norm():
    H=np.array([[1.,2.],[2.,4.]]);b=np.array([1.,2.])
    v=fixed_reference(H,b)
    assert v['rank']==1 and v['solve_flag']=='NONUNIQUE_DIAGNOSTIC_MINIMUM_NORM'
    np.testing.assert_allclose(v['coeff'],[.2,.4],atol=1e-15)
    assert v['normal_residual']<1e-14
    assert v['rank_cutoff']>=v['gram_roundoff_floor']


def test_zero_design_and_zero_reference():
    X=np.zeros((4,2),np.complex128);r=np.ones(4,np.complex128)
    v=real_oracle(X,r);f=fixed_reference(np.zeros((2,2)),np.zeros(2))
    assert v['rank']==f['rank']==0
    assert v['error']==4
    np.testing.assert_array_equal(v['coeff'],[0,0])
    np.testing.assert_array_equal(f['coeff'],[0,0])


@pytest.mark.parametrize('H,b,match',[
    (np.diag([-1.,2.]),np.zeros(2),'NON_PSD'),
    (np.array([[1.,2.],[0.,1.]]),np.zeros(2),'NONSYMMETRIC'),
    (np.diag([1.,0.]),np.array([0.,1.]),'INCONSISTENT')])
def test_invalid_normal_objective_refused(H,b,match):
    with pytest.raises(ValueError,match=match):fixed_reference(H,b)


def test_gram_diagnostics_normalized_eigenvalue_and_condition():
    v=gram_diagnostics(np.diag([2.,8.]))
    assert v['rank']==2 and v['condition']==4
    assert v['minimum_normalized_eigenvalue']==.25


def test_eval_fixed_risk_difference_requires_nonzero_gradient():
    H=np.array([[2.,.2],[.2,1.]]);b=np.array([3.,-2.]);cp=np.array([.1,.2]);cd=np.array([1.,0.])
    t=risk_transfer(H,b,cp,cd)
    assert abs(t['gradient'])>.1
    assert t['distance']>=0
    actual=quadratic_risk(H,b,10,cp)-quadratic_risk(H,b,10,cd)
    assert t['exactdiff']==pytest.approx(actual,abs=1e-14)
    assert t['distance']+t['gradient']==pytest.approx(actual,abs=1e-14)
    assert abs(actual-t['distance'])>.1


def test_reference_optimal_on_other_sample_not_eval():
    ref=fixed_reference(np.eye(2),np.array([1.,0.]))['coeff']
    H=np.eye(2);b=np.zeros(2);cp=np.zeros(2)
    t=risk_transfer(H,b,cp,ref)
    assert t['distance']==1 and t['gradient']==-2 and t['exactdiff']==-1


def test_negative_cross_term_and_negative_compression_penalty_valid():
    z=compression_terms(np.array([1+0j]),np.array([-.5+0j]))
    assert z['A']==.25 and z['B']==-1.
    assert z['total']==-.75 and z['difference']==-.75
    assert z['identity_residual']==0


def test_complex_compression_direct_identity():
    e=np.array([1+2j,-.3+.4j,3-.2j]);d=np.array([.5-.3j,.2+.1j,-1+.5j])
    z=compression_terms(e,d)
    assert z['total']==pytest.approx(np.sum(abs(e+d)**2)-np.sum(abs(e)**2),abs=1e-14)
    assert z['A']>=0


def test_joint_gram_matches_stacked_actual_design():
    X,r=fixture();Y=np.roll(X,1,axis=0)*(.2+.7j)
    H,b,k=real_sufficient(X,r);J,bb,kk=real_sufficient(Y,r)
    C=(X.conj().T@Y).real
    expected,_,_=real_sufficient(np.c_[X,Y],r)
    np.testing.assert_allclose(joint_gram(H,J,C),expected,atol=1e-14)
    assert np.linalg.eigvalsh(expected).min()>0


def test_output_spaces_need_not_be_nested_or_oracle_ordered():
    XD=np.array([[1,0],[0,1],[0,0]],np.complex128)
    XL=np.array([[1,0],[0,0],[0,1]],np.complex128)
    assert real_oracle(XL,np.array([0,0,1]))['error']<real_oracle(XD,np.array([0,0,1]))['error']
    assert real_oracle(XD,np.array([0,1,0]))['error']<real_oracle(XL,np.array([0,1,0]))['error']


def test_streaming_small_immutable_sufficient_outputs():
    XD,r=fixture();XL=XD*np.array([.6,1.2])[None,:];g=np.array([1,2,3,4],np.complex128)
    online=np.array([.4,-.3]);S=np.array([[2.,.1],[.1,1.]]);b=np.array([.3,.6])
    result=streaming_one_frame(g,g+r,XD,XL,online,S,b,10.)
    assert result['data_gram'].shape==(2,2,2)
    assert result['data_rhs'].shape==(2,2)
    assert result['cross_gram'].shape==(2,2)
    assert result['oracle_coordinates'].shape==(2,2)
    assert result['oracle_rank'].tolist()==[2,2]
    assert result['oracle_bound_max_violation'].max()<1e-10
    assert abs(result['audit']['compression_identity_residual'])<1e-12
    assert abs(result['audit']['direct_quadratic_max_abs'])<1e-12
    assert result['pilot_loss']==pytest.approx(quadratic_risk(S,b,10,online))


def test_batched_quadratic():
    X,r=fixture();H,b,k=real_sufficient(X,r);cs=np.array([[0,0],[1,.3],[-3,2]])
    vals=quadratic_risk(H,b,k,cs)
    expected=np.array([np.vdot(X@c-r,X@c-r).real for c in cs])
    np.testing.assert_allclose(vals,expected,atol=1e-13)


@pytest.mark.parametrize('obj',[np.array([np.inf+0j]),np.array([np.nan+1j])])
def test_nonfinite_complex_vectors_refused(obj):
    with pytest.raises(ValueError):compression_terms(obj,np.zeros_like(obj))
