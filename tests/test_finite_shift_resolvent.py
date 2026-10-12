"""Independent complex128 exact identities and predefined decision boundaries."""
from fractions import Fraction
import inspect
import ast
import numpy as np
import pytest
from src import finite_shift_resolvent as exact


def scalar():
    d=dict(zip(exact.NAMES,[np.array([[v]],complex) for v in (3,5,2,6)]))
    p=dict(zip(exact.NAMES,[np.array([[v]],complex) for v in (4,6,3,8)]))
    return exact.ExactWienerMap.build(d,p,1.)


def complex_fixture():
    p0=np.array([[2,.2+.1j],[.2-.1j,1]],complex)
    ep=np.array([[.5,.15j],[-.15j,.4]])
    em=np.array([[.2,.1],[.1,.6]])
    ei=np.array([[.03,.02j],[-.02j,.08]])
    d0=np.array([[1+.3j,.4j],[.2,.5-.2j],[.1j,.3]])
    fp=np.array([[.1j,.2],[.2j,.1],[-.2,.1j]])
    fm=np.array([[.3,.1j],[.05j,.2],[.1,.2j]])
    fi=np.array([[.02j,.01],[.03,.02j],[.01j,.04]])
    return exact.ExactWienerMap.build(dict(zip(exact.NAMES,(d0,d0+fp,d0+fm,d0+fp+fm+fi))),
        dict(zip(exact.NAMES,(p0,p0+ep,p0+em,p0+ep+em+ei))),.3)


def test_scalar_four_terms_against_independent_rational_arithmetic():
    m=scalar();r,c,a=m.decomposition()
    expected=[Fraction(2,9),Fraction(-4,63),Fraction(5,36),Fraction(-103,420)]
    for value,target in zip(c.values(),expected):np.testing.assert_allclose(value,float(target),rtol=2e-14,atol=2e-16)
    np.testing.assert_allclose(r,float(Fraction(11,210)),rtol=2e-14,atol=2e-16)
    assert a['four_term_relative_residual']<2e-13


def test_noncommuting_complex_four_term_and_filter_equations():
    m=complex_fixture();r,c,a=m.decomposition()
    assert np.linalg.norm(m.p['P']@m.p['M']-m.p['M']@m.p['P'])>.01
    assert a['four_term_relative_residual']<2e-13
    for k,w in m.w.items():
        np.testing.assert_allclose(w@(m.p[k]+m.sigma2*np.eye(2)),m.d[k],rtol=2e-14,atol=2e-15)
        assert w.dtype==np.complex128 and not w.flags.writeable


@pytest.mark.parametrize('pair',range(4))
def test_each_resolvent_identity_noncommuting_action(pair):
    m=complex_fixture()
    assert m.resolvent_audit(m.d['0'])[pair]['action_relative_residual']<2e-13


def test_weighted_gram_and_cancellation_hand_sentinel():
    c={k:np.array([[v]],complex) for k,v in zip(exact.TERMS,(1,-.5,.25,-.25))}
    r,rows=exact.weighted_bookkeeping(np.array([[.5]]),np.array([[1.]]),c,np.array([[2.]]))
    assert [x['weighted_energy'] for x in rows]==[4.,1.,.25,.25]
    assert r['cancellation_index']==5.5 and r['cross_fraction']==-4.5
    assert r['energy_identity_relative_residual']==0
    assert r['weighted_gram_real'][0][1]==-2 and r['cross_energy_matrix'][0][1]==-4
    assert r['cross_energy_matrix'][0][0]==8


def test_compensated_Gram_sum_retains_small_exact_remainder():
    c={k:np.array([[v]],complex) for k,v in zip(exact.TERMS,(1e8,-1e8,1.,0.))}
    r,_=exact.weighted_bookkeeping(np.array([[1.]]),np.array([[1.]]),c,np.array([[1.]]))
    assert r['reconstructed_Gram_energy']==1. and r['energy_identity_relative_residual']==0.
    # Ordinary grouped sums round the 1-energy contribution away.
    assert sum(np.diag(r['weighted_gram_real']))+(-2e16)==0.


@pytest.mark.parametrize('pd',[True,False])
def test_optional_counterfactual_PD_and_unavailable_no_regularization(pd):
    m=scalar() if pd else exact.ExactWienerMap.build(
        {k:np.array([[v]]) for k,v in zip(exact.NAMES,(3,5,2,6))},
        {k:np.array([[v]]) for k,v in zip(exact.NAMES,(10,1,1,2))},1.)
    r,_,_=m.decomposition();record=m.counterfactual(r,m.lower())
    assert record['additive_statistics_counterfactual_available']==pd
    assert not record['regularization_used']
    if pd:assert record['decomposition_relative_residual']<2e-13 and record['energy_identity_relative_residual']<2e-13
    else:assert 'covariance_nonadditivity_energy' not in record and record['minimum_eigenvalue']<0


@pytest.mark.parametrize('closure,index,size,outcome',[(.9,2.,.5,'supported'),(.9,10.,.4,'cancellation'),
    (.9,3.,1.,'cancellation'),(.9,2.01,.5,'partial'),(.899,10.,2.,'partial'),(.95,.7,.2,'supported')])
def test_preregistered_gate_boundaries(closure,index,size,outcome):
    rows=[dict(snr_db=s,exact_affine_closure=closure,cancellation_index=index,
               maximum_contribution_norm_ratio=size) for s in (5,10,20)]
    r=exact.decision(rows)
    assert r==dict(finite_shift_separability_supported=outcome=='supported',
        affine_additivity_cancellation_dominated=outcome=='cancellation',finite_shift_separability_partial=outcome=='partial')


def test_any_primary_SNR_cancellation_rule_not_five_dB():
    rows=[dict(snr_db=s,exact_affine_closure=.95,cancellation_index=1.,maximum_contribution_norm_ratio=.1) for s in (5,10,20)]
    rows[0]['cancellation_index']=100.
    assert exact.decision(rows)['finite_shift_separability_supported']
    rows[1]['cancellation_index']=10.
    assert exact.decision(rows)['affine_additivity_cancellation_dominated']


@pytest.mark.parametrize('noise',[0.,-1.,np.inf,np.nan])
def test_invalid_noise_rejected(noise):
    m=scalar()
    with pytest.raises(ValueError):exact.ExactWienerMap.build(m.d,m.p,noise)


def test_no_inverse_identity_rhs_generator_fit_or_Taylor_calls():
    source=inspect.getsource(exact)
    calls={getattr(n.func,'id',getattr(n.func,'attr','')) for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)}
    assert not calls&{'inv','pinv','first','second','cross','WienerSensitivity','generate','fit','default_rng'}
    assert 'cho_solve' in calls
