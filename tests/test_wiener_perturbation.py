"""Independent deterministic complex-matrix sentinels; no stochastic channels."""
import inspect
import numpy as np
import pytest
from scipy.linalg import cholesky
from src.wiener_perturbation import WienerSensitivity, compare, real_scalar, spectrum, principal_angles, mechanism_gate


@pytest.fixture
def model():
    p = np.array([[2,.1+.2j,.05j],[.1-.2j,1.5,.15],[-.05j,.15,1.]])
    d = np.array([[1+.2j,.5j,.3],[.2,1-.3j,.1j],[.7,.2j,1.]])
    return WienerSensitivity.build(d,p,.4)


@pytest.fixture
def perturb():
    f = np.array([[.1j,.2,.03],[.04,.1j,.05],[.02j,.03,.08]])
    e = np.array([[.1,.02j,.01],[-.02j,-.05,.03],[.01,.03,.06]])
    return f,e


def test_cholesky_right_solve_and_frozen(model):
    s = model.p0+.4*np.eye(3)
    np.testing.assert_allclose(model.w0@s,model.d0,rtol=2e-15,atol=2e-15)
    assert not model.w0.flags.writeable and not model.factor[0].flags.writeable
    with pytest.raises(ValueError): model.w0[0,0]=0


@pytest.mark.parametrize('h',[1e-2,1e-3,1e-4])
def test_first_order_finite_difference(model,perturb,h):
    f,e=perturb
    derivative=(model.exact(h*f,h*e)-model.exact(-h*f,-h*e))/(2*h)
    np.testing.assert_allclose(derivative,model.first(f,e),rtol=3e-6,atol=3e-8)


def test_second_order_small_epsilon_cubic_convergence(model,perturb):
    f,e=perturb; errors=[]; first_errors=[]
    for h in (.1,.05,.025):
        exact=model.exact(h*f,h*e)
        first=model.w0+h*model.first(f,e)
        errors.append(np.linalg.norm(exact-first-h*h*model.second(f,e)))
        first_errors.append(np.linalg.norm(exact-first))
    assert all(7.8<a/b<8.2 for a,b in zip(errors,errors[1:]))
    assert all(3.9<a/b<4.1 for a,b in zip(first_errors,first_errors[1:]))


def test_second_order_coefficients_independent_scalar_formula():
    # W(epsilon)=(3+2epsilon)/(5+epsilon): L=.28, Q=-.056.
    m=WienerSensitivity.build([[3]],[[4]],1.)
    assert m.first([[2]],[[1]])[0,0]==pytest.approx(.28)
    assert m.second([[2]],[[1]])[0,0]==pytest.approx(-.056)


def test_cross_identity_and_linear_interaction(model,perturb):
    fp,ep=perturb; fm=np.flip(fp,axis=0)*(.4+.2j); em=np.diag([.03,-.02,.01])
    direct=model.second(fp+fm,ep+em)-model.second(fp,ep)-model.second(fm,em)
    np.testing.assert_allclose(direct,model.cross(fp,ep,fm,em),rtol=2e-14,atol=2e-17)
    fi=fp*.2; ei=ep*.1
    np.testing.assert_allclose(model.first(fp+fm+fi,ep+em+ei),
                               model.first(fp,ep)+model.first(fm,em)+model.first(fi,ei),rtol=3e-15,atol=2e-17)


def test_exact_module_interaction_decomposition(model,perturb):
    f,e=perturb; fm=f*.6; em=e*-.3; fi=f*.1; ei=e*.05
    wp=model.exact(f,e); wm=model.exact(fm,em); wpm=model.exact(f+fm+fi,e+em+ei)
    residual=wpm-(wp+wm-model.w0)
    pred=model.first(fi,ei)+model.cross(f,e,fm,em)
    remainder=residual-pred
    np.testing.assert_allclose(residual,pred+remainder,rtol=3e-15,atol=1e-17)


def test_weighted_metric_hand_formula_and_negative_closure():
    actual=np.array([[1,1j],[2,0]],complex); pred=actual*3
    lower=np.diag([2,3])
    r=compare(actual,pred,lower)
    assert r['reference_weighted_energy']==29
    assert r['residual_weighted_energy']==116
    assert r['relative_weighted_residual']==2
    assert r['closure']==-3
    assert r['weighted_cosine_real']==pytest.approx(1)


def test_zero_reference_is_explicitly_undefined():
    r=compare(np.zeros((2,2)),np.zeros((2,2)),np.eye(2))
    assert r['closure'] is None and r['weighted_cosine_real'] is None


def test_real_scalar_normal_equation_not_complex_fit():
    b=np.array([[1,2j],[2,1j]]); a=(.6+.4j)*b
    r=real_scalar(a,b)
    assert r['gamma']==pytest.approx(.6)
    assert r['relative_residual']==pytest.approx(.4/np.sqrt(.52))
    assert r['normal_equation_relative_residual']<1e-15


def test_c75_sensitivity_weighted_gamma(model,perturb):
    f,e=perturb; lower=cholesky(model.p0+model.sigma2*np.eye(3),lower=True)
    for a,b in [(f*.6,f),(e*.6,e)]:
        raw=real_scalar(a,b)
        mapped_a=model.right_solve(a if a.shape==model.d0.shape and np.array_equal(b,f) else model.w0@a)@lower
        mapped_b=model.right_solve(b if np.array_equal(b,f) else model.w0@b)@lower
        assert raw['gamma']==pytest.approx(.6)
        assert real_scalar(mapped_a,mapped_b)['gamma']==pytest.approx(.6)


def test_subspace_and_effective_rank():
    a=np.diag(np.arange(8,0,-1)).astype(complex)
    u,s,capture,diag=spectrum(a)
    assert all(np.diff(capture)>=0) and capture[-1]==pytest.approx(1)
    assert diag['numerical_rank']==8
    assert 1<diag['entropy_singular_value_effective_rank']<8
    assert max(principal_angles(u,u,4))<1e-5


@pytest.mark.parametrize('first,second,state',[(.70,.90,'supported'),(.299,.499,'contradicted'),(.30,.49,'partial'),(.69,.95,'partial'),(-5,-20,'contradicted')])
def test_fixed_gate_boundaries_and_unclipped_negative(first,second,state):
    rows=[dict(snr_db=s,first_order_additive_closure=first,interaction_corrected_closure=second) for s in (10,20)]
    g=mechanism_gate(rows)
    assert g['wiener_perturbation_mechanism_'+state]
    assert sum(g.values())==1


@pytest.mark.parametrize('bad',[-1,0,float('nan')])
def test_invalid_noise(model,bad):
    with pytest.raises(ValueError): WienerSensitivity.build(model.d0,model.p0,bad)


def test_no_inverse_or_channel_path():
    import src.wiener_perturbation as module
    source=inspect.getsource(module)
    assert 'np.linalg.inv(' not in source and 'np.linalg.pinv(' not in source
    assert '.generate(' not in source and 'StreamingStatistics' not in source


def test_nonhermitian_and_nonpositive_rejected(model):
    p=model.p0.copy();p[0,1]+=1
    with pytest.raises(ValueError): WienerSensitivity.build(model.d0,p,.4)
    with pytest.raises(np.linalg.LinAlgError): WienerSensitivity.build(model.d0,-10*np.eye(3),.4)


def test_neumann_radius_zero_and_known_scalar():
    m=WienerSensitivity.build([[3]],[[4]],1.)
    assert m.neumann_radius([[0]])==0
    assert m.neumann_radius([[10]])==pytest.approx(2)


def test_historical_w0_mismatch_rejected(model):
    with pytest.raises(ValueError):
        WienerSensitivity.build(model.d0,model.p0,.4,historical_w0=model.w0+1e-14)
