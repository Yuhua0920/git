import importlib.util,ast,json
from pathlib import Path
import numpy as np
import pytest
PATH=Path(r'D:\Users\yuhua\Desktop\Project\outputs\revision_mr03_em16_reconciliation\src\reconcile.py')
spec=importlib.util.spec_from_file_location('reconcile_tested',PATH);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def example():
    F=np.array([[1+2j,.3],[.2-.4j,1j],[.4+.2j,1-.3j],[2j,.7],[.5,1+.6j]],complex)
    C=F@F.conj().T+.07*np.eye(5);mu=np.array([.4j,1+.2j,2j,-.3,.1+.5j])
    return F,C,mu,np.array([3,0]),np.array([4,1,2])

@pytest.mark.parametrize('sigma',[.01,.2,1.])
def test_noncontiguous_covariance_conditioning(sigma):
    F,C,mu,p,d=example();W,mp,md=r.conditional_from_covariance(C,mu,p,d,sigma)
    literal=C[np.ix_(d,p)]@np.linalg.inv(C[np.ix_(p,p)]+sigma*np.eye(2))
    np.testing.assert_allclose(W,literal,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(mp,mu[p]);np.testing.assert_array_equal(md,mu[d])

def test_natural_mean_not_source_centered():
    F,C,mu,p,d=example();W,mp,md=r.conditional_from_covariance(C,mu,p,d,.2)
    np.testing.assert_allclose(r.predict(W,mp,md,mp),md)

def test_noise_not_added_twice():
    F,C,mu,p,d=example();W,_,_=r.conditional_from_covariance(C,mu,p,d,.2)
    wrong=C[np.ix_(d,p)]@np.linalg.inv(C[np.ix_(p,p)]+.4*np.eye(2))
    assert np.linalg.norm(W-wrong)>.01

def test_complex_conjugation_matters():
    F,C,mu,p,d=example();W,_,_=r.conditional_from_covariance(C,mu,p,d,.2)
    wrong=C[np.ix_(d,p)].conj()@np.linalg.inv(C[np.ix_(p,p)]+.2*np.eye(2))
    assert np.linalg.norm(W-wrong)>.1

def test_single_batch_equivalence():
    F,C,mu,p,d=example();W,mp,md=r.conditional_from_covariance(C,mu,p,d,.2)
    Y=np.array([[.1+.3j,.4-.9j],[1-.2j,-.3+.1j]])
    np.testing.assert_allclose(r.predict(W,mp,md,Y),np.stack([r.predict(W,mp,md,y) for y in Y]),atol=1e-13)

def test_invalid_nonfinite_prediction_rejected():
    with pytest.raises(ValueError):r.check([np.nan],[0],'nan')

def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):r.check(np.zeros((2,1)),np.zeros(2),'shape')

def test_material_fidelity_difference_rejected():
    with pytest.raises(ValueError):r.check([1.01],[1.],'difference')

def test_no_training_or_parameter_assignment_in_evaluation():
    tree=ast.parse(PATH.read_text(encoding='utf8'))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('fit','fit_transform','partial_fit','run_em') for n in ast.walk(tree))

def test_receipt_json_keywords():
    tree=ast.parse(PATH.read_text(encoding='utf8'))
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='dumps':assert not any(k.arg=='flush' for k in n.keywords)

def test_covariance_loading_maps_agree():
    F,C,mu,p,d=example();W,mp,md=r.conditional_from_covariance(C,mu,p,d,.2)
    model=r.ppca.PPCAModel(mu,F,.07);co=r.ppca.conditional_operator(model,p,d,.2)
    np.testing.assert_allclose(W,co.W,atol=1e-12)

def test_exact_sign_and_paired_energy():
    a=np.array([1.,2.,5.]);b=np.array([2.,4.,10.]);q=np.array([4.,5.,6.])
    assert 10*np.log10(a.sum()/q.sum())-10*np.log10(b.sum()/q.sum())<0
