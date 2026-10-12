"""B5A fidelity only: no project preparation/evaluation datasets."""
from functools import lru_cache
import inspect
import numpy as np
import pytest
from scripts.sp_u12b5a_cases import (
    posterior_case,em_case,seed_case,initialization_case,vector_case,
    observation_case,grid_case,firewall_case,known_model,reference_modules,
    ProjectPilotInterface,ProjectMFAReceiver,make_reference_model,
    MFAReferenceAdapter,FROZEN_POLICY)


def test_original_reference_import():
    core,estimator,utilities=reference_modules()
    assert issubclass(estimator.MfaEstimator,core.MFA_cplx)
    z=np.array([[1+2j,3-4j]])
    assert np.array_equal(utilities.real2cplx(utilities.cplx2real(z,1),1),z)
    assert not hasattr(np,'Inf') and core.np.Inf==np.inf


@pytest.mark.parametrize('K',[1,2,4])
@pytest.mark.parametrize('kind',['identity','selector','colored'])
def test_posterior_shapes_dtype_responsibilities_dense_reference(K,kind):
    report=posterior_case(K,kind)
    assert report['PASS']
    assert report['dtype']=='complex128'
    assert report['raw_adapter']['max_absolute_difference']==0


def test_original_em_update_order_against_independent_dense_moments():
    assert em_case()['PASS']


def test_initialization_draw_order_and_rng_restore():
    assert initialization_case()['PASS']


@lru_cache(maxsize=1)
def _seed_report():
    return seed_case()


def test_training_author_adapter_bitwise_parity():
    r=_seed_report()
    assert r['PASS']
    assert all(x['max_absolute_difference']==0 for x in r['reference_adapter'].values())


def test_seed_replay_bitwise():
    assert _seed_report()['bitwise_replay']


def test_objective_progression_finite_without_invented_monotonicity_requirement():
    r=_seed_report()
    assert r['history_finite']
    assert 7<=r['iterations']<=100
    assert not r['strict_monotonicity_required']


def test_covariance_structure_and_weights():
    m=known_model()
    assert np.allclose(m.covs,m.covs.conj().transpose(0,2,1))
    assert np.all(np.linalg.eigvalsh(m.covs)>0)
    assert np.all(m.psis==m.psis[:,:1])
    assert np.all(m.amps>=0) and m.amps.sum()==1


def test_original_clipping_deviation_recorded_not_repaired():
    model=known_model();model._means[1]+=100;model.N=3
    data=np.zeros((3,6),complex)
    _,r=model._calc_probs(data)
    assert np.allclose(r.sum(axis=0),1,rtol=0,atol=1e-10)
    model.rs_clip=1e-6  # Only this source-behavior test, never the baseline policy.
    _,clipped=model._calc_probs(data)
    assert np.all(clipped.sum(axis=0)>1+1e-7)
    assert FROZEN_POLICY['rs_clip']==0


def test_vectorization_roundtrip_full_and_explicit_sidecar():
    r=vector_case()
    assert r['PASS'] and r['selector_matches_full'] and r['data_order_matches']
    assert r['full4096_bitwise_roundtrip'] and r['joint4032_with_sidecar64_bitwise_roundtrip']


def test_same_production_pilots_and_noise_transport():
    assert observation_case()['PASS']


@pytest.mark.parametrize('K',[1,2,4])
@pytest.mark.parametrize('L',[4,8,16])
def test_structure_semantics_no_formal_training(K,L):
    m=make_reference_model(K,L)
    assert m.n_components==K and m.M==L
    assert m.PPCA and not m.lock_psis and m.rs_clip==0
    assert m.maxiter==100 and m.tol==1e-4
    assert m.D is None


def test_all_nine_grid_tiny_inference_support():
    assert grid_case()['PASS']


def test_target_information_firewall_and_online_contract():
    assert firewall_case()['PASS']
    assert list(inspect.signature(ProjectMFAReceiver.estimate_data).parameters)==['self','received_owner_pilot_symbols','noise_variance']


def test_full_mixture_not_integer_default():
    m=known_model();A=np.eye(6)[[0,2,5]];y=np.zeros((1,3),complex);Cn=.37*np.eye(3)
    full=MFAReferenceAdapter(m,A).estimate(y,Cn)
    hard=m.estimate(y,Cn,A,1)
    assert np.max(abs(full-hard))>1e-6


def test_project_receiver_only_maps_pilots_and_data_order():
    class RecordingReference:
        def estimate(self,**kwargs):
            self.kwargs=kwargs
            return np.arange(4032,dtype=np.complex128)[None]
    raw=RecordingReference();receiver=ProjectMFAReceiver(raw)
    symbols=receiver.interface.owner_symbols
    out=receiver.estimate_data(2*symbols,.37)
    assert out.shape==(1,3968) and out.dtype==np.complex128
    assert np.array_equal(out[0],receiver.interface.data_positions)
    assert np.allclose(raw.kwargs['y'],2,atol=1e-15,rtol=0)
    assert np.array_equal(raw.kwargs['A'],receiver.interface.A)
    assert type(raw.kwargs['n_summands_or_proba']) is float and raw.kwargs['n_summands_or_proba']==1.0
    assert np.array_equal(raw.kwargs['Cn'],.37*np.eye(64))


def test_no_native_implementation_or_algorithm_enhancement():
    assert FROZEN_POLICY==dict(PPCA=True,lock_psis=False,rs_clip=0.0,max_condition_number=1e6,maxiter=100,tol=1e-4,verbose=False)
