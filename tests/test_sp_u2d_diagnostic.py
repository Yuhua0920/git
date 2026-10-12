"""Pre-analysis real-entry, symmetry, numerical and firewall calibration."""
import ast
import inspect
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2d_diagnostic as d
d.install_guard()


@pytest.fixture(scope='module')
def inputs():
    pre=d.load(d.OUT/'sp_u2d_preregistration.json')
    metadata={(a['trajectory_id'],a['repeat_id']):a for a in pre['repeat_caches']}
    method=d.u.frozen_method();fixed=d.fixed_cache(0,pre['fixed_cache_identity']);pilots=d.repeat_cache(0,0,metadata)
    energy={(int(a['trajectory_id']),int(a['snr_db']),int(a['frame_id'])):a for a in d.read_rows(d.OUT/'sp_u2d_energy_manifest.csv')}
    return pre,method,fixed,pilots,d.production_references(),energy


def test_exact_cache_identity(inputs):
    pre,_,fixed,pilots,_,_=inputs
    assert d.array_hash(fixed['H'])==pre['fixed_cache_identity'][0]['H_sha256']
    assert not pilots.flags.writeable and not fixed['H'].flags.writeable


def test_ids_exact():assert d.allocation()[0]==tuple(range(20))


def test_all_twenty_repeats():
    pre=d.load(d.OUT/'sp_u2d_preregistration.json')
    assert d.allocation()[1]==tuple(range(20))
    assert {(a['trajectory_id'],a['repeat_id']) for a in pre['repeat_caches']}=={(i,r) for i in range(20) for r in range(20)}


def test_no_generator_call_source():
    tree=ast.parse(inspect.getsource(d))
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{'noisy_inputs','fixed_inputs','generate_upgrade','generate','complex_gaussian_noise','default_rng'}


def test_moments_reconstruct_full(inputs):
    _,method,_,pilots,_,_=inputs
    cv=method[1][10];c,diag=cv.fit(pilots[0,:5],modules=('P','M'))
    H,b,ha,ba,e=d.chronological_moments(cv,pilots[0,:5])
    assert np.array_equal(H,diag['gram']) and np.array_equal(b,diag['rhs'])
    assert e<d.TOL


def test_age_index_correct(inputs):
    _,method,_,pilots,_,_=inputs;cv=method[1][10]
    H,b,ha,ba,e=d.chronological_moments(cv,pilots[0,:5])
    for a in range(5):
        moment=cv.frame_objective(pilots[0,4-a])
        assert np.array_equal(ha[a],moment[0].real) and np.array_equal(ba[a],moment[1].real)


def synthetic_moments():
    H=np.stack([np.array([[2+a*.3,.1*(a-2)],[.1*(a-2),3+a*.1]]) for a in range(5)])
    b=np.array([[.2*a,-.1*a+.3] for a in range(5)])
    return H,b


def test_permutation_equivariance():
    h,b=synthetic_moments();fullH=sum(h);fullb=sum(b)
    base=d.lofo_moments(fullH,fullb,h,b)
    perm=[3,0,4,1,2];changed=d.lofo_moments(fullH,fullb,h[perm],b[perm])
    for k,a in enumerate(perm):assert np.array_equal(changed[k]['c'],base[a]['c'])


def test_stationary_identical_moments():
    h=np.tile(np.array([[3.,.2],[.2,2.]])[None],(5,1,1));b=np.tile(np.array([.4,.6]),(5,1))
    result=d.lofo_moments(sum(h),sum(b),h,b)
    assert all(np.array_equal(a['c'],result[0]['c']) for a in result)


def test_four_frame_count(inputs):
    _,method,_,pilots,_,_=inputs
    from scripts.sp_u2b_diagnostic import explicit_design
    for a in range(5):
        keep=np.arange(5)!=4-a
        assert keep.sum()==4
        G,z=explicit_design(method[1][10],pilots[0,:5][keep])
        assert G.shape==(512,2) and z.shape==(512,)


def test_real_solve_direct_reference(inputs):
    _,method,_,pilots,_,_=inputs;cv=method[1][10]
    H,b,ha,ba,_=d.chronological_moments(cv,pilots[0,:5])
    for a,variant in enumerate(d.lofo_moments(H,b,ha,ba)):
        direct,_=cv.fit(pilots[0,:5][np.arange(5)!=4-a],modules=('P','M'))
        d.close(variant['c'],direct)


def test_no_ridge_clip_fallback():
    source=inspect.getsource(d.lofo_moments)
    assert 'np.linalg.solve' in source
    assert not any(x in source for x in ('pinv','lstsq','clip','ridge','np.eye'))


def test_singular_preserved_no_fallback():
    H=np.tile(np.array([[1.,0.],[0.,0.]])[None],(5,1,1));b=np.ones((5,2))
    results=d.lofo_moments(sum(H),sum(b),H,b)
    assert len(results)==5 and all(a['singular'] and a['c'] is None for a in results)


@pytest.mark.parametrize('snr,si',[(10,0),(20,1)])
def test_real_entry_full_replay_risk_and_generator_sentinels(inputs,monkeypatch,snr,si):
    pre,method,fixed,pilots,refs,energy=inputs
    def forbidden(*args,**kwargs):raise AssertionError('new generator forbidden')
    monkeypatch.setattr(d.u,'noisy_inputs',forbidden)
    monkeypatch.setattr(d.u,'fixed_inputs',forbidden)
    import src.numerics
    monkeypatch.setattr(src.numerics,'complex_gaussian_noise',forbidden)
    rows=d.record(0,0,snr,4,pilots[si],fixed,method,refs[(0,0,snr,4)],energy[(0,snr,4)])
    assert len(rows)==5 and all(a['retained_frames']==4 and a['removed_frame']==4-a['age'] for a in rows)
    assert max(a['risk_identity_error'] for a in rows)<d.TOL
    assert max(a['direct_four_frame_error'] for a in rows)<d.TOL
    assert d.paired(rows)['valid_all_ages']


def test_cosine_same_safety():
    assert d.u.cosine(np.zeros(2),np.ones(2),np.eye(2))['alignment_undefined']
    assert d.u.cosine(np.array([1.,0]),np.array([-1.,0]),np.eye(2))['alignment_cosine']==-1


@pytest.mark.parametrize('name',['t20b_forbidden_sentinel.json','sp_u2c_analysis.json','sp_u2b_development_cache/trajectory_020.npz'])
def test_formal_firewall(name):
    with pytest.raises(PermissionError):d.sha(d.OUT/name)


def test_old_inputs_readonly():
    with pytest.raises(PermissionError):
        d.output_guard('open',(str(d.OUT/'sp_u2c_repeat_cache/trajectory_000_repeat_00.npz'),'w',os_flags()))


def os_flags():
    import os
    return os.O_WRONLY|os.O_TRUNC


def test_production_hashes_unchanged():
    from scripts.run_sp_u2d import verify
    pre=verify();assert 'src/pilot_cv.py' in pre['protected_sources']


def test_energy_rule_pre_registered(inputs):
    pre,_,_,_,_,energy=inputs
    assert pre['energy_strata_outcome_free']
    assert len(energy)==240
    assert sum(a['energy_tercile']==1 for a in energy.values())==80
    assert all(a['similar_clean_energy']==(a['clean_max_min_ratio']<=1.10) for a in energy.values())
