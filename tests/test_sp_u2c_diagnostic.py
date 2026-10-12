"""Pre-run numerical and firewall calibration; synthetic noise only."""
import inspect
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2c_diagnostic as d
d.install_guard()


@pytest.fixture(scope='module')
def inputs():
    p=d.load(d.OUT/'sp_u2c_preregistration.json')
    return p,d.cache(0,p['cache_identity']),d.frozen_method()


def test_fixed_cache_identity(inputs):
    p,x,_=inputs
    assert d.array_hash(x['H'])==p['cache_identity'][0]['H_sha256']
    assert not x['H'].flags.writeable
    assert d.sha(d.OUT/'sp_u2b_development_cache/trajectory_000.npz')==p['cache_identity'][0]['cache_sha256']


def test_selected_ids_exact():
    assert d.selected_ids()==tuple(range(20))
    with pytest.raises(ValueError):d.cache(20,[])


def test_selection_no_performance_parameter():
    assert not inspect.signature(d.selected_ids).parameters
    assert inspect.getsource(d.selected_ids).split('return ')[1].split()[0]=='IDS'


def test_seed_unique_and_old_disjoint():
    import csv
    new={d.noise_seed(i,r,j) for i in d.IDS for r in range(20) for j in range(10)}
    with (d.OUT/'sp_u2b_seed_manifest.csv').open(encoding='utf8') as f:
        old={int(a[k]) for a in csv.DictReader(f) for k in a if k.endswith('seed') or k.startswith('bits_seed')}
    assert len(new)==4000 and not new.intersection(old)


def test_paired_snr_standard_atom():
    from src.numerics import complex_gaussian_noise
    atom=complex_gaussian_noise((2,64,16),1.,np.random.default_rng(717))
    assert np.allclose(np.sqrt(d.VARIANCES[10])*atom/np.sqrt(10),np.sqrt(d.VARIANCES[20])*atom,rtol=2e-15,atol=2e-15)
    source=inspect.getsource(d.noisy_inputs)
    assert source.count('complex_gaussian_noise(')==1
    assert 'for s in SNRS' in source and 'atoms[j]' in source


def test_clean_coordinate_invariance(inputs):
    _,x,method=inputs
    for s in d.SNRS:
        c,_=method[1][s].fit(x['clean_pilots'][:5],modules=('P','M'))
        assert np.array_equal(c,x[f'c0_{s}_4'])
        assert len({d.array_hash(c) for _ in range(20)})==1


def synthetic():
    rng=np.random.default_rng(918)
    X=rng.normal(size=(37,2))+1j*rng.normal(size=(37,2))
    truth=rng.normal(size=37)+1j*rng.normal(size=37)
    base=rng.normal(size=37)+1j*rng.normal(size=37)
    return X,truth,base


def test_oracle_direct_ls():
    from scripts.sp_u2b_diagnostic import channel_oracle
    X,t,b=synthetic();c,M,e=channel_oracle(t,b,X)
    assert e<d.TOL
    assert np.allclose(M@c,(X.conj().T@(t-b)).real)


def test_risk_decomposition_and_sign():
    from scripts.sp_u2b_diagnostic import channel_oracle
    X,t,b=synthetic();co,M,_=channel_oracle(t,b,X)
    c0=np.array([.5,-.2]);n=np.array([-.1,.2]);s=c0-co
    risk=lambda c:float(np.vdot(b+X@c-t,b+X@c-t).real/37)
    Q=float(n@M@n/37);C=float(2*s@M@n/37)
    d.close(risk(c0+n)-risk(c0),Q+C)
    d.close(risk(c0)-risk(co),s@M@s/37)
    assert (Q+C<0)==(C < -Q)


@pytest.mark.parametrize('s,n,undefined',[([0,0],[1,2],True),([1,0],[0,0],True),([1,0],[-1,0],False),([1e-20,0],[1,0],True)])
def test_cosine_safety(s,n,undefined):
    value=d.cosine(np.array(s,float),np.array(n,float),np.eye(2))
    assert value['alignment_undefined']==undefined
    assert (value['alignment_cosine'] is None)==undefined


def test_empirical_covariance():
    c=np.array([[0.,1.],[2.,-1.],[4.,3.]])
    value=d.empirical(c,np.array([1.,2.]))
    cov=np.cov(c,rowvar=False,ddof=1)
    assert value['variance_alpha']==cov[0,0] and value['variance_beta']==cov[1,1]
    assert value['covariance_alpha_beta']==cov[0,1]


@pytest.mark.parametrize('name',['t20b_forbidden_sentinel.json','sp_u2b_development_windows.csv','sp_u2b_development_cache/trajectory_020.npz'])
def test_firewall_blocks_before_open(name):
    with pytest.raises(PermissionError):d.sha(d.OUT/name)


def test_production_protection():
    from scripts.run_sp_u2c import verify
    pre=verify()
    assert any(k=='src/pilot_cv.py' for k in pre['protected_sources'])


def test_deterministic_replay_synthetic_noise():
    from src.numerics import complex_gaussian_noise
    # Calibration namespace; scientific repeat seeds are not consumed here.
    for seed in (719,721,723):
        a=complex_gaussian_noise((2,64,16),1.,np.random.default_rng(seed))
        b=complex_gaussian_noise((2,64,16),1.,np.random.default_rng(seed))
        assert np.array_equal(a,b)


def test_real_fixed_and_noisy_input_entry(inputs,monkeypatch):
    _,x,method=inputs
    useful,hashes=d.fixed_inputs(0,x)
    monkeypatch.setattr(d,'noise_seed',lambda i,r,j:800+j)
    pilots,atoms=d.noisy_inputs(0,0,useful)
    again,again_atoms=d.noisy_inputs(0,0,useful)
    assert np.array_equal(pilots,again) and atoms==again_atoms
    assert len(hashes)==10 and pilots.shape==(2,10,64)
    d.close((pilots[0]-x['clean_pilots'])/np.sqrt(10),pilots[1]-x['clean_pilots'])
    rows=d.diagnose(0,0,pilots,x,method)
    assert len(rows)==12 and max(a['identity_error_max'] for a in rows)<d.TOL


def test_no_channel_generator_import():
    source=(ROOT/'scripts/sp_u2c_diagnostic.py').read_text(encoding='utf8')
    assert 'from src.cdl_benchmark' not in source and 'from sionna' not in source


def test_conditional_covariance_summary_schema():
    from scripts.analyze_sp_u2c import stats
    assert stats([])['n']==0 and stats([])['mean'] is None
    assert d.normal({'x':np.int64(3)})=={'x':3}
