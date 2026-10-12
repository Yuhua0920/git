"""SP-U4A protocol, independent numerical fixtures and actual prerequisite gates."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from scripts.sp_u4a_common import *
from src.cdl_benchmark import SYSTEM


@pytest.fixture(scope='module')
def synthetic():
    rng=np.random.default_rng(804120)
    corpus={e:(rng.normal(size=(10,10,4096))+1j*rng.normal(size=(10,10,4096))) for e in ('Source','P','M')}
    stats={n:{e:build_statistics(x,range(n)) for e,x in corpus.items()} for n in (5,10)}
    ops={n:build_operators(s)[0] for n,s in stats.items()}
    return corpus,stats,ops


def test_permutations_frozen_before_nmse():
    p=prereg();assert p['frozen_before_nmse']
    for e,s in p['config']['permutation_seeds'].items():assert p['permutations'][e]==np.random.default_rng(s).permutation(500).tolist()


def test_permutation_hashes():
    p=prereg()
    assert all(array_hash(np.array(ids,dtype=np.int64))==p['permutation_sha256'][e] for e,ids in p['permutations'].items())


def test_nested_subsets():
    for ids in prereg()['permutations'].values():
        assert set(ids[:50])<set(ids[:100])<set(ids[:250])<set(ids[:500])


def test_exact_counts():
    for ids in prereg()['permutations'].values():
        assert sorted(ids)==list(range(500))
        for n in config()['budgets']:assert len(set(ids[:n]))==n


def test_freeze_independent_of_channel_or_results():
    src=Path('scripts/sp_u4a_common.py').read_text(encoding='utf8')
    tree=ast.parse(src);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='freeze')
    body=ast.unparse(node)
    assert 'default_rng(s).permutation(500)' in body and 'np.load' not in body and 'norm(' not in body


def test_eval_namespace_role_disjointness():
    values={c:{tuple(development_seeds(c,i,j).values()) for i in range(100) for j in range(10)} for c in config()['conditions']}
    assert not values['ID']&values['Composed']
    from src.cdl_benchmark import seeds
    assert not {tuple(seeds(i,j).values()) for i in range(500) for j in range(10)} & (values['ID']|values['Composed'])


@pytest.mark.parametrize('field',['mean_p','mean_d'])
def test_means_rebuilt_from_only_selected_rows(synthetic,field):
    corpus,stats,_=synthetic;g=observation_geometry()
    mask=g.selected if field=='mean_p' else g.data
    for n in (5,10):np.testing.assert_allclose(getattr(stats[n]['Source'],field),corpus['Source'][:n].reshape(-1,4096)[:,mask].mean(0),rtol=1e-12,atol=1e-12)
    assert not np.array_equal(getattr(stats[5]['Source'],field),getattr(stats[10]['Source'],field))


@pytest.mark.parametrize('field',['rpp','rdp'])
def test_centered_population_covariance(synthetic,field):
    corpus,stats,_=synthetic;g=observation_geometry()
    for n in (5,10):
        x=corpus['Source'][:n].reshape(-1,4096);x=x-x.mean(0)
        left=x[:,g.selected] if field=='rpp' else x[:,g.data]
        expected=left.T@x[:,g.selected].conj()/len(x)
        np.testing.assert_allclose(getattr(stats[n]['Source'],field),expected,rtol=1e-12,atol=1e-12)


def test_validation_ids_rejected(synthetic):
    with pytest.raises(ValueError):build_statistics(synthetic[0]['Source'],[500])


def test_duplicate_ids_rejected(synthetic):
    with pytest.raises(ValueError):build_statistics(synthetic[0]['Source'],[1,1])


@pytest.mark.parametrize('env',['Source','P','M'])
def test_n_specific_W_normal_equation(synthetic,env):
    _,stats,ops=synthetic
    for n in (5,10):
        for s in (10,20):
            v=stats[n][env];np.testing.assert_allclose(ops[n][f'W{env}_{s}']@(v.rpp+config()['noise_variances'][str(s)]*np.eye(64)),v.rdp,rtol=1e-11,atol=1e-11)
    assert not np.array_equal(ops[5][f'W{env}_10'],ops[10][f'W{env}_10'])


@pytest.mark.parametrize('env',['P','M'])
def test_same_budget_corrections(synthetic,env):
    ops=synthetic[2]
    for n in (5,10):
        for s in (10,20):assert np.array_equal(ops[n][f'DeltaW{env}_{s}'],ops[n][f'W{env}_{s}']-ops[n][f'WSource_{s}'])


@pytest.mark.parametrize('env',['P','M'])
def test_rank4_same_budget_weighted_svd(synthetic,env):
    from scipy.linalg import cholesky
    _,stats,ops=synthetic
    for n in (5,10):
        a=ops[n];delta=a[f'DeltaW{env}_10'];L=cholesky(stats[n][env].rpp+config()['noise_variances']['10']*np.eye(64),lower=True)
        U,S,V=np.linalg.svd(delta@L,full_matrices=False)
        np.testing.assert_allclose((a[f'A_{env}_10']@a[f'Q_{env}_10'])@L,(U[:,:4]*S[:4])@V[:4],rtol=1e-10,atol=1e-11)
        assert a[f'A_{env}_10'].shape==(3968,4) and a[f'Q_{env}_10'].shape==(4,64)


@pytest.mark.parametrize('env',['Source','P','M'])
def test_n_specific_cv_kernels(synthetic,env):
    _,stats,ops=synthetic;folds,_=frozen_folds(observation_geometry().selected)
    for n in (5,10):
        for f,(observed,hold) in enumerate(folds):
            R=stats[n][env].rpp
            np.testing.assert_allclose(ops[n][f'K{env}_10'][f]@(R[np.ix_(observed,observed)]+config()['noise_variances']['10']*np.eye(48)),R[np.ix_(hold,observed)],rtol=1e-11,atol=1e-11)
    assert not np.array_equal(ops[5][f'K{env}_10'],ops[10][f'K{env}_10'])


def test_no_historical_reuse_inside_builder():
    import inspect
    s=inspect.getsource(build_operators)
    assert 'np.load' not in s and 'historical_arrays' not in s and 'stats[e].rpp' in s


def test_fixed_rank_window_folds_coordinates():
    c=config();assert (c['rank'],c['window'],c['folds'])==(4,5,4)
    assert c['coordinate_model']=='unrestricted real; no ridge, no clipping'
    g=observation_geometry();folds,_=frozen_folds(g.selected)
    assert all((len(o),len(h))==(48,16) for o,h in folds)


def test_fixed_variances_reference_power():
    c=config();assert c['p_ref']==9.941134822523972
    assert c['noise_variances']=={'10':.9941134822523973,'20':.09941134822523973}


def test_sampling_rate_and_geometry():
    assert SYSTEM['sample_rate_hz']==SYSTEM['nfft']*SYSTEM['delta_f_hz']
    g=observation_geometry();assert g.selected.size==64 and np.count_nonzero(g.data)==3968


def test_standardized_noise_shared_between_snr():
    from scripts.run_sp_u4a import evaluate
    import inspect
    src=inspect.getsource(evaluate)
    assert src.index('atom=')<src.index('for si,s in enumerate((10,20))')
    assert 'np.sqrt(cfg[\'noise_variances\'][str(s)])*atom' in src


def test_input_pairing_across_budget_arms():
    from scripts.run_sp_u4a import evaluate
    import inspect
    s=inspect.getsource(evaluate)
    assert s.index('pilots[si,j]=')<s.index('for n,a in arrays.items():')
    assert 'model.generate' not in s[s.index('for n,a in arrays.items():'):]


def test_trajectory_metric_uses_energy_ratio_not_frame_average():
    got=trajectory_metric([1,9],[1,99]);assert got['nmse_linear']==.1
    assert got['nmse_db']==-10 and got['nmse_linear']!=np.mean([1,9/99])


def test_permutation_reordering_roundoff_fixture(synthetic):
    x=synthetic[0]['Source'];a=build_statistics(x,range(10));b=build_statistics(x,range(9,-1,-1))
    for k in ('mean_p','mean_d','rpp','rdp'): assert rel_error(getattr(a,k),getattr(b,k))<1e-12


def test_effective_operator_ignores_basis_phase(synthetic):
    a=synthetic[2][5];A=a['A_P_10'];Q=a['Q_P_10'];phase=np.exp(1j*np.arange(4))
    np.testing.assert_allclose((A*phase)@(Q/phase[:,None]),A@Q,rtol=1e-12,atol=1e-12)


def test_formal_and_old_dev_firewall():
    code="from scripts.sp_u4a_common import *; firewall(); "
    for filename in ('t20b_formal_outcomes.csv','sp_u2b_development_windows.csv','sp_u3a_source_smoke_cache.npz'):
        p=subprocess.run([sys.executable,'-c',code+f"(OUT/'{filename}').read_bytes()"],cwd=ROOT,capture_output=True,text=True)
        assert p.returncode!=0 and 'firewall blocked' in p.stderr


def test_historical_write_firewall():
    code="from scripts.sp_u4a_common import *; firewall(); (OUT/'t16e_frozen_operators.npz').open('wb')"
    p=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode!=0 and 'firewall blocked' in p.stderr


def test_historical_protected_files_unchanged():
    assert all(sha(ROOT/p)==h for p,h in protected_files().items())


def test_actual_n500_provenance_gate():
    p=OUT/'sp_u4a_historical500_gate.json'
    if not p.exists():pytest.skip('actual clean500 reconstruction not run yet; synthetic preparation only')
    g=readjson(p);assert g['passed'] and g['historical_full500_per_environment'] and g['effective_products_compared']


def test_actual_all_budgets_consistent():
    if not (OUT/'sp_u4a_construction_500.json').exists():pytest.skip('actual budget preparation pending')
    for n in config()['budgets']:
        m=readjson(OUT/f'sp_u4a_construction_{n}.json');assert m['budget']==m['all_object_budget']==n and m['rank']==4
        assert m['operators_sha256']==sha(OUT/f'sp_u4a_operators_{n}.npz')
        with np.load(OUT/f'sp_u4a_statistics_{n}.npz') as f:
            for e in config()['environments']:
                assert f[f'{e}_samples']==n*10
                assert f[f'{e}_ids'].tolist()==prereg()['permutations'][e][:n]


def test_actual_development_pairing_counts():
    if not (OUT/'sp_u4a_development_gate.json').exists():pytest.skip('development evaluation pending')
    gate=readjson(OUT/'sp_u4a_development_gate.json');assert gate['passed']
    rows=csvread(OUT/'sp_u4a_trajectories.csv');assert len(rows)==4000
    for c in config()['conditions']:
        for s in (10,20):
            for n in ('50','100','250','500','Original500'):
                for m in ('Source','LR4'):
                    ids=[int(r['trajectory_id']) for r in rows if (r['condition'],int(r['snr_db']),r['budget'],r['method'])==(c,s,n,m)]
                    assert sorted(ids)==list(range(100))


def test_actual_original500_regression():
    if not (OUT/'sp_u4a_trajectories.csv').exists():pytest.skip('actual original500 inference not yet run')
    r={(v['condition'],v['snr_db'],v['trajectory_id'],v['method'],v['budget']):v for v in csvread(OUT/'sp_u4a_trajectories.csv')}
    for k,v in r.items():
        if k[-1]!='500':continue
        ref=r[(*k[:-1],'Original500')]
        assert abs(float(v['nmse_db'])-float(ref['nmse_db']))<1e-8

