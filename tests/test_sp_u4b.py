import inspect
import subprocess
import sys
import numpy as np
import pytest
from scripts.sp_u4b_common import *


def test_R0_exact_reuse():
    p=prereg();old=readjson(OUT/'sp_u4a_preregistration.json')
    assert p['permutations']['R0']==old['permutations'] and p['permutation_sha256']['R0']==old['permutation_sha256']


@pytest.mark.parametrize('r',range(1,5))
def test_prescribed_seeds_and_frozen_permutations(r):
    p=prereg();assert p['frozen_before_scientific_result_reads']
    for j,e in enumerate(('Source','P','M'),1):
        s=2026100500+10*r+j;assert p['permutation_seeds'][f'R{r}'][e]==s
        a=np.random.default_rng(s).permutation(500);assert a.tolist()==p['permutations'][f'R{r}'][e]
        assert array_hash(a)==p['permutation_sha256'][f'R{r}'][e]


def test_unique_permutations_and_nested():
    p=prereg();hashes=[]
    for rep,v in p['permutations'].items():
        for e,ids in v.items():
            assert sorted(ids)==list(range(500));assert set(ids[:50])<set(ids[:100])<set(ids[:250])<set(ids)
            hashes.append(p['permutation_sha256'][rep][e])
    assert len(set(hashes))==15


def test_selection_has_no_scientific_readers():
    s=inspect.getsource(freeze);assert 'np.load' not in s and 'trajectories.csv' not in s and 'norm(' not in s


def test_common500_identity():
    paths=[artifact(r,500) for r in prereg()['replicates']]
    assert len(set(paths))==1 and paths[0]==OUT/'sp_u4a_operators_500.npz'


def test_no_new_budget_or500_construction():
    from scripts.run_sp_u4b import prepare
    s=inspect.getsource(prepare);assert 'for n in (50,100,250)' in s
    assert prereg()['new_budgets']==[50,100,250]


def test_artifact_names_isolated():
    assert len({artifact(r,n) for r in ('R1','R2','R3','R4') for n in (50,100,250)})==12


def test_actual_preparation_identity_and_counts():
    for r in ('R1','R2','R3','R4'):
        for n in (50,100,250):
            m=readjson(OUT/f'sp_u4b_construction_{r}_{n}.json');assert m['object_identity']==[r,n] and m['rank']==4
            assert not m['N500_used_to_construct'] and not m['other_replicate_used_to_construct']
            a=load_arrays(r,n,'statistics')
            for e in ('Source','P','M'):
                assert a[e+'_ids'].tolist()==prereg()['permutations'][r][e][:n] and a[e+'_samples']==10*n


@pytest.mark.parametrize('rep',['R1','R2','R3','R4'])
def test_operator_construction_identities(rep):
    n=50;a=load_arrays(rep,n);stats=load_arrays(rep,n,'statistics')
    for s in (10,20):
        for e in ('Source','P','M'):
            np.testing.assert_allclose(a[f'W{e}_{s}']@(stats[f'{e}_rpp']+prereg()['config']['noise_variances'][str(s)]*np.eye(64)),stats[f'{e}_rdp'],rtol=1e-10,atol=1e-10)
        for e in ('P','M'):
            assert np.array_equal(a[f'DeltaW{e}_{s}'],a[f'W{e}_{s}']-a[f'WSource_{s}'])
            assert np.array_equal(a[f'DeltaK{e}_{s}'],a[f'K{e}_{s}']-a[f'KSource_{s}'])
            assert a[f'A_{e}_{s}'].shape==(3968,4) and a[f'Q_{e}_{s}'].shape==(4,64)


def test_exact_input_replay():
    g=readjson(OUT/'sp_u4b_input_replay_gate.json');assert g['passed'] and g['new_channel_generations']==0
    assert g['all_H_and_communication_digests_exact'] and g['all4000_frame_snr_pilots_exact']


def test_pairing_no_generation_inside_evaluator():
    from scripts.run_sp_u4b import evaluate
    s=inspect.getsource(evaluate)
    assert '.generate(' not in s and 'default_rng' not in s and "pilots=f['pilots']" in s


def test_trajectory_metrics():
    assert trajectory_metric([1,9],[1,99])['nmse_linear']==.1


def test_deterministic_tail_and_sign():
    t=signed_tail([.2,-.3,.5,.5,.1],[8,9,10,11,12])
    assert t['top5_positive_ids']==[10,11,8,12] and t['maximum_positive_penalty']==.5
    assert signed_tail([-1,-2],[1,2])['p95_positive_penalty'] is None


def test_firewall_formal_u2_mfa_and_old_writes():
    for filename,mode in [('t20b_formal_outcomes.csv','rb'),('sp_u2b_analysis.json','rb'),('sp_u3a_baseline_gate.json','rb'),('sp_u4a_operators_500.npz','wb')]:
        code=f"from scripts.sp_u4b_common import *;firewall();(OUT/'{filename}').open('{mode}')"
        p=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        assert p.returncode!=0 and 'firewall blocked' in p.stderr


def test_original_sources_and_U4A_unchanged():
    b=readjson(OUT/'sp_u4b_before_manifest.json')
    for p,h in {**b['protected'],**b['U4A']['source_sha256'],**b['U4A']['artifact_sha256']}.items():assert sha(ROOT/p)==h


def test_actual_R0_raw_reuse_and_common_reference():
    p=OUT/'sp_u4b_trajectories.csv'
    if not p.exists():pytest.skip('new replicate inference not run yet')
    new={(r['replicate'],r['condition'],r['snr_db'],r['budget'],r['method'],r['trajectory_id']):r for r in csvread(p)}
    for r in csvread(OUT/'sp_u4a_trajectories.csv'):
        if r['budget']=='Original500':continue
        rep='Common500' if r['budget']=='500' else 'R0';v=new[rep,r['condition'],r['snr_db'],r['budget'],r['method'],r['trajectory_id']]
        assert all(v[k]==x for k,x in r.items())
    assert len(new)==12800


def test_actual_replicate_level_no_pooling():
    if not (OUT/'sp_u4b_summary.csv').exists():pytest.skip('replicate analysis pending')
    rows=csvread(OUT/'sp_u4b_summary.csv');assert len(rows)==160
    assert all(int(r['trajectories'])==100 for r in rows)
