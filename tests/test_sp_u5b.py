"""Measurement-control invariants, independent sentinels and actual frozen replay."""
import ast,inspect,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pytest
from scripts.sp_u5b_common import *
from scripts.run_sp_u5b import replay_trajectory,indexes

@pytest.fixture(scope='module')
def domains():return common_domain()

@pytest.mark.parametrize('df',[8,16,32])
def test_actual_mask_intersection_and_membership(domains,df):
    gs,m,full,ids,pos=domains
    expected=np.logical_and.reduce([gs[d].layout.data_mask for d in (8,16,32)])
    assert np.array_equal(m,expected) and np.array_equal(m,gs[8].layout.data_mask)
    assert np.all(m<=gs[df].layout.data_mask) and np.all(m[0]==m[1])
    assert m[0].sum()==960 and full.sum()==3840 and m.sum()*2==3840

@pytest.mark.parametrize('df',[8,16,32])
def test_same_truth_indices_and_target_mapping(domains,df):
    gs,m,full,ids,pos=domains;truth=np.arange(4096)+1j*np.arange(4096)[::-1]
    assert np.array_equal(truth[gs[df].data][pos[df]],truth[ids])
    assert pos[df].size==3840

@pytest.mark.parametrize('df',[8,16,32])
def test_same_saved_bits_symbols_and_noise(domains,df):
    from src.link import assemble_pilot_data_frame,extract_reference_data_bits
    gs,m,full,ids,pos=domains
    with np.load(OUT/'sp_u5a_inputs/ID_000.npz',allow_pickle=False) as f:
        x8=assemble_pilot_data_frame(f['symbols'][4],gs[8].layout);xx=assemble_pilot_data_frame(f['symbols'][4],gs[df].layout)
        assert np.array_equal(x8[m],xx[m])
        b=extract_reference_data_bits(f['bits'][4],m)
        assert sum(len(v) for v in b)==3840
        # Every density indexes the same stored full-grid field with the identical RE mask.
        common_ids=np.flatnonzero(m[0].ravel());native_ids=np.flatnonzero(gs[df].layout.data_mask[0].ravel())
        mapped=np.searchsorted(native_ids,common_ids);noise=f['noise_atoms'][4].reshape(2,-1)
        assert np.array_equal(noise[:,native_ids][:,mapped],noise[:,common_ids])

def test_independent_energy_aggregation_sentinel():
    r=trajectory_metric([1,9],[2,98])
    assert r['nmse_linear']==.1
    assert r['nmse_db']==-10
    assert r['nmse_linear']!=np.mean([1/2,9/98])

def test_no_random_no_refit_no_rebuild_calls():
    forbidden={'fit','build_ops','build_stats','build_fdm_pilot_layout','generate','generate_master_qpsk','default_rng'}
    for n in ('sp_u5b_common.py','run_sp_u5b.py'):
        tree=ast.parse((ROOT/'scripts'/n).read_text(encoding='utf8'))
        names=[c.func.id if isinstance(c.func,ast.Name) else c.func.attr if isinstance(c.func,ast.Attribute) else '' for c in ast.walk(tree) if isinstance(c,ast.Call)]
        assert not set(names)&forbidden

@pytest.mark.parametrize('condition',['ID','Composed'])
def test_actual_deterministic_native_replay_and_no_scientific_mutation(condition,domains):
    block_scientific_mutations()
    gs,m,full,ids,pos=domains;aa={df:operators(df) for df in (8,16,32)};fi,ti,hi=indexes();coords=coordinate_index()
    rr=replay_trajectory(condition,0,aa,gs,m,ids,pos,coords,fi,ti,hi)
    ff,tt,checks,reuse,pa=rr
    assert len(ff)==144 and len(tt)==24 and len(reuse)==36
    assert all(r['max_energy_difference']==r['max_nmse_difference']==0 for r in checks)
    assert all(r['BER_counts_exact'] for r in checks)
    assert pa[0]['common_received_exact'] and pa[0]['stored_full_grid_digests_exact']
    for r in reuse:
        old=coords[condition,0,r['snr_db'],r['dF'],r['frame_id']]
        assert r['alpha']==float(old['alpha']) and r['beta']==float(old['beta']) and not r['refit']
    ix={(r['snr_db'],r['dF'],r['method'],r['domain']):r for r in tt}
    for s in (10,20):
        for method in ('Source','LR4'):
            n=ix[s,8,method,'native'];c=ix[s,8,method,'common']
            assert all(n[k]==c[k] for k in ('error_energy','truth_energy','nmse_linear','nmse_db','bit_errors','bit_count','BER'))
    assert all(r['bit_count']==3840 for r in ff if r['domain']=='common')

@pytest.mark.parametrize('name',['t20b_results.json','sp_u4a_trajectories.csv','sp_u4b_trajectories.csv','sp_u2c_trajectories.csv'])
def test_formal_and_unrelated_outcome_firewall(name):
    firewall()
    with pytest.raises(PermissionError):(OUT/name).open('rb')

def test_U5A_write_firewall():
    firewall()
    with pytest.raises(PermissionError):(OUT/'sp_u5a_operators_16.npz').open('wb')

def test_protected_historical_sources_and_artifacts():
    assert protect()>600

def test_frozen_sample_rate_and_protocol():
    p=prereg();assert p['frozen_before_common_results']
    assert SYSTEM['sample_rate_hz']==SYSTEM['nfft']*SYSTEM['delta_f_hz']
    assert p['config']['new_random_inputs'] is False and p['config']['operator_rebuild'] is False

def test_full_actual_population_and_native_identity():
    p=OUT/'sp_u5b_replay_gate.json'
    if not p.exists():pytest.skip('post-run control population is not available before replay')
    g=readjson(p);assert g['passed'] and g['native_max_metric_difference']==0 and g['native_BER_counts_exact']
    rows=csvread(OUT/'sp_u5b_trajectories.csv');assert len(rows)==4800
    assert len({(r['condition'],r['trajectory_id']) for r in rows})==200
    cc=[r for r in rows if r['domain']=='common'];assert len(cc)==2400 and all(int(r['bit_count'])==23040 for r in cc)
    coords=coordinate_index();reuse=csvread(OUT/'sp_u5b_coordinate_reuse.csv');assert len(reuse)==7200
    for r in reuse:
        old=coords[r['condition'],int(r['trajectory_id']),int(r['snr_db']),int(r['dF']),int(r['frame_id'])]
        assert float(r['alpha'])==float(old['alpha']) and float(r['beta'])==float(old['beta']) and r['refit']=='False'

