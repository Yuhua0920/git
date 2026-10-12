"""Actual SP-U5A engineering prerequisites and independent metric checks."""
import subprocess,sys,inspect
import numpy as np
import pytest
from scripts.sp_u5a_common import *
from src.link import generate_master_qpsk,assemble_pilot_data_frame,apply_frequency_mimo_channel,detect_data_zf,demodulate_data_bits,extract_reference_data_bits
from src.estimators import pilot_ls_estimate

@pytest.mark.parametrize('df,union,pdim,ddim',[(8,64,128,3840),(16,32,64,3968),(32,16,32,4032)])
def test_actual_masks_dimensions(df,union,pdim,ddim):
    g=geometry(df);m=g.layout
    assert m.pilot_union.sum()==union and len(g.selected)==pdim and g.data.sum()==ddim
    assert g.counts['union_ratio']==union/1024
    assert not np.any(m.pilot_mask.sum(axis=0)>1)
    assert np.all(m.pilot_mask.astype(int)+m.mute_mask+m.data_mask==1)
    assert np.array_equal(m.data_mask[0],m.data_mask[1])
    assert m.pilot_mask[0].sum()==m.pilot_mask[1].sum()==union//2
    assert m.mute_mask.sum()==union
    for q in (0,1):
        expected=np.zeros((64,16),bool);expected[np.ix_(np.arange(q*df//2,64,df),[0,4,8,12])]=True
        assert np.array_equal(expected,m.pilot_mask[q])

@pytest.mark.parametrize('df',[8,16,32])
def test_fold_balance_and_exclusivity(df):
    g=geometry(df);fs,definition=folds_for(g);rx,tx,k,t=np.unravel_index(g.selected,(2,2,64,16));union=[]
    for o,h in fs:
        assert len(h)==len(g.selected)//4 and len(o)==3*len(h)
        assert not set(o)&set(h) and set(o)|set(h)==set(range(len(g.selected)))
        for r in (0,1):
            for q in (0,1):assert sum((rx[h]==r)&(tx[h]==q))==len(h)//4
        union.extend(h)
    assert sorted(union)==list(range(len(g.selected)))

def test_dF16_mask_fold_identity():
    from src.spatial_information import observation_geometry
    g=geometry(16);old=observation_geometry()
    assert np.array_equal(g.selected,old.selected) and np.array_equal(g.data,old.data)
    assert folds_for(g)[1]==frozen_folds(old.selected)[1]

def test_dF16_actual_regression_gate():
    q=readjson(OUT/'sp_u5a_regression_gate.json');assert q['passed'] and q['replay_rows']==96
    assert all(q['masks'].values()) and q['maximum_relative_error']<=cfg()['historical_relative_roundoff_gate']

@pytest.mark.parametrize('df',[8,16,32])
def test_self_consistent_artifacts_normal_equations(df):
    g=geometry(df);a=arrays(df);fs,_=folds_for(g);n=len(g.selected);d=g.data.sum()
    with np.load(OUT/f'sp_u5a_statistics_{df}.npz') as st:
        for e in ('Source','P','M'):
            assert sorted(st[e+'_ids'])==list(range(500)) and st[e+'_samples']==5000
            assert st[e+'_rpp'].shape==(n,n) and st[e+'_rdp'].shape==(d,n)
            for s in (10,20):
                sigma=cfg()['noise_variances'][str(s)];assert rel_error(a[f'W{e}_{s}']@(st[e+'_rpp']+sigma*np.eye(n)),st[e+'_rdp'])<1e-11
                for f,(o,h) in enumerate(fs):assert rel_error(a[f'K{e}_{s}'][f]@(st[e+'_rpp'][np.ix_(o,o)]+sigma*np.eye(len(o))),st[e+'_rpp'][np.ix_(h,o)])<1e-11
        for e in ('P','M'):
            for s in (10,20):
                assert a[f'A_{e}_{s}'].shape==(d,4) and a[f'Q_{e}_{s}'].shape==(4,n)
                assert np.array_equal(a[f'DeltaW{e}_{s}'],a[f'W{e}_{s}']-a[f'WSource_{s}'])
                assert np.array_equal(a[f'DeltaK{e}_{s}'],a[f'K{e}_{s}']-a[f'KSource_{s}'])

def test_no_cross_density_mix():
    a=arrays(16);g=geometry(16)
    with pytest.raises(ValueError):cv_model(a,g,10).fit(np.ones((5,32)),modules=('P','M'))
    with pytest.raises(ValueError):RankFactors(a['A_P_10'],a['Q_P_10']).apply(np.ones(128))

def test_frozen_500_builder_not_historical_loader():
    text=inspect.getsource(build_stats)+inspect.getsource(build_ops)
    assert 'np.load' not in text and 'historical_arrays' not in text and 'range(0,500,10)' in text

def test_namespace_and_sampling_configuration():
    assert cfg()['namespace']=='SP_U5A_PILOT_DEV_v1' and cfg()['densities']==[8,16,32]
    assert SYSTEM['sample_rate_hz']==SYSTEM['nfft']*SYSTEM['delta_f_hz']
    tuples={tuple(seed_map(c,i,j).values()) for c in cfg()['conditions'] for i in range(100) for j in range(10)}
    assert len(tuples)==2000
    assert seed_map('ID',0)!=seed_map('Composed',0)

def test_actual_input_gate_no_U4_evaluation():
    q=readjson(OUT/'sp_u5a_input_gate.json');assert q['passed'] and q['distinct_H']==200 and not q['U4_evaluation_read']
    assert q['namespace']==cfg()['namespace']

@pytest.mark.parametrize('df',[8,16,32])
def test_paired_full_grid_fields_and_true_transmission(df):
    g=geometry(df)
    with np.load(OUT/'sp_u5a_inputs/Composed_000.npz') as f:
        j=4;truth=frames(f['H'])[j];bits=f['bits'][j];sy=f['symbols'][j];atom=f['noise_atoms'][j]
        ss=seed_map('Composed',0,j);b,s=generate_master_qpsk(bits_seeds=(ss['bits_seed_tx0'],ss['bits_seed_tx1']))
        assert np.array_equal(bits,b) and np.array_equal(sy,s)
        rng=np.random.default_rng(ss['noise_seed']);expected=(rng.standard_normal((2,64,16))+1j*rng.standard_normal((2,64,16)))/np.sqrt(2)
        assert np.array_equal(atom,expected)
        x=assemble_pilot_data_frame(sy,g.layout);assert np.all(x[g.layout.mute_mask]==0)
        assert np.allclose(abs(x[g.layout.pilot_mask])**2,1,atol=1e-12,rtol=0)
        assert np.allclose(np.sum(abs(x)**2,axis=0),1,atol=1e-12,rtol=0)
        for other in (8,16,32):
            gg=geometry(other);xx=assemble_pilot_data_frame(sy,gg.layout);common=g.layout.data_mask&gg.layout.data_mask
            assert np.array_equal(x[common],xx[common])
        for si,snr in enumerate((10,20)):
            y=apply_frequency_mimo_channel(truth,x)+np.sqrt(cfg()['noise_variances'][str(snr)])*atom
            p,_=pilot_ls_estimate(y,x,g.layout.pilot_mask);assert np.array_equal(p.ravel()[g.selected],f[f'pilots_{df}'][si,j])

@pytest.mark.parametrize('df',[8,16,32])
def test_OLS_real_expansion_and_causality(df):
    g=geometry(df);a=arrays(df);model=cv_model(a,g,10)
    with np.load(OUT/'sp_u5a_inputs/ID_000.npz') as f:pilots=f[f'pilots_{df}'][0,:5]
    c,diag=model.fit(pilots,modules=('P','M'));columns=[];responses=[]
    for p in pilots:
        z=p-a['source_mean_p']
        for fold,(o,h) in enumerate(model.folds):
            columns.append(np.column_stack((a['DeltaKP_10'][fold]@z[o],a['DeltaKM_10'][fold]@z[o])))
            responses.append(z[h]-a['KSource_10'][fold]@z[o])
    d=np.concatenate(columns);r=np.concatenate(responses);dr=np.r_[d.real,d.imag];rr=np.r_[r.real,r.imag]
    oracle=np.linalg.lstsq(dr,rr,rcond=None)[0];assert np.allclose(c,oracle,rtol=1e-11,atol=1e-11)
    assert diag['gram_rank']==2 and diag['window_frames']==5
    with pytest.raises(ValueError):model.fit(np.r_[pilots,pilots[:1]],modules=('P','M'))
    if df==16:
        old=PilotCV(model.folds,model.source_mean_p,model.source_k,model.profile_delta_k,model.mobility_delta_k)
        co,do=old.fit(pilots,modules=('P','M'));assert np.array_equal(c,co) and diag['gram']==do['gram']

@pytest.mark.parametrize('df',[8,16,32])
def test_actual_data_domain_and_BER_counts(df):
    g=geometry(df);bits,sy=generate_master_qpsk(bits_seeds=(111,222));x=assemble_pilot_data_frame(sy,g.layout)
    h=np.broadcast_to(np.eye(2)[:,:,None,None],(2,2,64,16)).astype(complex);y=apply_frequency_mimo_channel(h,x)
    dec=demodulate_data_bits(detect_data_zf(y,h,g.layout.data_mask)*np.sqrt(2),g.layout.data_mask);ref=extract_reference_data_bits(bits,g.layout.data_mask)
    assert all(np.array_equal(a,b) for a,b in zip(dec,ref))
    assert sum(len(b) for b in ref)==2*g.counts['two_stream_data_symbols']==int(g.data.sum())
    truth=h.ravel()[g.data];estimate=truth+1
    assert np.sum(abs(estimate-truth)**2)==g.data.sum()

def test_trajectory_aggregation_energy_not_mean_frame_ratio():
    q=trajectory_metric([1.,9.],[1.,99.]);assert q['nmse_linear']==.1 and q['nmse_linear']!=np.mean([1.,9/99])

@pytest.mark.parametrize('name',['t20b_formal_outcomes.csv','sp_u4a_trajectories.csv','sp_u4b_trajectories.csv','sp_u2c_windows.csv','sp_u3b_mfa_performance.csv'])
def test_formal_and_old_evaluation_firewall(name):
    code=f"from scripts.sp_u5a_common import *;firewall();(OUT/{name!r}).open('rb')"
    p=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True);assert p.returncode!=0 and b'SP-U5A firewall' in p.stderr

def test_historical_write_firewall():
    code="from scripts.sp_u5a_common import *;firewall();(OUT/'t16e_frozen_operators.npz').open('wb')"
    p=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True);assert p.returncode!=0 and b'SP-U5A firewall' in p.stderr

def test_historical_files_unchanged():
    assert all(sha(ROOT/k)==v for k,v in readjson(OUT/'sp_u5a_before_manifest.json').items())

def test_actual_scientific_population_if_present():
    if not (OUT/'sp_u5a_evaluation_gate.json').exists():pytest.skip('scientific scan not run yet')
    q=readjson(OUT/'sp_u5a_evaluation_gate.json');assert q['passed'] and q['new_trajectories']==200 and q['rank_failures']==q['fallback_count']==0
    rr=csvread(OUT/'sp_u5a_trajectories.csv');assert len(rr)==3200
    for c in cfg()['conditions']:
        for key in ('8','16','32','Historical16'):
            for s in (10,20):
                for m in ('Source','LR4'):
                    selected=[r for r in rr if (r['condition'],r['artifact'],r['snr_db'],r['method'])==(c,key,str(s),m)]
                    assert sorted(int(r['trajectory_id']) for r in selected)==list(range(100))

