"""Engineering and preservation gates for the post-validation receiver audit."""
import inspect
from pathlib import Path
import numpy as np
import pytest
from scripts import sp_u8a_common as c

@pytest.fixture(scope='module')
def fixtures():return c.fixtures()

@pytest.mark.parametrize('name',['identity','diagonal','full_complex'])
def test_algebra(fixtures,name):
    row=next(r for r in fixtures['algebra'] if r['fixture']==name)
    assert row['bit_errors']==0 and row['max_symbol_residual']<1e-12

@pytest.mark.parametrize('name',['TX_swap','RX_swap','H_transpose','H_Hermitian','KT_swap'])
def test_ordering_sentinel(fixtures,name):assert fixtures['ordering_mutation_bit_errors'][name]>0

def test_noise_scaling(fixtures):
    for name in ('identity','diagonal'):
        rows=[r for r in fixtures['noise_sweep'] if r['fixture']==name]
        assert rows[0]['BER']>rows[1]['BER']>=rows[2]['BER']
        assert np.allclose([r['standardized_atom_energy'] for r in rows],1.,rtol=0,atol=1e-14)
        assert rows[0]['variance']/rows[1]['variance']==pytest.approx(10.)

def test_perfect_H_identity_and_saved_inputs():
    h,comm,raw,proof=c.ingredients(0)
    assert c.array_hash(h[0])==proof['inputs'][0]['H_sha256']
    assert c.array_hash(comm[0][0])==proof['inputs'][0]['bits_sha256']
    assert c.array_hash(comm[0][2])==proof['inputs'][0]['unit_noise_sha256']

def test_data_mask_and_bitcount_identity():
    g=c.observation_geometry()
    assert np.array_equal(g.layout.data_mask[0],g.layout.data_mask[1])
    assert g.layout.data_mask[0].sum()==992
    selected=np.broadcast_to(g.layout.data_mask[None],(2,2,64,16))
    assert np.array_equal(selected.ravel(),g.data)
    _,comm,raw,_=c.ingredients(0)
    assert sum(v.size for v in c.extract_reference_data_bits(comm[4][0],g.layout.data_mask))==3968
    assert np.all(raw['frame_total_bits']==3968) and np.all(raw['total_bits']==23808)

def test_frozen_BER_replay_one_saved_frame():
    c.install_guard();a,f,matched=c.load_methods();g=c.observation_geometry()
    hs,comm,raw,proof=c.ingredients(0);j=4;bits,sym,unit=comm[j]
    clean=c.apply_frequency_mimo_channel(hs[j],c.assemble_pilot_data_frame(sym,g.layout))
    ref=c.extract_reference_data_bits(bits,g.layout.data_mask)
    for si,s in enumerate(c.SNRS):
        y=clean+np.sqrt(c.VARIANCES[s])*unit
        sparse,linear=c.estimate_ls_linear_channel(y,g.layout);pilot=sparse.ravel()[g.selected]
        pred,grids=c.replay_prediction(pilot,linear,sparse,s,raw['coordinates'][si*6,2:4],a,f,matched)
        for m in c.METHODS[1:]:
            witness=next(v for v in proof['outputs'] if v['snr_db']==s and v['frame_id']==j and v['method']==m)
            assert c.array_hash(pred[m])==witness['prediction_sha256']
            be,_=c.count_bits(y,grids[m],g.layout.data_mask,ref)
            assert be==raw['frame_bit_errors'][si,c.frozen.METHODS.index(m),0]

def test_no_communication_regeneration_or_refit():
    c.install_guard()
    with pytest.raises(RuntimeError):c.frozen.communication({})
    with pytest.raises(RuntimeError):c.frozen.infer_pilots(None,None,None,None,10)

def test_no_OOD_seed_use_and_history_firewall():
    c.install_guard()
    with pytest.raises(RuntimeError):np.random.default_rng(1)
    with pytest.raises(PermissionError):(c.R1/'seed_manifest.csv').read_text()
    with pytest.raises(PermissionError):(c.ROOT/'outputs/sp_u7b/analysis.json').read_text()

def test_primary_analysis_files_unchanged():
    before=c.read(c.OUT/'before.json')['hashes']
    for name in ('analysis.json','table_A.csv','table_B.csv','unseal.json','state.json'):
        rel='outputs/sp_u7b_r1/'+name
        assert c.sha(c.ROOT/rel)==before[rel]

def test_full_replay_if_complete():
    if not (c.OUT/'replay_gate.json').exists():return
    gate=c.read(c.OUT/'replay_gate.json')
    assert gate['PASS'] and gate['prediction_hash_comparisons']==48000
    assert gate['max_bit_count_error']==0 and gate['max_aggregate_BER_error']==0
    assert gate['total_bits']==[23808000,23808000]
    assert all(v==0 for v in gate['no_regeneration'].values())

def test_source_replay_has_no_fitting():
    code=inspect.getsource(c.replay_prediction)
    assert '.fit(' not in code and 'default_rng' not in code

def test_full_conditioning_if_complete():
    if not (c.OUT/'conditioning_summary.json').exists():return
    s=c.read(c.OUT/'conditioning_summary.json')
    assert s['REs']==5952000 and s['singular_REs']==0 and s['nonfinite_values']==0
    assert s['mismatch_identity_relative_energy_residual_max']<1e-10
