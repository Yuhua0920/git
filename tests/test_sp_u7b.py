"""U7B wrapper tests use synthetic data exclusively; CDL coefficient calls forbidden."""
import ast
import copy
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from scripts import sp_u7b_common as c
from scripts import run_sp_u7b as runner
from scripts import analyze_sp_u7b as analysis
from scripts import sp_u7a_statistics as stats
from src.cdl_benchmark import frames,StreamingStatistics,observation_geometry
from src.pilot_cv import frozen_folds,deployment
from src.lowrank_modules import factored_affine_predict

@pytest.fixture(autouse=True)
def no_scientific_CDL(monkeypatch):
    # No real library coefficient generation in any wrapper test, even one sample.
    from sionna.phy.channel.tr38901 import CDL
    def forbidden(*args,**kwargs):raise AssertionError('scientific CDL generation forbidden in Phase A tests')
    monkeypatch.setattr(CDL,'__call__',forbidden)

@pytest.fixture(scope='module')
def frozen_objects():return c.load_deployables()

def fake_proof(i=0):
    inputs=[dict(snr_db=s,frame_id=j,**{k:'a'*64 for k in ('H_sha256','bits_sha256','symbols_sha256','unit_noise_sha256','Y_sha256','pilot_sha256')}) for s in c.SNRS for j in range(10)]
    outputs=[dict(snr_db=s,frame_id=j,method=m,prediction_sha256='b'*64,shared_Y_sha256='a'*64,shared_pilot_sha256='a'*64) for s in c.SNRS for j in range(4,10) for m in c.METHODS]
    windows=[dict(snr_db=s,frame_id=j,frames=list(range(j-4,j+1)),rank_P=4,rank_M=4,clipped=False,regularized=False,solve_success=True,dense_lr4_same_c=True) for s in c.SNRS for j in range(4,10)]
    return dict(trajectory_id=i,group='EVAL',frames=10,states=160,coefficient_calls=1,counts=dict(trajectory_metrics=14,frame_metrics=84,inputs=20,coordinates=12),
       inputs=inputs,outputs=outputs,windows=windows,target_clean_CSI_passed_to_deployables=False,matched_reference_passed_to_deployables=False,required_finite=True,paired_inputs_immutable=True)

def fake_reference(i=0):return dict(trajectory_id=i,group='MATCHED_REF',reference_only=True,finite=True,coefficient_calls=1,frames=10,states=160)

def test_seed_manifest_unique_nohistory():
    result=c.seed_audit();assert result['rows']==34501;assert result['historical_overlap_count']==0

def test_seed_role_consumption():
    index=c.seed_index();a=c.consumed(index,'EVAL',0,0);b=c.consumed(index,'EVAL',0,1);r=c.consumed(index,'MATCHED_REF',0)
    assert set(a)==set(c.ROLES);assert set(r)==set(c.ROLES[:3])
    assert all(a[k]==b[k] for k in c.ROLES[:3]);assert all(a[k]!=b[k] for k in c.ROLES[3:]);assert set(a.values()).isdisjoint(r.values())

def test_sionna_rng_derivation_source():
    source=(c.ROOT/'.venv-cdl/Lib/site-packages/sionna/phy/config.py').read_text(encoding='utf8')
    assert 'device_index' in source and 'manual_seed' in source
    assert 'sionna_seed' in c.seed_audit()['rng_roles']

def test_frozen_U7A_byte_registry():c.verify(c.read(c.ROOT/'outputs/sp_u7a_protected_inputs.json')['hashes'])

def test_protocol_exact():
    p=c.read(c.PROTOCOL);d=p['condition'];assert d['sample_rate_hz']==d['K']*d['spacing_hz'];assert d['profile']=='CDL-A';assert d['speed_kmh']==120
    assert p['evaluation']['methods']==list(c.METHODS);assert p['statistics']['bootstrap']['seed']==764389531

def test_constructor_metadata_no_coefficient_call():
    adapter=runner.SealedCDL('MATCHED_REF',c.seed_index());m=adapter.metadata();assert not adapter.calls
    assert m['states_per_call']==160 and m['profile']=='CDL-A' and m['normalize_H'] is False

def test_original_generator_body_and_continuous_call(monkeypatch):
    adapter=runner.SealedCDL('EVAL',c.seed_index());seen=[]
    def fake(self,i):
        from src.cdl_benchmark import seeds
        seen.append(seeds(i));return np.zeros((2,2,64,160),complex)
    monkeypatch.setattr(runner.OfficialCDL,'generate',fake)
    assert adapter.generate(0).shape==(2,2,64,160)
    assert seen[0]==c.consumed(c.seed_index(),'EVAL',0)
    with pytest.raises(ValueError):adapter.generate(0)
    assert len(seen)==1

def test_state_mapping():
    h=np.arange(2*2*64*160).reshape(2,2,64,160)
    f=frames(h)
    for j in range(10):assert np.array_equal(f[j],h[:,:,:,j*16:(j+1)*16])

def test_original_communication_semantics():
    from src.link import generate_master_qpsk
    from src.numerics import complex_gaussian_noise
    s=dict(bits_seed_tx0=42,bits_seed_tx1=43,noise_seed=44);bits,sym,z=c.communication(s)
    b,x=generate_master_qpsk(bits_seeds=(42,43));n=complex_gaussian_noise((2,64,16),1,np.random.default_rng(44))
    assert np.array_equal(bits,b) and np.array_equal(sym,x) and np.array_equal(z,n)
    assert not bits.flags.writeable and not z.flags.writeable

def test_reference_welford_population_semantics():
    g=observation_geometry();idx=np.arange(20*4096).reshape(20,4096);x=np.sin(idx*.021)+1j*np.cos(idx*.033)
    acc=StreamingStatistics(g.selected,g.data);acc.update(x[:10],[0]);acc.update(x[10:],[1]);f=acc.freeze()
    xp=x[:,g.selected];xd=x[:,g.data];pc=xp-xp.mean(0);dc=xd-xd.mean(0)
    np.testing.assert_allclose(f.rpp,pc.T@pc.conj()/20,rtol=1e-12,atol=1e-13)
    np.testing.assert_allclose(f.rdp,dc.T@pc.conj()/20,rtol=1e-12,atol=1e-13)
    assert f.samples==20 and f.train_trajectory_ids==(0,1)

def test_pilot_only_boundary_and_reference_taint(frozen_objects):
    arrays,factors,cvs=frozen_objects
    i=np.arange(640).reshape(10,64);p=np.sin(i*.013)+1j*np.cos(i*.057)
    before={k:c.sha(c.ROOT/k) for k in c.read(c.PROTOCOL)['deployable']['objects']}
    first=c.infer_pilots(p,arrays,factors,cvs,10)
    # Arbitrary reference/truth objects cannot enter the pilot-only call.
    reference_a=np.full(4096,1e9);reference_b=np.full(4096,-1e9)
    second=c.infer_pilots(p.copy(),arrays,factors,cvs,10)
    for a,b in zip(first,second):
        for k in a['predictions']:assert np.array_equal(a['predictions'][k],b['predictions'][k])
    with pytest.raises(TypeError):c.infer_pilots(p,arrays,factors,cvs,10,reference_a)
    assert all(c.sha(c.ROOT/k)==v for k,v in before.items())
    assert 'matched' not in inspect.signature(c.infer_pilots).parameters and 'truth' not in inspect.signature(c.infer_pilots).parameters

def test_dense_lr4_exact_coordinate_and_frozen_prediction(frozen_objects):
    a,f,cv=frozen_objects;i=np.arange(640).reshape(10,64);p=np.sin(i*.015)+1j*np.cos(i*.041)
    for s in c.SNRS:
        rows=c.infer_pilots(p,a,f,cv,s)
        assert [r['frame_id'] for r in rows]==list(range(4,10))
        for r in rows:
            coeff,diag=cv[s].fit(p[r['frame_id']-4:r['frame_id']+1],modules=('P','M'))
            assert r['dense_coefficients'] is r['lr4_coefficients'];assert np.array_equal(coeff,r['dense_coefficients'])
            args=(p[r['frame_id']],a['source_mean_p'],a['source_mean_d'],a[f'Wsource_{s}'])
            assert np.array_equal(r['predictions']['dense-affine'],deployment(*args,a[f'DeltaWP_{s}'],a[f'DeltaWM_{s}'],coeff,('P','M')))
            assert np.array_equal(r['predictions']['lr4-affine'],factored_affine_predict(*args,coeff,f['P',s,4],f['M',s,4]))

def test_normal_equation_solve_failure_no_fallback(monkeypatch,frozen_objects):
    a,f,cv=frozen_objects
    def fail(*args,**kwargs):raise np.linalg.LinAlgError('synthetic singular')
    monkeypatch.setattr(np.linalg,'solve',fail)
    with pytest.raises(np.linalg.LinAlgError):c.infer_pilots(np.ones((10,64),complex),a,f,cv,10)

def test_factor_rank_prefix_byte_exact(frozen_objects):
    a,f,cv=frozen_objects
    with np.load(c.ROOT/'outputs/t16g_frozen_module_factors.npz',allow_pickle=False) as z:
        for s in c.SNRS:
            for m in ('P','M'):
                assert f[m,s,4].rank==4;assert np.array_equal(f[m,s,4].A,z[f'A_{m}_{s}'][:,:4]);assert np.array_equal(f[m,s,4].Q,z[f'Q_{m}_{s}'][:4,:])

def test_structural_exact1000_complete():assert runner.structural_validate([fake_proof(i) for i in range(1000)],[fake_reference(i) for i in range(500)])

@pytest.mark.parametrize('mutation',['missing','count','future','taint','reference','rank','clip','nonfinite','pairing','solve','same_c'])
def test_integrity_mutation_rejected(mutation):
    p=fake_proof();r=fake_reference()
    if mutation=='missing':p['trajectory_id']=1
    elif mutation=='count':p['counts']['frame_metrics']=83
    elif mutation=='future':p['windows'][0]['frames']=[0,1,2,3,5]
    elif mutation=='taint':p['target_clean_CSI_passed_to_deployables']=True
    elif mutation=='reference':p['matched_reference_passed_to_deployables']=True
    elif mutation=='rank':p['windows'][0]['rank_P']=8
    elif mutation=='clip':p['windows'][0]['clipped']=True
    elif mutation=='nonfinite':p['required_finite']=False
    elif mutation=='pairing':p['outputs'][0]['shared_Y_sha256']='z'*64
    elif mutation=='solve':p['windows'][0]['solve_success']=False
    elif mutation=='same_c':p['windows'][0]['dense_lr4_same_c']=False
    with pytest.raises(ValueError):runner.structural_validate([p],[r],1,1)

def test_integrity_source_no_numeric_columns():
    source=inspect.getsource(runner.integrity)
    assert 'np.load=' in source and 'numeric_outcome_columns_read=False' in source
    for token in ("['error']","['bit_errors']","['coordinates']","['power']"):assert token not in source

def test_archive_header_hash_without_numeric_load(tmp_path,monkeypatch):
    path=tmp_path/'raw.npz';np.savez(path,error=np.ones((2,7)),coordinates=np.zeros((12,16)))
    monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('numeric load forbidden')))
    assert runner.headers(path)['error']==dict(shape=[2,7],dtype='float64');assert len(c.sha(path))==64

def test_pooled_energy_not_mean_db():
    e=np.ones((3,2,7))*np.array([1.,100.,4.])[:,None,None];q=np.ones((3,2))*np.array([1.,2.,4.])[:,None]
    actual=stats.reduce_nmse(e,q);expected=10*np.log10(105/7)
    assert np.all(actual==expected);assert not np.isclose(actual[0,0],np.mean(10*np.log10(e[:,0,0]/q[:,0])))

def test_bootstrap_saved_before_use_exact_pairing(tmp_path):
    n=5;e=np.arange(n*2*7).reshape(n,2,7)+1.;q=np.arange(n*2).reshape(n,2)+2.;be=np.ones_like(e,dtype=np.int64);nb=np.full((n,2),20)
    proof=analysis.draw_table(tmp_path/'draws.npy',n,100,764389531)
    result=stats.paired_bootstrap(e,q,be,nb,764389531,replicates=100,expected_n=n)
    assert proof['saved_before_first_use'] and proof['array_sha256']==result['indices_sha256']
    draws=np.load(tmp_path/'draws.npy');db=10*np.log10(e[draws].sum(1)/q[draws].sum(1)[...,None]);gain=db[:,:,1]-db[:,:,5]
    np.testing.assert_array_equal(result['primary_lcb'],np.quantile(gain,.025,axis=0,method='linear'))

@pytest.mark.parametrize('point,lo,hi,expected',[([1,2],[.1,.2],[2,3],'A'),([1,-1],[-1,-2],[2,0],'C'),([1,1],[-.1,.2],[2,3],'B')])
def test_case_order_bonferroni(point,lo,hi,expected):assert stats.decision(point,lo,hi).startswith(expected);assert stats.PRIMARY_QUANTILE==.025

def test_unseal_denied_before_gate(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path)
    c.write(tmp_path/'state.json',dict(phase='B_RAW_COMPLETE',UNSEAL=False))
    monkeypatch.setattr(c,'verify_frozen',lambda:dict(hashes={}))
    c.write(tmp_path/'unseal.json',dict(UNSEAL=False))
    with pytest.raises(RuntimeError):analysis.assert_unseal()

def test_state_machine_no_resume_after_invalidation(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path);c.write(tmp_path/'state.json',dict(phase='INVALIDATED'))
    with pytest.raises(RuntimeError):c.transition('A_FROZEN','B_REFERENCE_STARTED')

def test_byte_drift_rejected(tmp_path):
    p=tmp_path/'sentinel.txt';p.write_text('a',encoding='utf8');h=c.sha(p);p.write_text('b',encoding='utf8')
    with pytest.raises(RuntimeError):c.verify({str(p):h})

def test_firewall_blocks_outcome_read_subprocess():
    code="from scripts import sp_u7b_common as c; c.guard(); (c.ROOT/'outputs/t20b_results.json').read_text()"
    result=subprocess.run([sys.executable,'-c',code],cwd=c.ROOT,capture_output=True,text=True)
    assert result.returncode!=0 and 'historical outcome firewall' in result.stderr

def test_method_helpers_remain_production_unmodified():
    tree=ast.parse(inspect.getsource(c));names={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
    assert 'svd' not in names and 'pinv' not in names
    assert c.METHODS==stats.METHODS and c.SNRS==stats.SNRS

def test_full_synthetic_receiver_truth_and_reference_isolation(frozen_objects):
    arrays,factors,cvs=frozen_objects;g=observation_geometry()
    idx=np.arange(2*2*64*160).reshape(2,2,64,160)
    h=(np.sin(idx*.017)+1j*np.cos(idx*.019)).astype(np.complex128)
    comm=[c.communication(dict(bits_seed_tx0=42+j*3,bits_seed_tx1=43+j*3,noise_seed=44+j*3)) for j in range(10)]
    class Reference:
        def __init__(self,offset):self.offset=offset
        def estimate(self,p):return arrays['source_mean_d']+(p-arrays['source_mean_p'])@arrays['Wsource_10'].T+self.offset
    first,p=c.evaluate(h,comm,arrays,factors,cvs,{s:Reference(.1) for s in c.SNRS})
    altered=h.copy();views=frames(altered)
    for j in range(10):
        # Data-only clean truth perturbation leaves every pilot received atom exact.
        flat=views[j].copy().ravel();flat[g.data]+=.2;views[j][:]=flat.reshape(2,2,64,16)
    second,p2=c.evaluate(altered,comm,arrays,factors,cvs,{s:Reference(.3) for s in c.SNRS})
    assert len(p['outputs'])==84 and len(p['inputs'])==20 and len(p['windows'])==12
    assert p['required_finite'] and not p['target_clean_CSI_passed_to_deployables']
    for a,b in zip(p['outputs'],p2['outputs']):
        if a['method']!='Matched':assert a['prediction_sha256']==b['prediction_sha256']
    assert any(a['prediction_sha256']!=b['prediction_sha256'] for a,b in zip(p['outputs'],p2['outputs']) if a['method']=='Matched')
    for key,frame_key in (('error','frame_error'),('truth','frame_truth'),('bit_errors','frame_bit_errors'),('total_bits','frame_total_bits')):
        expected=np.zeros_like(first[key])
        for j in range(6):expected+=first[frame_key][...,j]
        assert np.array_equal(first[key],expected)
    assert np.all(first['total_bits']==23808) and np.all(first['frame_total_bits']==3968)
    for j in range(10):
        a=next(r for r in p['inputs'] if r['snr_db']==10 and r['frame_id']==j)
        b=next(r for r in p['inputs'] if r['snr_db']==20 and r['frame_id']==j)
        assert a['unit_noise_sha256']==b['unit_noise_sha256'] and a['bits_sha256']==b['bits_sha256']

@pytest.mark.parametrize('value',[0.,np.nan,-1.])
def test_invalid_energy_no_floor_no_filter(value):
    e=np.ones((5,2,7));q=np.ones((5,2));e[0,0,0]=value
    with pytest.raises(ValueError):stats.reduce_nmse(e,q)
