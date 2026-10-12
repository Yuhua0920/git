"""Synthetic fixtures only; no stored scientific pilots/results/models."""
import copy,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import base as b
import science as s
import paired_stats as st
def test_contract_grid():
    p=b.contract();assert tuple(p['conditions'])==st.CONDITIONS and tuple(p['snrs'])==st.SNRS
    assert p['N']*len(p['score_frames'])*len(p['snrs'])*len(p['conditions'])==210000
    assert p['statistics']['tail_probability']==.05/70
def test_top_manifest_binding():
    # The hash is the independently checked PERSONAL top manifest, not NPY claims.
    assert b.contract()['final_manifest_sha256']=='8fc8793053c45a00e17e77328bac456670fd088c45916609c3ba0f0515b7b475'
@pytest.mark.parametrize('name',['../escape','C:/absolute','../../B2'])
def test_path_escape(name,tmp_path):
    with pytest.raises(RuntimeError):b.safe(tmp_path,name)
def test_pooled_not_average_ratios():
    e=np.array([1.,9.]);t=np.array([1.,3.]);assert st.pooled(e,t)==2.5;assert st.pooled(e,t)!=np.mean(e/t)
@pytest.mark.parametrize('e,t',[(np.array([np.nan]),np.ones(1)),(np.ones(1),np.zeros(1)),(-np.ones(1),np.ones(1))])
def test_invalid_energy(e,t):
    with pytest.raises(RuntimeError):st.pooled(e,t)
def test_paired_bootstrap_sign_and_quantile():
    t=np.arange(1,4,dtype=float)[:,None]*np.ones((1,7));e=np.stack((2*t,4*t),axis=2)
    rows,draws=st.summarize(e,t,0)
    assert draws.shape==(100000,7)
    assert np.allclose(draws,10*np.log10(2),rtol=0,atol=1e-14)
    assert all(r['classification']=='LR4_SUPPORTED' for r in rows)
    e=e[:,:,::-1];rows,_=st.summarize(e,t,0);assert all(r['classification']=='MFA_SUPPORTED' for r in rows)
def test_bootstrap_original_index_stream():
    t=np.ones((3,7));x=np.arange(21,dtype=float).reshape(3,7)+1;e=np.stack((x,x*x),axis=2)
    _,draws=st.summarize(e,t,2)
    rng=np.random.default_rng(np.random.SeedSequence([2997555629,2]));ids=rng.integers(0,3,(1000,3),dtype=np.int32)
    for j in range(10):
        num=e[ids[j]].sum(axis=0);db=10*np.log10(num/3)
        assert np.allclose(draws[j],db[:,1]-db[:,0],rtol=0,atol=1e-14)
def test_no_weak_bootstrap():
    with pytest.raises(RuntimeError):st.summarize(np.ones((3,7,2)),np.ones((3,7)),0,99999)
def test_null_intervals():
    rows,_=st.summarize(np.ones((2,7,2)),np.ones((2,7)),4)
    assert all(r['classification']=='INDETERMINATE' and r['MFA_minus_LR4_dB']==0 for r in rows)
def proof_fixture():
    cfg={'fold_hash':'fold','pilot_geometry_hash':'geometry','operator_cache_sha256':'operator'}
    ix={('ID',0,-1,r):k+1 for k,r in enumerate(s.ROLES[:3])}
    ix.update({('ID',0,j,r):100+j*3+k for j in range(10) for k,r in enumerate(s.ROLES[3:])})
    frames=[{r:ix['ID',0,-1 if r in s.ROLES[:3] else j,r] for r in s.ROLES} for j in range(10)]
    proof={'condition':'ID','trajectory_id':0,'group':'EVAL','snr_db':list(st.SNRS),'estimators':b.contract()['B2_methods'],'scored_frames':list(range(4,10)),
       'rank':4,'window_frames':5,'fold_hash':'fold','pilot_geometry_hash':'geometry','operator_sha256':'operator',
       **{k:True for k in ('method_pairing','CRN_pairing','dense_lr4_shared_coordinates','finite_estimates','valid_coordinate_solve')},
       'channel_seeds':{r:ix['ID',0,-1 if r in s.ROLES[:3] else 0,r] for r in s.ROLES},'communication_seeds':frames,
       'inputs':[{'snr_db':snr,'frame_id':j,'truth_sha256':'truth','unit_noise_sha256':'noise','pilot_sha256':'pilot','Y_sha256':'y'} for snr in st.SNRS for j in range(10)]}
    return proof,cfg,ix
def test_identity_grid():
    proof,cfg,ix=proof_fixture();assert len(s.proof_check(proof,'ID',0,cfg,ix))==70
@pytest.mark.parametrize('field,value',[('trajectory_id',1),('condition','Boundary'),('rank',8),('window_frames',4),('valid_coordinate_solve',False),('fold_hash','bad')])
def test_identity_mismatch(field,value):
    p,c,ix=proof_fixture();p[field]=value
    with pytest.raises(RuntimeError):s.proof_check(p,'ID',0,c,ix)
def test_duplicate_input_identity():
    p,c,ix=proof_fixture();p['inputs'][-1]=p['inputs'][0]
    with pytest.raises(RuntimeError):s.proof_check(p,'ID',0,c,ix)
def test_noise_seed_mismatch():
    p,c,ix=proof_fixture();p['communication_seeds'][0]['noise_seed']+=1
    with pytest.raises(RuntimeError):s.proof_check(p,'ID',0,c,ix)
@pytest.fixture(scope='module')
def author_model():
    s.modules()
    from src.baselines.mfa_reference_adapter import make_reference_model,MFAReferenceAdapter
    m=make_reference_model(1,16);D=32;m.D=D;m.N=3
    m.means=(np.arange(D)/100+1j*np.arange(D)/200)[None,:].astype(np.complex128)
    m.lambdas=np.eye(D,16,dtype=np.complex128)[None,:,:]*.25
    m.psis=np.ones((1,D))*.5;m.amps=np.ones(1);m.covs=m.lambdas@m.lambdas.conj().transpose(0,2,1)+.5*np.eye(D)[None,:,:]
    m.inv_covs=np.linalg.inv(m.covs);A=np.eye(D,dtype=np.complex128)[:8]
    return m,MFAReferenceAdapter(m,A)
def test_original_proper_complex_posterior(author_model):
    m,a=author_model;y=np.arange(24).reshape(3,8).astype(np.complex128)/10+0.2j
    got,timing=s.infer(a,y,.3)
    expected=m.means[0]+(m.covs[0]@a.A.conj().T@np.linalg.solve(a.A@m.covs[0]@a.A.conj().T+.3*np.eye(8),(y-m.means[0,:8]).T)).T
    assert np.allclose(got,expected,rtol=1e-13,atol=1e-13);assert timing['conditional_calls']==3
def test_batch_has_no_future_row_dependence(author_model):
    _,a=author_model;y=np.ones((6,8),np.complex128);first=s.infer(a,y,.7)[0]
    y[-1]*=100;second=s.infer(a,y,.7)[0];single=s.infer(a,y[:1],.7)[0]
    assert np.array_equal(first[:-1],second[:-1]);assert np.array_equal(first[:1],single)
def test_float1_not_integer_interface(author_model):
    _,a=author_model
    import inspect
    assert 'n_summands_or_proba=1.0' in inspect.getsource(type(a).estimate)
    assert 'target' not in inspect.signature(s.infer).parameters
def test_parameter_arrays_not_mutated(author_model):
    m,a=author_model;_,ops,_,_=s.modules();prior=ops.parameter_hashes(m);s.infer(a,np.zeros((6,8),np.complex128),.9)
    assert prior==ops.parameter_hashes(m)
def test_hash_tamper(tmp_path):
    q=tmp_path/'file';q.write_bytes(b'original');before=b.sha(q);q.write_bytes(b'changed');assert b.sha(q)!=before
def test_h_storage():
    h=(np.arange(2*2*64*160).reshape(2,2,64,160)+1j).astype(np.complex128)
    native=np.ascontiguousarray(h.transpose(1,0,3,2)).transpose(1,0,3,2)
    assert np.array_equal(native,h) and s.array_hash(native)==s.array_hash(h)
    assert native.strides!=(h.strides)
def test_worker_firewall_in_subprocess(tmp_path):
    # Installed guard denies raw historical B2/B3, prep reads, and model writes.
    code="""import control,base as b
control.scope(b.OUT/'synthetic',False)
cases=[(b.ROOT/'outputs/sp_u12b2b_sealed/raw_evaluation/ID/0000.npz','rb'),(b.ROOT/'outputs/sp_u12b3h_sealed/raw.npz','rb'),(b.ROOT/'outputs/sp_u12b5b/full_retraining.npy','rb'),(b.FINAL/'model/means.npy','wb')]
for path,mode in cases:
 try:open(path,mode)
 except RuntimeError:continue
 raise AssertionError('FIREWALL_EXPANDED')
print('FOUR_BOUNDARY_DENIALS_PASS')
"""
    env=os.environ.copy();env['PYTHONPATH']=str(b.HERE)
    r=subprocess.run([sys.executable,'-B','-c',code],env=env,cwd=b.ROOT,capture_output=True,text=True)
    assert r.returncode==0,r.stderr;assert 'FOUR_BOUNDARY_DENIALS_PASS' in r.stdout
def test_formal_requires_explicit_switch():
    import control
    with pytest.raises(RuntimeError):
        with pytest.MonkeyPatch.context() as m:m.setattr(sys,'argv',['control.py','Run']);control.main()
def test_repeated_start_rejected(tmp_path,monkeypatch):
    import control
    monkeypatch.setattr(b,'OUT',tmp_path);(tmp_path/'FORMAL_START_CONSUMED.json').write_text('{}')
    with pytest.raises(RuntimeError):control.absent()
def test_no_selection_fit_entrypoints():
    text=(b.HERE/'control.py').read_text();assert "choices=('Verify','Preflight','ToyTests','MonitorToy','Status','Stop','Export','Run')" in text
    assert 'CompleteSelection' not in text and 'StartFinal' not in text
def test_report_keeps_negative_results():
    from report import result_draft
    rows=[{'condition':c,'snr_db':v,'classification':'MFA_SUPPORTED','LR4_NMSE_dB':-2.,'MFA_NMSE_dB':-3.,'MFA_minus_LR4_dB':-1.,'simultaneous_CI_lower_dB':-2.,'simultaneous_CI_upper_dB':-.5} for c in st.CONDITIONS for v in st.SNRS]
    text=result_draft(rows,{'adapter_seconds':1.,'posterior_setup_seconds':.3})
    assert 'MFA in 35' in text;assert 'Boundary' in text and '30 dB' in text
    assert 'SP-U' not in text and 'PASS' not in text
def synthetic_cache(tmp_path,monkeypatch):
    # Create original B2-shaped archives entirely from analytic arrays.
    from src.cdl_benchmark import frames,observation_geometry
    from src.link import apply_frequency_mimo_channel,assemble_pilot_data_frame,estimate_ls_linear_channel
    monkeypatch.setattr(b,'DATA',tmp_path);p,cfg,ix=proof_fixture();cfg.update(operator_cache_sha256='operator')
    rx,tx,k,t=np.indices((2,2,64,160));h=(1+.1*rx+.2*tx+.003*k+.002*t+1j*(.3+.001*t)).astype(np.complex128)
    h=np.ascontiguousarray(h.transpose(1,0,3,2)).transpose(1,0,3,2)
    symbols=np.full((10,2,64,16),(1+1j)/np.sqrt(2),np.complex128)
    j,rx,k,t=np.indices((10,2,64,16));noise=(.01*(j+rx+k)+.02j*(j+t)).astype(np.complex128)
    bits=np.zeros((10,2,64,16,2),np.uint8);comm={'bits':bits,'symbols':symbols,'unit_noise':noise}
    g=observation_geometry();truths=frames(h);inputs=[];truth=np.empty((7,6),np.float64)
    for j in range(10):
        clean=apply_frequency_mimo_channel(truths[j],assemble_pilot_data_frame(symbols[j],g.layout))
        for si,snr in enumerate(st.SNRS):
            y=clean+np.sqrt(b.contract()['power']/10**(snr/10))*noise[j]
            sparse,_=estimate_ls_linear_channel(y,g.layout);pilot=sparse.ravel()[g.selected]
            inputs.append({'snr_db':snr,'frame_id':j,'truth_sha256':s.array_hash(truths[j]),'unit_noise_sha256':s.array_hash(noise[j]),'pilot_sha256':s.array_hash(pilot),'Y_sha256':s.array_hash(y)})
            if j>=4:truth[si,j-4]=float(np.sum(abs(truths[j].ravel()[g.data])**2))
    p['inputs']=inputs;p['H_sha256']=s.array_hash(h)
    cache=tmp_path/'outputs/sp_u12b2b_sealed/cache/ID';raw=tmp_path/'outputs/sp_u12b2b_sealed/raw_evaluation/ID';cache.mkdir(parents=True);raw.mkdir(parents=True)
    np.savez(cache/'H_0000.npz',H=h);np.savez(cache/'comm_0000.npz',**comm)
    error=np.ones((7,7,6));status=np.zeros((7,7,6),np.int8);status[:,4:6]=1
    np.savez(raw/'0000.npz',frame_error=error,frame_truth=truth,trajectory_error=error.sum(axis=2),trajectory_truth=truth.sum(axis=1),solve_status=status)
    def record(q):return {'path':q.relative_to(tmp_path).as_posix(),'sha256':b.sha(q),'bytes':q.stat().st_size}
    hi={'condition':'ID','trajectory_id':0,'group':'EVAL','H_sha256':s.array_hash(h),'export_strides_bytes':list(h.strides),'channel_seeds':p['channel_seeds'],'artifacts':[record(cache/'H_0000.npz')]}
    ci={'condition':'ID','trajectory_id':0,'shared_across_SNR':True,'frame_seeds':p['communication_seeds'],'artifacts':[record(cache/'comm_0000.npz')],**{n+'_sha256':[s.array_hash(comm[n][j]) for j in range(10)] for n in comm}}
    p.update(H_artifact_sha256=b.sha(cache/'H_0000.npz'),communication_sha256=b.sha(cache/'comm_0000.npz'),artifacts=[record(raw/'0000.npz')])
    for path,obj in [(cache/'H_0000.json',hi),(cache/'comm_0000.json',ci),(raw/'0000.json',p)]:path.write_text(json.dumps(obj),encoding='utf8')
    return cfg,ix,truth
def test_exact_input_archive_transport(tmp_path,monkeypatch):
    cfg,ix,truth=synthetic_cache(tmp_path,monkeypatch)
    h,comm,ins,lr4,t,links,old_trajectory=s.checked_input('ID',0,cfg,ix);pilots,target,energy=s.pilot_inputs(h,comm,ins)
    assert pilots.shape==(7,10,64) and target.shape==(6,3968)
    assert np.array_equal(truth,t) and np.array_equal(truth,np.broadcast_to(energy,(7,6)))
    assert not pilots.flags.writeable and not target.flags.writeable and lr4.shape==(7,6)
    e=np.stack((lr4,lr4*2),axis=2);assert np.array_equal(s.paired_reduction(e)[:,0],old_trajectory)
def test_exact_pilot_pairing_rejects_single_bad_hash(tmp_path,monkeypatch):
    cfg,ix,_=synthetic_cache(tmp_path,monkeypatch);h,comm,ins,*_=s.checked_input('ID',0,cfg,ix)
    ins[10,4]['pilot_sha256']='0'*64
    with pytest.raises(RuntimeError,match='PAIRING_FAILED'):s.pilot_inputs(h,comm,ins)
def test_archive_corruption_rejected(tmp_path,monkeypatch):
    cfg,ix,_=synthetic_cache(tmp_path,monkeypatch)
    q=tmp_path/'outputs/sp_u12b2b_sealed/cache/ID/comm_0000.npz';q.write_bytes(q.read_bytes()+b'changed')
    with pytest.raises(RuntimeError,match='HASH_DRIFT'):s.checked_input('ID',0,cfg,ix)
def test_unverified_gate_cannot_release(tmp_path,monkeypatch):
    from report import analyze
    monkeypatch.setattr(b,'OUT',tmp_path);monkeypatch.setattr(b,'verify',lambda:True)
    (tmp_path/'integrity_gate.json').write_text(json.dumps({'PASS':False}))
    (tmp_path/'status.json').write_text(json.dumps({'PASS':False,'state':'FAILED'}))
    with pytest.raises(RuntimeError):analyze(tmp_path)
    assert not(tmp_path/'analysis').exists()
def test_exact_original_sum_with_large_dynamic_range():
    original=np.ones((7,7,6),np.float64);original[:,5,0]=1e16
    paired=np.stack((original[:,5,:],original[:,5,:]*.7),axis=2)
    assert np.array_equal(s.paired_reduction(paired)[:,0],original.sum(axis=2)[:,5])
