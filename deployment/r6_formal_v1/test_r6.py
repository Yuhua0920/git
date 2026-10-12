"""Only bounded synthetic arrays and identity metadata; never real fitting."""
from pathlib import Path
import sys,os,json,subprocess,threading,time,inspect,ast,csv
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import numpy as np
import pytest
import common as c
import model_ops as m

def toy_data():
    rng=np.random.default_rng(928716);x=(rng.normal(size=(96,32))+1j*rng.normal(size=(96,32)))/np.sqrt(2)
    x[:48]+=1.1;return x

@pytest.mark.parametrize('K',[1,2,4])
@pytest.mark.parametrize('r',[4,8,16])
def test_original_toy_fit_parity_roundtrip(K,r):
    job=c.LOCAL/'toy_models'/f'K{K}_r{r}';job.mkdir(parents=True,exist_ok=False)
    x=toy_data();seed=c.candidate(f'K{K}_r{r}')['seed']
    np.save(job/'toy_input.npy',x,allow_pickle=False)
    transported=np.load(job/'toy_input.npy',mmap_mode='r',allow_pickle=False)
    core,est,_=m.reference_modules();direct=est.MfaEstimator(n_components=K,latent_dim=r,**m.FROZEN_POLICY)
    model=m.make_reference_model(K,r)
    with c.threads(6,1) as pools:
        m.fit_reference(direct,x,seed)
        with m.Observe(job) as observer:m.fit_reference(model,transported,seed)
        assert observer.steps==len(model.L_all) and observer.initial
        assert m.parameter_hashes(model)==m.parameter_hashes(direct)
        assert np.array_equal(model.L_all,direct.L_all)
        m.parameter_validity(model)
        artifact=m.save_model(model,job/'model',{'synthetic_only':True,'seed':seed},observer.initial)
        restored,_=m.load_model(job/'model');m.parameter_validity(restored)
        assert m.parameter_hashes(model)==m.parameter_hashes(restored)
        smoke=m.inference_smoke(restored)
        assert smoke['component_calls']==K and smoke['API_calls']==[['float',1.0]]
        # Separate independent dense conditional posterior, without calling
        # any author's covariance/posterior implementation for the reference.
        A=np.eye(32,dtype=complex)[[0,3,7,12]];y=x[:6]@A.T;Cn=.37*np.eye(4)
        log=[];estimates=[]
        for k in range(K):
            C=model.covs[k];mu=model.means[k];Cy=A@C@A.conj().T+Cn;inv=np.linalg.pinv(Cy,hermitian=True)
            diff=y-mu@A.T;_,ld=np.linalg.slogdet(Cy)
            log.append(np.log(model.amps[k])-ld-np.real(np.sum(diff.conj()*(inv@diff.T).T,axis=1)))
            estimates.append(mu[None,:]+(C@A.conj().T@inv@diff.T).T)
        logs=np.stack(log,axis=1);weights=np.exp(logs-logs.max(axis=1,keepdims=True));weights/=weights.sum(axis=1,keepdims=True)
        expected=sum(weights[:,k,None]*estimates[k] for k in range(K))
        actual=m.MFAReferenceAdapter(restored,A).estimate(y,Cn)
        assert np.allclose(actual,expected,atol=1e-10,rtol=1e-9)
    c.write(job/'toy_report.json',{'PASS':True,'N':96,'D':32,'K':K,'r':r,'seed':seed,'bitwise_author_fit_parity':True,
            'initialization_hash':artifact['initialization_hash'],'parameter_hash':artifact['parameter_hash'],'iterations':artifact['iterations'],
            'convergence':artifact['convergence'],'independent_dense_posterior_PASS':True,'smoke':smoke,'synthetic_only':True,'actual_fit_threadpools':pools})

def test_registry_and_frozen_policy():
    cfg=c.science();rows=c.registry();assert rows==cfg['candidate_order'] and len(rows)==9
    import hashlib
    for row in rows:
        for key,role in [('seed','candidate'),('final_seed','full_retraining')]:
            text=f"SP_U12B5A_MFA_AUTHOR2023_V1|{role}|K={row['K']}|L={row['rank']}|fit=0"
            assert row[key]==int.from_bytes(hashlib.sha256(text.encode()).digest()[:4],'big')
    assert all(cfg['policy'][k]==v for k,v in m.FROZEN_POLICY.items())

def identity_rows():
    return [{'K':K,'rank':r,'origin':origin,'snr_db':snr,'trajectory_id':i,'frame_id':f}
            for (K,r),origin,snr,i,f in sorted(m.canonical().metadata_contract(c.science()))]

def test_canonical_cardinality_32400():
    rows=identity_rows();assert m.total_cardinality(rows)['scored_frame_predictions']==32400
    for item in c.registry(): assert m.candidate_cardinality([v for v in rows if (v['K'],v['rank'])==(item['K'],item['rank'])],item['K'],item['rank'])==3600

@pytest.mark.parametrize('fault',['missing','duplicate','unexpected'])
def test_cardinality_rejects(fault):
    rows=identity_rows()
    if fault=='missing':rows.pop()
    elif fault=='duplicate':rows.append(rows[0])
    else:rows[0]={**rows[0],'frame_id':3}
    with pytest.raises((AssertionError,RuntimeError)):m.total_cardinality(rows)

def test_linear_pooled_then_macro_not_db_or_trajectory_mean():
    E=np.array([[1.,9.],[4.,0.]]);T=np.array([[1.,3.],[2.,20.]])
    assert m.pooled_ratio(E,T)==14/26
    assert m.pooled_ratio(E,T)!=np.mean(E/T)
    cells=[.01,.02,.03,.1,.2,.3]
    # Match the frozen NumPy float64 reduction, not Python's different
    # rounding order. This does not change the scientific metric.
    assert m.macro_score(cells)==float(np.asarray(cells,dtype=np.float64).sum(dtype=np.float64)/6)
    assert 10*np.log10(m.macro_score(cells))!=np.mean(10*np.log10(cells))

@pytest.mark.parametrize('E,T',[([1],[0]),([-1],[1]),([float('nan')],[1]),([1,2],[1])])
def test_invalid_metric_stop(E,T):
    with pytest.raises(RuntimeError):m.pooled_ratio(E,T)

def test_exact_tie_and_one_ulp_not_tie():
    rows=[{**row,'macro_selection_score':1.} for row in c.registry()]
    assert m.automatic_select(rows)['name']=='K1_r4'
    rows[-1]['macro_selection_score']=np.nextafter(1.,0.).item()
    assert m.automatic_select(rows)['name']=='K4_r16'
    with pytest.raises(RuntimeError):m.automatic_select(rows[:-1])

def test_maxiter_legal_not_restart():
    assert m.convergence(np.linspace(-1000,-200,100))=='MAX_ITER_REACHED'
    with pytest.raises(RuntimeError):m.convergence([np.nan]*100)
    with pytest.raises(RuntimeError):m.convergence([1.]*6)

def test_corrupt_serialization_refused():
    folder=c.LOCAL/'corrupt_model';folder.mkdir()
    for p in (c.LOCAL/'toy_models/K1_r4/model').iterdir():(folder/p.name).write_bytes(p.read_bytes())
    p=folder/'means.npy';p.write_bytes(p.read_bytes()+b'CORRUPTION')
    with pytest.raises(RuntimeError,match='SERIALIZATION_FILE_CORRUPTION'):m.load_model(folder)

def test_nonpositive_covariance_rejected_even_with_scale_tolerance():
    model=m.make_reference_model(1,1);model.D=3;model.N=12
    model.means=np.zeros((1,3),complex);model.lambdas=np.zeros((1,3,1),complex);model.lambdas[0,0,0]=1e5
    model.psis=np.ones((1,3))*1e-6;model.amps=np.ones(1)
    model.covs=model.lambdas@model.lambdas.conj().transpose(0,2,1)+1e-6*np.eye(3)[None]
    model.inv_covs=np.eye(3)[None];model.covs[0,2,2]=-1e-6
    with pytest.raises(RuntimeError,match='ILLEGAL_MODEL_PARAMETERS'):m.parameter_validity(model)

def child(code,*args):
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['TEMP']=env['TMP']=str(c.LOCAL/'temporary')
    (c.LOCAL/'temporary').mkdir(exist_ok=True)
    return subprocess.run([sys.executable,'-B','-X','utf8','-c',code,str(HERE),*args],cwd=c.ROOT,env=env,text=True,capture_output=True,encoding='utf8')

@pytest.mark.parametrize('path',['outputs/sp_u12b2b/never_read.npy','outputs/sp_u12b3h/never_read.json','paper/main.tex','outputs/sp_u12b5b/selection_train.npy','outputs/sp_u12b5b/validation_pilots.npy'])
def test_firewall_real_data_denied_before_open(path):
    r=child("import sys; sys.path.insert(0,sys.argv[1]); import common as c; c.scope(); (c.ROOT/sys.argv[2]).read_bytes()",path)
    assert r.returncode!=0 and ('ALLOWLIST_REJECTED' in r.stderr or 'READ_REJECTED' in r.stderr)

def test_historical_write_denied():
    r=child("import sys; sys.path.insert(0,sys.argv[1]); import common as c; c.scope(); (c.ROOT/'src/THIS_MUST_NOT_EXIST.txt').write_text('x')")
    assert r.returncode!=0 and 'WRITE_FORBIDDEN' in r.stderr and not (c.ROOT/'src/THIS_MUST_NOT_EXIST.txt').exists()

@pytest.mark.parametrize('role',['runtime','output','external'])
def test_no_numeric_backdoor_through_runtime_or_outputs(role):
    r=child("import sys,pathlib; sys.path.insert(0,sys.argv[1]); import common as c; c.scope(); p={'runtime':pathlib.Path(sys.prefix)/'UNAPPROVED.npy','output':c.OUT/'UNAPPROVED.npy','external':pathlib.Path('C:/UNAPPROVED.txt')}[sys.argv[2]]; p.read_bytes()",role)
    assert r.returncode!=0 and any(v in r.stderr for v in ('READ_FORBIDDEN','NUMERICAL_READ','ALLOWLIST_REJECTED'))

@pytest.mark.parametrize('action',['train','validate','select','final'])
def test_pending_approval_exact_formal_entry_refused(action):
    r=subprocess.run([sys.executable,'-B','-X','utf8',str(HERE/'runner.py'),action,'--candidate','K1_r4'],cwd=c.ROOT,capture_output=True,text=True,encoding='utf8')
    assert r.returncode!=0 and 'MENTOR_APPROVAL_PENDING' in r.stderr
    assert not (c.OUT/'candidates').exists() and not (c.OUT/'SELECTION_MANIFEST.json').exists() and not (c.OUT/'final').exists()

def test_no_optimized_or_old_orchestrator_imports():
    for file in ['common.py','runner.py','worker.py','model_ops.py','supervisor.py']:
        source=(HERE/file).read_text(encoding='utf8');tree=ast.parse(source)
        names=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not any(n and ('sp_u12b5b_common' in n or 'sp_u12b5b_model' in n or 'low_memory' in n) for n in names)
        assert not any('run_sp_u12b5b' in n.name for node in ast.walk(tree) if isinstance(node,ast.Import) for n in node.names)
    assert 'fit_reference(m,x,seed)' in (HERE/'worker.py').read_text(encoding='utf8')

@pytest.mark.parametrize('mode',['success','exception','timeout','RSS','commit','RAM_floor','commit_floor','user_stop','tree_timeout'])
def test_actual_windows_monitor_faults(mode):
    from supervisor import run_worker
    from windows_resources import frozen_probe
    job=c.LOCAL/'monitor'/mode;job.mkdir(parents=True)
    limits={'RSS_bytes':256*c.GIB//1024,'private_commit_bytes':512*c.GIB//1024,'min_available_physical_bytes':1,'min_remaining_commit_bytes':1}
    extra=['--probe-mode','success'];timeout=8
    if mode=='exception':extra=['--probe-mode','exception']
    if mode in ('timeout','tree_timeout'):extra=['--probe-mode','tree' if mode=='tree_timeout' else 'allocate'];timeout=1.2
    if mode=='RSS':extra=['--probe-mode','allocate'];limits['RSS_bytes']=32*2**20
    if mode=='commit':extra=['--probe-mode','allocate'];limits['private_commit_bytes']=48*2**20
    if mode in ('RAM_floor','commit_floor','user_stop'):
        def inject_after_handshake():
            end=time.monotonic()+5
            while not (job/'monitor_ready.json').exists() and time.monotonic()<end:time.sleep(.01)
            if mode=='RAM_floor':limits['min_available_physical_bytes']=frozen_probe().memory()['total_physical_bytes']+1
            if mode=='commit_floor':limits['min_remaining_commit_bytes']=2**60
            if mode=='user_stop':c.write(job/'STOP_REQUESTED.json',{'reason':'toy injection'})
        threading.Thread(target=inject_after_handshake,daemon=True).start()
    os.environ['R6_PERSONAL_SELFTEST']='1'
    if mode=='success':
        state,_=run_worker('unit-probe',job,local=True,local_limits=limits,extra=extra,timeout=timeout)
        assert state['PASS'] and state['actual_interpreter_PID']!=state['venv_launcher_PID']
    else:
        with pytest.raises(RuntimeError):run_worker('unit-probe',job,local=True,local_limits=limits,extra=extra,timeout=timeout)
        state=c.read(job/'status.json');assert not state['PASS'] and state['state'] in ('STOP','FAILED')
        assert state['actual_interpreter_PID']!=state['venv_launcher_PID']
        if mode=='tree_timeout': assert state['actual_process_tree_max_members']>=3
    assert not (c.OUT/'STOP.json').exists() and not (c.OUT/'active_worker.json').exists()

def test_monitor_copy_and_historical_contract_unchanged():
    assert (HERE/'windows_resources.py').read_bytes()==(c.ROOT/'deployment/r5_128gb_v1/windows_resources.py').read_bytes()
    p=c.protocol();assert p['status']=='MENTOR_APPROVAL_PENDING' and p['BLAS_threads']==6 and p['OpenMP_threads']==1
    assert c.science()['resource_contract']['BLAS_threads']==1 and c.science()['resource_contract']['max_process_bytes']==6*2**30

def test_technical_export_no_preparation_or_model_arrays():
    import runner,zipfile,hashlib
    (c.LOCAL/'pytest_export_fixture.log').write_text('Synthetic ZIP export fixture only; not an actual pytest or background completion report.\n',encoding='utf8')
    report=runner.export_results(False)
    with zipfile.ZipFile(report['path']) as z:
        names=z.namelist()
        assert 'r6_formal/preparation_personal/pytest_export_fixture.log' in names
        assert all(Path(n).suffix.lower()not in('.npy','.npz','.pkl','.pickle')for n in names)
        assert all('historical_before.json'not in n and 'source_snapshot'not in n for n in names)
        manifest=json.loads(z.read('result_manifest.json'))
        for name,row in manifest['files'].items():assert hashlib.sha256(z.read('r6_formal/'+name)).hexdigest()==row['sha256']
