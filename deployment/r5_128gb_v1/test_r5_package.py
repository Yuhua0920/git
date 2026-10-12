"""Safe local checks: tiny synthetic only, no laboratory qualification claims."""
from pathlib import Path
import sys,os,json,subprocess,ast,hashlib,uuid
os.environ['R5_PERSONAL_SELFTEST']='1';sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import pytest
from common import ROOT,OUT,LOCAL,read,write,protocol,threads,sha,GIB
from windows_resources import capacity,exceeded

def test_protocol_exact_authorized_limits():
    p=protocol();assert p['thread_candidates']==[1,2,4,6,8,10,20]
    assert p['limits']=={'RSS_bytes':48*GIB,'private_commit_bytes':64*GIB,'min_available_physical_bytes':32*GIB,'min_remaining_commit_bytes':32*GIB}
    assert p['launch_minimum']['available_physical_bytes']==80*GIB and p['sample_seconds']==.1
    assert p['historical_formal_RSS_bytes']==6*GIB and p['historical_formal_job_seconds']==86400
    assert not p['formal_execution_allowed']and not p['automatic_retries']

def fake_hardware():
    return {'CPU_exact_model':'Intel(R) Core(TM) i9-10900KF CPU','topology':{'physical_cores':10,'logical_threads':20},
            'memory':{'total_physical_bytes':128*GIB,'available_physical_bytes':108*GIB,'commit_available_bytes':108*GIB},'disk_free_bytes':390*GIB}
def test_capacity_pass_and_each_failure():
    v=fake_hardware();assert capacity(v)['PASS']
    for section,key,value in(('memory','available_physical_bytes',79*GIB),('memory','commit_available_bytes',79*GIB),('topology','physical_cores',9)):
        v=fake_hardware();v[section][key]=value;assert not capacity(v)['PASS']
    v=fake_hardware();v['CPU_exact_model']='personal 16GB host';assert not capacity(v)['PASS']

@pytest.mark.parametrize('field',['RSS_bytes','private_commit_bytes','physical','commit'])
def test_each_hard_stop_boundary(field):
    limits=protocol()['limits'];tree={'RSS_bytes':1,'private_commit_bytes':1,'members':[]};system={'available_physical_bytes':100*GIB,'commit_available_bytes':100*GIB}
    if field in tree:tree[field]=limits[field]+1
    elif field=='physical':system['available_physical_bytes']=32*GIB-1
    else:system['commit_available_bytes']=32*GIB-1
    assert exceeded(tree,system,limits)

def test_balanced_order():
    c=protocol()['thread_candidates'];orders=[c[i:]+c[:i]for i in range(7)]
    assert all(set(row)==set(c)for row in orders)
    assert all(set(row[position]for row in orders)==set(c)for position in range(7))

def test_full_shapes_block_before_allocation():
    from mfa_ops import synthetic,enforce_shape
    for n in(12000,15000):
        with pytest.raises(AssertionError,match='PERSONAL_COMPUTER_FULL_SCALE_FORBIDDEN'):synthetic(n,4032,4,16,0)

def test_synthetic_seed_and_layout():
    import numpy as np
    from mfa_ops import synthetic,seed,array_hash
    x=synthetic(64,32,2,4,seed('unit_data'));y=synthetic(64,32,2,4,seed('unit_data'))
    assert array_hash(x)==array_hash(y)and x.dtype==np.complex128 and x.flags.c_contiguous

def test_legacy_rng_restored_and_native_initialization_replayed():
    import numpy as np
    from mfa_ops import synthetic,initialize_native,parameters,seed,compare,Observe
    from src.baselines.mfa_reference_adapter import make_reference_model
    with threads(1,1):
        x=synthetic(64,32,2,4,seed('rng_fixture'));state=np.random.get_state();a=make_reference_model(2,4);b=make_reference_model(2,4)
        observation=Observe()
        with observation:initialize_native(a,x,seed('rng_init'))
        initialize_native(b,x,seed('rng_init'));after=np.random.get_state()
    assert state[0]==after[0]and np.array_equal(state[1],after[1])and state[2:]==after[2:]
    assert all(compare(value,parameters(b)[name])['PASS']for name,value in parameters(a).items())
    assert observation.kmeans[0]['n_init']==1 and observation.kmeans[0]['random_state']is None
    assert observation.kmeans[0]['nested_BLAS_cap_verified']

@pytest.mark.parametrize('K',[1,2,4])
def test_author_adapter_and_independent_posterior(K):
    from mfa_ops import fidelity_state
    with threads(1,1):_,_,checks=fidelity_state(K,4)
    assert all(v['PASS']for v in checks)

def test_tracing_preserves_native_numeric_state():
    from mfa_ops import synthetic,seed,initialize_native,parameters,compare,Observe,finalize_author
    from src.baselines.mfa_reference_adapter import make_reference_model
    with threads(1,1):
        x=synthetic(64,32,2,4,seed('trace_data'));a=make_reference_model(2,4);b=make_reference_model(2,4)
        initialize_native(a,x,seed('trace_init'));initialize_native(b,x,seed('trace_init'))
        observation=Observe()
        with observation:a._EM_per_component(x,a.PPCA)
        b._EM_per_component(x,b.PPCA)
        before=parameters(a);assert all(compare(value,parameters(b)[name])['PASS']for name,value in before.items())
        finalize_author(a)
    assert sum(v['kind']=='loading_contraction_complete'for v in observation.events)==2
    assert all(v['broadcast_shape']==[32,4,64]for v in observation.events if v['kind']=='loading_contraction_complete')
    assert a.maxiter==100 and a.tol==1e-4
    assert compare(a.covs,b._covs)['PASS']and compare(a.means,b._means)['PASS']

def test_guards_block_legacy_values_and_writes():
    code="from common import scope,ROOT;scope();(ROOT/'outputs/sp_u12b5b/selection_train.npy').open('rb')"
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=HERE,capture_output=True,text=True,encoding='utf8')
    assert p.returncode!=0 and'R5_SCIENTIFIC_INPUT_FIREWALL'in p.stderr
    code="from common import scope,ROOT;scope();(ROOT/'src/baselines/mfa_reference_adapter.py').open('wb')"
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=HERE,capture_output=True,text=True,encoding='utf8')
    assert p.returncode!=0 and'R5_HISTORICAL_WRITE_FORBIDDEN'in p.stderr

@pytest.mark.parametrize('action',['train','validation','retrain','target-evaluation'])
def test_no_formal_CLI_actions(action):
    p=subprocess.run([sys.executable,'-B','-X','utf8',str(HERE/'driver.py'),action],cwd=HERE,capture_output=True,text=True,encoding='utf8')
    assert p.returncode!=0 and'invalid choice'in p.stderr

def test_no_low_memory_math_import_or_policy_reassignment():
    for name in('worker.py','mfa_ops.py','driver.py'):
        text=(HERE/name).read_text(encoding='utf8');tree=ast.parse(text)
        assert 'low.em_step' not in text and 'mfa_low_memory_fidelity_r4' not in text
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign):assert not any(isinstance(t,ast.Attribute)and t.attr in('maxiter','tol','rs_clip','PPCA','lock_psis')for t in node.targets)

def test_frozen_reference_complete_tree():
    source=read(ROOT/'outputs/sp_u12b5a/historical_reference.json');combined={}
    for repo,info in source.items():
        for name,h in info['files'].items():assert sha(ROOT/'reference/sp_u12b5a/paper_period'/repo/name)==h;combined[repo+'/'+name]=h
    assert len(combined)==15
    assert hashlib.sha256(json.dumps(combined,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='e6c27180ba2abb07c699e943ba581290371a4dea04583b19b63037c552c3db5f'

def local_limits(RSS=256*2**20):
    return {'RSS_bytes':RSS,'private_commit_bytes':512*2**20,'min_available_physical_bytes':256*2**20,'min_remaining_commit_bytes':256*2**20}
def test_actual_PID_and_descendant_monitor():
    from supervisor import run_worker
    job=LOCAL/('pid_tree_'+uuid.uuid4().hex)
    state,result=run_worker('unit-probe',job,local=True,local_limits=local_limits(),timeout=30)
    assert state['PASS']and state['actual_process_tree_max_members']>=3 and state['resource_samples']>=5
    assert state['actual_interpreter_PID']==result['actual_PID']
    assert state['actual_interpreter_PID']!=state['venv_launcher_PID']
    assert state['all_descendants_kernel_contained']and state['peak_private_commit_bytes']>32*2**20
    write(LOCAL/'actual_PID_tree_test.json',state)

def test_actual_resource_STOP_and_exception_record():
    from supervisor import run_worker
    job=LOCAL/('resource_stop_'+uuid.uuid4().hex)
    with pytest.raises(RuntimeError):run_worker('unit-probe',job,local=True,local_limits=local_limits(32*2**20),timeout=30)
    state=read(job/'status.json');assert state['state']=='STOP'and not state['PASS']and state['stop_reason']
    assert(job/'supervisor_exception.json').is_file()
    write(LOCAL/'actual_resource_STOP_test.json',state)

def test_actual_worker_exception_record():
    from supervisor import run_worker
    job=LOCAL/('exception_'+uuid.uuid4().hex);job.mkdir(parents=True);(job/'intentional_failure.flag').write_text('probe',encoding='utf8')
    with pytest.raises(RuntimeError):run_worker('unit-probe',job,local=True,local_limits=local_limits(),timeout=30)
    state=read(job/'status.json');assert state['state']=='FAILED'and(job/'exception.json').is_file()
    assert read(job/'exception.json')['type']=='RuntimeError'

def synthetic_records():
    names=['slogdet4032','covariance_Woodbury','beta_contraction','dense_likelihood_GEMM','latent_and_old_loading','latent_moment','loading_contraction','residual_update','native_KMeans_initialization','small_complete_native_EM','medium_complete_native_EM']
    return [{'threads':t,'kernels':{n:{'wall_seconds':1+(0 if t in(2,4)else 1),'PASS':True,'native_iterations':8 if n.endswith('EM')else None}for n in names},
             'native_KMeans':{'native_KMeans_with_real_conversion_wall_seconds':1+(0 if t in(2,4)else 1)},
             'resources':{'peak_RSS_bytes':1,'peak_private_commit_bytes':2}}for t in protocol()['thread_candidates']for block in range(7)]
def test_composite_near_tie_uses_less_resources():
    from analysis import select_threads
    v=select_threads(synthetic_records());assert v['RECOMMENDED_BLAS_THREADS']==2 and v['RECOMMENDED_OPENMP_THREADS']==2
    assert v['no_single_GEMM_selection']
def test_failure_not_selected_away():
    from analysis import select_threads
    r=synthetic_records();r[0]['kernels']['slogdet4032']['PASS']=False
    with pytest.raises(AssertionError):select_threads(r)

def test_gate_missing_or_modified_prerequisite_stops(tmp_path):
    import driver
    stage='unit_missing_'+uuid.uuid4().hex
    with pytest.raises(FileNotFoundError):driver.gate(stage)

def test_cost_reporting_schema_and_exact_counts():
    from analysis import select_threads,training_budget,inference_cost
    rows=synthetic_records();choice=select_threads(rows)
    def smoke(n):
        return {'N':n,'initialization_seconds':1.,'EM_iteration_seconds':[10.,11.],
            'finalization_seconds':1.,'native_KMeans':{'n_iter':3},
            'phase_events':[{'kind':'return','function':'_calc_probs','wall_seconds':2.,'full_phase':'EM_update_1'},
                            {'kind':'return','function':'_update_covs','wall_seconds':2.,'full_phase':'EM_update_2'}]}
    result=training_budget(smoke(12000),smoke(15000),choice,rows)
    assert len(result['EXTRAPOLATED']['nine_candidate_training'])==9
    assert not result['formal_24h_safety_certified']and result['UNKNOWN']
    records=[{'blas':2,'K':K,'report':{'rows':[{'batch_six_including_setup_seconds':1.}for _ in range(6)]}}for K in(1,2,4)]
    costs=inference_cost(records,2)
    assert costs['validation_single_candidate_frames']==3600 and costs['validation_nine_candidates_frames']==32400
    assert costs['future_formal_frames']==210000 and costs['nine_candidate_validation_central_seconds']==5400

def test_synthetic_parameter_public_copy_and_float_contract():
    from mfa_ops import synthetic_parameters,clone_for_inference,posterior_independent
    with threads(1,1):
        m=synthetic_parameters(2,32,4);public=clone_for_inference(m)
        from src.baselines.mfa_reference_adapter import MFAReferenceAdapter
        import numpy as np
        y=np.zeros((6,4),dtype=np.complex128)
        h=MFAReferenceAdapter(public,np.eye(32,dtype=np.complex128)[:4]).estimate(y,.37*np.eye(4))
    assert h.shape==(6,32)and np.isfinite(h).all()

def test_multithread_tiny_numeric_parity():
    from mfa_ops import fidelity_state,compare
    with threads(1,1):_,a,_=fidelity_state(2,4)
    with threads(2,2):_,b,_=fidelity_state(2,4)
    assert all(compare(v,b[k])['PASS']for k,v in a.items()if v.dtype.kind not in('U','S'))

def test_actual_auditable_timeout_STOP():
    from supervisor import run_worker
    job=LOCAL/('timeout_'+uuid.uuid4().hex)
    with pytest.raises(RuntimeError):run_worker('unit-probe',job,local=True,local_limits=local_limits(),timeout=.5)
    state=read(job/'status.json');assert state['state']=='STOP'and state['stop_reason']=='AUDITABLE_TECHNICAL_TIME_BUDGET_EXCEEDED'

def test_named_job_manual_termination():
    import threading,time
    from supervisor import run_worker
    from windows_resources import stop_named_job
    job=LOCAL/('manual_stop_'+uuid.uuid4().hex);terminated=[]
    def stop():
        for _ in range(100):
            if(job/'probe_child.json').exists():
                terminated.append(stop_named_job(read(job/'status.json')['JobObject_name']));return
            time.sleep(.05)
    thread=threading.Thread(target=stop);thread.start()
    with pytest.raises(RuntimeError):run_worker('unit-probe',job,local=True,local_limits=local_limits(),timeout=15)
    thread.join(timeout=10);assert terminated==[True]
    assert not read(job/'status.json')['PASS']

def test_exact_preflight_body_and_golden_fixtures(tmp_path):
    job=LOCAL/('tiny_preflight_'+uuid.uuid4().hex)
    code="from common import scope,threads,write;scope();from worker import preflight;from pathlib import Path;\nwith threads(1,1):\n result=preflight(Path("+repr(str(job))+"))\n assert result['PASS']\n write(Path("+repr(str(job))+" )/'direct_tiny_preflight.json',result)"
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],cwd=HERE,capture_output=True,text=True,encoding='utf8',timeout=90)
    assert p.returncode==0,p.stderr
    assert read(job/'direct_tiny_preflight.json')['PASS']

def test_diagnostic_kernels_tiny_anchor_and_thread_parity(monkeypatch,tmp_path):
    import copy,mfa_ops
    cfg=copy.deepcopy(protocol())
    cfg['kernel_shape'].update(N=64,D=32,rank=4)
    cfg['kmeans_shape'].update(N=64,D=32,rank=4)
    cfg['small_em'].update(N=64,D=32,rank=4)
    cfg['medium_em'].update(N=128,D=64,rank=4)
    monkeypatch.setattr(mfa_ops,'protocol',lambda:cfg)
    with threads(1,1):
        kernels,h=mfa_ops.benchmark_kernels()
        for name,fn in kernels.items():
            folder=tmp_path/name;folder.mkdir();mfa_ops.checked_timing(fn,folder,save=True)
    with threads(2,2):
        kernels,h2=mfa_ops.benchmark_kernels();assert h==h2
        for name,fn in kernels.items():assert mfa_ops.checked_timing(fn,tmp_path/name)['PASS']

def test_actual_kernel_commit_cap_denies_allocation():
    from supervisor import run_worker
    job=LOCAL/('commit_denial_'+uuid.uuid4().hex);job.mkdir(parents=True)
    (job/'probe_allocate_bytes.flag').write_text(str(128*2**20),encoding='ascii')
    limits=local_limits();limits['private_commit_bytes']=64*2**20
    with pytest.raises(RuntimeError):run_worker('unit-probe',job,local=True,local_limits=limits,timeout=30)
    state=read(job/'status.json');assert not state['PASS']
    assert'MemoryError'in read(job/'probe_child_error.json')['stderr']
    assert state['state']=='STOP'and'KERNEL_COMMIT_CAP'in state['stop_reason']

def test_OpenMP_uses_KMeans_clock_not_covariance_initialization():
    from analysis import select_threads
    rows=synthetic_records()
    for row in rows:
        row['native_KMeans']['native_KMeans_with_real_conversion_wall_seconds']=.5 if row['threads']==1 else 1.
        row['kernels']['native_KMeans_initialization']['wall_seconds']=.1 if row['threads']==20 else 1.
    assert select_threads(rows)['RECOMMENDED_OPENMP_THREADS']==1
