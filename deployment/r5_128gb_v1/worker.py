"""Isolated synthetic workers. No scientific datasets or formal runner imports."""
from pathlib import Path
import os,sys,time,argparse,json,traceback,subprocess,gc
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import ROOT,OUT,read,write,scope,environment,threads,protocol,scientific_counters

def preflight(job):
    import numpy as np
    from mfa_ops import fidelity_state,compare,array_hash,seed
    reports=[];cfg=protocol()['fidelity_shape']
    for K in cfg['K']:
        for r in cfg['rank']:
            write(job/'phase.json',{'phase':'cross_host_author_adapter_fidelity','K':K,'rank':r})
            x,state,checks=fidelity_state(K,r);x2,replay,checks2=fidelity_state(K,r)
            checks +=checks2;checks.append({'field':'seed_input_replay','PASS':array_hash(x)==array_hash(x2)})
            anchor=HERE/'golden'/f'K{K}_r{r}.npz'
            with np.load(anchor,allow_pickle=False)as gold:
                assert set(gold.files)==set(state),'GOLDEN_FIELDS_DIFFER'
                for name,a in state.items():
                    checks.append({'field':name,'PASS':str(a)==str(gold[name])}if a.dtype.kind in('U','S')else{'field':name,**compare(a,gold[name])})
                    checks.append({'field':'replay/'+name,'PASS':str(a)==str(replay[name])}if a.dtype.kind in('U','S')else{'field':'replay/'+name,**compare(a,replay[name])})
            check={'K':K,'rank':r,'PASS':all(c['PASS']for c in checks),'checks':checks,
                   'input_sha256':array_hash(x),'initialization_seed':seed(f'fidelity_init_K{K}_r{r}')}
            reports.append(check);write(job/'fidelity_progress.json',reports);assert check['PASS'],'CROSS_HOST_FIDELITY_STOP'
    from mfa_ops import Observe,initialize_native,synthetic
    from src.baselines.mfa_reference_adapter import make_reference_model
    probe=Observe();x=synthetic(64,32,2,4,seed('native_KMeans_probe'))
    with probe:initialize_native(make_reference_model(2,4),x,seed('native_KMeans_probe_init'))
    assert probe.kmeans[0]['nested_BLAS_cap_verified'],'NATIVE_KMEANS_NESTED_THREAD_CONTROL_UNVERIFIED'
    command=[sys.executable,'-B','-X','utf8','-m','pytest',str(ROOT/'tests/test_mfa_reference.py'),'-q','-p','no:cacheprovider','--basetemp',str(job/'pytest_tmp')]
    original_tests=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace')
    write(job/'original_reference_pytest.json',{'exit_code':original_tests.returncode,'stdout':original_tests.stdout,'stderr':original_tests.stderr})
    assert original_tests.returncode==0,'ORIGINAL_REFERENCE_UNIT_TESTS_FAILED'
    result={'PASS':True,'cases':reports,'atol':1e-10,'rtol':1e-9,'golden_origin':'personal-side independent synthetic fixtures; fixed author source/adapter',
            'native_KMeans_policy_and_threads':probe.kmeans[0],'original_reference_pytest_exit':original_tests.returncode,**scientific_counters()}
    write(job/'fidelity_report.json',result);return result

def benchmark(job,anchor,save):
    from mfa_ops import benchmark_kernels,checked_timing,Observe
    kernels,input_hash=benchmark_kernels();rows=[]
    for name,fn in kernels.items():
        write(job/'phase.json',{'phase':'kernel_warmup_timing_and_elementwise_parity','kernel':name})
        folder=anchor/name
        if save:folder.mkdir(parents=True,exist_ok=False)
        result=checked_timing(fn,folder,save=save);result.update(kernel=name,input_sha256=input_hash)
        rows.append(result);write(job/'benchmark_progress.json',rows);gc.collect()
    native=Observe()
    with native:kernels['native_KMeans_initialization']()
    assert native.kmeans[0]['nested_BLAS_cap_verified'],'NESTED_KMEANS_THREADS_NOT_VERIFIED'
    result={'PASS':all(r['PASS']for r in rows),'rows':rows,'native_KMeans':native.kmeans[0],
            'anchor_creation':save,'input_sha256':input_hash,**scientific_counters()}
    write(job/'benchmark_report.json',result);return result

def inference(job,K):
    import numpy as np
    from mfa_ops import synthetic_parameters,seed,cn,clone_for_inference,compare,Observe
    from src.baselines.mfa_reference_adapter import ProjectPilotInterface,MFAReferenceAdapter
    from threadpoolctl import threadpool_info
    cfg=protocol()['inference'];m=clone_for_inference(synthetic_parameters(K,cfg['D'],cfg['rank']))
    interface=ProjectPilotInterface();A=interface.A;rng=np.random.default_rng(seed('fixed_inference_observations'))
    y=cn(rng,(cfg['batch'],cfg['observations']));adapter=MFAReferenceAdapter(m,A);rows=[]
    # One baseline per noise state on the same in-memory synthetic parameters.
    with threads(1,1):references={str(noise):adapter.estimate(y,noise*np.eye(64))for noise in cfg['noise_variances']}
    for noise in cfg['noise_variances']:
        Cn=noise*np.eye(64,dtype=np.complex128)
        write(job/'phase.json',{'phase':'all_components_author_inference','K':K,'noise_variance':noise})
        adapter.estimate(y,Cn)
        for block in range(cfg['timing_blocks']):
            t=time.perf_counter();m._prepare_for_prediction(A,Cn);setup=time.perf_counter()-t
            t=time.perf_counter();estimated=adapter.estimate(y,Cn);batch=time.perf_counter()-t
            parity=compare(estimated,references[str(noise)]);assert parity['PASS'],'INFERENCE_THREAD_PARITY_STOP'
            rows.append({'K':K,'noise_variance':noise,'block':block,'noise_setup_seconds':setup,
                'batch_six_including_setup_seconds':batch,'per_frame_amortized_seconds':batch/6,
                'parity':parity,'n_summands_or_proba_type':'float','n_summands_or_proba':1.0})
    # Observe an extra un-timed adapter call to verify the API and all-component path.
    import types
    calls=[];terms=[];original=m.estimate;term=m._lmmse
    def spy(self,*args,**kwargs):calls.append((type(kwargs['n_summands_or_proba']).__name__,kwargs['n_summands_or_proba']));return original(*args,**kwargs)
    def lmmse(self,*args,**kwargs):terms.append(1);return term(*args,**kwargs)
    m.estimate=types.MethodType(spy,m);m._lmmse=types.MethodType(lmmse,m)
    adapter.estimate(y,cfg['noise_variances'][0]*np.eye(64))
    assert calls==[('float',1.0)]and len(terms)==6*K,'ALL_COMPONENTS_FLOAT1_CONTRACT_STOP'
    result={'PASS':True,'K':K,'rows':rows,'adapter_API_calls':calls,'all_component_calls':len(terms),
            'synthetic_parameters_only':True,'timed_batch_includes_native_noise_setup':True,**scientific_counters()}
    write(job/'inference_report.json',result);return result

def unit_probe(job):
    # Standard-library PID/process-tree probe, no MFA or NumPy imports.
    count=int((job/'probe_allocate_bytes.flag').read_text(encoding='ascii')) if (job/'probe_allocate_bytes.flag').exists() else 64*2**20
    child_code=f"import os,time; a=bytearray({count}); a[::4096]=bytes([1])*len(a[::4096]); print(os.getpid(),flush=True); time.sleep(2)"
    p=subprocess.Popen([sys.executable,'-B','-c',child_code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    line=p.stdout.readline().strip()
    if not line:
        error=p.stderr.read();p.wait(timeout=15);write(job/'probe_child_error.json',{'stderr':error,'exit_code':p.returncode})
        if'MemoryError'in error:raise MemoryError('KERNEL_CAPPED_STDLIB_CHILD_ALLOCATION')
        raise RuntimeError('STDLIB_CHILD_FAILED_BEFORE_PID_REPORT:'+error)
    child_pid=int(line);write(job/'probe_child.json',{'actual_child_PID':child_pid})
    p.wait(timeout=15)
    if (job/'intentional_failure.flag').exists():raise RuntimeError('INTENTIONAL_EXCEPTION_PROBE_NO_MFA')
    return {'PASS':True,'actual_child_PID':child_pid,'MFA_calls':0}

def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('preflight','pair-fidelity','anchor','benchmark','inference','smoke12000','smoke15000','unit-probe'),required=True)
    p.add_argument('--job',required=True);p.add_argument('--blas',type=int,default=1);p.add_argument('--openmp',type=int,default=1)
    p.add_argument('--K',type=int);p.add_argument('--anchor');args=p.parse_args()
    job=Path(args.job).resolve();assert job.is_relative_to(OUT);job.mkdir(parents=True,exist_ok=True)
    scope();write(job/'actual_pid.json',{'PID':os.getpid(),'executable':sys.executable,'time':time.time()})
    start=time.monotonic()
    while not(job/'monitor_ready.json').exists():
        assert time.monotonic()-start<60,'NO_RESOURCE_MONITOR_HANDSHAKE';time.sleep(.02)
    if args.action=='unit-probe':
        assert os.environ.get('R5_PERSONAL_SELFTEST')=='1','UNIT_PROBE_NOT_A_LAB_STAGE'
        result=unit_probe(job)
    else:
        for name in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]=str(args.blas)
        os.environ['OMP_NUM_THREADS']=str(args.openmp)
        env=environment();write(job/'environment.json',env)
        with threads(args.blas,args.openmp)as actual:
            write(job/'actual_threadpools.json',actual)
            if args.action in('preflight','pair-fidelity'):result=preflight(job)
            elif args.action in('benchmark','anchor'):
                anchor=Path(args.anchor).resolve();assert anchor.is_relative_to(OUT/'_scratch')
                result=benchmark(job,anchor,args.action=='anchor')
            elif args.action=='inference':result=inference(job,args.K)
            else:
                from mfa_ops import full_smoke
                result=full_smoke(12000 if args.action=='smoke12000'else 15000,job)
                write(job/'full_scale_report.json',result)
            result['actual_threadpools']=actual
    from windows_resources import frozen_probe
    result.update(PASS=True,actual_PID=os.getpid(),native_final_memory=frozen_probe().process_memory(),**scientific_counters())
    write(job/'worker_result.json',result)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if '--job'in sys.argv:
            job=Path(sys.argv[sys.argv.index('--job')+1]).resolve()
            if job.is_relative_to(OUT):write(job/'exception.json',{'type':type(e).__name__,'traceback':traceback.format_exc(),**scientific_counters()})
        traceback.print_exc();sys.exit(1)
