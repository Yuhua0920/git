"""Only synthetic Fidelity/resource workers; original MFA kernels are imported."""
from pathlib import Path
import os,sys,json,time,argparse,traceback,subprocess
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from runtime import ROOT,RUNS,read,write,utc,environment,install_scope,host_module

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',required=True,
        choices=('host','fidelity','self-test','smoke-variant','smoke-compare','plumbing-fail'))
    parser.add_argument('--threads',type=int,default=1);parser.add_argument('--job',required=True)
    parser.add_argument('--variant',choices=('source','low'));parser.add_argument('--parent-job');args=parser.parse_args()
    job=Path(args.job).resolve();assert job.is_relative_to(RUNS);job.mkdir(parents=True,exist_ok=True)
    install_scope();write(job/'actual_pid.json',{'PID':os.getpid(),'time':utc()})
    start=time.monotonic()
    while not(job/'monitor_ready.json').exists():
        assert time.monotonic()-start<60,'NO_MONITOR_HANDSHAKE';time.sleep(.05)
    for name in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]=str(args.threads)
    import numpy as np
    from threadpoolctl import threadpool_limits,threadpool_info
    from experiments.mfa_low_memory_fidelity_r4.fidelity import (model,initialize,synthetic,low,
        run_case,compare_case,independent_identities,boundary_stop_test,inference_contract,discrepancy)
    host=host_module();env=environment();write(job/'environment.json',env);assert env['PASS'],'EXACT_ENVIRONMENT_GATE_FAILED'
    passed=True
    with threadpool_limits(args.threads):
        assert all(p['num_threads']==args.threads for p in threadpool_info()),'THREAD_CONTROL_FAILED'
        if args.action=='host':write(job/'host_inventory.json',host.inventory())
        elif args.action=='self-test':
            write(job/'stage.json',{'stage':'deployment plumbing only; no MFA step'})
            time.sleep(1);write(job/'self_test.json',{'PASS':True,'logging':True,'PID_handshake':True,'formal_fit_calls':0})
        elif args.action=='plumbing-fail':raise RuntimeError('INTENTIONAL_PLUMBING_EXCEPTION_SENTINEL; NO_MFA_STEP')
        elif args.action=='fidelity':
            identities=independent_identities();boundary=boundary_stop_test();assert identities['PASS']and boundary['PASS']
            write(job/'independent_identities.json',identities);write(job/'stop_boundary.json',boundary)
            # Original B5A unit tests are unchanged and do not access scientific datasets.
            command=[sys.executable,'-B','-X','utf8','-m','pytest','tests/test_mfa_reference.py','-q','-p','no:cacheprovider','--basetemp',str(job/'pytest_tmp')]
            p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf8');write(job/'B5A_pytest.json',{'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr});assert p.returncode==0
            cfg=read(ROOT/'configs/sp_u12b5b_r4_synthetic.json');checks=[];allrows=[]
            for case in cfg['cases']:
                write(job/'stage.json',{'case':case['id'],'stage':'author1 / low requested threads'})
                reference=job/case['id']/'source';reference.mkdir(parents=True)
                with threadpool_limits(1):status,m,x=run_case(case,False,1,reference)
                write(reference/'status.json',status);del m,x
                path=job/case['id']/'low';path.mkdir();status,m,x=run_case(case,True,args.threads,path);write(path/'status.json',status)
                interface=inference_contract(m,x,path);write(path/'inference.json',interface);del m,x
                rows,check=compare_case(reference,path,status);check['case']=case['id'];checks.append(check);allrows+=rows
                assert check['PASS']and interface['PASS'],'FIDELITY_STOP; NO_RELAXATION_OR_HISTORY_TRUNCATION'
                print(json.dumps({'case':case['id'],'PASS':True}),flush=True)
            write(job/'fidelity.json',{'PASS':all(c['PASS']for c in checks),'atol':1e-10,'rtol':1e-9,'checks':checks,
                'comparison_rows':len(allrows),'worst_absolute':max(allrows,key=lambda x:x['max_absolute']),
                'worst_relative':max(allrows,key=lambda x:x['max_relative_with_absolute_floor']),
                'initialization_threads':1,'formal_data_reads':0,'formal_candidate_fits':0})
        elif args.action=='smoke-variant':
            c=read(HERE/'deployment_config.json')['proposed_R5_smoke'];write(job/'stage.json',{'stage':'synthetic fixture','configuration':c})
            # Only one EM step on synthetic N128, never formal N12000/15000.
            x=synthetic(c['N'],c['D'],c['K'],c['r'],c['data_seed']);m=initialize(model(c['K'],c['r'],args.variant=='low'),x,c['initialization_seed'])
            write(job/'stage.json',{'stage':'one synthetic EM step','variant':args.variant})
            begin=time.perf_counter();m._EM_per_component(x,m.PPCA);elapsed=time.perf_counter()-begin
            for n,a in [('weights',m.amps),('means',m._means),('loadings',m._lambdas),('residuals',m._psis),('covariances',m._covs),('inverse_covariances',m._inv_covs),('objective',np.array(m.logLs.sum()))]:
                assert np.isfinite(a).all();np.save(job/(n+'.npy'),a,allow_pickle=False)
            write(job/'smoke_step.json',{'PASS':True,'variant':args.variant,'seconds':elapsed,'configuration':c,'synthetic_only':True,
                'formal_fit_calls':0,'validation_scores':0,'maximum_iterations_changed':False,'steps_executed':1})
        else:
            parent=Path(args.parent_job).resolve();assert parent.is_relative_to(RUNS);checks=[]
            for name in('weights','means','loadings','residuals','covariances','inverse_covariances','objective'):
                a=np.load(parent/'low'/(name+'.npy'),mmap_mode='r',allow_pickle=False);b=np.load(parent/'source'/(name+'.npy'),mmap_mode='r',allow_pickle=False)
                # Chunk diagnostics only, never changes full-batch training.
                chunks=[discrepancy(a[i:i+1],b[i:i+1])for i in range(a.shape[0])]if a.ndim else[discrepancy(a,b)]
                checks.append({'tensor':name,'PASS':all(v['PASS']for v in chunks),'max_absolute':max(v['max_absolute']for v in chunks),
                    'max_relative':max(v['max_relative_with_absolute_floor']for v in chunks)});del a,b
            passed=all(v['PASS']for v in checks);write(job/'smoke_comparison.json',{'PASS':passed,'checks':checks,'atol':1e-10,'rtol':1e-9,'scope':'one-step full-feature synthetic resource preflight; not full-scale certification'})
            assert passed,'SYNTHETIC_FIDELITY_FAILURE'
        write(job/'worker_final.json',{'PASS':passed,'memory':host.process_memory(),'actual_threadpools':threadpool_info(),
            'formal_candidate_fits':0,'validation_scores':0,'final_retraining':0,'target_evaluation':0,'GPU_used':False})

if __name__=='__main__':
    try:main()
    except Exception:
        if '--job'in sys.argv:
            p=Path(sys.argv[sys.argv.index('--job')+1]).resolve()
            if p.is_relative_to(RUNS):write(p/'exception.json',{'time':utc(),'traceback':traceback.format_exc(),'formal_training_started':False})
        traceback.print_exc();sys.exit(1)
