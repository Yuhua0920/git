"""EXPERIMENTAL_NOT_FORMAL_BASELINE. Actual-PID monitored synthetic worker."""
from pathlib import Path
import os,sys,json,time,argparse,traceback
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.sp_u12b5b_r4_common import firewall,read,write,csvwrite
firewall()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--role',choices=('parity','measure','inventory'),required=True);parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--optimized',action='store_true');parser.add_argument('--reference');parser.add_argument('--kind');args=parser.parse_args()
    cfg=read(args.config);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    # Use frozen R3 atomic metadata helper only; no R3 runner/import side effects.
    sys.path.insert(0,str(ROOT/'experiments/mfa_low_memory_prototype/portable'))
    from runner import write as atomic_write
    atomic_write(out/'actual_pid.json',{'actual_PID':os.getpid()})
    while not(out/'monitor_ready.json').exists():time.sleep(.02)
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]=str(args.threads)
    sys.path.append(str(ROOT/'.venv-gmm/Lib/site-packages'))
    import numpy as np
    from threadpoolctl import threadpool_limits,threadpool_info
    from host_inventory import process_memory,inventory
    from experiments.mfa_low_memory_fidelity_r4.fidelity import (run_case,compare_case,
        independent_identities,boundary_stop_test,inference_contract)
    with threadpool_limits(args.threads):
        assert all(p['num_threads']==args.threads for p in threadpool_info())
        write(out/'environment.json',{'threads':args.threads,'pools':threadpool_info(),'Python':sys.version,'NumPy':np.__version__})
        if args.role=='parity':
            identities=independent_identities();stopbranch=boundary_stop_test()
            write(out/'independent_identities.json',identities);write(out/'stop_branch_sentinel.json',stopbranch)
            assert identities['PASS'] and stopbranch['PASS'],'INDEPENDENT_IDENTITY_OR_STOP_BRANCH_FAILURE'
            checks=[];rows=[];inference=[]
            for case in cfg['cases']:
                path=out/case['id'];path.mkdir();atomic_write(out/'current_stage.json',{'case':case['id'],'role':args.role})
                status,m,x=run_case(case,args.optimized,args.threads,path);write(path/'status.json',status)
                interface=inference_contract(m,x,path);write(path/'inference.json',interface);inference.append(interface)
                if args.reference:
                    rec,check=compare_case(Path(args.reference)/case['id'],path,status);rows+=rec;checks.append({'case':case['id'],**check})
                del m,x
                import gc;gc.collect()
                print(json.dumps({'case':case['id'],'iterations':status['iterations'],'threads':args.threads,'comparison':None if not checks else checks[-1]['PASS']}),flush=True)
            if rows:csvwrite(out/'comparisons.csv',rows)
            write(out/'parity.json',{'PASS':all(c['PASS'] for c in checks) and all(i['PASS'] for i in inference),
                'checks':checks,'inference':inference,'threads':args.threads,'optimized':args.optimized,
                'comparison_scope':'reference1 vs frozen R3 low-memory; no parameter/state alignment; no history truncation'})
        elif args.role=='measure':
            from experiments.mfa_low_memory_fidelity_r4.measure import measure
            atomic_write(out/'current_stage.json',{'kind':args.kind,'implementation':'low_memory' if args.optimized else 'original'})
            value=measure(args.kind,args.optimized,process_memory);write(out/'measurements.json',value)
        else:write(out/'host_inventory.json',inventory())
        atomic_write(out/'worker_final.json',{'memory':process_memory(),'pools':threadpool_info(),'formal_candidate_fits':0,'validation_scores':0,'target_reads':0})
    
if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
