"""Windows portable launcher. Only integrity/host/fidelity/synthetic actions."""
from pathlib import Path
import sys,os,json,time,argparse,subprocess,traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from runtime import ROOT,RUNS,read,write,sha,utc,integrity,host_module,optional,stop_process

def supervise(action,threads,job,extra=None):
    host=host_module();top=host.topology();assert 1<=threads<=top['physical_cores'],'THREADS_EXCEED_ACTUAL_PHYSICAL_CORES'
    memory=host.memory();assert memory['available_physical_bytes']>4*2**30,'SAFE_LAUNCH_HEADROOM_FAILED'
    cfg=read(HERE/'deployment_config.json');cmd=[sys.executable,'-B','-X','utf8',str(HERE/'worker.py'),
        '--action',action,'--threads',str(threads),'--job',str(job)]
    if extra:cmd+=extra
    job.mkdir(parents=True,exist_ok=True)
    state={'state':'STARTING','action':action,'threads':threads,'started_UTC':utc(),'supervisor_PID':os.getpid(),
        'limits':cfg['resource_limits'],'formal_candidate_training':0,'validation_scores':0,'target_evaluation':0}
    write(job/'status.json',state);log=(job/'worker.log').open('w',encoding='utf8');p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    actual=None;peak=commit=0;minimum=memory['available_physical_bytes'];start=time.monotonic();reason=None
    with(job/'resources.jsonl').open('w',encoding='utf8')as resources:
        while p.poll()is None:
            marker=optional(job/'actual_pid.json')
            if actual is None and marker:
                actual=marker['PID'];write(job/'monitor_ready.json',{'actual_PID':actual,'launcher_PID':p.pid})
                state.update(state='RUNNING',worker_PID=actual,launcher_PID=p.pid);write(job/'status.json',state)
            if actual is not None:
                try:m=host.process_memory(actual);s=host.memory()
                except OSError:
                    try:code=p.wait(timeout=2)
                    except subprocess.TimeoutExpired:code=None
                    if code==0 and(job/'worker_final.json').exists():break
                    reason='MEMORY_MONITOR_FAILED';break
                peak=max(peak,m['peak_working_set_bytes']);commit=max(commit,m['peak_commit_bytes']);minimum=min(minimum,s['available_physical_bytes'])
                resources.write(json.dumps({'elapsed_seconds':time.monotonic()-start,'process':m,'system':s,'stage':optional(job/'stage.json')})+'\n');resources.flush()
                if peak>=cfg['near_limit_stop']['RSS_bytes']:reason='NEAR_RSS_CAP_STOP'
                if minimum<cfg['near_limit_stop']['available_bytes']:reason='NEAR_HEADROOM_STOP'
            if actual is None and time.monotonic()-start>60:reason='PID_HANDSHAKE_TIMEOUT'
            if time.monotonic()-start>cfg['timeout_seconds']:reason='SYNTHETIC_TIMEOUT'
            if reason:break
            time.sleep(.25)
    if reason:
        if actual:stop_process(actual)
        else:p.terminate()
    code=p.wait(timeout=30);log.close();final=optional(job/'worker_final.json')
    if final:
        peak=max(peak,final['memory']['peak_working_set_bytes']);commit=max(commit,final['memory']['peak_commit_bytes'])
    success=code==0 and reason is None and final is not None and final.get('PASS',False)
    state.update(state='COMPLETE'if success else'STOPPED',completed_UTC=utc(),exit_code=code,stop_reason=reason,
        peak_RSS_bytes=peak,peak_private_commit_bytes=commit,minimum_available_physical_bytes=minimum,
        wall_seconds=time.monotonic()-start,PASS=success)
    write(job/'status.json',state)
    if not success:raise RuntimeError('WORKER_STOPPED: inspect status.json, worker.log and exception.json; no automatic retry')
    return state

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('verify','host','fidelity','self-test','smoke','status'))
    parser.add_argument('--threads',type=int,default=1);parser.add_argument('--job');parser.add_argument('--out');args=parser.parse_args()
    if args.action=='status':
        latest=read(RUNS/'latest.json');p=ROOT/latest['job'];print(json.dumps(optional(p/'status.json')or latest,indent=2));return
    check=integrity();assert check['PASS'],check
    if args.action=='verify':print(json.dumps(check,indent=2));return
    name=time.strftime('%Y%m%d_%H%M%S')+f'_{args.action}_{time.time_ns()%1000000:06d}'
    job=Path(args.job).resolve()if args.job else RUNS/name
    assert job.is_relative_to(RUNS),'JOB_PATH_MUST_BE_UNDER_RELOCATED_ROOT_LAB_RUNS'
    job.mkdir(parents=True,exist_ok=True);write(RUNS/'latest.json',{'job':job.relative_to(ROOT).as_posix(),'action':args.action,'time':utc()})
    write(job/'integrity_before.json',check)
    if args.action=='smoke':
        # Prepared R5 resource preflight candidate, not execution of an approved R5 protocol.
        # Data seeds are technical-only and disjoint from registered formal candidates.
        write(job/'status.json',{'state':'RUNNING','PASS':False,'action':'smoke','formal_training_authorized':False})
        for variant in('source','low'):
            supervise('smoke-variant',1 if variant=='source'else args.threads,job/variant,['--variant',variant])
        supervise('smoke-compare',1,job/'compare',['--parent-job',str(job)])
        write(job/'status.json',{'state':'COMPLETE','PASS':True,'action':'smoke','scope':'PROPOSED_SYNTHETIC_R5_PREFLIGHT_ONLY',
            'formal_training_authorized':False,'STOP':'MENTOR_REVIEW'})
    else:supervise(args.action,args.threads,job)
    write(job/'integrity_after.json',integrity())
    print(json.dumps({'job':job.relative_to(ROOT).as_posix(),'status':str(job/'status.json'),'STOP':'MENTOR_REVIEW'},indent=2))

if __name__=='__main__':
    try:main()
    except Exception:
        if '--job'in sys.argv:
            p=Path(sys.argv[sys.argv.index('--job')+1]).resolve()
            if p.is_relative_to(RUNS):
                write(p/'supervisor_exception.json',{'time':utc(),'traceback':traceback.format_exc(),'formal_training_started':False})
                state=optional(p/'status.json')or{}
                state.update(state='STOPPED',PASS=False,completed_UTC=utc(),supervisor_exception=True,formal_training_authorized=False)
                write(p/'status.json',state)
        traceback.print_exc();sys.exit(1)
