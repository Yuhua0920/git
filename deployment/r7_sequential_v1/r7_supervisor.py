"""R7 bridge entry; R6 supervision copied prospectively, mathematical worker unchanged."""
from pathlib import Path
import os,sys,time,json,subprocess,uuid,traceback
import r7_common as q
from common import protocol, scientific_counters
HERE, ROOT, OUT = q.HERE, q.ROOT, q.OUT
read, utc = q.read, q.utc

def write(path, value):
    if Path(path).resolve().is_relative_to(q.R6OUT.resolve()):
        from common import write as r6_write
        return r6_write(path, value)
    return q.write(path, value)
from windows_resources import Job,frozen_probe,exceeded,capacity,hardware

def optional(path):
    try:return read(path)
    except(FileNotFoundError,PermissionError):return None

def run_worker(action,job,blas=1,openmp=1,extra=None,local=False,local_limits=None,timeout=None):
    assert job.resolve().is_relative_to(OUT/'technical') or job.resolve().is_relative_to(q.R6OUT/'candidates')
    cfg=protocol();limits=local_limits or cfg['limits']
    if local:
        assert action=='unit-probe','PERSONAL_FORMAL_EXECUTION_FORBIDDEN'
        assert os.environ.get('R7_SYNTHETIC_TEST')=='1'
    else:
        check=capacity(hardware());write(job/'capacity_before.json',check);assert check['PASS'],check
    job.mkdir(parents=True,exist_ok=True);start=time.monotonic();samples=0;peak_RSS=peak_commit=0
    minimum_RAM=minimum_commit=None;max_members=0;actual=None;reason=None;code=None;worker_result=None
    native_name='Local\\SP_U12B5_R7_'+uuid.uuid4().hex
    container=Job(native_name,limits['private_commit_bytes']);p=None
    state={'state':'RUNNING','PASS':False,'action':action,'started_UTC':utc(),'supervisor_PID':os.getpid(),
           'JobObject_name':native_name,'blas_threads':blas,'openmp_threads':openmp,'limits':limits,**scientific_counters()}
    active_path=job/'active_worker.json' if local else q.R6OUT/'active_worker.json'
    stop_path=job/'STOP_REQUESTED.json' if local else q.R6OUT/'STOP_REQUESTED.json'
    write(job/'status.json',state);write(active_path,{'job':job.relative_to(ROOT).as_posix(),'JobObject_name':native_name})
    env=os.environ.copy();env['TEMP']=env['TMP']=str(job/'temporary');(job/'temporary').mkdir(exist_ok=True)
    env['PYTHONDONTWRITEBYTECODE']='1';env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    command=[sys.executable,'-B','-X','utf8',str(HERE/('toy_worker.py' if local else 'worker_bridge.py')),'--action',action,'--job',str(job),'--blas',str(blas),'--openmp',str(openmp)]
    if extra:command+=extra
    timeout=timeout or cfg['worker_seconds'].get(action,1800)
    previous_cpu=0.;previous_time=start;deltas=[]
    try:
        with(job/'stdout.log').open('w',encoding='utf8')as stdout,(job/'stderr.log').open('w',encoding='utf8')as stderr,(job/'resources.jsonl').open('w',encoding='utf8')as stream:
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            container.assign(p.pid);state['venv_launcher_PID']=p.pid
            while p.poll()is None:
                marker=optional(job/'actual_pid.json')
                if actual is None and marker:
                    actual=int(marker['PID'])
                    if actual not in container.members():container.assign(actual)
                    members=container.members();assert actual in members,'INTERPRETER_NOT_IN_KERNEL_JOB'
                    state['actual_interpreter_PID']=actual;write(job/'status.json',state)
                    write(job/'monitor_ready.json',{'actual_PID':actual,'venv_launcher_PID':p.pid,'kernel_job_bound':True})
                tree=container.sample();system=frozen_probe().memory();now=time.monotonic()
                sample={'elapsed_seconds':now-start,'UTC':utc(),'process_tree':tree,'system':system,'phase':optional(job/'phase.json')}
                dt=now-previous_time;cpu_delta=max(0,tree['CPU_seconds']-previous_cpu)
                sample['CPU_utilization_percent_one_core_units']=100*cpu_delta/dt if dt>0 else None
                sample['CPU_utilization_percent_20_logical_threads']=sample['CPU_utilization_percent_one_core_units']/20 if dt>0 else None
                previous_cpu=tree['CPU_seconds'];previous_time=now;deltas.append(dt)
                samples+=1;max_members=max(max_members,len(tree['members']))
                peak_RSS=max(peak_RSS,tree['RSS_bytes'],max((v['peak_RSS_bytes']for v in tree['members']),default=0))
                peak_commit=max(peak_commit,tree['private_commit_bytes'],max((v['native_peak_commit_bytes']for v in tree['members']),default=0))
                minimum_RAM=system['available_physical_bytes']if minimum_RAM is None else min(minimum_RAM,system['available_physical_bytes'])
                minimum_commit=system['commit_available_bytes']if minimum_commit is None else min(minimum_commit,system['commit_available_bytes'])
                stream.write(json.dumps(sample)+'\n');stream.flush()
                reason=exceeded(tree,system,limits)
                if stop_path.exists():reason='USER_STOP_REQUESTED'
                if actual is None and now-start>60:reason='ACTUAL_PID_HANDSHAKE_TIMEOUT'
                if now-start>timeout:reason='AUDITABLE_TECHNICAL_TIME_BUDGET_EXCEEDED'
                if reason:container.kill();break
                time.sleep(cfg['sample_seconds'])
            code=p.wait(timeout=30)
        worker_result=optional(job/'worker_result.json')
        exception=optional(job/'exception.json')
        if exception and exception.get('type')in('MemoryError','_ArrayMemoryError'):
            reason='WORKER_ALLOCATION_FAILED_WITH_KERNEL_COMMIT_CAP; inspect exception; exact allocation cause not inferred'
        if worker_result:
            native=worker_result['native_final_memory'];peak_RSS=max(peak_RSS,native['peak_working_set_bytes']);peak_commit=max(peak_commit,native['peak_commit_bytes'])
            if peak_RSS>limits['RSS_bytes']:reason='NATIVE_FINAL_PEAK_RSS_EXCEEDED_LIMIT'
            if peak_commit>limits['private_commit_bytes']:reason='NATIVE_FINAL_PEAK_COMMIT_EXCEEDED_LIMIT'
        success=code==0 and reason is None and worker_result is not None and worker_result.get('PASS',False)and samples>0 and actual is not None
        state.update(state='COMPLETE'if success else'STOP'if reason else'FAILED',PASS=success,completed_UTC=utc(),exit_code=code,
            stop_reason=reason,peak_RSS_bytes=peak_RSS,peak_private_commit_bytes=peak_commit,
            minimum_available_physical_bytes=minimum_RAM,minimum_remaining_commit_bytes=minimum_commit,
            actual_process_tree_max_members=max_members,resource_samples=samples,sample_target_seconds=.1,
            sample_max_observed_gap_seconds=max(deltas,default=0),wall_seconds=time.monotonic()-start,
            all_descendants_kernel_contained=True,monitor_not_assumed_venv_launcher_PID=True)
        state['CPU_utilization_recorded_in']='resources.jsonl: one-core units and percent of 20 logical threads'
        write(job/'status.json',state)
        if not success:raise RuntimeError('WORKER_FAILED_OR_STOPPED; inspect original logs; NO_AUTOMATIC_RETRY')
        return state,worker_result
    except Exception:
        if p and p.poll()is None:
            # The owned child is still in its handshake if Job assignment failed.
            container.kill()
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
        if state.get('state')=='RUNNING':
            state.update(state='FAILED',PASS=False,completed_UTC=utc(),stop_reason='SUPERVISOR_OR_MONITOR_FAILURE')
            write(job/'status.json',state)
        write(job/'supervisor_exception.json',{'traceback':traceback.format_exc(),**scientific_counters()});raise
    finally:container.close()
