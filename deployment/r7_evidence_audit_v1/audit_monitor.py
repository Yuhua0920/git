"""New audit-only JobObject/actual-PID monitor; frozen Windows primitives reused."""
from pathlib import Path
import os,sys,time,subprocess,uuid,traceback
import audit_core as a
def optional(p):
    try:return a.read(p)
    except(FileNotFoundError,PermissionError):return None
def alive_identity(wr,identity):
    import ctypes
    if not identity:return False
    h=wr.K.OpenProcess(0x1000,False,int(identity['PID']))
    if not h:return False
    try:
        code=wr.w.DWORD();created,ended,kernel,user=wr.FILETIME(),wr.FILETIME(),wr.FILETIME(),wr.FILETIME()
        a.require(wr.K.GetExitCodeProcess(h,ctypes.byref(code)),'PROCESS_STATUS_QUERY_FAILED')
        if code.value!=259:return False
        a.require(wr.K.GetProcessTimes(h,ctypes.byref(created),ctypes.byref(ended),ctypes.byref(kernel),ctypes.byref(user)),'PID_CREATION_QUERY_FAILED')
        return ((created.high<<32)|created.low)==identity['creation_time_100ns']
    finally:wr.K.CloseHandle(h)
def idle_processes(cfg):
    _,_,wr=a.original_modules();identities=[]
    q=a.read(a.ROOT/'lab_runs/r7_sequential/queue.json');identities.append(q['controller_identity'])
    for row in cfg['candidate_order']:
        for action in ('train','validate'):
            base=a.ROOT/'lab_runs/r6_formal/candidates'/row['name']/action;s=a.read(base/'status.json');actual=s['actual_interpreter_PID'];creation=None
            with(base/'resources.jsonl').open(encoding='utf8')as f:
                for line in f:
                    import json
                    for x in json.loads(line)['process_tree']['members']:
                        if x['PID']==actual:creation=x['creation_time_100ns'];break
                    if creation is not None:break
            a.require(creation is not None,'NO_HISTORICAL_INTERPRETER_IDENTITY');identities.append({'PID':actual,'creation_time_100ns':creation})
    a.require(not any(alive_identity(wr,x)for x in identities),'SCIENTIFIC_PROCESS_STILL_ALIVE');return identities
def run(job,mode):
    import json
    _,_,wr=a.original_modules();cfg=a.inputs();limits=dict(cfg['resources']['limits']);real=mode=='lab'
    if real:
        a.old_is_idle(a.ROOT,cfg);a.write(job/'idle_processes.json',idle_processes(cfg),True)
        cap=wr.capacity(wr.hardware());a.write(job/'capacity_before.json',cap,True);a.require(cap['PASS'],'LAB_CAPACITY_NOT_QUALIFIED')
    else:
        # Explicit technical fixture; real laboratory thresholds never amended.
        limits['min_available_physical_bytes']=limits['min_remaining_commit_bytes']=0
    name='Local\\SP_U12B5_R7_EVIDENCE_'+uuid.uuid4().hex;container=wr.Job(name,limits['private_commit_bytes']);p=None;start=time.monotonic();reason=None;actual=None
    samples=0;rss=commit=0;ram=available=None;previous=0.;gap=0.
    state={'state':'RUNNING','PASS':False,'mode':mode,'started_UTC':a.utc(),'supervisor_PID':os.getpid(),'JobObject_name':name,'limits':limits,'blas_threads':6,'openmp_threads':1,'lab_models_verified':0}
    a.write(job/'status.json',state)
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    (job/'temporary').mkdir();env['TEMP']=env['TMP']=str(job/'temporary')
    try:
        with(job/'stdout.log').open('x',encoding='utf8')as out,(job/'stderr.log').open('x',encoding='utf8')as err,(job/'resources.jsonl').open('x',encoding='utf8')as stream:
            p=subprocess.Popen([sys.executable,'-B','-X','utf8',str(a.HERE/'audit_worker.py'),'--job',str(job),'--mode',mode],cwd=a.ROOT,env=env,stdout=out,stderr=err);container.assign(p.pid);state['venv_launcher_PID']=p.pid
            while p.poll()is None:
                marker=optional(job/'actual_pid.json')
                if marker and actual is None:
                    actual=marker['PID']
                    if actual not in container.members():container.assign(actual)
                    a.require(actual in container.members(),'ACTUAL_AUDIT_INTERPRETER_NOT_CONTAINED');state['actual_interpreter_PID']=actual
                    a.write(job/'monitor_ready.json',{'actual_PID':actual,'venv_launcher_PID':p.pid,'kernel_job_bound':True},True);a.write(job/'status.json',state)
                tree=container.sample();system=wr.frozen_probe().memory();elapsed=time.monotonic()-start;gap=max(gap,elapsed-previous);previous=elapsed;samples+=1
                rss=max(rss,tree['RSS_bytes'],max((x['peak_RSS_bytes']for x in tree['members']),default=0));commit=max(commit,tree['private_commit_bytes'],max((x['native_peak_commit_bytes']for x in tree['members']),default=0))
                ram=system['available_physical_bytes']if ram is None else min(ram,system['available_physical_bytes']);available=system['commit_available_bytes']if available is None else min(available,system['commit_available_bytes'])
                stream.write(json.dumps({'UTC':a.utc(),'elapsed_seconds':elapsed,'process_tree':tree,'system':system})+'\n');stream.flush()
                reason=wr.exceeded(tree,system,limits)
                if(job/'STOP_AUDIT_REQUESTED.json').exists():reason='USER_STOP_AUDIT_ONLY'
                if actual is None and elapsed>60:reason='ACTUAL_INTERPRETER_HANDSHAKE_TIMEOUT'
                if elapsed>(86400 if real else 600):reason='AUDIT_TIME_LIMIT'
                if reason:container.kill();break
                time.sleep(.1)
            code=p.wait(timeout=30)
        result=optional(job/'worker_result.json')
        if result:
            n=result['native_final_memory'];rss=max(rss,n['peak_working_set_bytes']);commit=max(commit,n['peak_commit_bytes'])
            if rss>limits['RSS_bytes']or commit>limits['private_commit_bytes']:reason='NATIVE_AUDIT_PEAK_LIMIT_EXCEEDED'
        ok=code==0 and reason is None and result is not None and result['PASS']and actual is not None and samples>0
        state.update(state='COMPLETE'if ok else'STOP'if reason else'FAILED',PASS=ok,completed_UTC=a.utc(),exit_code=code,stop_reason=reason,
          peak_RSS_bytes=rss,peak_private_commit_bytes=commit,minimum_available_physical_bytes=ram,minimum_remaining_commit_bytes=available,
          resource_samples=samples,sample_target_seconds=.1,sample_max_observed_gap_seconds=gap,wall_seconds=time.monotonic()-start,
          all_descendants_kernel_contained=True,actual_interpreter_not_assumed_launcher=True,lab_models_verified=9 if ok and real else 0)
        a.write(job/'status.json',state);return state
    except BaseException as e:
        if p and p.poll()is None:
            container.kill()
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
        state.update(state='FAILED',PASS=False,stop_reason=str(e),completed_UTC=a.utc());a.write(job/'status.json',state)
        a.write(job/'supervisor_exception.json',{'traceback':traceback.format_exc()},True);raise
    finally:container.close()
