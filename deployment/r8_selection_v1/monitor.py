"""Windows PID/JobObject monitor using unchanged frozen resource primitives."""
from pathlib import Path
import sys,os,time,subprocess,uuid,json,traceback
import selection_core as a
def optional(p):
    try:return a.read(p)
    except(FileNotFoundError,PermissionError):return None
def run(job,request):
    *_,wr,q=a.modules();limits=dict(a.cfg()['resources']['limits'])
    # Selection is metadata/opaque hashing, no EM or preparation-array decoding.
    # Original private/RSS limits are retained; no training capacity claim is made.
    container=wr.Job('Local\\SP_U12B5_R8_SELECTION_'+uuid.uuid4().hex,limits['private_commit_bytes']);child=None;started=time.monotonic();reason=None
    state={'state':'RUNNING','PASS':False,'mode':request['mode'],'JobObject_name':container.name if hasattr(container,'name')else None,'started_UTC':a.utc(),'actual_interpreter_PID':None,'real_selection_executed':False,'final_model_trained':False,'target_access':0,'automatic_retry':False,'limits':limits}
    # wr.Job has no public name guarantee; retain exactly the supplied kernel name.
    state['JobObject_name']=getattr(container,'name',state['JobObject_name'])
    a.write(job/'status.json',state);env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    env.update(OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6',NUMEXPR_NUM_THREADS='6',OMP_NUM_THREADS='1');(job/'temporary').mkdir();env['TEMP']=env['TMP']=str(job/'temporary')
    count=0;peak_rss=peak_commit=0
    try:
        with(job/'stdout.log').open('x',encoding='utf8')as out,(job/'stderr.log').open('x',encoding='utf8')as err,(job/'resources.jsonl').open('x',encoding='utf8')as resources:
            child=subprocess.Popen([sys.executable,'-B','-X','utf8',str(a.HERE/'worker.py'),'--job',str(job)],cwd=a.ROOT,env=env,stdout=out,stderr=err);container.assign(child.pid);state['venv_launcher_PID']=child.pid
            while child.poll()is None:
                marker=optional(job/'actual_pid.json')
                if marker and state['actual_interpreter_PID']is None:
                    actual=marker['PID']
                    if actual not in container.members():container.assign(actual)
                    a.require(actual in container.members(),'ACTUAL_WORKER_NOT_KERNEL_CONTAINED');state['actual_interpreter_PID']=actual
                    a.write(job/'monitor_ready.json',{'actual_PID':actual,'kernel_job_bound':True},True)
                tree=container.sample();system=wr.frozen_probe().memory();elapsed=time.monotonic()-started
                count+=1;peak_rss=max(peak_rss,tree['RSS_bytes']);peak_commit=max(peak_commit,tree['private_commit_bytes'])
                resources.write(json.dumps({'elapsed_seconds':elapsed,'process_tree':tree,'system':system})+'\n');resources.flush()
                reason='USER_STOP'if(a.OUT/'STOP_REQUESTED.json').exists()else'RSS_LIMIT'if peak_rss>limits['RSS_bytes']else'PRIVATE_COMMIT_LIMIT'if peak_commit>limits['private_commit_bytes']else'TIME_LIMIT'if elapsed>7200 else None
                if state['actual_interpreter_PID']is None and elapsed>60:reason='ACTUAL_PID_HANDSHAKE_TIMEOUT'
                if reason:container.kill();break
                state.update(resource_samples=count,peak_RSS_bytes=peak_rss,peak_private_commit_bytes=peak_commit);a.write(job/'status.json',state);time.sleep(.1)
            code=child.wait(timeout=30);result=optional(job/'worker_result.json');passed=not reason and code==0 and result and result['PASS'] and state['actual_interpreter_PID'] is not None and count>0
            state.update(state='COMPLETE'if passed else'STOP'if reason else'FAILED',PASS=bool(passed),exit_code=code,stop_reason=reason,completed_UTC=a.utc(),resource_samples=count,wall_seconds=time.monotonic()-started,all_descendants_kernel_contained=True,actual_interpreter_not_assumed_launcher=True)
            if result:state.update({k:result[k]for k in('real_selection_executed','final_model_trained','target_access')if k in result})
            a.write(job/'status.json',state);return state
    except BaseException as e:
        if child and child.poll()is None:container.kill();child.wait(timeout=30)
        state.update(state='FAILED',PASS=False,stop_reason=str(e));a.write(job/'status.json',state)
        a.write(job/'monitor_exception.json',{'message':str(e),'traceback':traceback.format_exc()},True);raise
    finally:container.close()
