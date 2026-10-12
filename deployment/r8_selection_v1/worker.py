"""R8 role-scoped child. No training/inference/target entry points."""
from pathlib import Path
import sys,os,time,argparse,traceback,subprocess
sys.dont_write_bytecode=True
import selection_core as a
def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);args=p.parse_args();job=args.job.resolve()
    a.require(job.is_relative_to(a.OUT),'WORKER_JOB_SCOPE');a.verify();request=a.read(job/'request.json')
    a.write(job/'actual_pid.json',{'PID':os.getpid(),'executable':sys.executable,'UTC':a.utc()},True)
    start=time.monotonic()
    while not(job/'monitor_ready.json').exists():a.require(time.monotonic()-start<60,'PID_HANDSHAKE_TIMEOUT');time.sleep(.02)
    try:
        if request['mode']=='toy':
            with(job/'toy.log').open('x',encoding='utf8')as f:
                code=subprocess.call([sys.executable,'-B','-X','utf8',str(a.HERE/'toy_case.py'),str(job/'toy_project'),'success'],stdout=f,stderr=subprocess.STDOUT)
            a.require(code==0,'TOY_SELECTION_FAILED')
            result={'PASS':True,'synthetic_only':True,'real_selection_executed':False,'final_model_trained':False,'target_access':0,'actual_PID':os.getpid()}
        else:
            evidence=Path(request['evidence']);archive=Path(request['archive']);mode=request['mode'];a.require(mode in('preflight','complete'),'R8_ACTION_FORBIDDEN')
            approval=Path(request['approval'])if mode=='complete'else None;instruction=Path(request['instruction'])if mode=='complete'else None
            if mode=='complete':a.authorize(approval,instruction,request['approval_sha'])
            c,_,_,_,_=a.modules();a.write(job/'environment.json',c.environment(),True)
            with c.threads(6,1)as threads:
                a.write(job/'actual_threads.json',threads,True)
                a.install_guard(job,evidence,archive,()if mode=='preflight'else(approval,instruction))
                if mode=='preflight':
                    with a.lease():result=a.preflight(evidence,archive,True);result['idle']=a.idle_processes()
                else:result=a.complete_once(job,evidence,archive,approval,instruction,request['approval_sha'])
        a.write(job/'worker_result.json',result,True)
    except BaseException as e:
        a.write(job/'exception.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'real_selection_committed':(a.R6OUT/'SELECTION_MANIFEST.json').exists()if request['mode']=='complete'else False},True);raise
if __name__=='__main__':main()
