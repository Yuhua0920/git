"""Only a final fit or a bounded synthetic launch handshake."""
from pathlib import Path
import argparse,os,sys,time,traceback
sys.dont_write_bytecode=True
import final_once as a

def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('final','unit-probe'),required=True)
    p.add_argument('--job',type=Path,required=True);p.add_argument('--candidate');p.add_argument('--blas',type=int,required=True);p.add_argument('--openmp',type=int,required=True)
    args=p.parse_args();job=args.job.resolve();a.require(args.blas==6 and args.openmp==1,'FINAL_FROZEN_THREADS_REQUIRED')
    c,w,_,_,wr=a.modules()
    if args.action=='final':
        a.require(job==a.FINAL and args.candidate=='K1_r16','ONLY_CANONICAL_SELECTED_FINAL_ALLOWED')
        a.check_confirmation(a.read(job/'operator_confirmation.json'))
    else:
        a.require(job.is_relative_to(c.LOCAL/'final_once_synthetic')and os.environ.get('R6_PERSONAL_SELFTEST')=='1','TOY_SCOPE_REQUIRED')
    a.write(job/'actual_pid.json',{'PID':os.getpid(),'interpreter':sys.executable,'UTC':a.utc()},True)
    begin=time.monotonic()
    while not(job/'monitor_ready.json').exists():
        a.require(time.monotonic()-begin<60,'MONITOR_HANDSHAKE_TIMEOUT');time.sleep(.02)
    try:
        if args.action=='final':
            # Exact authenticated SourceFileLoader code, never stale .pyc.
            a.guard(job)
            row=c.candidate('K1_r16');a.require(row==a.contract()['selected'],'FINAL_REGISTRY_DRIFT')
            a.write(job/'environment.json',c.environment(),True)
            with c.threads(6,1)as pools:
                a.write(job/'actual_threads.json',pools,True)
                result=a.fit_callable(w.train)(job,row,True)
        else:
            time.sleep(.5)
            result={'PASS':True,'synthetic_only':True,'target_access':0,'real_training_executed':False}
        result.update(native_final_memory=wr.frozen_probe().process_memory(),actual_PID=os.getpid())
        a.write(job/'worker_result.json',result,True)
    except BaseException as e:
        a.write(job/'exception.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},True);raise
if __name__=='__main__':main()
