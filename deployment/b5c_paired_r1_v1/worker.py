"""Disposable LAB inference worker, or bounded synthetic monitor fixture."""
from pathlib import Path
import argparse,os,sys,time,traceback
sys.dont_write_bytecode=True
import base as b
def main():
    a=argparse.ArgumentParser();a.add_argument('--action',choices=('evaluate','unit-probe'),required=True);a.add_argument('--job',type=Path,required=True);a.add_argument('--blas',type=int,required=True);a.add_argument('--openmp',type=int,required=True);a.add_argument('--probe',default='success');x=a.parse_args()
    b.require(x.action=='unit-probe','MENTOR_NEW_ATTEMPT_REQUIRED_R1_TECHNICAL_ONLY')
    job=x.job.resolve();b.require(job.is_relative_to(b.OUT),'NEW_B5C_JOB_ONLY');b.require((x.blas,x.openmp)==(6,1),'FROZEN_THREADS_REQUIRED')
    for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[n]='6'
    os.environ['OMP_NUM_THREADS']='1'
    from science import modules
    c,_,_,wr=modules();b.write(job/'actual_pid.json',{'PID':os.getpid(),'UTC':b.utc(),'interpreter':sys.executable},True)
    begin=time.monotonic()
    while not(job/'monitor_ready.json').exists():
        b.require(time.monotonic()-begin<60,'MONITOR_HANDSHAKE_TIMEOUT');time.sleep(.02)
    try:
        if x.action=='unit-probe':
            b.require(job.is_relative_to(b.OUT/'synthetic') and os.environ.get('R6_PERSONAL_SELFTEST')=='1','SYNTHETIC_SCOPE_REQUIRED')
            if x.probe=='exception':raise RuntimeError('INTENTIONAL_SYNTHETIC_EXCEPTION')
            if x.probe=='long':time.sleep(20)
            else:time.sleep(.5)
            result={'PASS':True,'synthetic_only':True,'formal_predictions':0,'formal_training_executed':False}
        else:
            from control import scope,environment
            b.require(job==b.OUT/'formal','CANONICAL_FORMAL_OUTPUT_REQUIRED')
            receipt=b.read(b.OUT/'FORMAL_START_CONSUMED.json')
            b.require(receipt['tool_manifest_sha256']==b.sha(b.HERE/'package_manifest.json') and receipt['contract_sha256']==b.sha(b.HERE/'contract.json'),'EXECUTION_FREEZE_CHANGED')
            scope(job,True);b.write(job/'environment.json',environment(),True)
            from science import evaluate
            with c.threads(6,1) as pools:
                b.write(job/'threads.json',pools,True);result=evaluate(job)
        result.update(native_final_memory=wr.frozen_probe().process_memory(),actual_PID=os.getpid());b.write(job/'worker_result.json',result,True)
    except BaseException as e:
        b.write(job/'exception.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},True);raise
if __name__=='__main__':main()
