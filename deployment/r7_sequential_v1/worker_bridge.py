"""Prospective timing-only entry around frozen R6 main, no model modification."""
from pathlib import Path
import sys,os,time,json,traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import r7_common as c

def main():
    c.verify();c.approval()
    require_args=sys.argv
    c.require('--action' in require_args and require_args[require_args.index('--action')+1] in ('train','validate'),'R7_ONLY_TRAIN_VALIDATE')
    job=Path(require_args[require_args.index('--job')+1]).resolve()
    c.require(job.is_relative_to(c.R6OUT/'candidates'),'R7_FORMAL_JOB_PATH_REQUIRED')
    c.require(c.read(c.OUT/'queue.json')['state']=='QUEUE_RUNNING' and (c.OUT/'execution.lock').exists() and (c.R6OUT/'pipeline.lock').exists(),'R7_QUEUE_LEASE_REQUIRED')
    r6,runner,_,_=c.r6_imports()
    # Initialize only pinned runtime/platform discovery before R6's stricter
    # scientific scope; no numerical preparation input is opened.
    r6.environment()
    import worker
    from timing_observer import TimingObserver
    observer=TimingObserver(lambda value:r6.write(job/'r7_phase.json',value))
    try:
        with observer:worker.main()
    finally:
        r6.write(job/'r7_timing.json',observer.result,exclusive=True)

if __name__=='__main__':main()
