"""One prospective attempt, fixed order, verified completion, no retries."""
from pathlib import Path
import os,time,traceback
from r7_common import write,read,utc,require,lease,process_identity

def run_queue(directory, rows, backend):
    directory=Path(directory)
    with lease(directory):
        require(not (directory/'queue.json').exists(),'DUPLICATE_QUEUE_OR_PRIOR_ATTEMPT')
        state={'state':'QUEUE_RUNNING','controller_PID':os.getpid(),'controller_identity':process_identity(os.getpid()),'started_UTC':utc(),
               'current_candidate':None,'current_stage':'preflight','completed_candidates':[],
               'remaining_candidates':[r['name'] for r in rows],'last_checkpoint_status':'NONE',
               'checkpoint_resume_allowed':False,'automatic_retry':False,'STOP_reason':None,'elapsed_seconds':0}
        write(directory/'queue.json',state,exclusive=True); start=time.monotonic()
        try:
            with backend.guard():
                backend.prepare()
                for row in rows:
                    state['current_candidate']=row['name']
                    for stage in ('train','validate'):
                        backend.admit(row,stage)
                        require(not (directory/'STOP_REQUESTED.json').exists(),'USER_STOP_REQUESTED')
                        state.update(current_stage=stage,elapsed_seconds=time.monotonic()-start,last_checkpoint_status='NO_FAILED_CHECKPOINT_REUSE')
                        write(directory/'queue.json',state)
                        backend.stage(row,stage)
                        backend.verify_stage(row,stage)
                        require(not (directory/'STOP_REQUESTED.json').exists(),'USER_STOP_REQUESTED')
                        state['last_checkpoint_status']=stage.upper()+'_VERIFIED_COMPLETE'
                        write(directory/'queue.json',state)
                    receipt=backend.verify_candidate(row)
                    write(directory/'completed'/ (row['name']+'.json'),{'PASS':True,'candidate':row,'receipt':receipt,'UTC':utc()},exclusive=True)
                    state['completed_candidates'].append(row['name']);state['remaining_candidates'].pop(0)
                    write(directory/'queue.json',state)
                require(len(state['completed_candidates'])==9,'NINE_CANDIDATES_REQUIRED')
                backend.finish()
                state.update(state='QUEUE_COMPLETE',completion='QUEUE_COMPLETE_9_OF_9',current_stage='STOP_AFTER_NINE',elapsed_seconds=time.monotonic()-start,completed_UTC=utc())
                write(directory/'completion.json',{'PASS':True,**state,'selection_performed':False,'final_fit_performed':False,'target_access':0},exclusive=True)
                write(directory/'queue.json',state)
                return state
        except BaseException as e:
            reason=str(e)
            # Conservatively stop the entire attempt, including audit failures.
            state.update(state='QUEUE_STOPPED' if isinstance(e,(RuntimeError,AssertionError)) else 'QUEUE_FAILED',
                         completion='QUEUE_STOPPED_AT_CANDIDATE',STOP_reason=reason,elapsed_seconds=time.monotonic()-start,completed_UTC=utc())
            try: backend.abort(reason)
            except BaseException as stop_error: state['abort_error']=repr(stop_error)
            write(directory/'failure.json',{'PASS':False,**state,'traceback':traceback.format_exc(),'no_retry':True},exclusive=True)
            write(directory/'queue.json',state)
            raise
