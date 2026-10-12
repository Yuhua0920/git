"""R7 scheduling only. Frozen R6 workers own every mathematical operation."""
import sys,os,time,csv,json,contextlib,math
from pathlib import Path
import r7_common as q

def monitor_complete(job):
    s=q.read(job/'status.json')
    q.require(s['state']=='COMPLETE' and s['PASS'] and s['exit_code']==0 and s['stop_reason'] is None,'JOB_EXIT_OR_STOP_INVALID')
    q.require(s['all_descendants_kernel_contained'] and s['actual_interpreter_PID']>0,'WINDOWS_JOB_OR_ACTUAL_PID_INVALID')
    cfg=q.contract()['resource_contract']; limits=cfg['limits']
    q.require(s['limits']==limits and s['blas_threads']==6 and s['openmp_threads']==1,'ACTUAL_RESOURCE_OR_THREADS_DRIFT')
    q.require(s['peak_RSS_bytes']<=limits['RSS_bytes'] and s['peak_private_commit_bytes']<=limits['private_commit_bytes'],'RECORDED_RESOURCE_LIMIT_EXCEEDED')
    q.require(s['minimum_available_physical_bytes']>=limits['min_available_physical_bytes'] and s['minimum_remaining_commit_bytes']>=limits['min_remaining_commit_bytes'],'RECORDED_SYSTEM_FLOOR_VIOLATION')
    count=0;last=-1.;pid_seen=False;actual_identity=None
    with (job/'resources.jsonl').open(encoding='utf8') as f:
        for line in f:
            v=json.loads(line);q.require(v['elapsed_seconds']>last,'RESOURCE_TIME_SERIES_NOT_MONOTONIC');last=v['elapsed_seconds'];count+=1
            pid_seen |= any(r['PID']==s['actual_interpreter_PID'] for r in v['process_tree']['members'])
            for r in v['process_tree']['members']:
                if r['PID']==s['actual_interpreter_PID']:actual_identity={'PID':r['PID'],'creation_time_100ns':r['creation_time_100ns']}
    q.require(count==s['resource_samples'] and count>0 and pid_seen,'RESOURCE_TIME_SERIES_INCOMPLETE')
    q.require(q.process_identity(s['actual_interpreter_PID'])!=actual_identity,'PREVIOUS_SCIENTIFIC_WORKER_STILL_RUNNING')
    for n in ('stdout.log','stderr.log','actual_pid.json','monitor_ready.json','capacity_before.json','actual_threads.json','environment.json','r7_timing.json'):
        q.require((job/n).is_file(),'MISSING_STAGE_AUDIT:'+n)
    timing=q.read(job/'r7_timing.json');q.require(timing['error'] is None,'BRIDGE_TIMING_EXCEPTION')
    q.require(math.isclose(sum(timing['exclusive_seconds'].values()),timing['wall_seconds_in_bridge'],abs_tol=1e-6),'NONEXCLUSIVE_TIMING_ACCOUNTING')
    q.require(q.read(job/'capacity_before.json')['PASS'],'STAGE_LAUNCH_CAPACITY_FAILED')
    return {'PASS':True,'status_SHA256':q.digest(job/'status.json'),'resource_SHA256':q.digest(job/'resources.jsonl'),'timing':timing}

class FormalBackend:
    def __init__(self):
        self.c,self.r,self.s,self.w=q.r6_imports();self.current_job=None
    def guard(self): return contextlib.nullcontext() # main owns the full R6 lease
    def prepare(self):
        q.approval();q.verify();self.c.approval();self.c.integrity()
        q.require((q.R6OUT/'pipeline.lock').exists(),'FULL_QUEUE_R6_LEASE_REQUIRED')
    def admit(self,row,stage):
        q.approval();q.verify(); self.c.approval()
        cap=self.w.capacity(self.w.hardware());q.require(cap['PASS'],'NEXT_STAGE_LAUNCH_CAPACITY_FAILED:'+str(cap['reasons']))
        q.write(q.OUT/'admission'/ (row['name']+'_'+stage+'.json'),{'PASS':True,'capacity':cap,'UTC':q.utc()},exclusive=True)
        q.require(not (q.R6OUT/'candidates'/row['name']/stage).exists(),'EXISTING_ATTEMPT_NO_RETRY')
        if stage=='validate':self.r.valid_candidate(row,False)
        else:
            for previous in q.contract()['candidate_order'][:q.contract()['candidate_order'].index(row)]:self.r.valid_candidate(previous)
    def stage(self,row,stage):
        from r7_supervisor import run_worker
        job=q.R6OUT/'candidates'/row['name']/stage;job.mkdir(parents=True,exist_ok=False);self.current_job=job
        self.c.write(job/'execution_context.json',{'action':stage,'candidate':row,'package_manifest_sha256':self.c.sha(q.R6/'package_manifest.json'),
                 'resource_proposal_sha256':self.c.sha(q.R6/'resource_amendment.json'),'approval':self.c.approval(),
                 'environment':self.c.environment(),'UTC':q.utc(),'R7_bindings':q.bindings()})
        self.c.write(q.R6OUT/'PHASE_A_STATUS.json',{'state':'RUNNING','action':stage,'candidate':row['name'],'UTC':q.utc(),'scheduling':'R7_SEQUENTIAL'})
        timeout=q.contract()['resource_contract']['worker_seconds'][stage]
        if stage=='validate':
            # Preserve R6's stricter frozen train+validation combined budget.
            timeout-=q.read(job.parent/'train/status.json')['wall_seconds'];q.require(timeout>0,'R6_COMBINED_CANDIDATE_24H_EXHAUSTED')
        try:state,result=run_worker(stage,job,blas=6,openmp=1,extra=['--candidate',row['name']],timeout=timeout)
        except BaseException as error:
            status=q.read(job/'status.json') if (job/'status.json').exists() else {}
            detail=q.read(job/'exception.json') if (job/'exception.json').exists() else {}
            raise RuntimeError(stage.upper()+': '+str(status.get('stop_reason') or detail.get('message') or error)) from error
        result_name='training_result.json' if stage=='train' else 'validation_result.json'
        state.update(worker_result_sha256=self.c.sha(job/'worker_result.json'),scientific_result_sha256=self.c.sha(job/result_name))
        self.c.write(job/'status.json',state)
    def verify_stage(self,row,stage):
        job=q.R6OUT/'candidates'/row['name']/stage;monitor_complete(job)
        q.require(q.read(job/'execution_context.json')['R7_bindings']==q.bindings(),'R7_STAGE_BINDING_DRIFT')
        if stage=='train':
            t=self.r.valid_candidate(row,False)
            q.require(t['parameter_validity']['PASS'] and t['parameter_validity']['full_covariance_check'] and t['serialization_roundtrip'] and t['all_components_smoke']['PASS'],'MODEL_LEGALITY_OR_ROUNDTRIP_FAILED')
            with (job/'iteration_history.jsonl').open(encoding='utf8') as f: history=[json.loads(line) for line in f]
            q.require(len(history)==t['iterations'] and [v['iteration'] for v in history]==list(range(1,t['iterations']+1)) and all(math.isfinite(v['author_log_likelihood']) for v in history),'TRAINING_HISTORY_INCOMPLETE')
            q.require(t['convergence'] in ('CONVERGED','MAX_ITER_REACHED'),'ILLEGAL_CONVERGENCE_STATUS')
        else:self.r.valid_candidate(row)
    def verify_candidate(self,row):
        result,_=self.r.valid_candidate(row)
        # Controller audits integrity and coverage; score never drives schedule.
        return {'PASS':True,'prediction_count':result['prediction_count'],'cells':len(result['cells']),
                'model_manifest_SHA256':result['model_manifest_sha256'],
                'validation_result_SHA256':q.digest(q.R6OUT/'candidates'/row['name']/'validate/validation_result.json'),
                'seed':row['seed'],'score_used_to_change_schedule':False}
    def finish(self):
        q.approval()
        from model_ops import total_cardinality
        frames=[]
        for row in q.contract()['candidate_order']:
            _,f=self.r.valid_candidate(row);frames.extend(f)
        q.require(total_cardinality(frames)['scored_frame_predictions']==32400,'NINE_CANDIDATE_TOTAL_CARDINALITY_FAILED')
        q.require(not any((q.R6OUT/n).exists() for n in ('SELECTION_MANIFEST.json','final','FINAL_MODEL_MANIFEST.json')),'UNAUTHORIZED_LATER_STAGE')
        self.c.write(q.R6OUT/'PHASE_A_STATUS.json',{'state':'COMPLETE','complete_candidate_validations':9,'STOP_AFTER_NINE':True,'UTC':q.utc()})
    def abort(self,reason):
        # No retry, no lock deletion of an unrelated controller, no checkpoint load.
        if self.current_job:
            active=self.s.optional(q.R6OUT/'active_worker.json')
            if active and active['job']==self.current_job.relative_to(q.ROOT).as_posix():self.w.stop_named_job(active['JobObject_name'])
        self.c.write(q.R6OUT/'STOP.json',{'UTC':q.utc(),'reason':reason,'source':'R7_QUEUE_ABORT','no_retry':True})
        q.write(q.OUT/'STOP.json',{'UTC':q.utc(),'reason':reason,'no_retry':True})

def install_parent_firewall():
    """Parent can hash authorized NPY bytes via frozen c.sha, never decode them."""
    c,_,_,_=q.r6_imports();spec=q.read(q.R6/'input_contract.json')
    permitted={(q.ROOT/n).resolve() for n in spec['frozen_hashes']}
    permitted |= {(q.ROOT/n).resolve() for n in spec['metadata_allowlist']+list(spec['lab_prerequisite_hashes'])}
    permitted.add(q.ROOT/'migration_manifest.json')
    runtime={Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()}
    formats={'.npy','.npz','.mat','.pkl','.pickle','.h5','.hdf5'}
    def check(path,writing=False):
        p=Path(path).resolve()
        if writing:
            allowed=p==q.R6OUT or p.is_relative_to(q.OUT) or p.is_relative_to(q.R6OUT/'candidates') or p in {q.R6OUT/n for n in ('pipeline.lock','active_worker.json','active_worker.json.tmp','STOP.json','STOP.json.tmp','PHASE_A_STATUS.json','PHASE_A_STATUS.json.tmp')}
            q.require(allowed,'R7_HISTORICAL_WRITE_FORBIDDEN:'+str(p));return
        if any(p.is_relative_to(root) for root in runtime):q.require(p.suffix.lower() not in formats,'RUNTIME_NUMERICAL_DATA_DENIED');return
        if p.is_relative_to(q.HERE) or p.is_relative_to(q.R6) or p.is_relative_to(q.OUT):q.require(p.suffix.lower() not in formats,'DEPLOYMENT_NUMERICAL_DATA_DENIED');return
        if p.is_relative_to(q.R6OUT):q.require(p.suffix.lower() not in formats or c._opaque,'PARENT_MODEL_VALUES_DENIED');return
        if p in permitted:
            q.require(p.suffix.lower() not in formats or c._opaque,'PARENT_PREPARATION_VALUES_DENIED');return
        raise PermissionError('R7_READ_ALLOWLIST_REJECTED:'+str(p))
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            if os.fsdecode(args[0]).lower()==os.devnull.lower():return
            mode,flags=args[1:3];writing=isinstance(mode,str) and any(v in mode for v in 'wax+') or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            check(os.fsdecode(args[0]),writing)
        elif event in ('os.remove','os.rename','os.rmdir','os.mkdir','os.utime'):
            for raw in args[:2] if event=='os.rename' else args[:1]:
                if isinstance(raw,(str,bytes,os.PathLike)):check(os.fsdecode(raw),True)
    sys.addaudithook(hook)
