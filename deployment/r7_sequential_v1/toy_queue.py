"""Real Windows process/JobObject queue exercises with tiny synthetic models."""
from pathlib import Path
import sys,os,json,contextlib,time,argparse
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import r7_common as q
from queue_engine import run_queue

class SyntheticBackend:
    def __init__(self,folder,fault='none'):
        self.folder=Path(folder);self.fault=fault;self.started=[];self.results=[];self.current=None
    def guard(self):return contextlib.nullcontext()
    def prepare(self):
        q.verify();q.require(self.folder.is_relative_to(q.OUT/'technical'),'TOY_FOLDER_ONLY')
        q.require(not (q.R6OUT/'pipeline.lock').exists() and not (q.R6OUT/'candidates').exists() and not (q.OUT/'queue.json').exists(),'TOY_DURING_FORMAL_EXECUTION_FORBIDDEN')
    def admit(self,row,stage):
        q.require(not (self.folder/'jobs'/row['name']/stage).exists(),'TOY_NO_RETRY')
    def stage(self,row,stage):
        os.environ['R7_SYNTHETIC_TEST']='1'
        q.r6_imports()
        from r7_supervisor import run_worker
        index=q.contract()['candidate_order'].index(row);self.started.append((row['name'],stage))
        job=self.folder/'jobs'/row['name']/stage;self.current=job
        mode=stage;fault='none'
        limits={'RSS_bytes':2*2**30,'private_commit_bytes':3*2**30,'min_available_physical_bytes':0,'min_remaining_commit_bytes':0};timeout=90
        if index==2:
            if self.fault=='third_train' and stage=='train':mode='exception'
            if self.fault in ('missing_frame','duplicate_frame') and stage=='validate':fault=self.fault
            if self.fault=='resource' and stage=='train':mode='allocate';limits['RSS_bytes']=20*2**20
            if self.fault=='timeout' and stage=='train':mode='timeout';timeout=.3
        run_worker('unit-probe',job,blas=6,openmp=1,extra=['--probe-mode',mode,'--candidate',row['name'],'--fault',fault],local=True,local_limits=limits,timeout=timeout)
    def verify_stage(self,row,stage):
        job=self.folder/'jobs'/row['name']/stage;s=q.read(job/'status.json')
        q.require(s['PASS'] and s['exit_code']==0 and s['stop_reason'] is None and s['resource_samples']>0 and s['actual_interpreter_PID']>0,'TOY_MONITOR_FAILURE')
        result=q.read(job/'toy_result.json');q.require(result['seed']==row['seed'],'TOY_SEED_DRIFT')
        if stage=='train':
            q.require(result['legal']['PASS'],'TOY_ILLEGAL_MODEL')
            for n,h in result['file_hashes'].items():q.require(q.digest(job/'model'/n)==h,'TOY_SERIALIZATION_CORRUPT')
        else:
            _,r,_,_=q.r6_imports()
            from model_ops import candidate_cardinality
            candidate_cardinality(result['frames'],row['K'],row['rank'])
            q.require(len(result['cells'])==6,'TOY_SIX_CELLS_INCOMPLETE')
    def verify_candidate(self,row):return {'PASS':True,'synthetic_only':True,'seed':row['seed']}
    def finish(self):q.require(len(self.started)==18,'TOY_EXACT_18_STAGES_REQUIRED')
    def abort(self,reason):q.write(self.folder/'STOP.json',{'reason':reason,'synthetic_only':True},exclusive=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--fault',default='none');a=p.parse_args()
    backend=SyntheticBackend(a.folder,a.fault)
    result=run_queue(a.folder,q.contract()['candidate_order'],backend)
    print(json.dumps({'PASS':True,'state':result,'synthetic_only':True,'formal_training':0},indent=2),flush=True)
if __name__=='__main__':main()
