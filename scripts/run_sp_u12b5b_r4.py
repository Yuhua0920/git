"""R4 technical-only certification orchestration; no formal entrypoint calls."""
from pathlib import Path
import sys,os,time,json,subprocess,argparse
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u12b5b_r4_common import firewall,read,write,sha,OUT,check_frozen,LIMITS
PORT=ROOT/'experiments/mfa_low_memory_fidelity_r4';CONFIG=ROOT/'configs/sp_u12b5b_r4_synthetic.json'

def freeze():
    firewall();check_frozen()
    assert (OUT/'preservation_before.json').exists()
    assert not CONFIG.exists(),'DO_NOT_OVERWRITE_SYNTHETIC_PREREGISTRATION'
    cases=[]
    for i,(K,r) in enumerate(( (k,r) for k in (1,2,4) for r in (4,8,16))):
        cases.append({'id':f'nine_K{K}_r{r}','K':K,'r':r,'D':64,'N':256,'mode':'short','steps':8,
            'data_seed':42480000+i*10,'initialization_seed':42480001+i*10})
    for i,(K,r) in enumerate(((1,4),(2,8),(4,16),(1,16))):
        cases.append({'id':f'convergence_K{K}_r{r}','K':K,'r':r,'D':64,'N':256 if i<3 else 128,
            'mode':'convergence','steps':100,'data_seed':42481000+i*10,'initialization_seed':42481001+i*10,
            'stop_sensitive_role':'extra high-rank smaller-sample case + strict-threshold branch sentinel' if i==3 else 'representative'})
    cases.append({'id':'medium_K4_r16','K':4,'r':16,'D':256,'N':1024,'mode':'short','steps':8,
        'data_seed':42482000,'initialization_seed':42482001})
    seeds={c[n] for c in cases for n in ('data_seed','initialization_seed')}
    registry=read(ROOT/'outputs/sp_u12b5b_candidate_seed_registry.json')
    def integers(v):
        if isinstance(v,dict):return set().union(*(integers(x) for x in v.values()))
        if isinstance(v,list):return set().union(*(integers(x) for x in v))
        return {v} if type(v)is int else set()
    assert not seeds&integers(registry),'FORMAL_SEED_REUSE_FORBIDDEN'
    write(CONFIG,{'scope':'EXPERIMENTAL_NOT_FORMAL_BASELINE','cases':cases,'tolerance':{'atol':1e-10,'rtol':1e-9},
        'thread_comparisons':[1,4,8,12],'resource_limits':LIMITS,'near_limit_stop':{'RSS_bytes':int(5.5*2**30),'available_bytes':int(2.25*2**30)},
        'worker_timeout_seconds':1200,'measurements':{'kinds':['loading','latent','quadratic','residual','small_EM','medium_EM'],'blocks':3,'threads':1},
        'formal_fit_authorized':False,'scientific_score_authorized':False,'policy':'original author max100 stopping for convergence, only four frozen R3 transforms'})
    files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(PORT.glob('*.py'))}
    for p in (Path(__file__),ROOT/'scripts/sp_u12b5b_r4_common.py',CONFIG):files[p.relative_to(ROOT).as_posix()]=sha(p)
    write(OUT/'execution_freeze.json',{'files':files,'R3_immutable':check_frozen(),'state':'FROZEN_BEFORE_SYNTHETIC_RUN'})
    print('R4_SYNTHETIC_EXECUTION_FROZEN')

def monitored(out,role,threads=1,optimized=False,reference=None,kind=None):
    firewall()
    sys.path.insert(0,str(ROOT/'experiments/mfa_low_memory_prototype/portable'))
    from runner import write as atomic_write,optional_marker,clean_exit_race,terminate
    from host_inventory import process_memory,memory
    cfg=read(CONFIG);assert not out.exists(),'FRESH_JOB_REQUIRED';out.mkdir(parents=True)
    assert memory()['available_physical_bytes']>4*2**30,'INSUFFICIENT_SAFE_LAUNCH_HEADROOM'
    command=[sys.executable,'-B','-X','utf8',str(PORT/'worker.py'),'--config',str(CONFIG),'--out',str(out),'--role',role,'--threads',str(threads)]
    if optimized:command+=['--optimized']
    if reference:command+=['--reference',str(reference)]
    if kind:command+=['--kind',kind]
    start=time.monotonic();log=(out/'worker.log').open('w',encoding='utf8');proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    pid=None;peak=commit=0;minfree=memory()['available_physical_bytes'];reason=None;records=[]
    while proc.poll()is None:
        if pid is None:
            marker=optional_marker(out/'actual_pid.json')
            if marker:pid=marker['actual_PID'];atomic_write(out/'monitor_ready.json',{'actual_PID':pid,'launcher_PID':proc.pid})
        if pid:
            try:m=process_memory(pid);system=memory()
            except OSError:
                if clean_exit_race(proc,out/'worker_final.json'):break
                reason='ACTUAL_PID_MEMORY_READ_FAILED';break
            peak=max(peak,m['peak_working_set_bytes']);commit=max(commit,m['peak_commit_bytes']);minfree=min(minfree,system['available_physical_bytes'])
            records.append({'elapsed':time.monotonic()-start,'process':m,'available_physical_bytes':system['available_physical_bytes'],'stage':optional_marker(out/'current_stage.json')})
            if peak>=cfg['near_limit_stop']['RSS_bytes']:reason='NEAR_PROCESS_LIMIT_STOP'
            if minfree<cfg['near_limit_stop']['available_bytes']:reason='NEAR_HEADROOM_LIMIT_STOP'
        if time.monotonic()-start>cfg['worker_timeout_seconds']:reason='SYNTHETIC_TIMEOUT'
        if not pid and time.monotonic()-start>60:reason='PID_HANDSHAKE_TIMEOUT'
        if reason:
            if pid:terminate(pid)
            else:proc.terminate()
            break
        time.sleep(.05)
    code=proc.wait(timeout=30);log.close()
    final=read(out/'worker_final.json') if(out/'worker_final.json').exists()else None
    if final:peak=max(peak,final['memory']['peak_working_set_bytes']);commit=max(commit,final['memory']['peak_commit_bytes'])
    write(out/'resource_summary.json',{'PASS':code==0 and reason is None and final is not None,'exit_code':code,'reason':reason,
        'peak_RSS':peak,'peak_private_commit':commit,'minimum_available_physical_bytes':minfree,'wall_seconds':time.monotonic()-start,
        'actual_PID':pid,'launcher_PID':proc.pid,'samples':len(records),'limits_unchanged':LIMITS})
    write(out/'resource_samples.json',records)
    assert code==0 and reason is None and final is not None,'TECHNICAL_OR_RESOURCE_STOP; RETAIN_RECORDS'
    print(json.dumps({'job':out.name,'completed':True,'peak_RSS_GiB':peak/2**30}),flush=True)

def run():
    firewall();check_frozen();f=read(OUT/'execution_freeze.json')
    assert all(sha(ROOT/n)==h for n,h in f['files'].items()),'R4_EXECUTION_BYTES_CHANGED'
    ref=OUT/'reference_1';monitored(ref,'parity',1)
    for t in (1,4,8,12):monitored(OUT/f'low_{t}','parity',t,True,ref)
    for kind in read(CONFIG)['measurements']['kinds']:
        for optimized in (False,True):monitored(OUT/f"measure_{kind}_{'low' if optimized else 'original'}",'measure',1,optimized,kind=kind)
    monitored(OUT/'host_inventory','inventory',1)
    print('R4_SAFE_SYNTHETIC_RUN_COMPLETE; FORMAL_FITS=0')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','run'));args=parser.parse_args()
    freeze()if args.action=='freeze'else run()
