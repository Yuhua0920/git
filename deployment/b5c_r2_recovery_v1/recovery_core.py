"""Validate all stored sufficient statistics before any outcome aggregation."""
import json,math,zipfile
from pathlib import Path
import numpy as np
import recovery_base as r

FIELDS={'frame_error.npy','frame_truth.npy','trajectory_error.npy','trajectory_truth.npy'}
def load_raw(path,reduction):
    with zipfile.ZipFile(path) as z:
        names=z.namelist();r.require(len(names)==4 and set(names)==FIELDS,'RAW_NPZ_SCHEMA_OR_DUPLICATE:'+str(path))
        r.require(z.testzip() is None,'RAW_NPZ_CRC:'+str(path))
    with np.load(path,allow_pickle=False) as z:
        fe,ft,e,t=(z[n] for n in ('frame_error','frame_truth','trajectory_error','trajectory_truth'))
    r.require(fe.shape==(7,6,2) and ft.shape==(7,6) and e.shape==(7,2) and t.shape==(7,),'RAW_ARRAY_SHAPE:'+str(path))
    r.require(all(v.dtype==np.float64 and np.isfinite(v).all() and np.all(v>0) for v in (fe,ft,e,t)),'RAW_NONFINITE_OR_INVALID:'+str(path))
    r.require(np.array_equal(e,reduction(fe)) and np.array_equal(t,ft.sum(axis=1)),'RAW_SUFFICIENT_STATISTIC_DRIFT:'+str(path))
    r.require(np.array_equal(ft,np.broadcast_to(ft[0],ft.shape)),'RAW_CROSS_SNR_TRUTH_DRIFT:'+str(path))
    return fe,ft,e,t
def coverage(job,p):
    wanted={f'{c}/{i:04d}.{ext}' for c in p['conditions'] for i in range(p['N']) for ext in ('json','npz')}
    raw=job/'raw';r.plain(raw,r.SOURCE)
    actual=set()
    for path in raw.rglob('*'):
        r.plain(path,raw)
        if path.is_file():actual.add(path.relative_to(raw).as_posix());r.require(path.stat().st_size>0,'EMPTY_RAW_FILE:'+str(path))
    r.require(actual==wanted,'EXACT_RAW_INVENTORY_MISMATCH: missing='+str(sorted(wanted-actual)[:12])+' extra='+str(sorted(actual-wanted)[:12]))
    return len(wanted)//2
def runtime_grid(path,p):
    seen=set();count=0
    with path.open(encoding='utf8') as f:
        for number,line in enumerate(f,1):
            r.require(line.endswith('\n') and line.strip(),'RUNTIME_TRUNCATED_LINE:'+str(number))
            # Enforce duplicate-field and nonfinite JSON rules used by r.read.
            def pairs(items):
                d={}
                for k,v in items:r.require(k not in d,'RUNTIME_DUPLICATE_FIELD');d[k]=v
                return d
            v=json.loads(line,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(RuntimeError('RUNTIME_NONFINITE_JSON')))
            key=(v['condition'],v['snr_db'],v['trajectory_id'])
            r.require(type(v['trajectory_id']) is int and type(v['snr_db']) is int and key not in seen,'RUNTIME_DUPLICATE_OR_INVALID_ID:'+str(key));seen.add(key)
            r.require(v['condition'] in p['conditions'] and v['snr_db'] in p['snrs'] and v['trajectory_id'] in range(p['N']),'RUNTIME_UNEXPECTED_ID:'+str(key))
            r.require(type(v['rows']) is int and type(v['conditional_calls']) is int and v['rows']==6 and v['conditional_calls']==6 and v['observer_profile_included'] is True,'RUNTIME_INFERENCE_INTERFACE_DRIFT:'+str(key))
            for k in ('adapter_wall_seconds','posterior_setup_seconds'):r.require(type(v[k]) in (int,float) and math.isfinite(v[k]) and v[k]>=0,'RUNTIME_INVALID_DURATION:'+str(key))
            r.require(v['posterior_setup_seconds']<=v['adapter_wall_seconds']+1e-6,'RUNTIME_SETUP_EXCEEDS_CALL:'+str(key));count+=1
    expected={(c,s,i) for c in p['conditions'] for s in p['snrs'] for i in range(p['N'])}
    r.require(seen==expected and count==35000,'RUNTIME_EXACT_35000_GRID_REQUIRED');return count
def failed_end_gate(b,s,h):
    job=r.SOURCE;state=r.read(job/'status.json');exception=r.read(job/'exception.json')
    r.require(state['state']=='FAILED' and state['PASS'] is False and state['exit_code']==1 and state['stop_reason'] is None,'FAILED_ATTEMPT_NOT_EXPECTED_POSTINFERENCE_ERROR')
    expected=str(b.HERE/'package_manifest.json')
    r.require(exception['type']=='RuntimeError' and exception['message']=='B5C_READ_FIREWALL_DENIED:'+expected,'FAILURE_NOT_EXACT_MANIFEST_ACCESS')
    trace=exception['traceback'];r.require('science.py' in trace and 'b.verify()' in trace and 'evaluate' in trace,'FAILURE_PHASE_NOT_PROVEN')
    r.require((r.OLD/'STOP.json').is_file() and (r.OLD/'FORMAL_START_CONSUMED.json').is_file(),'ORIGINAL_FAILURE_RECEIPTS_MISSING')
    lease=h.Inspection()
    try:
        lease.closed_status(job/'status.json','b5c_paired_r2')
        if (r.OLD/'active_worker.json').exists():lease.pointer(r.OLD/'active_worker.json',r.OLD)
        if (r.OLD/'execution.lock').exists():lease.classify_lock(r.OLD/'execution.lock','b5c_paired_r2')
    finally:lease.close()
    phase=r.read(job/'phase.json');p=b.contract()
    r.require(phase['phase']=='FORMAL_EVALUATION' and phase['condition']==list(p['conditions'])[-1] and phase['trajectory_id']==999 and phase['MFA_predictions_complete']==phase['expected']==210000,'LAST_PERSISTED_PHASE_NOT_FULL_GRID')
    freeze=r.read(job/'execution_freeze.json');consumed=r.read(r.OLD/'FORMAL_START_CONSUMED.json')
    r.require(freeze==consumed and freeze['output']=='lab_runs/b5c_paired_r2/formal' and freeze['automatic_retry'] is False,'EXECUTION_FREEZE_OR_CONSUMPTION_DRIFT')
    paths={'tool_manifest_sha256':b.HERE/'package_manifest.json','contract_sha256':b.ROOT/'deployment/b5c_paired_v1/contract.json',
           'data_manifest_sha256':b.ROOT/'deployment/b5c_paired_v1/data_manifest.json','model_manifest_sha256':b.ROOT/'lab_runs/r6_formal/FINAL_MODEL_MANIFEST.json'}
    for k,path in paths.items():r.require(freeze[k]==r.sha(path),'EXECUTION_HASH_BINDING_DRIFT:'+k)
    r.require(freeze['tool_manifest_sha256']==r.bindings()['source_manifest_sha256'],'ORIGINAL_EXECUTED_CODE_NOT_CERTIFIED')
    r.require(freeze['preflight']['PASS'] is True and freeze['preflight']['tool_freeze_sha256']==freeze['tool_manifest_sha256'],'ORIGINAL_PREFLIGHT_NOT_PASS')
    from r2_gate import receipt_unchanged
    receipt_unchanged(freeze['preflight']['R1_identity_audit'])
    identity=s.final_identity();gate=r.read(job/'model_gate.json')
    r.require(gate['identity']==identity and gate['legal']['PASS'] and gate['legal']['full_covariance_check'] and gate['all_components']['PASS'],'ORIGINAL_MODEL_GATE_NOT_PASS_OR_CHANGED')
    r.require(state['all_descendants_kernel_contained'] is True and state['monitor_not_assumed_venv_launcher_PID'] is True and state['actual_interpreter_PID']>0 and state['resource_samples']>0,'RESOURCE_MONITOR_EVIDENCE_INCOMPLETE')
    limits=state['limits'];res=p['resource'];required={'RSS_bytes':res['RSS_GiB']*2**30,'private_commit_bytes':res['private_commit_GiB']*2**30,'min_available_physical_bytes':res['headroom_GiB']*2**30,'min_remaining_commit_bytes':res['headroom_GiB']*2**30}
    r.require(limits==required and (state['blas_threads'],state['openmp_threads'])==(6,1),'RESOURCE_CONTRACT_DRIFT')
    r.require(state['peak_RSS_bytes']<=limits['RSS_bytes'] and state['peak_private_commit_bytes']<=limits['private_commit_bytes'] and state['minimum_available_physical_bytes']>=limits['min_available_physical_bytes'] and state['minimum_remaining_commit_bytes']>=limits['min_remaining_commit_bytes'] and state['wall_seconds']<=res['timeout_seconds'],'RESOURCE_LIMIT_NOT_SATISFIED')
    samples=0
    with (job/'resources.jsonl').open(encoding='utf8') as f:
        for line in f:
            r.require(line.endswith('\n'),'RESOURCE_LOG_TRUNCATED');v=json.loads(line);tree=v['process_tree'];system=v['system']
            r.require(tree['RSS_bytes']<=limits['RSS_bytes'] and tree['private_commit_bytes']<=limits['private_commit_bytes'] and system['available_physical_bytes']>=limits['min_available_physical_bytes'] and system['commit_available_bytes']>=limits['min_remaining_commit_bytes'],'RESOURCE_SAMPLE_VIOLATION')
            samples+=1
    r.require(samples==state['resource_samples'],'RESOURCE_SAMPLE_COUNT_DRIFT')
    return state
def trajectory_gate(job,p,s,index,cfg,cancel=lambda:None):
    count=coverage(job,p);raw_hashes={};cells={(c,v):0 for c in p['conditions'] for v in p['snrs']}
    for condition in p['conditions']:
        for i in range(p['N']):
            cancel();path=job/'raw'/condition/f'{i:04d}.npz';receipt=r.read(path.with_suffix('.json'))
            r.require(type(receipt['trajectory_id']) is int and receipt['trajectory_id']==i and receipt['condition']==condition,'NEW_RESULT_IDENTITY_DRIFT:'+str((condition,i)))
            r.require(receipt['snrs']==p['snrs'] and receipt['frames']==p['score_frames'] and receipt['methods']==['LR4','MFA'] and receipt['pairing_verified'] is True and receipt['model_manifest_sha256']==p['final_model_manifest_sha256'],'RESULT_AXIS_MODEL_OR_PAIRING_DRIFT:'+str((condition,i)))
            raw_hashes[path.relative_to(job).as_posix()]=r.sha(path)
            r.require(receipt['raw_sha256']==raw_hashes[path.relative_to(job).as_posix()],'RESULT_NPZ_RECEIPT_HASH_DRIFT:'+str((condition,i)))
            fe,ft,e,t=load_raw(path,s.paired_reduction)
            h,comm,inputs,lr4,original_truth,links,old_sum=s.checked_input(condition,i,cfg,index)
            # Pure replay from frozen H/symbols/unit_noise; no RNG, no MFA infer.
            _,_,truth=s.pilot_inputs(h,comm,inputs)
            r.require(receipt['original_inputs']==links,'ORIGINAL_INPUT_LINK_DRIFT:'+str((condition,i)))
            expected=[inputs[v,j] for v in p['snrs'] for j in p['score_frames']]
            r.require(receipt['scored_pilot_and_truth_identities']==expected,'EXACT_42_SCORED_IDENTITIES_DRIFT:'+str((condition,i)))
            r.require(np.array_equal(fe[:,:,0],lr4) and np.array_equal(ft,original_truth) and np.array_equal(e[:,0],old_sum) and np.array_equal(ft,np.broadcast_to(truth,(7,6))),'ORIGINAL_LR4_TRUTH_SUFFICIENT_STATISTIC_NOT_EXACT:'+str((condition,i)))
            for v in p['snrs']:cells[condition,v]+=1
    r.require(count==5000 and len(raw_hashes)==5000 and all(v==1000 for v in cells.values()),'FULL_GRID_GATE_INCOMPLETE')
    runtime=runtime_grid(job/'inference_runtime.jsonl',p)
    return {'PASS':True,'raw_hashes':raw_hashes,'raw_trajectories':count,'raw_JSON':count,'raw_NPZ':count,
            'condition_SNR_cells':35,'MFA_scored_positions':count*42,'runtime_calls':runtime,
            'original_LR4_truth_and_input_pairing_exact':True,'performance_summaries_released':False,'new_MFA_inference_calls':0,'synthetic_only':bool(p.get('synthetic_only',False))}

def analyze(source,job,p,s,stats,report,gate,state,cancel=lambda:None):
    r.require(gate['PASS'] is True and gate['snapshot_sha256']==r.sha(job/'original_snapshot.json'),'RECOVERY_GATE_NOT_AUTHENTICATED')
    dest=job/'analysis';dest.mkdir(exist_ok=False);rows=[];frames=0
    for ci,condition in enumerate(p['conditions']):
        cancel();r.status(job,'ANALYZING',condition=condition,replicates=100000,old_attempt_state='FAILED',new_inference_calls=0)
        error=np.empty((1000,7,2),np.float64);truth=np.empty((1000,7),np.float64)
        for i in range(1000):
            n=f'raw/{condition}/{i:04d}.npz';path=source/n
            r.require(r.sha(path)==gate['raw_hashes'][n],'SNAPSHOT_RAW_CHANGED_BEFORE_ANALYSIS')
            fe,ft,e,t=load_raw(path,s.paired_reduction);error[i]=e;truth[i]=t;frames+=42
        # Original frozen program, defaults, master seed, order and quantiles.
        values,draws=stats.summarize(error,truth,ci,cancel=cancel);rows.extend(values)
        with (dest/(condition+'_bootstrap.npz')).open('xb') as f:np.savez(f,MFA_minus_LR4_dB=draws)
        with (dest/(condition+'_paired_trajectory_energies.npz')).open('xb') as f:np.savez(f,error=error,truth=truth,trajectory_id=np.arange(1000))
    r.require(len(rows)==35 and frames==210000,'RECOVERY_ANALYSIS_COVERAGE')
    import csv
    with (dest/'nmse_comparison_35_cells.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    calls=[json.loads(line) for line in (source/'inference_runtime.jsonl').open(encoding='utf8')]
    times=np.array([v['adapter_wall_seconds'] for v in calls]);setup=np.array([v['posterior_setup_seconds'] for v in calls])
    runtime={'adapter_seconds':float(times.sum()),'posterior_setup_seconds':float(setup.sum()),
             'per_six_frame_call_seconds_median':float(np.median(times)),'per_six_frame_call_seconds_p95':float(np.quantile(times,.95)),
             'calls':35000,'predictions':210000,'peak_RSS_bytes':state['peak_RSS_bytes'],'peak_private_commit_bytes':state['peak_private_commit_bytes'],
             'evaluation_wall_seconds':state['wall_seconds'],'wall_endpoint':'original supervisor total until technical failure, not successful-run latency',
             'no_fair_cross_hardware_speedup_claim':True,'LR4_dominant_complex_MAC_proxy_per_frame':332288,'new_inference_calls':0}
    r.write(dest/'runtime_summary.json',runtime)
    (dest/'results_draft.txt').write_text(report.result_draft(rows,runtime),encoding='utf8')
    r.write(dest/'analysis_manifest.json',{'PASS':True,'rows':35,'bootstrap_replicates':100000,'family':35,'difference':'MFA minus LR4 dB; positive favors LR4',
            'original_attempt_state':'FAILED','source_snapshot_sha256':gate['snapshot_sha256'],'independent_recovery_gate_sha256':r.sha(job/'integrity_gate.json'),
            'files':{q.name:r.sha(q) for q in dest.iterdir() if q.is_file()},'UTC':r.utc(),'new_inference_calls':0,'synthetic_only':gate.get('synthetic_only',False)})
    return dest
