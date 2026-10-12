"""Release only complete, resource-qualified paired data; never tune a model."""
import csv,json
import numpy as np
import base as b
def result_draft(rows,runtime):
    npos=sum(v['classification']=='LR4_SUPPORTED' for v in rows);nneg=sum(v['classification']=='MFA_SUPPORTED' for v in rows)
    text=['The frozen source-anchored estimator with rank-four corrections and the independently selected mixture-of-factor-analyzers baseline were evaluated on identical channel trajectories and received pilot observations. The MFA used one component with loading rank 16, fitted to 1500 preparation trajectories. Each of the five channel conditions was evaluated at seven SNR levels from 0 to 30 dB, with 1000 independent trajectories per condition and six scored frames per trajectory.',
          'Both estimators used clean preparation channels from the same three origin conditions. Their development interfaces differ: MFA structure selection used preparation-origin validation, whereas the frozen correction rank used separate channel-development validation that included the composed condition. Identical test observations therefore establish a paired comparison of the deployed estimators, rather than equality of their complete development-information budgets.',
          'NMSE was computed by pooling squared estimation-error energy and clean-channel energy in the data domain. Uncertainty in MFA-minus-LR4 NMSE differences was assessed using 100000 paired trajectory bootstrap replicates. Two-sided intervals used a Bonferroni adjustment across all 35 condition-SNR comparisons. A positive difference favors LR4.',
          f'The adjusted intervals favored LR4 in {npos} comparisons and MFA in {nneg}; the remaining {35-npos-nneg} intervals contained zero. The complete numerical comparisons are reported below. These results characterize the evaluated conditions and SNR levels.',
          'Condition-specific results:']
    for condition in ('ID','Profile','Mobility','Composed','Boundary'):
        cells=[v for v in rows if v['condition']==condition]
        lines=[]
        for r in cells:
            lines.append(f"{r['snr_db']} dB: LR4 {r['LR4_NMSE_dB']:.4f} dB, MFA {r['MFA_NMSE_dB']:.4f} dB; difference {r['MFA_minus_LR4_dB']:+.4f} dB, adjusted interval [{r['simultaneous_CI_lower_dB']:+.4f}, {r['simultaneous_CI_upper_dB']:+.4f}] dB")
        text.append(condition+' — '+'; '.join(lines)+'.')
    text.append(f"On the evaluation host, original-author MFA adapter calls required {runtime['adapter_seconds']:.3f} seconds in total, including {runtime['posterior_setup_seconds']:.3f} seconds of observed noise-dependent posterior preparation. Each call processed six pilot-observation rows. Monitoring and profiling overhead are included. This endpoint is distinct from the arithmetic count used for LR4 and does not establish a wall-clock speedup between the methods.")
    text.append('The comparison holds both estimators fixed after preparation. Test observations do not change the MFA structure, its initialization or residual covariance, or the LR4 rank and cross-pilot fitting rule. The channel simulation is symbol-wise quasi-static and includes no explicit intercarrier interference. Further attribution of any condition-dependent advantage requires separate evidence.')
    return '\n\n'.join(text)+'\n'
def analyze(job):
    b.verify();gate=b.read(job/'integrity_gate.json');state=b.read(job/'status.json')
    b.require(gate['PASS'] and gate['all_pairs_exact'] and gate['paired_frame_predictions_per_method']==210000 and state['PASS'] and state['state']=='COMPLETE','NO_AGGREGATE_RELEASE_BEFORE_ALL_GATES')
    b.require(len(gate['raw_hashes'])==5000 and not(b.OUT/'STOP.json').exists(),'INCOMPLETE_OR_STOPPED_SEALED_EVALUATION')
    dest=job/'analysis';dest.mkdir(exist_ok=False)
    from paired_stats import summarize
    from science import paired_reduction
    p=b.contract();rows=[];frame_count=0
    def cancel():b.require(not(b.OUT/'STOP_REQUESTED.json').exists(),'ANALYSIS_USER_STOP_REQUESTED')
    for ci,condition in enumerate(p['conditions']):
        cancel();b.write(job/'phase.json',{'phase':'PAIRED_BOOTSTRAP','condition':condition,'replicates':100000,'family':35})
        error=np.empty((1000,7,2),np.float64);truth=np.empty((1000,7),np.float64)
        for i in range(1000):
            n=f'raw/{condition}/{i:04d}.npz';q=job/n
            b.require(b.sha(q)==gate['raw_hashes'][n],'NEW_RAW_CHANGED_BEFORE_RELEASE')
            receipt=b.read(q.with_suffix('.json'))
            b.require(receipt['pairing_verified'] and receipt['condition']==condition and receipt['trajectory_id']==i and receipt['model_manifest_sha256']==p['final_model_manifest_sha256'],'NEW_IDENTITY_DRIFT')
            with np.load(q,allow_pickle=False) as z:
                fe=z['frame_error'];ft=z['frame_truth'];e=z['trajectory_error'];t=z['trajectory_truth']
            b.require(fe.shape==(7,6,2) and ft.shape==(7,6) and np.array_equal(e,paired_reduction(fe)) and np.array_equal(t,ft.sum(axis=1)),'NEW_RAW_REDUCTION_DRIFT')
            error[i]=e;truth[i]=t;frame_count+=42
        values,draws=summarize(error,truth,ci,cancel=cancel);rows.extend(values)
        with (dest/(condition+'_bootstrap.npz')).open('xb') as f:np.savez(f,MFA_minus_LR4_dB=draws)
        with (dest/(condition+'_paired_trajectory_energies.npz')).open('xb') as f:np.savez(f,error=error,truth=truth,trajectory_id=np.arange(1000))
    b.require(len(rows)==35 and frame_count==210000,'ANALYSIS_FULL_GRID_REQUIRED')
    cancel()
    with (dest/'nmse_comparison_35_cells.csv').open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    calls=[]
    with (job/'inference_runtime.jsonl').open(encoding='utf8') as f:
        for line in f:calls.append(json.loads(line))
    keys={(r['condition'],r['snr_db'],r['trajectory_id']) for r in calls}
    b.require(len(calls)==len(keys)==35000 and keys=={(c,s,i) for c in p['conditions'] for s in p['snrs'] for i in range(1000)},'RUNTIME_GRID_IDENTITY_DRIFT')
    times=np.array([r['adapter_wall_seconds'] for r in calls]);setup=np.array([r['posterior_setup_seconds'] for r in calls])
    runtime={'adapter_seconds':float(times.sum()),'posterior_setup_seconds':float(setup.sum()),'per_six_frame_call_seconds_median':float(np.median(times)),
        'per_six_frame_call_seconds_p95':float(np.quantile(times,.95)),'calls':35000,'predictions':210000,
        'peak_RSS_bytes':state['peak_RSS_bytes'],'peak_private_commit_bytes':state['peak_private_commit_bytes'],
        'evaluation_wall_seconds':gate['evaluation_wall_seconds'],'no_fair_cross_hardware_speedup_claim':True,
        'LR4_dominant_complex_MAC_proxy_per_frame':332288,'MFA_endpoint':'original float1.0 adapter; posterior preparation included; profiling included',
        'offline_final_training_excluded':True,'one_time_model_integrity_setup_included_in_evaluation_wall':True}
    b.write(dest/'runtime_summary.json',runtime,True)
    (dest/'results_draft.txt').write_text(result_draft(rows,runtime),encoding='utf8')
    b.write(dest/'analysis_manifest.json',{'PASS':True,'rows':35,'bootstrap_replicates':100000,'family':35,
       'source_integrity_sha256':b.sha(job/'integrity_gate.json'),'resource_status_sha256':b.sha(job/'status.json'),
       'files':{q.name:b.sha(q) for q in dest.iterdir() if q.is_file()},'figures':'Run publication_plots.py with this CSV in an existing matplotlib environment; all 35 cells retained','UTC':b.utc()},True)
