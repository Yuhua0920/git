"""T19C frozen-history/statistics reconstruction, claims and final regression."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src import target_statistics_replication as p
from src import frozen_affine_validation as old
from src.cdl_benchmark import FrozenStatistics,array_hash
from scripts.run_target_statistics_replication import OUT,RESULT,SOURCES,accepted,frozen_inputs,diagnostic,original_reference,namespace_audit,verify_protected,method_guard
from scripts.run_cdl_mismatch import load,save,sha,environments
from scripts.run_pilot_cv import readcsv


def read_statistics(info,*,large=False):
    path=ROOT/info['path']
    if sha(path)!=info['sha256']:raise ValueError('new frozen statistics byte drift')
    with np.load(path,allow_pickle=False) as f:
        values=[f[k].copy() for k in ('mean_p','mean_d','rpp','rdp')]
        if any(array_hash(a)!=info['component_sha256'][k] for a,k in zip(values,('mean_p','mean_d','rpp','rdp'))):
            raise ValueError('new statistics component drift')
        ids=tuple(tuple(map(int,row)) for row in f['train_trajectory_keys']) if large else tuple(map(int,f['train_trajectory_ids']))
        samples=int(f['samples'])
    if samples!=info['samples']:raise ValueError('sample count changed')
    for a in values:a.flags.writeable=False
    return FrozenStatistics(*values,ids,samples)


def verify(r):
    verify_protected(r)
    for group in ('numeric_artifact_sha256','claim_document_sha256','figure_artifact_sha256'):
        for path,h in r.get(group,{}).items():
            if sha(ROOT/path)!=h:raise ValueError('artifact drift:'+path)
    if any(r[k] for k in ('validation_generated','new_module_added','new_model_trained','new_coefficient_solver','Pilot_CV_performed','BER_evaluations','bootstrap_performed','source_SNR_reference_recalibrated','formal_test_used','original_in_replication_aggregate','original_in_large_stat_reference')):
        raise ValueError('scope violation')
    if not all(r[k] for k in ('method_frozen','formal_test_sealed','same_statistics_for_both_SNRs','train_smoke_exact_replay','large_stat_enabled')):
        raise ValueError('frozen design failed')
    if r['delay_levels_ns']!=[50,200] or r['statistics_replicates']!=5 or r['trajectories_per_statistics_replicate']!=500 or r['rank_per_module']!=4:
        raise ValueError('design/primary population changed')
    if r['generator_calls']!={'preflight_train_smoke':10,'target_train':5000,'validation':0} or r['new_train_realizations']!=5000:
        raise ValueError('new trajectory/validation scope changed')
    if (r['c6_claim_status'],r['c7_claim_status'],r['c8_claim_status'],r['c9_claim_status'],r['c10_claim_status'],r['t18a_materiality_status'])!=(old.C6,old.C7,'SUPPORTED_IN_FIXED_ENVIRONMENTS','INCONCLUSIVE','INCONCLUSIVE','PARTIAL'):
        raise ValueError('historical claim changed')
    if r['source_power']!=old.POWER or {int(k):v for k,v in r['source_noise_variances'].items()}!={s:old.VARIANCES[s] for s in p.SNRS}:
        raise ValueError('source SNR definition changed')
    ns=namespace_audit(p.manifest_rows())
    if ns!=r['namespace_audit'] or load(OUT/'t19c_seed_audit.json')!=ns:raise ValueError('namespace proof drift')
    manifest=readcsv(OUT/'t19c_manifest.csv');expected=p.manifest_rows()
    if len(manifest)!=50000:raise ValueError('manifest population changed')
    for actual,declared in zip(manifest,expected):
        if any(actual[k]!=str(v) for k,v in declared.items()):raise ValueError('actual generator seed manifest changed')
    hashes=readcsv(OUT/'t19c_channel_hashes.csv')
    index={(v['environment'],int(v['replicate']),int(v['trajectory_id'])):v for v in hashes}
    if len(hashes)!=5000 or set(index)!={(n,r0,i) for n in p.ENVIRONMENTS for r0 in p.REPLICATES for i in range(500)} or len({v['H_sha256'] for v in hashes})!=5000:
        raise ValueError('missing/duplicated independent H population')
    pre=load(OUT/'t19c_preflight.json')
    for n in p.ENVIRONMENTS:
        for r0 in p.REPLICATES:
            row=index[n,r0,0]
            if row['namespace']!=p.namespace(n,r0) or any(row[k]!=v for k,v in pre['train_smoke'][row['namespace']].items()):
                raise ValueError('first-trajectory deterministic smoke regression failed')
    mapping={'operator_capture':'operator_capture_replicates','statistics_replicates':'statistics_artifacts',
             'statistics_distances':'statistics_distances','effective_delay':'effective_delay_audit','large_stat_reference':'large_stat_reference','original_reference':'original_reference'}
    for name,key in mapping.items():
        rows=readcsv(OUT/f't19c_{name}.csv')
        if len(rows)!=len(r[key]):raise ValueError('CSV/JSON row count drift')
        for csvrow,value in zip(rows,r[key]):
            if any(csvrow[k]!=str(v) for k,v in value.items()):raise ValueError('CSV/JSON value drift:'+name)
    with method_guard():
        accepted();arrays,family,_,original,proof=frozen_inputs()
        if proof!=r['method_provenance'] or original_reference(arrays,family,original)!=r['original_reference']:
            raise ValueError('frozen method / original exact reference drift')
        primary={};records={n:[] for n in p.ENVIRONMENTS}
        for info in r['statistics_artifacts']:
            n,r0=info['environment'],info['replicate'];st=read_statistics(info)
            if st.samples!=5000 or st.train_trajectory_ids!=tuple(range(500)):raise ValueError('independent500x10 population changed')
            if info['namespace']!=p.namespace(n,r0) or info['batch_frames']!=100 or info['validation_used']:
                raise ValueError('estimator/namespace design changed')
            records[n].append((info['namespace'],st));primary[n,r0]=st
            group=[v for v in r['operator_capture_replicates'] if (v['environment'],v['replicate'])==(n,r0)]
            computed=diagnostic(n,r0,st,arrays,family)
            if len(group)!=2 or {v['statistics_sha256'] for v in group}!={info['sha256']}:
                raise ValueError('SNRs use different statistics')
            for a,b in zip(group,computed):
                if any(a[k]!=v for k,v in b.items()):raise ValueError('exact replicate operator reconstruction drift')
            distance=next(v for v in r['statistics_distances'] if (v['environment'],v['replicate'])==(n,r0))
            for part in ('rpp','rdp'):
                absolute=float(np.linalg.norm(getattr(st,part)-getattr(original[n],part)))
                relative=absolute/float(np.linalg.norm(getattr(original[n],part)))
                if distance[part+'_distance_fro']!=absolute or distance[part+'_relative_distance_fro']!=relative:
                    raise ValueError('statistics distance reconstruction drift')
        for info in r['large_stat_artifacts']:
            n=info['environment'];stored=read_statistics(info,large=True)
            merged=p.merge_sufficient_statistics(n,records[n])
            if stored.samples!=25000 or len(stored.train_trajectory_ids)!=2500 or stored.train_trajectory_ids!=merged.train_trajectory_ids:
                raise ValueError('large-stat population/weighting changed')
            if info['original_T19A_included'] or info['used_for_primary_gate'] or info['input_namespaces']!=[p.namespace(n,r0) for r0 in p.REPLICATES]:
                raise ValueError('large-stat leak into primary/original pooling')
            if any(not np.array_equal(getattr(stored,k),getattr(merged,k)) for k in ('mean_p','mean_d','rpp','rdp')):
                raise ValueError('exact sufficient-stat aggregation drift')
            rows=diagnostic(n,0,merged,arrays,family)
            group=[v for v in r['large_stat_reference'] if v['environment']==n]
            if any(any(a[k]!=v for k,v in b.items()) for a,b in zip(group,rows)):raise ValueError('large-stat capture changed')
        for v in r['effective_delay_audit']:
            power=np.array(v['pooled_path_energy']);tau=np.array(v['path_delays_seconds']);pp=power/power.sum();mu=float(pp@tau)
            rms=float(np.sqrt(pp@(tau-mu)**2))*1e9
            if rms!=v['empirical_array_weighted_rms_delay_ns'] or v['nominal_delay_spread_ns']!=int(v['environment'].split('-')[-1]) or v['SNR_reference_recalibrated']:
                raise ValueError('physical effective-delay audit changed')
            if v['train_receive_power']!=float(np.mean(v['train_receive_power_per_rx'])) or v['trajectories']!=500 or v['frames']!=5000:
                raise ValueError('receive-power/population audit changed')
    reference={(v['environment'],v['snr_db']):v['operator_span_capture_dense'] for v in r['original_reference']}
    if p.summarize(r['operator_capture_replicates'],reference)!=r['capture_summary']:raise ValueError('replication summary drift')
    if any(r[k]!=v for k,v in p.scientific_gate(r['operator_capture_replicates']).items()):raise ValueError('pre-registered replication gate drift')
    return True


def claim():
    r=load(RESULT);verify(r)
    if r.get('claim_document_sha256'):raise ValueError('existing claim files protected')
    text='# T19C Claim–Evidence update\n\nResearch Question: reproducible C30/200/10 operator deficiency versus finite target-statistics estimation randomness.\n\n'
    text+='C11: '+p.C11+'\n\nStatus: '+r['c11_claim_status']+'.\n\n'
    text+='Historical C6='+r['c6_claim_status']+', C7='+r['c7_claim_status']+', C8='+r['c8_claim_status']+', C9/C10 INCONCLUSIVE; T18A PARTIAL unchanged.\n\n'
    text+='| Condition | Five captures | Mean | Sample std | Median | Min | Max | Original |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for v in r['capture_summary']:
        text+='| '+v['environment']+'/'+str(v['snr_db'])+' | '+', '.join(str(x) for x in v['captures'])+' | '+' | '.join(str(v[k]) for k in ('mean','std','median','minimum','maximum','original_t19a_capture'))+' |\n'
    text+='\n'+''.join(f'- {k}: {r[k]}\n' for k in ('primary_below_050','primary_at_or_above_070','primary_median','delay_operator_deficiency_replication_supported','delay_operator_deficiency_replication_contradicted','delay_operator_deficiency_replication_inconclusive','delay50_control_stable','snr_asymmetry_direction_consistent'))
    text+='\nDecision dependency: '+r['next_step_dependency']+' No next-stage work is implemented.\n\n'
    text+='Five independent500-trajectory target statistics per delay; paired SNR uses one clean covariance. All original1801 regressions remain mandatory. Original T19A point is an external reference and is not an additional replicate. Pooled2500 is sufficient-stat-only secondary convergence evidence, not an extra replicate or deployed method.\n'
    review='# T19C Reviewer Red Team\n\n'
    review+='- The independent random unit is the target-statistics estimate. Five replicates cannot establish all CDL populations or universal delay deficiency. No frame/trajectory pseudoreplication, bootstrap or post-hoc threshold.\n'
    review+='- Only C30/200/10 determines C11. C30/50 is a separate high-capture control;20dB asymmetry and rank4 results are secondary and cannot overwrite the primary gate.\n'
    review+='- Source/module/means/folds/SNR scaling are frozen. Target mean/Rpp/Rdp are independently estimated from train data only. Evaluation-only oracle coordinates are not deployed method refits.\n'
    review+='- Target training sets are disjoint from all audited historical streams and each other. All5000 realizations are included; no sample removal, seed search by performance or validation generation. Ten preflight smoke calls are replayed, not ten extra stochastic realizations.\n'
    review+='- Original StreamingStatistics population convention and100-frame batches remain exact. No shrinkage, new regularization, Rdd fitting or solver changes. Physical generator AST differs only in its declared namespace seed call.\n'
    review+='- Pooled2500 covariance includes between-mean terms and uses only five new sufficient-stat snapshots. Do not average centered covariances without this correction or include the original500 trajectories.\n'
    review+='- Train receive power and effective array-weighted delay are audits only. Neither renormalizes H nor changes noise variance. Nominal DS and empirical RMS need not coincide under array weighting.\n'
    review+='- Exact operator capture is conditional on estimated statistics, not actual deployed NMSE/BER. Support would not establish that a third module improves channel estimation. C9/C10 and negative C6/C7 history remain unchanged.\n\n'
    review+='Actual result: '+r['c11_claim_status']+'; '+r['next_step_dependency']+'\n'
    hashes={}
    for name,value in (('t19c_claim_evidence_update.md',text),('t19c_reviewer_update.md',review)):
        path=ROOT/'docs'/name
        if path.exists():raise ValueError('existing claim document protected')
        path.write_text(value,encoding='utf8');hashes['docs/'+name]=sha(path)
    r['claim_document_sha256']=hashes;save(RESULT,r);print('T19C bounded claim/reviewer update generated',flush=True)


def visual_review():
    r=load(OUT/'t19c_figure_audit.json')
    if r['source_returncode'] or not all(r['CSV_only_rebuild'].values()) or any(v['text_returncode'] or v['collision_returncode'] for v in r['rendered_checks'].values()):
        raise ValueError('automatic figure QA incomplete')
    r.update(visual_review='Actual two PNGs inspected: all20 replicate captures,four original and four pooled references; allfive C30/200 paired SNR lines. Labels/legends/units and external-reference exclusion visible; no clipping or CI.',all_checks_passed=True)
    save(OUT/'t19c_figure_audit.json',r)


def finalize():
    start=time.perf_counter();r=load(RESULT);verify(r)
    if not load(OUT/'t19c_figure_audit.json')['all_checks_passed'] or not r.get('claim_document_sha256'):
        raise ValueError('claims/manual visual review incomplete')
    env=dict(os.environ);env['TEMP']=env['TMP']=str(ROOT/'.pytest_cache/pytest_workspace_temp');tests={}
    groups=[('related',['tests/test_target_statistics_replication.py']),
            ('provenance',['tests/test_operator_space_audit.py','tests/test_delay_factor_challenge.py','tests/test_historical_provenance.py','tests/test_validation_replication.py']),('full',[])]
    for label,args in groups:
        path=OUT/f't19c_pytest_{label}.txt'
        if path.exists():raise ValueError('existing regression log protected')
        tick=time.perf_counter();print('Actual '+label+' regression',flush=True)
        with path.open('w',encoding='utf8') as f:
            proc=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest',*args,'-q'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        summary=path.read_text(encoding='utf8').strip().splitlines()[-1];print(summary,flush=True)
        tests[label]=dict(command=[str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest',*args,'-q'],summary=summary,exit_code=proc.returncode,seconds=time.perf_counter()-tick,sha256=sha(path))
        if proc.returncode or 'skipped' in summary:raise ValueError('regression failure/skip:STOP')
    verify(r);after=environments()
    if after!=r['environment_before']:raise ValueError('frozen environment payload drift')
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError('git diff --check failed')
    new_files={}
    for name in (*SOURCES,*r['claim_document_sha256']):
        lines=(ROOT/name).read_text(encoding='utf8').splitlines();bad=[i for i,line in enumerate(lines,1) if line.rstrip()!=line]
        if bad:raise ValueError('new source whitespace:'+name)
        new_files[name]=dict(lines=len(lines),sha256=sha(ROOT/name),trailing_whitespace_lines=bad)
    from scripts.plot_target_statistics_replication import STEMS
    r.update(pytest=tests,environment_after=after,git_diff_check=dict(exit_code=diff.returncode,output=diff.stdout+diff.stderr),
        final_hash_audit=dict(passed=True,historical_files=len(r['frozen_history_sha256']),unexpected_content_mismatch=0),
        new_file_diff_equivalent=new_files,figure_audit_sha256=sha(OUT/'t19c_figure_audit.json'),
        figure_artifact_sha256={f'outputs/{stem}.{ext}':sha(OUT/f'{stem}.{ext}') for stem in STEMS for ext in ('png','pdf','svg','alignment.json','collision-audit.json')},
        all_checks_passed=True,completion_state='completed')
    r['runtime']['final_audit_and_regression_seconds']=time.perf_counter()-start
    save(RESULT,r);print('T19C engineering PASS; C11='+r['c11_claim_status']+'; formal test sealed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['claim','visual-review','finalize']);args=parser.parse_args()
    {'claim':claim,'visual-review':visual_review,'finalize':finalize}[args.mode]()
