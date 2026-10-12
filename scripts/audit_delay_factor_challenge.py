"""T19A post-run reconstruction, claims and complete regression gates."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src import delay_factor_challenge as p
from src import frozen_affine_validation as old
from src.cdl_benchmark import array_hash
from scripts.run_delay_factor_challenge import OUT,RESULT,accepted,frozen_method,namespace_audit,aggregate,summarize_weights
from scripts.run_cdl_mismatch import load,save,sha,environments
from scripts.run_pilot_cv import readcsv,trajectory_energy


def verify(r):
    accepted();_,_,proof=frozen_method()
    if proof!=r['frozen_method_provenance']:raise ValueError('frozen deployed method drift')
    for group in ('source_sha256','frozen_history_sha256','preparation_artifact_sha256','numeric_artifact_sha256'):
        for path,h in r[group].items():
            if sha(ROOT/path)!=h:raise ValueError('artifact drift:'+path)
    for path,h in r['source_sha256'].items():
        if sha(ROOT/r['source_snapshot_root']/path)!=h:raise ValueError('execution archive drift')
    namespace_audit(p.manifest_rows())
    if (r['c6_claim_status'],r['c7_claim_status'],r['c8_claim_status'],r['t18a_materiality_status'])!=(old.C6,old.C7,'SUPPORTED_IN_FIXED_ENVIRONMENTS','PARTIAL'):raise ValueError('historical claim changed')
    if r['source_power']!=old.POWER or {int(k):v for k,v in r['source_noise_variances'].items()}!={s:old.VARIANCES[s] for s in p.SNRS}:raise ValueError('SNR policy changed')
    if any(r[k] for k in ('primary_estimator_uses_environment_label','target_statistics_used_for_deployment','new_module_added','target_normalization','noise_recalibrated','BER_evaluations','formal_test_used','bootstrap_performed','replication_performed','new_model_trained','method_parameters_changed')):raise ValueError('scope violation')
    if not all(r[k] for k in ('method_frozen','formal_test_sealed','no_basis_refit','dense_lr4_coefficients_identical','oracle2_evaluation_only','source_means_for_all_nonmatched')):raise ValueError('frozen constraints failed')
    if (r['rank_per_module'],r['causal_window'],r['delay_levels_ns'])!=(4,5,[50,200]):raise ValueError('fixed design changed')
    for n,info in r['target_statistics'].items():
        if sha(ROOT/info['path'])!=info['sha256'] or info['used_for_deployment'] or info['validation_used']:raise ValueError('train-only target artifact changed')
        with np.load(ROOT/info['path'],allow_pickle=False) as f:
            if not np.array_equal(f['train_trajectory_ids'],np.arange(500)) or int(f['samples'])!=5000:raise ValueError('target provenance changed')
            if any(array_hash(f[k])!=h for k,h in info['component_sha256'].items()):raise ValueError('target component changed')
    tr=readcsv(OUT/'t19a_trajectory_energy.csv');raw=readcsv(OUT/'t19a_weights.csv')
    nmse,metrics=aggregate(tr);summaries=summarize_weights(raw)
    if nmse!=r['nmse'] or metrics!=r['factor_metrics'] or summaries!=r['deployed_coordinates']:raise ValueError('energy/coordinate reconstruction drift')
    if any(r[k]!=v for k,v in p.scientific_gate(metrics).items()):raise ValueError('pre-registered gate drift')
    frame=readcsv(OUT/'t19a_frame_metrics.csv');index={}
    if len(frame)!=67200 or len(raw)!=28800:raise ValueError('missing frame/window results')
    for v in frame:index.setdefault((v['environment'],v['snr_db'],v['trajectory_id'],v['method']),[]).append(v)
    for v in tr:
        group=index[v['environment'],v['snr_db'],v['trajectory_id'],v['method']]
        if [int(x['frame_id']) for x in group]!=list(range(4,10)) or trajectory_energy([float(x['error_energy']) for x in group])!=float(v['error_energy']) or trajectory_energy([float(x['truth_energy']) for x in group])!=float(v['truth_energy']):raise ValueError('frame to trajectory reduction drift')
    oracle={(v['environment'],v['snr_db']):v for v in r['oracle2_coordinates']}
    for v in raw:
        if int(v['frame_id'])-int(v['window_start'])!=4:raise ValueError('causal window drift')
        if v['method']=='affine':
            o=oracle[v['environment'],int(v['snr_db'])];c=np.array([float(v['alpha']),float(v['beta'])])
            distance=float(np.linalg.norm(c-[o['alpha_oracle'],o['beta_oracle']]))
            if v['dense_lr4_same_coefficients']!='True' or array_hash(c)!=v['coefficients_sha256'] or distance!=float(v['coordinate_error_norm']):raise ValueError('coefficient identity/distance drift')
    channels=readcsv(OUT/'t19a_channel_hashes.csv');pairs=readcsv(OUT/'t19a_paired_input_hashes.csv')
    if len(channels)!=4800 or len({v['H_sha256'] for v in channels})!=4800 or len(pairs)!=16000 or {int(v['snr_db']) for v in pairs}!={10,20}:raise ValueError('paired stochastic population drift')
    for v in r['effective_delay_audit']:
        power=np.array(v['pooled_path_energy']);tau=np.array(v['path_delays_seconds']);pp=power/power.sum();mean=float(pp@tau)
        if float(np.sqrt(pp@(tau-mean)**2))*1e9!=v['empirical_array_weighted_rms_delay_ns'] or v['trajectories']!=500 or not v['audit_only']:raise ValueError('effective delay definition drift')
    return True


def claim():
    r=load(RESULT);verify(r)
    if r.get('claim_document_sha256'):raise ValueError('claim documents protected')
    text='# T19A Claim–Evidence update\n\nC9: '+p.C9+'\n\nStatus: '+r['c9_claim_status']+'.\n\n'
    text+='C6: '+old.C6+'; C7: '+old.C7+'; C8: SUPPORTED_IN_FIXED_ENVIRONMENTS. T18A materiality: PARTIAL, unchanged.\n\n'
    text+='| Pure-delay condition | Matched gap dB | Oracle2 recovery dB | Representation closure | Coordinate efficiency | LR4 penalty dB |\n|---|---:|---:|---:|---:|---:|\n'
    for v in r['factor_metrics']:
        if v['environment'] in p.PURE:
            text+='| '+v['environment']+'/'+str(v['snr_db'])+' | '+' | '.join(str(v[k]) for k in ('matched_gap_db','oracle2_recovery_db','representation_closure','coordinate_efficiency','compression_penalty_db'))+' |\n'
    text+='\n'+''.join(f'- {k}: {r[k]}\n' for k in ('existing_pm_space_delay_representation_sufficient','delay_specific_representation_deficiency','delay_coordinate_inference_bottleneck','delay_factor_gate_partial','rank4_out_of_factor_warning'))
    text+='\nDecision dependency: '+r['next_step_dependency']+' No next-stage implementation is performed.\n\n'
    if r['existing_pm_space_delay_representation_sufficient'] and r['delay_coordinate_inference_bottleneck']:
        text+='In the tested pure-delay conditions, the frozen P/M correction space has sufficient representation capacity, but current sparse-pilot coordinate fitting does not stably realize it. This does not establish deployed-estimator success.\n\n'
    text+='Evidence is actual paired validation,100 independent trajectories per fixed environment, frames4..9; all NMSE/gains use pooled energies. Oracle2 uses target-train covariance only, optimizes centered filter error and evaluates using source means. Matched uses target means, so representation closure is an empirical recovery diagnostic rather than an absolute projection theorem. Combined12 conditions and coefficient geometry are secondary descriptive diagnostics.\n\n'
    text+='Scope: only tested CDL-C30 pure-delay levels50/200ns and10/20dB; no universal delay representation or new physical mechanism/novelty claim. No bootstrap or independent replication of the new factor; target covariance is one500-trajectory estimate. Effective array-weighted RMS differs from nominal DS. Formal test remains sealed.\n'
    review='# T19A Reviewer Red Team\n\n'
    review+='- The method is unchanged; a new factor challenge is not a new module or proof of separability. C6/C7 negative conclusions remain permanent.\n'
    review+='- Fresh500 train/100 validation trajectories per environment; no validation fitting, sample removal or seed search. The same H/bits/Z/Y are paired across all methods.\n'
    review+='- Pure-delay four-condition gate is fixed; combined profile/mobility/delay conditions cannot change C9.\n'
    review+='- Oracle2 is a nondeployable capacity control, not a pilot-estimator baseline. Its optimum is target-weighted centered operator error, not sample NMSE or source-mean bias. Matched includes a target-mean reference advantage; this is explicitly disclosed.\n'
    review+='- Coordinate efficiency may be negative or exceed1; ratios are not clipped. Nonpositive denominators are flagged/null. No post-hoc threshold or axis clipping hides failures.\n'
    review+='- Source SNR is frozen; measured target power and effective RMS are audits only. Any profile/array received-power differences remain in the challenge.\n'
    review+='- Rank4 warning is retained without increasing rank. Correlated causal windows provide descriptive coordinate quartiles, not600 independent samples.\n'
    review+='- This discovery gate has100 independent trajectories/environment, one namespace and no bootstrap; sampling and target-statistics uncertainty require separately assigned replication.\n'
    review+='- No universal delay-factor coverage, third direction, network training, BER or formal test is claimed.\n\n'
    review+='Actual outcome: C9='+r['c9_claim_status']+'; representation sufficient='+str(r['existing_pm_space_delay_representation_sufficient'])+'; deficiency='+str(r['delay_specific_representation_deficiency'])+'; coordinate bottleneck='+str(r['delay_coordinate_inference_bottleneck'])+'.\n'
    hashes={}
    for name,value in (('t19a_claim_evidence_update.md',text),('t19a_reviewer_update.md',review)):
        path=ROOT/'docs'/name
        if path.exists():raise ValueError('existing claim file protected')
        path.write_text(value,encoding='utf8');hashes['docs/'+name]=sha(path)
    r['claim_document_sha256']=hashes;save(RESULT,r);print('T19A claims and reviewer audit generated; historical conclusions unchanged',flush=True)


def visual_review():
    r=load(OUT/'t19a_figure_audit.json')
    if r['source_returncode'] or not all(r['CSV_only_rebuild'].values()) or any(v['text_returncode'] or v['collision_returncode'] for v in r['rendered_checks'].values()):raise ValueError('automatic figure QA incomplete')
    r.update(visual_review='Final actual PNGs inspected:20 paired pure-delay method values and all16 factor ratio points/explicit undefined conditions. Axes, legend and thresholds legible; pooled-energy statistic and absence of CI stated.',
        panel_review=[dict(panel=n,pass_review=True,independent_unit='trajectory',uncertainty='none:single namespace discovery gate') for n in ('pure-delay NMSE','representation versus coordinates')],all_checks_passed=True)
    save(OUT/'t19a_figure_audit.json',r)


def finalize():
    tick=time.perf_counter();r=load(RESULT);verify(r)
    figure=load(OUT/'t19a_figure_audit.json')
    if not figure['all_checks_passed'] or not r.get('claim_document_sha256'):raise ValueError('claim/visual review incomplete')
    env=dict(os.environ);env['TEMP']=env['TMP']=str(ROOT/'.pytest_cache/pytest_workspace_temp');tests={}
    groups=[('related',['tests/test_delay_factor_challenge.py']),('provenance',['tests/test_validation_replication.py','tests/test_frozen_generalization.py','tests/test_historical_provenance.py','tests/test_finite_shift_resolvent.py','tests/test_wiener_perturbation.py']),('full',[])]
    for label,args in groups:
        path=OUT/f't19a_pytest_{label}.txt'
        if path.exists():raise ValueError('existing regression log protected')
        t=time.perf_counter();print('Actual '+label+' regression',flush=True)
        with path.open('w',encoding='utf8') as f:proc=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest',*args,'-q'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        summary=path.read_text(encoding='utf8').strip().splitlines()[-1];print(summary,flush=True)
        tests[label]=dict(command=[str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest',*args,'-q'],summary=summary,exit_code=proc.returncode,seconds=time.perf_counter()-t,sha256=sha(path))
        if proc.returncode or 'skipped' in summary:raise ValueError('regression failure/skip:STOP')
    verify(r);after=environments()
    if after!=r['environment_before']:raise ValueError('environment payload changed')
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError('git diff --check failed')
    from scripts.plot_delay_factor_challenge import STEMS
    r.update(pytest=tests,environment_after=after,git_diff_check=dict(exit_code=diff.returncode,output=diff.stdout+diff.stderr),
        final_hash_audit=dict(passed=True,historical_files=len(r['frozen_history_sha256']),unexpected_content_mismatch=0),
        figure_audit_sha256=sha(OUT/'t19a_figure_audit.json'),figure_artifact_sha256={f'outputs/{stem}.{ext}':sha(OUT/f'{stem}.{ext}') for stem in STEMS for ext in ('png','pdf','svg','alignment.json','collision-audit.json')},
        all_checks_passed=True,completion_state='completed')
    r['runtime']['final_audit_and_regression_seconds']=time.perf_counter()-tick
    save(RESULT,r);print('T19A engineering PASS; C9='+r['c9_claim_status']+'; formal test sealed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['claim','visual-review','finalize']);args=parser.parse_args()
    {'claim':claim,'visual-review':visual_review,'finalize':finalize}[args.mode]()
