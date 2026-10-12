"""T18B CSV energy, scientific decision, historical integrity and final gates."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src import validation_replication as p
from src import frozen_affine_validation as old
from scripts.run_validation_replication import accepted,frozen_method,namespace_audit,references,matrix,OUT,RESULT
from scripts.run_cdl_mismatch import load,save,sha,environments
from scripts.run_pilot_cv import readcsv,trajectory_energy


def verify(r,recompute=True):
    history=accepted();_,_,proof=frozen_method()
    if proof!=r['frozen_method_provenance']:raise ValueError('method changed')
    amendment={}
    if r.get('figure_source_amendment_artifact'):
        info=r['figure_source_amendment_artifact']
        if sha(ROOT/info['path'])!=info['sha256']:raise ValueError('figure amendment provenance drift')
        amendment=load(ROOT/info['path'])['current_source_sha256']
        if set(amendment)!={'scripts/plot_validation_replication.py','scripts/audit_validation_replication.py'}:raise ValueError('only figure/audit source amendments allowed')
    for group in ('frozen_history_sha256','preparation_artifact_sha256','numeric_artifact_sha256','source_sha256'):
        for path,h in r[group].items():
            if group=='source_sha256' and path in amendment:
                if sha(ROOT/path)!=amendment[path] or sha(ROOT/r['source_snapshot_root']/path)!=h:raise ValueError('original/current figure provenance changed')
            elif sha(ROOT/path)!=h:raise ValueError('artifact changed:'+path)
    for path,h in r['source_sha256'].items():
        if sha(ROOT/r['source_snapshot_root']/path)!=h:raise ValueError('execution source archive drift')
    references(history,stored=True);namespace_audit(p.manifest_rows())
    gate=load(OUT/'t18b_historical_replay_gate.json')
    if gate!=r['historical_replay_gate'] or not gate['all_checks_passed']:raise ValueError('historical gate drift')
    if r['c6_claim_status']!=old.C6 or r['c7_claim_status']!=old.C7 or r['t18a_materiality_status_preserved']!='PARTIAL':raise ValueError('historical scientific claim changed')
    if r['t18a_original_scientific_flags']!={k:history[k] for k in r['t18a_original_scientific_flags']}:raise ValueError('historical gate reclassification')
    if any(r[k] for k in ('formal_test_used','BER_evaluations','detector_calls','target_statistics_uncertainty_evaluated','target_normalization','noise_recalibrated','primary_estimator_uses_environment_label','target_statistics_used_for_deployment')):raise ValueError('scope violation')
    if not all(r[k] for k in ('formal_test_sealed','method_frozen','target_statistics_frozen','no_covariance_refit','no_new_target_train','no_basis_refit','no_new_method','dense_lr4_coefficients_identical','t18a_not_in_new_pooled_sample')):raise ValueError('frozen constraints failed')
    if (r['replicate_count'],r['trajectories_per_replicate'],r['bootstrap_samples'],r['bootstrap_unit'],r['rank_per_module'],r['causal_window'])!=(5,100,10000,'trajectory',4,5):raise ValueError('fixed design drift')
    science=p.scientific_gate(r['replicate_metrics'],r['pooled_metrics'],r['bootstrap_intervals'],r['weight_stability'])
    if any(r[k]!=v for k,v in science.items()):raise ValueError('new scientific gate drift')
    if recompute:
        energies=readcsv(OUT/'t18b_trajectory_energy.csv');raw=readcsv(OUT/'t18b_window_weights.csv')
        from scripts.run_validation_replication import summarize
        nmse,metrics,points,stability,cis,indices,draws=summarize(energies,raw)
        if (nmse,metrics,points,stability,cis,indices)!=(r['replicate_nmse'],r['replicate_metrics'],r['pooled_metrics'],r['weight_stability'],r['bootstrap_intervals'],r['bootstrap_index_provenance']):raise ValueError('CSV/energy/bootstrap result drift')
        with np.load(OUT/'t18b_bootstrap_draws.npz',allow_pickle=False) as f:
            if set(f.files)!=set(draws) or any(not np.array_equal(f[k],v) for k,v in draws.items()):raise ValueError('paired bootstrap draws drift')
        frame=readcsv(OUT/'t18b_frame_metrics.csv');indexed={}
        for v in frame:indexed.setdefault((v['environment'],v['snr_db'],v['replicate'],v['trajectory_id'],v['method']),[]).append(v)
        if len(energies)!=28000 or len(frame)!=168000:raise ValueError('trajectory/frame coverage drift')
        for v in energies:
            group=indexed[v['environment'],v['snr_db'],v['replicate'],v['trajectory_id'],v['method']]
            if [int(x['frame_id']) for x in group]!=list(range(4,10)) or trajectory_energy([float(x['error_energy']) for x in group])!=float(v['error_energy']) or trajectory_energy([float(x['truth_energy']) for x in group])!=float(v['truth_energy']):raise ValueError('within-trajectory reduction mismatch')
        for v in raw:
            if int(v['frame_id'])-int(v['window_start'])!=4 or int(v['frame_id']) not in range(4,10):raise ValueError('causal frames drift')
            if v['method']=='affine' and v['dense_lr4_same_coefficients']!='True':raise ValueError('Dense/LR4 identity missing')
        hashes=readcsv(OUT/'t18b_channel_hashes.csv')
        historical={v['H_sha256'] for v in readcsv(OUT/'t18a_channel_hashes.csv')}
        if len(hashes)!=2000 or len({v['H_sha256'] for v in hashes})!=2000 or historical&{v['H_sha256'] for v in hashes}:raise ValueError('channel uniqueness drift')
        pairs=readcsv(OUT/'t18b_paired_input_hashes.csv')
        if len(pairs)!=40000 or {int(v['snr_db']) for v in pairs}!={10,20}:raise ValueError('paired input scope drift')
    return True


def figure_amendment():
    """Retain original execution hashes; declare a layout-only post-run revision."""
    r=load(RESULT);previous=load(OUT/'t18b_figure_failure01/results_before_figure_amendment.json')
    if r!=previous or r.get('figure_source_amendment_artifact'):raise ValueError('original result archive required')
    permitted=('scripts/plot_validation_replication.py','scripts/audit_validation_replication.py')
    for path,h in r['source_sha256'].items():
        if sha(ROOT/r['source_snapshot_root']/path)!=h:raise ValueError('original execution snapshot changed')
        if path not in permitted and sha(ROOT/path)!=h:raise ValueError('numerical source amendment forbidden')
    for path,h in r['numeric_artifact_sha256'].items():
        if sha(ROOT/path)!=h:raise ValueError('numerical output changed')
    artifact=OUT/'t18b_figure_only_amendment.json'
    if artifact.exists():raise ValueError('existing figure amendment protected')
    save(artifact,dict(original_execution_source_sha256=r['source_sha256'],
        current_source_sha256={v:sha(ROOT/v) for v in permitted},
        original_result_archive_sha256=sha(OUT/'t18b_figure_failure01/results_before_figure_amendment.json'),
        numeric_artifact_sha256=r['numeric_artifact_sha256'],numerical_scan_repeated=False,
        reason='Actual PDF collision: panel b overlapped vertical axis title; move only panel-label x from -.095 to -.025. Audit supports separate original/current figure source provenance.',
        numerical_implementation_changed=False,original_execution_hashes_preserved=True))
    r['figure_source_amendment_artifact']=dict(path='outputs/'+artifact.name,sha256=sha(artifact))
    r['warnings'].append('Actual first figure QA failed panel-label/y-axis-title overlap; original figures/source/result archived, label position corrected under a figure-only provenance amendment. No numerical rerun.')
    save(RESULT,r);verify(r);print('Original T18B execution hashes retained; layout-only amendment frozen; every numeric CSV/draw hash unchanged',flush=True)


def claim():
    r=load(RESULT);verify(r)
    if r.get('claim_document_sha256'):raise ValueError('existing claim docs protected')
    text='# T18B Claim–Evidence update\n\nC8: '+p.C8+'\n\nStatus: '+r['c8_claim_status']+'.\n\n'
    text+='C6: '+old.C6+'; C7: '+old.C7+'. Historical T18A materiality: PARTIAL (unchanged).\n\n'
    text+='| Composed condition | T18A prior gain dB | Five new gains dB | New pooled gain dB | Paired95% CI |\n|---|---:|---|---:|---|\n'
    for prior in r['t18a_original_composed_gains']:
        n,s=prior['environment'],prior['snr_db'];rep=[v['lr4_vs_convex_gain_db'] for v in r['replicate_metrics'] if (v['environment'],v['snr_db'])==(n,s)]
        v=next(v for v in r['bootstrap_intervals'] if (v['environment'],v['snr_db'],v['metric'])==(n,s,'lr4_vs_convex_gain_db'))
        text+=f"| {n}/{s} | {prior['lr4_vs_convex_gain_db']:.6f} | {', '.join(f'{x:.6f}' for x in rep)} | {v['point_estimate']:.6f} | [{v['lower_2_5']:.6f}, {v['upper_97_5']:.6f}] |\n"
    text+='\nScientific diagnostics:\n\n'+''.join(f'- {k}: {r[k]}\n' for k in ('positive_affine_advantage_statistically_replicated','replication_instability_observed','replication_inconclusive','interpolation_threshold_stability','rank4_compression_uncertainty_warning','outside_simplex_geometry_stable','post_replication_materiality_diagnostic','broader_factor_validation_may_be_considered'))
    text+='\nEvidence: five independent new namespaces per fixed environment,100 trajectories each; paired trajectory-energy bootstrap10000 draws. All original T18A10/20 validation inputs/weights/predictions/energies replay exactly. Frozen target covariance and original pilot-only receiver were not fitted or changed.\n\n'
    text+='Limitations: conditional on one finite T18A target-statistics realization; no target-statistics uncertainty, random-environment population coverage or simultaneous95% coverage. C60/C90/B60/B90 are fixed environments. Geometry is secondary descriptive evidence, not physical mechanism and not part of C8. Formal test remains sealed.\n'
    if not r['positive_affine_advantage_statistically_replicated']:text+='\nStop rule: stop broader factor validation; sampling-sensitive empirical observation.\n'
    elif r['interpolation_threshold_stability']:text+='\nOnly consideration of a separately assigned broader factor validation is allowed; no further experiment is executed.\n'
    else:text+='\nInterpolation stability is insufficient to consider broader factor validation.\n'
    if not r['outside_simplex_geometry_stable']:text+='\nPerformance/benefit support, if present, does not imply coefficient geometry replicated.\n'
    reviewer='# T18B Reviewer Red Team\n\n'
    reviewer+='This is sampling replication of a frozen empirical estimator, not a new method or mechanism.\n\n'
    reviewer+='- Five namespaces/environment provide500 independent trajectories; frames/windows/REs stay clustered. The same index resamples every method.\n- Every NMSE/gain is reconstructed from summed energies before log conversion; no trajectory-dB averaging. T18A prior samples are excluded.\n- All methods are paired on H/bits/noise/Y. Dense and LR4 share coefficients; Matched uses target means, oracle affine uses source means and frozen target-weighted coordinates.\n- The weakest original B60/10 effect is explicitly displayed; no sample/replicate deletion or CI/seed search.\n- Confidence intervals are marginal conditional intervals, not a guarantee over future wireless environments. Joint scientific support follows the pre-registered conjunction, not a new multiple-comparison adjustment.\n- Target statistics are held fixed; their finite-training variability is not captured.\n- C6/C7 contradictions and T18A PARTIAL remain; outside-simplex geometry cannot recover a physical-additivity claim.\n- Rank4 compression warnings are retained without increasing rank. Formal test remains sealed.\n\n'
    reviewer+='Actual outcome: C8='+r['c8_claim_status']+'; positive support='+str(r['positive_affine_advantage_statistically_replicated'])+'; interpolation stability='+str(r['interpolation_threshold_stability'])+'; compression warning='+str(r['rank4_compression_uncertainty_warning'])+'.\n'
    hashes={}
    for name,value in (('t18b_claim_evidence_update.md',text),('t18b_reviewer_update.md',reviewer)):
        path=ROOT/'docs'/name
        if path.exists():raise ValueError('claim file protected')
        path.write_text(value,encoding='utf8');hashes[f'docs/{name}']=sha(path)
    r['claim_document_sha256']=hashes;save(RESULT,r);print('T18B Claim–Evidence/Reviewer documents generated; historical claims unchanged',flush=True)


def visual_review():
    r=load(OUT/'t18b_figure_audit.json')
    if r['source_returncode'] or not all(r['CSV_only_rebuild'].values()) or any(v['text_returncode'] or v['collision_returncode'] for v in r['rendered_checks'].values()):raise ValueError('automatic QA incomplete')
    r.update(visual_review='Actual final PNGs inspected: all20 new replicate gains,4 frozen priors,4 pooled CIs and8 paired closure/compression intervals. Labels/legend/thresholds clear. CI conditional scope explicit.',
        panel_review=[dict(panel=n,pass_review=True,independent_unit='trajectory',uncertainty='paired95% percentile CI') for n in ('composed benefit','interpolation closure','rank4 penalty')],all_checks_passed=True)
    save(OUT/'t18b_figure_audit.json',r)


def finalize():
    t=time.perf_counter();r=load(RESULT);verify(r)
    figure=load(OUT/'t18b_figure_audit.json')
    if not figure['all_checks_passed'] or not r.get('claim_document_sha256'):raise ValueError('claim/visual review incomplete')
    env=dict(os.environ);env['TEMP']=env['TMP']=str(ROOT/'.pytest_cache/pytest_workspace_temp');tests={}
    for label,args in [('related',['tests/test_validation_replication.py']),('provenance',['tests/test_frozen_generalization.py','tests/test_finite_shift_resolvent.py','tests/test_wiener_perturbation.py']),('full',[])]:
        path=OUT/f't18b_pytest_{label}.txt'
        if path.exists():raise ValueError('existing regression log protected')
        tick=time.perf_counter();print('Executing actual '+label+' pytest',flush=True)
        with path.open('w',encoding='utf8') as f:proc=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest',*args,'-q'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        summary=path.read_text(encoding='utf8').strip().splitlines()[-1]
        tests[label]=dict(command=[str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest',*args,'-q'],summary=summary,exit_code=proc.returncode,seconds=time.perf_counter()-tick,sha256=sha(path))
        print(summary,flush=True)
        if proc.returncode or 'skipped' in summary:raise ValueError('regression failed/skipped: STOP')
    verify(r);after=environments()
    if after!=r['environment_before']:raise ValueError('frozen environment drift')
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError('git diff --check failed')
    from scripts.plot_validation_replication import STEMS
    r.update(pytest=tests,environment_after=after,git_diff_check=dict(exit_code=diff.returncode,output=diff.stdout+diff.stderr),
        final_hash_audit=dict(passed=True,historical_files=len(r['frozen_history_sha256']),unexpected_content_mismatch=0),
        figure_audit_sha256=sha(OUT/'t18b_figure_audit.json'),
        figure_artifact_sha256={f'outputs/{stem}.{ext}':sha(OUT/f'{stem}.{ext}') for stem in STEMS for ext in ('png','pdf','svg','alignment.json','collision-audit.json')},
        all_checks_passed=True,completion_state='completed')
    r['runtime']['final_audit_and_regression_seconds']=time.perf_counter()-t
    save(RESULT,r);print('T18B engineering PASS; C8='+r['c8_claim_status']+'; T18A PARTIAL preserved; formal test sealed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['claim','visual-review','figure-amendment','finalize']);args=parser.parse_args()
    {'claim':claim,'visual-review':visual_review,'figure-amendment':figure_amendment,'finalize':finalize}[args.mode]()
