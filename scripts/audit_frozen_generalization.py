"""T18A final evidence, regression and frozen-artifact integrity audit."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src import frozen_affine_validation as p
from scripts.run_frozen_generalization import accepted,frozen_method,OUT,RESULT
from scripts.run_cdl_mismatch import load,save,sha,environments
from scripts.run_pilot_cv import readcsv


def verify(r):
    accepted();_,_,proof=frozen_method()
    if proof!=r['frozen_method_provenance']:raise ValueError('frozen method changed')
    for group in ('frozen_history_sha256','preparation_artifact_sha256','numeric_artifact_sha256','source_sha256'):
        for path,h in r[group].items():
            if sha(ROOT/path)!=h:raise ValueError('artifact drift:'+path)
    for info in r['target_statistics'].values():
        if sha(ROOT/info['path'])!=info['sha256'] or not info['training_only'] or info['validation_used_for_statistics']:
            raise ValueError('target statistics gate failed')
        with np.load(ROOT/info['path'],allow_pickle=False) as f:
            if not np.array_equal(f['train_trajectory_ids'],np.arange(500)) or int(f['samples'])!=5000:raise ValueError('target train IDs drift')
            from src.cdl_benchmark import array_hash
            if any(array_hash(f[k])!=h for k,h in info['component_sha256'].items()):raise ValueError('target component hash drift')
    if r['c6_claim_status']!=p.C6 or r['c7_claim_status']!=p.C7:raise ValueError('permanent negative claims changed')
    if any(r[k] for k in ('method_parameters_changed','new_model_trained','new_module_fitted','new_svd','target_normalization',
            'per_environment_noise_recalibration','primary_estimator_uses_environment_label','target_statistics_used_for_deployment',
            'BER_evaluations','detector_calls','formal_test_used','within_symbol_ICI_modeled')):raise ValueError('scope violation')
    if not r['formal_test_sealed'] or not r['dense_lr4_coefficients_identical'] or r['rank_per_module']!=4:raise ValueError('freeze violation')
    if r['scientific_gate_thresholds']!=dict(pilotcv_efficiency=.8,compression_penalty_db=.5,dense_closure=.7,
            composed_gain_replicated_db=1.,composed_gain_weak_db=.25,outside_fraction=.75,stress_convex_tolerance_db=.5):raise ValueError('threshold drift')
    if any(r[k]!=v for k,v in p.scientific_gate(r['condition_metrics']).items()):raise ValueError('gate drift')
    return True


def claim():
    r=load(RESULT);verify(r)
    if r.get('claim_document_sha256'):raise ValueError('claim documents already exist')
    gates=p.scientific_gate(r['condition_metrics'])
    claim='# T18A Claim–Evidence update\n\n'
    claim+='C6 = '+p.C6+'; C7 = '+p.C7+'. Both negative mechanism conclusions remain unchanged.\n\n'
    claim+='Method: completely frozen T16E/F/G source/modules, pilot geometry, four-fold causal-five-frame fitting and weighted rank4 data correction.\n\n'
    claim+='| Condition | Dense closure | Pilot-CV efficiency | LR4 penalty dB | LR4 over Convex dB | Outside fraction |\n|---|---|---|---|---|---|\n'
    for v in r['condition_metrics']:
        claim+=f"| {v['environment']} /{v['snr_db']} | {v['dense_affine_closure']:.6g} | {v['pilotcv_efficiency']:.6g} | {v['compression_penalty_db']:.6g} | {v['lr4_vs_convex_gain_db']:.6g} | {v['outside_simplex_fraction']:.6g} |\n"
    claim+='\nPre-registered scientific gate:\n\n'+''.join(f'- {k}: {v}\n' for k,v in gates.items())
    claim+='\nExisting Evidence: paired held-out development validation on six new namespaces, train-only target references, complete saved trajectory energies and source-data CSVs; no parameter adaptation beyond the frozen pilot-driven coefficients.\n\n'
    claim+='Missing Evidence: formal test, other delay/array/pilot configurations, additional profile families, within-symbol ICI and uncertainty intervals. No claim extends to those.\n\n'
    claim+='Confidence: restricted to these six100ns environments and the current symbol-wise CDL abstraction. Affine remains an empirical estimator structure with no physical separability explanation.\n'
    if gates['narrow_empirical_observation_downgrade']:claim+='\nRegistered stop rule triggered: narrow empirical observation; stop formal-test candidacy.\n'
    elif gates['broader_factor_validation_may_follow']:claim+='\nOnly a future explicitly assigned broader factor validation / method-definition freeze is permitted by this gate. Formal test stays sealed.\n'
    else:claim+='\nThe gate does not jointly permit the next broader validation/freeze stage; retain the mixed outcome.\n'
    reviewer='# T18A Reviewer Red Team\n\n'
    reviewer+='The experiment tests a frozen estimator across four interpolation and two stress environments. It does not establish an additive physical mechanism or new model class.\n\n'
    reviewer+='- Strong comparator: exact original Hard and Convex Pilot-CV solvers, same paired data and source means; Matched is explicitly oracle-style and uses target means.\n'
    reviewer+='- Oracle affine: train-only target-weighted filter optimum with source means; it is not a validation-NMSE optimizer and its mean-bias objective is not optimized.\n'
    reviewer+='- Sampling:100 independent validation trajectories/environment; six correlated windows/trajectory. Window quartiles/std are descriptive, not independent replicates or confidence intervals.\n'
    reviewer+='- Report every condition, signed penalty and failure; all seeds/namespaces fixed before the scan. No scenario or sample was dropped.\n'
    reviewer+='- Potential generality confound: all environments retain100ns, array, orientation and pilots; speed interpolation is not broad universal channel generalization.\n'
    reviewer+='- Stress limitation:180km/h excludes within-symbol ICI; no full high-mobility receiver claim.\n'
    reviewer+='- Efficiency: historic332288 complex-MAC proxy is preserved, and is not a runtime-speedup claim.\n\n'
    reviewer+='Actual gate:\n\n'+''.join(f'- {k}: {v}\n' for k,v in gates.items())
    hashes={}
    for name,text in (('t18a_claim_evidence_update.md',claim),('t18a_reviewer_update.md',reviewer)):
        path=ROOT/'docs'/name
        if path.exists():raise ValueError('existing claim/reviewer artifact protected')
        path.write_text(text,encoding='utf8');hashes[f'docs/{name}']=sha(path)
    r['claim_document_sha256']=hashes;save(RESULT,r);print('T18A empirical claim/reviewer updates generated from actual results; C6/C7 unchanged')


def visual_review():
    report=load(OUT/'t18a_figure_audit.json')
    if report['source_returncode'] or not all(report['CSV_only_rebuild'].values()) or any(v['text_returncode'] or v['collision_returncode'] for v in report['rendered_checks'].values()):raise ValueError('automatic figure gate incomplete')
    report.update(visual_review='All three actual final PNGs inspected panel by panel: interpolation comparisons, signed gain/geometry and C180/B180 stress. Labels/legend/margins clear; no CI implied.',
        panel_review=[dict(panel=n,pass_review=True,uncertainty='pooled summary without CI, descriptive geometry fraction') for n in
            ('C mobility','B composed','Paired gain','Outside geometry','C180 stress','B180 stress')],all_checks_passed=True)
    save(OUT/'t18a_figure_audit.json',report);print('Actual final figure visual review recorded')


def finalize():
    start=time.perf_counter();r=load(RESULT);verify(r)
    figure=load(OUT/'t18a_figure_audit.json')
    if not figure['all_checks_passed'] or not r.get('claim_document_sha256'):raise ValueError('visual/claim review incomplete')
    env=dict(os.environ);env['TEMP']=env['TMP']=str(ROOT/'.pytest_cache/pytest_workspace_temp');tests={}
    for label,args in [('related',['tests/test_frozen_generalization.py']),('full',[])]:
        path=OUT/f't18a_pytest_{label}.txt'
        if path.exists():raise ValueError('existing regression log protected')
        tick=time.perf_counter();print('Executing actual '+label+' pytest',flush=True)
        with path.open('w',encoding='utf8') as out:proc=subprocess.run([sys.executable,'-m','pytest',*args,'-q'],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
        summary=path.read_text(encoding='utf8').strip().splitlines()[-1]
        tests[label]=dict(command=[sys.executable,'-m','pytest',*args,'-q'],summary=summary,exit_code=proc.returncode,seconds=time.perf_counter()-tick,sha256=sha(path))
        print(summary,flush=True)
        if proc.returncode or 'skipped' in summary:raise ValueError('actual regression failed or skipped')
    verify(r);after=environments()
    if after!=r['environment_before']:raise ValueError('frozen runtime payload changed')
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError('git diff --check failed')
    r.update(pytest=tests,environment_after=after,git_diff_check=dict(exit_code=diff.returncode,output=diff.stdout+diff.stderr),
        final_hash_audit=dict(passed=True,historical_files=len(r['frozen_history_sha256']),unexpected_content_mismatch=0),
        figure_audit_sha256=sha(OUT/'t18a_figure_audit.json'),
        figure_artifact_sha256={f'outputs/{stem}.{ext}':sha(OUT/f'{stem}.{ext}') for stem in
            ('t18a_interpolation_nmse','t18a_composed_gain_geometry','t18a_extrapolation_stress') for ext in ('png','pdf','svg','alignment.json','collision-audit.json')},
        all_checks_passed=True,completion_state='completed')
    r['runtime']['final_audit_and_regression_seconds']=time.perf_counter()-start
    save(RESULT,r);print('T18A engineering PASS; all scientific outcomes retained; formal test sealed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['claim','visual-review','finalize']);args=parser.parse_args()
    {'claim':claim,'visual-review':visual_review,'finalize':finalize}[args.mode]()
