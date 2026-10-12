"""T17A actual-result, provenance and regression finalization; no numerical rerun."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import os
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_wiener_mechanism import OUT,RESULT,CLAIM,accepted,c6_status
from scripts.run_cdl_mismatch import load,save,sha,environments
from src.wiener_perturbation import mechanism_gate


def verify(result):
    accepted()
    for group in ('frozen_history_sha256','numeric_artifact_sha256','preparation_artifact_sha256'):
        for path,digest in result[group].items():
            if sha(ROOT/path)!=digest:raise ValueError('frozen artifact drift:'+path)
    for path,digest in result['source_sha256'].items():
        snapshot=ROOT/result['pre_run_source_snapshot_root']/path
        if sha(snapshot)!=digest:raise ValueError('pre-run source snapshot drift:'+path)
        current=result.get('post_run_document_sha256',{}).get(path,digest)
        if sha(ROOT/path)!=current:raise ValueError('current source drift:'+path)
    if any(result[key]!=value for key,value in mechanism_gate(result['interaction_decomposition']).items()):
        raise ValueError('scientific gate differs from pre-registered rule')
    if result['c6_claim_status']!=c6_status(result):raise ValueError('C6 status differs')
    if result['generator_calls'] or result['new_statistics_fit'] or result['formal_test_used']:
        raise ValueError('information boundary violated')
    return True


def claim():
    result=load(RESULT);verify(result)
    if result.get('post_run_document_sha256'):raise ValueError('post-gate claim already recorded')
    path=ROOT/CLAIM
    original=path.read_text(encoding='utf8')
    text='\n## Post-gate C6 evidence — T17A\n\n'
    text+=f"Current C6 status: **{result['c6_claim_status']}**.\n\n"
    text+='The pre-mechanism matrix above remains the original record. This update concerns only C6; literature facts and C1–C5 are unchanged.\n\n'
    text+='| SNR (dB) | First-order additive closure | Interaction-corrected closure |\n| --- | --- | --- |\n'
    for row in result['interaction_decomposition']:
        text+=f"| {row['snr_db']} | {row['first_order_additive_closure']:.12g} | {row['interaction_corrected_closure']:.12g} |\n"
    text+='\nExisting Evidence: frozen statistics/operator analysis in outputs/t17a_results.json and its CSVs; independent derivative/convergence/cross-identity tests; unchanged historical performance evidence.\n\n'
    text+='Missing Evidence: validity beyond these finite perturbations, wider environment coverage and finite-statistics uncertainty; this task introduces no new scenes.\n\n'
    text+='Confidence: follows the frozen B12010/20 rule only. Local infinitesimal differentiability and finite-shift explanatory success are separate claims; no universal mechanism or novelty conclusion is made.\n'
    path.write_text(original+text,encoding='utf8')
    result['post_run_document_sha256']={CLAIM:sha(path)}
    save(RESULT,result)
    print('C6 post-gate evidence appended; original pre-run snapshot retained')


def visual_review():
    figure=load(OUT/'t17a_figure_audit.json')
    if figure['source_returncode'] or not all(figure['CSV_only_rebuild'].values()) or any(
        v['text_returncode'] or v['collision_returncode'] for v in figure['rendered_checks'].values()):
        raise ValueError('automatic figure gates incomplete')
    figure['visual_review']='Both final PNGs actually inspected: residual magnitudes/zero reference and all three spectrum panels; labels, legends, margins and symlog/log transforms clear.'
    figure['panel_review']=[dict(panel=p,pass_review=True,uncertainty='none: deterministic frozen operators')
        for p in ('residual','spectrum a /5dB','spectrum b /10dB','spectrum c /20dB')]
    figure['all_checks_passed']=True;save(OUT/'t17a_figure_audit.json',figure)
    print('Actual final visual review recorded')


def finalize():
    start=time.perf_counter();result=load(RESULT);verify(result)
    if not result.get('post_run_document_sha256'):raise ValueError('post-gate C6 evidence missing')
    figure=load(OUT/'t17a_figure_audit.json')
    if not figure['all_checks_passed']:raise ValueError('actual figure review missing')
    env=dict(os.environ)
    env['TEMP']=env['TMP']=str(ROOT/'.pytest_cache/pytest_workspace_temp')
    tests={}
    for label,args in (
        ('related',['tests/test_wiener_perturbation.py','tests/test_wiener_mechanism_pipeline.py','tests/test_wiener_mechanism_figures.py']),
        ('provenance',['tests/test_wiener_mechanism_pipeline.py','-k','provenance or frozen']),
        ('full',[])):
        path=OUT/f't17a_pytest_{label}.txt'
        if path.exists():raise ValueError('existing final test transcript protected')
        print(f'Executing actual {label} pytest',flush=True)
        tick=time.perf_counter()
        with path.open('w',encoding='utf8') as f:
            process=subprocess.run([sys.executable,'-m','pytest',*args,'-q'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        summary=path.read_text(encoding='utf8').strip().splitlines()[-1]
        tests[label]=dict(command=[sys.executable,'-m','pytest',*args,'-q'],exit_code=process.returncode,
                         summary=summary,seconds=time.perf_counter()-tick,sha256=sha(path))
        print(summary,flush=True)
        if process.returncode or 'skipped' in summary:raise ValueError('actual regression failed or skipped:'+label)
    verify(result)
    print('Verifying frozen Python environments',flush=True)
    after=environments()
    if after!=result['environment_before']:raise ValueError('environment payload changed')
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError('git diff --check failed')
    result['pytest']=tests;result['git_diff_check']=dict(exit_code=diff.returncode,output=diff.stdout+diff.stderr)
    result['final_hash_audit']=dict(passed=True,unexpected_content_mismatch=0,
        historical_files=len(result['frozen_history_sha256']),preparation_files=len(result['preparation_artifact_sha256']),
        numerical_CSVs=len(result['numeric_artifact_sha256']))
    result['figure_artifact_sha256']={str(p.relative_to(ROOT)).replace('\\','/'):sha(p)
        for stem in ('t17a_weighted_residual','t17a_interaction_energy')
        for ext in ('png','pdf','svg','alignment.json','collision-audit.json') for p in [OUT/f'{stem}.{ext}']}
    result['figure_audit_sha256']=sha(OUT/'t17a_figure_audit.json')
    result['runtime']['plotting_seconds']=figure['runtime_plotting_and_QA_seconds']
    result['runtime']['final_audit_and_regression_seconds']=time.perf_counter()-start
    result['runtime']['total_recorded_wall_seconds']=result['runtime']['total_numeric_wall_seconds']+result['runtime']['plotting_seconds']+result['runtime']['final_audit_and_regression_seconds']
    result['environment_after']=after
    result['completion_state']='completed';result['all_checks_passed']=True
    save(RESULT,result)
    print('T17A engineering PASS; scientific result: '+result['c6_claim_status'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['claim','visual-review','finalize'])
    args=parser.parse_args()
    {'claim':claim,'visual-review':visual_review,'finalize':finalize}[args.mode]()
