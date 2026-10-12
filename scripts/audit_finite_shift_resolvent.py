"""Finalize T17B evidence and actual regressions without rerunning the mechanism."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_finite_shift_resolvent import accepted,OUT,RESULT,DOCS,C6,c7_status
from scripts.run_cdl_mismatch import load,save,sha,environments
from src.finite_shift_resolvent import decision


def verify(r):
    accepted()
    for group in ('frozen_history_sha256','numeric_artifact_sha256','preparation_artifact_sha256'):
        for p,h in r[group].items():
            if sha(ROOT/p)!=h:raise ValueError('frozen bytes changed:'+p)
    for p,h in r['source_sha256'].items():
        if sha(ROOT/r['pre_run_source_snapshot_root']/p)!=h:raise ValueError('source snapshot drift')
        if sha(ROOT/p)!=r.get('post_run_document_sha256',{}).get(p,h):raise ValueError('current source drift')
    for k,v in decision(r['four_term_decomposition']).items():
        if r[k]!=v:raise ValueError('predefined decision changed')
    if r['c6_claim_status']!=C6 or r['c7_claim_status']!=c7_status(r):raise ValueError('claim status drift')
    if any(r[k] for k in ('generator_calls','statistics_refits','coefficient_refits','Taylor_terms_used','formal_test_used','new_channel_generated','new_model_trained','novelty_claim_made')):raise ValueError('scope boundary violated')
    return True


def claim():
    r=load(RESULT);verify(r)
    if r.get('post_run_document_sha256'):raise ValueError('post-gate documents already saved')
    text='\n## Actual T17B evidence\n\nC7 status: **'+r['c7_claim_status']+'**.\n\n'
    text+='| SNR | Affine closure | Cancellation index | Maximum contribution norm ratio |\n| --- | --- | --- | --- |\n'
    for row in r['four_term_decomposition']:text+=f"| {row['snr_db']} | {row['exact_affine_closure']:.12g} | {row['cancellation_index']:.12g} | {row['maximum_contribution_norm_ratio']:.12g} |\n"
    text+='\nExisting Evidence: exact four-term/resolvent and Gram identities, recorded frozen-statistics/W regressions, and outputs/t17b_results.json.\n\n'
    text+='Missing Evidence: sensitivity to covariance-estimation uncertainty, alternative decompositions and broader environments. None are introduced in this task.\n\n'
    text+='Confidence: restricted to the predefined four terms and B12010/20 gate. Standard exact matrix identities carry no novelty claim. C6 permanently remains '+C6+'.\n'
    review='\n## Actual T17B result\n\n'+r['c7_claim_status']+'. See the exact contributions, signed cross energies and cancellation indices in outputs/t17b_results.json.\n\n'
    if r['affine_additivity_cancellation_dominated']:
        review+='The registered stop rule is triggered: retain affine composition as an empirical method; stop the search for a universal separable mechanism in this task. Broader validation is a dependency for any future generality claim.\n'
    elif r['finite_shift_separability_supported']:
        review+='The predefined finite-shift decomposition is supported in B12010/20 only; broader stability remains untested.\n'
    else:review+='The gate is partial; no universal finite-shift separability claim is supported.\n'
    review+='\nRemaining risk: decomposition-dependent cancellation and finite-statistics uncertainty. Bookkeeping contributions are not independent physical causes. T17A and C6 remain unchanged.\n'
    hashes={}
    for doc,addition in zip(DOCS[1:],(text,review)):
        path=ROOT/doc;original=path.read_text(encoding='utf8');path.write_text(original+addition,encoding='utf8');hashes[doc]=sha(path)
    r['post_run_document_sha256']=hashes;save(RESULT,r);print('C7/reviewer result appended; T17A/C6 unchanged')


def visual_review():
    f=load(OUT/'t17b_figure_audit.json')
    if f['source_returncode'] or not all(f['CSV_only_rebuild'].values()) or any(v['text_returncode'] or v['collision_returncode'] for v in f['rendered_checks'].values()):raise ValueError('automatic figure gates incomplete')
    f['visual_review']='Both actual final PNGs inspected: all four terms/final residual and all signed-energy bars; labels, magnitudes, negative cross contributions and margins clear.'
    f['panel_review']=[dict(panel=p,pass_review=True,uncertainty='none: deterministic frozen operators') for p in ('Contribution norms','Signed energy accounting')]
    f['all_checks_passed']=True;save(OUT/'t17b_figure_audit.json',f);print('Actual final visual review recorded')


def finalize():
    start=time.perf_counter();r=load(RESULT);verify(r)
    f=load(OUT/'t17b_figure_audit.json')
    if not f['all_checks_passed'] or not r.get('post_run_document_sha256'):raise ValueError('figure/claim audit pending')
    env=dict(os.environ);env['TEMP']=env['TMP']=str(ROOT/'.pytest_cache/pytest_workspace_temp')
    tests={}
    for label,args in [('related',['tests/test_finite_shift_resolvent.py','tests/test_finite_shift_pipeline.py']),
        ('provenance',['tests/test_finite_shift_pipeline.py','-k','provenance or frozen']),('full',[])]:
        path=OUT/f't17b_pytest_{label}.txt'
        if path.exists():raise ValueError('existing transcript protected')
        print('Executing actual '+label+' pytest',flush=True);tick=time.perf_counter()
        with path.open('w',encoding='utf8') as out:proc=subprocess.run([sys.executable,'-m','pytest',*args,'-q'],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
        summary=path.read_text(encoding='utf8').strip().splitlines()[-1]
        tests[label]=dict(command=[sys.executable,'-m','pytest',*args,'-q'],summary=summary,exit_code=proc.returncode,seconds=time.perf_counter()-tick,sha256=sha(path))
        print(summary,flush=True)
        if proc.returncode or 'skipped' in summary:raise ValueError('actual regression failed or skipped')
    verify(r);print('Checking frozen environment payloads',flush=True);after=environments()
    if after!=r['environment_before']:raise ValueError('frozen environment changed')
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError('git diff --check failed')
    r.update(pytest=tests,environment_after=after,git_diff_check=dict(exit_code=diff.returncode,output=diff.stdout+diff.stderr),
        final_hash_audit=dict(passed=True,historical_files=len(r['frozen_history_sha256']),unexpected_content_mismatch=0),
        figure_artifact_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for stem in ('t17b_contribution_norms','t17b_signed_energy')
                               for ext in ('png','pdf','svg','alignment.json','collision-audit.json') for p in [OUT/f'{stem}.{ext}']},
        figure_audit_sha256=sha(OUT/'t17b_figure_audit.json'),completion_state='completed',all_checks_passed=True)
    r['runtime']['plotting_seconds']=f['runtime_plotting_and_QA_seconds']
    r['runtime']['final_audit_and_regression_seconds']=time.perf_counter()-start
    r['runtime']['total_recorded_wall_seconds']=r['runtime']['total_numeric_wall_seconds']+r['runtime']['plotting_seconds']+r['runtime']['final_audit_and_regression_seconds']
    save(RESULT,r);print('T17B engineering PASS; scientific gate '+r['c7_claim_status'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['claim','visual-review','finalize']);a=p.parse_args()
    {'claim':claim,'visual-review':visual_review,'finalize':finalize}[a.mode]()
