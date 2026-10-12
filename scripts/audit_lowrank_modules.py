"""T16G saved-result/figure/environment audit; no compressed performance rerun."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_lowrank_modules import OUT,RESULT,accepted,frozen_factors
from scripts.run_cdl_mismatch import load,save,sha,environments


def visual_review():
    figure=load(OUT/'t16g_figure_audit.json')
    if figure['source_returncode'] or not all(figure['CSV_only_rebuild'].values()) or any(
        v['text_returncode'] or v['collision_returncode'] for v in figure['rendered_checks'].values()):
        raise ValueError('automatic figure QA incomplete')
    figure['visual_review']='Both final PNGs inspected: paired5-frame two-shift rank NMSE and all four primary compression penalties; panels/labels/reference line/legends/margins clear.'
    figure['reviewed_warnings']=['PNG300dpi/PDF/SVG satisfy task contract; TIFF not required','Rank grid is fixed; no points discarded; pooled NMSE has no inferred CI']
    figure['all_checks_passed']=True;save(OUT/'t16g_figure_audit.json',figure)
    print('T16G actual final visual review recorded')


def finalize():
    start=time.perf_counter();result=load(RESULT);accepted();frozen_factors()
    hashes={**result['source_sha256'],**result['frozen_history_sha256'],**result['numeric_artifact_sha256'],
        **result['preparation_artifact_sha256'],**result['preflight_schema_recovery']['artifact_sha256']}
    for p,h in hashes.items():
        if sha(ROOT/p)!=h:raise ValueError(f'artifact drift:{p}')
    figure=load(OUT/'t16g_figure_audit.json')
    if not figure['all_checks_passed']:raise ValueError('actual visual review pending')
    for label,arguments in (('provenance',['tests/test_lowrank_modules_pipeline.py','-k','snapshot or frozen or actual or regression']),('full',[])):
        command=[str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest','-q',*arguments]
        completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace')
        output=completed.stdout+completed.stderr;(OUT/f't16g_pytest_{label}.txt').write_text(output,encoding='utf8')
        result[f'pytest_{label}']=dict(command=' '.join(command),returncode=completed.returncode,output=output);save(RESULT,result)
        if completed.returncode or 'skipped' in output:raise ValueError('final tests failed/skipped; numeric results preserved')
        print(output.strip().splitlines()[-1],flush=True)
    for p,h in hashes.items():
        if sha(ROOT/p)!=h:raise ValueError('artifact changed during tests')
    after=environments()
    if after!=result['environment_before']:raise ValueError('frozen environment changed')
    result['environment_after']=after;result['figure_audit']=figure
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError('git diff check failed')
    result['git_diff_check']=dict(returncode=diff.returncode,output=diff.stdout+diff.stderr)
    result['runtime']['final_audit_seconds']=time.perf_counter()-start
    result['all_checks_passed']=True;result['completion_state']='completed';save(RESULT,result)
    print('T16G final engineering audit PASS; fixed scientific gate retained; formal test sealed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--record-visual-review',action='store_true')
    if parser.parse_args().record_visual_review:visual_review()
    else:finalize()
