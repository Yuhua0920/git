"""T16F saved-result/figure/environment audit; no performance rerun."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_statistics_selection import OUT,RESULT,accepted,frozen_candidates
from scripts.run_cdl_mismatch import load,save,sha,environments


def visual_review():
    r=load(OUT/'t16f_figure_audit.json')
    if r['source_returncode'] or not all(r['CSV_only_rebuild'].values()) or any(
        v['text_returncode'] or v['collision_returncode'] for v in r['rendered_checks'].values()):
        raise ValueError('automatic figure QA incomplete')
    r['visual_review']='Both final PNGs inspected: equal-frame primary NMSE and B120 convex/affine weight geometry; all panels, labels, IQRs, margins and legends clear.'
    r['reviewed_warnings']=['PNG300dpi with PDF/SVG; no TIFF required','Correlated windows: IQR is descriptive, not CI']
    r['all_checks_passed']=True;save(OUT/'t16f_figure_audit.json',r)
    print('T16F actual final visual review recorded')


def finalize():
    start=time.perf_counter();r=load(RESULT);accepted();frozen_candidates()
    hashes={**r['source_sha256'],**r['frozen_history_sha256'],**r['numeric_artifact_sha256'],
        **r['preflight_failure_recovery']['artifact_sha256']}
    for p,h in hashes.items():
        if sha(ROOT/p)!=h:raise ValueError(f'file drift:{p}')
    figure=load(OUT/'t16f_figure_audit.json')
    if not figure['all_checks_passed']:raise ValueError('actual visual review pending')
    for label,args in (('provenance',['tests/test_statistics_selection.py','-k','snapshot or frozen or oracle or actual or replay']),('full',[])):
        command=[str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest','-q',*args]
        test=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace')
        output=test.stdout+test.stderr;(OUT/f't16f_pytest_{label}.txt').write_text(output,encoding='utf8')
        r[f'pytest_{label}']=dict(command=' '.join(command),returncode=test.returncode,output=output);save(RESULT,r)
        if test.returncode or 'skipped' in output:raise ValueError('final tests failed/skipped; numeric artifacts preserved')
        print(output.strip().splitlines()[-1],flush=True)
    for p,h in hashes.items():
        if sha(ROOT/p)!=h:raise ValueError('file changed during tests')
    environment=environments()
    if environment!=r['environment_before']:raise ValueError('frozen environment drift')
    r['environment_after']=environment;r['figure_audit']=figure
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError('git diff check failed')
    r['git_diff_check']=dict(returncode=diff.returncode,output=diff.stdout+diff.stderr)
    r['runtime']['final_audit_seconds']=time.perf_counter()-start
    r['all_checks_passed']=True;r['completion_state']='completed';save(RESULT,r)
    print('T16F final engineering audit PASS; frozen scientific gate retained; formal test sealed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--record-visual-review',action='store_true')
    if p.parse_args().record_visual_review:visual_review()
    else:finalize()
