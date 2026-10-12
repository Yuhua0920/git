"""T16E final saved-artifact audit; no fitting, performance replay or generator."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_pilot_cv import OUT,RESULT,accepted,frozen_operators
from scripts.run_cdl_mismatch import load,save,sha,environments


def visual_review():
    r=load(OUT/'t16e_pilot_cv_figure_audit.json')
    if r['source_returncode'] or not all(r['CSV_only_rebuild'].values()) or any(v['text_returncode'] or v['collision_returncode'] for v in r['rendered_checks'].values()):
        raise ValueError('automatic figure QA incomplete')
    r['visual_review']='Both final PNGs inspected: same-frame NMSE and oracle efficiency; method/style labels, all SNRs/conditions, axes, legends and margins clear.'
    r['reviewed_warnings']=['PNG300dpi/no TIFF; editable vectors supplied','Pooled ratios, trajectory std saved; no CI']
    r['all_checks_passed']=True;save(OUT/'t16e_pilot_cv_figure_audit.json',r)
    print('Actual final T16E figure review recorded')


def finalize():
    start=time.perf_counter();r=load(RESULT);accepted();frozen_operators()
    hashes={**r['source_sha256'],**r['frozen_history_sha256'],**r['numeric_artifact_sha256']}
    for p,h in hashes.items():
        if sha(ROOT/p)!=h:raise ValueError(f'file drift: {p}')
    for p,h in r['preparation_artifact_sha256'].items():
        if sha(OUT/p)!=h:raise ValueError('original preparation artifact changed')
    figure=load(OUT/'t16e_pilot_cv_figure_audit.json')
    if not figure['all_checks_passed']:raise ValueError('actual figure review incomplete')
    for label,args in (('provenance',['tests/test_pilot_cv.py','-k','snapshot or frozen or label or actual_results or replay']),('full',[])):
        command=[str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest','-q',*args]
        test=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace')
        output=test.stdout+test.stderr;(OUT/f't16e_pilot_cv_pytest_{label}.txt').write_text(output,encoding='utf8')
        r[f'pytest_{label}']=dict(command=' '.join(command),returncode=test.returncode,output=output);save(RESULT,r)
        if test.returncode or 'skipped' in output:raise ValueError('final tests failed or skipped; numerical artifacts retained')
        print(output.strip().splitlines()[-1],flush=True)
    for p,h in hashes.items():
        if sha(ROOT/p)!=h:raise ValueError('file changed during tests')
    env=environments()
    if env!=r['environment_before']:raise ValueError('frozen environment changed')
    r['environment_after']=env;r['figure_audit']=figure
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError('git diff check failed')
    r['git_diff_check']=dict(returncode=diff.returncode,output=diff.stdout+diff.stderr)
    r['runtime']['final_audit_seconds']=time.perf_counter()-start
    r['all_checks_passed']=True;r['completion_state']='completed';save(RESULT,r)
    print('T16E engineering audit PASS; science gate retained; formal test sealed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--record-visual-review',action='store_true')
    if p.parse_args().record_visual_review:visual_review()
    else:finalize()
