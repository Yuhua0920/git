"""Record the minimal post-run auditor fix; retain original execution hashes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_operator_space_audit import RESULT,OUT
from scripts.run_cdl_mismatch import load,save,sha
from scripts.operator_space_audit_provenance import verify_sources,HELPER
from scripts.audit_operator_space_audit import finalize


def main():
    r=load(RESULT)
    if r['all_checks_passed']:raise ValueError('completed result protected')
    r['post_run_auditor_revision']=verify_sources(r)
    r['post_run_source_sha256']={path:sha(ROOT/path) for path in (HELPER,'scripts/finalize_operator_space_audit.py')}
    r['engineering_warnings']=[
        'Preparation attempt01:5 failed,24 passed,1 skipped. Synthetic phase-shift columns were rank deficient and a broad AST rule mistook dictionary update for statistics fitting. Fixed before scientific execution; logs retained.',
        'Figure calibration attempt01: source regex mistook scientific_figures=3 for resolution. Actual PNG was300dpi; rendered checks/rebuilds passed. Changed metadata key before scientific execution; failed fixture archive retained.',
        'Initial post-run verifier reassociated residual subtraction; maximum coefficient difference5.551115123125783e-17. Corrected only verifier grouping; reverse patch exactly reproduces original archived auditor. No numerical CSV/NPZ or execution hash rewritten; no scientific rerun.',
        'Independent real-stack/Cholesky LS contractions are not bitwise identical to historical vdot/numpy solve; numerical errors are explicitly recorded, not claimed exact.',
        'Some rank4 capture losses are negative: approximated P/M directions define a different two-dimensional span, not a nested subset of the original dense P/M span. All signed values retained.',
        'Virtual environment emits stale base-Python-location warning for D:/Users/yuhua/AppData/Local/Programs/Python/Python313/python.exe; no environment payload changes.',
        'Figure source QA warns PNG300dpi is below its600dpi default and has no submission TIFF; requested PNG/PDF/SVG research exports pass.',
        'Git staged historical SVG/log whitespace and LF/CRLF warnings retained when committing T19A; historical content preserved. Current git diff --check is tested separately.',
        'Skill lesson helper recommends pruning its long log; no pruning or skill-rule promotion performed.'
    ]
    r['preparation_artifact_sha256']={f'outputs/{name}':sha(OUT/name) for name in (
        't19b_preflight.json','t19b_startup_pytest.txt','t19b_preparation_pytest_attempt01.txt',
        't19b_preparation_pytest_attempt02.txt','t19b_preparation_pytest_final.txt','t19b_fixture_figure_gate.json',
        't19b_claim_run.txt','t19b_claim_run_attempt02.txt','t19b_audit_reassociation_diagnosis.json')}
    save(RESULT,r)
    finalize()


if __name__=='__main__':main()
