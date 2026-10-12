"""Record the T17A preflight only; never run the scientific mechanism."""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_wiener_mechanism import (OUT, SOURCES, accepted,
    load_frozen_statistics, prepare_operators)
from scripts.run_cdl_mismatch import history, save, sha


def check():
    commit, accepted_result = accepted()
    before = history()
    stats, variances, stat_proof = load_frozen_statistics()
    _, operator_proof, sentinel = prepare_operators(stats, variances)
    calls = set()
    for path in ('src/wiener_perturbation.py', 'scripts/run_wiener_mechanism.py'):
        tree = ast.parse((ROOT/path).read_text(encoding='utf8'))
        calls.update(getattr(n.func, 'id', getattr(n.func, 'attr', ''))
                     for n in ast.walk(tree) if isinstance(n, ast.Call))
    # dict.update() serializes metrics; it is not a covariance-fitting call.
    forbidden = calls & {'generate', 'communication', 'fit', 'StreamingStatistics', 'inv', 'pinv'}
    if forbidden:
        raise ValueError('forbidden scientific path: '+str(forbidden))
    from scripts.finalize_nearest_validation_provenance import verify_current_registry
    verify_current_registry(ROOT)
    baseline = OUT/'t17a_startup_accepted_pytest.txt'
    math = OUT/'t17a_postcommit_math_pytest.txt'
    text = baseline.read_text(encoding='utf8')
    # Only pytest's terminal summary, never intermediate dots or a prior log.
    summary = next((line for line in reversed(text.splitlines())
                    if re.search(r'\d+ (?:passed|failed|error)', line)
                    and ' in ' in line), '')
    counts = {key: int(match.group(1)) if (match := re.search(
        rf'(\d+) {key}\b', summary)) else 0
        for key in ('passed', 'failed', 'errors', 'error', 'skipped', 'warnings', 'warning')}
    baseline_passed = counts['passed'] >= 1610 and not any(
        counts[key] for key in ('failed', 'errors', 'error', 'skipped'))
    if before != history():
        raise ValueError('historical bytes drifted during preflight')
    diff = subprocess.run(['git', 'diff', '--check'], cwd=ROOT,
                          capture_output=True, text=True)
    result = dict(task='T17A', stage='post-commit execution preflight',
        accepted_T16G_commit=commit, T16G_clean_immediately_after_commit=True,
        accepted_T16G_frozen_file_count=941, accepted_T16G_hashes_unchanged=True,
        startup_pytest_summary=summary, startup_pytest_counts=counts,
        startup_log_sha256=sha(baseline), startup_pytest_passed=baseline_passed,
        math_pytest_summary=math.read_text(encoding='utf8').strip().splitlines()[-1],
        math_pytest_log_sha256=sha(math), statistics_provenance=stat_proof,
        operator_provenance=operator_proof,
        historical_operator_quadratic_sentinels=sentinel,
        historical_raw_target_W_hash_available=False,
        frozen_history_sha256=before,
        environment_audit='attempt01 completed environments() before its AST dict.update false positive; not repeated after failed baseline',
        source_sha256={p:sha(ROOT/p) for p in (*SOURCES, 'scripts/check_wiener_preflight.py')},
        generator_calls=0, new_channel_generated=False, new_statistics_fit=False,
        mechanism_run_performed=False, scientific_gate_evaluated=False,
        formal_test_sealed=True, formal_test_used=False,
        git_diff_check_exit_code=diff.returncode,
        git_diff_check_output=diff.stdout+diff.stderr,
        all_checks_passed=baseline_passed and diff.returncode == 0,
        completion_state='preflight passed' if baseline_passed else 'STOP: full pytest preflight failed',
        warnings=['Earlier failed permission attempts are retained.',
            'Audit attempt01 misclassified dict.update as statistics fitting; guard corrected, no scientific run performed.',
            'Math tests have a pytest cache PermissionError warning.',
            'Target standalone raw-W historical hashes are unavailable; replay and exact quadratic sentinels are separate evidence.'])
    save(OUT/'t17a_execution_preflight.json', result)
    print(json.dumps({key:result[key] for key in ('accepted_T16G_commit',
        'startup_pytest_summary', 'all_checks_passed', 'completion_state')}, indent=2))
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    sys.exit(check())
