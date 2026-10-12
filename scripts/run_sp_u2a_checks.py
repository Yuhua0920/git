"""Execute this task's validator and output-guarded, synthetic scoped pytest."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

TESTS = ['tests/test_sp_u2a_theory.py',
         'tests/test_pilot_cv.py::test_fold_geometry_partition_and_no_heldout_observed_overlap',
         'tests/test_pilot_cv.py::test_cross_pilot_K_rank_one_analytic_reference_without_inverse',
         'tests/test_pilot_cv.py::test_real_LS_independent_stacked_reference_and_brute_force',
         'tests/test_pilot_cv.py::test_no_environment_or_oracle_features_in_fit_interfaces',
         'tests/test_pilot_cv.py::test_source_means_and_frozen_full_module_deployment',
         'tests/test_pilot_cv.py::test_causal_windows_and_future_frame_perturbation',
         'tests/test_pilot_cv.py::test_unconstrained_extreme_coefficients_are_not_clipped',
         'tests/test_pilot_cv.py::test_primary_disallows_oracle_target_inputs_and_always_two_modes']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', default='')
    args = parser.parse_args()
    if args.attempt and (not args.attempt.startswith('_attempt') or not args.attempt[8:].isdigit()):
        parser.error('attempt must be empty or _attemptNN')
    records = []
    argv = [sys.executable, 'scripts/validate_sp_u2a.py']
    temp = ROOT / f'outputs/sp_u2a_pytest_tmp{args.attempt}'
    assert temp.resolve().parent == (ROOT / 'outputs').resolve() and not temp.exists(), 'use a fresh scoped pytest temp path'
    guarded = ('import sys; from scripts.validate_sp_u2a import output_guard; '
               'sys.addaudithook(output_guard); import pytest; '
               f'sys.exit(pytest.main({["-q", "-p", "no:cacheprovider", "--basetemp", str(temp), *TESTS]!r}))')
    commands = [('validator', argv), ('scoped_pytest', [sys.executable, '-c', guarded]),
                ('tracked_diff_check', ['git', 'diff', '--check'])]
    for label, command in commands:
        start = time.perf_counter()
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
        log = OUT / f'sp_u2a_{label}_execution{args.attempt}.txt'
        if log.exists():
            raise RuntimeError('execution log already exists; preserve it and use a new attempt filename')
        text = run.stdout + '\nSTDERR:\n' + run.stderr
        log.write_text(text, encoding='utf-8')
        records.append(dict(label=label, argv=command, actually_executed=True, exit_code=run.returncode,
                            elapsed_seconds=time.perf_counter() - start,
                            log=str(log.relative_to(ROOT)).replace('\\', '/'),
                            sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
        print(label, 'exit=', run.returncode, '\n', run.stdout, run.stderr)
        if run.returncode:
            break
    report = dict(task='SP-U2A', executions=records, full_pytest_run=False,
                  scope='new algebra tests plus explicitly selected synthetic-only historical nodes',
                  old_scientific_outputs_blocked_by_audit_hook=True,
                  all_exit_zero=all(r['exit_code'] == 0 for r in records))
    record_path = OUT / f'sp_u2a_execution_results{args.attempt}.json'
    if record_path.exists():
        raise RuntimeError('execution record already exists; use a new attempt filename')
    record_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if not report['all_exit_zero']:
        sys.exit(1)


if __name__ == '__main__':
    main()
