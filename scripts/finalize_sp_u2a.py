"""Generate delivery inventory from actual logs; preserve frozen history."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import re

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def whitespace_issues(paths, root=ROOT):
    authored, source_snapshot = [], []
    for name in paths:
        if Path(name).suffix in ('.py', '.md', '.json', '.txt'):
            for line, text in enumerate((root / name).read_text(encoding='utf-8').splitlines(), 1):
                if text.rstrip(' \t') != text:
                    record = dict(path=name, line=line)
                    destination = source_snapshot if name.startswith('outputs/') and Path(name).suffix == '.txt' else authored
                    destination.append(record)
    return authored, source_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execution', default='outputs/sp_u2a_execution_results.json')
    args = parser.parse_args()
    before = json.loads((ROOT / 'outputs/sp_u2a_before_manifest.json').read_text(encoding='utf-8'))
    protected_changes = [p for p, digest in before['protected_sha256'].items() if sha(ROOT / p) != digest]
    assert not protected_changes, protected_changes
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert head == before['head']
    assert not subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).strip()
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).strip()
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
    paths = sorted(p for p in untracked if ('sp_u2a_' in p or 'sp_u2a.' in p))
    scratch = [p for p in paths if p.startswith('outputs/sp_u2a_pytest_tmp')]
    paths = [p for p in paths if p not in scratch]
    own_manifest = 'outputs/sp_u2a_delivery_manifest.json'
    paths = [p for p in paths if p != own_manifest]
    warnings = []
    execution = json.loads((ROOT / args.execution).read_text(encoding='utf-8'))
    assert execution['all_exit_zero']
    for record in execution['executions']:
        log = ROOT / record['log']
        assert sha(log) == record['sha256']
        for line in log.read_text(encoding='utf-8').splitlines():
            if 'Failed to find real location' in line:
                warnings.append(dict(log=record['log'], warning=line))
    pytest_record = next(r for r in execution['executions'] if r['label'] == 'scoped_pytest')
    pytest_log = (ROOT / pytest_record['log']).read_text(encoding='utf-8')
    pytest_summary = re.search(r'\d+ passed(?:, \d+ skipped)?', pytest_log)
    assert pytest_summary, 'actual pytest summary missing'
    # git diff does not inspect untracked additions; check new text bytes separately.
    whitespace, raw_source_whitespace = whitespace_issues(paths)
    assert not whitespace, whitespace
    files = [dict(path=p, bytes=(ROOT / p).stat().st_size, sha256=sha(ROOT / p)) for p in paths]
    result = dict(task='SP-U2A', head=head, protected_files=len(before['protected_sha256']),
                  original_sp_u1_count=len(before['original_sp_u1_files']), protected_changes=protected_changes,
                  tracked_diff_empty=True, original_sp_u1_unchanged=True, manuscript_unchanged=True,
                  new_files_excluding_manifest=len(files), files=files, untracked_whitespace_issues=whitespace,
                  raw_text_whitespace_preserved=raw_source_whitespace,
                  audit_failure_preserved='outputs/sp_u2a_finalizer_failure01',
                  pytest_scratch_not_delivery_artifacts=scratch,
                  runtime_warnings=warnings, scoped_pytest=pytest_summary.group(), full_pytest_run=False,
                  execution_record=args.execution,
                  formal_outcomes_used=False, baseline_code_executed=False, new_channel_experiments_run=False,
                  equation_reference='docs/sp_u2a_equation_inventory.md',
                  raw_derivation='docs/sp_u2a_coordinate_theory.md',
                  first_layer_connection_self_assessment='PASS', novel_general_theorem='NOT_ESTABLISHED',
                  stop_rule='advisor may declare theory branch insufficient; no autonomous U3',
                  next_task_authorized=False, committed=False)
    (ROOT / own_manifest).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(new_files_excluding_manifest=len(files), protected_files=result['protected_files'],
                          protected_changes=protected_changes, original_sp_u1_unchanged=True,
                          tracked_diff_empty=True, runtime_warning_count=len(warnings)), indent=2))


if __name__ == '__main__':
    main()
