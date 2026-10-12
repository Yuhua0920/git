"""Assemble R2 preparation evidence; never open scientific performance arrays."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_sp_u12b5c_r2 import OUT, NEW, sha, read, dump, need


def main():
    package = read(OUT / 'package_report.json')
    science = read(OUT / 'science_preservation.json')
    preservation = read(OUT / 'preservation_after.json')
    windows = read(OUT / 'windows_qa.json')
    before = read(OUT / 'preservation_before.json')
    need(all(x['PASS'] for x in (science, preservation, windows)), 'PREPARATION_EVIDENCE_NOT_PASS')
    need(len(windows['tests']) == 17 and all(x['PASS'] for x in windows['tests']), 'WINDOWS_QA_INCOMPLETE')
    candidates = list((ROOT / 'lab_runs/b5c_paired_r2/synthetic').glob('pytest_*/pytest.xml'))
    need(bool(candidates), 'ACTUAL_PYTEST_REPORT_MISSING')
    xml = max(candidates, key=lambda q: q.stat().st_mtime_ns)
    suites = list(ET.parse(xml).getroot().iter('testsuite'))
    totals = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    need(totals == {'tests': 54, 'failures': 0, 'errors': 0, 'skipped': 0}, 'ACTUAL_PYTEST_NOT_54_PASS')
    need(read(xml.parent / 'status.json')['PASS'], 'ACTUAL_PYTEST_STATUS_NOT_PASS')

    zpath = Path(package['path'])
    need(sha(zpath) == package['sha256'] == zpath.with_suffix('.zip.sha256').read_text().strip(), 'FINAL_ZIP_HASH_DRIFT')
    with (OUT / 'SP_U12B5C_R2_File_Manifest.csv').open(encoding='utf8', newline='') as f:
        records = list(csv.DictReader(f))
    need(len(records) == package['files'] == 14, 'ZIP_FILE_COUNT_DRIFT')
    with zipfile.ZipFile(zpath) as z:
        need(z.testzip() is None and set(z.namelist()) == {v['relative_path'] for v in records}, 'ZIP_INVENTORY_DRIFT')
        for v in records:
            payload = z.read(v['relative_path'])
            need(len(payload) == int(v['bytes']) and hashlib.sha256(payload).hexdigest() == v['sha256'], 'ZIP_PAYLOAD_DRIFT')
            need(sha(ROOT / v['relative_path']) == v['sha256'], 'DEPLOYMENT_FILE_DRIFT')
    need(sha(OUT / 'Install-R2.ps1') == sha(NEW / 'Install-R2.ps1'), 'EXTERNAL_INSTALLER_DRIFT')

    dependency = read(NEW / 'package_manifest.json')['files']
    for name, v in dependency.items():
        need(sha(ROOT / name) == v['sha256'], 'FROZEN_DEPENDENCY_DRIFT:' + name)
    git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    tracked = hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT)).hexdigest()
    staged = hashlib.sha256(subprocess.check_output(['git', 'diff', '--cached', '--binary'], cwd=ROOT)).hexdigest()
    need((git_head, tracked, staged) == (before['git_HEAD'], before['tracked_diff'], before['staged_diff']), 'GIT_TRACKED_OR_COMMIT_DRIFT')
    status = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT).decode('utf8', errors='replace')
    with (OUT / 'git_status.txt').open('x', encoding='utf8') as f:
        f.write(status)

    personal = ROOT / 'lab_runs/b5c_paired_r2'
    need(not (personal / 'formal').exists() and not (personal / 'FORMAL_START_CONSUMED.json').exists(), 'PERSONAL_FORMAL_BOUNDARY_VIOLATED')
    result = {
        'verdict': 'R2_EXECUTION_PACKAGE_PREPARED_LAB_GATE_PENDING',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'package': package,
        'pytest': {**totals, 'PASS': True, 'xml': str(xml), 'log': str(xml.parent / 'pytest.log')},
        'windows_technical_checks': {'PASS': True, 'checks': len(windows['tests']), 'QA_root': windows['QA_root'], 'scope': windows['scope']},
        'science': science,
        'preservation': preservation,
        'current_authenticated_dependency_files': len(dependency),
        'tracked_and_staged_diff_unchanged': True,
        'HEAD_unchanged': True,
        'git_HEAD': git_head,
        'real_formal_evaluation_executed': False,
        'real_model_or_data_migrated': False,
        'real_LAB_R1_PASS_verified_on_personal': False,
        'LAB_R1_IdentityAudit_required_before_Preflight_and_Run': True,
        'mentor_review_required_before_Run': True,
        'approval_or_signature_record_created': False,
        'original_attempt_resumed': False,
        'failed_attempt_performance_used': False,
        'next_action': 'STOP; mentor reviews package, then LAB executes actual R1 IdentityAudit and prerequisites before one R2 launch',
    }
    dump(OUT / 'final_report.json', result)
    doc = ROOT / 'docs/sp_u12b5c_r2_execution_preparation.md'
    need(not doc.exists(), 'DELIVERY_REPORT_ALREADY_EXISTS')
    text = f'''# SP-U12B5C-R2 execution preparation

Verdict: R2_EXECUTION_PACKAGE_PREPARED_LAB_GATE_PENDING.

This report records personal-computer technical preparation only. Actual LAB R1
IdentityAudit remains a prerequisite; this task has not certified a real LAB PASS,
accessed the LAB model, or run a formal evaluation.

## Frozen science and independent execution

The new science.py is byte-identical to certified R1. Relative to original B5C,
only the complete six-field channel seed representation is corrected. Original
paired_stats.py, report.py and publication_plots.py are byte-identical. The shared
contract and data manifest hashes remain unchanged.

Output: lab_runs/b5c_paired_r2/formal. Old lab_runs/b5c_paired is never resumed.
Old STOP, consumption marker and logs must be present and match actual R1's
opaque preservation inventory. Canonical LAB R1 status/report and both original
preservation files are checked and their hashes bound to the new execution.
Existing authenticated model and 30008 input files are verified before launch.
Preflight and the exclusively locked startup both reject Stop/consumption/concurrent
tasks. Failed execution does not retry or clear a consumption marker.

Five conditions, seven SNRs, 1000 trajectories/cell, frames 4--9 and 210000
positions/method remain frozen. Paired trajectory bootstrap uses 100000 replicates
and one 35-cell two-sided Bonferroni family. Difference is MFA NMSE dB minus LR4
NMSE dB, so positive favors LR4. No failed-attempt outcomes are used.

## Actual technical evidence

- Pytest: {totals['tests']} passed, 0 failed/errors/skips; {xml.relative_to(ROOT).as_posix()}.
- Actual relocated Windows PowerShell checks: 17/17 PASS. These use synthetic old
  failure fixtures; no real R1 PASS/model/data is transferred into relocation QA.
- Original local preservation: {preservation['protected_files']} files,
  {preservation['bytes']} bytes, zero changed/additions. B2/B3 are opaque hashes;
  no performance arrays are decoded.
- Final dependency authentication: {len(dependency)} files, all PASS.
- HEAD, tracked and staged diffs unchanged; no commit or manuscript/source change.

## Transfer four files

From outputs/sp_u12b5c_r2 to D:\\Research\\Incoming\\B5C_R2:

1. SP_U12B5C_R2_Execution.zip
2. SP_U12B5C_R2_Execution.zip.sha256
3. SP_U12B5C_R2_File_Manifest.csv
4. Install-R2.ps1

ZIP bytes: {package['bytes']}; members: {package['files']}.
SHA256: {package['sha256']}.
No B2 data or MFA parameter file is bundled. Existing LAB CPU Python environment
is reused: Python 3.13.15, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1,
threadpoolctl 3.7.0, pytest 9.1.1. No GPU migration.

## PowerShell commands

Run each step separately and stop on any error.

```powershell
$root = 'D:\\Research\\Project_MFA_Lab'
$incoming = 'D:\\Research\\Incoming\\B5C_R2'
powershell -NoProfile -ExecutionPolicy Bypass -File "$incoming\\Install-R2.ps1" -ProjectRoot $root -ZipPath "$incoming\\SP_U12B5C_R2_Execution.zip" -ExpectedSHA256 '{package['sha256']}'
Set-Location $root
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r2_v1\\R2.ps1 -Action Verify
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r2_v1\\R2.ps1 -Action ToyTests
```

Only after actual LAB R1 COMPLETE/PASS:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r2_v1\\R2.ps1 -Action Preflight
```

Only after mentor review and all prerequisites pass, launch once:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r2_v1\\R2.ps1 -Action Run -ConfirmFormalEvaluation -Background
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r2_v1\\R2.ps1 -Action Status
```

Stop if needed, then export only after termination/completion:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r2_v1\\R2.ps1 -Action Stop
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r2_v1\\R2.ps1 -Action Status
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r2_v1\\R2.ps1 -Action Export
```

Background PID is launch information, not scientific success. Status must reach
COMPLETE; failed/STOP records must remain. Export prints exact new ZIP/SHA paths.
Default R1 receipt selection uses the most recent canonical COMPLETE/PASS job;
its exact relative path and four SHA256 values are recorded. Optional -R1AuditJob
may name the actual canonical relative identity directory. No fabricated mentor
signature or prior authorization file is created.

STOP after preparation. LAB execution and actual scientific outcomes await
operator action after mentor review. Remaining risks are actual LAB gate pass,
model/data/capacity checks, and avoiding concurrent manual scientific starts.
'''
    with doc.open('x', encoding='utf8') as f:
        f.write(text)
    print(json.dumps({k: result[k] for k in ('verdict', 'package', 'pytest', 'windows_technical_checks', 'current_authenticated_dependency_files', 'real_formal_evaluation_executed')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
