"""Technical-only installer repair evidence; no IdentityAudit or scientific run."""
from pathlib import Path
import argparse
import csv
import difflib
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / 'deployment/b5c_paired_r2_v1'
OLD_OUT = ROOT / 'outputs/sp_u12b5c_r2'
OUT = ROOT / 'outputs/sp_u12b5c_r2_installer_fix'
EXPORT = ROOT / 'R1_Technical_Export_1f901d0f1b77.zip'
PACKAGE = OUT / 'SP_U12B5C_R2_Execution_InstallerFix.zip'


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def git_state():
    return {
        'HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'tracked_diff': hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT)).hexdigest(),
        'staged_diff': hashlib.sha256(subprocess.check_output(['git', 'diff', '--cached', '--binary'], cwd=ROOT)).hexdigest(),
    }


def before():
    need(not OUT.exists(), 'FIX_PREPARATION_ALREADY_EXISTS')
    OUT.mkdir()
    archive = OUT / 'original_r2_deployment'
    shutil.copytree(DEPLOY, archive)
    paths = [p for p in DEPLOY.rglob('*') if p.is_file()]
    paths += [p for p in OLD_OUT.glob('*') if p.is_file()]
    paths += [EXPORT]
    dump(OUT / 'before.json', {
        'files': {p.relative_to(ROOT).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths},
        'git': git_state(),
        'historical_baseline': 'outputs/sp_u12b5c_r2/preservation_before.json',
    })
    print('Original R2 deployment archived; no scientific execution')


def evidence():
    binding = read(DEPLOY / 'bindings.json')
    with zipfile.ZipFile(EXPORT) as z:
        need(z.testzip() is None and len(set(z.namelist())) == len(z.namelist()), 'EXPORT_ZIP_INTEGRITY_FAILED')
        records = json.loads(z.read('EXPORT_SHA256.json').decode('utf-8-sig'))
        need(set(z.namelist()) == set(records) | {'EXPORT_SHA256.json'}, 'EXPORT_INVENTORY_MISMATCH')
        for n, h in records.items():
            need(hashlib.sha256(z.read(n)).hexdigest() == h, 'EXPORT_MEMBER_HASH_MISMATCH:' + n)
        prefix = 'technical/identity_4a0a010d221e/'
        def j(n):
            return json.loads(z.read(prefix + n).decode('utf-8-sig'))
        s = j('status.json')
        r = j('identity_report.json')
        a = j('original_preservation_after.json')
        b = j('original_preservation_before.json')
        need(s == {'PASS': True, 'state': 'COMPLETE', 'formal_predictions': 0, 'corrected_proof_passes': 5000}, 'LAB_IDENTITY_STATUS_NOT_PASS')
        for k, v in binding['required_R1_counts'].items():
            need(type(r.get(k)) is int and r[k] == v, 'LAB_COVERAGE_NOT_FROZEN:' + k)
        for k, v in binding['input_identity_digests'].items():
            need(r[k] == v, 'LAB_INPUT_DIGEST_NOT_FROZEN:' + k)
        need(r['cells'] == binding['cells'], 'LAB_35_CELLS_NOT_FROZEN')
        need(r['PASS'] and r['real_scientific_seed_mismatches'] == 0 and r['formal_predictions'] == 0 and r['formal_scientific_arrays_deserialized'] == 0, 'LAB_IDENTITY_NOT_PASS')
        need(r['lab_failed_files_opaque_hashes_preserved'] and r['original_H_noise_pilot_checks_unchanged'], 'LAB_PRESERVATION_NOT_PASS')
        need(a == {'PASS': True, 'files': len(b), 'changed': []}, 'LAB_OLD_FILES_NOT_PRESERVED')
        for n in ('lab_runs/b5c_paired/STOP.json', 'lab_runs/b5c_paired/FORMAL_START_CONSUMED.json'):
            need(n in b, 'LAB_OLD_FAILURE_MARKER_NOT_RECORDED:' + n)
        toy_prefix = 'technical/pytest_48cf352a60e5/'
        toy = json.loads(z.read(toy_prefix + 'status.json').decode('utf-8-sig'))
        suites = list(ET.fromstring(z.read(toy_prefix + 'pytest.xml')).iter('testsuite'))
        totals = {k: sum(int(q.attrib.get(k, 0)) for q in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
        need(toy['PASS'] and toy['state'] == 'COMPLETE' and toy['exit_code'] == 0 and totals == {'tests': 86, 'failures': 0, 'errors': 0, 'skipped': 0}, 'EXPORTED_TOY_TESTS_NOT_PASS')
    dump(OUT / 'lab_export_review.json', {
        'PASS': True, 'export_sha256': sha(EXPORT), 'export_member_hashes_verified': len(records),
        'identity_job': 'lab_runs/b5c_paired_r1/' + prefix.rstrip('/'),
        'identity_state': s['state'], 'corrected_proof_passes': 5000,
        'scientific_seed_mismatches': 0, 'cells': 35, 'scored_input_identities': 210000,
        'old_files_opaque_preserved': len(b), 'old_STOP_and_consumption_recorded': True,
        'exported_toy_tests': totals,
        'identity_audit_rerun': False, 'formal_predictions_on_personal': 0,
        'LAB_current_files_directly_accessed': False,
        'Verify_evidence': 'User reports manual Verify PASS; separate Verify log is absent from this eight-file export',
        'receipt_hashes': {n: records[prefix + n] for n in ('status.json', 'identity_report.json', 'original_preservation_before.json', 'original_preservation_after.json')},
    })
    print('LAB export receipt/hash review PASS; no seed or scientific array replay')


def relocate(parent):
    root = parent / 'Windows Root with spaces'
    root.mkdir(parents=True)
    files = dict(read(ROOT / 'deployment/b5c_paired_r1_v1/package_manifest.json')['files'])
    files['deployment/b5c_paired_r1_v1/package_manifest.json'] = {}
    for n in files:
        q = root / n
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / n, q)
    for n in ('STOP.json', 'FORMAL_START_CONSUMED.json', 'formal/exception.json', 'formal/stderr.log'):
        q = root / 'lab_runs/b5c_paired' / n
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text('SYNTHETIC_PRESERVED_FAILURE_ONLY\n', encoding='utf8')
    r = subprocess.run([sys.executable, '-B', '-m', 'venv', '--without-pip', str(root / '.venv-lab')], capture_output=True, encoding='utf8', errors='replace', timeout=60)
    need(r.returncode == 0, 'SYNTHETIC_DEFAULT_VENV_CREATION_FAILED:' + r.stderr)
    frozen = {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*') if p.is_file() and '.venv-lab' not in p.parts}
    return root, frozen


def install(root, zip_path, installer, mode='omitted', expected=None):
    ps = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass']
    args = ['-File', str(installer), '-ProjectRoot', str(root), '-ZipPath', str(zip_path), '-ExpectedSHA256', expected or sha(zip_path)]
    if mode in ('empty', 'whitespace'):
        # In-process binding preserves empty strings. Native -File cannot carry a
        # missing argument, which is exactly the original child-launch defect.
        quote = lambda v: "'" + str(v).replace("'", "''") + "'"
        value = '' if mode == 'empty' else '   '
        command = '& ' + quote(installer) + ' -ProjectRoot ' + quote(root) + ' -ZipPath ' + quote(zip_path) + ' -ExpectedSHA256 ' + quote(expected or sha(zip_path)) + ' -Python ' + quote(value)
        args = ['-Command', command]
    elif mode in ('explicit', 'missing_explicit'):
        args += ['-Python', str(root / ('.venv-lab/Scripts/python.exe' if mode == 'explicit' else 'missing/Python.exe'))]
    r = subprocess.run(ps + args, cwd=root, capture_output=True, encoding='utf8', errors='replace', timeout=120)
    return r


def unchanged(root, frozen):
    for n, h in frozen.items():
        need(sha(root / n) == h, 'SYNTHETIC_HISTORY_CHANGED:' + n)
    need(not (root / 'lab_runs/b5c_paired_r2/formal').exists(), 'SYNTHETIC_FORMAL_DIRECTORY_CREATED')
    need(not (root / 'lab_runs/b5c_paired_r2/FORMAL_START_CONSUMED.json').exists(), 'SYNTHETIC_FORMAL_CONSUMED')


def reproduce():
    qa = OUT / ('original_reproduction_' + uuid.uuid4().hex[:8])
    root, frozen = relocate(qa)
    installer = qa / 'Install-R2.ps1'
    shutil.copy2(OUT / 'original_r2_deployment/Install-R2.ps1', installer)
    r = install(root, OLD_OUT / 'SP_U12B5C_R2_Execution.zip', installer)
    (qa / 'stdout.log').write_text(r.stdout, encoding='utf8')
    (qa / 'stderr.log').write_text(r.stderr, encoding='utf8')
    need(r.returncode != 0 and 'Python' in r.stderr and 'MissingArgument' in r.stderr, 'ORIGINAL_EMPTY_PYTHON_FAILURE_NOT_REPRODUCED')
    unchanged(root, frozen)
    dump(OUT / 'original_failure_reproduction.json', {
        'PASS': True, 'expected_failure_reproduced': True, 'exit_code': r.returncode,
        'reason': 'Native powershell -File receives -Python without an argument when installer default is empty',
        'QA_root': str(root), 'stderr': str(qa / 'stderr.log'),
        'default_python_exists': (root / '.venv-lab/Scripts/python.exe').is_file(),
        'old_files_unchanged': len(frozen), 'formal_predictions': 0,
    })
    print('Original default-Python wrapper failure reproduced; logs preserved')


def package():
    old = read(OUT / 'before.json')['files']
    changed = [n for n, v in old.items() if sha(ROOT / n) != v['sha256']]
    need(changed == ['deployment/b5c_paired_r2_v1/Install-R2.ps1'], 'NOT_INSTALLER_ONLY_BEFORE_REFREEZE:' + str(changed))
    manifest_path = DEPLOY / 'package_manifest.json'
    manifest = read(manifest_path)
    name = 'deployment/b5c_paired_r2_v1/Install-R2.ps1'
    manifest['files'][name] = {'sha256': sha(DEPLOY / 'Install-R2.ps1'), 'bytes': (DEPLOY / 'Install-R2.ps1').stat().st_size}
    with manifest_path.open('w', encoding='utf8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    for n, v in manifest['files'].items():
        need(sha(ROOT / n) == v['sha256'], 'DEPENDENCY_HASH_DRIFT:' + n)
    paths = sorted(p for p in DEPLOY.rglob('*') if p.is_file())
    need(len(paths) == 14, 'DEPLOYMENT_FILE_COUNT_DRIFT')
    records = {p.relative_to(ROOT).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths}
    with zipfile.ZipFile(PACKAGE, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in paths:
            z.write(p, p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(PACKAGE) as z:
        need(z.testzip() is None, 'NEW_ZIP_CRC_FAILED')
        for n, v in records.items():
            need(hashlib.sha256(z.read(n)).hexdigest() == v['sha256'], 'ZIP_MEMBER_HASH_DRIFT')
    PACKAGE.with_suffix('.zip.sha256').write_text(sha(PACKAGE) + '\n', encoding='ascii')
    with (OUT / 'SP_U12B5C_R2_InstallerFix_File_Manifest.csv').open('x', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=('relative_path', 'bytes', 'sha256'))
        w.writeheader()
        for n, v in records.items():
            w.writerow({'relative_path': n, **v})
    shutil.copy2(DEPLOY / 'Install-R2.ps1', OUT / 'Install-R2.ps1')
    diff = ''.join(difflib.unified_diff(
        (OUT / 'original_r2_deployment/Install-R2.ps1').read_text(encoding='utf8').splitlines(True),
        (DEPLOY / 'Install-R2.ps1').read_text(encoding='utf8').splitlines(True),
        fromfile='original/Install-R2.ps1', tofile='fixed/Install-R2.ps1'))
    (OUT / 'minimal_fix.diff').write_text(diff, encoding='utf8')
    dump(OUT / 'package_report.json', {'path': str(PACKAGE), 'sha256': sha(PACKAGE), 'bytes': PACKAGE.stat().st_size, 'files': len(paths), 'unpacked_bytes': sum(v['bytes'] for v in records.values()), 'models_or_B2_data_bundled': False})
    print(json.dumps(read(OUT / 'package_report.json'), ensure_ascii=False, indent=2))


def finalize():
    before_record = read(OUT / 'before.json')
    allowed = {'deployment/b5c_paired_r2_v1/Install-R2.ps1', 'deployment/b5c_paired_r2_v1/package_manifest.json'}
    changed = [n for n, v in before_record['files'].items() if sha(ROOT / n) != v['sha256']]
    need(set(changed) == allowed, 'UNAUTHORIZED_FILE_CHANGE:' + str(changed))
    historical = read(ROOT / before_record['historical_baseline'])['files']
    drift = [n for n, v in historical.items() if sha(ROOT / n) != v['sha256']]
    need(not drift, 'HISTORICAL_FILE_DRIFT')
    need(git_state() == before_record['git'], 'GIT_DRIFT')
    suites = list(ET.parse(OUT / 'pytest.xml').getroot().iter('testsuite'))
    totals = {k: sum(int(q.attrib.get(k, 0)) for q in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    need(totals == {'tests': 10, 'failures': 0, 'errors': 0, 'skipped': 0}, 'TARGETED_PYTEST_NOT_PASS')
    p = read(OUT / 'package_report.json')
    need(sha(PACKAGE) == p['sha256'] == PACKAGE.with_suffix('.zip.sha256').read_text().strip(), 'FINAL_ZIP_HASH_DRIFT')
    for n, v in read(DEPLOY / 'package_manifest.json')['files'].items():
        need(sha(ROOT / n) == v['sha256'], 'FINAL_DEPENDENCY_DRIFT:' + n)
    r = {
        'verdict': 'R2_INSTALLER_EMPTY_PYTHON_FIX_READY', 'PASS': True,
        'changed_files': changed, 'pytest': totals, 'package': p,
        'lab_export_review': read(OUT / 'lab_export_review.json'),
        'historical_files_opaque_checked': len(historical), 'historical_drift': drift,
        'historical_bytes': sum(v['bytes'] for v in historical.values()),
        'original_R2_package_unchanged': True, 'tracked_staged_and_HEAD_unchanged': True,
        'science_model_data_statistics_changed': False, 'IdentityAudit_rerun': False,
        'formal_evaluation_executed': False, 'old_failed_attempt_resumed': False,
        'notes': 'Actual LAB audit receipts reviewed; LAB current state is rechecked by existing R2 gate. No scientific replay or new authorization.',
    }
    dump(OUT / 'final_report.json', r)
    doc = ROOT / 'docs/sp_u12b5c_r2_installer_fix.md'
    need(not doc.exists(), 'FIX_DOCUMENT_ALREADY_EXISTS')
    with doc.open('x', encoding='utf8') as f:
        f.write(f'''# R2 default-Python installer repair

Verdict: R2_INSTALLER_EMPTY_PYTHON_FIX_READY.

Original child powershell -File received -Python without an argument when the
installer parameter was omitted. This was actually reproduced using the original
R2 ZIP and a Windows root with a valid default test interpreter. The earlier R2
deployment QA supplied an explicit Python path, so it missed the default branch.

Only two executable lines were added to Install-R2.ps1: resolve empty/whitespace
Python to ProjectRoot/.venv-lab/Scripts/python.exe, then reject a missing interpreter
before deployment. Only this script and its derived package_manifest hash changed.
Original R2 package and archived deployment bytes remain available and unchanged.
No change to R2.ps1, control.py, worker.py, science.py, statistics, models, protocol
or the original failed attempt. Targeted actual Windows pytest: 10 passed.

The supplied LAB export has seven payload hashes matching EXPORT_SHA256.json.
Identity job: lab_runs/b5c_paired_r1/technical/identity_4a0a010d221e.
COMPLETE/PASS, 5000 corrected proofs, zero scientific seed mismatches, 35 cells,
210000 scored identities, 64 old files with no changes, old STOP and consumed
marker in the preservation inventory. Exported ToyTests: 86 passed. Manual Verify
PASS is user-reported; this export has no separate Verify log. No IdentityAudit
was rerun and no scientific arrays were decoded on the personal computer.

New ZIP: {p['path']}
Bytes: {p['bytes']}; members: {p['files']}.
SHA256: {p['sha256']}
No B2 data, MFA parameters or virtual environment is bundled.

Copy these four files from outputs/sp_u12b5c_r2_installer_fix to
D:\\Research\\Incoming\\B5C_R2_InstallerFix:

- SP_U12B5C_R2_Execution_InstallerFix.zip
- SP_U12B5C_R2_Execution_InstallerFix.zip.sha256
- SP_U12B5C_R2_InstallerFix_File_Manifest.csv
- Install-R2.ps1

For a LAB project where R2 has not already been installed:

```powershell
$root = 'D:\\Research\\Project_MFA_Lab'
$incoming = 'D:\\Research\\Incoming\\B5C_R2_InstallerFix'
powershell -NoProfile -ExecutionPolicy Bypass -File "$incoming\\Install-R2.ps1" -ProjectRoot $root -ZipPath "$incoming\\SP_U12B5C_R2_Execution_InstallerFix.zip" -ExpectedSHA256 '{p['sha256']}'
```

The default interpreter now works without -Python. Installation performs Verify.
If R2 already exists, preserve it and STOP; do not clear a marker or overwrite a
partial installation. Existing R1 installation and its passed audit are retained.

After mentor review, use the existing audit receipt without rerunning IdentityAudit:

```powershell
Set-Location 'D:\\Research\\Project_MFA_Lab'
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r2_v1\\R2.ps1 -Action Preflight -R1AuditJob 'lab_runs/b5c_paired_r1/technical/identity_4a0a010d221e'
```

Preflight retains the frozen current-file hash, model, resource and concurrency
checks. Only after mentor review and PASS may the operator use the unchanged once
launch command, which was not executed during this repair:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r2_v1\\R2.ps1 -Action Run -R1AuditJob 'lab_runs/b5c_paired_r1/technical/identity_4a0a010d221e' -ConfirmFormalEvaluation -Background
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r2_v1\\R2.ps1 -Action Status
```

Stop if needed; export after termination/completion:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r2_v1\\R2.ps1 -Action Stop
powershell -NoProfile -ExecutionPolicy Bypass -File .\\deployment\\b5c_paired_r2_v1\\R2.ps1 -Action Export
```

Historical local preservation: {len(historical)} opaque hashes, zero drift.
No formal evaluation, scientific verification replay, model change, target tuning,
manuscript change, old attempt restoration or Git commit. STOP after repair.
''')
    print(json.dumps({k: r[k] for k in ('verdict', 'changed_files', 'pytest', 'package', 'historical_files_opaque_checked', 'historical_drift', 'formal_evaluation_executed')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('action', choices=('before', 'evidence', 'reproduce', 'package', 'finalize'))
    args = a.parse_args()
    {'before': before, 'evidence': evidence, 'reproduce': reproduce, 'package': package, 'finalize': finalize}[args.action]()
