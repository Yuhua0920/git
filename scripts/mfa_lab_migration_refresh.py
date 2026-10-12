"""Record and refresh a deployment-only staged installer revision before QA."""
from pathlib import Path
import sys, json, csv, shutil
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.mfa_lab_migration_common import OUT, read, write, sha
from scripts.mfa_lab_migration_prepare import STAGE

assert not (OUT / 'Project_MFA_Lab_v1.zip').exists()
names = ['deployment/mfa_lab_v1/' + n for n in
         ('Install-Environment.ps1', 'Verify-Package.ps1', 'Start-LabJob.ps1', 'lab.py')]
manifest = read(STAGE / 'migration_manifest.json')
old_manifest_hash = sha(STAGE / 'migration_manifest.json')
rows = read(OUT / 'file_inventory.json')
changes = []
for name in names:
    old_file_hash = manifest['files'][name]['sha256']
    shutil.copyfile(ROOT / name, STAGE / name)
    info = {'bytes': (STAGE / name).stat().st_size, 'sha256': sha(STAGE / name)}
    manifest['files'][name].update(info)
    for row in rows:
        if row['path'] == name:
            row.update(info)
    if old_file_hash != info['sha256']:
        changes.append({'path': name, 'old_sha256': old_file_hash, 'new_sha256': info['sha256']})
write(OUT / 'file_inventory.json', rows)
with (OUT / 'necessary_files.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
shutil.copyfile(OUT / 'necessary_files.csv', STAGE / 'necessary_files.csv')
manifest['files']['necessary_files.csv'].update(
    bytes=(STAGE / 'necessary_files.csv').stat().st_size,
    sha256=sha(STAGE / 'necessary_files.csv'))
write(STAGE / 'migration_manifest.json', manifest)
(STAGE / 'SHA256SUMS.txt').write_text(
    ''.join(f"{v['sha256']}  {n}\n" for n, v in sorted(manifest['files'].items()))
    + sha(STAGE / 'migration_manifest.json') + '  migration_manifest.json\n', encoding='utf8')
ready = read(OUT / 'stage_ready.json')
ready.update(payload_bytes=sum(v['bytes'] for v in manifest['files'].values()),
             manifest_sha256=sha(STAGE / 'migration_manifest.json'))
write(OUT / 'stage_ready.json', ready)
write(OUT / 'stage_refresh_record_2.json', {
    'reason': 'PowerShell stream hashing compatibility, immediate job status and supervisor exception bookkeeping.',
    'changes': changes, 'old_manifest_sha256': old_manifest_hash,
    'new_manifest_sha256': ready['manifest_sha256'],
    'scientific_files_modified': 0, 'before_relocation_validation': True})
print(json.dumps({'PASS': True, 'deployment_only_refresh': changes}))
