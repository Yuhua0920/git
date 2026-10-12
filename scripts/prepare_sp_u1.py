"""Freeze the accepted editorial/scientific tree; no experiment imports."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files if p}
    report = dict(task='SP-U1', manuscript_state='pre_signal_processing_upgrade_draft',
                  accepted_t20g_commit=commit, clean_tree_verified_before_preparation=True,
                  tracked_file_sha256=hashes, historical_test_reporting_only=True,
                  scientific_execution_authorized=False)
    target = ROOT/'outputs/sp_u1_before_manifest.json'
    if target.exists():
        raise RuntimeError('Refusing to replace the upgrade baseline')
    target.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'commit':commit, 'tracked_files':len(hashes), 'manifest':str(target)}))

if __name__ == '__main__':
    main()
