"""Read accepted pre-unseal evidence from its committed snapshot, never rewrite it."""
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'e454d11884ec864486b1de7b5ce9a05e5f643479'
EVOLVING = frozenset(('scripts/audit_formal_preflight.py', 'scripts/assign_formal_seeds.py',
                      'tests/test_formal_preflight.py', 'tests/test_formal_seed_governance.py'))


def blob(path):
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', COMMIT, 'HEAD'], cwd=ROOT)
    return subprocess.check_output(['git', 'show', f'{COMMIT}:{path}'], cwd=ROOT)


def historical_hashes(hashes):
    for path, expected in hashes.items():
        content = blob(path) if path in EVOLVING else (ROOT / path).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('Historical snapshot/content SHA256 mismatch: ' + path)


def sealed_snapshot():
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', COMMIT], cwd=ROOT, text=True).splitlines()
    if 'outputs/t20b_formal_unseal.json' in paths:
        raise ValueError('Accepted preparation snapshot already unsealed')
    return True
