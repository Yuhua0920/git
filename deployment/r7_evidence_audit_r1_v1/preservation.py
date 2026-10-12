"""Controller-only opaque preservation evidence, never scientific decoding."""
from pathlib import Path
import hashlib

FAILED_ATTEMPT = 'lab_20261009_131543_e9d32a05'
REQUIRED = ('exception.json', 'cli_exception.json', 'stderr.log',
            'final_report.json', 'status.json')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}

def snapshot(root):
    root = Path(root).resolve()
    failed = root / 'lab_runs/r7_evidence_audit/runs' / FAILED_ATTEMPT
    if not failed.is_dir() or any(not (failed / n).is_file() for n in REQUIRED):
        raise RuntimeError('ORIGINAL_FAILED_ATTEMPT_MISSING: STOP; preserve originals')
    files = {}
    for p in sorted(failed.rglob('*')):
        if p.is_file():
            if not p.resolve().is_relative_to(failed.resolve()):
                raise RuntimeError('FAILED_ATTEMPT_LINK_OUTSIDE_SCOPE')
            files[p.relative_to(root).as_posix()] = digest(p)
    caches = {}
    for repo, stems in [('cplx-mfa', ('utils', 'MFA_cplx')),
                        ('MFA_estimator', ('MFA_estimator',))]:
        parent = root / 'reference/sp_u12b5a/paper_period' / repo / '__pycache__'
        for stem in stems:
            for p in sorted(parent.glob(stem + '.*.pyc')):
                if not p.resolve().is_relative_to(parent.resolve()):
                    raise RuntimeError('REFERENCE_CACHE_LINK_OUTSIDE_SCOPE')
                caches[p.relative_to(root).as_posix()] = digest(p)
    return {'failed_attempt': FAILED_ATTEMPT, 'files': files,
            'reference_cache_opaque_hashes': caches,
            'bytecode_decoded_or_executed': False}
