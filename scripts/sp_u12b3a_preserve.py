"""Opaque artifact protection for B3A; never parse scientific outcomes."""
from pathlib import Path
import hashlib, json, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/sp_u12b3a'

def owned(path):
    return any(part.startswith(('sp_u12b3a', 'run_sp_u12b3a', 'test_sp_u12b3a'))
               for part in path.parts)

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def paths():
    skip = {'__pycache__', '.pytest_cache', 'pytest_cache', 'pytest_tmp'}
    result = [p for folder in ('src', 'scripts', 'tests', 'configs', 'docs', 'paper', 'outputs')
              for p in (ROOT / folder).rglob('*') if p.is_file()
              and not owned(p.relative_to(ROOT)) and not (set(p.parts) & skip)]
    result += [ROOT / name for name in ('main.pdf', 'Sun_2024_SSRN_4944730.pdf')
               if (ROOT / name).is_file()]
    return sorted(result)

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')

def run(mode):
    OUT.mkdir(exist_ok=True)
    baseline_path = OUT / 'before.json'
    tick = time.monotonic()
    if mode == 'freeze':
        if baseline_path.exists():
            raise RuntimeError('BASELINE_EXISTS')
        hashes = {}
        for path in paths():
            hashes[path.relative_to(ROOT).as_posix()] = sha(path)
            if time.monotonic() - tick > 25:
                print({'protected_hashes': len(hashes)}, flush=True)
                tick = time.monotonic()
        save(baseline_path, dict(hashes=hashes, git_head=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
        print({'protection_frozen': len(hashes)}, flush=True)
        return
    baseline = json.loads(baseline_path.read_text(encoding='utf8'))
    changed = []
    for i, (name, digest) in enumerate(baseline['hashes'].items(), 1):
        if not (ROOT / name).exists() or sha(ROOT / name) != digest:
            changed.append(name)
        if time.monotonic() - tick > 25:
            print({'protected_hashes_checked': i}, flush=True)
            tick = time.monotonic()
    added = [p.relative_to(ROOT).as_posix() for p in paths()
             if p.relative_to(ROOT).as_posix() not in baseline['hashes']]
    same_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                        text=True).strip() == baseline['git_head']
    audit = dict(PASS=not (changed or added) and same_head,
                 protected_files=len(baseline['hashes']), changed_protected=changed,
                 unauthorized_additions=added, git_HEAD_unchanged=same_head,
                 manuscript_drift=sum(p.startswith('paper/') for p in changed),
                 production_semantic_drift=sum(p.startswith('src/') for p in changed),
                 old_artifact_drift=sum(p.startswith('outputs/') for p in changed),
                 unrelated_config_drift=sum(p.startswith('configs/') for p in changed))
    save(ROOT / 'outputs/sp_u12b3a_preservation.json', audit)
    print(audit, flush=True)
    if not audit['PASS']:
        raise RuntimeError('PROTECTED_ARTIFACT_DRIFT')

if __name__ == '__main__':
    run(sys.argv[1])
