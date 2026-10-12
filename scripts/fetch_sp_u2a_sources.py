"""Read-only public-source retrieval; never import or execute author code."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/sp_u2a_sources'
REPO = 'TaeJun1999/Attention-aided-MMSE'


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'research-source-audit'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    OUT.mkdir(exist_ok=True)
    records = []

    def save(name, url):
        data = fetch(url)
        path = OUT / name
        if path.exists() and path.read_bytes() != data:
            raise RuntimeError(f'refusing to overwrite a different source snapshot: {name}')
        path.write_bytes(data)
        records.append(dict(path=str(path.relative_to(ROOT)).replace('\\', '/'),
                            url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        return data

    repo = json.loads(save('repository.json', f'https://api.github.com/repos/{REPO}'))
    commit = json.loads(save('commit.json', f'https://api.github.com/repos/{REPO}/commits/{repo["default_branch"]}'))
    pin = commit['sha']
    for name in ('README.md', 'A-MMSE/AMMSE.py', 'A-MMSE/main.py', 'A-MMSE/data.py', 'A-MMSE/train_ra.py'):
        save(name.replace('/', '__') + '.txt', f'https://raw.githubusercontent.com/{REPO}/{pin}/{name}')
    tree = json.loads(save('tree.json', f'https://api.github.com/repos/{REPO}/git/trees/{pin}?recursive=1'))
    old = json.loads((ROOT / 'outputs/sp_u1_author_code_access.json').read_text(encoding='utf-8'))[0]
    comparisons = {}
    for name, info in old['read_files'].items():
        data = (OUT / (name.replace('/', '__') + '.txt')).read_bytes()
        comparisons[name] = dict(old_sha256=info['sha256'], current_sha256=hashlib.sha256(data).hexdigest(),
                                 byte_identical=info['sha256'] == hashlib.sha256(data).hexdigest())
    result = dict(date='2026-10-04', repository=REPO, current_commit=pin,
                  old_sp_u1_commit=old['commit'], reproduction_target='2506.00452v5',
                  license=repo.get('license'), license_paths=[x['path'] for x in tree['tree'] if 'license' in x['path'].lower()],
                  author_code_executed=False, sources=records, previous_snapshot_comparison=comparisons)
    (ROOT / 'outputs/sp_u2a_source_access.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(commit=pin, source_count=len(records), license=repo.get('license'),
                          comparisons=comparisons), indent=2))


if __name__ == '__main__':
    main()
