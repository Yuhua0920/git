"""Freeze public primary sources only; never imports an experimental runner."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/sp_u6a_sources'

def get(url):
    with urlopen(Request(url, headers={'User-Agent': 'SP-U6A-source-audit'}), timeout=45) as response:
        return response.read()

def main():
    OUT.mkdir(exist_ok=True)
    if any(OUT.iterdir()):
        raise RuntimeError('Refusing to update a nonempty source freeze')
    rows = []
    def save(name, url):
        raw = get(url)
        (OUT / name).write_bytes(raw)
        rows.append({'path': name, 'url': url, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
        return raw
    base = 'https://api.github.com/repos/TaeJun1999/Attention-aided-MMSE'
    repo = json.loads(save('repository.json', base))
    commit = json.loads(save('commit.json', base + '/commits/' + repo['default_branch']))
    sha = commit['sha']
    tree = json.loads(save('tree.json', base + '/git/trees/' + sha + '?recursive=1'))
    files = [row['path'] for row in tree['tree'] if row['type'] == 'blob']
    selected = [p for p in files if p.endswith('.py') or p.endswith('README.md') or 'license' in p.lower() or 'copying' in p.lower()]
    for p in selected:
        save(p.replace('/', '__'), 'https://raw.githubusercontent.com/TaeJun1999/Attention-aided-MMSE/' + sha + '/' + p)
    for paper in ['2506.00452', '2608.15945']:
        save(paper + '_abs.html', 'https://arxiv.org/abs/' + paper)
        version = 'v5' if paper == '2506.00452' else 'v1'
        save(paper + version + '.pdf', 'https://arxiv.org/pdf/' + paper + version)
        save(paper + version + '.html', 'https://arxiv.org/html/' + paper + version)
    ledger = {'task': 'SP-U6A', 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'commit': sha, 'commit_date': commit['commit']['committer']['date'], 'license': repo.get('license'), 'all_repository_paths': files, 'sources': rows, 'immutable': True, 'author_code_executed': False}
    (ROOT / 'outputs/sp_u6a_source_freeze.json').write_text(json.dumps(ledger, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: ledger[k] for k in ['commit', 'commit_date', 'license', 'author_code_executed']}, indent=2))
    print('Sources frozen:', len(rows))

if __name__ == '__main__':
    main()
