"""Public author-repository audit. Downloads no dataset and executes no author code."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]

def get(url):
    with urlopen(Request(url, headers={'User-Agent':'SP-U1-literature-audit'}),timeout=30) as r:
        return r.read()

def main():
    reports=[]
    for repo in ['TaeJun1999/Attention-aided-MMSE','dianixn/Channelformer','benediktfesl/cplx-mfa']:
        record={'repository':repo,'date':'2026-10-04','author_code_executed':False}
        try:
            meta=json.loads(get('https://api.github.com/repos/'+repo))
            sha=json.loads(get('https://api.github.com/repos/'+repo+'/commits/'+meta['default_branch']))['sha']
            tree=json.loads(get('https://api.github.com/repos/'+repo+'/git/trees/'+sha+'?recursive=1'))
            paths=[v['path'] for v in tree['tree'] if v['type']=='blob']
            license_meta=meta.get('license')
            record.update(commit=sha,license=license_meta,paths=paths)
            select=['README.md','LICENSE','pyproject.toml','A-MMSE/main.py','A-MMSE/AMMSE.py','A-MMSE/data.py','OFDM.py','Online_pruning.m']
            files={}
            for p in select:
                if p in paths:
                    raw=get('https://raw.githubusercontent.com/'+repo+'/'+sha+'/'+p)
                    lines=raw.decode('utf-8',errors='replace').splitlines()
                    selected=[i+1 for i,l in enumerate(lines) if any(k in l for k in ['import ','load_channel','USE_','epochs','BATCH','GNU GENERAL','Version 3','train_test_split','num_val','num_test','CLEAN_PATH','optimizer','sharing_mode'])]
                    files[p]={'sha256':hashlib.sha256(raw).hexdigest(),'selected_lines':selected[:60]}
            record['read_files']=files
        except Exception as e:
            record['retrieval_error']=str(e)
        reports.append(record)
    target=ROOT/'outputs/sp_u1_author_code_access.json'
    target.write_text(json.dumps(reports,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'repository':r['repository'],'commit':r.get('commit'),
                      'license':r.get('license'),'files_read':list(r.get('read_files',{})),
                      'retrieval_error':r.get('retrieval_error')} for r in reports],indent=2))

if __name__=='__main__':main()
