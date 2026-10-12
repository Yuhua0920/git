import json,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]
url='https://api.crossref.org/works/10.1109%2FISCAS.2010.5538026'
with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SP-U9A-R1-audit/1.0'}),timeout=30) as r: data=r.read()
(root/'outputs/sp_u9a_r1/crossref_LS2010.json').write_bytes(data)
m=json.loads(data)['message']
print(json.dumps({k:m.get(k) for k in ['title','author','published','page','container-title']},ensure_ascii=False))
