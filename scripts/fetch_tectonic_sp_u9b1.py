"""Fetch an official portable build executable into this task's output directory."""
import hashlib,json,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u9b1/toolchain'
def fetch(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SP-U9B-1-manuscript-build'}),timeout=60) as r:return r.read()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 data=fetch('https://api.github.com/repos/tectonic-typesetting/tectonic/releases/tags/tectonic%400.17.0')
 (OUT/'release.json').write_bytes(data);release=json.loads(data)
 candidates=[a for a in release['assets'] if 'x86_64-pc-windows-msvc' in a['name'] and a['name'].endswith('.zip')]
 if len(candidates)!=1:raise ValueError([a['name'] for a in release['assets']])
 asset=candidates[0];raw=fetch(asset['browser_download_url']);archive=OUT/asset['name'];archive.write_bytes(raw)
 digest=hashlib.sha256(raw).hexdigest()
 if asset.get('digest') and asset['digest']!='sha256:'+digest:raise ValueError('Release asset digest mismatch')
 with zipfile.ZipFile(archive) as z:
  names=[n for n in z.namelist() if Path(n).name=='tectonic.exe']
  if len(names)!=1:raise ValueError(z.namelist())
  executable=OUT/'tectonic.exe';executable.write_bytes(z.read(names[0]))
 record=dict(version=release['tag_name'],url=asset['browser_download_url'],archive_sha256=digest,
  executable=str(executable),executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),github_digest=asset.get('digest'),scope='Portable task-local executable; no system TeX installation or PATH edits')
 (OUT/'toolchain.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record))
if __name__=='__main__':main()
