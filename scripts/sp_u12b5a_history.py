"""Fetch only public author history; do not execute downloaded code."""
import io,json,zipfile,urllib.request
from pathlib import Path
from scripts.sp_u12b5a_common import ROOT,OUT,REF,sha,write,firewall
PINS={'cplx-mfa':'184e97c641cd0d1383d28bce102a53b10a527c89','MFA_estimator':'93ce8eec4770840a377581e70de14c1a447a628d'}
def get(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SP-U12B5A-provenance','Accept':'application/vnd.github+json'}),timeout=60).read()
def main():
    firewall();result={}
    for repo,commit in PINS.items():
        target=REF/'paper_period'/repo
        assert not target.exists(),'historical reference is immutable'
        raw=get(f'https://codeload.github.com/benediktfesl/{repo}/zip/{commit}')
        archive=OUT/f'{repo}_2023_commit.zip';archive.write_bytes(raw)
        z=zipfile.ZipFile(io.BytesIO(raw))
        for name in z.namelist():
            parts=Path(name).parts[1:]
            if not parts or name.endswith('/'):continue
            assert '..' not in parts
            dst=target.joinpath(*parts);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(z.read(name))
        info=json.loads(get(f'https://api.github.com/repos/benediktfesl/{repo}/git/trees/{commit}?recursive=1'))
        write(OUT/f'{repo}_2023_git_tree.json',info)
        hashes={p.relative_to(target).as_posix():sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
        import hashlib
        result[repo]=dict(commit=commit,archive_sha256=sha(archive),files=hashes,tree_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest(),submodules=[v for v in info['tree'] if v['mode']=='160000'])
    write(OUT/'historical_reference.json',result)
    print(json.dumps({k:{n:v for n,v in r.items() if n!='files'} for k,r in result.items()}))
if __name__=='__main__':main()
