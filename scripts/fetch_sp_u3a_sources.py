"""Pin official public source bytes without installing or executing packages."""
import datetime
import io
from pathlib import Path
import sys
import zipfile
import subprocess
from urllib.error import URLError
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u3a_common as c

def get(url):
    try:
        with urlopen(Request(url,headers={'User-Agent':'SP-U3A-source-audit'}),timeout=45) as response:return response.read()
    except URLError as error:
        # Windows' certificate store is used; TLS verification is never disabled.
        target=c.OUT/'sp_u3a_sources'/('http_'+c.hashlib.sha256(url.encode()).hexdigest()+'.bin')
        assert not target.exists()
        command="Invoke-WebRequest -Uri '"+url.replace("'","''")+"' -OutFile '"+str(target).replace("'","''")+"'"
        subprocess.run(['powershell','-NoProfile','-Command',command],check=True,capture_output=True,text=True)
        return target.read_bytes()

def main():
    c.verify();base=c.OUT/'sp_u3a_sources'
    base.mkdir(exist_ok=True)
    reports=[]
    for name in ('cplx-mfa','MFA_estimator'):
        pin=base/name/'repository_pin.json'
        if pin.exists():
            record=c.load(pin)
            assert all(c.sha(base/name/p)==h for p,h in record['files'].items())
            reports.append(record);print(name,'preserved exact',record['commit'],flush=True);continue
        repo='benediktfesl/'+name
        meta=c.json.loads(get('https://api.github.com/repos/'+repo))
        commit=c.json.loads(get('https://api.github.com/repos/'+repo+'/commits/'+meta['default_branch']))
        sha=commit['sha'];root=base/name;root.mkdir()
        archive=get('https://codeload.github.com/'+repo+'/zip/'+sha)
        (root/'source.zip').write_bytes(archive)
        files={}
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            for item in z.infolist():
                if item.is_dir():continue
                relative=Path(*Path(item.filename).parts[1:])
                assert not relative.is_absolute() and '..' not in relative.parts
                target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(item))
                files[relative.as_posix()]=c.sha(target)
        record=dict(repository='https://github.com/'+repo,commit=sha,commit_timestamp=commit['commit']['committer']['date'],
          accessed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),license_metadata=meta['license'],files=files)
        c.save(root/'repository_pin.json',record);reports.append(record)
        print(name,sha,'files',len(files),flush=True)
    for name,url in [('paper_v2.pdf','https://arxiv.org/pdf/2304.14809v2'),
      ('dissertation_2025.pdf','https://mediatum.ub.tum.de/doc/1748775/1748775.pdf'),
      ('paper_metadata.html','https://arxiv.org/abs/2304.14809v2'),
      ('publisher_metadata.json','https://api.crossref.org/works/10.1109/IEEECONF59524.2023.10477088'),
      ('cplx_mfa_pypi.json','https://pypi.org/pypi/cplx-mfa/json'),
      ('mfa_estimator_pypi.json','https://pypi.org/pypi/mfa-estimator/json')]:
        if (base/name).exists():raw=(base/name).read_bytes()
        else:raw=get(url);(base/name).write_bytes(raw)
        print(name,len(raw),'bytes',flush=True)
    c.save(c.OUT/'sp_u3a_source_retrieval.json',dict(repositories=reports,
      files={p.name:c.sha(p) for p in base.iterdir() if p.is_file()},retrieved_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      paper='arXiv:2304.14809v2',software_executed=False,dependencies_installed=False,
      retrieval_attempts=['Attempt1 sandbox network WinError10013, no files retrieved',
       'Attempt2 permitted network downloaded both repositories; arXiv Python TLS issuer verification failed',
       'Attempt3 uses verified existing repositories and Windows certificate-store transport; TLS verification retained'],access=c.ACCESS))

if __name__=='__main__':main()
