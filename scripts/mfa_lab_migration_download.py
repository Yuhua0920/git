"""Download exact public Windows wheels and Python installer; no installation."""
from pathlib import Path
import sys,json,hashlib,urllib.request,importlib.metadata as md
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'.venv-gmm/Lib/site-packages'))
from scripts.mfa_lab_migration_common import DEPLOY,write
from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename

def download():
    dest=DEPLOY/'wheelhouse';dest.mkdir(parents=True,exist_ok=True)
    todo=['numpy','scipy','scikit-learn','threadpoolctl','pytest','PyYAML'];versions={}
    while todo:
        name=todo.pop(0);dist=md.distribution(name);canonical=dist.metadata['Name'].lower().replace('_','-')
        if canonical in versions:continue
        versions[canonical]=dist.version
        for s in dist.requires or []:
            r=Requirement(s)
            if r.marker is None or r.marker.evaluate({'extra':''}):todo.append(r.name)
    rows=[]
    for name,version in sorted(versions.items()):
        api=f'https://pypi.org/pypi/{name}/{version}/json'
        with urllib.request.urlopen(api,timeout=60)as response:meta=json.load(response)
        candidates=[]
        for row in meta['urls']:
            if row['packagetype']!='bdist_wheel' or row.get('yanked'):continue
            _,_,_,tags=parse_wheel_filename(row['filename'])
            if any(t.platform=='win_amd64'and t.interpreter=='cp313'and t.abi in ('cp313','abi3')for t in tags):candidates.append((0,row))
            elif any(t.platform=='any'and t.interpreter in ('py3','py2.py3')for t in tags):candidates.append((1,row))
            elif any(t.platform=='win_amd64'and t.abi=='abi3'and t.interpreter.startswith('cp3')and int(t.interpreter[2:])<=313 for t in tags):candidates.append((2,row))
        assert candidates,'NO_APPROVED_WINDOWS_WHEEL:'+name
        row=sorted(candidates,key=lambda x:(x[0],x[1]['filename']))[0][1];path=dest/row['filename']
        if not path.exists():
            with urllib.request.urlopen(row['url'],timeout=120)as r,path.open('xb')as f:
                while block:=r.read(1024*1024):f.write(block)
        digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==row['digests']['sha256']
        rows.append({'package':name,'version':version,'file':path.name,'sha256':digest,'bytes':path.stat().st_size,'source':api,'download_url':row['url']})
        print(json.dumps({'downloaded':path.name,'bytes':path.stat().st_size}),flush=True)
    requirements='\n'.join(f"{r['package']}=={r['version']} --hash=sha256:{r['sha256']}"for r in rows)+'\n'
    (DEPLOY/'requirements-win-cpu.lock').write_text(requirements,encoding='utf8')
    installers=DEPLOY/'installers';installers.mkdir(exist_ok=True)
    url='https://www.python.org/ftp/python/3.13.15/python-3.13.15-amd64.exe';path=installers/'python-3.13.15-amd64.exe'
    if not path.exists():
        with urllib.request.urlopen(url,timeout=120)as r,path.open('xb')as f:
            while b:=r.read(1024*1024):f.write(b)
    digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest=='edec09c4853aeae9ac36efb8c9f95b6b8e2fee65eee56d9767a8b7c69c574403'
    write(DEPLOY/'environment_source.json',{'Python':'3.13.15','architecture':'CPython standard-GIL Windows x86-64',
        'wheels':rows,'installer':{'file':path.name,'sha256':digest,'bytes':path.stat().st_size,'source':url},
        'installation_on_personal_computer':False,'personal_virtual_environment_included':False})
    print(json.dumps({'OFFLINE_DEPENDENCIES_READY':True,'wheels':len(rows)}))

if __name__=='__main__':download()
