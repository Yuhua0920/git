"""Task-local provenance and strict output access boundary; no old outcomes."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
HEAD='ca83982b1d0e4bdca0215a3ecbe3a7a8f8f8ec55'
ACCESS=[]
ALLOWED={'t15a_cdl_manifest.csv','t15c_cdl_manifest.csv','t15a_cdl_statistics_500.npz',
 't15c_cdl_statistics_CDL-B-30.npz','t15c_cdl_statistics_CDL-C-120.npz',
 't16e_pilot_cv_preflight.json','t16e_frozen_operators.npz','t16f_preflight.json',
 't16f_frozen_candidates.npz','t16g_factor_preflight.json','t16g_frozen_module_factors.npz'}
ALLOWED.add('t15b_gmm_probe_dataset_manifest.json') # H hashes only, no performance outcomes.
ALLOWED.add('t16c_heldout_manifest.csv') # Identity/count ledger only; no target arrays.

def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    path=Path(os.fsdecode(args[0])).resolve()
    try:rel=path.relative_to(OUT.resolve()).as_posix()
    except ValueError:return
    flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
    writing=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
    allowed=rel.split('/')[0].startswith('sp_u3a_') or (rel in ALLOWED and not writing)
    ACCESS.append(dict(path=rel,writing=writing,allowed=allowed))
    if not allowed:raise PermissionError('SP-U3A firewall:'+rel)

sys.addaudithook(guard)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf8'))
def save(path,value):
    path=Path(path);assert not path.exists(),str(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n',encoding='utf8')
def csvsave(path,rows):
    path=Path(path);assert not path.exists()
    with path.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def verify():
    pre=load(OUT/'sp_u3a_before_manifest.json')
    assert git('rev-parse','HEAD')==HEAD and not git('diff','--name-only') and not git('diff','--cached','--name-only')
    assert all(sha(ROOT/p)==h for p,h in pre['protected_sources'].items())
    return pre

def before():
    assert git('rev-parse','HEAD')==HEAD
    paths=set(git('ls-files').splitlines())|set(git('ls-files','--others','--exclude-standard').splitlines())
    protected={p:sha(ROOT/p) for p in sorted(paths) if p.startswith(('src/','scripts/','tests/','configs/','docs/','paper/'))
      and 'sp_u3a' not in p and '__pycache__' not in p and (ROOT/p).is_file()}
    save(OUT/'sp_u3a_before_manifest.json',dict(task='SP-U3A',head=HEAD,protected_sources=protected,
       formal_read=False,old_U2_outcomes_read=False,production_change=False))
    print('BEFORE',len(protected),'protected sources;formal andU2 outcomes denied')

if __name__=='__main__':before()
