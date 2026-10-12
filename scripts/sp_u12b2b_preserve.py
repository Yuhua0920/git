"""Opaque before/after protection; never decode historical scientific outputs."""
from pathlib import Path
import hashlib,json,time,sys,subprocess
R=Path(__file__).resolve().parents[1];O=R/'outputs/sp_u12b2b'
def owned(p):return any(x.startswith(('sp_u12b2b','run_sp_u12b2b','test_sp_u12b2b')) for x in p.parts)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def paths():
 result=[p for folder in ('src','scripts','tests','configs','docs','paper','outputs') for p in (R/folder).rglob('*') if p.is_file() and not owned(p.relative_to(R)) and not any(x in p.parts for x in ('__pycache__','.pytest_cache','pytest_cache','pytest_tmp'))]
 result += [R/name for name in ('main.pdf','Sun_2024_SSRN_4944730.pdf') if (R/name).is_file()]
 return sorted(result)
def save(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def run(mode):
 O.mkdir(exist_ok=True);before=O/'before.json';tick=time.monotonic()
 if mode=='freeze':
  if before.exists():raise RuntimeError('BASELINE_EXISTS')
  hashes={}
  for p in paths():
   hashes[p.relative_to(R).as_posix()]=sha(p)
   if time.monotonic()-tick>25:print({'protected_hashes':len(hashes)},flush=True);tick=time.monotonic()
  save(before,dict(hashes=hashes,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()))
  print({'protection_frozen':len(hashes)},flush=True)
 else:
  baseline=json.loads(before.read_text(encoding='utf8'));bad=[]
  for i,(name,digest) in enumerate(baseline['hashes'].items(),1):
   if not (R/name).exists() or sha(R/name)!=digest:bad.append(name)
   if time.monotonic()-tick>25:print({'protected_hashes_checked':i},flush=True);tick=time.monotonic()
  additions=[p.relative_to(R).as_posix() for p in paths() if p.relative_to(R).as_posix() not in baseline['hashes']]
  git_same=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==baseline['git_head']
  audit=dict(PASS=not(bad or additions) and git_same,protected_files=len(baseline['hashes']),changed_protected=bad,unauthorized_additions=additions,manuscript_drift=sum(n.startswith('paper/') for n in bad),production_estimator_drift=sum(n.startswith('src/') for n in bad),old_scientific_artifact_drift=sum(n.startswith('outputs/') for n in bad),unrelated_config_drift=sum(n.startswith('configs/') for n in bad),git_HEAD_unchanged=git_same)
  save(R/'outputs/sp_u12b2b_preservation.json',audit);print(audit,flush=True)
  if not audit['PASS']:raise RuntimeError('PROTECTED_ARTIFACT_DRIFT')
if __name__=='__main__':run(sys.argv[1])
