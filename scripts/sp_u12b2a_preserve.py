"""Opaque preservation audit. No historical scientific payload decoding."""
from pathlib import Path
import hashlib,json,sys,time
R=Path(__file__).resolve().parents[1];O=R/'outputs/sp_u12b2a'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def owned(p):return any(x.startswith('sp_u12b2a') or x.startswith('run_sp_u12b2a') or x.startswith('test_sp_u12b2a') for x in p.parts)
def inventory():
 return sorted(p for name in ('src','scripts','tests','configs','docs','paper','outputs') for p in (R/name).rglob('*') if p.is_file() and not owned(p.relative_to(R)) and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and 'pytest_cache' not in p.parts)
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def run(mode):
 O.mkdir(exist_ok=True);path=O/'before.json';tick=time.monotonic()
 if mode=='freeze':
  if path.exists():raise RuntimeError('Preservation baseline already exists')
  hs={}
  for p in inventory():
   hs[p.relative_to(R).as_posix()]=sha(p)
   if time.monotonic()-tick>20:print({'opaque_hashes_frozen':len(hs)},flush=True);tick=time.monotonic()
  save(path,hs);print({'protected_files':len(hs)},flush=True)
 else:
  hs=json.loads(path.read_text(encoding='utf8'));bad=[]
  for i,(n,h) in enumerate(hs.items(),1):
   if not (R/n).exists() or sha(R/n)!=h:bad.append(n)
   if time.monotonic()-tick>20:print({'opaque_hashes_checked':i},flush=True);tick=time.monotonic()
  added=[p.relative_to(R).as_posix() for p in inventory() if p.relative_to(R).as_posix() not in hs]
  result=dict(protected_files=len(hs),changed_protected=bad,unauthorized_additions=added,manuscript_drift=sum(n.startswith('paper/') for n in bad),production_semantic_drift=sum(n.startswith('src/') for n in bad),scientific_artifact_drift=sum(n.startswith('outputs/') for n in bad),pass_gate=not(bad or added))
  save(R/'outputs/sp_u12b2a_preservation.json',result);print(result,flush=True)
  if not result['pass_gate']:raise RuntimeError('Preservation failure')
if __name__=='__main__':run(sys.argv[1])
