"""Opaque preservation, with one explicitly authorized preregistration amendment."""
from pathlib import Path
import sys,json,hashlib,time,subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u12b3b'
AMEND='docs/sp_u12b3a_preregistration.md'
def owned(p):return any(v.startswith(('sp_u12b3b','run_sp_u12b3b','test_sp_u12b3b')) for v in p.parts)
def paths():
 skip={'__pycache__','.pytest_cache','pytest_cache','pytest_tmp'}
 return sorted(p for d in ('src','scripts','tests','configs','docs','paper','outputs') for p in (ROOT/d).rglob('*') if p.is_file() and not owned(p.relative_to(ROOT)) and not(set(p.parts)&skip))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def save(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2)+'\n',encoding='utf8')
def run(mode):
 OUT.mkdir(exist_ok=True);base=OUT/'before.json';tick=time.monotonic();hashes={}
 if mode=='freeze' and base.exists():raise RuntimeError('BASELINE_EXISTS')
 before=json.loads(base.read_text(encoding='utf8')) if mode!='freeze' else None
 entries=paths() if mode=='freeze' else [ROOT/n for n in before['hashes']]
 for p in entries:
  hashes[p.relative_to(ROOT).as_posix()]=sha(p) if p.exists() else None
  if time.monotonic()-tick>25:print(json.dumps(dict(stage='preservation',protected_files=len(hashes))),flush=True);tick=time.monotonic()
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 if mode=='freeze':save(base,dict(hashes=hashes,git_head=head));print(json.dumps(dict(stage='preservation',protected_files=len(hashes),PASS=True)));return
 amendment=json.loads((OUT/'mentor_spec_correction.json').read_text(encoding='utf8'))
 changed=[n for n,h in hashes.items() if h!=before['hashes'][n]]
 allowed=changed==[AMEND] and before['hashes'][AMEND]==amendment['before_sha256'] and hashes[AMEND]==amendment['after_sha256']
 unexpected=[n for n in changed if n!=AMEND or not allowed]
 added=[p.relative_to(ROOT).as_posix() for p in paths() if p.relative_to(ROOT).as_posix() not in before['hashes']]
 audit=dict(PASS=not unexpected and not added and head==before['git_head'],protected_files=len(hashes),authorized_preregistration_amendment=allowed,unexpected_changes=unexpected,unauthorized_additions=added,git_HEAD_unchanged=head==before['git_head'],manuscript_drift=sum(n.startswith('paper/') for n in unexpected),production_semantic_drift=sum(n.startswith('src/') for n in unexpected),old_artifact_drift=sum(n.startswith('outputs/') for n in unexpected))
 save(ROOT/'outputs/sp_u12b3b_preservation.json',audit);save(OUT/'after_hashes.json',dict(hashes=hashes));print(json.dumps(audit));
 if not audit['PASS']:raise RuntimeError('PRESERVATION_FAILURE')
if __name__=='__main__':run(sys.argv[1])
