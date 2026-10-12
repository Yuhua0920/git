"""B5A opaque preservation and strict outcome firewall; no formal training."""
from pathlib import Path
import sys,os,json,time,subprocess,concurrent.futures,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b4a_preserve as p
OUT=ROOT/'outputs/sp_u12b5a';REF=ROOT/'reference/sp_u12b5a'
B4M=ROOT/'outputs/sp_u12b4b_manifest.json'
B4SHA='6187d63452947c18ab2a26d60ec7359489477701e58626bfd8c139598cd252e6'
NEW_EXACT={'src/baselines/mfa_reference_adapter.py','tests/test_mfa_reference.py','scripts/validate_mfa_fidelity.py'}
_installed=False
read=p.read;write=p.write;sha=p.sha;tree=p.tree
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def owned(path):
 rel=path.relative_to(ROOT).as_posix()
 return rel in NEW_EXACT or any(v.startswith(('sp_u12b5a','run_sp_u12b5a','test_sp_u12b5a')) for v in path.relative_to(ROOT).parts)
def firewall():
 global _installed
 if _installed:return
 allowed={p.B2M,p.B3M,B4M};old_reference=ROOT/'outputs/sp_u3a_sources'
 def hook(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   path=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
   writing=isinstance(mode,str) and any(v in mode for v in 'wax+') or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
   if not path.is_relative_to(ROOT):return
   rel=path.relative_to(ROOT)
   if writing and rel.parts[0] in ('src','scripts','tests','configs','docs','paper','outputs','reference') and not owned(path) and '__pycache__' not in rel.parts:raise PermissionError('B5A_HISTORICAL_WRITE_FORBIDDEN')
   if rel.parts[0]=='outputs' and not writing and not p._opaque and path not in allowed and not owned(path) and not path.is_relative_to(old_reference):raise PermissionError('B5A_SCIENTIFIC_OUTCOME_FIREWALL')
  if event in ('os.remove','os.rename','os.rmdir','os.mkdir','os.utime'):
   for raw in args[:2] if event=='os.rename' else args[:1]:
    if isinstance(raw,(str,bytes,os.PathLike)):
     path=Path(os.fsdecode(raw)).resolve()
     parent_creation=event=='os.mkdir' and path.is_relative_to(ROOT) and path.relative_to(ROOT).as_posix() in ('reference','src/baselines') and not path.exists()
     if path.is_relative_to(ROOT) and path.relative_to(ROOT).parts[0] in ('src','scripts','tests','configs','docs','paper','outputs','reference') and not owned(path) and not parent_creation and '__pycache__' not in path.parts:raise PermissionError('B5A_HISTORICAL_MUTATION_FORBIDDEN')
 sys.addaudithook(hook);_installed=True
def protected():
 return sorted(x for d in ('src','scripts','tests','configs','docs','paper','outputs','reference') for x in (ROOT/d).rglob('*') if x.is_file() and not owned(x) and not(set(x.parts)&{'__pycache__','.pytest_cache','pytest_cache','pytest_tmp'}))
def fullhash():
 result={};last=time.monotonic();old=p._opaque;p._opaque=True
 try:
  paths=protected()
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   for path,value in zip(paths,pool.map(sha,paths)):
    result[path.relative_to(ROOT).as_posix()]=value
    if time.monotonic()-last>25:print(json.dumps(dict(stage='B5A_opaque_hash',files=len(result))),flush=True);last=time.monotonic()
 finally:p._opaque=old
 return result
def report(name,value):write(ROOT/f'outputs/sp_u12b5a_mfa_{name}.json',value)
def preserve(stage):
 OUT.mkdir(parents=True,exist_ok=True);base=OUT/'preservation_before.json'
 assert stage!='before' or not base.exists(),'B5A_BASELINE_EXISTS'
 assert sha(B4M)==B4SHA,'B4_MANIFEST_DRIFT'
 hashes=fullhash();seals=p.seals(hashes)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True,encoding='utf8');(OUT/f'git_status_{stage}.txt').write_text(status,encoding='utf8')
 check=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf8');write(OUT/f'git_diff_check_{stage}.json',dict(exit_code=check.returncode,stdout=check.stdout,stderr=check.stderr));assert check.returncode==0
 if stage=='before':write(base,dict(hashes=hashes,seals=seals,git_HEAD=head,created_utc=utc()))
 else:
  before=read(base);changed=[n for n,v in before['hashes'].items() if hashes.get(n)!=v];added=[n for n in hashes if n not in before['hashes']]
  report('preservation',dict(PASS=not changed and not added and head==before['git_HEAD'],protected_files=len(before['hashes']),changed=changed,unauthorized_added=added,production_semantic_drift=sum(n.startswith(('src/','scripts/')) for n in changed),scientific_artifact_drift=sum(n.startswith('outputs/') for n in changed),manuscript_drift=sum(n.startswith('paper/') for n in changed),Git_HEAD=head,Git_HEAD_unchanged=head==before['git_HEAD'],B4_manifest_unchanged=True))
  report('b2_b3_seal_audit',dict(PASS=seals==before['seals'],before=before['seals'],after=seals,scientific_outcomes_viewed=False))
  assert not changed and not added and head==before['git_HEAD'] and seals==before['seals']
 print(json.dumps(dict(stage='B5A_preservation_'+stage,PASS=True,protected_files=len(hashes),B2_files=seals['B2']['files'],B3_files=seals['B3']['files'])),flush=True)
if __name__=='__main__':preserve(sys.argv[1])
