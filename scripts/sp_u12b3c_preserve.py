"""Opaque hash preservation and immutable FAILED_FORMAL_RUN_1 byte archive."""
from pathlib import Path
import sys,json,hashlib,time,subprocess,shutil,concurrent.futures
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u12b3c'
def owned(p):return any(v.startswith(('sp_u12b3c','run_sp_u12b3c','test_sp_u12b3c','sp_u12b3b_sealed_run2')) for v in p.parts)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2)+'\n',encoding='utf8')
def tree(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def paths():
 skip={'__pycache__','.pytest_cache','pytest_cache','pytest_tmp'}
 return sorted(p for d in ('src','scripts','tests','configs','docs','paper','outputs') for p in (ROOT/d).rglob('*') if p.is_file() and not owned(p.relative_to(ROOT)) and not(set(p.parts)&skip))
def b2check(hashes):
 mpath=ROOT/'outputs/sp_u12b2b_sealed_manifest.json';m=json.loads(mpath.read_text(encoding='utf8'));leaves={}
 for n,r in m['files'].items():
  p=ROOT/'outputs/sp_u12b2b_sealed'/n;v=dict(sha256=hashes[p.relative_to(ROOT).as_posix()],bytes=p.stat().st_size)
  if v!=r:raise RuntimeError('B2_LEAF_CHANGED')
  leaves[n]=v
 if tree(leaves)!=m['all_artifact_hash_tree'] or tree({n:v for n,v in leaves.items() if n.startswith('raw_evaluation/')})!=m['raw_data_hash_tree']:raise RuntimeError('B2_TREE_CHANGED')
 if set(leaves)!={p.relative_to(ROOT/'outputs/sp_u12b2b_sealed').as_posix() for p in (ROOT/'outputs/sp_u12b2b_sealed').rglob('*') if p.is_file()}:raise RuntimeError('B2_FILE_SET_CHANGED')
 return dict(PASS=True,files=len(leaves),raw_hash_tree=m['raw_data_hash_tree'],artifact_hash_tree=m['all_artifact_hash_tree'],manifest_sha256=hashes[mpath.relative_to(ROOT).as_posix()],certificate_sha256=hashes['docs/sp_u12b2b_seal_certificate.md'],numerical_outcomes_decoded=False)
def run(mode):
 OUT.mkdir(exist_ok=True);base=OUT/'preservation_before.json'
 if mode=='before' and base.exists():raise RuntimeError('BASELINE_EXISTS')
 pp=paths();hashes={};last=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for p,digest in zip(pp,pool.map(sha,pp)):
   hashes[p.relative_to(ROOT).as_posix()]=digest
   if time.monotonic()-last>25:print(json.dumps(dict(stage='preservation',files=len(hashes))),flush=True);last=time.monotonic()
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();check=b2check(hashes)
 if mode=='before':
  write(base,dict(hashes=hashes,git_HEAD=head,B2=check));failed={};archive=OUT/'FAILED_FORMAL_RUN_1'
  for p in pp:
   n=p.relative_to(ROOT).as_posix()
   if any(v.startswith(('sp_u12b3b','run_sp_u12b3b','test_sp_u12b3b')) for v in p.relative_to(ROOT).parts):
    dest=archive/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    if sha(dest)!=hashes[n]:raise RuntimeError('FAILED_ARCHIVE_COPY')
    st=p.stat();failed[n]=dict(sha256=hashes[n],bytes=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,archive_path=dest.relative_to(ROOT).as_posix())
  oldmanifest='outputs/sp_u12b3b_manifest.json'
  if hashes[oldmanifest]!='530cc5bc124bfc945a381fb6a3d30ad2ec70666c2887a48aa9b69f9847d8a250':raise RuntimeError('FAILED_MANIFEST_CHANGED')
  meta=dict(run='FAILED_FORMAL_RUN_1',status='SEALED_TECHNICAL_FAILURE_UNVIEWED',immutable=True,scientific_use='PERMANENTLY_EXCLUDED',original_paths_preserved=True,original_manifest_sha256=hashes[oldmanifest],files=failed,hash_tree=tree(failed),created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),exception_state=json.loads((ROOT/'outputs/sp_u12b3b/PERMANENT_STOP.json').read_text(encoding='utf8')))
  write(ROOT/'outputs/sp_u12b3c_failed_run1_archive.json',meta);write(archive/'IMMUTABLE_MANIFEST.json',meta);write(OUT/'B2_before.json',check)
  print(json.dumps(dict(stage='archive_before',PASS=True,protected_files=len(hashes),archived_files=len(failed))),flush=True);return
 prior=json.loads(base.read_text(encoding='utf8'));changed=[n for n,h in prior['hashes'].items() if hashes.get(n)!=h];added=[n for n in hashes if n not in prior['hashes']]
 failed=json.loads((ROOT/'outputs/sp_u12b3c_failed_run1_archive.json').read_text(encoding='utf8'))
 for n,r in failed['files'].items():
  if sha(ROOT/r['archive_path'])!=r['sha256']:raise RuntimeError('FAILED_ARCHIVE_CHANGED')
 audit=dict(PASS=not changed and not added and prior['git_HEAD']==head,protected_files=len(hashes),changed=changed,unauthorized_added=added,production_semantic_drift=sum(n.startswith('src/') for n in changed),scientific_artifact_drift=sum(n.startswith('outputs/') for n in changed),manuscript_drift=sum(n.startswith('paper/') for n in changed),failed_run1_original_and_archive_unchanged=True,git_HEAD_unchanged=prior['git_HEAD']==head)
 write(ROOT/'outputs/sp_u12b3c_preservation.json',audit);write(ROOT/'outputs/sp_u12b3c_b2_seal_audit.json',dict(PASS=check==prior['B2'],before=prior['B2'],after=check));print(json.dumps(dict(stage='preservation_after',**audit)),flush=True)
 if not audit['PASS'] or check!=prior['B2']:raise RuntimeError('PRESERVATION_FAILURE')
if __name__=='__main__':run(sys.argv[1])
