"""Opaque full preservation, including Run2 caches and immutable failure archive."""
from pathlib import Path
import sys,json,hashlib,time,subprocess,concurrent.futures,shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u12b3e'
def owned(p):return any(v.startswith(('sp_u12b3e','sp_u12b3f','run_sp_u12b3e','run_sp_u12b3f','test_sp_u12b3e')) for v in p.parts)
def is_run2(p):return any(v.startswith(('sp_u12b3d','run_sp_u12b3d','test_sp_u12b3d')) for v in p.parts)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2)+'\n',encoding='utf8')
def tree(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def paths():
 skip={'__pycache__','.pytest_cache','pytest_cache','pytest_tmp'}
 return sorted(p for d in ('src','scripts','tests','configs','docs','paper','outputs') for p in (ROOT/d).rglob('*') if p.is_file() and not owned(p.relative_to(ROOT)) and (is_run2(p.relative_to(ROOT)) or not(set(p.parts)&skip)))
def b2check(hashes):
 mpath=ROOT/'outputs/sp_u12b2b_sealed_manifest.json';m=json.loads(mpath.read_text(encoding='utf8'));leaves={}
 for n,r in m['files'].items():
  p=ROOT/'outputs/sp_u12b2b_sealed'/n;v=dict(sha256=hashes[p.relative_to(ROOT).as_posix()],bytes=p.stat().st_size)
  if v!=r:raise RuntimeError('B2_LEAF_CHANGED')
  leaves[n]=v
 assert len(leaves)==35031 and tree(leaves)==m['all_artifact_hash_tree'] and tree({n:v for n,v in leaves.items() if n.startswith('raw_evaluation/')})==m['raw_data_hash_tree']
 assert set(leaves)=={p.relative_to(ROOT/'outputs/sp_u12b2b_sealed').as_posix() for p in (ROOT/'outputs/sp_u12b2b_sealed').rglob('*') if p.is_file()}
 return dict(PASS=True,files=len(leaves),raw_hash_tree=m['raw_data_hash_tree'],artifact_hash_tree=m['all_artifact_hash_tree'],manifest_sha256=hashes[mpath.relative_to(ROOT).as_posix()],certificate_sha256=hashes['docs/sp_u12b2b_seal_certificate.md'],outcomes_decoded=False)
def run(mode):
 OUT.mkdir(exist_ok=True);base=OUT/'preservation_before.json'
 if mode=='before' and base.exists():raise RuntimeError('BASELINE_EXISTS')
 pp=paths();hashes={};last=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for p,digest in zip(pp,pool.map(sha,pp)):
   hashes[p.relative_to(ROOT).as_posix()]=digest
   if time.monotonic()-last>25:print(json.dumps(dict(stage='preservation',files=len(hashes))),flush=True);last=time.monotonic()
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();check=b2check(hashes)
 assert hashes['outputs/sp_u12b3b_manifest.json']=='530cc5bc124bfc945a381fb6a3d30ad2ec70666c2887a48aa9b69f9847d8a250'
 assert hashes['outputs/sp_u12b3d_run2_manifest.json']=='a22d244473336acda5ab8b75d9a848e44f8b1fcab19fea2391b93649d361e7bb'
 archives=[]
 for ledger in ('sp_u12b3c_failed_run1_archive.json','sp_u12b3c_failed_run1_archive_supplement.json'):
  f=ROOT/'outputs'/ledger;o=json.loads(f.read_text(encoding='utf8'))
  for n,r in o['files'].items():assert sha(ROOT/n)==r['sha256'] and sha(ROOT/r['archive_path'])==r['sha256']
  archives.append(dict(path=str(f),sha256=sha(f),files=len(o['files'])))
 if mode=='before':
  failed={};archive=OUT/'RUN2_FAILURE_IMMUTABLE_ARCHIVE'
  for p in pp:
   n=p.relative_to(ROOT).as_posix()
   if is_run2(p.relative_to(ROOT)):
    dest=archive/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);assert sha(dest)==hashes[n]
    failed[n]=dict(sha256=hashes[n],bytes=p.stat().st_size,archive_path=dest.relative_to(ROOT).as_posix())
  write(OUT/'run2_archive_manifest.json',dict(status='TECHNICAL_FAILURE_BEFORE_SCIENTIFIC_EXECUTION',original_failure_manifest_sha256=hashes['outputs/sp_u12b3d_run2_manifest.json'],original_bytes_modified=False,files=failed,hash_tree=tree(failed),outcomes_viewed=False))
  write(base,dict(hashes=hashes,git_HEAD=head,B2=check,run1_archives=archives));write(OUT/'B2_before.json',check)
  print(json.dumps(dict(stage='preservation_before',PASS=True,protected_files=len(hashes),B2_files=check['files'],run2_archive_files=len(failed))),flush=True);return
 prior=json.loads(base.read_text(encoding='utf8'));changed=[n for n,h in prior['hashes'].items() if hashes.get(n)!=h];added=[n for n in hashes if n not in prior['hashes']]
 failed=json.loads((OUT/'run2_archive_manifest.json').read_text(encoding='utf8'))
 for n,r in failed['files'].items():assert sha(ROOT/r['archive_path'])==r['sha256'] and hashes[n]==r['sha256']
 audit=dict(PASS=not changed and not added and prior['git_HEAD']==head,protected_files=len(hashes),changed=changed,unauthorized_added=added,production_semantic_drift=sum(n.startswith('src/') for n in changed),scientific_artifact_drift=sum(n.startswith('outputs/') for n in changed),manuscript_drift=sum(n.startswith('paper/') for n in changed),git_HEAD_unchanged=prior['git_HEAD']==head)
 write(ROOT/'outputs/sp_u12b3e_preservation.json',audit);write(ROOT/'outputs/sp_u12b3e_b2_seal_audit.json',dict(PASS=check==prior['B2'],before=prior['B2'],after=check))
 write(ROOT/'outputs/sp_u12b3e_run1_run2_preservation.json',dict(PASS=archives==prior['run1_archives'],Run1_unchanged=True,Run2_original_and_archive_unchanged=True,Run2_status='TECHNICAL_FAILURE_BEFORE_SCIENTIFIC_EXECUTION',Run2_failure_manifest_sha256=hashes['outputs/sp_u12b3d_run2_manifest.json'],Run2_archive_manifest_sha256=sha(OUT/'run2_archive_manifest.json'),Run2_files=len(failed['files']),Run1_archives=archives))
 print(json.dumps(dict(stage='preservation_after',**audit)),flush=True)
 if not audit['PASS'] or check!=prior['B2'] or archives!=prior['run1_archives']:raise RuntimeError('PRESERVATION_FAILURE')
if __name__=='__main__':run(sys.argv[1])
