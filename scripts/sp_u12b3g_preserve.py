"""Opaque before/after protection; archive authorized original technical bytes."""
from pathlib import Path
import sys,json,hashlib,time,concurrent.futures,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/sp_u12b3g'
ALLOWED={'scripts/run_sp_u12b3f.py','scripts/sp_u12b3e_io.py'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def tree(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,o):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2)+'\n',encoding='utf8')
def owned(p):return any(v.startswith(('sp_u12b3g','test_sp_u12b3g','run_sp_u12b3g')) for v in p.parts)
def failed_cache(p):return any(v.startswith(('sp_u12b3b','sp_u12b3c','sp_u12b3d','sp_u12b3e','sp_u12b3f','run_sp_u12b3d','test_sp_u12b3d')) for v in p.parts)
def paths():
 skip={'__pycache__','.pytest_cache','pytest_cache','pytest_tmp'}
 return sorted(p for d in ('src','scripts','tests','configs','docs','paper','outputs') for p in (ROOT/d).rglob('*') if p.is_file() and not owned(p.relative_to(ROOT)) and (failed_cache(p.relative_to(ROOT)) or not(set(p.parts)&skip)))
def b2check(hashes):
 mpath=ROOT/'outputs/sp_u12b2b_sealed_manifest.json';m=read(mpath);base=ROOT/'outputs/sp_u12b2b_sealed';leaves={}
 for n,r in m['files'].items():
  p=base/n;v=dict(sha256=hashes[p.relative_to(ROOT).as_posix()],bytes=p.stat().st_size);assert v==r,'B2_LEAF_DRIFT';leaves[n]=v
 assert len(leaves)==35031 and tree(leaves)==m['all_artifact_hash_tree']
 assert set(leaves)=={p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}
 assert tree({n:v for n,v in leaves.items() if n.startswith('raw_evaluation/')})==m['raw_data_hash_tree']
 return dict(PASS=True,files=35031,raw_hash_tree=m['raw_data_hash_tree'],artifact_hash_tree=m['all_artifact_hash_tree'],manifest_sha256=hashes[mpath.relative_to(ROOT).as_posix()],certificate_sha256=hashes['docs/sp_u12b2b_seal_certificate.md'],outcomes_decoded=False)
def history(h):
 expected={'outputs/sp_u12b3b_manifest.json':'530cc5bc124bfc945a381fb6a3d30ad2ec70666c2887a48aa9b69f9847d8a250','outputs/sp_u12b3d_run2_manifest.json':'a22d244473336acda5ab8b75d9a848e44f8b1fcab19fea2391b93649d361e7bb','outputs/sp_u12b3f_run3_manifest.json':'8dc4d3246e46efa65fec709b59943e50bf6eccebca08c5c61e5545e6fd174454','outputs/sp_u12b3e_run3_execution_freeze.json':'4e79f8f8bb267f4bf8be463bd2cee6b7d25914218a0c36936543e96d52f45ce6'}
 for n,v in expected.items():assert h[n]==v,'HISTORICAL_MANIFEST_DRIFT'
 for n in ('outputs/sp_u12b3c_failed_run1_archive.json','outputs/sp_u12b3c_failed_run1_archive_supplement.json','outputs/sp_u12b3e/run2_archive_manifest.json'):
  for p,r in read(ROOT/n)['files'].items():assert sha(ROOT/p)==sha(ROOT/r['archive_path'])==r['sha256'],'FAILED_ARCHIVE_DRIFT'
 return dict(PASS=True,Run1_unchanged=True,Run2_unchanged=True,Run3_B3F_unchanged=True,B3E_freeze_unchanged=True,hashes=expected,scientific_use='PROHIBITED',outcomes_viewed=False)
def run(mode):
 OUT.mkdir(exist_ok=True);base=OUT/'preservation_before.json';assert mode!='before' or not base.exists()
 pp=paths();h={};last=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for p,v in zip(pp,pool.map(sha,pp)):
   h[p.relative_to(ROOT).as_posix()]=v
   if time.monotonic()-last>25:print(json.dumps(dict(stage='opaque_preservation',files=len(h))),flush=True);last=time.monotonic()
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();b2=b2check(h);hist=history(h)
 if mode=='before':
  archives={}
  for n in h:
   if n in ALLOWED or Path(n).name.startswith('sp_u12b3f') or n.startswith(('outputs/sp_u12b3f','docs/sp_u12b3f','configs/sp_u12b3f')) or ('__pycache__' in n and Path(n).name.startswith(('sp_u12b3e_io.','run_sp_u12b3f.'))):
    dest=OUT/'original_bytes'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest);assert sha(dest)==h[n];archives[n]=dict(sha256=h[n],archive_path=dest.relative_to(ROOT).as_posix())
  write(OUT/'original_bytes_manifest.json',dict(files=archives,scientific_use='PROHIBITED',outcomes_viewed=False))
  write(base,dict(hashes=h,git_HEAD=head,B2=b2,history=hist))
  (OUT/'git_status_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True,encoding='utf8'),encoding='utf8')
  print(json.dumps(dict(stage='preservation_before',PASS=True,protected_files=len(h),B2_files=35031,archived_files=len(archives))),flush=True);return
 prior=read(base);changed=[n for n,v in prior['hashes'].items() if h.get(n)!=v];added=[n for n in h if n not in prior['hashes']]
 cache=lambda n:'__pycache__' in n and Path(n).name.startswith(('sp_u12b3e_io.','run_sp_u12b3f.'))
 bad=[n for n in changed if n not in ALLOWED and not cache(n)];badadded=[n for n in added if not cache(n)]
 for n,r in read(OUT/'original_bytes_manifest.json')['files'].items():assert sha(ROOT/r['archive_path'])==r['sha256'],'ORIGINAL_ARCHIVE_DRIFT'
 audit=dict(PASS=not bad and not badadded and head==prior['git_HEAD'],protected_files=len(prior['hashes']),authorized_technical_changes=[n for n in changed if n in ALLOWED],derived_live_bytecode_changes=[n for n in changed if cache(n)],unauthorized_changes=bad,unauthorized_added=badadded,production_semantic_drift=0 if not bad else None,scientific_artifact_drift=sum(n.startswith('outputs/') for n in bad),manuscript_drift=sum(n.startswith('paper/') for n in changed),git_HEAD_unchanged=head==prior['git_HEAD'],historical_original_bytes_archived=True,outcomes_viewed=False)
 write(ROOT/'outputs/sp_u12b3g_preservation.json',audit);write(ROOT/'outputs/sp_u12b3g_b2_seal_audit.json',dict(PASS=b2==prior['B2'],before=prior['B2'],after=b2));hist['PASS']=hist==prior['history'];write(ROOT/'outputs/sp_u12b3g_history_preservation.json',hist)
 (OUT/'git_status_after.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True,encoding='utf8'),encoding='utf8')
 r=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf8');write(OUT/'git_diff_check.json',dict(exit_code=r.returncode,output=r.stdout+r.stderr));assert r.returncode==0,'GIT_DIFF_CHECK'
 print(json.dumps(dict(stage='preservation_after',PASS=audit['PASS'],protected_files=audit['protected_files'],unauthorized_changes=len(bad),B2_PASS=b2==prior['B2'],history_PASS=hist['PASS'])),flush=True)
 assert audit['PASS'] and b2==prior['B2'] and hist['PASS']
if __name__=='__main__':run(sys.argv[1])
