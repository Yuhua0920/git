"""Opaque preservation checks; no B2/B3 scientific field decoding."""
from pathlib import Path
import sys,json,hashlib,time,concurrent.futures,subprocess,os,contextlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/sp_u12b4a'
B2=ROOT/'outputs/sp_u12b2b_sealed';B3=ROOT/'outputs/sp_u12b3g_next_sealed'
B2M=ROOT/'outputs/sp_u12b2b_sealed_manifest.json';B3M=ROOT/'outputs/sp_u12b3h_manifest.json'
B2SHA='54e5346a943e35b66e4a4f51a267b4db765430da1f27c6f0bc1bdbc95d1aa67e'
B3SHA='12f9406a7b9d88b7a8437d04c082eb68aa4a79b5b4880b6108acd1be6f200c65'
_opaque=False;_installed=False
def owned(p):return any(v.startswith(('sp_u12b4','run_sp_u12b4','test_sp_u12b4')) for v in p.parts)
def write(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 global _opaque
 old=_opaque;_opaque=True
 try:
  h=hashlib.sha256()
  with Path(p).open('rb') as f:
   for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
  return h.hexdigest()
 finally:_opaque=old
def tree(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def firewall():
 global _installed
 if _installed:return
 outputs=ROOT/'outputs';allowed={B2M,B3M,ROOT/'outputs/sp_u12b2a/operators.npz',ROOT/'outputs/sp_u12b2a_seed_registry.json',ROOT/'outputs/sp_u12b3b/bootstrap_seed_registry.json'}
 def hook(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
   writing=isinstance(mode,str) and any(v in mode for v in 'wax+') or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
   if not p.is_relative_to(ROOT):return
   if writing and any(p.is_relative_to(ROOT/d) for d in ('src','scripts','tests','configs','docs','paper','outputs')) and not owned(p.relative_to(ROOT)) and '__pycache__' not in p.parts:raise PermissionError('B4_HISTORICAL_WRITE_FORBIDDEN')
   if p.is_relative_to(outputs) and not writing and not _opaque and p not in allowed and not owned(p.relative_to(ROOT)):raise PermissionError('B4_SCIENTIFIC_OUTCOME_FIREWALL')
  if event in ('os.remove','os.rename','os.rmdir','os.mkdir','os.utime'):
   for raw in args[:2] if event=='os.rename' else args[:1]:
    if isinstance(raw,(str,bytes,os.PathLike)):
     p=Path(os.fsdecode(raw)).resolve()
     if p.is_relative_to(B2) or p.is_relative_to(B3):raise PermissionError('B4_SEALED_MUTATION_FORBIDDEN')
 sys.addaudithook(hook);_installed=True
def paths():
 return sorted(p for d in ('src','scripts','tests','configs','docs','paper','outputs') for p in (ROOT/d).rglob('*') if p.is_file() and not owned(p.relative_to(ROOT)) and not(set(p.parts)&{'__pycache__','.pytest_cache','pytest_cache','pytest_tmp'}))
def seals(h):
 assert sha(B2M)==B2SHA and sha(B3M)==B3SHA,'AUTHORITATIVE_MANIFEST_DRIFT'
 a=read(B2M);b=read(B3M)
 assert a['status']=='SEALED_COMPLETE_UNVIEWED' and b['status']=='B3_SEALED_FORMAL_COMPLETE_UNVIEWED'
 assert not a['outcomes_viewed'] and not b['outcomes_viewed'] and b['scientific_use']=='AUTHORITATIVE_B3_DATASET'
 answer={}
 for name,root,m in [('B2',B2,a),('B3',B3,b)]:
  actual={n:r for n,r in m['files'].items() if h.get((root/ n).relative_to(ROOT).as_posix())==r['sha256'] and (root/n).stat().st_size==r['bytes']}
  assert set(actual)==set(m['files']),name+'_SEALED_LEAF_DRIFT'
  disk={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
  assert disk==set(m['files']),name+'_SEALED_POPULATION_DRIFT'
  answer[name]=dict(PASS=True,files=len(actual),manifest_sha256=sha(B2M if name=='B2' else B3M),raw_hash_tree=m.get('raw_data_hash_tree'),all_artifact_hash_tree=m['all_artifact_hash_tree'],all_leaf_hashes_PASS=True,outcomes_viewed=False)
 return answer
def run(mode):
 OUT.mkdir(exist_ok=True);baseline=OUT/'preservation_before.json'
 assert mode!='before' or not baseline.exists(),'BASELINE_EXISTS'
 pp=paths();h={};last=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for p,v in zip(pp,pool.map(sha,pp)):
   h[p.relative_to(ROOT).as_posix()]=v
   if time.monotonic()-last>25:print(json.dumps(dict(stage='opaque_preservation',files=len(h))),flush=True);last=time.monotonic()
 seal=seals(h);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True,encoding='utf8')
 (OUT/('git_status_'+mode+'.txt')).write_text(status,encoding='utf8')
 if mode=='before':
  write(baseline,dict(hashes=h,seals=seal,git_HEAD=head))
  print(json.dumps(dict(stage='B4A_preservation_before',PASS=True,protected_files=len(h),B2_files=seal['B2']['files'],B3_files=seal['B3']['files'])),flush=True);return
 old=read(baseline);changed=[n for n,v in old['hashes'].items() if h.get(n)!=v];added=[n for n in h if n not in old['hashes']]
 o=dict(PASS=not changed and not added and head==old['git_HEAD'],protected_files=len(old['hashes']),changed=changed,unauthorized_added=added,production_semantic_drift=sum(n.startswith(('src/','scripts/')) for n in changed),scientific_artifact_drift=sum(n.startswith('outputs/') for n in changed),manuscript_drift=sum(n.startswith('paper/') for n in changed),git_HEAD_unchanged=head==old['git_HEAD'],outcomes_viewed=False)
 write(ROOT/'outputs/sp_u12b4a_preservation.json',o);write(ROOT/'outputs/sp_u12b4a_b2_b3_seal_audit.json',dict(PASS=seal==old['seals'],before=old['seals'],after=seal,outcomes_viewed=False))
 r=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf8');write(OUT/'git_diff_check.json',dict(exit_code=r.returncode,output=r.stdout+r.stderr))
 assert o['PASS'] and seal==old['seals'] and r.returncode==0,'PRESERVATION_FAILURE'
 print(json.dumps(dict(stage='B4A_preservation_after',PASS=True,protected_files=len(h),changed=0,B2_PASS=True,B3_PASS=True)),flush=True)
if __name__=='__main__':run(sys.argv[1])
