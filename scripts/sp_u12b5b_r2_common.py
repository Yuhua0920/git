"""R2 technical-only scope, byte preservation, and current-host safety."""
from pathlib import Path
import sys, os, json, csv, time, subprocess, concurrent.futures
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import sp_u12b4a_preserve as p
from scripts.sp_u12b5b_r1_common import host_snapshot, process_handle, process_memory, close_handle, threshold_reason
OUT = ROOT/'outputs/sp_u12b5b_r2'
CORE = ROOT/'reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py'
ADAPTER = ROOT/'src/baselines/mfa_reference_adapter.py'
ARCHIVES = {
 'outputs/sp_u12b5a_mfa_fidelity_manifest.json': ('56d0f541b328559ed83920ef45fb30fab2a7a29a3e37e3d43d825522a1ec4468','artifacts'),
 'outputs/sp_u12b5b_manifest.json': ('81e33520448d6b3dd9276208e59b3dd68367ddf1e1e31add4216a043bd9b342a','files'),
 'outputs/sp_u12b5b_r1_manifest.json': ('243f60fb07cd35186d5680f80853e19850329c9fd0821e5d36c0026af2504029','files')}
ALLOW = {ROOT/n for n in ARCHIVES} | {p.B2M,p.B3M,ROOT/'outputs/sp_u12b5a/historical_reference.json',
 ROOT/'outputs/sp_u12b5b_candidate_seed_registry.json', ROOT/'outputs/sp_u12b5b/input_manifest.json',
 ROOT/'outputs/sp_u12b5b/STOP.json', ROOT/'outputs/sp_u12b5b/monitor_K1_r4_train.json',
 ROOT/'outputs/sp_u12b5b_r1_resource_probe_results.json', ROOT/'outputs/sp_u12b5b_r1_host_capacity.json',
 ROOT/'outputs/sp_u12b5b_r1_allocation_model.json'}
ACCESS=[]; DENIED=[]; _installed=False
read=p.read; write=p.write; sha=p.sha; tree=p.tree
LIMITS={'max_process_bytes':6*2**30,'min_system_available_physical_bytes':2*2**30}
def owned(path):
 return any(s.startswith(('sp_u12b5b_r2','run_sp_u12b5b_r2','test_sp_u12b5b_r2')) for s in path.relative_to(ROOT).parts)
def firewall():
 global _installed
 if _installed:return
 def hook(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   path=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
   writing=isinstance(mode,str) and any(v in mode for v in 'wax+') or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
   if not path.is_relative_to(ROOT):return
   rel=path.relative_to(ROOT)
   if writing and rel.parts[0] in ('src','scripts','tests','configs','docs','paper','outputs','reference') and not owned(path) and '__pycache__' not in rel.parts:raise PermissionError('R2_HISTORICAL_WRITE_FORBIDDEN')
   if rel.parts[0]=='outputs' and not writing and not p._opaque:
    if not (owned(path) or path in ALLOW):
     DENIED.append(rel.as_posix());raise PermissionError('R2_SCIENTIFIC_VALUE_ACCESS_FORBIDDEN')
    ACCESS.append(rel.as_posix())
  if event in ('os.remove','os.rename','os.rmdir','os.mkdir','os.utime'):
   for raw in args[:2] if event=='os.rename' else args[:1]:
    if isinstance(raw,(str,bytes,os.PathLike)):
     path=Path(os.fsdecode(raw)).resolve()
     if path.is_relative_to(ROOT) and path.relative_to(ROOT).parts[0] in ('src','scripts','tests','configs','docs','paper','outputs','reference') and not owned(path) and '__pycache__' not in path.parts:raise PermissionError('R2_HISTORICAL_MUTATION_FORBIDDEN')
 sys.addaudithook(hook);_installed=True
def csvwrite(path,rows):
 rows=list(rows);assert rows
 with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def check_frozen():
 result={}
 for n,(expected,field) in ARCHIVES.items():
  assert sha(ROOT/n)==expected,n
  obj=read(ROOT/n)
  for leaf,h in obj[field].items():assert sha(ROOT/leaf)==h,'ARCHIVE_DRIFT:'+leaf
  result[n]={'sha256':expected,'leaves_verified':len(obj[field])}
 result['author_sha256']=sha(CORE);result['adapter_sha256']=sha(ADAPTER)
 assert result['adapter_sha256']=='f92729e5f4b623775a1ffdd4013fe7f85d8ca86913d05383fe4274fb4c38b0cc'
 return result
def preserve(stage):
 firewall();OUT.mkdir(parents=True,exist_ok=True);base=OUT/'preservation_before.json'
 assert stage!='before' or not base.exists()
 check_frozen()
 paths=sorted(v for d in ('src','scripts','tests','configs','docs','paper','outputs','reference') for v in (ROOT/d).rglob('*') if v.is_file() and not owned(v) and not(set(v.parts)&{'__pycache__','.pytest_cache','pytest_cache','pytest_tmp'}))
 hashes={};last=time.monotonic();old=p._opaque;p._opaque=True
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   for v,h in zip(paths,pool.map(sha,paths)):
    hashes[v.relative_to(ROOT).as_posix()]=h
    if time.monotonic()-last>25:print(json.dumps({'stage':'R2_opaque_integrity','files':len(hashes)}),flush=True);last=time.monotonic()
 finally:p._opaque=old
 seals=p.seals(hashes);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True,encoding='utf8');(OUT/f'git_status_{stage}.txt').write_text(status,encoding='utf8')
 diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf8');write(OUT/f'git_diff_check_{stage}.json',{'exit_code':diff.returncode,'stdout':diff.stdout,'stderr':diff.stderr});assert diff.returncode==0
 if stage=='before':write(base,{'hashes':hashes,'seals':seals,'Git_HEAD':head,'frozen':check_frozen()})
 else:
  prev=read(base);changed=[n for n,h in prev['hashes'].items() if hashes.get(n)!=h];added=[n for n in hashes if n not in prev['hashes']]
  result={'PASS':not changed and not added and head==prev['Git_HEAD'],'protected_files':len(prev['hashes']),'changed':changed,'unauthorized_added':added,'production_drift':sum(n.startswith(('src/','scripts/')) for n in changed),'scientific_artifact_drift':sum(n.startswith('outputs/') for n in changed),'manuscript_drift':sum(n.startswith('paper/') for n in changed),'Git_HEAD':head,'Git_HEAD_unchanged':head==prev['Git_HEAD'],'frozen':check_frozen()}
  write(ROOT/'outputs/sp_u12b5b_r2_preservation.json',result)
  write(ROOT/'outputs/sp_u12b5b_r2_b2_b3_seal_audit.json',{'PASS':seals==prev['seals'],'before':prev['seals'],'after':seals,'scientific_value_reads':0,'opaque_hashing_only':True})
  assert result['PASS'] and seals==prev['seals']
 print(json.dumps({'stage':'R2_preservation_'+stage,'PASS':True,'protected_files':len(hashes)}),flush=True)
if __name__=='__main__':preserve(sys.argv[1])
