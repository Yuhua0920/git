"""Full opaque protection of all historical artifacts; no outcome decoding."""
from pathlib import Path
import sys,json,time,hashlib,concurrent.futures,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u12b3g_preserve import b2check,history
OUT=ROOT/'outputs/sp_u12b3h';SCI=ROOT/'outputs/sp_u12b3g_next_sealed';CONTROL=ROOT/'outputs/sp_u12b3g/next_control'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(8*1024*1024),b''):h.update(v)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,o):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n',encoding='utf8')
def owned(p):return any(v.startswith(('sp_u12b3h','test_sp_u12b3h')) for v in p.parts)
def history_cache(p):return any(v.startswith(('sp_u12b3','run_sp_u12b3','test_sp_u12b3')) for v in p.parts)
def paths():
 skip={'__pycache__','.pytest_cache','pytest_cache','pytest_tmp'}
 return sorted(p for d in ('src','scripts','tests','configs','docs','paper','outputs') for p in (ROOT/d).rglob('*') if p.is_file() and not owned(p.relative_to(ROOT)) and not p.is_relative_to(SCI) and (history_cache(p.relative_to(ROOT)) or not(set(p.parts)&skip)))
def permitted_addition(n):
 p=ROOT/n
 if p.is_relative_to(CONTROL):return True
 names={'integrity_audit','both_replay_audit','bootstrap_audit','manifest','resource_log'}
 return p.parent==ROOT/'outputs' and p.name in {'sp_u12b3g_next_'+x+'.json' for x in names}
def run(mode):
 OUT.mkdir(exist_ok=True);base=OUT/'preservation_before.json'
 assert mode!='before' or not base.exists(),'BASELINE_EXISTS'
 pp=paths();h={};last=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for p,v in zip(pp,pool.map(sha,pp)):
   h[p.relative_to(ROOT).as_posix()]=v
   if time.monotonic()-last>25:print(json.dumps(dict(stage='opaque_preservation',files=len(h))),flush=True);last=time.monotonic()
 b2=b2check(h);hist=history(h);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 assert h['outputs/sp_u12b3g_next_execution_freeze.json']=='ba528450bc8c822759b025e71cd81fc334368feb2526307bf51b53e8f9bb4845'
 if mode=='before':
  assert not any(p.is_file() for p in SCI.rglob('*')),'FRESH_SCIENTIFIC_ROOT_REQUIRED'
  assert not CONTROL.exists() or not any(CONTROL.rglob('*')),'FRESH_CONTROL_REQUIRED'
  write(base,dict(hashes=h,B2=b2,history=hist,git_HEAD=head,scientific_files_before=0))
  (OUT/'git_status_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True,encoding='utf8'),encoding='utf8')
  print(json.dumps(dict(stage='preservation_before',PASS=True,protected_files=len(h),B2_files=35031,scientific_files_before=0)),flush=True);return
 old=read(base);changed=[n for n,v in old['hashes'].items() if h.get(n)!=v];added=[n for n in h if n not in old['hashes']];bad=[n for n in added if not permitted_addition(n)]
 o=dict(PASS=not changed and not bad and head==old['git_HEAD'],protected_files=len(old['hashes']),changed=changed,unauthorized_added=bad,authorized_new_runtime_metadata=len(added)-len(bad),production_semantic_drift=sum(n.startswith(('src/','scripts/')) for n in changed),scientific_artifact_drift=sum(n.startswith('outputs/') for n in changed),manuscript_drift=sum(n.startswith('paper/') for n in changed),git_HEAD_unchanged=head==old['git_HEAD'],outcomes_viewed=False)
 write(ROOT/'outputs/sp_u12b3h_preservation.json',o);write(ROOT/'outputs/sp_u12b3h_b2_seal_audit.json',dict(PASS=b2==old['B2'],before=old['B2'],after=b2));hist['PASS']=hist==old['history'];write(ROOT/'outputs/sp_u12b3h_history_preservation.json',hist)
 (OUT/'git_status_after.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True,encoding='utf8'),encoding='utf8')
 diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf8');write(OUT/'git_diff_check.json',dict(exit_code=diff.returncode,output=diff.stdout+diff.stderr))
 print(json.dumps(dict(stage='preservation_after',PASS=o['PASS'],protected_files=o['protected_files'],changed_count=len(changed),unauthorized_added_count=len(bad),B2_PASS=b2==old['B2'],history_PASS=hist['PASS'])),flush=True)
 assert o['PASS'] and b2==old['B2'] and hist['PASS'] and diff.returncode==0,'PRESERVATION_FAILURE'
if __name__=='__main__':run(sys.argv[1])
