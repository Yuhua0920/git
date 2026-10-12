"""Resource forensics only: immutable scientific archives and byte-only seals."""
from pathlib import Path
import sys,os,json,csv,time,datetime,subprocess,concurrent.futures,ctypes
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts import sp_u12b4a_preserve as p
from scripts.sp_u12b5b_common import process_handle,process_memory,system_memory,close_handle
OUT=ROOT/'outputs/sp_u12b5b_r1'
FID=ROOT/'outputs/sp_u12b5a_mfa_fidelity_manifest.json'
FAIL=ROOT/'outputs/sp_u12b5b_manifest.json'
FID_SHA='56d0f541b328559ed83920ef45fb30fab2a7a29a3e37e3d43d825522a1ec4468'
FAIL_SHA='81e33520448d6b3dd9276208e59b3dd68367ddf1e1e31add4216a043bd9b342a'
CORE=ROOT/'reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py'
ADAPTER=ROOT/'src/baselines/mfa_reference_adapter.py'
CONFIG=ROOT/'outputs/sp_u12b5b_r1_resource_probe_config.json'
read=p.read;write=p.write;sha=p.sha;tree=p.tree
ACCESS=[];DENIED=[];_installed=False;HEADER_ONLY=False
ALLOW={FID,FAIL,p.B2M,p.B3M,ROOT/'outputs/sp_u12b5a/historical_reference.json',
 ROOT/'outputs/sp_u12b5b/STOP.json',ROOT/'outputs/sp_u12b5b/monitor_K1_r4_train.json',
 ROOT/'outputs/sp_u12b5b/resource_K1_r4_train.jsonl',ROOT/'outputs/sp_u12b5b/worker_K1_r4_train.log',
 ROOT/'outputs/sp_u12b5b/worker_K1_r4_train_PID.json',ROOT/'outputs/sp_u12b5b/worker_K1_r4_train_MONITOR_READY.json',
 ROOT/'outputs/sp_u12b5b_candidates/K1_r4/fit_entry.json',ROOT/'outputs/sp_u12b5b_candidate_seed_registry.json',
 ROOT/'outputs/sp_u12b5b/input_manifest.json'}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def owned(path):return any(s.startswith(('sp_u12b5b_r1','run_sp_u12b5b_r1','test_sp_u12b5b_r1')) for s in path.relative_to(ROOT).parts)
def firewall():
 global _installed
 if _installed:return
 def hook(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   path=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
   writing=isinstance(mode,str) and any(v in mode for v in 'wax+') or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
   if not path.is_relative_to(ROOT):return
   rel=path.relative_to(ROOT)
   if writing and rel.parts[0] in ('src','scripts','tests','configs','docs','paper','outputs','reference') and not owned(path) and '__pycache__' not in rel.parts:raise PermissionError('R1_HISTORICAL_WRITE_FORBIDDEN')
   if rel.parts[0]=='outputs' and not writing and not p._opaque:
    header=HEADER_ONLY and path==ROOT/'outputs/sp_u12b5b/selection_train.npy'
    if not (owned(path) or path in ALLOW or header):
     DENIED.append(rel.as_posix());raise PermissionError('R1_SCIENTIFIC_VALUE_ACCESS_FORBIDDEN')
    if len(ACCESS)<20000:ACCESS.append(dict(path=rel.as_posix(),purpose='NPY_HEADER_ONLY' if header else 'resource/metadata'))
  if event in ('os.remove','os.rename','os.rmdir','os.mkdir','os.utime'):
   for raw in args[:2] if event=='os.rename' else args[:1]:
    if isinstance(raw,(str,bytes,os.PathLike)):
     path=Path(os.fsdecode(raw)).resolve()
     if path.is_relative_to(ROOT) and path.relative_to(ROOT).parts[0] in ('src','scripts','tests','configs','docs','paper','outputs','reference') and not owned(path) and '__pycache__' not in path.parts:raise PermissionError('R1_HISTORICAL_MUTATION_FORBIDDEN')
 sys.addaudithook(hook);_installed=True
def csvwrite(path,rows):
 rows=list(rows);assert rows
 with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def check_frozen():
 assert sha(FID)==FID_SHA and sha(FAIL)==FAIL_SHA
 a=read(FID);b=read(FAIL)
 for manifest,field in ((a,'artifacts'),(b,'files')):
  for n,h in manifest[field].items():assert sha(ROOT/n)==h,'IMMUTABLE_ARCHIVE_DRIFT:'+n
 return dict(B5A_manifest_sha256=FID_SHA,B5B_failure_manifest_sha256=FAIL_SHA,reference_tree_sha256=a['reference']['combined_tree_sha256'],adapter_sha256=sha(ADAPTER),B5A_artifact_count=len(a['artifacts']),B5B_failure_artifact_count=len(b['files']))
def preserve(stage):
 firewall();OUT.mkdir(parents=True,exist_ok=True);base=OUT/'preservation_before.json'
 assert stage!='before' or not base.exists();check_frozen()
 paths=sorted(v for d in ('src','scripts','tests','configs','docs','paper','outputs','reference') for v in (ROOT/d).rglob('*') if v.is_file() and not owned(v) and not(set(v.parts)&{'__pycache__','.pytest_cache','pytest_cache','pytest_tmp'}))
 hashes={};last=time.monotonic();old=p._opaque;p._opaque=True
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   for v,h in zip(paths,pool.map(sha,paths)):
    hashes[v.relative_to(ROOT).as_posix()]=h
    if time.monotonic()-last>25:print(json.dumps(dict(stage='R1_integrity_hashing',files=len(hashes))),flush=True);last=time.monotonic()
 finally:p._opaque=old
 seals=p.seals(hashes);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True,encoding='utf8');(OUT/f'git_status_{stage}.txt').write_text(status,encoding='utf8')
 check=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf8');write(OUT/f'git_diff_check_{stage}.json',dict(exit_code=check.returncode,stdout=check.stdout,stderr=check.stderr));assert check.returncode==0
 if stage=='before':write(base,dict(hashes=hashes,seals=seals,Git_HEAD=head,created_UTC=utc(),frozen=check_frozen()))
 else:
  previous=read(base);changed=[n for n,h in previous['hashes'].items() if hashes.get(n)!=h];added=[n for n in hashes if n not in previous['hashes']]
  result=dict(PASS=not changed and not added and head==previous['Git_HEAD'],protected_files=len(previous['hashes']),changed=changed,unauthorized_added=added,production_semantic_drift=sum(n.startswith(('src/','scripts/')) for n in changed),scientific_artifact_drift=sum(n.startswith('outputs/') for n in changed),manuscript_drift=sum(n.startswith('paper/') for n in changed),Git_HEAD=head,Git_HEAD_unchanged=head==previous['Git_HEAD'],frozen=check_frozen())
  write(ROOT/'outputs/sp_u12b5b_r1_preservation.json',result);write(ROOT/'outputs/sp_u12b5b_r1_b2_b3_seal_audit.json',dict(PASS=seals==previous['seals'],before=previous['seals'],after=seals,scientific_value_reads=0,opaque_hashing_only=True))
  assert result['PASS'] and seals==previous['seals']
 print(json.dumps(dict(stage='R1_preservation_'+stage,PASS=True,protected_files=len(hashes))),flush=True)
class Performance(ctypes.Structure):
 _fields_=[('cb',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[(n,ctypes.c_ulong) for n in ('HandleCount','ProcessCount','ThreadCount')]
def host_snapshot():
 x=Performance();x.cb=ctypes.sizeof(x);fn=ctypes.windll.psapi.GetPerformanceInfo;fn.argtypes=[ctypes.POINTER(Performance),ctypes.c_ulong];fn.restype=ctypes.c_int
 if not fn(ctypes.byref(x),x.cb):raise ctypes.WinError()
 result=dict(UTC=utc(),**system_memory(),commit_used_bytes=int(x.CommitTotal*x.PageSize),commit_limit_bytes=int(x.CommitLimit*x.PageSize),commit_peak_bytes=int(x.CommitPeak*x.PageSize),system_cache_bytes=int(x.SystemCache*x.PageSize),process_count=int(x.ProcessCount))
 result['OS_background_unavailable_physical_bytes']=result['total_physical_bytes']-result['available_physical_bytes']
 return result
def threshold_reason(process,host,limits):
 if process['peak_working_set_bytes']>=limits['max_process_bytes']:return 'PROCESS_RSS_CAP'
 if host['available_physical_bytes']<limits['min_system_available_physical_bytes']:return 'SYSTEM_PHYSICAL_HEADROOM'
 return None
if __name__=='__main__':preserve(sys.argv[1])
