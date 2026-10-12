"""B5B target-blind I/O, preservation and existing Windows resource contract."""
from pathlib import Path
import sys,os,json,csv,hashlib,time,datetime,subprocess,concurrent.futures,ctypes
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts import sp_u12b4a_preserve as p
OUT=ROOT/'outputs/sp_u12b5b';CONFIG=ROOT/'configs/sp_u12b5b_mfa_selection.yaml'
FID=ROOT/'outputs/sp_u12b5a_mfa_fidelity_manifest.json'
FID_SHA='56d0f541b328559ed83920ef45fb30fab2a7a29a3e37e3d43d825522a1ec4468'
PREPARATION={'sp_u4a_clean.npy','sp_u4a_clean_gate.json','sp_u4a_clean_hashes.csv',
 't15a_fix_replay_h_hashes.csv','t15c_cdl_channel_hashes.csv',
 't15a_cdl_manifest.csv','t15c_cdl_manifest.csv'}
_installed=False;ACCESS=[];DENIED=[]
read=p.read;write=p.write;sha=p.sha;tree=p.tree
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def owned(path):
 return any(x.startswith(('sp_u12b5b','run_sp_u12b5b','test_sp_u12b5b')) for x in path.relative_to(ROOT).parts)
def firewall():
 global _installed
 if _installed:return
 allowed={p.B2M,p.B3M,FID}
 def hook(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   path=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
   writing=isinstance(mode,str) and any(v in mode for v in 'wax+') or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
   if not path.is_relative_to(ROOT):return
   rel=path.relative_to(ROOT)
   if writing and rel.parts[0] in ('src','scripts','tests','configs','docs','paper','outputs','reference') and not owned(path) and '__pycache__' not in rel.parts:
    raise PermissionError('B5B_HISTORICAL_WRITE_FORBIDDEN')
   if rel.parts[0]=='outputs' and not writing:
    if p._opaque:return  # Integrity-only hashing, never decoded/used as scientific data.
    permitted=owned(path) or path in allowed or rel.parts[1].startswith('sp_u12b5a') or rel.as_posix()=='outputs/sp_u12b5a/historical_reference.json' or path.name in PREPARATION
    if not permitted:
     DENIED.append(str(rel));raise PermissionError('B5B_TARGET_INFORMATION_FIREWALL')
    if len(ACCESS)<20000:ACCESS.append(str(rel))
  if event in ('os.remove','os.rename','os.rmdir','os.mkdir','os.utime'):
   for raw in args[:2] if event=='os.rename' else args[:1]:
    if isinstance(raw,(str,bytes,os.PathLike)):
     path=Path(os.fsdecode(raw)).resolve()
     if path.is_relative_to(ROOT) and path.relative_to(ROOT).parts[0] in ('src','scripts','tests','configs','docs','paper','outputs','reference') and not owned(path) and '__pycache__' not in path.parts:raise PermissionError('B5B_HISTORICAL_MUTATION_FORBIDDEN')
 sys.addaudithook(hook);_installed=True
def csvwrite(path,rows):
 rows=list(rows);assert rows
 with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def csvread(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def check_fidelity():
 assert sha(FID)==FID_SHA,'B5A_FIDELITY_MANIFEST_DRIFT'
 m=read(FID)
 for n,h in m['artifacts'].items():assert sha(ROOT/n)==h,'B5A_ARTIFACT_DRIFT:'+n
 return m
def protected():
 return sorted(v for d in ('src','scripts','tests','configs','docs','paper','outputs','reference') for v in (ROOT/d).rglob('*') if v.is_file() and not owned(v) and not(set(v.parts)&{'__pycache__','.pytest_cache','pytest_cache','pytest_tmp'}))
def fullhash():
 result={};last=time.monotonic();old=p._opaque;p._opaque=True
 try:
  paths=protected()
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   for path,h in zip(paths,pool.map(sha,paths)):
    result[path.relative_to(ROOT).as_posix()]=h
    if time.monotonic()-last>25:print(json.dumps(dict(stage='B5B_opaque_preservation',files=len(result))),flush=True);last=time.monotonic()
 finally:p._opaque=old
 return result
def preserve(stage):
 firewall();OUT.mkdir(parents=True,exist_ok=True);before=OUT/'preservation_before.json'
 assert stage!='before' or not before.exists(),'B5B_BASELINE_EXISTS'
 fid=check_fidelity();hashes=fullhash();seals=p.seals(hashes)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 (OUT/f'git_status_{stage}.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True,encoding='utf8'),encoding='utf8')
 chk=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf8');write(OUT/f'git_diff_check_{stage}.json',dict(exit_code=chk.returncode,stdout=chk.stdout,stderr=chk.stderr));assert chk.returncode==0
 if stage=='before':write(before,dict(hashes=hashes,seals=seals,Git_HEAD=head,created_UTC=utc(),B5A_manifest_sha256=FID_SHA))
 else:
  original=read(before);changed=[n for n,h in original['hashes'].items() if hashes.get(n)!=h];added=[n for n in hashes if n not in original['hashes']]
  result=dict(PASS=not changed and not added and head==original['Git_HEAD'],protected_files=len(original['hashes']),changed=changed,unauthorized_added=added,production_semantic_drift=sum(n.startswith(('src/','scripts/')) for n in changed),scientific_artifact_drift=sum(n.startswith('outputs/') for n in changed),manuscript_drift=sum(n.startswith('paper/') for n in changed),Git_HEAD=head,Git_HEAD_unchanged=head==original['Git_HEAD'],B5A_reference_adapter_unchanged=True)
  write(ROOT/'outputs/sp_u12b5b_preservation.json',result)
  write(ROOT/'outputs/sp_u12b5b_b2_b3_seal_audit.json',dict(PASS=seals==original['seals'],before=original['seals'],after=seals,scientific_values_read=0,opaque_integrity_hashing_only=True))
  assert result['PASS'] and seals==original['seals']
 print(json.dumps(dict(stage='B5B_preservation_'+stage,PASS=True,protected_files=len(hashes))),flush=True)

class Counters(ctypes.Structure):
 _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
def process_handle(pid):
 fn=ctypes.windll.kernel32.OpenProcess;fn.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];fn.restype=ctypes.c_void_p
 handle=fn(0x410,False,int(pid));assert handle,'WORKER_PID_OPEN_FAILED';return handle
def process_memory(handle):
 v=Counters();v.cb=ctypes.sizeof(v);fn=ctypes.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong];fn.restype=ctypes.c_int
 if not fn(handle,ctypes.byref(v),v.cb):raise ctypes.WinError()
 return dict(working_set_bytes=int(v.WorkingSetSize),peak_working_set_bytes=int(v.PeakWorkingSetSize),private_bytes=int(v.PrivateUsage),peak_commit_bytes=int(v.PeakPagefileUsage))
def system_memory():
 class Status(ctypes.Structure):
  _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total_physical','available_physical','total_commit','available_commit','total_virtual','available_virtual','available_extended_virtual')]
 v=Status();v.length=ctypes.sizeof(v);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(v))
 return dict(total_physical_bytes=int(v.total_physical),available_physical_bytes=int(v.available_physical),available_commit_bytes=int(v.available_commit))
def close_handle(handle):
 fn=ctypes.windll.kernel32.CloseHandle;fn.argtypes=[ctypes.c_void_p];fn.restype=ctypes.c_int;fn(handle)
if __name__=='__main__':preserve(sys.argv[1])
