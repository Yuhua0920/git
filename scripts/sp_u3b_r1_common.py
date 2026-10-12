"""Read-only prior evidence, immutable new results and Windows memory metrics."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
sys.dont_write_bytecode = True
ACCESS = []
OLD = {'t15a_fix_replay_h_hashes.csv', 't15c_cdl_channel_hashes.csv',
       't15a_cdl_manifest.csv','t15c_cdl_manifest.csv'}

def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)): return
    p=Path(os.fsdecode(args[0])).resolve()
    try: rel=p.relative_to(OUT.resolve()).as_posix()
    except ValueError: return
    flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
    writing=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
    allowed=rel.split('/')[0].startswith('sp_u3b_r1_') or (not writing and
      (rel.split('/')[0].startswith(('sp_u3a_','sp_u3b_')) or rel in OLD))
    # Additional old files are accepted ONLY if listed in the U3A immutable input
    # seal: caller verify_prior hashes their bytes; no statistical values loaded.
    allowed=allowed or (not writing and rel in {
      't15a_cdl_statistics_500.npz','t15c_cdl_statistics_CDL-B-30.npz',
      't15c_cdl_statistics_CDL-C-120.npz','t15b_gmm_probe_dataset_manifest.json',
      't16c_heldout_manifest.csv','t16e_pilot_cv_preflight.json','t16e_frozen_operators.npz',
      't16f_preflight.json','t16f_frozen_candidates.npz','t16g_factor_preflight.json',
      't16g_frozen_module_factors.npz'})
    ACCESS.append(dict(path=rel,writing=writing,allowed=allowed))
    if not allowed: raise PermissionError('SP-U3B-R1 firewall:'+rel)
sys.addaudithook(guard)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):
    p=Path(p);assert not p.exists(),str(p)
    p.write_text(json.dumps(v,indent=2,allow_nan=False,ensure_ascii=False)+'\n',encoding='utf8')
def cfg(): return load(ROOT/'configs/sp_u3b_r1_protocol.json')
def path(name): return OUT/('sp_u3b_r1_'+name)
def git(*a): return subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()

def verify_prior():
    s=load(OUT/'sp_u3a_delivery_manifest.json')
    for group in ('artifact_sha256','execution_source_sha256','protected_sources'):
        for p,h in s[group].items(): assert sha(ROOT/p)==h,p
    for p,h in s['frozen_input_sha256'].items(): assert sha(OUT/p)==h,p
    t=load(OUT/'sp_u3b_delivery_manifest.json')
    for group in ('artifacts','sources'):
        for p,h in t[group].items(): assert sha(ROOT/p)==h,p
    assert git('rev-parse','HEAD')=='ca83982b1d0e4bdca0215a3ecbe3a7a8f8f8ec55'
    assert not git('diff','--name-only') and not git('diff','--cached','--name-only')
    return dict(passed=True,u3a_artifacts=len(s['artifact_sha256']),
      u3a_sources=len(s['execution_source_sha256']),protected_sources=len(s['protected_sources']),
      old_input_hashes=len(s['frozen_input_sha256']),u3b_artifacts=len(t['artifacts']),
      u3b_sources=len(t['sources']),production_bytes_unchanged=True)

class Counters(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[
      (n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize',
      'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
      'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
def open_process(pid=None):
    k=ctypes.windll.kernel32
    k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    k.OpenProcess.restype=ctypes.c_void_p
    k.GetCurrentProcess.restype=ctypes.c_void_p
    h=k.GetCurrentProcess() if pid is None else k.OpenProcess(0x410,False,pid)
    if not h: raise ctypes.WinError()
    return h
def memory(h=None):
    h=open_process() if h is None else h
    p=Counters();p.cb=ctypes.sizeof(p)
    fn=ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    fn.restype=ctypes.c_int
    if not fn(h,ctypes.byref(p),p.cb): raise ctypes.WinError()
    return dict(peak_working_set_bytes=int(p.PeakWorkingSetSize),
      working_set_bytes=int(p.WorkingSetSize),peak_commit_bytes=int(p.PeakPagefileUsage),
      commit_bytes=int(p.PagefileUsage),private_bytes=int(p.PrivateUsage))
def close_process(h):
    fn=ctypes.windll.kernel32.CloseHandle;fn.argtypes=[ctypes.c_void_p];fn.restype=ctypes.c_int
    assert fn(h)
def stop_reason(mem,seconds,limit=None):
    q=cfg() if limit is None else limit
    if max(mem['peak_working_set_bytes'],mem['peak_commit_bytes'])>=q['max_process_bytes']:
        return 'MEMORY_CAP'
    if seconds>=q['max_fit_seconds']: return 'FIT_TIME_CAP'
    return None

def frozen_sources():
    amended=path('execution_freeze_monitor_amendment.json')
    if not amended.exists():amended=path('execution_freeze_amendment.json')
    freeze=load(amended if amended.exists() else path('execution_freeze.json'))
    for p,h in freeze['hashes'].items(): assert sha(ROOT/p)==h,p
    return freeze
