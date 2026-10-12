"""Read-only Windows inventory and kernel-owned process-tree containment."""
import ctypes as c,ctypes.wintypes as w,os,time,importlib.util,shutil,functools
from common import ROOT,GIB,protocol
K=c.WinDLL('kernel32',use_last_error=True)
P=c.WinDLL('psapi',use_last_error=True)

class BASIC(c.Structure):
    _fields_=[('process_time',c.c_longlong),('job_time',c.c_longlong),('flags',w.DWORD),
              ('min_ws',c.c_size_t),('max_ws',c.c_size_t),('active',w.DWORD),('affinity',c.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
class IO(c.Structure):_fields_=[(n,c.c_ulonglong)for n in('read_count','write_count','other_count','read_bytes','write_bytes','other_bytes')]
class EXT(c.Structure):
    _fields_=[('basic',BASIC),('io',IO),('process_memory',c.c_size_t),('job_memory',c.c_size_t),('peak_process_memory',c.c_size_t),('peak_job_memory',c.c_size_t)]
class MEM(c.Structure):
    _fields_=[('cb',w.DWORD),('page_faults',w.DWORD)]+[(n,c.c_size_t)for n in
        ('peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]
class IDS(c.Structure):_fields_=[('assigned',w.DWORD),('count',w.DWORD),('ids',c.c_size_t*4096)]
class FILETIME(c.Structure):_fields_=[('low',w.DWORD),('high',w.DWORD)]
def seconds(v):return((v.high<<32)|v.low)/1e7

K.CreateJobObjectW.argtypes=[c.c_void_p,w.LPCWSTR];K.CreateJobObjectW.restype=w.HANDLE
K.SetInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD];K.SetInformationJobObject.restype=w.BOOL
K.QueryInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD,c.c_void_p];K.QueryInformationJobObject.restype=w.BOOL
K.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];K.AssignProcessToJobObject.restype=w.BOOL
K.TerminateJobObject.argtypes=[w.HANDLE,w.UINT];K.TerminateJobObject.restype=w.BOOL
K.OpenJobObjectW.argtypes=[w.DWORD,w.BOOL,w.LPCWSTR];K.OpenJobObjectW.restype=w.HANDLE
K.CloseHandle.argtypes=[w.HANDLE];K.CloseHandle.restype=w.BOOL
K.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];K.OpenProcess.restype=w.HANDLE
K.GetExitCodeProcess.argtypes=[w.HANDLE,c.POINTER(w.DWORD)];K.GetExitCodeProcess.restype=w.BOOL
K.GetProcessTimes.argtypes=[w.HANDLE,c.POINTER(FILETIME),c.POINTER(FILETIME),c.POINTER(FILETIME),c.POINTER(FILETIME)];K.GetProcessTimes.restype=w.BOOL
P.GetProcessMemoryInfo.argtypes=[w.HANDLE,c.POINTER(MEM),w.DWORD];P.GetProcessMemoryInfo.restype=w.BOOL

@functools.lru_cache(maxsize=1)
def frozen_probe():
    path=ROOT/'experiments/mfa_low_memory_prototype/portable/host_inventory.py'
    spec=importlib.util.spec_from_file_location('_r5_frozen_host_probe',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def hardware():
    probe=frozen_probe();v=probe.inventory(False)
    v.update(scope='ACTUAL_EXECUTION_HOST_DETECTION_ONLY',disk_free_bytes=shutil.disk_usage(ROOT).free)
    return v
def capacity(v,full=True):
    cfg=protocol();memory=v['memory'];top=v['topology'];reasons=[]
    if 'i9-10900KF'not in v['CPU_exact_model']:reasons.append('CPU_MODEL_MISMATCH')
    if top['physical_cores']!=10 or top['logical_threads']!=20:reasons.append('CPU_TOPOLOGY_MISMATCH')
    if memory['total_physical_bytes']<120*GIB:reasons.append('NOT_128GB_CLASS_HOST')
    if full:
        if memory['available_physical_bytes']<cfg['launch_minimum']['available_physical_bytes']:reasons.append('AVAILABLE_RAM_BELOW_80_GiB')
        if memory['commit_available_bytes']<cfg['launch_minimum']['remaining_commit_bytes']:reasons.append('REMAINING_COMMIT_BELOW_80_GiB')
    if v['disk_free_bytes']<cfg['launch_minimum']['disk_free_bytes']:reasons.append('DISK_HEADROOM_FAILED')
    return {'PASS':not reasons,'reasons':reasons,'hardware':v,'thresholds':cfg['launch_minimum']}

class Job:
    def __init__(self,name,private_limit):
        self.name=name;self.handle=K.CreateJobObjectW(None,name);self.processes={};self.cpu_seen={}
        if not self.handle:raise c.WinError(c.get_last_error())
        limits=EXT();limits.basic.flags=0x2000|0x100|0x200 # kill-on-close; process/job commit limits
        limits.process_memory=private_limit;limits.job_memory=private_limit
        if not K.SetInformationJobObject(self.handle,9,c.byref(limits),c.sizeof(limits)):
            error=c.get_last_error();self.close();raise c.WinError(error)
    def assign(self,pid):
        handle=K.OpenProcess(0x100|0x1|0x400|0x10|0x100000,False,int(pid))
        if not handle:raise c.WinError(c.get_last_error())
        if not K.AssignProcessToJobObject(self.handle,handle):
            error=c.get_last_error();K.CloseHandle(handle);raise c.WinError(error)
        self.processes[int(pid)]=handle
    def members(self):
        value=IDS()
        if not K.QueryInformationJobObject(self.handle,3,c.byref(value),c.sizeof(value),None):raise c.WinError(c.get_last_error())
        if value.assigned>4096:raise RuntimeError('UNBOUNDED_PROCESS_TREE_STOP')
        return [int(value.ids[i])for i in range(value.count)]
    def sample(self):
        answer=[]
        for pid in self.members():
            if pid not in self.processes:
                handle=K.OpenProcess(0x400|0x10|0x100000,False,pid)
                if not handle:raise c.WinError(c.get_last_error())
                self.processes[pid]=handle
            handle=self.processes[pid];exit_code=w.DWORD()
            if not K.GetExitCodeProcess(handle,c.byref(exit_code)):raise c.WinError(c.get_last_error())
            if exit_code.value!=259:continue
            memory=MEM();memory.cb=c.sizeof(memory)
            if not P.GetProcessMemoryInfo(handle,c.byref(memory),memory.cb):
                if K.GetExitCodeProcess(handle,c.byref(exit_code))and exit_code.value!=259:continue
                raise c.WinError(c.get_last_error())
            created,ended,kernel,user=FILETIME(),FILETIME(),FILETIME(),FILETIME()
            if not K.GetProcessTimes(handle,c.byref(created),c.byref(ended),c.byref(kernel),c.byref(user)):raise c.WinError(c.get_last_error())
            answer.append({'PID':pid,'creation_time_100ns':(created.high<<32)|created.low,
                'RSS_bytes':int(memory.ws),'peak_RSS_bytes':int(memory.peak_ws),'private_commit_bytes':int(memory.private),
                'native_peak_commit_bytes':int(memory.peak_pagefile),'CPU_seconds':seconds(kernel)+seconds(user)})
            self.cpu_seen[pid]=answer[-1]['CPU_seconds']
        return {'members':answer,'RSS_bytes':sum(v['RSS_bytes']for v in answer),
                'private_commit_bytes':sum(v['private_commit_bytes']for v in answer),
                'CPU_seconds':sum(self.cpu_seen.values())}
    def kill(self):
        if self.handle and not K.TerminateJobObject(self.handle,77):raise c.WinError(c.get_last_error())
    def close(self):
        if getattr(self,'handle',None):
            K.CloseHandle(self.handle);self.handle=None
        for handle in self.processes.values():K.CloseHandle(handle)
        self.processes.clear()

def stop_named_job(name):
    handle=K.OpenJobObjectW(0x8,False,name)
    if not handle:return False
    try:
        if not K.TerminateJobObject(handle,77):raise c.WinError(c.get_last_error())
        return True
    finally:K.CloseHandle(handle)

def exceeded(tree,system,limits):
    if tree['RSS_bytes']>limits['RSS_bytes']:return 'PROCESS_TREE_RSS_HARD_LIMIT'
    if tree['private_commit_bytes']>limits['private_commit_bytes']:return 'PROCESS_TREE_PRIVATE_COMMIT_HARD_LIMIT'
    if system['available_physical_bytes']<limits['min_available_physical_bytes']:return 'SYSTEM_RAM_HEADROOM_BELOW_32_GiB'
    if system['commit_available_bytes']<limits['min_remaining_commit_bytes']:return 'SYSTEM_COMMIT_HEADROOM_BELOW_32_GiB'
    if any(v['peak_RSS_bytes']>limits['RSS_bytes']for v in tree['members']):return 'NATIVE_PROCESS_PEAK_RSS_EXCEEDED_LIMIT'
    if any(v['native_peak_commit_bytes']>limits['private_commit_bytes']for v in tree['members']):return 'NATIVE_PROCESS_PEAK_COMMIT_EXCEEDED_LIMIT'
    return None
