"""Deployment-only logging, integrity and resource monitoring; no MFA math."""
from pathlib import Path
import os,sys,json,time,hashlib,importlib.util,contextlib
ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
MANIFEST=ROOT/'migration_manifest.json';RUNS=ROOT/'lab_runs'
_opaque=False

def utc():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    for attempt in range(101):
        try:os.replace(tmp,p);return
        except PermissionError as e:
            if getattr(e,'winerror',None)not in(5,32,33)or attempt==100:raise
            time.sleep(.01)

def sha(p):
    global _opaque
    previous=_opaque;_opaque=True
    try:
        h=hashlib.sha256()
        with Path(p).open('rb')as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    finally:_opaque=previous

def host_module():
    p=ROOT/'experiments/mfa_low_memory_prototype/portable/host_inventory.py'
    spec=importlib.util.spec_from_file_location('_frozen_host_probe',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def integrity():
    m=read(MANIFEST);bad=[]
    for name,v in m['files'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT)or not p.is_file()or p.stat().st_size!=v['bytes']or sha(p)!=v['sha256']:bad.append(name)
    forbidden=[]
    for name in ('outputs/sp_u12b2b_sealed','outputs/sp_u12b3g_next_sealed','.git','.env','.venv-cdl','.venv-gmm'):
        if(ROOT/name).exists():forbidden.append(name)
    return {'PASS':not bad and not forbidden,'payload_files_verified':len(m['files']),'bad':bad,'forbidden_present':forbidden,
        'manifest_sha256':sha(MANIFEST),'formal_training_authorized':False,'scientific_data_values_read':0}

def install_scope():
    # Hashing preparation bytes is permitted; using their numeric values is blocked.
    def hook(event,args):
        if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
            writing=isinstance(mode,str)and any(c in mode for c in 'wax+')or isinstance(flags,int)and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if p.is_relative_to(ROOT):
                if writing and not p.is_relative_to(RUNS):raise PermissionError('DEPLOYMENT_FROZEN_INPUT_WRITE_BLOCKED')
                if not writing and p.is_relative_to(ROOT/'outputs')and p.suffix in('.npy','.npz')and not _opaque:
                    raise PermissionError('PREPARATION_VALUE_READ_NOT_AUTHORIZED_IN_MIGRATION_OR_SMOKE')
                if not writing and any(p.is_relative_to(ROOT/n)for n in('outputs/sp_u12b2b_sealed','outputs/sp_u12b3g_next_sealed')):
                    raise PermissionError('SEALED_OUTCOME_ACCESS_FORBIDDEN')
        if event in('os.remove','os.rename','os.rmdir','os.mkdir','os.utime'):
            for raw in args[:2]if event=='os.rename'else args[:1]:
                if isinstance(raw,(str,bytes,os.PathLike)):
                    p=Path(os.fsdecode(raw)).resolve()
                    if p.is_relative_to(ROOT)and not p.is_relative_to(RUNS):raise PermissionError('DEPLOYMENT_FROZEN_MUTATION_BLOCKED')
    sys.addaudithook(hook)

def environment():
    import importlib.metadata as md,platform
    source=read(ROOT/'deployment/mfa_lab_v1/environment_source.json');actual={};bad=[]
    for row in source['wheels']:
        try:v=md.version(row['package'])
        except md.PackageNotFoundError:v='MISSING'
        actual[row['package']]=v
        if v!=row['version']:bad.append(row['package'])
    import numpy,scipy,sklearn
    from threadpoolctl import threadpool_info
    ok=sys.version_info[:3]==(3,13,15)and sys.maxsize>2**32 and not bad
    return {'PASS':ok,'Python':sys.version,'platform':platform.platform(),'packages':actual,'version_mismatch':bad,
        'threadpools':threadpool_info(),'CPU_only':True,'GPU_used':False,'formal_threads_unchanged':True}

def stop_process(pid):
    import ctypes
    k=ctypes.windll.kernel32;k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];k.OpenProcess.restype=ctypes.c_void_p
    h=k.OpenProcess(1,False,int(pid))
    if not h:return
    try:
        k.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint];k.TerminateProcess(h,77)
    finally:k.CloseHandle.argtypes=[ctypes.c_void_p];k.CloseHandle(h)

def optional(p):
    try:return read(p)
    except(FileNotFoundError,PermissionError):return None
