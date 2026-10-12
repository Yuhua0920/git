"""Candidate I/O observability/isolation hardening; not a proven historical repair."""
from pathlib import Path
import os,sys,json,threading,uuid,traceback,datetime,re,ctypes
_worker_context=None
def guarded_worker(condition,trajectory_id):
 """Picklable technical exception envelope; scientific worker is untouched."""
 g,original_worker=_worker_context
 try:return original_worker(condition,trajectory_id)
 except Exception as exc:
  info=g['technical_exception_record'](exc);info['trajectory_key']=dict(condition=condition,trajectory_id=trajectory_id)
  try:g['write'](g['TECH']/f'worker_exception_{os.getpid()}_{uuid.uuid4().hex}.json',info)
  except OSError:pass
  raise
def install_io(g):
 global _worker_context
 root=g['ROOT'];tech=g['TECH'];tls=threading.local();original_read=g['read'];original_atomic=g['atomic'];original_write=g['write']
 real_b2=(root/'outputs/sp_u12b2b_sealed').resolve();failed=(root/'outputs/sp_u12b3b_sealed').resolve()
 def route(path):
  p=Path(path)
  if p.parent==root/'outputs' and p.name.startswith('sp_u12b3b_'):return g['TECH']/'reports'/p.name
  if p.parent==root/'docs' and p.name.startswith('sp_u12b3b_'):return g['TECH']/'docs'/p.name
  return p
 def record(op,path,**extra):
  tls.last=dict(operation=op,exact_absolute_path=str(Path(path).resolve()),worker_pid=os.getpid(),thread_id=threading.get_ident(),timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=g['_phase'],**extra)
 def exception_record(exc):
  if getattr(tls,'exception_for',None) is exc:return tls.exception
  operation=dict(tls.last) if hasattr(tls,'last') else None
  safe=str(exc) if isinstance(exc,(OSError,g['TechnicalFailure'])) else type(exc).__name__
  frames=[dict(source_file=str(Path(v.filename).resolve()),source_line=v.lineno,function=v.name,source_text=v.line) for v in traceback.extract_tb(exc.__traceback__)]
  return dict(exception_type=type(exc).__name__,errno=getattr(exc,'errno',None),winerror=getattr(exc,'winerror',None),exception_message=safe,filesystem_operation=operation,call_stack=frames,full_python_traceback=''.join(traceback.format_tb(exc.__traceback__))+type(exc).__name__+': '+safe,worker_pid=os.getpid(),timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 def deny(p,op,code):
  record(op,p,denial_origin='PYTHON_AUDIT_POLICY',reason=code)
  e=PermissionError(code);e.filename=str(Path(p).resolve());raise e
 def atomic(path,data):
  path=route(path);tmp=path.with_name(path.name+f'.partial.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}')
  try:
   record('MKDIR',path.parent);path.parent.mkdir(parents=True,exist_ok=True)
   # Only dead-writer temps for this exact destination may be quarantined.
   for stale in path.parent.glob(path.name+'.partial.*'):
    m=re.fullmatch(re.escape(path.name)+r'\.partial\.(\d+)\.\d+\.[0-9a-f]+',stale.name)
    if m is None:raise g['TechnicalFailure']('UNRECOGNIZED_PARTIAL_OWNER')
    pid=int(m[1]);kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.restype=ctypes.c_void_p;kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if handle:
     code=ctypes.c_ulong();ok=kernel.GetExitCodeProcess(handle,ctypes.byref(code));kernel.CloseHandle(handle)
     if not ok or code.value==259:raise g['TechnicalFailure']('ACTIVE_WRITER_SAME_KEY')
    elif ctypes.get_last_error()!=87:raise g['TechnicalFailure']('PARTIAL_OWNER_STATE_UNKNOWN')
    quarantine=g['TECH']/'orphan_temp_quarantine';quarantine.mkdir(parents=True,exist_ok=True);os.replace(stale,quarantine/(stale.name+'.'+uuid.uuid4().hex))
   record('CREATE',tmp)
   with tmp.open('xb') as f:
    record('WRITE',tmp);f.write(data);record('FLUSH',tmp);f.flush();record('FSYNC',tmp);os.fsync(f.fileno())
   record('REPLACE',tmp,destination=str(path.resolve()));os.replace(tmp,path)
   return dict(path=path.relative_to(root).as_posix(),sha256=g['sha'](path),bytes=path.stat().st_size)
  except Exception as exc:
   # Retain interrupted temp bytes; never promote a partial file into a checkpoint.
   info=exception_record(exc);tls.exception=info;tls.exception_for=exc
   try:
    destination=g['TECH']/f'io_exception_{os.getpid()}_{uuid.uuid4().hex}.json';destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x',encoding='utf8') as f:json.dump(info,f,indent=2)
   except OSError:pass
   raise
 def read(path):
  p=route(path)
  try:return original_read(p)
  except Exception as exc:tls.exception=exception_record(exc);tls.exception_for=exc;raise
 def firewall():
  if g['_guard']:return
  allowed={g['MANIFEST'],g['c'].OUT/'operators.npz',g['c'].OUT/'execution_freeze.json',g['c'].OUT/'development_seed_registry.json',root/'outputs/sp_u12b3a/execution_freeze.json',root/'outputs/sp_u12b3a/decision.json'};allowed={p.resolve() for p in allowed}
  def hook(event,args):
   if not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
   if event not in ('open','os.mkdir','os.rename','os.remove','os.rmdir','os.utime','os.link','os.symlink','os.chmod','os.chown'):return
   p=Path(os.fsdecode(args[0])).resolve()
   writing=False;op='OTHER';targets=[p]
   if event=='open':
    mode=args[1];flags=args[2];writing=bool(isinstance(mode,str) and any(v in mode for v in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC));op='WRITE' if writing else 'READ'
   else:
    writing=True;op={'os.mkdir':'MKDIR','os.rename':'RENAME','os.remove':'DELETE','os.rmdir':'DELETE','os.utime':'OTHER','os.link':'CREATE','os.symlink':'CREATE','os.chmod':'OTHER','os.chown':'OTHER'}[event]
    if event in ('os.rename','os.link','os.symlink'):targets.append(Path(os.fsdecode(args[1])).resolve())
   b2roots=(real_b2,g['B2'].resolve());sealed=g['SEAL'].resolve();metadata=g['TECH'].resolve()
   for target in targets:
    if writing and any(target.is_relative_to(v) for v in (*b2roots,failed)):deny(target,op,'B2_OR_FAILED_RUN_MUTATION_FORBIDDEN')
    if writing and target.is_relative_to(root/'outputs') and not(target.is_relative_to(sealed) or target.is_relative_to(metadata)):deny(target,op,'OLD_ARTIFACT_MUTATION_FORBIDDEN')
   if event!='open':return
   if p.is_relative_to(root):record(op,p,audit_event=event,mode=args[1],flags=args[2])
   if g['_opaque'] or g['c']._hashing:
    if writing:deny(p,op,'OPAQUE_WRITE_FORBIDDEN')
    return
   if not p.is_relative_to(root/'outputs'):return
   if p.is_relative_to(failed):deny(p,op,'FAILED_FORMAL_RUN_PERMANENTLY_EXCLUDED')
   for b2root in b2roots:
    if p.is_relative_to(b2root):
     rel=p.relative_to(b2root).parts
     if not writing and rel[0]=='cache' and g['_phase'] in ('RAW','INTEGRITY','SYNTHETIC'):return
     if not writing and rel[0]=='raw_evaluation' and g['_phase']=='REPLAY':return
     deny(p,op,'B2_OUTCOME_ACCESS_FORBIDDEN')
   if p.is_relative_to(sealed):
    if g['_phase'] not in ('RAW','COMPUTE','INTEGRITY','SEAL_CHECK','SYNTHETIC'):deny(p,op,'B3_SEALED_ACCESS_FORBIDDEN')
    return
   if p.is_relative_to(metadata):return
   if not writing and p in allowed:return
   deny(p,op,'HISTORICAL_OUTCOME_FIREWALL')
  sys.addaudithook(hook);g['_guard']=True
 original_init=g['init_worker']
 def init_worker():
  if g['B2'].resolve()==real_b2:
   from scripts.sp_u12b3c_authorization import require_formal_authorization
   require_formal_authorization('run')
  elif g['B2'].resolve()!=(root/'outputs/sp_u12b3c/synthetic_B2_topology').resolve():raise PermissionError('UNREGISTERED_SYNTHETIC_INPUT_ROOT')
  return original_init()
 _worker_context=(g,g['worker'])
 g.update(atomic=atomic,read=read,firewall=firewall,init_worker=init_worker,worker=guarded_worker,technical_exception_record=exception_record,technical_last_operation=lambda:getattr(tls,'last',None),technical_record_operation=record,technical_route=route)
