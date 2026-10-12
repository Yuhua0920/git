"""Run2 I/O operations, strict read-only firewall and durable technical forensics."""
from pathlib import Path
import os,sys,json,threading,uuid,traceback,datetime,re,ctypes,inspect,collections
_worker_context=None
def guarded_worker(condition,trajectory_id):
 g,original=_worker_context
 g['technical_context'](condition=condition,trajectory_id=trajectory_id,SNR=list(g['SNRS']))
 try:
  g['need'](not(g['TECH']/'PERMANENT_STOP.json').exists(),'PERMANENT_STOP')
  value=original(condition,trajectory_id)
  g['write'](g['TECH']/'worker_io'/f'{os.getpid()}.json',dict(PID=os.getpid(),operations=g['technical_operation_counts'](),formal=g['B2']==g['ROOT']/'outputs/sp_u12b2b_sealed',filesystem_exceptions=0))
  return value
 except Exception as exc:
  info=g['technical_exception_record'](exc)
  try:g['write'](g['TECH']/f'worker_exception_{os.getpid()}_{uuid.uuid4().hex}.json',info)
  except OSError:pass
  raise
def install_io(g):
 global _worker_context
 root=g['ROOT'];tls=threading.local();counts=collections.Counter();original_read=g['read']
 real_b2=(root/'outputs/sp_u12b2b_sealed').resolve();failed=(root/'outputs/sp_u12b3b_sealed').resolve()
 def route(path):
  p=Path(path)
  if p.parent==root/'outputs' and p.name.startswith('sp_u12b3b_'):return root/'outputs'/p.name.replace('sp_u12b3b_','sp_u12b3d_run2_',1)
  return p
 def context(**kwargs):tls.context=kwargs
 def record(op,path,**extra):
  frame=inspect.currentframe().f_back;ctx=getattr(tls,'context',{})
  counts[op]+=1
  writer=getattr(tls,'writer',{})
  tls.last=dict(operation=op,exact_absolute_path=str(Path(path).resolve()),PID=os.getpid(),worker_id=os.getpid(),thread_id=threading.get_ident(),timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_file=str(Path(frame.f_code.co_filename).resolve()),source_line=frame.f_lineno,current_stage=g['_phase'],trajectory_key=ctx,condition=ctx.get('condition'),SNR=ctx.get('SNR'),temp_path=writer.get('temp_path'),final_path=writer.get('final_path'),**extra)
 def exception_record(exc):
  if getattr(tls,'exception_for',None) is exc:return tls.exception
  op=dict(getattr(tls,'last',{}));ctx=getattr(tls,'context',{})
  safe=str(exc) if isinstance(exc,(OSError,g['TechnicalFailure'])) else type(exc).__name__
  frames=[dict(source_file=str(Path(v.filename).resolve()),source_line=v.lineno,function=v.name,source_text=v.line) for v in traceback.extract_tb(exc.__traceback__)]
  return dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),PID=os.getpid(),worker_id=os.getpid(),thread_id=threading.get_ident(),absolute_path=op.get('exact_absolute_path') or getattr(exc,'filename',None),operation=op.get('operation'),source_file=frames[-1]['source_file'] if frames else op.get('source_file'),source_line=frames[-1]['source_line'] if frames else op.get('source_line'),full_traceback=''.join(traceback.format_exception(type(exc),exc,exc.__traceback__)),errno=getattr(exc,'errno',None),WinError=getattr(exc,'winerror',None),trajectory_key=ctx,condition=ctx.get('condition'),SNR=ctx.get('SNR'),current_stage=op.get('current_stage',g['_phase']),temp_path=op.get('temp_path'),final_path=op.get('final_path'),filesystem_operation=op,call_stack=frames,exception_type=type(exc).__name__,exception_message=safe)
 def save_failure(exc):
  info=exception_record(exc);tls.exception=info;tls.exception_for=exc
  try:
   p=g['TECH']/f'io_exception_{os.getpid()}_{uuid.uuid4().hex}.json';p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('x',encoding='utf8') as f:json.dump(info,f,indent=2)
  except OSError:pass
  return info
 def deny(p,op,code):
  record(op,p,denial_origin='PYTHON_AUDIT_POLICY',reason=code)
  e=PermissionError(code);e.filename=str(Path(p).resolve());raise e
 def atomic(path,data):
  path=route(path);ctx=getattr(tls,'context',{});key=str(ctx.get('condition','metadata'))+'_'+str(ctx.get('trajectory_id','NA'))
  tmp=path.with_name(path.name+f'.partial.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.{key}')
  tls.writer=dict(temp_path=str(tmp.resolve()),final_path=str(path.resolve()))
  try:
   record('MKDIR',path.parent);path.parent.mkdir(parents=True,exist_ok=True)
   for stale in path.parent.glob(path.name+'.partial.*'):
    match=re.fullmatch(re.escape(path.name)+r'\.partial\.(\d+)\.\d+\.[0-9a-f]+\..+',stale.name)
    if match is None:raise g['TechnicalFailure']('UNRECOGNIZED_PARTIAL_OWNER')
    pid=int(match[1]);kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.restype=ctypes.c_void_p;kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if handle:
     code=ctypes.c_ulong();ok=kernel.GetExitCodeProcess(handle,ctypes.byref(code));kernel.CloseHandle(handle)
     if not ok or code.value==259:raise g['TechnicalFailure']('ACTIVE_WRITER_SAME_KEY')
    elif ctypes.get_last_error()!=87:raise g['TechnicalFailure']('PARTIAL_OWNER_STATE_UNKNOWN')
    q=g['TECH']/'quarantine';record('MKDIR',q);q.mkdir(parents=True,exist_ok=True)
    destination=q/(stale.name+'.'+uuid.uuid4().hex);record('REPLACE',stale,destination=str(destination));os.replace(stale,destination)
   kind='CHECKPOINT_WRITE' if path.suffix=='.json' and path.is_relative_to(g['SEAL']/'raw') else 'MANIFEST_WRITE' if 'manifest' in path.name else 'OPEN_WRITE'
   record(kind,path);record('TEMP_CREATE',tmp);record('OPEN_WRITE',tmp)
   with tmp.open('xb') as f:
    record('OPEN_WRITE',tmp);f.write(data);record('FLUSH',tmp);f.flush();record('FSYNC',tmp);os.fsync(f.fileno())
   record('REPLACE',tmp);tls.last.update(temp_path=str(tmp.resolve()),final_path=str(path.resolve()));os.replace(tmp,path)
   digest=g['sha'](path);g['need'](digest==__import__('hashlib').sha256(data).hexdigest(),'ATOMIC_HASH_VERIFICATION')
   return dict(path=path.relative_to(root).as_posix(),sha256=digest,bytes=path.stat().st_size)
  except Exception as exc:save_failure(exc);raise
  finally:tls.writer={}
 def read(path):
  p=route(path)
  try:
   record('CHECKPOINT_READ' if p.suffix=='.json' and p.is_relative_to(g['SEAL']/'raw') else 'OPEN_READ',p)
   return original_read(p)
  except Exception as exc:save_failure(exc);raise
 def firewall():
  if g['_guard']:return
  allowed={g['MANIFEST'],g['c'].OUT/'operators.npz',g['c'].OUT/'execution_freeze.json',g['c'].OUT/'development_seed_registry.json',root/'outputs/sp_u12b3a/execution_freeze.json',root/'outputs/sp_u12b3a/decision.json'};allowed={p.resolve() for p in allowed}
  def hook(event,args):
   if not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
   if event not in ('open','os.mkdir','os.rename','os.remove','os.rmdir','os.utime','os.link','os.symlink','os.chmod','os.chown'):return
   p=Path(os.fsdecode(args[0])).resolve();targets=[p]
   if event=='open':
    mode=args[1];flags=args[2];writing=bool(isinstance(mode,str) and any(v in mode for v in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC));op='OPEN_WRITE' if writing else 'OPEN_READ'
   else:
    writing=True;op={'os.mkdir':'MKDIR','os.rename':'RENAME','os.remove':'DELETE','os.rmdir':'DELETE'}.get(event,'MUTATION')
    if event in ('os.rename','os.link','os.symlink'):targets.append(Path(os.fsdecode(args[1])).resolve())
   b2roots=(real_b2,g['B2'].resolve());sealed=g['SEAL'].resolve();metadata=g['TECH'].resolve()
   for target in targets:
    if writing and any(target.is_relative_to(v) for v in b2roots):deny(target,op,'B3_RUN2_B2_WRITE_VIOLATION')
    if writing and target.is_relative_to(failed):deny(target,op,'FAILED_RUN_MUTATION_FORBIDDEN')
    ownedpublic=target.parent==root/'outputs' and target.name.startswith('sp_u12b3d_')
    owneddoc=target.parent==root/'docs' and target.name.startswith('sp_u12b3d_')
    if writing and target.is_relative_to(root/'outputs') and not(target.is_relative_to(sealed) or target.is_relative_to(metadata) or ownedpublic):deny(target,op,'OLD_ARTIFACT_MUTATION_FORBIDDEN')
    if writing and target.is_relative_to(root/'docs') and not owneddoc:deny(target,op,'OLD_DOCUMENT_MUTATION_FORBIDDEN')
    if writing and target.is_relative_to(root/'paper'):deny(target,op,'MANUSCRIPT_MUTATION_FORBIDDEN')
   if event!='open':
    if event=='os.rename' and getattr(tls,'last',{}).get('operation')=='REPLACE':counts['RENAME']+=1
    elif p.is_relative_to(root):record(op,p)
    return
   if p.is_relative_to(root):record(op,p,audit_event=event)
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
   if p.is_relative_to(metadata) or p.parent==root/'outputs' and p.name.startswith('sp_u12b3d_'):return
   if not writing and p in allowed:return
   deny(p,op,'HISTORICAL_OUTCOME_FIREWALL')
  sys.addaudithook(hook);g['_guard']=True
 original_init=g['init_worker']
 def init_worker():
  try:
   if g['B2'].resolve()==real_b2:
    from scripts.sp_u12b3d_authorization import require_formal_authorization
    require_formal_authorization()
   elif g['B2'].resolve()!=(root/'outputs/sp_u12b3d/synthetic_B2_topology').resolve():raise PermissionError('UNREGISTERED_SYNTHETIC_INPUT_ROOT')
   parent=__import__('multiprocessing').parent_process()
   g['write'](g['TECH']/'actual_worker_pids'/f'{os.getpid()}.json',dict(PID=os.getpid(),parent_PID=os.getppid(),controller_PID=parent.pid if parent else os.getpid(),role='actual interpreter PID;not launcher'))
   return original_init()
  except Exception as exc:
   info=exception_record(exc)
   try:g['write'](g['TECH']/f'worker_exception_init_{os.getpid()}_{uuid.uuid4().hex}.json',info)
   except OSError:pass
   raise
 _worker_context=(g,g['worker'])
 g.update(atomic=atomic,read=read,firewall=firewall,init_worker=init_worker,worker=guarded_worker,technical_exception_record=exception_record,technical_last_operation=lambda:getattr(tls,'last',None),technical_record_operation=record,technical_route=route,technical_context=context,technical_operation_counts=lambda:dict(counts))
