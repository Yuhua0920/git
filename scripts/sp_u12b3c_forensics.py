"""Technical synthetic-only reproduction of the frozen B3B I/O entry path."""
from pathlib import Path
import sys,os,json,traceback,time,threading,concurrent.futures
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b3b_common as old
import numpy as np
OUT=ROOT/'outputs/sp_u12b3c';DUMMY=OUT/'synthetic_B2_topology';DRY=ROOT/'outputs/sp_u12b3b_sealed_run2'
_events={};_installed=False
def bind():
 old.TECH=OUT/'old_wrapper_technical';old.B2=DUMMY;old.SEAL=DRY
def observe():
 global _installed
 if _installed:return
 def audit(event,args):
  if event not in ('open','os.mkdir','os.rename','os.remove','os.rmdir') or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  if not p.is_relative_to(ROOT):return
  op={'open':'OPEN','os.mkdir':'MKDIR','os.rename':'REPLACE_OR_RENAME','os.remove':'DELETE','os.rmdir':'DELETE'}[event]
  record=dict(event=event,path=str(p),operation=op,worker_pid=os.getpid(),thread_id=threading.get_ident(),timestamp_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),phase=old._phase)
  if event=='open':
   mode=args[1];flags=args[2];record.update(mode=mode,flags=flags);record['operation']='WRITE' if isinstance(mode,str) and any(v in mode for v in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC) else 'READ'
  if event=='os.rename':record['destination']=str(Path(args[1]).resolve())
  _events[threading.get_ident()]=record
 sys.addaudithook(audit);_installed=True
def capture(exc,stage):
 event=dict(_events.get(threading.get_ident(),{}));tb=traceback.format_exception(type(exc),exc,exc.__traceback__);frames=[dict(source_file=str(Path(v.filename).resolve()),source_line=v.lineno,function=v.name,source_text=v.line) for v in traceback.extract_tb(exc.__traceback__)]
 return dict(stage=stage,exception_type=type(exc).__name__,errno=getattr(exc,'errno',None),winerror=getattr(exc,'winerror',None),exception_message=str(exc),traceback=''.join(tb),call_stack=frames,denied_operation=event,worker_pid=os.getpid(),timestamp_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),data_scope='SYNTHETIC_ONLY;no real B2 or failed B3 data decoded')
def init():
 bind();observe();old.init_worker()
def job(i):
 try:return dict(PASS=True,identity=i,result=old.worker('ID',i))
 except Exception as exc:return dict(PASS=False,identity=i,forensics=capture(exc,'frozen_worker_synthetic'))
def prepare():
 # No formal inputs or failed raw arrays are decoded here.
 if not(ROOT/'outputs/sp_u12b3c_failed_run1_archive.json').exists():raise RuntimeError('FAILED_ARCHIVE_MUST_PRECEDE_TEST')
 old.TECH=OUT/'old_wrapper_technical';old.TECH.mkdir(parents=True,exist_ok=True)
 old.atomic(old.TECH/'execution_freeze.json',(ROOT/'outputs/sp_u12b3b/execution_freeze.json').read_bytes())
 rx,tx,k,t=np.indices((2,2,64,160));h=(1+.4*rx+.3*tx+.01*k+.002*t+1j*(.2+.006*k+.004*t+.1*rx*tx)).astype(np.complex128)
 temp=OUT/'synthetic_H.npz';old.archive(temp,H=h)
 from scripts.resume_formal_test import load_native_h
 from src.cdl_benchmark import array_hash,OfficialCDL
 OfficialCDL.generate=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('SCIENTIFIC_CHANNEL_GENERATION_FORBIDDEN'))
 h=load_native_h(temp);operators=old.c.load_operators();cvs,factors=old.c.family(operators,old.SNRS)
 seeds=[dict(bits_seed_tx0=5520001+j*3,bits_seed_tx1=5520002+j*3,noise_seed=5520003+j*3) for j in range(10)];comm=[old.c.communicate(v) for v in seeds]
 raw,inputs,residual=old.evaluate(h,comm,operators,cvs,factors)
 old.archive(OUT/'synthetic_expected.npz',**raw)
 b2raw=dict(coordinates=np.zeros((7,6,2)),frame_error=np.ones((7,7,6)),frame_truth=np.ones((7,6)))
 for si,s in enumerate(old.SNRS):
  idx=old.c.SNRS.index(s);b2raw['coordinates'][idx]=raw['coordinates'][si,2];b2raw['frame_truth'][idx]=raw['frame_truth'][si]
  for di,m in enumerate(('Dense','LR4')):b2raw['frame_error'][idx,old.c.METHODS.index(m)]=raw['frame_error'][si,2,di]
 for i in range(100):
  p=DUMMY/'cache/ID'/f'H_{i:04d}.npz';rec=old.archive(p,H=h);old.write(p.with_suffix('.json'),dict(H_sha256=array_hash(h),export_strides_bytes=list(h.strides),channel_seeds={'SYNTHETIC_ONLY':True},artifacts=[rec]))
  p=DUMMY/'cache/ID'/f'comm_{i:04d}.npz';rec=old.archive(p,bits=np.stack([v[0] for v in comm]),symbols=np.stack([v[1] for v in comm]),unit_noise=np.stack([v[2] for v in comm]));old.write(p.with_suffix('.json'),dict(frame_seeds=seeds,artifacts=[rec],**{k+'_sha256':[array_hash(v[j]) for v in comm] for j,k in enumerate(('bits','symbols','unit_noise'))}))
  p=DUMMY/'raw_evaluation/ID'/f'{i:04d}.npz';rec=old.archive(p,**b2raw);old.write(p.with_suffix('.json'),dict(inputs=inputs,artifacts=[rec],data_scope='SYNTHETIC_ONLY'))
 old.write(OUT/'synthetic_fixture_receipt.json',dict(PASS=True,synthetic_trajectories=100,new_scientific_H=0,formal_inputs_decoded=False,failed_partial_arrays_decoded=False,expected_raw_sha256=old.sha(OUT/'synthetic_expected.npz'),normal_residual_gate_PASS=residual<=old.c.SOLVE_TOL))
 print(json.dumps(dict(stage='synthetic_prepare',PASS=True)),flush=True)
def reproduce():
 bind();observe();old.firewall();tick=time.monotonic();results=[]
 with old.phase('RAW'):
  with concurrent.futures.ProcessPoolExecutor(max_workers=2,initializer=init) as pool:
   jobs=[pool.submit(job,i) for i in range(100)]
   for n,f in enumerate(concurrent.futures.as_completed(jobs),1):
    try:results.append(f.result())
    except Exception as exc:results.append(dict(PASS=False,forensics=capture(exc,'parent_or_initializer')))
    if n%20==0:print(json.dumps(dict(stage='old_wrapper_reproduction',cycles=n)),flush=True)
 errors=[v for v in results if not v['PASS']];report=dict(original_traceback_saved=False,original_failure_traceback_recoverable=False,original_exception_state=old.read(ROOT/'outputs/sp_u12b3b/PERMANENT_STOP.json') if False else dict(exception_type='PermissionError',failure_code='SANITIZED_TECHNICAL_FAILURE'),synthetic_cycles=len(results),errors=errors,passed=sum(v['PASS'] for v in results),elapsed_seconds=time.monotonic()-tick,original_formal_root_cause_confirmed=False,original_denied_path=None,original_denied_operation=None,formal_data_processing=False)
 old.write(old.TECH/'exception_forensics.json',report);old.write(old.TECH/'synthetic_results.json',results);print(json.dumps(dict(stage='old_wrapper_reproduction',cycles=len(results),errors=len(errors))),flush=True)
if __name__=='__main__':{'prepare':prepare,'reproduce':reproduce}[sys.argv[1]]()
