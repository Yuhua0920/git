"""Sealed B2B storage, integrity, and bootstrap infrastructure.

Scientific inference is exclusively scripts.sp_u12b2a_common.evaluate.
Console emits a closed whitelist of technical metadata only.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib,time,contextlib,csv,io,zipfile,subprocess
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import sp_u12b2a_common as c
from scripts import run_sp_u12b2a as a
TECH=ROOT/'outputs/sp_u12b2b';SEAL=ROOT/'outputs/sp_u12b2b_sealed';CFG=ROOT/'configs/sp_u12b2b_sealed.json'
PROTOCOL=ROOT/'docs/sp_u12b2b_protocol.md'
SOURCE_FILES=('scripts/sp_u12b2b_common.py','scripts/run_sp_u12b2b.py','scripts/sp_u12b2b_preserve.py','scripts/sp_u12b2b_resource.py','scripts/test_sp_u12b2b.py','tests/test_sp_u12b2b.py','tests/test_sp_u12b2b_integrity.py')
_phase='METADATA';_opaque=False;_guard=False;_state=None
class IntegrityError(RuntimeError):pass
def need(condition,code):
 if not condition:raise IntegrityError(code)
def timestamp():return datetime.now(timezone.utc).isoformat()
def sha(path):
 global _opaque
 previous=_opaque;_opaque=True
 try:return c.sha(path)
 finally:_opaque=previous
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 data=(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8')
 return atomic(path,data)
def atomic(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.partial')
 with temp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
 return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size)
def archive(path,**arrays):
 # Stable ZIP timestamps and fixed member order make same-seed repair auditable.
 buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_STORED) as z:
  for key in sorted(arrays):
   value=np.asarray(arrays[key]);need(value.dtype.hasobject is False,'OBJECT_DTYPE')
   member=io.BytesIO();np.lib.format.write_array(member,value,allow_pickle=False)
   info=zipfile.ZipInfo(key+'.npy',(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED;info.external_attr=0o600<<16;z.writestr(info,member.getvalue())
 return atomic(path,buf.getvalue())
def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def csvwrite(path,rows,fields):
 rows=list(rows);need(all(set(row)==set(fields) for row in rows),'CSV_SCHEMA')
 f=io.StringIO(newline='');writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 return atomic(path,f.getvalue().encode('utf8'))
@contextlib.contextmanager
def phase(name):
 global _phase
 previous=_phase;_phase=name
 try:yield
 finally:_phase=previous
def firewall():
 global _guard
 if _guard:return
 allowed={c.OUT/'operators.npz',c.OUT/'execution_freeze.json',c.OUT/'historical_seed_inventory.json',c.OUT/'development_seed_registry.json',ROOT/'outputs/sp_u12b2a_formal_manifest.json',ROOT/'outputs/sp_u12b2a_seed_registry.json',ROOT/'outputs/sp_u12b2a_seed_overlap_audit.json',ROOT/'outputs/sp_u12b2a_output_schema.json'}
 allowed|={ROOT/'outputs'/name for name in c.STATS.values()}
 allowed|={c.FORMAL/name for name in ('config.json','seed_registry.json','schemas.json','manifest.json')}
 allowed={p.resolve() for p in allowed}
 def hook(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  path=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
  if _opaque or c._hashing:return
  if not path.is_relative_to((ROOT/'outputs').resolve()):return
  writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  owned=path.is_relative_to(TECH.resolve()) or path.name.startswith('sp_u12b2b_')
  sealed=path.is_relative_to(SEAL.resolve())
  if sealed:
   if _phase not in ('REFERENCE','RAW','INTEGRITY','SEALED_COMPUTE','SEAL_CHECK','SYNTHETIC'):raise PermissionError('SEALED_ACCESS_DENIED')
   return
  if writing:
   if not owned:raise PermissionError('HISTORICAL_WRITE_DENIED')
  elif not(owned or path in allowed):raise PermissionError('OUTCOME_FIREWALL_DENIED')
 sys.addaudithook(hook);_guard=True
LOG_KEYS={'stage','status','condition','trajectory_count','reference_count','evaluation_count','cells_completed','replicates_completed','replicates_expected','elapsed_seconds','worker_pid','worker_count','peak_working_set_bytes','peak_commit_bytes','hash','files','rows','tests','failed','skipped','retry_count','error_type','failure_code','formal_H_generated','outcomes_viewed','reference_generated','evaluation_generated','device','protected_files','PASS','verdict'}
def emit(**metadata):
 need(set(metadata)<=LOG_KEYS,'CONSOLE_SCHEMA')
 print(json.dumps(metadata,ensure_ascii=False,allow_nan=False),flush=True)
 if TECH.exists():
  with (TECH/'technical_progress.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(timestamp_utc=timestamp(),**metadata),ensure_ascii=False,allow_nan=False)+'\n')
def peak():
 from scripts.sp_u4a_common import peak_memory
 return peak_memory()
def memory_available():
 import ctypes
 class MEMORY(ctypes.Structure):
  _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(key,ctypes.c_ulonglong) for key in ('total_physical','available_physical','total_pagefile','available_pagefile','total_virtual','available_virtual','extended')]
 value=MEMORY();value.length=ctypes.sizeof(value)
 need(bool(ctypes.WinDLL('kernel32').GlobalMemoryStatusEx(ctypes.byref(value))),'MEMORY_PROBE')
 return dict(total_physical_bytes=int(value.total_physical),available_physical_bytes=int(value.available_physical),probe='Win32 GlobalMemoryStatusEx;CIM unavailable in client')
def config():return read(CFG)
def original_gate():
 c.require_freeze();manifest=read(ROOT/'outputs/sp_u12b2a_formal_manifest.json')
 need(manifest['verdict']=='B2_PIPELINE_READY_FOR_SEALED_RUN','B2A_NOT_READY')
 need(sha(ROOT/manifest['operator_cache']['path'])==manifest['operator_cache']['sha256'],'OPERATOR_HASH_DRIFT')
 for path,digest in manifest['formal_metadata_hashes'].items():need(sha(ROOT/path)==digest,'B2A_METADATA_DRIFT')
 need({p.relative_to(c.FORMAL).as_posix() for p in c.FORMAL.rglob('*') if p.is_file()}=={'config.json','manifest.json','seed_registry.json','schemas.json'},'B2A_FORMAL_POPULATION_NOT_EMPTY')
 return manifest
def check_freeze():
 original_gate();f=read(TECH/'execution_freeze.json')
 for path,digest in f['hashes'].items():need(sha(ROOT/path)==digest,'B2B_EXECUTION_DRIFT')
 need(not(TECH/'PERMANENT_STOP.json').exists(),'PERMANENT_STOP')
 return f
def registry():return read(ROOT/'outputs/sp_u12b2a_seed_registry.json')
def seed_index():return a.seed_index(registry())
def load_h(path,strides=None):
 from scripts.resume_formal_test import load_native_h
 h=load_native_h(path)
 need(h.shape==(2,2,64,160) and h.dtype==np.complex128 and np.isfinite(h).all(),'H_SHAPE_FINITE')
 if strides is not None:need(tuple(h.strides)==tuple(strides),'H_LAYOUT')
 return h
def committed(path):
 if not Path(path).exists():return None
 info=read(path)
 for artifact in info.get('artifacts',[]):need(sha(ROOT/artifact['path'])==artifact['sha256'],'COMMITTED_ARTIFACT_CORRUPTION')
 return info
def channel_path(group,condition,i):return SEAL/('reference' if group=='MATCHED_REF' else 'cache')/condition/f'H_{i:04d}.npz'
def raw_path(condition,i):return SEAL/'raw_evaluation'/condition/f'{i:04d}.npz'
def raw_receipt(condition,i):return SEAL/'raw_evaluation'/condition/f'{i:04d}.json'
def validate_raw(raw,expected_bits):
 primary={k:raw[k] for k in ('frame_error','frame_truth','frame_bit_errors','frame_bit_total','coordinates','perfect_frame_bit_errors')}
 c.validate_raw(primary,c.SNRS)
 need(raw['method_coordinates'].shape==(7,7,6,2),'METHOD_COORDINATE_SHAPE')
 need(raw['coordinate_present'].shape==(7,7,6) and raw['coordinate_present'].dtype==np.bool_,'COORDINATE_NULL_MASK')
 need(raw['solve_status'].shape==(7,7,6) and raw['solve_status'].dtype==np.int8,'SOLVE_STATUS_SCHEMA')
 expected=np.zeros((7,7,6),bool);expected[:,2:6,:]=True
 need(np.array_equal(raw['coordinate_present'],expected),'COORDINATE_NULL_SEMANTICS')
 need(np.isfinite(raw['method_coordinates']).all(),'COORDINATE_NONFINITE')
 need(np.array_equal(raw['method_coordinates'][:,4],raw['method_coordinates'][:,5]) and np.array_equal(raw['coordinates'],raw['method_coordinates'][:,5]),'DENSE_LR4_COORDINATE_SHARING')
 expected_status=np.zeros((7,7,6),np.int8);expected_status[:,4:6]=1
 need(np.array_equal(raw['solve_status'],expected_status),'SOLVE_STATUS_INVALID')
 need(np.all(raw['frame_bit_total']==expected_bits),'BIT_TOTAL')
 need(all(np.array_equal(raw['frame_truth'][0],raw['frame_truth'][s]) for s in range(7)),'CROSS_SNR_TRUTH_PAIRING')
 E,T,B,N=c.raw_to_trajectory(primary,c.SNRS)
 for key,value in [('trajectory_error',E),('trajectory_truth',T),('trajectory_bit_errors',B),('trajectory_bit_total',N)]:need(np.array_equal(raw[key],value),'SUFFICIENT_STATISTICS_REDUCTION')
 return True
def pack_raw(raw,rows,expected_bits):
 need(len(rows)==294,'FRAME_ROWS')
 raw=dict(raw);coordinate=np.zeros((7,7,6,2));present=np.zeros((7,7,6),bool);status=np.zeros((7,7,6),np.int8);keys=set()
 for row in rows:
  need(set(row)==set(c.RAW_FIELDS),'RAW_FIELD_SCHEMA')
  key=(row['snr_db'],row['estimator'],row['frame_id']);need(key not in keys,'RAW_DUPLICATE_KEY');keys.add(key)
  s=c.SNRS.index(row['snr_db']);m=c.METHODS.index(row['estimator']);j=row['frame_id']-4
  need(0<=j<6,'RAW_FRAME_ID')
  need(row['error_energy']==raw['frame_error'][s,m,j] and row['truth_energy']==raw['frame_truth'][s,j] and row['bit_errors']==raw['frame_bit_errors'][s,m,j] and row['bit_total']==raw['frame_bit_total'][s,j],'ROW_ARRAY_MAPPING')
  if row['alpha'] is not None:
   coordinate[s,m,j]=[row['alpha'],row['beta']];present[s,m,j]=True
  else:need(row['beta'] is None,'NULL_COORDINATE_PAIR')
  status[s,m,j]=1 if row['solve_status']=='PASS' else 0
 need(keys=={(s,m,j) for s in c.SNRS for m in c.METHODS for j in range(4,10)},'RAW_KEY_POPULATION')
 E,T,B,N=c.raw_to_trajectory(raw,c.SNRS)
 raw.update(method_coordinates=coordinate,coordinate_present=present,solve_status=status,trajectory_error=E,trajectory_truth=T,trajectory_bit_errors=B,trajectory_bit_total=N)
 validate_raw(raw,expected_bits);return raw
def validate_inputs(inputs):
 need(len(inputs)==70,'INPUT_CARDINALITY');keys={(r['snr_db'],r['frame_id']) for r in inputs}
 need(keys=={(s,j) for s in c.SNRS for j in range(10)},'INPUT_KEYS')
 for j in range(10):
  values=[r for r in inputs if r['frame_id']==j]
  need(len({r['truth_sha256'] for r in values})==len({r['unit_noise_sha256'] for r in values})==1,'CRN_INPUT_ID')
 return True
def bootstrap_draw(E,T,index):
 # Paired across every method and SNR: resample trajectories, then pooled ratio.
 need(E.ndim==3 and E.shape[1:]==(7,7) and T.shape==(E.shape[0],7),'BOOTSTRAP_INPUT_SHAPE')
 need(index.ndim==2 and index.shape[1]==E.shape[0] and np.all((index>=0)&(index<E.shape[0])),'BOOTSTRAP_INDICES')
 counts=np.stack([np.bincount(row,minlength=len(E)) for row in index]);pooled_E=(counts@E.reshape(len(E),49)).reshape(len(index),7,7);pooled_T=counts@T
 need(np.isfinite(pooled_E).all() and np.isfinite(pooled_T).all() and np.all(pooled_E>0) and np.all(pooled_T>0),'BOOTSTRAP_NONFINITE')
 return 10*np.log10(pooled_E/pooled_T[:,:,None])
def bootstrap_array(E,T,seed,ordinal,total,contrasts=None,progress=None):
 rng=np.random.default_rng(np.random.SeedSequence([seed,ordinal]));shape=(total,7,7) if contrasts is None else (total,len(contrasts));draws=np.empty(shape);digest=hashlib.sha256()
 for start in range(0,total,1000):
  size=min(1000,total-start);index=rng.integers(0,len(E),(size,len(E)),dtype=np.int32);digest.update(index.tobytes(order='C'));db=bootstrap_draw(E,T,index)
  if contrasts is None:draws[start:start+size]=db
  else:
   for k,row in enumerate(contrasts):
    s=c.SNRS.index(row['snr_db']);draws[start:start+size,k]=db[:,s,c.METHODS.index(row['first'])]-db[:,s,c.METHODS.index(row['second'])]
  if progress is not None and (start+size)%10000==0:progress(start+size,total)
 need(np.isfinite(draws).all(),'BOOTSTRAP_DRAWS_NONFINITE');return draws,digest.hexdigest()

def init_eval_worker():
 global _state
 firewall()
 with phase('RAW'):
  check_freeze();cfg=config();operators=c.load_operators();matched={}
  for condition in c.CONDITIONS:
   stat=c.statistics(SEAL/'reference'/condition/'statistics.npz')
   need(stat.samples==5000 and stat.train_trajectory_ids==tuple(range(500)),'MATCHED_CORPUS_POPULATION')
   matched[condition]={s:stat.prepare(c.variance(s)) for s in c.SNRS}
  _state=(operators,matched,cfg)
def evaluate_worker(condition,i):
 firewall()
 with phase('RAW'):
  need(_state is not None,'WORKER_NOT_INITIALIZED');operators,matched,cfg=_state
  prior=committed(raw_receipt(condition,i))
  if prior is not None:return dict(condition=condition,trajectory_id=i,completed=True,resumed=True,worker_pid=prior['worker_pid'],peak=prior['peak'])
  hinfo=committed(channel_path('EVAL',condition,i).with_suffix('.json'));need(hinfo is not None,'EVAL_H_COMMIT_MISSING')
  h=load_h(channel_path('EVAL',condition,i),hinfo['export_strides_bytes'])
  commpath=SEAL/'cache'/condition/f'comm_{i:04d}.npz';comm_info=committed(commpath.with_suffix('.json'));need(comm_info is not None,'COMM_COMMIT_MISSING')
  z=arrays(commpath);comm=list(zip(z['bits'],z['symbols'],z['unit_noise']))
  for trio in comm:
   for value in trio:value.flags.writeable=False
  raw,rows,evidence=c.evaluate(h,comm,operators,matched[condition],c.SNRS)
  validate_inputs(evidence['inputs']);packed=pack_raw(raw,rows,cfg['expected_bit_total_per_frame'])
  record=archive(raw_path(condition,i),**packed)
  proof=dict(condition=condition,trajectory_id=i,group='EVAL',snr_db=list(c.SNRS),estimators=list(c.METHODS),scored_frames=list(range(4,10)),logical_frame_rows=294,logical_trajectory_rows=49,artifacts=[record],H_sha256=hinfo['H_sha256'],H_artifact_sha256=hinfo['artifacts'][0]['sha256'],communication_sha256=comm_info['artifacts'][0]['sha256'],inputs=evidence['inputs'],channel_seeds=hinfo['channel_seeds'],communication_seeds=comm_info['frame_seeds'],method_pairing=True,CRN_pairing=True,dense_lr4_shared_coordinates=True,finite_estimates=True,valid_coordinate_solve=True,matched_reference_statistics_sha256=sha(SEAL/'reference'/condition/'statistics.npz'),operator_sha256=cfg['operator_cache_sha256'],rank=4,fold_hash=cfg['fold_hash'],window_frames=5,pilot_geometry_hash=cfg['pilot_geometry_hash'],worker_pid=os.getpid(),peak=peak(),created_utc=timestamp(),outcomes_viewed=False)
  write(raw_receipt(condition,i),proof)
  return dict(condition=condition,trajectory_id=i,completed=True,resumed=False,worker_pid=proof['worker_pid'],peak=proof['peak'])

def synthetic_worker(output_path):
 """Analytic fixture exclusively; no scientific channel calls or result disclosure."""
 from src.cdl_benchmark import OfficialCDL,observation_geometry
 def forbidden(*args,**kwargs):raise IntegrityError('SCIENTIFIC_CDL_FORBIDDEN_IN_PREFLIGHT')
 OfficialCDL.generate=forbidden
 firewall()
 with phase('SYNTHETIC'):
  operators=c.load_operators();stat=c.statistics(ROOT/'outputs'/c.STATS['Source']);matched={s:stat.prepare(c.variance(s)) for s in c.SNRS}
  rx,tx,k,t=np.indices((2,2,64,160));h=(1+.4*rx+.3*tx+.01*k+.002*t+1j*(.2+.006*k+.004*t+.1*rx*tx)).astype(np.complex128)
  comm=[c.communicate(dict(bits_seed_tx0=4420001+j*3,bits_seed_tx1=4420002+j*3,noise_seed=4420003+j*3)) for j in range(10)]
  raw,rows,evidence=c.evaluate(h,comm,operators,matched,c.SNRS);validate_inputs(evidence['inputs']);raw=pack_raw(raw,rows,int(2*np.count_nonzero(observation_geometry().layout.data_mask)))
  record=archive(Path(output_path),**raw);return dict(sha256=record['sha256'],rows=len(rows),finite=True,scientific_CDL_generated=0)
