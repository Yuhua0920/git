"""B3 sealed input reuse, independent arm fitting and metadata-only reporting."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
import sys,json,hashlib,contextlib,time,io,zipfile,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import sp_u12b2a_common as c
from scripts.sp_u12b3a_ablation import ARMS,deployed_arms
TECH=ROOT/'outputs/sp_u12b3d';SEAL=ROOT/'outputs/sp_u12b3d_run2_sealed';B2=ROOT/'outputs/sp_u12b2b_sealed'
CFG=ROOT/'configs/sp_u12b3b_sealed.json';PROTOCOL=ROOT/'docs/sp_u12b3b_protocol.md'
SNRS=(10,20);CONDITIONS=tuple(c.CONDITIONS);DEPLOY=('Dense','LR4')
MANIFEST=ROOT/'outputs/sp_u12b2b_sealed_manifest.json'
MANIFEST_SHA='54e5346a943e35b66e4a4f51a267b4db765430da1f27c6f0bc1bdbc95d1aa67e'
CERT=ROOT/'docs/sp_u12b2b_seal_certificate.md'
CERT_SHA='f01b24730524744dcd75d8f1f4ec937bcc79f078bb26faf03f3fffed94b5a576'
_opaque=False;_guard=False;_phase='METADATA';_state=None
class TechnicalFailure(RuntimeError):pass
def need(value,code):
 if not value:raise TechnicalFailure(code)
def sha(path):
 global _opaque
 previous=_opaque;_opaque=True
 try:
  h=hashlib.sha256()
  with Path(path).open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  return h.hexdigest()
 finally:_opaque=previous
def tree(leaves):return hashlib.sha256(json.dumps(leaves,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def atomic(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
 with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
 return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size)
def write(path,obj):return atomic(path,(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def archive(path,**arrays):
 buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_STORED) as z:
  for key in sorted(arrays):
   v=np.asarray(arrays[key]);need(not v.dtype.hasobject,'OBJECT_DTYPE');member=io.BytesIO();np.lib.format.write_array(member,v,allow_pickle=False)
   info=zipfile.ZipInfo(key+'.npy',(1980,1,1,0,0,0));info.external_attr=0o600<<16;z.writestr(info,member.getvalue())
 return atomic(path,buf.getvalue())
def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def csvwrite(path,rows):
 rows=list(rows);need(bool(rows) and all(set(r)==set(rows[0]) for r in rows),'CSV_SCHEMA')
 f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);return atomic(path,f.getvalue().encode())
@contextlib.contextmanager
def phase(name):
 global _phase
 old=_phase;_phase=name
 try:yield
 finally:_phase=old
def firewall():
 global _guard
 if _guard:return
 allowed={MANIFEST,c.OUT/'operators.npz',c.OUT/'execution_freeze.json',ROOT/'outputs/sp_u12b3a/execution_freeze.json',ROOT/'outputs/sp_u12b3a/decision.json'}
 allowed={p.resolve() for p in allowed}
 def hook(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
  writing=(isinstance(mode,str) and any(s in mode for s in 'wax+')) or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
  if _opaque or c._hashing:
   need(not writing,'OPAQUE_WRITE_FORBIDDEN');return
  if p==CERT.resolve() and writing:raise PermissionError('B2_CERTIFICATE_WRITE_FORBIDDEN')
  if not p.is_relative_to((ROOT/'outputs').resolve()):return
  if p.is_relative_to(B2.resolve()):
   if writing:raise PermissionError('B2_WRITE_FORBIDDEN')
   rel=p.relative_to(B2.resolve()).parts
   if rel and rel[0]=='cache' and _phase in ('RAW','INTEGRITY'):return
   if rel and rel[0]=='raw_evaluation' and _phase=='REPLAY':return
   raise PermissionError('B2_OUTCOME_ACCESS_FORBIDDEN')
  if p.is_relative_to(SEAL.resolve()):
   if _phase not in ('RAW','COMPUTE','INTEGRITY','SEAL_CHECK','SYNTHETIC'):raise PermissionError('B3_SEALED_ACCESS_FORBIDDEN')
   return
  owned=p.is_relative_to(TECH.resolve()) or p.name.startswith('sp_u12b3b_')
  if writing and not owned:raise PermissionError('OLD_ARTIFACT_WRITE_FORBIDDEN')
  if not writing and not(owned or p in allowed):raise PermissionError('HISTORICAL_OUTCOME_FIREWALL')
 sys.addaudithook(hook);_guard=True
LOG_KEYS={'stage','status','condition','trajectory_count','cells_completed','elapsed_seconds','worker_count','worker_pid','files','rows','tests','failed','skipped','PASS','verdict','failure_code','error_type','replicates_completed','replicates_expected','outcomes_viewed','hash','protected_files'}
def emit(**metadata):
 need(set(metadata)<=LOG_KEYS,'CONSOLE_SCHEMA');print(json.dumps(metadata,allow_nan=False),flush=True)
 if TECH.exists():
  with (TECH/'technical_progress.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(metadata,allow_nan=False)+'\n')
def peak():
 from scripts.sp_u4a_common import peak_memory
 return peak_memory()
def check_freeze():
 need(not(TECH/'PERMANENT_STOP.json').exists(),'PERMANENT_STOP')
 f=read(TECH/'execution_freeze.json')
 for name,digest in f['hashes'].items():need(sha(ROOT/name)==digest,'EXECUTION_BYTES_DRIFT')
 need(sha(MANIFEST)==MANIFEST_SHA and sha(CERT)==CERT_SHA,'B2_SEAL_METADATA_DRIFT')
 return f
def validate(raw):
 shapes={'frame_error':(2,3,2,6),'frame_truth':(2,6),'coordinates':(2,3,6,2),'coordinate_present':(2,3,6,2),'deployment_coordinates':(2,3,2,6,2),'solve_status':(2,3,2,6),'trajectory_error':(2,3,2),'trajectory_truth':(2,)}
 need(set(raw)==set(shapes),'RAW_SCHEMA')
 for key,shape in shapes.items():need(raw[key].shape==shape and np.isfinite(raw[key]).all(),'RAW_SHAPE_FINITE')
 mask=np.ones((2,3,6,2),bool);mask[:,0,:,1]=False;mask[:,1,:,0]=False
 need(raw['coordinate_present'].dtype==np.bool_ and np.array_equal(raw['coordinate_present'],mask),'NULL_MASK')
 need(np.all(raw['coordinates'][~mask]==0),'NULL_PAYLOAD_NOT_ZERO')
 need(np.array_equal(raw['deployment_coordinates'][:,:,0],raw['coordinates']) and np.array_equal(raw['deployment_coordinates'][:,:,1],raw['coordinates']),'COORDINATE_SHARING')
 need(raw['solve_status'].dtype==np.int8 and np.all(raw['solve_status']==1),'SOLVE_STATUS')
 need(np.all(raw['frame_error']>0) and np.all(raw['frame_truth']>0),'ENERGY_INVALID')
 need(np.array_equal(raw['frame_truth'][0],raw['frame_truth'][1]),'TRUTH_PAIRING')
 E=np.zeros((2,3,2));T=np.zeros(2)
 for j in range(6):E+=raw['frame_error'][:,:,:,j];T+=raw['frame_truth'][:,j]
 need(np.array_equal(E,raw['trajectory_error']) and np.array_equal(T,raw['trajectory_truth']),'TRAJECTORY_REDUCTION')
 return True
def evaluate(h,comm,a,cvs,factors):
 from src.cdl_benchmark import observation_geometry,frames,array_hash
 from src.link import assemble_pilot_data_frame,apply_frequency_mimo_channel,estimate_ls_linear_channel
 g=observation_geometry();truths=frames(h);pilots={s:[] for s in SNRS};inputs=[]
 for j,(_,symbols,z) in enumerate(comm):
  clean=apply_frequency_mimo_channel(truths[j],assemble_pilot_data_frame(symbols,g.layout))
  for s in SNRS:
   y=clean+np.sqrt(c.variance(s))*z;y.flags.writeable=False
   sparse,_=estimate_ls_linear_channel(y,g.layout);p=sparse.ravel()[g.selected];p.flags.writeable=False;pilots[s].append(p)
   inputs.append(dict(snr_db=s,frame_id=j,truth_sha256=array_hash(truths[j]),unit_noise_sha256=array_hash(z),pilot_sha256=array_hash(p),Y_sha256=array_hash(y)))
 E=np.zeros((2,3,2,6));T=np.zeros((2,6));coords=np.zeros((2,3,6,2));mask=np.ones(coords.shape,bool);mask[:,0,:,1]=False;mask[:,1,:,0]=False;residuals=[]
 for si,s in enumerate(SNRS):
  stream=np.stack(pilots[s]);rows=deployed_arms(stream,a,s,cvs[s],factors['P',s],factors['M',s]);original=c.infer(stream,a,s,cvs,factors)
  need([r['frame_id'] for r in rows]==list(range(4,10)),'FRAME_ORDER')
  for ji,row in enumerate(rows):
   need(row['window_start']==ji,'CAUSAL_WINDOW');truth=truths[ji+4].ravel()[g.data];T[si,ji]=float(np.sum(abs(truth)**2))
   both=row['arms']['Both'];old=original[ji]
   need(np.array_equal(both['coefficients'],old['coefficients']['affine']),'BOTH_CURRENT_COORDINATE_REPLAY')
   for d,key in [('Dense','dense-affine'),('LR4','lr4-affine')]:need(np.array_equal(both['predictions'][d],old['predictions'][key]),'BOTH_CURRENT_PREDICTION_REPLAY')
   for ai,arm in enumerate(ARMS):
    v=row['arms'][arm];diag=v['diagnostics'];coef=v['coefficients']
    need(v['dense_coefficients'] is coef and v['lr4_coefficients'] is coef,'COORDINATE_OBJECT_SHARING')
    need(not diag['clipped'] and not diag['regularized'] and not diag.get('fallback',False),'METHOD_SEMANTICS')
    residuals.append(diag['normal_equation_relative_residual']);need(residuals[-1]<=c.SOLVE_TOL,'NORMAL_RESIDUAL')
    coords[si,ai,ji,mask[si,ai,ji]]=coef
    for di,d in enumerate(DEPLOY):
     pred=v['predictions'][d];need(pred.shape==(3968,) and np.isfinite(pred).all(),'PREDICTION_FINITE');E[si,ai,di,ji]=float(np.sum(abs(pred-truth)**2))
 te=np.zeros((2,3,2));tt=np.zeros(2)
 for j in range(6):te+=E[:,:,:,j];tt+=T[:,j]
 raw=dict(frame_error=E,frame_truth=T,coordinates=coords,coordinate_present=mask,deployment_coordinates=np.repeat(coords[:,:,None],2,axis=2),solve_status=np.ones((2,3,2,6),np.int8),trajectory_error=te,trajectory_truth=tt)
 validate(raw);return raw,inputs,max(residuals)
def init_worker():
 global _state
 firewall();check_freeze()
 from src.cdl_benchmark import OfficialCDL
 def forbidden(*args,**kwargs):raise TechnicalFailure('NEW_CHANNEL_GENERATION_FORBIDDEN')
 OfficialCDL.generate=forbidden
 a=c.load_operators();cvs,factors=c.family(a,SNRS);_state=(a,cvs,factors)
def worker(condition,i):
 firewall();need(_state is not None,'WORKER_INIT');a,cvs,factors=_state
 path=SEAL/'raw'/condition/f'{i:04d}.npz';receipt=path.with_suffix('.json')
 with phase('RAW'):
  if receipt.exists():
   proof=read(receipt);need(sha(path)==proof['artifact']['sha256'],'CHECKPOINT_HASH');need(proof['execution_freeze_sha256']==sha(TECH/'execution_freeze.json'),'CHECKPOINT_FREEZE');validate(arrays(path))
   return dict(condition=condition,trajectory_id=i,resumed=True,worker_pid=proof['worker_pid'],peak=proof['peak'])
  hpath=B2/'cache'/condition/f'H_{i:04d}.npz';cp=B2/'cache'/condition/f'comm_{i:04d}.npz';hi=read(hpath.with_suffix('.json'));ci=read(cp.with_suffix('.json'))
  for p,info in [(hpath,hi),(cp,ci)]:need(sha(p)==info['artifacts'][0]['sha256'],'INPUT_COMMIT_HASH')
  from scripts.resume_formal_test import load_native_h
  from src.cdl_benchmark import array_hash
  h=load_native_h(hpath);need(tuple(h.strides)==tuple(hi['export_strides_bytes']) and array_hash(h)==hi['H_sha256'],'NATIVE_H_REPLAY')
  z=arrays(cp);comm=list(zip(z['bits'],z['symbols'],z['unit_noise']))
  for j,trio in enumerate(comm):
   for v,k in zip(trio,('bits','symbols','unit_noise')):need(array_hash(v)==ci[k+'_sha256'][j],'COMM_VALUE_HASH');v.flags.writeable=False
  raw,inputs,residual=evaluate(h,comm,a,cvs,factors)
 with phase('REPLAY'):
  bpath=B2/'raw_evaluation'/condition/f'{i:04d}.npz';br=read(bpath.with_suffix('.json'));need(sha(bpath)==br['artifacts'][0]['sha256'],'B2_RAW_COMMIT_HASH')
  with np.load(bpath,allow_pickle=False) as stored:old={k:stored[k] for k in ('coordinates','frame_error','frame_truth')}
  expected=[v for v in br['inputs'] if v['snr_db'] in SNRS];need(inputs==expected,'B2_PILOT_INPUT_REPLAY')
  for si,s in enumerate(SNRS):
   idx=c.SNRS.index(s)
   need(np.array_equal(raw['coordinates'][si,2],old['coordinates'][idx]),'B2_COORDINATE_REPLAY')
   need(np.array_equal(raw['frame_truth'][si],old['frame_truth'][idx]),'B2_TRUTH_REPLAY')
   for di,m in enumerate(('Dense','LR4')):need(np.array_equal(raw['frame_error'][si,2,di],old['frame_error'][idx,c.METHODS.index(m)]),'B2_ERROR_REPLAY')
 with phase('RAW'):
  record=archive(path,**raw);proof=dict(condition=condition,trajectory_id=i,artifact=record,inputs=inputs,channel_seeds=hi['channel_seeds'],communication_seeds=ci['frame_seeds'],H_sha256=hi['H_sha256'],H_artifact_sha256=sha(hpath),communication_artifact_sha256=sha(cp),operator_sha256=read(CFG)['operator_sha256'],execution_freeze_sha256=sha(TECH/'execution_freeze.json'),rank=4,folds=4,window=5,scored_frames=list(range(4,10)),Dense_LR4_coordinates_shared=True,Both_current_predictions_bitwise=True,Both_B2_coordinates_energies_bitwise=True,normal_residual_max=residual,unexpected_solve_failures=0,nonfinite_outputs=0,worker_pid=os.getpid(),peak=peak(),outcomes_viewed=False)
  write(receipt,proof)
 return dict(condition=condition,trajectory_id=i,resumed=False,worker_pid=proof['worker_pid'],peak=proof['peak'])
def draw(E,T,index):
 need(E.shape==(len(E),2,3,2) and T.shape==(len(E),2),'BOOT_SHAPE')
 need(index.ndim==2 and index.shape[1]==len(E) and np.all((index>=0)&(index<len(E))),'BOOT_INDEX')
 counts=np.stack([np.bincount(v,minlength=len(E)) for v in index]);pe=(counts@E.reshape(len(E),12)).reshape(len(index),2,3,2);pt=counts@T
 need(np.all(pe>0) and np.all(pt>0),'BOOT_ENERGY');return 10*np.log10(pe/pt[:,:,None,None])
def bootstrap(E,T,seed,ordinal):
 rng=np.random.default_rng(np.random.SeedSequence([seed,ordinal]));values=np.empty((10000,2,3,2));digest=hashlib.sha256()
 for start in range(0,10000,500):
  index=rng.integers(0,len(E),(500,len(E)),dtype=np.int32);digest.update(index.tobytes());values[start:start+500]=draw(E,T,index)
 need(np.isfinite(values).all(),'BOOT_FINITE');return values,digest.hexdigest()
def classify(lb,ub):return 'POSITIVE' if lb>0 else 'NEGATIVE' if ub<0 else 'INDETERMINATE'

# Authorized technical overrides; preceding scientific function sources are unchanged.
from scripts.sp_u12b3d_io import install_io
install_io(globals())
