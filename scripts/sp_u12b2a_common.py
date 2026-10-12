"""Full-SNR infrastructure. Frozen scientific estimators are imported, never redefined.

B2A may instantiate operators, replay old anchors and run development-only smoke.
Formal generation and analysis require separate, future authorization records.
"""
from __future__ import annotations
from pathlib import Path
import contextlib,csv,hashlib,json,os,sys,time,ast
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/sp_u12b2a';FORMAL=OUT/'formal';CONFIG=ROOT/'configs/sp_u12b2a_snr_sweep.yaml'
SNRS=(0,5,10,15,20,25,30);EDGE=(0,15,30);POWER=9.941134822523972
CONDITIONS={'ID':'CDL-C-30-100','Profile':'CDL-B-30-100','Mobility':'CDL-C-120-100','Composed':'CDL-B-120-100','Boundary':'CDL-C-30-200'}
METHODS=('LS-LIN','Source','Hard','Convex','Dense','LR4','Matched')
ROLES=('sionna_seed','torch_seed','numpy_seed','bits_seed_tx0','bits_seed_tx1','noise_seed')
STATS={'Source':'t15a_cdl_statistics_500.npz','P':'t15c_cdl_statistics_CDL-B-30.npz','M':'t15c_cdl_statistics_CDL-C-120.npz'}
OLD_OBJECTS=('t16e_frozen_operators.npz','t16f_frozen_candidates.npz','t16g_frozen_module_factors.npz')
MATCHED=('t15a_cdl_statistics_500.npz','t15c_cdl_statistics_CDL-B-30.npz','t15c_cdl_statistics_CDL-C-120.npz','t16c_heldout_statistics_CDL-B-120.npz','t19a_matched_statistics_CDL-C-30-200.npz')
TOL=1e-10;SOLVE_TOL=1e-11
RAW_FIELDS=('condition','snr_db','estimator','trajectory_id','frame_id','error_energy','truth_energy','bit_errors','bit_total','alpha','beta','solve_status')
SOURCE_FILES=('scripts/sp_u12b2a_common.py','scripts/run_sp_u12b2a.py','scripts/sp_u12b2a_preserve.py','tests/test_sp_u12b2a.py')
_phase='PREPARATION';_hashing=False;_guard=False;_denials=[]

def sha(p):
 global _hashing
 previous=_hashing;_hashing=True
 try:
  h=hashlib.sha256()
  with Path(p).open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  return h.hexdigest()
 finally:_hashing=previous
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,obj,exclusive=True):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x' if exclusive else 'w',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvwrite(p,rows):
 rows=list(rows);p=Path(p)
 if not rows or any(set(x)!=set(rows[0]) for x in rows):raise ValueError('CSV row schema')
 with p.open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def csvread(p):
 with Path(p).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def immutable_archive(p,**arrays):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:np.savez(f,**arrays)
 return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
def rel(a,b):return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),np.finfo(float).tiny))
def variance(s):return float(POWER/10**(s/10))
@contextlib.contextmanager
def phase(name):
 global _phase
 prior=_phase;_phase=name
 try:yield
 finally:_phase=prior
def firewall():
 global _guard
 if _guard:return
 base={(ROOT/'outputs'/n).resolve() for n in (*STATS.values(),*OLD_OBJECTS)}
 anchors={(ROOT/'outputs'/n).resolve() for n in MATCHED}
 anchors|={(ROOT/'outputs'/n).resolve() for n in ('t20b_formal_seed_manifest.csv','t20b_formal_frame_metrics.csv','t20b_formal_trajectory_metrics.csv','t20b_formal_nmse.csv','t20b_formal_ber.csv')}
 def hook(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
  writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  if _hashing:return
  if not p.is_relative_to((ROOT/'outputs').resolve()):return
  if writing:
   if p.is_relative_to(FORMAL.resolve()) and _phase not in ('PREPARATION','FORMAL_METADATA','SEALED_GENERATION','AUTHORIZED_ANALYSIS'):raise PermissionError('Formal B2 write blocked during B2A validation')
   if not p.is_relative_to(OUT.resolve()) and not p.name.startswith('sp_u12b2a_'):raise PermissionError('Historical output write blocked')
   return
  if p.is_relative_to(FORMAL.resolve()) and _phase not in ('PREPARATION','FORMAL_METADATA','SEALED_GENERATION','AUTHORIZED_ANALYSIS'):
   _denials.append(str(p));raise PermissionError('Formal B2 read blocked during B2A validation')
  if p.is_relative_to(OUT.resolve()) or p.name.startswith('sp_u12b2a_') or p in base:return
  if _phase=='ANCHOR' and (p in anchors or p.is_relative_to((ROOT/'outputs/t20b_formal_data').resolve())):return
  _denials.append(str(p));raise PermissionError('Outcome firewall: '+p.name)
 sys.addaudithook(hook);_guard=True

def statistics(path):
 from src.cdl_benchmark import FrozenStatistics
 with np.load(path,allow_pickle=False) as f:
  arrays=[f[k].copy() for k in ('mean_p','mean_d','rpp','rdp')];ids=tuple(map(int,f['train_trajectory_ids']));n=int(f['samples'])
 for a in arrays:a.flags.writeable=False
 return FrozenStatistics(*arrays,ids,n)
def instantiate(stats,snrs=SNRS):
 from src.pilot_cv import frozen_folds,cross_pilot_operators
 from src.cdl_benchmark import observation_geometry,array_hash
 from src.lowrank_modules import WeightedModule
 folds,definition=frozen_folds(observation_geometry().selected)
 a={'source_mean_p':stats['Source'].mean_p,'source_mean_d':stats['Source'].mean_d};audit=[]
 for s in snrs:
  sig=variance(s)
  for origin,st in stats.items():
   if st.samples!=5000 or st.train_trajectory_ids!=tuple(range(500)):raise ValueError('Preparation corpus identity')
   if st.mean_p.shape!=(64,) or st.mean_d.shape!=(3968,) or st.rpp.shape!=(64,64) or st.rdp.shape!=(3968,64):raise ValueError('Preparation shape/order gate')
   if any(not np.isfinite(v).all() for v in (st.mean_p,st.mean_d,st.rpp,st.rdp)):raise ValueError('Preparation nonfinite')
   estimator=st.prepare(sig);kernels,kdiag=cross_pilot_operators(st.rpp,sig,folds)
   label='source' if origin=='Source' else origin
   a[f'W{label}_{s}']=estimator.weight;a[f'K{origin}_{s}']=np.stack(kernels)
   C=st.rpp+sig*np.eye(64);hermitian=rel(C,C.conj().T)
   row=dict(snr_db=s,origin=origin,sigma2=sig,statistics_sha256=sha(ROOT/'outputs'/STATS[origin]),mean_p_sha256=array_hash(st.mean_p),mean_d_sha256=array_hash(st.mean_d),W_shape=list(estimator.weight.shape),K_shape=list(a[f'K{origin}_{s}'].shape),hermitian_relative_residual=hermitian,linear_solve_relative_residual=estimator.solve_relative_residual,kernel_solve_max_residual=max(d['solve_relative_residual'] for d in kdiag),covariance_condition=float(np.linalg.cond(C)),rank=None,full_factor_reconstruction_residual=None,rank4_right_solve_residual=None,weighted_tail_identity_residual=None,finite=True,fallback=False,rank_reselection=False,status='PASS')
   if hermitian>SOLVE_TOL or row['linear_solve_relative_residual']>SOLVE_TOL or row['kernel_solve_max_residual']>SOLVE_TOL:raise ValueError('Operator solve/Hermitian gate')
   if origin!='Source':
    a[f'DeltaW{origin}_{s}']=a[f'W{origin}_{s}']-a[f'Wsource_{s}'];a[f'DeltaK{origin}_{s}']=a[f'K{origin}_{s}']-a[f'KSource_{s}']
    mod=WeightedModule.build(a[f'DeltaW{origin}_{s}'],C);f=mod.factors(4);diag=mod.diagnostics()
    a[f'A_{origin}_{s}']=f.A;a[f'Q_{origin}_{s}']=f.Q
    tail=float(np.sum(mod.singular_values[4:]**2));res=float(np.linalg.norm((mod.delta-f.A@f.Q)@mod.lower)**2)
    row.update(rank=f.rank,full_factor_reconstruction_residual=diag['full_rank_correction_relative_residual'],rank4_right_solve_residual=rel(f.Q@mod.lower,mod.right_h[:4]),weighted_tail_identity_residual=abs(res-tail)/max(tail,np.finfo(float).tiny))
    if np.linalg.matrix_rank(f.A)!=4 or np.linalg.matrix_rank(f.Q)!=4:raise ValueError('Factor rank drift')
    if max(row[k] for k in ('full_factor_reconstruction_residual','rank4_right_solve_residual','weighted_tail_identity_residual'))>TOL:raise ValueError('Factor construction gate')
   audit.append(row)
  if any(not np.isfinite(v).all() for v in a.values()):raise ValueError('Operator NaN/Inf')
 for v in a.values():v.flags.writeable=False
 return a,audit,definition
def family(a,snrs):
 from src.pilot_cv import PilotCV,frozen_folds
 from src.cdl_benchmark import observation_geometry
 from src.lowrank_modules import RankFactors
 folds,_=frozen_folds(observation_geometry().selected)
 return {s:PilotCV(folds,a['source_mean_p'],tuple(a[f'KSource_{s}']),tuple(a[f'DeltaKP_{s}']),tuple(a[f'DeltaKM_{s}'])) for s in snrs},{(m,s):RankFactors(a[f'A_{m}_{s}'],a[f'Q_{m}_{s}']) for m in ('P','M') for s in snrs}
def load_operators():
 with np.load(OUT/'operators.npz',allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
 return a
def historical_operators():
 a={}
 for n in OLD_OBJECTS:
  with np.load(ROOT/'outputs'/n,allow_pickle=False) as f:
   a.update({k:f[k].copy() for k in f.files if k!='metadata_json'})
 for m in ('P','M'):
  for s in (10,20):a[f'A_{m}_{s}']=a[f'A_{m}_{s}'][:,:4];a[f'Q_{m}_{s}']=a[f'Q_{m}_{s}'][:4,:]
 return a
def communicate(seeds):
 from src.link import generate_master_qpsk
 from src.numerics import complex_gaussian_noise
 b,x=generate_master_qpsk(bits_seeds=(seeds['bits_seed_tx0'],seeds['bits_seed_tx1']))
 z=complex_gaussian_noise((2,64,16),1.,np.random.default_rng(seeds['noise_seed']))
 for v in (b,x,z):v.flags.writeable=False
 return b,x,z
def infer(pilots,a,s,cvs,factors):
 from src.frozen_affine_validation import deployed
 return deployed(pilots,a,s,cvs[s],factors['P',s],factors['M',s])
def evaluate(h,comm,a,matched,snrs=SNRS,reference_a=None):
 """Seven NMSE paths, frozen ZF BER, separate Perfect CSI BER reference.

 Only infer() is the deployment boundary; H/condition/matched never enter it.
 reference_a optionally checks every deployed prediction against old operators.
 """
 from src.cdl_benchmark import observation_geometry,frames,array_hash
 from src.link import assemble_pilot_data_frame,apply_frequency_mimo_channel,estimate_ls_linear_channel,extract_reference_data_bits,detect_data_zf,demodulate_data_bits
 g=observation_geometry();truths=frames(h);cvs,factors=family(a,snrs)
 if reference_a is not None:old_cvs,old_factors=family(reference_a,snrs)
 cache={};pilots={s:[] for s in snrs};inputs=[]
 for j,(bits,symbols,z) in enumerate(comm):
  clean=apply_frequency_mimo_channel(truths[j],assemble_pilot_data_frame(symbols,g.layout));ref=extract_reference_data_bits(bits,g.layout.data_mask)
  for s in snrs:
   y=clean+np.sqrt(variance(s))*z;y.flags.writeable=False
   sparse,linear=estimate_ls_linear_channel(y,g.layout);p=sparse.ravel()[g.selected];p.flags.writeable=False
   pilots[s].append(p);cache[s,j]=(y,sparse,linear,ref)
   inputs.append(dict(snr_db=s,frame_id=j,truth_sha256=array_hash(truths[j]),unit_noise_sha256=array_hash(z),pilot_sha256=array_hash(p),Y_sha256=array_hash(y)))
 n=len(snrs);fe=np.zeros((n,7,6));fq=np.zeros((n,6));fb=np.zeros((n,7,6),np.int64);fn=np.zeros((n,6),np.int64);perfect=np.zeros((n,6),np.int64);coords=np.zeros((n,6,2));normal=[];prediction_max=0.;coord_max=0.;rows=[]
 mapping={'source':'Source','hard':'Hard','convex':'Convex','dense-affine':'Dense','lr4-affine':'LR4'}
 for si,s in enumerate(snrs):
  fitted=infer(np.stack(pilots[s]),a,s,cvs,factors)
  if reference_a is not None:old=infer(np.stack(pilots[s]),reference_a,s,old_cvs,old_factors)
  if [r['frame_id'] for r in fitted]!=list(range(4,10)):raise ValueError('Score/warmup drift')
  for ji,row in enumerate(fitted):
   j=ji+4;y,sparse,linear,ref=cache[s,j];truth=truths[j];truth_d=truth.ravel()[g.data];diag=row['diagnostics']['affine'];cc=row['coefficients']['affine']
   if row['window_start']!=j-4 or row['dense_coefficients'] is not row['lr4_coefficients']:raise ValueError('Window/shared-coordinate drift')
   if diag['clipped'] or diag['regularized'] or not np.isfinite(np.r_[cc,np.array(diag['gram']).ravel(),diag['rhs'],diag['normal_equation_relative_residual']]).all():raise ValueError('Coordinate solve gate')
   if diag['normal_equation_relative_residual']>SOLVE_TOL:raise ValueError('Normal solve residual gate')
   coords[si,ji]=cc;normal.append(diag['normal_equation_relative_residual'])
   predictions={mapping[k]:v for k,v in row['predictions'].items()};predictions['LS-LIN']=linear.ravel()[g.data];predictions['Matched']=matched[s].estimate(pilots[s][j])
   if reference_a is not None:
    coord_max=max(coord_max,rel(cc,old[ji]['coefficients']['affine']))
    legacy={mapping[key]:old[ji]['predictions'][key] for key in mapping}
    legacy['LS-LIN']=estimate_ls_linear_channel(y,g.layout)[1].ravel()[g.data]
    legacy['Matched']=matched[s].estimate(pilots[s][j])
    for m in METHODS:prediction_max=max(prediction_max,rel(predictions[m],legacy[m]))
   fq[si,ji]=float(np.sum(abs(truth_d)**2));fn[si,ji]=sum(v.size for v in ref);before=array_hash(y)
   for mi,m in enumerate(METHODS):
    pred=predictions[m]
    if pred.shape!=(3968,) or not np.isfinite(pred).all():raise ValueError('Estimate shape/finite')
    if m=='LS-LIN':grid=linear
    else:
     v=sparse.copy().ravel();v[g.data]=pred;grid=v.reshape(2,2,64,16)
    detected=detect_data_zf(y,grid,g.layout.data_mask)*np.sqrt(2)
    if not np.isfinite(detected[:,g.layout.data_mask[0]]).all():raise ValueError('ZF nonfinite')
    bits_hat=demodulate_data_bits(detected,g.layout.data_mask)
    fe[si,mi,ji]=float(np.sum(abs(pred-truth_d)**2));fb[si,mi,ji]=sum(int(np.count_nonzero(v!=r)) for v,r in zip(bits_hat,ref))
    if array_hash(y)!=before:raise ValueError('Shared input mutation')
    coef=row['coefficients']['hard' if m=='Hard' else 'convex' if m=='Convex' else 'affine'] if m in ('Hard','Convex','Dense','LR4') else (None,None)
    rows.append(dict(condition=None,snr_db=s,estimator=m,trajectory_id=None,frame_id=j,error_energy=fe[si,mi,ji],truth_energy=fq[si,ji],bit_errors=int(fb[si,mi,ji]),bit_total=int(fn[si,ji]),alpha=None if coef[0] is None else float(coef[0]),beta=None if coef[1] is None else float(coef[1]),solve_status='PASS' if m in ('Dense','LR4') else 'NOT_APPLICABLE'))
   bits_hat=demodulate_data_bits(detect_data_zf(y,truth,g.layout.data_mask)*np.sqrt(2),g.layout.data_mask)
   perfect[si,ji]=sum(int(np.count_nonzero(v!=r)) for v,r in zip(bits_hat,ref))
 if prediction_max>TOL or coord_max>TOL:raise ValueError('ANCHOR_REPLAY_FAILURE prediction/coordinate')
 result=dict(frame_error=fe,frame_truth=fq,frame_bit_errors=fb,frame_bit_total=fn,coordinates=coords,perfect_frame_bit_errors=perfect)
 validate_raw(result,snrs)
 return result,rows,dict(finite=True,solve_success=True,normal_residual_max=max(normal),prediction_relative_error_max=prediction_max,coordinate_relative_error_max=coord_max,inputs=inputs,shared_dense_lr4_coordinates=True)
def validate_raw(raw,snrs):
 n=len(snrs);shapes={'frame_error':(n,7,6),'frame_truth':(n,6),'frame_bit_errors':(n,7,6),'frame_bit_total':(n,6),'coordinates':(n,6,2),'perfect_frame_bit_errors':(n,6)}
 if set(raw)!=set(shapes):raise ValueError('Raw bootstrap-input fields')
 for k,shape in shapes.items():
  if raw[k].shape!=shape or not np.isfinite(raw[k]).all():raise ValueError('Raw schema shape/finite')
  if k in ('frame_bit_errors','frame_bit_total','perfect_frame_bit_errors') and raw[k].dtype!=np.int64:raise ValueError('Bit count dtype must be int64')
 if np.any(raw['frame_error']<=0) or np.any(raw['frame_truth']<=0) or np.any(raw['frame_bit_errors']<0) or np.any(raw['frame_bit_errors']>raw['frame_bit_total'][:,None,:]):raise ValueError('Raw energy/bit validity')
 if np.any(raw['perfect_frame_bit_errors']<0) or np.any(raw['perfect_frame_bit_errors']>raw['frame_bit_total']):raise ValueError('Perfect-reference counts')
 return True
def planned_contrasts():
 rows=[]
 for condition in ('Profile','Mobility','Composed'):
  for s in SNRS:rows.append(dict(condition=condition,snr_db=s,comparison='source_gain',first='Source',second='LR4',positive_means='LR4 better'))
 for comparison,first,second,meaning in [('convex_gain','Convex','LR4','LR4 better'),('compression_penalty','LR4','Dense','compression worse')]:
  for s in SNRS:rows.append(dict(condition='Composed',snr_db=s,comparison=comparison,first=first,second=second,positive_means=meaning))
 return rows
def classify(lower,upper):
 if not np.isfinite([lower,upper]).all() or lower>upper:raise ValueError('Interval ordering')
 return 'POSITIVE' if lower>0 else 'NEGATIVE' if upper<0 else 'INDETERMINATE'
def raw_to_trajectory(raw,snrs):
 validate_raw(raw,snrs);n=len(snrs);e=np.zeros((n,7));q=np.zeros(n);be=np.zeros((n,7),np.int64);bt=np.zeros(n,np.int64)
 for j in range(6):e+=raw['frame_error'][:,:,j];q+=raw['frame_truth'][:,j];be+=raw['frame_bit_errors'][:,:,j];bt+=raw['frame_bit_total'][:,j]
 return e,q,be,bt
def require_freeze():
 f=read(OUT/'execution_freeze.json')
 for path,h in f['hashes'].items():
  if sha(ROOT/path)!=h:raise RuntimeError('Frozen execution bytes changed: '+path)
 if (OUT/'STOP.json').exists():raise RuntimeError('Permanent validation STOP')
 return f
def require_future_authorization(path,purpose):
 if path is None:raise PermissionError('A separate future user-authorized B2 record is required')
 a=read(path);manifest=read(ROOT/'outputs/sp_u12b2a_formal_manifest.json')
 if a.get('task')!='SP-U12B2' or a.get('purpose')!=purpose or a.get('manifest_sha256')!=sha(ROOT/'outputs/sp_u12b2a_formal_manifest.json'):raise PermissionError('Future authorization identity')
 require_freeze()
 if sha(ROOT/manifest['operator_cache']['path'])!=manifest['operator_cache']['sha256']:raise PermissionError('Frozen operator cache drift')
 for p,h in manifest['formal_metadata_hashes'].items():
  if sha(ROOT/p)!=h:raise PermissionError('Frozen formal metadata drift')
 if purpose=='unseal-analysis':
  if not a.get('mentor_release') or a.get('mfa_status')!='FINAL_MODEL_FROZEN':raise PermissionError('B2 stays sealed until final MFA freeze / explicit mentor release')
  p=ROOT/a['mfa_model_path']
  if sha(p)!=a['mfa_model_sha256']:raise PermissionError('Final MFA model hash')
 return manifest
