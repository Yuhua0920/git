"""SP-U7B orchestration boundaries. No scientific algorithm is implemented here."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'  # fixed resource plan, selected before scientific generation
import csv
import hashlib
import json
import inspect
import sys
import time
import ctypes
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT = ROOT/'outputs/sp_u7b'
PROTOCOL = ROOT/'configs/sp_u7a_protocol.json'
SEEDS = ROOT/'outputs/sp_u7a_future_seed_manifest.csv'
METHODS = ('LS','Source','Hard','Convex','Dense','LR4','Matched')
SNRS = (10,20)
VARIANCES = {10:0.9941134822523973,20:0.09941134822523973}
ROLES = ('sionna_seed','torch_seed','numpy_seed','bits_seed_tx0','bits_seed_tx1','noise_seed')
NAMESPACE = 'SP_U7B_OOD_CDLA120_V1'
NEW_SOURCES = ('scripts/sp_u7b_common.py','scripts/run_sp_u7b.py','scripts/analyze_sp_u7b.py',
               'scripts/plot_sp_u7b.py','tests/test_sp_u7b.py')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def utc():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def state_path():return OUT/'state.json'

def transition(expected,new,**extra):
    s=read(state_path())
    if s['phase']!=expected:raise RuntimeError('illegal phase transition '+s['phase']+' -> '+new)
    s.update(phase=new,updated_utc=utc(),**extra)
    temp=OUT/'state.tmp'
    temp.write_text(json.dumps(s,indent=2,allow_nan=False)+'\n',encoding='utf8')
    temp.replace(state_path())

def verify(hashes):
    bad=[p for p,h in hashes.items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
    if bad:raise RuntimeError('frozen byte drift: '+repr(bad))

def verify_frozen():
    f=read(OUT/'freeze.json');verify(f['hashes']);return f

def seed_rows():
    with SEEDS.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))

def seed_index():
    rows=seed_rows()
    return {(r['group'],int(r['trajectory_id']),int(r['frame_id']),r['role']):int(r['seed']) for r in rows}

def consumed(index,group,i,j=0):
    return {r:index[group,i,-1 if r in ROLES[:3] else j,r] for r in ROLES if (group,i,-1 if r in ROLES[:3] else j,r) in index}

def seed_audit():
    rows=seed_rows();values=[int(r['seed']) for r in rows]
    history=read(ROOT/'outputs/sp_u7a_historical_seed_union.json')
    keys={(r['group'],int(r['trajectory_id']),int(r['frame_id']),r['role']) for r in rows}
    expected={(g,i,-1,r) for g,n in (('EVAL',1000),('MATCHED_REF',500)) for i in range(n) for r in ROLES[:3]}
    expected|={('EVAL',i,j,r) for i in range(1000) for j in range(10) for r in ROLES[3:]}
    bootstrap=[r for r in rows if r['role']=='bootstrap_seed']
    # The bootstrap group/key is retained exactly as allocated by U7A.
    remaining=keys-expected
    valid=(len(rows)==34501 and len(set(values))==34501 and len(keys)==34501 and len(remaining)==1
           and expected<=keys and len(bootstrap)==1 and int(bootstrap[0]['seed'])==764389531
           and all(r['namespace']==NAMESPACE for r in rows))
    overlap=set(values)&set(history['forbidden_seeds'])
    if not valid or overlap:raise RuntimeError('immutable seed allocation audit failed')
    return dict(rows=len(rows),unique_individual_seeds=len(set(values)),historical_overlap_count=len(overlap),
                namespace=NAMESPACE,manifest_sha256=sha(SEEDS),historical_union_sha256=sha(ROOT/'outputs/sp_u7a_historical_seed_union.json'),
                internal_Sionna_derivation='config.seed resets explicit generator per available device to (sionna_seed + device_index) mod 2^64; cuda:0 index=1. Derived SDK state, not an independently allocated seed.',
                rng_roles={'sionna_seed':'Sionna explicit torch.Generator(device): ray coupling and initial ray phases',
                           'torch_seed':'torch.manual_seed global generators: frozen adapter compatibility; explicit Sionna generators unaffected',
                           'numpy_seed':'local default_rng PCG64: frozen adapter compatibility, no coefficient sampling uses it',
                           'bits_seed_tx0':'NumPy PCG64 -> Tx0 QPSK bits', 'bits_seed_tx1':'NumPy PCG64 -> Tx1 QPSK bits',
                           'noise_seed':'NumPy PCG64 -> unit complex Gaussian real then imaginary samples, same atom for both SNRs',
                           'bootstrap_seed':'NumPy PCG64 -> one 10000 x 1000 trajectory index table'})

def historical_snapshot():
    """Hash file inventory; read only newly changed seed metadata and pure H hashes."""
    from scripts.sp_u7a_metadata import project_json,csv_metadata,source_seeds
    forbidden=set(read(ROOT/'outputs/sp_u7a_historical_seed_union.json')['forbidden_seeds'])
    baseline=read(ROOT/'outputs/sp_u7a_metadata_inventory.json') if (ROOT/'outputs/sp_u7a_metadata_inventory.json').exists() else None
    # Fresh metadata projection includes current sources/seed manifests; no numeric outcome conversion.
    additions=set();hdigests=set();inventory={}
    import re
    for parent in ('scripts','tests','configs','outputs'):
        for p in (ROOT/parent).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or 'sp_u7b' in p.parts or 'sp_u7a' in p.name or 'sp_u7b' in p.name:continue
            rel=p.relative_to(ROOT).as_posix()
            inventory[rel]=dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
            # Pure seed manifests and pure H-hash manifests only. Never decode any outcome column.
            if p.suffix=='.csv' and ('seed' in p.name or ('hash' in p.name and ('_h_' in p.name or 'channel' in p.name))):
                with p.open(encoding='utf8',newline='') as f:
                    reader=csv.DictReader(f)
                    keep=[k for k in (reader.fieldnames or []) if 'seed' in k.lower() and 'sha' not in k.lower()]
                    hashes=[k for k in (reader.fieldnames or []) if 'sha256' in k.lower() and ('_h' in k.lower() or k.lower().startswith('h_') or 'channel' in k.lower())]
                    for row in reader:
                        for k in keep:
                            v=row.get(k,'')
                            if re.fullmatch(r'\d+',v or ''):additions.add(int(v))
                        for k in hashes:
                            v=row.get(k,'')
                            if re.fullmatch(r'[0-9a-f]{64}',v or ''):hdigests.add(v)
            elif p.suffix=='.py':additions|=source_seeds(p.read_text(encoding='utf8'))
            elif p.suffix in ('.json','.jsonl'):
                # Token projection converts ONLY seed numeric fields. All outcome
                # numerics are discarded as uninterpreted lexical tokens.
                payload=p.read_text(encoding='utf8')
                additions|=project_json(payload)['seeds']
                hdigests.update(re.findall(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])',payload))
    selected={int(r['seed']) for r in seed_rows()}
    if selected&(additions|forbidden):raise RuntimeError('fresh historical seed overlap')
    return dict(inventory=inventory,additional_seed_count=len(additions-forbidden),historical_H_hashes=sorted(hdigests),
                historical_H_hash_count=len(hdigests),seed_overlap_count=0,projection='only seed columns/AST seed literals/JSON seed token projection/H-hash columns and conservative SHA256 superset; no historical outcome conversion')

def guard(mode='generation'):
    """Runtime file-open firewall; only explicitly frozen historical arrays are allowed."""
    allow={str((ROOT/p).resolve()).lower() for p in read(ROOT/'outputs/sp_u7a_protected_inputs.json')['hashes'] if p.endswith('.npz')}
    # Only the six deployable arrays, never target/reference artifacts from history.
    allow={str((ROOT/p).resolve()).lower() for p in read(PROTOCOL)['deployable']['objects']}
    meta={str((ROOT/'outputs'/n).resolve()).lower() for n in ('sp_u7a_protected_inputs.json','sp_u7a_future_seed_manifest.csv',
          'sp_u7a_historical_seed_union.json','sp_u7a_model_audit.json','sp_u7a_delivery_manifest.json',
          'sp_u7a_exposure_audit.json','sp_u7a_seed_audit.json','sp_u7a_artifact_audit.json')}
    outputroot=str((ROOT/'outputs').resolve()).lower()+os.sep
    own=str(OUT.resolve()).lower()+os.sep
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=str(Path(os.fsdecode(args[0])).resolve()).lower()
        omode=args[1] if len(args)>1 else 'r'
        flags=args[2] if len(args)>2 else 0
        writing=(isinstance(omode,str) and any(c in omode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR))
        if writing:return
        if p.startswith(own):
            return
        if p.startswith(outputroot) and p not in allow and p not in meta:
            raise PermissionError('historical outcome firewall: '+Path(p).name)
    sys.addaudithook(hook)

def resources():
    """Windows RAM status without new package installation."""
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total','avail','page_total','page_avail','virtual','virtual_avail','extended')]
    m=Memory();m.length=ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return dict(physical_RAM_load_percent=m.load,available_RAM_GiB=m.avail/2**30)

def load_deployables():
    from src.lowrank_modules import RankFactors
    from src.pilot_cv import PilotCV,frozen_folds
    from src.cdl_benchmark import observation_geometry
    arrays={}
    for name in ('t16e_frozen_operators.npz','t16f_frozen_candidates.npz'):
        with np.load(ROOT/'outputs'/name,allow_pickle=False) as f:
            for k in f.files:
                if k=='metadata_json':continue
                arrays[k]=f[k].copy();arrays[k].flags.writeable=False
    family={}
    with np.load(ROOT/'outputs/t16g_frozen_module_factors.npz',allow_pickle=False) as f:
        for s in SNRS:
            for m in ('P','M'):family[m,s,4]=RankFactors(f[f'A_{m}_{s}'][:,:4],f[f'Q_{m}_{s}'][:4,:])
    folds,_=frozen_folds(observation_geometry().selected)
    cvs={s:PilotCV(folds,arrays['source_mean_p'],tuple(arrays[f'KSource_{s}']),tuple(arrays[f'DeltaKP_{s}']),tuple(arrays[f'DeltaKM_{s}'])) for s in SNRS}
    return arrays,family,cvs

def infer_pilots(pilots,arrays,family,cvs,snr):
    """Deployment boundary: no clean H, target label or reference object accepted."""
    from src.frozen_affine_validation import deployed
    return deployed(pilots,arrays,snr,cvs[snr],family['P',snr,4],family['M',snr,4])

def communication(s):
    from src.link import generate_master_qpsk
    from src.numerics import complex_gaussian_noise
    bits,symbols=generate_master_qpsk(bits_seeds=(s['bits_seed_tx0'],s['bits_seed_tx1']))
    z=complex_gaussian_noise((2,64,16),1.,np.random.default_rng(s['noise_seed']))
    for a in (bits,symbols,z):a.flags.writeable=False
    return bits,symbols,z

def evaluate(h,comm,arrays,family,cvs,matched):
    """Frozen seven-method orchestration; numerical components reused verbatim."""
    from src.cdl_benchmark import observation_geometry,frames,array_hash
    from src.link import (assemble_pilot_data_frame,apply_frequency_mimo_channel,estimate_ls_linear_channel,
                         detect_data_zf,demodulate_data_bits,extract_reference_data_bits)
    g=observation_geometry();truth_frames=frames(h);cache={};pilots={s:[] for s in SNRS}
    inputs=[];output_hashes=[];windows=[];coordinate=[]
    power=dict(signal_energy=0.,unit_noise_energy=0.,receive_samples=0)
    for j,(bits,symbols,z) in enumerate(comm):
        truth=truth_frames[j];x=assemble_pilot_data_frame(symbols,g.layout)
        clean=apply_frequency_mimo_channel(truth,x)
        if j>=4:
            mask=g.layout.data_mask[0]
            power['signal_energy']+=float(np.sum(abs(clean[:,mask])**2));power['unit_noise_energy']+=float(np.sum(abs(z[:,mask])**2));power['receive_samples']+=2*int(mask.sum())
        ref=extract_reference_data_bits(bits,g.layout.data_mask)
        for s in SNRS:
            y=clean+np.sqrt(VARIANCES[s])*z;y.flags.writeable=False
            sparse,linear=estimate_ls_linear_channel(y,g.layout)
            pilot=sparse.ravel()[g.selected];pilot.flags.writeable=False
            pilots[s].append(pilot);cache[s,j]=(y,sparse,linear,ref)
            inputs.append(dict(frame_id=j,snr_db=s,H_sha256=array_hash(truth),bits_sha256=array_hash(bits),
                symbols_sha256=array_hash(symbols),unit_noise_sha256=array_hash(z),Y_sha256=array_hash(y),pilot_sha256=array_hash(pilot)))
    fe=np.zeros((2,7,6));fq=np.zeros((2,6));fb=np.zeros((2,7,6),np.int64);fn=np.zeros((2,6),np.int64)
    shared=True
    for si,s in enumerate(SNRS):
        rows=infer_pilots(np.stack(pilots[s]),arrays,family,cvs,s)
        if [r['frame_id'] for r in rows]!=list(range(4,10)):raise ValueError('causal score rows')
        for row in rows:
            j=row['frame_id'];ji=j-4;y,sparse,linear,ref=cache[s,j]
            if row['dense_coefficients'] is not row['lr4_coefficients'] or not np.array_equal(row['dense_coefficients'],row['lr4_coefficients']):raise ValueError('coordinate identity drift')
            if row['window_start']!=j-4:raise ValueError('window drift')
            diag=row['diagnostics']['affine'];gram=np.array(diag['gram']);c=row['coefficients']['affine'];ev=np.linalg.eigvalsh(gram)
            values=np.r_[c,gram.ravel(),diag['rhs'],ev,diag['condition_number'],diag['normal_equation_relative_residual']]
            if not np.isfinite(values).all():raise ValueError('nonfinite coordinate/Gram diagnostics')
            coordinate.append(dict(snr_db=s,frame_id=j,alpha=float(c[0]),beta=float(c[1]),norm=float(np.linalg.norm(c)),
                 gram=diag['gram'],rhs=diag['rhs'],eigenvalues=ev.tolist(),gram_rank=int(np.count_nonzero(ev>max(ev)*2*np.finfo(float).eps)),
                 condition=diag['condition_number'],normal_residual=diag['normal_equation_relative_residual']))
            windows.append(dict(snr_db=s,frame_id=j,frames=list(range(j-4,j+1)),dense_lr4_same_c=True,rank_P=family['P',s,4].rank,rank_M=family['M',s,4].rank,
                                clipped=diag['clipped'],regularized=diag['regularized'],solve_success=True))
            mapping={'source':'Source','hard':'Hard','convex':'Convex','dense-affine':'Dense','lr4-affine':'LR4'}
            predictions={mapping[k]:v for k,v in row['predictions'].items()}
            predictions['LS']=linear.ravel()[g.data]
            predictions['Matched']=matched[s].estimate(pilots[s][j])
            true=truth_frames[j].ravel()[g.data];energy=float(np.sum(abs(true)**2))
            fq[si,ji]=energy;fn[si,ji]=sum(a.size for a in ref)
            before=array_hash(y)
            for mi,m in enumerate(METHODS):
                pred=predictions[m]
                if pred.shape!=(3968,) or not np.isfinite(pred).all():raise ValueError('prediction schema/finite')
                if m=='LS':grid=linear
                else:
                    value=sparse.copy().ravel();value[g.data]=pred;grid=value.reshape(2,2,64,16)
                detected=detect_data_zf(y,grid,g.layout.data_mask)*np.sqrt(2)
                if not np.isfinite(detected[:,g.layout.data_mask[0]]).all():raise ValueError('nonfinite data ZF')
                bhat=demodulate_data_bits(detected,g.layout.data_mask)
                fe[si,mi,ji]=float(np.sum(abs(pred-true)**2))
                fb[si,mi,ji]=sum(int(np.count_nonzero(a!=b)) for a,b in zip(bhat,ref))
                shared=shared and array_hash(y)==before
                output_hashes.append(dict(snr_db=s,frame_id=j,method=m,prediction_sha256=array_hash(pred),shared_Y_sha256=before,
                                          shared_pilot_sha256=array_hash(pilots[s][j])))
    if not shared:raise ValueError('paired input mutation')
    # Frozen original sequential frame accumulation, no dB reduction here.
    e=np.zeros((2,7));q=np.zeros(2);be=np.zeros((2,7),np.int64);nb=np.zeros(2,np.int64)
    for j in range(6):e+=fe[:,:,j];q+=fq[:,j];be+=fb[:,:,j];nb+=fn[:,j]
    if not np.isfinite(fe).all() or not np.isfinite(fq).all() or np.any(e<=0) or np.any(q<=0) or np.any(be>nb[:,None]):raise ValueError('invalid raw energies/counts')
    result=dict(error=e,truth=q,bit_errors=be,total_bits=nb,frame_error=fe,frame_truth=fq,frame_bit_errors=fb,frame_total_bits=fn,
                coordinates=np.array([[c['snr_db'],c['frame_id'],c['alpha'],c['beta'],c['norm'],c['gram_rank'],c['condition'],c['normal_residual'],*np.array(c['gram']).ravel(),*c['rhs'],*c['eigenvalues']] for c in coordinate]),
                power=np.array([power[k] for k in ('signal_energy','unit_noise_energy','receive_samples')]))
    evidence=dict(inputs=inputs,outputs=output_hashes,windows=windows,required_finite=True,paired_inputs_immutable=True,
                  deployable_boundary='infer_pilots(pilots, frozen_arrays, frozen_factors, frozen_cvs, snr)',target_clean_CSI_passed_to_deployables=False,
                  matched_reference_passed_to_deployables=False,counts=dict(trajectory_metrics=14,frame_metrics=84,inputs=20,coordinates=12),
                  coordinate_schema=['snr_db','frame_id','alpha','beta','norm','gram_rank','condition','normal_residual','gram00','gram01','gram10','gram11','rhs0','rhs1','eigen_min','eigen_max'])
    return result,evidence
