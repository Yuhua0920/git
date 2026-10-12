"""Frozen MR08A trajectory-paired bootstrap; no channels or predictions here.

Three separately interpreted35-comparison families. The bootstrap draws for a
condition are shared across every SNR and method. Raw int32 RNG outputs are
losslessly stored as uint16 NPY indices. All outputs are exclusive new files.
"""
from pathlib import Path
import csv,hashlib,json,math,zipfile
import numpy as np

CONDITIONS=('ID','Profile','Mobility','Composed','Boundary')
SNRS=(0,5,10,15,20,25,30)
METHODS=('SOURCE','POOLED_NATURAL','PPCA_DIRECT_R32','GMM_PREPARED_3','HARD_TW5','CONVEX_TW5','LR4_TW1','LR4_TW5','DENSE_TW5','WHOLE_POOL_M2_R24')
FAMILIES=(('A','POOLED_NATURAL','LR4_TW5'),('B','GMM_PREPARED_3','LR4_TW1'),('C','LR4_TW5','WHOLE_POOL_M2_R24'))
REPLICATES=100000
FAMILY_SIZE=35
TAIL=.05/(2*FAMILY_SIZE)

def sha(path):
    with Path(path).open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as h:json.dump(value,h,indent=2,ensure_ascii=False,allow_nan=False)

def csvwrite(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def exclusive_array(path,shape,dtype):
    """Create an exclusive valid NPY and return its writable data mapping."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);dtype=np.dtype(dtype)
    with path.open('xb') as h:
        np.lib.format.write_array_header_1_0(h,dict(descr=np.lib.format.dtype_to_descr(dtype),fortran_order=False,shape=tuple(shape)))
        offset=h.tell();h.truncate(offset+math.prod(shape)*dtype.itemsize)
    return np.memmap(path,dtype=dtype,mode='r+',offset=offset,shape=shape,order='C')

def store_indices(path,master,condition_index,n,replicates,block=1000):
    """Same trusted SeedSequence/default_rng/int32 stream, new frozen master."""
    if not isinstance(master,(int,np.integer)) or isinstance(master,(bool,np.bool_)) or master<0:raise ValueError('BOOTSTRAP_MASTER_INTEGER')
    if condition_index not in range(5) or n<=0 or n>65536 or replicates<=0 or block<=0:raise ValueError('INDEX_GRID')
    rng=np.random.default_rng(np.random.SeedSequence([int(master),int(condition_index)]))
    result=exclusive_array(path,(replicates,n),np.uint16)
    for start in range(0,replicates,block):
        take=min(block,replicates-start);ids=rng.integers(0,n,size=(take,n),dtype=np.int32)
        result[start:start+take]=ids.astype(np.uint16,copy=False)
    result.flush();del result
    return dict(path=str(path),sha256=sha(path),shape=[replicates,n],stored_dtype='uint16',generation_dtype='int32',rng='np.random.default_rng(np.random.SeedSequence([bootstrap_master,condition_index]))',master=int(master),condition_index=int(condition_index),n=n,replicates=replicates,lossless_cast=True)

def aggregate(fe,fq):
    if np.asarray(fe).dtype!=np.float64 or np.asarray(fq).dtype!=np.float64:raise ValueError('ENERGY_FLOAT64_REQUIRED')
    fe=np.asarray(fe,dtype=np.float64);fq=np.asarray(fq,dtype=np.float64)
    if fe.ndim!=4 or fq.shape!=(fe.shape[0],fe.shape[-1]):raise ValueError('FRAME_ENERGY_AXES')
    if not np.isfinite(fe).all() or not np.isfinite(fq).all() or np.any(fe<0) or np.any(fq<=0):raise ValueError('FRAME_ENERGY_FINITE_OR_SIGN')
    te=fe.sum(axis=-1);tq=fq.sum(axis=-1);linear=te.sum(axis=0)/tq.sum()
    if not np.isfinite(linear).all() or np.any(linear<=0):raise ValueError('POOLED_NMSE_NOT_POSITIVE')
    return te,tq,linear

def bootstrap_values(te,tq,ids):
    """Counts multiply sufficient trajectory energies; no frame resampling."""
    te=np.asarray(te,dtype=np.float64);tq=np.asarray(tq,dtype=np.float64);ids=np.asarray(ids)
    n,s,m=te.shape
    if tq.shape!=(n,) or ids.ndim!=2 or ids.shape[1]!=n or ids.dtype.kind not in 'iu' or ids.size==0 or ids.min()<0 or ids.max()>=n:raise ValueError('PAIRED_DRAW_AXES_OR_RANGE')
    counts=np.stack([np.bincount(row.astype(np.int64,copy=False),minlength=n) for row in ids])
    num=counts@te.reshape(n,s*m);den=counts@tq
    result=(num/den[:,None]).reshape(len(ids),s,m)
    if not np.isfinite(result).all() or np.any(result<=0):raise ValueError('BOOTSTRAP_NMSE_NOT_POSITIVE')
    return result

def primary_differences(linear,methods=METHODS):
    linear=np.asarray(linear,dtype=np.float64)
    if not np.isfinite(linear).all() or np.any(linear<=0) or linear.shape[-1]!=len(methods):raise ValueError('DIFFERENCE_NMSE_AXES')
    db=10*np.log10(linear)
    return np.stack([db[...,methods.index(first)]-db[...,methods.index(second)] for _,first,second in FAMILIES],axis=-1)

def condition_statistics(fe,fq,indices_path,linear_path,difference_path,*,block=512):
    te,tq,point=aggregate(fe,fq);indices=np.load(indices_path,mmap_mode='r',allow_pickle=False)
    n,s,m=te.shape;r=len(indices)
    if indices.shape!=(r,n) or indices.dtype!=np.uint16 or block<=0:raise ValueError('PERSISTED_INDEX_AXES')
    linear=exclusive_array(linear_path,(r,s,m),np.float64);difference=exclusive_array(difference_path,(r,s,3),np.float64)
    for start in range(0,r,block):
        stop=min(start+block,r);bl=bootstrap_values(te,tq,indices[start:stop]);linear[start:stop]=bl;difference[start:stop]=primary_differences(bl)
    linear.flush();difference.flush();point_difference=primary_differences(point)
    del linear,difference,indices
    return dict(trajectory_error=te,trajectory_truth=tq,point_linear=point,point_db=10*np.log10(point),point_difference=point_difference,replicates=r)

def adjusted_intervals(difference_path):
    difference=np.load(difference_path,mmap_mode='r',allow_pickle=False)
    if difference.ndim!=3 or difference.shape[-1]!=3 or not np.isfinite(difference).all():raise ValueError('PRIMARY_DIFFERENCE_ARRAY')
    return np.quantile(difference,[TAIL,1-TAIL],axis=0,method='linear')

def load_raw(out,mode,n):
    receipt=read(out/'reports'/f'{mode}_receipt.json')
    if receipt.get('PASS') is not True or receipt.get('failures',0)!=0 or receipt.get('nonfinite',0)!=0 or receipt.get('missing',0)!=0:raise ValueError('RAW_NOT_COMPLETE_OR_FAILED')
    entries=receipt['receipts'];ix={(z['condition'],int(z['trajectory_id'])):z for z in entries};expected={(c,i) for c in CONDITIONS for i in range(n)}
    if len(entries)!=n*5 or set(ix)!=expected:raise ValueError('RAW_EXACT_IDENTITY_GRID')
    # A formal dataset needs explicit successful input-identity qualification.
    if mode=='formal' and receipt.get('identity_PASS') is not True:raise ValueError('FORMAL_IDENTITY_NOT_PASS')
    errors=np.empty((5,n,7,10,6),np.float64);truth=np.empty((5,n,6),np.float64)
    for ci,c in enumerate(CONDITIONS):
        for i in range(n):
            r=ix[c,i];path=(out/r['path']).resolve();path.relative_to(out);jp=path.with_suffix('.json')
            if sha(path)!=r['SHA256'] or sha(jp)!=r['JSON_SHA256']:raise ValueError('RAW_HASH')
            meta=read(jp)
            if (meta['mode'],meta['condition'],int(meta['trajectory_id']),meta['complete'])!=(mode,c,i,True):raise ValueError('RAW_JSON_IDENTITY')
            with zipfile.ZipFile(path) as z:
                if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):raise ValueError('RAW_NPZ_CRC')
            with np.load(path,allow_pickle=False) as z:
                if list(z['method_names'])!=list(METHODS) or list(z['snr_db'])!=list(SNRS) or list(z['frame_ids'])!=list(range(4,10)):raise ValueError('RAW_METHOD_SNR_FRAME_AXES')
                if z['frame_error'].dtype!=np.float64 or z['frame_truth'].dtype!=np.float64 or z['frame_error'].shape!=(7,10,6) or z['frame_truth'].shape!=(6,) or np.any(z['failure_flags']):raise ValueError('RAW_SHAPE_OR_FAILED_FRAME')
                errors[ci,i]=z['frame_error'];truth[ci,i]=z['frame_truth']
    for ci in range(5):aggregate(errors[ci],truth[ci])
    return errors,truth,receipt

def analyze_dataset(out,mode,bootstrap_master,replicates=REPLICATES,n=1000):
    """Root-controlled entry; call only after all raw blocks and identity PASS."""
    out=Path(out).resolve()
    if mode not in ('formal','smoke'):raise ValueError('MODE_FORMAL_OR_SMOKE')
    if mode=='formal' and (replicates!=REPLICATES or n!=1000):raise ValueError('FROZEN_FORMAL_COUNTS')
    if replicates<=0 or n<=0:raise ValueError('POSITIVE_COUNTS')
    errors,truth,raw=load_raw(out,mode,n);nmse_rows=[];primary_rows=[];trajectory_rows=[];records=[]
    for ci,c in enumerate(CONDITIONS):
        directory=out/'data/bootstrap'/mode/c;indices=directory/'indices.npy';linear=directory/'bootstrap_linear.npy';difference=directory/'primary_differences.npy'
        draw_meta=store_indices(indices,bootstrap_master,ci,n,replicates)
        draw_meta['path']=str(indices.relative_to(out))
        z=condition_statistics(errors[ci],truth[ci],indices,linear,difference)
        bounds=adjusted_intervals(difference) if mode=='formal' else None
        for si,s in enumerate(SNRS):
            for mi,m in enumerate(METHODS):
                nmse_rows.append(dict(mode=mode,condition=c,snr_db=s,method=m,error_energy=float(z['trajectory_error'][:,si,mi].sum()),truth_energy=float(z['trajectory_truth'].sum()),nmse_linear=float(z['point_linear'][si,mi]),nmse_db=float(z['point_db'][si,mi]),trajectory_count=n,scored_frames=n*6,scope='independent confirmatory trajectories' if mode=='formal' else 'synthetic/independent smoke only; no performance inference'))
                for t in range(n):trajectory_rows.append(dict(mode=mode,condition=c,trajectory_id=t,snr_db=s,method=m,error_energy=float(z['trajectory_error'][t,si,mi]),truth_energy=float(z['trajectory_truth'][t])))
            for fi,(family,first,second) in enumerate(FAMILIES):
                lo=None if bounds is None else float(bounds[0,si,fi]);hi=None if bounds is None else float(bounds[1,si,fi])
                classification='SMOKE_ONLY' if bounds is None else ('SECOND_SUPPORTED' if lo>0 else 'FIRST_SUPPORTED' if hi<0 else 'INDETERMINATE')
                primary_rows.append(dict(mode=mode,family=family,condition=c,snr_db=s,first=first,second=second,estimate_db=float(z['point_difference'][si,fi]),CI_lower=lo,CI_upper=hi,classification=classification,trajectory_count=n,scored_frames=n*6,bootstrap=replicates,family_size=FAMILY_SIZE,tail_probability=TAIL if bounds is not None else None,sign='first NMSE(dB) minus second; positive favors second',interval='Separate35-comparison family Bonferroni-adjusted two-sided percentile interval; nominal95% within family; no combined105-comparison guarantee' if bounds is not None else 'No confidence interval or inferential conclusion for smoke'))
        records.append(dict(condition=c,indices=draw_meta,bootstrap_linear=dict(path=str(linear.relative_to(out)),SHA256=sha(linear),shape=[replicates,7,10]),primary_differences=dict(path=str(difference.relative_to(out)),SHA256=sha(difference),shape=[replicates,7,3])))
    prefix='formal' if mode=='formal' else 'smoke'
    csvwrite(out/'tables'/f'{prefix}_nmse.csv',nmse_rows);csvwrite(out/'tables'/f'{prefix}_primary_contrasts.csv',primary_rows);csvwrite(out/'tables'/f'{prefix}_trajectory_energies.csv',trajectory_rows)
    answer=dict(PASS=True,mode=mode,bootstrap_master=int(bootstrap_master),methods=list(METHODS),conditions=list(CONDITIONS),snrs=list(SNRS),NMSE_rows=len(nmse_rows),primary_rows=len(primary_rows),families=[dict(family=f,comparisons=35,first=a,second=b) for f,a,b in FAMILIES],physical_trajectories=n*5,scored_positions_per_method=n*5*7*6,bootstrap_replicates=replicates,paired_across_all_SNRs_and_methods=True,independent_unit='trajectory',not_three_independent_experiments=True,no_combined105_familywise_guarantee=True,smoke_uncertainty_disabled=mode=='smoke',records=records,raw_receipt_SHA256=sha(out/'reports'/f'{mode}_receipt.json'))
    write(out/'reports'/f'{mode}_analysis.json',answer)
    return answer



