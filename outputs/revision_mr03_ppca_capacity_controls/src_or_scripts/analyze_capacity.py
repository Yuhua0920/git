"""Recompute exploratory statistics from persisted, authenticated small energies."""
from pathlib import Path
import json,csv,hashlib,zipfile,argparse,time
import numpy as np

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def csvout(p,rows):
    with Path(p).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def npz(p,**x):
    with Path(p).open('xb') as f:np.savez_compressed(f,**x)
INTERVAL='Unadjusted exploratory 95% trajectory-paired percentile interval; not confirmatory'

def definitions(methods):
    d=[]
    if 'MFA_K1_R16_EM_FROZEN' in methods:d.append(('EM16_minus_Direct16','MFA_K1_R16_EM_FROZEN','PPCA_DIRECT_R16'))
    d += [('Direct16_minus_Direct32','PPCA_DIRECT_R16','PPCA_DIRECT_R32'),('Direct32_minus_Direct64','PPCA_DIRECT_R32','PPCA_DIRECT_R64')]
    for r in (16,32,64):
        m=f'PPCA_DIRECT_R{r}'
        for first in ('POOLED_NATURAL','LR4_TW1','LR4_TW5','DENSE_TW5','WHOLE_POOL_M2_R23','WHOLE_POOL_M2_R24'):
            d.append((first+'_minus_'+m,first,m))
    d += [('LR4_TW5_minus_LR4_TW1','LR4_TW5','LR4_TW1'),('DENSE_TW5_minus_DENSE_TW1','DENSE_TW5','DENSE_TW1')]
    return d

def paired(E,Q,draws):
    if E.ndim!=3 or Q.shape!=(len(E),) or draws.shape!=(2000,len(E)) or draws.dtype!=np.int64:raise ValueError('BOOTSTRAP_AXES')
    if draws.min()<0 or draws.max()>=len(E) or np.any(Q<=0) or np.any(E<0) or not np.isfinite(E).all() or not np.isfinite(Q).all():raise ValueError('ENERGY_OR_DRAW')
    point=E.sum(0)/Q.sum();bl=E[draws].sum(1)/Q[draws].sum(1)[:,None,None]
    if np.any(point<=0) or np.any(bl<=0):raise ValueError('NONPOSITIVE_NMSE')
    return point,10*np.log10(point),bl,10*np.log10(bl)

def analyze(out,dataset):
    out=Path(out);root=out.parents[1]
    for p,h in read(out/'reports/execution_freeze.json')['hashes'].items():
        if sha(out/p)!=h:raise ValueError('ANALYSIS_FROZEN_INPUT_CHANGED:'+p)
    for p,h in read(out/'reports/preservation_before.json')['hashes'].items():
        if sha(root/p)!=h:raise ValueError('ANALYSIS_ORIGINAL_INPUT_CHANGED:'+p)
    cfg=read(out/'config.json');methods=cfg['methods'];conditions=cfg['conditions'];snrs=cfg['snrs']
    receipt=read(out/f'reports/{dataset}_receipt.json')
    if not receipt['PASS'] or receipt['physical_trajectories']!=500 or not receipt['raw_saved_before_statistics']:raise ValueError('INCOMPLETE_RAW')
    with np.load(out/f'data/bootstrap_{dataset}.npz',allow_pickle=False) as z:draws={k:z[k] for k in z.files}
    fe=np.empty((5,100,3,len(methods),6));fq=np.empty((5,100,6));identities=[]
    rows=receipt['receipts'];by={(r['condition'],r['trajectory_id']):r for r in rows}
    if len(rows)!=500 or len(by)!=500:raise ValueError('IDENTITY_CARDINALITY')
    for ci,c in enumerate(conditions):
        for i in range(100):
            r=by[c,i];p=out/r['path'];jp=p.with_suffix('.json')
            if sha(p)!=r['sha256'] or sha(jp)!=r['json_sha256']:raise ValueError('RAW_HASH')
            m=read(jp)
            if m['condition']!=c or m['trajectory_id']!=i or m['dataset']!=dataset or not m['complete']:raise ValueError('RAW_IDENTITY')
            with zipfile.ZipFile(p) as zz:
                if zz.testzip() is not None:raise ValueError('CRC')
            with np.load(p,allow_pickle=False) as z:
                if list(z['method_names'])!=methods or list(z['snr_db'])!=snrs or list(z['frame_ids'])!=list(range(4,10)):raise ValueError('RAW_AXES')
                if z['frame_error'].shape!=(3,len(methods),6) or z['frame_truth'].shape!=(6,):raise ValueError('RAW_ENERGY_SHAPE')
                fe[ci,i]=z['frame_error'];fq[ci,i]=z['frame_truth']
                if np.any(z['fit_failed']) or not np.isfinite(z['coordinates_tw1']).all() or not np.isfinite(z['coordinates_tw5']).all():raise ValueError('FIT_COVERAGE')
            identities.append(dict(dataset=dataset,condition=c,trajectory_id=i,H_sha256=m['input']['H_sha256'],pilot_sha256=m['input']['pilot_sha256'],raw_path=r['path'],raw_sha256=r['sha256'],json_sha256=r['json_sha256']))
    E=fe.sum(-1);Q=fq.sum(-1);nd=[];cd=[];tr=[];defs=definitions(methods)
    bl=np.empty((5,2000,3,len(methods)));bd=np.empty_like(bl);bc=np.empty((5,2000,3,len(defs)))
    for ci,c in enumerate(conditions):
        p,d,b,bdb=paired(E[ci],Q[ci],draws[c]);bl[ci]=b;bd[ci]=bdb
        limits=np.quantile(bdb,[.025,.975],axis=0,method='linear')
        for si,s in enumerate(snrs):
            for mi,m in enumerate(methods):
                nd.append(dict(dataset=dataset,condition=c,snr_db=s,method=m,error_energy=float(E[ci,:,si,mi].sum()),truth_energy=float(Q[ci].sum()),nmse_linear=float(p[si,mi]),nmse_db=float(d[si,mi]),CI_lower=float(limits[0,si,mi]),CI_upper=float(limits[1,si,mi]),trajectory_count=100,scored_frames=600,bootstrap=2000,interval=INTERVAL))
            for k,(name,first,second) in enumerate(defs):
                a,bb=methods.index(first),methods.index(second);diff=bdb[:,si,a]-bdb[:,si,bb];bc[ci,:,si,k]=diff
                lo,hi=np.quantile(diff,[.025,.975],method='linear')
                cd.append(dict(dataset=dataset,condition=c,snr_db=s,contrast=name,first=first,second=second,estimate_db=float(d[si,a]-d[si,bb]),CI_lower=float(lo),CI_upper=float(hi),trajectory_count=100,bootstrap=2000,interval=INTERVAL,sign='first minus second NMSE(dB); positive favors second',first_worse_trajectory_fraction=float(np.mean(E[ci,:,si,a]>E[ci,:,si,bb]))))
        for i in range(100):
            for si,s in enumerate(snrs):
                for mi,m in enumerate(methods):tr.append(dict(dataset=dataset,condition=c,trajectory_id=i,snr_db=s,method=m,error_energy=float(E[ci,i,si,mi]),truth_energy=float(Q[ci,i]),scored_frames=6))
    npz(out/f'data/analysis_{dataset}.npz',frame_error=fe,frame_truth=fq,error=E,truth=Q,methods=np.array(methods),conditions=np.array(conditions),snrs=np.array(snrs))
    npz(out/f'data/bootstrap_statistics_{dataset}.npz',nmse_linear=bl,nmse_db=bd,contrast_db=bc,contrast_names=np.array([v[0] for v in defs]),contrast_first=np.array([v[1] for v in defs]),contrast_second=np.array([v[2] for v in defs]),methods=np.array(methods),conditions=np.array(conditions),snrs=np.array(snrs))
    for name,rr in [('nmse',nd),('contrasts',cd),('trajectory_energies',tr),('input_identities',identities)]:csvout(out/f'tables/{name}_{dataset}.csv',rr)
    if dataset=='V_new':
        macros=[];recommendations={}
        for scope,cs in (('origin-only',cfg['origin_conditions']),('expanded',conditions)):
            cis=[conditions.index(c) for c in cs]
            for r in (16,32,64):
                mi=methods.index(f'PPCA_DIRECT_R{r}');val=float(np.mean([rr['nmse_linear'] for rr in nd if rr['method']==methods[mi] and rr['condition'] in cs]))
                boots=bl[cis,:,:,mi].mean((0,2));lo,hi=np.quantile(boots,[.025,.975],method='linear')
                macros.append(dict(scope=scope,rank=r,macro_linear_pooled_nmse=val,db_of_macro_linear=float(10*np.log10(val)),CI_lower_linear=float(lo),CI_upper_linear=float(hi),cells=len(cs)*3,physical_trajectories=len(cs)*100,bootstrap=2000,interval=INTERVAL,selection_rule='minimum equal-cell arithmetic mean of linear pooled NMSE; tie lower rank; recommendation only'))
            rowscope=[row for row in macros if row['scope']==scope]
            best=min(rowscope,key=lambda a:(a['macro_linear_pooled_nmse'],a['rank']))
            recommendations[scope]=best
        csvout(out/'tables/validation_capacity_macros.csv',macros)
        write(out/'reports/validation_recommendations.json',dict(PASS=True,recommendations=recommendations,fixed_scopes_before_V=True,D_eval_not_yet_evaluated=True,final_deployment_selected=False))
    answer=dict(PASS=True,dataset=dataset,physical_trajectories=500,scored_positions_per_method=9000,method_count=len(methods),nmse_rows=len(nd),contrast_rows=len(cd),trajectory_rows=len(tr),raw_NPZ_CRC=500,raw_JSON_hashes=500,fit_failure_count=0,finite_energy=True,bootstrap=2000,interval=INTERVAL,independent_unit='physical trajectory',no_final_confirmation=True,files=[dict(path=f'tables/{name}_{dataset}.csv',sha256=sha(out/f'tables/{name}_{dataset}.csv')) for name in ('nmse','contrasts','trajectory_energies','input_identities')])
    write(out/f'reports/analysis_{dataset}.json',answer);return answer

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('dataset',choices=['V_new','D_eval']);a.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]);args=a.parse_args();print(json.dumps(analyze(args.out,args.dataset),indent=2))
