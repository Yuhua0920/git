"""Read-only reconciliation of persisted MR03B energies, identities and draws.

No models, generators, fit functions, prediction functions or random sources
are imported. Bootstrap is recomputed via literal draw indexing (the primary
analysis used multiplicity matrix multiplication). All historical inputs stay
read-only. The sole new output is an exclusive JSON review report.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math, os, zipfile
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np

CONDITIONS=('ID','Profile','Mobility','Composed','Boundary')
SNRS=(10,20,30)
PAIRS=('PPCA_DIRECT_R16','PPCA_DIRECT_R32','PPCA_DIRECT_R64',
       'POOLED_NATURAL','LR4_TW1','LR4_TW5',
       'WHOLE_POOL_M2_R23','WHOLE_POOL_M2_R24')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def table(p,fields):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    d={tuple(r[k] for k in fields):r for r in rows}
    if len(d)!=len(rows):raise ValueError('DUPLICATE_TABLE_KEYS:'+str(p))
    return d
def load_npz(p):
    with zipfile.ZipFile(p) as z:
        if len(z.namelist())!=len(set(z.namelist())) or z.testzip() is not None:
            raise ValueError('NPZ_CRC_OR_DUPLICATE:'+str(p))
    with np.load(p,allow_pickle=False) as z:return dict(z)
def exact(a,b,label):
    if not np.array_equal(a,b):raise ValueError('EXACT_REUSE_OR_IDENTITY:'+label)
def close(a,b,label,report):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('FINITE_SHAPE:'+label)
    e=float(np.max(abs(a-b),initial=0.))
    scale=max(float(np.max(abs(a),initial=0.)),float(np.max(abs(b),initial=0.)))
    if e>1e-10+1e-9*scale:raise ValueError('NUMERICAL:'+label+':'+str((e,scale)))
    report['numerical_comparisons']+=1
    report['max_absolute_numerical_difference']=max(report['max_absolute_numerical_difference'],e)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(r'D:\Users\yuhua\Desktop\Project'))
    p.add_argument('--out',type=Path)
    args=p.parse_args();root=args.root
    old=root/'outputs/revision_mr03_ppca_capacity_controls'
    new=root/'outputs/revision_mr03_em16_reconciliation'
    target=args.out or new/'reports/independent_numerical_review.json'
    config=read(new/'config.json');methods=config['methods']
    oldmethods=read(old/'config.json')['methods']
    if methods!=['ORIGINAL_EM16']+oldmethods or len(methods)!=12:
        raise ValueError('METHOD_ORDER')
    nms=table(new/'tables/capacity_comparison.csv',('dataset','condition','snr_db','method'))
    cons=table(new/'tables/paired_contrasts.csv',('dataset','condition','snr_db','first','second'))
    if len(nms)!=360 or len(cons)!=240:raise ValueError('FULL_TABLE_GRID')
    analysis=read(new/'reports/analysis.json')
    stored_boot=load_npz(new/'data/paired_bootstrap_statistics.npz')
    report=dict(PASS=False,review_code_sha256=sha(__file__),
        numerical_comparisons=0,max_absolute_numerical_difference=0.,
        new_random_draws=0,new_predictions=0,new_training=0,
        original_inputs_modified=False,bootstrap_route='literal persisted trajectory draw indexing',
        interval='unadjusted exploratory 95% paired trajectory percentile',datasets={})
    range_rows=[]
    for ds in ('V_new','D_eval'):
        recs=read(new/'reports'/f'{ds}_receipt.json')
        original=read(old/'reports'/f'{ds}_receipt.json')
        ix={(r['condition'],r['trajectory_id']):r for r in recs['receipts']}
        orig={(r['condition'],r['trajectory_id']):r for r in original['receipts']}
        expected={(c,i) for c in CONDITIONS for i in range(100)}
        if set(ix)!=expected or len(recs['receipts'])!=500 or set(orig)!=expected:
            raise ValueError('EXACT_500_IDENTITIES')
        draws_path=old/'data'/f'bootstrap_{ds}.npz'
        if sha(draws_path)!=analysis['original_draws_SHA256'][ds]:raise ValueError('FROZEN_DRAW_HASH')
        draws=load_npz(draws_path)
        dsr=dict(physical_trajectories=500,EM_scored_positions=9000,
                 hash_checked_new_NPZ=0,hash_checked_new_JSON=0,old_11_energy_reuse=0,
                 old_truth_reuse=0,original_raw_hash_and_JSON_checked=0,
                 paired_H_pilot_metadata=0,CRC_checked_new_and_old_NPZ=0,
                 condition_SNR_cells=15)
        for c in CONDITIONS:
            es=[];qs=[]
            for i in range(100):
                r=ix[c,i];o=orig[c,i]
                npz=new/r['raw_path'];js=npz.with_suffix('.json')
                op=old/o['path'];oj=op.with_suffix('.json')
                if sha(npz)!=r['raw_sha256'] or sha(js)!=r['JSON_sha256']:raise ValueError('NEW_HASH')
                if sha(op)!=o['sha256'] or sha(oj)!=o['json_sha256']:raise ValueError('OLD_HASH')
                if r['original_MR03_npz_sha256']!=o['sha256'] or r['original_MR03_json_sha256']!=o['json_sha256']:
                    raise ValueError('OLD_RAW_BINDING')
                info=read(js);oldinfo=read(oj)
                if (info['condition'],info['trajectory_id'],info['dataset'])!=(c,i,ds):raise ValueError('JSON_IDENTITY')
                if info['identity']!=oldinfo['identity'] or info['H_sha256']!=oldinfo['input']['H_sha256'] or info['pilot_sha256']!=oldinfo['input']['pilot_sha256']:
                    raise ValueError('PHYSICAL_PILOT_METADATA')
                raw=load_npz(npz);oldraw=load_npz(op)
                exact(raw['methods'],np.array(methods),'METHODS')
                exact(raw['snr_db'],np.array(SNRS),'SNRS')
                exact(raw['frame_ids'],np.arange(4,10),'FRAME_IDS')
                if raw['frame_error'].shape!=(3,12,6) or raw['frame_truth'].shape!=(6,):raise ValueError('ENERGY_SHAPE')
                if not np.isfinite(raw['frame_error']).all() or not np.isfinite(raw['frame_truth']).all() or (raw['frame_error']<0).any() or (raw['frame_truth']<=0).any():raise ValueError('INVALID_ENERGY')
                exact(raw['frame_error'][:,1:],oldraw['frame_error'],'OLD_METHOD_ENERGY')
                exact(raw['frame_truth'],oldraw['frame_truth'],'OLD_TRUTH')
                dsr['hash_checked_new_NPZ']+=1;dsr['hash_checked_new_JSON']+=1
                dsr['old_11_energy_reuse']+=1;dsr['old_truth_reuse']+=1
                dsr['original_raw_hash_and_JSON_checked']+=1;dsr['paired_H_pilot_metadata']+=1
                dsr['CRC_checked_new_and_old_NPZ']+=2
                es.append(raw['frame_error']);qs.append(raw['frame_truth'])
            fe=np.asarray(es);fq=np.asarray(qs);te=fe.sum(axis=-1);tq=fq.sum(axis=-1)
            pooled=te.sum(axis=0)/tq.sum();db=10*np.log10(pooled)
            dr=draws[c]
            if dr.shape!=(2000,100) or dr.dtype!=np.int64 or dr.min()<0 or dr.max()>=100:raise ValueError('DRAW_DOMAIN')
            boot=np.empty((2000,3,12))
            for start in range(0,2000,32):
                ixdraw=dr[start:start+32]
                boot[start:start+32]=te[ixdraw].sum(axis=1)/tq[ixdraw].sum(axis=1)[:,None,None]
            close(boot,stored_boot[ds+'_'+c+'_bootstrap_linear'],'STORED_BOOT:'+ds+c,report)
            bootdb=10*np.log10(boot);ci=np.quantile(bootdb,[.025,.975],axis=0,method='linear')
            for si,s in enumerate(SNRS):
                for mi,m in enumerate(methods):
                    row=nms[ds,c,str(s),m]
                    for field,value in [('nmse_linear',pooled[si,mi]),('nmse_db',db[si,mi]),('CI_lower',ci[0,si,mi]),('CI_upper',ci[1,si,mi])]:
                        close(float(row[field]),value,ds+c+str(s)+m+field,report)
                    if row['trajectory_count']!='100' or row['scored_frames']!='600':raise ValueError('TABLE_UNIT')
                for second in PAIRS:
                    mi=methods.index(second);dif=db[si,0]-db[si,mi]
                    bounds=np.quantile(bootdb[:,si,0]-bootdb[:,si,mi],[.025,.975],method='linear')
                    row=cons[ds,c,str(s),'ORIGINAL_EM16',second]
                    for field,value in [('estimate_db',dif),('CI_lower',bounds[0]),('CI_upper',bounds[1]),('EM_worse_trajectory_fraction',np.mean(te[:,si,0]>te[:,si,mi]))]:
                        close(float(row[field]),value,ds+c+str(s)+second+field,report)
                    if row['bootstrap']!='2000' or row['trajectory_count']!='100':raise ValueError('BOOTSTRAP_UNIT')
                    if second.startswith('PPCA_DIRECT_R'):
                        range_rows.append(dict(dataset=ds,condition=c,snr_db=s,second=second,estimate_db=float(dif),CI_lower=float(bounds[0]),CI_upper=float(bounds[1])))
        report['datasets'][ds]=dsr
    summaries={}
    for second in ('PPCA_DIRECT_R16','PPCA_DIRECT_R32','PPCA_DIRECT_R64'):
        rs=[r for r in range_rows if r['second']==second]
        summaries[second]=dict(cells=len(rs),positive_point_cells=sum(r['estimate_db']>0 for r in rs),CI_entirely_positive_cells=sum(r['CI_lower']>0 for r in rs),minimum=min(rs,key=lambda r:r['estimate_db']),maximum=max(rs,key=lambda r:r['estimate_db']))
    report['EM_minus_Direct_development_summary']=summaries
    report['NMSE_rows_checked']=len(nms);report['contrast_rows_checked']=len(cons)
    report['PASS']=True
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
