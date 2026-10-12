"""Post-run independent MR08A numerical review. No model/generator/random imports.

Reads actual raw energies and actual persisted trajectory-index draws. Repeats
all bootstrap ratios by direct-index blocks, unlike the main counts-matrix route.
No formal invocation should occur until root confirms complete raw execution.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,csv,hashlib,json,math,zipfile
import numpy as np

CONDS=('ID','Profile','Mobility','Composed','Boundary');SNRS=(0,5,10,15,20,25,30)
METHODS=('SOURCE','POOLED_NATURAL','PPCA_DIRECT_R32','GMM_PREPARED_3','HARD_TW5','CONVEX_TW5','LR4_TW1','LR4_TW5','DENSE_TW5','WHOLE_POOL_M2_R24')
FAMILIES=(('A','POOLED_NATURAL','LR4_TW5'),('B','GMM_PREPARED_3','LR4_TW1'),('C','LR4_TW5','WHOLE_POOL_M2_R24'))
TAIL=.05/(2*35)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def js(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def csvindex(path,keys):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    ix={tuple(row[k] for k in keys):row for row in rows}
    if len(ix)!=len(rows):raise ValueError('DUPLICATE_CSV_KEYS')
    return ix

def close(a,b,label,r):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('FINITE_SHAPE:'+label)
    error=float(np.max(abs(a-b),initial=0));scale=max(float(np.max(abs(a),initial=0)),float(np.max(abs(b),initial=0)))
    if error>1e-10+1e-10*scale:raise ValueError('INDEPENDENT_NUMERIC_MISMATCH:'+label+':'+str((error,scale)))
    r['numerical_comparisons']+=1;r['maximum_absolute_difference']=max(r['maximum_absolute_difference'],error);r['maximum_scaled_difference']=max(r['maximum_scaled_difference'],error/(1+scale))

def exact(a,b,label):
    if not np.array_equal(a,b):raise ValueError('EXACT_IDENTITY:'+label)

def differences(linear):
    db=10*np.log10(linear)
    return np.stack([db[...,METHODS.index(a)]-db[...,METHODS.index(b)] for _,a,b in FAMILIES],axis=-1)

def bounds_explicit(values,probabilities=(TAIL,1-TAIL)):
    values=np.sort(np.asarray(values,dtype=np.float64),axis=0)
    if not np.isfinite(values).all() or len(values)==0:raise ValueError('BOUNDS_NONFINITE')
    result=[]
    for p in probabilities:
        index=(len(values)-1)*p;lo=math.floor(index);hi=math.ceil(index);result.append(values[lo]+(index-lo)*(values[hi]-values[lo]))
    return np.asarray(result)

def direct_bootstrap(te,tq,ids):
    ids=np.asarray(ids)
    if ids.dtype.kind not in 'iu' or ids.ndim!=2 or ids.shape[1]!=len(te) or ids.size==0 or ids.min()<0 or ids.max()>=len(te):raise ValueError('STORED_INDEX_DOMAIN')
    result=te[ids].sum(axis=1)/tq[ids].sum(axis=1)[:,None,None]
    if not np.isfinite(result).all() or np.any(result<=0):raise ValueError('BOOTSTRAP_NMSE_INVALID')
    return result

def tiny_tests():
    fe=np.arange(1.,4*7*10*6+1).reshape(4,7,10,6);fq=np.arange(1.,25.).reshape(4,6);te=fe.sum(axis=-1);tq=fq.sum(axis=-1);ids=np.array([[0,1,2,3],[3,3,0,1],[2,2,2,2]],np.uint16);actual=direct_bootstrap(te,tq,ids)
    for k,row in enumerate(ids):
        ref=np.zeros((7,10));den=0.
        for trajectory in row:
            den+=float(fq[trajectory].sum())
            for si in range(7):
                for mi in range(10):ref[si,mi]+=float(fe[trajectory,si,mi].sum())
        assert np.allclose(actual[k],ref/den,rtol=1e-14,atol=1e-14)
    assert np.array_equal(bounds_explicit(np.arange(100000.)),np.quantile(np.arange(100000.),[TAIL,1-TAIL],method='linear'))
    x=np.ones((7,10));x[:,METHODS.index('POOLED_NATURAL')]=4;x[:,METHODS.index('LR4_TW5')]=2;x[:,METHODS.index('GMM_PREPARED_3')]=8;x[:,METHODS.index('LR4_TW1')]=4;x[:,METHODS.index('WHOLE_POOL_M2_R24')]=1
    assert np.allclose(differences(x),10*np.log10(2))
    return dict(PASS=True,test_groups=3,new_physical_samples=0,new_draws=0)

def review(out,mode='formal',block=16):
    out=Path(out).resolve();formal=mode=='formal';n=1000 if formal else 2;replicates=100000 if formal else 100
    if mode not in ('formal','smoke') or block<=0:raise ValueError('REVIEW_MODE')
    protocol=js(out/'protocol.json');receipt=js(out/f'reports/{mode}_receipt.json');analysis=js(out/f'reports/{mode}_analysis.json');frozen=js(out/'reports/execution_freeze.json')
    if receipt['PASS'] is not True or receipt['identity_PASS'] is not True or any(receipt[k] for k in ('failures','nonfinite','missing','substituted')) or receipt['smoke_mixed_into_formal'] is not False:raise ValueError('RAW_RECEIPT_NOT_PASS')
    if analysis['PASS'] is not True or analysis['mode']!=mode or analysis['bootstrap_replicates']!=replicates or analysis['physical_trajectories']!=5*n or analysis['NMSE_rows']!=350 or analysis['primary_rows']!=105 or analysis['no_combined105_familywise_guarantee'] is not True:raise ValueError('ANALYSIS_SCOPE')
    if tuple(protocol['methods'])!=METHODS or tuple(protocol['snrs'])!=SNRS:raise ValueError('PROTOCOL_METHODS_OR_SNR')
    for path,h in frozen['new_hashes'].items():
        p=(out/path).resolve();p.relative_to(out)
        if sha(p)!=h:raise ValueError('FROZEN_EXECUTION_SOURCE_OR_ASSET_CHANGED:'+path)
    nm=csvindex(out/f'tables/{mode}_nmse.csv',('condition','snr_db','method'));cs=csvindex(out/f'tables/{mode}_primary_contrasts.csv',('family','condition','snr_db'));tr=csvindex(out/f'tables/{mode}_trajectory_energies.csv',('condition','trajectory_id','snr_db','method'))
    if (len(nm),len(cs),len(tr))!=(350,105,5*n*7*10):raise ValueError('RESULT_GRID')
    rawix={(r['condition'],int(r['trajectory_id'])):r for r in receipt['receipts']}
    if len(receipt['receipts'])!=5*n or set(rawix)!={(c,i) for c in CONDS for i in range(n)}:raise ValueError('RAW_GRID')
    recordix={r['condition']:r for r in analysis['records']}
    if set(recordix)!=set(CONDS):raise ValueError('BOOTSTRAP_CONDITION_RECORDS')
    historical=set(js(out/'data/historical_H_hashes.json')['H_sha256']);forbidden=set(historical)
    if formal:forbidden.update(r['H_sha256'] for r in js(out/'reports/smoke_receipt.json')['receipts'])
    seen=set();report=dict(PASS=False,review_code_SHA256=sha(__file__),mode=mode,numerical_comparisons=0,maximum_absolute_difference=0.,maximum_scaled_difference=0.,new_draws=0,new_predictions=0,new_training=0,new_physical_trajectories=0,route='Actual persisted indices, direct trajectory indexing in16-draw blocks; independent of primary counts route',bounds_route='Explicit sorted-order linear interpolation at0.05/(2*35) and1-tail',no_combined105_familywise_guarantee=True,primary_families_separate=True,all_frame_energies_preserved=True,conditions={})
    for ci,c in enumerate(CONDS):
        fe=np.empty((n,7,10,6));fq=np.empty((n,6));raw_checked=0
        for i in range(n):
            r=rawix[c,i];p=(out/r['path']).resolve();p.relative_to(out);jp=p.with_suffix('.json')
            if sha(p)!=r['SHA256'] or sha(jp)!=r['JSON_SHA256']:raise ValueError('RAW_HASH')
            meta=js(jp)
            if (meta['mode'],meta['condition'],int(meta['trajectory_id']),meta['complete'])!=(mode,c,i,True) or meta['failures'] or meta['nonfinite'] or meta['common_truth_for_all_methods_and_SNR'] is not True:raise ValueError('RAW_JSON_IDENTITY')
            if meta['protocol_SHA256']!=sha(out/'protocol.json') or meta['seed_registry_SHA256']!=sha(out/f'data/{mode}_seed_registry.json'):raise ValueError('RAW_PROTOCOL_OR_SEED_BINDING')
            h=meta['input']['H_sha256']
            if h!=r['H_sha256'] or h in seen or h in forbidden:raise ValueError('PHYSICAL_IDENTITY_DUPLICATE_OR_HISTORICAL_OVERLAP')
            seen.add(h)
            with zipfile.ZipFile(p) as z:
                if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):raise ValueError('RAW_CRC_OR_DUPLICATE')
            with np.load(p,allow_pickle=False) as z:
                exact(z['method_names'],np.array(METHODS),'methods');exact(z['snr_db'],np.array(SNRS),'SNRs');exact(z['frame_ids'],np.arange(4,10),'frames')
                if z['frame_error'].shape!=(7,10,6) or z['frame_truth'].shape!=(6,) or z['frame_error'].dtype!=np.float64 or z['frame_truth'].dtype!=np.float64 or np.any(z['failure_flags']):raise ValueError('RAW_ENERGY_AXES_OR_FAILURE')
                fe[i]=z['frame_error'];fq[i]=z['frame_truth']
            raw_checked+=1
        if not np.isfinite(fe).all() or not np.isfinite(fq).all() or np.any(fe<0) or np.any(fq<=0):raise ValueError('RAW_ENERGY_FINITE_OR_SIGN')
        te=fe.sum(axis=-1);tq=fq.sum(axis=-1);point=te.sum(axis=0)/tq.sum();pointdb=10*np.log10(point);pd=differences(point);rec=recordix[c]
        paths={name:(out/rec[name]['path']).resolve() for name in ('indices','bootstrap_linear','primary_differences')}
        for name,path in paths.items():
            path.relative_to(out);h=rec[name].get('SHA256',rec[name].get('sha256'))
            if sha(path)!=h:raise ValueError('PERSISTED_BOOTSTRAP_HASH:'+name)
        ids=np.load(paths['indices'],mmap_mode='r',allow_pickle=False);bl=np.load(paths['bootstrap_linear'],mmap_mode='r',allow_pickle=False);bd=np.load(paths['primary_differences'],mmap_mode='r',allow_pickle=False)
        if ids.shape!=(replicates,n) or ids.dtype!=np.uint16 or bl.shape!=(replicates,7,10) or bd.shape!=(replicates,7,3):raise ValueError('PERSISTED_DRAW_STAT_AXES')
        for start in range(0,replicates,block):
            stop=min(start+block,replicates);expected=direct_bootstrap(te,tq,ids[start:stop]);close(bl[start:stop],expected,'allbootstrap linear ratios:'+c,report);close(bd[start:stop],differences(expected),'allbootstrap primary diffs:'+c,report)
        bounds=bounds_explicit(bd) if formal else None
        for si,s in enumerate(SNRS):
            for mi,m in enumerate(METHODS):
                row=nm[c,str(s),m]
                if int(row['trajectory_count'])!=n or int(row['scored_frames'])!=n*6:raise ValueError('NMSE_UNIT')
                for field,value in (('error_energy',te[:,si,mi].sum()),('truth_energy',tq.sum()),('nmse_linear',point[si,mi]),('nmse_db',pointdb[si,mi])):close(float(row[field]),value,'all350NMSE point/energy rows',report)
                for i in range(n):
                    row=tr[c,str(i),str(s),m];close(float(row['error_energy']),te[i,si,mi],'trajectory numerator',report);close(float(row['truth_energy']),tq[i],'commontrajectory denominator',report)
            for fi,(family,first,second) in enumerate(FAMILIES):
                row=cs[family,c,str(s)]
                if row['first']!=first or row['second']!=second or int(row['family_size'])!=35 or int(row['bootstrap'])!=replicates or 'positive favors second' not in row['sign']:raise ValueError('PRIMARY_FAMILY_DEFINITION')
                close(float(row['estimate_db']),pd[si,fi],'all105primary point differences',report)
                if formal:
                    if float(row['tail_probability'])!=TAIL:raise ValueError('BONFERRONI_35_TAIL')
                    close(float(row['CI_lower']),bounds[0,si,fi],'all105lower endpoints',report);close(float(row['CI_upper']),bounds[1,si,fi],'all105upper endpoints',report)
                    classification='SECOND_SUPPORTED' if bounds[0,si,fi]>0 else 'FIRST_SUPPORTED' if bounds[1,si,fi]<0 else 'INDETERMINATE'
                    if row['classification']!=classification:raise ValueError('CLASSIFICATION')
                elif row['CI_lower']!='' or row['CI_upper']!='' or row['classification']!='SMOKE_ONLY':raise ValueError('SMOKE_INFERENCE_FORBIDDEN')
        report['conditions'][c]=dict(raw_JSON_NPZ_hash_CRC_pairs=raw_checked,physical_trajectories=n,bootstrap_draws_checked=replicates,bootstrap_SNR_method_positions_checked=replicates*7*10,primary_differences_checked=replicates*7*3,all105family_endpoints_checked=formal,identities_disjoint=True)
        del ids,bl,bd,fe,fq,te,tq
    report.update(PASS=True,NMSE_rows=350,primary_contrasts=105,physical_trajectories=len(seen),scored_positions_per_method=5*n*7*6,bootstrap_replicates_per_condition=replicates,toy=tiny_tests());return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--mode',choices=('formal','smoke'),default='formal');p.add_argument('--report',type=Path);p.add_argument('--toy',action='store_true');a=p.parse_args();answer=tiny_tests() if a.toy else review(a.out,a.mode);text=json.dumps(answer,indent=2,allow_nan=False)
    if a.report:
        path=a.report.resolve()
        if path.is_relative_to((a.out/'data').resolve()):raise ValueError('REPORT_CANNOT_MODIFY_DATA')
        path.parent.mkdir(exist_ok=True,parents=True)
        with path.open('x',encoding='utf8') as f:f.write(text+'\n')
    print(text)
if __name__=='__main__':main()
