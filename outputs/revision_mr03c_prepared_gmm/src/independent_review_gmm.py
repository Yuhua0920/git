"""Independent persisted MR03C energy/posterior review; no inference or new draws.

Bootstrap uses literal stored-index batches, unlike primary counts matrices.
Percentile endpoints use explicit sorted-order interpolation. No original
analyzer or prediction code is imported and only a new review report is written.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,zipfile
import numpy as np

CONDITIONS=('ID','Profile','Mobility','Composed','Boundary')
SNRS=(10,20,30)
METHODS=('SOURCE','POOLED_NATURAL','GMM_PREPARED_3','PPCA_DIRECT_R16','PPCA_DIRECT_R32','LR4_TW1','LR4_TW5','DENSE_TW5','WHOLE_POOL_M2_R24')
SECOND=('POOLED_NATURAL','LR4_TW1','LR4_TW5','WHOLE_POOL_M2_R24','DENSE_TW5')
ATOL=1e-10
RTOL=1e-10

def sha(path):
    with Path(path).open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()

def js(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def npz(path):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):raise ValueError('NPZ_CRC_OR_DUPLICATE')
    with np.load(path,allow_pickle=False) as z:return dict(z)

def table(path,fields):
    with Path(path).open(encoding='utf-8-sig',newline='') as h:rows=list(csv.DictReader(h))
    ix={tuple(z[f] for f in fields):z for z in rows}
    if len(ix)!=len(rows):raise ValueError('DUPLICATE_TABLE')
    return ix

def close(actual,expected,label,r):
    a=np.asarray(actual);b=np.asarray(expected)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('NUMERIC_SHAPE_OR_FINITE:'+label)
    err=float(np.max(abs(a-b),initial=0));scale=max(float(np.max(abs(a),initial=0)),float(np.max(abs(b),initial=0)))
    if err>ATOL+RTOL*scale:raise ValueError('INDEPENDENT_DIFFERENCE:'+label+':'+str((err,scale)))
    r['numeric_comparisons']+=1;r['max_scaled_error']=max(r['max_scaled_error'],err/(1+scale));r['max_absolute_error']=max(r['max_absolute_error'],err)

def exact(a,b,label):
    if not np.array_equal(a,b):raise ValueError('EXACT_REUSE_IDENTITY:'+label)

def percentiles(values):
    a=np.sort(np.asarray(values,dtype=float),axis=0)
    if not len(a) or not np.isfinite(a).all():raise ValueError('PERCENTILE_NONFINITE')
    result=[]
    for p in (.025,.975):
        q=(len(a)-1)*p;lo=math.floor(q);hi=math.ceil(q);result.append(a[lo]+(q-lo)*(a[hi]-a[lo]))
    return np.asarray(result)

def stats(fe,fq,draws):
    n,s,m,f=fe.shape
    if fq.shape!=(n,f) or draws.shape!=(2000,n) or draws.dtype!=np.int64 or draws.min()<0 or draws.max()>=n:raise ValueError('AXES_OR_DRAWS')
    if not np.isfinite(fe).all() or not np.isfinite(fq).all() or np.any(fe<0) or np.any(fq<=0):raise ValueError('ENERGY_FINITE_NONNEGATIVE')
    te=fe.sum(axis=-1);tq=fq.sum(axis=-1);counts=np.stack([np.bincount(z,minlength=n) for z in draws]);numerator=te.sum(axis=0);denominator=float(tq.sum())
    linear=numerator/denominator
    boot=np.empty((2000,s,m),float)
    for start in range(0,2000,32):
        ix=draws[start:start+32]
        boot[start:start+32]=te[ix].sum(axis=1)/tq[ix].sum(axis=1)[:,None,None]
    if (linear<=0).any() or (boot<=0).any():raise ValueError('NMSE_NOT_POSITIVE')
    return dict(trajectory_error=te,trajectory_truth=tq,error_energy=numerator,truth_energy=denominator,linear=linear,db=10*np.log10(linear),bootstrap_linear=boot,bootstrap_db=10*np.log10(boot))

def posterior_check(raw,r):
    gamma=raw['posterior_weights'];score=raw['log_scores'];quad=raw['quadratic_forms'];entropy=raw['posterior_entropy_nats'];maximum=raw['max_posterior_weight']
    if gamma.shape!=(3,6,3) or score.shape!=gamma.shape or quad.shape!=gamma.shape or entropy.shape!=(3,6) or maximum.shape!=(3,6):raise ValueError('POSTERIOR_AXES')
    if not all(np.isfinite(z).all() for z in (gamma,score,quad,entropy,maximum)) or np.any(gamma<0) or np.any(quad<0):raise ValueError('POSTERIOR_NONFINITE_OR_SIGN')
    close(gamma.sum(axis=-1),np.ones((3,6)),'gamma simplex',r)
    shifted=score-score.max(axis=-1,keepdims=True);exp=np.exp(shifted);reference=exp/exp.sum(axis=-1,keepdims=True)
    close(gamma,reference,'literal exponent/normalizer posterior',r)
    loggamma=np.zeros_like(gamma);np.log(gamma,out=loggamma,where=gamma>0);reference_entropy=-(gamma*loggamma).sum(axis=-1)
    close(entropy,reference_entropy,'natural entropy',r);close(maximum,gamma.max(axis=-1),'posterior maximum',r)
    if np.any(entropy < -1e-12) or np.any(entropy > math.log(3)+1e-12):raise ValueError('ENTROPY_RANGE')

def rank_average(values):
    v=np.asarray(values);order=np.argsort(v,kind='stable');ranks=np.empty(len(v),float);i=0
    while i<len(v):
        j=i+1
        while j<len(v) and v[order[j]]==v[order[i]]:j+=1
        ranks[order[i:j]]=(i+j-1)/2+1;i=j
    return ranks

def spearman(a,b):
    a=rank_average(a);b=rank_average(b);a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return None if den==0 else float(a@b/den)

def toy():
    e=np.arange(1.,49.).reshape(4,2,3,2);q=np.arange(1.,9.).reshape(4,2);draws=np.tile(np.array([[0,1,2,3],[3,3,0,1],[2,2,2,2],[1,0,1,0]],np.int64),(500,1));s=stats(e,q,draws)
    for k,d in enumerate(draws[:4]):
        expected=e[d].sum(axis=(0,3))/q[d].sum()
        if not np.allclose(expected,s['bootstrap_linear'][k],rtol=1e-14,atol=1e-14):raise ValueError('TOY_LITERAL_ROUTE')
    if not np.allclose(percentiles(np.arange(2000.)),np.quantile(np.arange(2000.),[.025,.975],method='linear')):raise ValueError('TOY_PERCENTILE')
    if not np.array_equal(rank_average([2,1,1,3]),[3,1.5,1.5,4]):raise ValueError('TOY_TIED_RANKS')
    return dict(PASS=True,test_groups=3,new_predictions=0,new_draws=0)


def review(root,out):
    root=Path(root).resolve();out=Path(out).resolve();old=root/'outputs/revision_mr03_ppca_capacity_controls'
    cfg=js(out/'config.json');oldcfg=js(old/'config.json');oldmethods=oldcfg['methods']
    if tuple(cfg['methods'])!=METHODS or tuple(cfg['conditions'])!=CONDITIONS or tuple(cfg['snrs'])!=SNRS or cfg['per_condition']!=100 or cfg['bootstrap']['replicates']!=2000 or cfg['bootstrap']['adjusted']:raise ValueError('FROZEN_GRID')
    freeze=js(out/'reports/execution_freeze.json')
    for name,expected in freeze['hashes'].items():
        path=(out/name).resolve();path.relative_to(out)
        if sha(path)!=expected:raise ValueError('FROZEN_BYTE_DRIFT:'+name)
    receipt=js(out/'reports/analysis.json')
    if receipt['PASS'] is not True or receipt['NMSE_rows']!=270 or receipt['contrast_rows']!=150 or receipt['posterior_rows']!=30 or receipt['trajectory_blocks']!=1000 or receipt['new_bootstrap_draws']!=0:raise ValueError('ANALYSIS_NOT_COMPLETE')
    nms=table(out/'tables/nmse.csv',('dataset','condition','snr_db','method'))
    cons=table(out/'tables/paired_contrasts.csv',('dataset','condition','snr_db','first','second'))
    diag=table(out/'tables/posterior_diagnostics.csv',('dataset','condition','snr_db'))
    trajectories=table(out/'tables/trajectory_energies.csv',('dataset','condition','trajectory_id','snr_db','method'))
    if (len(nms),len(cons),len(diag),len(trajectories))!=(270,150,30,27000):raise ValueError('CSV_CARDINALITIES')
    stored=npz(out/'data/bootstrap_statistics.npz')
    setup=js(out/'reports/preparation.json')
    audits={int(x['snr_db']):x['setup_audit']['components'] for x in setup['setup']}
    report=dict(PASS=False,independent_review_code_sha256=sha(__file__),numeric_comparisons=0,max_absolute_error=0.,max_scaled_error=0.,new_predictions=0,new_random_draws=0,new_model_fits=0,bootstrap_route='literal original persisted indices in32-draw blocks; primary uses multiplicity matrices',percentile_route='independent explicit sorted-order linear interpolation',original_files_modified=0,NMSE_rows=270,contrast_rows=150,posterior_rows=30,trajectory_energy_rows=27000,freeze_files_checked=len(freeze['hashes']),datasets={})
    for ds in ('V_new','D_eval'):
        rec=js(out/f'reports/{ds}_receipt.json');orec=js(old/f'reports/{ds}_receipt.json')
        ix={(r['condition'],int(r['trajectory_id'])):r for r in rec['receipts']};oi={(r['condition'],int(r['trajectory_id'])):r for r in orec['receipts']}
        expected={(c,i) for c in CONDITIONS for i in range(100)}
        if rec['PASS'] is not True or rec['GMM_scored_positions']!=9000 or rec['failures']!=0 or rec['nonfinite']!=0 or len(rec['receipts'])!=500 or set(ix)!=expected or set(oi)!=expected:raise ValueError('COMPLETE_TRAJECTORY_GRID')
        drawp=out/f'data/bootstrap_{ds}.npz';originaldrawp=old/f'data/bootstrap_{ds}.npz'
        if sha(drawp)!=sha(originaldrawp) or sha(drawp)!=receipt['existing_draws_SHA256'][ds]:raise ValueError('ORIGINAL_BOOTSTRAP_HASH')
        draws=npz(drawp)
        dsr=dict(physical_trajectories=500,GMM_scored_positions=9000,condition_SNR_cells=15,new_JSON_NPZ_hash_pairs_checked=0,new_and_original_NPZ_CRC_checked=0,old_energy_reuse_checked=0,pilot_truth_identity_pairs_checked=0,posterior_frames_checked=0,posteriors_labels_used_online=False,draws_exact_original=True)
        for c in CONDITIONS:
            energies=[];truths=[];weights=[];entropy=[];maximum=[]
            for i in range(100):
                r=ix[c,i];o=oi[c,i];p=out/r['path'];op=old/o['path'];j=p.with_suffix('.json');oj=op.with_suffix('.json')
                if sha(p)!=r['SHA256'] or sha(j)!=r['JSON_SHA256'] or sha(op)!=o['sha256'] or sha(oj)!=o['json_sha256']:raise ValueError('RAW_OR_ORIGINAL_HASH')
                info=js(j);orig=js(oj)
                if (info['dataset'],info['condition'],int(info['trajectory_id']),info['complete'])!=(ds,c,i,True):raise ValueError('RAW_JSON_IDENTITY')
                if info['identity']!=orig['identity'] or info['old_raw_SHA256']!=o['sha256'] or info['old_JSON_SHA256']!=o['json_sha256'] or info['H_SHA256']!=orig['input']['H_sha256'] or info['pilot_full_SHA256']!=orig['input']['pilot_sha256']:raise ValueError('ORIGINAL_IDENTITY_BINDING')
                if info['posterior_labels_used_online'] is not False or info['failures']!=0 or info['nonfinite']!=0:raise ValueError('ONLINE_INFO_OR_FAILURE')
                raw=npz(p);oldraw=npz(op)
                exact(raw['method_names'],np.array(METHODS),'method order');exact(raw['snr_db'],np.array(SNRS),'SNR order');exact(raw['frame_ids'],np.arange(4,10),'frame order')
                if raw['frame_error'].shape!=(3,9,6) or raw['frame_truth'].shape!=(6,) or raw['pilots'].shape!=(3,6,64):raise ValueError('RAW_SHAPE')
                if np.any(raw['failure_flags']):raise ValueError('FAILED_RAW_FRAME')
                exact(raw['frame_truth'],oldraw['frame_truth'],'common truth')
                exact(raw['pilots'],oldraw['pilots'][:,4:10],'scored pilot exact')
                if hashlib.sha256(np.ascontiguousarray(raw['pilots']).tobytes()).hexdigest()!=info['scored_pilot_SHA256']:raise ValueError('SCORED_PILOT_HASH')
                for mi,m in enumerate(METHODS):
                    if m!='GMM_PREPARED_3':exact(raw['frame_error'][:,mi],oldraw['frame_error'][:,oldmethods.index(m)],'old energy:'+m)
                posterior_check(raw,report)
                for si,s in enumerate(SNRS):
                    ldet=np.array([a['logdet'] for a in audits[s]])
                    close(raw['log_scores'][si],-math.log(3)-ldet-raw['quadratic_forms'][si],'frozen uniform-prior logscore',report)
                energies.append(raw['frame_error']);truths.append(raw['frame_truth']);weights.append(raw['posterior_weights']);entropy.append(raw['posterior_entropy_nats']);maximum.append(raw['max_posterior_weight'])
                dsr['new_JSON_NPZ_hash_pairs_checked']+=1;dsr['new_and_original_NPZ_CRC_checked']+=2;dsr['old_energy_reuse_checked']+=8;dsr['pilot_truth_identity_pairs_checked']+=1;dsr['posterior_frames_checked']+=18
            fe,fq,gw,h,mx=map(np.asarray,(energies,truths,weights,entropy,maximum));z=stats(fe,fq,draws[c]);close(stored[ds+'_'+c],z['bootstrap_linear'],'persisted bootstrap:'+ds+c,report);limits=percentiles(z['bootstrap_db']);gi=METHODS.index('GMM_PREPARED_3');pi=METHODS.index('POOLED_NATURAL')
            for si,s in enumerate(SNRS):
                for mi,m in enumerate(METHODS):
                    row=nms[ds,c,str(s),m]
                    if row['trajectory_count']!='100' or row['scored_frames']!='600':raise ValueError('INDEPENDENT_UNIT')
                    for field,value in (('nmse_linear',z['linear'][si,mi]),('nmse_db',z['db'][si,mi]),('CI_lower',limits[0,si,mi]),('CI_upper',limits[1,si,mi])):close(float(row[field]),value,'NMSE:'+ds+c+str(s)+m+field,report)
                for second in SECOND:
                    bi=METHODS.index(second);row=cons[ds,c,str(s),'GMM_PREPARED_3',second];lim=percentiles(z['bootstrap_db'][:,si,gi]-z['bootstrap_db'][:,si,bi])
                    if row['bootstrap']!='2000' or row['trajectory_count']!='100' or 'positive favors comparator' not in row['sign'] or not all(t in row['interval'].lower() for t in ('exploratory','unadjusted','95%','trajectory','development')):raise ValueError('DIFFERENCE_SCOPE')
                    for field,value in (('estimate_db',z['db'][si,gi]-z['db'][si,bi]),('CI_lower',lim[0]),('CI_upper',lim[1]),('GMM_worse_trajectory_fraction',np.mean(z['trajectory_error'][:,si,gi]>z['trajectory_error'][:,si,bi]))):close(float(row[field]),value,'DIFFERENCE:'+ds+c+str(s)+second+field,report)
                row=diag[ds,c,str(s)];selection=gw[:,si].argmax(axis=-1);correspondence={'ID':0,'Profile':1,'Mobility':2}.get(c);en=h[:,si].mean(axis=-1);gain=10*np.log10(z['trajectory_error'][:,si,gi]/z['trajectory_error'][:,si,pi])
                for field,value in (('mean_weight_Source',gw[:,si,:,0].mean()),('mean_weight_Profile',gw[:,si,:,1].mean()),('mean_weight_Mobility',gw[:,si,:,2].mean()),('mean_entropy_nats',h[:,si].mean()),('mean_max_weight',mx[:,si].mean()),('hardlike_fraction_gamma_ge_095',np.mean(mx[:,si]>=.95)),('argmax_Source_fraction',np.mean(selection==0)),('argmax_Profile_fraction',np.mean(selection==1)),('argmax_Mobility_fraction',np.mean(selection==2))):close(float(row[field]),value,'POSTERIOR:'+ds+c+str(s)+field,report)
                rho=spearman(en,gain);written=row['Spearman_entropy_vs_GMM_minus_Pooled']
                if rho is None:
                    if written!='':raise ValueError('CONSTANT_CORRELATION')
                else:close(float(written),rho,'independent tied-rank entropy Spearman',report)
                if correspondence is None:
                    if any(row[k]!='' for k in ('correspondence_origin','correspondence_mismatch_frame_fraction','Spearman_mismatch_vs_GMM_minus_Pooled')):raise ValueError('NO_UNIQUE_ORIGIN')
                else:
                    mis=(selection!=correspondence).mean(axis=-1);close(float(row['correspondence_mismatch_frame_fraction']),mis.mean(),'origin correspondence descriptive fraction',report);rho=spearman(mis,gain);written=row['Spearman_mismatch_vs_GMM_minus_Pooled']
                    if rho is None:
                        if written!='':raise ValueError('CONSTANT_CORRELATION')
                    else:close(float(written),rho,'independent tied-rank correspondence Spearman',report)
                for t in range(100):
                    for mi,m in enumerate(METHODS):
                        rr=trajectories[ds,c,str(t),str(s),m]
                        for field,value in (('error_energy',z['trajectory_error'][t,si,mi]),('truth_energy',z['trajectory_truth'][t]),('GMM_entropy_nats',en[t]),('GMM_mean_max_weight',mx[t,si].mean()),('GMM_weight_Source',gw[t,si,:,0].mean()),('GMM_weight_Profile',gw[t,si,:,1].mean()),('GMM_weight_Mobility',gw[t,si,:,2].mean())):close(float(rr[field]),value,'trajectory energy/posterior row',report)
        report['datasets'][ds]=dsr
    report['PASS']=True;report['toy_tests']=toy();return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(r'D:\Users\yuhua\Desktop\Project'));p.add_argument('--out',type=Path);p.add_argument('--report',type=Path);p.add_argument('--toy',action='store_true');a=p.parse_args()
    out=a.out or a.root/'outputs/revision_mr03c_prepared_gmm';r=toy() if a.toy else review(a.root,out);text=json.dumps(r,indent=2,allow_nan=False)
    if a.report:
        a.report.parent.mkdir(parents=True,exist_ok=True)
        with a.report.open('x',encoding='utf8') as f:f.write(text+'\n')
    print(text)

if __name__=='__main__':main()

