"""Package-only persisted MR03C numerical review. No external project needed.

Original8-method entity reuse is referenced to the completed independent review;
this entry does not pretend to re-read absent historical source files. It freshly
checks package raw hashes, CRC, identities, all energies/posteriors, unchanged
stored draw identities, NMSE/paired intervals and descriptive diagnostics.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,json,math
import numpy as np
from independent_review_gmm import sha,js,npz,table,close,exact,stats,percentiles,posterior_check,spearman,toy,CONDITIONS,SNRS,METHODS,SECOND

def review(out):
    out=Path(out).resolve();prior_path=out/'reports/independent_numerical_review.json';prior=js(prior_path)
    if prior['PASS'] is not True or prior['new_predictions']!=0 or prior['new_random_draws']!=0 or prior['NMSE_rows']!=270 or prior['contrast_rows']!=150 or prior['posterior_rows']!=30:raise ValueError('ORIGINAL_REUSE_CERTIFICATION_NOT_PASS')
    if prior['independent_review_code_sha256']!=sha(Path(__file__).parent/'independent_review_gmm.py'):raise ValueError('CERTIFIED_REVIEW_SOURCE_CHANGED')
    cfg=js(out/'config.json');frozen=js(out/'reports/execution_freeze.json');analysis=js(out/'reports/analysis.json')
    if tuple(cfg['methods'])!=METHODS or tuple(cfg['conditions'])!=CONDITIONS or tuple(cfg['snrs'])!=SNRS or cfg['bootstrap']['replicates']!=2000 or cfg['bootstrap']['adjusted'] or analysis['PASS'] is not True:raise ValueError('PACKAGE_GRID_OR_STATISTICS')
    for name,h in frozen['hashes'].items():
        path=(out/name).resolve();path.relative_to(out)
        if sha(path)!=h:raise ValueError('PACKAGE_FROZEN_HASH:'+name)
    nm=table(out/'tables/nmse.csv',('dataset','condition','snr_db','method'));cs=table(out/'tables/paired_contrasts.csv',('dataset','condition','snr_db','first','second'));di=table(out/'tables/posterior_diagnostics.csv',('dataset','condition','snr_db'));te_rows=table(out/'tables/trajectory_energies.csv',('dataset','condition','trajectory_id','snr_db','method'))
    if (len(nm),len(cs),len(di),len(te_rows))!=(270,150,30,27000):raise ValueError('PACKAGE_TABLE_CARDINALITY')
    boot=npz(out/'data/bootstrap_statistics.npz');audits={int(z['snr_db']):z['setup_audit']['components'] for z in js(out/'reports/preparation.json')['setup']}
    r=dict(PASS=False,standalone_code_sha256=sha(__file__),source_root=str(out),numeric_comparisons=0,max_absolute_error=0.,max_scaled_error=0.,new_predictions=0,new_random_draws=0,new_model_fits=0,external_files_read=0,original8_method_entity_reuse_rechecked=False,prior_original_entity_reuse_certification_SHA256=sha(prior_path),prior_original_entity_reuse_certification='Referenced PASS; original project raw files not accessed',bootstrap_route='literal persisted trajectory indices in32-draw blocks',percentile_route='explicit sorted-order linear interpolation',interval='Unadjusted exploratory95% paired trajectory percentile; disclosed development data',table_rows=dict(NMSE=270,contrasts=150,posterior=30,trajectory_energy=27000),datasets={})
    for ds in ('V_new','D_eval'):
        rec=js(out/f'reports/{ds}_receipt.json');ix={(z['condition'],int(z['trajectory_id'])):z for z in rec['receipts']};expected={(c,i) for c in CONDITIONS for i in range(100)}
        if rec['PASS'] is not True or rec['failures'] or rec['nonfinite'] or rec['GMM_scored_positions']!=9000 or len(rec['receipts'])!=500 or set(ix)!=expected:raise ValueError('PACKAGE_RAW_GRID')
        drawpath=out/f'data/bootstrap_{ds}.npz'
        if sha(drawpath)!=analysis['existing_draws_SHA256'][ds]:raise ValueError('PACKAGE_DRAW_BINDING')
        draws=npz(drawpath);dr=dict(JSON_NPZ_hash_pairs=0,CRC_NPZ_checked=0,posterior_frames=0,physical_trajectories=500,GMM_scored_positions=9000)
        for c in CONDITIONS:
            E=[];Q=[];GW=[];H=[];MX=[]
            for i in range(100):
                row=ix[c,i];path=(out/row['path']).resolve();path.relative_to(out);jp=path.with_suffix('.json')
                if sha(path)!=row['SHA256'] or sha(jp)!=row['JSON_SHA256']:raise ValueError('PACKAGE_RAW_HASH')
                meta=js(jp)
                if (meta['dataset'],meta['condition'],int(meta['trajectory_id']),meta['complete'])!=(ds,c,i,True) or meta['posterior_labels_used_online'] is not False or meta['failures'] or meta['nonfinite']:raise ValueError('PACKAGE_IDENTITY_OR_FAILURE')
                raw=npz(path);exact(raw['method_names'],np.array(METHODS),'methods');exact(raw['snr_db'],np.array(SNRS),'SNRs');exact(raw['frame_ids'],np.arange(4,10),'frames')
                if raw['frame_error'].shape!=(3,9,6) or raw['frame_truth'].shape!=(6,) or raw['pilots'].shape!=(3,6,64) or np.any(raw['failure_flags']):raise ValueError('PACKAGE_RAW_AXES')
                posterior_check(raw,r)
                for si,s in enumerate(SNRS):close(raw['log_scores'][si],-math.log(3)-np.array([a['logdet'] for a in audits[s]])-raw['quadratic_forms'][si],'proper-complex uniform-prior score',r)
                E.append(raw['frame_error']);Q.append(raw['frame_truth']);GW.append(raw['posterior_weights']);H.append(raw['posterior_entropy_nats']);MX.append(raw['max_posterior_weight']);dr['JSON_NPZ_hash_pairs']+=1;dr['CRC_NPZ_checked']+=1;dr['posterior_frames']+=18
            fe,fq,gw,h,mx=map(np.asarray,(E,Q,GW,H,MX));z=stats(fe,fq,draws[c]);close(boot[ds+'_'+c],z['bootstrap_linear'],'package storedbootstrap',r);lim=percentiles(z['bootstrap_db']);gi=METHODS.index('GMM_PREPARED_3');pi=METHODS.index('POOLED_NATURAL')
            for si,s in enumerate(SNRS):
                for mi,m in enumerate(METHODS):
                    row=nm[ds,c,str(s),m]
                    if row['trajectory_count']!='100' or row['scored_frames']!='600':raise ValueError('PACKAGE_UNIT')
                    for key,value in (('nmse_linear',z['linear'][si,mi]),('nmse_db',z['db'][si,mi]),('CI_lower',lim[0,si,mi]),('CI_upper',lim[1,si,mi])):close(float(row[key]),value,'package NMSE',r)
                for second in SECOND:
                    bi=METHODS.index(second);row=cs[ds,c,str(s),'GMM_PREPARED_3',second];ci=percentiles(z['bootstrap_db'][:,si,gi]-z['bootstrap_db'][:,si,bi])
                    if row['bootstrap']!='2000' or row['trajectory_count']!='100' or 'positive favors comparator' not in row['sign'] or not all(k in row['interval'].lower() for k in ('unadjusted','exploratory','trajectory','development')):raise ValueError('PACKAGE_CONTRAST_SCOPE')
                    for key,value in (('estimate_db',z['db'][si,gi]-z['db'][si,bi]),('CI_lower',ci[0]),('CI_upper',ci[1]),('GMM_worse_trajectory_fraction',np.mean(z['trajectory_error'][:,si,gi]>z['trajectory_error'][:,si,bi]))):close(float(row[key]),value,'package contrast',r)
                row=di[ds,c,str(s)];sel=gw[:,si].argmax(axis=-1);en=h[:,si].mean(axis=-1);gain=10*np.log10(z['trajectory_error'][:,si,gi]/z['trajectory_error'][:,si,pi])
                for key,value in (('mean_weight_Source',gw[:,si,:,0].mean()),('mean_weight_Profile',gw[:,si,:,1].mean()),('mean_weight_Mobility',gw[:,si,:,2].mean()),('mean_entropy_nats',h[:,si].mean()),('mean_max_weight',mx[:,si].mean()),('hardlike_fraction_gamma_ge_095',np.mean(mx[:,si]>=.95)),('argmax_Source_fraction',np.mean(sel==0)),('argmax_Profile_fraction',np.mean(sel==1)),('argmax_Mobility_fraction',np.mean(sel==2))):close(float(row[key]),value,'package posterior summary',r)
                rho=spearman(en,gain)
                if rho is None:
                    if row['Spearman_entropy_vs_GMM_minus_Pooled']!='':raise ValueError('PACKAGE_CONSTANT_CORRELATION')
                else:close(float(row['Spearman_entropy_vs_GMM_minus_Pooled']),rho,'package entropy Spearman',r)
                cor={'ID':0,'Profile':1,'Mobility':2}.get(c)
                if cor is None:
                    if any(row[k]!='' for k in ('correspondence_origin','correspondence_mismatch_frame_fraction','Spearman_mismatch_vs_GMM_minus_Pooled')):raise ValueError('PACKAGE_NONEXISTENT_ORIGIN_LABEL')
                else:
                    mis=(sel!=cor).mean(axis=-1);close(float(row['correspondence_mismatch_frame_fraction']),mis.mean(),'package correspondence fraction',r);rho=spearman(mis,gain)
                    if rho is None:
                        if row['Spearman_mismatch_vs_GMM_minus_Pooled']!='':raise ValueError('PACKAGE_CONSTANT_CORRELATION')
                    else:close(float(row['Spearman_mismatch_vs_GMM_minus_Pooled']),rho,'package mismatch Spearman',r)
                for t in range(100):
                    for mi,m in enumerate(METHODS):
                        tr=te_rows[ds,c,str(t),str(s),m]
                        for key,value in (('error_energy',z['trajectory_error'][t,si,mi]),('truth_energy',z['trajectory_truth'][t]),('GMM_entropy_nats',en[t]),('GMM_mean_max_weight',mx[t,si].mean()),('GMM_weight_Source',gw[t,si,:,0].mean()),('GMM_weight_Profile',gw[t,si,:,1].mean()),('GMM_weight_Mobility',gw[t,si,:,2].mean())):close(float(tr[key]),value,'package trajectory row',r)
        r['datasets'][ds]=dr
    r.update(PASS=True,toy_checks=toy());return r

def standalone_toy():
    import tempfile
    result=toy()
    with tempfile.TemporaryDirectory(prefix='mr03c_standalone_toy_') as directory:
        root=Path(directory);(root/'reports').mkdir()
        prior=dict(PASS=False,new_predictions=0,new_random_draws=0,NMSE_rows=270,contrast_rows=150,posterior_rows=30,independent_review_code_sha256='0'*64)
        (root/'reports/independent_numerical_review.json').write_text(json.dumps(prior),encoding='utf8')
        try:review(root)
        except ValueError as e:
            assert str(e)=='ORIGINAL_REUSE_CERTIFICATION_NOT_PASS'
        else:raise AssertionError('NonPASS prior certification was accepted')
        prior['PASS']=True;(root/'reports/independent_numerical_review.json').write_text(json.dumps(prior),encoding='utf8')
        try:review(root)
        except ValueError as e:
            assert str(e)=='CERTIFIED_REVIEW_SOURCE_CHANGED'
        else:raise AssertionError('Incorrect source hash was accepted')
    result.update(test_groups=5,new_standalone_negative_tests=['nonPASS prior review rejected','wrong certified source hash rejected'])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--report',type=Path);p.add_argument('--toy',action='store_true');a=p.parse_args();result=standalone_toy() if a.toy else review(a.out);text=json.dumps(result,indent=2,allow_nan=False)
    if a.report:
        path=a.report.resolve()
        if path.is_relative_to((a.out/'data').resolve()):raise ValueError('REPORT_CANNOT_MODIFY_RAW_DATA')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf8') as f:f.write(text+'\n')
    print(text)

if __name__=='__main__':main()

