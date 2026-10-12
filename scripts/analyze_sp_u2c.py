"""All conditional windows and trajectory-cluster descriptive statistics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2c_diagnostic as d
d.install_guard()
import numpy as np
from scipy.stats import spearmanr

GEOMETRY=('sigma_min','lambda_min','kappa_G','column_norm_P','column_norm_M','excitation_fro','rho')
FIELDS=('delta_R_noise','Q','C','surrogate_gap','total_oracle_gap','output_lambda_min','output_lambda_max','output_condition','channel_strength')


def stats(values):
    v=np.asarray(values,dtype=float)
    if not len(v):
        return dict(n=0,mean=None,median=None,q25=None,q75=None,min=None,max=None,sd=None)
    return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),q25=float(np.quantile(v,.25)),
                q75=float(np.quantile(v,.75)),min=float(v.min()),max=float(v.max()),sd=float(v.std(ddof=1)) if len(v)>1 else 0.)


def build_conditional(rows):
    summaries=[]
    for i in d.IDS:
        for s in d.SNRS:
            for j in d.FRAMES:
                group=sorted([a for a in rows if a['trajectory_id']==i and a['snr_db']==s and a['frame_id']==j],key=lambda a:a['repeat_id'])
                assert len(group)==20 and [a['repeat_id'] for a in group]==list(range(20))
                row=dict(trajectory_id=i,snr_db=s,frame_id=j,repeat_count=20)
                for key in FIELDS:
                    row['mean_'+key]=float(np.mean([a[key] for a in group]))
                row['median_delta_R_noise']=float(np.median([a['delta_R_noise'] for a in group]))
                row['negative_fraction']=sum(a['negative_penalty'] for a in group)/20
                row['negative_cross_fraction']=sum(a['negative_cross'] for a in group)/20
                valid=[a['alignment_cosine'] for a in group if not a['alignment_undefined']]
                row['mean_cosine']=float(np.mean(valid)) if valid else None
                row['median_cosine']=float(np.median(valid)) if valid else None
                row['undefined_count']=20-len(valid)
                row['first_half_mean_delta']=float(np.mean([a['delta_R_noise'] for a in group[:10]]))
                row['second_half_mean_delta']=float(np.mean([a['delta_R_noise'] for a in group[10:]]))
                row['both_halves_negative']=bool(row['first_half_mean_delta']<0 and row['second_half_mean_delta']<0)
                row['repeat_sd_delta']=float(np.std([a['delta_R_noise'] for a in group],ddof=1))
                row['noise_MCSE_delta']=row['repeat_sd_delta']/np.sqrt(20)
                row.update(d.empirical([[a['alpha_r'],a['beta_r']] for a in group],[group[0]['alpha0'],group[0]['beta0']]))
                row.update({'clean_'+k:group[0]['clean_'+k] for k in GEOMETRY})
                row['clean_abs_rho']=abs(row['clean_rho'])
                d.close(row['mean_delta_R_noise'],row['mean_Q']+row['mean_C'])
                summaries.append(row)
    assert len(summaries)==240
    return summaries


def trajectory_means(rows):
    means=[]
    keys=[k for k,v in rows[0].items() if k not in ('trajectory_id','snr_db','frame_id','repeat_count') and isinstance(v,(int,float,bool))]
    for s in d.SNRS:
        for i in d.IDS:
            group=[a for a in rows if a['trajectory_id']==i and a['snr_db']==s]
            assert len(group)==6
            means.append(dict(trajectory_id=i,snr_db=s,windows=6,**{k:float(np.mean([a[k] for a in group])) for k in keys}))
    return means


def main():
    from scripts.run_sp_u2c import verify
    pre=verify()
    gate=d.load(d.OUT/'sp_u2c_run_gate.json')
    assert gate['passed'] and d.sha(d.OUT/'sp_u2c_repeat_rows.csv')==gate['rows_sha256']
    rows=d.read_rows(d.OUT/'sp_u2c_repeat_rows.csv')
    conditional=build_conditional(rows)
    tm=trajectory_means(conditional)
    d.csvsave(d.OUT/'sp_u2c_conditional_windows.csv',conditional)
    d.csvsave(d.OUT/'sp_u2c_trajectory_means.csv',tm)
    associations=[]
    for s in d.SNRS:
        group=[a for a in tm if a['snr_db']==s]
        for pred in ['clean_'+k for k in GEOMETRY if k!='rho']+['clean_abs_rho']:
            for outcome in ('RMS_c','V_c','trace_covariance'):
                rho=float(spearmanr([a[pred] for a in group],[a[outcome] for a in group]).statistic)
                associations.append(dict(snr_db=s,predictor=pred,outcome=outcome,unit='trajectory mean of six conditional windows',n=20,spearman=rho))
    d.csvsave(d.OUT/'sp_u2c_associations.csv',associations)
    report={};table1=[];table2=[];descriptive=[];examples=[];contrasts=[]
    for s in d.SNRS:
        raw=[a for a in rows if a['snr_db']==s]
        windows=[a for a in conditional if a['snr_db']==s]
        tr=[a for a in tm if a['snr_db']==s]
        r=dict(records=2400,conditional_windows=120,trajectories=20,
               negative_records=sum(a['negative_penalty'] for a in raw),negative_cross_records=sum(a['negative_cross'] for a in raw),
               negative_conditional_means=sum(a['mean_delta_R_noise']<0 for a in windows),
               negative_conditional_mean_cross=sum(a['mean_C']<0 for a in windows),
               negative_trajectory_means=sum(a['mean_delta_R_noise']<0 for a in tr),
               both_halves_negative=sum(a['both_halves_negative'] for a in windows),
               negative_fraction_gt_half=sum(a['negative_fraction']>.5 for a in windows),
               negative_fraction_ge_075=sum(a['negative_fraction']>=.75 for a in windows),
               all_repeats_negative=sum(a['negative_fraction']==1 for a in windows),
               undefined_cosines=sum(a['alignment_undefined'] for a in raw),
               surrogate_exceeds_abs_mean_noise=sum(a['mean_surrogate_gap']>abs(a['mean_delta_R_noise']) for a in windows),
               surrogate_exceeds_mean_Q=sum(a['mean_surrogate_gap']>a['mean_Q'] for a in windows))
        for key in ('delta_R_noise','Q','C','surrogate_gap','total_oracle_gap'):
            r['raw_'+key]=stats([a[key] for a in raw])
            r['conditional_mean_'+key]=stats([a['mean_'+key] for a in windows])
            r['trajectory_mean_'+key]=stats([a['mean_'+key] for a in tr])
        for key in ('negative_fraction','RMS_c','V_c','trace_covariance','variance_alpha','variance_beta','covariance_alpha_beta','bias_norm'):
            r['conditional_'+key]=stats([a[key] for a in windows])
            r['trajectory_'+key]=stats([a[key] for a in tr])
        for sign in ('negative','positive'):
            subset=[a for a in raw if (a['delta_R_noise']<0 if sign=='negative' else a['delta_R_noise']>0)]
            r['cosine_'+sign]=stats([a['alignment_cosine'] for a in subset if not a['alignment_undefined']])
            r['Q_'+sign]=stats([a['Q'] for a in subset]);r['C_'+sign]=stats([a['C'] for a in subset])
        report[str(s)]=r
        table1.append(dict(snr_db=s,trajectories=20,conditional_windows=120,repeats=20,
             mean_RMS_c=r['conditional_RMS_c']['mean'],median_RMS_c=r['conditional_RMS_c']['median'],
             mean_trace_covariance=r['conditional_trace_covariance']['mean'],
             mean_alpha_variance=r['conditional_variance_alpha']['mean'],mean_beta_variance=r['conditional_variance_beta']['mean'],
             mean_alpha_beta_covariance=r['conditional_covariance_alpha_beta']['mean'],negative_record_fraction=r['negative_records']/2400,
             negative_conditional_mean_fraction=r['negative_conditional_means']/120))
        table2.append(dict(snr_db=s,mean_Q=r['raw_Q']['mean'],mean_C=r['raw_C']['mean'],mean_delta_R_noise=r['raw_delta_R_noise']['mean'],
             mean_surrogate_gap=r['raw_surrogate_gap']['mean'],mean_total_oracle_gap=r['raw_total_oracle_gap']['mean']))
        median=float(np.median([a['mean_delta_R_noise'] for a in windows]))
        for sign in ('negative','positive'):
            candidates=[a for a in windows if (a['mean_delta_R_noise']<0 if sign=='negative' else a['mean_delta_R_noise']>0)]
            if candidates:
                winner=min(candidates,key=lambda a:(abs(a['mean_delta_R_noise']-median),a['trajectory_id'],a['frame_id']))
                examples.append(dict(snr_db=s,sign=sign,trajectory_id=winner['trajectory_id'],frame_id=winner['frame_id'],
                     population_median=median,selected_mean_delta=winner['mean_delta_R_noise'],selection='closest to within-SNR population median, constrained by sign; ties ID then frame'))
        for label,predicate in (('negative_mean',lambda a:a['mean_delta_R_noise']<0),('nonnegative_mean',lambda a:a['mean_delta_R_noise']>=0)):
            group=[a for a in windows if predicate(a)]
            for key in ('clean_sigma_min','clean_kappa_G','RMS_c','mean_surrogate_gap','mean_output_lambda_min','mean_output_lambda_max','mean_output_condition','mean_channel_strength','mean_Q','mean_C'):
                st=stats([a[key] for a in group])
                contrasts.append(dict(snr_db=s,group=label,quantity=key,**st))
    d.csvsave(d.OUT/'sp_u2c_table1_stability.csv',table1)
    d.csvsave(d.OUT/'sp_u2c_table2_risk_geometry.csv',table2)
    d.csvsave(d.OUT/'sp_u2c_representatives.csv',examples)
    d.csvsave(d.OUT/'sp_u2c_sign_group_contrasts.csv',contrasts)
    for s,r in report.items():
        for k,v in r.items():
            if isinstance(v,dict) and 'n' in v:
                descriptive.append(dict(snr_db=int(s),quantity=k,**v))
    d.csvsave(d.OUT/'sp_u2c_descriptive_statistics.csv',descriptive)
    d.save(d.OUT/'sp_u2c_analysis.json',dict(task='SP-U2C',report=report,associations=associations,
           raw_sha256=gate['rows_sha256'],conditional_sha256=d.sha(d.OUT/'sp_u2c_conditional_windows.csv'),
           clusters='20 trajectories; six overlapping windows; 20 independent receiver-noise repeats per fixed window; paired SNR atoms',
           no_p_values=True,no_bootstrap=True,no_formal_inference=True,pre_registered_protocol_sha256=pre['protocol_sha256'],access=d.ACCESS))
    print('PASS conditional summaries 240; trajectory summaries 40; associations',len(associations),flush=True)


if __name__=='__main__':main()
