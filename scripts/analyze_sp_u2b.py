"""CSV-only low-assumption SP-U2B summaries; trajectory is the independent unit."""
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2b_diagnostic as d
d.install_guard()
import numpy as np
from scipy.stats import spearmanr,rankdata

METRICS=['clean_sigma_min','clean_lambda_min','clean_column_norm_P','clean_column_norm_M',
         'clean_excitation_fro','clean_abs_rho','clean_kappa_G','noisy_sigma_min','noisy_kappa_G',
         'relative_E_spectral','relative_E_fro','joint_bound','joint_forcing_norm','exact_score_change_norm',
         'current_clean_pilot_norm','clean_channel_mean_power','output_lambda_max']
PAIRS=[(x,'delta_noise_norm') for x in METRICS]+[
    ('delta_noise_norm','coord_output_mse'),('delta_noise_norm','delta_R_noise'),
    ('coord_output_mse','delta_R_noise'),('delta_surrogate_norm','delta_R_surrogate'),
    ('delta_total_norm','delta_R_total'),('coord_output_quadratic','coord_output_energy'),
    ('clean_channel_mean_power','MSE_noisyCV')]


def read_rows(path):
    with Path(path).open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for key,value in list(r.items()):
            if value=='':r[key]=None
            elif value in ('True','False'):r[key]=value=='True'
            else:
                try:r[key]=float(value)
                except ValueError:pass
        r['clean_abs_rho']=abs(r['clean_rho'])
        delta=np.array([r['delta_surrogate_alpha'],r['delta_surrogate_beta']])
        M=np.array([[r['M_PP'],r['M_PM']],[r['M_PM'],r['M_MM']]])
        r['surrogate_output_mse']=float(delta@M@delta/r['ndata'])
        d.close(r['surrogate_output_mse'],r['delta_R_surrogate'])
    return rows


def distribution(values):
    v=np.array(values,float)
    assert v.size and np.isfinite(v).all()
    quant=np.percentile(v,[0,5,25,50,75,90,95,100])
    return dict(n=len(v),mean=float(v.mean()),std=float(v.std(ddof=1)) if len(v)>1 else 0.,
                minimum=float(quant[0]),p05=float(quant[1]),q25=float(quant[2]),median=float(quant[3]),
                q75=float(quant[4]),IQR=float(quant[4]-quant[2]),p90=float(quant[5]),p95=float(quant[6]),maximum=float(quant[7]))


def corr(rows,x,y):
    a,b=np.array([r[x] for r in rows]),np.array([r[y] for r in rows])
    if len(set(a))<2 or len(set(b))<2:return None
    return float(spearmanr(a,b).statistic)


def trajectory_means(rows):
    out=[]
    for s in d.SNRS:
        for i in range(100):
            subset=[r for r in rows if r['snr_db']==s and r['trajectory_id']==i]
            assert len(subset)==6
            value={k:float(np.mean([r[k] for r in subset])) for k in subset[0]
                   if isinstance(subset[0][k],(float,int)) and not isinstance(subset[0][k],bool)}
            value.update(snr_db=s,trajectory_id=i,scored_windows=6)
            out.append(value)
    return out


def within_rank_association(rows,x,y):
    a=rankdata([r[x] for r in rows]);b=rankdata([r[y] for r in rows])
    for i in range(100):
        idx=[j for j,r in enumerate(rows) if r['trajectory_id']==i]
        a[idx]-=a[idx].mean();b[idx]-=b[idx].mean()
    return float(np.corrcoef(a,b)[0,1])


def quartile_rows(rows,s,scope,x,y):
    order=sorted(rows,key=lambda r:(r[x],r['trajectory_id'],r.get('frame_id',0)))
    out=[]
    for q,indices in enumerate(np.array_split(np.arange(len(order)),4),1):
        group=[order[i] for i in indices]
        out.append(dict(snr_db=s,scope=scope,predictor=x,outcome=y,quartile=q,n=len(group),
                        predictor_min=min(r[x] for r in group),predictor_max=max(r[x] for r in group),
                        outcome_median=float(np.median([r[y] for r in group])),
                        outcome_q25=float(np.percentile([r[y] for r in group],25)),
                        outcome_q75=float(np.percentile([r[y] for r in group],75))))
    return out


def main():
    gate=d.load(d.OUT/'sp_u2b_development_gate.json')
    assert gate['all_checks_passed'] and gate['trajectories']==100
    rows=read_rows(d.OUT/'sp_u2b_development_windows.csv')
    assert len(rows)==1200
    trajectories=trajectory_means(rows)
    d.csvsave(d.OUT/'sp_u2b_trajectory_means.csv',trajectories)
    allmetrics=METRICS+['delta_noise_norm','delta_surrogate_norm','delta_total_norm',
                       'delta_R_noise','delta_R_surrogate','delta_R_total','coord_output_mse',
                       'MSE_noisyCV','MSE_cleanCV','MSE_channelOracle',
                       'compression_output_mse','signed_compression_MSE_penalty','MSE_clean_input_rank4_oracle',
                       'MSE_noisy_input_dense_oracle','rank4_vs_dense_oracle_MSE',
                       'representation_vector_mse','coordinate_vector_mse','compression_vector_mse','observation_noise_vector_mse',
                       'source_clean_MSE','source_noisy_MSE','noise_penalty_cross_term']
    stats,assoc,quartiles=[],[],[]
    for s in d.SNRS:
        window=[r for r in rows if r['snr_db']==s]
        traj=[r for r in trajectories if r['snr_db']==s]
        for scope,group in (('windows_descriptive',window),('trajectory_means',traj)):
            for metric in allmetrics:
                stats.append(dict(snr_db=s,scope=scope,metric=metric,**distribution([r[metric] for r in group])))
            for x,y in PAIRS:
                assoc.append(dict(snr_db=s,scope=scope,predictor=x,outcome=y,n=len(group),spearman=corr(group,x,y),
                                  within_trajectory_demeaned_rank_association=within_rank_association(group,x,y) if scope=='windows_descriptive' else None))
                quartiles.extend(quartile_rows(group,s,scope,x,y))
        # Gain-stratified sensitivity check, descriptive n=25 trajectories per stratum.
        ordered=sorted(traj,key=lambda r:r['clean_channel_mean_power'])
        for q,indices in enumerate(np.array_split(np.arange(100),4),1):
            group=[ordered[i] for i in indices]
            for x in ('clean_sigma_min','clean_kappa_G','relative_E_spectral','joint_bound'):
                assoc.append(dict(snr_db=s,scope=f'trajectory_gain_quartile_{q}',predictor=x,outcome='delta_noise_norm',n=25,
                                  spearman=corr(group,x,'delta_noise_norm'),within_trajectory_demeaned_rank_association=None))
    d.csvsave(d.OUT/'sp_u2b_statistics.csv',stats)
    d.csvsave(d.OUT/'sp_u2b_associations.csv',assoc)
    d.csvsave(d.OUT/'sp_u2b_quartiles.csv',quartiles)
    table1=[r for r in stats if r['scope']=='trajectory_means' and r['metric'] in
            ('clean_sigma_min','clean_kappa_G','relative_E_spectral','delta_noise_norm')]
    table2=[r for r in stats if r['scope']=='trajectory_means' and r['metric'] in
            ('delta_noise_norm','delta_surrogate_norm','delta_R_noise','delta_R_surrogate')]
    d.csvsave(d.OUT/'sp_u2b_table1_geometry.csv',table1)
    d.csvsave(d.OUT/'sp_u2b_table2_gaps.csv',table2)
    anomaly,bounds=[],[]
    for s in d.SNRS:
        group=[r for r in rows if r['snr_db']==s]
        cutoff=float(np.percentile([r['delta_noise_norm'] for r in group],75))
        near=[r for r in group if r['near_zero_coordinate_change']]
        ratios=[r['bound_actual_ratio'] for r in group if not r['near_zero_coordinate_change']]
        bounds.append(dict(snr_db=s,near_zero_count=len(near),near_zero_bounds=[r['joint_bound'] for r in near],
                           nonzero_ratio_statistics=distribution(ratios)))
        improved=[r for r in group if r['conditioning_improved']]
        inc=[r for r in group if r['minimum_singular_increased']]
        both=[r for r in group if (r['conditioning_improved'] or r['minimum_singular_increased']) and r['delta_noise_norm']>=cutoff]
        candidates=sorted(both,key=lambda r:(abs(r['delta_noise_norm']-np.median([v['delta_noise_norm'] for v in both])),r['trajectory_id'],r['frame_id']))
        # Representative near subgroup median; no hand-picked maximum.
        representative=candidates[0] if candidates else None
        lower=sorted(group,key=lambda r:r['delta_noise_norm'])[:150]
        anomaly.append(dict(snr_db=s,windows=600,upper_quartile_noise_norm=cutoff,
                            conditioning_improved_count=len(improved),conditioning_improved_fraction=len(improved)/600,
                            sigma_min_increased_count=len(inc),sigma_min_increased_fraction=len(inc)/600,
                            improved_or_increased_and_large_count=len(both),fraction_of_all_windows=len(both)/600,
                            affected_trajectories=len({r['trajectory_id'] for r in both}),
                            representative=representative,
                            large_subgroup_noise_penalty_negative=sum(r['delta_R_noise']<0 for r in both),
                            low_noise_quartile_surrogate_norm=distribution([r['delta_surrogate_norm'] for r in lower]),
                            low_noise_quartile_surrogate_penalty=distribution([r['delta_R_surrogate'] for r in lower]),
                            surrogate_larger_than_noise_coordinate_count=sum(r['delta_surrogate_norm']>r['delta_noise_norm'] for r in group),
                            surrogate_penalty_larger_than_abs_noise_penalty_count=sum(r['delta_R_surrogate']>abs(r['delta_R_noise']) for r in group),
                            signed_compression_penalty_larger_than_abs_noise_penalty_count=sum(abs(r['signed_compression_MSE_penalty'])>abs(r['delta_R_noise']) for r in group),
                            negative_noise_penalties=sum(r['delta_R_noise']<0 for r in group),
                            negative_surrogate_penalties=sum(r['delta_R_surrogate']<0 for r in group),
                            negative_total_penalties=sum(r['delta_R_total']<0 for r in group),
                            negative_compression_penalties=sum(r['signed_compression_MSE_penalty']<0 for r in group),
                            weyl_positive=sum(r['weyl_positive'] for r in group)))
    paired={s:{(r['trajectory_id'],r['frame_id']):r for r in rows if r['snr_db']==s} for s in d.SNRS}
    keys=list(paired[10])
    same_window=dict(windows=600,noise_coordinate_norm_larger_at_10db=sum(paired[10][k]['delta_noise_norm']>paired[20][k]['delta_noise_norm'] for k in keys),
                     noise_MSE_penalty_larger_at_10db=sum(paired[10][k]['delta_R_noise']>paired[20][k]['delta_R_noise'] for k in keys))
    power=gate['power_and_channel_hashes']
    sig=np.sum([r['signal_energy_per_rx'] for r in power],axis=0)
    noise=np.sum([r['unit_noise_energy_per_rx'] for r in power],axis=0)
    count=sum(r['useful_samples_per_rx'] for r in power)
    snr_audit=[dict(snr_db=s,source_power=d.POWER,sigma2=d.VARIANCES[s],
                     development_useful_power_per_rx=(sig/count).tolist(),
                     development_realized_noise_power_per_rx=(d.VARIANCES[s]*noise/count).tolist(),
                     realized_receive_snr_per_rx_db=(10*np.log10(sig/(d.VARIANCES[s]*noise))).tolist(),
                     recalibrated=False,normalization=False) for s in d.SNRS]
    identity_keys=['normal_equation_identity_error_max','joint_identity_error','propagation_identity_error',
                   'propagation_pure_relative_error','oracle_LS_identity_error','decomposition_identity_error',
                   'clean_condition_square_relative_error','noisy_condition_square_relative_error']
    identities={key:max(r[key] for r in rows) for key in identity_keys}
    # Descriptive outlier flags use preregistered quartiles; no rows removed.
    outliers=[]
    for s in d.SNRS:
        group=[r for r in rows if r['snr_db']==s]
        q1,q3=np.percentile([r['delta_noise_norm'] for r in group],[25,75])
        outliers.append(dict(snr_db=s,convention='Tukey q75+1.5*IQR descriptive',cutoff=float(q3+1.5*(q3-q1)),
                             count=sum(r['delta_noise_norm']>q3+1.5*(q3-q1) for r in group),excluded=0))
    pooled=[]
    for x,y in PAIRS:
        pooled.append(dict(predictor=x,outcome=y,pooled_snr_window_spearman=corr(rows,x,y),
                           caveat='SNR confounding; primary inference uses separate trajectory associations'))
    result=dict(task='SP-U2B',trajectories=100,unique_windows=600,paired_snr_window_rows=1200,
                statistical_unit='trajectory',smoke_used_for_science=False,bootstrap=False,CI=False,
                distributions=stats,associations=assoc,quartiles=quartiles,bound_ratios=bounds,
                conditioning_anomalies=anomaly,same_window_SNR_pairing=same_window,
                snr_audit=snr_audit,identity_error_max=identities,
                near_singular_count=sum(r['clean_near_singular'] or r['noisy_near_singular'] for r in rows),
                rank_failures=0,NaN_count=0,bound_failures=sum(r['bound_failure'] for r in rows),outliers=outliers,
                pooled_snr_associations=pooled,access=d.ACCESS,
                raw_sha256=d.sha(d.OUT/'sp_u2b_development_windows.csv'),
                analysis_source_sha256=d.sha(Path(__file__)))
    d.save(d.OUT/'sp_u2b_analysis.json',result)
    print('CSV-only analysis complete; all 1200 rows retained; independent units=100; no CI/p-values',flush=True)


if __name__=='__main__':main()
