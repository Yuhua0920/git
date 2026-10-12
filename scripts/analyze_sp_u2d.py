"""Trajectory-first summaries, paired contrasts and frozen energy strata."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2d_diagnostic as d
d.install_guard()
import numpy as np

AGE_FIELDS=('influence_norm','Q','C','delta_R','alignment_cosine','H_age_norm','H_age_fro','b_age_norm','noisy_frame_energy','clean_frame_energy')
CONTRAST_FIELDS=('D_coord','D_C','D_R','D_cosine','D_Hnorm','D_bnorm','D_noisy_energy')


def stats(values):
    x=np.asarray([v for v in values if v is not None],float)
    if not len(x):return dict(n=0,mean=None,median=None,q25=None,q75=None,min=None,max=None,sd=None,negative=0,positive=0,zero=0)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),q25=float(np.quantile(x,.25)),q75=float(np.quantile(x,.75)),
                min=float(x.min()),max=float(x.max()),sd=float(x.std(ddof=1)) if len(x)>1 else 0.,
                negative=int(np.sum(x<0)),positive=int(np.sum(x>0)),zero=int(np.sum(x==0)))


def mean(group,key):
    values=[a[key] for a in group if a[key] is not None]
    return float(np.mean(values)) if values else None


def aggregate(rows,fields,extra=()):
    groups={}
    for a in rows:
        key=tuple(a[k] for k in ('snr_db','trajectory_id',*extra));groups.setdefault(key,[]).append(a)
    result=[]
    for key,group in sorted(groups.items()):
        row=dict(zip(('snr_db','trajectory_id',*extra),key));row['records']=len(group)
        row.update({k:mean(group,k) for k in fields})
        result.append(row)
    return result


def main():
    from scripts.run_sp_u2d import verify
    pre=verify();gate=d.load(d.OUT/'sp_u2d_run_gate.json');assert gate['passed']
    assert d.sha(d.OUT/'sp_u2d_age_rows.csv')==gate['raw_sha256']
    assert d.sha(d.OUT/'sp_u2d_paired_rows.csv')==gate['paired_sha256']
    raw=d.read_rows(d.OUT/'sp_u2d_age_rows.csv');paired=d.read_rows(d.OUT/'sp_u2d_paired_rows.csv')
    conditional=aggregate(raw,AGE_FIELDS,('frame_id','age'))
    conditional_pair=aggregate(paired,CONTRAST_FIELDS,('frame_id',))
    trajectory_age=aggregate(raw,AGE_FIELDS,('age',))
    trajectory_pair=aggregate(paired,CONTRAST_FIELDS)
    assert len(conditional)==1200 and len(conditional_pair)==240 and len(trajectory_age)==200 and len(trajectory_pair)==40
    d.csvsave(d.OUT/'sp_u2d_conditional_ages.csv',conditional)
    d.csvsave(d.OUT/'sp_u2d_conditional_contrasts.csv',conditional_pair)
    d.csvsave(d.OUT/'sp_u2d_trajectory_ages.csv',trajectory_age)
    d.csvsave(d.OUT/'sp_u2d_trajectory_contrasts.csv',trajectory_pair)
    age_stats=[];table1=[];table2=[]
    for s in d.SNRS:
        for age in d.AGES:
            group=[a for a in raw if a['snr_db']==s and a['age']==age]
            tr=[a for a in trajectory_age if a['snr_db']==s and a['age']==age]
            for key in AGE_FIELDS:
                for label,data in [('raw_repeated_records',group),('trajectory_means',tr)]:
                    age_stats.append(dict(snr_db=s,age=age,quantity=key,unit=label,**stats([a[key] for a in data])))
            influence=stats([a['influence_norm'] for a in tr]);raw_inf=stats([a['influence_norm'] for a in group])
            table1.append(dict(snr_db=s,age=age,trajectories=len(tr),raw_records=len(group),
                   mean_influence=influence['mean'],median_trajectory_influence=influence['median'],
                   q25_trajectory_influence=influence['q25'],q75_trajectory_influence=influence['q75'],raw_median_influence=raw_inf['median'],
                   singular_count=sum(a['singular'] for a in group),undefined_cosine=sum(a['alignment_undefined'] for a in group)))
            Q,C,R,cos=[stats([a[k] for a in tr]) for k in ('Q','C','delta_R','alignment_cosine')]
            table2.append(dict(snr_db=s,age=age,mean_Q=Q['mean'],median_trajectory_Q=Q['median'],
                   mean_C=C['mean'],median_trajectory_C=C['median'],mean_delta_R=R['mean'],median_trajectory_delta_R=R['median'],
                   mean_cosine=cos['mean'],median_trajectory_cosine=cos['median'],
                   raw_median_cosine=stats([a['alignment_cosine'] for a in group])['median'],
                   negative_record_fraction=sum(a['delta_R'] is not None and a['delta_R']<0 for a in group)/len(group),
                   negative_trajectory_mean_count=R['negative'],singular_rate=sum(a['singular'] for a in group)/len(group)))
    d.csvsave(d.OUT/'sp_u2d_age_statistics.csv',age_stats);d.csvsave(d.OUT/'sp_u2d_table1_influence.csv',table1);d.csvsave(d.OUT/'sp_u2d_table2_risk.csv',table2)
    contrast_stats=[];pairwise=[];monotonic=[]
    for s in d.SNRS:
        group=[a for a in trajectory_pair if a['snr_db']==s]
        for key in CONTRAST_FIELDS:contrast_stats.append(dict(snr_db=s,quantity=key,unit='trajectory means',**stats([a[key] for a in group])))
        profiles={i:{int(a['age']):a for a in trajectory_age if a['snr_db']==s and a['trajectory_id']==i} for i in d.IDS}
        for age in (1,2,3,4):
            for key in ('influence_norm','C','delta_R','alignment_cosine'):
                differences=[profiles[i][0][key]-profiles[i][age][key] for i in d.IDS if profiles[i][0][key] is not None and profiles[i][age][key] is not None]
                pairwise.append(dict(snr_db=s,past_age=age,quantity=key,**stats(differences)))
        for key,expected in [('influence_norm','decreasing'),('C','increasing'),('delta_R','increasing'),('alignment_cosine','increasing')]:
            count=0;extreme=0
            for i in d.IDS:
                values=np.array([profiles[i][a][key] for a in d.AGES])
                delta=np.diff(values)
                count+=bool(np.all(delta<=0) if expected=='decreasing' else np.all(delta>=0))
                extreme+=bool(values[0]==(values.max() if expected=='decreasing' else values.min()))
            monotonic.append(dict(snr_db=s,quantity=key,expected_age_profile=expected,trajectory_count=20,monotonic_count=count,current_extreme_count=extreme))
    d.csvsave(d.OUT/'sp_u2d_table3_paired_contrasts.csv',contrast_stats)
    d.csvsave(d.OUT/'sp_u2d_current_vs_each_age.csv',pairwise);d.csvsave(d.OUT/'sp_u2d_recency_checks.csv',monotonic)
    energy_rows=[];energy_clusters=[]
    for s in d.SNRS:
        allrows=[a for a in paired if a['snr_db']==s]
        selectors=[('all',lambda a:True),('similar_clean_energy_10pct',lambda a:a['similar_clean_energy']),
                   ('similar_noisy_energy_10pct',lambda a:a['noisy_similar_energy'])]
        selectors += [(f'clean_energy_ratio_tercile_{q}',lambda a,q=q:a['energy_tercile']==q) for q in (1,2,3)]
        for label,predicate in selectors:
            subset=[a for a in allrows if predicate(a)]
            tr=aggregate(subset,CONTRAST_FIELDS+('clean_current_past_ratio',)) if subset else []
            for a in tr:energy_clusters.append(dict(stratum=label,**a))
            windows=len({(a['trajectory_id'],a['frame_id']) for a in subset})
            for key in CONTRAST_FIELDS+('clean_current_past_ratio',):
                energy_rows.append(dict(snr_db=s,stratum=label,quantity=key,raw_records=len(subset),fixed_windows=windows,
                       unique_trajectories=len(tr),unit='within-stratum trajectory means',**stats([a[key] for a in tr])))
    d.csvsave(d.OUT/'sp_u2d_energy_strata.csv',energy_rows)
    d.csvsave(d.OUT/'sp_u2d_energy_trajectory_contrasts.csv',energy_clusters)
    # Separate normalized profiles retain every trajectory and nonmonotonicity.
    normalized=[]
    for s in d.SNRS:
        for i in d.IDS:
            group=sorted([a for a in trajectory_age if a['snr_db']==s and a['trajectory_id']==i],key=lambda a:a['age'])
            for key in ('influence_norm','C','delta_R'):
                values=np.array([a[key] for a in group]);scale=float(np.mean(abs(values)))
                assert scale>0
                for age,v in enumerate(values):normalized.append(dict(snr_db=s,trajectory_id=i,age=age,quantity=key,value=float(v/scale),scale=scale))
    d.csvsave(d.OUT/'sp_u2d_normalized_profiles.csv',normalized)
    d.save(d.OUT/'sp_u2d_analysis.json',dict(task='SP-U2D',age_summary=table1,risk_summary=table2,
          trajectory_contrasts=contrast_stats,current_vs_each_age=pairwise,recency=monotonic,energy_strata=energy_rows,
          primary_unit='20 trajectories per SNR; contrasts computed within record before aggregation',
          no_p_values=True,no_bootstrap=True,no_new_hypotheses=True,protocol_sha256=pre['protocol_sha256'],access=d.ACCESS))
    print('PASS200 trajectory-age means;40 trajectory contrasts;1200 conditional-age summaries;240 conditional contrasts')


if __name__=='__main__':main()
