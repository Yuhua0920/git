"""CSV-only SP-U4A descriptive analysis; no refit, seed selection or tuning."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u4a_common import *
from scipy.stats import spearmanr


def operator_analysis():
    cfg=config();ops={};stats={}
    for n in cfg['budgets']:
        with np.load(OUT/f'sp_u4a_operators_{n}.npz') as f:ops[n]={k:f[k] for k in f.files}
        with np.load(OUT/f'sp_u4a_statistics_{n}.npz') as f:stats[n]={k:f[k] for k in f.files}
    rows=[]
    for n in cfg['budgets']:
        d=dict(budget=n,source_mean_relative_error=rel_error(np.r_[stats[n]['Source_mean_p'],stats[n]['Source_mean_d']],
            np.r_[stats[500]['Source_mean_p'],stats[500]['Source_mean_d']],cfg['mean_denominator_epsilon']))
        for e in ('Source','P','M'):
            for k in ('mean_p','mean_d','rpp','rdp'):
                d[f'{e}_{k}_relative_error']=rel_error(stats[n][f'{e}_{k}'],stats[500][f'{e}_{k}'],cfg['mean_denominator_epsilon'] if k.startswith('mean') else 0.)
        for s in (10,20):
            row=dict(**d,snr_db=s)
            for e in ('Source','P','M'):
                row[f'W{e}_relative_error']=rel_error(ops[n][f'W{e}_{s}'],ops[500][f'W{e}_{s}'])
                row[f'K{e}_relative_error']=rel_error(ops[n][f'K{e}_{s}'],ops[500][f'K{e}_{s}'])
            for e in ('P','M'):
                row[f'B{e}_relative_error']=rel_error(ops[n][f'DeltaW{e}_{s}'],ops[500][f'DeltaW{e}_{s}'])
                row[f'B{e}4_relative_error']=rel_error(ops[n][f'A_{e}_{s}']@ops[n][f'Q_{e}_{s}'],ops[500][f'A_{e}_{s}']@ops[500][f'Q_{e}_{s}'])
                row[f'DeltaK{e}_relative_error']=rel_error(ops[n][f'DeltaK{e}_{s}'],ops[500][f'DeltaK{e}_{s}'])
            rows.append(row)
    csvsave(OUT/'sp_u4a_table5_operators.csv',rows)
    return rows


def main():
    prereg();firewall()
    if not readjson(OUT/'sp_u4a_development_gate.json')['passed']:raise ValueError('development gate FAIL')
    raw=csvread(OUT/'sp_u4a_trajectories.csv');windows=csvread(OUT/'sp_u4a_windows.csv');ops=operator_analysis()
    index={(r['condition'],int(r['snr_db']),r['budget'],r['method'],int(r['trajectory_id'])):r for r in raw}
    nmse=[];penalty=[];gain=[];paired=[];coord=[];co_raw=[];associations=[]
    for cond in config()['conditions']:
        for s in (10,20):
            for n in ('50','100','250','500','Original500'):
                for method in ('Source','LR4'):
                    selected=[index[cond,s,n,method,i] for i in range(100)]
                    errors=sum(float(r['error_energy']) for r in selected);truth=sum(float(r['truth_energy']) for r in selected)
                    nmse.append(dict(condition=cond,snr_db=s,budget=n,method=method,trajectories=100,aggregate_nmse_linear=errors/truth,
                        aggregate_nmse_db=float(10*np.log10(errors/truth)),**distribution([float(r['nmse_db']) for r in selected])))
                if n=='Original500':continue
                values=np.array([float(index[cond,s,n,'Source',i]['nmse_db'])-float(index[cond,s,n,'LR4',i]['nmse_db']) for i in range(100)])
                gain.append(dict(condition=cond,snr_db=s,budget=int(n),**distribution(values),fraction_positive=float(np.mean(values>0))))
                for method in ('Source','LR4'):
                    deltas=np.array([float(index[cond,s,n,method,i]['nmse_db'])-float(index[cond,s,'500',method,i]['nmse_db']) for i in range(100)])
                    if n!='500':penalty.append(dict(condition=cond,snr_db=s,budget=int(n),method=method,**distribution(deltas)))
                    for i,v in enumerate(deltas):paired.append(dict(condition=cond,snr_db=s,budget=int(n),method=method,trajectory_id=i,penalty_db=float(v)))
                wi={(int(r['trajectory_id']),int(r['frame_id'])):r for r in windows if (r['condition'],int(r['snr_db']),r['budget'])==(cond,s,n)}
                ref={(int(r['trajectory_id']),int(r['frame_id'])):r for r in windows if (r['condition'],int(r['snr_db']),r['budget'])==(cond,s,'500')}
                devs=[];alphas=[];betas=[]
                for i in range(100):
                    delta=[];aa=[];bb=[]
                    for j in range(4,10):
                        r=wi[i,j];r0=ref[i,j];c=np.array([float(r['alpha']),float(r['beta'])]);c0=np.array([float(r0['alpha']),float(r0['beta'])]);d=float(np.linalg.norm(c-c0))
                        delta.append(d);aa.append(c[0]);bb.append(c[1])
                        co_raw.append(dict(condition=cond,snr_db=s,budget=int(n),trajectory_id=i,frame_id=j,coordinate_deviation=d,alpha=float(c[0]),beta=float(c[1])))
                    devs.append(float(np.mean(delta)));alphas.append(float(np.mean(aa)));betas.append(float(np.mean(bb)))
                coord.append(dict(condition=cond,snr_db=s,budget=int(n),statistic='trajectory mean of six window deviations',**distribution(devs),alpha_mean=float(np.mean(alphas)),beta_mean=float(np.mean(betas))))
                if n!='500':
                    deltas=[float(index[cond,s,n,'LR4',i]['nmse_db'])-float(index[cond,s,'500','LR4',i]['nmse_db']) for i in range(100)]
                    rho=float(spearmanr(devs,deltas).statistic)
                    associations.append(dict(condition=cond,snr_db=s,budget=int(n),trajectory_spearman_coordinate_vs_lr4_penalty=rho,
                        interpretation='exploratory descriptive association; no causal attribution or U2 theory reopening'))
    csvsave(OUT/'sp_u4a_nmse_summary.csv',nmse)
    csvsave(OUT/'sp_u4a_table2_ID.csv',[r for r in nmse if r['condition']=='ID'])
    csvsave(OUT/'sp_u4a_table3_Composed.csv',[r for r in nmse if r['condition']=='Composed'])
    csvsave(OUT/'sp_u4a_budget_penalties.csv',penalty);csvsave(OUT/'sp_u4a_paired_trajectory_penalties.csv',paired)
    csvsave(OUT/'sp_u4a_table4_gains.csv',gain);csvsave(OUT/'sp_u4a_coordinate_summary.csv',coord)
    csvsave(OUT/'sp_u4a_coordinate_deviations.csv',co_raw);csvsave(OUT/'sp_u4a_exploratory_associations.csv',associations)
    def get(rows,**kwargs):return next(r for r in rows if all(str(r[k])==str(v) for k,v in kwargs.items()))
    facts={};hyp=[]
    for s in (10,20):
        source_errors=[get(ops,budget=n,snr_db=s)['WSource_relative_error'] for n in (50,100,250,500)]
        id_median=[get(nmse,condition='ID',snr_db=s,budget=n,method='Source')['median'] for n in (50,100,250,500)]
        composed=[get(nmse,condition='Composed',snr_db=s,budget=n,method='LR4')['median'] for n in (50,100,250,500)]
        step=[composed[i]-composed[i+1] for i in range(3)]
        facts[str(s)]=dict(source_operator_errors=source_errors,id_source_medians=id_median,composed_lr4_medians=composed,
            composed_adjacent_improvement_db=step,correction_error_exceeds_source={str(n):{e:get(ops,budget=n,snr_db=s)[f'B{e}_relative_error']>get(ops,budget=n,snr_db=s)['WSource_relative_error'] for e in ('P','M')} for n in (50,100,250)},
            composed_gains_100_250=[get(gain,condition='Composed',snr_db=s,budget=n)['median'] for n in (100,250)],
            id_lr4_gains=[get(gain,condition='ID',snr_db=s,budget=n)['median'] for n in (50,100,250,500)])
    H1=all(all(a>=b for a,b in zip(f['source_operator_errors'],f['source_operator_errors'][1:])) and all(a>=b for a,b in zip(f['id_source_medians'],f['id_source_medians'][1:])) for f in facts.values())
    H2=all(all(all(v.values()) for v in f['correction_error_exceeds_source'].values()) for f in facts.values())
    H3=all(f['composed_adjacent_improvement_db'][1]>=f['composed_adjacent_improvement_db'][2]>=0 for f in facts.values())
    H4=all(min(f['composed_gains_100_250'])>0 for f in facts.values())
    descriptions={
        'H1':('Source operator and ID Source trajectory median approach500 monotonically' if H1 else 'At least one Source/operator or ID median sequence is nonmonotonic', 'supported for this nested realization' if H1 else 'mixed / subset-replicate consideration'),
        'H2':('All dense correction relative errors exceed Source W errors' if H2 else 'Corrections do not uniformly exceed Source W errors','descriptive support only; relative denominators differ' if H2 else 'not uniformly supported'),
        'H3':('100 to250 median improvement exceeds250 to500 at both SNR' if H3 else 'Saturation pattern is not consistent at both SNR','descriptive support; no sufficient N claim' if H3 else 'mixed / subset-replicate consideration'),
        'H4':('Composed median same-budget gain positive at100 and250 for both SNR' if H4 else 'At least one100/250 Composed median gain is nonpositive','supported for this realization' if H4 else 'not uniformly supported'),
        'H5':('ID paired penalties, gain distributions and worst trajectories retained; no arbitrary large-degradation threshold defined','safeguard audited descriptively; mentor judgment required')}
    alternatives={'H1':'single nested subset; environment gain variations; finite-sample centered covariance',
        'H2':'difference of independently estimated operators; differing reference norms; origin weighted compression',
        'H3':'single subset permutation; compression/CV kernels co-vary with N; condition channel-strength differences',
        'H4':'pilot-dependent coordinates and compression co-vary; weak trajectories may have negative gains',
        'H5':'current pilot observation noise and small-budget kernel changes; outliers retained'}
    for h,(observation,verdict) in descriptions.items():hyp.append(dict(hypothesis=h,observation=observation,interpretation=verdict,alternative_explanation=alternatives[h],evidence_strength='development descriptive, single nested realization',verdict=verdict))
    csvsave(OUT/'sp_u4a_table6_hypotheses.csv',hyp)
    cases='Case C: nonmonotonic or mixed sensitivity; replicated subsets should be considered' if not (H1 and H3) else 'Case A/B: monotonic pattern; mentor to distinguish magnitude without an invented sufficiency cutoff'
    original_diffs=[]
    for cond in config()['conditions']:
        for s in (10,20):
            for m in ('Source','LR4'):
                original_diffs.extend(abs(float(index[cond,s,'500',m,i]['nmse_db'])-float(index[cond,s,'Original500',m,i]['nmse_db'])) for i in range(100))
    anomalies=sorted([dict(condition=r['condition'],snr_db=int(r['snr_db']),budget=int(r['budget']),trajectory_id=int(r['trajectory_id']),method=r['method'],penalty_db=float(r['penalty_db'])) for r in paired if r['budget']!=500],key=lambda r:abs(r['penalty_db']),reverse=True)[:12]
    powers=csvread(OUT/'sp_u4a_pair_hashes.csv')
    save(OUT/'sp_u4a_analysis.json',dict(passed=True,hypothesis_facts=facts,hypothesis_audit=hyp,case=cases,subset_replicate_consideration=not (H1 and H3),
        original500_max_trajectory_db_difference=max(original_diffs),largest_absolute_paired_penalties=anomalies,
        condition_score_truth_energy={c:distribution([float(r['score_truth_energy']) for r in powers if r['condition']==c]) for c in config()['conditions']},
        condition_score_useful_receive_power={c:distribution([float(r['score_useful_receive_power']) for r in powers if r['condition']==c]) for c in config()['conditions']},
        statistical_unit='trajectory100 per condition/SNR; six overlapping windows summarized within trajectory; budgets paired; no p-values, formal CI or claim of independent validation',
        means_epsilon=config()['mean_denominator_epsilon'],epsilon_engineering_only=True,no_outliers_removed=True,no_theory_reopened=True,
        aggregate_definition='sum100 trajectory error energies / sum100 trajectory truth energies; dB trajectory summaries separately',
        next_action='STOP for mentor review; subset replication recommendation only; no automatic execution'))
    print(json.dumps(dict(case=cases,facts=facts,original500_max_db_difference=max(original_diffs)),indent=2),flush=True)


if __name__=='__main__':main()
