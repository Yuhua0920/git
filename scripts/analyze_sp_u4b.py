"""Replicate-first SP-U4B descriptive analysis. No formal inference or refitting."""
import collections
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u4b_common import *


def main():
    p=prereg();firewall()
    if not readjson(OUT/'sp_u4b_evaluation_gate.json')['passed']:raise ValueError('evaluation incomplete')
    tr=csvread(OUT/'sp_u4b_trajectories.csv');wr=csvread(OUT/'sp_u4b_windows.csv')
    ix={(r['replicate'],r['condition'],int(r['snr_db']),int(r['budget']),r['method'],int(r['trajectory_id'])):r for r in tr}
    summary=[];gains=[];tails=[];individual=[];coordinates=[];orders=[];across=[];operator=[];descriptors=[]
    windowix={(r['replicate'],r['condition'],int(r['snr_db']),int(r['budget']),int(r['trajectory_id']),int(r['frame_id'])):r for r in wr}
    for cond in p['config']['conditions']:
        for s in (10,20):
            for rep in p['replicates']:
                for n in (50,100,250,500):
                    keyrep='Common500' if n==500 else rep
                    v={m:np.array([float(ix[keyrep,cond,s,n,m,i]['nmse_db']) for i in range(100)]) for m in ('Source','LR4')}
                    gain=v['Source']-v['LR4'];degradation=-gain
                    gains.append(dict(replicate=rep,condition=cond,snr_db=s,budget=n,**distribution(gain),positive_fraction=float(np.mean(gain>0)),
                        degradation_median=float(np.median(degradation)),degradation_p95=float(np.quantile(degradation,.95)),degradation_maximum=float(max(0,degradation.max()))))
                    for method in ('Source','LR4'):
                        ref=np.array([float(ix['Common500',cond,s,500,method,i]['nmse_db']) for i in range(100)]);penalty=v[method]-ref
                        rows=[ix[keyrep,cond,s,n,method,i] for i in range(100)]
                        agg=sum(float(r['error_energy']) for r in rows)/sum(float(r['truth_energy']) for r in rows)
                        tail=signed_tail(penalty,range(100))
                        summary.append(dict(replicate=rep,condition=cond,snr_db=s,budget=n,method=method,trajectories=100,aggregate_nmse_linear=agg,
                            **{'nmse_'+k:x for k,x in distribution(v[method]).items()},**{'penalty_'+k:x for k,x in distribution(penalty).items()},
                            p95_positive_penalty=tail['p95_positive_penalty'],maximum_positive_penalty=tail['maximum_positive_penalty']))
                        if n!=500:
                            tails.append(dict(replicate=rep,condition=cond,snr_db=s,budget=n,method=method,**{k:json.dumps(x) if isinstance(x,list) else x for k,x in tail.items()}))
                            individual.extend(dict(replicate=rep,condition=cond,snr_db=s,budget=n,method=method,trajectory_id=i,penalty_db=float(x)) for i,x in enumerate(penalty))
                    if n!=500:
                        deviations=[]
                        for i in range(100):
                            ds=[]
                            for j in range(4,10):
                                w=windowix[rep,cond,s,n,i,j];w0=windowix['Common500',cond,s,500,i,j]
                                ds.append(np.linalg.norm([float(w['alpha'])-float(w0['alpha']),float(w['beta'])-float(w0['beta'])]))
                            deviations.append(float(np.mean(ds)))
                        coordinates.append(dict(replicate=rep,condition=cond,snr_db=s,budget=n,**distribution(deviations)))
                for method in ('Source','LR4'):
                    med={n:next(r['nmse_median'] for r in summary if (r['replicate'],r['condition'],r['snr_db'],r['budget'],r['method'])==(rep,cond,s,n,method)) for n in (50,100,250,500)}
                    orders.append(dict(replicate=rep,condition=cond,snr_db=s,method=method,ordering_worst_to_best=json.dumps(sorted(med,key=lambda n:-med[n])),
                        strict_50_100_250_500=med[50]>med[100]>med[250]>med[500],strict_reverse=med[50]<med[100]<med[250]<med[500]))
            for method in ('Source','LR4'):
                reference=next(r['nmse_median'] for r in summary if (r['replicate'],r['condition'],r['snr_db'],r['budget'],r['method'])==('R0',cond,s,500,method))
                for n in (50,100,250,500):
                    selected=[next(r for r in summary if (r['replicate'],r['condition'],r['snr_db'],r['budget'],r['method'])==(rep,cond,s,n,method)) for rep in p['replicates']]
                    vals=np.array([r['nmse_median'] for r in selected]);shift=vals-reference;paired=np.array([r['penalty_median'] for r in selected])
                    gain5=np.array([next(r['median'] for r in gains if (r['replicate'],r['condition'],r['snr_db'],r['budget'])==(rep,cond,s,n)) for rep in p['replicates']])
                    across.append(dict(condition=cond,snr_db=s,budget=n,method=method,raw5_central_medians=json.dumps(vals.tolist()),raw5_paired_penalty_medians=json.dumps(paired.tolist()),
                        budget_displacement_B=float(np.median(shift)),subset_spread_IQR_V=float(np.quantile(vals,.75)-np.quantile(vals,.25)),
                        central_replicate_median=float(np.median(vals)),central_range_min=float(vals.min()),central_range_max=float(vals.max()),central_range_width=float(np.ptp(vals)),
                        central_displacement_positive_count=int(np.sum(shift>0)),paired_penalty_positive_count=int(np.sum(paired>0)),
                        paired_penalty_replicate_median=float(np.median(paired)),paired_penalty_replicate_iqr=float(np.quantile(paired,.75)-np.quantile(paired,.25)),
                        paired_penalty_range_min=float(paired.min()),paired_penalty_range_max=float(paired.max()),
                        raw5_gain_medians=json.dumps(gain5.tolist()),gain_replicate_median=float(np.median(gain5)),gain_replicate_iqr=float(np.quantile(gain5,.75)-np.quantile(gain5,.25)),
                        gain_range_min=float(gain5.min()),gain_range_max=float(gain5.max()),gain_positive_sign_count=int(np.sum(gain5>0))))
    common=load_arrays('R0',500);commonstats=load_arrays('R0',500,'statistics');clean=np.load(OUT/'sp_u4a_clean.npy',mmap_mode='r')
    for rep in p['replicates']:
        for n in (50,100,250):
            a=load_arrays(rep,n);st=load_arrays(rep,n,'statistics')
            for s in (10,20):
                row=dict(replicate=rep,budget=n,snr_db=s,Ws=rel_error(a[f'Wsource_{s}'],common[f'Wsource_{s}']))
                for e in ('P','M'):
                    row['B'+e]=rel_error(a[f'DeltaW{e}_{s}'],common[f'DeltaW{e}_{s}'])
                    row['B'+e+'4']=rel_error(a[f'A_{e}_{s}']@a[f'Q_{e}_{s}'],common[f'A_{e}_{s}']@common[f'Q_{e}_{s}'])
                    row['DeltaK'+e]=rel_error(a[f'DeltaK{e}_{s}'],common[f'DeltaK{e}_{s}'])
                operator.append(row)
            for ei,e in enumerate(('Source','P','M')):
                ids=p['permutations'][rep][e][:n]
                descriptors.append(dict(replicate=rep,budget=n,environment=e,clean_power=float(np.mean(abs(clean[ei,ids])**2)),
                    pilot_covariance_trace_per_dimension=float(np.trace(st[f'{e}_rpp']).real/64),mean_norm=float(np.linalg.norm(np.r_[st[f'{e}_mean_p'],st[f'{e}_mean_d']]))))
    for ei,e in enumerate(('Source','P','M')):descriptors.append(dict(replicate='Common500',budget=500,environment=e,clean_power=float(np.mean(abs(clean[ei])**2)),
        pilot_covariance_trace_per_dimension=float(np.trace(commonstats[f'{e}_rpp']).real/64),mean_norm=float(np.linalg.norm(np.r_[commonstats[f'{e}_mean_p'],commonstats[f'{e}_mean_d']]))))
    frequency=[]
    for cond in p['config']['conditions']:
        for s in (10,20):
            for n in (50,100,250):
                for m in ('Source','LR4'):
                    selected=[r for r in tails if (r['condition'],r['snr_db'],r['budget'],r['method'])==(cond,s,n,m)]
                    count=collections.Counter(i for r in selected for i in json.loads(r['top5_positive_ids']))
                    frequency.append(dict(condition=cond,snr_db=s,budget=n,method=m,tail_occurrences_sorted=json.dumps(sorted(count.items(),key=lambda x:(-x[1],x[0]))),
                        unique_top5_ids=len(count),ids_in_all5=json.dumps(sorted(i for i,c in count.items() if c==5))))
    comparisons={f:dict(greater_than_Ws=sum(r[f]>r['Ws'] for r in operator),total=len(operator)) for f in ('BP','BM','BP4','BM4')}
    compg=[r for r in gains if r['condition']=='Composed' and r['budget']!=500]
    fact=dict(composed_all30_gain_medians_positive=all(r['median']>0 for r in compg),composed_minimum_individual_gain=min(r['minimum'] for r in compg),
        composed_minimum_positive_fraction=min(r['positive_fraction'] for r in compg),operator_sensitivity_counts=comparisons,
        composed_strict_ordering_count=sum(r['strict_50_100_250_500'] for r in orders if r['condition']=='Composed' and r['method']=='LR4'),
        composed_reverse_ordering_count=sum(r['strict_reverse'] for r in orders if r['condition']=='Composed' and r['method']=='LR4'))
    bv=[r for r in across if r['condition']=='Composed' and r['method']=='LR4' and r['budget']!=500]
    fact['composed_B_greater_than_V_groups']=[dict(snr_db=r['snr_db'],budget=r['budget']) for r in bv if abs(r['budget_displacement_B'])>r['subset_spread_IQR_V']]
    tail_order={}
    for cond in p['config']['conditions']:
        for s in (10,20):
            for metric in ('p95_penalty','maximum_positive_penalty'):
                count=0
                for rep in p['replicates']:
                    vals=[next(r[metric] for r in tails if (r['condition'],r['snr_db'],r['replicate'],r['budget'],r['method'])==(cond,s,rep,n,'LR4')) for n in (50,100,250)]
                    count+=vals[0]>vals[1]>vals[2]
                tail_order[f'{cond}_{s}_{metric}']=count
    fact['LR4_tail_strict_50_100_250_counts_of5']=tail_order
    idg=[r for r in gains if r['condition']=='ID' and r['budget']!=500]
    fact['ID_method_median_degradation_range']=[min(r['degradation_median'] for r in idg),max(r['degradation_median'] for r in idg)]
    fact['ID_maximum_method_degradation']=max(r['degradation_maximum'] for r in idg)
    fact['Composed_maximum_budget_penalty']=max(r['maximum_positive_penalty'] for r in tails if r['condition']=='Composed' and r['method']=='LR4')
    case='A' if fact['composed_all30_gain_medians_positive'] and fact['composed_minimum_positive_fraction']==1 and fact['composed_strict_ordering_count']<10 else 'mentor review: B/C distinction requires magnitude and tail assessment'
    audit=[dict(hypothesis=h,observation=o,interpretation=t,alternative_explanation=alt,evidence_strength='5 subset realizations on fixed finite corpus and reused development trajectories',verdict=v) for h,o,t,alt,v in [
        ('H1',str({k:fact[k] for k in ('composed_all30_gain_medians_positive','composed_minimum_individual_gain','composed_minimum_positive_fraction')}),'Gain direction checked in each preparation','fixed evaluation conditions; overlapping finite-corpus subsets','supported on tested development inputs' if fact['composed_all30_gain_medians_positive'] else 'not uniformly supported'),
        ('H2',json.dumps(dict(strict_orders=fact['composed_strict_ordering_count'],B_greater_than_V=fact['composed_B_greater_than_V_groups'])),'Central ordering inconsistent; low-budget signal also present','median ordering differs from paired-penalty median; finite-corpus overlap','partial support: inconsistent ordering, not universal subset-spread dominance'),
        ('H3',json.dumps(comparisons),'Relative denominator differences retained','compression and CV kernels co-vary; no causal performance attribution','supported only to observed comparison counts'),
        ('H4',json.dumps(tail_order),'Low-budget LR4 tail expansion in19/20 condition/SNR/replicate series; R1 Composed20 has N100>N50','channel/subset interaction; maximum depends on fixed100 trajectories','predominantly monotonic tails; universal monotonicity has one counterexample'),
        ('H5',json.dumps(dict(median_range=fact['ID_method_median_degradation_range'],maximum=fact['ID_maximum_method_degradation'])),'Central method degradation persists without low-N systematic amplification; individual tails remain','current pilot observations and kernel changes jointly affect estimates','central tendency supported; tail safeguard is limited')]]
    for name,rows in [('summary',summary),('table2_Composed',[r for r in summary if r['condition']=='Composed']),('table3_gains',gains),
        ('table4_ID',[r for r in summary if r['condition']=='ID']),('tails',tails),('tail_frequency',frequency),('paired_penalties',individual),
        ('across_replicates',across),('ordering',orders),('table5_operators',operator),('subset_descriptors',descriptors),('coordinates',coordinates),('table6_hypotheses',audit)]:csvsave(OUT/f'sp_u4b_{name}.csv',rows)
    save(OUT/'sp_u4b_analysis.json',dict(passed=True,facts=fact,case=case,subset_replicates=5,evaluation_trajectories_per_condition=100,
        no_500_independent_repetitions_claim=True,no_formal_inference=True,no_exclusions=True,branch_state='FREEZE pending mentor review; no further default development experiments',
        raw_metric_definition=p['central_metric'],positive_tail_definition=p['tail_definitions']))
    print(json.dumps(dict(case=case,facts=fact),indent=2),flush=True)


if __name__=='__main__':main()
