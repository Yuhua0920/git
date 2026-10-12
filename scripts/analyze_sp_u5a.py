"""Paired trajectory-first pilot-overhead analysis; no tuning or formal inference."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u5a_common import *
def main():
    prereg();firewall()
    if not readjson(OUT/'sp_u5a_evaluation_gate.json')['passed']:raise ValueError('scan gate')
    tr=csvread(OUT/'sp_u5a_trajectories.csv');wr=csvread(OUT/'sp_u5a_windows.csv');ix={(r['condition'],int(r['trajectory_id']),int(r['snr_db']),r['artifact'],r['method']):r for r in tr}
    nmse=[];gains=[];paired=[];individual=[];ber=[];stability=[];tails=[];reg=[];sensitivity=[]
    for c in cfg()['conditions']:
        for s in (10,20):
            for df in (8,16,32):
                vs={m:np.array([float(ix[c,i,s,str(df),m]['nmse_db']) for i in range(100)]) for m in ('Source','LR4')}
                gain=vs['Source']-vs['LR4'];degrad=-gain
                gains.append(dict(condition=c,snr_db=s,dF=df,**distribution(gain),positive_fraction=float(np.mean(gain>0)),
                    degradation_median=float(np.median(degrad)),degradation_p95=float(np.quantile(degrad,.95)),degradation_maximum=float(degrad.max())))
                for m in ('Source','LR4'):
                    rr=[ix[c,i,s,str(df),m] for i in range(100)];agg=sum(float(r['error_energy']) for r in rr)/sum(float(r['truth_energy']) for r in rr)
                    nmse.append(dict(condition=c,snr_db=s,dF=df,method=m,trajectories=100,aggregate_nmse_linear=agg,aggregate_nmse_db=float(10*np.log10(agg)),**distribution(vs[m])))
                    errors=sum(int(r['bit_errors']) for r in rr);bits=sum(int(r['bit_count']) for r in rr)
                    ber.append(dict(condition=c,snr_db=s,dF=df,method=m,BER=errors/bits,bit_errors=errors,decoded_bits=bits,
                        bits_per_scored_frame=geometry(df).counts['two_stream_data_symbols']*2,scored_frames=600,data_REs_per_frame=int(geometry(df).layout.data_mask[0].sum()),
                        **{'trajectory_BER_'+k:v for k,v in distribution([float(r['BER']) for r in rr]).items()}))
                    if df!=16:
                        ref=np.array([float(ix[c,i,s,'16',m]['nmse_db']) for i in range(100)])
                        diff=vs[m]-ref if df==32 else ref-vs[m];kind='reduction32_minus16' if df==32 else 'increase16_minus8'
                        paired.append(dict(condition=c,snr_db=s,contrast=kind,method=m,**distribution(diff),positive_fraction=float(np.mean(diff>0))))
                        individual.extend(dict(condition=c,snr_db=s,contrast=kind,method=m,trajectory_id=i,difference_db=float(d)) for i,d in enumerate(diff))
                        order=np.argsort(-diff,kind='stable')[:5]
                        tails.append(dict(condition=c,snr_db=s,contrast=kind,method=m,top5_ids=json.dumps(order.tolist()),top5_differences=json.dumps(diff[order].tolist()),
                            bottom5_ids=json.dumps(np.argsort(diff,kind='stable')[:5].tolist()),maximum=float(diff.max()),minimum=float(diff.min())))
                windows=[r for r in wr if (r['condition'],r['snr_db'],r['artifact'])==(c,str(s),str(df))]
                row=dict(condition=c,snr_db=s,dF=df,windows=len(windows),rank_failure_count=sum(int(r['gram_rank'])<2 for r in windows),fallback_count=sum(r['fallback']=='True' for r in windows))
                for field in ('condition_number','alpha','beta','coordinate_norm','normal_equation_residual'):
                    row.update({field+'_'+k:v for k,v in distribution([float(r[field]) for r in windows]).items()})
                for field in ('condition_number','coordinate_norm'):
                    vals=[np.mean([float(r[field]) for r in windows if int(r['trajectory_id'])==i]) for i in range(100)]
                    row.update({'trajectory_mean_'+field+'_'+k:v for k,v in distribution(vals).items()})
                stability.append(row)
            for m in ('Source','LR4'):
                delta=np.array([float(ix[c,i,s,'16',m]['nmse_db'])-float(ix[c,i,s,'Historical16',m]['nmse_db']) for i in range(100)])
                bitdif=[int(ix[c,i,s,'16',m]['bit_errors'])-int(ix[c,i,s,'Historical16',m]['bit_errors']) for i in range(100)]
                reg.append(dict(condition=c,snr_db=s,method=m,maximum_abs_trajectory_nmse_difference=float(max(abs(delta))),all_BER_error_counts_equal=all(v==0 for v in bitdif)))
            for contrast in ('reduction32_minus16','increase16_minus8'):
                src=np.array([r['difference_db'] for r in individual if (r['condition'],r['snr_db'],r['contrast'],r['method'])==(c,s,contrast,'Source')])
                lr=np.array([r['difference_db'] for r in individual if (r['condition'],r['snr_db'],r['contrast'],r['method'])==(c,s,contrast,'LR4')])
                sensitivity.append(dict(condition=c,snr_db=s,contrast=contrast,**distribution(lr-src),positive_fraction=float(np.mean(lr>src))))
    diminishing=[]
    for c in cfg()['conditions']:
        for s in (10,20):
            for m in ('Source','LR4'):
                values=[]
                for i in range(100):
                    v={df:float(ix[c,i,s,str(df),m]['nmse_db']) for df in (8,16,32)}
                    values.append((v[32]-v[16])-(v[16]-v[8]))
                diminishing.append(dict(condition=c,snr_db=s,method=m,contrast='reduction_penalty_minus_increase_gain',**distribution(values),positive_fraction=float(np.mean(np.array(values)>0))))
    csvsave(OUT/'sp_u5a_diminishing_diagnostic.csv',diminishing)
    for name,rows in [('nmse',nmse),('table2_ID',[r for r in nmse if r['condition']=='ID']),('table3_Composed',[r for r in nmse if r['condition']=='Composed']),
        ('table4_gains',gains),('table5_paired',paired),('paired_individual',individual),('table6_BER',ber),('stability',stability),('tails',tails),('full_dF16_regression',reg),('relative_sensitivity',sensitivity)]:csvsave(OUT/f'sp_u5a_{name}.csv',rows)
    facts=dict(low_overhead_composed_gain_all_positive=all(r['minimum']>0 for r in gains if r['condition']=='Composed' and r['dF']==32),
        low_overhead_composed_gain_minimum=min(r['minimum'] for r in gains if r['condition']=='Composed' and r['dF']==32),
        rank_failures=sum(r['rank_failure_count'] for r in stability),fallback_count=sum(r['fallback_count'] for r in stability),
        full_dF16_NMSE_max_abs=max(r['maximum_abs_trajectory_nmse_difference'] for r in reg),full_dF16_BER_counts_equal=all(r['all_BER_error_counts_equal'] for r in reg))
    # No post-hoc numeric cutoff for 'limited', 'meaningful', or 'significant'.
    hypotheses=[]
    definitions=[
        ('H1',[r for r in paired if r['method']=='Source' and r['contrast']=='reduction32_minus16'],
            'Lower overhead worsens Source central paired NMSE in all four groups; one Composed10 trajectory has negative penalty','supported in central tendency; not universal per trajectory'),
        ('H2',[r for r in gains if r['condition']=='Composed'],
            'All low-overhead Composed gains positive; magnitude shrinks compared with dF16','benefit survival supported; unchanged-benefit interpretation rejected'),
        ('H3',[r for r in sensitivity if r['contrast']=='reduction32_minus16'],
            'Composed LR4 minus Source reduction cost is positive on all100 trajectories at both SNRs; ID consistency weaker','Composed sensitivity limitation supported; causal coordinate explanation not identified'),
        ('H4',diminishing,'LR4 central paired comparison favors diminishing return; Source Composed20 is a counterexample','partial support only; no universal saturation or fitted curve'),
        ('H5',[r for r in gains if r['condition']=='ID'],
            'ID low-overhead central degradation and P95/max grow; paired method-penalty increase also retained','practical ID risk rises descriptively; no formal significance claim')]
    for h,relevant,interpretation,verdict in definitions:
        hypotheses.append(dict(hypothesis=h,evidence_json=json.dumps(relevant),interpretation=interpretation,
            alternative='data targets, pilot locations, fold counts, Source/correction spaces co-vary with geometry',evidence_strength='100 new paired development trajectories per condition; one shared offline corpus',verdict=verdict))
    csvsave(OUT/'sp_u5a_table7_hypotheses.csv',hypotheses)
    save(OUT/'sp_u5a_analysis.json',dict(passed=True,facts=facts,trajectories_per_condition=100,densities=3,
        formal_inference=False,no_performance_threshold_selected=True,no_exclusions=True,stop=True))
    print(json.dumps(facts,indent=2))
if __name__=='__main__':main()
