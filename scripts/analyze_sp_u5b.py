"""Paired common scoring controls, without inference or mechanism attribution."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u5b_common import *

def main():
    firewall();p=prereg()
    if not readjson(OUT/'sp_u5b_replay_gate.json')['passed']:raise ValueError('native replay gate')
    tr=csvread(OUT/'sp_u5b_trajectories.csv');ix={(r['condition'],int(r['trajectory_id']),int(r['snr_db']),int(r['dF']),r['method'],r['domain']):r for r in tr}
    nm=[];ga=[];ber=[];paired=[];control=[];extra=[];atten=[];individual=[];tails=[];idtrade=[]
    def vals(c,s,df,m,domain):return np.array([float(ix[c,i,s,df,m,domain]['nmse_db']) for i in range(100)])
    def tail(c,s,kind,m,v):
        order=np.argsort(-v,kind='stable')[:5];low=np.argsort(v,kind='stable')[:5]
        tails.append(dict(condition=c,snr_db=s,contrast=kind,method=m,top5_ids=json.dumps(order.tolist()),top5_values=json.dumps(v[order].tolist()),
            bottom5_ids=json.dumps(low.tolist()),bottom5_values=json.dumps(v[low].tolist()),p95=float(np.quantile(v,.95)),minimum=float(v.min()),maximum=float(v.max())))
    for c in ('ID','Composed'):
        for s in (10,20):
            for df in (8,16,32):
                gain=vals(c,s,df,'Source','common')-vals(c,s,df,'LR4','common')
                ga.append(dict(condition=c,snr_db=s,dF=df,**distribution(gain),positive_fraction=float(np.mean(gain>0))))
                if c=='ID':
                    v=-gain;idtrade.append(dict(condition=c,snr_db=s,dF=df,**distribution(v),positive_fraction=float(np.mean(v>0))))
                    tail(c,s,f'ID_degradation_dF{df}','LR4_minus_Source',v)
                for m in ('Source','LR4'):
                    rr=[ix[c,i,s,df,m,'common'] for i in range(100)];v=vals(c,s,df,m,'common');agg=sum(float(r['error_energy']) for r in rr)/sum(float(r['truth_energy']) for r in rr)
                    nm.append(dict(condition=c,snr_db=s,dF=df,method=m,aggregate_nmse_linear=agg,aggregate_nmse_db=float(10*np.log10(agg)),**distribution(v)))
                    be=sum(int(r['bit_errors']) for r in rr);bc=sum(int(r['bit_count']) for r in rr)
                    ber.append(dict(condition=c,snr_db=s,dF=df,method=m,bit_errors=be,decoded_bits=bc,BER=be/bc,
                        **{'trajectory_BER_'+k:v for k,v in distribution([float(r['BER']) for r in rr]).items()}))
            penalties={}
            for contrast,hi,lo in (('reduction32_minus16',32,16),('increase16_minus8',16,8)):
                for m in ('Source','LR4'):
                    common=vals(c,s,hi,m,'common')-vals(c,s,lo,m,'common');native=vals(c,s,hi,m,'native')-vals(c,s,lo,m,'native')
                    penalties[contrast,m]=common
                    paired.append(dict(condition=c,snr_db=s,contrast=contrast,method=m,**distribution(common),positive_fraction=float(np.mean(common>0))))
                    individual.extend(dict(condition=c,snr_db=s,contrast=contrast,method=m,trajectory_id=i,native_db=float(n),common_db=float(v),native_minus_common_db=float(n-v)) for i,(n,v) in enumerate(zip(native,common)))
                    difference=native-common;safe=abs(native)>p['ratio_guard'];ratio=common[safe]/native[safe]
                    atten.append(dict(condition=c,snr_db=s,contrast=contrast,method=m,**{'difference_'+k:v for k,v in distribution(difference).items()},
                        ratio_count=int(safe.sum()),ratio_unsafe_count=int((~safe).sum()),ratio_guard=p['ratio_guard'],**{'ratio_'+k:v for k,v in distribution(ratio).items()}))
                    tail(c,s,contrast,m,common)
                    if contrast=='reduction32_minus16':control.append(dict(condition=c,snr_db=s,method=m,native_median_penalty=float(np.median(native)),common_median_penalty=float(np.median(common)),
                        common_positive_fraction=float(np.mean(common>0)),median_of_paired_native_minus_common=float(np.median(difference)),difference_of_medians=float(np.median(native)-np.median(common))))
            e=penalties['reduction32_minus16','LR4']-penalties['reduction32_minus16','Source']
            en=(vals(c,s,32,'LR4','native')-vals(c,s,16,'LR4','native'))-(vals(c,s,32,'Source','native')-vals(c,s,16,'Source','native'))
            extra.append(dict(condition=c,snr_db=s,**{'common_'+k:v for k,v in distribution(e).items()},common_positive_fraction=float(np.mean(e>0)),
                **{'native_'+k:v for k,v in distribution(en).items()},native_positive_fraction=float(np.mean(en>0)),paired_native_minus_common_median=float(np.median(en-e))))
            tail(c,s,'extra_LR4_sensitivity','LR4_minus_Source',e)
    for name,r in (('nmse',nm),('gains',ga),('paired',paired),('control',control),('extra_sensitivity',extra),('attenuation',atten),('paired_individual',individual),('BER',ber),('tails',tails),('ID_tradeoff',idtrade)):csvsave(OUT/f'sp_u5b_{name}.csv',r)
    cp=[r for r in paired if r['condition']=='Composed' and r['contrast']=='reduction32_minus16'];ex=[r for r in extra if r['condition']=='Composed'];cg=[r for r in ga if r['condition']=='Composed' and r['dF']==32]
    facts=dict(common_composed_reduction_medians_positive=all(r['median']>0 for r in cp),common_composed_extra_positive_all=all(r['common_minimum']>0 for r in ex),
        common_composed_dF32_gains_positive_all=all(r['minimum']>0 for r in cg),score_mask_control_only=True,formal=False,mechanism_identified=False,no_exclusions=True)
    save(OUT/'sp_u5b_analysis.json',dict(passed=True,facts=facts,threshold_selected=False,decision_pending_qualitative_review=True,pilot_branch='FROZEN',no_U5C=True))
    print(json.dumps(dict(facts=facts,control=control,extra=extra),indent=2))

if __name__=='__main__':main()
