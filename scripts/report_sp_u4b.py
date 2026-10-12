"""Read-only, copyable compact numerical report; values stay in generated CSVs."""
import csv,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'outputs'
def rows(n):return list(csv.DictReader((P/('sp_u4b_'+n+'.csv')).open(encoding='utf8')))
def f(x):return f'{float(x):.5f}'
def get(rr,**kw):return next(r for r in rr if all(r[k]==str(v) for k,v in kw.items()))
REPS=['R0','R1','R2','R3','R4'];NS=[50,100,250]
GROUPS=[('ID',10),('ID',20),('Composed',10),('Composed',20)]
def main(section):
    if section=='C':
        for rep in REPS[1:]:print(rep,[(n,round(json.loads((P/f'sp_u4b_construction_{rep}_{n}.json').read_text())['wall_seconds'],5)) for n in NS])
    if section=='D':
        rr=rows('summary');print('r N | ID10 S/L | ID20 S/L | C10 S/L | C20 S/L ; median(mean)')
        for rep in REPS:
            for n in NS+[500]:
                v=[]
                for c,s in GROUPS:
                    v.append('/'.join(f(q['nmse_median'])+'('+f(q['nmse_mean'])+')' for m in ['Source','LR4'] for q in [get(rr,replicate=rep,budget=n,condition=c,snr_db=s,method=m)]))
                print(rep,n,' | '.join(v))
    if section=='E':
        rr=rows('summary');print('r N | ID10 | ID20 | C10 | C20 ; median/IQR/P95/max')
        for m in ['Source','LR4']:
            print(m)
            for rep in REPS:
                for n in NS:
                    v=[]
                    for c,s in GROUPS:
                        q=get(rr,replicate=rep,budget=n,condition=c,snr_db=s,method=m)
                        v.append('/'.join(f(q[k]) for k in ['penalty_median','penalty_iqr','penalty_p95','maximum_positive_penalty']))
                    print(rep,n,' | '.join(v))
    if section=='F':
        rr=rows('table3_gains');print('r N | ID10 | ID20 | C10 | C20 ; median/P05/P95/positivefraction')
        for rep in REPS:
            for n in NS+[500]:
                v=[]
                for c,s in GROUPS:
                    q=get(rr,replicate=rep,budget=n,condition=c,snr_db=s)
                    v.append('/'.join(f(q[k]) for k in ['median','p05','p95','positive_fraction']))
                print(rep,n,' | '.join(v))
        print('Extra: r N | C10/C20 minimum | ID10 median/P95/max | ID20 median/P95/max')
        for rep in REPS:
            for n in NS+[500]:
                mins=[get(rr,replicate=rep,budget=n,condition='Composed',snr_db=s)['minimum'] for s in [10,20]]
                vals=[]
                for s in [10,20]:
                    q=get(rr,replicate=rep,budget=n,condition='ID',snr_db=s);vals.append('/'.join(f(q[k]) for k in ['degradation_median','degradation_p95','degradation_maximum']))
                print(rep,n,'/'.join(map(f,mins)),' | ',' | '.join(vals))
    if section=='G':
        rr=rows('across_replicates');print('c s m N | raw5 central | B V rangeWidth | central+/paired+ | paired repmedian/IQR/min/max')
        for q in rr:
            if q['budget']=='500':continue
            print(q['condition'],q['snr_db'],q['method'],q['budget'],','.join(map(f,json.loads(q['raw5_central_medians']))),' | ',
                '/'.join(f(q[k]) for k in ['budget_displacement_B','subset_spread_IQR_V','central_range_width']),q['central_displacement_positive_count']+'/'+q['paired_penalty_positive_count'],' | ',
                '/'.join(f(q[k]) for k in ['paired_penalty_replicate_median','paired_penalty_replicate_iqr','paired_penalty_range_min','paired_penalty_range_max']))
        print('Across gain: c s N | median/IQR/min/max/positivecount')
        for q in rr:
            if q['method']=='LR4' and q['budget']!='500':print(q['condition'],q['snr_db'],q['budget'],'/'.join(f(q[k]) for k in ['gain_replicate_median','gain_replicate_iqr','gain_range_min','gain_range_max'])+'/'+q['gain_positive_sign_count'])
    if section=='H':
        rr=rows('tails');print('r N | ID10 S,L | ID20 S,L | C10 S,L | C20 S,L ; top5 positive IDs')
        for rep in REPS:
            for n in NS:
                vals=[]
                for c,s in GROUPS:
                    vals.append(';'.join(','.join(map(str,json.loads(get(rr,replicate=rep,budget=n,condition=c,snr_db=s,method=m)['top5_positive_ids']))) for m in ['Source','LR4']))
                print(rep,n,' | '.join(vals))
        print('Conditional positive P95: r N | ID10 S/L | ID20 S/L | C10 S/L | C20 S/L')
        for rep in REPS:
            for n in NS:
                print(rep,n,' | '.join('/'.join(f(get(rr,replicate=rep,budget=n,condition=c,snr_db=s,method=m)['p95_positive_penalty']) for m in ['Source','LR4']) for c,s in GROUPS))
        print('Recurrence: c s N m | unique IDs | each ID:count')
        for q in rows('tail_frequency'):print(q['condition'],q['snr_db'],q['budget'],q['method'],q['unique_top5_ids'],','.join(f'{i}:{count}' for i,count in json.loads(q['tail_occurrences_sorted'])))
    if section=='I':
        for q in rows('table5_operators'):print(q['replicate'],q['budget'],q['snr_db'],'/'.join(f(q[k]) for k in ['Ws','BP','BM','BP4','BM4','DeltaKP','DeltaKM']))
    if section=='J':
        rr=rows('subset_descriptors');print('r N | Source | B30 | C120 ; power/trace64/meannorm')
        for rep in REPS+['Common500']:
            for n in ([500] if rep=='Common500' else NS):
                print(rep,n,' | '.join('/'.join(f(get(rr,replicate=rep,budget=n,environment=e)[k]) for k in ['clean_power','pilot_covariance_trace_per_dimension','mean_norm']) for e in ['Source','P','M']))
    if section=='K':
        rr=rows('coordinates');print('r N | ID10 | ID20 | C10 | C20 ; mean/median/P95')
        for rep in REPS:
            for n in NS:print(rep,n,' | '.join('/'.join(f(get(rr,replicate=rep,budget=n,condition=c,snr_db=s)[k]) for k in ['mean','median','p95']) for c,s in GROUPS))
if __name__=='__main__':main(sys.argv[1])
