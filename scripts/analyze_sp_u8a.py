"""Descriptive summaries only: no bootstrap, p-values, new confidence or primary test."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import sp_u8a_common as c
import numpy as np
from scipy.stats import rankdata

def distribution(v):
    x=np.asarray(v,float)
    return dict(P05=float(np.quantile(x,.05)),median=float(np.median(x)),P95=float(np.quantile(x,.95)),
        min=float(np.min(x)),max=float(np.max(x)),mean=float(np.mean(x)))

def main():
    c.install_guard()
    assert c.read(c.OUT/'replay_gate.json')['PASS'] and c.read(c.OUT/'conditioning_summary.json')['PASS']
    rows=c.csv_read(c.OUT/'trajectories.csv');channel=c.csv_read(c.OUT/'channel_trajectories.csv')
    channel={int(r['trajectory_id']):r for r in channel};bank={}
    for r in rows:bank[int(r['snr_db']),r['method'],int(r['trajectory_id'])]=r
    summaries=[];differences=[];correlations=[];paired_rows=[]
    for s in c.SNRS:
        for m in c.METHODS:
            values=[float(bank[s,m,i]['BER']) for i in range(1000)]
            errors=sum(int(bank[s,m,i]['bit_errors']) for i in range(1000))
            summaries.append(dict(snr_db=s,method=m,aggregate_BER=errors/23808000,bit_errors=errors,total_bits=23808000,
                trajectory_distribution=distribution(values),worst_ID=int(np.argmax(values)),
                label='POST-HOC DIAGNOSTIC ONLY' if m=='Perfect' else 'Frozen comparator replay'))
        perfect=np.array([float(bank[s,'Perfect',i]['BER']) for i in range(1000)])
        for m in ('LS','Source','LR4','Matched'):
            delta=np.array([float(bank[s,m,i]['BER']) for i in range(1000)])-perfect
            differences.append(dict(snr_db=s,comparison=m+' minus Perfect',distribution=distribution(delta),
                lower_BER_count=int(np.count_nonzero(delta<0)),equal_BER_count=int(np.count_nonzero(delta==0)),
                higher_BER_count=int(np.count_nonzero(delta>0))))
            for i in range(1000):paired_rows.append(dict(trajectory_id=i,snr_db=s,method=m,BER_minus_Perfect=float(delta[i])))
        for metric in ('sigma_min','log10_condition','noise_enhancement'):
            for summary in ('median','P95'):
                key=metric+'_'+summary
                values=np.array([float(channel[i][key]) for i in range(1000)])
                assert np.isfinite(values).all() and np.isfinite(perfect).all()
                rho=float(np.corrcoef(rankdata(values),rankdata(perfect))[0,1])
                correlations.append(dict(snr_db=s,channel_statistic=key,Spearman=rho,trajectories=1000,
                    interpretation='descriptive association only; no causal or confidence claim'))
    fixed=c.csv_read(c.OUT/'fixed_ID_diagnostics.csv')
    fixed_numeric=[{k:(v if k=='method' else int(v) if k in ('trajectory_id','snr_db') else float(v)) for k,v in r.items()} for r in fixed]
    c.csv_write(c.OUT/'paired_BER_differences.csv',paired_rows)
    c.csv_write(c.OUT/'correlations.csv',correlations)
    c.write(c.OUT/'analysis.json',dict(task='SP-U8A',primary_case='A_STRONG_TRANSFER FROZEN',primary_impact='NONE',
        BER_summary=summaries,paired_differences=differences,conditioning=c.read(c.OUT/'conditioning_summary.json'),
        correlations=correlations,fixed_ID_diagnostics=fixed_numeric,
        inference_scope='post-hoc descriptive audit of USED R1 validation; no p-values, bootstrap or case retest',
        optional_identity='xhat-x=(solve(Hhat,H)-I)x+solve(Hhat,I)n; error energy=mismatch+noise+2Re<mismatch,noise>',
        assumptions=['square nonsingular per-RE channel; frozen solve rejects singular matrices',
            'unit complex noise scaled by frozen per-RX variance; saved atoms reused',
            'data transmitted QPSK/sqrt(2); detector output actual TX scale',
            'mismatch and transformed noise need not be orthogonal; estimated H depends on current pilot noise',
            'conditioning association is descriptive; condition number alone omits absolute channel scale']))
    print(c.NOTICE,flush=True)
    print('descriptive analysis complete; no primary NMSE computation',flush=True)

if __name__=='__main__':main()
