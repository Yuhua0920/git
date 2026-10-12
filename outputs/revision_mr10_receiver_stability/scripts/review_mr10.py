"""Independent recount and read-only physical checks of saved diagnostic quantities."""
import sys,json,csv,zipfile
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from run_mr10 import OUT,ROOT,R1,MR2,COND,METHODS,PREF,auth,read,write,table,csvrows,need,qstats
def main():
    count=np.zeros((2,7),np.int64);power=[];noise=[];errs=[];truths=[];maxgain=0.
    for i in range(100):
        p=OUT/f'data/receiver/{i:04d}.npz';meta=read(p.with_suffix('.json'));auth(p,meta['raw_sha256'])
        with zipfile.ZipFile(p) as z:need(z.testzip() is None,'RECEIVER_CRC')
        with np.load(p,allow_pickle=False) as z:
            need(z['bit_error_mask'].shape==(2,7,6,992,4),'BIT_MASK_SHAPE')
            count+=np.count_nonzero(z['bit_error_mask'],axis=(2,3,4))
            need(z['reference_bits'].shape==(6,992,4),'REFERENCE_BITS_SHAPE')
            for name in ('singular_values','condition_number','zf_noise_gain','zf_effective_esn0_db','conditional_zf_bit_ber'):
                need(np.isfinite(z[name]).all(),'NONFINITE_DIAGNOSTIC:'+name)
            need(np.all(z['singular_values'][...,1]>0),'SINGULAR_RE')
            need(np.allclose(z['zf_noise_gain'].sum(axis=-1),np.sum(1/z['singular_values']**2,axis=-1),rtol=2e-11,atol=1e-12),'INVERSE_GAIN_SVD_IDENTITY')
            snr=np.array([meta['actual_receive_snr_db'][si] for si in range(2)])
            realized=np.array([s+10*np.log10(meta['target_receive_power_realized']/PREF) for s in (10,20)])
            for si,s in enumerate((10,20)):
                power.append({'trajectory_id':i,'snr_db':s,'P_rx_expected':meta['target_receive_power_expected'],
                  'P_rx_realized':meta['target_receive_power_realized'],'gamma_actual_expected_db':snr[si],
                  'gamma_actual_realized_db':realized[si],'post_ZF_below_0dB_fraction':float(np.mean(z['zf_effective_esn0_db'][si]<0)),
                  'H_condition_above_100_fraction':float(np.mean(z['condition_number']>100))})
        with np.load(R1/f'evaluation/raw_{i:04d}.npz',allow_pickle=False) as z:
            need(abs(meta['target_receive_power_realized']-z['power'][0]/z['power'][2])<1e-12*meta['target_receive_power_realized'],'HISTORICAL_REALIZED_POWER')
            for si,s in enumerate((10,20)):
                for name,mi in (('LS',3),('LR4',4)):
                    need(meta['bit_errors'][si][mi]==int(z['bit_errors'][si,('LS','Source','Hard','Convex','Dense','LR4','Matched').index(name)]),'HISTORICAL_TRAJECTORY_COUNT')
            errs.append(z['error']);truths.append(z['truth'])
        noise.append([meta['noise_real_second_moment'],meta['noise_imag_second_moment'],meta['noise_complex_second_moment'],meta['noise_re_im_cross_moment']])
    for r in csvrows(OUT/'tables/detector_BER.csv'):
        si=(10,20).index(int(r['snr_db']));mi=METHODS.index(r['method']);need(int(r['bit_errors'])==count[si,mi],'TABLE_RAW_RECOUNT')
        need(float(r['BER'])==count[si,mi]/2380800,'BER_RECOUNT')
    table(OUT/'tables/receive_power_actual_vs_reference.csv',power)
    energy=np.array(errs);den=np.array(truths);quality=[]
    for si,s in enumerate((10,20)):
        for name,mi in (('LS',0),('LR4',5)):
            nmse=float(energy[:,si,mi].sum()/den[:,si].sum())
            mm=(5 if name=='LS' else 6);ber=count[si,mm]/2380800
            pernmse=energy[:,si,mi]/den[:,si]
            berper=np.array([read(OUT/f'data/receiver/{i:04d}.json')['bit_errors'][si][mm]/23808 for i in range(100)])
            quality.append({'snr_db':s,'channel_estimate':name,'data_NMSE_linear':nmse,'data_NMSE_db':10*np.log10(nmse),
                'same_policy_MMSE_BER':ber,'trajectory_NMSE_BER_Spearman':float(spearmanr(pernmse,berper).statistic),
                'role':'existing channel error energies, fixed-100 diagnostic only'})
    table(OUT/'tables/estimated_channel_quality_vs_MMSE_BER.csv',quality)
    with np.load(OUT/'data/coordinate_windows.npz',allow_pickle=False) as z:
        S=z['S'];b=z['b'];coef=z['coordinates'];need(S.shape==(5,100,3,6,2,2),'GRAM_DIMENSIONS')
        evals=np.linalg.eigvalsh(S);need(np.all(evals[...,0]>0),'GRAM_POSITIVITY')
        # Explicit two-by-two inverse identity verifies saved coordinates without refitting pilots.
        det=S[...,0,0]*S[...,1,1]-S[...,0,1]*S[...,1,0]
        xc=np.stack(((S[...,1,1]*b[...,0]-S[...,0,1]*b[...,1])/det,
                     (S[...,0,0]*b[...,1]-S[...,1,0]*b[...,0])/det),axis=-1)
        maxdiff=float(np.max(abs(xc-coef)))
        need(np.allclose(xc,coef,atol=1e-13,rtol=1e-13),'EXPLICIT_NORMAL_EQUATION_IDENTITY')
        tail=[]
        for ci,co in enumerate(COND):
            for si,s in enumerate((10,20,30)):
                cc=coef[ci,:,si];e=z['LR4_error'][ci,:,si]/z['truth'][ci,:,si]
                tail.append({'condition':co,'snr_db':s,'windows':600,'alpha_min':float(cc[...,0].min()),
                   'alpha_max':float(cc[...,0].max()),'beta_min':float(cc[...,1].min()),'beta_max':float(cc[...,1].max()),
                   'max_coordinate_norm':float(np.linalg.norm(cc,axis=-1).max()),
                   'abs_any_coordinate_above_1_windows':int(np.sum(np.any(abs(cc)>1,axis=-1))),
                   'abs_any_coordinate_above_2_windows':int(np.sum(np.any(abs(cc)>2,axis=-1))),
                   'max_frame_data_NMSE':float(e.max()),'role':'thresholds 1/2 are descriptive coordinate scales, not acceptance criteria'})
        table(OUT/'tables/coordinate_tail_ledger.csv',tail)
    pow_summary=[]
    for s in (10,20):
        rr=[r for r in power if r['snr_db']==s]
        pp=np.array([r['P_rx_realized'] for r in rr]);expected=np.array([r['P_rx_expected'] for r in rr])
        pow_summary.append({'snr_db':s,'mean_P_rx_realized':float(pp.mean()),'mean_P_rx_expected':float(expected.mean()),
          'P_source_reference':PREF,'pooled_realized_SNR_db':float(s+10*np.log10(pp.mean()/PREF)),
          'median_trajectory_realized_SNR_db':float(np.median([r['gamma_actual_realized_db'] for r in rr])),
          'realized_SNR_q05':float(np.quantile([r['gamma_actual_realized_db'] for r in rr],.05)),
          'realized_SNR_q95':float(np.quantile([r['gamma_actual_realized_db'] for r in rr],.95)),
          'post_ZF_below_0dB_fraction':float(np.mean([r['post_ZF_below_0dB_fraction'] for r in rr])),
          'H_condition_above_100_fraction':float(np.mean([r['H_condition_above_100_fraction'] for r in rr]))})
    table(OUT/'tables/power_summary.csv',pow_summary)
    write(OUT/'reports/independent_numeric_review.json',{'PASS':True,'receiver_blocks_CRC_checked':100,'raw_bit_count_matches_all_tables':True,
       'historical_replayed_trajectory_counts_exact':True,'realized_receive_power_matches_saved_history':True,
       'ZF_inverse_noise_gain_matches_SVD_identity':True,'coordinate_explicit_2x2_solution_max_abs_difference':maxdiff,
       'mean_unit_noise_real_second_moment':float(np.mean(noise,axis=0)[0]),
       'mean_unit_noise_imag_second_moment':float(np.mean(noise,axis=0)[1]),
       'mean_unit_noise_complex_second_moment':float(np.mean(noise,axis=0)[2]),
       'mean_unit_noise_real_imag_cross_moment':float(np.mean(noise,axis=0)[3]),
       'new_generator_calls':0,'coordinate_refits':0,'new_bootstrap_draws':0,'scientific_scope':'post-hoc receiver diagnostic and saved-window numerical review'})
    print('INDEPENDENT NUMERIC REVIEW PASS',flush=True)
if __name__=='__main__':main()
