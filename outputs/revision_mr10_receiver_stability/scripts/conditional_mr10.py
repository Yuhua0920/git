"""Conditional BER calculation on the same fixed saved 100 IDs. No noise draws."""
import numpy as np
from run_mr10 import OUT,c,METHODS,PREF,npz,read,write,table,need
from mr10_detectors import conditional_mismatched_zf
from src.lowrank_modules import factored_affine_predict
def main():
    g=c.observation_geometry();mask=g.layout.data_mask[0];a,f,_=c.frozen.load_deployables()
    output=OUT/'data/conditional_ZF';output.mkdir(exist_ok=False)
    for i in range(100):
        hf,comm,raw,proof=c.ingredients(i);po={(v['snr_db'],v['frame_id'],v['method']):v for v in proof['outputs']}
        expected=np.empty((2,3,6,992));gain=np.empty((2,3,6,992,2));mismatch=np.empty((2,3,6,992))
        for ji,j in enumerate(range(4,10)):
            h=hf[j];bits,symbols,z=comm[j];x=c.assemble_pilot_data_frame(symbols,g.layout);clean=c.apply_frequency_mimo_channel(h,x);hh=h[:,:,mask].transpose(2,0,1)
            for si,s in enumerate((10,20)):
                y=clean+np.sqrt(c.VARIANCES[s])*z;sparse,ls=c.estimate_ls_linear_channel(y,g.layout);pilot=sparse.ravel()[g.selected]
                pred=factored_affine_predict(pilot,a['source_mean_p'],a['source_mean_d'],a[f'Wsource_{s}'],raw['coordinates'][si*6+ji,2:4],f['P',s,4],f['M',s,4])
                for method,v in (('LS',ls.ravel()[g.data]),('LR4',pred)):need(c.array_hash(v)==po[s,j,method]['prediction_sha256'],'CONDITIONAL_PREDICTION_DRIFT')
                lr=sparse.copy().ravel();lr[g.data]=pred;lr=lr.reshape(2,2,64,16)
                for mi,est in enumerate((h,ls,lr)):
                    expected[si,mi,ji],gain[si,mi,ji],mismatch[si,mi,ji]=conditional_mismatched_zf(hh,est[:,:,mask].transpose(2,0,1),c.VARIANCES[s])
        npz(output/f'{i:04d}.npz',expected_bit_BER=expected,noise_gain=gain,mismatch_symbol_MSE=mismatch,trajectory_id=i,
            methods=np.array(['Perfect','LS','LR4']),snrs=np.array([10,20]),frames=np.arange(4,10))
    # Raw conditional risks saved for every trajectory before the following summaries.
    rows=[];checks=[]
    from scipy.stats import spearmanr
    for si,s in enumerate((10,20)):
        for mi,method,oi in ((0,'Perfect',0),(1,'LS',3),(2,'LR4',4)):
            vals=[];obs=[];gg=[];mm=[]
            for i in range(100):
                with np.load(output/f'{i:04d}.npz',allow_pickle=False) as z:
                    vv=float(z['expected_bit_BER'][si,mi].mean());ng=float(z['noise_gain'][si,mi].sum(axis=-1).mean());mis=float(z['mismatch_symbol_MSE'][si,mi].mean())
                oo=read(OUT/f'data/receiver/{i:04d}.json')['bit_errors'][si][oi]/23808
                rows.append({'trajectory_id':i,'snr_db':s,'method':method,'conditional_expected_BER':vv,'observed_BER':oo,
                     'mean_inverse_noise_gain_trace':ng,'mean_mismatch_symbol_MSE_per_stream':mis})
                vals.append(vv);obs.append(oo);gg.append(ng);mm.append(mis)
            checks.append({'snr_db':s,'method':method,'mean_conditional_expected_BER':float(np.mean(vals)),
              'observed_BER':float(np.mean(obs)),'observed_minus_expected':float(np.mean(np.array(obs)-vals)),
              'residual_trajectory_SD':float(np.std(np.array(obs)-vals,ddof=1)),'trajectory_Spearman':float(spearmanr(vals,obs).statistic),
              'mean_inverse_noise_gain_trace':float(np.mean(gg)),'mean_mismatch_symbol_MSE_per_stream':float(np.mean(mm))})
    table(OUT/'tables/estimated_channel_conditional_ZF_trajectory.csv',rows)
    table(OUT/'tables/estimated_channel_conditional_ZF_check.csv',checks)
    write(OUT/'reports/conditional_ZF_integrity.json',{'PASS':True,'raw_trajectories':100,'scored_frames':1200,
       'role':'exact conditional Gaussian marginal-bit risk, uniform QPSK averaged over 16 data vectors; not a new receiver run',
       'prediction_hash_checks':2400,'data_noise_regenerated':False,'coordinate_refitting':False,
       'scope':'joint signal-mismatch and transformed-noise effect; noise gain alone cannot identify an LS regularization mechanism'})
    print('CONDITIONAL ZF RISK PASS: all methods, same 100 frozen inputs, raw before summaries')
if __name__=='__main__':main()
