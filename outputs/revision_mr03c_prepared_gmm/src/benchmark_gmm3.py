"""Same-host prevalidated estimator-core timing, not a new performance dataset."""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time,json
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
ROOT=Path(r'D:\Users\yuhua\Desktop\Project');OUT=ROOT/'outputs/revision_mr03c_prepared_gmm';MR=ROOT/'outputs/revision_mr03_ppca_capacity_controls'
sys.path[:0]=[str(OUT/'src'),str(ROOT),str(MR/'src_or_scripts'),str(ROOT/'outputs/revision_mr05_compression_controls/src_or_scripts')]
from run_gmm3 import read,write,csvout,close,verify,models,sha,SNRS,CONDS
from run_mr05 import load_bank
from controls import prevalidated_deployment
from scripts import sp_u12b2a_common as common
from scripts.sp_u12b4a_adapter import rss,blas_threads
METHODS=('GMM_PREPARED_3','POOLED_NATURAL','PPCA_DIRECT_R32','LR4_TW1','LR4_TW5','DENSE_TW5','WHOLE_POOL_M2_R24')
def gmm_kernel(model):
    # Same posterior/map computation, excluding entropy/logging/validation extras.
    def kernel(p):
        scores=[];outputs=[]
        for q,c in enumerate(model.components):
            delta=p-c.mu_p;z=solve_triangular(c.chol_lower,delta,lower=True,check_finite=False)
            quad=np.sum(z.real**2+z.imag**2,dtype=np.float64)
            scores.append(model.log_priors[q]-c.logdet-quad);outputs.append(c.mu_d+delta@c.W.T)
        scores=np.array(scores);scores-=scores.max();w=np.exp(scores-logsumexp(scores));out=np.zeros(model.n_d,np.complex128)
        for weight,y in zip(w,outputs):out+=weight*y
        return out
    return kernel
def main():
    before=verify();cfg=read(OUT/'config.json')['timing'];g=models();op,bank=load_bank();cvs,_=common.family(op,SNRS)
    with np.load(MR/'data/conditional_operators.npz',allow_pickle=False) as z:ppca={k:z[k] for k in z.files if k.startswith('PPCA_DIRECT_R32_')}
    deploy={(s,m):prevalidated_deployment(m,op,bank[s],s) for s in SNRS for m in ('POOLED_NATURAL','CORR_ORIGIN_R4','DENSE_TW5','WHOLE_POOL_M2_R24')};gm={s:gmm_kernel(g[s]) for s in SNRS}
    windows={s:[] for s in SNRS}
    for c in CONDS:
        for i in range(20):
            p=MR/'data/D_eval'/f'{c}_{i:04d}.npz'
            with np.load(p,allow_pickle=False) as z:pilots=z['pilots']
            for si,s in enumerate(SNRS):
                win=pilots[si,:5].copy();close(gm[s](win[-1]),g[s].predict(win[-1]).estimate,'PURE_CORE_GMM_EQUIVALENCE')
                windows[s].append(win)
    def call(s,m,win):
        p=win[-1]
        if m=='GMM_PREPARED_3':return gm[s](p)
        if m=='PPCA_DIRECT_R32':return ppca[f'PPCA_DIRECT_R32_{s}_mean_d']+(p-ppca[f'PPCA_DIRECT_R32_{s}_mean_p'])@ppca[f'PPCA_DIRECT_R32_{s}_W'].T
        u=p-op['source_mean_p']
        if m=='POOLED_NATURAL':return deploy[s,m](u,None)
        coordinate,_=cvs[s].fit(win[-1:] if m=='LR4_TW1' else win,modules=('P','M'))
        name='CORR_ORIGIN_R4' if m in ('LR4_TW1','LR4_TW5') else m
        return deploy[s,name](u,coordinate)
    threads=blas_threads()
    if any(x['threads']!=1 for x in threads):raise ValueError('SINGLE_THREAD_REQUIRED')
    # Window preparation and original-fit check are outside measured region.
    for s in SNRS:
        if len(windows[s])!=100:raise ValueError('TIMING_WINDOWS')
        for win in windows[s]:
            for m in METHODS:
                if not np.isfinite(call(s,m,win)).all():raise ValueError('CORE_NONFINITE')
    records=[];rng=np.random.default_rng(cfg['order_seed']);start=time.monotonic()
    for s in SNRS:
        for m in METHODS:
            for k in range(cfg['warmup_calls']):call(s,m,windows[s][k%100])
        for block in range(cfg['blocks']):
            for pos,mi in enumerate(rng.permutation(len(METHODS))):
                m=METHODS[mi];t=time.perf_counter_ns()
                for win in windows[s]:y=call(s,m,win)
                ns=time.perf_counter_ns()-t
                if not np.isfinite(y).all():raise ValueError('TIMED_NONFINITE')
                records.append(dict(snr_db=s,block=block,order_position=pos,method=m,windows=100,elapsed_ns=ns,ns_per_call=ns/100))
        print(json.dumps(dict(stage='timing',snr_db=s,blocks=cfg['blocks'])),flush=True)
    summary=[];ratios=[]
    for s in SNRS:
        vals={m:np.array([r['ns_per_call'] for r in records if r['snr_db']==s and r['method']==m]) for m in METHODS}
        for m,v in vals.items():
            summary.append(dict(snr_db=s,method=m,median_us=float(np.median(v)/1000),q25_us=float(np.quantile(v,.25)/1000),q75_us=float(np.quantile(v,.75)/1000),blocks=10,windows_per_block=100,warmup=30))
            ratio=vals['GMM_PREPARED_3']/v;ratios.append(dict(snr_db=s,comparator=m,GMM_over_comparator_median=float(np.median(ratio)),q25=float(np.quantile(ratio,.25)),q75=float(np.quantile(ratio,.75)),scope='descriptive same-host paired block ratio; above1 means GMM slower'))
    csvout(OUT/'tables/timing_blocks.csv',records);csvout(OUT/'tables/timing_summary.csv',summary);csvout(OUT/'tables/timing_paired_ratios.csv',ratios)
    Q,np_,nd,M,F,no,nh,r=3,64,3968,2,4,48,16,4
    costs=[]
    for method in METHODS:
        if method=='GMM_PREPARED_3':maps=Q*nd*np_;fit=0;tri=Q*np_*(np_-1)//2;coeff=maps;means=Q*(nd+np_);chol=Q*np_**2;weighted=Q*nd;historical=0
        elif method in ('POOLED_NATURAL','PPCA_DIRECT_R32'):maps=nd*np_;fit=tri=chol=weighted=0;coeff=maps;means=nd+np_;historical=0
        else:
            historical=0 if method=='LR4_TW1' else 4;Tw=1+historical;fit=(M+1)*F*Tw*no*nh;tri=chol=weighted=0;means=nd+np_
            maps=(nd*np_+M*r*(nd+np_)) if method.startswith('LR4') else (3*nd*np_ if method=='DENSE_TW5' else 3*24*(nd+np_));coeff=maps
        costs.append(dict(method=method,history_frames=historical,conditional_deployment_complex_MAC=maps,cross_pilot_fit_complex_MAC=fit,posterior_triangular_complex_MAC=tri,total_dominant_complex_MAC=maps+fit+tri,mixture_complex_by_real_output_products=weighted,operator_complex_coefficients=coeff,mean_complex_coefficients=means,posterior_Cholesky_stored_complex_elements=chol,deployment_arrays_bytes=(coeff+means+chol)*16,proxy_excludes='real norms/divisions, exp/logsumexp, additions, Gram/small solve, covariance/statistics retained separately',PPCA_scope='actual cached dense conditional W; loading rank32 is not dense cache size' if method=='PPCA_DIRECT_R32' else ''))
    csvout(OUT/'tables/costs.csv',costs)
    stat_count=Q*(np_+nd+np_**2+nd*np_)
    write(OUT/'reports/timing.json',dict(PASS=True,config=cfg,methods=METHODS,threads=threads,measured_calls=len(records)*100,scored_dataset_predictions_not_recomputed=True,prior_and_model_tuned=False,resource=rss(),elapsed_seconds=time.monotonic()-start,prepared_clean_statistics_complex_elements=stat_count,prepared_clean_statistics_bytes=stat_count*16,GMM_noise_dependent_actual_instance_bytes_per_snr=g[10].setup_audit['components'][0]['persistent_array_bytes']*Q,numerical_core_vs_production_checks=300,same_endpoint='prevalidated mathematical full estimator core; GMM posterior/map/weighted mean included; entropy diagnostics and shape/finite guards excluded; original adaptive fit remains unchanged',pooled_and_PPCA_only_current_frame=True,whole24_and_LR4_Tw5_history_extra=True,preservation=verify()))
    print(json.dumps(dict(stage='timing',PASS=True,measured_calls=len(records)*100)),flush=True)
if __name__=='__main__':main()
