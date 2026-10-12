"""MR05 timing endpoints; no statistical selection or target generation."""
from __future__ import annotations
import time,csv,json,hashlib
from pathlib import Path
import numpy as np
from run_mr05 import OUT,ROOT,SNRS,CONDITIONS,METHODS,CONTROL,load_bank,source_input,read,write,csvout,archive,verify_sources,sha,close,common,rss,blas_threads
from controls import prevalidated_deployment

def fit_both(cv,window):
    return cv.fit(window,modules=('P','M'))

def costs():
    rows=[];M,F,Tw,np_,nd,no,nh=2,4,5,64,3968,48,16
    fit=(M+1)*F*Tw*no*nh
    for method in METHODS:
        background=method in METHODS[:2]
        if background:deploy=nd*np_;coeff=deploy;kind='one full operator';rank='full';factor_shape='3968x64'
        elif method=='DENSE_TW5':deploy=3*nd*np_;coeff=deploy;kind='full Source + two dense differences';rank='full';factor_shape='3 x 3968x64'
        elif method.startswith('CORR'):
            deploy=nd*np_+2*4*(nd+np_);coeff=deploy;kind='full Source + two rank4 factors';rank=4;factor_shape='3968x64 + 2 x (3968x4,4x64)'
        else:
            rank=int(method.rsplit('R',1)[1]);deploy=3*rank*(nd+np_);coeff=deploy;kind='three whole-operator factors';factor_shape=f'3 x (3968x{rank},{rank}x64)'
        # Source-centered API accepts u; end-to-end retained means included for deployment.
        # Natural Pooled needs only its own p/d means in a stand-alone implementation.
        means=nd+np_;kernels=0 if background else (M+1)*F*no*nh
        buffer_=np_ if background else Tw*np_
        rows.append(dict(method=method,rank=rank,assets=kind,factor_shapes=factor_shape,
            fit_complex_MAC=0 if background else fit,deployment_complex_MAC=deploy,
            total_dominant_complex_MAC=deploy+(0 if background else fit),
            operator_factor_complex_elements=coeff,operator_factor_bytes=coeff*16,
            mean_complex_elements=means,mean_bytes=means*16,
            minimal_deployment_complex_elements=coeff+means,minimal_deployment_bytes=(coeff+means)*16,
            source_centered_input_conversion_extra_bytes=1024 if method=='POOLED_NATURAL' else 0,
            common_kernel_complex_elements=kernels,common_kernel_bytes=kernels*16,
            causal_buffer_complex_elements=buffer_,causal_buffer_bytes=buffer_*16,
            gram_rhs_bytes=0 if background else 6*8,coordinate_bytes=0 if background else 2*8,
            fold_index_bytes=0 if background else F*(no+nh)*8,
            deployed_plus_fullcore_bytes=(coeff+means+kernels+buffer_)*16+(0 if background else 64+F*(no+nh)*8)+(1024 if method=='POOLED_NATURAL' else 0),
            proxy_excludes='Gram O(F Tw nh M^2), real solve O(M^3), scalar weighting and additions',
            total_MAC_percent_vs_corr4=100*(deploy+(0 if background else fit)-332288)/332288))
    return rows

def main():
    op,bank=load_bank();cvs,_=common.family(op,SNRS)
    cfg=read(OUT/'config.json')['timing'];rng=np.random.default_rng(cfg['order_seed'])
    kernels={(s,m):prevalidated_deployment(m,op,bank[s],s) for s in SNRS for m in METHODS}
    windows={s:[] for s in SNRS}
    for condition in CONDITIONS:
        for i in range(20):
            raw,_=source_input(condition,i)
            for si,s in enumerate(SNRS):
                window=raw['pilots'][si,:5].copy();u=window[-1]-op['source_mean_p'];cc=raw['coordinates'][si,0].copy()
                windows[s].append((window,u,cc,condition,i))
                for method in METHODS:
                    from controls import deployed
                    close(kernels[s,method](u,cc),deployed(method,u,cc,op,bank[s],s),'PREVALIDATED_KERNEL')
                fitted,diag=fit_both(cvs[s],window)
                close(fitted,cc,'TIMING_ORIGINAL_FIT_REPLAY')
                if diag['normal_equation_relative_residual']>common.SOLVE_TOL:raise ValueError('TIMING_SOLVE')
    if any(len(windows[s])!=100 for s in SNRS):raise ValueError('TIMING_WINDOWS')
    threads=blas_threads();before=rss()
    if any(item['threads']!=1 for item in threads):raise ValueError('TIMING_THREADS')
    warm=cfg['warmup_calls'];blocks=cfg['blocks'];records=[];out=None
    def call(endpoint,s,method,row):
        window,u,cc,_,_=row
        if endpoint=='full-estimator-core' and method not in METHODS[:2]:cc,_=fit_both(cvs[s],window)
        return kernels[s,method](u,cc)
    for endpoint in cfg['endpoint_names']:
        for s in SNRS:
            for method in METHODS:
                for k in range(warm):out=call(endpoint,s,method,windows[s][k%100])
            for block in range(blocks):
                order=[METHODS[x] for x in rng.permutation(len(METHODS))]
                for position,method in enumerate(order):
                    t=time.perf_counter_ns()
                    for row in windows[s]:out=call(endpoint,s,method,row)
                    elapsed=time.perf_counter_ns()-t
                    if not np.isfinite(out).all():raise ValueError('TIMING_NONFINITE')
                    records.append(dict(endpoint=endpoint,snr_db=s,block=block,order_position=position,method=method,calls=100,elapsed_ns=elapsed,ns_per_call=elapsed/100))
            print(json.dumps(dict(endpoint=endpoint,snr_db=s,blocks=blocks)),flush=True)
    summaries=[];ratios=[]
    for endpoint in cfg['endpoint_names']:
        for s in SNRS:
            vals={m:np.array([r['ns_per_call'] for r in records if r['method']==m and r['snr_db']==s and r['endpoint']==endpoint]) for m in METHODS}
            for m,v in vals.items():
                summaries.append(dict(endpoint=endpoint,snr_db=s,method=m,median_us=float(np.median(v)/1000),q25_us=float(np.quantile(v,.25)/1000),q75_us=float(np.quantile(v,.75)/1000),blocks=len(v),windows_per_block=100,warmup_calls=warm))
                ratio=vals['CORR_POOL_M2_R4']/v
                ratios.append(dict(endpoint=endpoint,snr_db=s,method=m,reference='CORR_POOL_M2_R4',paired_reference_over_method_median=float(np.median(ratio)),ratio_q25=float(np.quantile(ratio,.25)),ratio_q75=float(np.quantile(ratio,.75)),description='same-host paired block descriptive ratio; above1 means method faster'))
    csvout(OUT/'tables/timing_blocks.csv',records);csvout(OUT/'tables/timing_summary.csv',summaries);csvout(OUT/'tables/timing_paired_ratios.csv',ratios);csvout(OUT/'tables/costs.csv',costs())
    factor_rows=read(OUT/'reports/factor_construction.json')['audits'];offline=[]
    for s in SNRS:
        for method in CONTROL:
            rows=[r for r in factor_rows if r['method']==method and r['snr_db']==s]
            construction=sum(r['original_module_construction_seconds'] for r in rows)
            offline.append(dict(method=method,snr_db=s,construction_seconds=construction,svd_seconds=sum(r['original_module_svd_seconds'] for r in rows),audit_seconds=sum(r['audit_and_optional_construction_seconds'] for r in rows),status='baseline verified by repeat construction, frozen factors reused' if method=='CORR_ORIGIN_R4' else ('shared SVD across rank23/24; do not add arm times' if method.startswith('WHOLE') else 'new control constructed')))
    csvout(OUT/'tables/offline_construction.csv',offline)
    loaded=int(sum(v.nbytes for v in op.values())+sum(f.A.nbytes+f.Q.nbytes for banks in bank.values() for fs in banks.values() for f in fs)+sum(row[0].nbytes+row[1].nbytes+row[2].nbytes for rows in windows.values() for row in rows))
    write(OUT/'reports/timing.json',dict(PASS=True,threads=threads,resource_before=before,resource_after=rss(),measured_calls=len(records)*100,warmup_calls=2*3*9*warm,loaded_op_factor_window_bytes=loaded,
        minimal_assets_scope='per operating SNR, complex128 operators/factors and necessary means only; common full kernels and window separately',
        extra_validation_assets='all SNR controls, all frozen op archive arrays, originals and 300 fixed windows are resident for experiment verification; RSS is not deployment storage',
        after_preservation=verify_sources(),kernel_validation='prevalidated per-method closures match checked mathematical deployed; no matrix scan inside timed kernel',
        no_inference_data_generated=True,no_incremental_fit=True,fullcore_coordinate_reuse=False))
    print(json.dumps(dict(action='benchmark',PASS=True,measured_calls=len(records)*100)),flush=True)
if __name__=='__main__':main()
