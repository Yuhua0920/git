"""Generate R3 technical reports from bounded synthetic measurements only."""
from scripts.sp_u12b5b_r3_common import *
import math, statistics as st
GIB=2**30
PORT=ROOT/'experiments/mfa_low_memory_prototype/portable'
PREFIX='sp_u12b5b_r3_'
def save(name,value):write(ROOT/'outputs'/f'{PREFIX}{name}.json',value)
def document(name,value):
 (ROOT/'docs'/f'{PREFIX}{name}.md').write_text(value,encoding='utf8')
def percentile(values,p):
 s=sorted(values);x=(len(s)-1)*p;i=int(x);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(x-i)
def original(n,k,r):
 # Same source-lifetime expression as frozen R2; no historical science decoded.
 d=4032;a=16*n*d;c=16*k*d*d;pars=2*c+16*k*d+16*k*d*r+8*k*d+8*k
 small=8*k*n+16*k*r*d+16*r*n+32*r*r*n+16*r*r+8*k
 stages={'input':a,'KMeans':4*a+2*c+16*k*d+24*n,
 'covariance_update':3*a+pars+small+2*c,'likelihood':5*a+pars+16*k*n+8*n,
 'M_step_loading':3*a+pars+small+16*d*r*n,'M_step_residual':6*a+pars+small,
 'public_parameter_copies':a+2*pars,'serialization_validity':a+2*pars+8*k*d*d+16*64*d}
 return dict(N=n,D=d,K=k,rank=r,input_bytes=a,private_parameter_bytes=pars,
 largest_temporary_bytes=16*d*r*n,mandatory_input_plus_largest_temporary_bytes=a+16*d*r*n,
 source_live_bytes=max(stages.values()),peak_stage=max(stages,key=stages.get),stages=stages)
def low(n,k,r):
 o=original(n,k,r);d=o['D'];a=o['input_bytes'];pbytes=o['private_parameter_bytes'];c=16*k*d*d
 # Conservative overlapping expression buffers; explicit deletion before covariance update.
 small=8*k*n+16*k*r*d+48*r*n+64*r*r+16*d*r+8*k
 stages={'input':a,'KMeans':o['stages']['KMeans'],
 'likelihood':3*a+pbytes+16*k*n+24*n,
 'M_step_means':4*a+pbytes+small,
 'M_step_loading':3*a+pbytes+small,
 'M_step_residual':int(3.5*a)+pbytes+small,
 'covariance_update':a+pbytes+2*c+8*d*d+small,
 'public_parameter_copies':a+2*pbytes,
 'serialization_validity':o['stages']['serialization_validity']}
 return o|dict(largest_temporary_bytes=max(a,c,8*k*d*d),
 largest_covariance_temporary_overlap_bytes=2*c,
 weighted_latent_buffers_bytes=32*r*n,
 source_live_bytes=max(stages.values()),peak_stage=max(stages,key=stages.get),stages=stages,
 large_DrN_tensor_materialized=False)
def source_line(needle,path=CORE):
 found=[i for i,s in enumerate(path.read_text(encoding='utf8').splitlines(),1) if needle in s]
 assert found;return found[-1] if needle.startswith(('X1 =','X2 =','p = np.sum(X1')) else found[0]
def memory_reports():
 rows=[];models={}
 for method,fn in [('UNCHANGED_AUTHOR_IMPLEMENTATION',original),('MATHEMATICALLY_EQUIVALENT_LOW_MEMORY_IMPLEMENTATION',low)]:
  models[method]=[]
  for n in (12000,15000):
   for k in (1,2,4):
    for r in (4,8,16):
     m=fn(n,k,r);models[method].append(m)
     rows.append({'implementation':method,'N':n,'D':4032,'K':k,'rank':r,
      'input_GiB':m['input_bytes']/GIB,'persistent_private_GiB':m['private_parameter_bytes']/GIB,
      'largest_temporary_GiB':m['largest_temporary_bytes']/GIB,
      'source_simultaneous_live_GiB':m['source_live_bytes']/GIB,'peak_stage':m['peak_stage'],
      'native_workspace_GiB':'NOT_MEASURED_FULL_FIT; planning reserve 2..8 original / 1..4 prototype',
      'allocator_Python_GiB':'NOT_SEPARATELY_MEASURED; included in planning reserve',
      'actual_full_fit_RSS':'NOT_MEASURED','actual_full_fit_private_commit':'NOT_MEASURED',
      'evidence':'SOURCE_BASED_LIVE_ESTIMATE; not measured full-fit RSS'})
 csvwrite(ROOT/'outputs'/f'{PREFIX}memory_model.csv',rows)
 a=original(15000,4,16);b=low(15000,4,16)
 tiers={'UNCHANGED_AUTHOR_IMPLEMENTATION':{'16':'ESTIMATED_INSUFFICIENT','32':'BORDERLINE','48':'ESTIMATED_FEASIBLE','64':'ESTIMATED_FEASIBLE'},
 'MATHEMATICALLY_EQUIVALENT_LOW_MEMORY_IMPLEMENTATION':{'16':'BORDERLINE','32':'ESTIMATED_FEASIBLE','48':'ESTIMATED_FEASIBLE','64':'ESTIMATED_FEASIBLE'}}
 # All RAM ratings conditional on installed usable RAM and fresh host inventory.
 details={}
 for method,m,native,osreserve,headroom,recommended,launch in [
  ('UNCHANGED_AUTHOR_IMPLEMENTATION',a,8,8,4,48,36),
  ('MATHEMATICALLY_EQUIVALENT_LOW_MEMORY_IMPLEMENTATION',b,4,6,4,32,20)]:
  details[method]={'worst_candidate_source_live_GiB':max(v['source_live_bytes']/GIB for v in models[method] if v['N']==12000),
   'worst_final_source_live_GiB':m['source_live_bytes']/GIB,
   'actual_full_fit_peak_RSS':'NOT_MEASURED','actual_full_fit_peak_private_commit':'NOT_MEASURED',
   'lean_conditional_system_estimate_GiB':m['source_live_bytes']/GIB+(2 if method.startswith('UNCHANGED') else 1)+4+4,
   'conservative_planning_system_GiB':m['source_live_bytes']/GIB+native+osreserve+headroom,
   'native_allocator_runtime_reserve_GiB':native,'OS_background_reserve_GiB':osreserve,'physical_safety_GiB':headroom,
   'planning_process_private_commit_GiB':m['source_live_bytes']/GIB+native,
   'minimum_recommended_installed_RAM_GiB':recommended,'minimum_available_RAM_at_launch_GiB':launch,
   'required_free_storage_GiB':64,'RAM_classification':tiers[method],
   'current_busy_16GB_host':'ESTIMATED_INSUFFICIENT under unchanged 6 GiB process cap; no full-fit authorization',
   'laboratory_32GB_host':'NOT_VERIFIED; CPU, BLAS, usable/available RAM, commit, disk unknown',
   'recommended_RAM_explanation':'48 GiB lean-host original option; 64 GiB retains frozen R2 conservative recommendation' if method.startswith('UNCHANGED') else '32 GiB conditional on separate engineering fidelity and host-specific safety approval',
   'full_fit_MEASURED_READY':False}
 result={'units':'GiB throughout; nominal 16/32/48/64 classes must be checked against OS usable RAM',
  'models':models,'estimates':details,'measurement_vs_estimation':'Full-sized covariance kernels measured, full-N EM never allocated/run',
  'native_workspace_bound':'UNKNOWN; planning reserves are explicit assumptions, not guaranteed upper bounds',
  'laboratory_host_tested':False}
 save('minimum_ram_estimates',result);return result
def cpu_audit():
 specs=[
 ('KMeans','(N,2D),K','float64','O(I_KM NDK); kmeans++ scans','once/fit','4 input-sized arrays + covariances','OpenMP and implementation-specific BLAS','memory/compute mixed','Kmeans = cluster.KMeans'),
 ('dense covariance + Woodbury','K,D,r','complex128','O(KD^2r+Kr^3+KDr^2)','initial + every EM','dense covariance + inverse + expression temporaries','GEMM, small SVD/pinv','low-rank GEMM memory/compute mixed','self._covs = self._lambdas @'),
 ('slogdet','K,D,D','complex128','K O(D^3) LU','every EM; batch K factorizations','native LU workspaces UNKNOWN','LAPACK/OpenBLAS','compute heavy','sgn, logdet = np.linalg.slogdet(self._covs)'),
 ('dense likelihood','D,D times D,N, per component','complex128','K D^2N complex MAC','every EM; K matmuls','centered data, product, quadratic temporaries','GEMM','large compute and bandwidth','X2 = self._inv_covs[k] @ X1'),
 ('beta','K,r,D times K,D,D','complex128','K rD^2 complex MAC','every EM','K rD and conjugate arrays','GEMM','low-rank shape parallel overhead possible','betas = np.transpose'),
 ('latent posterior','r,D times D,N','complex128','K rDN; K r^2N moments','every EM; per component','rN, r^2N, centered DN','GEMM; original elementwise + dot','mixed','latents = betas[k]'),
 ('M-step loading tensor','D,r,N','complex128','K DrN + K r^2N + K r^3','every EM; per component','16 DrN bytes eager, not a view','dot reduction + small SciPy inv','large allocation/bandwidth; dot BLAS path depends strides','self._lambdas[k] = np.dot'),
 ('residual','D,N','complex128 inputs / real output','O(KDN) + old loading DrN','every EM; per component','up to three complex DN temporaries','weighted dot; prototype einsum','bandwidth/reduction heavy','psis = np.real(np.dot'),
 ('parameter copies','K D^2','complex128','O(KD^2) bytes','once/fit','private/public duplicates','memory copy, not useful BLAS','bandwidth limited','self.means = self._means.copy')]
 rows=[]
 for name,shape,dtype,ops,freq,alloc,blas,bound,needle in specs:
  rows.append(dict(kernel=name,source=CORE.relative_to(ROOT).as_posix(),line=source_line(needle),input_shape=shape,dtype=dtype,
  operations=ops,execution_frequency=freq,allocation=alloc,BLAS_or_OpenMP=blas,
  multithreaded='actual synthetic measurements and pools in benchmark summary; not assumed',bottleneck=bound,
  single_thread_limit='elementwise and optimize=False reductions largely single-thread; inspect measured CPU/wall',full_scale_measured=False))
 csvwrite(ROOT/'outputs'/f'{PREFIX}cpu_kernel_audit.csv',rows)
 return rows
def extract_measurements():
 cfg=read(ROOT/'outputs'/f'{PREFIX}thread_benchmark_config.json');assert read(OUT/'benchmarks_complete.json')['PASS']
 raw=[];resources=[];allmetadata={};parities={};missing_annotations=0
 for t in cfg['thread_candidates']:
  parities[str(t)]=read(OUT/f'parity_{t}/parity.json');resources.append(read(OUT/f'parity_{t}/resource_summary.json'))
 for job in cfg['schedule']:
  complete=read(OUT/'benchmarks_complete.json');path=ROOT/complete['job_paths'][str(job['job'])] if 'job_paths' in complete else OUT/f"job_{job['job']:03d}"
  resource=read(path/'resource_summary.json');resources.append(resource)
  samples=[json.loads(s) for s in (path/'resource.jsonl').read_text(encoding='utf8').splitlines()]
  missing_annotations+=sum(s['marker'] is None for s in samples)
  for line in (path/'timings.jsonl').read_text(encoding='utf8').splitlines():
   v=json.loads(line);assert v['job']==job['job'] and v['threads']==job['threads']
   assert all(p['num_threads']==v['threads'] for p in v['actual_pools'])
   assert v['metadata']['input_hashes']==allmetadata.setdefault(v['kind'],v['metadata']['input_hashes'])
   stage=[s for s in samples if s['marker'] and s['marker']['kind']==v['kind']]
   v['sampled_case_peak_RSS']=max((s['process']['working_set_bytes'] for s in stage),default=max(v['RSS_before'],v['RSS_after']))
   v['sampled_case_peak_private_commit']=max((s['process']['private_bytes'] for s in stage),default=v['private_commit_after'])
   v['job_peak_RSS']=resource['peak_RSS'];v['job_min_available']=resource['minimum_available_physical_bytes'];raw.append(v)
 assert len(raw)==len(cfg['schedule'])*len(cfg['cases'])
 for a,b in [('loading_direct','loading_low'),('latent_direct','latent_low'),('quadratic_direct','quadratic_low'),('residual_direct','residual_low'),('em_original','em_low')]:assert allmetadata[a]==allmetadata[b]
 csvwrite(ROOT/'outputs'/f'{PREFIX}thread_benchmark_raw.csv',[
  {k:(json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v) for k,v in row.items()} for row in raw])
 summaries=[]
 for t in cfg['thread_candidates']:
  for case in cfg['cases']:
   kind=case['kind'];a=[v for v in raw if v['kind']==kind and v['threads']==t];assert len(a)==5
   durations=[v['wall_ns']/1e9 for v in a];reference=st.median([v['wall_ns']/1e9 for v in raw if v['kind']==kind and v['threads']==1]);median=st.median(durations)
   peak=st.median([v['sampled_case_peak_RSS'] for v in a]);basepeak=st.median([v['sampled_case_peak_RSS'] for v in raw if v['kind']==kind and v['threads']==1])
   summaries.append({'kernel':kind,'threads':t,'blocks':5,'median_seconds':median,'min_seconds':min(durations),'max_seconds':max(durations),
    'IQR_seconds':percentile(durations,.75)-percentile(durations,.25),'MAD_seconds':st.median([abs(x-median) for x in durations]),
    'sample_CV':st.stdev(durations)/st.mean(durations),'max_min_ratio':max(durations)/min(durations),
    'one_thread_seconds':reference,'speedup':reference/median,'parallel_efficiency':reference/median/t,
    'sampled_case_peak_RSS_median_GiB':peak/GIB,'RSS_increase_over_1thread_GiB':(peak-basepeak)/GIB,
    'whole_worker_peak_RSS_max_GiB':max(v['job_peak_RSS'] for v in a)/GIB,
    'private_commit_after_median_GiB':st.median(v['private_commit_after'] for v in a)/GIB,
    'CPU_core_equivalent_median':st.median(v['effective_CPU_cores'] for v in a),
    'CPU_time_resolution_note':'Windows process CPU time is quantized despite ns API; sub15.6ms kernels may report0; do not interpret as idle',
    'host_CPU_percent_median':st.median(v['CPU_utilization_percent_of_logical_host'] for v in a),
    'numerical_parity_PASS':all(v['numerical_parity_PASS'] for v in a),
    'measurement_scope':'case RSS includes fixture, warmup, reference check, allocator retention; worker peak includes all cases',
    'actual_threads_verified':True,'formal_full_fit':False})
 csvwrite(ROOT/'outputs'/f'{PREFIX}thread_benchmark_summary.csv',summaries)
 def compact(value):
  v=value['variants'];return {'PASS':value['PASS'],'expression_parity':value['expression_parity'],
  'B5A_exact_analytical_case':value['B5A_exact_analytical_case'],
  'variants':{name:{'PASS':vv['PASS'],'same_stop_decisions':vv['same_stop_decisions'],
    'checks_count':len(vv['checks']),'max_absolute':max(x['max_absolute'] for x in vv['checks'].values()),
    'max_relative':max(x['max_relative'] for x in vv['checks'].values()),
    'failures':{n:x for n,x in vv['checks'].items() if not x['PASS']}} for name,vv in v.items()},
  'stopping':value['stopping'],'actual_pools':value['actual_pools']}
 parity={'predeclared_tolerance':cfg['parity_tolerance'],'threads':{t:compact(v) for t,v in parities.items()},
 'all_thread_EM_and_expression_PASS':all(v['PASS'] for v in parities.values()),
 'all_benchmark_kernel_PASS':all(v['numerical_parity_PASS'] for v in raw),
 'no_tolerance_relaxation':True,'formal_full_scale_fidelity_certified':False}
 save('multithread_parity',parity)
 monitor={'actual_synthetic_peak_RSS_GiB':max(v['peak_RSS'] for v in resources)/GIB,
 'actual_synthetic_peak_private_commit_GiB':max(v['peak_private_commit'] for v in resources)/GIB,
 'actual_minimum_available_physical_GiB':min(v['minimum_available_physical_bytes'] for v in resources)/GIB,
 'all_workers_completed':all(v['completed'] for v in resources),'resource_limits_unchanged':cfg['resource_limits'],
 'process_concurrency':1,'measurement_rows':len(raw),'completed_measurement_workers':len(resources),
 'technical_failed_worker_attempts':3,'total_worker_attempts_including_failed':len(resources)+3,
 'all_workers_completed_scope':'selected measurement workers only; three monitor failures archived and replaced',
 'missing_optional_stage_annotations_samples':missing_annotations,
 'optional_annotation_missing_does_not_omit_process_or_system_memory_sampling':True}
 assert monitor['actual_synthetic_peak_RSS_GiB']<6 and monitor['actual_minimum_available_physical_GiB']>=2
 return cfg,raw,summaries,parity,monitor
def projected_iteration(rows,n,k,r,implementation):
 t={v['kernel']:v['median_seconds'] for v in rows};lowpath=implementation=='low'
 # Dominant source call counts; two rDN GEMMs for posterior and old W*H.
 # covariance_init includes Woodbury: DO NOT add independent woodbury measurement again.
 terms={'dense_LU':t['slogdet'], 'dense_GEMM':t['likelihood_gemm']*n/512,
 'quadratic':t['quadratic_low' if lowpath else 'quadratic_direct']*n/512,
 'covariance_and_Woodbury':t['covariance_init']*r/16,'beta':t['beta']*r/16,
 'posterior_and_old_loadings_GEMMs':2*t['loading_low']*(4032/512)*(n/1024)*(r/16),
 'loading_numerator':t['loading_low' if lowpath else 'loading_direct']*(4032/512)*(n/1024)*(r/16),
 'latent_covariance':t['latent_low' if lowpath else 'latent_direct']*(n/2048)*(r/16)**2,
 'residual':t['residual_low' if lowpath else 'residual_direct']*n/512,
 'mean_and_centering_allowance':t['residual_low']*n/512}
 return {'seconds':k*sum(terms.values()),'component_terms_seconds':terms,'K':k,'rank':r,'N':n,'implementation':implementation}
def training_and_selection(cfg,raw,summary,parity):
 projects=[]
 for t in cfg['thread_candidates']:
  rows=[v for v in summary if v['threads']==t]
  for impl in ('original','low'):
   pairs=[projected_iteration(rows,12000,k,r,impl) for k in (1,2,4) for r in (4,8,16)]
   block_times=[]
   for block in range(5):
    blockrows=[{'kernel':v['kind'],'median_seconds':v['wall_ns']/1e9} for v in raw if v['threads']==t and v['block']==block]
    block_times.append(sum(projected_iteration(blockrows,12000,k,r,impl)['seconds'] for k in (1,2,4) for r in (4,8,16)))
   required=['slogdet','likelihood_gemm','covariance_init']
   stable=all(v['max_min_ratio']<=2 for v in rows if v['kernel'] in required)
   eligible=parity['threads'][str(t)]['variants'][impl]['PASS'] and parity['threads'][str(t)]['PASS'] and all(v['numerical_parity_PASS'] for v in rows) and stable
   projects.append({'threads':t,'implementation':impl,'sum_nine_candidate_iteration_seconds':sum(v['seconds'] for v in pairs),
    'block_projected_seconds':block_times,'projected_relative_IQR':(percentile(block_times,.75)-percentile(block_times,.25))/st.median(block_times),
    'stable_dominant_measurements':stable,'numerical_and_safety_eligible':eligible,'candidate_iterations':pairs})
 recommendation={}
 for impl in ('original','low'):
  eligible=[v for v in projects if v['implementation']==impl and v['numerical_and_safety_eligible']]
  if not eligible:recommendation[impl]=None;continue
  best=min(eligible,key=lambda x:x['sum_nine_candidate_iteration_seconds']);margin=max(.05,best['projected_relative_IQR'])
  near=[v for v in eligible if v['sum_nine_candidate_iteration_seconds']<=best['sum_nine_candidate_iteration_seconds']*(1+margin)]
  chosen=min(near,key=lambda x:x['threads']);recommendation[impl]=chosen['threads']
 rec={'recommended_threads':recommendation,'scope':'THIS_HOST_ONLY; laboratory CPU rebenchmark required; official B5B remains one thread',
 'projects':projects,'rule':cfg['stable_measurement_rule'],'formal_config_changed':False,
 'projection_model':'source-frequency weighted kernels, not largest single GEMM speedup; five blocked predictions',
 'uncertainties':['N scaling measured at N512/1024/2048, not full N12000/15000','r interpolation from r16 assumes dominant linear/quadratic operation scaling','sustained thermal state unknown','initialization NDK scaling approximate','memory allocator, BLAS packing, convergence data dependency unknown']}
 save('cpu_thread_recommendation',rec)
 scenarios={'optimistic':{'iterations':7,'execution_multiplier':.8},'central':{'iterations':100,'execution_multiplier':1.5},'conservative':{'iterations':100,'execution_multiplier':4.0}}
 paths={}
 for impl in ('original','low'):
  for threads in sorted({1,recommendation[impl]}-{None}):
   rows=[v for v in summary if v['threads']==threads];init=next(v['median_seconds'] for v in rows if v['kernel']=='initialization')
   # Extrapolate initialization by N*D*K; no EM factor used for initialization.
   initfit=lambda n,k:init*(n/2048)*(4032/256)*(k/4)
   source=sum(projected_iteration(rows,12000,k,r,impl)['seconds'] for k in (1,2,4) for r in (4,8,16))
   final=[projected_iteration(rows,15000,k,r,impl) for k in (1,2,4) for r in (4,8,16)]
   details={}
   for name,s in scenarios.items():
    nine=s['execution_multiplier']*(s['iterations']*source+sum(initfit(12000,k) for k in (1,2,4) for r in (4,8,16)))/3600
    last=[s['execution_multiplier']*(s['iterations']*v['seconds']+initfit(15000,v['K']))/3600 for v in final]
    details[name]={'nine_candidates_hours':nine,'final_retrain_hours_range':[min(last),max(last)],'total_hours_range':[nine+min(last),nine+max(last)],**s}
   paths[f'{impl}_{threads}_threads']={'evidence':'EXTRAPOLATED; no complete formal fit measured','scenarios':details,'initialization_scaling':'N*D*K, from D256 N2048 K4','parameter_copy_serialization':'not separately measured; margin covers unmodeled operations, not an upper bound'}
 cost={'paths':paths,'scenarios':scenarios,'recommended_threads':recommendation,
  'unchanged_max_iterations':100,'old_R2_total_central_hours':[34.7829,42.0168],
  'old_R2_comparison':'R3 estimates use current blocked measurements and more explicit kernels; no measured full-fit speedup claim',
  'no_convergence_prediction':'7-iteration optimistic is earliest permitted stop, not forecast; 100-iteration central/conservative are budget scenarios'}
 save('training_cost_model',cost);return rec,cost
def inference(summary,rec):
 result={}
 for t in sorted({1,*[v for v in rec['recommended_threads'].values() if v is not None]}):
  rows={v['kernel']:v for v in summary if v['threads']==t};details={}
  for k in (1,2,4):
   def interp(kind):
    if k in (1,4):return rows[f'{kind}_K{k}']['median_seconds']
    return rows[f'{kind}_K1']['median_seconds']+(rows[f'{kind}_K4']['median_seconds']-rows[f'{kind}_K1']['median_seconds'])/3
   setup=interp('inference_setup');batch=interp('inference_batch');per= batch/6
   details[str(k)]={'setup_seconds':setup,'batch6_including_noise_setup_seconds':batch,'per_scored_frame_seconds':per,
    'validation_16200_hours':per*16200/3600,'validation_source_32400_hours':per*32400/3600,
    'formal_210000_hours':per*210000/3600,
    'evidence':'EXTRAPOLATED_FROM_MEASURED_SYNTHETIC_D4032_Q64_BATCH6' if k!=2 else 'EXTRAPOLATED K2 linear interpolation K1..K4',
    'no_setup_double_counting':True}
  result[str(t)]={'per_K':details,
    'balanced_nine_candidate_validation_hours_16200':sum(v['per_scored_frame_seconds'] for v in details.values())*(16200/3)/3600,
    'balanced_nine_candidate_validation_hours_source_32400':sum(v['per_scored_frame_seconds'] for v in details.values())*(32400/3)/3600,
    'allocation_of_validation_budget':'equal by K because three ranks per K; rank16 timing applied to same dense inference interface, K2 interpolated',
    'per_K_validation_budget_fields':'hypothetical all stated frames using that K; actual balanced nine-candidate total given separately'}
 out={'threads':result,'validation_counts':{'task':16200,'frozen_worker_derived':32400,'derivation':'9 candidates * 3 origins * 2 SNR * 100 trajectories * 6 scored frames = 32400','status':'COUNT_MISMATCH; no protocol amendment'},
 'formal_frames':210000,'setup_scope':'author estimator re-prepares noise-dependent quantities on each estimate(batch6), already included in batch latency',
 'repeated_dense_C_AH':'author _lmmse recomputes covariance @ A^H for each observation/component; retained faithfully',
 'low_memory_training_changes_inference':False,'actual_target_predictions':0,'actual_validation_scores':0,
 'limitations':'synthetic covariances/observations; mixture posterior weights can affect optional component processing; baseline float1.0 full mixture retained; no noise-setup caching invented'}
 save('inference_cost_model',out);return out
def equivalence(parity,summary,monitor,memory):
 transformations=[]
 for n in (12000,15000):
  d=4032;r=16
  transformations.append({'N':n,'D':d,'r':r,'loading_original_tensor_bytes':16*d*r*n,
   'loading_original_plus_conjugate_bytes':16*d*r*n+16*r*n,
   'loading_replacement_two_weighted_latent_buffers_bytes':32*r*n,
   'loading_numerator_output_bytes':16*d*r,
   'latent_original_two_r_r_N_bytes':32*r*r*n,
   'latent_low_weighted_buffers_and_r_r_bytes':32*r*n+48*r*r,
   'quadratic_original_two_DN_temporaries_bytes':32*d*n,'quadratic_replacement_real_views_extra_bytes':24*n,
   'residual_original_three_complex_DN_bytes':48*d*n,'residual_low_one_real_DN_plus_output_bytes':8*d*n+24*d,
   'evidence':'exact array shape arithmetic; BLAS workspace/allocator extra UNKNOWN'})
 eq=r'''# R3 low-memory full-batch identities (experimental only)

For one component, X is N-by-D complex128, q_n are its original real responsibilities,
H is r-by-N posterior latent means computed from OLD parameters, W^- is OLD loading,
R^- = I - beta W^-, V^- = W^-H. Compute new mean exactly in the author order,
mu^+ = sum_n q_n (x_n - V^-_n) / sum_n q_n, and Z = X^T - mu^+.

1. Loading numerator: A_da = sum_n q_n Z_dn conjugate(H_an).
Original dot(Z[:,None,:] * conjugate(H)[None,:,:],q) materializes (D,r,N).
Equivalent A = Z (conjugate(H) diag(q))^T. The transpose is ordinary, not Hermitian:
the H conjugation has already occurred. Both are full-batch, O(DrN).
Largest replacement numerator input buffers are two complex r-by-N arrays, plus D-by-r output.
The actual low-memory code uses a weighted GEMM, not a minibatch or changed precision.

2. Latent second moment: Gamma = sum_n q_n [R^- + H_n H_n^H]
= (sum_n q_n)R^- + H (conjugate(H) diag(q))^T.
Then W^+ = A inv(Gamma), using the same SciPy small inverse as the author.
The r-by-r-by-N second-moment tensor is eliminated. OLD beta and W remain paired.

3. Likelihood quadratic: Re sum_d conjugate(X1_dn)X2_dn
= sum_d Re(X1_dn)Re(X2_dn) + Im(X1_dn)Im(X2_dn).
Two optimize=False einsum reductions use real views and length-N outputs;
the existing D-by-D inverse @ centered data and dense slogdet are retained.

4. Residual: psi_d^raw = Re sum_n q_n (Z_dn - V^-_dn) conjugate(Z_dn) / sum_n q_n
= [sum_n q_n (Re Z_dn - Re V^-_dn)Re Z_dn
 + sum_n q_n (Im Z_dn - Im V^-_dn)Im Z_dn] / sum_n q_n.
V^- is OLD loading-times-latents, including after W^+ has been stored. Replacing it
with W^+H changes the EM update and is NOT_EQUIVALENT. Floor1e-6, PPCA mean,
amplitudes, lock_psis=False, rs_clip=0, zero restarts, and original stopping remain.

5. Delete dead component-local arrays after every component and beta before covariance
update. This changes lifetime, not a numerical expression or update order.

All identities are MATHEMATICALLY_EQUIVALENT in exact arithmetic. Finite precision
reassociation and thread reduction order mean NUMERICALLY_CLOSE, not bitwise same.
R3 fidelity concerns synthetic/toy traces only, not full-scale formal certification.
Blockwise sufficient statistics, determinant lemma, matrix-free likelihood, online
noise estimation, minibatch EM, changed dtype, new residual update are NOT implemented.
'''
 (ROOT/'outputs'/f'{PREFIX}low_memory_equations.md').write_text(eq,encoding='utf8')
 lowpass=all(v['variants']['low']['PASS'] for v in parity['threads'].values())
 measured={}
 for family in ('loading','latent','quadratic','residual'):
  measured[family]=[]
  for t in sorted({v['threads'] for v in summary}):
   a=next(v for v in summary if v['threads']==t and v['kernel']==family+'_direct');b=next(v for v in summary if v['threads']==t and v['kernel']==family+'_low')
   measured[family].append({'threads':t,'direct_seconds':a['median_seconds'],'low_seconds':b['median_seconds'],'speedup':a['median_seconds']/b['median_seconds'],
    'direct_sampled_case_RSS_GiB':a['sampled_case_peak_RSS_median_GiB'],'low_sampled_case_RSS_GiB':b['sampled_case_peak_RSS_median_GiB'],
    'RSS_scope':'includes warmup/fixture/retained allocator; not isolated expression peak'})
 result={'mathematical_class':'MATHEMATICALLY_EQUIVALENT','finite_precision_class':'NUMERICALLY_CLOSE' if lowpass else 'NOT_ESTABLISHED',
 'all_toy_and_multistep_PARITY_PASS':lowpass,'formal_baseline_fidelity_certification':False,
 'temporary_models':transformations,'measured_paired_kernel_tradeoff':measured,'monitor':monitor,
 'full_N_EM_peak_RSS':'NOT_MEASURED','full_N_EM_runtime':'NOT_MEASURED',
 'means_or_update_order_changed':False,'old_residual_loading_preserved':True,'full_batch':True,'dtype':'complex128',
 'formal_runner_imports_prototype':False,'author_and_adapter_edits':0,
 'original_simultaneous_live_estimate_GiB':original(15000,4,16)['source_live_bytes']/GIB,
 'prototype_simultaneous_live_estimate_GiB':low(15000,4,16)['source_live_bytes']/GIB,
 'engineering_complexity':'moderate: isolated expression substitutions + explicit lifetimes; full-scale fidelity gate still required'}
 save('low_memory_prototype_audit',result)
 document('low_memory_equivalence_review',eq+'\n\n'+json.dumps(result,indent=2))
 return result
def decision_and_docs(memory,rec,cost,inf,eq,parity,monitor):
 verdict='MFA_32GB_EQUIVALENT_OPTIMIZATION_PROMISING' if eq['all_toy_and_multistep_PARITY_PASS'] and all(v is not None for v in rec['recommended_threads'].values()) else 'MFA_CPU_OPTIMIZATION_NEEDS_REVISION'
 decision={'verdict':verdict,'recommended_path':'PATH_B: equivalent low-memory implementation on 32GB after separate fidelity approval' if verdict.endswith('PROMISING') else 'PATH_C pending CPU/fidelity repair; higher-memory original fallback',
 'laboratory_32GB_actual_tested':False,'laboratory_host_status':'NOT_VERIFIED',
 'original_32GB':'BORDERLINE; do not launch full fit without fresh available RAM and separately approved full-scale smoke',
 'low_memory_32GB':'ESTIMATED_FEASIBLE conditional; prototype NOT formal baseline',
 'current_resource_limits_unchanged':LIMITS,'formal_candidate_fits':0,'validation_scores':0,'selected_structure':None,'final_model':None,
 'target_evaluations':0,'target_scientific_reads':0,'B2_B3_outcome_reads':0,'formal_CPU_config_changed':False,
 'MFA_scientific_semantic_change':False,'floating_bitwise_identity_claimed':False,
 'next_action':'STOP for mentor review. Proposed next gate: laboratory inventory/safe synthetic measurements, independent full-scale engineering fidelity and host-specific safety approval; no training authorization here'}
 save('32gb_feasibility_decision',decision)
 proposed={'status':'PROPOSAL_ONLY_NOT_APPLIED_REQUIRES_SEPARATE_MENTOR_APPROVAL','current_limits':LIMITS,
 'low_memory_32GB_proposal':{'minimum_installed_RAM_GiB':32,'minimum_available_physical_GiB_at_launch':20,
 'max_RSS_GiB':12,'minimum_physical_headroom_GiB':4,'max_private_commit_GiB':16,
 'minimum_available_commit_GiB_at_launch':20,'minimum_free_disk_GiB':64,'sampling_seconds':.05,
 'single_process_one_candidate_at_a_time':True,'thread_policy':'lab-specific benchmark+predeclared parity first; personal-host recommendation does not transfer automatically',
 'source_model_GiB':low(15000,4,16)['source_live_bytes']/GIB,'native_allocator_reserve_GiB':4,
 'cap_explanation':'ceil(source-live + 4 GiB planning reserve) + 2 GiB monitor margin; not proof of an upper bound',
 'kill_conditions':['RSS cap','private commit cap','physical/commit headroom','nonfinite/parity failure','timeout'],
 'full_scale_probe_allowed_now':False},
 'original_higher_RAM_fallback':{'minimum_recommended_GiB':48,'preferred_R2_conservative_GiB':64,'minimum_available_GiB_at_launch':36,
 'original_32GB_special_case':'requires substantially cleaner host, measured overhead, separate protocol; no immediate recommendation'},
 'formal_scientific_config_and_caps_changed':False}
 save('resource_safety_proposal',proposed)
 trade=[]
 for impl in ('original','low'):
  for t in (1,rec['recommended_threads'][impl]):
   if t is None:continue
   name=f'{impl}_{t}_threads'
   if any(v['path']==name for v in trade):continue
   trade.append({'path':name,'implementation':impl,'threads':t,
    'worst_candidate_source_live_GiB':(original if impl=='original' else low)(12000,4,16)['source_live_bytes']/GIB,
    'worst_final_source_live_GiB':(original if impl=='original' else low)(15000,4,16)['source_live_bytes']/GIB,
    'full_fit_RSS':'NOT_MEASURED','full_fit_private_commit':'NOT_MEASURED','full_fit_runtime':'NOT_MEASURED',
    'estimated_central_total_hours_min':cost['paths'][name]['scenarios']['central']['total_hours_range'][0],
    'estimated_central_total_hours_max':cost['paths'][name]['scenarios']['central']['total_hours_range'][1],
    'estimated_runtime_status':'EXTRAPOLATED','small_EM_median_seconds':next(v['median_seconds'] for v in SUMMARY if v['threads']==t and v['kernel']==('em_original' if impl=='original' else 'em_low')),
    'formal_baseline_approved':impl=='original','resource_protocol_changed':False})
 csvwrite(ROOT/'outputs'/f'{PREFIX}memory_runtime_tradeoff.csv',trade)
 document('hardware_recommendation','# R3 hardware recommendation\n\n'+json.dumps(memory['estimates'],indent=2)+'\n\nCurrent execution host is personal 16GB. Laboratory32GB is NOT_VERIFIED. Original 32GB is not ruled out mathematically, but its19.188GiB source-live final leaves little room for workspace, background and safety. Prefer separate low-memory fidelity gate on32GB; retain48/64GB original fallback. CPU core count alone is insufficient: sustained BLAS throughput and lab parity determine threads. Provision64GiB free disk; current free disk below this, sufficient for current synthetic probes only.\n')
 document('cpu_multicore_optimization_review','# R3 CPU measurements and recommendation\n\n'+json.dumps({'recommendation':rec,'resource_measurements':monitor,'cost':cost,'inference':inf},indent=2)+
 '\n\nTimers exclude fixture/reset/hash/parity; five balanced randomized blocks, one warmup per kernel. Default library20threads replaced only inside synthetic workers by actual requested1/2/4/8/12/14. NumPy and SciPy OpenBLAS plus sklearn OpenMP queried. No Python multi-worker oversubscription. Thermal sensor unavailable; OS MHz reports do not establish sustained clock or absence of throttling. Stage RSS includes allocator history, so allocation identities are the precise temporary-memory comparison.\n')
 document('recommended_research_path','# R3 recommended path and portable instructions\n\n'+json.dumps(decision,indent=2)+r'''

Portable bundle: experiments/mfa_low_memory_prototype/portable (independent of project scientific values).
Copy this directory to the laboratory computer manually; no remote deployment was performed.
Install pinned requirements.txt into an isolated Python environment with compatible wheels.
Inventory only: python -B -X utf8 runner.py --role inventory --out C:\mfa_r3_lab_inventory.json
Safe synthetic run: python -B -X utf8 runner.py --role portable-run --out C:\mfa_r3_lab_synthetic_fresh
The destination must be NEW; preserve failures and do not overwrite old attempts.
These commands inventory actual laboratory cores and create a host-specific balanced thread schedule.
They retain the existing RSS<6GiB and available physical>=2GiB safety thresholds.
No full-fit, preparation validation, model selection, final retrain, or target evaluation command exists.
Run only after checking inventory and at least4GiB available physical RAM for these small probes.
Check dependency/BLAS identity and parity again on the new host; do not inherit personal-host threads.
The copied author files/adapter are byte-identical frozen reference snapshots for local technical use.
Prototype results are not new scientific evidence and not certification of a deployable formal baseline.

Validation count discrepancy: task16200 versus frozen source9*3*2*100*6=32400.
Report both cost budgets; mentor must resolve a future protocol count before formal scoring. No scoring/config change here.
''')
 document('technical_lessons','# R3 technical lessons\n\nWindows venv launchers and actual Python worker PIDs have separate lifetimes. Monitoring must query the actual PID; a lost handle is a successful exit only after verified launcher exit0 and a complete final resource record. Preserve failed measurements before any repair.\n\nAtomic rename on Windows can fail while readers hold a target without delete sharing. Use bounded retry for the specific sharing errors, outside timed kernels, and test it; retain failures rather than silently reselecting measurements. Stage-marker reads are optional annotation: transient sharing misses must not disable independent primary RSS/headroom sampling.\n\nResource benchmark repairs must preserve computational bytes, fixed seeds, balanced schedule, tolerance and timer scope; repeat only technically incomplete jobs in fresh directories and retain the originals. Portable entry scripts need an explicit import-root bootstrap.\n')
 document('change_log','# SP-U12B5B-R3 changes\n\nOnly R3 scripts/tests/reports and independent experiments/mfa_low_memory_prototype added. Original source, adapter, R1/R2, all old outputs, manuscript and scientific protocol untouched. Run preflight, independent synthetic parity, 30timing jobs/600rows, script reports, postflight, opaque byte preservation and manifest closure. No commits.\n\nTechnical failure1: after job11 the actual interpreter exited normally while the venv launcher still appeared live. OpenProcess raised WinError87. Initial freeze/source and job11 retained in technical_attempt0_archive; jobs0..10 reused, job11 repeated in fresh job_011_repaired. Monitor repair accepts a lost PID only after launcher exit0 plus final record, checks final RSS cap, and makes state writes atomic. Technical failure2: job13 status os.replace encountered WinError5 while its target file was open for a monitoring read. Repair1 sources/partial job13 retained in technical_attempt1_archive; all13 completed jobs reused and job13 repeated in fresh job_013_repaired2. Only status writes get bounded1-second sharing-conflict retry outside timed kernel intervals. Computation, input, timer, seeds, tolerance, schedule unchanged. Initial9 tests passed; repair1 adds2 tests,11 passed; repair2 adds1 test,12 passed. A helper initial launch lacked root import bootstrap, failed before mutations, then corrected; no measurements existed from that invocation.\n\n'+json.dumps(monitor,indent=2))
 return decision
SUMMARY=[]
def reports():
 global SUMMARY
 firewall();check_frozen();memory=memory_reports();cpu_audit()
 cfg,raw,SUMMARY,parity,monitor=extract_measurements();rec,cost=training_and_selection(cfg,raw,SUMMARY,parity)
 inf=inference(SUMMARY,rec);eq=equivalence(parity,SUMMARY,monitor,memory)
 decision=decision_and_docs(memory,rec,cost,inf,eq,parity,monitor)
 log=ROOT/'docs'/f'{PREFIX}change_log.md'
 log.write_text(log.read_text(encoding='utf8')+'\n\nTechnical failure3: job22 optional current_case read encountered a sharing PermissionError during replacement. Repair2 source and incomplete job retained in technical_attempt2_archive. Jobs0..21 with complete resource records reused; fresh replacement job_022_repaired3. Optional annotation misses now returnNone while primary process/system memory sampling continues. Added a regression test: repair3 preflight13 passed. Three failed timing slots11/13/22 are excluded solely because resource records are incomplete; their original times remain archived. No latency-based sample reselection.\n',encoding='utf8')
 with log.open('a',encoding='utf8') as stream:stream.write('\nReport-only error: first training-cost report stored final projected seconds as floats instead of structured K/r/seconds records. TypeError raised before a cost report was written. Failed report source and error retained in report_generation_failure; corrected structured list, regenerated derived R3 reports, no benchmark rerun and no raw timing edit.\n')
 save('report_completion',{'PASS':True,'decision':decision,'monitor':monitor,'technical_failures_archived':3})
 print(json.dumps({'stage':'R3_reports','verdict':decision['verdict'],'recommendation':rec['recommended_threads'],'monitor':monitor}),flush=True)
def close():
 firewall();check_frozen();freeze=read(OUT/'execution_freeze_repaired3.json')
 for n,h in freeze['sources'].items():assert sha(ROOT/n)==h,n
 assert read(OUT/'pytest_postflight.json')['PASS']
 assert read(ROOT/'outputs'/f'{PREFIX}preservation.json')['PASS']
 assert read(ROOT/'outputs'/f'{PREFIX}b2_b3_seal_audit.json')['PASS']
 required=['host_inventory.json','memory_model.csv','minimum_ram_estimates.json','cpu_kernel_audit.csv','thread_benchmark_config.json',
 'thread_benchmark_raw.csv','thread_benchmark_summary.csv','cpu_thread_recommendation.json','multithread_parity.json',
 'low_memory_equations.md','low_memory_prototype_audit.json','memory_runtime_tradeoff.csv','training_cost_model.json','inference_cost_model.json',
 '32gb_feasibility_decision.json','resource_safety_proposal.json','b2_b3_seal_audit.json','preservation.json']
 assert all((ROOT/'outputs'/f'{PREFIX}{name}').is_file() for name in required)
 files={v.relative_to(ROOT).as_posix():sha(v) for d in ('outputs','docs','scripts','tests','experiments/mfa_low_memory_prototype') for v in (ROOT/d).rglob('*')
 if v.is_file() and owned(v) and v.name not in (f'{PREFIX}manifest.json',f'{PREFIX}manifest.sha256') and not(set(v.parts)&{'__pycache__','pytest_tmp_preflight','pytest_tmp_repair_preflight','pytest_tmp_repair2_preflight','pytest_tmp_repair3_preflight','pytest_tmp_postflight'})}
 value={'task':'SP-U12B5B-R3','files':files,'tree_sha256':tree(files),
 'verdict':read(ROOT/'outputs'/f'{PREFIX}32gb_feasibility_decision.json')['verdict'],
 'preflight_pytest':read(OUT/'pytest_preflight.json'),'postflight_pytest':read(OUT/'pytest_postflight.json'),
 'preservation':read(ROOT/'outputs'/f'{PREFIX}preservation.json'),'seals':read(ROOT/'outputs'/f'{PREFIX}b2_b3_seal_audit.json'),
 'formal_candidate_fits':0,'validation_scores':0,'selected_structure':None,'final_model':None,'target_data_reads':0,'STOP':True}
 path=ROOT/'outputs'/f'{PREFIX}manifest.json';write(path,value);digest=sha(path)
 (ROOT/'outputs'/f'{PREFIX}manifest.sha256').write_text(digest+'\n',encoding='utf8')
 for n,h in value['files'].items():assert sha(ROOT/n)==h,n
 print(json.dumps({'stage':'R3_close','PASS':True,'verdict':value['verdict'],'manifest':str(path),'SHA256':digest,'files':len(files)}),flush=True)
