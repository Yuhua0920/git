"""Technical timing summaries and explicitly conditional cost scenarios."""
import math,statistics as s
from common import protocol,scientific_counters
def stats(values):
    values=list(values);mean=s.mean(values);std=s.stdev(values)if len(values)>1 else None
    return {'n':len(values),'median':s.median(values),'mean':mean,'std':std,
            'coefficient_of_variation':None if std is None or mean==0 else std/mean,'minimum':min(values),'maximum':max(values)}
def forecast_em(kernels):
    q=protocol();scale=12000/q['kernel_shape']['N']
    return 4*(kernels['slogdet4032']+kernels['covariance_Woodbury']+kernels['beta_contraction']+
              scale*sum(kernels[n]for n in('dense_likelihood_GEMM','latent_and_old_loading','latent_moment','loading_contraction','residual_update')))
def select_threads(records):
    candidates=protocol()['thread_candidates'];summaries=[]
    for t in candidates:
        rows=[r for r in records if r['threads']==t];assert len(rows)==7
        by_kernel={n:stats(r['kernels'][n]['wall_seconds']for r in rows)for n in rows[0]['kernels']}
        med={n:v['median']for n,v in by_kernel.items()}
        pure_kmeans=stats(r['native_KMeans']['native_KMeans_with_real_conversion_wall_seconds']for r in rows)
        summaries.append({'threads':t,'kernels':by_kernel,'predicted_full_EM_seconds':forecast_em(med),
            'medium_native_complete_fit_seconds':med['medium_complete_native_EM'],
            'native_KMeans_seconds':pure_kmeans['median'],'native_KMeans_timing':pure_kmeans,
            'native_full_initialization_seconds':med['native_KMeans_initialization'],
            'peak_RSS_bytes':max(r['resources']['peak_RSS_bytes']for r in rows),
            'peak_private_commit_bytes':max(r['resources']['peak_private_commit_bytes']for r in rows),
            'all_numeric_parity_PASS':all(all(v['PASS']for v in r['kernels'].values())for r in rows)})
    assert all(v['all_numeric_parity_PASS']for v in summaries),'NO_SELECTING_SURVIVING_CONFIGURATIONS'
    minforecast=min(v['predicted_full_EM_seconds']for v in summaries);minmedium=min(v['medium_native_complete_fit_seconds']for v in summaries)
    for v in summaries:v['composite_score']=.8*v['predicted_full_EM_seconds']/minforecast+.2*v['medium_native_complete_fit_seconds']/minmedium
    best=min(v['composite_score']for v in summaries);near=[v for v in summaries if v['composite_score']<=best*1.05]
    for v in summaries:
        v['fit_variation_CV']=v['kernels']['medium_complete_native_EM']['coefficient_of_variation']
    allowance=protocol()['near_tie_stability_CV_allowance']
    min_cv=min(v['fit_variation_CV']for v in near)
    stable_near=[v for v in near if v['fit_variation_CV']<=min_cv+allowance]
    recommended=min(stable_near,key=lambda v:(v['threads'],v['peak_private_commit_bytes'],v['fit_variation_CV']))
    mininit=min(v['native_KMeans_seconds']for v in summaries)
    init_near=[v for v in summaries if v['native_KMeans_seconds']<=mininit*1.05]
    init_cv=min(v['native_KMeans_timing']['coefficient_of_variation']for v in init_near)
    omp=min(v['threads']for v in init_near if v['native_KMeans_timing']['coefficient_of_variation']<=init_cv+allowance)
    baseline=next(v for v in summaries if v['threads']==1)
    for v in summaries:
        v['projected_EM_speedup']=baseline['predicted_full_EM_seconds']/v['predicted_full_EM_seconds']
        v['projected_parallel_efficiency']=v['projected_EM_speedup']/v['threads']
        for name,item in v['kernels'].items():
            item['measured_speedup']=baseline['kernels'][name]['median']/item['median']
            item['parallel_efficiency']=item['measured_speedup']/v['threads']
    return {'PASS':True,'RECOMMENDED_BLAS_THREADS':recommended['threads'],'RECOMMENDED_OPENMP_THREADS':omp,
        'near_tie_relative':.05,'near_tie_BLAS_candidates':[v['threads']for v in near],'summaries':summaries,
        'stable_order':'cyclic balanced seven-block design','recommendation_status':'PROPOSAL_NOT_APPLIED_TO_FORMAL_CONFIG',
        'stability':{'variation_reported':True,'absolute_CV_allowance':allowance,'stable_near_BLAS_candidates':[v['threads']for v in stable_near],
                    'rule':'within 5% latency tie, reject CV more than 0.05 above smallest tied CV, then prefer fewer threads',
                    'stability_conflicts_require_review':True},
        'forecast_label':'EXTRAPOLATED_OPERATION_COMPOSITION','no_single_GEMM_selection':True,**scientific_counters()}

def inference_cost(records,recommended_blas):
    chosen=[r for r in records if r['blas']==recommended_blas];byK={}
    for K in(1,2,4):
        timings=[x['batch_six_including_setup_seconds']for r in chosen if r['K']==K for x in r['report']['rows']]
        assert len(timings)==6
        v=stats(timings);byK[str(K)]={'batch_six_seconds':v,'single_candidate_validation_seconds':600*v['median'],
            'future_210000_frames_seconds':35000*v['median'],'estimate_label':'EXTRAPOLATED_FROM_SYNTHETIC_PARAMETERS',
            'setup_included_once_per_six_frame_native_batch':True,'formal_data_dependent_latency':'UNKNOWN'}
    return {'status':'EXTRAPOLATED','by_K':byK,'validation_single_candidate_frames':3600,
        'validation_nine_candidates_frames':32400,'nine_candidate_validation_central_seconds':1800*sum(byK[str(K)]['batch_six_seconds']['median']for K in(1,2,4)),
        'nine_candidate_validation_optimistic_seconds':1800*sum(byK[str(K)]['batch_six_seconds']['minimum']for K in(1,2,4)),
        'nine_candidate_validation_conservative_scenario_seconds':1800*1.5*sum(byK[str(K)]['batch_six_seconds']['maximum']for K in(1,2,4)),
        'rank_inference_runtime_effect_not_directly_measured':True,'future_formal_frames':210000,
        'two_noise_states_tested_full_seven_SNR_inference_cost_not_measured':True}

def training_budget(smokeA,smokeB,recommendation,records):
    t=recommendation['RECOMMENDED_BLAS_THREADS'];chosen=[r for r in records if r['threads']==t]
    counts=[r['kernels'][name]['native_iterations']for r in chosen for name in('small_complete_native_EM','medium_complete_native_EM')]
    central_iterations=max(7,min(100,int(round(s.median(counts)))))
    summary=next(v for v in recommendation['summaries']if v['threads']==t)
    med={n:v['median']for n,v in summary['kernels'].items()}
    quadratic_weight=med['latent_moment']/sum(med[n]for n in('latent_moment','loading_contraction','latent_and_old_loading','residual_update'))
    def for_case(smoke,K,r):
        steps=smoke['EM_iteration_seconds'];events=smoke['phase_events']
        dense=sum(e['wall_seconds']for e in events if e['kind']=='return'and e.get('function')in('_calc_probs','_update_covs')and e.get('full_phase','').startswith('EM_update'))/2
        sensitive=s.mean(steps)-dense
        assert sensitive>=0,'TRACE_COST_COMPONENT_ACCOUNTING_INVALID'
        scale_r=(1-quadratic_weight)*r/16+quadratic_weight*(r/16)**2
        iteration=(K/4)*(dense+sensitive*scale_r)
        init=smoke['initialization_seconds']*K/4;final=smoke['finalization_seconds']*K/4
        km_iterations=smoke['native_KMeans']['n_iter'];km_upper=max(1,300/km_iterations)
        spread=max(steps)/s.mean(steps)
        return {'K':K,'rank':r,'N':smoke['N'],'label':'EXTRAPOLATED_SYNTHETIC_SCENARIOS',
            'optimistic_seconds':init+7*iteration+final,'central_seconds':init+central_iterations*iteration+final,
            'conservative_scenario_seconds':init*km_upper+100*iteration*spread*1.5+final*1.5,
            'scenario_iterations':{'optimistic':7,'central':central_iterations,'conservative':100},
            'formal_iteration_count':'UNKNOWN','late_iteration_cost':'UNKNOWN',
            'init_K_scaling':'approximate, not a measured KMeans law','conservative_is_not_a_proven_upper_bound':True}
    grid=[for_case(smokeA,K,r)for K in(1,2,4)for r in(4,8,16)]
    final=[for_case(smokeB,K,r)for K in(1,2,4)for r in(4,8,16)]
    return {'MEASURED':{'N12000':{n:smokeA[n]for n in('initialization_seconds','EM_iteration_seconds','finalization_seconds')},
        'N15000':{n:smokeB[n]for n in('initialization_seconds','EM_iteration_seconds','finalization_seconds')}},
        'EXTRAPOLATED':{'nine_candidate_training':grid,'nine_candidate_total_seconds':{n:sum(v[n]for v in grid)for n in('optimistic_seconds','central_seconds','conservative_scenario_seconds')},
                        'final_retraining_by_possible_selected_structure':final},
        'UNKNOWN':['true prepared-data KMeans iterations','formal convergence iterations','late EM cost and degeneracy','selected K/rank'],
        'historical_per_job_seconds_limit_unchanged':86400,
        'any_conservative_job_exceeds_24h':any(v['conservative_scenario_seconds']>86400 for v in grid+final),
        'no_simple_short_step_times_100_claim':True,'central_iteration_source':'observed small/medium synthetic native complete fits',
        'late_phase_allowance_factor':1.5,'quadratic_rank_weight_from_measured_kernel_mix':quadratic_weight,
        'formal_24h_safety_certified':False}
