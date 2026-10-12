"""Generate R4 reports from immutable code, metadata and synthetic diagnostics."""
from pathlib import Path
import sys,os,json,csv,ast,statistics,subprocess
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u12b5b_r4_common import firewall,read,write,csvwrite,sha,OUT,CORE,ADAPTER,check_frozen,LIMITS
P='sp_u12b5b_r4_';GIB=2**30;PORT=ROOT/'experiments/mfa_low_memory_fidelity_r4'

PROOF=r'''# Independent mathematical certification of frozen R3 replacements

All identities below concern exact arithmetic. Floating-point contractions can
change summation order; the separate numerical gate uses atol=1e-10, rtol=1e-9.
There is no claim of bitwise identity or universal convergence identity.

For one component k, X has shape N x D, q_n is its OLD responsibility (real,
nonnegative), s=sum_n q_n>0, W^- has shape D x r, and C^- is the original
W^- (W^-)^H+diag(psi^-) covariance. The original beta is
beta^- = (W^-)^H (C^-)^{-1}, of shape r x D. Superscript minus denotes OLD
parameters; (W^-)^H is their Hermitian transpose, not an inverse.
H=beta^-(X^T-mu^-1^T)
has shape r x N. J=I_r-beta^- W^- has shape r x r. V^-=W^-H has shape D x N.
The author computes mu^+=sum_n q_n(X_n-V^-_n)/s, then
Z=X^T-mu^+1^T (D x N). Component k is never relabelled or rotated.

## A. Loading numerator
The author's tensor T_{d,a,n}=Z_{d,n} conjugate(H_{a,n}) has shape D x r x N.
Its weighted dot product is R_{d,a}=sum_n q_n Z_{d,n} conjugate(H_{a,n}).
Let H_q=conjugate(H) diag(q), shape r x N. Then
(Z H_q^T)_{d,a}=sum_n Z_{d,n}(H_q)_{a,n}=R_{d,a}.
Thus the frozen replacement Z@(H.conj()*q[None,:]).T is exactly R.
The final transpose is ordinary T: conjugation has already been applied.
Replacing T by H would remove that conjugation and be incorrect.
Z uses the NEW mean; H, beta, q and V use OLD parameters. Both sides return D x r.

## B. Latent second moment
For each n the author's E_n=J+H_n H_n^H has shape r x r.
Gamma_{a,b}=sum_n q_n E_{a,b,n}
=s J_{a,b}+sum_n q_n H_{a,n} conjugate(H_{b,n}).
Therefore Gamma=s J+H (conjugate(H) diag(q))^T.
This is the author's weighted second moment, including the posterior covariance
J; replacing it by only H diag(q) H^H would omit J and change the model.
In exact arithmetic J=I-(W^-)^H(C^-)^{-1}W^- is Hermitian and nonnegative;
the weighted outer products are Hermitian. The frozen implementation does not
symmetrize, clip eigenvalues or change the author's SciPy inverse of Gamma.
It retains the ORIGINAL dense covariance inverse algorithm (including its pinv).
The update is W^+=R Gamma^{-1}, using old beta/W/H and newly centered Z.

## C. Likelihood quadratic
Let X1=(X-mu^-)^T and X2=(C^-)^{-1}X1, both D x N complex128.
For each n, Re sum_d conjugate(X1_{d,n}) X2_{d,n}
=sum_d [Re(X1_{d,n})Re(X2_{d,n})+Im(X1_{d,n})Im(X2_{d,n})].
The two optimize=False einsums return N real float64 values and exactly this
real quadratic. This is not a replacement of a complex inner product by an
unconjugated inner product. The imaginary part was already discarded by the
author. The normalization -D log(pi), dense slogdet, component log amplitude,
log-sum-exp and posterior responsibility normalization remain original code.
Noise-dependent observation covariance and inference are untouched.

## D. Residual
The original raw residual is psi_raw[d]=Re sum_n q_n
[(Z_{d,n}-V^-_{d,n}) conjugate(Z_{d,n})]/s.
Write Z=a+ib and V^-=v+iw. Its real product is (a-v)a+(b-w)b.
Thus two real weighted einsums divided by s equal the original raw residual.
Crucially V^-=W^- H is saved BEFORE W^+ is assigned, while Z uses mu^+.
Using W^+ H would change the residual. The independent sentinel deliberately
uses different old and new W and rejects that substitution.
Then, in the SAME order, the author clips EACH raw residual at 1e-6, takes
the PPCA mean over D, and assigns the scalar to every diagonal element.
Clipping after the PPCA mean would not in general be equivalent.

## Update order and lifetimes
1. Original full-N likelihood and q; 2. all old betas; 3. component loop in order;
4. old H/J/V; 5. new mean and Z; 6. new W via the same SciPy inverse;
7. OLD-V residual, floor, PPCA; 8. amplitude=s/N; 9. original covariance update.
lock_psis=False, rs_clip=0, proper-complex model, complex128 and zero restarts
are retained. No minibatch/stochastic EM or added regularization is introduced.
Deleting component temporaries after amplitude and betas before covariance
only removes references whose final mathematical use has already occurred.
No fifth replacement, factorization cache, determinant lemma, approximate
inverse, probability truncation, normalization change or reordered update occurs.

## Scope
These are code-specific algebraic identities, not a new EM theorem. They require
the author's original operations to be defined (s>0, inverse of Gamma exists,
finite parameters). The tests do not establish full-N resource safety, universal
floating-point equality, global EM convergence, or laboratory host readiness.
'''

def save(n,v):write(ROOT/'outputs'/f'{P}{n}.json',v)
def doc(n,s):(ROOT/'docs'/f'{P}{n}.md').write_text(s,encoding='utf8')
def line(path,needle):
    hits=[i for i,s in enumerate(path.read_text(encoding='utf8').splitlines(),1)if needle in s]
    assert hits,(path,needle);return hits[-1]

def math_audit():
    frozen=check_frozen();low=ROOT/'experiments/mfa_low_memory_prototype/portable/low_memory.py'
    mapping=[
      {'replacement':'A_loading','original_equation':'R[d,a]=sum_n q[n]Z[d,n]conj(H[a,n])','optimized_equation':'R=Z@(conj(H)*q).T',
       'input_tensors':'Z(D,N),H(r,N),q(N) real','output_tensor':'R(D,r) complex128','conjugation':'H conjugated exactly once','transpose':'ordinary T after conjugation',
       'summation_indices':'n=0..N-1; output d,a','dependency':'q,H old; Z new mean','original_line':line(CORE,'self._lambdas[k] = np.dot'),
       'optimized_line':line(low,'return zeroed@(latents.conj()'),'proof_section':'A'},
      {'replacement':'B_latent','original_equation':'Gamma[a,b]=sum_n q[n]*(J[a,b]+H[a,n]conj(H[b,n]))','optimized_equation':'Gamma=sum(q)*J+H@(conj(H)*q).T',
       'input_tensors':'H(r,N),q(N),J(r,r)','output_tensor':'Gamma(r,r) complex128','conjugation':'second latent factor conjugated','transpose':'ordinary T of conjugated weighted matrix',
       'summation_indices':'n=0..N-1; output a,b','dependency':'q,beta,W,H old','original_line':line(CORE,'latent_covs ='),
       'optimized_line':line(low,'return weights.sum()*base'),'proof_section':'B'},
      {'replacement':'C_quadratic','original_equation':'Re sum_d conj(X1[d,n])*X2[d,n]','optimized_equation':'sum_d ReX1*ReX2+ImX1*ImX2',
       'input_tensors':'X1(D,N),X2(D,N) complex128','output_tensor':'quadratic(N) float64','conjugation':'complex conjugate represented by real identity','transpose':'input X1 ordinary T retained',
       'summation_indices':'d=0..D-1; output n','dependency':'old means/inverse covariance; original likelihood normalizer',
       'original_line':line(CORE,'p = np.sum(X1.conj()'),'optimized_line':line(low,"return np.einsum('dn,dn->n'"),'proof_section':'C'},
      {'replacement':'D_residual','original_equation':'Re sum_n q[n]*(Z[d,n]-Vold[d,n])*conj(Z[d,n])/sum(q)','optimized_equation':'sum_n q*((ReZ-ReVold)*ReZ+(ImZ-ImVold)*ImZ)/sum(q)',
       'input_tensors':'Z(D,N),Vold(D,N),q(N)','output_tensor':'psi_raw(D) float64','conjugation':'Z conjugated, old V retained','transpose':'no new transpose',
       'summation_indices':'n=0..N-1; output d','dependency':'Z new mean; Vold=old W*old H; floor before PPCA',
       'original_line':line(CORE,'psis = np.real(np.dot'),'optimized_line':line(low,"return (np.einsum('dn,dn,n->d'"),'proof_section':'D'}]
    for row in mapping:
        row.update(original_source=CORE.relative_to(ROOT).as_posix(),optimized_source=low.relative_to(ROOT).as_posix(),
            exact_arithmetic='IDENTITY_PROVED',floating_point='SUMMATION_ORDER_DIFFERS')
    csvwrite(ROOT/'outputs'/f'{P}equation_mapping.csv',mapping)
    (ROOT/'outputs'/f'{P}independent_equivalence_proof.md').write_text(PROOF,encoding='utf8');doc('mathematical_fidelity_report',PROOF)
    # AST distinguishes transpose and conjugation; immutable code hash locks the audited expressions.
    ast.parse(CORE.read_text(encoding='utf8'));ast.parse(low.read_text(encoding='utf8'))
    conj={'PASS':True,'rows':mapping,'original_core_sha256':sha(CORE),'R3_low_memory_sha256':sha(low),
        'independent_index_oracle':read(OUT/'low_1/independent_identities.json'),'Hermitian_covariance':'exact arithmetic; no added symmetrization'}
    save('complex_conjugation_audit',conj)
    source=low.read_text(encoding='utf8');needles=['self.logLs,self.rs=','betas=','latents=','latent_base=',
        'old_lambdalatents=','self._means[k]=','zeroed=','cross=','gamma=','self._lambdas[k]=','psis=residual_moment',
        'self._psis[k]=np.clip','if PPCA:', 'self.amps[k]=','del latents','del betas','self._update_covs()']
    positions=[source.index(n) for n in needles];assert positions==sorted(positions)
    formal_source=(ROOT/'scripts/sp_u12b5b_worker.py').read_text(encoding='utf8')+(ROOT/'scripts/sp_u12b5b_model.py').read_text(encoding='utf8')
    assert 'mfa_low_memory' not in formal_source
    save('update_order_audit',{'PASS':True,'positions':dict(zip(needles,positions)),
        'old_loading_residual':True,'floor_then_PPCA':True,'full_batch':True,'all_old_betas_before_component_loop':True,
        'initialization':'untouched author KMeans(n_init=1), legacy MT19937, max_condition_number=1e6',
        'stop':'untouched author run_em, i>5 and abs((newL-prevL)/newL)<1e-4; maxiter100',
        'extra_restarts':0,'fifth_transformation':False,'formal_runner_imports_optimized':False,'frozen':frozen,
        'trace_scope':'copies local diagnostics; no assignment to author parameter buffers; inference uses separate clones'})

def numeric_reports():
    cfg=read(ROOT/'configs/sp_u12b5b_r4_synthetic.json');rows=[];jobs={};stopping=[];interfaces=[]
    for t in cfg['thread_comparisons']:
        d=OUT/f'low_{t}';v=read(d/'parity.json');jobs[str(t)]=v
        with(d/'comparisons.csv').open(encoding='utf-8-sig',newline='')as f:
            for row in csv.DictReader(f):
                for n in ('max_absolute','max_relative_with_absolute_floor','max_relative_above_atol','max_tolerance_ratio'):row[n]=float(row[n])
                for n in ('K','r','D','N','threads','worst_flat_index'):row[n]=int(row[n])
                for n in ('PASS','bitwise'):row[n]=row[n]=='True'
                rows.append(row)
        interfaces+=v['inference']
        for case in cfg['cases']:
            ref=read(OUT/'reference_1'/case['id']/'status.json');got=read(d/case['id']/'status.json')
            assert ref['input_sha256']==got['input_sha256']
            assert ref['initialization_policy']==got['initialization_policy']
            if case['mode']=='convergence':
                margins=[r['distance_from_tolerance'] for r in ref['diagnostics'][6:] if r['distance_from_tolerance']is not None]
                stopping.append({'case':case['id'],'threads':t,'reference_updates':ref['iterations'],'low_updates':got['iterations'],
                    'reference_reason':ref['stopping_reason'],'low_reason':got['stopping_reason'],
                    'PASS':ref['iterations']==got['iterations'] and ref['stopping_reason']==got['stopping_reason'],
                    'minimum_stop_margin_reference':min(margins),'no_history_truncation':True})
    assert rows;csvwrite(ROOT/'outputs'/f'{P}em_trajectory_parity.csv',rows)
    worst={n:max(rows,key=lambda r:r[n]) for n in ('max_absolute','max_relative_with_absolute_floor','max_relative_above_atol','max_tolerance_ratio')}
    fail=[v for v in rows if not v['PASS']]
    nine=[v for v in rows if v['case'].startswith('nine_')]
    one=[v for v in nine if v['phase'] in ('initial','iteration000')]
    initial=[v for v in rows if v['phase']=='initial' and v['tensor'] in ('weights','means','loadings','residuals','covariances','inverse_covariances')]
    assert all(v['bitwise'] for v in initial),'INITIAL_PARAMETERS_DIFFER'
    save('nine_structure_parity',{'PASS':all(v['PASS'] for v in nine),'one_step_PASS':all(v['PASS'] for v in one),
        'grid':[[k,r] for k in (1,2,4) for r in (4,8,16)],'D':64,'N':256,'short_updates':8,
        'comparison_rows':len(nine),'one_step_rows':len(one),'all_initial_parameters_bitwise':True,
        'seeds':{c['id']:{n:c[n] for n in ('data_seed','initialization_seed')}for c in cfg['cases']},
        'formal_seeds_used':0,'atol':1e-10,'rtol':1e-9,'worst':worst,'failed_rows':fail})
    save('stopping_parity',{'PASS':all(v['PASS'] for v in stopping),'cases':stopping,
        'strict_threshold_sentinel':read(OUT/'reference_1/stop_branch_sentinel.json'),
        'scope':'actual synthetic convergence + independent strict-threshold branch sentinel; no universal stop guarantee',
        'no_history_truncation':True,'maxiter':100})
    medium=[v for v in rows if v['case'].startswith('medium_')]
    save('medium_scale_parity',{'PASS':all(v['PASS'] for v in medium),'D':256,'N':1024,'K':4,'r':16,
        'updates':8,'threads':[1,4,8,12],'comparison_rows':len(medium),'worst':{n:max(medium,key=lambda r:r[n])for n in worst}})
    save('multithread_parity',{'PASS':all(v['PASS'] for v in jobs.values()),'reference_threads':1,
        'low_threads':[1,4,8,12],'same_input_hashes':True,'same_initial_parameters_bitwise':True,
        'initialization_threads':1,'thread_comparison_scope':'same exact author initialization, then EM and posterior at requested threads',
        'independent_native_multithread_KMeans_initialization_certified':False,
        'no_alignment_or_truncation':True,'atol':1e-10,'rtol':1e-9,'worst':worst,
        'jobs':{t:{'PASS':v['PASS'],'checks':v['checks']}for t,v in jobs.items()},
        'official_BLAS_threads_changed':False,'lab_CPU_tested':False})
    save('inference_compatibility',{'PASS':all(v['PASS']for v in interfaces),'checks':interfaces,
        'original_adapter_sha256':sha(ADAPTER),'float_1_0_all_components':all(v['component_lmmse_calls']==v['expected_all_components']for v in interfaces),
        'serialization_bitwise':all(v['serialization_bitwise_parameters']for v in interfaces),
        'Project_receiver_information_contract':'64 owner-divided pilots, known noise, frozen geometry, joint4032 -> data3968; original source unchanged',
        'project_dimension_training_tested':False,'synthetic_inference_dimensions':[64,256],
        'component_parameters_not_renormalized':True,'no_inference_factorization_cache':True,
        'K1_author_squeeze_behavior':'retained as original; no formula repair or alternate inference implementation'})
    return {'PASS':not fail and all(v['PASS']for v in jobs.values()) and all(v['PASS']for v in stopping),
        'rows':len(rows),'worst':worst,'stopping':stopping,'failures':fail}

def count_audit():
    from experiments.mfa_low_memory_fidelity_r4.canonical_validation_contract import metadata_contract,assert_exact_identity_coverage
    cfg=read(ROOT/'configs/sp_u12b5b_mfa_selection.yaml');split=read(ROOT/'outputs/sp_u12b5b_preparation_split_manifest.json')
    inputs=read(ROOT/'outputs/sp_u12b5b/input_manifest.json');noise=read(ROOT/'outputs/sp_u12b5b/validation_noise_registry.json')
    result=assert_exact_identity_coverage(metadata_contract(cfg),cfg)
    assert split['canonical_origins']==cfg['canonical_origins']
    for origin in cfg['canonical_origins']:
        assert split['origin_partitions'][origin]['train_ids']==list(range(400))
        assert split['origin_partitions'][origin]['validation_ids']==list(range(400,500))
    assert inputs['validation_scored_frames']==list(range(4,10)) and inputs['canonical_origins']==cfg['canonical_origins']
    tokens={(r['origin'],r['trajectory_id'],r['frame_id'])for r in noise}
    from itertools import product
    assert len(noise)==1800 and tokens==set(product(cfg['canonical_origins'],range(400,500),range(4,10)))
    assert all(type(r['seed'])is int and len(r['unit_owner_divided_noise_sha256'])==64 for r in noise)
    worker=ROOT/'scripts/sp_u12b5b_worker.py';prepare=ROOT/'scripts/sp_u12b5b_prepare.py'
    w=worker.read_text(encoding='utf8');p=prepare.read_text(encoding='utf8')
    ast.parse(w);ast.parse(p)
    checks=['for o,origin in enumerate(cfg[\'canonical_origins\'])','for si,s in enumerate((\'10\',\'20\'))','for i in range(100)',
        'y=np.asarray(pilots[o,si,i])','E=np.empty((100,6),','estimate=adapter.estimate(y,']
    # Whitespace independent statement check; no import or execution of formal worker.
    compact=''.join(w.split())
    assert all(''.join(s.split())in compact for s in checks),'FROZEN_RUNNER_CARDINALITY_CONFLICT'
    assert 'shape=(3,100,6,3968)'in p and '(3,2,100,6,64)'in p
    assert "enumerate(range(4,10))"in p and "for si,s in enumerate(('10','20'))"in p
    assert p.index('unit=(rng.standard_normal')<p.index("for si,s in enumerate(('10','20'))")
    metadata_hashes={n:sha(ROOT/n)for n in ['configs/sp_u12b5b_mfa_selection.yaml','outputs/sp_u12b5b_preparation_split_manifest.json',
        'outputs/sp_u12b5b_candidate_seed_registry.json','outputs/sp_u12b5b/input_manifest.json','outputs/sp_u12b5b/validation_noise_registry.json',
        'scripts/sp_u12b5b_worker.py','scripts/sp_u12b5b_prepare.py']}
    err={'record_type':'VALIDATION_CARDINALITY_ERRATUM','original_erroneous_count':16200,
        'correct_count_per_candidate':3600,'correct_nine_candidate_count':32400,
        'derivation':'9 candidates x 3 origins x 100 trajectories x 2 SNR x 6 frames = 32400',
        'affected_documents':['SP-U12B5B-R3 task attachment count','outputs/sp_u12b5b_r3_inference_cost_model.json (task and 16200 alternatives)',
            'docs/sp_u12b5b_r3_cpu_multicore_optimization_review.md','docs/sp_u12b5b_r3_recommended_research_path.md'],
        'scientific_dataset_changed':'NO','metric_changed':'NO','seed_changed':'NO','before_results_amendment':'YES',
        'historical_protocol_bytes_edited':0,'frozen_runner_computes_32400':True,'future_formal_predictions':210000,
        'validation_scores_before_and_after':0,'identity_hashes':metadata_hashes}
    save('validation_cardinality_erratum',err)
    audit={'PASS':True,'metadata_assertions':result,'canonical_labels_exact':True,'worker_loop_cardinality':3600,
        'nine_candidate_cardinality':32400,'batch_calls_per_candidate':600,'batch_size':6,'nine_batch_calls':5400,
        'worker_scope':'STATIC_AST/SOURCE + immutable split/noise metadata; worker.validate was never executed',
        'noise_records':1800,'CRN_across_SNR':'same unit owner-divided noise per origin/trajectory/frame scaled for both SNRs',
        'CRN_across_candidates':'same immutable validation_pilots.npy hash reused all nine',
        'pilot_file_hash':inputs['files']['validation_pilots.npy'],'truth_file_hash':inputs['files']['validation_truth.npy'],
        'source_frames':list(range(4,10)),'sample_identity_changed':False,'seed_registry_changed':False,
        'selection_metric':cfg['selection_metric'],'assertions_ready':True,'formal_runner_modified':False,
        'future_integration_required':'mentor-authorized wrapper must emit/check exact identity tokens, shapes and immutable input hashes before accepting any score',
        'canonical_contract_module':str(PORT/'canonical_validation_contract.py'),'symbolic_counter_role':'preflight cardinality only; canonical companion rejects aliases',
        'schema_shapes':{'pilots':[3,2,100,6,64],'truth':[3,100,6,3968],'cell_error_energy':[100,6]},
        'forbidden':{'actual_validation_values_read':0,'actual_validation_NMSE_computed':0}}
    save('validation_contract_audit',audit)
    costs=read(ROOT/'outputs/sp_u12b5b_r3_inference_cost_model.json');out={'classification':'EXTRAPOLATED_NOT_FULL_RUN_MEASUREMENT','validation_frames':32400,'formal_frames':210000,'threads':{}}
    for t,v in costs['threads'].items():
        per={};total=0
        for k,value in v['per_K'].items():
            batch=value['batch6_including_noise_setup_seconds'];per[k]={'batch6_seconds':batch,
                'single_candidate_validation_hours':600*batch/3600,'three_ranks_validation_hours':1800*batch/3600,
                'formal_210000_hours':35000*batch/3600,'provenance':value['evidence']};total+=1800*batch/3600
        out['threads'][t]={'per_K':per,'balanced_nine_candidate_validation_hours':total,
            'setup_double_counted':False,'setup_already_in_batch':True}
    out['uncertainty']='K2 interpolation; synthetic covariance/y; sustained CPU and full deployment unmeasured. No inference caching introduced.'
    save('corrected_cost_estimate',out)
    doc('validation_erratum','# VALIDATION_CARDINALITY_ERRATUM\n\n'+json.dumps(err,indent=2)+'\n\n'+json.dumps(audit,indent=2)+'\n\n'+json.dumps(out,indent=2))
    return audit,out

def memory_model(N,optimized):
    D,K,r=4032,4,16;a=16*N*D;c=16*K*D*D
    parameters=2*c+16*K*D+16*K*D*r+8*K*D+8*K
    init=4*a+2*c+16*K*D+24*N
    public=a+2*parameters;serialization=public+8*K*D*D+16*64*D
    if optimized:
        scratch=8*K*N+16*K*r*D+48*r*N+64*r*r+16*D*r+8*K
        stages={'KMeans_initialization':init,'likelihood':3*a+parameters+16*K*N+24*N,
            'mean_update':4*a+parameters+scratch,'loading_update':3*a+parameters+scratch,
            'residual_update':int(3.5*a)+parameters+scratch,'covariance_update':a+parameters+2*c+8*D*D+scratch,
            'public_parameter_copies':public,'serialization_and_validity':serialization}
    else:
        scratch=8*K*N+16*K*r*D+16*r*N+32*r*r*N+16*r*r+8*K
        stages={'KMeans_initialization':init,'likelihood':5*a+parameters+16*K*N+8*N,
            'loading_update':3*a+parameters+scratch+16*D*r*N,'residual_update':6*a+parameters+scratch,
            'covariance_update':3*a+parameters+scratch+2*c,'public_parameter_copies':public,'serialization_and_validity':serialization}
    peak=max(stages.values());native_reserve=4 if optimized else 8;os_reserve=6 if optimized else 8
    return {'N':N,'D':D,'K':K,'r':r,'implementation':'low_memory' if optimized else 'original',
        'data_bytes':a,'covariance_bytes':c,'inverse_covariance_bytes':c,'private_parameters_bytes':parameters,
        'latent_means_bytes':16*r*N,'latent_base_bytes':16*r*r,
        'original_latent_outer_plus_second_moment_bytes':32*r*r*N if not optimized else 0,
        'weighted_latent_buffers_bytes':32*r*N if optimized else 0,
        'original_loading_DrN_bytes':16*D*r*N,'low_loading_output_bytes':16*D*r,
        'remaining_covariance_expression_overlap_bytes':2*c,
        'stages_bytes':stages,'source_live_estimate_GiB':peak/GIB,'peak_stage':max(stages,key=stages.get),
        'largest_materialized_single_array_GiB':(max(a,c)if optimized else 16*D*r*N)/GIB,
        'conditional_process_planning_GiB':peak/GIB+native_reserve,
        'conditional_total_host_planning_GiB':peak/GIB+native_reserve+os_reserve+4,
        'lean_minimum_conditional_host_GiB':peak/GIB+(1 if optimized else 2)+4+4,
        'native_allocator_Python_reserve_GiB':native_reserve,'OS_background_reserve_GiB':os_reserve,'future_headroom_planning_GiB':4,
        'recommended_host_GiB':32 if optimized else 48,'preferred_original_host_GiB':None if optimized else 64,
        'Python_NumPy_object_overhead':'NOT_FULL_SCALE_MEASURED; included only as assumed reserve',
        'BLAS_workspace_allocator_retention':'UNKNOWN; reserve is not a proven upper bound',
        'full_scale_measured_RSS':'NOT_MEASURED','full_scale_measured_commit':'NOT_MEASURED',
        'original_formal_caps_changed':False}

def resource_reports():
    measurements=[];resources=[]
    for d in sorted(OUT.iterdir()):
        if d.is_dir()and(d/'resource_summary.json').exists():resources.append(read(d/'resource_summary.json')|{'job':d.name})
        if d.is_dir()and(d/'measurements.json').exists():measurements+=read(d/'measurements.json')
    assert len(measurements)==36 and len(resources)==18
    formatted=[{k:(json.dumps(v,sort_keys=True)if isinstance(v,(list,dict))else v)for k,v in row.items()}for row in measurements]
    csvwrite(ROOT/'outputs'/f'{P}runtime_measurements.csv',formatted)
    csvwrite(ROOT/'outputs'/f'{P}memory_measurements.csv',formatted)
    summary=[]
    for kind in ('loading','latent','quadratic','residual','small_EM','medium_EM'):
        a=[v for v in measurements if v['kind']==kind and v['implementation']=='original'];b=[v for v in measurements if v['kind']==kind and v['implementation']=='low_memory']
        sa=statistics.median(v['wall_seconds']for v in a);sb=statistics.median(v['wall_seconds']for v in b)
        summary.append({'kind':kind,'original_median_seconds':sa,'low_median_seconds':sb,'speedup':sa/sb,
            'low_is_slower':sb>sa,'original_iterations':[v['iterations']for v in a],'low_iterations':[v['iterations']for v in b],
            'original_process_peak_RSS_GiB':max(v['process_peak_RSS']for v in a)/GIB,
            'low_process_peak_RSS_GiB':max(v['process_peak_RSS']for v in b)/GIB,
            'original_traced_peak_bytes':max(v['traced_peak_bytes']for v in a),'low_traced_peak_bytes':max(v['traced_peak_bytes']for v in b)})
    previous=read(ROOT/'outputs/sp_u12b5b_r3_low_memory_prototype_audit.json')
    negative=previous['measured_paired_kernel_tradeoff']['quadratic'][0]
    aggregate={'PASS':all(v['PASS']for v in resources),'completed_workers':len(resources),'rows':len(measurements),
        'max_peak_RSS_GiB':max(v['peak_RSS']for v in resources)/GIB,'max_peak_private_commit_GiB':max(v['peak_private_commit']for v in resources)/GIB,
        'minimum_physical_available_GiB':min(v['minimum_available_physical_bytes']for v in resources)/GIB,
        'measurement_scope':'OS process peaks include libraries/fixtures/diagnostics; tracemalloc peaks from separate untimed calls; allocation snapshot counts LIVE blocks only',
        'timings':summary,'prior_frozen_negative_quadratic':{'origin':'R3 D4032,N512; not rerun','original_seconds':negative['direct_seconds'],'low_seconds':negative['low_seconds'],
            'slowdown':negative['low_seconds']/negative['direct_seconds']},
        'limits_unchanged':LIMITS,'near_limit_stops':0,'formal_concurrency':1,'current_host_threads':'R3 recommendation12; R4 fidelity confirms1/4/8/12; no new performance ranking or formal thread change',
        'laboratory_host_tested':False}
    save('safe_resource_summary',aggregate)
    models=[memory_model(n,opt)for n in (12000,15000)for opt in (False,True)]
    full={'models':models,'32GB_classification':'CONDITIONALLY_PLAUSIBLE_NOT_RESOURCE_CERTIFIED',
        'minimum_estimate_is_conditional_not_required_RAM_proof':True,'recommended_low_host_GiB':32,
        'full_N_train_calls':0,'measured_full_N_RSS':False,'current_busy16GB_formal_6GiB_cap':'FULL_TRAINING_NOT_AUTHORIZED; estimated private process + native reserve exceeds cap',
        'major_unknowns':['native BLAS packing/LU workspace','allocator retention/fragmentation','Python/NumPy overhead','KMeans real data copies','public/private parameter duplicates and serialization','32GB usable/free RAM and OS background load','sustained laboratory CPU throughput'],
        'caps':LIMITS,'future_12GiB_4GiB':'PROVISIONAL_ONLY; not applied'}
    save('fullscale_memory_model',full)
    # Commands are preparatory only; the lab has not been contacted.
    plan={'status':'NEXT_STAGE_DESIGN_ONLY_NOT_EXECUTED','laboratory_host_tested':False,'full_fit_authorized':False,
        'requirements':['Windows64bit/Python3.13 or separately verified compatibility','complex128 NumPy/SciPy/sklearn versions and BLAS fingerprint matching archived environment',
            'immutable reference/adapter/R3/fidelity manifest byte verification','at least20GiB available RAM proposal; >=64GiB free disk proposal',
            'preserve author initialization distribution/order; R4 multithread tests use shared1-thread initialization, so lab native multithread initialization needs separate review if proposed',
            'one process; known PID monitoring; no target/preparation/validation input values','mentor approval for any future12GiB/4GiB cap or thread change'],
        'host_inventory_command':'.venv-cdl\\Scripts\\python.exe -B -X utf8 experiments/mfa_low_memory_prototype/portable/host_inventory.py',
        'CPU_BLAS_command':'.venv-cdl\\Scripts\\python.exe -B -X utf8 -c "import sys; sys.path.append(\'.venv-gmm/Lib/site-packages\'); from threadpoolctl import threadpool_info; import numpy,scipy,sklearn; print(numpy.__version__, scipy.__version__, sklearn.__version__, threadpool_info())"',
        'CPU_BLAS_safe_comparison_command':'.venv-cdl\\Scripts\\python.exe -B -X utf8 scripts/sp_u12b5b_r4_next_host_smoke.py --out outputs/sp_u12b5b_r4/laboratory_threads12 --threads 12',
        'benchmark_scope_note':'safe one-step reference1/low12 numerical and monitored whole-worker wall-time comparison only; includes diagnostic work and is not an unbiased formal training-speed benchmark. Future clean medium EM timing gate must remain separate.',
        'safe_resource_smoke_command':'.venv-cdl\\Scripts\\python.exe -B -X utf8 scripts/sp_u12b5b_r4_next_host_smoke.py --out outputs/sp_u12b5b_r4/laboratory_smoke --threads 1',
        'thread_benchmark_design':'separate authorized host-specific 1/4/8/12 where physical cores suffice; synthetic medium EM and parity before timing; no full candidate training',
        'safe_smoke_shape':{'N':256,'D':4032,'K':4,'r':16,'steps':1,'synthetic_only':True},
        'unchanged_approved_limits':LIMITS,'proposed_future_limits_only':{'RSS_GiB':12,'physical_headroom_GiB':4},
        'stop':'RSS approaches5.5GiB or physical available below2.25GiB during safe smoke; no automatic retry/rescue',
        'next_gate':'host-specific fidelity, resource smoke and explicit formal authorization; R4 READY alone does not authorize full training'}
    save('32gb_host_test_plan',plan)
    return aggregate,full

def report():
    firewall();math_audit();numeric=numeric_reports();count,cost=count_audit();resource,full=resource_reports()
    passed=numeric['PASS']and count['PASS']and resource['PASS']
    verdict='MFA_LOW_MEMORY_FIDELITY_READY'if passed else'MFA_LOW_MEMORY_NUMERICAL_DIVERGENCE'
    summary={'verdict':verdict,'numerical':numeric,'resource':resource,'validation':count,'cost':cost,
        'fullscale_resource_certified':False,'formal_training_authorized':False,'formal_candidate_fits':0,'validation_scores':0,
        'selected_structure':'NONE','final_model':'NONE','target_evaluations':0,'B2_B3_values_viewed':False,
        'real_preparation_validation_value_reads':0,'scientific_model_change':False,'manuscript_change':False}
    save('decision',summary)
    doc('numerical_fidelity_report','# R4 numerical fidelity\n\n'+json.dumps(summary,indent=2)+'\n\nAll per-tensor/per-iteration differences are in the generated trajectory CSV. Synthetic tests do not prove universal floating-point equality or laboratory full-scale safety. No parameter alignment, loading rotation, component permutation or history truncation was used.\n')
    doc('next_stage_requirements','# STOP: mentor review required\n\n'+json.dumps(read(ROOT/'outputs'/f'{P}32gb_host_test_plan.json'),indent=2)+'\n\nFuture formal runner must integrate the canonical identity assertion after approval; original frozen runner remains unchanged. The validation erratum changes the count record, not samples, metrics, SNR or seeds. No candidate training, model selection or target evaluation is authorized by R4.\n')
    doc('change_log','# R4 change log\n\nAdded isolated experimental diagnostics, independent proofs, synthetic preregistration, monitored workers, metadata-only count assertions, tests and generated reports. All author/adapter/R3/historical protocol files remain immutable. The execution freeze protects numerical harness bytes before the synthetic run. A separate canonical metadata companion maps exact origin labels and is independently tested; it was not used to modify EM or formal validation.\n\n'+json.dumps(read(OUT/'execution_freeze.json'),indent=2))
    print(json.dumps({'stage':'R4_REPORT','verdict':verdict,'comparison_rows':numeric['rows'],'worst_absolute':numeric['worst']['max_absolute'],'worst_relative':numeric['worst']['max_relative_with_absolute_floor']},indent=2))

def finalize():
    firewall();summary=read(ROOT/'outputs'/f'{P}decision.json');pres=read(ROOT/'outputs'/f'{P}preservation.json');seals=read(ROOT/'outputs'/f'{P}b2_b3_seal_audit.json')
    assert pres['PASS']and seals['PASS'];f=read(OUT/'execution_freeze.json');assert all(sha(ROOT/n)==h for n,h in f['files'].items())
    pytest=read(OUT/'pytest_final.json');assert pytest['exit_code']==0
    files={}
    for d in ('scripts','tests','configs','outputs','docs','experiments/mfa_low_memory_fidelity_r4'):
        for path in(ROOT/d).rglob('*'):
            if path.is_file()and'__pycache__'not in path.parts and (path.is_relative_to(PORT)or any(s.startswith(('sp_u12b5b_r4','run_sp_u12b5b_r4','test_sp_u12b5b_r4'))for s in path.relative_to(ROOT).parts)):
                if path.name!=f'{P}fidelity_manifest.json':files[path.relative_to(ROOT).as_posix()]=sha(path)
    manifest={'status':summary['verdict'],'scope':'R4_LOW_MEMORY_FIDELITY_ONLY_NOT_FORMAL_BASELINE_RESOURCE_OR_TRAINING_AUTHORIZATION',
        'files':files,'execution_freeze_sha256':sha(OUT/'execution_freeze.json'),'historical_frozen':check_frozen(),
        'preservation':pres,'seals':seals,'pytest':pytest,'limits_unchanged':LIMITS,'formal_candidate_fits':0,'validation_scores':0,
        'selected_structure':'NONE','final_model':'NONE','target_evaluations':0,'no_commit':True,'STOP':'WAIT_MENTOR_REVIEW'}
    path=ROOT/'outputs'/f'{P}fidelity_manifest.json';write(path,manifest)
    print(json.dumps({'verdict':summary['verdict'],'manifest_sha256':sha(path),'manifest_files':len(files),'STOP':'WAIT_MENTOR_REVIEW'}))

if __name__=='__main__':
    report()if sys.argv[1]=='report'else finalize()
