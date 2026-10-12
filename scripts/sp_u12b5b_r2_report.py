"""Generate the R2 source/timing-backed planning artifacts; no scientific scores."""
from scripts.sp_u12b5b_r2_common import *
from scripts.sp_u12b5b_r2_model import sizes, GIB, N, D, line
import math, shutil, statistics
PFX='sp_u12b5b_r2_'
def emit(name,obj):write(ROOT/'outputs'/(PFX+name+'.json'),obj)
def md(name,text):(ROOT/'docs'/(PFX+name+'.md')).write_text(text.rstrip()+'\n',encoding='utf8')
def duration_model(bench):
 rows=bench['kernels']
 def find(kind,d=None,n=None,r=None):
  return next(v for v in rows if v['case']['kind']==kind and (d is None or v['case']['D']==d) and (n is None or v['case']['N']==n) and (r is None or v['case']['rank']==r))
 def measured(kind,d=None,n=None,r=None):return find(kind,d,n,r)['median_seconds']
 def interp(kind,r):
  if r in (4,16):return measured(kind,r=r)
  # rank8 was not measured for these kernels; affine interpolation is planning only.
  return measured(kind,r=4)+(r-4)/12*(measured(kind,r=16)-measured(kind,r=4))
 lu=measured('slogdet',d=4032)
 gemm_per_sample=measured('likelihood_matmul',n=512)/512
 io_per_byte=measured('array_io')/(16*1024*1024)
 km=find('kmeans',d=512)
 km_iteration=km['median_seconds']/km['last_kmeans_iteration_count']
 def job(n,k,r):
  components={
   'dense_logdet':k*lu,
   'likelihood_matmul':k*n*gemm_per_sample,
   'likelihood_center_conjugate_quadratic_and_means':k*3*measured('residual')*n/512,
   'beta':k*interp('beta',r),
   'posterior_latents_and_lambdalatents':k*2*measured('latent_mean')*n/512*r/16,
   'posterior_latent_covariance_and_aggregation':k*2*measured('latent_covariance')*n/1024*(r/16)**2,
   'M_step_loading_tensor':k*measured('loading_tensor',r=r)*(D/512)*(n/1024),
   'M_step_residual':k*measured('residual')*n/512,
   'covariance_and_Woodbury':k*(interp('covariance',r)+interp('woodbury',r))}
  iteration=sum(components.values());mem=sizes(n,D,k,r);output=mem['private_parameter_bytes']
  # KMeans convergence on scientific data is unknown; 30/100/300 iteration scenarios.
  km_per_iter=km_iteration*(n/4096)*(D/512)*(k/4)
  io=io_per_byte*(output+mem['input_bytes'])
  init_cov=components['covariance_and_Woodbury']
  scenarios={
   'optimistic':{'EM_iterations':7,'throughput_multiplier':.8,'KMeans_iterations':30,'seconds':7*iteration*.8+30*km_per_iter+io+init_cov},
   'central':{'EM_iterations':100,'throughput_multiplier':1.5,'KMeans_iterations':100,'seconds':100*iteration*1.5+100*km_per_iter+2*io+init_cov},
   'conservative':{'EM_iterations':100,'throughput_multiplier':4.0,'KMeans_iterations':300,'seconds':100*iteration*4+300*km_per_iter+10*io+init_cov}}
  return {'N':n,'D':D,'K':k,'rank':r,'per_EM_component_seconds':components,'per_EM_nominal_seconds':iteration,'initialization_Kmeans_per_iteration_EXTRAPOLATED_seconds':km_per_iter,
   'load_hash_serialize_proxy_seconds':io,'scenarios':scenarios,'status':'EXTRAPOLATED_NOT_MEASURED_TRAINING'}
 candidates=[job(N,k,r) for k in (1,2,4) for r in (4,8,16)]
 finals=[job(15000,k,r) for k in (1,2,4) for r in (4,8,16)]
 totals={}
 for key in ('optimistic','central','conservative'):
  cand=sum(v['scenarios'][key]['seconds'] for v in candidates);f=[v['scenarios'][key]['seconds'] for v in finals]
  totals[key]={'nine_candidates_hours':cand/3600,'final_selected_structure_unknown_hours_range':[min(f)/3600,max(f)/3600],
   'ten_training_jobs_hours_range':[(cand+min(f))/3600,(cand+max(f))/3600]}
 return {'basis':'single-thread current CPU synthetic kernels; high-memory host CPU equivalence NOT measured',
 'candidate_jobs':candidates,'final_structure_alternatives_NOT_selected':finals,'totals':totals,
 'scenarios':{'optimistic':'conditional fastest legal 7-iteration convergence, 0.8 x nominal throughput; NOT expected convergence',
 'central':'100 EM iterations, 1.5 x nominal extrapolated time to allow sustained/cache overhead; no early convergence assumed',
 'conservative':'100 EM iterations, 4 x nominal extrapolated time, worst KMeans iterations and cached-IO caveat'},
 'uncertainty':['No full N x D fit measured; rank16 M-step strided reduction at 12-15GB may slow beyond small cache-scale kernels.',
 'Warm kernels differ from sustained thermal/paging pressure and new-host CPU/BLAS.',
 'Native allocator retention/packing, KMeans data-dependent iteration count, residual small solves/dispatch and filesystem hashing are not separately measured.',
 'Multipliers are engineering budgeting assumptions, not confidence intervals or guaranteed upper bounds.',
 'Validation scoring and 35-cell target inference are excluded from TEN TRAINING JOBS; must obtain a separate safe inference cost gate before promising end-to-end completion.'],
 'recommended_CPU_reservation':'up to 5 single-thread CPU days for ten fits on equivalent-or-faster CPU; actual host smoke must replace this forecast and preserve old 24h/job gate',
 'per_job_timeout_risk':'Worst final conservative projection is about23.82h, close to unchanged24h cap. Proposal: authorize restart only if host smoke projects worst100-iteration job <=19.2h, leaving20% time reserve. Otherwise obtain faster CPU or mentor budget decision; never silently raise timeout or assume early convergence.',
 'unknown_convergence_not_used_to_select_or_reduce_grid':True,'parameters/objectives/exported':False}
def reports():
 firewall();bench=read(ROOT/'outputs/sp_u12b5b_r2_timing_microbenchmark.json');assert bench['resource_summary']['completed']
 proof=next(v['materialization_proof'] for v in bench['kernels'] if 'materialization_proof' in v)
 assert proof['owns_data'] and proof['base_is_none'] and proof['all_entries_consumed_by_dot']
 source=str(CORE.relative_to(ROOT))+':'+str(line('self._lambdas[k] = np.dot'))
 gate={'verdict':'ORIGINAL_6GIB_RSS_GATE_STRUCTURALLY_INCOMPATIBLE','scope':'complete frozen nine-candidate resident execution of original author implementation',
 'maximum_candidate_tensor':{'shape':[D,16,N],'dtype':'complex128','bytes':16*D*16*N,'GiB':16*D*16*N/GIB,'source':source,'eager_allocation':True,'full_producer_write_and_dot_read':True,'broadcast_view_only':False},
 'maximum_possible_final_retraining_tensor':{'shape':[D,16,15000],'bytes':16*D*16*15000,'GiB':16*D*16*15000/GIB},
 'synthetic_materialization_proof':proof,'formal_large_tensor_actually_allocated_in_R2':False,
 'lifespan':'binary multiply produces entire ndarray, inner np.dot contracts N after argument evaluation; released after inner dot returns; not multiplied by K since component loop is sequential',
 'simultaneously_alive':'training input, zeroed, lambdalatents, private K covariances and inverses, beta, latents and latent_covs',
 'resource_limit_origin':{'classification':'ENVIRONMENT_SPECIFIC_SAFETY_LIMIT','science_constraint':False,
 'lineage':['docs/sp_u12a_r1_plan_b_protocol.md:128','configs/sp_u12b5b_mfa_selection.yaml:1252','docs/sp_u12b5b_change_log.md:5'],
 'author_code_contains_RAM_or_RSS_requirement':False,'historical_quantitative_derivation_of_6GiB':'NOT_DOCUMENTED in surviving protocol; inherited engineering cap. Mentor R2 explicitly identifies it as resource-safety class C.',
 'RSS_6GiB':'current-host engineering/process safety bound','physical_headroom_2GiB':'system safety threshold, environment-specific numerical setting'},
 'evidence_semantics':'Full materialization/access is proved by immutable NumPy source expression plus independent identical smaller expression, not by an unsafe 12GB run. Allocation bytes are not a universal theorem about OS RSS under forced paging; a paging-dependent execution cannot establish the resident safety gate compatible.',
 'old_R1_metadata_preserved':'R1 OPEN allocation-vs-RSS caution remains in archived bytes; R2 closes materialization and prospectively classifies resident execution incompatibility per mentor instruction.',
 'old_limits_modified':False,'high_RAM_alone_resolves_old_gate':False}
 emit('rss_gate_compatibility',gate)
 costs=duration_model(bench);emit('total_training_cost_estimate',costs)
 worst=sizes(15000,D,4,16);native=8*GIB;rounded=math.ceil((worst['SOURCE_BASED_LIVE_ESTIMATE_bytes']+native)/GIB)*GIB
 process_limit=rounded+4*GIB
 assert process_limit==32*GIB
 disk_parameters=sum(sizes(N,D,k,r)['private_parameter_bytes'] for k in (1,2,4) for r in (4,8,16))+worst['private_parameter_bytes']
 hostnow=host_snapshot();disk=shutil.disk_usage(ROOT)
 hosts={'status':'PROVISIONAL_CAPACITY_PLAN_UNMEASURED_TARGET_HOSTS','current_host':hostnow,'current_disk_free_bytes':disk.free,
 'source_final_peak_bytes':worst['SOURCE_BASED_LIVE_ESTIMATE_bytes'],'native_allocator_budget_reserve_bytes':native,'reserve_is_NOT_measured_bound':True,
 'planned_process_peak_bytes':worst['SOURCE_BASED_LIVE_ESTIMATE_bytes']+native,
 '32GiB':{'total_RAM_bytes':32*GIB,'OS_background_reserve_bytes':8*GIB,'minimum_headroom_bytes':4*GIB,
 'live_plus_native_estimate_bytes':worst['SOURCE_BASED_LIVE_ESTIMATE_bytes']+native,
 'capacity_slack_bytes':32*GIB-8*GIB-4*GIB-worst['SOURCE_BASED_LIVE_ESTIMATE_bytes']-native,
 'verdict':'NOT_RECOMMENDED_MARGIN_INSUFFICIENT','reason':'Even source peak alone leaves about0.81GiB after8GiB OS and4GiB headroom; native/retained workspace is unbounded by current evidence. Not a proof no32GiB machine can ever finish.'},
 '64GiB':{'total_RAM_bytes':64*GIB,'OS_background_planning_reserve_bytes':12*GIB,'maximum_background_at_launch_bytes':16*GIB,
 'minimum_available_at_launch_bytes':48*GIB,'headroom_threshold_bytes':8*GIB,'process_RSS_limit_bytes':process_limit,
 'physical_capacity_slack_to_estimated_peak_bytes':64*GIB-12*GIB-8*GIB-worst['SOURCE_BASED_LIVE_ESTIMATE_bytes']-native,
 'minimum_free_after_RSS_limit_given_launch_free_bytes':48*GIB-process_limit,
 'expected_commit_budget_bytes':worst['SOURCE_BASED_LIVE_ESTIMATE_bytes']+native,'min_commit_remaining_at_launch_bytes':44*GIB,
 'verdict':'PREFERRED_PROVISIONAL; target-host validation REQUIRED'},
 'disk':{'ten_model_parameter_bytes':disk_parameters,'training_inputs_bytes':16*D*(12000+15000),'scratch_allowance_bytes':4*GIB,
 'failed_attempt_reserve_bytes':16*GIB,'recommended_free_disk_bytes':64*GIB,'current_free_meets_proposed_future_host_gate':disk.free>=64*GIB,
 'notes':'Technical plan retains failed archives and one selected final, not nine final retrains. Do not copy/decode sealed scientific arrays to make capacity evidence. Cached IO timing is not durable disk guarantee.'}}
 emit('host_capacity_plan',hosts)
 protocol={'name':'PROPOSED_MFA_EXECUTION_RESOURCE_PROTOCOL','status':'PROVISIONAL_MENTOR_APPROVAL_REQUIRED','applied':False,
 'target_host_RAM_bytes':64*GIB,'minimum_available_RAM_at_launch_bytes':48*GIB,'process_RSS_hard_limit_bytes':process_limit,
 'basis':{'source_worst_final_live_bytes':worst['SOURCE_BASED_LIVE_ESTIMATE_bytes'],'unmeasured_native_allocator_reserve_bytes':native,
 'ceil_live_plus_reserve_to_GiB_bytes':rounded,'sampling_peak_margin_bytes':4*GIB,'formula':'ceil((source worst-final live +8GiB workspace reserve)/GiB)*GiB +4GiB monitor margin =32GiB; target64GiB, launchfree48GiB and headroom8GiB',
 'actual_host_allocation_smoke_needed':True},
 'system_physical_headroom_threshold_bytes':8*GIB,'min_commit_space_remaining_at_launch_bytes':44*GIB,
 'process_private_commit_limit_bytes':36*GIB,'min_remaining_system_commit_threshold_bytes':8*GIB,
 'commit_monitor':'GetPerformanceInfo CommitLimit-CommitTotal in bytes, plus actual interpreter private/peak commit; never equate pagefile configuration to commit limit.',
 'sampling_interval_seconds':.1,'max_candidate_seconds_UNCHANGED':86400,'BLAS_threads_UNCHANGED':1,'concurrency_UNCHANGED':1,
 'proposed_CPU_launch_gate':'Actual authorized host smoke must project worst100-iteration job <=19.2h, leaving20% reserve under existing24h/job cap. Current conservative worst final23.82h is too close for launch approval. Need faster CPU or mentor review; no timeout adjustment applied.',
 'disk_free_launch_minimum_bytes':64*GIB,
 'termination':'Actual interpreter PID handshake before import/fit; monitor handle tied to reported PID. Any peakRSS>=32GiB, privatecommit>=36GiB, physicalfree<8GiB, commitremaining<8GiB, timeout, monitor failure or correctness exception terminates actual interpreter and awaits launcher exit. No restart, survivor selection, caps bypass or app shutdown.',
 'logging':'100ms append-only JSONL UTC+elapsed, actual/launcher PID, current/peakRSS, private/peakcommit, total/availablephysical, used/limitcommit, code/data hashes, stage. Periodic flush and worker handshakes; process-exit kernel peak as final sample where possible.',
 'exceptions':'capture full traceback and last-stage marker; retain monitor failures as STOP; no model considered trained until fit/check/serialization gates pass',
 'failed_run_preservation':'new unique attempt directory, never overwrite B5B/R1/R2 or any future failed prefix; keep partial input/provenance/logs/parameter files, hash archives. Whole nine grid halted on any failed gate; do not select survivors.',
 'fidelity_requalification':['B5A/reference/adapter byte hashes','toy posterior parity and author/adapter parity at frozen tolerances','exact seed replay and RNG consumption','shape4032 complex128 training input/order/split hashes','BLAS/thread behavior','report cross-host bitwise equality honestly; tolerance agreement cannot be called bitwise'],
 'formal_restart_prerequisites':['mentor authorizes prospective resource amendment','actual target-host total/free/commit/disk capacity passes','reference parity PASS','separately authorized full-scale resource smoke passes without target or scoring','measured per-step throughput validates accepted CPU budget','science/split/grid/seeds frozen','B2/B3 still SEALED','all failed archives intact','separate CLEAN_RESTART_FULL_9_CANDIDATES authorization'],
 'current_R2_limits_still':LIMITS,'formal_seeds_not_changed':True}
 emit('proposed_resource_protocol',protocol)
 options=[
 {'option':'tensor contraction','author_expression':'dot(zeroed[:,None,:]*latents[None,:,:].conj(),rs)','mathematical_expression':'sum_n rs[n] zeroed[d,n] conj(latents[a,n]); equivalently (zeroed * rs[None,:]) @ latents.conj().T',
 'full_batch_semantics':'can preserve in exact arithmetic','update_order':'must preserve new mean before zeroed/loadings, and OLD lambdalatents in residual update','summation_order':'changes dot/BLAS reduction order; complex128 rounding can diverge','numerical_bias':'not stochastic bias, floating trajectory and stopping can differ','memory':'removes D*r*N tensor; alternative weighted D*N temporary plus D*r output','tiny_parity':'analytical exact sum + original update parity + entire iteration and seed/stopping replay required','faithful_label':'mathematically equivalent adaptation only after NEW fidelity gate; not unchanged author bytes','implemented':False},
 {'option':'chunked sufficient statistics','mathematical_expression':'accumulate exact full sum for responsibilities/latent moments/numerators across all N then one parameter update',
 'full_batch_semantics':'possible only with frozen pre-update responsibilities and proper multipass means','update_order':'cannot update component parameters while consuming partial chunks; old lambdalatents retained conceptually','summation_order':'changes grouping; compensated or pairwise sum still differs from original','numerical_bias':'rounding/stopping may differ','memory':'O(D*r*chunk) or sufficient-statistic contraction; more passes/compute','tiny_parity':'derive all updates including author residual ordering; multi-iteration tolerances and seed replay','faithful_label':'requires independent fidelity gate','implemented':False},
 {'option':'blockwise E-step likelihood','mathematical_expression':'same component logdet; compute responsibilities per observation block using same whole model','full_batch_semantics':'posterior at each sample unchanged in exact arithmetic; accumulate all sufficient stats before M step','update_order':'keep global component/parameter order','summation_order':'EM likelihood sum and M-step aggregation order may change','numerical_bias':'rounding, BLAS panel sizes and convergence may diverge','memory':'reduces N D E-step buffers; DOES NOT by itself remove M-step D r N or private K D^2','tiny_parity':'posterior/logsumexp/complete iteration comparison, not only individual posterior','faithful_label':'author-code adaptation needs new gate','implemented':False},
 {'option':'temporary lifetime reduction','mathematical_expression':'release dead arrays after final use, no changed operations','full_batch_semantics':'preserved if aliases and author later uses are proved absent','update_order':'unchanged','summation_order':'potentially unchanged','numerical_bias':'potentially bitwise on same environment; must verify','memory':'small to moderate; cannot remove mandatory 11.536GiB tensor alone','tiny_parity':'liveness/alias audit and bitwise identical iterations','faithful_label':'byte changes require new engineering fidelity record','implemented':False},
 {'option':'out-of-core tensors/data','mathematical_expression':'same full-batch arrays mapped to files','full_batch_semantics':'possible with unchanged complete sums','update_order':'must preserve; merely mapping input insufficient because author binary multiply creates resident temporary','summation_order':'unchanged only for unchanged math calls; chunking often changes','numerical_bias':'may be same values but paging/IO overhead and stopping timing differ','memory':'allocated tensor still14.4GiB for final; memmap substitution requires expression/output rewrite; high IO costs','tiny_parity':'whole fit replay and actual RSS/commit/disk/termination audit','faithful_label':'requires new engineering/fidelity gate; not magic oldRSS compatibility','implemented':False}]
 emit('equivalent_engineering_options',{'status':'ANALYSIS_ONLY_NOT_IMPLEMENTED','options':options,
 'non_equivalent':{'mini_batch_EM':'SCIENTIFIC_METHOD_CHANGE','stochastic_EM':'SCIENTIFIC_METHOD_CHANGE','reduced_dimensions':'SCIENTIFIC_METHOD_CHANGE','per_origin_models':'SCIENTIFIC_METHOD_CHANGE'},
 'same_real_arithmetic_formula_not_same_floating_trajectory':True})
 decision={'verdict':'MFA_HIGH_MEMORY_EXECUTION_PLAN_READY','scope':'prospective PLAN_READY only; target host not obtained, resource amendment not approved, formal restart not authorized',
 'preferred_option':'A','Option_A':'unchanged author + adapter + complex128 + data/split/grid/seeds on provisional64GiB host; single-thread 10-fit credible extrapolated budget, reference/host smoke still required',
 'Option_B':'review only if A unavailable/unacceptable; contraction/full-batch sufficient stats plausible, new fidelity gate obligatory; no implementation now',
 'Option_C':'not activated; GMM channel-estimation / structured statistical covariance / adaptive low-rank estimator from prior-art ledger only via separate Literature/Fidelity Gate and mentor decision',
 'Option_C_feasibility_review':[
 {'candidate':'Koller2022 GMM channel estimation','prior_ledger':'outputs/sp_u12a_external_baseline_feasibility.csv','relevance':'high: learned statistical prior, pilot-dependent responsibilities and conditional LMMSE',
 'cost':'Full covariance remains K D^2 storage and dense D^2 N likelihood/factorization costs; removing MFA D r N temporary does NOT guarantee cheap training. Full covariance conditioning at this dimension needs original-author treatment.',
 'fidelity':'new exact-author/tiny posterior/mean/noise/general-A/fullsum and full-dimension resource gates required','fairness':'same prepared origins/clean budget, target-blind training and selection; no target CSI or labels','automatic_replacement':False},
 {'candidate':'structured-GMM / structured covariance estimator','relevance':'statistical estimation and potential reduced deployment cost','cost':'potentially smaller only if the physical joint4032 geometry supports the published structure',
 'fidelity':'structure assumptions and author equations must be verified; cannot quietly relabel as full GMM','fairness':'same offline data and noise/pilot information','automatic_replacement':False},
 {'candidate':'adaptive low-rank estimator from prior-art ledger','relevance':'adjacent adaptation/compression interface','cost':'not established here','fidelity':'source-specific details require a new Literature/Fidelity Gate; Özdemir ABSTRACT_ONLY / OPEN RISK remains unresolved, not treated as implementable full-text evidence','fairness':'target-information contract must be checked','automatic_replacement':False}],
 'research_value_cost':'Continue MFA as first choice: nearest statistical learned-prior/observation-dependent baseline, original-author fidelity already established, comparison completeness scientifically valuable. Reserve up to5 CPU days for training on equivalent CPU plus separate inference budget; do not stop based on unknown relative performance.',
 'cost_acceptance':'credible budget supplied; mentor must accept actual resource/cost and approve host-based gates. No claim that host access or budget approval exists.',
 'candidate_models_successfully_trained':0,'validation_scores':0,'selected_K_rank':None,'final_model_trained':False,'target_outcomes_viewed':False,
 'author_algorithm_modified':False,'adapter_full_training_data_duplication':False,'public_private_parameter_copies':'author fit intentionally keeps two sets; not new adapter copies',
 'downstream':'STOP, await mentor review; no training or migration in R2'}
 emit('baseline_value_cost_decision',decision)
 tables='\n'.join(f"| {k} | {r} | {sizes(N,D,k,r)['LOWER_BOUND_single_tensor_bytes']/GIB:.3f} | {sizes(N,D,k,r)['SOURCE_BASED_LIVE_ESTIMATE_bytes']/GIB:.3f} |" for k in (1,2,4) for r in (4,8,16))
 totals=costs['totals']; timingtable='\n'.join(f"| {key} | {v['nine_candidates_hours']:.1f} h | {v['final_selected_structure_unknown_hours_range'][0]:.1f}–{v['final_selected_structure_unknown_hours_range'][1]:.1f} h | {v['ten_training_jobs_hours_range'][0]:.1f}–{v['ten_training_jobs_hours_range'][1]:.1f} h |" for key,v in totals.items())
 md('mfa_feasibility_review',f'''# SP-U12B5B-R2 — Computational feasibility review

Verdict: **{decision['verdict']}**. This is a prospective execution plan, not formal training or resource approval.

## Materialization and gate closure

At `{source}`, eager NumPy multiplication creates an owned complex128 `(D,r,N)` buffer before `np.dot` consumes the whole contraction axis. The independently measured identical smaller expression returned OWNDATA=True, base=None, no input sharing; all entries were checked/read. Candidate rank16 is `{16*D*16*N:,}` bytes =11.53564453125GiB; final rank16 with15000 frames is14.4195556640625GiB. No dangerous large tensor was instantiated here. Broadcast operands are views; their product is an allocation.

Therefore **ORIGINAL_6GIB_RSS_GATE_STRUCTURALLY_INCOMPATIBLE** for resident full-grid original-code execution, independent of installed RAM. Paging changes working-set residence, so static storage is not an unconditional operating-system RSS theorem or measured full-fit RSS. Paging-dependent completion does not establish safe compatibility. R1's historical caution is retained unchanged. The6GiB cap and2GiB headroom are **ENVIRONMENT_SPECIFIC_SAFETY_LIMIT**, not MFA algorithm definitions. Earliest surviving local lineage is PLAN_B protocol line128 then B5B config line1252; no quantitative original6GiB derivation is documented. Mentor R2 clarifies its safety category.

## Source memory graph

| K | r | single cross tensor GiB | simultaneous source live peak GiB |
|---|---|---|---|
{tables}

Whole-fit allocation history is not summed as peak. Input .721GiB remains resident, and the M-step loading tensor overlaps zeroed data, OLD lambdalatents, dense covariance/inverse, beta and latent moment buffers. Component loop is sequential: the D*r*N tensor is not multiplied byK. Covariance update has dense temporaries plus last component locals; serialization retains original private and public copies, finite/abs temporaries and memoryview hash/save. Native work buffers, packing, allocator-retained blocks and Python runtime are unknown.

Worst candidate source-live = {sizes(N,D,4,16)['SOURCE_BASED_LIVE_ESTIMATE_bytes']/GIB:.3f}GiB. Worst possible selected final source-live = {worst['SOURCE_BASED_LIVE_ESTIMATE_bytes']/GIB:.3f}GiB. These are SOURCE-BASED_LIVE_ESTIMATE, not measured RSS. LOWER_BOUND and UNMEASURED_OVERHEAD are separate in machine-readable models.

32GiB leaves only about.81GiB beyond source-live peak,8GiB OS and4GiB headroom; credible native margin is absent. Prefer64GiB with at least48GiB free physical and44GiB remaining commit at launch. The8GiB workspace/allocator reserve is a planning assumption, not verified bound. All target-host statements remain PROVISIONAL.

## Compute paths and measurement

Each EM calls `_calc_probs -> np.linalg.slogdet(self._covs)` for dense4032x4032 LU-type factorizations, then K dense inverse-covariance times4032xN likelihood products. `_update_covs -> _invert_cov_all` uses Woodbury and rxr pinv; it does **not** repeatedly call a DxD inverse. Beta, latent posterior moments, loading sufficient statistics and dense inverse materialization add cost. One component likelihood is {D*D*N:,} complexMACs; sumK=21 across the nine fits.

20 independent synthetic kernels completed under original6GiB/2GiB gates; peakRSS={bench['resource_summary']['peak_RSS_bytes']/GIB:.3f}GiB, minimum systemfree={bench['resource_summary']['minimum_available_physical_bytes']/GIB:.3f}GiB. Dense4032 LU measured{next(v['median_seconds'] for v in bench['kernels'] if v['case']['kind']=='slogdet' and v['case']['D']==4032):.3f}s;4032x4032 times4032x512 measured{next(v['median_seconds'] for v in bench['kernels'] if v['case']['kind']=='likelihood_matmul' and v['case']['N']==512):.3f}s. Three timed repetitions after one warmup, single-thread, seeds4218352701–4218352720, no channel/preparation/validation/target samples or models.

| EXTRAPOLATED scenario | nine candidate fits | selected final, structure unknown | ten fits total |
|---|---|---|---|
{timingtable}

Optimistic assumes earliest legal7-iteration convergence only as conditional lower scenario. Central and conservative budget100 iterations; no unknown convergence history is assumed. Central1.5x and conservative4x nominal kernel projection are explicit engineering uncertainty margins, not statistical intervals. Full-strided12–15GiB reductions, sustained thermal pressure, new-host CPU/BLAS, KMeans and IO can differ. Reserve up to5 single-thread CPU days for ten training jobs on equivalent-or-faster CPU, replace forecast with actual authorized host smoke before restart. The worst conservative final projection23.82h is close to the unchanged24h cap: proposed launch requires actual-host projected worst100-iteration job<=19.2h for20% time reserve; otherwise faster CPU or mentor review, never silently extending timeout. Validation and full evaluation inference are excluded from these training totals and require their own cost gate.

## Path comparison

A is preferred: unchanged author/adapter, data, grid, dtype, seeds and selection; high-RAM environment and prospective safety limits only. B is plausible exact-arithmetic full-batch contraction/chunking, but rounding and update ordering must undergo a new fidelity gate; do not implement here. The author's residual uses OLD lambdalatents after updating loadings, which any rewrite must preserve. Lifetime changes alone cannot remove the large tensor. C remains conditional external-baseline replanning, not an MFA replacement decision.

No candidate training, scores, structure choice, final retraining, target outcome reads, threshold changes, manuscript changes, Git commit or migration. First preflight failed from direct-entry sys.path before any benchmark; failed sources/metadata retained, entry corrected, actual pytest passed before frozen synthetic schedule.
''')
 md('execution_safety_amendment_proposal','# Prospective resource amendment — NOT APPLIED\n\nThe old6GiB/2GiB gates remain active. Mentor approval and actual target-host evidence are required.\n\n```json\n'+json.dumps(protocol,indent=2,ensure_ascii=False)+'\n```\n\n32GiB is not recommended;64GiB is a capacity plan only. Do not migrate or perform full-scale smoke until separately authorized.')
 md('recommended_path','# Recommended path — STOP for mentor review\n\n'+json.dumps(decision,indent=2,ensure_ascii=False)+'\n\nA → B → C priority retained. The existing nine seeds/final seeds, split, three origins, maxiter100 and author semantics remain frozen. CPU reservation excludes validation and target inference. No results inform this decision.')
 md('change_log','''# SP-U12B5B-R2 change log

Added isolated R2 technical-only source allocation model, strict historical-output firewall, actual-interpreter monitored synthetic kernel harness, actual pytest preflight and frozen execution bytes. Created requested memory/compute/cost/host/safety/option/decision reports by scripts.

First preparation entry failed before numerical execution because workspace import path was absent. Preserved original sources and failure metadata under outputs/sp_u12b5b_r2/preflight_failure01; corrected only new R2 entry points. Tests passed before synthetic timing. No optimization implemented; no original source, adapter, scientific config, data, manuscript or prior archive modified.

Old RSS/headroom values remain unchanged. All proposed64GiB/32GiB-RSS/8GiB-headroom settings are PROVISIONAL and not applied. All timing projections are EXTRAPOLATED, separate from the20 measured safe kernels. Final byte preservation and manifest closure are required after reports.
''')
 print(json.dumps({'stage':'R2_reports_generated','verdict':decision['verdict'],'time_totals':totals,'process_limit_proposed_GiB':process_limit/GIB,'peak_final_source_GiB':worst['SOURCE_BASED_LIVE_ESTIMATE_bytes']/GIB}),flush=True)
def close():
 firewall();pres=read(ROOT/'outputs/sp_u12b5b_r2_preservation.json');assert pres['PASS']
 test=read(OUT/'pytest_postflight.json');assert test['PASS']
 freeze=read(OUT/'execution_freeze.json')
 for n,h in freeze['sources'].items():assert sha(ROOT/n)==h,'EXECUTION_DRIFT:'+n
 access=read(OUT/'worker_access.json');assert not access['denied_reads'] and not DENIED
 seal=read(ROOT/'outputs/sp_u12b5b_r2_b2_b3_seal_audit.json');assert seal['PASS']
 closure=f"\n## Final closure\n\nProtected historical files: {pres['protected_files']}; changed=0, unauthorized additions=0, Git HEAD unchanged. B2={seal['after']['B2']['files']} and B3={seal['after']['B3']['files']} leaves hash/population verified, SEALED/UNVIEWED. B5A/B5B/R1 archive hashes and all listed leaves unchanged. Actual pytest:8 passed preflight and8 passed postflight; safe synthetic timing:20/20 completed, no formal fits, validation, selection or target scientific reads. Execution-freeze source hashes verified; report-generation source is additionally included in final artifact manifest. STOP.\n"
 for suffix in ('mfa_feasibility_review','change_log'):
  path=ROOT/'docs'/(PFX+suffix+'.md')
  text=path.read_text(encoding='utf8');assert '## Final closure' not in text,'CLOSE_MUST_NOT_REPEAT'
  path.write_text(text+closure,encoding='utf8')
 report_paths=['memory_lower_bounds.json','author_allocation_lifetimes.csv','9candidate_memory_model.csv','rss_gate_compatibility.json','host_capacity_plan.json','proposed_resource_protocol.json','training_kernel_audit.csv','compute_complexity_model.json','timing_microbenchmark.json','total_training_cost_estimate.json','equivalent_engineering_options.json','baseline_value_cost_decision.json','b2_b3_seal_audit.json','preservation.json']
 for suffix in report_paths:assert (ROOT/'outputs'/(PFX+suffix)).is_file()
 write(OUT/'controller_access.json',{'metadata_reads':ACCESS,'denied_reads':DENIED,'B2_B3_scientific_value_reads':0,'formal_data_loaded':False})
 files={v.relative_to(ROOT).as_posix():sha(v) for d in ('scripts','tests','docs','outputs') for v in (ROOT/d).rglob('*') if v.is_file() and owned(v) and not(set(v.parts)&{'__pycache__','pytest_tmp_preflight','pytest_tmp_postflight'}) and v.name not in ('sp_u12b5b_r2_manifest.json','sp_u12b5b_r2_manifest.sha256')}
 manifest={'task':'SP-U12B5B-R2','verdict':read(ROOT/'outputs/sp_u12b5b_r2_baseline_value_cost_decision.json')['verdict'],'files':files,'file_count':len(files),'artifact_tree_sha256':tree(files),
 'preservation_PASS':True,'execution_freeze_verified':True,'postflight_pytest_PASS':True,'all_reports_generated_by_script':True,'thresholds_not_applied':True,'scientific_outcomes_unviewed':True,'STOP':True}
 path=ROOT/'outputs/sp_u12b5b_r2_manifest.json';write(path,manifest);digest=sha(path)
 (ROOT/'outputs/sp_u12b5b_r2_manifest.sha256').write_text(digest+'\n',encoding='ascii')
 for n,h in files.items():assert sha(ROOT/n)==h
 print(json.dumps({'stage':'R2_closed','verdict':manifest['verdict'],'manifest_sha256':digest,'files':len(files),'STOP':True}),flush=True)
