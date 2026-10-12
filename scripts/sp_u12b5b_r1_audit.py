"""Generate a source-based allocation audit without loading scientific values."""
from scripts.sp_u12b5b_r1_common import *
firewall()
import numpy as np,scipy
sys.path.append(str(ROOT/'.venv-gmm/Lib/site-packages'))
import sklearn
from threadpoolctl import threadpool_info
N,D=12000,4032
def number(path,needle):
 lines=Path(path).read_text(encoding='utf8').splitlines();hits=[i for i,s in enumerate(lines,1) if needle in s]
 return hits[-1] if needle.startswith(('X1 =','X2 =','p = np.sum(X1')) else hits[0]
def sizes(N,D,K,r):
 data=16*N*D;cov=16*K*D*D;parameter=2*cov+16*K*D+16*K*D*r+8*K*D+8*K
 small=8*K*N+16*K*r*D+16*r*N+32*r*r*N+16*r*r+8*K
 mstep=3*data+parameter+small+16*D*r*N
 likelihood=5*data+parameter+8*K*N+8*N
 residual=6*data+parameter+small
 kmeans=4*data+2*cov+16*K*D+8*N*3
 finalize=data+2*parameter
 return dict(input_bytes=data,private_parameter_bytes=parameter,loading_cross_tensor_bytes=16*D*r*N,component_latent_covariance_bytes=16*r*r*N,mstep_principal_live_bytes=mstep,likelihood_principal_live_bytes=likelihood,residual_principal_live_bytes=residual,Kmeans_principal_live_bytes=kmeans,final_public_private_principal_live_bytes=finalize,principal_live_peak_bytes=max(mstep,likelihood,residual,kmeans,finalize),unknown_extra_workspace='BLAS/LAPACK/allocator/runtime, not included; these are source live-array sums, NOT measured RSS')
def source_rows():
 rows=[];sk=ROOT/'.venv-gmm/Lib/site-packages/sklearn/cluster/_kmeans.py';nu=ROOT/'.venv-cdl/Lib/site-packages/numpy/_core/_methods.py';util=CORE.parent/'utils.py';worker=ROOT/'scripts/sp_u12b5b_worker.py';book=ROOT/'scripts/sp_u12b5b_model.py'
 def add(array,stage,shape,dtype,formula,path,needle,lifetime,coalive,classification,avoid,scales):
  row=dict(array=array,allocation_stage=stage,shape_formula=shape,dtype=dtype,bytes_formula=formula,source_path=Path(path).relative_to(ROOT).as_posix(),source_line=number(path,needle),lifetime=lifetime,simultaneously_alive=coalive,scales_with_N='N' in scales,scales_with_D='D' in scales,scales_with_K='K' in scales,scales_with_rank='r' in scales,allocation_owner=classification,avoidable_without_semantic_change=avoid)
  for K,r in ((1,4),(4,16)):row[f'bytes_K{K}_r{r}']=eval(formula,{'__builtins__':{}},dict(N=N,D=D,K=K,r=r)) if formula!='UNKNOWN' else None
  rows.append(row)
 add('training_data','worker materialization','(N,D)','complex128','16*N*D',worker,'data=np.load(path','entire fit','all fit allocations','SCIENTIFICALLY_REQUIRED_INTERFACE','No removal of mandatory input; staging changes require separate review','ND')
 add('adapter np.asarray input','fit_reference','(N,D)','complex128','0',ADAPTER,'data = np.asarray(clean_vectors','alias while fit','worker training_data','SCIENTIFICALLY_REQUIRED_INTERFACE','Already no copy for frozen complex128 ndarray','ND')
 add('Kmeans real/imag concatenation','initialization','(N,2D)','float64','16*N*D',util,'return np.concatenate','KMeans.fit call only','complex data; core calloc covariance addresses; KMeans copy','AUTHOR_REQUIRED','Do not change frozen representation','ND')
 add('KMeans copy_x=True validated X','Kmeans validation','(N,2D)','float64','16*N*D',sk,'copy=self.copy_x','KMeans.fit local','original concatenation plus complex data','AUTHOR_REQUIRED','Changing sklearn settings prohibited in this task','ND')
 add('Kmeans tolerance variance deviation','Kmeans _tolerance -> np.var','(N,2D)','float64','16*N*D',nu,'x = um.subtract','np.var call','KMeans copy and original real concatenation; complex data','AUTHOR_REQUIRED','Library allocation, no optimization here','ND')
 add('Kmeans centers','Kmeans','(K,2D)','float64','16*K*D',sk,'centers_init = self._init_centroids','KMeans.fit and result','large real inputs; extra centers_old/new of same size','AUTHOR_REQUIRED','No settings change','KD')
 add('Kmeans labels two arrays','Lloyd','2(N,)','int32','8*N',sk,'labels = np.full(X.shape[0]','Lloyd loop','centers, real X','AUTHOR_REQUIRED','No settings change','N')
 add('Kmeans sample weights and row norms','Kmeans','2(N,)','float64','16*N',sk,'sample_weight = _check_sample_weight','Kmeans.fit','real X, centers','AUTHOR_REQUIRED','No change','N')
 add('Kmeans native chunk/BLAS workspace','Kmeans','implementation-dependent','UNKNOWN','UNKNOWN',sk,'lloyd_iter(','native call','Kmeans inputs','UNKNOWN','Not inferred from RSS delta','NDK')
 add('means','initialization and EM','(K,D)','complex128','16*K*D',CORE,'self._means = ut.real2cplx','entire fit','data, loadings, covariance matrices','AUTHOR_REQUIRED','No change','KD')
 add('loadings','initialization and EM','(K,D,r)','complex128','16*K*D*r',CORE,'self._lambdas = (np.random','entire fit','data, residuals, means, covariance matrices','AUTHOR_REQUIRED','No change','KDr')
 add('residuals','initialization and EM','(K,D)','float64','8*K*D',CORE,'self._psis = np.tile','entire fit','means, loadings','AUTHOR_REQUIRED','No change','KD')
 add('weights','initialization and EM','(K,)','float64','8*K',CORE,'self.amps = np.random','entire fit','all components','AUTHOR_REQUIRED','No change','K')
 add('responsibilities','fit and EM','(K,N)','float64','8*K*N',CORE,'self.rs = np.zeros','fit to cleanup; old and new may overlap','logrs and covariance parameters','AUTHOR_REQUIRED','No change','KN')
 add('private covariance and inverse','fit; initialization; every EM','2(K,D,D)','complex128','32*K*D*D',CORE,'self._covs = np.zeros','entire fit; calloc bytes need not equal resident bytes','all data and per-component arrays','AUTHOR_REQUIRED','Author-defined dense buffers','KD')
 add('covariance replacement','_update_covs','(K,D,D)','complex128','16*K*D*D',CORE,'self._covs = self._lambdas @','new result overlaps old covariance until assignment','old covariance and inverse','AUTHOR_REQUIRED','No change','KD')
 add('diag residual matrix','_update_covs','(D,D)','float64','8*D*D',CORE,'self._covs[k] += np.diag','expression only','new covariance and old inverse','AUTHOR_REQUIRED','Equivalent diagonal indexing would modify author implementation; not authorized','D')
 add('inverse step matrix','_invert_cov_all','(K,D,D)','complex128','16*K*D*D',CORE,'step = psiI','expression to return; old inverse remains alive','covariance; intermediate products and inverse output','AUTHOR_REQUIRED','No change','KD')
 add('inverse broadcast multiply overlap','_invert_cov_all','(K,D,D)','complex128','16*K*D*D',CORE,'step = psiI','expression only; peak overlap of products','old inverse; covariance; new step','AUTHOR_REQUIRED','No change','KD')
 add('inverse return negation','_invert_cov_all','(K,D,D)','complex128','16*K*D*D',CORE,'return -step','return until assignment; distinct array from step','step, old inverse, new covariance','AUTHOR_REQUIRED','No change','KD')
 add('objective/logrs buffers','_calc_probs','(K,N) + (N,)','float64','8*K*N+8*N',CORE,'logrs = np.zeros((self.n_components, self.N))','probability call; small extra log-sum buffers','old/new responsibilities and data','AUTHOR_REQUIRED','Values not read or exported','KN')
 add('probability X1','_log_multi_gauss_nodet','(D,N)','complex128','16*D*N',CORE,'X1 = (data - self._means[k]).T','function','X2, conj(X1), product, data, private covariance pair','AUTHOR_REQUIRED','No change','ND')
 add('probability X2','_log_multi_gauss_nodet','(D,N)','complex128','16*D*N',CORE,'X2 = self._inv_covs[k] @ X1','function','X1, data and covariance pair','AUTHOR_REQUIRED','No change','ND')
 add('probability conjugate/product overlap','_log_multi_gauss_nodet','2(D,N)','complex128','32*D*N',CORE,'p = np.sum(X1.conj() * X2','expression only','X1 and X2 still live','AUTHOR_REQUIRED','No change','ND')
 add('betas','EM entry','(K,r,D)','complex128','16*K*r*D',CORE,'betas = np.transpose','EM call','covariances, data, per-component arrays','AUTHOR_REQUIRED','No change','KDr')
 add('latent posterior values','component E-step','(r,N)','complex128','16*r*N',CORE,'latents = betas[k]','component loop; previous component may overlap during reassignment','data centered temporary and betas','AUTHOR_REQUIRED','Actual per-component allocation, not docstring K*r*N','Nr')
 add('latent outer products step1','component E-step','(r,r,N)','complex128','16*r*r*N',CORE,'step1 = latents[:, None','component loop','latent_covs; previous step1 can overlap next assignment','AUTHOR_REQUIRED','No change','Nr')
 add('latent posterior covariance','component E-step','(r,r,N)','complex128','16*r*r*N',CORE,'latent_covs = np.eye','component loop','step1 remains alive','AUTHOR_REQUIRED','Actual per-component allocation, not K*r*r*N','Nr')
 add('lambdalatents','component M-step','(D,N)','complex128','16*D*N',CORE,'lambdalatents = self._lambdas[k] @','component loop','data and zeroed remain live','AUTHOR_REQUIRED','No change','ND')
 add('zeroed','component M-step','(D,N)','complex128','16*D*N',CORE,'zeroed = data.T','component loop','data and lambdalatents','AUTHOR_REQUIRED','No change','ND')
 add('loading cross-statistic broadcast','component M-step','(D,r,N)','complex128','16*D*r*N',CORE,'self._lambdas[k] = np.dot','entire np.dot reduction; expression then released','data; zeroed; lambdalatents; latent_covs; step1; covariance pair','AUTHOR_REQUIRED','Removing tensor would alter implementation; separate equivalence task required','NDr')
 add('residual subtraction/conjugate/product overlap','component M-step','3(D,N)','complex128','48*D*N',CORE,'psis = np.real(np.dot','expression only','data; zeroed; lambdalatents; latent arrays','AUTHOR_REQUIRED','No change','ND')
 add('public parameter copies','fit finalization','means,loadings,residuals,covariances','mixed complex128/float64','32*K*D*D+16*K*D+16*K*D*r+8*K*D',CORE,'self.means = self._means.copy()','trained model lifetime','all private parameters and worker input still retained','AUTHOR_REQUIRED','Author fit creates these copies; not adapter duplication','KD r')
 add('parameter finite masks','post-fit validity','(K,D,D)','bool','K*D*D',book,'finite={n:bool','each np.isfinite call','public and private parameter copies','SCIENTIFICALLY_REQUIRED_INTERFACE','Required validity audit; not source of initial failure','KD')
 add('covariance abs diagnostic','post-fit validity','(K,D,D)','float64','8*K*D*D',book,'covariance_valid=worst','expression only','public and private parameter copies','SCIENTIFICALLY_REQUIRED_INTERFACE','Not source of initial failure; no implementation change','KD')
 add('serialization array_hash ascontiguous','hashing','parameter shape','same dtype','0',book,'a=np.ascontiguousarray','alias for existing C-contiguous arrays; otherwise UNKNOWN extra copy','parameter arrays','SCIENTIFICALLY_REQUIRED_INTERFACE','No full parameter concat; actual layouts should be audited before future fit','KD r')
 add('NPY serialization streaming buffer','serialization','implementation buffer','bytes','16777216',book,'np.save(f,getattr','np.save call; standard numpy write_array buffersize','parameters; no concatenated model array','SCIENTIFICALLY_REQUIRED_INTERFACE','No serialization in R1 probes','')
 return rows
def header():
 import scripts.sp_u12b5b_r1_common as common
 path=ROOT/'outputs/sp_u12b5b/selection_train.npy';common.HEADER_ONLY=True
 try:
  with path.open('rb') as f:
   version=np.lib.format.read_magic(f)
   reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}[version]
   shape,order,dtype=reader(f);offset=f.tell()
 finally:common.HEADER_ONLY=False
 assert shape==(N,D) and dtype==np.dtype('complex128') and not order
 return dict(shape=list(shape),dtype=str(dtype),C_order=True,data_bytes=N*D*dtype.itemsize,MiB=N*D*dtype.itemsize/2**20,GiB=N*D*dtype.itemsize/2**30,header_bytes_read=offset,payload_values_read=0,file_bytes=path.stat().st_size,Kmeans_shape=[N,2*D],Kmeans_dtype='float64',Kmeans_representation_bytes=16*N*D,Kmeans_MiB=16*N*D/2**20,Kmeans_GiB=16*N*D/2**30,future_final_N=15000,future_final_input_bytes=15000*D*16)
def failed_timeline():
 stop=read(ROOT/'outputs/sp_u12b5b/STOP.json');monitor=read(ROOT/'outputs/sp_u12b5b/monitor_K1_r4_train.json');fit=read(ROOT/'outputs/sp_u12b5b_candidates/K1_r4/fit_entry.json')
 end=datetime.datetime.fromisoformat(stop['created_UTC']).timestamp();start=end-monitor['wall_seconds']
 fit_time=datetime.datetime.fromisoformat(fit['started_UTC']).timestamp()-start
 log=ROOT/'outputs/sp_u12b5b/worker_K1_r4_train.log';text=log.read_text(encoding='utf8');assert '"stage": "initialized"' in text and '"stage": "EM"' not in text
 initialized_time=log.stat().st_mtime-start
 records=[json.loads(line) for line in (ROOT/'outputs/sp_u12b5b/resource_K1_r4_train.jsonl').read_text(encoding='utf8').splitlines()]
 rows=[]
 for v in records:
  e=v['elapsed_seconds'];stage='WORKER_START_OR_DATA_MATERIALIZATION' if e<fit_time else 'AUTHOR_INITIALIZATION_KMEANS_OR_COVARIANCE_SUBSTAGE_UNKNOWN' if e<initialized_time else 'FIRST_EM_SUBSTAGE_UNKNOWN'
  rows.append(dict(elapsed_seconds=e,actual_PID=v['actual_PID'],launcher_PID=v['launcher_PID'],RSS_bytes=v['process']['working_set_bytes'],peak_RSS_bytes=v['process']['peak_working_set_bytes'],private_commit_bytes=v['process']['private_bytes'],peak_commit_bytes=v['process']['peak_commit_bytes'],available_physical_bytes=v['system']['available_physical_bytes'],available_commit_bytes=v['system']['available_commit_bytes'],bounded_stage=stage,stage_timing_basis='approximate parent start from STOP timestamp minus measured wall; initialization marker from technical log mtime, no scientific intermediate opened'))
 csvwrite(ROOT/'outputs/sp_u12b5b_r1_failed_run_timeline.csv',rows)
 summary=dict(resource_records=len(records),monitor=monitor,peak_stage='FIRST_EM_SUBSTAGE_UNKNOWN',exact_allocation_at_peak='UNKNOWN: old run lacks line-level allocation trace',fit_entry_elapsed_approx_seconds=fit_time,initialization_completed_elapsed_approx_seconds=initialized_time,Kmeans_start_end='UNKNOWN',MFA_covariance_initialization_start_end='UNKNOWN',data_materialization_start='UNKNOWN',data_materialization_completed_before_fit_entry=True,scientific_intermediate_values_read=0,failed_attempt_scientific_selection_use='PROHIBITED')
 write(OUT/'failed_timeline_summary.json',summary)
def host_idle():
 samples=[]
 for i in range(20):samples.append(host_snapshot());time.sleep(1)
 process=subprocess.run(['powershell','-NoProfile','-Command','Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 12 ProcessName,Id,WorkingSet64 | ConvertTo-Json -Compress'],capture_output=True,text=True,encoding='utf8')
 info=dict(status='FRESH_RESEARCH_IDLE_SNAPSHOT',definition='No formal training or synthetic probe running; existing user applications retained. This is not a clean-boot/minimum-background measurement.',process_inventory_scope='sandbox-visible only; not sufficient to attribute all host background usage',applications_closed=0,critical_services_closed=0,samples=samples,total_physical_bytes=samples[-1]['total_physical_bytes'],available_physical_min_bytes=min(v['available_physical_bytes'] for v in samples),available_physical_max_bytes=max(v['available_physical_bytes'] for v in samples),commit_limit_bytes=samples[-1]['commit_limit_bytes'],commit_used_bytes=samples[-1]['commit_used_bytes'],OS_background_unavailable_physical_bytes=samples[-1]['OS_background_unavailable_physical_bytes'],top_processes=json.loads(process.stdout) if process.returncode==0 else None,pagefile_specific_size='UNKNOWN: commit limit measured directly, not mislabeled as pagefile size')
 write(ROOT/'outputs/sp_u12b5b_r1_host_capacity.json',info)
def audit():
 check_frozen();cfg=read(ROOT/'configs/sp_u12b5b_mfa_selection.yaml');inputs=header();write(ROOT/'outputs/sp_u12b5b_r1_training_input_memory.json',inputs)
 rows=source_rows();csvwrite(ROOT/'outputs/sp_u12b5b_r1_memory_source_audit.csv',rows)
 scale=[dict(candidate=f'K{K}_r{r}',K=K,rank=r,**sizes(N,D,K,r),expected_relative_peak_category='largest rank16 / K-increasing covariance' if r==16 else 'rank8 tensor dominated' if r==8 else 'smallest rank; modeled Mstep live overlap dominates') for K in (1,2,4) for r in (4,8,16)]
 csvwrite(ROOT/'outputs/sp_u12b5b_r1_candidate_memory_scaling.csv',scale)
 worst=max(scale,key=lambda x:x['principal_live_peak_bytes']);model=dict(N=N,D=D,rows=len(rows),worst_case=worst['candidate'],certainty='UNAMBIGUOUS_FOR_SOURCE_ARRAY_SHAPES; actual RSS ordering subject to OS residency/allocator',shape_monotonicity='parameters grow with K, loading broadcast grows with rank; per-component D*r*N tensor is sequential and is NOT multiplied by K',memory_semantics='live-array nbytes are allocation bytes, not an unconditional RSS lower bound; paging, calloc zero pages and BLAS workspaces matter',candidate_models=scale,worst_final_full_data=sizes(15000,D,4,16),formal_method_modified=False)
 write(ROOT/'outputs/sp_u12b5b_r1_allocation_model.json',model)
 copies=[dict(array='fit_reference data=np.asarray(complex128)',classification='SCIENTIFICALLY_REQUIRED_INTERFACE',duplicate_bytes=0,why='Frozen worker np.load supplies an already complex128 ndarray; np.asarray preserves identity',possible_removal='None needed',algorithm_change_required=False),dict(array='4096 full tensor/vector and 64 sidecar in training worker',classification='SCIENTIFICALLY_REQUIRED_INTERFACE',duplicate_bytes=0,why='Formal worker loads selection_train.npy directly; does not call offline_pack or load original tensor/sidecar',possible_removal='Not present',algorithm_change_required=False),dict(array='KMeans real concat and copy_x=True',classification='AUTHOR_REQUIRED',duplicate_bytes=16*N*D,why='Original author requests cplx2real; installed sklearn default copy_x=True holds a second real matrix during fit',possible_removal='Would require separate author-path equivalence review; not an adapter staging fix',algorithm_change_required='Implementation change, prohibited here'),dict(array='author public/private covariance copies',classification='AUTHOR_REQUIRED',duplicate_bytes=32*4*D*D,why='Original fit explicitly copies _covs/_inv_covs to public covs/inv_covs after EM',possible_removal='Requires author implementation/inference compatibility review',algorithm_change_required='Implementation change, prohibited here'),dict(array='B5B parameter hashing',classification='SCIENTIFICALLY_REQUIRED_INTERFACE',duplicate_bytes=0,why='ascontiguousarray is alias for C-contiguous parameters; hashes stream memoryview; no concatenated copy',possible_removal='No observed large adapter copy; contingent on layout',algorithm_change_required=False)]
 csvwrite(ROOT/'outputs/sp_u12b5b_r1_adapter_copy_audit.csv',copies)
 failed_timeline()
 seed_registry=read(ROOT/'outputs/sp_u12b5b_candidate_seed_registry.json');formal={r[key] for r in seed_registry['candidates'] for key in ('seed','final_seed')}
 seeds=dict(synthetic_data_seed=4197123401,resource_initialization_seed=4197123402);assert not set(seeds.values())&formal
 config=dict(task='SP-U12B5B-R1',role='RESOURCE_ONLY_NO_SCIENTIFIC_SELECTION_USE',N=N,D=D,dtype='complex128',structures=[dict(K=4,rank=16,name='K4_r16')],selection_basis='source allocation worst-case only',maxiter=2,author_defaults_except_authorized_probe_iteration_cap=True,seeds=seeds,formal_seeds_used=False,preparation_IDs=[],target_IDs=[],input='independent shape-equivalent complex Gaussian synthetic; chunked input generation only, original full batch EM unchanged',validation=False,NMSE=False,parameter_serialization=False,model_selection=False,final_retraining=False,limits=cfg['resource_contract'],resource_exceed_action='terminate actual interpreter immediately; no further probes; no formal restart',trace='original author source line stages and array shape/dtype/nbytes only; no likelihood/parameter values',software=dict(Python=sys.version,NumPy=np.__version__,SciPy=scipy.__version__,scikit_learn=sklearn.__version__,threadpool_info=threadpool_info()),frozen=check_frozen())
 write(CONFIG,config);host_idle();print(json.dumps(dict(stage='R1_source_audit_complete',worst_case=worst['candidate'],input_bytes=inputs['data_bytes'],largest_broadcast_bytes=worst['loading_cross_tensor_bytes'],scientific_values_read=0)),flush=True)
if __name__=='__main__':audit()
