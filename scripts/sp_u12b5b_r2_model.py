"""Source live allocations and prospective costs; no scientific inputs."""
from scripts.sp_u12b5b_r2_common import *
import math
GIB=2**30; N=12000; D=4032
def sizes(n,d,k,r):
 a=16*n*d; c=16*k*d*d; pars=2*c+16*k*d+16*k*d*r+8*k*d+8*k
 small=8*k*n+16*k*r*d+16*r*n+32*r*r*n+16*r*r+8*k
 stages={'input':a,'KMeans':4*a+2*c+16*k*d+24*n,
 'covariance_update':3*a+pars+small+2*c,
 'likelihood':5*a+pars+16*k*n+8*n,
 'M_step_loading':3*a+pars+small+16*d*r*n,
 'M_step_residual':6*a+pars+small,
 'public_parameter_copies':a+2*pars,
 'serialization_validity':a+2*pars+8*k*d*d+16*64*d}
 return {'N':n,'D':d,'K':k,'rank':r,'input_bytes':a,'private_parameter_bytes':pars,
 'LOWER_BOUND_single_tensor_bytes':16*d*r*n,
 'LOWER_BOUND_tensor_plus_mandatory_input_bytes':16*d*r*n+a,
 'stage_live_bytes':stages,'SOURCE_BASED_LIVE_ESTIMATE_bytes':max(stages.values()),
 'peak_stage':max(stages,key=stages.get),'UNMEASURED_OVERHEAD':'native workspaces, packing, allocator retention, Python/runtime; not an RSS measurement'}
def line(needle,path=CORE):
 hits=[i for i,s in enumerate(path.read_text(encoding='utf8').splitlines(),1) if needle in s]
 assert hits,needle
 return hits[-1] if needle.startswith(('X1 =','X2 =','p = np.sum(X1')) else hits[0]
def allocation_rows():
 rows=[]
 def add(stage,name,shape,dtype,formula,needle,lifetime,overlap,confidence='HIGH',path=CORE):
  rows.append({'stage':stage,'allocation':name,'shape':shape,'dtype':dtype,'bytes_formula':formula,
   'bytes_K4_r16_N12000':None if formula=='UNKNOWN' else eval(formula,{'__builtins__':{}},{'N':N,'D':D,'K':4,'r':16}),
   'lifetime':lifetime,'peak_overlap':overlap,'evidence_source':str(path.relative_to(ROOT))+':'+str(line(needle,path)),
   'confidence':confidence,'actual_full_scale_measured':False})
 worker=ROOT/'scripts/sp_u12b5b_worker.py'; util=CORE.parent/'utils.py';sk=ROOT/'.venv-gmm/Lib/site-packages/sklearn/cluster/_kmeans.py'
 add('input','resident clean joint vector','(N,D)','complex128','16*N*D','data=np.load(path','whole fit','all stages',path=worker)
 add('adapter','np.asarray clean input alias','(N,D)','complex128','0','data = np.asarray(clean_vectors','whole fit','same C-order complex128 input, no duplicate',path=ADAPTER)
 add('KMeans','real/imag concatenate','(N,2D)','float64','16*N*D','return np.concatenate','KMeans.fit only','complex input + validated KMeans copy',path=util)
 add('KMeans','copy_x=True','(N,2D)','float64','16*N*D','copy=self.copy_x','fit only','input and real/imag concatenation',path=sk)
 add('KMeans','variance and native workspace','(N,2D) plus chunks','float64','16*N*D','self._tol = _tolerance','temporary','KMeans copy and input; extra chunk workspace UNKNOWN','MEDIUM',path=sk)
 add('parameters','private covariance and inverse','2(K,D,D)','complex128','32*K*D*D','self._covs = np.zeros','fit, replacement at updates','means/loadings/psis/input; initially calloc need not resident')
 add('parameters','means/loadings/psis/amplitudes','KD,KDr,KD,K','complex128/float64','16*K*D+16*K*D*r+8*K*D+8*K','self._lambdas = (np.random','whole fit','dense private covariance and inverse')
 add('covariance update','new covariance batch','(K,D,D)','complex128','16*K*D*D','self._covs = self._lambdas @','until next update','old covariance before assignment and old inverse; last M-step locals')
 add('covariance update','psi diagonal','(D,D)','float64','8*D*D','self._covs[k] += np.diag','each component temporary','new cov + old inverse + last M-step locals')
 add('Woodbury','small pseudoinverse','(K,r,r)','complex128','16*K*r*r','inv_inner = np.linalg.pinv','one update','KDr conjugate/weighted products and dense old buffers')
 add('Woodbury','dense covariance inverse construction intermediates','up to 2(K,D,D)','complex128','32*K*D*D','step = psiI[:, :, None]','multiply/negation expression','new cov + old inverse + input + last component locals; prior temporaries released')
 add('likelihood','slogdet native factorization','(D,D) per component workspace','complex128','UNKNOWN','sgn, logdet = np.linalg.slogdet(self._covs)','per EM native call','private K cov/inverse + input; native work buffer not in live-array model','UNKNOWN')
 add('likelihood','centered X1','(D,N)','complex128','16*D*N','X1 = (data - self._means[k]).T','one component likelihood','input + private parameters')
 add('likelihood','X2','(D,N)','complex128','16*D*N','X2 = self._inv_covs[k] @ X1','one component likelihood','X1 + input + private parameters')
 add('likelihood','X1 conjugate and quadratic product','2(D,N)','complex128','32*D*N','p = np.sum(X1.conj()','sum expression','X1 + X2 + input; elementwise intermediates eagerly allocated')
 add('E step','beta batch','(K,r,D)','complex128','16*K*r*D','betas = np.transpose','one EM call','input and private matrices')
 add('E step','latents','(r,N)','complex128','16*r*N','latents = betas[k]','per component; previous replaced','centered temporary D,N')
 add('E step','step1 + latent covariance','2(r,r,N)','complex128','32*r*r*N','step1 = latents','per component, through covariance update','latents, beta, zeroed and lambdalatents later; old loop locals can overlap replacements')
 add('M step','lambdalatents','(D,N)','complex128','16*D*N','lambdalatents =','per component through _update_covs','input + zeroed + latent covariance')
 add('M step','zeroed data','(D,N)','complex128','16*D*N','zeroed = data.T','per component through _update_covs','input + lambdalatents + latent covariance')
 add('M step','broadcast operands','(D,1,N),(1,r,N)','complex128','16*r*N','self._lambdas[k] = np.dot','expression only','zeroed indexing is view; latents.conj allocates rN; multiplication result is NOT view')
 add('M step','loading cross tensor','(D,r,N)','complex128','16*D*r*N','self._lambdas[k] = np.dot','until first np.dot consumes it','input + zeroed + lambdalatents + private matrices + r^2 N + beta')
 add('M step','loading numerator and small inverse','(D,r),(r,r)','complex128','16*D*r+16*r*r','inv(np.dot(latent_covs','outer dot expression','large cross tensor released after inner dot; r x r inverse, not D x D')
 add('M step','residual expression temporaries','up to 3(D,N)','complex128','48*D*N','psis = np.real(np.dot','one residual expression','input + zeroed + lambdalatents; no large D r N tensor simultaneously')
 add('finalize','public parameter copies','same as private','complex128/float64','32*K*D*D+16*K*D+16*K*D*r+8*K*D+8*K','self.means = self._means.copy','after fit until process exit','private arrays remain; adapter creates no extra full input')
 book=ROOT/'scripts/sp_u12b5b_model.py'
 add('serialization','parameter hashes and np.save','C-contiguous memoryviews + finite/abs temporary','complex128/float64','8*K*D*D+16*64*D','covariance_valid=worst','validation then serialization','public/private parameter sets + data; hashing ascontiguousarray aliases contiguous arrays','MEDIUM',path=book)
 add('native','BLAS/LAPACK/allocator','implementation-dependent','UNKNOWN','UNKNOWN','inv_inner = np.linalg.pinv','native calls / possibly retained','not all source arrays simultaneously live; reserve specified as planning uncertainty','UNKNOWN')
 return rows
def kernel_rows():
 specs=[
 ('KMeans initialization','(N,2D),K','O(I_KM N D K); kmeans++ extra initialization scans','once/fit','linear K; rank-independent','Kmeans = cluster.KMeans'),
 ('covariance construction','(K,D,r) @ (K,r,D)','K D^2 r complex MAC + K D^2 diagonal work','initialization + every EM','linear K,r','self._covs = self._lambdas @'),
 ('dense log determinant','(K,D,D)','K O(D^3) dense LU, despite low-rank mathematical structure','every EM','linear K; rank changes values, not dense shape','sgn, logdet = np.linalg.slogdet(self._covs)'),
 ('covariance inverse Woodbury','K r x r pinv; dense D x D materialization','O(K(r^3+D r^2+D^2 r+D^2))','initialization + every EM','linear K; rank r','inv_inner = np.linalg.pinv'),
 ('component likelihood','(D,D) @ (D,N)','K D^2 N complex MAC + O(KDN) elementwise','every EM','linear K; dominant term rank-independent','X2 = self._inv_covs[k] @ X1'),
 ('posterior beta','(K,r,D) @ (K,D,D)','K r D^2 complex MAC','every EM','linear K,r','betas = np.transpose'),
 ('posterior latent means','(r,D) @ (D,N)','K r D N complex MAC','every EM','linear K,r','latents = betas[k]'),
 ('posterior latent covariance','(r,r,N)','O(K r^2 N) + O(K r^2 D)','every EM','linear K, quadratic r','step1 = latents'),
 ('M-step loadings','(D,r,N) contracted with (N,)','O(K D r N) eagerly materialized + O(K r^2 N+r^3+D r^2)','every EM','linear K,r for dominant tensor','self._lambdas[k] = np.dot'),
 ('M-step means and residuals','(D,N),(r,N)','K D r N matmul + O(KDN) weighted reductions','every EM','linear K; low-rank multiplication r','lambdalatents ='),
 ('parameter copies/serialization','public/private K D^2 arrays','O(KD^2) bytes copied, hashed, checked, written','once/fit','linear K; weak rank dependence','self.means = self._means.copy')]
 return [{'kernel':name,'input_shape':shape,'scaling':scaling,'frequency':freq,'K_rank_dependence':dep,'source':str(CORE.relative_to(ROOT))+':'+str(line(needle)),'measured_in_R2':'see timing microbenchmark; full fit NEVER run'} for name,shape,scaling,freq,dep,needle in specs]
def static():
 firewall();OUT.mkdir(exist_ok=True)
 cfg=read(ROOT/'configs/sp_u12b5b_mfa_selection.yaml');assert cfg['resource_contract']['max_process_bytes']==LIMITS['max_process_bytes']
 models=[sizes(N,D,k,r) for k in (1,2,4) for r in (4,8,16)]
 rows=[{k:v for k,v in m.items() if k!='stage_live_bytes'}|{s+'_bytes':b for s,b in m['stage_live_bytes'].items()} for m in models]
 csvwrite(ROOT/'outputs/sp_u12b5b_r2_9candidate_memory_model.csv',rows)
 csvwrite(ROOT/'outputs/sp_u12b5b_r2_author_allocation_lifetimes.csv',allocation_rows())
 csvwrite(ROOT/'outputs/sp_u12b5b_r2_training_kernel_audit.csv',kernel_rows())
 write(ROOT/'outputs/sp_u12b5b_r2_memory_lower_bounds.json',{'candidate_models':models,'worst_final_full_retrain':sizes(15000,D,4,16),
 'definition':'storage bytes and source simultaneously-live array sums; not measured full-fit RSS',
 'tiers':{'LOWER_BOUND':'single eagerly materialized array and mandatory input; guaranteed allocation bytes','SOURCE_BASED_LIVE_ESTIMATE':'maximum of mutually exclusive stage live-array sums','UNMEASURED_OVERHEAD':'native work buffers, ndarray packing, allocator retained pages, Python runtime; budget reserve is assumption'},
 'covariance_update_liveness':'4 K D^2 arrays at inverse expression peak incl existing cov/old inverse and two expression temporaries; last component E/M locals persist until method returns',
 'no_K_multiplier_for_large_cross_tensor':True,'no_full_scale_allocation_performed':True})
 write(ROOT/'outputs/sp_u12b5b_r2_compute_complexity_model.json',{'N':N,'D':D,'kernels':kernel_rows(),'per_component_likelihood_complex_MAC':D*D*N,
 'sum_K_over_nine':21,'sum_Kr_over_nine':196,'max_iterations':100,'earliest_stopping_iteration_count':7,
 'dense_D_by_D_inverse_each_EM':False,'dense_D_by_D_LU_for_slogdet_each_EM':True,
 'covariance_update_inversion':'Woodbury + batched r x r np.linalg.pinv + dense inverse construction',
 'full_fit_and_convergence_history_unmeasured':True})
 return models
