"""Call frozen author methods; diagnostic kernels never replace EM methods."""
from pathlib import Path
import os,sys,hashlib,time,json,contextlib
import numpy as np
from common import ROOT,HERE,ATOL,RTOL,protocol,threads,write,read
from src.baselines.mfa_reference_adapter import (make_reference_model,reference_modules,
    recorded_initialization,fit_reference,MFAReferenceAdapter,FROZEN_POLICY)

ATTRS={'weights':'amps','means':'_means','loadings':'_lambdas','residuals':'_psis',
       'covariances':'_covs','inverse_covariances':'_inv_covs'}
PUBLIC={'means':'_means','lambdas':'_lambdas','psis':'_psis','covs':'_covs','inv_covs':'_inv_covs'}

def seed(role):
    cfg=protocol();s=f"{cfg['seed_namespace']}|base={cfg['synthetic_seed_base']}|{role}"
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4],'big')
def array_hash(value):
    a=np.ascontiguousarray(value);h=hashlib.sha256();h.update(memoryview(a).cast('B'));return h.hexdigest()
def compare(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:return {'PASS':False,'reason':'SHAPE_OR_HISTORY_LENGTH_DIFFER','a_shape':list(a.shape),'b_shape':list(b.shape)}
    maximum=ratio=0.;passed=True
    aa=a.reshape(-1);bb=b.reshape(-1)
    for start in range(0,aa.size,131072):
        left,right=aa[start:start+131072],bb[start:start+131072]
        d=abs(left-right);normal=d/(ATOL+RTOL*abs(right))
        maximum=max(maximum,float(np.max(d,initial=0)));ratio=max(ratio,float(np.max(normal,initial=0)))
        passed &=bool(np.isfinite(left).all()and np.isfinite(right).all()and np.all(normal<=1))
    return {'PASS':passed,'max_absolute':maximum,'max_tolerance_ratio':ratio,'atol':ATOL,'rtol':RTOL}
def enforce_shape(N,D):
    if os.environ.get('R5_PERSONAL_SELFTEST')=='1':
        assert N<=512 and D<=128,'PERSONAL_COMPUTER_FULL_SCALE_FORBIDDEN'
    elif N>1024 or D>256:
        from windows_resources import capacity,hardware
        check=capacity(hardware());assert check['PASS'],'LAB_128GB_CAPACITY_GATE:'+str(check['reasons'])
def cn(rng,shape):return(rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
def synthetic(N,D,K,r,random_seed):
    enforce_shape(N,D);rng=np.random.default_rng(random_seed)
    means=cn(rng,(K,D))*.65;loading=cn(rng,(K,D,r))*.2/np.sqrt(r)
    labels=np.arange(N)%K;rng.shuffle(labels);x=np.empty((N,D),dtype=np.complex128)
    # Generator batching only. Native author EM receives the complete C-order array.
    for start in range(0,N,256):
        stop=min(N,start+256);ids=labels[start:stop]
        x[start:stop]=means[ids]+np.einsum('ndr,nr->nd',loading[ids],cn(rng,(stop-start,r)))+cn(rng,(stop-start,D))*.7
    assert x.flags.c_contiguous and x.dtype==np.complex128
    return x

class Observe:
    """Line/call tracing only: no assignments to model or original functions."""
    def __init__(self,callback=None):
        core,_,_=reference_modules();self.filename=core.__file__;self.path=Path(self.filename).resolve();self.lines=self.path.read_text(encoding='utf8').splitlines()
        self.callback=callback;self.events=[];self.previous=None;self.current={};self.step=-1;self.kmeans=[];self.kmeans_inner_pools=[];self.kmeans_centers=[]
    def emit(self,kind,frame,**extra):
        record={'kind':kind,'elapsed_seconds':time.perf_counter()-self.start,'step':self.step,**extra}
        self.events.append(record)
        if self.callback:self.callback(record)
    def tracer(self,frame,event,arg):
        if event=='call'and frame.f_code.co_name in('_kmeans_single_lloyd','_kmeans_single_elkan'):
            from threadpoolctl import threadpool_info
            self.kmeans_inner_pools.append(threadpool_info())
        if frame.f_code.co_filename!=self.filename:return None
        name=frame.f_code.co_name
        if event=='call':
            self.current[id(frame)]=time.perf_counter()
            if name=='_EM_per_component':self.step+=1
            if name in('_initialize','_EM_per_component','_calc_probs','_update_covs','_invert_cov_all'):
                self.emit('call',frame,function=name)
        elif event=='return'and name in('_initialize','_EM_per_component','_calc_probs','_update_covs','_invert_cov_all'):
            self.emit('return',frame,function=name,wall_seconds=time.perf_counter()-self.current.pop(id(frame),time.perf_counter()))
        elif event=='line':
            s=self.lines[frame.f_lineno-1].strip();v=frame.f_locals
            if name=='_initialize'and s.startswith('self._means ='):
                km=v['Kmeans'];info={'n_iter':int(km.n_iter_),'n_init':km.n_init,'init':km.init,
                    'random_state':km.random_state,'center_hash':array_hash(km.cluster_centers_)}
                info['native_KMeans_with_real_conversion_wall_seconds']=time.perf_counter()-self.current[id(frame)]
                from threadpoolctl import threadpool_info
                info['threadpools_after_native_KMeans']=threadpool_info()
                info['actual_threadpools_inside_native_KMeans']=self.kmeans_inner_pools.copy()
                info['nested_BLAS_cap_verified']=bool(self.kmeans_inner_pools)and all(v['num_threads']==1 for pools in self.kmeans_inner_pools for v in pools if v['user_api']=='blas')
                self.kmeans.append(info)
                self.kmeans_centers.append(km.cluster_centers_.copy())
                self.emit('native_KMeans_complete',frame,**info)
            if name=='_EM_per_component':
                tags=[('latents =','E_step_start'),('step1 =','latent_moment_start'),
                      ('lambdalatents =','old_loading_start'),('self._means[k] =','mean_update_start'),
                      ('self._lambdas[k] =','loading_contraction_start'),('psis =','loading_contraction_complete'),
                      ('self._psis[k] = np.clip','residual_update_complete'),('self._update_covs()','covariance_finalization_start')]
                for prefix,tag in tags:
                    if s.startswith(prefix):
                        m=v['self'];extra={'component':int(v.get('k',-1)),'line':frame.f_lineno}
                        if tag in('loading_contraction_start','loading_contraction_complete'):
                            extra.update(zeroed_shape=list(v['zeroed'].shape),latent_shape=list(v['latents'].shape),
                                broadcast_shape=[m.D,m.M,m.N],broadcast_dtype='complex128',expected_broadcast_bytes=m.D*m.M*m.N*16)
                        self.emit(tag,frame,**extra);break
        return self.tracer
    def __enter__(self):
        self.start=time.perf_counter();self.previous=sys.gettrace();sys.settrace(self.tracer);return self
    def __exit__(self,*args):sys.settrace(self.previous)

def initialize_native(m,x,initialization_seed,observer=None):
    # Exact prefix of author fit; technical qualification intentionally bounds updates.
    m.N=x.shape[0];m.D=x.shape[1]
    m.rs=np.zeros((m.n_components,m.N))
    m._covs=np.zeros((m.n_components,m.D,m.D),dtype=complex);m._inv_covs=np.zeros_like(m._covs)
    with recorded_initialization(initialization_seed):m._initialize(x)
    return m
def finalize_author(m):
    # Exact cleanup and copy lines in the original author fit, without model export.
    del m.latents,m.latent_covs,m.rs,m.kmeans_rs,m.betas,m.logLs
    m.means=m._means.copy();m.covs=m._covs.copy();m.inv_covs=m._inv_covs.copy()
    m.psis=m._psis.copy();m.lambdas=m._lambdas.copy()
    return m
def finite_model(m,public=False):
    names=('means','covs','inv_covs','psis','lambdas','amps')if public else tuple(ATTRS.values())
    return all(np.isfinite(getattr(m,name)).all()for name in names)
def parameters(m):return {name:np.array(getattr(m,attr),copy=True)for name,attr in ATTRS.items()}
def clone_for_inference(m):
    q=make_reference_model(m.n_components,m.M);q.D=m.D;q.N=m.N;q.amps=m.amps.copy()
    for name,attr in PUBLIC.items():setattr(q,name,np.array(getattr(m,attr),copy=True))
    return q
def posterior_independent(m,x):
    D=m.D;no=min(4,D);A=np.eye(D,dtype=complex)[:no];y=x[:6,:no];Cn=.37*np.eye(no)
    out=np.zeros((len(y),D),dtype=complex);logs=[];estimates=[]
    for k in range(m.n_components):
        C=m._covs[k];mu=m._means[k];Cy=A@C@A.conj().T+Cn;inv=np.linalg.pinv(Cy,hermitian=True)
        diff=y-mu[:no];_,logdet=np.linalg.slogdet(Cy)
        logs.append(np.log(m.amps[k])-logdet-np.real(np.sum(diff.conj()*(inv@diff.T).T,axis=1)))
        estimates.append(mu[None,:]+(C@A.conj().T@inv@diff.T).T)
    log=np.stack(logs,axis=1);p=np.exp(log-log.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
    for k in range(m.n_components):out+=p[:,k,None]*estimates[k]
    q=clone_for_inference(m);calls=[];terms=[];original=q.estimate;term=q._lmmse
    import types
    def observe(self,*args,**kwargs):calls.append((type(kwargs['n_summands_or_proba']).__name__,kwargs['n_summands_or_proba']));return original(*args,**kwargs)
    def lmmse(self,*args,**kwargs):terms.append(1);return term(*args,**kwargs)
    q.estimate=types.MethodType(observe,q);q._lmmse=types.MethodType(lmmse,q)
    actual=MFAReferenceAdapter(q,A).estimate(y,Cn)
    check=compare(actual,out);check.update(float1_calls=calls,component_calls=len(terms),expected_component_calls=len(y)*m.n_components)
    check['PASS'] &=calls==[('float',1.0)]and len(terms)==len(y)*m.n_components
    return actual,check
def fidelity_state(K,r):
    cfg=protocol()['fidelity_shape'];x=synthetic(cfg['N'],cfg['D'],K,r,seed(f'fidelity_data_K{K}_r{r}'))
    core,estimator,_=reference_modules();direct=estimator.MfaEstimator(n_components=K,latent_dim=r,**FROZEN_POLICY)
    adapted=make_reference_model(K,r);state={};checks=[]
    for label,m in(('direct',direct),('adapter',adapted)):
        observer=Observe()
        with observer:initialize_native(m,x,seed(f'fidelity_init_K{K}_r{r}'))
        local={'initial/'+n:a for n,a in parameters(m).items()}
        for step in range(2):
            m._EM_per_component(x,m.PPCA)
            local.update({f'step{step}/'+n:a for n,a in parameters(m).items()})
            local[f'step{step}/responsibilities']=m.rs.copy();local[f'step{step}/objective']=np.array(m.logLs.sum())
        estimate,post=posterior_independent(m,x);local['posterior']=estimate
        checks.append(post);local['KMeans_centers']=observer.kmeans_centers[0]
        if label=='direct':state=local
        else:
            for n,a in local.items():
                checks.append({'PASS':str(a)==str(state[n]),'field':n}if a.dtype.kind in('U','S')else{'field':n,**compare(a,state[n])})
    assert all(c['PASS']for c in checks),'AUTHOR_ADAPTER_OR_POSTERIOR_PARITY_FAILED'
    return x,state,checks

def synthetic_parameters(K,D,r):
    enforce_shape(6,D);rng=np.random.default_rng(seed(f'inference_parameters_K{K}_D{D}_r{r}'))
    m=make_reference_model(K,r);m.D=D;m.N=6
    m._means=cn(rng,(K,D))*.3;m._lambdas=cn(rng,(K,D,r))*.15/np.sqrt(r)
    m._psis=np.ones((K,D),dtype=float);m.amps=np.arange(1,K+1,dtype=float);m.amps/=m.amps.sum()
    m._update_covs();return m

def benchmark_kernels():
    cfg=protocol();a=cfg['kernel_shape'];D,N,r=a['D'],a['N'],a['rank']
    rng=np.random.default_rng(seed('fixed_kernel_input'));x=synthetic(N,D,4,r,seed('fixed_kernel_data'))
    m=synthetic_parameters(1,D,r);Z=x.T-m._means[0,:,None];H=cn(rng,(r,N));q=rng.uniform(.1,1,N)
    beta=m._lambdas[0].conj().T@m._inv_covs[0];old=m._lambdas[0]@H
    latent_cov=np.eye(r)[:,:,None]-(beta@m._lambdas[0])[:,:,None]+H[:,None,:]*H[None,:,:].conj()
    from scipy.linalg import inv
    def covariance():m._update_covs();return {'covariances':m._covs,'inverse_covariances':m._inv_covs}
    def latent_old():
        h=beta@Z;product=m._lambdas[0]@h;mean=np.sum(q*(x.T-product),axis=1)/np.sum(q)
        return {'latents':h,'old_loading_latents':product,'mean':mean}
    def loading():return np.dot(np.dot(Z[:,None,:]*H[None,:,:].conj(),q),inv(np.dot(latent_cov,q)))
    def residual():
        psis=np.real(np.dot((Z-old)*Z.conj(),q)/np.sum(q));psis=np.clip(psis,1e-6,np.inf)
        return np.mean(psis)*np.ones(D)
    kcfg=cfg['kmeans_shape'];kx=synthetic(kcfg['N'],kcfg['D'],kcfg['K'],kcfg['rank'],seed('kmeans_input'))
    def kmeans():
        km=make_reference_model(kcfg['K'],kcfg['rank']);initialize_native(km,kx,seed('kmeans_init'))
        return parameters(km)
    def fit_case(name):
        c=cfg[name];data=synthetic(c['N'],c['D'],c['K'],c['rank'],seed(name+'_input'))
        model=make_reference_model(c['K'],c['rank']);fit_reference(model,data,seed(name+'_init'))
        return {'means':model.means,'loadings':model.lambdas,'covariances':model.covs,'inverse_covariances':model.inv_covs,
                'residuals':model.psis,'weights':model.amps,'history':np.asarray(model.L_all)}
    return {'slogdet4032':lambda:dict(zip(('sign','logdet'),np.linalg.slogdet(m._covs))),
            'dense_likelihood_GEMM':lambda:m._log_multi_gauss_nodet(0,x),
            'covariance_Woodbury':covariance,'beta_contraction':lambda:m._lambdas[0].conj().T@m._inv_covs[0],
            'latent_and_old_loading':latent_old,'latent_moment':lambda:np.dot(latent_cov,q),
            'loading_contraction':loading,'residual_update':residual,'native_KMeans_initialization':kmeans,
            'small_complete_native_EM':lambda:fit_case('small_em'),'medium_complete_native_EM':lambda:fit_case('medium_em')},array_hash(x)

def flatten_output(output):return output if isinstance(output,dict)else{'value':np.asarray(output)}
def checked_timing(fn,anchor,save=False):
    fn() # one frozen warmup; output is discarded
    start=time.perf_counter();out=flatten_output(fn());elapsed=time.perf_counter()-start
    checks={};metadata={}
    for name,value in out.items():
        a=np.asarray(value);metadata[name]={'shape':list(a.shape),'dtype':a.dtype.str,'sha256':array_hash(a)}
        if save:np.save(anchor/(name+'.npy'),a,allow_pickle=False)
        else:
            b=np.load(anchor/(name+'.npy'),mmap_mode='r',allow_pickle=False);checks[name]=compare(a,b);del b
    if not save:assert all(v['PASS']for v in checks.values()),'THREAD_NUMERICAL_PARITY_FAILURE'
    return {'wall_seconds':elapsed,'checks':checks,'output_metadata':metadata,
            'native_iterations':len(out['history'])if'history'in out else None,
            'warmup_calls':1,'timed_calls':1,'PASS':save or all(v['PASS']for v in checks.values())}

def full_smoke(N,job):
    cfg=protocol()['full_smoke'];assert N in(cfg['N_A'],cfg['N_B']);assert cfg['updates']==2
    assert os.environ.get('R5_PERSONAL_SELFTEST')!='1','PERSONAL_FULL_SCALE_FORBIDDEN'
    x=synthetic(N,cfg['D'],cfg['K'],cfg['rank'],seed(f'smoke_data_N{N}'))
    m=make_reference_model(cfg['K'],cfg['rank']);events=[];phase_id='initialization'
    def note(event):
        event['full_phase']=phase_id
        events.append(event);write(job/'phase.json',event)
        with(job/'allocation_trace.jsonl').open('a',encoding='utf8')as f:f.write(json.dumps(event)+'\n')
    observation=Observe(note);begin=time.perf_counter()
    with observation:initialize_native(m,x,seed(f'smoke_init_N{N}'))
    init=time.perf_counter()-begin;initial_hashes={n:array_hash(getattr(m,a))for n,a in ATTRS.items()}
    durations=[];objectives=[]
    for step in range(2):
        phase_id=f'EM_update_{step+1}'
        write(job/'phase.json',{'phase':'bounded_native_full_batch_EM','step':step,'N':N})
        observation=Observe(note);begin=time.perf_counter()
        with observation:m._EM_per_component(x,m.PPCA)
        durations.append(time.perf_counter()-begin);objectives.append(float(m.logLs.sum()))
        assert finite_model(m),'NONFINITE_NATIVE_EM_STOP'
    before={n:array_hash(getattr(m,a))for n,a in ATTRS.items()};begin=time.perf_counter();finalize_author(m);finish=time.perf_counter()-begin
    assert finite_model(m,True),'FINALIZATION_NONFINITE_STOP'
    categories={v['kind']for v in events}
    loading_done=[v for v in events if v['kind']=='loading_contraction_complete']
    coverage={'native_KMeans':bool([v for v in events if v['kind']=='native_KMeans_complete']),
        'likelihood':any(v.get('function')=='_calc_probs'and v['kind']=='return'for v in events),
        'E_step':'E_step_start'in categories,'M_step':len(loading_done)==8,
        'D_r_N_broadcast_source_expression':len(loading_done)==8 and all(v['broadcast_shape']==[4032,16,N]for v in loading_done),
        'residual_update':sum(v['kind']=='residual_update_complete'for v in events)==8,
        'covariance_Woodbury':any(v.get('function')=='_invert_cov_all'and v['kind']=='return'for v in events),
        'author_finalization':all(array_hash(getattr(m,public))==before[n]for n,a in ATTRS.items()for public in
            [{'weights':'amps','means':'means','loadings':'lambdas','residuals':'psis','covariances':'covs','inverse_covariances':'inv_covs'}[n]])}
    assert all(coverage.values()),'INCOMPLETE_ALLOCATION_PATH_STOP'
    return {'PASS':True,'N':N,'D':4032,'K':4,'rank':16,'dtype':'complex128','updates':2,
        'initialization_seconds':init,'EM_iteration_seconds':durations,'finalization_seconds':finish,
        'coverage':coverage,'unmeasured_paths':['iterations after update 2','formal-data-dependent KMeans/convergence histories'],
        'input_sha256':array_hash(x),'initial_parameter_hashes':initial_hashes,'last_parameter_hashes':before,
        'native_KMeans':next(v for v in events if v['kind']=='native_KMeans_complete'),
        'phase_events':events,
        'objectives_synthetic_technical_only':objectives,'trace_observer_changes_model':False,
        'model_saved':False,'NMSE_computed':False,'maxiter_unchanged':m.maxiter,'tol_unchanged':m.tol}
