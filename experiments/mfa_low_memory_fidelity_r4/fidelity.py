"""EXPERIMENTAL_NOT_FORMAL_BASELINE. Observe immutable author/R3 methods.

The numerical tracer copies diagnostics only and never assigns model parameters.
No production runner imports this module; all inputs are synthetic.
"""
from pathlib import Path
import sys, types, importlib.util, ast, json, contextlib, io
import numpy as np
from threadpoolctl import threadpool_limits
from src.baselines.mfa_reference_adapter import (make_reference_model,
    fit_reference, MFAReferenceAdapter, reference_modules, FROZEN_POLICY)

ROOT=Path(__file__).resolve().parents[2]
ATOL=1e-10; RTOL=1e-9
R3=ROOT/'experiments/mfa_low_memory_prototype/portable/low_memory.py'
spec=importlib.util.spec_from_file_location('_immutable_r3_low_memory',R3)
low=importlib.util.module_from_spec(spec);spec.loader.exec_module(low)

def synthetic(N,D,K,r,seed):
    rng=np.random.default_rng(seed)
    def cn(shape):return (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
    means=cn((K,D))*.55
    loading=cn((K,D,r))*.20/np.sqrt(r)
    labels=np.arange(N)%K; rng.shuffle(labels)
    x=means[labels]+np.einsum('ndr,nr->nd',loading[labels],cn((N,r)))+cn((N,D))*.65
    return np.ascontiguousarray(x,dtype=np.complex128)

def model(K,r,optimized=False):
    m=make_reference_model(K,r)
    if optimized:
        m._EM_per_component=types.MethodType(low.em_step,m)
        m._log_multi_gauss_nodet=types.MethodType(low.log_gaussian_no_det,m)
    return m

def initialize(m,x,seed):
    # Exact author initialization; one thread ensures identical input parameters.
    from src.baselines.mfa_reference_adapter import recorded_initialization
    m.N,m.D=x.shape; m.rs=np.zeros((m.n_components,m.N))
    m._covs=np.zeros((m.n_components,m.D,m.D),dtype=np.complex128)
    m._inv_covs=np.zeros_like(m._covs)
    with threadpool_limits(1), recorded_initialization(seed):m._initialize(x)
    return m

def parameters(m):
    return {name:np.array(getattr(m,attr),copy=True) for name,attr in
        [('weights','amps'),('means','_means'),('loadings','_lambdas'),
         ('residuals','_psis'),('covariances','_covs'),('inverse_covariances','_inv_covs')]}

def inference_clone(m):
    q=make_reference_model(m.n_components,m.M);q.D=m.D;q.N=m.N
    for n in ('means','lambdas','psis','covs','inv_covs'):
        a={'means':'_means','lambdas':'_lambdas','psis':'_psis','covs':'_covs','inv_covs':'_inv_covs'}[n]
        setattr(q,n,np.array(getattr(m,a),copy=True))
    q.amps=np.array(m.amps,copy=True)
    return q

def posterior(m,x):
    # Author inference mutates private observation-domain buffers; use a clone.
    q=inference_clone(m);n_o=min(8,m.D);A=np.eye(m.D,dtype=np.complex128)[:n_o]
    y=np.array(x[:6,:n_o],copy=True);Cn=.37*np.eye(n_o,dtype=np.complex128)
    est=MFAReferenceAdapter(q,A).estimate(y,Cn)
    return {'posterior_estimator':est,'observation_responsibilities':q.predict_proba(y)}

def capture_step(m,x):
    fn=m._EM_per_component.__func__;code=fn.__code__;source=Path(code.co_filename).read_text(encoding='utf8')
    lines=source.splitlines();marks={}
    for i,line in enumerate(lines,1):
        s=line.strip()
        if s.startswith('self._means[k] =') or s.startswith('self._means[k]='):marks[i]='before_mean'
        if s.startswith('self._lambdas[k] =') or s.startswith('self._lambdas[k]='):marks[i]='before_loading'
        if s.startswith('self._psis[k] = np.clip') or s.startswith('self._psis[k]=np.clip'):marks[i]='raw_residual'
    assert set(marks.values())=={'before_mean','before_loading','raw_residual'}
    captured={}
    def trace(frame,event,arg):
        if frame.f_code is not code:return None
        if event=='line' and frame.f_lineno in marks:
            v=frame.f_locals;k=v['k'];tag=marks[frame.f_lineno]
            def keep(n,a):captured[f'component{k}/{n}']=np.array(a,copy=True)
            if tag=='before_mean':
                keep('old_beta',v['betas'][k]);keep('latent_means',v['latents'])
                keep('old_loading_latent',v.get('old_lambdalatents',v.get('lambdalatents')))
                keep('latent_base',v.get('latent_base',np.eye(m.M)-v['betas'][k]@m._lambdas[k]))
            elif tag=='before_loading':
                q=m.rs[k];Z=v['zeroed'];H=v['latents']
                keep('loading_numerator',v.get('cross') if 'cross' in v else np.dot(Z[:,None,:]*H[None,:,:].conj(),q))
                keep('latent_second_moment',v.get('gamma') if 'gamma' in v else np.dot(v['latent_covs'],q))
            else:keep('raw_residual',v['psis'])
        return trace
    previous=sys.gettrace();sys.settrace(trace)
    try:m._EM_per_component(x,m.PPCA)
    finally:sys.settrace(previous)
    captured.update(parameters(m));captured['responsibilities']=m.rs.copy()
    captured['log_likelihood']=m.logLs.copy();captured['objective']=np.array(m.logLs.sum())
    captured.update(posterior(m,x))
    assert all(np.all(np.isfinite(a)) for a in captured.values()),'NONFINITE_SYNTHETIC_STATE'
    return captured

def discrepancy(a,b):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(a.shape,b.shape)
    d=abs(a-b);den=abs(b);mask=den>ATOL
    normalized=d/(ATOL+RTOL*den)
    wi=int(np.argmax(d)) if d.size else 0
    return {'PASS':bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(normalized<=1)),
        'max_absolute':float(np.max(d,initial=0)),
        'max_relative_with_absolute_floor':float(np.max(d/np.maximum(den,ATOL),initial=0)),
        'max_relative_above_atol':float(np.max(d[mask]/den[mask],initial=0)),
        'max_tolerance_ratio':float(np.max(normalized,initial=0)),
        'worst_flat_index':wi,'bitwise':bool(np.array_equal(a,b))}

def run_case(case,optimized,threads,out):
    x=synthetic(case['N'],case['D'],case['K'],case['r'],case['data_seed'])
    m=model(case['K'],case['r'],optimized);states={};order=[];diagnostics=[]
    original_init=m._initialize;original_step=m._EM_per_component
    # fit_reference calls the untouched author fit and run_em, including cleanup.
    def observed_init(self,data):
        with threadpool_limits(1):original_init(data)
        st=parameters(self);st.update(posterior(self,data));states.update({'initial/'+k:v for k,v in st.items()});order.append('initial')
    def observed_step(self,data,PPCA):
        self._EM_per_component=original_step
        try:st=capture_step(self,data)
        finally:self._EM_per_component=types.MethodType(observed_step,self)
        tag=f'iteration{len(diagnostics):03d}';states.update({tag+'/'+k:v for k,v in st.items()});order.append(tag)
        obj=float(st['objective']);previous=diagnostics[-1]['objective'] if diagnostics else None
        ratio=None if previous is None else abs((obj-previous)/obj)
        diagnostics.append({'iteration':len(diagnostics),'objective':obj,'relative_change':ratio,
            'original_stop_test':False if previous is None else len(diagnostics)>5 and ratio<self.tol,
            'distance_from_tolerance':None if ratio is None else abs(ratio-self.tol)})
    if case['mode']=='short':
        initialize(m,x,case['initialization_seed']);st=parameters(m);st.update(posterior(m,x))
        states.update({'initial/'+k:v for k,v in st.items()});order.append('initial')
        for i in range(case['steps']):
            st=capture_step(m,x);tag=f'iteration{i:03d}';states.update({tag+'/'+k:v for k,v in st.items()});order.append(tag)
            diagnostics.append({'iteration':i,'objective':float(st['objective']),'relative_change':None,'original_stop_test':None,'distance_from_tolerance':None})
    else:
        m._initialize=types.MethodType(observed_init,m);m._EM_per_component=types.MethodType(observed_step,m)
        with contextlib.redirect_stdout(io.StringIO()) as captured_stdout:fit_reference(m,x,case['initialization_seed'])
        assert len(m.L_all)==len(diagnostics)
    np.savez_compressed(out/'states.npz',**states)
    status={'case':case,'optimized':optimized,'threads':threads,'iterations':len(diagnostics),'order':order,
        'diagnostics':diagnostics,'stopping_reason':'SHORT_STEP_DIAGNOSTIC' if case['mode']=='short' else
        ('ORIGINAL_STOP_TRIGGERED' if diagnostics[-1]['original_stop_test'] else 'MAXITER_100'),
        'finite':all(np.all(np.isfinite(v)) for v in states.values()),'input_sha256':__import__('hashlib').sha256(x.tobytes()).hexdigest(),
        'initialization_policy':FROZEN_POLICY,'extra_restarts':0}
    return status,m,x

def compare_case(reference,out,status):
    refstatus=json.loads((reference/'status.json').read_text(encoding='utf8'))
    rows=[]
    with np.load(reference/'states.npz',allow_pickle=False) as ref,np.load(out/'states.npz',allow_pickle=False) as got:
        if set(ref.files)!=set(got.files):
            return [],{'PASS':False,'reason':'HISTORY_LENGTH_OR_STATE_KEYS_DIFFER; NO_TRUNCATION',
                'reference_iterations':refstatus['iterations'],'optimized_iterations':status['iterations']}
        for key in ref.files:
            tag,tensor=key.split('/',1);v=discrepancy(got[key],ref[key])
            rows.append({'case':status['case']['id'],'K':status['case']['K'],'r':status['case']['r'],
                'D':status['case']['D'],'N':status['case']['N'],'threads':status['threads'],
                'phase':tag,'tensor':tensor,**v})
    stop=refstatus['iterations']==status['iterations'] and refstatus['stopping_reason']==status['stopping_reason']
    return rows,{'PASS':all(v['PASS'] for v in rows) and stop,'stopping_PASS':stop,
        'reference_iterations':refstatus['iterations'],'optimized_iterations':status['iterations'],
        'reference_reason':refstatus['stopping_reason'],'optimized_reason':status['stopping_reason'],
        'rows':len(rows),'atol':ATOL,'rtol':RTOL,'no_history_truncation':True}

def independent_identities(seed=42490001):
    rng=np.random.default_rng(seed);D,r,N=7,3,11
    cn=lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
    Z,H,V=cn((D,N)),cn((r,N)),cn((D,N));q=rng.uniform(.1,1,N);base=np.eye(r)*.7
    # Explicit indexed sums are independent of R3 contraction code.
    cross=np.array([[sum(q[n]*Z[d,n]*H[a,n].conjugate() for n in range(N)) for a in range(r)] for d in range(D)])
    gamma=np.array([[sum(q[n]*(base[a,b]+H[a,n]*H[b,n].conjugate()) for n in range(N)) for b in range(r)] for a in range(r)])
    quad=np.array([sum((Z[d,n].conjugate()*V[d,n]).real for d in range(D)) for n in range(N)])
    res=np.array([sum(q[n]*((Z[d,n]-V[d,n])*Z[d,n].conjugate()).real for n in range(N))/sum(q) for d in range(D)])
    checks={n:discrepancy(a,b) for n,a,b in [('loading',low.loading_numerator(Z,H,q),cross),
       ('latent',low.latent_moment(H,q,base),gamma),('quadratic',low.real_quadratic(Z,V),quad),
       ('residual',low.residual_moment(Z,V,q),res)]}
    Wold,Wnew=cn((D,r)),cn((D,r))*4;old=Wold@H;new=Wnew@H
    expected=np.real(((Z-old)*Z.conj())@q/sum(q))
    correct=low.residual_moment(Z,old,q);wrong=low.residual_moment(Z,new,q)
    checks['old_loading_residual_regression']={'PASS':discrepancy(correct,expected)['PASS'] and not discrepancy(wrong,expected)['PASS'],
        'wrong_new_loading_max_absolute':float(np.max(abs(wrong-expected))),
        'correct':discrepancy(correct,expected),'old_new_product_difference':float(np.max(abs(old-new)))}
    return {'PASS':all(v['PASS'] for v in checks.values()),'checks':checks,'seed':seed,'shape':[D,r,N]}

def boundary_stop_test():
    core,_,_=reference_modules();records=[]
    for ratio in (0.999999e-4,1.000001e-4):
        m=model(1,4);m.maxiter=100;m.verbose=False;m.L_all=[];count=[0]
        # Synthetic scalar-objective sentinel tests the untouched run_em branch.
        def step(self,data,PPCA):
            i=count[0];v=-1000 if i<6 else (-1000/(1-ratio) if i==6 else -1000/(1-ratio))
            self.logLs=np.array([v]);count[0]+=1
        m._EM_per_component=types.MethodType(step,m)
        core.MFA_cplx.run_em(m,np.zeros((1,64),dtype=np.complex128))
        records.append({'ratio':ratio,'iterations':count[0],'expected':7 if ratio<1e-4 else 8})
    return {'PASS':all(v['iterations']==v['expected'] for v in records),'records':records,
        'scope':'Untouched author stopping branch only; no trained model or scientific score'}

def inference_contract(m,x,out):
    q=inference_clone(m);n_o=min(8,m.D);A=np.eye(m.D,dtype=np.complex128)[:n_o];y=x[:6,:n_o];Cn=.37*np.eye(n_o,dtype=np.complex128)
    calls=[];terms=[];original=q.estimate;origterm=q._lmmse
    def spy(self,*args,**kwargs):
        v=kwargs['n_summands_or_proba'];calls.append({'type':type(v).__name__,'value':v});return original(*args,**kwargs)
    def termspy(self,*args,**kwargs):terms.append(1);return origterm(*args,**kwargs)
    q.estimate=types.MethodType(spy,q);q._lmmse=types.MethodType(termspy,q)
    result=MFAReferenceAdapter(q,A).estimate(y,Cn)
    serial=out/'serialized_synthetic';serial.mkdir()
    before=parameters(m)
    for n,a in before.items():np.save(serial/(n+'.npy'),a,allow_pickle=False)
    q2=make_reference_model(m.n_components,m.M);q2.D=m.D;q2.N=m.N
    attrs={'weights':'amps','means':'means','loadings':'lambdas','residuals':'psis','covariances':'covs','inverse_covariances':'inv_covs'}
    exact=True
    for n,a in before.items():
        loaded=np.load(serial/(n+'.npy'),allow_pickle=False);exact &= np.array_equal(a,loaded);setattr(q2,attrs[n],loaded)
    result2=MFAReferenceAdapter(q2,A).estimate(y,Cn)
    q3=inference_clone(m);hard=q3.estimate(y=y,Cn=Cn,A=A,n_summands_or_proba=1)
    d=float(np.max(abs(hard-result)))
    return {'PASS':exact and discrepancy(result2,result)['PASS'] and calls==[{'type':'float','value':1.0}] and len(terms)==len(y)*m.n_components,
        'float1_contract':calls,'component_lmmse_calls':len(terms),'expected_all_components':len(y)*m.n_components,
        'integer1_result_difference':d,'integer1_is_distinct_when_K_gt1':None if m.n_components==1 else d>ATOL,
        'serialization_bitwise_parameters':bool(exact),'serialized_inference':discrepancy(result2,result),
        'adapter_unchanged':True,'inference_formula':'immutable author MfaEstimator + immutable MFAReferenceAdapter',
        'synthetic_only':True,'no_target_metric':True}
