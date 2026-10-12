"""Unmodified author packages + thin full-grid selector adapter + independent reference."""
import importlib.metadata as metadata
from pathlib import Path
import sys
import time
import warnings
import ctypes
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u3a_common as c
sys.dont_write_bytecode=True
SOURCE=c.OUT/'sp_u3a_sources'
for name in ('cplx-mfa','MFA_estimator'):
    pin=c.load(SOURCE/name/'repository_pin.json')
    assert all(c.sha(SOURCE/name/p)==h for p,h in pin['files'].items())
    sys.path.insert(0,str(SOURCE/name/'src'))
# Reuse installed, frozen dependencies without installation or environment writes.
sys.path.append(str(ROOT/'.venv-gmm/Lib/site-packages'))
import numpy as np
from mfa_estimator import MfaEstimator
from threadpoolctl import threadpool_limits

def protocol():return c.load(ROOT/'configs/sp_u3a_protocol.json')
def cn(shape,rng):return (rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2)
def error(a,b):return float(np.max(abs(a-b)))

def reference(means,covariances,weights,y,A,Cn):
    """Independent direct proper-complex equations; no author helpers or pinv."""
    component=[];logs=[]
    for mu,C,pi in zip(means,covariances,weights):
        S=A@C@A.conj().T+Cn;diff=(y-mu@A.T).T
        solved=np.linalg.solve(S,diff)
        sign,ld=np.linalg.slogdet(S);assert sign.real>0
        logs.append(np.log(pi)-len(S)*np.log(np.pi)-ld.real-np.sum(diff.conj()*solved,axis=0).real)
        component.append((mu[:,None]+C@A.conj().T@solved).T)
    logs=np.array(logs).T;likelihood=np.exp(logs-logs.max(axis=1,keepdims=True))
    probabilities=likelihood/likelihood.sum(axis=1,keepdims=True)
    component=np.stack(component,axis=1)
    return probabilities,component,np.sum(probabilities[:,:,None]*component,axis=1)

def known_model(K=2,N=8,L=2):
    rng=np.random.default_rng(202603201);model=MfaEstimator(K,L,random_state=202603201,verbose=False)
    model.means_=cn((K,N),rng)*.6;model.loadings_=cn((K,N,L),rng)*.3
    model.noise_variances_=np.arange(K*N,dtype=float).reshape(K,N)/100+.4
    model.covariances_=np.stack([w@w.conj().T+np.diag(psi) for w,psi in zip(model.loadings_,model.noise_variances_)])
    model.precisions_=np.stack([np.linalg.solve(C,np.eye(N)) for C in model.covariances_])
    model.weights_=np.array([.35,.65]) if K==2 else np.ones(1)
    return model

def compare(model,y,A,Cn):
    refp,refcomponent,refestimate=reference(model.means_,model.covariances_,model.weights_,y,A,Cn)
    obs=model._prepare_observation_model(A,Cn)
    p=model._predict_observation_proba(y,obs)
    component=np.stack([[model._lmmse(v,A,k,obs.precisions[k]) for k in range(model.n_components)] for v in y])
    estimate=model.estimate(y,Cn,A,n_summands_or_proba=1.0)
    return dict(responsibility_error=error(p,refp),component_estimate_error=error(component,refcomponent),
      mixture_estimate_error=error(estimate,refestimate),responsibility_sum_error=error(p.sum(axis=1),np.ones(len(y))),
      dtype=str(estimate.dtype),finite=bool(np.isfinite(estimate).all()))

def mapping():
    from src.spatial_information import observation_geometry
    g=observation_geometry();A=g.selector.toarray().astype(np.complex128)
    assert A.shape==(64,4096) and g.data.sum()==3968
    assert np.count_nonzero(A)==64 and np.all(A.sum(axis=1)==1)
    return g,A

def estimate_data(model,y,sigma2):
    # Online entry has no channel truth, labels, target covariance or future data.
    if not np.isfinite(sigma2) or sigma2<=0:raise ValueError('positive frozen complex variance required')
    g,A=mapping();out=model.estimate(y,np.eye(64)*sigma2,A,n_summands_or_proba=1.0)
    return out[:,g.data]

def mapping_check(H):
    from src.link import PILOT_SYMBOLS,apply_frequency_mimo_channel
    from src.pilots import map_pilot_symbols
    from src.estimators import pilot_ls_estimate
    g,A=mapping();x=map_pilot_symbols(g.layout,PILOT_SYMBOLS)
    assert np.all(x[g.layout.mute_mask]==0)
    clean=apply_frequency_mimo_channel(H,x)
    ls,_=pilot_ls_estimate(clean,x,g.layout.pilot_mask)
    truth=H.ravel(order='C');observed=ls.ravel(order='C')[g.selected]
    assert error(A@truth,observed)<1e-12
    # Fixed complex noise atom: actual owner division equals diagonal phase rotation.
    rng=np.random.default_rng(202603202);noise=cn(clean.shape,rng)*np.sqrt(.37)
    noisy,_=pilot_ls_estimate(clean+noise,x,g.layout.pilot_mask)
    rx,tx,k,t=np.unravel_index(g.selected,(2,2,64,16))
    rotated=noise[rx,k,t]/PILOT_SYMBOLS[tx]
    rotation=np.diag(1/PILOT_SYMBOLS[tx])
    covariance=.37*(rotation@rotation.conj().T)
    return dict(noiseless_error=error(A@truth,observed),owner_division_noise_error=error(noisy.ravel()[g.selected]-observed,rotated),
      noise_covariance_error=error(covariance,.37*np.eye(64)),selected_count=64,data_count=3968,muted_channel_count=64,
      data_extraction_exact=bool(np.array_equal(truth[g.data],H.reshape(-1,order='C')[g.data])),
      selector_sha256=c.hashlib.sha256(g.selected.tobytes()).hexdigest(),pilot_symbols=[[float(a.real),float(a.imag)] for a in PILOT_SYMBOLS])

def toy_results():
    cfg=protocol()['toy'];rng=np.random.default_rng(cfg['seed']);model=known_model()
    A=cn((3,8),rng)/np.sqrt(8);y=cn((9,3),rng)
    n=cn((3,3),rng);colored=n@n.conj().T*.1+.2*np.eye(3)
    fixtures=[]
    for label,op,Cn in [('rectangular_white',A,.37*np.eye(3)),('rectangular_colored',A,colored),
       ('selector',np.eye(8)[[0,3,7]],.37*np.eye(3)),('identity',np.eye(8),.37*np.eye(8))]:
        yy=y if len(op)==3 else cn((9,8),rng)
        fixtures.append(dict(case=label,**compare(model,yy,op,Cn)))
    single=known_model(K=1);single.means_=np.zeros((1,8),complex)
    single.covariances_=np.array([2*np.eye(8,dtype=complex)]);yy=cn((9,8),rng)
    k1=single.estimate(yy,.5*np.eye(8),np.eye(8),1.0)
    noise_check=dict(argument='Cn=E[n nH];complex variance,not real variance/std',expected_gain=.8,
      K1_LMMSE_error=error(k1,.8*yy),wrong_half_variance_error=error(single.estimate(yy,.25*np.eye(8),np.eye(8),1.0),.8*yy),
      real_imag_variance='.25each forcomplexsigma2=.5')
    atoms=cn((200000,),rng)*np.sqrt(.5)
    noise_check.update(empirical_complex_variance=float(np.mean(abs(atoms)**2)),
      empirical_real_variance=float(np.var(atoms.real)),empirical_imag_variance=float(np.var(atoms.imag)))
    # True author .fit path on small synthetic mixture; formula checked on learned parameters.
    labels=rng.choice(2,size=cfg['train_samples'],p=model.weights_)
    training=np.stack([model.means_[k]+np.linalg.cholesky(model.covariances_[k])@cn((8,),rng) for k in labels])
    fitted=[];warns=[];start=time.perf_counter()
    with threadpool_limits(limits=4):
        for _ in range(2):
            f=MfaEstimator(2,2,max_iter=cfg['max_iter'],tol=1e-4,random_state=cfg['seed'],verbose=False)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always');f.fit(training)
            fitted.append(f);warns.append([str(a.message) for a in w])
        fitted_check=compare(fitted[0],y,A,.37*np.eye(3))
    assert all(np.array_equal(getattr(fitted[0],key),getattr(fitted[1],key)) for key in ('means_','loadings_','noise_variances_','weights_','covariances_'))
    for row in fixtures+[fitted_check]:
        assert row['finite'] and max(row[k] for k in ('responsibility_error','component_estimate_error','mixture_estimate_error','responsibility_sum_error'))<cfg['tolerance']
    assert noise_check['K1_LMMSE_error']<1e-12 and noise_check['wrong_half_variance_error']>.01
    assert abs(noise_check['empirical_complex_variance']-.5)<.0075
    template=np.arange(4096).reshape(2,2,64,16).astype(complex)*(1+.2j)
    return dict(passed=True,fixtures=fixtures,noise=noise_check,learned_model_formula=fitted_check,
      actual_fit=dict(shape=[256,8],fits=2,iterations=[len(f.lower_bound_history_) for f in fitted],warnings=warns,
       runtime_seconds=time.perf_counter()-start,deterministic_exact=True),mapping=mapping_check(template),
      no_scientific_performance_claim=True)

def process_memory():
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(name,ctypes.c_size_t) for name in
          ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
           'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    p=Counters();p.cb=ctypes.sizeof(p)
    ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    handle=ctypes.windll.kernel32.GetCurrentProcess()
    assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle),ctypes.byref(p),p.cb)
    return dict(peak_working_set_bytes=int(p.PeakWorkingSetSize),working_set_bytes=int(p.WorkingSetSize),peak_pagefile_bytes=int(p.PeakPagefileUsage))
