"""Small, outcome-blind analytical/reference cases; never loads channel data."""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
from pathlib import Path
import sys,json,io,contextlib,runpy,inspect,copy,hashlib
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Reuse only already installed optional packages, after locking production NumPy/SciPy.
sys.path.append(str(ROOT/'.venv-gmm/Lib/site-packages'))
import sklearn
import threadpoolctl
from scipy.special import logsumexp
from src.baselines.mfa_reference_adapter import (
    reference_modules,make_reference_model,fit_reference,recorded_initialization,
    MFAReferenceAdapter,ProjectPilotInterface,ProjectMFAReceiver,tensor_to_vector,
    vector_to_tensor,FROZEN_POLICY,REFERENCE)

ATOL,RTOL=1e-10,1e-9
DATA_SEED,INIT_SEED=4190123500,4190123501

def cnormal(rng,shape):
    return (rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2)

def error(a,b):
    a,b=np.asarray(a),np.asarray(b)
    absolute=float(np.max(abs(a-b),initial=0))
    relative=float(np.max(abs(a-b)/np.maximum(abs(b),ATOL),initial=0))
    return dict(max_absolute_difference=absolute,max_relative_difference=relative,
                relative_denominator='max(abs(reference),1e-10)',PASS=bool(np.allclose(a,b,atol=ATOL,rtol=RTOL)))

def known_model(K=2,D=6,L=2):
    rng=np.random.default_rng(DATA_SEED+K+D+L)
    model=make_reference_model(K,L);model.D=D;model.N=11
    model.means=cnormal(rng,(K,D))*.4
    model.lambdas=cnormal(rng,(K,D,L))*.25
    model.psis=np.repeat(np.linspace(.5,.9,K)[:,None],D,axis=1)
    model.amps=np.arange(1,K+1,dtype=float);model.amps/=model.amps.sum()
    model.covs=model.lambdas@model.lambdas.conj().transpose(0,2,1)
    model.covs+=np.array([np.diag(p) for p in model.psis])
    model.inv_covs=np.linalg.inv(model.covs)
    model._means=model.means.copy();model._lambdas=model.lambdas.copy()
    model._psis=model.psis.copy();model._covs=model.covs.copy();model._inv_covs=model.inv_covs.copy()
    return model

def dense_posterior(model,y,Cn,A):
    """Independent direct solve of proper-complex Eq10-12 with general A."""
    y=np.asarray(y);N,q=y.shape;K,D=model.means.shape
    logp=np.empty((N,K));component=np.empty((N,K,D),complex)
    for k in range(K):
        S=A@model.covs[k]@A.conj().T+Cn
        sign,logdet=np.linalg.slogdet(S);assert np.allclose(sign,1)
        v=y-model.means[k]@A.T
        solved=np.linalg.solve(S,v.T)
        logp[:,k]=np.log(model.amps[k])-q*np.log(np.pi)-logdet-np.real(np.sum(v.conj()*solved.T,axis=1))
        component[:,k,:]=model.means[k]+(model.covs[k]@A.conj().T@solved).T
    ll=logsumexp(logp,axis=1)
    resp=np.exp(logp-ll[:,None])
    final=np.sum(resp[:,:,None]*component,axis=1)
    return resp,component,final,ll

def posterior_case(K=2,kind='selector'):
    model=known_model(K);rng=np.random.default_rng(DATA_SEED+17)
    if kind=='identity':A=np.eye(6,dtype=complex)
    elif kind=='selector':A=np.eye(6,dtype=complex)[[0,2,5]]
    else:A=cnormal(rng,(3,6))*.3
    q=len(A);noise=.37*np.eye(q,dtype=complex)
    if kind=='colored':
        B=cnormal(rng,(q,q))*.2;noise+=B@B.conj().T
    y=cnormal(rng,(11,q))
    expected_r,expected_components,expected,expected_ll=dense_posterior(model,y,noise,A)
    raw=model.estimate(y=y,Cn=noise,A=A,n_summands_or_proba=1.0)
    adapter=MFAReferenceAdapter(model,A).estimate(y,noise)
    raw_r=model.predict_proba(y)
    component=np.stack([np.stack([model._lmmse(yi,A,k,model._inv_covs[k]) for k in range(K)]) for yi in y])
    report=dict(K=K,D=6,L=2,q=q,kind=kind,raw_adapter=error(adapter,raw),responsibilities=error(raw_r,expected_r),component_means=error(component,expected_components),final=error(adapter,expected),finite=bool(np.all(np.isfinite(adapter))),responsibility_sum_error=float(np.max(abs(raw_r.sum(axis=1)-1))),weights_sum=float(model.amps.sum()),dtype=str(adapter.dtype))
    if K>1:
        rawll=np.array([model._log_sum(np.array([np.log(model.amps[k])+model._log_multi_gauss(k,y) for k in range(K)]))])[0]
        report['noisy_log_likelihood']=error(rawll,expected_ll)
    else:
        report['K1_likelihood_squeeze_note']='Original private noisy likelihood uses squeezed mean; not exposed/used for selection. K=1 responsibilities and final conditional estimate verified independently.'
    report['PASS']=all(v['PASS'] for v in report.values() if isinstance(v,dict) and 'PASS' in v) and report['finite'] and report['responsibility_sum_error']<ATOL
    return report

def dense_em_step(model,data):
    """Independently reproduce the exact original author update ordering."""
    K,D=model.means.shape;L=model.M
    r,_,_,ll=dense_posterior(model,data,np.zeros((D,D)),np.eye(D))
    mu=model.means.copy();W=model.lambdas.copy();psi=model.psis.copy();pi=model.amps.copy()
    means=[];loadings=[];noises=[]
    for k in range(K):
        C=model.covs[k];beta=np.linalg.solve(C,W[k]).conj().T
        z=(data-mu[k])@beta.T
        Rz=np.eye(L)-beta@W[k]
        old_Wz=z@W[k].T
        Nk=r[:,k].sum()
        newmean=np.sum(r[:,k,None]*(data-old_Wz),axis=0)/Nk
        zeroed=data-newmean
        cross=(zeroed.T*r[:,k])@z.conj()
        latent=Nk*Rz+(z.T*r[:,k])@z.conj()
        newW=np.linalg.solve(latent.T,cross.T).T
        newpsi=np.real(np.sum(r[:,k,None]*(zeroed-old_Wz)*zeroed.conj(),axis=0)/Nk)
        newpsi=np.clip(newpsi,1e-6,np.inf)
        if model.PPCA:newpsi=np.ones(D)*newpsi.mean()
        means.append(newmean);loadings.append(newW);noises.append(newpsi)
    means,W,psi=np.array(means),np.array(loadings),np.array(noises)
    C=W@W.conj().transpose(0,2,1)+np.array([np.diag(x) for x in psi])
    return dict(means=means,lambdas=W,psis=psi,amps=r.mean(axis=0),covs=C,inv_covs=np.linalg.inv(C),responsibilities=r.T,log_likelihood=ll)

def em_case():
    model=known_model(K=2,D=8,L=2)
    data=cnormal(np.random.default_rng(DATA_SEED),(160,8));model.N=len(data)
    expected=dense_em_step(model,data)
    model._EM_per_component(data,True)
    names={'means':'_means','lambdas':'_lambdas','psis':'_psis','amps':'amps','covs':'_covs','inv_covs':'_inv_covs','responsibilities':'rs','log_likelihood':'logLs'}
    checks={n:error(getattr(model,attr),expected[n]) for n,attr in names.items()}
    return dict(PASS=all(x['PASS'] for x in checks.values()),checks=checks,interpretation='Exact author sequence, including old-loading residual; no monotonicity theorem claimed')

def initialization_case():
    core,_,ut=reference_modules();data=cnormal(np.random.default_rng(DATA_SEED),(160,8))
    m=make_reference_model(2,2);m.N,m.D=data.shape
    with recorded_initialization(INIT_SEED):m._initialize(data)
    # Independent replay of precisely the original draw order and official KMeans call.
    with recorded_initialization(INIT_SEED):
        km=sklearn.cluster.KMeans(n_clusters=2,n_init=1).fit(np.concatenate([data.real,data.imag],axis=1))
        mu=km.cluster_centers_[:,:8]+1j*km.cluster_centers_[:,8:]
        W=(np.random.randn(2,8,2)+1j*np.random.randn(2,8,2))/np.sqrt(1e6)/np.sqrt(2)
        psi=np.tile(np.var(data,axis=0)[None],(2,1))
        pi=np.random.rand(2);pi/=pi.sum()
    checks={n:error(a,b) for n,a,b in [('means',m._means,mu),('loadings',m._lambdas,W),('residual',m._psis,psi),('weights',m.amps,pi)]}
    state_before=np.random.get_state()
    with recorded_initialization(INIT_SEED):np.random.rand(3)
    state_after=np.random.get_state()
    restored=state_before[0]==state_after[0] and np.array_equal(state_before[1],state_after[1]) and state_before[2:]==state_after[2:]
    return dict(PASS=all(x['PASS'] for x in checks.values()) and restored,checks=checks,global_RNG_restored=restored,seed=INIT_SEED)

def seed_case():
    data=cnormal(np.random.default_rng(DATA_SEED),(160,8))
    raw,wrapped,replay=[make_reference_model(2,2) for _ in range(3)]
    outputs=[]
    for m,direct in [(raw,True),(wrapped,False),(replay,False)]:
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            if direct:
                with recorded_initialization(INIT_SEED):m.fit(data.copy())
            else:fit_reference(m,data.copy(),INIT_SEED)
        outputs.append(stream.getvalue())
    attrs=('means','lambdas','psis','amps','covs','inv_covs','L_all')
    reference={n:error(getattr(wrapped,n),getattr(raw,n)) for n in attrs}
    repeat={n:error(getattr(replay,n),getattr(wrapped,n)) for n in attrs}
    history=np.array(raw.L_all)
    delta=np.diff(history)
    return dict(PASS=bool(all(x['PASS'] for x in list(reference.values())+list(repeat.values())) and np.all(np.isfinite(history))),data_seed=DATA_SEED,initialization_seed=INIT_SEED,training_seed=INIT_SEED,restart_seed=None,outer_fits=1,reference_adapter=reference,replay=repeat,bitwise_replay=all(np.array_equal(getattr(replay,n),getattr(wrapped,n)) for n in attrs),iterations=len(history),history_finite=bool(np.all(np.isfinite(history))),negative_objective_steps=int(np.sum(delta<0)),min_objective_change=float(delta.min()),strict_monotonicity_required=False,author_warnings=outputs,final_residual_isotropic=bool(np.all(raw.psis==raw.psis[:,:1])),no_target_data=True)

def vector_case():
    interface=ProjectPilotInterface();rng=np.random.default_rng(DATA_SEED)
    tensor=cnormal(rng,(3,2,2,64,16))
    full=tensor_to_vector(tensor);back=vector_to_tensor(full)
    joint,side=interface.offline_pack(tensor);again=interface.offline_unpack(joint,side)
    return dict(PASS=np.array_equal(back,tensor) and np.array_equal(again,tensor),full4096_bitwise_roundtrip=full.tobytes()==tensor.tobytes() and back.tobytes()==tensor.tobytes(),joint4032_with_sidecar64_bitwise_roundtrip=again.tobytes()==tensor.tobytes(),joint_shape=list(joint.shape),sidecar_shape=list(side.shape),selector_matches_full=bool(np.array_equal(joint@interface.A.T,full[:,interface.selected_full])),data_order_matches=bool(np.array_equal(joint[:,interface.data_positions],full[:,interface.data_full])))

def observation_case():
    from src.spatial_information import observation_geometry
    from src.pilots import map_pilot_symbols
    from src.link import apply_frequency_mimo_channel,PILOT_SYMBOLS
    from src.estimators import pilot_ls_estimate
    g=observation_geometry();interface=ProjectPilotInterface();rng=np.random.default_rng(DATA_SEED)
    H=cnormal(rng,(2,2,64,16));X=map_pilot_symbols(g.layout,PILOT_SYMBOLS)
    received=apply_frequency_mimo_channel(H,X)
    # This deterministic supplied array is not generated by any scientific channel/noise runner.
    noise=cnormal(rng,received.shape)*np.sqrt(.37)
    received+=noise
    hls,_=pilot_ls_estimate(received,X,g.layout.pilot_mask)
    original_p=hls.ravel(order='C')[g.selected]
    rx,tx,k,t=np.unravel_index(g.selected,(2,2,64,16),order='C')
    owner_received=received[rx,k,t]
    p=interface.divide_owner_pilots(owner_received)
    joint,_=interface.offline_pack(H)
    expected=interface.A@joint+noise[rx,k,t]/PILOT_SYMBOLS[tx]
    Cn=interface.noise_covariance(.37)
    # Deterministic covariance transport by unit pilot division, no estimated target variance.
    rotation=np.diag(1/PILOT_SYMBOLS[tx])
    covariance=rotation@(.37*np.eye(64))@rotation.conj().T
    checks=dict(same_pilot_values=error(p,original_p),observation_equation=error(p,expected),noise_covariance=error(Cn,covariance))
    return dict(PASS=all(v['PASS'] for v in checks.values()),checks=checks,extra_target_information=False,interpolated_input=False,noise_variance_semantics='E|complex n|^2=sigma2, per real part sigma2/2')

def grid_case():
    cases=[]
    for K in (1,2,4):
        for L in (4,8,16):
            model=known_model(K,20,L);rng=np.random.default_rng(DATA_SEED)
            A=np.eye(20,dtype=complex)[[0,3,9]];y=cnormal(rng,(2,3));Cn=.37*np.eye(3)
            expected=dense_posterior(model,y,Cn,A)[2]
            actual=MFAReferenceAdapter(model,A).estimate(y,Cn)
            cases.append(dict(K=K,L=L,D=20,loading_shape=list(model.lambdas.shape),covariance_full_rank=bool(np.all(np.linalg.matrix_rank(model.covs)==20)),posterior=error(actual,expected)))
    return dict(PASS=all(x['posterior']['PASS'] for x in cases),cases=cases,formal_feature_dimension=4032,all_latent_dimensions_less_than_formal_D=True,formal_candidate_training=False,final_K=None,final_loading_rank=None)

def reference_demo():
    core,_,ut=reference_modules()
    original=sys.modules.get('MFA_cplx');sys.modules['MFA_cplx']=core
    # Capture the author's entropy-created sampling generators for replay provenance.
    sample_rng=core.MFA_cplx.sample.__defaults__[1]
    normal_rng=ut.crandn.__kwdefaults__['rng']
    rng_states=dict(sample=copy.deepcopy(sample_rng.bit_generator.state),crandn=copy.deepcopy(normal_rng.bit_generator.state))
    try:
        with recorded_initialization(0),contextlib.redirect_stdout(io.StringIO()) as stream:
            globals_result=runpy.run_path(str(REFERENCE/'cplx-mfa/MFA_cplx_example.py'),run_name='__main__')
    finally:
        if original is None:sys.modules.pop('MFA_cplx',None)
        else:sys.modules['MFA_cplx']=original
    model=globals_result['mfa_est'];actual=stream.getvalue()
    unit_weights=abs(model.amps.sum()-1)<ATOL
    passed=model.means.shape==(16,32) and np.all(np.isfinite(globals_result['proba_soft'])) and 'Test completed.' in actual
    return dict(PASS=bool(passed),source=str(REFERENCE/'cplx-mfa/MFA_cplx_example.py'),source_modified=False,expected=dict(weights_sum='Printed by author, no specified assertion',means_shape=[16,32],sample_shape=[100,32],message='Test completed.',fixed_numerical_output='NOT_SPECIFIED'),actual=dict(weights_sum=float(model.amps.sum()),means_shape=list(model.means.shape),sample_shape=list(globals_result['samples'].shape),iterations=len(model.L_all),stdout=actual),unit_mixture_weight_diagnostic=dict(PASS=bool(unit_weights),absolute_error=float(abs(model.amps.sum()-1)),tolerance=ATOL,interpretation='FAIL retained: demo uses rs_clip=1e-6 without renormalization, unlike already-frozen default rs_clip=0. No repair or policy change.'),initialization_seed=0,author_data_seed=1235428719812346,sampling_rng_states_before=rng_states,tolerance=ATOL,publication_figure_replication='REFERENCE_NUMERICAL_REPLICATION_NOT_FULLY_SPECIFIED',runtime_comparison=False)

def firewall_case():
    from scripts.sp_u12b5a_common import firewall,OUT
    firewall()
    forbidden=ROOT/'outputs/sp_u12b2b/forbidden_scientific_result_probe.csv'
    try:forbidden.open('r')
    except PermissionError:blocked=True
    else:blocked=False
    params=list(inspect.signature(ProjectMFAReceiver.estimate_data).parameters)
    safe=params==['self','received_owner_pilot_symbols','noise_variance']
    return dict(PASS=blocked and safe,old_outcome_open_blocked_before_read=blocked,online_signature=params,target_information_arguments_absent=safe,scientific_outcomes_decoded=False)
