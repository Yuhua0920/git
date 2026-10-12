"""Exact B5A toy analytical functions; no project channel data."""
import numpy as np
from scipy.special import logsumexp
from kernels import adapter
make_reference_model=adapter().make_reference_model
DATA_SEED=4190123500
ATOL=1e-10
RTOL=1e-9
def cnormal(rng,shape):
    return (rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2)

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
