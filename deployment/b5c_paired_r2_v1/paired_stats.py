"""Predeclared paired trajectory bootstrap, without fitting or selection."""
import numpy as np
from base import require
SNRS=(0,5,10,15,20,25,30)
CONDITIONS=('ID','Profile','Mobility','Composed','Boundary')
R=100000
FAMILY=35
MASTER=2997555629
TAIL=.05/(2*FAMILY)
def pooled(e,t):
    e=np.asarray(e,dtype=np.float64); t=np.asarray(t,dtype=np.float64)
    require(e.shape==t.shape and np.isfinite(e).all() and np.isfinite(t).all() and np.all(e>0) and np.all(t>0),'INVALID_POSITIVE_FINITE_ENERGIES')
    return e.sum(axis=0,dtype=np.float64)/t.sum(axis=0,dtype=np.float64)
def summarize(error,truth,condition_index,replicates=R,cancel=None):
    # error[N,7,2], truth[N,7]; method order LR4,MFA; N=1000 in formal runner.
    e=np.asarray(error,dtype=np.float64); t=np.asarray(truth,dtype=np.float64)
    require(e.ndim==3 and e.shape[1:]==(7,2) and t.shape==e.shape[:2],'PAIRED_ENERGY_AXIS_DRIFT')
    require(condition_index in range(5) and replicates==R,'FROZEN_BOOTSTRAP_REQUIRED')
    nmse=pooled(e,np.repeat(t[:,:,None],2,axis=2))
    db=10*np.log10(nmse); point=db[:,1]-db[:,0]
    rng=np.random.default_rng(np.random.SeedSequence([MASTER,condition_index]))
    draws=np.empty((R,7),np.float64); n=len(e)
    for start in range(0,R,1000):
        if cancel is not None:cancel()
        ids=rng.integers(0,n,size=(min(1000,R-start),n),dtype=np.int32)
        # Same trajectory counts for both methods and every SNR in this condition.
        counts=np.empty((len(ids),n),dtype=np.int64)
        for k,row in enumerate(ids): counts[k]=np.bincount(row,minlength=n)
        num=counts@e.reshape(n,14); den=counts@t
        bnmse=num.reshape(len(ids),7,2)/den[:,:,None]
        bdb=10*np.log10(bnmse); draws[start:start+len(ids)]=bdb[:,:,1]-bdb[:,:,0]
    bounds=np.quantile(draws,[TAIL,1-TAIL],axis=0,method='linear')
    rows=[]
    for si,s in enumerate(SNRS):
        lo,hi=bounds[:,si]
        rows.append({'condition':CONDITIONS[condition_index],'snr_db':s,'N_trajectories':n,
            'LR4_linear_pooled_NMSE':float(nmse[si,0]),'MFA_linear_pooled_NMSE':float(nmse[si,1]),
            'LR4_NMSE_dB':float(db[si,0]),'MFA_NMSE_dB':float(db[si,1]),
            'MFA_minus_LR4_dB':float(point[si]),'simultaneous_CI_lower_dB':float(lo),'simultaneous_CI_upper_dB':float(hi),
            'classification':'LR4_SUPPORTED' if lo>0 else 'MFA_SUPPORTED' if hi<0 else 'INDETERMINATE',
            'bootstrap_replicates':R,'family_size':FAMILY,'tail_probability':TAIL})
    return rows,draws
