"""Pure preregistered U7B reduction; U7A tests use synthetic energies only.

No filesystem, simulator, estimator, historical outcome or refitting API.
Calling this module on scientific data requires a future passed unseal gate.
"""
import numpy as np

METHODS=('LS','Source','Hard','Convex','Dense','LR4','Matched')
SNRS=(10,20)
REPLICATES=10000
FAMILY_ALPHA=.05
PRIMARY_QUANTILE=FAMILY_ALPHA/len(SNRS)

def reduce_nmse(error,truth):
    e=np.asarray(error,dtype=np.float64);q=np.asarray(truth,dtype=np.float64)
    if e.ndim!=3 or e.shape[1:]!=(2,7) or q.shape!=(e.shape[0],2):raise ValueError('trajectory x SNR x method energies required')
    if not np.isfinite(e).all() or not np.isfinite(q).all() or np.any(e<=0) or np.any(q<=0):raise ValueError('finite positive NMSE energies required; no epsilon floor')
    return 10*np.log10(e.sum(axis=0)/q.sum(axis=0)[:,None])

def decision(point,lower,upper):
    p,l,u=[np.asarray(v,dtype=float) for v in (point,lower,upper)]
    if any(v.shape!=(2,) or not np.isfinite(v).all() for v in (p,l,u)):raise ValueError('two finite SNR results required')
    if np.any(l>u):raise ValueError('CI ordering')
    if np.all(p>0) and np.all(l>0):return 'A_STRONG_TRANSFER'
    if np.any(u<=0):return 'C_NONPOSITIVE_BOUNDARY'
    return 'B_MIXED_OR_INCONCLUSIVE'

def paired_bootstrap(error,truth,bit_errors,total_bits,seed,*,replicates=REPLICATES,expected_n=1000):
    e=np.asarray(error,dtype=float);q=np.asarray(truth,dtype=float)
    be=np.asarray(bit_errors);nb=np.asarray(total_bits)
    if e.shape!=(expected_n,2,7) or q.shape!=(expected_n,2) or be.shape!=e.shape or nb.shape!=(expected_n,2):raise ValueError('complete paired panel required; no missing-row filtering')
    if not np.isfinite(be).all() or not np.isfinite(nb).all() or np.any(be<0) or np.any(nb<=0) or np.any(be>nb[...,None]) or np.any(be!=np.floor(be)) or np.any(nb!=np.floor(nb)):raise ValueError('valid integer bit counts required')
    observed=reduce_nmse(e,q);observed_ber=be.sum(0)/nb.sum(0)[:,None]
    # Same trajectory draw across ALL seven methods AND both SNRs.
    rng=np.random.Generator(np.random.PCG64(int(seed)))
    indices=rng.integers(0,expected_n,size=(replicates,expected_n))
    db=np.empty((replicates,2,7));ber=np.empty_like(db)
    # Reuse original Paper 1 chunked, pooled-energy reduction semantics.
    for start in range(0,replicates,100):
        ii=indices[start:start+100]
        db[start:start+len(ii)]=10*np.log10(e[ii].sum(1)/q[ii].sum(1)[...,None])
        ber[start:start+len(ii)]=be[ii].sum(1)/nb[ii].sum(1)[...,None]
    gain=observed[:,1]-observed[:,5];draw=db[:,:,1]-db[:,:,5]
    lower=np.quantile(draw,PRIMARY_QUANTILE,axis=0,method='linear')
    upper=np.quantile(draw,1-PRIMARY_QUANTILE,axis=0,method='linear')
    secondary={}
    for label,left,right in (('LR4_minus_Dense',5,4),('LR4_minus_Matched',5,6),('Convex_minus_LR4',3,5)):
        secondary[label]={'point':observed[:,left]-observed[:,right],
                          'ci':np.quantile(db[:,:,left]-db[:,:,right],[.025,.975],axis=0,method='linear')}
    return {'nmse_db':observed,'nmse_ci':np.quantile(db,[.025,.975],axis=0,method='linear'),'ber':observed_ber,'gain':gain,'primary_lcb':lower,'descriptive_ucb':upper,
            'decision':decision(gain,lower,upper),'secondary':secondary,
            'ber_ci':np.quantile(ber,[.025,.975],axis=0,method='linear'),
            'ber_LR4_minus_Source_ci':np.quantile(ber[:,:,5]-ber[:,:,1],[.025,.975],axis=0,method='linear'),
            'indices_sha256':__import__('hashlib').sha256(indices.tobytes()).hexdigest()}
