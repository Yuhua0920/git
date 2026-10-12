"""Independent proper-complex detector diagnostics; no estimator changes."""
import itertools
import numpy as np
from scipy.special import logsumexp,erfc

BITS=np.array(list(itertools.product((0,1),repeat=4)),dtype=np.uint8)
SYMBOLS=((1-2*BITS.astype(float).reshape(16,2,2)[...,0])+
         1j*(1-2*BITS.astype(float).reshape(16,2,2)[...,1]))/2
PS=0.5

def check(h,y,variance):
    h=np.asarray(h,dtype=np.complex128);y=np.asarray(y,dtype=np.complex128)
    if h.ndim!=3 or h.shape[1:]!=(2,2) or y.shape!=(len(h),2):raise ValueError('RE_RX_TX_SHAPE')
    if not np.isfinite(h).all() or not np.isfinite(y).all():raise ValueError('NONFINITE_INPUT')
    if not np.isfinite(variance) or variance<=0:raise ValueError('POSITIVE_COMPLEX_VARIANCE_REQUIRED')
    return h,y

def linear(h,y,variance,mmse=False):
    h,y=check(h,y,variance)
    if mmse:
        hh=h.conj().transpose(0,2,1)
        gram=hh@h+(variance/PS)*np.eye(2)
        rhs=np.einsum('nij,nj->ni',hh,y)
        lower=np.linalg.cholesky(gram)
        a=np.linalg.solve(lower,rhs[...,None])
        x=np.linalg.solve(lower.conj().transpose(0,2,1),a)[...,0]
    else:x=np.linalg.solve(h,y[...,None])[...,0]
    if not np.isfinite(x).all():raise ValueError('NONFINITE_LINEAR_OUTPUT')
    bit=np.stack((x.real<0,x.imag<0),axis=-1).reshape(len(h),4).astype(np.uint8)
    return bit,x

def bit_map(h,y,variance):
    h,y=check(h,y,variance)
    means=np.einsum('nrt,ct->ncr',h,SYMBOLS)
    score=-np.sum(abs(y[:,None,:]-means)**2,axis=-1)/variance
    # Uniform16-vector prior; common proper-complex likelihood constants cancel.
    marginal=np.column_stack([logsumexp(score[:,BITS[:,k]==1],axis=1)-
        logsumexp(score[:,BITS[:,k]==0],axis=1) for k in range(4)])
    bit=(marginal>0).astype(np.uint8)  # equal posterior ->0, matching hard-slicer ties
    if not np.isfinite(marginal).all():raise ValueError('NONFINITE_MAP_LOG_ODDS')
    return bit,marginal,BITS[np.argmax(score,axis=1)]

def zf_risk(h,variance):
    h=np.asarray(h,dtype=np.complex128)
    sv=np.linalg.svd(h,compute_uv=False)
    inv=np.linalg.solve(h,np.broadcast_to(np.eye(2,dtype=np.complex128),h.shape))
    gain=np.sum(abs(inv)**2,axis=2)
    esn0=PS/(variance*gain)
    # QPSK bit BER conditional on H: Eb/N0=Ps/(2 sigma² diag(H^-1 H^-H)).
    ber=.5*erfc(np.sqrt(esn0/2))
    return sv,sv[:,0]/sv[:,1],gain,10*np.log10(esn0),ber

def conditional_mismatched_zf(h_true,h_est,variance):
    """Exact expected bit error given H and pilot-only Hhat, averaged over 16 data vectors.

    Data noise is independent of the pilots used to form Hhat. No channel or noise
    is drawn. Both signal mismatch and transformed-noise covariance enter margins.
    """
    n=len(h_true)
    inv=np.linalg.solve(h_est,np.broadcast_to(np.eye(2,dtype=np.complex128),h_est.shape))
    effective=inv@h_true
    mean=np.einsum('nrt,ct->ncr',effective,SYMBOLS)
    means=np.stack((mean.real,mean.imag),axis=-1).reshape(n,16,4)
    gain=np.sum(abs(inv)**2,axis=2)
    sign=(1-2*BITS.astype(float))
    denominator=np.sqrt(variance*np.repeat(gain,2,axis=1))[:,None,:]
    expected=.5*erfc(sign[None]*means/denominator)
    # Average over the equiprobable 16 transmit vectors and their four bits.
    mismatch_mse_per_stream=PS*np.sum(abs(effective-np.eye(2))**2,axis=(1,2))/2
    return np.mean(expected,axis=(1,2)),gain,mismatch_mse_per_stream
