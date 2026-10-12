"""Receiver-only exact Gaussian pilot likelihood and candidate LMMSE (T13C).

No true channel, data samples, bits, sample IDs or true regime enter this module.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor,cho_solve

from src import lmmse
from src.lmmse_train import raw_array_sha256
from src.pilots import build_fdm_pilot_layout,map_pilot_symbols

CANDIDATES=(("ID",30.,100.),("speed120",120.,100.),("DS300",30.,300.))
SNRS=(5,10,20)
WINDOWS=(1,5)
LAYOUT=build_fdm_pilot_layout(2,64,16,16,4)
PILOT_VALUES=map_pilot_symbols(LAYOUT,np.array([(1+1j)/np.sqrt(2),(1-1j)/np.sqrt(2)]))
KNOWN_VALUES=np.stack([PILOT_VALUES[q,LAYOUT.pilot_mask[q]] for q in range(2)])


def pilot_coordinates(window_frames,tx):
    if window_frames not in WINDOWS or tx not in (0,1):
        raise ValueError("one/five frames and TX0/1 required")
    mask=np.tile(LAYOUT.pilot_mask[tx],(1,window_frames))
    return np.argwhere(mask)


def pilot_covariance(rf,rt,window_frames,tx):
    coordinates=pilot_coordinates(window_frames,tx)
    if rf.shape!=(64,64) or rt.shape!=(16*window_frames,16*window_frames):
        raise ValueError("theoretical covariance axes differ from observation grid")
    k,t=coordinates.T
    return rf[k[:,None],k[None,:]]*rt[t[:,None],t[None,:]]


def build_candidate_bank():
    bank=[]; records=[]
    for name,speed,ds in CANDIDATES:
        rf=lmmse.build_frequency_covariance(desired_rms_delay_spread_ns=ds)
        times={w:lmmse.build_time_covariance(num_symbols=16*w,speed_kmh=speed) for w in WINDOWS}
        full=lmmse.build_full_grid_covariance(rf,times[1])
        priors={(w,q):pilot_covariance(rf,times[w],w,q) for w in WINDOWS for q in range(2)}
        for a in (rf,full,*times.values(),*priors.values()):
            a.setflags(write=False)
        bank.append({"name":name,"rf":rf,"rt":times,"full_frame":full,"pilot_prior":priors})
        records.append({"candidate":f"C{len(bank)-1}","condition":name,"speed_kmh":speed,"nominal_DS_ns":ds,
            "source":"existing exact theoretical TDL-A Rf and Jakes Rt; oracle-defined candidate bank",
            "full_frame_sha256":raw_array_sha256(full),"Rf_sha256":raw_array_sha256(rf),
            "Rt_sha256":{str(w):raw_array_sha256(times[w]) for w in WINDOWS},
            "pilot_covariance_sha256":{f"{w}frames_tx{q}":raw_array_sha256(a) for (w,q),a in priors.items()}})
    return tuple(bank),records


@dataclass(frozen=True)
class LikelihoodModel:
    factor: tuple
    logdet: float
    observation_covariance: np.ndarray
    noise_variance: float


def prepare_likelihood(prior,noise_variance):
    r=np.asarray(prior,dtype=np.complex128)
    if (r.ndim!=2 or r.shape[0]!=r.shape[1] or not np.all(np.isfinite(r))
            or not np.allclose(r,r.conj().T,rtol=0,atol=1e-12)
            or not np.isfinite(noise_variance) or noise_variance<=0):
        raise ValueError("Hermitian prior and positive known observation variance required")
    c=r+noise_variance*np.eye(r.shape[0]); factor=cho_factor(c,lower=True,check_finite=True)
    logdet=float(2*np.log(np.diag(factor[0]).real).sum())
    c.setflags(write=False); factor[0].setflags(write=False)
    return LikelihoodModel(factor,logdet,c,float(noise_variance))


def prepare_pools(bank):
    likelihood={}; estimators={}
    for snr in SNRS:
        noise=10**(-snr/10)
        for w in WINDOWS:
            likelihood[w,snr]=tuple(tuple(prepare_likelihood(c["pilot_prior"][w,q],noise) for q in range(2)) for c in bank)
        estimators[snr]=tuple(tuple(lmmse.prepare_oracle_lmmse(c["full_frame"],lmmse.pilot_flat_indices(LAYOUT.pilot_mask[q]),noise,(64,16)) for q in range(2)) for c in bank)
    return likelihood,estimators


def pilot_ls_from_received_pilots(received_pilots,known_pilots=KNOWN_VALUES):
    y=np.asarray(received_pilots,dtype=np.complex128); x=np.asarray(known_pilots,dtype=np.complex128)
    if (y.shape!=(2,2,16) or x.shape!=(2,16) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(x))
            or not np.allclose(np.abs(x),1.,rtol=0,atol=1e-12)):
        raise ValueError("four-link unit-modulus owner-pilot observations required")
    return y/x[None,:,:]


def pack_joint_pilots(pilot_frames):
    z=np.asarray(pilot_frames,dtype=np.complex128)
    if z.ndim!=4 or z.shape[0] not in WINDOWS or z.shape!=(z.shape[0],2,2,16) or not np.all(np.isfinite(z)):
        raise ValueError("finite Pilot-LS [one/five frames, RX2, TX2, pilots16] required")
    # frequency outer; then frame and within-frame time anchor.
    return z.reshape(z.shape[0],2,2,4,4).transpose(1,2,3,0,4).reshape(2,2,16*z.shape[0])


def score_links(packed_pilots,models):
    z=np.asarray(packed_pilots,dtype=np.complex128)
    if len(models)!=3 or z.ndim!=3 or z.shape[:2]!=(2,2) or not np.all(np.isfinite(z)):
        raise ValueError("three candidates and four finite independent pilot vectors required")
    scores=np.empty((3,2,2),dtype=float)
    for i,pair in enumerate(models):
        if len(pair)!=2:
            raise ValueError("exact per-TX observation models required")
        for q,model in enumerate(pair):
            if z.shape[-1]!=model.observation_covariance.shape[0]:
                raise ValueError("pilot vector ordering/length differs from covariance")
            rhs=z[:,q].T
            solved=cho_solve(model.factor,rhs,check_finite=True)
            quadratic=np.sum(rhs.conj()*solved,axis=0)
            if np.max(np.abs(quadratic.imag))>1e-9 or not np.all(np.isfinite(quadratic)):
                raise ValueError("likelihood quadratic numerical validation failed")
            scores[i,:,q]=model.logdet+quadratic.real
    return scores


def select_candidate(pilot_frames,models):
    per_link=score_links(pack_joint_pilots(pilot_frames),models)
    scores=per_link.sum(axis=(1,2))
    rank=np.argsort(scores,kind="stable")
    return {"candidate_index":int(rank[0]),"scores":scores,"scores_by_link":per_link,
        "margin":float(scores[rank[1]]-scores[rank[0]]),"tie":bool(scores[rank[0]]==scores[rank[1]])}


def estimate_pilot_frames(pilot_frames,candidate_index,estimators):
    z=np.asarray(pilot_frames,dtype=np.complex128)
    if z.shape!=(5,2,2,16) or not np.all(np.isfinite(z)) or candidate_index not in (0,1,2):
        raise ValueError("five-frame pilots and one fixed candidate required")
    pair=estimators[candidate_index]
    output=np.empty((5,2,2,64,16),dtype=np.complex128)
    for frame in range(5):
        for q,obj in enumerate(pair):
            output[frame,:,q]=obj.estimate(z[frame,:,q])
    return output
