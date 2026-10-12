"""T13D proper-complex covariance information; no parameter estimator.

Coordinates are the existing [K,global symbol] C ordering. The delay parameter
is the existing nominal TDL-A scaling parameter in seconds, without changing
the tabulated normalized delays or powers. Noise/power/shape are known.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import cholesky,cho_solve,solve_triangular
from scipy.special import j1

from src.doppler import (CARRIER_FREQUENCY_HZ,SPEED_OF_LIGHT_MPS,jakes_covariance,
    mobility_parameters,ofdm_time_grid_seconds)
from src.lmmse import build_frequency_covariance
from src.lmmse_train import raw_array_sha256
from src.pilots import build_fdm_pilot_layout,count_layout_resources
from src.tdl import TDL_A_NORMALIZED_DELAYS,centered_frequency_grid_hz,tdl_a_profile,rms_delay_spread_seconds

POINTS=(("P0",30.,100.),("P1",120.,100.),("P2",30.,300.),("P3",120.,300.))
DFS=(16,8); WINDOWS=(1,5); SNRS=(5,10,20)
SPEED_PER_FD=3.6*SPEED_OF_LIGHT_MPS/CARRIER_FREQUENCY_HZ
RMS_SCALE_FACTOR=rms_delay_spread_seconds(*tdl_a_profile(1.))/1e-9


def theta_from_physical(speed_kmh,nominal_ds_ns):
    if not np.isfinite(speed_kmh) or speed_kmh<=0 or not np.isfinite(nominal_ds_ns) or nominal_ds_ns<=0:
        raise ValueError("positive finite Doppler/speed and delay scale required")
    return np.array([mobility_parameters(speed_kmh)["maximum_doppler_hz"],nominal_ds_ns*1e-9])


@dataclass(frozen=True)
class Geometry:
    dF: int
    window_frames: int
    tx: int
    coordinates: np.ndarray
    flat_indices: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray
    pilot_ratio: float


def geometry(dF,window_frames,tx):
    if dF not in DFS or window_frames not in WINDOWS or tx not in (0,1):
        raise ValueError("only dF16/8, windows1/5 and TX0/1 required")
    layout=build_fdm_pilot_layout(2,64,16,dF,4)
    mask=np.tile(layout.pilot_mask[tx],(1,window_frames))
    coordinates=np.argwhere(mask); k,t=coordinates.T
    indices=np.flatnonzero(mask.ravel(order="C"))
    f=centered_frequency_grid_hz(64,30000)[k]
    times=ofdm_time_grid_seconds(num_symbols=16*window_frames)[t]
    for a in (coordinates,indices,f,times):
        a.setflags(write=False)
    return Geometry(dF,window_frames,tx,coordinates,indices,f,times,count_layout_resources(layout)["union_ratio"])


def factors(fd_hz,tau_s,window_frames,derivatives=True,timing=None):
    if (not np.isfinite(fd_hz) or fd_hz<=0 or not np.isfinite(tau_s) or tau_s<=0
            or window_frames not in WINDOWS):
        raise ValueError("positive family parameters and one/five frames required")
    stamp=perf_counter()
    # Reuse the original frequency builder, including its delay scaling.
    rf=build_frequency_covariance(desired_rms_delay_spread_ns=tau_s*1e9)
    times=ofdm_time_grid_seconds(num_symbols=16*window_frames)
    rt=jakes_covariance(times,fd_hz)
    if timing is not None:
        timing["covariance_construction_seconds"]+=perf_counter()-stamp
    if not derivatives:
        return rf,rt
    stamp=perf_counter(); lag=times[:,None]-times[None,:]
    drt=-2*np.pi*lag*j1(2*np.pi*fd_hz*lag)
    frequencies=centered_frequency_grid_hz(64,30000)
    df=frequencies[:,None]-frequencies[None,:]
    normalized=np.asarray(TDL_A_NORMALIZED_DELAYS,dtype=float)
    delays,powers=tdl_a_profile(tau_s*1e9)
    phase=-2j*np.pi*df[...,None]
    drf=np.sum(powers*phase*normalized*np.exp(phase*delays),axis=-1)
    if timing is not None:
        timing["derivatives_seconds"]+=perf_counter()-stamp
    return rf,rt,drf,drt


def observation_covariance(theta,g,noise_variance,derivatives=True,timing=None):
    if np.shape(theta)!=(2,) or not np.isfinite(noise_variance) or noise_variance<=0:
        raise ValueError("two real parameters and known positive noise required")
    terms=factors(*theta,g.window_frames,derivatives=derivatives,timing=timing)
    stamp=perf_counter(); rf,rt=terms[:2]; k,t=g.coordinates.T
    f=rf[k[:,None],k[None,:]]; time=rt[t[:,None],t[None,:]]
    covariance=f*time+noise_variance*np.eye(len(k))
    if timing is not None:
        timing["covariance_construction_seconds"]+=perf_counter()-stamp
    if not derivatives:
        return covariance
    stamp=perf_counter(); drf,drt=terms[2:]
    dc=np.stack((f*drt[t[:,None],t[None,:]],drf[k[:,None],k[None,:]]*time))
    if timing is not None:
        timing["derivatives_seconds"]+=perf_counter()-stamp
    return covariance,dc


@dataclass(frozen=True)
class PreparedInformation:
    covariance: np.ndarray
    factor: np.ndarray
    logdet: float
    covariance_sha256: str


def prepare_covariance(covariance):
    c=np.asarray(covariance,dtype=np.complex128)
    if c.ndim!=2 or c.shape[0]==0 or c.shape[0]!=c.shape[1] or not np.all(np.isfinite(c)) or not np.allclose(c,c.conj().T,rtol=0,atol=1e-12):
        raise ValueError("finite Hermitian observation covariance required")
    factor=cholesky(c,lower=True,check_finite=True)
    logdet=float(2*np.log(np.diag(factor).real).sum())
    c.setflags(write=False); factor.setflags(write=False)
    return PreparedInformation(c,factor,logdet,raw_array_sha256(c))


def whiten_derivatives(model,derivatives):
    matrices=np.asarray(derivatives,dtype=np.complex128)
    if matrices.shape!=(2,*model.covariance.shape) or not np.all(np.isfinite(matrices)):
        raise ValueError("two covariance derivatives matching observations required")
    whitened=[]
    for d in matrices:
        if not np.allclose(d,d.conj().T,rtol=1e-12,atol=1e-12):
            raise ValueError("covariance derivative must be Hermitian")
        left=solve_triangular(model.factor,d,lower=True,check_finite=True)
        whitened.append(solve_triangular(model.factor,left.conj().T,lower=True,check_finite=True).conj().T)
    return np.stack(whitened)


def fisher_link(model,derivatives):
    a=whiten_derivatives(model,derivatives)
    raw=np.einsum("aij,bji->ab",a,a)
    if np.max(np.abs(raw.imag))>1e-9 or not np.all(np.isfinite(raw)):
        raise ValueError("FIM trace must be finite and real")
    return raw.real


class SingularInformationError(ValueError):
    def __init__(self,record):
        self.record=record
        super().__init__("log-parameter FIM near singular; no pinv fallback")


def crlb_from_log_fim(j_log,theta):
    j=np.asarray(j_log,dtype=float); theta=np.asarray(theta,dtype=float)
    if j.shape!=(2,2) or theta.shape!=(2,) or not np.all(np.isfinite(j)) or not np.all(theta>0) or not np.allclose(j,j.T,rtol=0,atol=1e-10):
        raise ValueError("finite symmetric FIM and positive physical scales required")
    eigenvalues=np.linalg.eigvalsh(j); condition=float(eigenvalues[-1]/eigenvalues[0]) if eigenvalues[0]>0 else None
    record={"dimensionless_eigenvalues":eigenvalues.tolist(),"dimensionless_condition":condition,"pinv_used":False}
    if eigenvalues[0]<=1e-12*max(eigenvalues[-1],1.) or condition is None or condition>1e12:
        raise SingularInformationError(record)
    b_log=cho_solve((cholesky(j,lower=True),True),np.eye(2))
    b_physical=b_log*theta[:,None]*theta[None,:]
    correlation=float(b_log[0,1]/np.sqrt(b_log[0,0]*b_log[1,1]))
    return b_physical,b_log,{**record,"parameter_error_correlation":correlation,"cholesky_success":True}


def information_point(point,speed,ds,dF,window,snr,timing=None):
    theta=theta_from_physical(speed,ds); noise=10**(-snr/10)
    per_link=[]; geom_records=[]; covariances=[]; models=[]
    for nr in range(2):
        for tx in range(2):
            g=geometry(dF,window,tx); c,dc=observation_covariance(theta,g,noise,timing=timing)
            stamp=perf_counter(); model=prepare_covariance(c); j=fisher_link(model,dc*theta[:,None,None])
            if timing is not None:
                timing["FIM_seconds"]+=perf_counter()-stamp
            per_link.append(j); covariances.append(c); models.append(model)
            geom_records.append({"link":f"H{nr}{tx}","tx":tx,"pilot_count":len(g.coordinates),
                "first_coordinate":g.coordinates[0].tolist(),"last_coordinate":g.coordinates[-1].tolist(),
                "flat_indices_sha256":raw_array_sha256(g.flat_indices),"covariance_sha256":model.covariance_sha256,"J_log":j.tolist()})
    stamp=perf_counter(); j_log=np.sum(per_link,axis=0); j_physical=j_log/theta[:,None]/theta[None,:]
    bound,b_log,diag=crlb_from_log_fim(j_log,theta)
    if timing is not None:
        timing["FIM_seconds"]+=perf_counter()-stamp
    equality=max(float(np.max(np.abs(c-covariances[0]))) for c in covariances)
    link_equality=max(float(np.max(np.abs(j-per_link[0]))) for j in per_link)
    row={"point":point,"speed_kmh":speed,"dF":dF,"pilot_ratio":geometry(dF,window,0).pilot_ratio,"window_frames":window,
        "snr_db":snr,"fD_Hz":float(theta[0]),"tau_rms_ns":ds,"actual_profile_rms_ns":ds*RMS_SCALE_FACTOR,
        "pilots_per_link":models[0].covariance.shape[0],
        **{f"fim_{a}{b}":float(j_physical[a,b]) for a,b in ((0,0),(0,1),(1,1))},
        **{f"fim_log_{a}{b}":float(j_log[a,b]) for a,b in ((0,0),(0,1),(1,1))},
        **{f"crlb_{a}{b}":float(bound[a,b]) for a,b in ((0,0),(0,1),(1,1))},
        "fD_std_Hz":float(np.sqrt(bound[0,0])),"speed_std_kmh":float(SPEED_PER_FD*np.sqrt(bound[0,0])),
        "delay_std_ns":float(np.sqrt(bound[1,1])*1e9),"actual_profile_rms_std_ns":float(np.sqrt(bound[1,1])*1e9*RMS_SCALE_FACTOR),
        "relative_speed_std_percent":float(np.sqrt(b_log[0,0])*100),"relative_delay_std_percent":float(np.sqrt(b_log[1,1])*100),
        "parameter_error_correlation":diag["parameter_error_correlation"],"dimensionless_eigenvalue_min":diag["dimensionless_eigenvalues"][0],
        "dimensionless_eigenvalue_max":diag["dimensionless_eigenvalues"][1],"dimensionless_condition":diag["dimensionless_condition"],
        "raw_unit_condition_diagnostic_only":float(np.linalg.cond(j_physical)),"four_link_covariance_equality_max_error":equality,
        "four_link_FIM_equality_max_error":link_equality,"pinv_used":False}
    return row,{"point":point,"dF":dF,"window_frames":window,"snr_db":snr,"links":geom_records,
        "four_link_FIM_sum":j_log.tolist(),"all_checks_passed":True}


def kl_link(reference,candidate):
    if reference.covariance.shape!=candidate.covariance.shape:
        raise ValueError("KL requires identical pilot geometry/dimension")
    trace=np.trace(cho_solve((candidate.factor,True),reference.covariance,check_finite=True))
    if abs(trace.imag)>1e-9:
        raise ValueError("KL trace has significant imaginary error")
    value=float(candidate.logdet-reference.logdet+trace.real-reference.covariance.shape[0])
    if value < -1e-9 or not np.isfinite(value):
        raise ValueError("proper-complex KL failed nonnegativity tolerance")
    return value


def kl_four_links(references,candidates):
    if len(references)!=4 or len(candidates)!=4:
        raise ValueError("four independent-link distributions required")
    values=[kl_link(a,b) for a,b in zip(references,candidates)]
    return float(sum(values)),values
