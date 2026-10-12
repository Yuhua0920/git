"""Frozen T15B probe geometry, independent Gaussian reference and diagnostics.

Production GMM fitting/estimation is delegated to the unmodified author package.
All full-grid indices use C order [rx, tx, frequency, time].
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
from src.spatial_information import observation_geometry

KS = (1, 2, 4, 8)
EM_SEEDS = (20261631, 20261632, 20261633)
EM_PARAMETERS = dict(covariance_type="full", zero_mean=False, n_init=1,
                     init_params="kmeans", max_iter=100, tol=0.001, reg_covar=1e-6)
REGRESSION_ATOL = 1e-8
REGRESSION_RTOL = 1e-8


def probe_indices():
    """4x4 equal bins/link; choose lower midpoint, nearest data point if needed.

    Ties are resolved by frequency then time. This uses masks only, never H.
    """
    g = observation_geometry()
    data = []
    mask = g.data.reshape(2, 2, 64, 16)
    for rx in range(2):
        for tx in range(2):
            for fbin in range(4):
                for tbin in range(4):
                    flo, tlo = 16*fbin, 4*tbin
                    fc, tc = flo+7, tlo+1
                    candidates = [(abs(f-fc)+abs(t-tc), f, t)
                                  for f in range(flo, flo+16)
                                  for t in range(tlo, tlo+4) if mask[rx,tx,f,t]]
                    _, f, t = min(candidates)
                    data.append(np.ravel_multi_index((rx,tx,f,t), mask.shape))
    p, d = g.selected.copy(), np.asarray(data, dtype=np.int64)
    if len(p)!=64 or len(d)!=64 or len(set(d))!=64 or not g.data[d].all():
        raise ValueError("invalid fixed probe")
    return p, d


def selection_matrix():
    return np.eye(128, dtype=np.complex128)[:64]


def pilot_unit_noise(full_noise):
    """Apply the original owner-pilot phase division in exact selected order."""
    from src.link import PILOT_SYMBOLS
    p,_=probe_indices()
    rx,tx,k,t=np.unravel_index(p,(2,2,64,16))
    noise=np.asarray(full_noise,dtype=np.complex128)
    if noise.shape!=(2,64,16): raise ValueError("original RX grid noise required")
    return noise[rx,k,t]/PILOT_SYMBOLS[tx]


def gaussian_statistics(train, trajectory_ids, *, reg_covar=1e-6):
    """Independent population covariance + the package's declared regularizer."""
    ids=np.asarray(trajectory_ids)
    x=np.asarray(train, dtype=np.complex128)
    if x.ndim!=2 or x.shape[1]!=128 or len(ids)!=len(x) or not np.isfinite(x).all():
        raise ValueError("finite 128-dimensional rows required")
    if np.any((ids<0)|(ids>=500)):
        raise ValueError("train_dev only; validation cannot fit statistics")
    mean=x.mean(axis=0)
    centered=x-mean
    covariance=centered.T@centered.conj()/len(x)+reg_covar*np.eye(128)
    return mean, covariance


def gaussian_reference(z, mean, covariance, sigma2):
    """Independent R10-probe reference, including the identical regularizer."""
    if not np.isfinite(sigma2) or sigma2<=0: raise ValueError("positive noise")
    cross=covariance[:, :64]
    weight=cho_solve(cho_factor(covariance[:64,:64]+sigma2*np.eye(64),lower=True),cross.conj().T).conj().T
    return mean+(z-mean[:64])@weight.T


def choose_training_seed(records):
    if not records or any(r.get("selection_split")!="train_dev" for r in records):
        raise ValueError("selection must use training log likelihood only")
    viable=[r for r in records if r.get("fit_succeeded") and np.isfinite(r["train_log_likelihood"])]
    if not viable: raise ValueError("no stable model; stop this K")
    return max(viable, key=lambda r:(r["train_log_likelihood"], -r["seed"]))["seed"]


def paired_diagnostics(candidate, reference):
    delta=np.asarray(candidate)-np.asarray(reference)
    return dict(median_error_energy_difference=float(np.median(delta)),
                q25_error_energy_difference=float(np.quantile(delta,.25)),
                q75_error_energy_difference=float(np.quantile(delta,.75)),
                fraction_trajectories_improved=float(np.mean(delta<0)),
                difference_definition="K error energy minus K1 error energy")


def pilot_log_likelihood(z, weights, means, covariances, sigma2):
    """Independent noisy-pilot density audit only, never model selection."""
    components=[]
    for weight,mean,cov in zip(weights,means,covariances,strict=True):
        factor=cho_factor(cov[:64,:64]+sigma2*np.eye(64),lower=True)
        delta=z-mean[:64]
        quadratic=np.sum(delta.conj()*cho_solve(factor,delta.T).T,axis=1).real
        logdet=2*np.log(np.diag(factor[0]).real).sum()
        components.append(np.log(weight)-64*np.log(np.pi)-logdet-quadratic)
    return logsumexp(np.asarray(components),axis=0)


def scientific_flags(results):
    focus=[r for r in results if r["K"] in (4,8) and r["snr_db"] in (10,20)]
    if len(focus)!=4:
        return dict(nonlinear_prior_gain_small_in_probe=None,
                    nonlinear_prior_gain_material=None, nonlinear_prior_gain_partial=None,
                    interpretation="incomplete pre-registered K4/K8 focus; no scientific flag assigned")
    small=max(r["nonlinear_gain_db"] for r in focus)<.25
    material=any(r["nonlinear_gain_db"]>=1. and r["fraction_trajectories_improved"]>=.7 for r in focus)
    return dict(nonlinear_prior_gain_small_in_probe=small,
                nonlinear_prior_gain_material=material, nonlinear_prior_gain_partial=not(small or material))
