"""T14B controlled Kronecker information gate; no communication generator.

Project order is h[((rx*Nt+tx)*K+k)*T+t] = H[rx,tx,k,t].
Column-major single-point vec uses rx+Nr*tx; explicit permutation is required.
Only covariance blocks needed by selection/trace calculations are materialized.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix

from src import lmmse
from src.doppler import covariance_eigenfactor
from src.numerics import complex_gaussian_noise, noise_variance_from_snr
from src.pilots import build_fdm_pilot_layout, count_layout_resources
from src.tdl import tdl_a_profile, centered_frequency_grid_hz

T14A_COMMIT = "94b74ddad547180af02db7569b3225fec1799dee"
PREFIX = "t14b_spatial_information_gate"
RHOS = (0., .3, .6, .9)
SNRS = (5, 10, 20)
MODES = ("TX-only", "RX-only", "BOTH")
MC_SAMPLES = 10000
MC_CHANNEL_SEED = 20261418
MC_NOISE_SEED = 20261419
MC_RELATIVE_LIMIT = .02
SPATIAL_SAMPLES = 200000
SPATIAL_SEED = 20261417
SPATIAL_RELATIVE_LIMIT = .015
TOL = 1e-12


def correlation(rho):
    if not np.isreal(rho) or not np.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("rho must be real, nonnegative and strictly below one")
    return np.array([[1., rho], [rho, 1.]], dtype=np.complex128)


def spatial_covariances(rho_tx, rho_rx):
    tx, rx = correlation(rho_tx), correlation(rho_rx)
    column = np.kron(tx.T, rx)
    # Project pairs [(0,0),(0,1),(1,0),(1,1)] map to vec_F [0,2,1,3].
    permutation = np.array([0, 2, 1, 3])
    project = column[np.ix_(permutation, permutation)]
    return tx, rx, column, project


def pack_project(h):
    a = np.asarray(h)
    if a.ndim < 4 or a.shape[-4:-2] != (2, 2):
        raise ValueError("H must end in [Rx=2,Tx=2,K,T]")
    return a.reshape(a.shape[:-4] + (-1,), order="C")


@dataclass(frozen=True)
class Geometry:
    layout: object
    selected: np.ndarray
    data: np.ndarray
    selector: csr_matrix
    counts: dict


def observation_geometry():
    layout = build_fdm_pilot_layout(2, 64, 16, 16, 4)
    observed = np.broadcast_to(layout.pilot_mask[None], (2, 2, 64, 16))
    data = np.broadcast_to(layout.data_mask[None], observed.shape).ravel(order="C").copy()
    selected = np.flatnonzero(observed.ravel(order="C"))
    selection = csr_matrix((np.ones(selected.size), (np.arange(selected.size), selected)), shape=(selected.size, data.size))
    return Geometry(layout, selected, data, selection, count_layout_resources(layout))


@dataclass(frozen=True)
class JointCovariance:
    """R_joint = R_sp_project tensor Rf tensor Rt, in the documented order."""
    spatial: np.ndarray
    frequency: np.ndarray
    time: np.ndarray

    @property
    def size(self):
        return self.spatial.shape[0] * self.frequency.shape[0] * self.time.shape[0]

    def entries(self, rows, columns):
        rows, columns = np.asarray(rows), np.asarray(columns)
        k, t = self.frequency.shape[0], self.time.shape[0]
        a, b = rows // (k*t), columns // (k*t)
        f, g = (rows // t) % k, (columns // t) % k
        u, v = rows % t, columns % t
        return (self.spatial[a[:, None], b[None, :]] * self.frequency[f[:, None], g[None, :]]
                * self.time[u[:, None], v[None, :]])

    def diagonal(self):
        return np.kron(np.diag(self.spatial), np.kron(np.diag(self.frequency), np.diag(self.time))).real

    def psd_diagnostics(self):
        if not all(np.allclose(m, m.conj().T, rtol=0, atol=TOL) for m in (self.spatial, self.frequency, self.time)):
            raise ValueError("Hermitian covariance factors required")
        factors = [np.linalg.eigvalsh(m) for m in (self.spatial, self.frequency, self.time)]
        eigenvalues = np.kron(factors[0], np.kron(factors[1], factors[2]))
        if min(x.min() for x in factors) < -TOL or eigenvalues.min() < -1e-10:
            raise ValueError("joint covariance fails Kronecker-factor PSD gate")
        return {"factor_min_eigenvalues": [float(x.min()) for x in factors],
                "joint_min_eigenvalue": float(eigenvalues.min()),
                "joint_max_eigenvalue": float(eigenvalues.max()), "PSD_tolerance": 1e-10,
                "Hermitian_PSD_passed": True, "dense_joint_allocated": False}


@dataclass(frozen=True)
class Posterior:
    cross: np.ndarray
    observation: np.ndarray
    factor: tuple
    weights: np.ndarray
    diagonal: np.ndarray
    prior: np.ndarray
    selected: np.ndarray
    diagnostics: dict

    def estimate(self, z):
        a = np.asarray(z, dtype=np.complex128)
        if a.ndim != 2 or a.shape[1] != self.selected.size or not np.all(np.isfinite(a)):
            raise ValueError("finite z[samples, observations] required")
        return a @ self.weights.T


def prepare_posterior(joint, selected, sigma2):
    indices = np.asarray(selected)
    if indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu" or np.any(np.diff(indices) <= 0) or indices.min() < 0 or indices.max() >= joint.size:
        raise ValueError("sorted unique selected indices required")
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("positive complex noise variance required")
    prior = joint.diagonal()
    cross = joint.entries(np.arange(joint.size), indices)
    c = joint.entries(indices, indices) + sigma2*np.eye(indices.size)
    factor = cho_factor(c, lower=True)
    solved = cho_solve(factor, cross.conj().T)
    posterior = prior - np.sum(cross*solved.T, axis=1)
    imaginary = float(np.max(np.abs(posterior.imag)))
    if imaginary > TOL or posterior.real.min() < -TOL or np.any(posterior.real > prior+TOL):
        raise ValueError("posterior bounds failed; no clipping or fallback")
    eigen = np.linalg.eigvalsh(c)
    # If R=FF^H, Sigma=F(I-Fp^H C^-1 Fp)F^H is PSD.
    # Nontrivial latent eigenvalues are sigma2/lambda(C)>0; others equal one.
    diagnostics = {"cholesky_success": True, "observation_condition": float(eigen[-1]/eigen[0]),
                   "latent_posterior_PSD_lower_bound": float(sigma2/eigen[-1]),
                   "posterior_PSD_certificate": "factor PSD plus noisy Schur complement",
                   "prior_minus_posterior_PSD_certificate": "cross C^-1 cross^H Gram matrix",
                   "posterior_min": float(posterior.real.min()), "posterior_max_imaginary": imaginary,
                   "posterior_clipped": False, "jitter_used": False, "inverse_used": False, "pinv_used": False}
    return Posterior(cross, c, factor, solved.conj().T, posterior.real, prior, indices, diagnostics)


def metric(posterior, data):
    error, prior = float(posterior.diagonal[data].sum()), float(posterior.prior[data].sum())
    ratio = error/prior
    return {"posterior_trace": error, "prior_trace": prior, "nmse_linear": ratio, "nmse_db": float(10*np.log10(ratio))}


def per_link_reference(rf, rt, geometry, snr):
    full = lmmse.build_full_grid_covariance(rf, rt)
    errors, priors, records = [], [], []
    for tx in range(2):
        p = lmmse.prepare_oracle_lmmse(full, lmmse.pilot_flat_indices(geometry.layout.pilot_mask[tx]), noise_variance_from_snr(snr), (64, 16))
        mask = geometry.layout.data_mask[tx].ravel(order="C")
        errors.append(float(p.posterior_diagonal[mask].sum()))
        priors.append(float(p.prior_diagonal[mask].sum()))
        records.append(p)
    error, prior = 2*sum(errors), 2*sum(priors)
    return {"posterior_trace": error, "prior_trace": prior, "nmse_linear": error/prior,
            "nmse_db": float(10*np.log10(error/prior))}, records


def mode_rhos(mode, rho):
    if mode not in MODES:
        raise ValueError("unknown correlation mode")
    return (rho if mode != "RX-only" else 0., rho if mode != "TX-only" else 0.)


def ordering_audit(geometry):
    tx = np.array([[1., .2+.35j], [.2-.35j, 1.]])
    rx = np.array([[1., .6-.1j], [.6+.1j, 1.]])
    def root(m):
        e, v = np.linalg.eigh(m)
        return (v*np.sqrt(e))@v.conj().T
    generated = []
    for q in range(2):
        for r in range(2):
            basis = np.zeros((2, 2), complex); basis[r, q] = 1
            generated.append((root(rx)@basis@root(tx)).ravel(order="F"))
    f = np.stack(generated, axis=1)
    expected = np.kron(tx.T, rx)
    complex_error = float(np.abs(f@f.conj().T-expected).max())
    h = np.empty((2, 2, 64, 16), dtype=complex)
    explicit = np.empty(4096, dtype=complex)
    observed = []
    for r in range(2):
        for q in range(2):
            for k in range(64):
                for t in range(16):
                    value = 100000*r+10000*q+100*k+t+1j*(r+3*q+5*k+7*t)
                    h[r, q, k, t] = value
                    explicit[((r*2+q)*64+k)*16+t] = value
                    if geometry.layout.pilot_mask[q, k, t]: observed.append(value)
    project = pack_project(h)
    selector_exact = bool(np.array_equal(geometry.selector@project, np.array(observed)))
    column = h[:, :, 3, 7].ravel(order="F")
    permuted_exact = bool(np.array_equal(column[[0,2,1,3]], h[:, :, 3, 7].ravel(order="C")))
    passed = complex_error <= TOL and selector_exact and permuted_exact and np.array_equal(project, explicit)
    return {"complex_nonsymmetric_vec_covariance_max_error": complex_error,
            "project_index_explicit_loop_exact": bool(np.array_equal(project, explicit)),
            "single_point_permutation_exact": permuted_exact, "S_explicit_loop_observation_exact": selector_exact,
            "column_to_project_permutation": [0,2,1,3], "all_checks_passed": bool(passed)}


def spatial_generation_audit():
    # Unequal real correlations expose TX/RX swaps, without using a joint prior factor.
    tx, rx, expected, _ = spatial_covariances(.3, .6)
    # Symmetric Hermitian square roots for the independent matrix-generation identity.
    et, vt = np.linalg.eigh(tx); er, vr = np.linalg.eigh(rx)
    st = (vt*np.sqrt(et))@vt.conj().T; sr = (vr*np.sqrt(er))@vr.conj().T
    w = complex_gaussian_noise((SPATIAL_SAMPLES, 2, 2), 1., np.random.default_rng(SPATIAL_SEED))
    h = sr @ w @ st
    vec = h.transpose(0, 2, 1).reshape(-1, 4)
    empirical = vec.T@vec.conj()/SPATIAL_SAMPLES
    relative = float(np.linalg.norm(empirical-expected)/np.linalg.norm(expected))
    pseudo = vec.T@vec/SPATIAL_SAMPLES
    return {"samples": SPATIAL_SAMPLES, "seed": SPATIAL_SEED, "rho_tx": .3, "rho_rx": .6,
            "relative_Frobenius_error": relative, "relative_error_limit": SPATIAL_RELATIVE_LIMIT,
            "max_pseudocovariance_magnitude": float(np.abs(pseudo).max()),
            "all_checks_passed": relative <= SPATIAL_RELATIVE_LIMIT}


def monte_carlo_audit(joint, geometry, posterior):
    """Direct joint Gaussian factor draw, not a correlated TDL trajectory API."""
    fs, spatial_diag = covariance_eigenfactor(joint.spatial)
    ft, temporal_diag = covariance_eigenfactor(joint.time)
    delays, powers = tdl_a_profile(100.)
    ff = np.exp(-2j*np.pi*centered_frequency_grid_hz(nfft=64, delta_f_hz=30000.)[:, None]*delays)*np.sqrt(powers)
    frequency_error = float(np.abs(ff@ff.conj().T-joint.frequency).max())
    if max(frequency_error, spatial_diag["factor_reconstruction_max_error"], temporal_diag["factor_reconstruction_max_error"]) > TOL:
        raise ValueError("joint Gaussian factor reconstruction failed")
    channel_rng = np.random.default_rng(MC_CHANNEL_SEED)
    noise_rng = np.random.default_rng(MC_NOISE_SEED)
    energies = []
    truth = 0.
    for start in range(0, MC_SAMPLES, 256):
        n = min(256, MC_SAMPLES-start)
        # F_joint = F_sp_project tensor F_frequency tensor F_time.
        w = complex_gaussian_noise((n, 4, len(powers), 16), 1., channel_rng)
        temp = w@ft.T
        grid = np.einsum("kl,nalt->nakt", ff, temp, optimize=True)
        h = np.einsum("ba,nakt->nbkt", fs, grid, optimize=True).reshape(n, -1)
        z = h[:, geometry.selected] + complex_gaussian_noise((n, geometry.selected.size), noise_variance_from_snr(10), noise_rng)
        error = (posterior.estimate(z)-h)[:, geometry.data]
        energies.extend(np.sum(np.abs(error)**2, axis=1).tolist())
        truth += float(np.sum(np.abs(h[:, geometry.data])**2))
    samples = np.asarray(energies)
    empirical = float(samples.mean())
    theory = metric(posterior, geometry.data)["posterior_trace"]
    relative = abs(empirical/theory-1)
    return {"samples": MC_SAMPLES, "rho_tx": .6, "rho_rx": .6, "snr_db": 10,
            "channel_seed": MC_CHANNEL_SEED, "noise_seed": MC_NOISE_SEED,
            "seed_namespace": "T14B synthetic covariance correctness audit; no bits/training/formal data",
            "empirical_error_trace": empirical, "theoretical_error_trace": theory,
            "relative_error": relative, "relative_error_limit": MC_RELATIVE_LIMIT,
            "empirical_error_trace_standard_error": float(samples.std(ddof=1)/np.sqrt(MC_SAMPLES)),
            "empirical_truth_trace": truth/MC_SAMPLES, "prior_data_trace": float(posterior.prior[geometry.data].sum()),
            "empirical_NMSE_fixed_theoretical_denominator": empirical/float(posterior.prior[geometry.data].sum()),
            "empirical_NMSE_sample_energy_ratio": empirical/(truth/MC_SAMPLES),
            "frequency_factor_max_error": frequency_error, "temporal_factor": temporal_diag,
            "all_checks_passed": relative <= MC_RELATIVE_LIMIT, "communication_experiment": False}


def scientific_flags(rows):
    def gain(rho, snr):
        return next(r["spatial_gain_db"] for r in rows if r["mode"] == "BOTH" and r["rho_tx"] == rho and r["snr_db"] == snr)
    moderate, high = gain(.6, 20), gain(.9, 20)
    return {"moderate_spatial_correlation_material": moderate >= 1.,
            "spatial_information_gain_weak": all(gain(.6, s) < .5 for s in SNRS) and high < 1.,
            "spatial_gain_material_only_at_high_correlation": moderate < 1. and high >= 1.,
            "final_paper_should_include_spatially_correlated_evaluation": moderate >= .5,
            "high_only_definition": "moderate 20dB gain <1dB and high 20dB gain >=1dB",
            "scientific_thresholds_db": {"moderate_material": 1., "weak": .5, "high_material": 1., "reviewer": .5}}
