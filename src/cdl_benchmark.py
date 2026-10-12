"""T15A pinned official CDL adapter and train-only empirical statistics.

Sionna is imported lazily; historical main-environment modules stay untouched.
Project vector order is C order [rx,tx,k,t], spatial pair a=2*rx+tx.
"""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from src.spatial_information import observation_geometry

T14B_COMMIT = "f3c1d058fed3592cdb408038e0539a14b521489d"
SNRS = (5, 10, 20)
PREFIXES = (100, 250, 500)
SYMBOL_SECONDS = 80/1920000
EXPERIMENT_TYPE = "realistic 3GPP CDL-C benchmark integration and Gaussian-assumption audit for sparse-pilot 2x2 MIMO-OFDM"
SYSTEM = dict(nt=2,nr=2,nfft=64,num_symbols=16,cp_len=16,delta_f_hz=30000,
    sample_rate_hz=1920000,carrier_frequency_hz=3.5e9,total_power=1.,
    fft_norm="ortho",modulation="qpsk_gray",nominal_delay_spread_seconds=100e-9,
    speed_kmh=30.,symbol_duration_seconds=SYMBOL_SECONDS,snr_db=list(SNRS))
ARRAY = dict(rows=1,columns=2,panels=[1,1],local_ULA_axis="y",
    element_spacing_wavelengths=.5,polarization="single",polarization_type="V",
    element_pattern="38.901",bs_orientation_radians=[0.,0.,0.],
    ut_orientation_radians=[float(np.pi),0.,0.],ut_velocity_mps=[30/3.6,0.,0.])


def seeds(trajectory, frame=0):
    """New experiment namespace; roles are distinct SeedSequence inputs."""
    def one(role):
        return int(np.random.SeedSequence([20261530,15001,trajectory,frame,role]).generate_state(1)[0])
    return {"sionna_seed":one(1),"torch_seed":one(2),"numpy_seed":one(3),
        "bits_seed_tx0":one(4),"bits_seed_tx1":one(5),"noise_seed":one(6)}


def manifest_rows():
    rows=[]
    for i in range(600):
        trajectory_seeds=seeds(i,0)
        for j in range(10):
            used_seeds=seeds(i,j)
            for role in ("sionna_seed","torch_seed","numpy_seed"):
                used_seeds[role]=trajectory_seeds[role]
            rows.append(dict(namespace="T15A_CDL_DEV_v1",split="train_dev" if i<500 else "validation_dev",
                trajectory_id=i,frame_id=j,start_symbol=16*j,end_symbol=16*j+15,**used_seeds))
    return rows


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


class OfficialCDL:
    """One official 160-state call per trajectory, never one redraw per frame."""
    def __init__(self, device=None):
        if importlib.metadata.version("sionna-no-rt")!="2.2.0":
            raise ValueError("T15A requires sionna-no-rt 2.2.0")
        import torch
        from sionna.phy.channel import subcarrier_frequencies
        from sionna.phy.channel.tr38901 import CDL, PanelArray
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.array = PanelArray(num_rows_per_panel=1,num_cols_per_panel=2,
            polarization="single",polarization_type="V",antenna_pattern="38.901",
            carrier_frequency=3.5e9,element_horizontal_spacing=.5,element_vertical_spacing=.5,
            precision="double",device=self.device)
        self.model = CDL(model="C",delay_spread=100e-9,carrier_frequency=3.5e9,
            ut_array=self.array,bs_array=self.array,direction="downlink",
            ut_orientation=ARRAY["ut_orientation_radians"],bs_orientation=ARRAY["bs_orientation_radians"],
            ut_velocity=ARRAY["ut_velocity_mps"],min_speed=30/3.6,max_speed=30/3.6,
            normalize_delays=True,precision="double",device=self.device,spec_version="19.2")
        self.frequencies = subcarrier_frequencies(64,30000,precision="double",device=self.device)

    def generate(self, trajectory, *, details=False):
        import torch
        from sionna.phy import config
        from sionna.phy.channel import cir_to_ofdm_channel
        s = seeds(trajectory)
        config.seed=s["sionna_seed"]; torch.manual_seed(s["torch_seed"])
        # Explicit local NumPy seed; no mutation of global NumPy random state.
        self.numpy_rng=np.random.default_rng(s["numpy_seed"])
        with torch.no_grad():
            a,tau=self.model(batch_size=1,num_time_steps=160,sampling_frequency=1/SYMBOL_SECONDS)
            hf=cir_to_ofdm_channel(self.frequencies,a,tau,normalize=False)
        h=hf[0,0,:,0,:,:,:].permute(0,1,3,2).cpu().numpy().astype(np.complex128,copy=True)
        if h.shape!=(2,2,64,160) or not np.isfinite(h).all(): raise ValueError("invalid CDL export")
        if details:
            return h,a[0,0,:,0].cpu().numpy().astype(np.complex128),tau[0,0,0].cpu().numpy().astype(np.float64)
        return h

    def provenance(self):
        import inspect
        import torch
        from sionna.phy.channel.tr38901 import models
        from sionna.phy.channel import cir_to_ofdm_channel
        paths=[inspect.getfile(type(self.model)),inspect.getfile(type(self.array)),
            inspect.getfile(type(self.model._cir_gen)),inspect.getfile(cir_to_ofdm_channel),
            str(models.parameter_file("CDL-C.json","19.2"))]
        from pathlib import Path
        return dict(version="2.2.0",distribution="sionna-no-rt",spec_version="19.2",
            standard="3GPP TR 38.901 V19.2.0",profile="CDL-C",los=False,
            license="Apache-2.0",source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
            normalized_PDP_powers=self.model._powers.cpu().tolist(),
            scaled_cluster_delays_seconds=self.model._delays.cpu().tolist(),
            array_positions_m=self.array.ant_pos.cpu().tolist(),
            device=self.device,internal_dtype="torch.complex128 / torch.float64",
            export_dtype="numpy.complex128 / numpy.float64",ofdm_channel_normalize=False,
            torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            reproducibility_scope="same installed environment, device, seeds, precision and call shape only",
            time_states_per_call=160,sampling_frequency_hz=1/SYMBOL_SECONDS,
            official_docs="https://nvlabs.github.io/sionna/phy/api/channel/wireless/sionna.phy.channel.tr38901.CDL.html")


def frames(h):
    if h.shape!=(2,2,64,160): raise ValueError("ten continuous frames required")
    return h.reshape(2,2,64,10,16).transpose(3,0,1,2,4)


@dataclass(frozen=True)
class FrozenStatistics:
    mean_p: np.ndarray
    mean_d: np.ndarray
    rpp: np.ndarray
    rdp: np.ndarray
    train_trajectory_ids: tuple
    samples: int

    def prepare(self, sigma2):
        if not np.isfinite(sigma2) or sigma2<=0: raise ValueError("strictly positive noise variance required")
        factor=cho_factor(self.rpp+sigma2*np.eye(self.rpp.shape[0]),lower=True,check_finite=True)
        # A X = R_pd, followed by adjoint gives R_dp A^-1.
        weight=cho_solve(factor,self.rdp.conj().T).conj().T
        residual=np.linalg.norm(weight@(self.rpp+sigma2*np.eye(self.rpp.shape[0]))-self.rdp)/max(np.linalg.norm(self.rdp),1e-300)
        return EmpiricalEstimator(self,weight,float(sigma2),float(residual))


@dataclass(frozen=True)
class EmpiricalEstimator:
    statistics: FrozenStatistics
    weight: np.ndarray
    sigma2: float
    solve_relative_residual: float

    def estimate(self, z):
        z=np.asarray(z,dtype=np.complex128)
        if z.shape[-1]!=self.statistics.mean_p.size or not np.isfinite(z).all(): raise ValueError("invalid pilot vector")
        return self.statistics.mean_d+(z-self.statistics.mean_p)@self.weight.T


class StreamingStatistics:
    """Centered batch/Welford covariance, population convention 1/N.

    Reject validation IDs at update boundary. Full covariance is optional RAM-only
    audit scratch; Rpp/Rdp are independently streamed with the correct adjoints.
    """
    def __init__(self, selected, data, dimension=4096, *, full_audit=False):
        self.selected=np.asarray(selected); self.data=np.asarray(data)
        self.n=0; self.ids=[]; self.mean=np.zeros(dimension,np.complex128)
        self.mpp=np.zeros((len(selected),len(selected)),np.complex128)
        self.mdp=np.zeros((int(np.count_nonzero(data)),len(selected)),np.complex128)
        self.mfull=np.zeros((dimension,dimension),np.complex128) if full_audit else None

    def update(self, rows, trajectory_ids):
        if any(i<0 or i>=500 or i in self.ids for i in trajectory_ids): raise ValueError("only new train_dev trajectories may fit statistics")
        x=np.asarray(rows,np.complex128)
        if x.ndim!=2 or x.shape[1]!=self.mean.size or not np.isfinite(x).all(): raise ValueError("invalid training rows")
        count=len(x); mean=x.mean(axis=0); xc=x-mean; delta=mean-self.mean
        correction=self.n*count/(self.n+count)
        p,d=xc[:,self.selected],xc[:,self.data]
        self.mpp+=p.T@p.conj()+correction*np.outer(delta[self.selected],delta[self.selected].conj())
        self.mdp+=d.T@p.conj()+correction*np.outer(delta[self.data],delta[self.selected].conj())
        if self.mfull is not None:
            self.mfull+=xc.T@xc.conj()+correction*np.outer(delta,delta.conj())
        self.mean+=delta*count/(self.n+count); self.n+=count; self.ids.extend(trajectory_ids)

    def freeze(self):
        if self.n<1: raise ValueError("empty training statistics")
        values=[self.mean[self.selected].copy(),self.mean[self.data].copy(),self.mpp/self.n,self.mdp/self.n]
        for a in values: a.flags.writeable=False
        return FrozenStatistics(*values,tuple(self.ids),self.n)


def marginal_kronecker_audit(cov, shape=(4,64,16)):
    """Partial-trace/marginal Kronecker metric, not a best-fit claim.

    Exact for a separable covariance. Returns full-grid relative Frobenius
    residual and optimal scalar for these fixed marginal factors. No inverse.
    """
    a,k,t=shape; r=np.asarray(cov).reshape(a,k,t,a,k,t)
    rs=np.einsum("aktbkt->ab",r)/(k*t)
    rf=np.einsum("aktalt->kl",r)/(a*t)
    rt=np.einsum("aktaku->tu",r)/(a*k)
    power=float(np.trace(cov).real/(a*k*t))
    approximation=np.kron(np.kron(rs,rf),rt)/power**2
    scalar=float(np.vdot(approximation,cov).real/np.vdot(approximation,approximation).real)
    residual=float(np.linalg.norm(cov-scalar*approximation)/np.linalg.norm(cov))
    return dict(metric="partial-trace marginal Kronecker factors with least-squares scalar",
        best_full_Kronecker_fit_claimed=False,mean_power=power,optimal_scalar=scalar,
        relative_Frobenius_residual=residual,spatial=rs,frequency=rf,time=rt)


def moments(x):
    x=np.asarray(x,np.float64); centered=x-x.mean(); variance=float(np.mean(centered**2))
    return dict(mean=float(x.mean()),variance=variance,
        skewness=float(np.mean(centered**3)/variance**1.5),
        excess_kurtosis=float(np.mean(centered**4)/variance**2-3))


def normalized_correlation(x,y):
    x=np.asarray(x).ravel(); y=np.asarray(y).ravel()
    x=x-x.mean(); y=y-y.mean()
    return complex(np.vdot(y,x)/np.sqrt(np.vdot(x,x).real*np.vdot(y,y).real))
