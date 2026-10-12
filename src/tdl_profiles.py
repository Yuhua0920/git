"""T14A NLOS TDL profiles; A delegates to unchanged historical implementations."""
from dataclasses import dataclass

import numpy as np

from src import lmmse,tdl
from src.doppler import correlated_complex_gaussian
from src.trajectories import prepare_trajectory,slice_channel_frames

STANDARD_URL=tdl.STANDARD_URL
STANDARD_REFERENCE=tdl.STANDARD_REFERENCE
TABLES={"A":"7.7.2-1","B":"7.7.2-2","C":"7.7.2-3"}
NORMALIZED_DELAYS={"A":tdl.TDL_A_NORMALIZED_DELAYS,
    "B":(0.,.1072,.2155,.2095,.2870,.2986,.3752,.5055,.3681,.3697,.5700,.5283,
        1.1021,1.2756,1.5474,1.7842,2.0169,2.8294,3.0219,3.6187,4.1067,4.2790,4.7834),
    "C":(0.,.2099,.2219,.2329,.2176,.6366,.6448,.6560,.6584,.7935,.8213,.9336,
        1.2285,1.3083,2.1704,2.7105,4.2589,4.6003,5.4902,5.6077,6.3065,6.6374,7.0427,8.6523)}
POWERS_DB={"A":tdl.TDL_A_POWERS_DB,
    "B":(0.,-2.2,-4.,-3.2,-9.8,-1.2,-3.4,-5.2,-7.6,-3.,-8.9,-9.,-4.8,-5.7,-7.5,-1.9,-7.6,-12.2,-9.8,-11.4,-14.9,-9.2,-11.3),
    "C":(-4.4,-1.2,-3.5,-5.2,-2.5,0.,-2.2,-3.9,-7.4,-7.1,-10.7,-11.1,-5.1,-6.8,-8.7,-13.2,-13.9,-13.9,-15.8,-17.1,-16.,-15.7,-21.6,-22.8)}


def profile_parameters(profile,nominal_ds_ns=100.):
    if profile not in TABLES:
        raise ValueError("only NLOS TDL-A/B/C allowed")
    if profile=="A":
        return tdl.tdl_a_profile(nominal_ds_ns)
    if not np.isfinite(nominal_ds_ns) or nominal_ds_ns<=0:
        raise ValueError("positive finite nominal delay scale required")
    delays=np.asarray(NORMALIZED_DELAYS[profile])*nominal_ds_ns*1e-9
    raw=10**(np.asarray(POWERS_DB[profile])/10)
    return delays,raw/raw.sum()


def frequency_covariance(profile,nominal_ds_ns=100.):
    if profile=="A":
        return lmmse.build_frequency_covariance(desired_rms_delay_spread_ns=nominal_ds_ns)
    delays,powers=profile_parameters(profile,nominal_ds_ns)
    f=tdl.centered_frequency_grid_hz(64,30000)
    return np.sum(powers*np.exp(-2j*np.pi*(f[:,None]-f[None,:])[...,None]*delays),axis=-1)


def profile_diagnostics(profile):
    delays,powers=profile_parameters(profile)
    return {"profile":"TDL-"+profile,"table":TABLES[profile],"tap_count":len(delays),
        "nominal_delay_spread_ns":100.,"total_normalized_power":float(powers.sum()),
        "actual_RMS_delay_ns":tdl.rms_delay_spread_seconds(delays,powers)*1e9,
        "mean_delay_ns":float(np.sum(delays*powers)*1e9),"maximum_physical_delay_ns":float(delays.max()*1e9),
        "fading_distribution":"Rayleigh for every tap","random_realization_normalization":False}


@dataclass(frozen=True)
class PreparedProfileTrajectory:
    profile: str
    template: object
    delays_s: np.ndarray
    powers: np.ndarray
    phase_matrix: np.ndarray

    @property
    def covariance(self):
        return self.template.covariance

    @property
    def factorization_summary(self):
        return self.template.factorization_summary

    @property
    def mobility(self):
        return self.template.mobility

    def generate_taps(self,channel_seed):
        if isinstance(channel_seed,(bool,np.bool_)) or not isinstance(channel_seed,(int,np.integer)) or channel_seed<0:
            raise ValueError("nonnegative integer channel seed required")
        processes=correlated_complex_gaussian(np.random.default_rng(channel_seed),self.template.factor,(2,2,len(self.powers)))
        return processes*np.sqrt(self.powers)[None,None,:,None]

    def frequency_response(self,taps):
        taps=np.asarray(taps)
        if taps.dtype!=np.complex128 or taps.shape!=(2,2,len(self.powers),self.template.time_grid_s.size):
            raise ValueError("complex128 taps must match profile count and continuous time grid")
        return np.moveaxis(np.moveaxis(taps,-2,-1)@self.phase_matrix,-2,-1)


def prepare_profile_trajectory(profile,frames_per_trajectory=5):
    """A returns the historical object; B/C replace only PDP and tap count."""
    if profile not in TABLES:
        raise ValueError("only A/B/C")
    template=prepare_trajectory(frames_per_trajectory=frames_per_trajectory,speed_kmh=30.)
    if profile=="A":
        return template
    delays,powers=profile_parameters(profile)
    phase=tdl.fractional_delay_frequency_response(np.eye(len(powers),dtype=np.complex128),delays,template.frequencies_hz)
    for a in (delays,powers,phase):
        a.setflags(write=False)
    return PreparedProfileTrajectory(profile,template,delays,powers,phase)


def draw_profile_channel(model,channel_seed):
    taps=model.generate_taps(channel_seed); h=model.frequency_response(taps)
    taps.setflags(write=False); h.setflags(write=False)
    frames=slice_channel_frames(h)
    return taps,h,frames
