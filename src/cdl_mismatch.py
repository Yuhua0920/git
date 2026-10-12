"""T15C two declared CDL shifts, frozen source SNR, train-only diagnostics."""
import copy
import hashlib
import numpy as np
from scipy.linalg import cho_factor, cho_solve, subspace_angles
from src.cdl_benchmark import OfficialCDL, ARRAY, SYSTEM, SYMBOL_SECONDS

T15B_COMMIT = "448092c93b6bad39c9862929c8c4820c178cf83e"
SHIFTS = {"CDL-B-30": dict(profile="B", speed_kmh=30., namespace_code=15101),
          "CDL-C-120": dict(profile="C", speed_kmh=120., namespace_code=15102)}
METHODS = ("R3", "R10-C-frozen", "R10-matched", "Perfect-CSI")
SNRS = (5, 10, 20)
EXPERIMENT_TYPE = "realistic CDL covariance-mismatch gate for source-trained empirical LMMSE under very sparse pilots"


def shift_definition(name):
    s = SHIFTS[name]
    system, array = copy.deepcopy(SYSTEM), copy.deepcopy(ARRAY)
    system["speed_kmh"] = s["speed_kmh"]
    array["ut_velocity_mps"] = [s["speed_kmh"]/3.6, 0., 0.]
    return dict(profile=s["profile"], system=system, array=array,
                ofdm_channel_normalize=False, dF=16, spacing_time=4)


def seeds(name, split, trajectory, frame=0):
    limits = {"train_dev": range(500), "validation_dev": range(500,600)}
    if split not in limits or trajectory not in limits[split] or frame not in range(10):
        raise ValueError("dedicated train/validation namespace and ten frames required")
    role_names = ("sionna_seed", "torch_seed", "numpy_seed", "bits_seed_tx0", "bits_seed_tx1", "noise_seed")
    split_code = 1 if split == "train_dev" else 2
    return {role: int(np.random.SeedSequence([20261531, SHIFTS[name]["namespace_code"],
             split_code, trajectory, 0 if j<=3 else frame, j]).generate_state(1)[0])
            for j, role in enumerate(role_names, 1)}


def manifest_rows():
    return [dict(shift=name, namespace=f"T15C_{name}_OOD_v1", split=split,
                 trajectory_id=i, frame_id=j, start_symbol=16*j, end_symbol=16*j+15,
                 **seeds(name,split,i,j))
            for name in SHIFTS for split,ids in (("train_dev",range(500)),("validation_dev",range(500,600)))
            for i in ids for j in range(10)]


class ShiftCDL(OfficialCDL):
    """Same calibrated array/call/export, only declared profile and speed change."""
    def __init__(self, name, device=None):
        super().__init__(device)
        from sionna.phy.channel.tr38901 import CDL
        self.name=name
        s=shift_definition(name)
        self.model=CDL(model=s["profile"], delay_spread=100e-9, carrier_frequency=3.5e9,
            ut_array=self.array, bs_array=self.array, direction="downlink",
            ut_orientation=ARRAY["ut_orientation_radians"], bs_orientation=ARRAY["bs_orientation_radians"],
            ut_velocity=s["array"]["ut_velocity_mps"], min_speed=s["system"]["speed_kmh"]/3.6,
            max_speed=s["system"]["speed_kmh"]/3.6, normalize_delays=True,
            precision="double", device=self.device, spec_version="19.2")

    def generate(self, trajectory, *, split):
        import torch
        from sionna.phy import config
        from sionna.phy.channel import cir_to_ofdm_channel
        s=seeds(self.name,split,trajectory)
        config.seed=s["sionna_seed"];torch.manual_seed(s["torch_seed"])
        self.numpy_rng=np.random.default_rng(s["numpy_seed"])
        with torch.no_grad():
            a,tau=self.model(batch_size=1,num_time_steps=160,sampling_frequency=1/SYMBOL_SECONDS)
            hf=cir_to_ofdm_channel(self.frequencies,a,tau,normalize=False)
        h=hf[0,0,:,0,:,:,:].permute(0,1,3,2).cpu().numpy().astype(np.complex128,copy=True)
        if h.shape!=(2,2,64,160) or not np.isfinite(h).all():raise ValueError("invalid CDL export")
        return h

    def provenance(self):
        from pathlib import Path
        from sionna.phy.channel.tr38901 import models
        p=super().provenance()
        p["source_hashes"]={k:v for k,v in p["source_hashes"].items() if not k.endswith("CDL-C.json")}
        path=Path(models.parameter_file(f"CDL-{SHIFTS[self.name]['profile']}.json","19.2"))
        p["source_hashes"][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        p["profile"]=f"CDL-{SHIFTS[self.name]['profile']}"
        p["shift_definition"]=shift_definition(self.name)
        return p


def frozen_variances(source_result):
    """Production uses source training only, never target power."""
    power=float(source_result["train_reference_signal_power"])
    values={s:power/10**(s/10) for s in SNRS}
    saved={int(r["snr_db"]):float(r["new_sigma2"]) for r in source_result["noise_variances"]}
    if values!=saved:raise ValueError("source variance identity drift")
    return power,values


def gaussian_kl(mean_a,cov_a,mean_b,cov_b):
    """Proper-complex KL a||b (nats), SPD noisy pilot marginals, no 1/2."""
    fa=cho_factor(cov_a,lower=True);fb=cho_factor(cov_b,lower=True)
    log_a=2*np.log(np.diag(fa[0]).real).sum();log_b=2*np.log(np.diag(fb[0]).real).sum()
    delta=mean_b-mean_a
    return float(log_b-log_a-len(mean_a)+np.trace(cho_solve(fb,cov_a)).real+
                 np.vdot(delta,cho_solve(fb,delta)).real)


def covariance_diagnostics(source,target,variances):
    ea,ua=np.linalg.eigh(source.rpp);eb,ub=np.linalg.eigh(target.rpp)
    result=dict(rpp_relative_frobenius=float(np.linalg.norm(target.rpp-source.rpp)/np.linalg.norm(source.rpp)),
                rdp_relative_frobenius=float(np.linalg.norm(target.rdp-source.rdp)/np.linalg.norm(source.rdp)),
                denominator="source Frobenius norm",mean_p_distance=float(np.linalg.norm(target.mean_p-source.mean_p)),
                mean_d_distance=float(np.linalg.norm(target.mean_d-source.mean_d)),subspaces={},pilot_gaussian_kl={})
    for r in (8,16):
        result["subspaces"][str(r)]=dict(angles_degrees=np.rad2deg(subspace_angles(ua[:,-r:],ub[:,-r:])).tolist(),
            source_eigengap=float(ea[-r]-ea[-r-1]),target_eigengap=float(eb[-r]-eb[-r-1]))
    for snr,sigma in variances.items():
        a=source.rpp+sigma*np.eye(64);b=target.rpp+sigma*np.eye(64)
        result["pilot_gaussian_kl"][str(snr)]=dict(source_to_target_nats=gaussian_kl(source.mean_p,a,target.mean_p,b),
            target_to_source_nats=gaussian_kl(target.mean_p,b,source.mean_p,a),sigma2=sigma)
    result["KL_definition"]="proper-complex Gaussian LS-pilot marginals, covariance Rpp+production sigma2 I; mechanism diagnostic only, not joint Gaussianity proof"
    return result


def scientific_flags(penalties):
    material=any(r["snr_db"] in (10,20) and r["mismatch_penalty_db"]>=2. for r in penalties)
    weak=all(r["mismatch_penalty_db"]<.5 for r in penalties)
    return dict(realistic_cdl_adaptation_gap_material=material,
                realistic_cdl_adaptation_gap_weak=weak,
                realistic_cdl_adaptation_gap_partial=not material and not weak)


def paired_summary(frozen,matched):
    delta=np.array([a["error_energy"]-b["error_energy"] for a,b in zip(frozen,matched,strict=True)])
    gain=10*np.log10(np.array([a["nmse_linear"]/b["nmse_linear"] for a,b in zip(frozen,matched,strict=True)]))
    ber=np.array([a["ber"]-b["ber"] for a,b in zip(frozen,matched,strict=True)])
    return dict(independent_unit="trajectory",trajectories=len(delta),
        error_energy_difference_q25_median_q75=np.quantile(delta,[.25,.5,.75]).tolist(),
        penalty_db_q25_median_q75=np.quantile(gain,[.25,.5,.75]).tolist(),
        ber_difference_q25_median_q75=np.quantile(ber,[.25,.5,.75]).tolist(),
        fraction_matched_lower_error=float(np.mean(delta>0)),fraction_matched_lower_ber=float(np.mean(ber>0)))
