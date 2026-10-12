"""T14A profile-shift development pool and theoretical TDL-A oracle projection."""
from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.optimize import minimize_scalar

from src import link,lmmse,ood_mismatch as old,statistics_information as info,tdl_profiles as profiles
from src.lmmse_train import prepare_train_lmmse,raw_array_sha256
from src.lmmse_train_validation import batched_data_zf
from src.metrics import channel_error_energies,count_bit_errors
from src.numerics import complex_gaussian_noise,noise_variance_from_snr
from src.tdl import rms_delay_spread_seconds
from src.tdl_awgn_scan import scale_unit_noise

T13D_COMMIT="2045c837a66c94bd41b78b344db0e9538d99a2ac"
PREFIX="t14a_profile_shape_gate"
BASE_SEED=20261401; NUM_TRAJECTORIES=200; FRAMES=5
SNRS=(5,10,20); PROFILES=("B","C"); CODES={"B":1,"C":2}
LAYOUT=old.LAYOUT
METHODS=("R3","R4","R8","R5","B0")
TIMING_KEYS=("channel_generation_seconds","R3_seconds","R4_seconds","R8_seconds","R5_seconds","B0_inference_seconds","ZF_seconds")
LIMITS=dict(old.LIMITS)
COARSE_TAUS=tuple(range(20,601,10))


def seed_words(role,profile,tid,frame=0,tx=0,purpose=1):
    if (role not in (14001,14002,14003) or profile not in CODES or purpose not in (0,1)
            or any(isinstance(v,bool) or not isinstance(v,int) for v in (tid,frame,tx))
            or not 0<=tid<NUM_TRAJECTORIES or not 0<=frame<FRAMES or tx not in (0,1)):
        raise ValueError("dedicated profile-shape development coordinates required")
    return [role,BASE_SEED,purpose,CODES[profile],tid,frame,tx]


def seed(role,profile,tid,frame=0,tx=0,purpose=1):
    return int(np.random.SeedSequence(seed_words(role,profile,tid,frame,tx,purpose)).generate_state(1,dtype=np.uint64)[0])


def manifest():
    return [{"split":"profile_shape_dev","profile":p,"trajectory_id":tid,"channel_seed":seed(14001,p,tid),
        "frames_per_trajectory":5,"speed_kmh":30.,"nominal_delay_spread_ns":100.}
        for p in PROFILES for tid in range(NUM_TRAJECTORIES)]


def calibrate_channel(model,profile):
    power=0.; count=0; tap_power=np.zeros(len(model.powers))
    temporal={lag:np.zeros(3,dtype=complex) for lag in (1,4)}
    frequency={lag:np.zeros(4,dtype=complex) for lag in (1,4,8,16,32)}
    for tid in range(NUM_TRAJECTORIES):
        taps,h,_=profiles.draw_profile_channel(model,seed(14001,profile,tid,purpose=0))
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(taps)):
            raise ValueError("nonfinite profile calibration draw")
        power+=float(np.sum(np.abs(h)**2)); count+=h.size
        tap_power+=np.mean(np.abs(taps)**2,axis=(0,1,3))
        unit=taps/np.sqrt(model.powers)[None,None,:,None]
        for lag,values in temporal.items():
            a,b=unit[...,lag:],unit[...,:-lag]
            values+=[np.sum(a*b.conj()),np.sum(np.abs(a)**2),np.sum(np.abs(b)**2)]
        for lag,values in frequency.items():
            a,b=h[:,:,lag:],h[:,:,:-lag]
            values+=[np.sum(a*b.conj()),np.sum(np.abs(a)**2),np.sum(np.abs(b)**2),a.size]
    rf=profiles.frequency_covariance(profile); time_records=[]; freq_records=[]
    for lag,(cross,a,b) in temporal.items():
        empirical=cross/np.sqrt(a.real*b.real); theory=model.covariance[lag,0]
        error=float(abs(empirical-theory))
        time_records.append({"lag":lag,"empirical":old.complex_scalar(empirical),"theory":float(theory),
            "absolute_error":error,"passed":error<=LIMITS["temporal_correlation_abs"]})
    for lag,(cross,a,b,n) in frequency.items():
        raw=cross/n.real; corr=cross/np.sqrt(a.real*b.real); theory=rf[lag,0]
        raw_error=float(abs(raw-theory)); corr_error=float(abs(corr-theory))
        freq_records.append({"lag":lag,"empirical_covariance":old.complex_scalar(raw),"empirical_correlation":old.complex_scalar(corr),
            "theory":old.complex_scalar(theory),"covariance_error":raw_error,"correlation_error":corr_error,
            "passed":raw_error<=LIMITS["frequency_covariance_abs"] and corr_error<=LIMITS["frequency_correlation_abs"]})
    pdp=tap_power/tap_power.sum(); empirical_rms=rms_delay_spread_seconds(model.delays_s,pdp)*1e9
    actual_rms=rms_delay_spread_seconds(model.delays_s,model.powers)*1e9
    checks={"mean_channel_power":abs(power/count-1)<=LIMITS["power_abs"],
        "empirical_RMS_delay":abs(empirical_rms/actual_rms-1)<=LIMITS["rms_relative"],
        "frequency_covariance":all(r["passed"] for r in freq_records),"temporal_covariance":all(r["passed"] for r in time_records),
        "factor_reconstruction":model.factorization_summary["factor_reconstruction_max_error"]<=LIMITS["factor_reconstruction_abs"]}
    return {"profile":profile,"purpose":"independent calibration only","seed_purpose":0,"num_independent_trajectories":200,
        "frames_per_trajectory":5,"channel_power":power/count,"empirical_RMS_delay_ns":empirical_rms,"actual_RMS_delay_ns":actual_rms,
        "frequency_covariance":freq_records,"temporal_covariance":time_records,"limits":LIMITS,"checks":checks,
        "factorization":model.factorization_summary,"all_checks_passed":all(checks.values())}


def require_calibration(records):
    if set(records)!=set(PROFILES) or any(not r["all_checks_passed"] for r in records.values()):
        raise ValueError("profile calibration failed; STOP before performance scan")


def observation_models(rf,snr):
    rt=lmmse.build_time_covariance(num_symbols=80,speed_kmh=30.)
    models=[]
    for nr in range(2):
        for tx in range(2):
            g=info.geometry(16,5,tx); k,t=g.coordinates.T
            c=rf[k[:,None],k[None,:]]*rt[t[:,None],t[None,:]]+noise_variance_from_snr(snr)*np.eye(len(k))
            models.append(info.prepare_covariance(c))
    return tuple(models)


def covariance_distance(reference,candidate):
    return {"relative_Frobenius_distance":float(np.linalg.norm(reference-candidate)/np.linalg.norm(reference)),
        "normalized_eigenvalue_L2_distance":float(np.linalg.norm(np.linalg.eigvalsh(reference)-np.linalg.eigvalsh(candidate))/np.linalg.norm(np.linalg.eigvalsh(reference)))}


def project_to_A_family(true_profile,snr):
    """Known true theoretical covariance only; no pilot sample or true H argument."""
    if true_profile not in ("A","B","C") or snr not in SNRS:
        raise ValueError("prescribed profile/SNR required")
    true_rf=profiles.frequency_covariance(true_profile); truth=observation_models(true_rf,snr)
    identity=info.kl_four_links(truth,truth)[0]
    if abs(identity)>4e-9:
        raise ValueError("proper-complex KL identity failed")
    evaluations=[]
    def objective(tau,phase="refinement"):
        value,per_link=info.kl_four_links(truth,observation_models(profiles.frequency_covariance("A",float(tau)),snr))
        evaluations.append({"true_profile":true_profile,"snr_db":snr,"phase":phase,"tau_ns":float(tau),"KL_total":value,
            **{f"KL_H{nr}{tx}":per_link[2*nr+tx] for nr in range(2) for tx in range(2)}})
        return value
    coarse=[objective(t,"coarse") for t in COARSE_TAUS]
    index=int(np.argmin(coarse)); chosen=float(COARSE_TAUS[index]); best=coarse[index]
    lower=max(20.,chosen-10); upper=min(600.,chosen+10)
    local=minimize_scalar(objective,bounds=(lower,upper),method="bounded",options={"xatol":1e-3,"maxiter":100})
    if not local.success or not np.isfinite(local.fun):
        raise ValueError("deterministic local projection refinement failed")
    # Keep exact coarse controls when refinement differs only by roundoff.
    if local.fun<best-1e-10:
        chosen=float(local.x); best=float(local.fun)
    projected_rf=profiles.frequency_covariance("A",chosen); projected=observation_models(projected_rf,snr)
    stale_rf=profiles.frequency_covariance("A",100.); stale=observation_models(stale_rf,snr)
    row={"true_profile":true_profile,"snr_db":snr,"family":"TDL-A","known_true_speed_kmh":30.,"tau_eff_ns":chosen,
        "coarse_tau_ns":COARSE_TAUS[index],"coarse_KL":coarse[index],"projection_KL":best,
        "A100_KL":info.kl_four_links(truth,stale)[0],"projection_boundary_hit":abs(chosen-20)<1e-3 or abs(chosen-600)<1e-3,
        "KL_identity":identity,"local_refinement_bounds_ns":[lower,upper],"local_refinement_success":bool(local.success),
        "Rf_A100_distance":covariance_distance(true_rf,stale_rf),"Rf_projected_distance":covariance_distance(true_rf,projected_rf),
        "pilot_A100_distance":covariance_distance(truth[0].covariance,stale[0].covariance),
        "pilot_projected_distance":covariance_distance(truth[0].covariance,projected[0].covariance),
        "empirical_data_used":False,"true_channel_used":False,"deployable_estimator":False}
    if true_profile=="A" and (abs(chosen-100)>1e-3 or abs(best)>4e-9):
        raise ValueError("A100 self projection failed; STOP")
    return row,evaluations,projected_rf


@dataclass(frozen=True)
class PreparedAFamilyOracle:
    tau_eff_ns: float
    estimators: tuple
    family: str="TDL-A"


def estimate_A_family(pilot_ls,prepared):
    if not isinstance(prepared,PreparedAFamilyOracle) or prepared.family!="TDL-A":
        raise ValueError("R8 must use prepared TDL-A family covariance")
    return lmmse.estimate_mimo_from_pilot_ls(pilot_ls,prepared.estimators)


def prepare_estimators(frozen,profile,projections):
    rf=profiles.frequency_covariance(profile); rt=lmmse.build_time_covariance(speed_kmh=30.)
    r5=lmmse.build_full_grid_covariance(rf,rt); pools={}; records=[]
    for snr in SNRS:
        tau=projections[profile,snr]["tau_eff_ns"]
        r8=lmmse.build_full_grid_covariance(profiles.frequency_covariance("A",tau),rt)
        noise=noise_variance_from_snr(snr)
        def oracle(c):
            return tuple(lmmse.prepare_oracle_lmmse(c,lmmse.pilot_flat_indices(LAYOUT.pilot_mask[q]),noise,(64,16)) for q in range(2))
        pools[snr]={"R4":tuple(prepare_train_lmmse(frozen,LAYOUT.pilot_mask[q],noise) for q in range(2)),
            "R5":oracle(r5),"R8":PreparedAFamilyOracle(tau,oracle(r8))}
        records.append({"profile":profile,"snr_db":snr,"R5_true_profile":profile,"R5_full_grid_sha256":raw_array_sha256(r5),
            "R8_family":"TDL-A","tau_eff_ns":tau,"R8_full_grid_sha256":raw_array_sha256(r8),"shared_time_covariance_sha256":raw_array_sha256(rt),
            "true_channel_used":False,"empirical_covariance_used":False})
    return pools,records


def evaluate_trajectory(channel_model,profile,tid,b0,pools,historical,timing):
    stamp=perf_counter(); _,long,frames=profiles.draw_profile_channel(channel_model,seed(14001,profile,tid))
    timing["channel_generation_seconds"]+=perf_counter()-stamp
    details=[]; bases=[]; hashes=[]
    for frame,h in enumerate(frames):
        bits,symbols=link.generate_master_qpsk(bits_seeds=tuple(seed(14002,profile,tid,frame,q) for q in range(2)))
        z=complex_gaussian_noise((2,64,16),1,np.random.default_rng(seed(14003,profile,tid,frame)),dtype=np.complex128)
        x=link.assemble_pilot_data_frame(symbols,LAYOUT)
        record={"frame_index":frame,"H_sha256":raw_array_sha256(h),"bits_sha256":raw_array_sha256(bits),"unit_noise_sha256":raw_array_sha256(z),"Y_by_snr":{}}
        for snr in SNRS:
            noise=scale_unit_noise(z,noise_variance_from_snr(snr))
            y=link.add_received_noise(link.apply_frequency_mimo_channel(h,x),noise)
            hx=historical["assemble_pilot_data_frame"](symbols,LAYOUT)
            hy=historical["add_received_noise"](historical["apply_frequency_mimo_channel"](h,hx),noise)
            if not np.array_equal(x,hx) or not np.array_equal(y,hy):
                raise ValueError("old physical link semantics changed")
            y.setflags(write=False); stamp=perf_counter(); ls,r3=link.estimate_ls_linear_channel(y,LAYOUT)
            timing["R3_seconds"]+=perf_counter()-stamp; estimates={"R3":r3}
            for m in ("R4","R8","R5"):
                stamp=perf_counter()
                estimates[m]=estimate_A_family(ls,pools[snr][m]) if m=="R8" else lmmse.estimate_mimo_from_pilot_ls(ls,pools[snr][m])
                timing[m+"_seconds"]+=perf_counter()-stamp
            record["Y_by_snr"][str(snr)]=raw_array_sha256(y)
            record.setdefault("pilot_LS_by_snr",{})[str(snr)]=raw_array_sha256(ls)
            bases.append(r3); details.append((frame,snr,h,bits,z,y,estimates))
        hashes.append(record)
    predictions=old.infer_b0(b0,bases,timing)
    totals={s:{"truth_energy":0.,"total_bits":0,**{m+"_error_energy":0. for m in METHODS},**{m+"_bit_errors":0 for m in METHODS}} for s in SNRS}
    for index,(frame,snr,h,bits,z,y,estimates) in enumerate(details):
        estimates["B0"]=predictions[index]; accum=totals[snr]; truth=None; y_ids=[]
        reference=link.extract_reference_data_bits(bits,LAYOUT.data_mask)
        for m in METHODS:
            error,denominator=channel_error_energies(estimates[m],h,LAYOUT.data_mask)
            if truth is not None and denominator!=truth:
                raise ValueError("NMSE denominator differs across methods")
            truth=denominator; accum[m+"_error_energy"]+=error
            stamp=perf_counter(); y_ids.append(id(y)); detected=batched_data_zf(y,estimates[m],LAYOUT.data_mask)
            timing["ZF_seconds"]+=perf_counter()-stamp
            errors,n=count_bit_errors(link.demodulate_data_bits(np.sqrt(2)*detected,LAYOUT.data_mask),reference)
            if n!=2*np.count_nonzero(LAYOUT.data_mask):
                raise ValueError("BER denominator differs from actual mask")
            accum[m+"_bit_errors"]+=errors
        accum["truth_energy"]+=truth; accum["total_bits"]+=n
        rec=hashes[frame]
        if len(set(y_ids))!=1 or any(raw_array_sha256(a)!=rec[k] for a,k in ((h,"H_sha256"),(bits,"bits_sha256"),(z,"unit_noise_sha256"))) or raw_array_sha256(y)!=rec["Y_by_snr"][str(snr)]:
            raise ValueError("paired H/bits/unit-noise/Y mutated")
    rows=[{"profile":profile,"trajectory_id":tid,"split":"profile_shape_dev","snr_db":s,"dF":16,"pilot_ratio":1/32,
        "channel_seed":seed(14001,profile,tid),"frames_aggregated":5,"same_H_bits_noise_Y":True,**totals[s]} for s in SNRS]
    return rows,{"profile":profile,"trajectory_id":tid,"H_long_sha256":raw_array_sha256(long),"frame_realizations":hashes,"historical_link_exact_checks":15}


def aggregate(rows,expected_units=200):
    result=[]
    for p in PROFILES:
        for s in SNRS:
            group=[r for r in rows if r["profile"]==p and r["snr_db"]==s]
            if len(group)!=expected_units or {r["trajectory_id"] for r in group}!=set(range(expected_units)) or any(r["frames_aggregated"]!=5 or not r["same_H_bits_noise_Y"] for r in group):
                raise ValueError("complete paired profile-trajectory pool required")
            truth=sum(r["truth_energy"] for r in group); bits=sum(r["total_bits"] for r in group)
            row={"profile":p,"snr_db":s,"dF":16,"pilot_ratio":1/32,"num_trajectories":len(group),"frames_aggregated":5*len(group),"truth_energy":truth,"total_bits":bits}
            for m in METHODS:
                error=sum(r[m+"_error_energy"] for r in group); errors=sum(r[m+"_bit_errors"] for r in group)
                row.update({m+"_error_energy":error,m+"_nmse_db":float(10*np.log10(error/truth)),m+"_bit_errors":errors,m+"_ber":errors/bits})
            row.update({"stale_gap_db":row["R4_nmse_db"]-row["R5_nmse_db"],"in_family_residual_db":row["R8_nmse_db"]-row["R5_nmse_db"],
                "projection_recovery_db":row["R4_nmse_db"]-row["R8_nmse_db"],**{f"B0_minus_{m}_db":row["B0_nmse_db"]-row[m+"_nmse_db"] for m in ("R4","R8","R5")}})
            row["closure"]=row["projection_recovery_db"]/row["stale_gap_db"] if row["stale_gap_db"]!=0 else None
            result.append(row)
    return result


def paired_diagnostics(rows):
    records=[]
    for p in PROFILES:
        for s in SNRS:
            group=[r for r in rows if r["profile"]==p and r["snr_db"]==s]
            for a,b in combinations(("R4","R8","R5","B0"),2):
                errors=np.array([10*np.log10(r[a+"_error_energy"]/r[b+"_error_energy"]) for r in group])
                ber=np.array([(r[a+"_bit_errors"]-r[b+"_bit_errors"])/r["total_bits"] for r in group])
                q1=np.quantile(errors,[.25,.5,.75]); q2=np.quantile(ber,[.25,.5,.75])
                records.append({"profile":p,"snr_db":s,"comparison":a+" minus "+b,"trajectory_units":len(group),"frames_accumulated_first":5,
                    **{f"NMSE_db_{k}":float(v) for k,v in zip(("Q25","median","Q75"),q1)},
                    **{f"BER_{k}":float(v) for k,v in zip(("Q25","median","Q75"),q2)},"p_value":None})
    return records


def scientific_flags(rows):
    key=[r for r in rows if r["snr_db"]==20]
    if len(key)!=2 or {r["profile"] for r in key}!=set(PROFILES):
        raise ValueError("B/C 20dB gates required")
    adequate=all(r["in_family_residual_db"]<=.5 for r in key)
    material=any(r["in_family_residual_db"]>=1.5 for r in key)
    return {"tdla_parametric_family_adequate":adequate,"out_of_family_shape_mismatch_material":material,
        "partial_profile_model_mismatch":not adequate and not material,"gate_SNR_dB":20,"adequacy_threshold_db":.5,"material_threshold_db":1.5}
