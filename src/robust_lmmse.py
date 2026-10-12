"""Fixed support HR baseline and exact incremental validation replay (T13B)."""
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import quad

from src import link, lmmse, ood_mismatch as ood
from src.estimators import pilot_ls_estimate
from src.lmmse_train import formal_trajectory, raw_array_sha256
from src.lmmse_train_validation import batched_data_zf, bits_seed, noise_seed
from src.metrics import channel_error_energies, count_bit_errors
from src.nearest_validation import load_references, verify_frame
from src.numerics import complex_gaussian_noise, noise_variance_from_snr
from src.pilots import count_layout_resources, map_pilot_symbols
from src.stochastic_scan import correct_bit_rate_proxy, frame_duration_seconds
from src.tdl import TDL_A_NORMALIZED_DELAYS, centered_frequency_grid_hz, tdl_a_profile
from src.tdl_awgn_scan import scale_unit_noise

PREFIX = "t13b_robust_lmmse"
T13A_COMMIT = "e59090f1cd2e2fcc3a5cedaa8428a8a83c67125d"
FD_BOUND = 389.158111
TAU_BOUND = float(np.max(tdl_a_profile(300)[0]))
CONDITIONS = ("ID", "speed120", "DS300")
SNRS = (5, 10, 20)
LAYOUT = ood.LAYOUT
TOL = 1e-12


def envelope():
    return {"normalized_max_delay": float(np.max(TDL_A_NORMALIZED_DELAYS)),
        "nominal_DS_bound_ns": 300, "scaling": "normalized_tap_delay * DS_ns * 1e-9",
        "tau_max_seconds": TAU_BOUND, "tau_max_ns": TAU_BOUND*1e9,
        "fd_max_hz": FD_BOUND, "condition_label_used": False,
        "physical_speed120_fd_hz": 389.15811106451076,
        "bound_rounding_difference_hz": 389.15811106451076-FD_BOUND}


def frequency_covariance(frequencies_hz, tau_max=TAU_BOUND):
    f = np.asarray(frequencies_hz, dtype=float)
    if f.ndim != 1 or not np.all(np.isfinite(f)) or not np.isfinite(tau_max) or tau_max <= 0:
        raise ValueError("finite 1D frequencies and positive delay bound required")
    lag = f[:, None]-f[None, :]
    return np.asarray(np.exp(-1j*np.pi*lag*tau_max)*np.sinc(lag*tau_max), dtype=np.complex128)


def time_covariance(times_s, fd_max=FD_BOUND):
    t = np.asarray(times_s, dtype=float)
    if t.ndim != 1 or not np.all(np.isfinite(t)) or not np.isfinite(fd_max) or fd_max <= 0:
        raise ValueError("finite 1D times and positive Doppler bound required")
    return np.asarray(np.sinc(2*fd_max*(t[:, None]-t[None, :])), dtype=np.complex128)


def _quadrature(phase, low, high):
    # Independent adaptive integration of density; no sinc/closed-form used.
    real, re = quad(lambda u: np.cos(phase*u)/(high-low), low, high, epsabs=1e-13, epsrel=1e-13)
    imag, ie = quad(lambda u: np.sin(phase*u)/(high-low), low, high, epsabs=1e-13, epsrel=1e-13)
    return real+1j*imag, max(re, ie)


def validate_factor(matrix, coordinates, phase_scale, interval):
    lags = np.unique((coordinates[:, None]-coordinates[None, :]).reshape(-1))
    max_error, max_quad_error, records = 0., 0., []
    for lag in lags:
        i, j = np.argwhere(coordinates[:, None]-coordinates[None, :] == lag)[0]
        value, bound = _quadrature(phase_scale*lag, *interval)
        error = float(abs(value-matrix[i,j])); max_error = max(max_error,error)
        max_quad_error = max(max_quad_error,bound)
        records.append({"physical_lag": float(lag), "analytic_real": float(matrix[i,j].real),
            "analytic_imag": float(matrix[i,j].imag), "quadrature_real": float(value.real),
            "quadrature_imag": float(value.imag), "absolute_error": error})
    eigenvalues = np.linalg.eigvalsh(matrix)
    hermitian = float(np.max(np.abs(matrix-matrix.conj().T)))
    diagonal = float(np.max(np.abs(np.diag(matrix)-1)))
    passed = hermitian <= TOL and diagonal <= TOL and eigenvalues.min() >= -TOL and max_error <= TOL
    return {"shape": list(matrix.shape), "dtype": str(matrix.dtype), "Hermitian_max_error": hermitian,
        "diagonal_max_error": diagonal, "minimum_eigenvalue": float(eigenvalues.min()),
        "negative_roundoff_eigenvalues": int(np.count_nonzero(eigenvalues<0)),
        "eigenvalues_clipped": False, "quadrature_max_error": max_error,
        "quadrature_reported_error_max": max_quad_error, "distinct_lags_checked": len(lags),
        "lag_correlations": records, "tolerance": TOL, "all_checks_passed": bool(passed)}


def build_prior():
    f = centered_frequency_grid_hz(64, 30000)
    t = np.arange(16, dtype=float)*(80/1920000)
    rf, rt = frequency_covariance(f), time_covariance(t)
    rv = validate_factor(rf, f, -2*np.pi*TAU_BOUND, (0.,1.))
    tv = validate_factor(rt, t, 2*np.pi*FD_BOUND, (-1.,1.))
    full = lmmse.build_full_grid_covariance(rf,rt)
    sentinel = np.array([full[k*16+t1,k2*16+t2]-rf[k,k2]*rt[t1,t2]
        for k,t1,k2,t2 in ((0,0,63,15),(7,3,48,11),(63,0,0,15),(32,8,32,8))])
    order_error = float(np.max(np.abs(sentinel)))
    if not rv["all_checks_passed"] or not tv["all_checks_passed"] or order_error > TOL:
        raise ValueError("HR covariance calibration failed; performance prohibited")
    full.setflags(write=False)
    return full, {"rf_validation": rv, "rt_validation": tv,
        "covariance_hash": raw_array_sha256(full), "full_shape": list(full.shape),
        "Kronecker_order": "Rf kron Rt; C order k*16+t", "ordering_sentinel_max_error": order_error,
        "frequency_order": "fftshift ascending [-960000,930000] Hz", "symbol_step_s": 80/1920000,
        "full_PSD_proof": "Kronecker product of two calibrated PSD factors; tolerance 1e-12"}


def prepare_pool(full):
    # Condition-blind: no condition argument, statistics fitting or true-channel input.
    return {snr: tuple(lmmse.prepare_oracle_lmmse(full, lmmse.pilot_flat_indices(LAYOUT.pilot_mask[q]),
        noise_variance_from_snr(snr), (64,16)) for q in range(2)) for snr in SNRS}


def load_replay_references(outputs):
    outputs = Path(outputs)
    id_refs = load_references(outputs)
    old = json.loads((outputs/f"{ood.PREFIX}_results.json").read_text(encoding="utf8"))
    if not old["all_checks_passed"] or old["formal_test_used"] or not old["formal_test_sealed"]:
        raise ValueError("accepted sealed T13A required")
    index = {(r["shift"],r["trajectory_id"]):r for r in old["realization_hashes"]}
    if set(index) != {(s,i) for s in ood.SHIFTS for i in range(200)}:
        raise ValueError("exact 400 existing OOD trajectories required")
    rows = ood.read_csv(outputs/f"{ood.PREFIX}_trajectory_metrics.csv")
    old_index = {(r["shift"],int(r["trajectory_id"]),int(r["snr_db"])):r for r in rows}
    if len(old_index) != 1200:
        raise ValueError("1200 frozen OOD trajectory rows required")
    return {"ID": id_refs, "OOD": index, "OOD_trajectory_index": old_index, "t13a_report": old}


def evaluate_trajectory(condition, tid, model, pools, refs, historical, timing):
    """Only R6 inference/ZF; frozen old methods are never invoked here."""
    stamp = perf_counter()
    if condition == "ID":
        row = refs["ID"]["manifest"][tid-2000]
        if row["trajectory_id"] != tid or row["split"] != "validation_id":
            raise ValueError("ID validation manifest index mismatch")
        trajectory = formal_trajectory(model,row)
        h_long, frames = trajectory.h_long, trajectory.h_frames
        expected = refs["ID"]["realizations"][tid]; channel_seed = row["channel_seed"]
    elif condition in ood.SHIFTS:
        _, h_long, frames = ood.draw_channel(model,condition,tid)
        expected = refs["OOD"][condition,tid]; channel_seed = ood.seed(13001,condition,tid)
    else:
        raise ValueError("only sealed validation-ID and two frozen OOD namespaces allowed")
    timing["channel_replay_seconds"] += perf_counter()-stamp
    if raw_array_sha256(h_long) != expected["H_long_sha256"]:
        raise ValueError("frozen long-channel hash mismatch")
    nframes = 10 if condition == "ID" else 5
    if frames.shape != (nframes,2,2,64,16):
        raise ValueError("continuous frame count changed")
    sums = {s:{"nmse_truth_energy":0., "total_bits":0, "R6_HR_error_energy":0., "R6_HR_bit_errors":0} for s in SNRS}
    records = []
    pilots = map_pilot_symbols(LAYOUT,link.PILOT_SYMBOLS)
    for frame,h in enumerate(frames):
        bs = tuple(bits_seed(tid,frame,q) if condition == "ID" else ood.seed(13002,condition,tid,frame,q) for q in range(2))
        ns = noise_seed(tid,frame) if condition == "ID" else ood.seed(13003,condition,tid,frame)
        bits,symbols = link.generate_master_qpsk(bits_seeds=bs)
        z = complex_gaussian_noise((2,64,16),1.,np.random.default_rng(ns),dtype=np.complex128)
        exp = expected["frame_realizations"][frame]
        record = verify_frame({k:exp[k] for k in ("frame_index","H_sha256","bits_sha256","unit_noise_sha256")},frame,h,bits,z)
        record.update({"Y_by_snr":{},"shared_pilot_LS_by_snr":{}})
        reference_bits = link.extract_reference_data_bits(bits,LAYOUT.data_mask)
        x = link.assemble_pilot_data_frame(symbols,LAYOUT)
        old_x = historical["assemble_pilot_data_frame"](symbols,LAYOUT)
        if not np.array_equal(x,old_x):
            raise ValueError("accepted T10D X changed")
        for snr in SNRS:
            noise = scale_unit_noise(z,noise_variance_from_snr(snr))
            y = link.add_received_noise(link.apply_frequency_mimo_channel(h,x),noise)
            old_y = historical["add_received_noise"](historical["apply_frequency_mimo_channel"](h,old_x),noise)
            if not np.array_equal(y,old_y):
                raise ValueError("accepted T10D Y changed")
            y.setflags(write=False)
            yh = raw_array_sha256(y); record["Y_by_snr"][str(snr)] = yh
            if condition != "ID" and yh != exp["Y_by_snr"][str(snr)]:
                raise ValueError("frozen T13A Y hash mismatch")
            stamp = perf_counter(); observations,_ = pilot_ls_estimate(y,pilots,LAYOUT.pilot_mask)
            timing["Pilot_LS_seconds"] += perf_counter()-stamp
            ph = raw_array_sha256(observations); record["shared_pilot_LS_by_snr"][str(snr)] = ph
            if condition != "ID" and ph != exp["shared_pilot_LS_by_snr"][str(snr)]:
                raise ValueError("frozen T13A Pilot-LS hash mismatch")
            stamp = perf_counter(); estimate = lmmse.estimate_mimo_from_pilot_ls(observations,pools[snr])
            timing["R6_inference_seconds"] += perf_counter()-stamp
            error,truth = channel_error_energies(estimate,h,LAYOUT.data_mask)
            stamp = perf_counter(); detected = batched_data_zf(y,estimate,LAYOUT.data_mask)
            timing["ZF_seconds"] += perf_counter()-stamp
            errors,total = count_bit_errors(link.demodulate_data_bits(np.sqrt(2)*detected,LAYOUT.data_mask),reference_bits)
            if total != 3968 or raw_array_sha256(y) != yh:
                raise ValueError("data-mask denominator/input mutation")
            sums[snr]["nmse_truth_energy"] += truth; sums[snr]["R6_HR_error_energy"] += error
            sums[snr]["total_bits"] += total; sums[snr]["R6_HR_bit_errors"] += errors
        records.append(record)
    rows=[]
    for snr in SNRS:
        saved = refs["ID"]["trajectory_index"][tid,16,snr] if condition == "ID" else refs["OOD_trajectory_index"][condition,tid,snr]
        total_key = "total_data_bits" if condition == "ID" else "total_bits"
        if (sums[snr]["total_bits"] != int(saved[total_key]) or
                not np.isclose(sums[snr]["nmse_truth_energy"],float(saved["nmse_truth_energy"]),rtol=1e-14,atol=0)):
            raise ValueError("replayed denominator differs from frozen performance")
        rows.append({"condition":condition,"trajectory_id":tid,"split":"validation_id" if condition == "ID" else "ood_dev",
            "snr_db":snr,"dF":16,"pilot_ratio":count_layout_resources(LAYOUT)["union_ratio"],
            "channel_seed":channel_seed,"frames_aggregated":nframes,"exact_replay_verified":True,**sums[snr]})
    return rows,{"condition":condition,"trajectory_id":tid,"H_long_sha256":raw_array_sha256(h_long),
        "frame_realizations":records,"historical_link_X_Y_exact_checks":nframes*3}


def frozen_performance(outputs):
    outputs=Path(outputs)
    id_rows=ood.read_csv(outputs/"t10d_lmmse_train_validation_scan.csv")
    b0_rows=ood.read_csv(outputs/"t12d_b0_df16_dev60_validation.csv")
    ood_rows=ood.read_csv(outputs/f"{ood.PREFIX}.csv")
    result={}
    for snr in SNRS:
        a=next(r for r in id_rows if int(r["dF"])==16 and int(r["snr_db"])==snr)
        b=next(r for r in b0_rows if int(r["snr_db"])==snr)
        entry={f"{m}_{metric}":float(a[f"{old}_{metric}"]) for m,old in (("R3","ls"),("R4","train"),("R5","oracle")) for metric in ("nmse_db","ber")}
        entry.update({"B0_nmse_db":float(b["b0_best60_nmse_db"]),"B0_ber":float(b["b0_best60_ber"]),"Perfect_ber":float(a["perfect_ber"]),
            "old_truth_energy":float(a["nmse_truth_energy"]),"old_total_bits":int(a["total_data_bits"])})
        result["ID",snr]=entry
    for a in ood_rows:
        result[a["shift"],int(a["snr_db"])]= {**{f"{m}_{metric}":float(a[f"{old}_{metric}"])
            for m,old in (("R3","R3"),("R4","R4_frozen"),("R5","R5_ood_oracle"),("B0","B0")) for metric in ("nmse_db","ber")},
            "Perfect_ber":float(a["Perfect_ber"]),"old_truth_energy":float(a["nmse_truth_energy"]),"old_total_bits":int(a["total_bits"])}
    return result


def closure(r4,r6,r5):
    gap=r4-r5
    if not np.isfinite(gap) or gap<=0:
        raise ValueError("positive finite frozen mismatch gap required")
    recovery=r4-r6
    return {"R4_R5_gap_db":gap,"R6_recovery_db":recovery,"robust_closure":recovery/gap}


def aggregate(rows, references):
    result=[]
    for condition in CONDITIONS:
        ids=set(range(2000,2400)) if condition=="ID" else set(range(200))
        frames=10 if condition=="ID" else 5
        for snr in SNRS:
            group=[r for r in rows if r["condition"]==condition and r["snr_db"]==snr]
            if len(group)!=len(ids) or {r["trajectory_id"] for r in group}!=ids or any(r["frames_aggregated"]!=frames or not r["exact_replay_verified"] for r in group):
                raise ValueError("complete frozen replay pool required")
            truth=sum(r["nmse_truth_energy"] for r in group); bits=sum(r["total_bits"] for r in group)
            energy=sum(r["R6_HR_error_energy"] for r in group); errors=sum(r["R6_HR_bit_errors"] for r in group)
            old=references[condition,snr]
            if bits!=old["old_total_bits"] or not np.isclose(truth,old["old_truth_energy"],rtol=1e-14,atol=0):
                raise ValueError("aggregate denominator differs from frozen reference")
            a={"condition":condition,"snr_db":snr,"dF":16,"pilot_ratio":1/32,"num_trajectories":len(ids),
                "frames_aggregated":len(ids)*frames,"nmse_truth_energy":truth,"total_bits":bits,
                "R6_HR_error_energy":energy,"R6_HR_bit_errors":errors,"R6_HR_nmse_db":float(10*np.log10(energy/truth)),"R6_HR_ber":errors/bits,
                **{k:v for k,v in old.items() if not k.startswith("old_")},
                "R4_R5_gap_db":None,"R6_recovery_db":None,"robust_closure":None,"ID_penalty_vs_R4_db":None,"ID_penalty_vs_R5_db":None}
            if condition=="ID":
                a.update({"ID_penalty_vs_R4_db":a["R6_HR_nmse_db"]-a["R4_nmse_db"],"ID_penalty_vs_R5_db":a["R6_HR_nmse_db"]-a["R5_nmse_db"]})
            else:
                a.update(closure(a["R4_nmse_db"],a["R6_HR_nmse_db"],a["R5_nmse_db"]))
            for m in ("R3","R4","R6_HR","R5","B0","Perfect"):
                _,rate=correct_bit_rate_proxy(3968,a[f"{m}_ber"],frame_duration_seconds())
                a[f"{m}_correct_bit_rate_proxy_bps"]=rate
            result.append(a)
    return result


def decision_flags(rows):
    values=[r["robust_closure"] for r in rows if r["condition"] in ood.SHIFTS and r["snr_db"]==20]
    if len(values)!=2 or not np.all(np.isfinite(values)):
        raise ValueError("both 20-dB OOD closures required")
    material=all(v>=.50 for v in values); insufficient=all(v<.25 for v in values)
    return {"fixed_robust_baseline_materially_closes_gap":material,
        "fixed_robust_prior_insufficient":insufficient,"mixed_result":not material and not insufficient}


def feasibility():
    return {"mr_sdp_feasibility":{
        "reference":"Nisar et al. 2010 DOI 10.1109/TSP.2010.2043126, Sections IV/V and remarks1-6",
        "uncertainty_class":"stationary valid 2-D PSD correlation; bounded spectral mass; one-sided delay and two-sided Doppler support",
        "regular_per_TX_grid_applicable":True,"pilot_grid":[16,4],"pilot_observations_per_TX":16,
        "bounds":{"delay_s":TAU_BOUND,"doppler_hz":FD_BOUND,"mean_link_power":1.,"noise_variances":[noise_variance_from_snr(s) for s in SNRS]},
        "pilot_delay_alias_period_s":1/(16*30000),"delay_support_exceeds_pilot_alias_period":True,
        "extra_SDP_stack_needed":True,"cvxpy_installed":importlib.util.find_spec("cvxpy") is not None,
        "cvxopt_installed":importlib.util.find_spec("cvxopt") is not None,"dependencies_installed":False,
        "dimension_estimate":{"one_target_Schur_complex_order":17,"real_embedding_order":34,
            "conservative_dense_full_grid_PSD_order":1024,"conjugate_lag_real_parameters":127*31,
            "one_dense_complex128_matrix_MiB":16,"marginal_support_orders_approx":[63,15],
            "status":"project conservative estimate, not exact minimal SDP or measured solver runtime"},
        "offline_only_future_R7_possible":True,"numerical_solver_run":False,
        "remaining_work":["target/objective and edge neighborhoods","PSD support extension constraints","delay aliasing","SNR-specific LF solutions and solver validation"],
        "note_path":"docs/robust_lmmse_feasibility.md"},
        "ccf_selection_feasibility":{
        "reference":"Mei et al. 2021 DOI 10.1049/cmu2.12250, equations13-16, Algorithm1 and Section5",
        "criterion":"sampled residual MSE after leave-one-observation-out MMSE interpolation",
        "published_OFDM_example":"block pilots","current_grid":"4x4 per TX, 16 pilots per frame",
        "direct_1D_block_pilot_reproduction":False,"literature_itself_uses_leave_one_out":True,
        "possible_extension":"physical-coordinate 2-D leave-one-pilot-out candidate comparison; unvalidated",
        "selection_implemented":False,"extension_validated":False,
        "risks":["16-observation score variability","correlated residuals","candidate design","edge extrapolation","delay aliasing","pooling latency and time correlation"],
        "note_path":"docs/robust_lmmse_feasibility.md"}}
