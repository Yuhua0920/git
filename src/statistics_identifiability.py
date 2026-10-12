"""T13C evaluation: exact frozen replay, metrics and interpretation gates."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src import link,lmmse,ood_mismatch as ood,statistics_selection as receiver
from src.estimators import ls_linear_interpolate,pilot_ls_estimate
from src.lmmse_train import formal_trajectory,raw_array_sha256,prepare_train_lmmse
from src.lmmse_train_validation import batched_data_zf,bits_seed,noise_seed,load_frozen_inputs
from src.metrics import channel_error_energies,count_bit_errors
from src.nearest_validation import verify_frame
from src.numerics import complex_gaussian_noise,noise_variance_from_snr
from src.robust_lmmse import load_replay_references
from src.tdl_awgn_scan import scale_unit_noise

PREFIX="t13c_statistics_identifiability"
PERFORMANCE_PREFIX="t13c_bank_selection_lmmse"
T13B_COMMIT="a4e5e9b72191ca95d843cf0a10fb348679785a5b"
CONDITIONS=tuple(c[0] for c in receiver.CANDIDATES)
SNRS=receiver.SNRS
LAYOUT=receiver.LAYOUT
FRAMES=5
TIMING_KEYS=("replay_and_Pilot_LS_seconds","likelihood_selection_seconds","R7_LMMSE_seconds",
    "OracleBank_regression_seconds","R5_reference_seconds","ID_R3_R4_reference_seconds","ZF_seconds")


def validate_likelihood(bank,pools):
    """Independent direct-solve sentinel; never used for sample selection."""
    records=[]
    for window in receiver.WINDOWS:
        coordinates=receiver.pilot_coordinates(window,0)
        z=(np.arange(4*16*window).reshape(2,2,16*window)+1j*np.arange(4*16*window,0,-1).reshape(2,2,16*window))/(16*window)
        for snr in SNRS:
            scores=receiver.score_links(z,pools[window,snr]); reference=np.empty_like(scores)
            logdet_error=0.; ordering_error=0.
            for i,c in enumerate(bank):
                for q,model in enumerate(pools[window,snr][i]):
                    xy=receiver.pilot_coordinates(window,q); k,t=xy.T
                    direct=c["rf"][k[:,None],k[None,:]]*c["rt"][window][t[:,None],t[None,:]]
                    ordering_error=max(ordering_error,float(np.max(np.abs(direct-c["pilot_prior"][window,q]))))
                    sign,logdet=np.linalg.slogdet(model.observation_covariance)
                    if abs(sign-1)>1e-12:
                        raise ValueError("likelihood direct determinant is not positive")
                    logdet_error=max(logdet_error,abs(float(logdet)-model.logdet))
                    for rx in range(2):
                        vector=z[rx,q]
                        reference[i,rx,q]=logdet+np.vdot(vector,np.linalg.solve(model.observation_covariance,vector)).real
            error=float(np.max(np.abs(scores-reference)))
            if error>1e-9 or logdet_error>1e-10 or ordering_error>1e-12:
                raise ValueError("Cholesky/direct likelihood sentinel differs")
            records.append({"window_frames":window,"snr_db":snr,"pilot_count_per_link":len(coordinates),
                "score_direct_max_absolute_error":error,"logdet_direct_max_absolute_error":logdet_error,
                "ordering_max_absolute_error":ordering_error,"four_link_sum_max_absolute_error":float(np.max(np.abs(scores.sum((1,2))-reference.sum((1,2))))),
                "all_checks_passed":True})
    return records


def aliasing_context():
    return {"pilot_frequency_spacing_hz":16*30000,"delay_ambiguity_period_s":1/(16*30000),
        "HR_uncertainty_support_s":2.89758e-6,"HR_support_exceeds_delay_ambiguity_period":True,
        "interpretation":"the tested wide-envelope HR prior is insufficient",
        "scope":"does not establish insufficiency of all classical robust/MMSE approaches",
        "causality":"support exceeds ambiguity period is context, not a standalone causal proof"}


def references(outputs):
    outputs=Path(outputs); refs=load_replay_references(outputs)
    hr=json.loads((outputs/"t13b_robust_lmmse_results.json").read_text(encoding="utf8"))
    if not hr["all_checks_passed"] or hr["formal_test_used"] or not hr["formal_test_sealed"]:
        raise ValueError("accepted sealed T13B required")
    refs["ID_Y"]={r["trajectory_id"]:r for r in hr["realization_hashes"] if r["condition"]=="ID"}
    return refs


def prepare_R4(outputs):
    frozen,_,_,_=load_frozen_inputs(outputs)
    pools={s:tuple(prepare_train_lmmse(frozen,LAYOUT.pilot_mask[q],noise_variance_from_snr(s)) for q in range(2)) for s in SNRS}
    return pools,{"file_sha256":frozen.file_sha256,"raw_array_sha256":frozen.raw_array_sha256,
        "manifest_sha256":frozen.manifest_sha256,"covariance_reestimated":False}


def replay_trajectory(condition,tid,channel_model,refs,historical,timing):
    stamp=perf_counter()
    if condition=="ID":
        if not 2000<=tid<2200:
            raise ValueError("only first200 frozen ID validation trajectories allowed")
        trajectory=formal_trajectory(channel_model,refs["ID"]["manifest"][tid-2000])
        h_long,frames=trajectory.h_long,trajectory.h_frames[:5]
        saved=refs["ID"]["realizations"][tid]; saved_y=refs["ID_Y"][tid]
    elif condition in ood.SHIFTS and 0<=tid<200:
        _,h_long,frames=ood.draw_channel(channel_model,condition,tid)
        saved=refs["OOD"][condition,tid]; saved_y=saved
    else:
        raise ValueError("only frozen balanced ID/OOD namespaces allowed")
    if raw_array_sha256(h_long)!=saved["H_long_sha256"] or frames.shape!=(5,2,2,64,16):
        raise ValueError("original frozen continuous trajectory differs")
    observations={s:[] for s in SNRS}; details={s:[] for s in SNRS}; frame_records=[]
    for frame,h in enumerate(frames):
        bs=tuple(bits_seed(tid,frame,q) if condition=="ID" else ood.seed(13002,condition,tid,frame,q) for q in range(2))
        ns=noise_seed(tid,frame) if condition=="ID" else ood.seed(13003,condition,tid,frame)
        bits,symbols=link.generate_master_qpsk(bits_seeds=bs)
        unit=complex_gaussian_noise((2,64,16),1.,np.random.default_rng(ns),dtype=np.complex128)
        expected=saved["frame_realizations"][frame]
        rec=verify_frame({k:expected[k] for k in ("frame_index","H_sha256","bits_sha256","unit_noise_sha256")},frame,h,bits,unit)
        rec.update({"Y_by_snr":{},"shared_pilot_LS_by_snr":{}})
        x=link.assemble_pilot_data_frame(symbols,LAYOUT)
        old_x=historical["assemble_pilot_data_frame"](symbols,LAYOUT)
        if not np.array_equal(x,old_x):
            raise ValueError("accepted T10D X differs")
        for snr in SNRS:
            noise=scale_unit_noise(unit,noise_variance_from_snr(snr))
            y=link.add_received_noise(link.apply_frequency_mimo_channel(h,x),noise)
            old_y=historical["add_received_noise"](historical["apply_frequency_mimo_channel"](h,old_x),noise)
            if not np.array_equal(y,old_y):
                raise ValueError("accepted T10D Y differs")
            yh=raw_array_sha256(y)
            if yh!=saved_y["frame_realizations"][frame]["Y_by_snr"][str(snr)]:
                raise ValueError("frozen received Y hash differs")
            # Only these received pilot values cross the selector boundary.
            received_pilots=np.stack([y[:,LAYOUT.pilot_mask[q]] for q in range(2)],axis=1)
            pilots=receiver.pilot_ls_from_received_pilots(received_pilots)
            full_ls,_=pilot_ls_estimate(y,receiver.PILOT_VALUES,LAYOUT.pilot_mask)
            for q in range(2):
                if not np.array_equal(pilots[:,q],full_ls[:,q,LAYOUT.pilot_mask[q]]):
                    raise ValueError("packed Pilot-LS differs from accepted estimator")
            ph=raw_array_sha256(full_ls)
            if ph!=saved_y["frame_realizations"][frame]["shared_pilot_LS_by_snr"][str(snr)]:
                raise ValueError("frozen Pilot-LS hash differs")
            y.setflags(write=False); pilots.setflags(write=False)
            observations[snr].append(pilots); details[snr].append((h,bits,y,full_ls))
            rec["Y_by_snr"][str(snr)]=yh; rec["shared_pilot_LS_by_snr"][str(snr)]=ph
        frame_records.append(rec)
    timing["replay_and_Pilot_LS_seconds"]+=perf_counter()-stamp
    return {s:np.stack(z) for s,z in observations.items()},details,{"condition":condition,"trajectory_id":tid,
        "H_long_sha256":raw_array_sha256(h_long),"frame_realizations":frame_records,"historical_X_Y_checks":15}


def evaluate_trajectory(condition,tid,channel_model,likelihood,pools,r4,refs,historical,timing):
    pilots,details,realization=replay_trajectory(condition,tid,channel_model,refs,historical,timing)
    selections,performance=[],[]
    for snr in SNRS:
        chosen=None
        for window in receiver.WINDOWS:
            stamp=perf_counter(); result=receiver.select_candidate(pilots[snr][:window],likelihood[window,snr])
            timing["likelihood_selection_seconds"]+=perf_counter()-stamp
            selected=CONDITIONS[result["candidate_index"]]
            selections.append({"trajectory_id":tid,"true_condition":condition,"snr_db":snr,"window_frames":window,
                "score_id":float(result["scores"][0]),"score_speed120":float(result["scores"][1]),"score_ds300":float(result["scores"][2]),
                "selected_condition":selected,"correct":selected==condition,"margin":result["margin"],"exact_tie":result["tie"],
                "packed_pilots_sha256":raw_array_sha256(receiver.pack_joint_pilots(pilots[snr][:window]))})
            if window==5:
                chosen=result["candidate_index"]
        # The actual receiver estimate is fixed before true-class ceiling checks.
        stamp=perf_counter(); estimates=receiver.estimate_pilot_frames(pilots[snr],chosen,pools[snr])
        timing["R7_LMMSE_seconds"]+=perf_counter()-stamp
        true_index=CONDITIONS.index(condition)
        stamp=perf_counter(); ceiling=receiver.estimate_pilot_frames(pilots[snr],true_index,pools[snr])
        timing["OracleBank_regression_seconds"]+=perf_counter()-stamp
        methods=("R3","R4","R7_BankML","R7_OracleBank","R5")
        totals={"nmse_truth_energy":0.,"total_bits":0,**{f"{m}_error_energy":0. for m in methods},**{f"{m}_bit_errors":0 for m in methods}}
        max_oracle_difference=0.
        for frame,(h,bits,y,full_ls) in enumerate(details[snr]):
            stamp=perf_counter(); reference_r5=lmmse.estimate_mimo_from_pilot_ls(full_ls,pools[snr][true_index])
            timing["R5_reference_seconds"]+=perf_counter()-stamp
            difference=float(np.max(np.abs(reference_r5-ceiling[frame])))
            max_oracle_difference=max(max_oracle_difference,difference)
            if difference>1e-12:
                raise ValueError("OracleBank differs from the calibrated R5 interface")
            channels={"R7_BankML":estimates[frame],"R7_OracleBank":ceiling[frame],"R5":reference_r5}
            if condition=="ID":
                stamp=perf_counter(); channels["R3"]=ls_linear_interpolate(full_ls,LAYOUT.pilot_mask)
                channels["R4"]=lmmse.estimate_mimo_from_pilot_ls(full_ls,r4[snr])
                timing["ID_R3_R4_reference_seconds"]+=perf_counter()-stamp
            reference_bits=link.extract_reference_data_bits(bits,LAYOUT.data_mask)
            frame_truth=None
            for method,channel in channels.items():
                error,truth=channel_error_energies(channel,h,LAYOUT.data_mask)
                if frame_truth is not None and truth!=frame_truth:
                    raise ValueError("different receiver NMSE denominators")
                frame_truth=truth; totals[f"{method}_error_energy"]+=error
                stamp=perf_counter(); detected=batched_data_zf(y,channel,LAYOUT.data_mask)
                timing["ZF_seconds"]+=perf_counter()-stamp
                errors,n=count_bit_errors(link.demodulate_data_bits(np.sqrt(2)*detected,LAYOUT.data_mask),reference_bits)
                if n!=3968:
                    raise ValueError("pilot/data power or bit denominator changed")
                totals[f"{method}_bit_errors"]+=errors
            totals["nmse_truth_energy"]+=frame_truth; totals["total_bits"]+=n
        if totals["R7_OracleBank_bit_errors"]!=totals["R5_bit_errors"] or not np.isclose(totals["R7_OracleBank_error_energy"],totals["R5_error_energy"],rtol=1e-12,atol=1e-12):
            raise ValueError("OracleBank/R5 numerical performance regression failed")
        if condition!="ID":
            old=refs["OOD_trajectory_index"][condition,tid,snr]
            if (totals["total_bits"]!=int(old["total_bits"]) or not np.isclose(totals["nmse_truth_energy"],float(old["nmse_truth_energy"]),rtol=1e-14,atol=0)
                    or totals["R5_bit_errors"]!=int(old["R5_ood_oracle_bit_errors"])
                    or not np.isclose(totals["R5_error_energy"],float(old["R5_ood_oracle_error_energy"]),rtol=1e-12,atol=1e-12)):
                raise ValueError("frozen T13A R5 trajectory regression failed")
            for m,old_name in (("R3","R3"),("R4","R4_frozen")):
                totals[f"{m}_error_energy"]=float(old[f"{old_name}_error_energy"])
                totals[f"{m}_bit_errors"]=int(old[f"{old_name}_bit_errors"])
        performance.append({"condition":condition,"trajectory_id":tid,"split":"validation_id" if condition=="ID" else "ood_dev",
            "snr_db":snr,"frames_aggregated":5,"dF":16,"pilot_ratio":1/32,
            "selected_condition":CONDITIONS[chosen],"selection_correct":CONDITIONS[chosen]==condition,
            "OracleBank_R5_max_channel_difference":max_oracle_difference,**totals})
    return selections,performance,realization


def identification_flags(macro,recalls):
    ready=macro>=.90 and min(recalls)>=.85
    weak=macro<.70
    return {"pilot_statistics_readily_identifiable":bool(ready),"partial_identifiability":bool(not ready and not weak),
        "sparse_pilot_statistics_identifiability_weak":bool(weak)}


def confusion_results(rows):
    result=[]
    for window in receiver.WINDOWS:
        for snr in SNRS:
            group=[r for r in rows if r["window_frames"]==window and r["snr_db"]==snr]
            expected={(c,tid) for c in CONDITIONS for tid in (range(2000,2200) if c=="ID" else range(200))}
            if len(group)!=600 or {(r["true_condition"],r["trajectory_id"]) for r in group}!=expected:
                raise ValueError("600 balanced frozen trajectory-class samples required")
            matrix=np.zeros((3,3),dtype=int)
            for r in group:
                matrix[CONDITIONS.index(r["true_condition"]),CONDITIONS.index(r["selected_condition"])]+=1
                if r["correct"]!=(r["true_condition"]==r["selected_condition"]):
                    raise ValueError("selection correctness flag inconsistent")
            recall=np.diag(matrix)/matrix.sum(axis=1)
            record={"snr_db":snr,"window_frames":window,"condition_order":list(CONDITIONS),"confusion_matrix":matrix.tolist(),
                "sample_count":600,"overall_accuracy":float(np.trace(matrix)/matrix.sum()),"macro_accuracy":float(recall.mean()),
                "per_class_recall":{c:float(v) for c,v in zip(CONDITIONS,recall)}}
            if window==5:
                record.update(identification_flags(record["macro_accuracy"],recall))
            result.append(record)
    return result


def aggregate_performance(rows):
    result=[]
    for condition in CONDITIONS:
        expected=set(range(2000,2200)) if condition=="ID" else set(range(200))
        for snr in SNRS:
            group=[r for r in rows if r["condition"]==condition and r["snr_db"]==snr]
            if len(group)!=200 or {r["trajectory_id"] for r in group}!=expected or any(r["frames_aggregated"]!=5 for r in group):
                raise ValueError("complete200x5 balanced performance pool required")
            truth=sum(r["nmse_truth_energy"] for r in group); bits=sum(r["total_bits"] for r in group)
            if bits!=3968000 or truth<=0:
                raise ValueError("shared 200x5 data-bit denominator and positive truth energy required")
            a={"condition":condition,"snr_db":snr,"dF":16,"pilot_ratio":1/32,"num_trajectories":200,"frames_aggregated":1000,
                "nmse_truth_energy":truth,"total_bits":bits,"selection_accuracy":sum(r["selection_correct"] for r in group)/200,
                "R4_R5_gap_db":None,"R7_recovery_db":None,"gap_closure":None,"ID_penalty_vs_R4_db":None,"ID_penalty_vs_R5_db":None}
            for m in ("R3","R4","R7_BankML","R7_OracleBank","R5"):
                energy=sum(r[f"{m}_error_energy"] for r in group); errors=sum(r[f"{m}_bit_errors"] for r in group)
                a.update({f"{m}_error_energy":energy,f"{m}_nmse_db":float(10*np.log10(energy/truth)),f"{m}_bit_errors":errors,f"{m}_ber":errors/bits})
            if condition=="ID":
                a.update({"ID_penalty_vs_R4_db":a["R7_BankML_nmse_db"]-a["R4_nmse_db"],"ID_penalty_vs_R5_db":a["R7_BankML_nmse_db"]-a["R5_nmse_db"]})
            else:
                gap=a["R4_nmse_db"]-a["R5_nmse_db"]; recovery=a["R4_nmse_db"]-a["R7_BankML_nmse_db"]
                if gap<=0:
                    raise ValueError("positive frozen OOD mismatch gap required")
                a.update({"R4_R5_gap_db":gap,"R7_recovery_db":recovery,"gap_closure":recovery/gap})
            result.append(a)
    return result


def total_gate(performance,confusions,scope="20dB"):
    five=[r for r in confusions if r["window_frames"]==5]
    if scope=="20dB":
        metric=next(r for r in five if r["snr_db"]==20)["macro_accuracy"]
    elif scope=="pooled_SNR":
        metric=float(np.mean([r["macro_accuracy"] for r in five]))
    else:
        raise ValueError("explicit predeclared total gate scope required")
    closures=[r["gap_closure"] for r in performance if r["condition"]!="ID" and r["snr_db"]==20]
    if len(closures)!=2:
        raise ValueError("both OOD20 closures required")
    return {"decision_scope":scope,"five_frame_macro_accuracy_used":metric,
        "simple_covariance_selection_viable":bool(metric>=.90 and all(c>=.70 for c in closures)),
        "sparse_pilot_adaptation_bottleneck":bool(metric<.70 or any(c<.50 for c in closures))}
