"""Incremental R2 evaluation on the exact frozen T10D validation realizations.

Old receiver performance is read from CSV; no old estimator is prepared or run.
"""

import csv
import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np

from src.estimators import pilot_ls_estimate
from src.interpolation import ls_nearest_interpolate
from src.link import (PILOT_SYMBOLS, add_received_noise, apply_frequency_mimo_channel,
    assemble_pilot_data_frame, demodulate_data_bits, extract_reference_data_bits,
    generate_master_qpsk)
from src.lmmse_train import formal_trajectory, raw_array_sha256, read_manifest, sha256_file
from src.lmmse_train_validation import batched_data_zf, bits_seed, noise_seed
from src.metrics import channel_error_energies, count_bit_errors
from src.numerics import complex_gaussian_noise, noise_variance_from_snr
from src.pilots import count_layout_resources, map_pilot_symbols
from src.stochastic_scan import correct_bit_rate_proxy, frame_duration_seconds
from src.tdl_awgn_scan import scale_unit_noise
from src.tdl_noiseless_scan import NoiselessLinkFailure, complex_record, condition_numbers

T11A_COMMIT = "9fe2f165f9f511c9fe1c90585f2ec5023ef3c951"
DF_OPTIONS = (2, 4, 8, 16)
SNR_OPTIONS = (0, 5, 10, 15, 20)
FROZEN_HASHES = {
    "t10d_lmmse_train_validation_scan.csv": "e815f93a9ec0cbd35e5f49a21804f7665a035f13f65c112308029b02a2e06c76",
    "t10d_lmmse_train_validation_trajectory_metrics.csv": "5d87c74f41a83860ca24d0bbdf04ebc912ab7d1669b0ef1e0b9261db0116bb60",
    "t10d_lmmse_train_validation_results.json": "1e19c8d0bb0f1bab1658f5cad98bc7bd29ca8704aae121d6981e014ce41bfca5",
    "t10c_train_validation_manifest.csv": "03c4fc237c374741df0d3aa43956461e4870885cf851e1f63640f00041c488f4",
    "t10c_lmmse_train_covariance.npz": "b41ec0128243d07efb7577785e798994e5d8bc00905ab40d174ff1ee0da8dc3e",
}
OLD_TO_MERGED = {
    **{f"r{rid}_{label}_nmse_{scale}": f"{old}_nmse_{scale}"
       for rid, label, old in ((3,"linear","ls"),(4,"train","train"),(5,"oracle","oracle"))
       for scale in ("linear","db")},
    **{f"r{rid}_{label}_ber": f"{old}_ber"
       for rid, label, old in ((0,"perfect","perfect"),(3,"linear","ls"),(4,"train","train"),(5,"oracle","oracle"))},
    **{f"r{rid}_proxy_bps": f"{old}_proxy_bps"
       for rid, old in ((0,"perfect"),(3,"ls"),(4,"train"),(5,"oracle"))},
}


def verify_frozen(outputs):
    actual = {name: sha256_file(Path(outputs)/name) for name in FROZEN_HASHES}
    if actual != FROZEN_HASHES:
        raise ValueError("frozen T10D/T10C reference changed")
    return actual


def verify_accepted_sources(root):
    """Version historical detector/guard files; other numerical cores stay current."""
    from src.provenance import recorded_source_matches
    root = Path(root)
    subject = subprocess.check_output(["git","show","-s","--format=%s",T11A_COMMIT],cwd=root,text=True).strip()
    if subject != "feat: add calibrated LS-nearest baseline":
        raise ValueError("accepted standalone T11A commit missing")
    t11a = json.loads((root/"outputs/t11a_nearest_interpolation_results.json").read_text(encoding="utf-8"))
    t10d = json.loads((root/"outputs/t10d_lmmse_train_validation_results.json").read_text(encoding="utf-8"))
    hashes = {**t10d["provenance"]["source_sha256"],
        **{p:h for p,h in t11a["provenance"]["source_sha256"].items()
           if p in ("src/interpolation.py","scripts/validate_nearest_interpolation.py","configs/base.yaml")}}
    checks = {p:recorded_source_matches(root,p,h,
        "t11a" if p in t11a["provenance"]["source_sha256"] else "t10d",current_hash=sha256_file)
        for p,h in hashes.items()}
    if not all(checks.values()):
        raise ValueError("accepted numerical source/config changed")
    return checks


def read_csv(path):
    with Path(path).open(encoding="utf-8",newline="") as handle:
        return list(csv.DictReader(handle))


def load_references(outputs):
    outputs = Path(outputs)
    before = verify_frozen(outputs)
    old = json.loads((outputs/"t10d_lmmse_train_validation_results.json").read_text(encoding="utf-8"))
    if not old["all_checks_passed"] or old["formal_test_used"] or not old["formal_test_sealed"]:
        raise ValueError("accepted T10D held-out validation required")
    manifest = [r for r in read_manifest(outputs/"t10c_train_validation_manifest.csv") if r["split"]=="validation_id"]
    if [r["trajectory_id"] for r in manifest] != list(range(2000,2400)):
        raise ValueError("exact validation IDs 2000..2399 required")
    records = old["realization_hashes"]
    if len(records)!=400 or [r["trajectory_id"] for r in records]!=list(range(2000,2400)):
        raise ValueError("400 frozen trajectory provenance records required")
    for r in records:
        if (set(r)!={"trajectory_id","channel_seed","H_long_sha256","frame_realizations"}
                or r["channel_seed"]!=60261001+r["trajectory_id"]
                or [f["frame_index"] for f in r["frame_realizations"]]!=list(range(10))
                or any(set(f)!={"frame_index","H_sha256","bits_sha256","unit_noise_sha256"} for f in r["frame_realizations"])):
            raise ValueError("all actual T10D provenance fields must be verified; unexpected schema")
    scan = read_csv(outputs/"t10d_lmmse_train_validation_scan.csv")
    trajectories = read_csv(outputs/"t10d_lmmse_train_validation_trajectory_metrics.csv")
    conditions = {(df,snr) for df in DF_OPTIONS for snr in SNR_OPTIONS}
    if len(scan)!=20 or {(int(r["dF"]),int(r["snr_db"])) for r in scan}!=conditions:
        raise ValueError("20 unique frozen aggregate conditions required")
    index = {(int(r["trajectory_id"]),int(r["dF"]),int(r["snr_db"])):r for r in trajectories}
    if len(trajectories)!=8000 or set(index)!={(tid,df,snr) for tid in range(2000,2400) for df,snr in conditions}:
        raise ValueError("8000 unique frozen trajectory rows required")
    return {"before":before,"manifest":manifest,"old":old,"scan":scan,"trajectory_index":index,
        "realizations":{r["trajectory_id"]:r for r in records}}


def verify_frame(expected, frame, h, bits, unit_noise):
    actual = {"frame_index":frame,"H_sha256":raw_array_sha256(h),
        "bits_sha256":raw_array_sha256(bits),"unit_noise_sha256":raw_array_sha256(unit_noise)}
    if actual != expected:
        mismatches = [k for k in actual if actual[k]!=expected.get(k)]
        raise ValueError(f"exact replay mismatch at frame {frame}: {mismatches}")
    return actual


def _complex(value):
    return np.array(value["real"])+1j*np.array(value["imag"])


def manual_diagnostic(h, h_nearest, x, y, detected, old):
    records = []
    for k,t in ((1,2),(1,15)):
        reference = next(r for r in old["manual_diagnostic"]["fixed_case"] if (r["k"],r["t"])==(k,t))
        if any(not np.array_equal(value,_complex(reference[name]))
               for name,value in (("H_true",h[:,:,k,t]),("X_true",x[:,k,t]),("Y_noisy",y[:,k,t]))):
            raise ValueError(f"T10D sentinel H/X/Y exact mismatch at k={k},t={t}")
        linear = _complex(reference["H_ls"])
        records.append({"trajectory_id":2000,"frame_index":0,"dF":4,"snr_db":10,"k":k,"t":t,
            "H_true":complex_record(h[:,:,k,t]),"H_nearest":complex_record(h_nearest[:,:,k,t]),
            "H_linear":complex_record(linear),"linear_source":"frozen T10D fixed_case.H_ls",
            "X_true":complex_record(x[:,k,t]),"Y_noisy":complex_record(y[:,k,t]),
            "X_hat_nearest":complex_record(detected[:,k,t]),
            "absolute_channel_errors":{"nearest":np.abs(h_nearest[:,:,k,t]-h[:,:,k,t]).tolist(),
                "linear":np.abs(linear-h[:,:,k,t]).tolist()},"saved_H_X_Y_exact_match":True})
    return records


def evaluate_frame(h,bits,symbols,noise,layout,known_pilots,tid,frame,snr,timing,old):
    x = assemble_pilot_data_frame(symbols,layout)
    y = add_received_noise(apply_frequency_mimo_channel(h,x),noise)
    stamp = perf_counter()
    h_ls,mask = pilot_ls_estimate(y,known_pilots,layout.pilot_mask)
    timing["Pilot_LS_seconds"] += perf_counter()-stamp
    stamp = perf_counter()
    estimated = ls_nearest_interpolate(h_ls,mask)
    timing["nearest_interpolation_seconds"] += perf_counter()-stamp
    stamp = perf_counter()
    try:
        detected = batched_data_zf(y,estimated,layout.data_mask)
    except (ValueError,np.linalg.LinAlgError) as exc:
        raise NoiselessLinkFailure({"reason":str(exc),"path":"R2 LS-nearest","trajectory_id":tid,
            "channel_seed":60261001+tid,"frame_index":frame,"snr_db":snr,
            "dF":int(64/layout.pilot_mask[0,:,0].sum()),"noise_seed":noise_seed(tid,frame),
            "bits_seeds":[bits_seed(tid,frame,q) for q in range(2)]}) from exc
    timing["ZF_seconds"] += perf_counter()-stamp
    stamp = perf_counter()
    errors,nbits = count_bit_errors(demodulate_data_bits(np.sqrt(2)*detected,layout.data_mask),
        extract_reference_data_bits(bits,layout.data_mask))
    error,truth = channel_error_energies(estimated,h,layout.data_mask)
    if nbits!=2*np.count_nonzero(layout.data_mask):
        raise ValueError("actual data-mask bit denominator mismatch")
    timing["aggregation_seconds"] += perf_counter()-stamp
    row = {"nearest_nmse_error_energy":error,"nmse_truth_energy":truth,
        "nearest_bit_errors":errors,"total_data_bits":nbits}
    manual = manual_diagnostic(h,estimated,x,y,detected,old) if (tid,frame,snr)==(2000,0,10) and layout.offsets_by_tx==(0,2) else None
    return row,manual


class TrajectoryAccumulator:
    def __init__(self,manifest,df,snr,ratio):
        self.row = {"trajectory_id":manifest["trajectory_id"],"channel_seed":manifest["channel_seed"],
            "snr_db":snr,"noise_variance":noise_variance_from_snr(snr),"dF":df,"pilot_ratio":ratio,
            "frames_aggregated":0,"total_data_bits":0,"nearest_nmse_error_energy":0.0,
            "nmse_truth_energy":0.0,"nearest_bit_errors":0,"condition_number_max_over_10_frames":0.0}

    def add(self,frame,metrics,condition_max):
        if frame!=self.row["frames_aggregated"] or not 0<=frame<10:
            raise ValueError("frames 0..9 must be accumulated exactly once in order")
        for key,value in metrics.items():
            self.row[key] += value
        self.row["frames_aggregated"] += 1
        self.row["condition_number_max_over_10_frames"] = max(self.row["condition_number_max_over_10_frames"],condition_max)

    def finalize(self,old):
        if self.row["frames_aggregated"]!=10:
            raise ValueError("ten continuous frames required before writing trajectory row")
        for key in ("channel_seed","frames_aggregated","total_data_bits","nmse_truth_energy","condition_number_max_over_10_frames","noise_variance","pilot_ratio"):
            if float(old[key])!=self.row[key]:
                raise ValueError(f"paired T10D trajectory denominator/context mismatch: {key}")
        return {**self.row,"replay_h_hash_match":True,"replay_bits_hash_match":True,"replay_noise_hash_match":True}


def aggregate_r2(rows):
    result = []
    ids = sorted({r["trajectory_id"] for r in rows})
    expected = {(tid,df,snr) for tid in ids for df in DF_OPTIONS for snr in SNR_OPTIONS}
    if not ids or len(rows)!=20*len(ids) or {(r["trajectory_id"],r["dF"],r["snr_db"]) for r in rows}!=expected:
        raise ValueError("one row per trajectory/condition required")
    for df in DF_OPTIONS:
        for snr in SNR_OPTIONS:
            group = [r for r in rows if (r["dF"],r["snr_db"])==(df,snr)]
            if any(r["frames_aggregated"]!=10 or not all(r[k] for k in ("replay_h_hash_match","replay_bits_hash_match","replay_noise_hash_match")) for r in group):
                raise ValueError("only verified complete ten-frame trajectories may be aggregated")
            error = sum(r["nearest_nmse_error_energy"] for r in group)
            truth = sum(r["nmse_truth_energy"] for r in group)
            bits = sum(r["total_data_bits"] for r in group)
            errors = sum(r["nearest_bit_errors"] for r in group)
            ratio,ber = error/truth,errors/bits
            result.append({"dF":df,"pilot_ratio":group[0]["pilot_ratio"],"snr_db":snr,
                "noise_variance":noise_variance_from_snr(snr),"num_trajectories":len(group),
                "frames_per_trajectory":10,"num_correlated_frames":10*len(group),"total_data_bits":bits,
                "data_bits_per_frame":bits//(10*len(group)),"nearest_nmse_error_energy":error,
                "nmse_truth_energy":truth,"nearest_nmse_linear":ratio,"nearest_nmse_db":float(10*np.log10(ratio)),
                "nearest_bit_errors":errors,"nearest_ber":ber,
                "nearest_proxy_bps":correct_bit_rate_proxy(bits//(10*len(group)),ber,frame_duration_seconds())[1],
                "all_checks_passed":True})
    return result


def merge_baselines(r2,refs):
    """Copy frozen CSV strings verbatim; require paired 400-trajectory denominators."""
    old_index = {(int(r["dF"]),int(r["snr_db"])):r for r in refs["scan"]}
    if len(r2)!=20 or {(a["dF"],a["snr_db"]) for a in r2}!=set(old_index):
        raise ValueError("exact twenty R2 conditions required for baseline merge")
    merged = []
    for a in r2:
        old = old_index[a["dF"],a["snr_db"]]
        metadata = ("dF","pilot_ratio","snr_db","noise_variance","num_trajectories","frames_per_trajectory",
            "num_correlated_frames","total_data_bits","data_bits_per_frame","nmse_truth_energy")
        if a["num_trajectories"]!=400 or any(float(old[k])!=a[k] for k in metadata):
            raise ValueError("merge requires exact paired full-validation metadata/denominators")
        row = {k:old[k] for k in metadata}
        row.update({"R1_pilot_mse_or_NA_reference":"T09B: t09b_tdl_awgn_results.json#pilot_ls_theoretical_mse_checks",
            "r1_data_ber":"N/A","r2_nearest_nmse_linear":a["nearest_nmse_linear"],
            "r2_nearest_nmse_db":a["nearest_nmse_db"],"r2_nearest_ber":a["nearest_ber"],
            "r2_proxy_bps":a["nearest_proxy_bps"],
            **{key:old[source] for key,source in OLD_TO_MERGED.items()},
            "r2_minus_r3_nmse_db":a["nearest_nmse_db"]-float(old["ls_nmse_db"]),
            "r3_minus_r4_nmse_db":float(old["ls_nmse_db"])-float(old["train_nmse_db"]),
            "r4_minus_r5_nmse_db":float(old["train_nmse_db"])-float(old["oracle_nmse_db"]),
            "all_checks_passed":True})
        merged.append(row)
    return merged


def run_replay(manifest,model,layouts,refs,progress=None,generator=formal_trajectory):
    if (not manifest or any(r["split"]!="validation_id" or not 2000<=r["trajectory_id"]<2400
            or r["channel_seed"]!=60261001+r["trajectory_id"] for r in manifest)
            or len({r["trajectory_id"] for r in manifest})!=len(manifest)):
        raise ValueError("unique validation-only manifest rows required")
    started = perf_counter()
    timing = {key:0.0 for key in ("trajectory_generation_seconds","Pilot_LS_seconds",
        "nearest_interpolation_seconds","ZF_seconds","aggregation_seconds","replay_verification_seconds")}
    output,realizations,manual = [],[],None
    counters = {"trajectory_generator_calls":0,"master_bits_calls":0,"unit_noise_calls":0,
        "R2_frame_condition_evaluations":0,"old_method_performance_evaluations":0,
        "LMMSE_preparations":0,"shared_frame_mutation_checks":0}
    known = {df:map_pilot_symbols(layout,PILOT_SYMBOLS) for df,layout in layouts.items()}
    for manifest_row in manifest:
        tid = manifest_row["trajectory_id"]
        stamp = perf_counter(); trajectory = generator(model,manifest_row)
        timing["trajectory_generation_seconds"] += perf_counter()-stamp
        counters["trajectory_generator_calls"] += 1
        expected = refs["realizations"][tid]
        if (trajectory.trajectory_id!=tid or trajectory.channel_seed!=manifest_row["channel_seed"]
                or trajectory.h_long.shape!=(2,2,64,160) or trajectory.h_frames.shape!=(10,2,2,64,16)
                or trajectory.h_long.dtype!=np.complex128 or not np.shares_memory(trajectory.h_frames,trajectory.h_long)):
            raise ValueError("continuous complex128 160-symbol trajectory and shared ten-frame views required")
        h_long_hash = raw_array_sha256(trajectory.h_long)
        if h_long_hash!=expected["H_long_sha256"]:
            raise ValueError(f"exact H_long replay mismatch at trajectory {tid}")
        acc = {(df,snr):TrajectoryAccumulator(manifest_row,df,snr,count_layout_resources(layouts[df])["union_ratio"])
            for df in DF_OPTIONS for snr in SNR_OPTIONS}
        frame_hashes = []
        for frame,h in enumerate(trajectory.h_frames):
            conditions = condition_numbers(h,tid)
            bits,symbols = generate_master_qpsk(bits_seeds=tuple(bits_seed(tid,frame,q) for q in range(2)))
            unit_noise = complex_gaussian_noise((2,64,16),1,np.random.default_rng(noise_seed(tid,frame)))
            counters["master_bits_calls"] += 1; counters["unit_noise_calls"] += 1
            stamp = perf_counter()
            frame_hashes.append(verify_frame(expected["frame_realizations"][frame],frame,h,bits,unit_noise))
            timing["replay_verification_seconds"] += perf_counter()-stamp
            for snr in SNR_OPTIONS:
                noise = scale_unit_noise(unit_noise,noise_variance_from_snr(snr))
                for df,layout in layouts.items():
                    metrics,diagnostic = evaluate_frame(h,bits,symbols,noise,layout,known[df],tid,frame,snr,timing,refs["old"])
                    stamp = perf_counter(); acc[df,snr].add(frame,metrics,float(conditions.max()))
                    timing["aggregation_seconds"] += perf_counter()-stamp
                    counters["R2_frame_condition_evaluations"] += 1
                    if diagnostic is not None:
                        manual = diagnostic
            verify_frame(frame_hashes[-1],frame,h,bits,unit_noise)
            counters["shared_frame_mutation_checks"] += 1
        stamp = perf_counter()
        output.extend(a.finalize(refs["trajectory_index"][tid,df,snr]) for (df,snr),a in acc.items())
        timing["aggregation_seconds"] += perf_counter()-stamp
        actual = {"trajectory_id":tid,"channel_seed":manifest_row["channel_seed"],
            "H_long_sha256":raw_array_sha256(trajectory.h_long),"frame_realizations":frame_hashes}
        if actual!=expected:
            raise ValueError("complete trajectory provenance/mutation mismatch")
        realizations.append(actual)
        if progress:
            progress(len(realizations))
    stamp = perf_counter(); aggregate = aggregate_r2(output)
    timing["aggregation_seconds"] += perf_counter()-stamp
    timing["total_simulation_seconds"] = perf_counter()-started
    frames = 10*len(manifest)
    return {"r2_aggregate_results":aggregate,"realization_hashes":realizations,"manual_diagnostic":manual,
        "visited_validation_trajectory_ids":[r["trajectory_id"] for r in manifest],
        "replay_verification":{"num_h_hash_matches":len(manifest),"num_frame_h_hash_matches":frames,
            "num_bits_hash_matches":frames,"num_noise_hash_matches":frames,
            "num_complete_frame_provenance_matches":frames,"all_stored_fields_checked":True,"exact_match_fraction":1.0,
            "all_checks_passed":True},"execution_counters":counters,"runtime":timing,
        "num_filtered_trajectories":0,"numerical_failures":[]},output
