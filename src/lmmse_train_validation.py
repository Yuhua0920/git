"""T10D held-out full-link integration; frozen covariance and trajectory units."""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from src import lmmse
from src.link import (PILOT_SYMBOLS, add_received_noise, apply_frequency_mimo_channel,
    assemble_pilot_data_frame, demodulate_data_bits, estimate_ls_linear_channel,
    extract_reference_data_bits, generate_master_qpsk)
from src.lmmse_train import (formal_trajectory, load_frozen_covariance, read_manifest,
    prepare_train_lmmse, raw_array_sha256, sha256_file)
from src.metrics import channel_error_energies, count_bit_errors
from src.numerics import complex_gaussian_noise, noise_variance_from_snr
from src.pilots import build_fdm_pilot_layout, count_layout_resources
from src.stochastic_scan import correct_bit_rate_proxy, frame_duration_seconds
from src.tdl_awgn_scan import scale_unit_noise
from src.tdl_noiseless_scan import NoiselessLinkFailure, complex_record, condition_numbers, failure_context

DF_OPTIONS = (2,4,8,16)
SNR_OPTIONS = (0,5,10,15,20)
METHODS = ("perfect","ls","train","oracle")
ESTIMATED_METHODS = ("ls","train","oracle")
BITS_SEED_BASE = 80261001
NOISE_SEED_BASE = 90261001
THEORY_TOLERANCE_DB = 0.30
T10C_COMMIT = "d51dfb4df7c40298e5edb164a46168fe07111dc6"


def bits_seed(tid,frame,tx):
    _validation_coordinate(tid,frame)
    if tx not in (0,1):
        raise ValueError("tx must be 0 or 1")
    return BITS_SEED_BASE+100*tid+2*frame+tx


def noise_seed(tid,frame):
    _validation_coordinate(tid,frame)
    return NOISE_SEED_BASE+100*tid+frame


def _validation_coordinate(tid,frame):
    if (isinstance(tid,bool) or not isinstance(tid,int) or not 2000<=tid<2400
            or isinstance(frame,bool) or not isinstance(frame,int) or not 0<=frame<10):
        raise ValueError("validation trajectory 2000..2399 and frame 0..9 required")


def frozen_hashes(outputs, frozen):
    return {"manifest_sha256":sha256_file(outputs/"t10c_train_validation_manifest.csv"),
        "covariance_file_sha256":sha256_file(outputs/"t10c_lmmse_train_covariance.npz"),
        "R_train_raw_sha256":raw_array_sha256(frozen.R_train)}


def load_frozen_inputs(outputs):
    outputs = Path(outputs)
    reference = json.loads((outputs/"t10c_lmmse_train_results.json").read_text(encoding="utf-8"))
    if not reference["all_checks_passed"]:
        raise ValueError("accepted T10C calibration is not passed")
    manifest = read_manifest(outputs/"t10c_train_validation_manifest.csv")
    frozen = load_frozen_covariance(outputs/"t10c_lmmse_train_covariance.npz",reference["manifest"]["sha256"])
    expected = {"manifest_sha256":reference["manifest"]["sha256"],
        "covariance_file_sha256":reference["covariance_sha256"],"R_train_raw_sha256":reference["R_train_raw_array_sha256"]}
    before = frozen_hashes(outputs,frozen)
    if before != expected:
        raise ValueError("frozen T10C file/raw/manifest hashes changed")
    theory_path = outputs/"t10c_lmmse_train_theory.csv"
    if sha256_file(theory_path) != reference["provenance"]["theory_csv_sha256"]:
        raise ValueError("T10C mismatch-theory table changed")
    with theory_path.open(encoding="utf-8",newline="") as handle:
        rows = list(csv.DictReader(handle))
    theory = {(int(r["dF"]),int(r["snr_db"])):r for r in rows}
    if len(rows)!=20 or set(theory)!={(df,snr) for df in DF_OPTIONS for snr in SNR_OPTIONS}:
        raise ValueError("exactly 20 frozen theory conditions required")
    validation = [r for r in manifest if r["split"]=="validation_id"]
    return frozen,validation,theory,before


def prepare_pools(frozen, layouts, theory):
    """Count actual Cholesky calls separately, before channel/frame processing."""
    started = perf_counter()
    true_r = lmmse.build_full_grid_covariance(lmmse.build_frequency_covariance(),lmmse.build_time_covariance())
    train,oracle,diagnostics = {},{},[]
    with patch("src.lmmse.cho_factor",wraps=lmmse.cho_factor) as train_factor:
        for df,layout in layouts.items():
            for snr in SNR_OPTIONS:
                for q in range(2):
                    obj = prepare_train_lmmse(frozen,layout.pilot_mask[q],noise_variance_from_snr(snr))
                    train[df,snr,q] = obj
                    diagnostics.append({"method":"train","dF":df,"snr_db":snr,"tx":q,**obj.diagnostics})
        train_count = train_factor.call_count
    with patch("src.lmmse.cho_factor",wraps=lmmse.cho_factor) as oracle_factor:
        for df,layout in layouts.items():
            for snr in SNR_OPTIONS:
                total,count = 0.0,0
                for q in range(2):
                    obj = lmmse.prepare_oracle_lmmse(true_r,lmmse.pilot_flat_indices(layout.pilot_mask[q]),
                        noise_variance_from_snr(snr),(64,16))
                    oracle[df,snr,q] = obj
                    total += float(obj.posterior_variance()[layout.data_mask[q]].sum())
                    count += int(layout.data_mask[q].sum())
                    diagnostics.append({"method":"oracle","dF":df,"snr_db":snr,"tx":q,**obj.diagnostics})
                if not np.isclose(total/count,float(theory[df,snr]["oracle_nmse_data_linear"]),rtol=1e-12,atol=1e-15):
                    raise ValueError("matched Oracle T10A/T10C posterior regression changed")
        oracle_count = oracle_factor.call_count
    counters = {"prepared_train_estimators":len(train),"prepared_oracle_estimators":len(oracle),
        "train_Cholesky_calls":train_count,"oracle_Cholesky_calls":oracle_count,
        "prepared_outside_trajectory_loop":True,"diagnostics":diagnostics,"seconds":perf_counter()-started}
    if (len(train)!=40 or len(oracle)!=40 or train_count!=40 or oracle_count!=40
            or not all(d["cholesky_success"] and not d["jitter_used"] for d in diagnostics)):
        raise ValueError("expected exactly 40 stable Train and 40 stable Oracle preparations")
    return train,oracle,counters


def batched_data_zf(y,h,mask):
    """Batch the existing square 2x2 solve over data REs; no inverse/fallback.

    Existing src/detectors.py remains frozen. Tests compare this adapter against
    its individual-RE solves, including constellation decisions and failures.
    """
    if (h.shape!=(2,2,64,16) or y.shape!=(2,64,16) or mask.shape!=(2,64,16)
            or mask.dtype!=np.bool_ or not np.array_equal(mask[0],mask[1]) or not np.any(mask)
            or not np.all(np.isfinite(h)) or not np.all(np.isfinite(y))):
        raise ValueError("finite square full-grid Y/H and common data mask required")
    selected = mask[0]
    matrices = h[:,:,selected].transpose(2,0,1)
    vectors = y[:,selected].T[...,None]
    try:
        solved = np.linalg.solve(matrices,vectors)[...,0].T
    except np.linalg.LinAlgError as exc:
        bad = np.linalg.det(matrices)==0
        index = int(np.flatnonzero(bad)[0]) if np.any(bad) else 0
        k,t = np.argwhere(selected)[index]
        raise ValueError(f"batched ZF singular channel at physical RE k={k}, t={t}") from exc
    if not np.all(np.isfinite(solved)):
        raise ValueError("batched ZF produced nonfinite symbols")
    output = np.full((2,64,16),np.nan+1j*np.nan,dtype=np.complex128)
    output[:,selected] = solved
    return output


def estimated_receivers(y,layout,train_pair,oracle_pair,timing):
    """Estimated receiver boundary: no current truth, bits or data input."""
    stamp = perf_counter(); pilot_ls,lin = estimate_ls_linear_channel(y,layout,PILOT_SYMBOLS)
    timing["LS_seconds"] += perf_counter()-stamp
    stamp = perf_counter(); train = lmmse.estimate_mimo_from_pilot_ls(pilot_ls,train_pair)
    timing["Train_seconds"] += perf_counter()-stamp
    stamp = perf_counter(); oracle = lmmse.estimate_mimo_from_pilot_ls(pilot_ls,oracle_pair)
    timing["Oracle_seconds"] += perf_counter()-stamp
    return lin,train,oracle,pilot_ls


def manual_diagnostic(h,channels,x,y,detected,tid,frame,df,snr):
    records = []
    for label,k,t in (("interior_data",1,2),("trailing_data",1,15)):
        records.append({"trajectory_id":tid,"frame_index":frame,"dF":df,"snr_db":snr,
            "kind":label,"k":k,"t":t,"H_true":complex_record(h[:,:,k,t]),
            **{f"H_{m}":complex_record(channels[m][:,:,k,t]) for m in ESTIMATED_METHODS},
            "X_true":complex_record(x[:,k,t]),"Y_noisy":complex_record(y[:,k,t]),
            **{f"X_hat_{m}":complex_record(detected[m][:,k,t]) for m in METHODS},
            "absolute_channel_errors":{m:np.abs(channels[m][:,:,k,t]-h[:,:,k,t]).tolist() for m in ESTIMATED_METHODS}})
    return {"fixed_case":records,"relative_Train_Oracle_estimate_Frobenius_difference":
        float(np.linalg.norm(channels["train"]-channels["oracle"])/np.linalg.norm(channels["oracle"]))}


def evaluate_frame(h,bits,symbols,noise,layout,train_pair,oracle_pair,tid,frame,snr,timing):
    x = assemble_pilot_data_frame(symbols,layout)
    y = add_received_noise(apply_frequency_mimo_channel(h,x),noise)
    lin,train,oracle,pilot_ls = estimated_receivers(y,layout,train_pair,oracle_pair,timing)
    channels = {"perfect":h,"ls":lin,"train":train,"oracle":oracle}
    detected,y_ids = {},[]
    stamp = perf_counter()
    for method,channel in channels.items():
        try:
            y_ids.append(id(y))
            detected[method] = batched_data_zf(y,channel,layout.data_mask)
        except (ValueError,np.linalg.LinAlgError) as exc:
            import re
            match = re.search(r"k=(\d+), t=(\d+)",str(exc))
            k,t = (int(match[1]),int(match[2])) if match else tuple(np.argwhere(layout.data_mask[0])[0])
            record = failure_context(str(exc),tid,h,k,t,dF=int(64/layout.pilot_mask[0,:,0].sum()),x=x,h_used=channel,path=method)
            record.update({"channel_seed":60261001+tid,"frame_index":frame,"snr_db":snr,
                "noise_seed":noise_seed(tid,frame),"bits_seeds":[bits_seed(tid,frame,q) for q in range(2)]})
            raise NoiselessLinkFailure(record) from exc
    timing["ZF_seconds"] += perf_counter()-stamp
    reference = extract_reference_data_bits(bits,layout.data_mask)
    counts = {m:count_bit_errors(demodulate_data_bits(np.sqrt(2)*v,layout.data_mask),reference) for m,v in detected.items()}
    nbits = counts["perfect"][1]
    if nbits != 2*np.count_nonzero(layout.data_mask) or any(v[1]!=nbits for v in counts.values()):
        raise ValueError("four data-bit denominators differ from actual mask")
    energies = {m:channel_error_energies(channels[m],h,layout.data_mask) for m in ESTIMATED_METHODS}
    truth = energies["ls"][1]
    if any(v[1]!=truth for v in energies.values()):
        raise ValueError("three channel-energy denominators differ")
    row = {"data_bits":nbits,"nmse_truth_energy":truth,
        **{f"{m}_nmse_error_energy":v[0] for m,v in energies.items()},
        **{f"{m}_bit_errors":v[0] for m,v in counts.items()},"same_y_for_all_methods":len(set(y_ids))==1}
    diagnostic = manual_diagnostic(h,channels,x,y,detected,tid,frame,4,snr) if (tid,frame,snr)==(2000,0,10) and layout.offsets_by_tx==(0,2) else None
    return row,diagnostic


@dataclass
class TrajectoryAccumulator:
    manifest_row: dict
    df: int
    snr: int
    pilot_ratio: float
    frames: list = field(default_factory=list)
    totals: dict = field(default_factory=lambda:{k:0 for k in (
        "total_data_bits","ls_nmse_error_energy","train_nmse_error_energy","oracle_nmse_error_energy",
        "nmse_truth_energy","perfect_bit_errors","ls_bit_errors","train_bit_errors","oracle_bit_errors")})
    condition_max: float = 0.0
    same_y: bool = True

    def add(self,frame_index,row,condition_max):
        if frame_index!=len(self.frames) or frame_index>=10:
            raise ValueError("exactly frames 0..9 must be accumulated once in order")
        self.frames.append(frame_index)
        for name in self.totals:
            self.totals[name] += row["data_bits" if name=="total_data_bits" else name]
        self.condition_max = max(self.condition_max,condition_max)
        self.same_y = self.same_y and row["same_y_for_all_methods"]

    def finalize(self):
        if self.frames!=list(range(10)):
            raise ValueError("trajectory row cannot be written before all ten frames")
        return {"trajectory_id":self.manifest_row["trajectory_id"],"channel_seed":self.manifest_row["channel_seed"],
            "snr_db":self.snr,"noise_variance":noise_variance_from_snr(self.snr),"dF":self.df,
            "pilot_ratio":self.pilot_ratio,"frames_aggregated":10,**self.totals,
            "condition_number_max_over_10_frames":self.condition_max,"same_y_for_all_methods":self.same_y}


def aggregate_results(rows,theory):
    result = []
    for df in DF_OPTIONS:
        for snr in SNR_OPTIONS:
            group = [r for r in rows if (r["dF"],r["snr_db"])==(df,snr)]
            if not group or any(r["frames_aggregated"]!=10 for r in group):
                raise ValueError("every aggregate condition requires ten-frame trajectory rows")
            nbits = sum(r["total_data_bits"] for r in group)
            truth = sum(r["nmse_truth_energy"] for r in group)
            row = {"dF":df,"pilot_ratio":group[0]["pilot_ratio"],"snr_db":snr,
                "noise_variance":noise_variance_from_snr(snr),"num_trajectories":len(group),
                "frames_per_trajectory":10,"num_correlated_frames":10*len(group),"total_data_bits":nbits,
                "nmse_truth_energy":truth}
            for method in ESTIMATED_METHODS:
                error = sum(r[f"{method}_nmse_error_energy"] for r in group)
                row.update({f"{method}_nmse_error_energy":error,f"{method}_nmse_linear":error/truth,
                    f"{method}_nmse_db":float(10*np.log10(error/truth))})
            row.update({"train_theory_nmse_db":float(theory[df,snr]["train_predicted_nmse_data_db"]),
                "oracle_theory_nmse_db":float(theory[df,snr]["oracle_nmse_data_db"])})
            for method in ("train","oracle"):
                row[f"{method}_empirical_minus_theory_db"] = row[f"{method}_nmse_db"]-row[f"{method}_theory_nmse_db"]
            row["ls_minus_train_nmse_db"] = row["ls_nmse_db"]-row["train_nmse_db"]
            row["train_minus_oracle_nmse_db"] = row["train_nmse_db"]-row["oracle_nmse_db"]
            row["data_bits_per_frame"] = nbits//(10*len(group))
            for method in METHODS:
                errors = sum(r[f"{method}_bit_errors"] for r in group)
                row[f"{method}_bit_errors"] = errors
                row[f"{method}_ber"] = errors/nbits
                row[f"{method}_proxy_bps"] = correct_bit_rate_proxy(row["data_bits_per_frame"],errors/nbits,frame_duration_seconds())[1]
            row["all_checks_passed"] = all(abs(row[f"{m}_empirical_minus_theory_db"])<=THEORY_TOLERANCE_DB for m in ("train","oracle"))
            result.append(row)
    return result


def verify_aggregation(aggregate,rows,ids):
    expected = {(tid,df,snr) for tid in ids for df in DF_OPTIONS for snr in SNR_OPTIONS}
    if len(rows)!=len(ids)*20 or len(aggregate)!=20 or {(r["trajectory_id"],r["dF"],r["snr_db"]) for r in rows}!=expected:
        return False
    for a in aggregate:
        group = [r for r in rows if (r["dF"],r["snr_db"])==(a["dF"],a["snr_db"])]
        bits = sum(r["total_data_bits"] for r in group)
        truth = sum(r["nmse_truth_energy"] for r in group)
        if (len(group)!=len(ids) or any(r["frames_aggregated"]!=10 or not r["same_y_for_all_methods"] for r in group)
                or a["total_data_bits"]!=bits or a["nmse_truth_energy"]!=truth
                or a["num_correlated_frames"]!=10*len(ids)):
            return False
        for m in ESTIMATED_METHODS:
            error = sum(r[f"{m}_nmse_error_energy"] for r in group)
            if a[f"{m}_nmse_error_energy"]!=error or a[f"{m}_nmse_linear"]!=error/truth:
                return False
        for m in METHODS:
            errors = sum(r[f"{m}_bit_errors"] for r in group)
            proxy = (bits-errors)/(len(ids)*10*frame_duration_seconds())
            if a[f"{m}_bit_errors"]!=errors or a[f"{m}_ber"]!=errors/bits or not np.isclose(a[f"{m}_proxy_bps"],proxy,rtol=1e-12):
                return False
    return True


def run_validation(rows,model,layouts,train,oracle,theory,progress=None,generator=formal_trajectory):
    if (not rows or any(r["split"]!="validation_id" or not 2000<=r["trajectory_id"]<2400
            or r["channel_seed"]!=60261001+r["trajectory_id"] for r in rows)
            or len({r["trajectory_id"] for r in rows})!=len(rows)):
        raise ValueError("performance evaluation requires unique validation manifest IDs only")
    started = perf_counter()
    timing = {k:0.0 for k in ("trajectory_generation_seconds","frame_processing_seconds","LS_seconds","Train_seconds","Oracle_seconds","ZF_seconds")}
    output,realizations,visited,manual = [],[],[],None
    counters = {"trajectory_generator_calls":0,"master_bits_calls":0,"unit_noise_calls":0,"frame_condition_evaluations":0}
    mutation_checks = 0
    for manifest in rows:
        tid = manifest["trajectory_id"]
        stamp = perf_counter(); trajectory = generator(model,manifest)
        timing["trajectory_generation_seconds"] += perf_counter()-stamp
        counters["trajectory_generator_calls"] += 1
        if trajectory.trajectory_id!=tid or trajectory.channel_seed!=manifest["channel_seed"] or trajectory.h_long.shape!=(2,2,64,160) or not np.shares_memory(trajectory.h_long,trajectory.h_frames):
            raise ValueError("validation generator must preserve manifest ID/seed and continuous frame views")
        visited.append(tid)
        accumulators = {(df,snr):TrajectoryAccumulator(manifest,df,snr,count_layout_resources(layouts[df])["union_ratio"])
            for df in DF_OPTIONS for snr in SNR_OPTIONS}
        frame_hashes = []
        for frame,h in enumerate(trajectory.h_frames):
            stamp = perf_counter()
            try:
                conditions = condition_numbers(h,tid)
            except NoiselessLinkFailure as exc:
                exc.record.update({"channel_seed":manifest["channel_seed"],"frame_index":frame})
                raise
            bits,symbols = generate_master_qpsk(bits_seeds=tuple(bits_seed(tid,frame,q) for q in range(2)))
            unit_noise = complex_gaussian_noise((2,64,16),1,np.random.default_rng(noise_seed(tid,frame)))
            counters["master_bits_calls"] += 1; counters["unit_noise_calls"] += 1
            frame_hashes.append({"frame_index":frame,"H_sha256":raw_array_sha256(h),
                "bits_sha256":raw_array_sha256(bits),"unit_noise_sha256":raw_array_sha256(unit_noise)})
            for snr in SNR_OPTIONS:
                noise = scale_unit_noise(unit_noise,noise_variance_from_snr(snr))
                for df,layout in layouts.items():
                    metrics,diagnostic = evaluate_frame(h,bits,symbols,noise,layout,
                        tuple(train[df,snr,q] for q in range(2)),tuple(oracle[df,snr,q] for q in range(2)),tid,frame,snr,timing)
                    accumulators[df,snr].add(frame,metrics,float(conditions.max()))
                    counters["frame_condition_evaluations"] += 1
                    if diagnostic is not None:
                        manual = diagnostic
            if (raw_array_sha256(h)!=frame_hashes[-1]["H_sha256"]
                    or raw_array_sha256(bits)!=frame_hashes[-1]["bits_sha256"]
                    or raw_array_sha256(unit_noise)!=frame_hashes[-1]["unit_noise_sha256"]):
                raise ValueError("shared H/bits/unit noise mutated during condition processing")
            mutation_checks += 1
            timing["frame_processing_seconds"] += perf_counter()-stamp
        output.extend(a.finalize() for a in accumulators.values())
        realizations.append({"trajectory_id":tid,"channel_seed":manifest["channel_seed"],
            "H_long_sha256":raw_array_sha256(trajectory.h_long),"frame_realizations":frame_hashes})
        if progress:
            progress(len(visited))
    aggregate = aggregate_results(output,theory)
    comparisons = {m:[{"dF":r["dF"],"snr_db":r["snr_db"],"empirical_nmse_db":r[f"{m}_nmse_db"],
        "theory_nmse_db":r[f"{m}_theory_nmse_db"],"delta_db":r[f"{m}_empirical_minus_theory_db"],
        "tolerance_db":THEORY_TOLERANCE_DB,"passed":abs(r[f"{m}_empirical_minus_theory_db"])<=THEORY_TOLERANCE_DB} for r in aggregate] for m in ("train","oracle")}
    checks = {"validation_only":all(2000<=t<2400 for t in visited),"continuous_ten_frames":True,
        "paired_H_bits_unit_noise":counters["master_bits_calls"]==counters["unit_noise_calls"]==10*len(rows),
        "shared_input_hashes_unchanged":mutation_checks==10*len(rows),
        "same_y_for_all_methods":all(r["same_y_for_all_methods"] for r in output),
        "trajectory_first_aggregation":verify_aggregation(aggregate,output,visited),
        "train_20_theory_comparisons":all(r["passed"] for r in comparisons["train"]),
        "oracle_20_theory_comparisons":all(r["passed"] for r in comparisons["oracle"])}
    timing["total_simulation_seconds"] = perf_counter()-started
    return {"aggregate_results":aggregate,"theory_comparisons":comparisons,"checks":checks,
        "visited_validation_trajectory_ids":visited,"realization_hashes":realizations,"manual_diagnostic":manual,
        "runtime":timing,"execution_counters":counters,"numerical_failures":[],"num_filtered_trajectories":0},output
