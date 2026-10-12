"""T11C-B: six-condition paired detection on frozen held-out trajectories.

No new detector formula, covariance fit, NMSE computation or R2/R5 evaluation.
"""

import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from src import link
from src.lmmse import estimate_mimo_from_pilot_ls
from src.lmmse_train import (formal_trajectory, load_frozen_covariance,
                             prepare_train_lmmse, raw_array_sha256, sha256_file)
from src.lmmse_train_validation import batched_data_zf, bits_seed, noise_seed
from src.metrics import count_bit_errors
from src.nearest_validation import load_references, verify_frame
from src.numerics import complex_gaussian_noise, noise_variance_from_snr
from src.pilots import build_fdm_pilot_layout, count_layout_resources
from src.stochastic_scan import frame_duration_seconds
from src.tdl_awgn_scan import scale_unit_noise
from src.tdl_noiseless_scan import NoiselessLinkFailure

T11C_A_COMMIT = "58b7209beb1dfeb091596f3fe846c1978b5410cf"
CONDITIONS = tuple((df,snr) for df in (2,16) for snr in (0,10,20))
ESTIMATORS = ("R0", "R3", "R4")
OLD_METHOD = {"R0":"perfect", "R3":"ls", "R4":"train"}
BIN_EDGES = (1,2,4,8,16,float("inf"))
BIN_LABELS = ("[1,2)","[2,4)","[4,8)","[8,16)","[16,inf)")


def load_inputs(root):
    root = Path(root); outputs = root/"outputs"
    subject = subprocess.check_output(["git","show","-s","--format=%s",T11C_A_COMMIT],cwd=root,text=True).strip()
    if subject!="feat: add calibrated linear-MMSE detector sensitivity path":
        raise ValueError("accepted standalone T11C-A commit required")
    a = json.loads((outputs/"t11c_a_linear_mmse_detector_results.json").read_text(encoding="utf8"))
    if not a["all_checks_passed"] or not a["formal_test_sealed"] or a["main_detector_changed"]:
        raise ValueError("accepted T11C-A calibration required")
    expected_sources = {**a["provenance"]["current_source_sha256"],
                        **a["provenance"]["current_provenance_plumbing_sha256"]}
    if any(sha256_file(root/p)!=h for p,h in expected_sources.items()):
        raise ValueError("accepted T11C-A detector/core/provenance source changed")
    freeze = json.loads((outputs/"t11c_a_frozen_artifact_hashes.json").read_text(encoding="utf8"))["sha256"]
    if any(sha256_file(root/p)!=h for p,h in freeze.items()):
        raise ValueError("accepted historical output changed")
    before = {**freeze,**{f"outputs/{n}":sha256_file(outputs/n) for n in
        ("t11c_a_linear_mmse_detector_results.json","t11c_a_frozen_artifact_hashes.json")}}
    refs = load_references(outputs)
    b = json.loads((outputs/"t11b_nearest_validation_results.json").read_text(encoding="utf8"))
    if b["realization_hashes"]!=refs["old"]["realization_hashes"] or not b["all_checks_passed"]:
        raise ValueError("T10D/T11B frozen realization provenance differs")
    frozen = load_frozen_covariance(outputs/"t10c_lmmse_train_covariance.npz",
                                   sha256_file(outputs/"t10c_train_validation_manifest.csv"))
    refs.update({"root":root,"before_all":before,"accepted_sources":expected_sources,
                 "t11c_a":a,"frozen_covariance":frozen,"R_train_raw_before":raw_array_sha256(frozen.R_train)})
    return refs


def verify_unchanged(refs):
    after = {p:sha256_file(refs["root"]/p) for p in refs["before_all"]}
    if after!=refs["before_all"] or raw_array_sha256(refs["frozen_covariance"].R_train)!=refs["R_train_raw_before"]:
        raise ValueError("frozen result/manifest/covariance changed")
    return after


def prepare_train_pool(refs):
    layouts = {df:build_fdm_pilot_layout(2,64,16,df,4) for df in (2,16)}
    start = perf_counter()
    train = {(df,snr):tuple(prepare_train_lmmse(refs["frozen_covariance"],layouts[df].pilot_mask[q],
                 noise_variance_from_snr(snr)) for q in range(2)) for df,snr in CONDITIONS}
    return layouts,train,{"prepared_train_estimators":12,"prepared_oracle_estimators":0,
        "seconds":perf_counter()-start,"diagnostics":[{"dF":df,"snr_db":snr,"tx":q,**obj.diagnostics}
            for (df,snr),pair in train.items() for q,obj in enumerate(pair)]}


def historical_calibration_regression(refs):
    from scripts.validate_linear_mmse_detector import small_full_link_sentinel
    actual = small_full_link_sentinel()
    expected = refs["t11c_a"]["small_full_link_sentinel"]
    keys = ("trajectory_id","frame_index","dF","SNR_dB","channel_seed","bits_seeds","noise_seed")
    checks = {key:actual[key]==expected[key] for key in keys}
    records = []
    for new,old in zip(actual["records"],expected["records"],strict=True):
        passed = all(new[k]==old[k] for k in ("estimator","detector","Y_sha256","bits_sha256",
            "H_est_sha256","diagnostic_total_bit_errors","diagnostic_bit_errors_by_stream","data_bits"))
        records.append({"estimator":new["estimator"],"detector":new["detector"],
            "actual_bit_errors":new["diagnostic_total_bit_errors"],
            "historical_bit_errors":old["diagnostic_total_bit_errors"],"exact_match":passed})
    if not all(checks.values()) or not all(r["exact_match"] for r in records):
        raise ValueError("historical T11C-A calibration regression failed")
    return {"namespace":"historical calibration; separate from validation",**{k:actual[k] for k in keys},
        "checks":checks,"records":records,"historical_JSON_modified":False,"all_checks_passed":True}


def condition_bin_indices(values):
    values = np.asarray(values,dtype=float)
    if values.ndim!=1 or np.any(np.isnan(values)) or np.any(values<1):
        raise ValueError("condition numbers must be one-dimensional, >=1, and not NaN")
    return np.searchsorted(np.array(BIN_EDGES[1:-1]),values,side="right")


class ConditionBins:
    """All data REs, fixed bins, exact integer errors; retain values for medians."""

    def __init__(self):
        self.values = [[] for _ in BIN_LABELS]
        self.rows = [{"condition_bin":label,"num_RE":0,"num_bits":0,
                      "zf_bit_errors":0,"mmse_bit_errors":0} for label in BIN_LABELS]

    def add(self,condition,zf_errors,mmse_errors):
        ids = condition_bin_indices(condition)
        zf_errors,mmse_errors = np.asarray(zf_errors),np.asarray(mmse_errors)
        if (zf_errors.shape!=ids.shape or mmse_errors.shape!=ids.shape
                or any(v.dtype.kind not in "iu" or np.any(v<0) or np.any(v>4) for v in (zf_errors,mmse_errors))):
            raise ValueError("one integer error count in [0,4] required per two-stream RE")
        for i,row in enumerate(self.rows):
            selected = ids==i
            self.values[i].append(np.asarray(condition)[selected].copy())
            n = int(np.count_nonzero(selected))
            row["num_RE"]+=n; row["num_bits"]+=4*n
            row["zf_bit_errors"]+=int(zf_errors[selected].sum())
            row["mmse_bit_errors"]+=int(mmse_errors[selected].sum())

    def finalize(self):
        result = []
        for row,parts in zip(self.rows,self.values):
            values = np.concatenate(parts) if parts else np.empty(0)
            def finite_or_inf(v):
                return float(v) if np.isfinite(v) else "inf"
            result.append({**row,"zf_ber":row["zf_bit_errors"]/row["num_bits"] if row["num_bits"] else None,
                "mmse_ber":row["mmse_bit_errors"]/row["num_bits"] if row["num_bits"] else None,
                "mean_condition_number":finite_or_inf(np.mean(values)) if values.size else None,
                "median_condition_number":finite_or_inf(np.median(values)) if values.size else None})
        return result


def detect_pair(y,h_est,bits,mask,sigma2,ps,observer=None):
    reference = link.extract_reference_data_bits(bits,mask)
    inputs = (y,h_est,bits,mask)
    before = tuple(raw_array_sha256(v) for v in inputs)
    supplied = []; outputs = {}; timing = {}
    for name,detector in (("ZF",batched_data_zf),("Linear-MMSE",link.detect_data_linear_mmse)):
        supplied.append(tuple(id(v) for v in inputs))
        if observer: observer(name,*inputs)
        start = perf_counter()
        if name=="ZF":
            symbols = detector(y,h_est,mask); diagnostics = None
        else:
            symbols,diagnostics = detector(y,h_est,mask,sigma2,ps,return_diagnostics=True)
        timing[name] = perf_counter()-start
        bit_hat = link.demodulate_data_bits(np.sqrt(2)*symbols,mask)
        errors,nbits = count_bit_errors(bit_hat,reference)
        per_re = np.count_nonzero(np.stack(bit_hat).reshape(2,-1,2)!=
                                  np.stack(reference).reshape(2,-1,2),axis=(0,2))
        if int(per_re.sum())!=errors or nbits!=2*np.count_nonzero(mask):
            raise ValueError("per-RE errors and actual data-mask bit denominator mismatch")
        outputs[name] = {"errors":errors,"bits":nbits,"per_RE_errors":per_re,
            "symbols":symbols,"diagnostics":diagnostics}
        if tuple(raw_array_sha256(v) for v in inputs)!=before:
            raise ValueError("paired detector mutated Y/H_est/bits/mask")
    if supplied[0]!=supplied[1] or outputs["ZF"]["bits"]!=outputs["Linear-MMSE"]["bits"]:
        raise ValueError("paired receiver inputs differ")
    return outputs,timing


def validation_sentinel(refs,h,bits,symbols,unit_noise,y,channels,detected,layout):
    old_link = {}
    data = subprocess.check_output(["git","show","6129743ca9940ad41a5d9de2f70005c9160822c2:src/link.py"],cwd=refs["root"])
    exec(compile(data,"accepted_T10D_link","exec"),old_link)
    old_y = old_link["add_received_noise"](old_link["apply_frequency_mimo_channel"](
        h,old_link["assemble_pilot_data_frame"](symbols,layout)),scale_unit_noise(unit_noise,0.1))
    if not np.array_equal(old_y,y):
        raise ValueError("validation Y differs from accepted T10D link replay")
    records = []
    reference = link.extract_reference_data_bits(bits,layout.data_mask)
    for estimator,h_est in channels.items():
        # Independent per-RE D0 reference; main D0 is the accepted T10D batch adapter.
        reference_x = old_link["detect_data_zf"](y,h_est,layout.data_mask)
        old_errors,_ = count_bit_errors(link.demodulate_data_bits(np.sqrt(2)*reference_x,layout.data_mask),reference)
        actual = detected[estimator]
        if actual["ZF"]["errors"]!=old_errors:
            raise ValueError("validation sentinel D0 differs from accepted ZF reference")
        records.append({"estimator":estimator,"zf_bit_errors":actual["ZF"]["errors"],
            "accepted_ZF_reference_bit_errors":old_errors,"mmse_bit_errors":actual["Linear-MMSE"]["errors"],
            "H_est_sha256":raw_array_sha256(h_est),"ZF_integer_exact_match":True})
    return {"namespace":"validation_id","trajectory_id":2000,"frame_index":0,"dF":16,"snr_db":10,
        "channel_seed":60261001+2000,"bits_seeds":[bits_seed(2000,0,q) for q in range(2)],
        "unit_noise_seed":noise_seed(2000,0),"Y_sha256":raw_array_sha256(y),
        "H_sha256":raw_array_sha256(h),"bits_sha256":raw_array_sha256(bits),
        "unit_noise_sha256":raw_array_sha256(unit_noise),"Y_exact_accepted_link_replay":True,
        "original_archive_single_frame_dF16_errors_available":False,
        "ZF_reference":"accepted T10D link/individual ZF; ten-frame counts additionally match frozen T10D CSV",
        "D1_is_new_validation_result":True,"compared_to_calibration_trajectory0":False,
        "records":records,"all_checks_passed":True}


def aggregate_trajectories(rows):
    ids = sorted({r["trajectory_id"] for r in rows})
    expected = {(tid,df,snr,e) for tid in ids for df,snr in CONDITIONS for e in ESTIMATORS}
    if not ids or len(rows)!=len(expected) or {(r["trajectory_id"],r["dF"],r["snr_db"],r["estimator"]) for r in rows}!=expected:
        raise ValueError("exact six conditions and three estimators required for every trajectory")
    for r in rows:
        if (not 2000<=r["trajectory_id"]<2400 or r["total_bits"]<=0
                or any(not isinstance(r[k],(int,np.integer)) or not 0<=r[k]<=r["total_bits"] for k in ("zf_bit_errors","mmse_bit_errors"))
                or r["zf_ber"]!=r["zf_bit_errors"]/r["total_bits"] or r["mmse_ber"]!=r["mmse_bit_errors"]/r["total_bits"]):
            raise ValueError("valid integer errors and exact trajectory BER ratios required")
    aggregate,paired = [],[]
    for df,snr in CONDITIONS:
        for estimator in ESTIMATORS:
            group = [r for r in rows if (r["dF"],r["snr_db"],r["estimator"])==(df,snr,estimator)]
            if any(r["frames_aggregated"]!=10 or not all(r[k] for k in ("same_H_est","same_Y","same_bits","same_data_mask")) for r in group):
                raise ValueError("complete paired ten-frame trajectory rows required")
            bits = sum(r["total_bits"] for r in group)
            zf = sum(r["zf_bit_errors"] for r in group); mmse = sum(r["mmse_bit_errors"] for r in group)
            zber,mber = zf/bits,mmse/bits
            reduction = zber-mber
            duration = len(ids)*10*frame_duration_seconds()
            aggregate.append({"dF":df,"pilot_ratio":group[0]["pilot_ratio"],"snr_db":snr,"estimator":estimator,
                "num_trajectories":len(ids),"frames_per_trajectory":10,"total_bits":bits,
                "zf_bit_errors":zf,"zf_ber":zber,"mmse_bit_errors":mmse,"mmse_ber":mber,
                "absolute_ber_reduction":reduction,"relative_ber_reduction":reduction/zber if zber else None,
                "zf_proxy_bps":(bits-zf)/duration,"mmse_proxy_bps":(bits-mmse)/duration,"all_checks_passed":True})
            delta = np.array([r["zf_ber"]-r["mmse_ber"] for r in group])
            paired.append({"dF":df,"snr_db":snr,"estimator":estimator,"num_independent_trajectories":len(ids),
                "delta_BER_definition":"trajectory BER_ZF - BER_Linear_MMSE",
                "mean":float(np.mean(delta)),"median":float(np.median(delta)),
                "percentile_25":float(np.quantile(delta,0.25,method="linear")),
                "percentile_75":float(np.quantile(delta,0.75,method="linear")),
                "fraction_MMSE_errors_less":sum(r["mmse_bit_errors"]<r["zf_bit_errors"] for r in group)/len(ids),
                "fraction_equal":sum(r["mmse_bit_errors"]==r["zf_bit_errors"] for r in group)/len(ids),
                "p_value_computed":False,"bootstrap_CI_computed":False})
    return aggregate,paired


def run_scan(refs,manifest,model,layouts,train,ps=0.5,progress=None,observer=None):
    if (not manifest or set(layouts)!={2,16} or set(train)!=set(CONDITIONS) or ps!=0.5
            or any(r["split"]!="validation_id" or not 2000<=r["trajectory_id"]<2400
                   or r["channel_seed"]!=60261001+r["trajectory_id"] for r in manifest)
            or len({r["trajectory_id"] for r in manifest})!=len(manifest)):
        raise ValueError("frozen validation-only manifest, six conditions and Ps=0.5 required")
    start = perf_counter(); rows=[]; realizations=[]; bins=ConditionBins(); manual=None
    timing = {k:0.0 for k in ("channel_generation","R3_estimation","R4_estimation","ZF","Linear_MMSE","aggregation","replay_verification")}
    counters = {"trajectory_generator_calls":0,"master_bits_calls":0,"unit_noise_calls":0,
        "frame_condition_evaluations":0,"paired_estimator_evaluations":0,"Cholesky_successes":0,"Cholesky_failures":0}
    for mi,row in enumerate(manifest):
        tid=row["trajectory_id"]; expected=refs["realizations"][tid]
        stamp=perf_counter(); trajectory=formal_trajectory(model,row)
        timing["channel_generation"]+=perf_counter()-stamp; counters["trajectory_generator_calls"]+=1
        if raw_array_sha256(trajectory.h_long)!=expected["H_long_sha256"] or trajectory.channel_seed!=expected["channel_seed"]:
            raise ValueError("exact H_long/channel-seed replay mismatch")
        totals={(df,snr,e):{"trajectory_id":tid,"dF":df,"snr_db":snr,"estimator":e,
            "pilot_ratio":count_layout_resources(layouts[df])["union_ratio"],"frames_aggregated":0,"total_bits":0,
            "zf_bit_errors":0,"mmse_bit_errors":0,"same_H_est":True,"same_Y":True,"same_bits":True,"same_data_mask":True}
            for df,snr in CONDITIONS for e in ESTIMATORS}
        frame_hashes=[]
        for frame,h in enumerate(trajectory.h_frames):
            bits,symbols=link.generate_master_qpsk(bits_seeds=tuple(bits_seed(tid,frame,q) for q in range(2)))
            z=complex_gaussian_noise((2,64,16),1,np.random.default_rng(noise_seed(tid,frame)))
            counters["master_bits_calls"]+=1; counters["unit_noise_calls"]+=1
            stamp=perf_counter(); frame_hashes.append(verify_frame(expected["frame_realizations"][frame],frame,h,bits,z))
            timing["replay_verification"]+=perf_counter()-stamp
            for df,snr in CONDITIONS:
                layout=layouts[df]; sigma2=noise_variance_from_snr(snr)
                y=link.add_received_noise(link.apply_frequency_mimo_channel(h,link.assemble_pilot_data_frame(symbols,layout)),scale_unit_noise(z,sigma2))
                stamp=perf_counter(); pilot_ls,linear=link.estimate_ls_linear_channel(y,layout)
                timing["R3_estimation"]+=perf_counter()-stamp
                stamp=perf_counter(); trained=estimate_mimo_from_pilot_ls(pilot_ls,train[df,snr])
                timing["R4_estimation"]+=perf_counter()-stamp
                channels={"R0":h,"R3":linear,"R4":trained}; detected={}
                for estimator,h_est in channels.items():
                    try:
                        outputs,times=detect_pair(y,h_est,bits,layout.data_mask,sigma2,ps,observer)
                    except (ValueError,np.linalg.LinAlgError) as exc:
                        raise NoiselessLinkFailure({"reason":str(exc),"trajectory_id":tid,"frame_index":frame,
                            "dF":df,"snr_db":snr,"estimator":estimator,"channel_seed":row["channel_seed"],
                            "bits_seeds":[bits_seed(tid,frame,q) for q in range(2)],"unit_noise_seed":noise_seed(tid,frame)}) from exc
                    timing["ZF"]+=times["ZF"]; timing["Linear_MMSE"]+=times["Linear-MMSE"]
                    detected[estimator]=outputs
                    counters["Cholesky_successes"]+=outputs["Linear-MMSE"]["diagnostics"]["cholesky_successes"]
                    counters["Cholesky_failures"]+=outputs["Linear-MMSE"]["diagnostics"]["cholesky_failures"]
                    counters["paired_estimator_evaluations"]+=1
                    stamp=perf_counter(); t=totals[df,snr,estimator]
                    if frame!=t["frames_aggregated"]: raise ValueError("frames must accumulate once in order")
                    t["frames_aggregated"]+=1; t["total_bits"]+=outputs["ZF"]["bits"]
                    t["zf_bit_errors"]+=outputs["ZF"]["errors"]; t["mmse_bit_errors"]+=outputs["Linear-MMSE"]["errors"]
                    timing["aggregation"]+=perf_counter()-stamp
                if (df,snr)==(16,10):
                    condition=np.linalg.cond(h[:,:,layout.data_mask[0]].transpose(2,0,1))
                    bins.add(condition,detected["R0"]["ZF"]["per_RE_errors"],detected["R0"]["Linear-MMSE"]["per_RE_errors"])
                    if (tid,frame)==(2000,0):
                        manual=validation_sentinel(refs,h,bits,symbols,z,y,channels,detected,layout)
                counters["frame_condition_evaluations"]+=1
            verify_frame(frame_hashes[-1],frame,h,bits,z)
        stamp=perf_counter()
        for (df,snr,estimator),t in totals.items():
            old=refs["trajectory_index"][tid,df,snr]
            if t["frames_aggregated"]!=10 or t["total_bits"]!=int(old["total_data_bits"]) or t["zf_bit_errors"]!=int(old[f"{OLD_METHOD[estimator]}_bit_errors"]):
                raise ValueError(f"frozen T10D trajectory ZF/denominator mismatch: {tid}/{df}/{snr}/{estimator}")
            t.update({"zf_ber":t["zf_bit_errors"]/t["total_bits"],"mmse_ber":t["mmse_bit_errors"]/t["total_bits"],
                "zf_minus_mmse_bit_errors":t["zf_bit_errors"]-t["mmse_bit_errors"],"frozen_T10D_ZF_exact_match":True})
            rows.append(t)
        actual={"trajectory_id":tid,"channel_seed":trajectory.channel_seed,"H_long_sha256":raw_array_sha256(trajectory.h_long),"frame_realizations":frame_hashes}
        if actual!=expected: raise ValueError("complete trajectory provenance differs")
        realizations.append(actual); timing["aggregation"]+=perf_counter()-stamp
        if progress: progress(mi+1)
    stamp=perf_counter(); aggregate,paired=aggregate_trajectories(rows); bin_rows=bins.finalize()
    timing["aggregation"]+=perf_counter()-stamp
    perfect=next(r for r in aggregate if (r["dF"],r["snr_db"],r["estimator"])==(16,10,"R0"))
    if any(sum(r[k] for r in bin_rows)!=perfect[k] for k in ("zf_bit_errors","mmse_bit_errors")) or sum(r["num_bits"] for r in bin_rows)!=perfect["total_bits"]:
        raise ValueError("every Perfect-CSI data RE/error must belong to exactly one bin")
    if manual is not None: manual["ten_frame_ZF_counts_exact_frozen_CSV_match"]=True
    timing["total_simulation_seconds"]=perf_counter()-start
    n=len(manifest); verify_unchanged(refs)
    return {"aggregate_results":aggregate,"paired_trajectory_diagnostics":paired,"condition_number_bins":bin_rows,
        "validation_t10d_replay_regression":manual,"realization_hashes":realizations,
        "runtime":timing,"execution_counters":counters,
        "replay_verification":{"num_h_hash_matches":n,"num_frame_h_hash_matches":10*n,"num_bits_hash_matches":10*n,
            "num_noise_hash_matches":10*n,"all_stored_fields_checked":True,"T10D_and_T11B_identical":True,
            "exact_match_fraction":1.0,"all_checks_passed":True}},rows
