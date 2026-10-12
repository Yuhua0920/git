"""T15A-FIX: strict historical gate, then corrected receive-SNR evaluation.

Never alters the original CDL implementation or frozen clean statistics.
Historical hashes and new replay hashes are explicitly distinct evidence.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".pytest_cache/matplotlib-cdl"))
import numpy as np
from src.cdl_benchmark import (OfficialCDL,FrozenStatistics,frames,array_hash,seeds,
    manifest_rows,observation_geometry,SNRS,PREFIXES)
from src.cdl_snr import TrainReceivePowerAccumulator,SNR_DEFINITION
from src.link import (generate_master_qpsk,assemble_pilot_data_frame,apply_frequency_mimo_channel,
    estimate_ls_linear_channel,detect_data_zf,extract_reference_data_bits,demodulate_data_bits)
from src.numerics import complex_gaussian_noise,noise_variance_from_snr
from scripts.run_cdl_benchmark import save_csv,save_json,old_hashes
from scripts.check_cdl_snr_environment import verify as verify_environment
OUT=ROOT/"outputs"
ARCHIVE=OUT/"t15a_attempt01_nominal_tx_snr"
METHODS=("R0-Perfect-CSI","R3-LS-LIN","R10-100","R10-250","R10-500")
FIX_SOURCES=("src/cdl_snr.py","scripts/fix_cdl_snr.py","scripts/archive_cdl_nominal_snr.py",
    "scripts/check_cdl_snr_environment.py","scripts/plot_cdl_benchmark.py","tests/test_cdl_snr.py",
    "docs/cdl_snr_fix_protocol.md")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path): return json.loads(Path(path).read_text(encoding="utf8"))


def archive_verified():
    m=load_json(ARCHIVE/"superseded_metadata.json")
    checks={p:sha(ARCHIVE/p)==info["sha256"] for p,info in m["file_hashes"].items()}
    if not all(checks.values()): raise ValueError("historical archive changed")
    return checks


def load_frozen_statistics(old):
    statistics={};checks={}
    for n in PREFIXES:
        path=OUT/f"t15a_cdl_statistics_{n}.npz"
        if sha(path)!=old["empirical_lmmse_definition"]["snapshots"][str(n)]["sha256"] or sha(path)!=sha(ARCHIVE/"outputs"/path.name):
            raise ValueError("frozen clean statistics changed")
        with np.load(path) as f:
            arrays=[f[k].copy() for k in ("mean_p","mean_d","rpp","rdp")]
            ids=tuple(map(int,f["train_trajectory_ids"])); samples=int(f["samples"])
        if ids!=tuple(range(n)) or samples!=10*n: raise ValueError("statistics training namespace differs")
        for a in arrays:a.flags.writeable=False
        statistics[n]=FrozenStatistics(*arrays,ids,samples)
        checks[str(n)]=dict(file_sha256=sha(path),unchanged=True,
            array_sha256={k:array_hash(a) for k,a in zip(("mean_p","mean_d","rpp","rdp"),arrays)})
    return statistics,checks


def require_historical_gate(gate):
    if not gate.get("all_checks_passed",False): raise ValueError("STOP: historical attempt01 replay gate did not pass")


def compare_metric_rows(actual,path):
    with Path(path).open(encoding="utf8",newline="") as f: expected=list(csv.DictReader(f))
    diagnostics={};mismatches=[];compared=0
    if len(actual)!=len(expected): return dict(exact=False,row_count_actual=len(actual),row_count_expected=len(expected))
    for index,(a,b) in enumerate(zip(actual,expected,strict=True)):
        if list(a)!=list(b): raise ValueError("metric schema/order differs from attempt01")
        for key,value in a.items():
            written="" if value is None else str(value); original=b[key]; compared+=1
            if written!=original:
                if len(mismatches)<8:mismatches.append(dict(row=index,field=key,actual=written,original=original))
            try:
                v=float(written);o=float(original);absolute=abs(v-o)
                relative=absolute/max(abs(o),1e-300)
                d=diagnostics.setdefault(key,dict(max_abs_error=0.,max_rel_error=0.))
                d["max_abs_error"]=max(d["max_abs_error"],absolute);d["max_rel_error"]=max(d["max_rel_error"],relative)
            except ValueError:pass
    return dict(exact=not mismatches,rows=len(actual),fields_compared=compared,
        maximum_numeric_errors=diagnostics,mismatch_examples=mismatches,
        tolerance_used=False,comparison="exact serialized fields; numeric errors also reported")


def freeze_replay_inputs(h,communication):
    h.flags.writeable=False
    for values in communication:
        for a in values:a.flags.writeable=False


def replay_input_hashes(h,communication):
    result={"H":array_hash(h)}
    for name,j in (("bits",0),("symbols",1),("unit_noise",2)):
        digest=hashlib.sha256()
        for values in communication:digest.update(np.ascontiguousarray(values[j]).tobytes())
        result[name]=digest.hexdigest()
    return result


def assert_replay_unchanged(h,communication,before):
    after=replay_input_hashes(h,communication)
    if before!=after:raise ValueError("shared replay H/bits/symbols/unit-noise changed")
    return after


def evaluate_trajectory(i,h,communication,g,estimators,variances):
    """Same existing receive operations/order as the archived T15A scan."""
    accum={(snr,method):dict(error_energy=0.,truth_energy=0.,bit_errors=0,total_bits=0) for snr in SNRS for method in METHODS}
    for truth,(bits,symbols,unit_noise) in zip(frames(h),communication,strict=True):
        transmitted=assemble_pilot_data_frame(symbols,g.layout);noiseless=apply_frequency_mimo_channel(truth,transmitted)
        reference=extract_reference_data_bits(bits,g.layout.data_mask)
        truth_energy=float(np.sum(abs(truth.ravel()[g.data])**2))
        for snr in SNRS:
            y=noiseless+np.sqrt(variances[snr])*unit_noise
            sparse,linear=estimate_ls_linear_channel(y,g.layout);z=sparse.ravel()[g.selected]
            estimates={"R0-Perfect-CSI":truth,"R3-LS-LIN":linear}
            for n in PREFIXES:
                estimate=np.full(4096,np.nan+1j*np.nan,np.complex128)
                estimate[g.data]=estimators[n,snr].estimate(z)
                estimates[f"R10-{n}"]=estimate.reshape(2,2,64,16)
            for method,estimate in estimates.items():
                detected=detect_data_zf(y,estimate,g.layout.data_mask)*np.sqrt(2)
                predicted=demodulate_data_bits(detected,g.layout.data_mask)
                errors=sum(int(np.count_nonzero(a!=b)) for a,b in zip(predicted,reference))
                entry=accum[snr,method];entry["bit_errors"]+=errors;entry["total_bits"]+=sum(b.size for b in reference)
                entry["truth_energy"]+=truth_energy
                if method!="R0-Perfect-CSI":entry["error_energy"]+=float(np.sum(abs(estimate.ravel()[g.data]-truth.ravel()[g.data])**2))
    return [dict(trajectory_id=i,split="validation_dev",frames=10,dF=16,snr_db=snr,method=method,**entry,
        nmse_linear=entry["error_energy"]/entry["truth_energy"] if method!="R0-Perfect-CSI" else "",
        ber=entry["bit_errors"]/entry["total_bits"]) for (snr,method),entry in accum.items()]


def aggregate_rows(rows,g):
    result=[]
    for snr in SNRS:
        for method in METHODS:
            subset=[r for r in rows if r["snr_db"]==snr and r["method"]==method]
            e=sum(r["error_energy"] for r in subset);den=sum(r["truth_energy"] for r in subset)
            err=sum(r["bit_errors"] for r in subset);bitcount=sum(r["total_bits"] for r in subset)
            nmse=e/den if method!="R0-Perfect-CSI" else None
            result.append(dict(method=method,dF=16,pilot_ratio=float(g.layout.pilot_union.mean()),snr_db=snr,
                trajectories=100,frames_per_trajectory=10,nmse_linear=nmse,nmse_db=float(10*np.log10(nmse)) if nmse else None,
                trajectory_nmse_mean=float(np.mean([r["nmse_linear"] for r in subset])) if nmse else None,
                trajectory_nmse_std=float(np.std([r["nmse_linear"] for r in subset],ddof=1)) if nmse else None,
                error_energy=e,truth_energy=den,bit_errors=err,total_bits=bitcount,ber=err/bitcount,
                trajectory_ber_std=float(np.std([r["ber"] for r in subset],ddof=1))))
    return result


def corrected_run(gate,cache,statistics,reference,g):
    require_historical_gate(gate)  # No corrected filter/performance work before gateA.
    variances={snr:reference.noise_variance(snr) for snr in SNRS}
    estimators={(n,snr):statistics[n].prepare(variances[snr]) for n in PREFIXES for snr in SNRS}
    rows=[];identity=[];validation_power=np.zeros(2);signal_energy=np.zeros(2);noise_energy=np.zeros(2);samples=0
    for i,h,communication,before in cache:
        assert_replay_unchanged(h,communication,before)
        rows.extend(evaluate_trajectory(i,h,communication,g,estimators,variances))
        after=assert_replay_unchanged(h,communication,before)
        identity.append(dict(trajectory_id=i,**{name+"_hash_unchanged":before[name]==after[name] for name in before}))
        validation_power+=.5*np.mean(abs(h)**2,axis=(2,3)).sum(axis=1)
        # Post-hoc only. Expected symbol-power audit plus realized useful data RE
        # energies are distinct; pilots/mutes do not enter useful-data energy.
        for truth,(bits,symbols,z) in zip(frames(h),communication,strict=True):
            x=assemble_pilot_data_frame(symbols,g.layout);s=apply_frequency_mimo_channel(truth,x)
            data=g.layout.data_mask[0]
            signal_energy+=np.sum(abs(s[:,data])**2,axis=1)
            noise_energy+=np.sum(abs(z[:,data])**2,axis=1);samples+=int(data.sum())
        if (i-499)%10==0:print(f"corrected validation: {i-499}/100",flush=True)
    validation_power/=len(cache)
    audit=[]
    for snr in SNRS:
        sigma=variances[snr]
        audit.append(dict(target_snr_db=snr,train_reference_identity_db=float(10*np.log10(reference.reference/sigma)),
            validation_expected_useful_power_per_rx=validation_power.tolist(),
            validation_expected_receive_snr_per_rx_db=(10*np.log10(validation_power/sigma)).tolist(),
            validation_expected_receive_snr_mean_reference_db=float(10*np.log10(validation_power.mean()/sigma)),
            validation_realized_useful_power_per_rx=(signal_energy/samples).tolist(),
            validation_realized_noise_power_per_rx=(sigma*noise_energy/samples).tolist(),
            validation_realized_data_signal_to_noise_per_rx_db=(10*np.log10(signal_energy/(sigma*noise_energy))).tolist(),
            reference_recalibrated_from_validation=False))
    return rows,aggregate_rows(rows,g),identity,audit,[e.solve_relative_residual for e in estimators.values()]


def prefix_audit(aggregate):
    result=[]
    for snr in SNRS:
        values={n:next(r for r in aggregate if r["snr_db"]==snr and r["method"]==f"R10-{n}") for n in PREFIXES}
        result.append(dict(snr_db=snr,nmse_linear={str(n):values[n]["nmse_linear"] for n in PREFIXES},
            nmse_db={str(n):values[n]["nmse_db"] for n in PREFIXES},
            **{f"delta_{a}_to_{b}_{unit}":values[b]["nmse_"+unit]-values[a]["nmse_"+unit]
                for a,b in ((100,250),(250,500)) for unit in ("linear","db")}))
    return result


def run():
    if Path(sys.prefix).resolve()!=(ROOT/".venv-cdl").resolve():raise ValueError("use original isolated .venv-cdl")
    if (OUT/"t15a_fix_historical_replay_gate.json").exists():raise ValueError("existing replay gate protected; do not silently retry")
    started=time.perf_counter();old=load_json(ARCHIVE/"outputs/t15a_cdl_results.json")
    baseline=load_json(ARCHIVE/"superseded_metadata.json")
    archive_checks=archive_verified();verify_environment("before")
    for path in ("src/cdl_benchmark.py","scripts/run_cdl_benchmark.py"):
        if sha(ROOT/path)!=sha(ARCHIVE/path):raise ValueError("original CDL/numerical channel source changed")
    if old_hashes()!=old["provenance"]["historical_before"]:raise ValueError("older frozen history changed")
    manifest_path=OUT/"t15a_cdl_manifest.csv"
    if sha(manifest_path)!=sha(ARCHIVE/"outputs"/manifest_path.name):raise ValueError("manifest bytes changed")
    with manifest_path.open(encoding="utf8",newline="") as f:original_rows=list(csv.DictReader(f))
    mapping_exact=original_rows==[{k:str(v) for k,v in r.items()} for r in manifest_rows()]
    if not mapping_exact:raise ValueError("manifest seed mapping changed")
    statistics,stats_checks=load_frozen_statistics(old);g=observation_geometry();adapter=OfficialCDL()
    if adapter.provenance()!=old["sionna_provenance"]:raise ValueError("official CDL/device/provenance changed")
    accumulator=TrainReceivePowerAccumulator();h_hashes=[];six={};expected_h=old["calibration"]["selected_trajectory_hashes"]
    def record_h(i,h):
        value=array_hash(h);historical=expected_h.get(str(i))
        h_hashes.append(dict(trajectory_id=i,split="train_dev" if i<500 else "validation_dev",
            evidence="new deterministic replay H hash",replayed_h_sha256=value,
            historical_h_hash_available=historical is not None,historical_h_sha256=historical or "",
            historical_match=value==historical if historical else ""))
        if historical:six[str(i)]=value==historical
        if historical and value!=historical:raise ValueError(f"historical H hash mismatch: trajectory {i}")
    train_started=time.perf_counter()
    for i in range(500):
        h=adapter.generate(i);record_h(i,h);accumulator.update(i,h)
        if (i+1)%100==0:print(f"train power replay: {i+1}/500; no covariance fitting",flush=True)
    reference=accumulator.freeze();power_seconds=time.perf_counter()-train_started
    old_variances={snr:noise_variance_from_snr(snr) for snr in SNRS}
    old_estimators={(n,snr):statistics[n].prepare(old_variances[snr]) for n in PREFIXES for snr in SNRS}
    cache=[];historical=[];frame_hashes=[];replay_started=time.perf_counter()
    for i in range(500,600):
        h=adapter.generate(i);record_h(i,h);communication=[]
        for j in range(10):
            seed=seeds(i,j);bits,symbols=generate_master_qpsk(bits_seeds=(seed["bits_seed_tx0"],seed["bits_seed_tx1"]))
            z=complex_gaussian_noise((2,64,16),1.,np.random.default_rng(seed["noise_seed"]))
            communication.append((bits,symbols,z))
            frame_hashes.append(dict(trajectory_id=i,frame_id=j,evidence="new replay frame hashes; no original historical frame hashes",
                bits_sha256=array_hash(bits),symbols_sha256=array_hash(symbols),unit_noise_sha256=array_hash(z)))
        freeze_replay_inputs(h,communication);before=replay_input_hashes(h,communication)
        historical.extend(evaluate_trajectory(i,h,communication,g,old_estimators,old_variances))
        assert_replay_unchanged(h,communication,before);cache.append((i,h,communication,before))
        if (i-499)%10==0:print(f"historical attempt01 replay: {i-499}/100",flush=True)
    aggregate=aggregate_rows(historical,g)
    save_csv("t15a_fix_historical_trajectory_replay.csv",historical)
    save_csv("t15a_fix_historical_aggregate_replay.csv",aggregate)
    comparisons={"trajectory":compare_metric_rows(historical,ARCHIVE/"outputs/t15a_cdl_trajectory_metrics.csv"),
        "aggregate":compare_metric_rows(aggregate,ARCHIVE/"outputs/t15a_cdl_lmmse.csv")}
    bytes_exact={"trajectory":sha(OUT/"t15a_fix_historical_trajectory_replay.csv")==sha(ARCHIVE/"outputs/t15a_cdl_trajectory_metrics.csv"),
        "aggregate":sha(OUT/"t15a_fix_historical_aggregate_replay.csv")==sha(ARCHIVE/"outputs/t15a_cdl_lmmse.csv")}
    gate=dict(task="T15A-FIX gate A",historical_full_frame_hashes_available=False,
        deterministic_cdl_replay_used=True,historical_six_h_hashes_matched=len(six)==6 and all(six.values()),
        historical_six_h_hash_checks=six,manifest_bytes_unchanged=True,manifest_seed_mapping_exact=mapping_exact,
        clean_channel_statistics_unchanged=True,statistics=stats_checks,
        historical_attempt01_metrics_reproduced=all(c["exact"] for c in comparisons.values()) and all(bytes_exact.values()),
        comparisons=comparisons,metric_CSV_bytes_exact=bytes_exact,archive_files_preserved=all(archive_checks.values()),
        corrected_performance_started=False,tolerance_used=False,
        all_checks_passed=len(six)==6 and all(six.values()) and all(c["exact"] for c in comparisons.values()) and all(bytes_exact.values()))
    save_json("t15a_fix_historical_replay_gate.json",gate)
    save_csv("t15a_fix_replay_h_hashes.csv",h_hashes);save_csv("t15a_fix_replay_frame_hashes.csv",frame_hashes)
    require_historical_gate(gate)
    print("Historical gate PASS: six H hashes exact, all 1500+15 rows/fields and CSV bytes exact.",flush=True)
    replay_seconds=time.perf_counter()-replay_started
    # Train reference was frozen before validation; no calibration update below.
    corrected_started=time.perf_counter()
    rows,aggregate,identity,validation_audit,residuals=corrected_run(gate,cache,statistics,reference,g)
    corrected_seconds=time.perf_counter()-corrected_started
    _,stats_after=load_frozen_statistics(old)
    if stats_checks!=stats_after:raise ValueError("statistics changed during corrected evaluation")
    unaffected_names=("t15a_cdl_manifest.csv","t15a_cdl_calibration.json","t15a_cdl_distribution_audit.csv",
        "t15a_cdl_projection_audit.csv","t15a_cdl_marginal_covariances.npz","t15a_cdl_environment.json","t15a_cdl_packages_freeze.txt")
    unaffected={name:sha(OUT/name)==sha(ARCHIVE/"outputs"/name) for name in unaffected_names}
    if not all(unaffected.values()):raise ValueError("noise-independent diagnostics changed")
    archive_verified()
    save_csv("t15a_cdl_trajectory_metrics.csv",rows);save_csv("t15a_cdl_lmmse.csv",aggregate)
    save_csv("t15a_fix_replay_identity.csv",identity)
    result=copy.deepcopy(old)
    # Old completion metadata stays in the original archive, not as a current claim.
    for name in ("metadata_only_amendment","artifact_sha256","pytest","git_diff_check","figure_audit","main_environment_final"):
        result.pop(name,None)
    result.update(task="T15A-FIX",snr_definition=SNR_DEFINITION,
        train_reference_signal_power=reference.reference,per_rx_signal_power=list(reference.per_rx_signal_power),
        per_rx_reference_snr_offsets_db=reference.branch_offsets_db().tolist(),
        train_reference_definition=dict(stream_power=.5,train_trajectory_ids=list(reference.train_trajectory_ids),
            averaging="500 train trajectories x10 frames x64 frequencies x16 symbols; each trajectory equally weighted",
            common_noise_variance=True,validation_used_for_calibration=False),
        noise_variances=[dict(snr_db=snr,old_sigma2=old_variances[snr],new_sigma2=reference.noise_variance(snr),
            ratio=reference.noise_variance(snr)/old_variances[snr],
            target_identity_db=float(10*np.log10(reference.reference/reference.noise_variance(snr)))) for snr in SNRS],
        attempt01_superseded=True,attempt01_status="superseded_due_to_snr_semantics",
        attempt01_archive="outputs/t15a_attempt01_nominal_tx_snr",
        snr_semantics_repaired=True,historical_full_frame_hashes_available=False,deterministic_cdl_replay_used=True,
        historical_six_h_hashes_matched=gate["historical_six_h_hashes_matched"],
        historical_attempt01_metrics_reproduced=gate["historical_attempt01_metrics_reproduced"],
        historical_replay_gate=gate,clean_channel_statistics_unchanged=True,
        replay_shared_input_identity=dict(evidence="same readonly objects in gate A and corrected run B; hashes recorded newly in this task",
            trajectories=100,frames=1000,all_unchanged=all(all(v for k,v in row.items() if k!="trajectory_id") for row in identity),
            historical_bits_and_unit_noise_hashes_available=False,
            new_replay_h_hashes="outputs/t15a_fix_replay_h_hashes.csv",new_replay_frame_hashes="outputs/t15a_fix_replay_frame_hashes.csv",identity_path="outputs/t15a_fix_replay_identity.csv"),
        noise_independent_diagnostics_unchanged=unaffected,
        covariance_sample_size_audit=prefix_audit(aggregate),validation_results=aggregate,validation_rows=len(rows),
        validation_realized_snr_audit=validation_audit,cholesky_solve_relative_residuals=residuals,
        empirical_lmmse_stable=None,empirical_lmmse_stability_assessment="descriptive_only",
        runtime=dict(train_power_replay_seconds=power_seconds,historical_metric_replay_seconds=replay_seconds,
            corrected_validation_seconds=corrected_seconds,fix_wall_seconds=time.perf_counter()-started,
            temporary_validation_input_RAM_bytes=sum(h.nbytes+sum(a.nbytes for values in comm for a in values) for i,h,comm,before in cache),
            complete_raw_H_saved=False,clean_covariance_refitted=False,new_model_trained=False),
        all_checks_passed=False,completion_state="corrected numeric scan complete; full pytest/environment/figure checks pending")
    result["provenance"]["attempt01_original_source_sha256"]=old["provenance"]["source_sha256"]
    result["provenance"]["source_sha256"]={p:sha(ROOT/p) for p in set(old["provenance"]["source_sha256"])|set(FIX_SOURCES)}
    result["provenance"]["historical_after"]=old_hashes()
    result["warnings"]=[w for w in old["warnings"] if "fixed nominal transmit SNR is retained" not in w]
    result["warnings"] += ["Attempt01 nominal-transmit-SNR performance is superseded; archive bytes preserved.",
        "Historical hashes exist for six trajectories only; all newly recorded H/bits/noise hashes are replay evidence, not invented historical hashes.",
        "Train-only mean receive-power reference is common to both RX branches; branch asymmetry and validation SNR are descriptive audits only."]
    save_json("t15a_cdl_results.json",result)
    print(json.dumps({"per_rx_signal_power":result["per_rx_signal_power"],"reference":reference.reference,
        "noise_variances":result["noise_variances"],"prefix_audit":result["covariance_sample_size_audit"]},indent=2),flush=True)


def finalize():
    result=load_json(OUT/"t15a_cdl_results.json")
    if result.get("task")!="T15A-FIX":raise ValueError("corrected scan required")
    require_historical_gate(result["historical_replay_gate"]);archive_verified()
    if old_hashes()!=result["provenance"]["historical_before"]:raise ValueError("old history changed")
    if any(sha(ROOT/p)!=value for p,value in result["provenance"]["source_sha256"].items()):raise ValueError("fix run sources changed")
    p=subprocess.run([str(ROOT/".venv/Scripts/python.exe"),"-m","pytest","-q"],cwd=ROOT,capture_output=True,text=True,encoding="utf8",errors="replace")
    (OUT/"t15a_fix_pytest_full.txt").write_text(p.stdout+p.stderr,encoding="utf8")
    result["pytest"]=dict(command=".venv/Scripts/python.exe -m pytest -q",returncode=p.returncode,output=p.stdout+p.stderr)
    save_json("t15a_cdl_results.json",result)
    if p.returncode:raise ValueError("full regression failed; corrected numeric results retained")
    verify_environment("after");result["environment_regression"]=load_json(OUT/"t15a_fix_environment_after.json")
    figure=load_json(OUT/"t15a_cdl_figure_audit.json")
    if not figure["all_checks_passed"]:raise ValueError("figure visual review required")
    result["figure_audit"]=figure
    check=subprocess.run(["git","diff","--check"],capture_output=True,text=True)
    result["git_diff_check"]=dict(returncode=check.returncode,output=check.stdout+check.stderr)
    if check.returncode:raise ValueError("diff check failed")
    archive_verified();load_frozen_statistics(load_json(ARCHIVE/"outputs/t15a_cdl_results.json"))
    result["all_checks_passed"]=True;result["completion_state"]="completed"
    result["artifact_sha256"]={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for pattern in ("t15a_cdl_*","t15a_fix_*") for p in OUT.glob(pattern) if p.is_file() and p.name!="t15a_cdl_results.json"}
    save_json("t15a_cdl_results.json",result)
    print("T15A-FIX full pytest, both environments, archive, statistics and figures PASS",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--finalize",action="store_true")
    if p.parse_args().finalize:finalize()
    else:run()
