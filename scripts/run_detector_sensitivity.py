"""T11C-B: benchmark ten trajectories, then six-condition held-out sensitivity."""

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import scipy

from scripts.plot_detector_sensitivity import FIGURE_NAMES,plot_results
from scripts.run_lmmse_train_validation import qa_tools,save_csv,save_json
from scripts.validate_lmmse_train import check_config
from src.detector_sensitivity import (CONDITIONS,ESTIMATORS,T11C_A_COMMIT,historical_calibration_regression,
    load_inputs,prepare_train_pool,run_scan,verify_unchanged)
from src.lmmse_train import sha256_file
from src.tdl_noiseless_scan import NoiselessLinkFailure
from src.trajectories import prepare_trajectory

AGGREGATE="t11c_b_detector_sensitivity.csv"
TRAJECTORY="t11c_b_detector_sensitivity_trajectory_metrics.csv"
BINS="t11c_b_detector_sensitivity_condition_bins.csv"
RESULTS="t11c_b_detector_sensitivity_results.json"


def source_preflight(tools_dir):
    r=subprocess.run([sys.executable,str(Path(tools_dir)/"validate_figure.py"),
                      str(ROOT/"scripts/plot_detector_sensitivity.py"),"--json"],capture_output=True,text=True,encoding="utf8")
    if r.returncode: raise RuntimeError(f"plot preflight failed: {r.stdout}\n{r.stderr}")
    return json.loads(r.stdout)


def render_and_audit(aggregate,bins,directory,tools_dir=None):
    tools_dir,gate=qa_tools(tools_dir)
    paths,records=plot_results(aggregate,bins,directory,gate)
    for path,record in zip(paths,records):
        text=subprocess.run([sys.executable,str(tools_dir/"audit_pdf_text.py"),str(path.with_suffix(".pdf")),"--min-pt","5","--json"],capture_output=True,text=True,encoding="utf8")
        audit=path.with_suffix(".collision-audit.json")
        collision=subprocess.run([sys.executable,str(tools_dir/"audit_figure_collisions.py"),str(path.with_suffix(".pdf")),"--json-out",str(audit)],capture_output=True,text=True,encoding="utf8")
        if text.returncode or collision.returncode: raise RuntimeError(f"figure QA failed: {path.name}\n{text.stdout}\n{collision.stdout}\n{collision.stderr}")
        record.update({"path":str(path),"PNG_sha256":sha256_file(path),"PDF_text_audit":json.loads(text.stdout),
            "collision_audit":json.loads(audit.read_text(encoding="utf8")),
            "alignment_audit":json.loads(path.with_suffix(".alignment.json").read_text(encoding="utf8"))})
    return paths,records


def csv_only_rebuild(directory,tools_dir=None):
    directory=Path(directory)
    expected={name:sha256_file(directory/f"{name}.png") for name in FIGURE_NAMES}
    with tempfile.TemporaryDirectory(prefix="t11c_b_csv_only_") as temp:
        target=Path(temp)
        for name in (AGGREGATE,BINS): shutil.copyfile(directory/name,target/name)
        paths,_=render_and_audit(target/AGGREGATE,target/BINS,target,tools_dir)
        matches={p.stem:sha256_file(p)==expected[p.stem] for p in paths}
        if not all(matches.values()): raise ValueError("CSV-only PNG rebuild differs")
    return {"only_inputs":[AGGREGATE,BINS],"PNG_hash_matches":matches,"all_checks_passed":True}


def nmse_reference(refs):
    result=[]
    for df,snr in CONDITIONS:
        row=next(r for r in refs["scan"] if (int(r["dF"]),int(r["snr_db"]))==(df,snr))
        result.append({"dF":df,"snr_db":snr,"R0":"Perfect CSI; no numeric NMSE column in frozen T10D",
            "R3_nmse_linear":float(row["ls_nmse_linear"]),"R3_nmse_db":float(row["ls_nmse_db"]),
            "R4_nmse_linear":float(row["train_nmse_linear"]),"R4_nmse_db":float(row["train_nmse_db"])})
    return {"source":"outputs/t10d_lmmse_train_validation_scan.csv",
        "source_sha256":refs["before_all"]["outputs/t10d_lmmse_train_validation_scan.csv"],
        "NMSE_recomputed":False,"detector_NMSE_defined":False,"conditions":result}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark",action="store_true")
    parser.add_argument("--plot-only",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs")
    parser.add_argument("--figure-qa-tools-dir",type=Path)
    args=parser.parse_args(argv)
    directory=args.output_dir; directory.mkdir(exist_ok=True)
    tools_dir,_=qa_tools(args.figure_qa_tools_dir)
    preflight=source_preflight(tools_dir)
    if args.plot_only:
        _,figures=render_and_audit(directory/AGGREGATE,directory/BINS,directory,tools_dir)
        rebuild=csv_only_rebuild(directory,tools_dir)
        if (directory/RESULTS).exists():
            report=json.loads((directory/RESULTS).read_text(encoding="utf8"))
            for name,digest in report["output_csv_sha256"].items():
                if sha256_file(directory/name)!=digest: raise ValueError("stored numeric CSV changed")
            report.update({"figures":figures,"csv_only_rebuild":rebuild,"figure_source_preflight":preflight})
            report["figures_pending"]=False
            report["all_checks_passed"]=report["numeric_checks_passed"] and rebuild["all_checks_passed"]
            report["provenance"]["plot_source_sha256"]=sha256_file(ROOT/"scripts/plot_detector_sensitivity.py")
            save_json(directory/RESULTS,report)
        print(json.dumps({"plot_only":True,"csv_only_rebuild":rebuild},indent=2)); return
    started=perf_counter(); config=check_config(); refs=load_inputs(ROOT)
    historical=historical_calibration_regression(refs)
    layouts,train,preparation=prepare_train_pool(refs)
    stamp=perf_counter(); model=prepare_trajectory(); model_seconds=perf_counter()-stamp
    manifest=refs["manifest"][:10] if args.benchmark else refs["manifest"]
    if not args.benchmark and not (directory/"t11c_b_benchmark_results.json").is_file():
        raise ValueError("execute the ten-trajectory benchmark and figure QA before the full run")
    def progress(done):
        if done%10==0 or done==len(manifest): print(f"validation trajectories {done}/{len(manifest)}; elapsed {perf_counter()-started:.1f}s",flush=True)
    try:
        report,rows=run_scan(refs,manifest,model,layouts,train,config["system"]["total_power"]/config["system"]["nt"],progress)
    except NoiselessLinkFailure as exc:
        save_json(directory/"t11c_b_detector_sensitivity_failure.json",{"task":"T11C-B","failure":exc.record,
            "num_filtered_trajectories":0,"all_checks_passed":False}); raise
    report.update({"task":"T11C-B","experiment_type":"limited held-out detector-sensitivity validation for ZF versus linear-MMSE",
        "benchmark_only":args.benchmark,"conditions":[{"dF":df,"snr_db":snr} for df,snr in CONDITIONS],
        "estimators":list(ESTIMATORS),"detectors":["ZF","Linear-MMSE"],"num_independent_trajectories":len(manifest),
        "frames_per_trajectory":10,"num_correlated_frames":10*len(manifest),"visited_validation_trajectory_ids":[r["trajectory_id"] for r in manifest],
        "excluded_estimators":{"R1":"raw Pilot-LS has no data BER","R2":"previous weak interpolation reference",
            "R5":"near R4 under ID covariance; task isolates detector sensitivity"},
        "stream_symbol_power":0.5,"regularization_definition":"noise_variance / stream_symbol_power",
        "noise_variance_definition":"complex variance per RX, 10^(-SNR/10); no instantaneous-power adjustment",
        "historical_t11c_a_calibration_regression":historical,
        "t11c_a_fixed_case_regression":{"researcher_approved_namespace_amendment":True,"two_independent_namespaces":True},
        "channel_estimation_nmse_reference":nmse_reference(refs),"estimator_preparation":preparation,
        "main_detector":"ZF","main_detector_changed":False,"formal_test_used":False,"formal_test_sealed":True,
        "OOD_generated":False,"num_filtered_trajectories":0,"numerical_failures":[],"covariance_reestimated":False,
        "condition_diagnostic_scope":{"estimator":"R0","dF":16,"snr_db":10,"data_RE_only":True,
            "fixed_bins":["[1,2)","[2,4)","[4,8)","[8,16)","[16,inf)"],"clipping":False,"winsorizing":False},
        "seed_policy":refs["old"]["seed_policy"],"training_seed":None,
        "frozen_hash_before_after":{"before":refs["before_all"],"after":verify_unchanged(refs)},
        "frozen_R_train_raw_sha256":refs["R_train_raw_before"],
        "provenance":{"accepted_T11C_A_commit":T11C_A_COMMIT,"T11C_A_committed_cleanly_at_task_start":True,
            "D0_path":"accepted T10D batched_data_zf","D1_path":"unchanged T11C-A detect_data_linear_mmse",
            "python_version":platform.python_version(),"numpy_version":np.__version__,"scipy_version":scipy.__version__,
            "source_sha256":{p:sha256_file(ROOT/p) for p in ("src/detector_sensitivity.py","scripts/run_detector_sensitivity.py",
                "tests/test_detector_sensitivity.py")},"plot_source_sha256":sha256_file(ROOT/"scripts/plot_detector_sensitivity.py")}})
    report["runtime"].update({"train_estimator_preparation":preparation["seconds"],"trajectory_model_preparation":model_seconds})
    report["all_checks_passed"]=(len(rows)==18*len(manifest) and len(report["aggregate_results"])==18
        and report["replay_verification"]["all_checks_passed"] and historical["all_checks_passed"]
        and report["validation_t10d_replay_regression"]["all_checks_passed"] and report["execution_counters"]["Cholesky_failures"]==0)
    report["numeric_checks_passed"]=report["all_checks_passed"]
    if args.benchmark:
        save_csv(directory/"t11c_b_benchmark_trajectory_metrics.csv",rows)
        with tempfile.TemporaryDirectory(prefix="t11c_b_benchmark_figures_") as temp:
            save_csv(Path(temp)/AGGREGATE,report["aggregate_results"]); save_csv(Path(temp)/BINS,report["condition_number_bins"])
            _,figures=render_and_audit(Path(temp)/AGGREGATE,Path(temp)/BINS,temp,tools_dir)
        report.update({"benchmark_figure_QA":figures,"figure_source_preflight":preflight})
        report["runtime"]["total_command_seconds"]=perf_counter()-started
        save_json(directory/"t11c_b_benchmark_results.json",report)
    else:
        if not report["all_checks_passed"]: raise ValueError("numeric sensitivity checks failed")
        save_csv(directory/TRAJECTORY,rows); save_csv(directory/AGGREGATE,report["aggregate_results"])
        save_csv(directory/BINS,report["condition_number_bins"])
        report["output_csv_sha256"]={n:sha256_file(directory/n) for n in (TRAJECTORY,AGGREGATE,BINS)}
        # Numeric artifacts are recoverable before rendering; plot-only never replays.
        report["figures_pending"]=True; report["all_checks_passed"]=False; save_json(directory/RESULTS,report)
        _,report["figures"]=render_and_audit(directory/AGGREGATE,directory/BINS,directory,tools_dir)
        report["csv_only_rebuild"]=csv_only_rebuild(directory,tools_dir)
        report["figures_pending"]=False; report["figure_source_preflight"]=preflight
        report["all_checks_passed"]=report["numeric_checks_passed"] and report["csv_only_rebuild"]["all_checks_passed"]
        report["runtime"]["total_command_seconds"]=perf_counter()-started
        report["frozen_hash_before_after"]["after"]=verify_unchanged(refs)
        save_json(directory/RESULTS,report)
    print(json.dumps({"benchmark":args.benchmark,"runtime":report["runtime"],"rows":len(rows),
                      "aggregate_rows":18,"all_checks_passed":report["all_checks_passed"]},indent=2))
    return report


if __name__=="__main__":
    main()
