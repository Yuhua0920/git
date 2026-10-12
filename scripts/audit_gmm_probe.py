"""Finalize T15B without repeating CDL generation, EM, or numeric evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/"outputs"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding="utf8"))
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf8")


def record_visual_review():
    r=load(OUT/"t15b_gmm_figure_audit.json")
    if r["source_returncode"] or not all(r["CSV_only_rebuild"].values()) or any(v["text_returncode"] or v["collision_returncode"] for v in r["rendered_checks"].values()):
        raise ValueError("automated gates must pass before actual visual review")
    r["visual_review"]="both final PNGs inspected: labels/legend/curves/cropping; no overlap"
    r["reviewed_source_warnings"]=["300dpi PNG preview without TIFF: development export", 
                                   "No uncertainty band: pooled ratio and train-selected EM seed, no formal CI requested"]
    r["all_checks_passed"]=True;save(OUT/"t15b_gmm_figure_audit.json",r)
    print("T15B actual two-figure visual review recorded")


def finalize():
    if Path(sys.prefix).resolve()!=(ROOT/".venv-gmm").resolve():raise ValueError("isolated GMM environment required")
    from scripts.build_gmm_probe_dataset import frozen_environments
    from scripts.check_cdl_environment import environment_fingerprint
    from scripts.run_gmm_probe import check_package_bytes
    r=load(OUT/"t15b_gmm_results.json");dataset=load(OUT/"t15b_gmm_probe_dataset_manifest.json")
    figure=load(OUT/"t15b_gmm_figure_audit.json")
    if not r["validation_nmse"] or not r["k1_regression"]["all_checks_passed"] or not figure["all_checks_passed"]:
        raise ValueError("numeric / K1 / figure gate incomplete")
    numeric_paths=[OUT/f for f in ("t15b_gmm_training.csv","t15b_gmm_probe_validation.csv","t15b_gmm_trajectory_metrics.csv","t15b_gmm_probe_dataset.npz")]
    hashes={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in numeric_paths}
    history={p:sha(ROOT/p)==digest for p,digest in dataset["frozen_history_sha256"].items()}
    if not all(history.values()):raise ValueError("T15A or older frozen history drift")
    if sha(ROOT/dataset["dataset_path"])!=dataset["dataset_sha256"]:raise ValueError("probe dataset drift")
    if any(sha(ROOT/p)!=digest for p,digest in r["source_sha256"].items()):raise ValueError("run source drift")
    with (OUT/"t15b_gmm_training.csv").open(encoding="utf8") as f:training=list(csv.DictReader(f))
    if len(training)!=12:raise ValueError("all 12 declared initializations must be retained")
    for entry in training:
        if entry["fit_succeeded"]=="True" and sha(ROOT/entry["model_path"])!=entry["model_sha256"]:
            raise ValueError("model artifact drift")
    print("Frozen dataset, 12 initialization records, models and source provenance PASS",flush=True)
    test=subprocess.run([str(ROOT/".venv/Scripts/python.exe"),"-m","pytest","-q"],cwd=ROOT,capture_output=True,text=True,encoding="utf8",errors="replace")
    (OUT/"t15b_gmm_pytest_full.txt").write_text(test.stdout+test.stderr,encoding="utf8")
    r["pytest"]=dict(command=".venv/Scripts/python.exe -m pytest -q",returncode=test.returncode,output=test.stdout+test.stderr)
    save(OUT/"t15b_gmm_results.json",r)
    if test.returncode:raise ValueError("full regression failed; numeric results retained")
    print(test.stdout.strip().splitlines()[-1],flush=True)
    env=frozen_environments();check_package_bytes()
    old=load(OUT/"t15b_gmm_environment_payload.json");current=environment_fingerprint(ROOT/".venv-gmm")
    if old["files"]!=current["files"]:raise ValueError("GMM environment payload drift")
    env["gmm"]=dict(before_sha256=old["sha256"],after_sha256=current["sha256"],payload_unchanged=True)
    save(OUT/"t15b_gmm_environment_regression.json",env)
    after={p:sha(ROOT/p) for p in hashes}
    if after!=hashes:raise ValueError("numeric artifacts changed during final audit")
    diff=subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError("git diff --check failed")
    r["environment_regression"]=env;r["frozen_history_unchanged"]=True;r["frozen_history_files_verified"]=len(history)
    r["figure_audit"]=figure;r["git_diff_check"]=dict(returncode=diff.returncode,output=diff.stdout+diff.stderr)
    r["numeric_artifacts_unchanged_during_final_audit"]=True
    r["numeric_artifact_sha256"]=hashes
    r["convergence"]={k:v.get("convergence") for k,v in r["gmm_results_by_K"].items()}
    r["component_weights"]={k:v.get("component_weights") for k,v in r["gmm_results_by_K"].items()}
    r["likelihood_diagnostic_definition"]="train log-likelihood: clean joint 128D prior; validation: noisy 64D pilot marginal at each sigma2; different dimensions, no cross-domain numeric comparison or validation selection"
    r["structured_covariance_feasibility_note"]="Author package supports structured covariances; no structured variant evaluated in T15B"
    r["reference_baseline_distinction"]="R10-probe is the independent 128D single Gaussian plus identical package regularizer; not a numeric reproduction of R10-full"
    r["initial_pytest_summary"]="1286 passed in 122.30s (0:02:02) before the accepted T15A/FIX commit"
    r["preflight_history"]=dict(first_plot_preflight="1 failed, 19 passed in 5.72s; SVG random IDs differed; archived",corrected_preflight="20 passed in 4.97s before EM",archive="outputs/t15b_preflight_attempt01")
    r["warnings"].extend(["Dataset replay used a temporary Matplotlib cache after default cache permission warning; replay succeeded.",
                           "Optional Triton flop counter unavailable; original frozen CDL runtime retained.",
                           "Ten frames share one trajectory; responsibility sample counts are frame counts, not independent trajectory counts.",
                           "Peak RAM per K is sampled process working set (20ms); lifetime peak is separately recorded.",
                           "Inference timing includes full author estimate API and filter/posterior preparation, not a cached deployment kernel.",
                           "Validation development probe only; no BER or formal-test evaluation."])
    r["final_audit_source_sha256"]=sha(Path(__file__))
    r["completion_state"]="completed";r["all_checks_passed"]=True
    save(OUT/"t15b_gmm_results.json",r)
    print("T15B final engineering gates PASS",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--record-visual-review",action="store_true")
    if p.parse_args().record_visual_review:record_visual_review()
    else:finalize()
