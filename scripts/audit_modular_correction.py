"""T16D saved-artifact final audit; never replay or generate any channel."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_modular_correction import (OUT,RESULT,SOURCES,accepted_snapshot,environment,
    frozen_validation_tables,load,save,sha,environments)
from src.cdl_benchmark import array_hash
from src.heldout_basis import load_bases


def visual_review():
    r=load(OUT/"t16d_modular_figure_audit.json")
    if r["source_returncode"] or not all(r["CSV_only_rebuild"].values()) or any(v["text_returncode"] or v["collision_returncode"] for v in r["rendered_checks"].values()):
        raise ValueError("automated figure gates incomplete")
    r["visual_review"]="Actual final PNGs inspected: B120 equal-budget curves, both scalar NMSE panels; legend/style mapping, threshold, labels, axes, margins and all requested conditions clear."
    r["reviewed_warnings"]=["PNG300dpi preview with PDF/SVG; no TIFF","Pooled ratios, trajectory dispersion saved; no CI"]
    r["all_checks_passed"]=True;save(OUT/"t16d_modular_figure_audit.json",r)
    print("Actual T16D visual review recorded")


def finalize():
    environment();start=time.perf_counter();r=load(RESULT)
    for p,h in {**r["source_sha256"],**r["frozen_history_sha256"],**r["numeric_artifact_sha256"]}.items():
        if sha(ROOT/p)!=h:raise ValueError(f"source/history/numeric drift: {p}")
    accepted_snapshot();frozen_validation_tables();load_bases(ROOT)
    figure=load(OUT/"t16d_modular_figure_audit.json")
    if not figure["all_checks_passed"]:raise ValueError("actual figure review incomplete")
    for label,args in (("provenance",["tests/test_modular_correction.py","-k","snapshot or frozen or actual_results or replay"]),("full",[])):
        command=[str(ROOT/".venv/Scripts/python.exe"),"-m","pytest","-q",*args]
        test=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding="utf8",errors="replace")
        output=test.stdout+test.stderr;(OUT/f"t16d_modular_pytest_{label}.txt").write_text(output,encoding="utf8")
        r[f"pytest_{label}"]=dict(command=" ".join(command),returncode=test.returncode,output=output);save(RESULT,r)
        if test.returncode:raise ValueError("pytest failed; numerical artifacts retained")
        print(output.strip().splitlines()[-1],flush=True)
    for p,h in {**r["source_sha256"],**r["frozen_history_sha256"],**r["numeric_artifact_sha256"]}.items():
        if sha(ROOT/p)!=h:raise ValueError("file changed during final audit")
    info=r["union_basis_artifact"]
    if sha(ROOT/info["path"])!=info["file_sha256"]:raise ValueError("union artifact changed")
    with np.load(ROOT/info["path"],allow_pickle=False) as f:
        if any(array_hash(f[f"Q{q}"])!=h for q,h in info["Q_sha256"].items()):raise ValueError("Q array changed")
    load_bases(ROOT);r["environment_after"]=environments()
    if r["environment_before"]!=r["environment_after"]:raise ValueError("frozen environment changed")
    diff=subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError("diff check failed")
    r["git_diff_check"]=dict(returncode=diff.returncode,output=diff.stdout+diff.stderr)
    r["figure_audit"]=figure;r["startup_pytest"]=(ROOT/".pytest_cache/t16d_startup_pytest.txt").read_text(encoding="utf8")
    r["small_pytest"]=(ROOT/".pytest_cache/t16d_small_integrated.txt").read_text(encoding="utf8")
    r["runtime"]["final_audit_seconds"]=time.perf_counter()-start
    r["all_checks_passed"]=True;r["completion_state"]="completed";save(RESULT,r)
    print("T16D engineering audit PASS; scientific gate outcomes retained; formal test sealed")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--record-visual-review",action="store_true")
    if p.parse_args().record_visual_review:visual_review()
    else:finalize()
