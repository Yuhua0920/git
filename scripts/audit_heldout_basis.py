"""Finalize T16C saved artifacts; no channel generation or basis construction."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_heldout_basis import OUT,RESULT,load,save,sha,snapshot,environments,environment
from src.heldout_basis import load_bases


def visual_review():
    r=load(OUT/"t16c_heldout_figure_audit.json")
    if r["source_returncode"] or not all(r["CSV_only_rebuild"].values()) or any(v["text_returncode"] or v["collision_returncode"] for v in r["rendered_checks"].values()):
        raise ValueError("figure automated gates incomplete")
    r["visual_review"]="Actual final PNGs inspected panel-by-panel: heldout shared curves and oracle-versus-shared bars; labels, legends, ranks, reference lines, directions and margins clear."
    r["reviewed_source_warnings"]=["PNG300dpi/no TIFF: development preview with editable PDF/SVG","No pooled CI; trajectory dispersion saved"]
    r["all_checks_passed"]=True;save(OUT/"t16c_heldout_figure_audit.json",r);print("Actual T16C visual review recorded")


def finalize():
    environment();start=time.perf_counter();r=load(RESULT)
    for p,h in {**r["source_sha256"],**r["frozen_history_sha256"]}.items():
        if sha(ROOT/p)!=h:raise ValueError(f"source/history changed: {p}")
    snapshot();load_bases(ROOT)
    figure=load(OUT/"t16c_heldout_figure_audit.json")
    if not figure["all_checks_passed"]:raise ValueError("figure review incomplete")
    numeric={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in OUT.glob("t16c_*.csv")}
    for name,args in (("provenance",["tests/test_heldout_basis.py","-k","snapshot or frozen or actual_results"]),("full",[])):
        cmd=[str(ROOT/".venv/Scripts/python.exe"),"-m","pytest","-q",*args]
        t=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding="utf8",errors="replace");output=t.stdout+t.stderr
        (OUT/f"t16c_heldout_pytest_{name}.txt").write_text(output,encoding="utf8")
        r[f"pytest_{name}"]=dict(command=" ".join(cmd),returncode=t.returncode,output=output);save(RESULT,r)
        if t.returncode:raise ValueError("pytest failed; retained numerical results")
        print(output.strip().splitlines()[-1],flush=True)
    if any(sha(ROOT/p)!=h for p,h in numeric.items()):raise ValueError("numeric CSV changed during audit")
    r["environment_after"]=environments()
    if r["environment_after"]!=r["environment_before"]:raise ValueError("environment payload drift")
    if any(sha(ROOT/p)!=h for p,h in r["frozen_history_sha256"].items()):raise ValueError("historical files changed")
    load_bases(ROOT)
    for info in r["target_statistics_provenance"].values():
        if sha(ROOT/info["path"])!=info["sha256"]:raise ValueError("new train-only statistics changed")
    diff=subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise ValueError("diff check failed")
    r["git_diff_check"]=dict(returncode=diff.returncode,output=diff.stdout+diff.stderr)
    r["numeric_artifact_sha256"]=numeric;r["figure_audit"]=figure
    r["startup_pytest"]=(ROOT/".pytest_cache/t16c_startup_pytest.txt").read_text(encoding="utf8")
    r["small_pytest"]=(ROOT/".pytest_cache/t16c_small_pytest.txt").read_text(encoding="utf8")
    r["runtime"]["final_audit_seconds"]=time.perf_counter()-start
    r["all_checks_passed"]=True;r["completion_state"]="completed";save(RESULT,r)
    print("T16C engineering gates PASS; scientific outcomes retained; formal test sealed",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--record-visual-review",action="store_true")
    if p.parse_args().record_visual_review:visual_review()
    else:finalize()
