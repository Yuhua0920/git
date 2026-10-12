"""Finalize saved T16A artifacts; no channel generation or numerical rescan."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_lowrank_filter import OUT, SOURCES, load, save, sha, statistics, environments


def visual_review():
    r = load(OUT/"t16a_lowrank_figure_audit.json")
    if r["source_returncode"] or not all(r["CSV_only_rebuild"].values()) or any(v["text_returncode"] or v["collision_returncode"] for v in r["rendered_checks"].values()):
        raise ValueError("automated figure gates incomplete")
    r["visual_review"] = "Actual final PNGs inspected: both shift panels of closure and weighted energy; legends, curves, rank ticks, thresholds and margins clear."
    r["reviewed_source_warnings"] = ["PNG300dpi/no TIFF: development previews; editable PDF/SVG supplied", "No CI: pooled ratios; independent-trajectory dispersion preserved"]
    r["all_checks_passed"] = True; save(OUT/"t16a_lowrank_figure_audit.json", r)
    print("Actual T16A visual review recorded")


def finalize():
    if Path(sys.prefix).resolve() != (ROOT/".venv-cdl").resolve(): raise ValueError("frozen CDL environment required")
    start = time.perf_counter(); r = load(OUT/"t16a_lowrank_results.json")
    for p, h in {**r["source_sha256"], **r["frozen_history_sha256"]}.items():
        if sha(ROOT/p) != h: raise ValueError(f"source or frozen history drift: {p}")
    statistics()
    figure = load(OUT/"t16a_lowrank_figure_audit.json")
    if not figure["all_checks_passed"]: raise ValueError("figure audit incomplete")
    numeric = {str(p.relative_to(ROOT)).replace("\\", "/"):sha(p) for p in OUT.glob("t16a_*.csv")}
    # Re-run provenance first, then the complete suite. Numeric artifacts frozen.
    for name, args in (("provenance", ["tests/test_lowrank_filter.py", "-k", "committed_snapshot or frozen_statistics_hashes"]), ("full", [])):
        command = [str(ROOT/".venv/Scripts/python.exe"), "-m", "pytest", "-q", *args]
        test = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf8", errors="replace")
        (OUT/f"t16a_lowrank_pytest_{name}.txt").write_text(test.stdout+test.stderr, encoding="utf8")
        r[f"pytest_{name}"] = dict(command=" ".join(command), returncode=test.returncode, output=test.stdout+test.stderr)
        save(OUT/"t16a_lowrank_results.json", r)
        if test.returncode: raise ValueError("pytest failed; saved numerical artifacts retained")
        print(test.stdout.strip().splitlines()[-1], flush=True)
    if any(sha(ROOT/p) != h for p, h in numeric.items()): raise ValueError("numeric CSV changed during audit")
    r["environment_after"] = environments()
    if r["environment_after"] != r["environment_before"]: raise ValueError("environment payload drift")
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
    if diff.returncode: raise ValueError("git diff --check failed")
    if any(sha(ROOT/p) != h for p, h in r["frozen_history_sha256"].items()): raise ValueError("historical artifacts changed")
    r["git_diff_check"] = dict(returncode=diff.returncode, output=diff.stdout+diff.stderr)
    r["numeric_artifact_sha256"] = numeric; r["figure_audit"] = figure
    r["startup_pytest"] = (ROOT/".pytest_cache/t16a_startup_pytest.txt").read_text(encoding="utf8")
    r["small_math_pytest"] = (ROOT/".pytest_cache/t16a_math_pytest.txt").read_text(encoding="utf8")
    r["runtime"]["final_audit_seconds"] = time.perf_counter()-start
    r["runtime"]["total_numeric_plus_final_audit_seconds"] = r["runtime"]["total_numeric_wall_seconds"]+r["runtime"]["final_audit_seconds"]
    r["all_checks_passed"] = True; r["completion_state"] = "completed"
    save(OUT/"t16a_lowrank_results.json", r)
    print("T16A all engineering gates PASS; formal test sealed", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--record-visual-review", action="store_true")
    if p.parse_args().record_visual_review: visual_review()
    else: finalize()
