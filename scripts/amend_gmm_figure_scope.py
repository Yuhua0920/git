"""Record a plot-only provenance amendment; preserve every T15B numeric value."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
ARCHIVE=OUT/"t15b_final_audit_attempt01"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    meta=load(ARCHIVE/"preservation.json")
    for path,digest in meta["archived_sha256"].items():
        if sha(ARCHIVE/path)!=digest:raise ValueError("failed-attempt archive drift")
    numeric={p:sha(ROOT/p) for p in meta["numeric_sha256"]}
    if numeric!=meta["numeric_sha256"]:raise ValueError("numeric artifact drift")
    old=ast.parse((ARCHIVE/"scripts/plot_gmm_probe.py").read_text(encoding="utf8"))
    new=ast.parse((ROOT/"scripts/plot_gmm_probe.py").read_text(encoding="utf8"))
    old_func=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=="plot")
    new_func=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=="_plot")
    old_func=copy.deepcopy(old_func);new_func=copy.deepcopy(new_func);old_func.name=new_func.name="core"
    if ast.dump(old_func)!=ast.dump(new_func):raise ValueError("plot computation changed")
    original_style=next(n.value.args[0] for n in old.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=="update")
    current_style=next(n.value for n in new.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="STYLE" for t in n.targets))
    if ast.dump(original_style)!=ast.dump(current_style):raise ValueError("style parameters changed")
    figure_hashes={p:sha(ROOT/p)==digest for p,digest in meta["archived_sha256"].items() if p.endswith((".png",".pdf",".svg"))}
    if not all(figure_hashes.values()):raise ValueError("actual T15B figure bytes changed")
    r=load(OUT/"t15b_gmm_results.json")
    original_sources=r["source_sha256"].copy()
    changed=[p for p,h in original_sources.items() if sha(ROOT/p)!=h]
    if changed!=["scripts/plot_gmm_probe.py"]:raise ValueError("only plot scope may change")
    for group in r["gmm_results_by_K"].values():
        for trial in group["runs"]:
            if sha(ROOT/trial["model_path"])!=trial["model_sha256"]:raise ValueError("model drift")
    r["original_run_source_sha256"]=original_sources
    r["source_sha256"]["scripts/plot_gmm_probe.py"]=sha(ROOT/"scripts/plot_gmm_probe.py")
    r["plot_only_provenance_amendment"]=dict(reason=meta["reason"],archive=str(ARCHIVE.relative_to(ROOT)).replace("\\","/"),
        failed_full_pytest=meta["pytest"],original_source_sha256=original_sources["scripts/plot_gmm_probe.py"],
        corrected_source_sha256=r["source_sha256"]["scripts/plot_gmm_probe.py"],plot_core_AST_exact=True,
        style_parameters_AST_exact=True,all_six_export_bytes_unchanged=all(figure_hashes.values()),
        numeric_artifact_sha256=numeric,all_12_model_hashes_unchanged=True,
        CDL_EM_and_numeric_evaluation_repeated=False,amendment_script_sha256=sha(Path(__file__)))
    (OUT/"t15b_gmm_results.json").write_text(json.dumps(r,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print("Plot-only amendment: same plot AST/style, six exports, numeric artifacts and 12 model hashes PASS")


if __name__=="__main__":main()
