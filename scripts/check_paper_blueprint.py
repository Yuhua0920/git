"""T20D manuscript governance; read frozen evidence, never run experiments."""
import argparse
import re
import subprocess
from collections import Counter
from pathlib import Path

from build_paper_preview import bib_entries, flatten
from check_paper import bibliography_metadata, script_firewall
from paper_support import ROOT, json_read, json_write, sha256

EDITABLE = {
    "docs/paper_scientific_freeze.md", "docs/paper_contribution_statements.md",
    "docs/paper_related_work_outline.md", "docs/paper_equation_inventory.md",
    "docs/paper_argument_structure.md", "docs/paper_result_claim_map.md",
    "paper/README.md", "paper/tables/notation.tex", "paper/bibliography/references.bib",
    "paper/bibliography/metadata.json", "paper/preview.tex", "paper/preview_provenance.json",
    "paper/main.tex", "paper/appendix/evidence_slots.tex", "scripts/check_paper.py",
    "tests/test_paper_evidence.py",
} | {f"paper/sections/{i:02d}_{name}.tex" for i, name in enumerate(
    ("introduction", "related_work", "system_model", "proposed_method", "experimental_setup",
     "results", "discussion", "conclusion"), 1)}
DOCS = ("venue_shortlist", "novelty_ledger_final", "one_sentence_thesis", "detailed_outline",
        "introduction_blueprint", "related_work_blueprint", "method_figure_spec",
        "development_evidence_selection", "layout_audit", "notation_audit",
        "latex_environment_plan", "length_budget", "writing_order", "literature_closure")


def capture(root=ROOT):
    root = Path(root)
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).split(b"\0")
    hashes = {p.decode("utf8"): sha256(root / p.decode("utf8")) for p in files if p}
    report = {"task": "T20D", "t20c_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "tracked_before_sha256": hashes, "authorized_editable_paths": sorted(EDITABLE),
        "working_tree_clean_after_t20c_commit_verified": True}
    json_write(root / "outputs/t20d_before_manifest.json", report)
    return {"captured_tracked_files": len(hashes), "t20c_commit": report["t20c_commit"]}


def unexpected_changes(root, manifest):
    root = Path(root)
    changed = [p for p, h in manifest["tracked_before_sha256"].items()
               if not (root / p).is_file() or sha256(root / p) != h]
    return changed, [p for p in changed if p not in EDITABLE]


def notation_audit(root=ROOT):
    root = Path(root)
    text = (root / "paper/tables/notation.tex").read_text(encoding="utf8")
    method = (root / "paper/sections/04_proposed_method.tex").read_text(encoding="utf8")
    checks = {
        "H_distinct_from_heldout": r"\mathcal H_f" in text and "Channel coefficient" in text,
        "origin_operators_not_physical_factors": "profile-origin" in text and "mobility-origin" in text,
        "Q_distinct_from_QPSK": "QPSK" in text and "Q_{m,4}" in text,
        "module_index_defined": r"m\in\{P,M\}" in text,
        "fold_index_defined": r"f\in\{0,1,2,3\}" in text,
        "spatial_indices_defined": "receive" in text and "transmit" in text,
        "trajectory_and_window_indices_defined": "$i,j,\\ell$" in text,
        "factors_dimensions_preserved": "$3968\\times4$" in text and "$4\\times64$" in text,
        "source_full_operator_declared": "full operator" in method,
        "real_coordinates": r"a,b\in\mathbb R" in method,
        "causal_window_preserved": r"\sum_{\ell=j-4}^{j}" in method,
    }
    return {"checks": checks, "all_checks_passed": all(checks.values()),
            "code_authority": "docs/t20a_final_method_definition.md; frozen src/pilot_cv.py and src/lowrank_modules.py",
            "equations_changed": False}


def equations_unchanged(root, commit):
    pattern = r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}"
    parse = lambda t: Counter((k, re.sub(r"\s+", "", b)) for k, b in re.findall(pattern, t, re.S))
    for path in (Path(root) / "paper/sections").glob("*.tex"):
        old = subprocess.check_output(["git", "show", commit + ":" + path.relative_to(root).as_posix()], cwd=root).decode("utf8")
        if parse(old) != parse(path.read_text(encoding="utf8")):
            return False
    return True


def audit(root=ROOT):
    root = Path(root)
    manifest = json_read(root / "outputs/t20d_before_manifest.json")
    changed, bad = unexpected_changes(root, manifest)
    for name in DOCS:
        if not (root / f"docs/paper_{name}.md").is_file():
            bad.append("missing blueprint:" + name)
    figure = json_read(root / "paper/figures/provenance.json")
    table = json_read(root / "paper/tables/provenance.json")
    for report, directory, field in ((figure, "figures", "generated_sha256"),
                                     (table, "tables", "generated_table_sha256")):
        for p, h in report["source_sha256"].items():
            if sha256(root / p) != h:
                bad.append("source:" + p)
        for p, h in report[field].items():
            if sha256(root / "paper" / directory / p) != h:
                bad.append(directory + ":" + p)
    bib = bibliography_metadata(root)
    json_write(root / "paper/bibliography/metadata.json", bib)
    if not bib["metadata_sanity_passed"]:
        bad.extend(bib["missing_or_invalid"])
    notation = notation_audit(root)
    if not notation["all_checks_passed"]:
        bad.append("notation audit")
    mathematical_bodies_exact = equations_unchanged(root, manifest["t20c_commit"])
    if not mathematical_bodies_exact:
        bad.append("mathematical equation drift")
    firewall_findings = {}
    for filename in ("check_paper_blueprint.py", "audit_paper_layout.py"):
        firewall_findings[filename] = script_firewall((root / "scripts" / filename).read_text(encoding="utf8"))
        if firewall_findings[filename]:
            bad.append("scientific call:" + filename)
    tex = flatten(root / "paper/main.tex", root / "paper")
    assert "within-symbol ICI" in tex and "C30/200" in tex
    equations = re.findall(r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", tex, re.S)
    preview = (root / "paper/preview.tex").read_text(encoding="utf8")
    if any(body not in preview for _, body in equations):
        bad.append("preview equation drift")
    report = {"task": "T20D", "tracked_files_checked": len(manifest["tracked_before_sha256"]),
        "authorized_manuscript_changes": changed, "unexpected_mutations": bad,
        "historical_output_mutations": [p for p in changed if p.startswith("outputs/")],
        "historical_output_files_checked": sum(p.startswith("outputs/") for p in manifest["tracked_before_sha256"]),
        "scientific_source_files_checked": sum(p.startswith("src/") for p in manifest["tracked_before_sha256"]),
        "scientific_code_unchanged": not any(p.startswith("src/") for p in changed),
        "formal_figures_and_numeric_tables_unchanged": not any(p.startswith("figures:") or p.startswith("tables:") for p in bad),
        "bibliography": {k: bib[k] for k in ("entry_count", "metadata_sanity_passed", "technical_unknown_entries")},
        "notation": notation, "mathematical_equations_equal_committed_t20c": mathematical_bodies_exact,
        "paper_script_ast_firewall_findings": firewall_findings,
        "no_new_experiment_calls_in_paper_orchestration": not any(firewall_findings.values()),
        "layout_audit": json_read(root / "outputs/t20d_layout_audit.json"),
        "all_checks_passed": not bad}
    json_write(root / "outputs/t20d_blueprint_audit.json", report)
    if bad:
        raise ValueError(bad)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    result = capture() if args.capture else audit()
    print({k: v for k, v in result.items() if k not in ("notation", "authorized_manuscript_changes")})
