"""Audit table/figure/citation provenance and the paper-only data firewall."""
import ast
from pathlib import Path
import re

from build_paper_preview import bib_entries
from paper_support import ROOT, csv_read, json_read, json_write, require_sources, sha256

PAPER_SCRIPTS = ("build_paper_tables.py", "build_paper_figures.py", "build_paper_preview.py",
                 "freeze_paper_evidence.py", "paper_support.py", "check_paper.py")
REQUIRED_DOCS = ("scientific_freeze", "argument_structure", "contribution_statements", "result_claim_map",
                 "equation_inventory", "related_work_outline", "abstract_skeleton", "reproducibility_map", "data_firewall")
FORBIDDEN_CALLS = {"generate", "deployed", "bootstrap", "svd", "fit", "train", "evaluate"}


def script_firewall(text):
    tree = ast.parse(text)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            bad += [name for name in names if name == "src" or name.startswith("src.")]
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                node.func.id if isinstance(node.func, ast.Name) else "")
            if name in FORBIDDEN_CALLS:
                bad.append(name)
    return bad


def bibliography_metadata(root=ROOT):
    paper = Path(root) / "paper"
    entries = bib_entries((paper / "bibliography/references.bib").read_text(encoding="utf8"))
    closure_path = paper / "bibliography/technical_closure.json"
    closure = json_read(closure_path)["entries"] if closure_path.is_file() else {}
    missing, warnings, records = [], [], []
    for key, entry in entries.items():
        venue = next((entry[k] for k in ("journal", "booktitle", "howpublished", "institution") if k in entry), None)
        for field in ("author", "title", "year", "verified_source"):
            if not entry.get(field):
                missing.append(f"{key}:{field}")
        if not venue or not re.fullmatch(r"\d{4}", entry.get("year", "")):
            missing.append(f"{key}:venue/year")
        identifier = entry.get("doi") or entry.get("eprint")
        exception = False
        if not identifier:
            if key == "edfors1996report" and entry.get("identifier") == "urn:nbn:se:ltu:diva-24711":
                identifier = entry["identifier"]
                exception = True
                warnings.append("Edfors 1996 technical report: approved verified URN; no DOI/arXiv verified in the sources checked")
            elif (key == "3gpp38901" and entry.get("identifier") == "3GPP TR 38.901 V19.2.0"
                  and entry.get("verified_source") == "https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/19.02.00_60/tr_138901v190200p.pdf"):
                identifier = entry["identifier"]
                exception = True
                warnings.append("3GPP channel report: verified report number and official permanent PDF; no DOI/arXiv verified in the sources checked")
            else:
                missing.append(f"{key}:doi/arxiv")
        fulltext = "UNKNOWN" if key in ("ozdemir2006", "lee2026") else "T20A inspected source/sections only"
        fulltext = closure.get(key, {}).get("technical_fulltext_status", fulltext)
        if key == "3gpp38901":
            fulltext = "Official report: CDL and delay-scaling sections verified"
        records.append({"key": key, "author": entry.get("author"), "title": entry.get("title"),
                        "venue": venue, "year": entry.get("year"), "identifier": identifier,
                        "identifier_type": entry.get("identifier_type", "URN") if exception else ("DOI" if entry.get("doi") else "arXiv"),
                        "doi": entry.get("doi", "N/A"), "arxiv": entry.get("eprint", "N/A"),
                        "identifier_status": "no DOI/arXiv verified in the sources checked" if exception else "verified metadata",
                        "verified_source": entry.get("verified_source"),
                        "stable_urn_exception": exception and entry.get("identifier_type", "URN") == "URN",
                        "persistent_identifier_exception": exception,
                        "technical_fulltext_status": fulltext,
                        "metadata_only_supplement": key not in closure and key != "3gpp38901",
                        "technical_closure": closure.get(key)})
    # Registry records independently guard the title/author/DOI mismatch found in the old ledger.
    crossref = {r["record"]["DOI"].lower(): r["record"]
                for r in json_read(paper / "bibliography/doi_register_records.json")}
    normal = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    for key in ("mei2021", "gui2015journal", "ozdemir2006", "wang2025lorat"):
        e = entries[key]
        source = crossref[e["doi"].lower()]
        if normal(e["title"]) != normal(source["title"][0]):
            missing.append(key + ":DOI registered title mismatch")
        if len(e["author"].split(" and ")) != len(source["author"]):
            missing.append(key + ":DOI registered author count mismatch")
    all_tex = "\n".join(p.read_text(encoding="utf8") for p in (paper / "sections").glob("*.tex"))
    keys = {k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", all_tex) for k in group.split(",")}
    missing += ["undefined citation:" + key for key in sorted(keys - entries.keys())]
    return {"entry_count": len(entries), "entries": records, "warnings": warnings,
            "missing_or_invalid": missing, "metadata_sanity_passed": not missing,
            "doi_arxiv_entry_count": sum(bool(e.get("doi") or e.get("eprint")) for e in entries.values()),
            "verified_urn_entry_count": 1,
            "technical_unknown_entries": [r["key"] for r in records if
                r["technical_fulltext_status"] == "UNKNOWN" or
                closure.get(r["key"], {}).get("technical_unresolved", False)]}


def check(root=ROOT):
    root = Path(root)
    tables = json_read(root / "paper/tables/provenance.json")
    figures = json_read(root / "paper/figures/provenance.json")
    bad = []
    for report, directory, hash_key in ((tables, "tables", "generated_table_sha256"),
                                         (figures, "figures", "generated_sha256")):
        require_sources(root, report["source_sha256"].keys())
        for path, expected in report[hash_key].items():
            if sha256(root / "paper" / directory / path) != expected:
                bad.append(directory + "/" + path)
    raw_metric = {}
    for metric, path in (("nmse_db", "outputs/t20b_formal_nmse.csv"), ("ber", "outputs/t20b_formal_ber.csv")):
        for row in csv_read(root / path):
            raw_metric[(row["environment"], row["snr_db"], row["method"], metric)] = row[metric]
    intervals = {(r["environment"], r["snr_db"], r["comparison"], r["metric"]): r
                 for r in csv_read(root / "outputs/t20b_formal_bootstrap.csv")}
    seen = set()
    for cell in tables["cells"]:
        key = (cell["environment"], cell["snr_db"], cell["method"], cell["metric"])
        if key in seen or cell["raw_value"] != raw_metric[key]:
            bad.append("table cell mismatch:" + str(key))
        for field in ("ci95_lower", "ci95_upper"):
            if cell["raw_" + field] != intervals[key][field]:
                bad.append("table interval mismatch:" + str(key))
        seen.add(key)
    if seen != raw_metric.keys():
        bad.append("table coverage mismatch")
    for path in PAPER_SCRIPTS:
        findings = script_firewall((root / "scripts" / path).read_text(encoding="utf8"))
        bad += [path + ":" + finding for finding in findings]
    for doc in REQUIRED_DOCS:
        if not (root / f"docs/paper_{doc}.md").is_file():
            bad.append("missing doc:" + doc)
    bib = bibliography_metadata(root)
    json_write(root / "paper/bibliography/metadata.json", bib)
    report = {"task": "T20C", "table_cells_exact_to_frozen_csv": not bad,
              "table_cell_count": len(seen), "primary_endpoint_cells": tables["primary_cells"],
              "figure_geometry_exact_to_original": figures["original_figure_data_geometry_exact"],
              "figure_source_and_export_hashes_exact": not bad,
              "bibtex_metadata_sanity_passed": bib["metadata_sanity_passed"],
              "bibliography_identifier_exception": bib["warnings"],
              "paper_scripts_no_scientific_calls": not any(":" in p and ".py" in p for p in bad),
              "historical_scientific_data_recomputed": False, "problems": bad + bib["missing_or_invalid"],
              "all_checks_passed": not bad and bib["metadata_sanity_passed"]}
    json_write(root / "outputs/t20c_paper_validation.json", report)
    if not report["all_checks_passed"]:
        raise ValueError(report["problems"])
    return report


if __name__ == "__main__":
    print(check())
