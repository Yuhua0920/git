"""Paper engineering checks; no formal inference/channel generation is invoked."""
import sys
import hashlib
import subprocess
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paper_support import ROOT, accepted_results, csv_read, json_read, require_sources, sha256
from build_paper_tables import build
from build_paper_figures import capture_marks
from build_paper_preview import bib_entries, flatten
from check_paper import PAPER_SCRIPTS, bibliography_metadata, script_firewall
from paper_history import historical_module, historical_bytes_exact


def historical_paper_bytes(name):
    commit = json_read(ROOT / "outputs/t20e_before_manifest.json")["t20d_commit"]
    return subprocess.check_output(["git", "show", f"{commit}:paper/{name}"], cwd=ROOT)


def test_historical_tables_rebuild_byte_deterministically(tmp_path):
    commit = json_read(ROOT / "outputs/t20e_before_manifest.json")["t20d_commit"]
    historical_build = historical_module(commit, "scripts/build_paper_tables.py").build
    historical_build(output=tmp_path / "one")
    historical_build(output=tmp_path / "two")
    for path in (tmp_path / "one").iterdir():
        assert path.read_bytes() == (tmp_path / "two" / path.name).read_bytes()
        assert path.read_bytes() == historical_paper_bytes("tables/" + path.name)


def test_all_160_metric_cells_retain_raw_precision():
    report = json_read(ROOT / "paper/tables/provenance.json")
    assert len(report["cells"]) == 160
    for metric, file in (("nmse_db", "nmse"), ("ber", "ber")):
        rows = csv_read(ROOT / f"outputs/t20b_formal_{file}.csv")
        expected = {(r["environment"], r["snr_db"], r["method"]): r[metric] for r in rows}
        actual = {(c["environment"], c["snr_db"], c["method"]): c["raw_value"]
                  for c in report["cells"] if c["metric"] == metric}
        assert actual == expected


def test_rendered_geometry_is_exact_to_original_not_just_csv_hash():
    paper = json_read(ROOT / "paper/figures/provenance.json")
    assert capture_marks(paper_style=False) == capture_marks(paper_style=True) == paper["numeric_geometry"]


def test_frozen_source_mutation_is_rejected(tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/t20b_formal_results.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        accepted_results(tmp_path)


@pytest.mark.parametrize("directory,key", [("tables", "generated_table_sha256"), ("figures", "generated_sha256")])
def test_provenance_source_and_product_bytes(directory, key):
    report = json_read(ROOT / f"paper/{directory}/provenance.json")
    require_sources(ROOT, report["source_sha256"])
    commit = json_read(ROOT / "outputs/t20e_before_manifest.json")["t20d_commit"]
    for path, expected in report[key].items():
        checked = historical_bytes_exact(commit, f"paper/{directory}/{path}", expected)
        assert hashlib.sha256(checked).hexdigest() == expected


@pytest.mark.parametrize("path", PAPER_SCRIPTS)
def test_paper_data_firewall(path):
    assert script_firewall((ROOT / "scripts" / path).read_text(encoding="utf8")) == []


def test_firewall_detects_channel_or_refit_calls():
    assert script_firewall("from src.cdl_benchmark import OfficialCDL\nmodel.generate(1)\nmodel.fit(x)")


def test_bibliography_and_version_identity():
    result = bibliography_metadata()
    assert result["metadata_sanity_passed"]
    assert result["entry_count"] == 15
    bib = bib_entries((ROOT / "paper/bibliography/references.bib").read_text(encoding="utf8"))
    assert bib["gui2014preprint"]["author"] != bib["gui2015journal"]["author"]
    assert bib["gui2014preprint"]["title"] != bib["gui2015journal"]["title"]
    assert bib["ha2026mmse"]["eprint"] == "2506.00452v5"
    assert "doi" not in bib["edfors1996report"]


def test_unbalanced_bibliography_fails():
    with pytest.raises(ValueError, match="Unbalanced"):
        bib_entries("@misc{x, title={broken")


def test_historical_scaffold_snapshot_retains_eight_sections_and_math(tmp_path):
    commit = json_read(ROOT / "outputs/t20e_before_manifest.json")["t20d_commit"]
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit, "paper"], cwd=ROOT, text=True).splitlines()
    for name in names:
        if name.endswith(".tex"):
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT))
    text = flatten(tmp_path / "paper/main.tex", tmp_path / "paper")
    assert len(list((ROOT / "paper/sections").glob("*.tex"))) == 8
    for title in ("Introduction", "Related Work", "System Model", "Experimental Setup", "Formal Results", "Discussion", "Conclusion"):
        assert title in text
    assert r"\arg\min_{a,b\in\mathbb R}" in text
    assert "source/development" in text and "C30/200" in text
    assert "within-symbol ICI" in text


def test_built_in_preview_has_no_external_project_inputs():
    text = (ROOT / "paper/preview.tex").read_text(encoding="utf8")
    assert r"\input{" not in text and r"\includegraphics" not in text
    assert r"\begin{thebibliography}" in text
    assert r"Figure placeholder" in text
    assert r"\end{document}" in text
