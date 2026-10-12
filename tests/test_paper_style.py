"""Small scanner checks before manuscript-wide integration."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_paper_style import scan_text, printed_tex, audit
from audit_manuscript_draft import ROOT, claims, dimensions, latex, sentence_records
from paper_support import json_read, sha256
from render_manuscript_tables import render
from check_paper import script_firewall
from paper_history import prior_paper_commit, historical_bytes
import hashlib


@pytest.mark.parametrize("phrase,category", [
    ("T16A", "internal_id_hits"), ("T20", "internal_id_hits"),
    ("C12", "internal_id_hits"), ("B0", "internal_id_hits"),
    ("Endpoint A PASS", "internal_id_hits"), ("Codex", "meta_ai_hits"),
    ("AI-generated", "meta_ai_hits"), ("using AI", "meta_ai_hits"),
    ("first we", "lab_notebook_style_hits"), ("after several experiments", "lab_notebook_style_hits"),
    ("rank-4 LMMSE estimator", "overclaim_hits"), ("fully blind", "overclaim_hits"),
    ("gate", "forbidden_terminology_hits"), ("formal test", "forbidden_terminology_hits"),
    ("SHA256", "forbidden_terminology_hits"), ("detectors.py", "forbidden_terminology_hits")])
def test_forbidden_printed_language_is_detected(phrase, category):
    assert scan_text(phrase)[category]


def test_nonprinted_comments_and_structural_paths_are_excluded():
    text = "% T20 PASS Codex\n\\input{T20.py}\n\\includegraphics[width=\\linewidth]{T16A.json}\n\\label{C12}"
    assert not any(scan_text(text).values())


def test_legitimate_ai_and_formal_definition_are_not_globally_banned():
    assert not any(scan_text("AI-based channel estimation; formal definition; rank-4 correction operators.").values())


def test_printed_caption_and_escaped_percent_are_retained():
    assert scan_text(r"\caption{T20B PASS}")["internal_id_hits"]
    assert r"95\%" in printed_tex(r"95\% confidence interval % T20B")


def test_unannotated_prose_cannot_pass_claim_audit():
    entries, missing=sentence_records("Unsupported prose.","example.tex")
    assert not entries and missing==["Unsupported prose."]


def test_sentence_audit_excludes_display_math_but_counts_all_prose():
    entries, missing=sentence_records("% audit: DEFINITION | test\nDefine x. Define y.\n\\begin{equation}x=y\\end{equation}","example.tex")
    assert len(entries)==2 and not missing


def test_current_printed_language_firewall(tmp_path):
    assert audit(output=tmp_path/"style.json")["all_lexical_checks_passed"]


def test_current_draft_claim_coverage_and_evidence():
    r=json_read(ROOT/"outputs/t20e_claim_audit.json")
    assert r["all_checks_passed"] and len(r["sentences"])>80
    for sentence in r["sentences"]:
        assert sentence["evidence_file_sha256"]
        for name,digest in sentence["evidence_file_sha256"].items():assert sha256(ROOT/name)==digest


def test_frozen_dimensions_weighted_metric_and_factor_prefixes():
    r=dimensions()
    assert r["all_checks_passed"] and r["checked_frozen_arrays"]==50
    assert max(r["residuals"].values())<1e-11


def test_current_static_latex_labels_and_references():
    assert latex()["static_validation_passed"]


def test_historical_editorial_table_derivatives_preserve_numeric_rows(tmp_path):
    r=render(output=tmp_path)
    assert r["raw_metric_cells_unchanged"] and all(a["numeric_rows"]==80 for a in r["performance_row_audits"].values())
    for name,digest in r["generated_table_sha256"].items():
        assert hashlib.sha256(historical_bytes(prior_paper_commit(), "paper/tables/"+name)).hexdigest()==digest


def test_workflow_source_and_products_are_traceable_and_contain_no_data():
    r=json_read(ROOT/"paper/figures/method_workflow_provenance.json")
    assert sha256(ROOT/"scripts/build_method_workflow.py")==r["source_sha256"]
    assert not r["new_scientific_data_used"] and not r["scientific_code_called"]
    assert r["exact_pilot_fitting_separate_from_compression"]
    for name,digest in r["generated_sha256"].items():assert sha256(ROOT/"paper/figures"/name)==digest
    for label in r["diagram_labels"]:assert not any(scan_text(label).values())


@pytest.mark.parametrize("name",["audit_paper_style.py","audit_manuscript_draft.py","build_method_workflow.py","render_manuscript_tables.py","prepare_manuscript_editorial.py"])
def test_new_editorial_scripts_have_no_scientific_calls(name):
    assert not script_firewall((ROOT/"scripts"/name).read_text(encoding="utf8"))


def test_notation_relabels_do_not_change_real_coordinate_or_window_semantics():
    text=(ROOT/"paper/sections/04_proposed_method.tex").read_text(encoding="utf8")
    assert r"c\in\mathbb R^2" in text and r"\sum_{\ell=j-4}^{j}" in text
    assert "uncompressed" in text and "source means" in text
    assert r"e_{\ell f}" in text and r"r_{\ell f}" not in text
    assert r"Q_{m,4}L_m=V_{m,4}^H" in text


def test_style_guide_remains_at_frozen_digest():
    assert sha256(ROOT/"docs/paper_writing_style_guide.md")==json_read(ROOT/"outputs/t20e_before_manifest.json")["writing_style_guide_sha256"]
