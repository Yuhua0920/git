"""T20F reporting calibration, followed by frozen-evidence integration checks."""
import sys
import hashlib
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_paper_tables import compact_table, index_unique
from build_paper_tables import build
from build_paper_figures import style_metric_axes, geometry, ENCODING, clean_svg_export
from paper_support import ROOT, json_read, sha256
from audit_paper_results import resolve_macro, comparison_support
from audit_manuscript_draft import dimensions, latex
import matplotlib.pyplot as plt
import paper_history


def test_duplicate_numeric_source_is_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        index_unique([{"k": "a"}, {"k": "a"}], ("k",))


@pytest.mark.parametrize('case', ('blob', 'checkout', 'mismatch'))
def test_historical_provenance_requires_recorded_exact_bytes(monkeypatch, case):
    raw = b'{"value": 1}\n'
    checkout = b'{"value": 1}\r\n'
    calls = []
    def git_read(args, **kwargs):
        calls.append(args)
        return raw if args[1] == 'show' else checkout
    monkeypatch.setattr(paper_history.subprocess, 'check_output', git_read)
    expected = hashlib.sha256(raw if case == 'blob' else checkout).hexdigest()
    if case == 'mismatch':
        with pytest.raises(ValueError, match='Historical SHA256 mismatch'):
            paper_history.historical_bytes_exact('snapshot', 'paper/file.json', '0' * 64)
    else:
        checked, evidence = paper_history.historical_hash_evidence('snapshot', 'paper/file.json', expected)
        assert hashlib.sha256(checked).hexdigest() == expected
        assert evidence['exact_match'] and evidence['expected_sha256'] == expected
    assert len(calls) == (1 if case == 'blob' else 2)


def test_compact_table_preserves_unfavorable_values_and_rounding():
    rows = [{"environment": "test", "snr_db": "10", "method": m, "nmse_db": v}
            for m, v in (("Source LMMSE", "-2.3456"), ("Final LR4 Affine", "-2.1001"))]
    text, cells = compact_table(rows, "nmse_db", conditions=("test",),
                               snrs=(10,), methods=("Source LMMSE", "Final LR4 Affine"))
    assert "-2.346" in text and "-2.100" in text
    assert [c["raw_value"] for c in cells] == ["-2.3456", "-2.1001"]


def test_compact_table_missing_cell_raises():
    with pytest.raises(KeyError):
        compact_table([], "ber", conditions=("test",), snrs=(10,), methods=("Convex",))


def test_encoding_revision_preserves_all_numeric_marks():
    fig, axes = plt.subplots(1, 5)
    for ax in axes:
        for i, label in enumerate(ENCODING):
            ax.plot([10, 20], [i, i + 1], label=label)
            ax.vlines([10, 20], [i - .1, i + .9], [i + .1, i + 1.1])
    before = geometry(fig)
    style_metric_axes(fig)
    assert geometry(fig) == before
    assert len(set(line.get_marker() for line in axes[0].lines)) == 8
    plt.close(fig)


def test_svg_export_cleanup_preserves_draw_coordinates(tmp_path):
    path=tmp_path/'figure.svg'
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg"><path d="M 1 2 \nL 3 4 \n"/>\n</svg>\n')
    r=clean_svg_export(path)
    assert r['svg_drawing_tokens_exact'] and r['meaningful_text_exact']
    assert r['removed_trailing_whitespace_lines']==2
    assert all(line==line.rstrip(' \t') for line in path.read_text().splitlines())


def test_svg_export_cleanup_rejects_meaningful_text_mutation(tmp_path):
    path=tmp_path/'figure.svg'
    original='<svg xmlns="http://www.w3.org/2000/svg"><text>A \nB</text></svg>\n'
    path.write_text(original)
    with pytest.raises(ValueError,match='changed drawing or text semantics'):
        clean_svg_export(path)
    assert path.read_text()==original


def test_current_tables_rebuild_deterministically_without_overwriting_history(tmp_path):
    protected=('paper/tables/provenance.json','paper/tables/t20e_provenance.json')
    before={p:sha256(ROOT/p) for p in protected}
    one,two=tmp_path/'one',tmp_path/'two'
    r=build(output=one);build(output=two)
    for path in one.iterdir():
        assert path.read_bytes()==(two/path.name).read_bytes()
        assert path.read_bytes()==(ROOT/'paper/tables'/path.name).read_bytes()
    assert before=={p:sha256(ROOT/p) for p in protected}
    assert len(r['full_cells'])==160 and len(r['compact_cells'])==100


def test_bootstrap_metric_namespace_is_explicit():
    records=json_read(ROOT/'paper/tables/t20f_provenance.json')['number_macros']
    record=records['AffineGainComposedTwenty']
    assert record['selector']['metric']=='nmse_difference_db'
    raw,display=resolve_macro(record)
    assert raw==record['raw_value'] and display==record['display_value']


def test_every_published_number_macro_reconstructs_from_its_frozen_field():
    for record in json_read(ROOT/'paper/tables/t20f_provenance.json')['number_macros'].values():
        assert resolve_macro(record)==(record['raw_value'],record['display_value'])


def test_unfavorable_id_and_delay_boundary_are_retained():
    checks=comparison_support()
    assert checks['id_loss_retained']
    assert checks['boundary_higher_snr_compression_exceeds_margin']
    assert checks['boundary_lower_snr_no_compression_penalty']


@pytest.mark.parametrize('name', ('table_numeric','figure_numeric','claim','result_sentence','paper_style','notation_dimension','latex_static'))
def test_current_reporting_audits_are_complete(name):
    r=json_read(ROOT/f'outputs/t20f_{name}_audit.json')
    assert r.get('all_checks_passed',r.get('all_lexical_checks_passed',r.get('static_validation_passed')))


def test_sentence_records_have_specific_conditions_and_support():
    r=json_read(ROOT/'outputs/t20f_result_sentence_audit.json')
    assert r['sentences'] and not r['uncovered_prose']
    for sentence in r['sentences']:
        assert sentence['conditions'] and sentence['source_artifacts'] and sentence['metric']
        assert sentence['support_status'].startswith('SUPPORTED')


def test_current_math_and_latex_checks_do_not_rewrite_historical_reports():
    protected=('outputs/t20e_latex_static_audit.json','outputs/t20e_equation_dimension_audit.json')
    before={p:sha256(ROOT/p) for p in protected}
    assert dimensions()['all_checks_passed'] and latex()['static_validation_passed']
    assert before=={p:sha256(ROOT/p) for p in protected}


def test_final_size_figure_alignment_collision_and_glyph_checks():
    r=json_read(ROOT/'outputs/t20f_figure_layout_audit.json')
    assert r['all_automatic_checks_passed']
    for figure in r['figures'].values():
        assert figure['min_glyph_pt_at_174mm'] >= 6
        assert figure['alignment_report']['verdict']=='PASS'
        assert figure['collision_report']['summary']['fail']==figure['collision_report']['summary']['warn']==0


def test_t20f_historical_scope_and_current_all_method_tables():
    from paper_history import historical_bytes_exact
    commit=json_read(ROOT/'outputs/t20g_before_manifest.json')['t20f_commit']
    manifest=json_read(ROOT/'outputs/t20f_before_manifest.json')['tracked_before_sha256']
    for path in manifest:
        if path.startswith('paper/sections/') and path not in (
                'paper/sections/05_experimental_setup.tex','paper/sections/06_results.tex'):
            historical_bytes_exact(commit,path,manifest[path])
    for file in ('02_nmse.tex','03_ber.tex'):
        text=(ROOT/'paper/tables'/file).read_text(encoding='utf8')
        assert 'Perfect' in text and 'LS-LIN' in text and 'Hard' in text
