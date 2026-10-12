"""Small reporting calibration and frozen manuscript positioning regression."""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_paper_summary import conservative_bound, products


def test_reporting_upper_bound_contains_unrounded_values():
    assert conservative_bound([0.371, 0.37701, 0.12, 0.21]) == '0.38'
    assert conservative_bound([0.38]) == '0.38'
    assert conservative_bound([0.38001]) == '0.39'


def test_empty_reporting_bound_rejected():
    with pytest.raises(ValueError):
        conservative_bound([])


def test_missing_summary_cell_rejected():
    with pytest.raises(KeyError):
        products([])


def test_duplicate_summary_cell_rejected():
    row = dict(environment='CDL-C-30-100', snr_db='10', method='Source LMMSE', nmse_db='-1')
    with pytest.raises(ValueError):
        products([row, row])


from paper_support import json_read, sha256, csv_read
from audit_paper_positioning import (validate_citation, paragraphs, LITERATURE,
                                     results_edit_audit, summary_audit)
from check_paper import script_firewall


def test_metadata_only_cannot_support_inferred_algorithm():
    base=dict(citation='ozdemir2006', **LITERATURE['ozdemir2006'])
    validate_citation(dict(base,sentence='Ozdemir published work on adaptive low-rank LMMSE.'))
    with pytest.raises(ValueError,match='inferred algorithm'):
        validate_citation(dict(base,sentence='Ozdemir updates the full filter using pilot labels.'))


@pytest.mark.parametrize('claim', ('first','novel','unique','unprecedented','no existing work'))
def test_citation_does_not_clear_priority(claim):
    with pytest.raises(ValueError,match='priority'):
        validate_citation(dict(citation='mei2021',sentence='Our method is '+claim))


def test_uncovered_discussion_prose_is_rejected():
    with pytest.raises(ValueError,match='Unannotated'):
        paragraphs('A performance claim without evidence.')


def test_duplicate_claim_annotation_is_rejected():
    with pytest.raises(ValueError,match='duplicate'):
        paragraphs('% support-id: d01\nOne.\n\n% support-id: d01\nTwo.')


def test_current_table_retains_adverse_results_and_all_conditions():
    report=summary_audit()
    assert report['all_checks_passed'] and report['row_count']==10
    assert {r['environment'] for r in report['cells']}=={
        'CDL-C-30-100','CDL-B-30-100','CDL-C-120-100','CDL-B-120-100','CDL-C-30-200'}
    assert sum(r['comparison']=='source_gain' for r in report['cells'])==10
    assert len(report['single_shift_gaps'])==4


def test_results_changes_are_only_exact_authorized_editorial_edits():
    assert results_edit_audit()['all_checks_passed']


@pytest.mark.parametrize('name', ('discussion_claim','citation_support','related_work_comparison',
                                  'table_numeric','results_edit','integrity'))
def test_current_audit_complete(name):
    assert json_read(ROOT/f'outputs/t20g_{name}_audit.json')['all_checks_passed']


def test_every_discussion_sentence_has_scope_alternative_and_boundary():
    report=json_read(ROOT/'outputs/t20g_discussion_claim_audit.json')
    assert not report['uncovered_prose']
    for record in report['sentences']:
        assert all(record[k] for k in ('evidence_source','scope','alternative_explanation','boundary'))
        assert all(sha256(ROOT/p)==h for p,h in record['evidence_sha256'].items())
    assert any(r['category']=='FUTURE_DIRECTION' for r in report['sentences'])


def test_unresolved_literature_is_not_cleared_or_misclassified():
    report=json_read(ROOT/'outputs/t20g_citation_support_audit.json')
    oz=[r for r in report['sentences'] if r['citation']=='ozdemir2006']
    assert len(oz)==1 and oz[0]['evidence_type']=='METADATA_ONLY'
    assert not oz[0]['verified_from_abstract_only'] and not oz[0]['full_text_available']
    assert not json_read(ROOT/'outputs/t20g_related_work_comparison_audit.json')['priority_claim_cleared']


def test_citation_records_are_current_and_have_real_sources():
    report=json_read(ROOT/'outputs/t20g_citation_support_audit.json')
    assert report['current_draft_sha256']==sha256(ROOT/'paper/sections/02_related_work.tex')
    for record in report['sentences']:
        assert all(record[k] for k in ('sentence','citation','verified_source','support_scope','risk','read_version','locator'))
        assert record['verified_source'].startswith('https://')
        assert all(sha256(ROOT/p)==h for p,h in record['frozen_reading_records_sha256'].items())


def test_reporting_scripts_do_not_enter_scientific_chain():
    for script in ('prepare_paper_positioning.py','build_paper_summary.py','audit_paper_positioning.py'):
        assert not script_firewall((ROOT/'scripts'/script).read_text(encoding='utf8'))


def test_static_and_lexical_checks_pass_without_claiming_compilation():
    assert json_read(ROOT/'outputs/t20g_paper_style_audit.json')['all_lexical_checks_passed']
    report=json_read(ROOT/'outputs/t20g_latex_static_audit.json')
    assert report['static_validation_passed'] and report['compilation_is_not_inferred_from_static_checks']


def test_unrequested_sections_and_historical_outputs_stay_byte_exact():
    before=json_read(ROOT/'outputs/t20g_before_manifest.json')['tracked_before_sha256']
    # Comprehensive output/source hashing is saved by the executed integrity audit.
    report=json_read(ROOT/'outputs/t20g_integrity_audit.json')
    assert not report['unexpected_mutations'] and not report['historical_output_mutations']
    for p,h in before.items():
        if p.startswith(('paper/figures/','paper/tables/','paper/bibliography/')) or p in (
            'paper/sections/01_introduction.tex','paper/sections/03_system_model.tex',
            'paper/sections/04_proposed_method.tex','paper/sections/05_experimental_setup.tex',
            'paper/sections/08_conclusion.tex'):
            assert sha256(ROOT/p)==h
