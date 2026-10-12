"""Upgrade planning gates: metadata, immutable evidence and no science execution."""
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_sp_u1 import CLASSES, REQUIRED_DOCS, metadata_errors
from build_sp_u1 import FIELDS, WORKS
from check_paper import script_firewall
from sp_u1_plans import REVIEW_HASHES

def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))

def test_literature_scope_and_explicit_unknown_fields():
    assert len(FIELDS)==26
    assert metadata_errors({'fields':FIELDS,'works':WORKS})==[]
    oz=next(w for w in WORKS if w['id']=='oz2006')
    assert oz['source quality']=='ABSTRACT_ONLY'
    assert oz['coefficient/adaptation mechanism']=='UNKNOWN'
    assert 'tracking equations' in oz['uncertainty']

@pytest.mark.parametrize('mutation',['missing_field','duplicate_work','small_set','fake_identifier'])
def test_metadata_gate_rejects_incomplete_or_duplicate_evidence(mutation):
    db={'fields':FIELDS,'works':copy.deepcopy(WORKS)}
    if mutation=='missing_field':del db['works'][0]['pilot setting']
    if mutation=='duplicate_work':db['works'].append(copy.deepcopy(db['works'][0]))
    if mutation=='small_set':db['works']=db['works'][:5]
    if mutation=='fake_identifier':db['works'][0]['DOI/arXiv']='unverified'
    assert metadata_errors(db)

def test_independent_reviews_remain_frozen_and_have_ten_major_concerns_each():
    for path,expected in REVIEW_HASHES.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected
    for i in (1,2,3):
        report=read(f'outputs/sp_u1_reviewer{i}_frozen.json')
        assert len(report['concerns'])>=10
        assert sum(c.get('severity','').lower()=='major' for c in report['concerns'])>=10
        assert all(c['classification'] in CLASSES for c in report['concerns'])

def test_all_planning_deliverables_and_advisor_sections():
    for name in REQUIRED_DOCS:assert (ROOT/f'docs/sp_u1_{name}.md').is_file()
    packet=(ROOT/'docs/sp_u1_advisor_review_packet.md').read_text(encoding='utf-8')
    assert re.findall(r'^## ([A-M])\.',packet,re.M)==list('ABCDEFGHIJKLM')

def test_old_test_remains_reporting_only_and_next_stage_is_not_authorized():
    gate=read('outputs/sp_u1_upgrade_gate.json')
    assert gate['old_t20b_reporting_only'] is True
    assert gate['old_t20b_used_for_new_design'] is False
    assert gate['scientific_execution_authorized'] is False
    assert gate['new_experiments_run'] is False
    assert gate['next_task_started'] is False
    assert gate['baseline_implementation_go'] is False

def test_planning_scripts_have_no_scientific_import_or_execution_calls():
    for path in (ROOT/'scripts').glob('*sp_u1*.py'):
        assert not script_firewall(path.read_text(encoding='utf-8'))
    assert script_firewall('from src.channel import generate\ngenerate()')
    assert script_firewall('model.fit(data)')

def test_recorded_integrity_evidence_is_frozen_and_binds_the_accepted_git_snapshot():
    # The SP-U1 audit checks the working tree during this task. Historical
    # regression must not permanently lock later authorized manuscript/source edits.
    path=ROOT/'outputs/sp_u1_integrity_audit.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='6e6b6892487713073c3f7908f6a7553e4efac10d81e442e4dd018665ea37babf'
    report=read('outputs/sp_u1_integrity_audit.json')
    assert report['checked_tracked_files']==12537
    assert report['unexpected_historical_mutation_count']==0
    assert report['science_and_manuscript_unchanged']
    baseline=read('outputs/sp_u1_before_manifest.json')
    assert baseline['accepted_t20g_commit']==report['accepted_t20g_commit']
    subprocess.check_call(['git','merge-base','--is-ancestor',report['accepted_t20g_commit'],'HEAD'],cwd=ROOT)
    inventory=subprocess.check_output(['git','ls-tree','-r','--name-only','-z',report['accepted_t20g_commit']],cwd=ROOT).decode().split('\0')
    assert set(p for p in inventory if p)==set(baseline['tracked_file_sha256'])

def test_author_code_audit_is_pinned_readonly_and_licenses_are_not_invented():
    records=read('outputs/sp_u1_author_code_access.json')
    assert len(records)==3
    assert all(not r['author_code_executed'] and re.fullmatch('[0-9a-f]{40}',r['commit']) for r in records)
    ammse=next(r for r in records if r['repository']=='TaeJun1999/Attention-aided-MMSE')
    assert ammse['license'] is None

def test_new_plans_keep_theory_conditional_and_geometry_change_explicit():
    theory=(ROOT/'docs/sp_u1_theory_feasibility.md').read_text(encoding='utf-8')
    assert 'rank over the reals' in theory
    assert 'cross term' in theory
    assert 'not the deployed guarantee' in theory
    pilot=(ROOT/'docs/sp_u1_pilot_conditioning_study_plan.md').read_text(encoding='utf-8')
    assert 'newly proposed' in pilot and 'from the actual mask' in pilot
    assert 'common-data-intersection' in pilot
