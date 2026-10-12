"""SP-U9B-1R: provenance, language and timing-scope tests only."""
import csv
import hashlib
import importlib.util
import json
import re
import unicodedata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/sp_u9b1r'
spec=importlib.util.spec_from_file_location('u9b1r',ROOT/'scripts/run_sp_u9b1r.py')
revision=importlib.util.module_from_spec(spec);spec.loader.exec_module(revision)


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def text(name):return (ROOT/name).read_text(encoding='utf-8')
def before(name):return (OUT/'pre_source'/Path(name).relative_to('paper')).read_text(encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_three_files_changed_and_main_abstract_only():
    baseline=read(ROOT/'outputs/sp_u9b1r_preedit_manifest.json')
    actual={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'paper').rglob('*') if p.is_file()}
    assert set(actual)==set(baseline['paper_sources'])
    changed={name for name,digest in baseline['paper_sources'].items() if actual[name]!=digest}
    assert changed==set(revision.ALLOWED)
    assert revision.outer(text('paper/main.tex'))==revision.outer(before('paper/main.tex'))
    assert text('paper/metadata.tex')==before('paper/metadata.tex')
    assert text('paper/bibliography/references.bib')==before('paper/bibliography/references.bib')


def test_all_historical_scientific_and_u9b1_artifacts_preserved():
    baseline=read(ROOT/'outputs/sp_u9b1r_preedit_manifest.json')
    assert len(baseline['frozen_files'])==892
    for name,digest in baseline['frozen_files'].items():assert sha(ROOT/name)==digest,name


def test_numeric_claims_and_citation_sequences_preserved():
    for name in revision.ALLOWED:
        old,new=before(name),text(name)
        if name=='paper/main.tex':old,new=revision.abstract(old),revision.abstract(new)
        assert revision.numeric(old)==revision.numeric(new),name
        assert revision.citations(old)==revision.citations(new),name
    baseline=read(ROOT/'outputs/sp_u9b1r_preedit_manifest.json')
    assert sha(ROOT/baseline['frozen_number_audit'])==baseline['number_audit_sha256']
    with (ROOT/baseline['frozen_number_audit']).open(encoding='utf-8-sig',newline='') as file:rows=list(csv.DictReader(file))
    assert len(rows)==19
    new=revision.abstract(text('paper/main.tex'))
    for row in rows:assert re.search(r'(?<![\d.])'+re.escape(row['display_value'])+r'(?![\d.])',new),row['claim_id']
    assert '0.5-dB compression criterion' in new and '58.87\\%' in new


def test_governance_terms_context_audited_without_touching_later_sections():
    with (ROOT/'outputs/sp_u9b1r_language_audit.csv').open(encoding='utf-8-sig',newline='') as file:rows=list(csv.DictReader(file))
    assert {'term','file','line','context','classification','action'}<=set(rows[0])
    assert {'sealed','formal','preregistered','post-hoc','gate','evidence','validation','causal','online','real-time'}<={r['term'] for r in rows}
    assert {r['classification'] for r in rows}<={'INTERNAL_GOVERNANCE_REMOVE','SCIENTIFIC_NORMAL_KEEP','NOT_PRESENT'}
    assert not [r for r in rows if r['stage']=='POST' and r['scope']=='FRONT_HALF' and r['classification']=='INTERNAL_GOVERNANCE_REMOVE']
    outside_internal=[r for r in rows if r['scope']=='FROZEN_OUTSIDE_SCOPE' and r['classification']=='INTERNAL_GOVERNANCE_REMOVE']
    assert all(r['action']=='DEFER_TO_U9B2_FROZEN_OUTSIDE_FRONT_HALF' for r in outside_internal)
    assert any(r['stage']=='POST' and r['term']=='post-hoc' and r['classification']=='SCIENTIFIC_NORMAL_KEEP' for r in rows)
    assert any(r['term']=='real-time' and r['action']=='KEEP_PRIOR_WORK_TITLE' for r in rows)


def test_problem_source_definition_and_online_scope():
    abstract=revision.abstract(text('paper/main.tex'))
    assert 'When online pilots do not support reliable estimation' in abstract
    assert 'sparse pilots provide insufficient information' not in abstract
    assert 'designated source channel condition (Source)' in abstract
    assert 'online target-condition label' in abstract
    assert 'single independently evaluated unseen-profile condition' in abstract
    assert 'In-distribution degradation, condition-dependent compression effects' in abstract
    assert 'finite matched-statistics reference' in abstract
    intro=text('paper/sections/01_introduction.tex')
    assert intro.index('designated source channel condition (Source)')<intro.index('frozen Source')
    assert 'Online operation here denotes deployment-time adaptation from received pilot observations.' in intro
    assert 'no target clean channel state information (CSI), target covariance or target environment label' in intro
    assert 'requires clean channel statistics from labeled preparation conditions' in intro


def test_frame_causality_current_observations_and_buffering():
    intro=text('paper/sections/01_introduction.tex')
    assert 'five consecutive frames, including the current frame' in intro
    assert 'causal across frames and uses no future frame' in intro
    assert 'current-frame pilots enter the fit' in intro
    assert 'may require within-frame buffering' in intro
    assert 'neither symbol-immediate nor zero-latency estimation is claimed' in intro
    assert 'four-fold cross-pilot prediction objective' in intro
    assert 'cross-validation' not in intro and 'leave-one-out' not in intro and 'held-out fitting' not in intro
    assert 'does not validate full doubly-selective' in intro


def test_first_three_contributions_unchanged_and_fourth_scoped():
    intro=text('paper/sections/01_introduction.tex')
    old=re.findall(r'\\item (.+)',before('paper/sections/01_introduction.tex'))
    new=re.findall(r'\\item (.+)',intro)
    assert len(new)==4 and new[:3]==old[:3]
    assert 'independently evaluated unseen CDL-A profile condition' in new[3]
    assert 'Sensitivity experiments examine pilot-density changes and offline-statistics budgets' in new[3]
    assert 'in-distribution degradation, condition-dependent compression behavior and matched-statistics gaps' in new[3]
    assert not re.search(r'formal|sealed|post-hoc|evidence hierarchy',new[3],re.I)
    assert 'Additional sensitivity experiments examine' in intro
    assert 'without establishing a causal mechanism' in intro


def test_prior_art_and_source_limits_remain_without_process_narrative():
    related=text('paper/sections/02_related_work.tex')
    assert len(re.findall(r'\\noindent\\textbf\{',related))==6
    for key in ['ozdemir2005','ozdemir2006','azpicueta2010ls','gui2014preprint','gui2015journal','koller2022gmm','fesl2022structured','fesl2023mfa','mei2021','ha2026mmse']:
        assert key in related
    assert 'Their 2006 work also addresses comb-type pilots and subspace tracking' in related
    assert 'priority claims for adaptive or low-rank LMMSE estimation itself' in related
    assert 'two versions have identical equations or constraints' in related
    assert 'nonnegative responsibilities summing to one' in related
    assert 'low-rank loadings together with a residual covariance term' in related
    assert 'leave-one-out errors determine a discrete candidate selection' in related
    assert 'arXiv v5 preprint' in related
    for forbidden in ['full method-field audit','available public sources','inaccessible methods','unavailable method fields','fixed reproduction protocol','A faithful project-scale','unverified journal publication status','Unlike']:
        assert forbidden not in related
    assert not re.search(r'\bfirst\b|unprecedented|novel mixing|\bno prior\b|outperforms|superior to',related,re.I)
    requirements=text('docs/sp_u9b1r_discussion_requirements.md')
    assert 'matched-condition performance superiority evidence' in requirements
    assert '不能解释为对方算法性能失败' in requirements
    assert '不作 numerical superiority claim' in requirements
    assert text('paper/sections/07_discussion.tex')==before('paper/sections/07_discussion.tex')


def test_full_compilation_and_existing_warning_classification():
    report=read(ROOT/'outputs/sp_u9b1r_compile_report.json')
    assert report['execution']['exit_code']==0 and report['verdict'] in ['COMPILE_PASS','PASS']
    assert all(not value for value in report['checks'].values())
    assert not report['NEW_warnings'] and report['PREEXISTING_warnings']
    pdf=Path(report['pdf']);assert pdf.read_bytes().startswith(b'%PDF-')
    assert sha(pdf)==report['pdf_sha256']
    qa=read(OUT/'visual_qa.json')
    assert qa['review_status']=='PASS_DIRECT_VISUAL_INSPECTION'
    assert qa['pdf_metadata']['/Title']=='Source-Anchored LMMSE Estimation for MIMO-OFDM with Cross-Pilot Fitting and Low-Rank Corrections'
    assert {1,2,3,qa['references_start']}<={r['page'] for r in qa['reviewed_pages']}
    pdf_text=unicodedata.normalize('NFKC',text('outputs/sp_u9b1r/pdf_text.txt'))
    assert '15.72' in pdf_text and '58.87%' in pdf_text
    assert 'within-frame buffering' in pdf_text


def test_word_budget_and_prior_tracked_diff_preserved():
    edits=read(OUT/'edit_record.json')
    after=edits['after_words'];old=edits['before_words']
    assert 180<=after['abstract']<=250
    assert after['introduction']-old['introduction']<=60
    assert after['related_work']<old['related_work']
    old_tracked=read(ROOT/'outputs/sp_u9b1/final_audit.json')['git']['tracked_changed']
    current=read(OUT/'check_report.json')['tracked_diff_names']
    assert set(current)<=set(old_tracked)|set(revision.ALLOWED)
    assert read(OUT/'check_report.json')['git_diff_check_exit']==0
