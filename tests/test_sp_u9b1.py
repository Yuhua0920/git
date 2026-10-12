"""Source/provenance checks for the authorized manuscript front-half upgrade.

These tests never import or execute scientific simulations, result generation,
bootstrap, historical draft rebuilders, or figure regeneration.
"""
import csv
import hashlib
import importlib.util
import json
import re
import unicodedata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/sp_u9b1'
spec=importlib.util.spec_from_file_location('u9b1_finalize',ROOT/'scripts/finalize_sp_u9b1.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def table(path):
    with path.open(encoding='utf-8-sig',newline='') as file:
        return list(csv.DictReader(file))


def text(name):
    return (ROOT/name).read_text(encoding='utf-8')


def test_only_five_authorized_manuscript_files_changed():
    before=read(ROOT/'outputs/sp_u9b1_preedit_manifest.json')
    changed=[]
    for name,digest in before['paper_sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            changed.append(name)
    assert sorted(changed)==sorted(before['allowed'])
    assert audit.outside_abstract(text('paper/main.tex'))==audit.outside_abstract(text('outputs/sp_u9b1/pre_source/main.tex'))
    assert text('paper/metadata.tex').replace('\\newcommand{\\PaperTitle}{'+read(OUT/'visual_qa.json')['pdf_metadata']['/Title']+'}', 'TITLE').startswith('%')
    assert not (ROOT/'paper/references.bib').exists()


def test_frozen_scientific_and_prior_audit_bytes_unchanged():
    before=read(ROOT/'outputs/sp_u9b1_preedit_manifest.json')
    assert len(before['frozen_files'])==772
    for name,digest in before['frozen_files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name


def test_abstract_performance_values_and_one_sided_compression_bounds():
    rows=table(ROOT/'outputs/sp_u9b1_abstract_number_audit.csv')
    assert len(rows)==19 and len({row['claim_id'] for row in rows})==19
    formal=read(ROOT/'outputs/t20b_formal_results.json')
    sealed=read(ROOT/'outputs/sp_u7b_r1/analysis.json')
    abstract=re.search(r'(?s)\\begin\{abstract\}(.*?)\\end\{abstract\}',text('paper/main.tex'))[1]
    for comparison,digits in [('source_gain',2),('convex_gain',2),('compression_penalty',3)]:
        for snr in [10,20]:
            raw=next(row for row in formal['bootstrap_results'] if row['environment']=='CDL-B-120-100' and row['snr_db']==snr and row['comparison']==comparison and row['metric']=='nmse_difference_db')
            reported=next(row for row in rows if row['metric']==comparison and int(row['snr_db'])==snr)
            assert float(reported['raw_value'])==raw['estimate']
            assert reported['display_value']==f"{raw['estimate']:.{digits}f}"
            assert reported['display_value'] in abstract
            if comparison=='compression_penalty':
                assert float(reported['criterion_bound'])==raw['primary_upper']<0.5
    for index,snr in enumerate([10,20]):
        reported=next(row for row in rows if row['claim_id']==f'A_sealed_{snr}')
        assert float(reported['raw_value'])==sealed['results']['gain'][index]
        assert reported['display_value'] in abstract
    assert 'arithmetic proxy' in abstract and '58.87\\%' in abstract
    assert formal['complexity']['saving_vs_dense_affine_percent']==float(next(r for r in rows if r['claim_id']=='A_cost')['raw_value'])
    assert 'runtime' not in abstract and 'single independently evaluated unseen-profile condition' in abstract
    assert 'In-distribution degradation, condition-dependent compression' in abstract


def test_bibliography_fields_match_source_ledger():
    records=audit.bib_records(text('paper/bibliography/references.bib'))
    old=audit.bib_records(text('outputs/sp_u9b1/pre_source/bibliography/references.bib'))
    rows=read(OUT/'bibliography_audit.json')
    ledger={r['id']:r for r in read(ROOT/'outputs/sp_u9a_r1_source_ledger.json')['sources']}
    assert len(rows)==10
    changed_keys={r['key'] for r in rows}
    assert set(records)-set(old)=={'koller2022gmm','fesl2022structured','fesl2023mfa','ozdemir2005','azpicueta2010ls'}
    for key in set(old)-changed_keys:
        assert records[key]==old[key]
    for row in rows:
        fields=records[row['key']]['fields']
        source=ledger[row['source_id']]
        assert fields['title'].replace('{','').replace('}','')==source['title']
        assert fields['year']==str(source['year'])
        expected=' and '.join(unicodedata.normalize('NFKD',author).encode('ascii','ignore').decode() for author in source['authors'])
        assert fields['author']==expected
        assert fields.get('doi')==source['doi'] or (not fields.get('doi') and not source['doi'])
        assert fields['url']==source['url']
        if 'pages' in fields:
            assert fields['pages'].replace('--','-')==source['crossref']['page']
            crossref_id='P7' if row['source_id']=='Mei' else row['source_id']
            metadata=read(ROOT/f'outputs/sp_u9a_r1/crossref_{crossref_id}.json')['message']
            if 'journal' in fields:
                assert fields['journal']==metadata['container-title'][0]
                assert fields.get('volume')==metadata.get('volume')
                assert fields.get('number')==metadata.get('issue')
            elif row['source_id']=='P5':
                # Frozen primary metadata misspells Micrwave and omits "on".
                assert fields['booktitle']=='The 2005 IEEE Annual Conference on Wireless and Microwave Technology'
                assert 'Micrwave' in metadata['container-title'][0]
            else:
                assert fields['booktitle']==metadata['container-title'][0]
        assert fields==row['fields']
    assert records['ha2026mmse']['type']=='misc'
    assert records['ha2026mmse']['fields']['eprint']=='2506.00452v5'
    assert records['gui2014preprint']['fields']['eprint']=='1407.6083v1'


def test_all_active_citations_resolve_and_nearest_works_appear():
    active,paths=audit.active_inputs()
    assert 'paper/preview.tex' not in paths
    records=audit.bib_records(text('paper/bibliography/references.bib'))
    keys={key for group in re.findall(r'\\cite\{([^}]+)\}',active) for key in group.split(',')}
    assert keys<=set(records)
    assert {row['key'] for row in read(OUT/'bibliography_audit.json')}<=keys
    labels=re.findall(r'\\label\{([^}]+)\}',active)
    assert len(labels)==len(set(labels))


def test_source_limited_fields_are_not_inferred():
    related=text('paper/sections/02_related_work.tex')
    assert 'available public sources did not permit a full method-field audit' in related
    assert 'without inferring a contrasting object' in related
    assert 'Unlike' not in related
    assert 'two versions have identical equations or constraints' in related
    assert 'stochastic gradient' not in related.lower()
    assert 'arXiv v5' in related
    assert 'unverified journal publication status' in related
    assert 'ET2026' not in related


def test_information_risk_and_no_ici_contract():
    intro=text('paper/sections/01_introduction.tex')
    assert 'requires clean channel statistics from labeled preparation conditions' in intro
    assert 'no target clean channel state information (CSI), target covariance or target environment label' in intro
    assert 'additive base' in intro and 'does not invoke a Bayesian' in intro
    assert 'current frame and excluding future frames' in intro
    assert 'pilot prediction operators used for coordinate fitting are retained' in intro
    assert 'prediction columns themselves depend on received observations' in intro
    assert 'A coordinate that minimizes the former need not minimize the latter' in intro
    assert 'symbol-wise quasi-static abstraction' in intro
    assert 'does not validate full doubly-selective' in intro
    assert 'Post-hoc receiver diagnostics' in intro
    assert 'they are not additional formal replications' in intro


def test_no_component_priority_or_unsupported_external_superiority():
    active,_=audit.active_inputs()
    assert not re.search(r'\bfirst\b|unprecedented|novel mixing|\bno prior\b|outperforms (?:GMM|MFA|A-MMSE)|near[- ]oracle|near[- ]optimal|state[- ]of[- ]the[- ]art',active,re.I)
    contributions=re.findall(r'\\item (.+)',text('paper/sections/01_introduction.tex'))
    assert len(contributions)==4
    assert 'For this prepared correction space' in contributions[1]
    assert 'retain dense pilot predictors' in contributions[2]
    assert 'prepared from labeled channel-statistics conditions' in contributions[0]
    assert 'make no performance claim relative to those methods' in text('paper/sections/02_related_work.tex')


def test_rw_method_sentences_have_specific_citations():
    rows=table(ROOT/'outputs/sp_u9b1_related_work_collision_audit.csv')
    assert len(rows)==42 and {r['paragraph'] for r in rows}=={'RW1','RW2','RW3','RW4','RW5','RW6'}
    for row in rows:
        sentence=row['sentence']
        # Named external method assertions require their own citation, not a paragraph-end blanket citation.
        specific=(sentence.startswith(('Classical LMMSE','Adaptive low-rank','Azpicueta-Ruiz','Their reference-output','The separate author','GMM estimators','Fesl','These works already','Their component means','Mei et al.','In their estimator','This establishes','The inspected arXiv','Channelformer')))
        if specific:
            assert row['citations'],row['sentence_id']
        assert row['allowed']=='YES' and row['source_limit_preserved']=='YES'
    intro=table(ROOT/'outputs/sp_u9b1_intro_claim_audit.csv')
    assert len(intro)==51 and all(r['evidence_source'] and r['allowed']=='YES' for r in intro)


def test_full_compilation_actual_pdf_and_warning_classification():
    report=read(ROOT/'outputs/sp_u9b1_compile_report.json')
    assert report['verdict']=='PASS' and report['full_manuscript_build']
    assert report['post']['exit_code']==0
    for stage in ['pre','post']:
        for key in ['latex_errors','undefined_citations','undefined_references','overfull_boxes','duplicate_labels','bibtex_errors']:
            assert not report[stage][key]
    assert not report['warning_classification']['NEW']
    assert report['warning_classification']['PREEXISTING']
    pdf=Path(report['post']['pdf'])
    assert pdf.read_bytes().startswith(b'%PDF-') and len(pdf.read_bytes())>300000
    assert hashlib.sha256(pdf.read_bytes()).hexdigest()==report['post']['pdf_sha256']


def test_visual_receipt_and_final_abstract_not_truncated_after_percent():
    qa=read(OUT/'visual_qa.json')
    assert qa['review_status']=='PASS' and qa['pages_total']==19
    assert [r['page'] for r in qa['reviewed_pages']]==[1,2,3,4,15,16]
    assert qa['pdf_metadata']['/Title']=='Source-Anchored LMMSE Estimation for MIMO-OFDM with Cross-Pilot Fitting and Low-Rank Corrections'
    pdf_text=unicodedata.normalize('NFKC',text('outputs/sp_u9b1/pdf_text.txt'))
    assert '15.72' in pdf_text and '58.87%' in pdf_text
    assert 'In-distribution degradation' in pdf_text
    assert 'TODO: Authors' in pdf_text and 'TODO: Affiliations' in pdf_text


def test_tracked_diff_stays_in_manuscript_scope_and_word_budget():
    git=read(OUT/'git_audit.json')
    assert set(git['tracked_changed'])<=set(read(ROOT/'outputs/sp_u9b1_preedit_manifest.json')['allowed'])
    assert git['diff_check_exit_code']==0 and not git['commit_performed']
    words=read(OUT/'editorial_report.json')['word_counts']
    assert 180<=words['abstract']<=250 and 900<=words['introduction']<=1300 and 900<=words['related_work']<=1400
