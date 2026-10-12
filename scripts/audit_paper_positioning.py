"""T20G reporting audits: frozen evidence, reviewed prose and bounded comparisons."""
import argparse
import re
import subprocess
from pathlib import Path

from paper_support import ROOT, csv_read, json_read, json_write, sha256
from build_paper_summary import products
from audit_paper_style import audit as style, printed_tex
from audit_manuscript_draft import latex
from build_paper_preview import bib_entries
from check_paper import script_firewall
from paper_history import historical_hash_evidence

DRAFTS = ('paper/sections/07_discussion.tex', 'paper/sections/02_related_work.tex')
ALLOWED = {*DRAFTS, 'paper/sections/06_results.tex', 'paper/main.tex', 'paper/preview.tex',
           'paper/preview_provenance.json', 'tests/test_paper_results.py'}

# Each paragraph's classification is editorial metadata, not experimental data.
DISCUSSION = {
 'd01': ('CLAIM', ['outputs/t20b_formal_nmse.csv', 'outputs/t20b_formal_bootstrap.csv', 'paper/sections/04_proposed_method.tex'],
         'Paired composed condition at 10/20 dB; ID loss retained',
         'The comparison also depends on the prepared library and prediction objective',
         'No claim for arbitrary filter libraries or source-matched improvement'),
 'd02': ('CLAIM', ['outputs/t20b_formal_weight_diagnostics.json', 'paper/sections/04_proposed_method.tex'],
         'Descriptive composed-coordinate geometry in 6000 scored windows at each SNR',
         'Coordinates reflect the pilot objective and noise as well as distribution changes',
         'No covariance-hull or causal explanation'),
 'd03': ('METHOD_DESCRIPTION', ['paper/sections/04_proposed_method.tex', 'outputs/t20b_formal_bootstrap.csv', 'outputs/t20a_results.json'],
         'Correction-only weighted rank4 with full source and exact fitting',
         'Operator approximation is not channel rank', 'Fidelity conclusion limited to composed condition'),
 'd04': ('CLAIM', ['outputs/t17a_results.json', 'outputs/t17b_results.json', 'docs/t17b_claim_evidence_update.md'],
         'Frozen finite B120 shift, 10/20 dB and predefined decompositions',
         'Large cancelling terms can coexist with accurate affine approximation',
         'Rejects the tested explanation, not all perturbation theory'),
 'd05': ('METHOD_DESCRIPTION', ['paper/sections/04_proposed_method.tex', 'docs/paper_scientific_freeze.md'],
         'Current offline/online interface', 'Library quality affects deployment',
         'Clean offline statistics required; no target-statistics input online'),
 'd06': ('ASSUMPTION', ['docs/paper_scientific_freeze.md'],
         'Conditional possible acquisition routes, not evaluated deployment evidence',
         'Measurement noise and nonrepresentative calibration may limit the library',
         'No acquisition feasibility or performance claim'),
 'd07': ('CLAIM', ['paper/sections/05_experimental_setup.tex', 'outputs/t20b_formal_results.json'],
         'Independent realizations of prepared conditions, held-out composition and boundary',
         'Prepared-condition reuse is not transfer to a new family',
         '2x2 CDL-B/C and fixed geometry only'),
 'd08': ('CLAIM', ['outputs/t20b_formal_nmse.csv', 'outputs/t20b_formal_bootstrap.csv', 'paper/tables/t20f_provenance.json'],
         'C30/200 nominal delay boundary at 10/20 dB',
         'Matched gap includes operator and mean differences',
         'Does not imply universal delay failure or isolate operator deficiency from formal gap alone'),
 'd09': ('CLAIM', ['outputs/t19c_results.json', 'outputs/t19d_results.json', 'docs/t19d_claim_evidence_update.md'],
         'Development operator analysis: replicated C30/200/10 deficiency; predefined heterogeneous held-out contexts',
         'Residual directions may require context-dependent representation',
         'No deployed added-module NMSE or claim for all delay shifts'),
 'd10': ('ASSUMPTION', ['paper/sections/03_system_model.tex', 'paper/sections/05_experimental_setup.tex'],
         'Symbolwise ideal synchronization, no ICI and fixed source ensemble SNR',
         'Waveform impairments may change the comparison', 'No full high-mobility ICI conclusion'),
 'd11': ('CLAIM', ['outputs/t20b_formal_results.json', 'outputs/t20b_formal_nmse.csv', 'outputs/t20b_formal_ber.csv'],
         'Specified arithmetic/storage proxy and uncoded ZF results',
         'Runtime includes implementation and memory costs; detection gain need not scale with NMSE',
         'No measured timing/energy benefit or new detection mechanism'),
 'd12': ('FUTURE_DIRECTION', ['docs/paper_scientific_freeze.md', 'outputs/t19d_results.json'],
         'Conditional questions derived from current limitations',
         'An extension may not transfer or may worsen acquisition requirements',
         'No new experiment or module implemented'),
 'd13': ('FUTURE_DIRECTION', ['paper/sections/03_system_model.tex', 'outputs/t20b_formal_results.json'],
         'Conditional waveform/runtime evaluation questions',
         'Arithmetic saving may not produce system-level savings', 'No implemented runtime or ICI capability'),
}

LITERATURE = {
 'edfors1996report': dict(verified_source='https://www.diva-portal.org/smash/get/diva2:997763/FULLTEXT01.pdf',
     support_scope='Frequency-domain LMMSE; SVD reduced-rank full-estimator approximation',
     read_version='1996 technical report TULEA 1996:18, URN urn:nbn:se:ltu:diva-24711',
     locator='Sections 3.2/3.3 and Appendix A', full_text_available=True, risk='LOW',
     current_access='Primary report PDF rechecked; no DOI/arXiv verified in checked sources'),
 'mei2021': dict(verified_source='https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cmu2.12250',
     support_scope='Candidate correlation selection via interpolation-error comparison of noisy observations',
     read_version='Publisher-indexed article sections inspected in frozen literature review',
     locator='Sections 3-5; Algorithm 1', full_text_available=True, risk='MEDIUM',
     current_access='Direct publisher access failed this turn; technical support uses frozen inspected-section record'),
 'ha2026mmse': dict(verified_source='https://arxiv.org/html/2506.00452v5',
     support_scope='Offline learned MMSE filters, linear inference and rank-adaptive full-filter representation',
     read_version='arXiv 2506.00452v5, 2026 revision of 2025 submission',
     locator='Sections IV-VI; equations 29-32', full_text_available=True, risk='MEDIUM',
     current_access='Primary author HTML available; unknown online fitting/label requirements unasserted'),
 'ozdemir2006': dict(verified_source='https://ieeexplore.ieee.org/document/1705928',
     support_scope='Title, authors, year, journal and DOI metadata only; no technical comparison',
     read_version='DOI 10.1109/TWC.2006.04713 metadata', locator='Frozen Crossref record and institutional metadata',
     full_text_available=False, risk='HIGH_UNRESOLVED', current_access='Full text and primary abstract unverified',
     evidence_type='METADATA_ONLY', technical_comparison_limited=True, verified_from_abstract_only=False),
 'turan2024': dict(verified_source='https://arxiv.org/html/2403.17577',
     support_scope='GMM conditional LMMSE estimates, posterior combination and component pilot selection',
     read_version='arXiv 2403.17577v1', locator='Conditional estimator and pilot-design sections',
     full_text_available=True, risk='MEDIUM', current_access='Primary author HTML rechecked; equal-budget comparison absent'),
 'gui2014preprint': dict(verified_source='https://arxiv.org/abs/1407.6083v1',
     support_scope='Two adaptive sparse LMS filters; affine stochastic-gradient combination',
     read_version='Gui/Xu two-author arXiv 1407.6083v1, distinct from four-author 2015 journal version',
     locator='Affine combiner equations in inspected v1', full_text_available=True, risk='MEDIUM',
     current_access='Primary v1 record available; full text not fetched this turn; prior inspected v1 supports details'),
 'luan2023channelformer': dict(verified_source='https://arxiv.org/pdf/2302.04368v1',
     support_scope='Offline neural initialization; gradient online training; specialized feature/label double-symbol DMRS',
     read_version='arXiv 2302.04368v1, not assumed identical to final journal article',
     locator='Sections II, III-E, IV-C, V, VI-D; double-symbol discussion pp.15-16',
     full_text_available=True, risk='MEDIUM', current_access='Primary author PDF rechecked; no equal-budget benchmark'),
 'wang2025lorat': dict(verified_source='https://xubolee.github.io/files/papers/GLOBECOM25_LoRAT.pdf',
     support_scope='Low-rank neural attention weight adaptation and distribution transfer',
     read_version='Author 6-page GLOBECOM manuscript, DOI 10.1109/GLOBECOM59602.2025.11431847',
     locator='Sections II-IV', full_text_available=True, risk='MEDIUM',
     current_access='Direct PDF fetch failed this turn; prior inspected author version retained; pilot-CV API unknown'),
 'lee2026': dict(verified_source='https://www.researchgate.net/publication/405423813_Two-Tier_PEFT_Adapter_Switching_for_Environment-_and_SNR-Adaptive_Channel_Estimation',
     support_scope='Trained neural PEFT; environment bases and SNR adapter selection; SISO evaluation',
     read_version='Public accepted-author manuscript HTML; publisher-final/local PDF not obtained',
     locator='Sections III/IV/VII; Algorithms 1-2; equations 17-23; Table 2',
     full_text_available=True, risk='MEDIUM', current_access='Author HTML rechecked; label-free base choice/equal FDM budget unknown'),
}
CONTROLS = ('docs/t20a_closest_work_matrix.md', 'docs/paper_literature_closure.md',
            'paper/bibliography/technical_closure.json', 'paper/bibliography/references.bib')
PRIOR_PROPERTIES = {k: v['support_scope'] for k, v in LITERATURE.items()}
RELATED_PRIORS = {'r01':['edfors1996report'], 'r02':['mei2021'], 'r03':['edfors1996report','ha2026mmse'],
                 'r04':['ozdemir2006'], 'r05':['turan2024'], 'r06':['gui2014preprint'],
                 'r07':['luan2023channelformer'], 'r08':['wang2025lorat','lee2026'],
                 'r09':['mei2021','gui2014preprint','edfors1996report'], 'r10':['wang2025lorat','lee2026']}
OUR_PROPERTIES = {
 'r01':'Source empirical LMMSE and prepared corrections',
 'r02':'Continuous two-real-coordinate fitting over a fixed operator library',
 'r03':'Each correction alone is compressed; full source and exact cross-pilot operators retained',
 'r05':'Analytical real least-squares coordinates, not probabilistic mixture responsibilities',
 'r06':'Frozen empirical LMMSE operators with causal-window cross-pilot least squares',
 'r07':'Unchanged sparse FDM MIMO pilots and analytical coordinate inference',
 'r08':'Frozen statistical correction operators, not learned neural parameters',
 'r09':'Fixed filters, continuous real coordinates, held-out-pilot objective and correction-only compression',
 'r10':'Received pilots, known noise and frozen source means/library; clean statistics offline; 2x2 CDL-B/C evaluation'}


def paragraphs(text):
    result = {}
    for block in re.split(r'\n\s*\n', text):
        match = re.search(r'% support-id: ([dr]\d+)', block)
        body = re.sub(r'(?<!\\)%[^\n]*', '', block)
        body = re.sub(r'\\(?:section|subsection)\{[^}]+\}', '', body).strip()
        if body:
            if not match or match[1] in result:
                raise ValueError('Unannotated or duplicate prose paragraph')
            result[match[1]] = body
    return result


def sentences(body):
    text = re.sub(r'\s+', ' ', body).replace('Fig.~', 'Figure~').strip()
    return re.split(r'(?<=[.!?])\s+', text)


def validate_citation(record):
    if record['citation'] == 'ozdemir2006':
        if record.get('evidence_type') != 'METADATA_ONLY' or record['full_text_available']:
            raise ValueError('Inaccessible prior work cannot be technically cleared')
        if re.search(r'\b(?:uses|updates|requires|selects|compresses|optimizes)\b', record['sentence'], re.I):
            raise ValueError('Metadata cannot support an inferred algorithm')
    if re.search(r'\b(?:first|novel|unique|unprecedented)\b|no (?:prior|existing) work', record['sentence'], re.I):
        raise ValueError('Unsupported priority/gap claim')


def discussion_audit():
    blocks = paragraphs((ROOT/DRAFTS[0]).read_text(encoding='utf8'))
    if blocks.keys() != DISCUSSION.keys():
        raise ValueError('Claim metadata does not cover current Discussion')
    records=[]
    for key, body in blocks.items():
        category, sources, scope, alternative, boundary = DISCUSSION[key]
        hashes = {p: sha256(ROOT/p) for p in sources}
        for i, sentence in enumerate(sentences(body), 1):
            records.append(dict(sentence_id=f'{key}_{i}', sentence=sentence, category=category,
                evidence_source=sources, evidence_sha256=hashes, scope=scope,
                alternative_explanation=alternative, boundary=boundary,
                support_status='PROPOSAL_NOT_EXECUTED' if category=='FUTURE_DIRECTION' else
                               'CONDITIONAL_POSSIBILITY' if key=='d06' else 'SUPPORTED_WITH_STATED_SCOPE'))
    weights = [r for r in json_read(ROOT/'outputs/t20b_formal_weight_diagnostics.json')
               if r['environment']=='CDL-B-120-100']
    checks = dict(outside_simplex_description_exact=len(weights)==2 and
        all(r['outside_simplex_fraction']==1 and r['scored_windows']==6000 for r in weights),
        negative_local_mechanism_preserved=json_read(ROOT/'outputs/t17a_results.json')['wiener_perturbation_mechanism_contradicted'],
        cancellation_evidence_preserved=json_read(ROOT/'outputs/t17b_results.json')['affine_additivity_cancellation_dominated'],
        replicated_deficiency_preserved=json_read(ROOT/'outputs/t19c_results.json')['delay_operator_deficiency_replication_supported'],
        weak_residual_transfer_preserved=json_read(ROOT/'outputs/t19d_results.json')['delay_residual_subspace_transfer_weak'])
    return dict(sentences=records, uncovered_prose=[], checks=checks,
                current_draft_sha256=sha256(ROOT/DRAFTS[0]), all_checks_passed=all(checks.values()))


def related_audits():
    blocks = paragraphs((ROOT/DRAFTS[1]).read_text(encoding='utf8'))
    if blocks.keys()!=RELATED_PRIORS.keys():
        raise ValueError('Related Work metadata coverage mismatch')
    bib = bib_entries((ROOT/'paper/bibliography/references.bib').read_text(encoding='utf8'))
    control_hashes = {p: sha256(ROOT/p) for p in CONTROLS}
    citations, comparisons, paragraphs_reviewed = [], [], []
    for key, body in blocks.items():
        parasources = RELATED_PRIORS[key]
        for sentence in sentences(body):
            keys = [k for chunk in re.findall(r'\\cite\{([^}]+)\}', sentence) for k in chunk.split(',')]
            for citation in keys:
                assert citation in bib and citation in parasources
                record = dict(sentence=sentence, citation=citation, **LITERATURE[citation],
                    frozen_reading_records_sha256=control_hashes,
                    certainty_level='METADATA_ONLY' if citation=='ozdemir2006' else 'VERIFIED_READ_VERSION',
                    no_external_numerical_comparison=True)
                validate_citation(record); citations.append(record)
            # Include uncited comparative sentences via their paragraph's specific source set.
            if re.search(r'\b(?:here|present|rather than|relative to|different from|whereas|while their|comparison)\b',sentence,re.I) and key!='r04':
                comparisons.append(dict(sentence=sentence, our_property=OUR_PROPERTIES[key],
                    prior_work_property={k:PRIOR_PROPERTIES[k] for k in (keys or parasources)},
                    source={k:LITERATURE[k]['verified_source'] for k in (keys or parasources)},
                    our_source=['paper/sections/04_proposed_method.tex','docs/paper_scientific_freeze.md'],
                    certainty_level='VERIFIED_SPECIFIC_PROPERTIES; NO_PRIORITY_OR_PERFORMANCE_CLEARANCE',
                    unknown_fields_preserved=True))
        paragraphs_reviewed.append(dict(id=key, body=body, cited_sources=parasources,
            own_method_sha256=sha256(ROOT/'paper/sections/04_proposed_method.tex'),
            interpretation='metadata topic only' if key=='r04' else 'scoped synthesis of verified properties'))
    base = dict(current_draft_sha256=sha256(ROOT/DRAFTS[1]), manual_semantic_review_completed=True)
    citation_report = dict(**base, sentences=citations, paragraph_coverage=paragraphs_reviewed,
        unique_citations=sorted({r['citation'] for r in citations}), no_abstract_only_inference=True,
        ozdemir_status='METADATA_ONLY; TECHNICAL_COMPARISON_LIMITED; PRIORITY_RISK_OPEN',
        full_text_available_refers_to_inspected_version_not_current_fetch=True,
        all_checks_passed=True)
    comparison_report = dict(**base, comparisons=comparisons, literature=LITERATURE,
        unresolved_risks=[
            'Ozdemir adaptation object, update rule, online information, rank and pilot split UNKNOWN',
            'Lee environment-base selection/equal pilot budget UNKNOWN; accepted-author version only',
            'Gui exact training density/labels, Ha online fitting/labels, LoRAT pilot-CV interface UNKNOWN',
            'No external implementations reproduced or equal-information performance comparison run'],
        ingredients_already_established=['selection','affine combination','low-rank filtering','pilot-supported adaptation'],
        priority_claim_cleared=False, unsupported_priority_wording=[], all_checks_passed=True)
    return citation_report, comparison_report


def summary_audit():
    expected, cells, gaps, bound = products(csv_read(ROOT/'outputs/t20b_formal_nmse.csv'))
    provenance=json_read(ROOT/'paper/tables/t20g_provenance.json')
    checks = dict(products_reconstruct_exactly=all((ROOT/'paper/tables'/n).read_text(encoding='utf8')==t for n,t in expected.items()),
        cell_provenance_exact=provenance['cells']==cells, gaps_exact=provenance['single_shift_gaps']==gaps,
        bound_contains_all_four_gaps=len(gaps)==4 and all(g['gap_db']<=float(bound) for g in gaps),
        id_losses_retained=all(r['value_db']<0 for r in cells if r['comparison']=='source_gain' and r['environment']=='CDL-C-30-100'),
        boundary_penalties_retained=any(r['environment']=='CDL-C-30-200' and r['snr_db']=='20' and
            r['comparison']=='compression_penalty' and r['display']=='1.086' for r in cells),
        full_appendix_kept=all((ROOT/'paper/tables'/p).is_file() for p in ('02_nmse.tex','03_ber.tex')))
    return dict(checks=checks, row_count=10, cells=cells, single_shift_gaps=gaps,
                bound_db=bound, max_gap_db=max(g['gap_db'] for g in gaps),
                new_metric_estimated=False, all_checks_passed=all(checks.values()))


def results_edit_audit():
    before=json_read(ROOT/'outputs/t20g_before_manifest.json')
    path='paper/sections/06_results.tex'
    data, evidence=historical_hash_evidence(before['t20f_commit'],path,before['tracked_before_sha256'][path])
    old=data.decode('utf8').replace('\r\n','\n')
    replacements={
      'Fig.~\\ref{fig:nmse} shows all estimators, and Table~\\ref{tab:nmse-compact} gives the core comparisons across every condition.':
      'Fig.~\\ref{fig:nmse} shows all estimators, and Table~\\ref{tab:summary} gives the key NMSE differences across every condition.',
      '\\input{tables/02_nmse_compact}':'\\input{tables/06_summary}',
      'LR4 remained close to the Matched reference under these prepared single-shift conditions, as shown by the values in Table~\\ref{tab:nmse-compact}.':
      'LR4 was within \\SingleShiftMatchedBound{} dB of the Matched reference across all evaluated prepared single-shift conditions.',
      'Fig.~\\ref{fig:ber} retains all eight methods, while Table~\\ref{tab:ber-compact} gives the core BER comparisons.':
      'Fig.~\\ref{fig:ber} retains all eight methods, with their complete BER values and intervals in the appendix.',
      '\\input{tables/03_ber_compact}\n':''}
    changed=old
    for a,b in replacements.items():
        assert changed.count(a)==1, a
        changed=changed.replace(a,b)
    actual=(ROOT/path).read_text(encoding='utf8')
    return dict(historical_provenance=evidence, allowed_replacements=replacements,
                full_results_rewrite=False, all_checks_passed=changed==actual)


def integrity():
    before=json_read(ROOT/'outputs/t20g_before_manifest.json')
    hashes=before['tracked_before_sha256']
    changed=[p for p,h in hashes.items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=h]
    unexpected=[p for p in changed if p not in ALLOWED]
    scripts=('prepare_paper_positioning.py','build_paper_summary.py','audit_paper_positioning.py')
    firewall={p:script_firewall((ROOT/'scripts'/p).read_text(encoding='utf8')) for p in scripts}
    report=dict(tracked_files_checked=len(hashes), historical_outputs_checked=sum(p.startswith('outputs/') for p in hashes),
        scientific_source_files_checked=sum(p.startswith('src/') for p in hashes), changed_files=changed,
        unexpected_mutations=unexpected, historical_output_mutations=[p for p in changed if p.startswith('outputs/')],
        scientific_code_unchanged=not any(p.startswith('src/') for p in changed), paper_script_firewall=firewall,
        new_channel_generated=False,new_statistics_fit=False,new_model_trained=False,method_changed=False,
        formal_data_used_only_for_reporting=True, all_checks_passed=not unexpected and not any(firewall.values()))
    return report


def build(reviewed=False):
    if not reviewed:
        raise ValueError('Claim and citation semantic review must precede audit completion')
    before=json_read(ROOT/'outputs/t20g_before_manifest.json')
    # Source evidence bytes must be exact to the pre-writing snapshot, not just exist.
    protected=set(CONTROLS)
    protected.update(p for spec in DISCUSSION.values() for p in spec[1] if p.startswith(('outputs/','docs/')))
    for p in protected:
        if sha256(ROOT/p)!=before['tracked_before_sha256'][p]:
            raise ValueError('Frozen writing evidence changed: '+p)
    reports={'discussion_claim':discussion_audit(), 'table_numeric':summary_audit(),
             'results_edit':results_edit_audit(), 'latex_static':latex()}
    reports['citation_support'],reports['related_work_comparison']=related_audits()
    reports['paper_style']=style()
    for name,report in reports.items():
        json_write(ROOT/f'outputs/t20g_{name}_audit.json',report)
    hashes_report=integrity()
    json_write(ROOT/'outputs/t20g_integrity_audit.json',hashes_report)
    checks={k:r.get('all_checks_passed',r.get('all_lexical_checks_passed',r.get('static_validation_passed'))) for k,r in reports.items()}
    checks['integrity']=hashes_report['all_checks_passed']
    print(checks)
    if not all(checks.values()):
        raise ValueError('Editorial audit failed: '+str(checks))
    return reports


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewed',action='store_true')
    args=parser.parse_args()
    build(args.reviewed)
