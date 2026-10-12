"""Package reviewed current prose and comparison limits for advisor review."""
import re
from paper_support import ROOT, json_read, json_write, sha256
from audit_paper_positioning import paragraphs, sentences, LITERATURE, DRAFTS
from audit_paper_style import printed_tex


def headings(path):
    return re.findall(r'\\subsection\{([^}]+)\}', (ROOT/path).read_text(encoding='utf8'))


def build():
    discussion=paragraphs((ROOT/DRAFTS[0]).read_text(encoding='utf8'))
    related=paragraphs((ROOT/DRAFTS[1]).read_text(encoding='utf8'))
    citation=json_read(ROOT/'outputs/t20g_citation_support_audit.json')
    claims=json_read(ROOT/'outputs/t20g_discussion_claim_audit.json')
    comparisons=json_read(ROOT/'outputs/t20g_related_work_comparison_audit.json')
    numbers=(ROOT/'paper/tables/result_numbers.tex').read_text(encoding='utf8')
    macros=dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}', numbers))
    def readable(text):
        for k,v in macros.items(): text=text.replace('\\'+k+'{}',v)
        text=re.sub(r'\\cite\{([^}]+)\}',r' [\1]',text)
        text=re.sub(r'\\ref\{([^}]+)\}',r'[\1]',text)
        return re.sub(r'\s+',' ',text).replace('~',' ').strip()
    packet=['# T20G Advisor Review Packet', '', '## A. Discussion subsection structure', '']
    packet += ['- '+h for h in headings(DRAFTS[0])]
    packet += ['', '## B. Core conclusion of each subsection', '']
    conclusions=(
        'Unrestricted coordinates improve the held-out composition within the prepared library; outside-simplex geometry is descriptive and does not prove a physical cause.',
        'Rank4 applies only to the weighted corrections. The empirical span is useful under the composed shift, while tested local-perturbation and finite-shift-separability explanations remain unsupported.',
        'Deployment combines clean offline statistical preparation with received-pilot coordinate inference. Acquisition feasibility and representativeness remain unevaluated.',
        'Evidence separates source-matched, prepared single-shift, held-out composition and unseen-delay conditions. The tested delay boundary retains a Source benefit but has a large Matched gap and degraded high-SNR compression fidelity.',
        'Ideal synchronization and symbolwise no-ICI modeling limit mobility scope. Arithmetic/storage evidence does not establish timing/energy gains, and NMSE/BER gains differ in magnitude.',
        'Further claims depend on wider family/geometry evaluation, transferable compact delay representation, reliable measurement-derived statistics, waveform ICI evaluation and measured runtime costs.')
    for h,conclusion in zip(headings(DRAFTS[0]),conclusions):
        packet += ['### '+h,'']
        packet += [conclusion]
        packet.append('')
    # Exhaustive limitation inventory: record whole reviewed paragraphs wherever
    # they contain a scope restriction, so no omitted subordinate sentence is hidden.
    packet += ['## C. All limitation statements', '']
    for key in ('d01','d02','d03','d04','d05','d06','d07','d08','d09','d10','d11'):
        packet += ['- **'+key+'** '+readable(discussion[key])]
    packet += ['', '## D. All future-work statements', '']
    packet += ['- '+readable(discussion[k]) for k in ('d12','d13')]
    packet += ['', '## E. Related Work subsection structure', '']
    packet += ['- '+h for h in headings(DRAFTS[1])]
    packet += ['', '## F. Eight closest works and accurate differences', '',
               '| Work/version | Verified property | Specific difference / boundary |',
               '|---|---|---|']
    distinctions={
      'edfors1996report':'Full-estimator SVD approximation versus correction-only compression; full source and fitting retained.',
      'mei2021':'Candidate correlation selection versus analytical continuous real coordinates over frozen operators.',
      'gui2014preprint':'Adaptive sparse LMS filters and stochastic-gradient coefficient versus fixed empirical LMMSE corrections and cross-pilot real LS; no performance reproduction.',
      'ozdemir2006':'Technical comparison limited: topic overlap only. Adaptation object/update/information/rank/pilots UNKNOWN; cannot claim differentiation.',
      'ha2026mmse':'Offline learned full-filter representation versus prepared empirical corrections; no new low-rank theorem. Unknown online fitting/labels unasserted.',
      'wang2025lorat':'Neural weight factors versus statistical correction operators. Prepared-base/compact-library overlap acknowledged; pilot-CV API UNKNOWN.',
      'lee2026':'Environment neural bases/SNR PEFT selection versus two real coordinates; accepted-author version only; label-free base choice/equal FDM budget UNKNOWN.',
      'luan2023channelformer':'Gradient online neural training with specialized double-symbol feature/label DMRS versus unchanged sparse FDM MIMO pilots and analytical fitting; no equal-budget benchmark.'}
    for k,distinction in distinctions.items():
        source=LITERATURE[k]
        packet.append('| ['+k+']('+source['verified_source']+') / '+source['read_version']+' | '+source['support_scope']+' | '+distinction+' |')
    packet += ['', '## G. All unresolved prior-art risks', '']
    packet += ['- '+r for r in comparisons['unresolved_risks']]
    packet += ['- Edfors technical report uses its verified URN; no DOI/arXiv verified in checked sources. Do not borrow the identifier of another publication.',
               '- Current direct access failed for Mei/LoRAT; their technical statements remain tied to the previously inspected versions and frozen reading records.',
               '- Metadata/topic overlap with Ozdemir remains a substantial unresolved novelty risk, regardless of the success of the current experiments.']
    packet += ['', '## H. Potentially strong novelty wording', '',
               '- No priority/gap claim appears in the manuscript. Selection, affine combination, low-rank filtering and pilot-supported adaptation are acknowledged lineage.',
               '- The final sentence "These adaptation objects, information requirements and compression choices define the contribution examined in this paper" is a scoped positioning statement, not proof that no prior work has this combination.',
               '- "Their common advantage identifies the coordinate constraint as a consequential design choice" is limited to paired methods within this library; it is not a unique physical mechanism.',
               '- "Supports retaining a compact two-operator design" expresses the present evidence boundary; it does not establish that all extra directions are ineffective.']
    packet += ['', '## I. Results wording edits', '']
    edits=json_read(ROOT/'outputs/t20g_results_edit_audit.json')['allowed_replacements']
    for old,new in edits.items():
        packet += ['- Before: `'+old.replace('\n','')+'`', '  After: `'+(new.replace('\n','') or '(removed redundant main-table input)')+'`']
    packet += ['', '## J. Final main-table recommendation', '',
               '- Use one neutral 10-row summary table (five conditions x two SNRs), with Source−LR4 gain in every row, Convex−LR4 only for Composed, LR4−Dense penalty in every row and LR4−Matched for Boundary.',
               '- Negative ID gains and the 1.086-dB high-SNR Boundary penalty remain explicit.',
               '- Keep the three complete eight-method figures and full eight-method NMSE/BER appendix tables byte-exact. Existing compact table files are preserved but no longer inserted in the main text.',
               '- This choice reduces 100 main-table metric values to 24 differences without selecting favorable conditions; it is not venue-specific.',
               '- Final pagination and visual table width require a working compiler; current native compiler failed before source parsing.']
    (ROOT/'docs/t20g_advisor_review_packet.md').write_text('\n'.join(packet)+'\n',encoding='utf8',newline='\n')
    counts={}
    for path in DRAFTS:
        counts[path]=len(re.findall(r'\b[A-Za-z][A-Za-z-]*\b',printed_tex((ROOT/path).read_text(encoding='utf8'))))
    notes=['# T20G Writing Notes','',
       'Scope: English Discussion and Related Work; quantify one Results sentence and replace two dense main tables with one summary. Introduction, Conclusion, Abstract, method, protocol, statistics, seeds and scientific outputs are unchanged.', '',
       '## Evidence hierarchy', '',
       '- Formal independent evaluation supports empirical efficacy/composition/compression and the delay boundary.',
       '- Frozen development analyses constrain mechanism interpretation and residual-extension scope; they do not create new independent-test evidence.',
       '- Clean offline source/profile-origin/mobility-origin statistics remain required. Known noise and received pilots determine online coordinates; no target statistics, clean CSI or environment labels enter deployment.', '',
       '## Literature handling', '',
       '- Nine representative works are cited; eight closest works are compared in the packet. Technical facts are tied to exact inspected versions and locators.',
       '- Ozdemir: METADATA_ONLY, not ABSTRACT_ONLY. Technical comparison and priority clearance remain open.',
       '- Lee: accepted-author HTML, not a claimed publisher-final PDF. Channelformer: arXiv v1 feature/label pilot arrangement. Gui: two-author preprint, not a merged journal record.',
       '- Previously read primary versions remain controlling evidence where direct access failed; current-access status is recorded separately.', '',
       '## Editorial validation', '',
       '- Source annotations are nonprinted comments. Sentence records include evidence, scope, alternative explanations and boundaries.',
       '- The 0.38-dB sentence is a conservative upward-rounded bound of all four frozen single-shift gaps; no new metric/statistic is estimated.',
       '- T20F historical draft-scope check now validates its actual committed snapshot ec9b672; current scope is checked separately. No historical hash or result is rewritten.',
       '- Initial snapshot utility rejected its own new untracked files. Its three bootstrap utilities were explicitly enumerated; tracked accepted files stayed clean. Clean status had already been observed immediately after the T20F commit.',
       '- Small calibration: 4 passed before summary generation. Paper regression: see actual execution record. Full regression: see actual execution record after completion.',
       '- Native preview compilation attempted on the existing editor file; compiler returned "Unable to find standard directories for platform" before source parsing. Static checks do not imply successful compilation or validated pages.',
       '- Existing Python launcher-location notice and Git LF/CRLF notices remain engineering warnings.', '',
       '## Length and reviewer boundary', '',
       '- Approximate lexical counts (including headings): '+str(counts)+'. Actual 1.2–1.8 / 1.3–2.0 page targets remain unverified until native compilation is available.',
       '- Advisor review is required for the interpretation and novelty scope; the audits do not clear inaccessible prior art.',
       '- No conceptual figure is needed; no new scientific figure was produced.',
       '- This task stops at the requested drafts.']
    (ROOT/'docs/t20g_writing_notes.md').write_text('\n'.join(notes)+'\n',encoding='utf8',newline='\n')
    json_write(ROOT/'outputs/t20g_review_packet_audit.json',dict(
        expected_parts=list('ABCDEFGHIJ'), parts_present=all('## '+p+'.' in '\n'.join(packet) for p in 'ABCDEFGHIJ'),
        closest_works_count=len(distinctions), unresolved_literature_risks_preserved=True,
        discussion_sentence_count=len(claims['sentences']), citation_sentence_count=len(citation['sentences']),
        lexical_counts_including_headings=counts, actual_pages_verified=False,
        packet_sha256=sha256(ROOT/'docs/t20g_advisor_review_packet.md'),
        notes_sha256=sha256(ROOT/'docs/t20g_writing_notes.md')))
    print(dict(advisor_packet='docs/t20g_advisor_review_packet.md',closest_works=8,lexical_counts=counts))


if __name__=='__main__':
    build()
