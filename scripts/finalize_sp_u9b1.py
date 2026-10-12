"""Finalize the manuscript-only audit from actual source, build and review evidence.

No scientific code imports, experiment execution, bootstrap or figure generation.
The visual receipt records the six pages actually inspected in this chat.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/sp_u9b1'
GIT = Path(r'C:\Users\yuhua\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def uncomment(text):
    return re.sub(r'(?m)(?<!\\)%.*$', '', text)


def outside_abstract(text):
    return re.sub(r'(?s)(\\begin\{abstract\}).*?(\\end\{abstract\})', r'\1ABSTRACT\2', text)


def active_inputs():
    seen = set()
    def visit(path):
        relative = path.relative_to(ROOT).as_posix()
        if relative in seen:
            return ''
        seen.add(relative)
        text = uncomment(path.read_text(encoding='utf-8'))
        for name in re.findall(r'\\input\{([^}]+)\}', text):
            text += '\n' + visit(ROOT / 'paper' / (name + '.tex'))
        return text
    return visit(ROOT / 'paper/main.tex'), sorted(seen)


def bib_records(text):
    """Balanced-brace parser: accepts acronym protection and multi-line fields."""
    entries = {}
    for match in re.finditer(r'(?m)^@(\w+)\{([^,]+),', text):
        start = match.end()
        depth = 1
        index = start
        while depth:
            if index >= len(text):
                raise ValueError('Unbalanced bibliography entry')
            if text[index] == '{':
                depth += 1
            elif text[index] == '}':
                depth -= 1
            index += 1
        body = text[start:index-1]
        fields = {}
        cursor = 0
        while True:
            field = re.search(r'(\w+)\s*=\s*\{', body[cursor:])
            if not field:
                break
            value_start = cursor + field.end()
            pos = value_start
            depth = 1
            while depth:
                if body[pos] == '{':
                    depth += 1
                elif body[pos] == '}':
                    depth -= 1
                pos += 1
            fields[field[1]] = body[value_start:pos-1]
            cursor = pos
        key = match[2].strip()
        if key in entries:
            raise ValueError('Duplicate bibliography key: ' + key)
        entries[key] = dict(type=match[1], fields=fields)
    return entries


def build_checks(stage):
    report = read(OUT / f'build_{stage}/report.json')
    log = (OUT / f'build_{stage}/main.log').read_text(encoding='utf-8')
    blg = (OUT / f'build_{stage}/main.blg').read_text(encoding='utf-8')
    aux = (OUT / f'build_{stage}/main.aux').read_text(encoding='utf-8')
    labels = re.findall(r'\\newlabel\{([^}]+)\}', aux)
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    return dict(exit_code=report['exit_code'], pdf=report['pdf'], pdf_sha256=sha(Path(report['pdf'])),
                source_log=report['log'], console=report['console'],
                latex_errors=re.findall(r'(?m)^!.*', log),
                undefined_citations=re.findall(r"(?m)^.*Citation.*undefined.*$", log),
                undefined_references=re.findall(r"(?m)^.*(?:Reference.*undefined|undefined references).*$", log),
                overfull_boxes=re.findall(r'(?m)^Overfull.*$', log),
                duplicate_labels=duplicate_labels,
                bibtex_errors=re.findall(r'(?mi)^.*(?:warning--|error|I couldn.t|Repeated entry|not found).*$' ,blg),
                warnings=[line for line in log.splitlines() if 'Warning:' in line],
                fontconfig_stderr='Fontconfig error: Cannot load default config file' in (OUT / f'build_{stage}/console.txt').read_text(encoding='utf-8'))


def main():
    before = read(ROOT / 'outputs/sp_u9b1_preedit_manifest.json')
    current = {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / 'paper').rglob('*') if p.is_file()}
    changes = [name for name, digest in before['paper_sources'].items() if current.get(name) != digest]
    added = sorted(set(current) - set(before['paper_sources']))
    forbidden = sorted(set(changes) - set(before['allowed']))
    frozen_drift = [name for name, digest in before['frozen_files'].items() if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    main_scope = outside_abstract((ROOT / 'paper/main.tex').read_text(encoding='utf-8')) == outside_abstract((OUT / 'pre_source/main.tex').read_text(encoding='utf-8'))
    snapshot = dict(task='SP-U9B-1', paper_file_count=len(current),
                    allowed_files_changed=sorted(changes), forbidden_files_unchanged=len(current)-len(changes),
                    forbidden_drift=forbidden, added_paper_files=added, main_abstract_only=main_scope,
                    frozen_file_count=len(before['frozen_files']), frozen_drift=frozen_drift,
                    pre_post={name:dict(pre=before['paper_sources'][name],post=current.get(name)) for name in before['paper_sources']})
    dump(ROOT / 'outputs/sp_u9b1_postedit_manifest.json', snapshot)
    assert not forbidden and not added and not frozen_drift and main_scope

    pre, post = build_checks('pre'), build_checks('post')
    new_warnings = sorted(set(post['warnings']) - set(pre['warnings']))
    preserved_warnings = sorted(set(post['warnings']) & set(pre['warnings']))
    for build in [pre, post]:
        assert build['exit_code'] == 0
        assert not any(build[k] for k in ['latex_errors','undefined_citations','undefined_references','overfull_boxes','duplicate_labels','bibtex_errors'])
    assert not new_warnings

    qa = read(OUT / 'visual_qa.json')
    assert qa['pdf_sha256'] == post['pdf_sha256']
    assert [r['page'] for r in qa['reviewed_pages']] == [1,2,3,4,15,16]
    for record in qa['reviewed_pages']:
        assert sha(Path(record['image'])) == record['sha256']
    qa.update(review_status='PASS',review_method='Direct visual inspection by the writing agent in the current chat; not inferred from compilation',
              inspection_notes={
                  '1':'Two-line required title; full six-slot abstract including text after escaped percent; preserved TODO authors and affiliations; legible citations and heading hierarchy.',
                  '2':'Introduction paragraphs, all four contribution bullets and transition to Related Work; no clipping or overlap.',
                  '3':'All six Related Work paragraphs end before System Model; previously observed overflow removed; numeric citations render.',
                  '4':'System Model and original mathematical expressions render normally; no source changes to equations or system scope.',
                  '15':'References start with preceding frozen appendix table; new GMM/MFA/LS records and preprint version notes legible, title acronyms preserved.',
                  '16':'References continuation, Mei and both Ozdemir records legible; natural bibliography continuation with ample whitespace; no overflow.'},
              unresolved_layout_items=['Pre-existing TODO author/affiliation placeholders', 'Existing deferred figure float placement and unused appendix heading retained within frozen scope'])
    dump(OUT / 'visual_qa.json',qa)
    compile_report = dict(task='SP-U9B-1',compiler=read(OUT / 'toolchain/toolchain.json'),
                          full_manuscript_build=True,actual_existing_figures=True,pre=pre,post=post,
                          warning_classification=dict(PREEXISTING=preserved_warnings,NEW=new_warnings,
                              fontconfig='PREEXISTING in both before/after builds; toolchain stderr; visually legible PDF'),
                          visual_qa=qa,verdict='PASS')
    dump(ROOT / 'outputs/sp_u9b1_compile_report.json',compile_report)

    active, inputs = active_inputs()
    banned = r'\bfirst\b|for the first time|unprecedented|novel mixing|\bno prior\b|outperforms (?:MFA|GMM|A-MMSE)|superior to modern learned methods|near[- ]optimal|near[- ]oracle|state[- ]of[- ]the[- ]art|low[- ]latency|hardware[- ]efficient|sample[- ]efficient'
    hits = re.findall(banned,active,flags=re.I)
    bibliography = bib_records((ROOT / 'paper/bibliography/references.bib').read_text(encoding='utf-8'))
    citations = sorted(set(key for group in re.findall(r'\\cite\{([^}]+)\}',active) for key in group.split(',')))
    undefined = sorted(set(citations)-set(bibliography))
    own_cross_terms = re.findall(r'cross-validation|held-out pilot fitting',active,flags=re.I)
    assert not hits and not undefined and not own_cross_terms
    audit = dict(task='SP-U9B-1',active_input_files=inputs,forbidden_claim_hits=hits,undefined_citation_keys=undefined,
                 citations_used=citations,bibliography_entry_count=len(bibliography),
                 cross_pilot_ambiguous_synonyms=own_cross_terms,component_priority='PASS: contribution 2 explicitly specifies prepared correction space; 1/3/4 name the implemented objects and evaluation',
                 source_limits={'P4':'PUBLISHER_ABSTRACT; no inferred adaptation variables or rank construction',
                    'P5':'ABSTRACT_ONLY; no full method-field inference',
                    'P6':'PUBLISHER_ABSTRACT/partial publisher text; journal equations and coefficient constraints not inferred from 2014 preprint',
                    'Channelformer':'Existing title-level context only, explicitly authorized genuinely relevant existing learning work; not represented as U9A-R1 full-text ledger evidence'},
                 evidence_conflicts=[],scope_drift=[],ET2026_included=False,verdict='PASS')
    dump(OUT / 'claim_scope_audit.json',audit)
    status = subprocess.run([str(GIT),'status','--short'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True).stdout
    (OUT / 'git_status_after.txt').write_text(status,encoding='utf-8')
    diff = subprocess.run([str(GIT),'diff','--']+before['allowed'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True).stdout
    (OUT / 'manuscript.diff').write_text(diff,encoding='utf-8')
    diff_check = subprocess.run([str(GIT),'diff','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    (OUT / 'git_diff_check.txt').write_text(diff_check.stdout+diff_check.stderr,encoding='utf-8')
    tracked = subprocess.run([str(GIT),'diff','--name-only'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True).stdout.splitlines()
    assert set(tracked) <= set(before['allowed']) and diff_check.returncode == 0
    before_lines = set((OUT/'git_status_before.txt').read_text(encoding='utf-8').splitlines())
    after_lines = set(status.splitlines())
    git_audit = dict(tracked_changed=tracked,diff_check_exit_code=diff_check.returncode,
                     status_before_entries=len(before_lines),status_after_entries=len(after_lines),
                     new_status_entries=sorted(after_lines-before_lines),removed_status_entries=sorted(before_lines-after_lines),
                     commit_performed=False)
    dump(OUT/'git_audit.json',git_audit)
    editorial=read(OUT/'editorial_report.json')
    changelog = '# SP-U9B-1 change log\n\n'
    changelog += 'FRONT_HALF_PASS is conditional on the separately executed scoped pytest and final receipt. No U9B-2 work is authorized by this artifact.\n\n'
    changelog += '\n'.join('- '+name for name in sorted(changes))+'\n\n'
    changelog += ('Only main.tex Abstract changed outside the separately authorized metadata file. '
                  'The active bibliography is paper/bibliography/references.bib; the nonexistent task shorthand paper/references.bib was mapped without changing main.tex bibliography wiring.\n\n'
                  'Title/PDF title, six-slot single-paragraph Abstract, nine Introduction paragraphs plus four scoped contributions, six Related Work paragraphs and ten nearest-work bibliography records were updated. '
                  'Source anchor, cross-pilot fitting, prepared origin labels, correction-only compression, offline clean-statistics contract, online information budget, surrogate risk distinction and no-ICI scope are explicit.\n\n'
                  'Koller/Fesl GMM/MFA are closest statistical comparators; affine LS and Mei prediction selection precedents are acknowledged. Özdemir full method fields remain unresolved; Gui 2014/2015 version constraints are not conflated. A-MMSE is fixed to v5 preprint. ET2026 was omitted by default. Channelformer uses existing title-level context only. No external performance superiority is claimed.\n\n'
                  'T20B and U7B-R1 numbers were read only for frozen reporting; no result-dependent method, threshold or theory selection was performed. Complexity is an arithmetic proxy, not measured runtime. U4/U5 remain development sensitivity; U8 remains post-hoc.\n\n')
    changelog += f"Words: {editorial['word_counts']}. Number provenance rows: {editorial['abstract_number_rows']}; Introduction rows: {editorial['intro_claim_rows']}; Related Work rows: {editorial['related_work_sentence_rows']}.\n\n"
    changelog += f"Preservation: {len(before['paper_sources'])} paper files checked; {len(changes)} authorized changes; {len(before['frozen_files'])} frozen non-edit files unchanged. Full before/after builds passed; visually inspected PDF pages 1/2/3/4/15/16. PREEXISTING inputenc warning and Fontconfig toolchain stderr remain; NEW warnings zero.\n\n"
    changelog += 'Remaining manuscript risks: author/affiliation placeholders, historical later-section wording/structure and deferred figure placement require mentor-directed subsequent work; no full-manuscript readiness claim. No experiment, bootstrap, figure regeneration, production change or Git commit. STOP for mentor review.\n'
    (ROOT/'docs/sp_u9b1_change_log.md').write_text(changelog,encoding='utf-8')
    dump(OUT/'final_audit.json',dict(task='SP-U9B-1',scope='PASS',compile='PASS',visual='PASS',claim='PASS',
                                    tests='PENDING_ACTUAL_PYTEST',git=git_audit,counts=editorial['word_counts']))
    print(json.dumps(dict(paper_files=len(current),changed=changes,frozen_unchanged=len(before['frozen_files']),
                         pages=qa['pages_total'],new_warnings=len(new_warnings),tracked_changed=tracked,verdict='AUDIT_PASS_TEST_PENDING'),ensure_ascii=False))


if __name__ == '__main__':
    main()
