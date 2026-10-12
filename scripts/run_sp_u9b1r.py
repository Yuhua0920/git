"""SP-U9B-1R: bounded language edits, frozen-claim checks, full LaTeX build.

Never imports scientific code or recomputes a result. Existing number audit is
read only to verify the unchanged displayed values; historical outputs stay intact.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/sp_u9b1r'
ALLOWED=['paper/main.tex','paper/sections/01_introduction.tex','paper/sections/02_related_work.tex']
GIT=Path(r'C:\Users\yuhua\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe')
OLD=ROOT/'outputs/sp_u9b1'
TERMS=['sealed','formal','preregistered','pre-registered','post-hoc','gate','PASS','FAIL','evidence','validation','causal','online','real-time',
       'low-latency','instantaneous','symbol-immediate','zero-latency','evidence tier','evidence role','USED validation','reporting-only','fixed reproduction protocol','post-hoc evidence']
BUFFERING=('The window is causal across frames and uses no future frame. '
           'Because current-frame pilots enter the fit, practical implementation may require within-frame buffering; '
           'neither symbol-immediate nor zero-latency estimation is claimed.')
C4=('We characterize the frozen estimator across prepared profile and mobility mismatch and an independently evaluated unseen CDL-A profile condition. '
    'Sensitivity experiments examine pilot-density changes and offline-statistics budgets, while the characterization reports in-distribution degradation, '
    'condition-dependent compression behavior and matched-statistics gaps.')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def uncomment(text):return re.sub(r'(?m)(?<!\\)%.*$', '',text)
def abstract(text):return re.search(r'(?s)\\begin\{abstract\}(.*?)\\end\{abstract\}',text)[1].strip()
def outer(text):return re.sub(r'(?s)(\\begin\{abstract\}).*?(\\end\{abstract\})',r'\1ABSTRACT\2',text)
def citations(text):return re.findall(r'\\cite\{([^}]+)\}',uncomment(text))
def printed(text):
    text=uncomment(text)
    return re.sub(r'\\(?:input|includegraphics|label|ref|cite|section|subsection|begin|end)\*?(?:\[[^\]]*\])?\{[^}]*\}', '',text)
def numeric(text):return re.findall(r'(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:-dB)?',printed(text))
def words(text):
    text=uncomment(text)
    text=re.sub(r'\\cite\{[^}]+\}|\\(?:begin|end|section|subsection|paragraph|label)\{[^}]*\}', '',text)
    text=text.replace('\\%','%').replace('~',' ').replace('\\item','')
    text=re.sub(r'\\[a-zA-Z]+','',text).replace('{','').replace('}','')
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",text))
def git(*args):return subprocess.run([str(GIT),*args],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True).stdout
def replace_once(text,old,new):
    assert text.count(old)==1,old[:100]
    return text.replace(old,new,1)


def prepare():
    target=ROOT/'outputs/sp_u9b1r_preedit_manifest.json'
    assert not target.exists(),'Immutable before snapshot already exists'
    OUT.mkdir(exist_ok=True)
    paper={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'paper').rglob('*') if p.is_file()}
    frozen=dict(load(ROOT/'outputs/sp_u9b1_preedit_manifest.json')['frozen_files'])
    # Snapshot current historical artifacts, never reuse their old draft assertions.
    for prefix in ['outputs/sp_u9b1','scripts','tests','docs','src','configs']:
        for p in (ROOT/prefix).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and 'tex_cache' not in p.parts and 'sp_u9b1r' not in p.as_posix():
                name=p.relative_to(ROOT).as_posix()
                if prefix=='outputs/sp_u9b1' or name in frozen or 'sp_u9b1' in name:
                    frozen[name]=sha(p)
    for p in (ROOT/'outputs').glob('sp_u9b1_*'):
        if p.is_file():frozen[p.relative_to(ROOT).as_posix()]=sha(p)
    shutil.copytree(ROOT/'paper',OUT/'pre_source')
    (OUT/'git_status_before.txt').write_text(git('status','--short'),encoding='utf-8')
    dump(target,dict(task='SP-U9B-1R',allowed=ALLOWED,paper_sources=paper,frozen_files=frozen,
         frozen_number_audit='outputs/sp_u9b1_abstract_number_audit.csv',number_audit_sha256=sha(ROOT/'outputs/sp_u9b1_abstract_number_audit.csv'),
         no_numeric_recalculation=True,title_frozen=True,bibliography_frozen=True,
         compiler_cache_exception='Reusable TeX cache is runtime state, excluded from scientific/artifact preservation'))
    print(json.dumps(dict(stage='prepare',paper_files=len(paper),frozen_files=len(frozen))))


def edit():
    assert (ROOT/'outputs/sp_u9b1r_preedit_manifest.json').exists()
    main=(OUT/'pre_source/main.tex').read_text(encoding='utf-8')
    old=abstract(main)
    new=replace_once(old,
        'Source-statistics mismatch can degrade linear minimum mean-square error (LMMSE) channel estimation when sparse pilots provide insufficient information to estimate a target covariance and no target-condition label is available online. We study a frozen Source empirical LMMSE estimator augmented by prepared statistical correction directions.',
        'When online pilots do not support reliable estimation of target channel statistics, distribution changes can render a source-prepared linear minimum mean-square error (LMMSE) estimator mismatched. We augment a frozen empirical LMMSE estimator prepared for a designated source channel condition (Source) with prepared statistical correction directions, without assuming an online target-condition label.')
    new=replace_once(new,'formal composed-shift evaluation','composed-shift evaluation')
    new=replace_once(new,'preregistered 0.5-dB criterion','predefined 0.5-dB compression criterion')
    new=replace_once(new,'Independent sealed validation','Independent held-out evaluation')
    main=replace_once(main,old,new)
    intro=(OUT/'pre_source/sections/01_introduction.tex').read_text(encoding='utf-8')
    intro=replace_once(intro,'A filter prepared for a source condition can become mismatched',
                       'A filter prepared for a designated source channel condition (Source) can become mismatched')
    intro=replace_once(intro,
        'Cross-pilot fitting is defined here as predicting the observations in one of four deterministic pilot folds from the other three folds, and fitting the correction coordinates to the resulting prediction residuals.',
        'Cross-pilot fitting minimizes a four-fold cross-pilot prediction objective over the existing pilots. Observations in each deterministic pilot fold are predicted from the other three folds, and the resulting residuals determine the correction coordinates.')
    intro=replace_once(intro,
        'Their objectives are accumulated over five consecutive frames, including the current frame and excluding future frames.',
        'The objective is accumulated over five consecutive frames, including the current frame. '+BUFFERING)
    intro=replace_once(intro,'Online operation uses received pilot observations,',
        'Online operation here denotes deployment-time adaptation from received pilot observations. The receiver uses received pilot observations,')
    old_evidence=('The evaluation separates a formal test on specified source and shifted conditions from an independent sealed validation on an unseen CDL-A profile condition. '
        'The formal test supports Source-relative and evaluated convex-combination comparisons for the composed shift, together with a preregistered compression criterion. '
        'The independent validation tests the frozen receiver at one additional profile, speed and delay-spread combination. '
        'Separate development studies indicate sensitivity to offline statistical budgets, subset realizations and pilot geometry; they are not additional formal replications. '
        'Post-hoc receiver diagnostics describe how estimation and detection metrics can differ for the fixed receiver, without serving as primary evidence or establishing a causal mechanism.')
    new_evidence=('The primary evaluation examines specified source and shifted conditions, alongside an independent held-out evaluation of one unseen CDL-A profile condition. '
        'The primary evaluation compares the estimator with Source and the evaluated convex combination under the composed shift and assesses compression against a predefined criterion. '
        'The independent evaluation tests the frozen receiver at one additional profile, speed and delay-spread combination. '
        'Additional sensitivity experiments examine offline statistical budgets, subset realizations and pilot geometry. '
        'Post-hoc receiver diagnostics describe how estimation and detection metrics can differ for the fixed receiver, without establishing a causal mechanism.')
    intro=replace_once(intro,old_evidence,new_evidence)
    old_c4=('We evaluate the frozen receiver using a formal test and an independent sealed unseen-profile validation, and use development sensitivity studies and post-hoc receiver diagnostics '
             'to describe the statistical-budget, pilot-geometry and receiver-scope limitations.')
    intro=replace_once(intro,old_c4,C4)
    intro=replace_once(intro,'describes the frozen evaluation protocol','describes the evaluation protocol')
    related=(OUT/'pre_source/sections/02_related_work.tex').read_text(encoding='utf-8')
    old_oz=('Adaptive low-rank LMMSE estimation for MIMO-OFDM was studied previously by \\"Ozdemir, Arslan and Arvas~\\cite{ozdemir2005,ozdemir2006}, '
       'and the original abstract of their 2006 paper explicitly discusses comb-type pilots and subspace tracking~\\cite{ozdemir2006}. '
       'The available public sources did not permit a full method-field audit of these works, so we make no priority claim over their adaptation variables or rank construction.')
    new_oz=('Adaptive low-rank LMMSE estimation for MIMO-OFDM was studied previously by \\"Ozdemir, Arslan and Arvas~\\cite{ozdemir2005,ozdemir2006}. '
       'Their 2006 work also addresses comb-type pilots and subspace tracking~\\cite{ozdemir2006}. '
       'Accordingly, our positioning does not rely on priority claims for adaptive or low-rank LMMSE estimation itself.')
    related=replace_once(related,old_oz,new_oz)
    related=replace_once(related,' This defines our compression object without inferring a contrasting object for the inaccessible methods.','')
    related=replace_once(related,'GMM and MFA remain closest statistical comparators; these interface distinctions do not establish performance superiority.',
        'GMM and MFA remain closest statistical comparators for the interface studied here.')
    related=replace_once(related,'The inspected arXiv v5 manuscript of Ha et al. uses','The arXiv v5 preprint of Ha et al. uses')
    related=replace_once(related,' That manuscript is cited as a versioned preprint rather than assigned an unverified journal publication status.','')
    disclosure=(' A faithful project-scale numerical comparison with the examined public MFA and A-MMSE implementations was not established under our fixed reproduction protocol; '
                'accordingly, we make no performance claim relative to those methods.')
    related=replace_once(related,disclosure,'')
    related=replace_once(related,', and it does not resolve the unavailable method fields of the \\"Ozdemir works.','.')
    for name,body in zip(ALLOWED,[main,intro,related]):(ROOT/name).write_text(body,encoding='utf-8')
    requirements=('# SP-U9B-1R: Discussion requirements for future mentor-authorized U9B-2\n\n'
       '## Modern baseline comparison limitation\n\n'
       '- 本文没有 MFA/A-MMSE matched-condition performance superiority evidence。\n'
       '- Reproduction/portability gate 不能解释为对方算法性能失败。\n'
       '- 因此不作 numerical superiority claim，也不跨 setup 比较文献数字。\n\n'
       '原 Related Work 的复现过程披露已从前半稿移除。本文件只记录未来 Discussion 必须处理的限制；本轮不修改 Discussion，不启动 U9B-2。\n'
       'U4/U5 仍为 sensitivity evidence；U7B-R1 仍为独立 held-out evaluation。Özdemir OPEN_SOURCE_LIMIT 及既有 source ledger 保持原始 bytes。\n')
    (ROOT/'docs/sp_u9b1r_discussion_requirements.md').write_text(requirements,encoding='utf-8')
    dump(OUT/'edit_record.json',dict(task='SP-U9B-1R',abstract=new,contributions=re.findall(r'\\item (.+)',intro),buffering_sentence=BUFFERING,
         before_words=dict(abstract=words(old),introduction=words((OUT/'pre_source/sections/01_introduction.tex').read_text(encoding='utf-8')),related_work=words((OUT/'pre_source/sections/02_related_work.tex').read_text(encoding='utf-8'))),
         after_words=dict(abstract=words(new),introduction=words(intro),related_work=words(related))))
    print(json.dumps(dict(stage='edit',words=load(OUT/'edit_record.json')['after_words']),ensure_ascii=False))


def compile_manuscript():
    stage=OUT/'build';stage.mkdir(exist_ok=True)
    env=dict(os.environ);env['TECTONIC_CACHE_DIR']=str(OLD/'tex_cache');env['NO_COLOR']='1'
    command=[str(OLD/'toolchain/tectonic.exe'),'--untrusted','--keep-logs','--keep-intermediates','--print','--outdir',str(stage),'main.tex']
    started=time.monotonic()
    r=subprocess.run(command,cwd=ROOT/'paper',env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (stage/'console.txt').write_text(r.stdout+'\nSTDERR:\n'+r.stderr,encoding='utf-8')
    dump(stage/'execution.json',dict(command=command,cwd=str(ROOT/'paper'),exit_code=r.returncode,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(stage='compile',exit_code=r.returncode)))
    if r.returncode:raise SystemExit(r.returncode)


def qa_render():
    from pypdf import PdfReader
    pdf=OUT/'build/main.pdf';reader=PdfReader(pdf)
    texts=[p.extract_text() or '' for p in reader.pages]
    refs=next(i+1 for i,t in enumerate(texts) if '\nReferences\n' in t or t.startswith('References\n'))
    front_end=next(i+1 for i,t in enumerate(texts) if 'System Model' in t)
    pages=sorted(set(range(1,front_end+1))|{refs,refs+1})
    poppler=Path(r'C:\Users\yuhua\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe')
    images=[]
    for page in pages:
        prefix=OUT/f'qa_page_{page:02d}'
        command=[str(poppler),'-f',str(page),'-l',str(page),'-singlefile','-scale-to','1800','-png',str(pdf),str(prefix)]
        r=subprocess.run(command,capture_output=True,text=True,check=True)
        png=prefix.with_suffix('.png');images.append(dict(page=page,path=str(png),sha256=sha(png),exit_code=r.returncode))
    (OUT/'pdf_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n'+t for i,t in enumerate(texts)),encoding='utf-8')
    dump(OUT/'visual_qa.json',dict(pdf=str(pdf),pdf_sha256=sha(pdf),pages_total=len(reader.pages),pdf_metadata=dict(reader.metadata),
         reviewed_pages=images,review_status='PENDING_DIRECT_VISUAL_INSPECTION',front_end=front_end,references_start=refs))
    print(json.dumps(dict(stage='render',pages=pages,total=len(reader.pages))))


def record_review():
    """Record direct image inspection already performed in this chat."""
    qa=load(OUT/'visual_qa.json')
    assert [x['page'] for x in qa['reviewed_pages']]==[1,2,3,4,15,16]
    assert sha(Path(qa['pdf']))==qa['pdf_sha256']
    for image in qa['reviewed_pages']:assert sha(Path(image['path']))==image['sha256']
    qa.update(review_status='PASS_DIRECT_VISUAL_INSPECTION',review_method='Writing agent directly inspected all six rendered page images in this chat',
        notes={'1':'Frozen title/author metadata; qualified problem and Source definition; complete Abstract after percent; ordinary scientific evaluation wording and unchanged numbers; legible.',
               '2':'Buffering/frame-causality/zero-latency scope sentence visible and wraps cleanly; contributions 1-4 complete; no clipping or overlap.',
               '3':'All six RW paragraphs, prior-art citations, process-language removal and final positioning visible; no overflow.',
               '4':'Unmodified System Model equations and transition into Method render normally.',
               '15':'References start and GMM/MFA/LS/version notes render legibly; frozen bibliography unchanged.',
               '16':'References continuation, Mei and both Ozdemir records render normally.'},
        preserved_placeholders='TODO Authors/Affiliations unchanged',scope='LaTeX manuscript was compiled; PDF skill used for read-only rendering and inspection')
    dump(OUT/'visual_qa.json',qa)
    print(json.dumps(dict(stage='review',verdict=qa['review_status'])))


def pytest_checks():
    previous=OUT/'pytest_receipt.json'
    history_path=OUT/'pytest_attempts.json'
    if previous.exists():
        history=load(history_path) if history_path.exists() else []
        history.append(dict(receipt=load(previous),log=(OUT/'pytest.txt').read_text(encoding='utf-8')))
        dump(history_path,history)
    command=[str(ROOT/'.venv-cdl/Scripts/python.exe'),'-X','utf8','-m','pytest','tests/test_sp_u9b1r.py','-q','--junitxml='+str(OUT/'pytest.xml')]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (OUT/'pytest.txt').write_text(result.stdout+'\nSTDERR:\n'+result.stderr,encoding='utf-8')
    passed=re.search(r'(\d+) passed',result.stdout)
    dump(OUT/'pytest_receipt.json',dict(command=command,exit_code=result.returncode,passed=int(passed[1]) if passed else 0,
         scope='Ten SP-U9B-1R source/provenance/language/build checks; no scientific experiment or historical generation tests'))
    print(result.stdout)
    if result.returncode:raise SystemExit(result.returncode)


def language_rows(stage):
    paths=[p for p in (ROOT/'paper').rglob('*.tex') if p.name!='preview.tex']
    rows=[]
    for path in sorted(paths):
        name=path.relative_to(ROOT).as_posix()
        if stage=='PRE':path=OUT/'pre_source'/path.relative_to(ROOT/'paper')
        body=path.read_text(encoding='utf-8')
        if name=='paper/main.tex':
            # Main.tex rendered prose is the Abstract; do not count comment/path tokens.
            first=body.index('\\begin{abstract}')
            allowed_lines=set(range(body[:first].count('\n')+1,body[:body.index('\\end{abstract}')].count('\n')+2))
        else:allowed_lines=None
        for term in TERMS:
            found=False
            pattern=r'(?<![A-Za-z])'+re.escape(term)+r'(?![A-Za-z])'
            for line,context in enumerate(body.splitlines(),1):
                if allowed_lines is not None and line not in allowed_lines:continue
                visible=printed(context)
                if re.search(pattern,visible,re.I):
                    found=True
                    internal=term in ['sealed','formal','preregistered','pre-registered','gate','PASS','FAIL','evidence tier','evidence role','USED validation','reporting-only','fixed reproduction protocol','post-hoc evidence']
                    if term=='evidence' and re.search(r'primary evidence|evidence (?:tier|role|hierarchy|level)|post-hoc evidence',visible,re.I):internal=True
                    if term in ['validation','post-hoc'] and re.search(r'\bsealed\b|\bformal\b',visible,re.I):internal=True
                    classification='INTERNAL_GOVERNANCE_REMOVE' if internal else 'SCIENTIFIC_NORMAL_KEEP'
                    in_scope=name in ALLOWED
                    action=('REPLACED_OR_REMOVED_IN_REVISION' if stage=='PRE' and in_scope and internal else
                            'DEFER_TO_U9B2_FROZEN_OUTSIDE_FRONT_HALF' if not in_scope and internal else
                            'KEEP_SCIENTIFIC_CONTEXT')
                    if stage=='POST' and in_scope and internal:action='UNRESOLVED_FRONT_HALF'
                    rows.append(dict(term=term,file=name,line=line,context=context,classification=classification,action=action,stage=stage,scope='FRONT_HALF' if in_scope else 'FROZEN_OUTSIDE_SCOPE'))
            if not found:rows.append(dict(term=term,file=name,line='',context='',classification='NOT_PRESENT',action='NONE',stage=stage,scope='FRONT_HALF' if name in ALLOWED else 'FROZEN_OUTSIDE_SCOPE'))
    # Bibliography titles can contain real-time: keep literal prior-work title.
    bib=ROOT/'paper/bibliography/references.bib' if stage=='POST' else OUT/'pre_source/bibliography/references.bib'
    for line,context in enumerate(bib.read_text(encoding='utf-8').splitlines(),1):
        if 'real-time' in context:rows.append(dict(term='real-time',file='paper/bibliography/references.bib',line=line,context=context,classification='SCIENTIFIC_NORMAL_KEEP',action='KEEP_PRIOR_WORK_TITLE',stage=stage,scope='FROZEN_BIBLIOGRAPHY'))
    return rows


def check():
    before=load(ROOT/'outputs/sp_u9b1r_preedit_manifest.json')
    current={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'paper').rglob('*') if p.is_file()}
    changed=[name for name,h in before['paper_sources'].items() if current.get(name)!=h]
    forbidden=sorted(set(changed)-set(ALLOWED))
    frozen_drift=[name for name,h in before['frozen_files'].items() if not (ROOT/name).exists() or sha(ROOT/name)!=h]
    assert set(changed)==set(ALLOWED) and not forbidden and not frozen_drift
    assert set(current)==set(before['paper_sources'])
    assert outer((ROOT/'paper/main.tex').read_text(encoding='utf-8'))==outer((OUT/'pre_source/main.tex').read_text(encoding='utf-8'))
    claim_checks=[]
    for name in ALLOWED:
        old=(OUT/'pre_source'/Path(name).relative_to('paper')).read_text(encoding='utf-8')
        new=(ROOT/name).read_text(encoding='utf-8')
        if name=='paper/main.tex':old,new=abstract(old),abstract(new)
        assert numeric(old)==numeric(new),(name,numeric(old),numeric(new))
        assert citations(old)==citations(new),name
        claim_checks.append(dict(file=name,numeric_claim_tokens=numeric(new),numeric_drift=False,citation_sequence_unchanged=True))
    with (ROOT/before['frozen_number_audit']).open(encoding='utf-8-sig',newline='') as f:existing=list(csv.DictReader(f))
    final_abstract=abstract((ROOT/'paper/main.tex').read_text(encoding='utf-8'))
    number_rows=[]
    for row in existing:
        assert re.search(r'(?<![\d.])'+re.escape(row['display_value'])+r'(?![\d.])',final_abstract),row['claim_id']
        number_rows.append(dict(claim_id=row['claim_id'],display_value=row['display_value'],check='UNCHANGED_DISPLAY_PRESENT'))
    assert sha(ROOT/before['frozen_number_audit'])==before['number_audit_sha256']
    dump(OUT/'claim_preservation.json',dict(claims=claim_checks,number_audit_reused=before['frozen_number_audit'],
        number_audit_sha256=before['number_audit_sha256'],number_rows=number_rows,scientific_numbers_recomputed=False,drift=False))
    language=language_rows('PRE')+language_rows('POST')
    with (ROOT/'outputs/sp_u9b1r_language_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(language[0]));writer.writeheader();writer.writerows(language)
    unresolved=[r for r in language if r['action']=='UNRESOLVED_FRONT_HALF']
    assert not unresolved,unresolved
    post=dict(task='SP-U9B-1R',allowed_files_changed=changed,forbidden_file_drift=forbidden,
        forbidden_files_unchanged=len(current)-len(changed),paper_file_count=len(current),frozen_files_unchanged=len(before['frozen_files']),frozen_drift=frozen_drift,
        main_abstract_only=True,pre_post={name:dict(pre=h,post=current[name]) for name,h in before['paper_sources'].items()})
    dump(ROOT/'outputs/sp_u9b1r_postedit_manifest.json',post)
    build=OUT/'build';log=(build/'main.log').read_text(encoding='utf-8');blg=(build/'main.blg').read_text(encoding='utf-8');aux=(build/'main.aux').read_text(encoding='utf-8')
    labels=re.findall(r'\\newlabel\{([^}]+)\}',aux)
    checks=dict(latex_errors=re.findall(r'(?m)^!.*',log),undefined_citations=re.findall(r'(?m)^.*Citation.*undefined.*$',log),
         undefined_references=re.findall(r'(?m)^.*(?:Reference.*undefined|undefined references).*$',log),
         overfull_boxes=re.findall(r'(?m)^Overfull.*$',log),duplicate_labels=sorted({x for x in labels if labels.count(x)>1}),
         bibtex_errors=re.findall(r'(?mi)^.*(?:warning--|error|I couldn.t|Repeated entry|not found).*$',blg))
    warnings=[line for line in log.splitlines() if 'Warning:' in line]
    baseline=load(ROOT/'outputs/sp_u9b1_compile_report.json')['post']['warnings']
    new_warnings=sorted(set(warnings)-set(baseline))
    assert not any(checks.values()) and not new_warnings,checks
    execution=load(build/'execution.json');assert execution['exit_code']==0
    qa=load(OUT/'visual_qa.json');assert qa['pdf_sha256']==sha(build/'main.pdf')
    dump(ROOT/'outputs/sp_u9b1r_compile_report.json',dict(task='SP-U9B-1R',execution=execution,checks=checks,
        NEW_warnings=new_warnings,PREEXISTING_warnings=sorted(set(warnings)&set(baseline)),
        fontconfig='PREEXISTING same runtime stderr as U9B-1',pdf=qa['pdf'],pdf_sha256=qa['pdf_sha256'],pages=qa['pages_total'],
        visual_qa=qa['review_status'],verdict='COMPILE_PASS'))
    diffs=[]
    import difflib
    for name in ALLOWED:
        old=(OUT/'pre_source'/Path(name).relative_to('paper')).read_text(encoding='utf-8').splitlines(True)
        new=(ROOT/name).read_text(encoding='utf-8').splitlines(True)
        diffs.extend(difflib.unified_diff(old,new,fromfile='pre/'+name,tofile='post/'+name))
    (OUT/'revision.diff').write_text(''.join(diffs),encoding='utf-8')
    (OUT/'git_tracked.diff').write_text(git('diff'),encoding='utf-8')
    (OUT/'git_diff_stat.txt').write_text(git('diff','--stat'),encoding='utf-8')
    checked=subprocess.run([str(GIT),'diff','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8');assert checked.returncode==0,checked.stdout
    (OUT/'git_diff_check.txt').write_text(checked.stdout+checked.stderr,encoding='utf-8')
    status=git('status','--short');(OUT/'git_status_after.txt').write_text(status,encoding='utf-8')
    dump(OUT/'check_report.json',dict(task='SP-U9B-1R',scope='PASS',numbers='PASS',citations='PASS',language='PASS',compile='PASS',visual=qa['review_status'],
         language_rows=len(language),removed_front_half_occurrences=sum(r['stage']=='PRE' and r['scope']=='FRONT_HALF' and r['classification']=='INTERNAL_GOVERNANCE_REMOVE' for r in language),
         unresolved_front_half_governance=unresolved,git_diff_check_exit=checked.returncode,
         tracked_diff_names=git('diff','--name-only').splitlines(),word_counts=load(OUT/'edit_record.json')['after_words']))
    print(json.dumps(dict(stage='check',changed=changed,frozen_files=len(before['frozen_files']),language_rows=len(language),new_warnings=new_warnings)))


def finish():
    qa=load(OUT/'visual_qa.json')
    assert qa['review_status']=='PASS_DIRECT_VISUAL_INSPECTION'
    check()
    test=load(OUT/'pytest_receipt.json');assert test['exit_code']==0
    report=load(OUT/'check_report.json');report.update(verdict='FRONT_HALF_PASS',tests=test,stop='STOP; mentor review; no U9B-2')
    dump(OUT/'final_report.json',report)
    compile_report=load(ROOT/'outputs/sp_u9b1r_compile_report.json');compile_report.update(visual_qa=qa,verdict='PASS')
    dump(ROOT/'outputs/sp_u9b1r_compile_report.json',compile_report)
    edits=load(OUT/'edit_record.json');post=load(ROOT/'outputs/sp_u9b1r_postedit_manifest.json')
    note=('# SP-U9B-1R change log\n\nFRONT_HALF_PASS\n\n'
       'Only Abstract, Introduction and Related Work changed. Title/metadata/bibliography and all later sections are byte-preserved.\n\n'
       'Scoped edits: qualify pilot-statistics problem; define Source at first use; replace formal/preregistered/sealed wording; rewrite contribution 4 as empirical characterization with separate sensitivity experiments; '
       'remove literature-access/process prose; relocate modern-baseline limitation to future Discussion requirements; clarify frame causality/current pilots/within-frame buffering/non-symbol-immediate/non-zero-latency scope.\n\n'
       f"Word counts before: {edits['before_words']}; after: {edits['after_words']}.\n\n"
       f"All scientific numeric tokens and citation sequences unchanged; all 19 existing Abstract audit displays retained; no numbers recomputed. {post['forbidden_files_unchanged']} paper files and {post['frozen_files_unchanged']} historical frozen files preserved.\n\n"
       f"Full manuscript compiled; no errors/undefined citations/undefined references/overfull boxes/duplicate labels/NEW warnings. Direct visual review pages {[x['page'] for x in qa['reviewed_pages']]}. Actual pytest: {test['passed']} passed, exit {test['exit_code']}.\n\n"
       'Historical source-limit ledgers remain unchanged. Frozen later sections were scanned without edits; comments and file-path tokens do not constitute manuscript prose. '
       'Post-hoc diagnostics remain descriptive; evidence levels unchanged. No experiment, bootstrap, literature search, method/rank/threshold/figure change or commit. STOP.\n')
    (ROOT/'docs/sp_u9b1r_change_log.md').write_text(note,encoding='utf-8')
    (OUT/'git_status_after.txt').write_text(git('status','--short'),encoding='utf-8')
    artifacts=[p for p in OUT.rglob('*') if p.is_file() and 'pre_source' not in p.parts]
    artifacts += [ROOT/'outputs'/name for name in ['sp_u9b1r_language_audit.csv','sp_u9b1r_preedit_manifest.json','sp_u9b1r_postedit_manifest.json','sp_u9b1r_compile_report.json']]
    artifacts += [ROOT/'docs/sp_u9b1r_change_log.md',ROOT/'docs/sp_u9b1r_discussion_requirements.md',ROOT/'scripts/run_sp_u9b1r.py',ROOT/'tests/test_sp_u9b1r.py']
    artifacts += [ROOT/name for name in ALLOWED]
    dump(OUT/'delivery_manifest.json',dict(task='SP-U9B-1R',verdict='FRONT_HALF_PASS',sha256={p.relative_to(ROOT).as_posix():sha(p) for p in artifacts if p.name!='delivery_manifest.json'}))
    print(json.dumps(dict(verdict='FRONT_HALF_PASS',tests=test['passed'],word_counts=edits['after_words'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','edit','compile','qa','review','check','pytest','finish']);args=parser.parse_args()
    {'prepare':prepare,'edit':edit,'compile':compile_manuscript,'qa':qa_render,'review':record_review,'check':check,'pytest':pytest_checks,'finish':finish}[args.stage]()
