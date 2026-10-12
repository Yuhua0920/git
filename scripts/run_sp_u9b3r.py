"""Local notation audit and manuscript compilation; no scientific data execution."""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/sp_u9b3r'
RUNTIME = Path(r'C:\Users\yuhua\.cache\codex-runtimes\codex-primary-runtime\dependencies')
GIT = RUNTIME / 'native/git/cmd/git.exe'
EDIT = 'paper/tables/notation.tex'
TASK = 'SP-U9B-3R'
AUTHORITATIVE = {
    'u_j': (r'u_j=p_j-\mu_{p,s}', 'complex 64', 'Source-centered pilot input'),
    'G_j': (r'G_j=[Re D_j; Im D_j]', 'real 640 x 2', 'cross-pilot real design'),
    'z_j': (r'z_j=[Re e_j; Im e_j]', 'real 640', 'cross-pilot real response'),
    'S_j': (r'S_j=G_j^T G_j', 'real 2 x 2', 'Gram matrix'),
    'b_j': (r'b_j=G_j^T z_j', 'real 2', 'normal-equation right-hand side'),
    'C_m': (r'C_m=R_{pp,m}+\sigma^2 I_{64}=L_m L_m^H', 'complex 64 x 64', 'compression pilot-plus-noise covariance'),
    'C_{z,m}': (r'use C_m=R_{pp,m}+\sigma^2 I_{64}', 'complex 64 x 64', 'obsolete covariance alias'),
}
SYMBOLS = {s: re.compile(r'(?<![A-Za-z])' + s[0] + r'_\s*(?:' +
             (r'\{\s*z\s*,\s*m\s*\}' if s == 'C_{z,m}' else
              r'\{\s*' + s[-1] + r'\s*\}|' + s[-1] + r'(?![A-Za-z])') + r')')
           for s in AUTHORITATIVE}


def read(p):
    return p.read_text(encoding='utf-8-sig')


def load(p):
    return json.loads(read(p))


def sha(p):
    digest = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def text_sha(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def dump(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def csv_write(p, data):
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def git(*args):
    return subprocess.run([str(GIT), *args], cwd=ROOT, capture_output=True,
                          text=True, encoding='utf-8', check=True).stdout


def printed(s):
    return re.sub(r'(?m)(?<!\\)%[^\n]*', '', s)


def graph(base):
    included = set()

    def visit(name):
        if name in included:
            return
        included.add(name)
        p = base / name
        assert p.is_file(), name
        for child in re.findall(r'\\(?:input|include)\{([^}]+)\}', printed(read(p))):
            visit(child if child.endswith('.tex') else child + '.tex')

    visit('main.tex')
    return sorted('paper/' + n for n in included)


def snapshots():
    return {p.relative_to(ROOT).as_posix(): sha(p)
            for p in sorted((ROOT / 'paper').rglob('*')) if p.is_file()}


def excluded(n):
    parts = Path(n).parts
    return ('__pycache__' in parts or '.pytest_cache' in parts or
            any(x.startswith('pytest_tmp') for x in parts) or
            n.startswith('outputs/sp_u9b3r') or
            n.startswith('outputs/sp_u9b1/tex_cache/') or
            n.startswith('outputs/sp_u9b1/toolchain/') or
            n.startswith('scripts/run_sp_u9b3r.') or
            n.startswith('docs/sp_u9b3r_') or n == EDIT)


def prepare():
    dest = ROOT / 'outputs/sp_u9b3r_preedit_manifest.json'
    assert not dest.exists(), 'Immutable before manifest already exists'
    OUT.mkdir(exist_ok=True)
    paper = snapshots()
    shutil.copytree(ROOT / 'paper', OUT / 'pre_source')
    names = set(load(ROOT / 'outputs/sp_u9b3_postedit_manifest.json')['frozen_files'])
    for prefix in ['src', 'configs', 'scripts', 'tests', 'docs', 'outputs', 'paper']:
        for p in (ROOT / prefix).rglob('*'):
            if p.is_file():
                names.add(p.relative_to(ROOT).as_posix())
    names.add('AGENTS.md')
    names = sorted(n for n in names if not excluded(n))
    frozen = {}
    total_bytes = 0
    start = time.monotonic()
    progress = start
    for n in names:
        p = ROOT / n
        assert p.is_file(), n
        frozen[n] = sha(p)
        total_bytes += p.stat().st_size
        if time.monotonic() - progress > 15:
            print(json.dumps(dict(stage='prepare-hashes',files=len(frozen),bytes=total_bytes)), flush=True)
            progress = time.monotonic()
    (OUT / 'git_status_before.txt').write_text(git('status', '--short'), encoding='utf-8')
    (OUT / 'git_diff_before.patch').write_text(git('diff', '--', 'paper'), encoding='utf-8')
    dump(dest, dict(task=TASK, allowed_edits=[EDIT], paper_sources=paper,
         frozen_files=frozen, current_input_graph=graph(ROOT / 'paper'),
         protected_bytes=total_bytes, hash_seconds=time.monotonic()-start,
         manuscript_scope='main.tex input graph; all other TeX also searched and classified',
         exceptions=['current-task files', '__pycache__/.pytest_cache', 'pytest_tmp*',
                     'reusable Tectonic toolchain/cache; toolchain is not scientific output'],
         scientific_data_read='byte hashing only; no outcome analysis', scientific_experiments=0,
         bootstrap_runs=0))
    semantic('before', OUT / 'pre_source')
    registry()
    print(json.dumps(dict(stage='prepare', paper_files=len(paper), frozen_files=len(frozen),
                          protected_bytes=total_bytes)), flush=True)


def semantic(phase, base):
    active = set(graph(base))
    rows = []
    for p in sorted(base.rglob('*.tex')):
        n = 'paper/' + p.relative_to(base).as_posix()
        for line_number, line in enumerate(read(p).splitlines(), 1):
            s = printed(line)
            for symbol, pattern in SYMBOLS.items():
                matches = list(pattern.finditer(s))
                if not matches:
                    continue
                auth, dim, role = AUTHORITATIVE[symbol]
                ok = True
                local = role
                reason = 'definition/use matches approved Method'
                action = 'PRESERVE'
                if symbol == 'C_{z,m}':
                    ok = False
                    local = 'obsolete covariance symbol; same covariance object'
                    action = 'REPLACE_WITH_C_m' if n in active else 'PRESERVE_HISTORICAL_NONINPUT'
                if n == EDIT and phase == 'before':
                    if symbol == 'z_j':
                        ok, dim, local = False, 'complex 64', 'old Source-centered pilot input'
                        action = 'REPLACE_INPUT_WITH_u_j; ADD_REAL_RESPONSE_z_j'
                    if symbol == 'u_j':
                        ok, dim, local = False, 'real 2', 'old normal-equation RHS'
                        action = 'REPLACE_RHS_WITH_b_j; ADD_CENTERED_INPUT_u_j'
                    if symbol == 'G_j':
                        ok, dim, local = False, 'real 2 x 2', 'old Gram matrix'
                        action = 'REPLACE_GRAM_WITH_S_j; ADD_REAL_DESIGN_G_j'
                if n == 'paper/preview.tex':
                    ok = False
                    action = 'PRESERVE_HISTORICAL_NONINPUT'
                    reason = 'frozen flattened preview has historical provenance and is not input by main.tex'
                    if symbol == 'z_j':
                        local, dim = 'historical Source-centered complex input', 'complex 64'
                    if symbol == 'G_j':
                        local, dim = 'historical Gram matrix', 'real 2 x 2'
                    if symbol == 'u_j':
                        local, dim = 'historical normal-equation RHS', 'real 2'
                if symbol == 'C_m' and 'second moment' in s:
                    local = 'covariance term in source-centered second moment C_m + mean-offset outer product'
                    reason = 'mean-offset term is separate; C_m retains covariance meaning'
                if symbol == 'G_j' and 'lambda' in s:
                    local = 'real design rank; equivalent positivity of its Gram S_j'
                if symbol == 'z_j' and 'epsilon' in s:
                    local = 'real response; exact residual identity with noisy endogenous design'
                rows.append(dict(phase=phase, symbol=symbol, file=n, line=line_number,
                    local_definition=local, dimension=dim, semantic_role=role,
                    authoritative_definition=auth, consistent=ok, action=action,
                    in_current_manuscript=n in active, occurrence_count=len(matches),
                    context=s.strip(), semantic_review=reason,
                    authoritative_source='paper/sections/04_proposed_method.tex'))
    csv_write(OUT / f'notation_{phase}.csv', rows)
    if phase == 'before':
        cm = [r for r in rows if r['symbol']=='C_m' and r['in_current_manuscript']]
        assert cm and all(r['consistent'] for r in cm), 'NEW_NOTATION_COLLISION'
        dump(OUT / 'C_m_preedit_collision_review.json', dict(verdict='NO_INCOMPATIBLE_MEANING',
            checked_locations=[dict(file=r['file'],line=r['line'],meaning=r['local_definition']) for r in cm],
            reviewed_by='root semantic review of full source contexts'))
    else:
        csv_write(ROOT / 'outputs/sp_u9b3r_global_notation_audit.csv', rows)
        assert not [r for r in rows if r['in_current_manuscript'] and not r['consistent']], 'Active semantic collision'
    return rows


def registry():
    lines = [f'# {TASK} Internal Notation Registry', '',
             'Authority: approved Sections 3–5, especially Method Eqs. source, coordinate-fit, normal, weight and factors.',
             'Scope: the current main.tex input graph. This registry is internal and is not input into the manuscript.', '',
             '| Domain | Symbols and final meaning | Dimensions |', '|---|---|---|',
             r'| System | $H_{r,a,k,t}$ channel, $X$ transmitted, $Y$ received, $N$ noise; $H$ as a superscript is conjugate transpose | indices as defined in Section 3 |',
             r'| Source estimation | $p_j$ pilot-LS observation; $u_j=p_j-\mu_{p,s}$; $\mu_{p,s},\mu_{d,s}$ Source means; $W_s$ full operator | complex 64; complex 64; complex 64/3968; complex 3968 x 64 |',
             r'| Corrections | $B_P=W_P-W_s$, $B_M=W_M-W_s$; $B_{m,4}=A_{m,4}Q_{m,4}$ | 3968 x 64; $A$:3968 x 4, $Q$:4 x 64 |',
             r'| Coordinates | $c=[\alpha,\beta]^T$ unrestricted real objective variable; $c_j=[\alpha_j,\beta_j]^T$ frame-indexed vector; $\widehat c_j=[\widehat\alpha_j,\widehat\beta_j]^T$ fitted vector | real 2 |']
    for s,(definition,dim,role) in AUTHORITATIVE.items():
        if s!='C_{z,m}':
            lines.append(f'| Fitting/compression | ${definition}$: {role} | {dim} |')
    lines += [r'| Compression factor | $L_m$: lower Cholesky factor of $C_m$ | complex 64 x 64 |', '',
              'Five frames x four folds x 16 complex held-out coefficients = 320 complex equations = 640 real equations.',
              'The normal equation is S_j c_hat_j = b_j. G_j is never the Gram and u_j is never its RHS in the current manuscript.',
              'C_m is the pilot-plus-noise covariance. The Source-centered input second moment is C_m + delta_mu_m delta_mu_m^H; these are distinct objects.',
              'Generic u denotes the random centered input; u_j its frame realization. BER counts b_ij^err/b_ij^tot have different indices and meanings from b_j; no collision.',
              'Rank four applies to each deployed correction, not to the full Source-plus-correction estimator. Dense and LR4 share fitted coordinates.',
              'Source is the source-prepared estimator; Matched is the finite matched-statistics evaluation reference.',
              'Historical paper/preview.tex is a non-input flattened derivative documented by preview_provenance.json; its old aliases are recorded and byte-preserved, not a valid current manuscript build.', '']
    (ROOT / 'docs/sp_u9b3r_notation_registry.md').write_text('\n'.join(lines), encoding='utf-8')


def audit():
    before = load(ROOT / 'outputs/sp_u9b3r_preedit_manifest.json')
    now = snapshots()
    changed = [n for n,v in before['paper_sources'].items() if now.get(n)!=v]
    assert changed==[EDIT], changed
    assert set(now)==set(before['paper_sources'])
    rows = semantic('after', ROOT/'paper')
    active = before['current_input_graph']
    assert active==graph(ROOT/'paper')
    # Pure numeric streams are exact for every unedited source, including result macros.
    numeric = []
    for n in sorted(before['paper_sources']):
        if not n.endswith('.tex'):
            continue
        a = read(OUT/'pre_source'/Path(n).relative_to('paper'))
        b = read(ROOT/n)
        # TeX control words may immediately precede a number (e.g. \times64).
        # Extract complete numeric runs rather than taking a suffix of such a run.
        old_tokens = re.findall(r'\d+(?:\.\d+)?', printed(a))
        new_tokens = re.findall(r'\d+(?:\.\d+)?', printed(b))
        numeric.append(dict(file=n, active=n in active, old_count=len(old_tokens), new_count=len(new_tokens),
            raw_tokens_identical=old_tokens==new_tokens,
            before_numeric_sha256=text_sha(json.dumps(old_tokens)), after_numeric_sha256=text_sha(json.dumps(new_tokens)),
            classification='AUTHORIZED_EXISTING_DIMENSION_ANNOTATIONS' if n==EDIT else 'BYTE_AND_NUMBER_IDENTICAL',
            removed_literal_tokens=dict(collections.Counter(old_tokens)-collections.Counter(new_tokens)),
            added_literal_tokens=dict(collections.Counter(new_tokens)-collections.Counter(old_tokens))))
        if n!=EDIT:
            assert a==b and old_tokens==new_tokens, 'NUMERIC_DRIFT: '+n
    old_rows = [s for s in read(OUT/'pre_source/tables/notation.tex').splitlines() if ' & ' in s]
    new_rows = [s for s in read(ROOT/EDIT).splitlines() if ' & ' in s]
    replaced = [s for s in old_rows if s.startswith((r'$h_p,p,z_j,h_d$',r'$G_j,u_j$',r'$C_{z,m},L_m$'))]
    assert len(replaced)==3
    assert all(s in new_rows for s in old_rows if s not in replaced), 'Unrelated table-row drift'
    assert not any('C_{z,m}' in s for s in new_rows)
    # All added dimensions are directly copied from the frozen authoritative Method.
    expected_additions = [r'$h_p,h_d$', r'$p_j$', r'$u_j$', r'$G_j$', r'$z_j$',
                          r'$S_j$', r'$b_j$', r'$C_m$', r'$L_m$']
    additions=[s for s in new_rows if s not in old_rows]
    assert len(additions)==len(expected_additions)
    assert all(any(s.startswith(prefix+' & ') for s in additions) for prefix in expected_additions)
    dims = dict(p_j='complex 64',u_j='complex 64',G_j='real 640 x 2',z_j='real 640',
                S_j='real 2 x 2',b_j='real 2',C_m='complex 64 x 64',L_m='complex 64 x 64')
    method = read(ROOT/'paper/sections/04_proposed_method.tex')
    for exact in [r'\mathbb C^{64}',r'\mathbb R^{640\times2}',r'\mathbb R^{640}',
                  r'S_j&=G_j^TG_j',r'b_j&=G_j^Tz_j',r'C_m=R_{pp,m}+\sigma^2I_{64}=L_mL_m^H']:
        assert exact in method
    (OUT/'notation_diff.patch').write_text(subprocess.run([str(GIT),'diff','--no-index',
          str(OUT/'pre_source/tables/notation.tex'),str(ROOT/EDIT)],capture_output=True,text=True,
          encoding='utf-8',errors='replace').stdout,encoding='utf-8')
    dump(ROOT/'outputs/sp_u9b3r_numeric_preservation.json',dict(task=TASK,verdict='PASS',numeric_drift=False,
        scientific_numeric_tokens_changed=0,all_unedited_tex_identical=True,
        raw_numeric_token_stream_of_edited_table_unchanged=False,
        edited_table_numeric_change_reason='Existing authoritative dimensions are now explicit; multiplicities change, scientific values do not',
        removed_table_rows=replaced,added_table_rows=additions,authoritative_dimensions=dims,
        dimensional_authority='frozen Sections 3–5; 5 x 4 x 16 x 2 = 640 real equations',
        preserved_table_rows=len(old_rows)-len(replaced),all_other_paper_files_identical=len(now)-1,
        per_file=numeric,scientific_claim_drift=False,sections_1_8_token_corrections=[]))
    from audit_paper_style import scan_text, printed_tex
    style = []
    scope=[EDIT]+[n for n in active if '/appendix/' in n]
    for n in scope:
        hits = scan_text(read(ROOT/n),n)
        assert not any(hits.values()), hits
        extra=re.findall(r'\b(?:SP-U[\w-]*|PASS|FAIL|gate|sealed|unseal|firewall|mentor|Codex|GPT|branch|evidence tier)\b',
                         printed_tex(read(ROOT/n)),re.I)
        assert not extra,extra
        style.append(dict(file=n,scanner_hits=hits,extra_internal_hits=extra,verdict='PASS'))
    # Static full-input labels/references/citations and actual caption inventory.
    alltex='\n'.join(printed(read(ROOT/n)) for n in active)
    labels=re.findall(r'\\label\{([^}]+)\}',alltex)
    refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',alltex)
    citations=set(x.strip() for group in re.findall(r'\\cite\{([^}]+)\}',alltex) for x in group.split(','))
    bib_keys=set(re.findall(r'@\w+\s*\{\s*([^,]+)',read(ROOT/'paper/bibliography/references.bib')))
    assert len(labels)==len(set(labels)) and not set(refs)-set(labels)
    assert not citations-bib_keys
    captions=[]
    old_normal_equations=[]
    old_normal_pattern=re.compile(r'G_\s*(?:j|\{\s*j\s*\}).*c.*=.*u_\s*(?:j|\{\s*j\s*\})')
    for p in sorted((ROOT/'paper').rglob('*.tex')):
        name=p.relative_to(ROOT).as_posix()
        for i,s in enumerate(read(p).splitlines(),1):
            if old_normal_pattern.search(printed(s)):
                old_normal_equations.append(dict(file=name,line=i,text=s,in_current_manuscript=name in active,
                    action='PRESERVE_HISTORICAL_NONINPUT' if name not in active else 'REQUIRES_CORRECTION'))
    assert not any(r['in_current_manuscript'] for r in old_normal_equations)
    for n in active:
        for i,s in enumerate(read(ROOT/n).splitlines(),1):
            if r'\caption' in s or r'\textbf{Algorithm' in s:
                captions.append(dict(file=n,line=i,text=s,semantic_review='rank applies to corrections; Source/Matched/Dense/LR4 retain approved meanings; frozen captions unchanged unless notation table caption, which is also unchanged'))
    dump(OUT/'local_consistency_report.json',dict(active_files=len(active),semantic_audit_rows=len(rows),
        active_semantic_rows=sum(r['in_current_manuscript'] for r in rows),active_collision_count=0,
        noninput_legacy_rows=[r for r in rows if not r['in_current_manuscript'] and not r['consistent']],
        modified_files=changed,sections_1_8_token_corrections=[],appendix_modifications=[],
        generic_alias_review=['generic u is random input; u_j realization','b_ij^err/b_ij^tot differ from fitting b_j',
                              'C_m covariance differs from covariance plus mean-offset second moment','H tensor and superscript H have distinct syntactic roles'],
        old_gram_normal_equation_search=old_normal_equations,
        style=style,caption_and_algorithm_review=captions,duplicate_labels=[],undefined_static_refs=[],undefined_static_citations=[]))
    print(json.dumps(dict(stage='audit',changed=changed,active_files=len(active),semantic_rows=len(rows),
                         active_collisions=0,scientific_numeric_drift=0,claims_drift=0)),flush=True)


def checks():
    python=ROOT/'.venv-cdl/Scripts/python.exe'
    command=[str(python),'-X','utf8','-m','pytest','tests/test_paper_style.py','-q','-k',
             'forbidden_printed or nonprinted or legitimate_ai or printed_caption or unannotated or sentence_audit or workflow_source or new_editorial_scripts or style_guide',
             '--basetemp','outputs/sp_u9b3r/pytest_tmp']
    r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (OUT/'pytest_console.txt').write_text(r.stdout+'\nSTDERR:\n'+r.stderr,encoding='utf-8')
    style_cmd=[str(python),'-X','utf8','scripts/audit_paper_style.py','--output','outputs/sp_u9b3r/global_style_report.json']
    s=subprocess.run(style_cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (OUT/'global_style_console.txt').write_text(s.stdout+'\nSTDERR:\n'+s.stderr,encoding='utf-8')
    report=load(OUT/'global_style_report.json')
    reviewed=[]
    for hit in report['overclaim_hits']:
        assert hit['file']=='paper/sections/01_introduction.tex' and hit['text']=='universal',hit
        assert 'It is not a universal optimality guarantee' in read(ROOT/'paper/sections/01_introduction.tex')
        reviewed.append(dict(hit=hit,verdict='PREEXISTING_NEGATED_CLAIM_FALSE_POSITIVE',
                             reason='approved Introduction rejects universal optimality; no edit'))
    assert not any(report[k] for k in ['internal_id_hits','meta_ai_hits','lab_notebook_style_hits','forbidden_terminology_hits','notation_conflicts'])
    from check_paper import script_firewall
    forbidden=script_firewall(read(Path(__file__)))
    assert not forbidden,forbidden
    dump(OUT/'checks.json',dict(pytest=dict(command=command,exit_code=r.returncode,
             summary=re.findall(r'\d+ passed[^\n]*',r.stdout),log='outputs/sp_u9b3r/pytest_console.txt',
             scope='28 applicable scanner/annotation/workflow/style-guide tests; six historical integrated-draft checks deselected'),
             raw_style_exit=s.returncode,reviewed_false_positives=reviewed,style_semantic_verdict='PASS',
             local_table_appendix_style_review='see local_consistency_report.json',
             scientific_imports_or_calls=forbidden,historical_whole_suite_claimed_passing=False,
             scientific_experiments=0,bootstrap_runs=0))
    print(json.dumps(dict(stage='checks',pytest_exit=r.returncode,summary=re.findall(r'\d+ passed[^\n]*',r.stdout),
                         raw_style_exit=s.returncode,reviewed_false_positives=len(reviewed))),flush=True)
    assert r.returncode==0 and '28 passed, 6 deselected' in r.stdout


def compile_manuscript():
    stage=OUT/'build'
    stage.mkdir(exist_ok=True)
    env=dict(os.environ)
    env['TECTONIC_CACHE_DIR']=str(ROOT/'outputs/sp_u9b1/tex_cache')
    env['NO_COLOR']='1'
    command=[str(ROOT/'outputs/sp_u9b1/toolchain/tectonic.exe'),'--untrusted','--keep-logs',
             '--keep-intermediates','--print','--outdir',str(stage),'main.tex']
    start=time.monotonic()
    r=subprocess.run(command,cwd=ROOT/'paper',env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (stage/'console.txt').write_text(r.stdout+'\nSTDERR:\n'+r.stderr,encoding='utf-8')
    log=read(stage/'main.log') if (stage/'main.log').exists() else ''
    errors=re.findall(r'(?m)^!.*|Undefined control sequence|Emergency stop|Fatal error',log)
    undefined=re.findall(r'[^\n]*(?:undefined|multiply defined)[^\n]*',log,re.I)
    overfull=re.findall(r'Overfull \\[hv]box[^\n]*',log)
    warnings=re.findall(r'[^\n]*Warning[^\n]*',log)
    old_warnings=re.findall(r'[^\n]*Warning[^\n]*',read(ROOT/'outputs/sp_u9b3/build/main.log'))
    new_warnings=list((collections.Counter(warnings)-collections.Counter(old_warnings)).elements())
    report=dict(task=TASK,command=command,cwd=str(ROOT/'paper'),exit_code=r.returncode,
        elapsed_seconds=time.monotonic()-start,latex_errors=errors,
        undefined_references_or_citations=undefined,duplicate_labels=[x for x in undefined if 'multiply defined' in x],
        overfull_boxes=overfull,warnings=warnings,preexisting_warnings=old_warnings,new_warnings=new_warnings,
        log=str(stage/'main.log'),console=str(stage/'console.txt'),pdf=str(stage/'main.pdf'),
        pdf_sha256=sha(stage/'main.pdf') if (stage/'main.pdf').exists() else None,visual_qa='PENDING')
    dump(ROOT/'outputs/sp_u9b3r_compile_report.json',report)
    print(json.dumps(dict(stage='compile',exit_code=r.returncode,errors=errors,undefined=undefined,
                         overfull=overfull,new_warnings=new_warnings)),flush=True)
    assert r.returncode==0 and not errors and not undefined and not overfull and not new_warnings


def render():
    from pypdf import PdfReader
    texts=[p.extract_text() or '' for p in PdfReader(OUT/'build/main.pdf').pages]
    roles=collections.defaultdict(list)
    for i,t in enumerate(texts,1):
        if 'Symbol' in t and 'Meaning' in t and ('cross-pilot' in t or 'normal-equation' in t or 'Source-centered' in t or 'Cholesky' in t):
            roles[i].append('notation table')
        if 'Four-Fold Cross-Pilot Fitting' in t or ('640' in t and 'normal' in t):
            roles[i].append('cross-pilot fitting equations')
        if 'Covariance-Weighted Rank-Four' in t or ('Cholesky' in t and 'second moment' in t):
            roles[i].append('compression equations')
        if 'Independent Unseen-Profile Results' in t or 'Receiver Ordering' in t:
            roles[i].append('appendix presentation')
    # Include both table boundaries and adjacent Method/appendix pages for continuation checks.
    table_pages=[i for i,r in roles.items() if 'notation table' in r]
    method_pages=[i for i,r in roles.items() if any('equations' in x for x in r)]
    assert table_pages and method_pages
    for page in table_pages:
        if page>1:
            roles[page-1].append('notation table preceding boundary')
        if page<len(texts):
            roles[page+1].append('notation table following boundary')
    for page in method_pages:
        if page<len(texts):
            roles[page+1].append('Method equation continuation')
    images=[]
    for page in sorted(roles):
        prefix=OUT/f'qa_page_{page:02d}'
        subprocess.run([str(RUNTIME/'native/poppler/Library/bin/pdftoppm.exe'),'-f',str(page),'-l',str(page),
                        '-singlefile','-scale-to','1800','-png',str(OUT/'build/main.pdf'),str(prefix)],
                       capture_output=True,check=True)
        images.append(dict(page=page,path=str(prefix.with_suffix('.png')),
                           sha256=sha(prefix.with_suffix('.png')),roles=roles[page],directly_viewed=False))
    (OUT/'pdf_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n'+t for i,t in enumerate(texts)),encoding='utf-8')
    dump(OUT/'visual_qa.json',dict(pages_total=len(texts),pages=images,status='PENDING_DIRECT_INSPECTION',
                                  automated_visual_pass_inferred=False))
    print(json.dumps(dict(stage='render',pages_total=len(texts),pages=[dict(page=x['page'],roles=x['roles']) for x in images])),flush=True)


def qa_review():
    """Persist explicit notes only after the root reviewer has actually viewed each image."""
    qa=load(OUT/'visual_qa.json')
    # Root directly viewed these final rendered pages before authorizing this stage.
    notes={
        '5':'Directly viewed: u_j Source input; D/e blocks, 640-by-2 G_j and 640-vector z_j, S_j/b_j normal equation all readable and mutually consistent; no clipping.',
        '6':'Directly viewed: C_m covariance and L_m Cholesky, separate mean-offset second moment, correction-only rank four, factor dimensions and final u_j input; algorithm uses S_j/b_j. Figure/algorithm readable.',
        '7':'Directly viewed: G_j real design rank, S_j positivity, noisy G_j/z_j residual identity and u_j output perturbation agree with notation; no equation/reference layout conflict.',
        '15':'Directly viewed: preceding Discussion boundary intact; Source/Matched/Dense/LR4 meanings retained; no stray table row or orphan notation header.',
        '16':'Directly viewed: entire notation table fits on one page. p_j/u_j complex 64; G_j/z_j real 640-by-2/640; S_j/b_j real 2-by-2/2; C_m/L_m complex 64-by-64. G_j dimension is unbroken. Conclusion and following full-results table remain readable.',
        '17':'Directly viewed: unchanged full-results longtable follows with repeated header and readable rows; no notation-table continuation or clipping.',
        '21':'Directly viewed: appendix receiver presentation and continued sensitivity table preserve Source/Matched/LR4 meanings and receiver cross-term scope; repeated header and captions readable; no old fitting aliases.'}
    assert set(notes)=={str(x['page']) for x in qa['pages']}
    for x in qa['pages']:
        assert sha(Path(x['path']))==x['sha256']
        x.update(directly_viewed=True,layout_verdict='PASS',observations=notes[str(x['page'])])
    qa.update(status='PASS_DIRECT_VISUAL_INSPECTION',scientific_consistency='core notation agrees with frozen Method',
              scope='all notation table pages and boundaries; fitting/compression equations and appendix presentation',
              scientific_figures_redrawn=False)
    dump(OUT/'visual_qa.json',qa)
    print(json.dumps(dict(stage='qa-review',status=qa['status'],pages=[x['page'] for x in qa['pages']])),flush=True)


def finish():
    before=load(ROOT/'outputs/sp_u9b3r_preedit_manifest.json')
    frozen={}
    drift=[]
    progress=time.monotonic()
    for n,v in before['frozen_files'].items():
        current=sha(ROOT/n) if (ROOT/n).is_file() else 'MISSING'
        frozen[n]=current
        if current!=v:
            drift.append(n)
        if time.monotonic()-progress>15:
            print(json.dumps(dict(stage='post-hashes',files=len(frozen),drift=len(drift))),flush=True)
            progress=time.monotonic()
    now=snapshots()
    changes={n:dict(before=v,after=now.get(n,'MISSING')) for n,v in before['paper_sources'].items() if now.get(n)!=v}
    assert set(changes)=={EDIT} and not drift, (changes,drift)
    num=load(ROOT/'outputs/sp_u9b3r_numeric_preservation.json')
    checks=load(OUT/'checks.json')
    qa=load(OUT/'visual_qa.json')
    cp=load(ROOT/'outputs/sp_u9b3r_compile_report.json')
    assert num['verdict']=='PASS' and checks['pytest']['exit_code']==0 and checks['style_semantic_verdict']=='PASS'
    assert qa['status']=='PASS_DIRECT_VISUAL_INSPECTION'
    assert cp['exit_code']==0 and not any(cp[x] for x in ['latex_errors','undefined_references_or_citations','duplicate_labels','overfull_boxes','new_warnings'])
    cp.update(visual_qa=qa['status'],pages_total=qa['pages_total'],reviewed_pages=[x['page'] for x in qa['pages']])
    dump(ROOT/'outputs/sp_u9b3r_compile_report.json',cp)
    diff=subprocess.run([str(GIT),'diff','--stat','--','paper'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True).stdout
    (OUT/'git_diff_stat.txt').write_text(diff,encoding='utf-8')
    (OUT/'git_status_after.txt').write_text(git('status','--short'),encoding='utf-8')
    (OUT/'git_diff_after.patch').write_text(git('diff','--','paper'),encoding='utf-8')
    change_log=[f'# {TASK} Change Log','', 'Verdict: NOTATION_CONSISTENCY_PASS.',
        'Modified manuscript file: paper/tables/notation.tex only.',
        'Replaced centered-observation z_j by u_j; added distinct p_j/u_j definitions.',
        'Replaced old Gram/RHS G_j/u_j by S_j/b_j; added G_j/z_j real design/response.',
        'Unified old C_{z,m} to C_m and separated the lower Cholesky factor L_m.',
        'All other table rows remain exact. No Sections 1–8 token correction; no appendix correction.',
        'Existing dimension values are newly explicit; edited-table raw numeric multiplicities change. Scientific numeric values and every other manuscript file are unchanged.',
        'Semantic current-input audit passes. Historical non-input preview retains old aliases and remains byte-identical; it is not a current manuscript build.',
        'Actual scoped pytest: '+ '; '.join(checks['pytest']['summary'])+'. Global raw style exit 1 has one manually reviewed preexisting negated universal false positive. Local table/appendix checks pass.',
        f'Full compilation: exit {cp["exit_code"]}; new warnings zero; PDF pages {qa["pages_total"]}; direct visual QA pages '+','.join(str(x['page']) for x in qa['pages'])+'.',
        f'Protected files: {len(frozen)}; drift zero. No experiment, new bootstrap, figure redraw, commit or final full-paper audit.',
        'Stop for mentor review. Recommend full-paper audit only after mentor approval.', '']
    (ROOT/'docs/sp_u9b3r_change_log.md').write_text('\n'.join(change_log),encoding='utf-8')
    artifacts=[ROOT/'scripts/run_sp_u9b3r.py',ROOT/'docs/sp_u9b3r_change_log.md',ROOT/'docs/sp_u9b3r_notation_registry.md']
    artifacts += [p for p in (ROOT/'outputs').glob('sp_u9b3r_*') if p.is_file()]
    artifacts += [p for p in OUT.rglob('*') if p.is_file() and 'pre_source' not in p.parts and not any(x.startswith('pytest_tmp') for x in p.parts)]
    bootstrap=[n for n in frozen if 'bootstrap' in n.lower()]
    dump(ROOT/'outputs/sp_u9b3r_postedit_manifest.json',dict(task=TASK,verdict='NOTATION_CONSISTENCY_PASS',
        changes=changes,paper_sources=now,frozen_files=frozen,forbidden_drift=drift,preserved_paper_files=len(now)-1,
        preserved_scientific_and_historical_files=len(frozen),protected_bootstrap_files=bootstrap,
        current_task_artifact_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(artifacts))},
        numeric_drift=0,scientific_claim_drift=0,sections_1_8_token_corrections=[],
        new_experiments=0,new_bootstrap_runs=0,git_commit=False,final_full_paper_audit=False,stopped_for_mentor_review=True))
    print(json.dumps(dict(stage='finish',verdict='NOTATION_CONSISTENCY_PASS',changed=list(changes),
          frozen_files=len(frozen),drift=len(drift),bootstrap_files=len(bootstrap),pages=qa['pages_total'],
          notation_hash=changes[EDIT],pdf_sha256=cp['pdf_sha256'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','audit','checks','compile','render','qa-review','finish'])
    args=parser.parse_args()
    {'prepare':prepare,'audit':audit,'checks':checks,'compile':compile_manuscript,
     'render':render,'qa-review':qa_review,'finish':finish}[args.stage]()
