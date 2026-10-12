"""Frozen-aggregate manuscript formatting and audit; no scientific code imports."""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/sp_u9b3'
EDIT = ['paper/sections/06_results.tex', 'paper/sections/07_discussion.tex',
        'paper/sections/08_conclusion.tex', 'paper/appendix/evidence_slots.tex']
ADD = ['paper/tables/u9b3_evidence_numbers.tex', 'paper/tables/u9b3_independent_cdl_a.tex',
       'paper/tables/u9b3_sensitivity_summary.tex']
RUNTIME = Path(r'C:\Users\yuhua\.cache\codex-runtimes\codex-primary-runtime\dependencies')
GIT = RUNTIME/'native/git/cmd/git.exe'
CONDITIONS={'CDL-C-30-100':'ID','CDL-B-30-100':'Profile','CDL-C-120-100':'Mobility',
            'CDL-B-120-100':'Composed','CDL-C-30-200':'Boundary'}
EVIDENCE = ['t20b_formal_nmse.csv', 't20b_formal_ber.csv', 't20b_formal_bootstrap.csv',
            't20b_formal_results.json', 'sp_u7b_r1/analysis.json', 'sp_u7b_r1/table_A.csv',
            'sp_u7b_r1/table_B.csv', 'sp_u7b_r1/final_verification.json',
            'sp_u4b_analysis.json', 'sp_u4b_across_replicates.csv', 'sp_u4b_summary.csv',
            'sp_u4b_table6_hypotheses.csv', 'sp_u4b_tails.csv', 'sp_u5a_analysis.json',
            'sp_u5a_nmse.csv', 'sp_u5a_table1_geometry.csv', 'sp_u5a_relative_sensitivity.csv',
            'sp_u5a_table5_paired.csv', 'sp_u5b_analysis.json', 'sp_u5b_nmse.csv',
            'sp_u5b_control.csv', 'sp_u5b_extra_sensitivity.csv', 'sp_u5b_ID_tradeoff.csv',
            'sp_u5b_gains.csv', 'sp_u5b_feedback_correction.json', 'sp_u8a/analysis.json',
            'sp_u8a/conditioning_summary.json', 'sp_u8a/correlations.csv',
            'sp_u9a_claim_evidence_matrix.csv', 'sp_u9a_claim_evidence_matrix.json']

def read(p): return p.read_text(encoding='utf-8-sig')
def load(p): return json.loads(read(p))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(n): return list(csv.DictReader((ROOT/'outputs'/n).open(encoding='utf-8-sig')))
def write_csv(n, data):
    with (ROOT/f'outputs/sp_u9b3_{n}.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def git(*args):
    return subprocess.run([str(GIT),*args],cwd=ROOT,capture_output=True,text=True,
                          encoding='utf-8',check=True).stdout
def prepare():
    dest=ROOT/'outputs/sp_u9b3_preedit_manifest.json'
    assert not dest.exists(), 'Immutable before snapshot already exists'
    OUT.mkdir(exist_ok=True)
    paper={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'paper').rglob('*') if p.is_file()}
    old=load(ROOT/'outputs/sp_u9b2_preedit_manifest.json')['frozen_files']
    names=set(old)
    for prefix in ['src','configs','scripts','tests','docs','outputs/sp_u9b2']:
        for p in (ROOT/prefix).rglob('*'):
            n=p.relative_to(ROOT).as_posix()
            if p.is_file() and '__pycache__' not in p.parts and 'sp_u9b3' not in n:
                names.add(n)
    for p in (ROOT/'outputs').glob('sp_u9b2_*'):
        if p.is_file():names.add(p.relative_to(ROOT).as_posix())
    names.update('outputs/'+n for n in EVIDENCE)
    names.update(n for n in paper if n not in EDIT)
    names.difference_update(EDIT)
    frozen={n:sha(ROOT/n) for n in sorted(names)}
    shutil.copytree(ROOT/'paper',OUT/'pre_source')
    (OUT/'git_status_before.txt').write_text(git('status','--short'),encoding='utf-8')
    dump(dest,dict(task='SP-U9B-3',allowed_edits=EDIT,allowed_additions=ADD,
                   paper_sources=paper,frozen_files=frozen,
                   exceptions=['Reusable compiler runtime cache; current-task output artifacts'],
                   scientific_code_execution=False,aggregate_reporting_only=True))
    print(json.dumps(dict(stage='prepare',paper_files=len(paper),frozen_files=len(frozen))))

def evidence():
    assert (ROOT/'outputs/sp_u9b3_preedit_manifest.json').exists()
    # Build source ledger BEFORE manuscript edits; all numbers below are frozen cells.
    catalog=[]
    claims=[]
    def add(name,value,source,locator,field,condition='',snr='',comparator='',metric='dB',
            aggregation='pooled error/truth energy then 10log10',N=1000,
            interval='point',position='Results',wording='',digits=3):
        name=name.replace('4','Four')
        value=float(value)
        rendered=f'{value:.{digits}f}' if digits else f'{value:.0f}'
        entry=dict(macro=name,raw_value=value,rendered=rendered,digits=digits,
                   source='outputs/'+source,source_sha256=sha(ROOT/'outputs'/source),
                   row_locator=locator,source_field=field,condition=condition,snr_db=snr,
                   comparator=comparator,metric=metric,aggregation=aggregation,
                   sampling_unit='trajectory' if N else 'specified implementation',N=N,
                   interval_type=interval,planned_location=position,
                   allowed_wording=wording or 'Descriptive frozen value with stated direction and scope')
        catalog.append(entry)
        claims.append({k:v for k,v in entry.items() if k not in ['macro','raw_value','rendered','digits']} |
                      dict(claim_id=name,claim_value=value))
        return '\\'+name+'{}'
    boot=rows('t20b_formal_bootstrap.csv')
    for r in boot:
        if r['metric']!='nmse_difference_db':continue
        condition=CONDITIONS[r['environment']];snr=int(float(r['snr_db']));suffix='Ten' if snr==10 else 'Twenty'
        comparison=r['comparison'];prefix={'source_gain':'SourceGain','convex_gain':'ConvexGain','compression_penalty':'Compression'}[comparison]
        locator=f'environment={r["environment"]};snr_db={snr};comparison={comparison};metric=nmse_difference_db'
        for field,tail in [('estimate',''),('ci95_lower','Lo'),('ci95_upper','Hi')]:
            add('Back'+prefix+condition+suffix+tail,r[field],'t20b_formal_bootstrap.csv',locator,field,
                condition,snr,comparison,interval='descriptive 95% paired percentile' if tail else 'point')
        for field,tail in [('primary_lower','LCB'),('primary_upper','UCB')]:
            if r[field] not in ['', 'nan','NaN','None']:
                add('Back'+prefix+condition+suffix+tail,r[field],'t20b_formal_bootstrap.csv',locator,field,
                    condition,snr,comparison,interval='main ten-cell family one-sided quantile '+('0.005' if tail=='LCB' else '0.995'),
                    wording='Positive gain lower bound; compression upper bound applies only to original Composed cells')
    nmse=rows('t20b_formal_nmse.csv');ber=rows('t20b_formal_ber.csv')
    def mainrow(condition,snr,method):
        return next(r for r in nmse if CONDITIONS[r['environment']]==condition and int(float(r['snr_db']))==snr and r['method']==method)
    for snr in [10,20]:
        suff='Ten' if snr==10 else 'Twenty'
        a=mainrow('Boundary',snr,'Final LR4 Affine');b=mainrow('Boundary',snr,'Matched LMMSE reference')
        add('BackBoundaryGap'+suff,float(a['nmse_db'])-float(b['nmse_db']),'t20b_formal_nmse.csv',
            f'Boundary;SNR={snr};Final LR4 Affine - Matched LMMSE reference','nmse_db subtraction',
            'Boundary',snr,'LR4 minus finite Matched',interval='descriptive point; no primary inference')
        for method,key in [('Source LMMSE','Source'),('Convex','Convex'),('Dense Affine','Dense'),
                           ('Final LR4 Affine','LR'),('Matched LMMSE reference','Matched')]:
            r=next(x for x in ber if CONDITIONS[x['environment']]=='Composed' and int(float(x['snr_db']))==snr and x['method']==method)
            add('BackBerComposed'+key+suff,r['ber'],'t20b_formal_ber.csv',f'Composed;snr_db={snr};method={method}',
                'ber','Composed',snr,method,metric='BER',aggregation='pooled bit errors / pooled bits',digits=6)
    cp=load(ROOT/'outputs/t20b_formal_results.json')['complexity']
    for name,field,digits in [('BackLRCost','total_arithmetic_proxy',0),('BackDenseCost','dense_affine_total_arithmetic_proxy',0),
                              ('BackSourceCost','source_complex_MAC_per_frame',0),('BackCostReduction','saving_vs_dense_affine_percent',5)]:
        add(name,cp[field],'t20b_formal_results.json','complexity',field,comparator='specified LR4/Dense/Source operations',
            metric='percent' if digits else 'complex MAC proxy',aggregation='arithmetic count',N=0,digits=digits)
    A=rows('sp_u7b_r1/table_A.csv');B=rows('sp_u7b_r1/table_B.csv')
    for r in A:
        snr=int(float(r['snr_db']));suff='Ten' if snr==10 else 'Twenty'
        for field,tail in [('pooled_nmse_db',''),('descriptive_ci_lower','Lo'),('descriptive_ci_upper','Hi')]:
            add('BackCDLA'+r['method']+suff+tail,r[field],'sp_u7b_r1/table_A.csv',f'snr_db={snr};method={r["method"]}',field,
                'CDL-A,120km/h,100ns',snr,r['method'],interval='descriptive 95% paired percentile' if tail else 'point',position='Results table / Appendix')
        if r['method']=='LR4':
            for field,tail in [('primary_gain_db','Gain'),('adjusted_one_sided_LCB','LCB')]:
                add('BackCDLA'+tail+suff,r[field],'sp_u7b_r1/table_A.csv',f'snr_db={snr};method=LR4',field,
                    'CDL-A,120km/h,100ns',snr,'Source minus LR4',
                    interval='independent two-cell family one-sided quantile 0.025' if tail=='LCB' else 'point')
    for r in B:
        snr=int(float(r['snr_db']));suff='Ten' if snr==10 else 'Twenty'
        record=r['record'];key=re.sub('[^A-Za-z]','',record)
        if r['method']:key+=r['method']
        for field,tail in [('point',''),('ci_lower','Lo'),('ci_upper','Hi')]:
            add('BackCDLA'+key+suff+tail,r[field],'sp_u7b_r1/table_B.csv',f'snr_db={snr};record={record};method={r["method"]}',field,
                'CDL-A,120km/h,100ns',snr,record,metric=r['units'],
                aggregation='pooled bit errors / pooled bits' if 'BER' in record else 'pooled energy NMSE dB difference',
                interval='descriptive 95% paired percentile' if tail else 'point',position='Results / Appendix',digits=6 if 'BER' in record else 3)
    # Budget distribution summaries are saved trajectory-dB statistics, not pooled primary endpoints.
    budget=rows('sp_u4b_across_replicates.csv')
    for r in budget:
        if r['method']!='LR4':continue
        suff='Ten' if float(r['snr_db'])==10 else 'Twenty';base='BackBudget'+r['condition']+suff+'N'+{'50':'Fifty','100':'Hundred','250':'TwoFifty','500':'FiveHundred'}[r['budget']]
        for field,tail in [('paired_penalty_replicate_median','Median'),('paired_penalty_range_min','Min'),('paired_penalty_range_max','Max'),('gain_replicate_median','Gain')]:
            if r[field] not in ['', 'nan','NaN']:
                add(base+tail,r[field],'sp_u4b_across_replicates.csv',f'{r["condition"]};snr={r["snr_db"]};budget={r["budget"]};LR4',field,
                    r['condition'],r['snr_db'],'budget minus own nested 500; gain Source minus LR4',
                    aggregation='median/range of five saved preparation-specific trajectory medians',N=100,position='Appendix',interval='descriptive; no confidence interval')
    for source,domain in [('sp_u5a_nmse.csv','Native'),('sp_u5b_nmse.csv','Common')]:
        for r in rows(source):
            snr=int(float(r['snr_db']));suff='Ten' if snr==10 else 'Twenty';df={'8':'Eight','16':'Sixteen','32':'ThirtyTwo'}[r['dF']]
            add('BackPilot'+domain+r['condition']+suff+df+r['method'],r['aggregate_nmse_db'],source,
                f'{r["condition"]};snr={snr};dF={r["dF"]};method={r["method"]}','aggregate_nmse_db',r['condition'],snr,r['method'],
                N=100,position='Results / Appendix',interval='descriptive pooled point')
    for r in rows('sp_u5b_control.csv'):
        suff='Ten' if float(r['snr_db'])==10 else 'Twenty'
        for field,tail in [('native_median_penalty','Native'),('common_median_penalty','Common')]:
            add('BackReduction'+r['condition']+suff+r['method']+tail,r[field],'sp_u5b_control.csv',
                f'{r["condition"]};snr={r["snr_db"]};{r["method"]}',field,r['condition'],r['snr_db'],'1/64 minus 1/32',
                aggregation='median of paired trajectory-dB differences',N=100,position='Results / Appendix',interval='descriptive distribution statistic')
    for r in rows('sp_u5b_extra_sensitivity.csv'):
        suff='Ten' if float(r['snr_db'])==10 else 'Twenty'
        for field,tail in [('common_median','Median'),('common_p95','Pninetyfive'),('common_positive_fraction','Positive')]:
            add('BackExtra'+r['condition']+suff+tail,r[field],'sp_u5b_extra_sensitivity.csv',
                f'{r["condition"]};snr={r["snr_db"]}',field,r['condition'],r['snr_db'],'LR4 reduction penalty minus Source reduction penalty',
                aggregation='paired trajectory-dB distribution; common 3840-coefficient domain',N=100,
                position='Results / Appendix',interval='descriptive; NOT LR4-minus-Dense compression loss')
    for r in rows('sp_u5b_ID_tradeoff.csv'):
        if r['dF']!='32':continue
        suff='Ten' if float(r['snr_db'])==10 else 'Twenty'
        for field,tail in [('median','Median'),('p95','Pninetyfive'),('maximum','Max')]:
            add('BackPilotIDTradeoff'+suff+tail,r[field],'sp_u5b_ID_tradeoff.csv',f'ID;snr={r["snr_db"]};dF=32',field,
                'ID',r['snr_db'],'LR4 minus Source',aggregation='paired trajectory-dB distribution; common domain',N=100,
                position='Results / Appendix',interval='descriptive distribution statistic')
    audit=load(ROOT/'outputs/sp_u8a/analysis.json')
    for r in audit['BER_summary']:
        suff='Ten' if r['snr_db']==10 else 'Twenty'
        add('BackDiagnostic'+r['method']+suff,r['aggregate_BER'],'sp_u8a/analysis.json',
            f'BER_summary;snr_db={r["snr_db"]};method={r["method"]}','aggregate_BER','CDL-A,120km/h,100ns',r['snr_db'],r['method'],
            metric='BER',aggregation='pooled bit errors / pooled bits',position='Appendix receiver diagnostic',
            interval='post-hoc descriptive point; no new interval',digits=6)
    qualitative=[
      ('Prepared and evaluation populations','B120 participated in historical clean validation rank selection; not in three-origin operator construction',
       'paper/sections/05_experimental_setup.tex;outputs/sp_u9b2_information_budget_audit.csv','Results / Discussion',
       'Composed uses independent evaluation trajectories, with development involvement disclosed'),
      ('Pilot information contract','online received pilots;known noise;common Source means;offline clean library and rank validation',
       'paper/sections/04_proposed_method.tex;paper/sections/05_experimental_setup.tex','Discussion','pilot-only online coordinates within prepared operators'),
      ('Compression contrast','Dense/LR4 share coordinates;only corrections change;Source/pilot predictors remain full',
       'paper/sections/04_proposed_method.tex','Results / Discussion','isolates correction approximation, not whole-estimator low rank'),
      ('Sensitivity design','five nested preparation realizations;100 evaluation trajectories reused;native domains differ;common changes scoring only',
       'paper/sections/05_experimental_setup.tex;outputs/sp_u4b_analysis.json;outputs/sp_u5b_analysis.json','Results / Appendix','descriptive sensitivity; no independent N=500; no single-factor causal effect'),
      ('Extra sensitivity identity','contrast (LR4_32-LR4_16)-(Source_32-Source_16);Dense absent',
       'scripts/run_sp_u5b.py;outputs/sp_u5b_extra_sensitivity.csv','Results / Discussion','extra deployed compressed-adaptation sensitivity; cannot isolate compression'),
      ('Receiver population','seven estimated paths plus Perfect CSI; main reference and CDL-A post-hoc receiver separate',
       'outputs/t20b_formal_ber.csv;outputs/sp_u8a/analysis.json','Results / Appendix','metric-dependent ordering with fixed unregularized 2x2 ZF'),
      ('Modern baseline scope','MFA/A-MMSE same-setup performance comparison unestablished',
       'outputs/sp_u9a_claim_evidence_matrix.csv;paper/sections/02_related_work.tex','Discussion','no superiority from feasibility limitations'),
      ('Mathematical scope','fixed-input identities and rank uniqueness not risk/BER guarantees;pilot/data objectives differ',
       'paper/sections/04_proposed_method.tex','Discussion','partial empirical compensation, finite-library/model scope')]
    for title,statement,source,pos,words in qualitative:
        assert all((ROOT/n).is_file() for n in source.split(';')),source
        claims.append(dict(source=source,source_sha256='see frozen manifest',row_locator=statement,source_field='source statement',
            condition='specified evaluation conditions',snr_db='10/20',comparator=title,metric='interpretation',aggregation='none',
            sampling_unit='trajectory where empirical',N='specified per experiment',interval_type='no new inference',
            planned_location=pos,allowed_wording=words,claim_id=title,claim_value=''))
    assert len({r['macro'] for r in catalog})==len(catalog)
    dump(OUT/'number_catalog.json',catalog)
    write_csv('claim_evidence_audit',claims)
    print(json.dumps(dict(stage='evidence',numbers=len(catalog),claim_rows=len(claims),new_statistics=0)))

def format_tables():
    catalog=load(OUT/'number_catalog.json')
    (ROOT/ADD[0]).write_text('% Generated from frozen aggregates; no metric or interval estimation.\n'+
        '\n'.join('\\newcommand{\\'+r['macro']+'}{'+r['rendered']+'}' for r in catalog)+'\n',encoding='utf-8')
    def m(name):return '\\'+name.replace('4','Four')+'{}'
    def table(caption,label,layout,header,lines,notes):
        return ('\\begingroup\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{longtable}{'+layout+'}\n'+
            '\\caption{'+caption+'}\\label{'+label+'}\\\\\n\\toprule\n'+header+' \\\\\n\\midrule\\endfirsthead\n\\toprule\n'+header+' \\\\\n\\midrule\\endhead\n'+
            '\n'.join(' & '.join(line)+' \\\\' for line in lines)+'\n\\bottomrule\n\\end{longtable}\n'+notes+'\n\\endgroup\n')
    lines=[]
    for suff,snr in [('Ten','10'),('Twenty','20')]:
        lines.append([snr,m('BackCDLAGain'+suff),m('BackCDLALCB'+suff),
            m('BackCDLALRminusDense'+suff)+' ['+m('BackCDLALRminusDense'+suff+'Lo')+', '+m('BackCDLALRminusDense'+suff+'Hi')+']',
            m('BackCDLALRminusMatched'+suff)+' ['+m('BackCDLALRminusMatched'+suff+'Lo')+', '+m('BackCDLALRminusMatched'+suff+'Hi')+']'])
    main=table('Independent CDL-A evaluation: pooled-energy NMSE differences (dB)', 'tab:independent', 'rrrrr',
        'SNR & Source--LR4 & Lower bound & LR4--Dense [95\\% CI] & LR4--Matched [95\\% CI]',lines,
        'The Source gains use the two-comparison adjusted one-sided lower bounds. Other intervals are descriptive paired-trajectory percentile intervals; positive LR4--Dense denotes a compression loss. All comparisons condition on the fixed library and the separate finite Matched reference.')
    full=[]
    for suff,snr in [('Ten','10'),('Twenty','20')]:
        for method in ['LS','Source','Hard','Convex','Dense','LR4','Matched']:
            n='BackCDLA'+method+suff;b='BackCDLABER'+method+suff
            full.append([snr,method,m(n)+' ['+m(n+'Lo')+', '+m(n+'Hi')+']',m(b)+' ['+m(b+'Lo')+', '+m(b+'Hi')+']'])
    appendix=table('Complete independent CDL-A NMSE and BER', 'tab:independent-full', 'rlrr',
        'SNR & Method & NMSE (dB) [95\\% CI] & BER [95\\% CI]',full,
        'Intervals are descriptive 95\\% paired-trajectory percentile intervals for 1000 evaluation trajectories. LS denotes LS with linear interpolation. Perfect-CSI receiver diagnostics are reported separately in Table~\\ref{tab:receiver-diagnostic}.')
    # Define appendix material without a second input of macros or duplicate labels.
    (ROOT/ADD[1]).write_text(main+'\n\\newcommand{\\IndependentFullResults}{%\n'+appendix+'}\n',encoding='utf-8')
    b=[]
    for condition in ['ID','Composed']:
        for suff,snr in [('Ten','10'),('Twenty','20')]:
            for budget,key in [('50','Fifty'),('100','Hundred'),('250','TwoFifty'),('500','FiveHundred')]:
                base='BackBudget'+condition+suff+'N'+key
                b.append([condition,snr,budget,m(base+'Median'), '['+m(base+'Min')+', '+m(base+'Max')+']',m(base+'Gain')])
    budgettable=table('Preparation-budget sensitivity of LR4 on fixed evaluation trajectories','tab:budget-sensitivity','lrrrrr',
        'Condition & SNR & Budget & Median penalty & Range & Median gain',b,
        'All values are in dB. Penalty is budget minus the corresponding nested 500-trajectory preparation. The central value and range summarize the five preparation-specific medians of paired trajectory differences; gain is the median across preparations of the trajectory-median Source--LR4 gain. Each preparation uses the same 100 evaluation trajectories per condition. These are distribution summaries, not the main pooled-energy endpoints or confidence intervals.')
    p=[]
    for condition in ['ID','Composed']:
        for suff,snr in [('Ten','10'),('Twenty','20')]:
            for frac,key,dim in [('1/16','Eight','3840'),('1/32','Sixteen','3968'),('1/64','ThirtyTwo','4032')]:
                prefix='BackPilot';end=condition+suff+key
                p.append([condition,snr,frac,dim,m(prefix+'Native'+end+'Source'),m(prefix+'Native'+end+'LR4'),m(prefix+'Common'+end+'Source'),m(prefix+'Common'+end+'LR4')])
    pilottable=table('Pilot-overhead sensitivity: native and common-domain pooled NMSE (dB)','tab:pilot-sensitivity','lrrrrrrr',
        'Condition & SNR & Fraction & Native dim. & Source & LR4 & Source common & LR4 common',p,
        'The native targets contain 3840/3968/4032 channel coefficients. The common domain contains 3840 coefficients. Common scoring reuses each geometry\'s received pilots, fitted coordinates and operators. Each condition uses 100 shared evaluation trajectories across the configurations; the control changes the score mask only.')
    q=[]
    for condition in ['ID','Composed']:
        for suff,snr in [('Ten','10'),('Twenty','20')]:
            q.append([condition,snr,m('BackReduction'+condition+suff+'LR4Native'),m('BackReduction'+condition+suff+'LR4Common'),
                m('BackReduction'+condition+suff+'SourceCommon'),m('BackExtra'+condition+suff+'Median'),m('BackExtra'+condition+suff+'Pninetyfive')])
    contrast=table('Trajectory-level cost of reducing pilot fraction from 1/32 to 1/64','tab:pilot-contrast','lrrrrrr',
        'Condition & SNR & LR4 native & LR4 common & Source common & Extra median & Extra 95th',q,
        'Values are descriptive dB distribution summaries. The first three quantities are medians of paired trajectory NMSE changes. Extra is the trajectory-wise LR4 change minus the Source change, reported as its median and 95th percentile on the common domain. It is additional sensitivity of the deployed compressed adaptation relative to Source, not a Dense--LR4 compression contrast.')
    (ROOT/ADD[2]).write_text(budgettable+'\n'+pilottable+'\n'+contrast,encoding='utf-8')
    print(json.dumps(dict(stage='format',new_tables=3,scientific_values_formatted=len(catalog),new_metric_or_CI_estimates=0)))

def preservation():
    before=load(ROOT/'outputs/sp_u9b3_preedit_manifest.json')
    now={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'paper').rglob('*') if p.is_file()}
    pd=[n for n,h in before['paper_sources'].items() if n not in EDIT and now.get(n)!=h]
    pd+=sorted(set(now)-set(before['paper_sources'])-set(ADD))
    fd=[n for n,h in before['frozen_files'].items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
    return before,now,pd,fd

def audit():
    before,now,pd,fd=preservation();assert not pd and not fd,(pd,fd)
    from audit_paper_style import scan_text
    style=[]
    for n in EDIT+ADD:
        s=read(ROOT/n)
        for cat,hits in scan_text(s,n).items():
            style.append(dict(file=n,check=cat,finding_count=len(hits),verdict='PASS' if not hits else 'FAIL',detail=json.dumps(hits)))
        printed=re.sub(r'(?m)(?<!\\)%.*$', '',s)
        printed=re.sub(r'\\(?:input|includegraphics|label|ref|eqref|cite)\{[^}]*\}', '',printed)
        hits=re.findall(r'\b(?:SP-U[\w-]*|sealed|unseal|firewall|mentor|branch|gate|PASS|FAIL|GPT|Codex|reporting-only|evidence tier)\b',printed,re.I)
        style.append(dict(file=n,check='additional_internal_labels',finding_count=len(hits),verdict='PASS' if not hits else 'FAIL',detail=json.dumps(hits)))
    write_csv('manuscript_style_audit',style)
    assert all(x['verdict']=='PASS' for x in style),[x for x in style if x['verdict']!='PASS']
    old=read(OUT/'pre_source/sections/06_results.tex');new=read(ROOT/EDIT[0])
    assert re.findall(r'(?s)\\begin\{figure\}.*?\\end\{figure\}',old)==re.findall(r'(?s)\\begin\{figure\}.*?\\end\{figure\}',new),'Frozen figure/caption block drift'
    # Validate every rendered macro against ledger; input graph detects missing references.
    included=set()
    def visit(p):
        if p in included:return ''
        included.add(p);s=read(p)
        return s+'\n'+'\n'.join(visit(ROOT/'paper'/(n if n.endswith('.tex') else n+'.tex')) for n in re.findall(r'\\input\{([^}]+)\}',s))
    alltex=visit(ROOT/'paper/main.tex');labels=re.findall(r'\\label\{([^}]+)\}',alltex)
    refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',alltex)
    assert len(labels)==len(set(labels));assert not set(refs)-set(labels)
    definitions=dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}',alltex))
    number=[]
    for x in load(OUT/'number_catalog.json'):
        expected=f'{x["raw_value"]:.{x["digits"]}f}' if x['digits'] else f'{x["raw_value"]:.0f}'
        assert definitions[x['macro']]==expected
        number.append(x|dict(actual_macro_expansion=definitions[x['macro']],used_in_source=len(re.findall(r'\\'+x['macro']+r'\b',alltex))>1,
                            rounding_ok=True,source_hash_unchanged=sha(ROOT/x['source'])==x['source_sha256'],verdict='PASS'))
    main=rows('t20b_formal_nmse.csv');boots=rows('t20b_formal_bootstrap.csv')
    lookup={(r['environment'],float(r['snr_db']),r['method']):float(r['nmse_db']) for r in main if r['nmse_db'] not in ['nan','NaN','']}
    directions={'source_gain':('Source LMMSE','Final LR4 Affine'), 'convex_gain':('Convex','Final LR4 Affine'),
                'compression_penalty':('Final LR4 Affine','Dense Affine')}
    arithmetic=[]
    for r in boots:
        if r['metric']!='nmse_difference_db':continue
        a,b=directions[r['comparison']];value=lookup[r['environment'],float(r['snr_db']),a]-lookup[r['environment'],float(r['snr_db']),b]
        residual=value-float(r['estimate']);assert abs(residual)<1e-10
        arithmetic.append(dict(condition=r['environment'],snr=r['snr_db'],contrast=r['comparison'],residual=residual))
    cp=load(ROOT/'outputs/t20b_formal_results.json')['complexity']
    assert abs(100*(1-cp['total_arithmetic_proxy']/cp['dense_affine_total_arithmetic_proxy'])-cp['saving_vs_dense_affine_percent'])<1e-10
    independent=rows('sp_u7b_r1/table_A.csv');secondary=rows('sp_u7b_r1/table_B.csv')
    for snr in ['10','20']:
        vals={r['method']:float(r['pooled_nmse_db']) for r in independent if r['snr_db']==snr}
        gain=next(r for r in independent if r['snr_db']==snr and r['method']=='LR4')
        assert abs(vals['Source']-vals['LR4']-float(gain['primary_gain_db']))<1e-10
        for record,a,b in [('LR4_minus_Dense','LR4','Dense'),('LR4_minus_Matched','LR4','Matched'),('Convex_minus_LR4','Convex','LR4')]:
            r=next(r for r in secondary if r['snr_db']==snr and r['record']==record)
            assert abs(vals[a]-vals[b]-float(r['point']))<1e-10
    write_csv('number_audit',number)
    # Also validate the pre-existing numeric macros actually reused by the rewritten prose.
    legacy=[]
    texts='\n'.join(read(ROOT/n) for n in EDIT)
    def legacy_check(name,value,source,field,context,digits=3):
        expected=f'{value:.{digits}f}'
        assert definitions[name]==expected,(name,definitions[name],expected)
        legacy.append(dict(macro=name,raw_value=value,rendered=expected,source=source,
            source_field=field,context=context,actual_macro_expansion=definitions[name],verdict='PASS'))
    for suff,snr in [('Ten',10),('Twenty',20)]:
        r=next(x for x in boots if CONDITIONS[x['environment']]=='ID' and float(x['snr_db'])==snr and x['comparison']=='source_gain' and x['metric']=='nmse_difference_db')
        legacy_check('IDLoss'+suff,-float(r['estimate']),'outputs/t20b_formal_bootstrap.csv','-estimate',f'ID;SNR={snr};Source minus LR4 sign inverted to positive loss')
        for key,method in [('Convex','Convex'),('Dense','Dense Affine')]:
            legacy_check('NmseComposed'+key+suff,lookup['CDL-B-120-100',snr,method],
                'outputs/t20b_formal_nmse.csv','nmse_db',f'Composed;SNR={snr};method={method}')
    parameters={'EvaluationSnrs':'10 and 20','CompressionMargin':'0.5','NumTrajectories':'1000',
                'ScoredFrames':'4--9','ConfidenceLevel':'95','BoundaryChannel':'CDL-C/30 km/h/200 ns'}
    for name,value in parameters.items():
        assert definitions[name]==value
        legacy.append(dict(macro=name,raw_value=value,rendered=value,source='paper/sections/05_experimental_setup.tex',
            source_field='frozen scientific definition',context='parameters retained in existing prose/figure captions',actual_macro_expansion=value,verdict='PASS'))
    write_csv('legacy_number_audit',legacy)
    # Source annotations cover every prose block, including unchanged scientific captions.
    from audit_manuscript_draft import sentence_records
    sentences=[];missing=[]
    for n in EDIT:
        records,uncovered=sentence_records(read(ROOT/n),n);missing+=uncovered
        for r in records:
            evidence_names=[x.split('#',1)[0] for x in r['evidence']]
            assert all((ROOT/x).is_file() for x in evidence_names),evidence_names
            sentences.append(dict(file=n,sentence=r['sentence'],category=r['category'],
                evidence=';'.join(r['evidence']),evidence_sha256=json.dumps({x:sha(ROOT/x) for x in evidence_names}),
                manual_semantic_review='scoped observation or declared definition; no new causal/guarantee claim',verdict='PASS'))
    assert not missing,missing
    write_csv('sentence_claim_audit',sentences)
    assert len(re.findall(r'\\input\{tables/u9b3_evidence_numbers\}',alltex))==1
    assert read(ROOT/'paper/main.tex')==read(OUT/'pre_source/main.tex')
    scope=[
      ('Main primary','1000 independent trajectories/condition; six scored frames; 10000 paired trajectory resamples', 'ten-cell family quantiles .005/.995; fixed library/reference', 'Results A/B/C'),
      ('Independent profile','1000 evaluation;500 separate matched-reference trajectories;single CDL-A120/100ns combination', 'two Source-gain endpoints LCB quantile .025;other95%CI descriptive', 'Results D'),
      ('Composed development','B120 excluded from Source/Profile/Mobility operator construction;used in historical rank selection', 'independent evaluation does not erase validation role', 'Results / Discussion'),
      ('Offline information','clean construction library1500 trajectories;clean validation includes rank choice;reference evaluation-only', 'online pilots/knownnoise/Source means/operators;no target cleanCSI/covariance/labels', 'Discussion'),
      ('Preparation sensitivity','budgets50/100/250/500;five nested preparations;reuse100evaluation/condition', 'N=100,not500;trajectory-dB median summaries distinct from pooled primary', 'Results F / Appendix'),
      ('Pilot overhead','fractions1/16,1/32,1/64;native3840/3968/4032;common3840', 'score-maskcontrolonly;geometry/folds/operators jointly change', 'Results F / Appendix'),
      ('Extra sensitivity','LR4 change minus Source change under reducedpilots;no Dense in contrast', 'additional compressed-adaptation sensitivity,not isolated compression penalty', 'Results F / Appendix'),
      ('Receiver','fixed unregularized2x2ZF,GrayQPSK;sevenestimatedpaths plusPerfectreceiver', 'BER secondary/descriptive;mainreference andposthocCDLA separate;bits do notincreaseN', 'Results G / Appendix'),
      ('Compression criterion','0.5dB only originalComposed10/20;Dense/LR4 shared coordinates', 'mixedCDLA effects andBoundary20penalty retained;no new success threshold', 'Results C/D/E'),
      ('Cost','specifiedcomplexMAC/correction-onlystorage proxies;Source full', 'no latency/energymeasurement;no faster-than-Source claim', 'Results C / Discussion'),
      ('Theory and competitors','rank uniqueness/fixedinput identities;MFA/AMMSEsame-setupperformanceunestablished', 'no risk/BER guarantee;no performance ranking fromfeasibility', 'Discussion'),
      ('Existing frozen caption shorthand','formal_ber caption says all eight estimators', 'text explicitly defines sevenestimatedpaths+Perfectreceiver;all3figureblocks preserved verbatim', 'Results G')]
    write_csv('information_scope_audit',[dict(topic=a,information=b,boundary=c,location=d,verdict='PASS') for a,b,c,d in scope])
    # Material pre-existing contradiction discovered during visual inspection of the retained appendix.
    # This task forbids editing the existing notation table or the approved Method.
    notation=read(ROOT/'paper/tables/notation.tex');method=read(ROOT/'paper/sections/04_proposed_method.tex')
    assert 'Real normal-equation Gram matrix and right-hand side' in notation
    assert '640\\times2' in method and 'S_j&=G_j^TG_j' in method
    findings=[
      dict(file='paper/tables/notation.tex',line=12,original='$h_p,p,z_j,h_d$ : clean pilot, pilot-LS, source-centered observation, data channel',
           evidence='paper/sections/04_proposed_method.tex:18,69-73',
           approved_definition='u_j=p_j-mu_p,s is complex64; z_j stacks real/imag held-out residual and is real640',
           impact='Retained appendix assigns the old source-centered meaning to z_j, contradicting the fitting response and deployment input in Method',
           minimal_suggestion='Replace source-centered z_j with u_j; add separate z_j real-response row'),
      dict(file='paper/tables/notation.tex',line=24,original='$G_j,u_j$ : Gram and RHS (2x2,2)',
           evidence='paper/sections/04_proposed_method.tex:71-81',
           approved_definition='G_j real640x2 design; S_j=G_j^T G_j real2x2 Gram; b_j=G_j^T z_j real2 RHS',
           impact='Retained appendix gives G_j and u_j incompatible meanings and dimensions; manuscript-wide notation consistency remains unresolved',
           minimal_suggestion='Replace old G_j/u_j Gram row with S_j/b_j; add G_j design and source-centered u_j rows'),
      dict(file='paper/tables/notation.tex',line=25,original='$C_{z,m},L_m$ : development pilot-plus-noise covariance and Cholesky factor',
           evidence='paper/sections/04_proposed_method.tex:94',
           approved_definition='C_m=R_pp,m+sigma^2 I=L_m L_m^H',
           impact='Table uses an unexplained historical covariance name; physical definition agrees but symbol should be aligned',
           minimal_suggestion='Use C_m for the same covariance; no construction or number change')]
    dump(OUT/'frozen_content_contradictions.json',dict(status='OPEN_FROZEN_NOTATION_CONFLICT',findings=findings,
        frozen_table_sha256=sha(ROOT/'paper/tables/notation.tex'),table_bytes_preserved=True,
        no_new_claim_uses_old_symbol_definitions=True,
        scientific_numbers_and_new_results_unaffected=True,delivery_verdict='REVISION_REQUIRED',
        required_next_authorization='Mentor review of a minimal notation-table alignment; no edit or next task initiated'))
    dump(OUT/'static_checks.json',dict(frozen_paper_drift=pd,historical_drift=fd,scientific_experiments=0,
        bootstrap_runs=0,frozen_figure_blocks_preserved=3,duplicate_labels=[],unresolved_source_refs=[],
        arithmetic_checks=arithmetic,number_macros=len(number),source_files=len(included),
        reused_legacy_macros=len(legacy),sentence_records=len(sentences),uncovered_prose=[],
        optional_figures_copied=0,legacy_tables_and_macros_preserved=True))
    print(json.dumps(dict(stage='audit',style_rows=len(style),number_rows=len(number),source_files=len(included),forbidden_drift=0)))

def checks():
    python=ROOT/'.venv-cdl/Scripts/python.exe'
    command=[str(python),'-X','utf8','-m','pytest','tests/test_paper_style.py','-q','-k',
        'forbidden_printed or nonprinted or legitimate_ai or printed_caption or unannotated or sentence_audit or workflow_source or new_editorial_scripts or style_guide',
        '--basetemp','outputs/sp_u9b3/pytest_tmp']
    r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (OUT/'pytest_console.txt').write_text(r.stdout+'\nSTDERR:\n'+r.stderr,encoding='utf-8')
    style_cmd=[str(python),'-X','utf8','scripts/audit_paper_style.py','--output','outputs/sp_u9b3/global_style_report.json']
    s=subprocess.run(style_cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (OUT/'global_style_console.txt').write_text(s.stdout+'\nSTDERR:\n'+s.stderr,encoding='utf-8')
    report=load(OUT/'global_style_report.json');reviewed=[]
    for hit in report['overclaim_hits']:
        assert hit['file']=='paper/sections/01_introduction.tex' and hit['text']=='universal',hit
        assert 'It is not a universal optimality guarantee' in read(ROOT/'paper/sections/01_introduction.tex')
        reviewed.append(dict(hit=hit,verdict='PREEXISTING_NEGATED_CLAIM_FALSE_POSITIVE',
            reason='Frozen approved Introduction explicitly rejects universal optimality; left unchanged'))
    assert not any(report[x] for x in ['internal_id_hits','meta_ai_hits','lab_notebook_style_hits','forbidden_terminology_hits','notation_conflicts'])
    from check_paper import script_firewall
    forbidden=script_firewall(read(Path(__file__)));assert not forbidden,forbidden
    dump(OUT/'global_style_manual_review.json',dict(raw_exit_code=s.returncode,reviewed_hits=reviewed,
        current_sections_zero_hits=True,appendix_and_new_tables_scanned=True,manual_semantic_verdict='PASS'))
    dump(OUT/'checks.json',dict(pytest=dict(command=command,exit_code=r.returncode,log='outputs/sp_u9b3/pytest_console.txt',
        scope='28 applicable scanner/annotation/workflow/style-guide tests;6 historical integrated draft checks deselected',
        summary=re.findall(r'\d+ passed[^\n]*',r.stdout)),global_style=dict(command=style_cmd,raw_exit_code=s.returncode,
        log='outputs/sp_u9b3/global_style_console.txt',manual_context_review='PASS'),
        new_script_forbidden_scientific_calls=forbidden,historical_whole_suite_run_this_task=False,
        all_historical_tests_claimed_passing=False,scientific_experiments=0,bootstrap_runs=0))
    print(json.dumps(dict(stage='checks',pytest_exit=r.returncode,pytest_summary=re.findall(r'\d+ passed[^\n]*',r.stdout),
        raw_style_exit=s.returncode,reviewed_false_positives=len(reviewed),scientific_calls=forbidden)))
    assert r.returncode==0 and '28 passed, 6 deselected' in r.stdout

def compile_manuscript():
    stage=OUT/'build';stage.mkdir(exist_ok=True)
    env=dict(os.environ);env['TECTONIC_CACHE_DIR']=str(ROOT/'outputs/sp_u9b1/tex_cache');env['NO_COLOR']='1'
    cmd=[str(ROOT/'outputs/sp_u9b1/toolchain/tectonic.exe'),'--untrusted','--keep-logs','--keep-intermediates','--print','--outdir',str(stage),'main.tex']
    start=time.monotonic();r=subprocess.run(cmd,cwd=ROOT/'paper',env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (stage/'console.txt').write_text(r.stdout+'\nSTDERR:\n'+r.stderr,encoding='utf-8')
    log=read(stage/'main.log') if (stage/'main.log').exists() else ''
    errors=re.findall(r'(?m)^!.*|Undefined control sequence|Emergency stop|Fatal error',log)
    undefined=re.findall(r'[^\n]*(?:undefined|multiply defined)[^\n]*',log,re.I)
    overfull=re.findall(r'Overfull \\[hv]box[^\n]*',log)
    dump(ROOT/'outputs/sp_u9b3_compile_report.json',dict(task='SP-U9B-3',command=cmd,cwd=str(ROOT/'paper'),exit_code=r.returncode,
        elapsed_seconds=time.monotonic()-start,latex_errors=errors,undefined_references_or_citations=undefined,
        duplicate_labels=[x for x in undefined if 'multiply defined' in x],overfull_boxes=overfull,
        warnings=re.findall(r'[^\n]*Warning[^\n]*',log),pdf=str(stage/'main.pdf'),log=str(stage/'main.log'),
        pdf_sha256=sha(stage/'main.pdf') if (stage/'main.pdf').exists() else None,visual_qa='PENDING'))
    print(json.dumps(dict(stage='compile',exit_code=r.returncode,errors=errors,undefined=undefined,overfull=overfull)))
    assert r.returncode==0 and not errors and not undefined and not overfull

def render():
    from pypdf import PdfReader
    texts=[p.extract_text() or '' for p in PdfReader(OUT/'build/main.pdf').pages]
    start=next(i+1 for i,t in enumerate(texts) if re.search(r'6\s+Results',t))
    bibliography=next(i+1 for i,t in enumerate(texts) if 'References' in t)
    # All pages from the first edited section through appendix (including boundary bibliography page).
    images=[]
    for page in range(start,len(texts)+1):
        prefix=OUT/f'qa_page_{page:02d}'
        subprocess.run([str(RUNTIME/'native/poppler/Library/bin/pdftoppm.exe'),'-f',str(page),'-l',str(page),
            '-singlefile','-scale-to','1800','-png',str(OUT/'build/main.pdf'),str(prefix)],capture_output=True,check=True)
        images.append(dict(page=page,path=str(prefix.with_suffix('.png')),sha256=sha(prefix.with_suffix('.png'))))
    (OUT/'pdf_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n'+t for i,t in enumerate(texts)),encoding='utf-8')
    dump(OUT/'visual_qa.json',dict(pages_total=len(texts),changed_start=start,bibliography_start=bibliography,
                                pages=images,status='PENDING_DIRECT_INSPECTION'))
    print(json.dumps(dict(stage='render',pages_total=len(texts),pages=[x['page'] for x in images])))

def qa_review():
    """Record the root reviewer's completed direct image inspection, not automated layout inference."""
    qa=load(OUT/'visual_qa.json')
    assert qa['pages_total']==22 and [x['page'] for x in qa['pages']]==list(range(10,23))
    notes={10:'Results and unchanged summary/cost tables; cost-table notes stay with table',
        11:'Independent profile table, boundary, sensitivities and BER readable; signs and intervals distinct',
        12:'Original NMSE figure and caption; no clipping',
        13:'Original paired comparisons figure and caption; primary/descriptive encodings retained',
        14:'Original BER figure and frozen caption; seven estimates plus reference explained in prose',
        15:'Discussion and three future questions; no orphan heading',
        16:'Both Conclusion paragraphs together; retained notation table readable but has documented frozen scientific symbol conflict',
        17:'Unchanged full NMSE longtable continuation; repeated header; no missing rows or clipping',
        18:'End of full NMSE and unchanged BER table; longtable continuation readable',
        19:'Full BER continuation and complete independent-profile table with NMSE/BER intervals',
        20:'Three sensitivity tables readable, native/common columns separate; table notes continue as normal prose',
        21:'Sensitivity notes, receiver diagnostic and bibliography transition readable',
        22:'Remaining frozen bibliography readable; no figure floats after references'}
    for x in qa['pages']:
        assert sha(Path(x['path']))==x['sha256']
        x.update(directly_viewed=True,layout_verdict='PASS',observations=notes[x['page']])
    qa.update(status='PASS_DIRECT_VISUAL_INSPECTION',scope='layout only; every changed page plus figures/bibliography boundary directly viewed',
        scientific_consistency='REVISION_REQUIRED: retained frozen notation conflicts with approved Method',
        automated_visual_pass_inferred=False,scientific_figures_redrawn=False,copied_optional_figures=0)
    dump(OUT/'visual_qa.json',qa)
    print(json.dumps(dict(stage='qa-review',layout_status=qa['status'],reviewed_pages=list(notes),scientific_status='REVISION_REQUIRED')))

def finish():
    before,now,pd,fd=preservation();assert not pd and not fd
    qa=load(OUT/'visual_qa.json');assert qa['status']=='PASS_DIRECT_VISUAL_INSPECTION'
    tests=load(OUT/'checks.json');assert tests['pytest']['exit_code']==0
    cp=load(ROOT/'outputs/sp_u9b3_compile_report.json');assert cp['exit_code']==0 and not cp['overfull_boxes']
    cp.update(visual_qa=qa['status'],pages_total=qa['pages_total'],reviewed_pages=[x['page'] for x in qa['pages']])
    cp['scientific_consistency']=qa['scientific_consistency']
    dump(ROOT/'outputs/sp_u9b3_compile_report.json',cp)
    changes={n:dict(before=before['paper_sources'].get(n,'ABSENT'),after=now[n]) for n in EDIT+ADD}
    (OUT/'git_status_after.txt').write_text(git('status','--short'),encoding='utf-8')
    artifacts=['scripts/run_sp_u9b3.py','outputs/sp_u9b3_preedit_manifest.json','outputs/sp_u9b3_compile_report.json']
    artifacts += [p.relative_to(ROOT).as_posix() for p in (ROOT/'outputs').glob('sp_u9b3_*.csv')]
    artifacts += ['outputs/sp_u9b3/static_checks.json','outputs/sp_u9b3/visual_qa.json','outputs/sp_u9b3/checks.json']
    contradictions=load(OUT/'frozen_content_contradictions.json')
    verdict=contradictions['delivery_verdict']
    artifacts.append('outputs/sp_u9b3/frozen_content_contradictions.json')
    dump(ROOT/'outputs/sp_u9b3_postedit_manifest.json',dict(task='SP-U9B-3',verdict=verdict,
        changes=changes,paper_sources=now,frozen_files=before['frozen_files'],paper_forbidden_drift=pd,historical_forbidden_drift=fd,
        preserved_paper_files=len(before['paper_sources'])-len(EDIT),preserved_historical_files=len(before['frozen_files']),
        current_task_artifact_sha256={n:sha(ROOT/n) for n in artifacts},scientific_experiments=0,new_bootstrap_runs=0,
        optional_figures_copied=0,git_commit=False,stopped_for_mentor_review=True,
        open_frozen_content_findings=contradictions['findings']))
    lines=['# SP-U9B-3 change log','',
      'REVISION_REQUIRED. Authorized Results, Discussion, Conclusion and appendix rewrite complete; three frozen-aggregate table files added. A retained frozen notation-table conflict blocks unconditional delivery approval.',
      'Claim-to-source and immutable before manifest generated before editing. No scientific code executed, statistical fit, new channel, bootstrap, experiment or threshold.',
      'Results follows Source adaptation -> unrestricted Composed comparison -> correction compression/cost -> independent CDL-A -> Boundary -> budget/pilot sensitivity -> secondary BER.',
      'Original 06: "under the held-out combination of profile and mobility shifts"; original 07: "held-out composed shift". Historical B120 clean validation participated in rank choice. Authorized revision distinguishes three-origin operator construction from rank selection and independent evaluation. Frozen Sections1–5 already disclose this; no frozen contradiction requiring an edit.',
      'U5 extra sensitivity is (LR4_32-LR4_16)-(Source_32-Source_16), not LR4-Dense. No Dense comparator exists in these pilot-overhead aggregates; prose records additional sensitivity of the compressed deployment without attributing its cause to compression.',
      'Retained ID losses, single-shift Convex ordering, Boundary20 penalty, CDL-A mixed Dense/LR4 effects, finite Matched gaps, ID sensitivity tails and receiver metric-dependent ordering.',
      'All original numerical/notation appendix tables retained. All existing figures/captions and macros unchanged; existing BER-caption shorthand all eight estimators contextualized as seven estimates plus Perfect-CSI receiver in new prose.',
      'No optional figure copy: independent CDL-A differences and full numerical appendix suffice. No scientific figure redraw.',
      'Discussion discloses clean library/rank-validation/reference roles, within-frame buffering, no explicit ICI, surrogate/data-risk difference, single unseen-profile scope and unestablished same-setup MFA/A-MMSE performance comparison.',
      'Future directions reduced to three evidence-driven questions; Conclusion written as two bounded paragraphs.', '', '## Before/after SHA256']
    lines += [f'- {n}: {v["before"]} -> {v["after"]}' for n,v in changes.items()]
    lines += ['',f'Frozen drift: paper {len(pd)}, historical {len(fd)}; preserved {len(before["paper_sources"])-len(EDIT)} existing paper files and {len(before["frozen_files"])} immutable files.',
      'OPEN_FROZEN_NOTATION_CONFLICT: notation.tex:12 uses z_j for Source-centered input;:24 uses G_j/u_j for Gram/RHS;:25 retains C_z,m. Approved Method defines u_j complex64 input, G_j real640x2 design, z_j real640 response, S_j/b_j Gram/RHS and C_m covariance. Original table preserved as instructed. See frozen_content_contradictions.json for original rows, evidence, impact and minimal proposed alignment. No new numerical claim uses the old definitions; no table/core-section change performed.',
      'Actual applicable pytest, global raw scan/manual review, compiler logs and direct visual QA: see current checks and compile report. Historical whole suite was not rerun; its prior failures are not claimed fixed.',
      'Stop for mentor review. No full-paper final audit, next task, submission, publishing or Git commit.']
    (ROOT/'docs/sp_u9b3_change_log.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    post=load(ROOT/'outputs/sp_u9b3_postedit_manifest.json')
    post['current_task_artifact_sha256']['docs/sp_u9b3_change_log.md']=sha(ROOT/'docs/sp_u9b3_change_log.md')
    post['current_task_artifact_sha256'].update({p.relative_to(ROOT).as_posix():sha(p) for p in OUT.rglob('*')
        if p.is_file() and 'pre_source' not in p.parts and 'pytest_tmp' not in p.parts})
    dump(ROOT/'outputs/sp_u9b3_postedit_manifest.json',post)
    print(json.dumps(dict(stage='finish',verdict=verdict,changes=changes,preserved_historical=len(before['frozen_files']))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','evidence','format','audit','checks','compile','render','qa-review','finish']);args=parser.parse_args()
    {'prepare':prepare,'evidence':evidence,'format':format_tables,'audit':audit,'checks':checks,'compile':compile_manuscript,'render':render,'qa-review':qa_review,'finish':finish}[args.stage]()
