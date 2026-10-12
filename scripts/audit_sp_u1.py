"""Read-only science/metadata audit; writes only new SP-U1 audit outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from check_paper import bibliography_metadata, script_firewall
from build_paper_preview import bib_entries
from audit_paper_style import audit as style_audit
from sp_u1_plans import REVIEW_HASHES

ROOT=Path(__file__).resolve().parents[1]
REQUIRED_DOCS=('signal_processing_venue_profile','literature_search_log','closest_work_matrix',
    'signal_processing_recent_paper_patterns','novelty_collision_gate','reviewer_attack_matrix',
    'theory_feasibility','modern_baseline_gate','offline_statistics_study_plan',
    'pilot_conditioning_study_plan','ood_study_gate','upgrade_value_cost_matrix',
    'paper1_paper2_boundary','title_positioning_review','projected_contributions',
    'recommended_upgrade_plan','advisor_review_packet')
CLASSES={'ALREADY_ANSWERED','WRITING_FIX','THEORY_NEEDED','SMALL_EXPERIMENT',
         'MODERATE_EXPERIMENT','LARGE_NEW_RESEARCH','OUT_OF_SCOPE'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def metadata_errors(database):
    fields=database['fields']
    works=database['works']
    errors=[]
    if len(works)<15:errors.append('fewer than15 unique relevant records')
    if sum(2023<=w['year']<=2026 for w in works)<8:errors.append('fewer than8 recent works')
    if sum(bool(w.get('high_core_proximity')) for w in works)<5:errors.append('fewer than5 core works')
    if len({w['id'] for w in works})!=len(works):errors.append('duplicate work IDs')
    if len({w['title'].casefold() for w in works})!=len(works):errors.append('duplicate version/title counting')
    identifiers=[]
    for w in works:
        errors += [w['id']+':missing:'+f for f in fields if f not in w or w[f] in ('',None)]
        identifier=w.get('DOI/arXiv','')
        identifiers.append(identifier.casefold())
        if not re.search(r'10\.\d{4,9}/\S+|\d{4}\.\d{4,5}(?:v\d+)?|urn:nbn:',identifier):
            errors.append(w['id']+':unrecognized identifier')
        if not w.get('read_locators') or not w.get('source_url','').startswith('https://'):
            errors.append(w['id']+':no inspected source/locator')
        if w['year']>2026:errors.append(w['id']+':after cutoff')
    if len(set(identifiers))!=len(identifiers):errors.append('duplicate identifiers')
    oz=next((w for w in works if w['id']=='oz2006'),None)
    if oz is None or oz['source quality']!='ABSTRACT_ONLY' or 'UNKNOWN' not in oz['uncertainty']:
        errors.append('legacy full-text uncertainty lost')
    return errors

def integrity(root):
    baseline=read(root/'outputs/sp_u1_before_manifest.json')
    expected=baseline['tracked_file_sha256']
    mismatch=[p for p,h in expected.items() if not (root/p).is_file() or sha(root/p)!=h]
    groups={prefix:sum(p.startswith(prefix) for p in expected) for prefix in ('outputs/','paper/','src/','configs/','tests/')}
    return {'accepted_t20g_commit':baseline['accepted_t20g_commit'],
            'checked_tracked_files':len(expected),'group_file_counts':groups,
            'byte_sha256_mismatches':mismatch,'unexpected_historical_mutation_count':len(mismatch),
            'science_and_manuscript_unchanged':not mismatch}

def audit(root=ROOT,finalize=False):
    root=Path(root)
    problems=[]
    database=read(root/'outputs/sp_u1_literature_database.json')
    problems += metadata_errors(database)
    for name in REQUIRED_DOCS:
        path=root/f'docs/sp_u1_{name}.md'
        if not path.is_file() or len(path.read_text(encoding='utf-8'))<100:
            problems.append('missing/empty deliverable:'+name)
    packet=(root/'docs/sp_u1_advisor_review_packet.md').read_text(encoding='utf-8')
    if re.findall(r'^## ([A-M])\.',packet,re.M)!=list('ABCDEFGHIJKLM'):
        problems.append('advisor packet must have only A–M ordered sections')
    reports=[read(root/f'outputs/sp_u1_reviewer{i}_frozen.json') for i in (1,2,3)]
    review_mismatches=[p for p,h in REVIEW_HASHES.items() if sha(root/p)!=h]
    problems += ['frozen reviewer mutated:'+p for p in review_mismatches]
    for i,r in enumerate(reports,1):
        if len(r['concerns'])<10:problems.append(f'reviewer{i}:under10 concerns')
        if sum(c.get('severity','').lower()=='major' for c in r['concerns'])<10:
            problems.append(f'reviewer{i}:under10 major concerns')
        if any(c['classification'] not in CLASSES for c in r['concerns']):
            problems.append(f'reviewer{i}:unrecognized feasibility class')
    firewall=[]
    for path in sorted((root/'scripts').glob('*sp_u1*.py')):
        firewall += [path.name+':'+s for s in script_firewall(path.read_text(encoding='utf-8'))]
    problems += firewall
    status=integrity(root)
    problems += ['historical byte mutation:'+p for p in status['byte_sha256_mismatches']]
    bib=bibliography_metadata(root)  # read-only function; do not call check_paper.check().
    bibtext=(root/'paper/bibliography/references.bib').read_text(encoding='utf-8')
    keys=re.findall(r'@\w+\s*\{\s*([^,]+),',bibtext)
    entries=bib_entries(bibtext)
    duplicate_keys=sorted({k for k in keys if keys.count(k)>1})
    seen={};duplicate_identifiers=[]
    for key,entry in entries.items():
        for kind in ('doi','eprint','identifier'):
            if value:=entry.get(kind):
                norm=(kind,value.lower().removeprefix('https://doi.org/').strip())
                if norm in seen:duplicate_identifiers.append([seen[norm],key,kind,value])
                seen[norm]=key
    problems += ['bibliography:'+s for s in bib['missing_or_invalid']]
    if duplicate_keys or duplicate_identifiers:problems.append('BibTeX duplicates')
    save(root/'outputs/sp_u1_bibliography_audit.json',dict(bib,
        duplicate_keys=duplicate_keys,duplicate_identifiers=duplicate_identifiers,
        manuscript_bibtex_mutated=False))
    style=style_audit(root,output=root/'outputs/sp_u1_paper_style_audit.json')
    if not style['all_lexical_checks_passed']:problems.append('existing manuscript style audit failed')
    code=read(root/'outputs/sp_u1_author_code_access.json')
    if len(code)!=3 or any(c['author_code_executed'] for c in code):problems.append('author-code access audit')
    queries=read(root/'outputs/sp_u1_search_queries.json')
    old_gate=read(root/'outputs/sp_u1_upgrade_gate.json')
    if old_gate['scientific_execution_authorized'] or old_gate['new_experiments_run'] or old_gate['old_t20b_used_for_new_design']:
        problems.append('scientific boundary violated')
    # Forensic pass after independent reviewers. No new metric/aggregate is calculated.
    forensic=[
      {'audit_id':'AUD-01','status':'verified_internal_consistency','location':'paper/sections/04_proposed_method.tex',
       'claim_pointer':'full Source with two factored rank4 corrections','evidence_pointer':'method dimensions3968×64 and factors3968×4,4×64',
       'check':'dimension arithmetic only','result':{'source_entries':3968*64,'rank4_correction_entries':2*4*(3968+64),
          'dense_correction_entries':2*3968*64,'rank4_correction_complex128_bytes':2*4*(3968+64)*16},
       'impact':'compression concerns only the corrections; full Source/K storage retained',
       'resolution_test':'later measured runtime must include fitting, storage and all SNR libraries','reviewer_overlap':'R2-m01/R3-m3'},
      {'audit_id':'AUD-02','status':'verified_internal_consistency','location':'paper/sections/03_system_model.tex and04_proposed_method.tex',
       'claim_pointer':'1/32 union overhead,64 selected coefficients,3968 data coefficients',
       'evidence_pointer':'Nt2,Nr2,64×16,dF16,dT4',
       'check':'layout dimension arithmetic only','result':'union32/1024;Nr×union64;4×(1024−32)=3968',
       'impact':'consistent observation/data-domain units; not proof of empirical identifiability',
       'resolution_test':'future geometry must enumerate real masks and partitions','reviewer_overlap':'conditioning and pilot-budget questions'},
      {'audit_id':'AUD-03','status':'unresolved_analysis_gap','location':'origin-weighted low-rank definition versus Source means',
       'claim_pointer':'fixed centered-origin truncation versus actual adaptive Source-centered error',
       'evidence_pointer':'method C_z,m=Rpp_m+sigma2I and z=p−mu_ps',
       'check':'first/second-moment convention and algebra','result':'noncentral second moment and adaptive module cross terms require separate analysis',
       'impact':'no contradiction of weighted-SVD optimization; not a general target/adaptive-risk bound',
       'resolution_test':'SP-U2A derive conditional coupling and cross terms','reviewer_overlap':'R1-M05/R1-M06,R2-M07/R2-M08,R3-M4/R3-M5'},
      {'audit_id':'AUD-04','status':'verified_provenance','location':'all tracked historical files and frozen reviewer outputs',
       'claim_pointer':'SP-U1 only editorial/audit additions','evidence_pointer':'sp_u1_before_manifest.json and REVIEW_HASHES',
       'check':'byte SHA256 across tracked baseline; independent reports frozen',
       'result':status,'impact':'no result or scientific path changed; existing regressions remain required',
       'resolution_test':'paper/full tests plus final integrity audit','reviewer_overlap':'provenance concerns'},
      {'audit_id':'AUD-05','status':'access_limit_not_numeric_error','location':'Özdemir record and full journal author guide',
       'claim_pointer':'exact legacy distinction and complete submission rules',
       'evidence_pointer':'SP-U1 search/retrieval log',
       'check':'reconcile new source access with frozen prior closure without rewriting it',
       'result':'new abstract verified; algorithm UNKNOWN; full author guide403',
       'impact':'priority and complete format clearance remain unavailable',
       'resolution_test':'legal full text/publisher guide when available; retain limited claims','reviewer_overlap':'R2-M01/R3-m1'},
      {'audit_id':'AUD-06','status':'confirmed_internal_error_resolved','location':'new SP-U1 modern-baseline planning draft',
       'claim_pointer':'PEFT backbone size in first generated plan',
       'evidence_pointer':'Lee/Choi author copy Tables2,7,8 rechecked2026-10-04',
       'check':'separate parameter count from MB and nominal versus serialized payload',
       'result':'corrected1.71M parameters versus7.168 MB backbone;192 KB nominal versus0.196 MB serialized LoRA',
       'impact':'planning metadata correction only; all frozen scientific/manuscript files unchanged',
       'resolution_test':'final generated plan cites author measurements/hardware and no own timing claim',
       'reviewer_overlap':'none; root post-review check'}]
    save(root/'outputs/sp_u1_post_review_forensic_audit.json',{'reviewers_frozen_before_audit':True,
        'findings':forensic,'formal_outputs_read_for_new_design':False,
        'new_aggregate_or_scientific_analysis_run':False,
        'existing_numeric_provenance_gate':'paper regression tests; no new evaluation generation'})
    literature={'task':'SP-U1','search_date':'2026-10-04','horizon':'2021–2026 plus necessary classics',
        'query_count':len(queries),'engine_result_totals_available':False,
        'work_count':len(database['works']),
        'recent_2023_2026_count':sum(2023<=w['year']<=2026 for w in database['works']),
        'high_core_proximity_count':sum(w['high_core_proximity'] for w in database['works']),
        'metadata_errors':metadata_errors(database),'source_records':'outputs/sp_u1_literature_database.json',
        'author_code_access':'outputs/sp_u1_author_code_access.json',
        'legacy_fulltext_available':False,'legacy_technical_status':'ABSTRACT_ONLY',
        'journal_full_guide_available':False,'bibtex_duplicates':duplicate_keys+duplicate_identifiers,
        'reviewer_hash_exact':not review_mismatches,'integrity':status,
        'scientific_script_firewall_findings':firewall,
        'bibliography_warnings':bib['warnings'],
        'all_metadata_and_integrity_checks_passed':not problems,'problems':problems}
    save(root/'outputs/sp_u1_literature_audit.json',literature)
    save(root/'outputs/sp_u1_integrity_audit.json',status)
    if finalize:
        execution=read(root/'outputs/sp_u1_execution_results.json')
        required=('focused_pytest','build','audit','paper_pytest','style','full_pytest','diff_check')
        for name in required:
            rec=execution.get(name,{})
            if rec.get('exit_code')!=0 or not rec.get('actually_executed'):
                problems.append('missing/failed actual execution:'+name)
            if rec.get('log') and sha(root/rec['log'])!=rec.get('log_sha256'):
                problems.append('execution log changed:'+name)
            for path,expected in rec.get('execution_source_sha256',{}).items():
                if sha(root/path)!=expected:problems.append('execution source changed:'+name+':'+path)
        full=(root/execution['full_pytest']['log']).read_text(encoding='utf-8')
        match=re.search(r'(\d+) passed',full)
        if not match or int(match[1])<2103 or re.search(r'\b\d+ (?:failed|error|errors|skipped)\b',full):
            problems.append('full historical regression count/status')
        old_gate.update(all_checks_passed=not problems,
            validation_state='PASS' if not problems else 'FAIL',
            validation_problems=problems,execution_results='outputs/sp_u1_execution_results.json',
            integrity_audit='outputs/sp_u1_integrity_audit.json',
            frozen_review_exact=not review_mismatches,literature_record_count=len(database['works']))
        save(root/'outputs/sp_u1_upgrade_gate.json',old_gate)
    if problems:raise ValueError(problems)
    return literature

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    result=audit(finalize=args.finalize)
    print(json.dumps({'metadata_integrity_passed':result['all_metadata_and_integrity_checks_passed'],
        'works':result['work_count'],'historical_mutations':result['integrity']['unexpected_historical_mutation_count'],
        'finalize':args.finalize}))
