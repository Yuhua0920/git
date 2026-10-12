"""Summarize actual editorial executions and final hash evidence only."""
import re
import subprocess
from paper_support import ROOT, json_read, json_write, sha256
from audit_paper_positioning import integrity
from check_paper import script_firewall


def execution(name, minimum):
    record=json_read(ROOT/f'outputs/t20g_{name}_pytest_execution.json')
    log=ROOT/record['log']
    text=log.read_text(encoding='utf-8-sig')
    summary=next(line for line in reversed(text.splitlines()) if re.search(r'\d+ passed',line))
    count=int(re.search(r'(\d+) passed',summary)[1])
    if record['exit_code']!=0 or count<minimum or re.search(r'\d+ (?:failed|errors?|skipped)',summary):
        raise ValueError('Actual regression did not pass: '+summary)
    return dict(record,summary=summary,passed=count,log_sha256=sha256(log))


def build():
    paper=execution('paper',123)
    full=execution('full',2103)
    before=json_read(ROOT/'outputs/t20g_before_manifest.json')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=before['t20f_commit']:
        raise ValueError('Accepted T20F baseline commit changed')
    reports={}
    for name in ('discussion_claim','citation_support','related_work_comparison','paper_style',
                 'table_numeric','results_edit','latex_static'):
        p=ROOT/f'outputs/t20g_{name}_audit.json'
        report=json_read(p)
        ok=report.get('all_checks_passed',report.get('all_lexical_checks_passed',report.get('static_validation_passed')))
        if not ok: raise ValueError('Audit failed: '+name)
        reports[name]=dict(path=p.relative_to(ROOT).as_posix(),sha256=sha256(p),passed=True)
    hashes=integrity()
    firewall={p:script_firewall((ROOT/'scripts'/p).read_text(encoding='utf8')) for p in
              ('write_paper_positioning_packet.py','finalize_paper_positioning.py')}
    if not hashes['all_checks_passed'] or any(firewall.values()):
        raise ValueError('Final historical/file-content integrity failure')
    hashes['additional_reporting_script_firewall']=firewall
    allowed_new={
        'scripts/prepare_paper_positioning.py','scripts/build_paper_summary.py',
        'scripts/audit_paper_positioning.py','scripts/write_paper_positioning_packet.py',
        'scripts/finalize_paper_positioning.py','tests/test_paper_positioning.py',
        'paper/tables/06_summary.tex','paper/tables/summary_numbers.tex','paper/tables/t20g_provenance.json'}
    new=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    badnew=[p for p in new if p not in allowed_new and not p.startswith(('outputs/t20g_','docs/t20g_'))]
    if badnew: raise ValueError('Unexpected new files: '+str(badnew))
    json_write(ROOT/'outputs/t20g_integrity_audit.json',hashes)
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    (ROOT/'outputs/t20g_git_diff_check.txt').write_text(diff.stdout+diff.stderr,encoding='utf8',newline='\n')
    if diff.returncode: raise ValueError('git diff --check failed')
    stat=subprocess.check_output(['git','diff','--stat'],cwd=ROOT,text=True)
    (ROOT/'outputs/t20g_git_diff_stat.txt').write_text(stat,encoding='utf8',newline='\n')
    packet=json_read(ROOT/'outputs/t20g_review_packet_audit.json')
    if not packet['parts_present'] or packet['closest_works_count']!=8:
        raise ValueError('Advisor packet incomplete')
    report=dict(task='T20G',status='EDITORIAL_AND_REGRESSION_PASS_NATIVE_COMPILE_UNVERIFIED',
        t20f_commit=before['t20f_commit'],working_tree_clean_after_t20f_commit=True,
        baseline_paper_pytest=before['paper_acceptance'],baseline_full_pytest=before['full_acceptance'],
        paper_pytest=paper,full_pytest=full,new_test_count=full['passed']-2077,
        new_test_source='tests/test_paper_positioning.py: four small reporting calibrations plus 22 citation/scope/integrity tests',
        current_integrity=hashes,validation_reports=reports,
        native_compile_attempt=json_read(ROOT/'outputs/t20g_latex_compile_attempt.json'),
        actual_pages_verified=False,unresolved_prior_art_risk=True,
        current_stage_requires_advisor_review=True,all_required_editorial_and_regression_checks_passed=True,
        git_diff_check_exit_code=diff.returncode,git_diff_stat=stat,
        modified_files=hashes['changed_files'],new_files=sorted(set(new)|{'outputs/t20g_completion_report.json'}),
        new_channel_generated=False,new_statistics_fit=False,new_model_trained=False,method_changed=False,
        formal_data_used_only_for_reporting=True,
        actual_task_commands=[
            'git add -A','git commit -m "feat: draft evaluation setup and empirical results"',
            '.venv/Scripts/python.exe scripts/prepare_paper_positioning.py',
            '.venv/Scripts/python.exe -m pytest -q tests/test_paper_positioning.py',
            '.venv/Scripts/python.exe scripts/build_paper_summary.py',
            '.venv/Scripts/python.exe scripts/build_paper_preview.py',
            '.venv/Scripts/python.exe scripts/audit_paper_style.py --output outputs/t20g_paper_style_audit.json',
            '.venv/Scripts/python.exe scripts/audit_paper_positioning.py --reviewed',
            paper['command'],full['command'],
            '.venv/Scripts/python.exe scripts/write_paper_positioning_packet.py',
            '.venv/Scripts/python.exe scripts/finalize_paper_positioning.py','git diff --check'],
        engineering_attempt_issues=[
            'Initial snapshot utility rejected its new untracked bootstrap files; clean tracked baseline verified and utilities explicitly inventoried before snapshot generation.',
            'Initial prose patch rejected duplicate target delete/add operations atomically; no partial mutation, corrected write completed before auditing.',
            'Native compiler failed before source parsing; source remains in the same open editor.',
            'Existing Python launcher location notice and Git LF/CRLF notices remain.'])
    json_write(ROOT/'outputs/t20g_completion_report.json',report)
    print(dict(status=report['status'],paper=paper['summary'],full=full['summary'],
               unexpected_mutations=len(hashes['unexpected_mutations']),git_diff_check=diff.returncode))


if __name__=='__main__':
    build()
