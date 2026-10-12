"""Package actual T20F editorial validation records without scientific execution."""
import argparse
import json
import re
import subprocess
from pathlib import Path

from paper_support import ROOT, json_read, json_write, sha256


def pytest_record(path, minimum):
    record=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    log=ROOT/record['log']
    text=log.read_text(encoding='utf-8-sig')
    summary=next(line for line in reversed(text.splitlines()) if re.search(r'\d+ passed',line))
    count=int(re.search(r'(\d+) passed',summary)[1])
    assert record['exit_code']==0 and count>=minimum, record
    assert not re.search(r'\d+ (?:failed|errors?|skipped)',summary), summary
    return dict(record,summary=summary,passed=count,log_sha256=sha256(log))


def build(manual_review_completed=False):
    assert manual_review_completed, 'Final rendered views must be inspected first'
    before=json_read(ROOT/'outputs/t20f_before_manifest.json')
    reports={}
    for name in ('table_numeric','figure_numeric','result_sentence','claim','paper_style',
                 'notation_dimension','latex_static','historical_editorial_provenance','integrity'):
        path=ROOT/f'outputs/t20f_{name}_audit.json'
        record=json_read(path)
        passed=record.get('all_checks_passed',record.get('all_lexical_checks_passed',record.get('static_validation_passed')))
        assert passed, name
        reports[name]=dict(all_checks_passed=True,path=path.relative_to(ROOT).as_posix(),sha256=sha256(path))
    layout=json_read(ROOT/'outputs/t20f_figure_layout_audit.json')
    assert layout['all_automatic_checks_passed']
    views=[]
    for name in ('formal_nmse','formal_ber','formal_endpoints'):
        for view in ('color','grayscale','deuteranopia_approx'):
            path=ROOT/f'outputs/t20f_figure_qa/{name}_{view}.png'
            views.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha256(path),
                              manually_inspected=True))
    check=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    (ROOT/'outputs/t20f_git_diff_check.txt').write_text(check.stdout+check.stderr,encoding='utf8')
    assert check.returncode==0, check.stdout+check.stderr
    stat=subprocess.check_output(['git','diff','--stat'],cwd=ROOT,text=True)
    (ROOT/'outputs/t20f_git_diff_stat.txt').write_text(stat,encoding='utf8')
    current_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert current_commit==before['t20e_commit']
    existing_changes=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
    new_files=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    compile_record=json_read(ROOT/'outputs/t20f_latex_compile_attempt.json')
    new_files=sorted(set(new_files)|{'outputs/t20f_completion_report.json'})
    out=dict(task='T20F',status='PASS_WITH_UNVERIFIED_NATIVE_COMPILATION',
        t20e_commit=current_commit,baseline_working_tree_clean=before['working_tree_clean_before_t20f'],
        baseline_paper_acceptance=before['paper_acceptance'],baseline_full_acceptance=before['full_acceptance'],
        paper_pytest=pytest_record(ROOT/'outputs/t20f_paper_pytest_execution.json',73),
        full_pytest=pytest_record(ROOT/'outputs/t20f_full_pytest_execution.json',2053),
        validation_reports=reports,manual_figure_views=views,
        all_13_panels_inspected_in_color_grayscale_and_cvd_approximation=True,
        legitimate_near_equal_curve_overlap_remains=True,
        native_compilation_verified=False,native_compile_record=compile_record,
        actual_manuscript_pagination_verified=False,
        git_diff_check_exit_code=check.returncode,git_diff_stat=stat,
        modified_files=existing_changes,new_files=new_files,
        actual_task_commands=[
            'git add -A',
            'git commit -m "feat: draft system model and pilot-driven affine LMMSE method"',
            '.venv/Scripts/python.exe scripts/build_paper_tables.py',
            '.venv/Scripts/python.exe scripts/build_paper_figures.py',
            '.venv/Scripts/python.exe scripts/build_paper_preview.py',
            '.venv/Scripts/python.exe scripts/audit_paper_figures.py',
            '.venv/Scripts/python.exe scripts/audit_paper_results.py',
            '.venv/Scripts/python.exe scripts/audit_paper_results.py --integrity',
            '.venv/Scripts/python.exe scripts/audit_paper_style.py --output outputs/t20f_paper_style_audit.json',
            '.venv/Scripts/python.exe scripts/write_t20f_review_packet.py',
            '.venv/Scripts/python.exe scripts/finalize_paper_results.py --manual-reviewed',
            'git diff --check', 'git diff --stat'],
        manuscript_number_macros={name:json_read(ROOT/'paper/tables/t20f_provenance.json')['number_macros'][name]
            for name in json_read(ROOT/'outputs/t20f_writing_audit.json')['used_number_macros']},
        new_channel_generated=False,new_statistics_fit=False,new_model_trained=False,
        method_changed=False,formal_test_reused_for_design=False,
        warning_summary=[
            'Native compiler unavailable before source parsing: Unable to find standard directories for platform',
            'Existing Python launcher location notice and Git LF/CRLF notices retained in logs',
            'Early editorial formatter syntax, source-namespace, Git representation and SVG export whitespace failures repaired; attempt logs preserved',
            'Historical bibliography full-text-access qualifications retained; new channel reference uses verified official report identifier',
            'Some near-equal curves naturally overlap; exact values remain in full appendix tables'],
        all_required_editorial_checks_passed=True)
    json_write(ROOT/'outputs/t20f_completion_report.json',out)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manual-reviewed',action='store_true')
    args=parser.parse_args()
    r=build(args.manual_reviewed)
    print({'status':r['status'],'paper_pytest':r['paper_pytest']['summary'],
           'full_pytest':r['full_pytest']['summary'],'git_diff_check':r['git_diff_check_exit_code']})
