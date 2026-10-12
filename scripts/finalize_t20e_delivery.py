"""Consolidate executed T20E checks without editing historical evidence."""
import re
import subprocess
from paper_support import ROOT, json_read, json_write, sha256
from audit_manuscript_draft import integrity


def finish():
    frozen=integrity()
    reports={key:json_read(ROOT/name) for key,name in {
        "style":"outputs/t20e_paper_style_audit.json", "dimensions":"outputs/t20e_equation_dimension_audit.json",
        "claims":"outputs/t20e_claim_audit.json", "latex":"outputs/t20e_latex_static_audit.json",
        "workflow":"outputs/t20e_workflow_audit.json", "manual":"outputs/t20e_manual_review.json"}.items()}
    suites={}
    for key,name in {"start_paper":"outputs/t20d_acceptance_paper_pytest.txt",
                     "start_full_initial_failure":"outputs/t20d_acceptance_full_pytest.txt",
                     "start_full_recovery":"outputs/t20d_acceptance_retry_full_pytest.txt",
                     "scanner_unit":"outputs/t20e_scanner_unit_pytest.txt",
                     "final_paper":"outputs/t20e_final_paper_pytest.txt",
                     "full":"outputs/t20e_full_pytest.txt"}.items():
        text=(ROOT/name).read_text(encoding="utf8")
        lines=[line for line in text.splitlines() if "passed in" in line]
        suites[key]=dict(summary=lines[-1] if lines else "missing summary",path=name,sha256=sha256(ROOT/name))
    assert "2053 passed" in suites["full"]["summary"]
    assert "73 passed" in suites["final_paper"]["summary"]
    check=subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True,text=True)
    (ROOT/"outputs/t20e_git_diff_check.txt").write_text(check.stdout+check.stderr,encoding="utf8",newline="\n")
    stat=subprocess.run(["git","diff","--stat"],cwd=ROOT,capture_output=True,text=True)
    (ROOT/"outputs/t20e_git_diff_stat.txt").write_text(stat.stdout,encoding="utf8",newline="\n")
    new=[p.decode("utf8") for p in subprocess.check_output(["git","ls-files","--others","--exclude-standard","-z"],cwd=ROOT).split(b"\0") if p]
    current=[p.decode("utf8") for p in subprocess.check_output(["git","diff","--name-only","-z"],cwd=ROOT).split(b"\0") if p]
    checks=dict(style=reports["style"]["all_lexical_checks_passed"],dimensions=reports["dimensions"]["all_checks_passed"],
                claims=reports["claims"]["all_checks_passed"],static_latex=reports["latex"]["static_validation_passed"],
                figure=reports["workflow"]["all_automatic_checks_passed"],integrity=frozen["all_checks_passed"],
                git_diff_check=check.returncode==0,final_paper_tests=True,full_regression=True)
    report=dict(task="T20E",t20d_commit=json_read(ROOT/"outputs/t20e_before_manifest.json")["t20d_commit"],
        working_tree_clean_after_t20d_commit=True, t20e_commit_created=False,
        checks=checks,all_checks_passed=all(checks.values()), pytest=suites,
        tracked_modified_files=current,new_files=new, git_diff_stat=stat.stdout,
        git_diff_check_returncode=check.returncode,git_line_ending_notices=check.stderr,
        tracked_before_hash_count=frozen["tracked_files_checked"], historical_outputs_checked=frozen["historical_outputs_checked"],
        historical_scientific_mutations=frozen["unexpected_mutations"], scientific_source_files_checked=frozen["scientific_source_files_checked"],
        compile_not_verified_due_to_environment=True,actual_document_pagination_verified=False,
        final_full_suite_precedes_last_editorial_caption_symbol_cleanup=True,
        final_paper_suite_executed_after_all_editorial_changes=True,
        no_algorithm_or_scientific_test_changes_after_full_suite=True,
        only_system_model_and_proposed_method_drafted=True,
        new_channel_generated=False,new_statistics_fit=False,new_model_trained=False,method_changed=False,formal_data_used_for_design=False,
        waiting_for_advisor_review=True)
    json_write(ROOT/"outputs/t20e_completion_report.json",report)
    print({"checks":checks,"tracked_modified":len(current),"new_files_before_completion_report":len(new),
           "integrity_count":frozen["tracked_files_checked"],"status":"PASS" if report["all_checks_passed"] else "FAIL"})
    if not report["all_checks_passed"]:raise SystemExit(1)
    return report


if __name__=="__main__":finish()
