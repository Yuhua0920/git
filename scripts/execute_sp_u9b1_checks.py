"""Execute manuscript-only final checks and record actual pytest completion."""
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import finalize_sp_u9b1 as audit
OUT=audit.OUT


def main():
    started=time.monotonic()
    previous=OUT/'execution_receipt.json'
    history_path=OUT/'check_attempts.json'
    history=audit.read(history_path) if history_path.exists() else []
    if previous.exists():
        history.append(dict(receipt=audit.read(previous),
                            pytest_log=(OUT/'pytest.txt').read_text(encoding='utf-8'),
                            test_source_sha256=audit.sha(ROOT/'tests/test_sp_u9b1.py')))
        audit.dump(history_path,history)
    audit.main()
    command=[str(ROOT/'.venv-cdl/Scripts/python.exe'),'-X','utf8','-m','pytest','tests/test_sp_u9b1.py','-q',
             '--junitxml='+str(OUT/'pytest.xml')]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (OUT/'pytest.txt').write_text(result.stdout+'\nSTDERR:\n'+result.stderr,encoding='utf-8')
    count=re.search(r'(\d+) passed',result.stdout)
    receipt=dict(task='SP-U9B-1',command=command,pytest_exit_code=result.returncode,
                 tests_passed=int(count[1]) if count else None,test_scope='Twelve manuscript source/provenance/build checks; not the entire historical repository suite',
                 elapsed_seconds=time.monotonic()-started,
                 earlier_attempt='Initial check caught two Related Work trailing spaces. Initial pytest: 10 passed / 2 failed (TODO colon assertion; downstream git audit absent after whitespace stop). Next run: 11 passed / 1 failed (PDF extraction used Unicode ffi ligature in Affiliations). Corrected formatting and normalized extracted text without altering the manuscript wording, rebuilt complete PDF, regenerated audits and reran all twelve checks.',
                 experiments_executed=False,bootstrap_executed=False,figure_regeneration=False,
                 production_modification=False,commit_performed=False,
                 verdict='FRONT_HALF_PASS' if result.returncode==0 and count and int(count[1])==12 else 'FRONT_HALF_REVISION_REQUIRED')
    audit.dump(OUT/'execution_receipt.json',receipt)
    final=audit.read(OUT/'final_audit.json')
    final.update(tests='PASS' if result.returncode==0 else 'FAIL',test_count=receipt['tests_passed'],
                 verdict=receipt['verdict'],stop='STOP; wait for mentor review; do not start U9B-2')
    audit.dump(OUT/'final_audit.json',final)
    changelog=ROOT/'docs/sp_u9b1_change_log.md'
    note=changelog.read_text(encoding='utf-8').replace(
        'FRONT_HALF_PASS is conditional on the separately executed scoped pytest and final receipt. No U9B-2 work is authorized by this artifact.',
        receipt['verdict']+': actual scoped pytest '+str(receipt['tests_passed'])+' passed, exit '+str(result.returncode)+'. No U9B-2 work is authorized by this artifact.')
    changelog.write_text(note,encoding='utf-8')
    # Refresh the final status after all task scripts/docs exist.
    status=subprocess.run([str(audit.GIT),'status','--short'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True).stdout
    (OUT/'git_status_after.txt').write_text(status,encoding='utf-8')
    prior=set((OUT/'git_status_before.txt').read_text(encoding='utf-8').splitlines())
    now=set(status.splitlines())
    git=audit.read(OUT/'git_audit.json')
    git.update(status_after_entries=len(now),new_status_entries=sorted(now-prior),removed_status_entries=sorted(prior-now))
    audit.dump(OUT/'git_audit.json',git)
    final['git']=git
    audit.dump(OUT/'final_audit.json',final)
    artifacts=[ROOT/'outputs'/name for name in [
        'sp_u9b1_preedit_manifest.json','sp_u9b1_postedit_manifest.json','sp_u9b1_abstract_number_audit.csv',
        'sp_u9b1_intro_claim_audit.csv','sp_u9b1_related_work_collision_audit.csv','sp_u9b1_compile_report.json']]
    artifacts += [changelog]+[p for p in OUT.glob('*') if p.is_file()]+list((OUT/'build_post').glob('*'))
    artifacts += list((ROOT/'scripts').glob('*sp_u9b1*.py'))+[ROOT/'tests/test_sp_u9b1.py']
    artifacts += [ROOT/name for name in audit.read(ROOT/'outputs/sp_u9b1_preedit_manifest.json')['allowed']]
    audit.dump(OUT/'delivery_manifest.json',dict(task='SP-U9B-1',verdict=receipt['verdict'],
                 sha256={p.relative_to(ROOT).as_posix():audit.sha(p) for p in artifacts if p.is_file() and p.name!='delivery_manifest.json'}))
    print(result.stdout.strip())
    print(json.dumps(dict(verdict=receipt['verdict'],tests=receipt['tests_passed'],exit_code=result.returncode,
                         tracked_files=len(git['tracked_changed']),git_status_entries=len(now)),ensure_ascii=False))
    if result.returncode or receipt['verdict']!='FRONT_HALF_PASS':
        raise SystemExit(result.returncode or 1)


if __name__=='__main__':
    main()
