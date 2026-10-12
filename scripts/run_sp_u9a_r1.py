"""Actual literature task script + scoped pytest + preservation/format checks."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
from build_sp_u9a_r1 import main,preservation
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u9a_r1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run_record(command,name):
 start=time.monotonic()
 result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
 log=OUT/(name+'.txt');log.write_text(result.stdout+'\nSTDERR:\n'+result.stderr,encoding='utf-8')
 return dict(command=command,exit_code=result.returncode,elapsed_seconds=time.monotonic()-start,log=str(log),sha256=sha(log))
def main_run():
 started=time.time();decision=main()
 python=ROOT/'.venv-cdl/Scripts/python.exe'
 tests=run_record([str(python),'-X','utf8','-m','pytest','-q','tests/test_sp_u9a_r1.py','-p','no:cacheprovider'],'pytest_execution')
 git=Path(r'C:\Users\yuhua\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe')
 diff=run_record([str(git),'diff','--check'],'diff_check_execution')
 audit=preservation()
 result=dict(task='SP-U9A-R1',started_unix=started,finished_unix=time.time(),build='ACTUALLY_EXECUTED',pytest=tests,diff_check=diff,integrity=audit,
             scientific_runs=0,passed=tests['exit_code']==0 and diff['exit_code']==0 and audit['all_unchanged'])
 write(OUT/'execution_results.json',result);write(OUT/'integrity_audit.json',audit)
 decision['execution_validation']='PASS' if result['passed'] else 'FAIL'
 decision['integrity']=audit
 decision['execution_receipt']=str(OUT/'execution_results.json')
 write(ROOT/'outputs/sp_u9a_r1_decision.json',decision)
 artifacts=[ROOT/'outputs/sp_u9a_r1_source_ledger.json',ROOT/'outputs/sp_u9a_r1_collision_matrix.csv',ROOT/'outputs/sp_u9a_r1_decision.json']
 artifacts+=sorted((ROOT/'docs').glob('sp_u9a_r1_*.md'))
 artifacts+=sorted((ROOT/'scripts').glob('*sp_u9a_r1*.py'))+[ROOT/'tests/test_sp_u9a_r1.py']
 write(OUT/'delivery_manifest.json',dict(task='SP-U9A-R1',files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in artifacts],execution=result))
 print(json.dumps(dict(task='SP-U9A-R1',actual_pytest_exit=tests['exit_code'],diff_check_exit=diff['exit_code'],preserved=audit['checked_files'],changed=audit['changed_files'],execution_validation=decision['execution_validation'],readiness=decision['final_readiness']),ensure_ascii=False))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main_run()
