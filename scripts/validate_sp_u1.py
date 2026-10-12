"""Run authorized editorial/regression commands and record their real exits."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PAPER_TESTS=['tests/test_paper_tables.py','tests/test_paper_evidence.py','tests/test_paper_blueprint.py',
             'tests/test_paper_style.py','tests/test_paper_results.py','tests/test_paper_positioning.py']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','focused','complete'])
    args=parser.parse_args()
    steps={
      'prepare':[('build',[sys.executable,'scripts/build_sp_u1.py'])],
      'focused':[('focused_pytest',[sys.executable,'-m','pytest','-q','tests/test_sp_u1_upgrade.py']),
                 ('audit',[sys.executable,'scripts/audit_sp_u1.py'])],
      'complete':[('paper_pytest',[sys.executable,'-m','pytest','-q',*PAPER_TESTS]),
                  ('style',[sys.executable,'scripts/audit_paper_style.py','--output','outputs/sp_u1_paper_style_audit.json']),
                  ('full_pytest',[sys.executable,'-m','pytest','-q']),
                  ('diff_check',['git','diff','--check'])]}[args.stage]
    record_path=ROOT/'outputs/sp_u1_execution_results.json'
    records=json.loads(record_path.read_text(encoding='utf-8')) if record_path.exists() else {}
    for name,cmd in steps:
        print('Starting '+name,flush=True)
        sources=sorted((ROOT/'scripts').glob('*sp_u1*.py'))+[ROOT/'tests/test_sp_u1_upgrade.py']
        source_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        start=time.monotonic()
        proc=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        previous=records.get(name)
        history=previous.get('attempts',[]) if previous else []
        if previous:
            history=history+[{k:v for k,v in previous.items() if k!='attempts'}]
        suffix=f'_attempt{len(history)+1:02}' if history else ''
        path='outputs/sp_u1_'+name+'_execution'+suffix+'.txt'
        (ROOT/path).write_bytes(proc.stdout)
        records[name]={'argv':cmd,'actually_executed':True,'exit_code':proc.returncode,
                       'duration_seconds':time.monotonic()-start,'log':path,
                       'log_sha256':hashlib.sha256(proc.stdout).hexdigest(),'attempts':history,
                       'execution_source_sha256':source_hashes}
        record_path.write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
        print(proc.stdout.decode('utf-8',errors='replace')[-1600:],flush=True)
        print(f'{name}: exit {proc.returncode}',flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)

if __name__=='__main__':main()
