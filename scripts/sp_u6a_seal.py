"""Seal SP-U6A audit artifacts after actual pytest; no performance data access."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    raw_log=(OUT/'sp_u6a_pytest_execution.txt').read_bytes()
    log=raw_log.decode('utf-16' if raw_log.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8')
    assert '13 passed, 4 skipped' in log
    post=(OUT/'sp_u6a_pytest_post.txt').read_bytes()
    post_log=post.decode('utf-16' if post.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8')
    assert '13 passed, 4 skipped' in post_log
    pin=json.loads((OUT/'sp_u6a_source_freeze.json').read_text(encoding='utf-8'))
    for row in pin['sources']:
        path=OUT/'sp_u6a_sources'/row['path']
        assert sha(path)==row['sha256']
        if path.suffix=='.pdf':
            assert path.read_bytes().startswith(b'%PDF')
    provenance=json.loads((OUT/'sp_u6a_input_provenance.json').read_text(encoding='utf-8'))
    for name,digest in provenance['permitted_input_sha256'].items():
        assert sha(ROOT/name)==digest
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True,check=True)
    tracked=subprocess.run(['git','diff','--numstat'],cwd=ROOT,capture_output=True,text=True,check=True)
    assert not tracked.stdout.strip()
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    artifacts=sorted([p for p in OUT.glob('sp_u6a_*') if p.is_file() and p.name!='sp_u6a_delivery_manifest.json']+
                     list((OUT/'sp_u6a_sources').glob('*'))+
                     list((ROOT/'scripts').glob('sp_u6a_*.py'))+
                     list((ROOT/'tests').glob('test_sp_u6a_*.py')))
    # The seal stdout is redirected by the caller and is still being written.
    artifacts=[p for p in artifacts if p.name!='sp_u6a_seal_execution_attempt02.txt']
    rows=[{'path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size} for p in artifacts]
    result={'task':'SP-U6A','verdict':'FAIL / STOP-D','HEAD':head,'tracked_diff':'EMPTY','pytest':{'passed':13,'skipped':4,'scope':'tests/test_sp_u6a_gate.py','post_log_verified':True},
            'source_freeze_unchanged':True,'permitted_inputs_unchanged':True,'author_sources_modified':False,
            'next_task_started':False,'commit_created':False,'source_or_artifact_hashes':rows,
            'excluded_self_and_open_log':['sp_u6a_delivery_manifest.json','sp_u6a_seal_execution_attempt02.txt'],
            'firewall_limits':'Dedicated scripts/tests use allowlist. Shell reads restricted to literature, protocol, source and own audit artifacts. No historical performance suite run.'}
    (OUT/'sp_u6a_delivery_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'verdict':result['verdict'],'artifacts':len(rows),'HEAD':head,'tracked_diff':'EMPTY','pytest':'13 passed,4 explicit skipped','protected_allowed_inputs':len(provenance['permitted_input_sha256']),'author_sources_unchanged':True},indent=2))

if __name__=='__main__':
    main()
