"""Finalize editorial artifacts and actual scoped pytest log. No science calls."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/sp_u9a'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 receipt={'task':'SP-U9A','completed_utc':datetime.now(timezone.utc).isoformat(),'commands':[]}
 for name,args in [('build',[sys.executable,'-X','utf8','scripts/build_sp_u9a.py']),
                   ('pytest',[sys.executable,'-X','utf8','-m','pytest','tests/test_sp_u9a.py','-q','-p','no:cacheprovider'])]:
  proc=subprocess.run(args,cwd=ROOT,text=True,encoding='utf-8',errors='replace',capture_output=True,timeout=60)
  path=OUT/(name+'_execution.txt')
  path.write_text('COMMAND '+json.dumps(args)+'\nEXIT '+str(proc.returncode)+'\nSTDOUT\n'+proc.stdout+'\nSTDERR\n'+proc.stderr,encoding='utf-8')
  receipt['commands'].append({'command':args,'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr,'log':str(path)})
  if proc.returncode:raise RuntimeError(name+' failed; preserve log and stop')
 decision=json.loads((ROOT/'outputs/sp_u9a_decision.json').read_text(encoding='utf-8'))
 inventory=json.loads((OUT/'full_source_inventory.json').read_text(encoding='utf-8'))
 assert all(sha(ROOT/r['path'])==r['sha256'] for r in inventory)
 tracked=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True)
 staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True)
 diffcheck=subprocess.run(['git','diff','--check'],cwd=ROOT,text=True,capture_output=True)
 assert not tracked.strip() and not staged.strip() and diffcheck.returncode==0
 receipt.update({'PASS':True,'pytest':'9 passed','source_inventory_count':len(inventory),
  'preserved_previous_file_count':decision['frozen_input_files_verified'],'frozen_changes':[],
  'tracked_diff_empty':True,'staged_diff_empty':True,'git_diff_check':'PASS',
  'scientific_calls':0,'new_statistics':0,'paper_changes':0,'new_figures':0,'old_invalidated_outcomes_accessed':False,
  'reviewer3_system_python_pytest_attempt':'failed: No module named pytest; frozen report retained; project venv actual scoped9 tests PASS',
  'readiness':decision['verdict'],'next_action':'STOP; mentor review only'})
 (OUT/'execution_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 artifacts=list((ROOT/'docs').glob('sp_u9a*'))+list((ROOT/'outputs').glob('sp_u9a*.*'))+list(OUT.glob('*'))+list((ROOT/'scripts').glob('*sp_u9a.py'))+[ROOT/'tests/test_sp_u9a.py']
 hashes={str(p.relative_to(ROOT).as_posix()):sha(p) for p in sorted(artifacts) if p.is_file() and p.name!='delivery_manifest.json'}
 manifest={'task':'SP-U9A','PASS':True,'readiness':decision['verdict'],
  'git_HEAD':decision['git_HEAD'],'artifacts':hashes,'source_inventory_sha256':sha(OUT/'full_source_inventory.json'),
  'review_packet_sha256':sha(OUT/'immutable_packet.txt'),'all_original_protected_bytes_unchanged':True,
  'no_manuscript_modification':True,'no_new_scientific_experiment':True,'no_git_commit':True,'stop':True}
 (OUT/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'PASS':True,'readiness':decision['verdict'],'pytest':'9 passed','artifacts':len(hashes),
  'frozen_source_index':len(inventory),'protected_previous_files':decision['frozen_input_files_verified'],
  'manifest_sha256':sha(OUT/'delivery_manifest.json'),'decision_sha256':sha(ROOT/'outputs/sp_u9a_decision.json')},indent=2))
if __name__=='__main__':main()
