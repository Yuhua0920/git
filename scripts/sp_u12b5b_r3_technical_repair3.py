"""Archive optional status read sharing failure; resource sampling stays primary."""
from pathlib import Path
import sys,shutil
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sp_u12b5b_r3_common import *
def archive():
 firewall();freeze=read(OUT/'execution_freeze_repaired2.json');dest=OUT/'technical_attempt2_archive';assert not dest.exists();dest.mkdir()
 for n,h in freeze['sources'].items():
  assert sha(ROOT/n)==h;target=dest/'sources'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,target)
 shutil.copyfile(OUT/'execution_freeze_repaired2.json',dest/'execution_freeze.json');shutil.copytree(OUT/'job_022',dest/'job_022')
 write(dest/'failure.json',{'kind':'OPTIONAL_MARKER_READ_SHARING','error':'PermissionError reading current_case.json during atomic replacement',
 'failed_job':22,'resource_monitor_record_incomplete':True,'worker_final_exists':(OUT/'job_022/worker_final.json').exists(),
 'action':'keep completed jobs0..21; fresh replacement22; transient optional marker read miss must not stop primary RSS/headroom sampling',
 'scientific_or_computation_changed':False})
 print('R3_OPTIONAL_MARKER_FAILURE_ARCHIVED')
def freeze():
 firewall();assert read(OUT/'pytest_repair3_preflight.json')['PASS'];old=read(OUT/'technical_attempt2_archive/execution_freeze.json')
 files={n:sha(ROOT/n) for n in old['sources']};files['scripts/sp_u12b5b_r3_technical_repair3.py']=sha(Path(__file__))
 changed=[n for n,h in old['sources'].items() if files[n]!=h]
 assert set(changed)=={'experiments/mfa_low_memory_prototype/portable/runner.py','scripts/run_sp_u12b5b_r3.py','tests/test_sp_u12b5b_r3.py'}
 paths={str(i):f'outputs/sp_u12b5b_r3/job_{i:03d}' for i in range(22)};paths['11']='outputs/sp_u12b5b_r3/job_011_repaired';paths['13']='outputs/sp_u12b5b_r3/job_013_repaired2'
 for path in paths.values():assert read(ROOT/path/'resource_summary.json')['completed']
 write(OUT/'execution_freeze_repaired3.json',{'sources':files,'tree_sha256':tree(files),'preflight_PASS':True,
 'technical_changes':changed,'previous_freeze_sha256':sha(OUT/'technical_attempt2_archive/execution_freeze.json'),
 'algorithm_fixture_timing_seeds_tolerance_changed':False,'completed_job_paths':paths,'resume_from':22})
 print('R3_OPTIONAL_MARKER_REPAIR_FROZEN')
if __name__=='__main__':archive() if sys.argv[1]=='archive' else freeze()
