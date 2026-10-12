"""Archive Windows status-file sharing failure; no computational changes."""
from pathlib import Path
import sys,shutil
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sp_u12b5b_r3_common import *
def archive():
 firewall();freeze=read(OUT/'execution_freeze_repaired.json');dest=OUT/'technical_attempt1_archive';assert not dest.exists();dest.mkdir()
 for n,h in freeze['sources'].items():
  assert sha(ROOT/n)==h;target=dest/'sources'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,target)
 shutil.copyfile(OUT/'execution_freeze_repaired.json',dest/'execution_freeze.json');shutil.copytree(OUT/'job_013',dest/'job_013')
 write(dest/'failure.json',{'kind':'WINDOWS_ATOMIC_REPLACE_SHARING','error':'WinError5, current_case.json being read during os.replace',
 'completed_jobs':list(range(13)),'failed_job':13,'action':'bounded retry for WinError5/32/33 before timer; keep completed measurements; repeat13 in fresh folder',
 'changed_scientific_or_timing_logic':False})
 print('R3_STATUS_SHARING_FAILURE_ARCHIVED')
def freeze():
 firewall();assert read(OUT/'pytest_repair2_preflight.json')['PASS'];old=read(OUT/'technical_attempt1_archive/execution_freeze.json')
 files={n:sha(ROOT/n) for n in old['sources']};files['scripts/sp_u12b5b_r3_technical_repair2.py']=sha(Path(__file__))
 changed=[n for n,h in old['sources'].items() if files[n]!=h]
 assert set(changed)=={'experiments/mfa_low_memory_prototype/portable/runner.py','scripts/run_sp_u12b5b_r3.py','tests/test_sp_u12b5b_r3.py'}
 paths={str(i):f'outputs/sp_u12b5b_r3/job_{i:03d}' for i in range(13)};paths['11']='outputs/sp_u12b5b_r3/job_011_repaired'
 for path in paths.values():assert read(ROOT/path/'resource_summary.json')['completed']
 write(OUT/'execution_freeze_repaired2.json',{'sources':files,'tree_sha256':tree(files),'preflight_PASS':True,
 'technical_changes':changed,'previous_freeze_sha256':sha(OUT/'technical_attempt1_archive/execution_freeze.json'),
 'algorithm_fixture_timing_seeds_tolerance_changed':False,'completed_job_paths':paths,'resume_from':13,'fresh_job13_suffix':'_repaired2'})
 print('R3_STATUS_SHARING_REPAIR_FROZEN')
if __name__=='__main__':archive() if sys.argv[1]=='archive' else freeze()
