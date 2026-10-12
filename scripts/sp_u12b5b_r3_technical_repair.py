"""Archive monitor-only failed attempt; freeze a justified technical repair."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sp_u12b5b_r3_common import *
import shutil
def archive():
 firewall();freeze=read(OUT/'execution_freeze.json');dest=OUT/'technical_attempt0_archive';assert not dest.exists();dest.mkdir()
 for n,h in freeze['sources'].items():
  assert sha(ROOT/n)==h;npath=dest/'sources'/n;npath.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,npath)
 shutil.copyfile(OUT/'execution_freeze.json',dest/'execution_freeze.json')
 shutil.copytree(OUT/'job_011',dest/'job_011')
 write(dest/'failure.json',{'kind':'MONITOR_EXIT_RACE','error':'OpenProcess actual worker PID already exited: WinError87, launcher poll still None',
 'worker_final_exists':True,'completed_timing_rows':len((OUT/'job_011/timings.jsonl').read_text(encoding='utf8').splitlines()),
 'resource_monitor_log_not_flushed':True,'action':'retain jobs0..10; repeat job11 only in fresh directory then continue original schedule',
 'failed_job_timings_excluded':'technical incomplete resource record, not latency selection','scientific_data_used':False})
 print('R3_MONITOR_FAILED_ATTEMPT_ARCHIVED')
def freeze():
 firewall();assert read(OUT/'pytest_repair_preflight.json')['PASS']
 old=read(OUT/'technical_attempt0_archive/execution_freeze.json')
 files={n:sha(ROOT/n) for n in old['sources']}
 files['scripts/sp_u12b5b_r3_technical_repair.py']=sha(Path(__file__))
 changed=[n for n,h in old['sources'].items() if files[n]!=h]
 assert set(changed)=={'experiments/mfa_low_memory_prototype/portable/runner.py','scripts/run_sp_u12b5b_r3.py','tests/test_sp_u12b5b_r3.py'}
 write(OUT/'execution_freeze_repaired.json',{'sources':files,'tree_sha256':tree(files),'preflight_PASS':True,'technical_changes':changed,
 'original_freeze_sha256':sha(OUT/'technical_attempt0_archive/execution_freeze.json'),'algorithm_fixture_timing_seeds_tolerance_changed':False,
 'reuse_completed_jobs':list(range(11)),'fresh_repeated_job':11,'resume_schedule_from':11})
 print('R3_MONITOR_REPAIR_FROZEN')
if __name__=='__main__':archive() if sys.argv[1]=='archive' else freeze()
