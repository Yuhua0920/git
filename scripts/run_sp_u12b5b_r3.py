"""R3 authorized resource/CPU/prototype audit only; formal fitting absent."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sp_u12b5b_r3_common import *
PORT=ROOT/'experiments/mfa_low_memory_prototype/portable'
def benchmarks():
 firewall();freeze=read(OUT/'execution_freeze.json');assert freeze['preflight_PASS']
 assert (OUT/'preservation_before.json').exists()
 for n,h in freeze['sources'].items():assert sha(ROOT/n)==h
 sys.path.insert(0,str(PORT));from runner import monitored
 cfg=ROOT/'outputs/sp_u12b5b_r3_thread_benchmark_config.json';config=read(cfg)
 for t in config['thread_candidates']:
  result=monitored(cfg,OUT/f'parity_{t}','parity',t,reference=None if t==1 else OUT/'parity_1',project_root=ROOT)
  report=read(OUT/f'parity_{t}/parity.json');print(json.dumps({'stage':'R3_parity_complete','threads':t,'PASS':report['PASS'],'peak_RSS':result['peak_RSS']}),flush=True)
 for job in config['schedule']:
  result=monitored(cfg,OUT/f"job_{job['job']:03d}",'kernels',job['threads'],job=job['job'],project_root=ROOT)
  print(json.dumps({'stage':'R3_timing_block_complete','job':job['job'],'block':job['block'],'threads':job['threads'],'peak_RSS':result['peak_RSS'],'seconds':result['wall_seconds']}),flush=True)
 write(OUT/'benchmarks_complete.json',{'PASS':True,'parity_workers':len(config['thread_candidates']),'timing_jobs':len(config['schedule']),'formal_training':False})
def resume_monitor_repair():
 firewall();freeze=read(OUT/'execution_freeze_repaired3.json');assert freeze['preflight_PASS']
 for n,h in freeze['sources'].items():assert sha(ROOT/n)==h
 sys.path.insert(0,str(PORT));from runner import monitored
 cfg=ROOT/'outputs/sp_u12b5b_r3_thread_benchmark_config.json';config=read(cfg);paths={}
 for job in config['schedule']:
  number=job['job'];path=ROOT/freeze['completed_job_paths'][str(number)] if str(number) in freeze['completed_job_paths'] else OUT/(f'job_{number:03d}_repaired3' if number==22 else f'job_{number:03d}')
  if number<freeze['resume_from']:assert read(path/'resource_summary.json')['completed']
  else:
   result=monitored(cfg,path,'kernels',job['threads'],job=number,project_root=ROOT)
   print(json.dumps({'stage':'R3_timing_block_complete_after_monitor_repair','job':number,'block':job['block'],'threads':job['threads'],'peak_RSS':result['peak_RSS'],'seconds':result['wall_seconds']}),flush=True)
  paths[str(number)]=path.relative_to(ROOT).as_posix()
 write(OUT/'benchmarks_complete.json',{'PASS':True,'parity_workers':len(config['thread_candidates']),'timing_jobs':len(config['schedule']),
 'job_paths':paths,'formal_training':False,'technical_failures_preserved':True,'failed_jobs11_13_22_timings_not_selected':True})
def main():
 firewall();stage=sys.argv[1]
 if stage=='bench':benchmarks()
 elif stage=='bench-resume':resume_monitor_repair()
 elif stage=='reports':
  from scripts.sp_u12b5b_r3_report import reports
  reports()
 elif stage=='close':
  from scripts.sp_u12b5b_r3_report import close
  close()
 else:raise ValueError(stage)
if __name__=='__main__':main()
