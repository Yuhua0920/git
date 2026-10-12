"""Actual tests and source/config freeze before R3 benchmarks."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sp_u12b5b_r3_common import *
def run(label):
 firewall();env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
 cmd=[sys.executable,'-B','-X','utf8','-m','pytest','tests/test_sp_u12b5b_r3.py','-q','-p','no:cacheprovider','--basetemp',str(OUT/('pytest_tmp_'+label))]
 result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8',env=env)
 (OUT/('pytest_'+label+'.txt')).write_text(result.stdout+result.stderr,encoding='utf8');write(OUT/('pytest_'+label+'.json'),{'PASS':result.returncode==0,'exit_code':result.returncode,'command':cmd,'output':result.stdout+result.stderr})
 print(result.stdout,flush=True);assert result.returncode==0
 if label=='preflight':
  files={v.relative_to(ROOT).as_posix():sha(v) for d in ('scripts','tests','experiments/mfa_low_memory_prototype') for v in (ROOT/d).rglob('*') if v.is_file() and owned(v) and '__pycache__' not in v.parts}
  cfg=ROOT/'outputs/sp_u12b5b_r3_thread_benchmark_config.json';files[cfg.relative_to(ROOT).as_posix()]=sha(cfg)
  write(OUT/'execution_freeze.json',{'sources':files,'tree_sha256':tree(files),'preflight_PASS':True,'historical_archives':check_frozen()})
if __name__=='__main__':run(sys.argv[1])
