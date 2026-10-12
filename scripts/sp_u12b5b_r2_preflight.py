"""Actual pytest and immutable microbenchmark execution freeze."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sp_u12b5b_r2_common import *
def run(label):
 firewall();env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
 command=[sys.executable,'-B','-X','utf8','-m','pytest','tests/test_sp_u12b5b_r2.py','-q','-p','no:cacheprovider','--basetemp',str(OUT/('pytest_tmp_'+label))]
 result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf8')
 (OUT/('pytest_'+label+'.txt')).write_text(result.stdout+result.stderr,encoding='utf8')
 obj={'exit_code':result.returncode,'command':command,'output':result.stdout+result.stderr,'PASS':result.returncode==0}
 write(OUT/('pytest_'+label+'.json'),obj);print(result.stdout,flush=True);assert obj['PASS']
 if label=='preflight':
  sources={str(v.relative_to(ROOT)).replace('\\','/'):sha(v) for d in ('scripts','tests') for v in (ROOT/d).glob('*sp_u12b5b_r2*.py')}
  sources['outputs/sp_u12b5b_r2_microbenchmark_config.json']=sha(ROOT/'outputs/sp_u12b5b_r2_microbenchmark_config.json')
  write(OUT/'execution_freeze.json',{'preflight_passed':True,'sources':sources,'source_tree_sha256':tree(sources),'resource_thresholds_UNCHANGED':LIMITS,'reference':check_frozen()})
if __name__=='__main__':run(sys.argv[1])
