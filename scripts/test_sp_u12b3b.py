"""Actual pytest, captured output and only technical summaries to console."""
from pathlib import Path
import sys,subprocess,json,os,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u12b3b'
def run(stage):
 OUT.mkdir(exist_ok=True);name='pytest_'+stage
 files=['tests/test_sp_u12b3a.py','tests/test_sp_u12b3b.py'] if stage=='preflight' else ['tests/test_sp_u12b3b_integrity.py']
 cmd=[sys.executable,'-X','utf8','-m','pytest',*files,'-q','--assert=plain','-o','cache_dir=outputs/sp_u12b3b/pytest_cache','--basetemp=outputs/sp_u12b3b/pytest_tmp/'+stage,'--junitxml=outputs/sp_u12b3b/'+name+'.xml']
 env=os.environ.copy()
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
 proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8',env=env);(OUT/(name+'.txt')).write_text(proc.stdout+proc.stderr,encoding='utf8')
 xml=OUT/(name+'.xml');suites=ET.parse(xml).getroot().findall('testsuite') if xml.exists() else []
 result=dict(exit_code=proc.returncode,tests=sum(int(s.attrib['tests']) for s in suites),failed=sum(int(s.attrib['failures'])+int(s.attrib['errors']) for s in suites),skipped=sum(int(s.attrib['skipped']) for s in suites),seconds=sum(float(s.attrib['time']) for s in suites),stage=stage,command=cmd,console_sanitized=True,formal_scientific_values_disclosed=False)
 (OUT/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:result[k] for k in ('exit_code','tests','failed','skipped','seconds','stage')}),flush=True);raise SystemExit(proc.returncode)
if __name__=='__main__':run(sys.argv[1])
