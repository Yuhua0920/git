"""Actual pytest execution receipt, not a substitute for executing tests."""
from pathlib import Path
import sys,os,subprocess,xml.etree.ElementTree as ET,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u12b4a_preserve import OUT,write,sha
def run():
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','GOTO_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[key]='1'
 cmd=[str(ROOT/'.venv-cdl/Scripts/python.exe'),'-X','utf8','-m','pytest','tests/test_sp_u12b4a.py','-q','-p','no:cacheprovider','--basetemp='+str(OUT/'pytest_tmp'),'--junitxml='+str(OUT/'pytest.xml')]
 r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8')
 (OUT/'pytest.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf8');rows=list(ET.parse(OUT/'pytest.xml').getroot().iter('testsuite'))
 cal=ROOT/'outputs/sp_u12b4a_timing_calibration.json'
 o=dict(exit_code=r.returncode,passed=sum(int(x.get('tests','0'))-int(x.get('failures','0'))-int(x.get('errors','0'))-int(x.get('skipped','0')) for x in rows),failed=sum(int(x.get('failures','0'))+int(x.get('errors','0')) for x in rows),skipped=sum(int(x.get('skipped','0')) for x in rows),command=cmd,timing_calibration_sha256=sha(cal) if cal.exists() else None)
 write(OUT/'pytest.json',o);print(json.dumps({k:v for k,v in o.items() if k not in ('command','timing_calibration_sha256')}),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
if __name__=='__main__':run()
