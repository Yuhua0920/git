"""Actual pytest orchestration with sanitized console; logs never expose formal values."""
from pathlib import Path
import sys,subprocess,json,xml.etree.ElementTree as ET,argparse
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u12b2b';OUT.mkdir(exist_ok=True)
def run(stage):
 name='pytest_preflight' if stage=='preflight' else 'pytest_final'
 files=['tests/test_sp_u12b2a.py','tests/test_sp_u12b2b.py'] if stage=='preflight' else ['tests/test_sp_u12b2b_integrity.py']
 command=[sys.executable,'-X','utf8','-m','pytest',*files,'-q','--assert=plain','-o','cache_dir=outputs/sp_u12b2b/pytest_cache','--basetemp=outputs/sp_u12b2b/pytest_tmp_'+stage,'--junitxml=outputs/sp_u12b2b/'+name+'.xml']
 proc=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf8')
 (OUT/(name+'.txt')).write_text(proc.stdout+proc.stderr,encoding='utf8')
 suites=ET.parse(OUT/(name+'.xml')).getroot().findall('testsuite')
 result=dict(exit_code=proc.returncode,tests=sum(int(s.attrib['tests']) for s in suites),failed=sum(int(s.attrib['failures'])+int(s.attrib['errors']) for s in suites),skipped=sum(int(s.attrib['skipped']) for s in suites),seconds=sum(float(s.attrib['time']) for s in suites),phase=stage,command=command,console_sanitized=True,formal_scientific_values_printed=False)
 (OUT/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
 print(json.dumps({key:result[key] for key in ('exit_code','tests','failed','skipped','seconds','phase','console_sanitized')}),flush=True)
 raise SystemExit(proc.returncode)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['preflight','final']);run(parser.parse_args().phase)
