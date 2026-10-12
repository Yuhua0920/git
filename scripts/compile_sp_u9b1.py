"""Build the complete authoritative multi-file manuscript, preserving all input bytes."""
import argparse,hashlib,json,os,re,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u9b1'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['pre','post']);args=parser.parse_args()
 stage=OUT/('build_'+args.stage);stage.mkdir(exist_ok=True)
 source=OUT/'pre_source' if args.stage=='pre' else ROOT/'paper'
 exe=OUT/'toolchain/tectonic.exe'
 env=dict(os.environ);env['TECTONIC_CACHE_DIR']=str(OUT/'tex_cache');env['NO_COLOR']='1'
 command=[str(exe),'--untrusted','--keep-logs','--keep-intermediates','--print','--outdir',str(stage),'main.tex']
 start=time.monotonic()
 result=subprocess.run(command,cwd=source,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
 console=result.stdout+'\nSTDERR:\n'+result.stderr
 (stage/'console.txt').write_text(console,encoding='utf-8')
 log=(stage/'main.log').read_text(encoding='utf-8',errors='replace') if (stage/'main.log').exists() else ''
 pdf=stage/'main.pdf'
 report=dict(stage=args.stage,command=command,cwd=str(source),exit_code=result.returncode,elapsed_seconds=time.monotonic()-start,
  console=str(stage/'console.txt'),log=str(stage/'main.log'),pdf=str(pdf) if pdf.exists() else None,
  pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest() if pdf.exists() else None,
  warnings=re.findall(r'(?m)^.*(?:Warning|Overfull|Underfull|multiply defined|undefined).*(?:\n[^\n]*)?',log),
  errors=re.findall(r'(?m)^!.*',log),
  all_actual_figures=True,preview_or_placeholder_build=False)
 (stage/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:report[k] for k in ['stage','exit_code','elapsed_seconds','pdf','errors']},ensure_ascii=False))
 if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
