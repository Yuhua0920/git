"""Immutable before-edit snapshot and exact logical bibliography path mapping."""
import hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u9b1'
ALLOWED=['paper/metadata.tex','paper/main.tex','paper/sections/01_introduction.tex','paper/sections/02_related_work.tex','paper/bibliography/references.bib']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 target=ROOT/'outputs/sp_u9b1_preedit_manifest.json'
 if target.exists():raise RuntimeError('Before-edit manifest already exists; do not replace it')
 OUT.mkdir(exist_ok=True)
 paper={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'paper').rglob('*') if p.is_file()}
 prior=json.loads((ROOT/'outputs/sp_u9a_r1/preservation_before.json').read_text(encoding='utf-8'))
 frozen={name:sha(ROOT/name) for name in prior if name not in ALLOWED and (ROOT/name).is_file()}
 for parent in ['src','configs','outputs/sp_u9a_r1']:
  for p in (ROOT/parent).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:frozen[p.relative_to(ROOT).as_posix()]=sha(p)
 for p in (ROOT/'outputs').glob('sp_u9a_r1_*'):
  if p.is_file():frozen[p.relative_to(ROOT).as_posix()]=sha(p)
 shutil.copytree(ROOT/'paper',OUT/'pre_source')
 git=Path(r'C:\Users\yuhua\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe')
 status=subprocess.run([str(git),'status','--short'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8').stdout
 (OUT/'git_status_before.txt').write_text(status,encoding='utf-8')
 record=dict(task='SP-U9B-1',paper_sources=paper,frozen_files=frozen,allowed=ALLOWED,
  bibliography_mapping=dict(requested='paper/references.bib',active='paper/bibliography/references.bib',reason='main.tex already cites bibliography/references; necessary bibliography editing explicitly authorized; no main-document bibliography path changed.'),
  main_tex_scope='Only abstract block may change',git_status_before=str(OUT/'git_status_before.txt'))
 target.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'paper_files':len(paper),'frozen_files':len(frozen),'snapshot':str(target)}))
if __name__=='__main__':main()
