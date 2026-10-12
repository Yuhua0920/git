"""Read-only preservation snapshot for the literature gate."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/sp_u9a_r1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=True)
 target=OUT/'preservation_before.json'
 if target.exists():raise RuntimeError('Preservation baseline already exists')
 files={}
 for name in ('paper','src','configs','scripts','tests','docs'):
  for p in (ROOT/name).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and 'sp_u9a_r1' not in p.name:files[p.relative_to(ROOT).as_posix()]=sha(p)
 prior=json.loads((ROOT/'outputs/sp_u9a/full_source_inventory.json').read_text(encoding='utf-8'))
 # Hash existing evidence without parsing scientific outcomes.
 rows=prior if isinstance(prior,list) else prior.get('files',[])
 for row in rows:
  if isinstance(row,dict) and row.get('path'):
   p=ROOT/row['path']
   if p.is_file():files[row['path']]=sha(p)
 for p in (ROOT/'outputs').glob('sp_u9a*'):
  if p==OUT:continue
  candidates=[p] if p.is_file() else list(p.rglob('*'))
  for q in candidates:
   if q.is_file():files[q.relative_to(ROOT).as_posix()]=sha(q)
 target.write_text(json.dumps(files,indent=2),encoding='utf-8')
 print(json.dumps({'preserved_files':len(files),'baseline_sha256':sha(target)}))
if __name__=='__main__':main()
