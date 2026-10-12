"""Render requested manuscript regions with Poppler and retain actual PDF evidence."""
import hashlib,json,re,subprocess
from pathlib import Path
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u9b1'
POPPLER=Path(r'C:\Users\yuhua\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin')
def main():
 pdf=OUT/'build_post/main.pdf';reader=PdfReader(pdf)
 texts=[p.extract_text() or '' for p in reader.pages]
 # All three front-half pages plus the actual bibliography start and continuation.
 refs=next(i+1 for i,t in enumerate(texts) if '\nReferences\n' in t or t.startswith('References\n'))
 front_end=next(i+1 for i,t in enumerate(texts) if 'System Model' in t)
 pages=sorted(set(range(1,front_end+1))|{refs,refs+1})
 records=[]
 for page in pages:
  prefix=OUT/f'qa_page_{page:02d}'
  command=[str(POPPLER/'pdftoppm.exe'),'-f',str(page),'-l',str(page),'-singlefile','-scale-to','1800','-png',str(pdf),str(prefix)]
  r=subprocess.run(command,capture_output=True,text=True)
  if r.returncode:raise RuntimeError(r.stderr)
  png=prefix.with_suffix('.png')
  records.append(dict(page=page,image=str(png),sha256=hashlib.sha256(png.read_bytes()).hexdigest(),render_exit_code=r.returncode))
 (OUT/'pdf_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n'+t for i,t in enumerate(texts)),encoding='utf-8')
 report=dict(pdf=str(pdf),pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),pages_total=len(reader.pages),pdf_metadata=dict(reader.metadata),
  reviewed_pages=records,front_end=front_end,references_start=refs,actual_complete_multifile_build=True,
  review_status='PENDING visual inspection',author_metadata='Existing TODO Authors/Affiliations preserved, not invented',
  checks_required=['title','abstract','introduction','six related-work paragraphs','references','citation rendering','equations','line overflow','heading hierarchy','author metadata','page breaks'])
 (OUT/'visual_qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(pages=pages,total=len(reader.pages),metadata=dict(reader.metadata)),ensure_ascii=False))
if __name__=='__main__':main()
