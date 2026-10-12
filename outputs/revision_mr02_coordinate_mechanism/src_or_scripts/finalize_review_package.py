"""Repair report table spacing and refresh the review ZIP; no science changes."""
import csv,hashlib,json,os,re,zipfile
from pathlib import Path
OUT=Path(r'D:\Users\yuhua\Desktop\Project\outputs\revision_mr02_coordinate_mechanism')
ROOT=OUT.parents[1];MR=ROOT/'outputs/revision_mr01_pooled_lmmse'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if OUT.resolve()!=Path(r'D:\Users\yuhua\Desktop\Project\outputs\revision_mr02_coordinate_mechanism'):raise ValueError('Scope')
rp=OUT/'reports/chinese_scientific_report.md'
rp.write_text(re.sub(r'\n\n(?=\|)','\n',rp.read_text(encoding='utf8')),encoding='utf8')
mp=OUT/'results_review_file_manifest.csv'
with mp.open(encoding='utf8',newline='') as f:previous=list(csv.DictReader(f))
items={}
permitted={'reports/chinese_scientific_report.md','src_or_scripts/review_mr02.py','reports/review_export_console.log'}
changed=[]
for row in previous:
    name=row['path']
    p=MR/name[len('MR01_key/'):] if name.startswith('MR01_key/') else ROOT/Path(name).name if name.startswith('taskbook/') else OUT/name
    digest=sha(p)
    if digest!=row['sha256']:
        if name not in permitted:raise ValueError('Unexpected scientific review material change:'+name)
        changed.append(name)
    items[name]=p
newscript=OUT/'src_or_scripts/finalize_review_package.py';items[newscript.relative_to(OUT).as_posix()]=newscript
manifest=[dict(path=name,bytes=p.stat().st_size,sha256=sha(p)) for name,p in sorted(items.items())]
with mp.open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(manifest)
tmp=OUT/'results_review_final_build.zip';zp=OUT/'results_review.zip'
with zipfile.ZipFile(tmp,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,p in sorted(items.items()):z.write(p,name)
    z.write(mp,'results_review_file_manifest.csv')
with zipfile.ZipFile(tmp) as z:
    if z.testzip():raise ValueError('Final package CRC')
    for row in manifest:
        if hashlib.sha256(z.read(row['path'])).hexdigest()!=row['sha256']:raise ValueError('Final member SHA')
os.replace(tmp,zp)
digest=sha(zp);zp.with_suffix('.zip.sha256').write_text(digest+'  results_review.zip\n',encoding='ascii')
record=dict(path=str(zp),bytes=zp.stat().st_size,sha256=digest,files=len(manifest)+1,CRC_and_all_member_SHA256='PASS',excluded='full H/noise, operators, literature PDFs, historical formal outcomes',role='lightweight independent numeric review, not complete vector reproduction',final_report_layout_only=True,changed_members=changed,scientific_members_changed=0,initial_package_preserved='results_review_initial.zip',manifest_scope='members of final ZIP')
(OUT/'review_export.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps(record))
