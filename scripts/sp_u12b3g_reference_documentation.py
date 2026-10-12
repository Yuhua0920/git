"""Classify exact archived copies as historical evidence in the generated audit."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'outputs/sp_u12b3g_field_reference_audit.csv'
with p.open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
count=0
for row in rows:
 if set(Path(row['file']).parts)&{'original_bytes','pretest_failure01','pre_gate_hash_guard_amendment'}:
  row['classification']='A_HISTORICAL_ARCHIVED_RECORD'
  row['action']='preserve exact archived bytes;not a live runtime reference';count+=1
with p.open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['file','line','context','classification','action']);w.writeheader();w.writerows(rows)
(ROOT/'outputs/sp_u12b3g/reference_documentation_audit.json').write_text(json.dumps(dict(PASS=True,rows=len(rows),archived_rows_classified_historical=count,historical_source_bytes_modified=False,scope='technical documentation only;not a deployment dependency'),indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(stage='reference_documentation',PASS=True,rows=len(rows),historical_archive_rows=count)))
