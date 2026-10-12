"""Extract only public scholarly fields; redact anonymous redirect IDs."""
import json,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u9a_r1'
def main():
 p=OUT/'access_attempts.json';attempts=json.loads(p.read_text(encoding='utf-8'))
 for row in attempts:
  if 'final_url' in row:row['final_url']=re.sub(r';jsessionid=[^?;/]+',';jsessionid=REDACTED',row['final_url'])
 p.write_text(json.dumps(attempts,ensure_ascii=False,indent=2),encoding='utf-8')
 records=[]
 for pid in ('P4','P5'):
  text=(OUT/(pid+'_publisher_link0.html')).read_text(encoding='utf-8')
  match=re.search(r'xplGlobal.document.metadata=(\{.*?\});',text)
  m=json.loads(match.group(1));keep={k:m.get(k) for k in ('title','authors','doi','abstract','publicationTitle','publicationYear','publicationDate','conferenceDate','startPage','endPage','volume','issue','isOpenAccess','pdfPath','pdfUrl','sections')}
  records.append({'id':pid,**keep})
 (OUT/'publisher_oz_metadata.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(records,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
