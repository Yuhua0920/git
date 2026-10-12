"""Lawful, bounded OA locator and publisher access audit."""
import json,urllib.request,urllib.parse,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u9a_r1'
def main():
 attempts=[]
 urls={
 'P4_openalex':'https://api.openalex.org/works/https://doi.org/10.1109/TWC.2006.04713',
 'P5_openalex':'https://api.openalex.org/works/https://doi.org/10.1109/WAMIC.2005.1528390',
 'P4_s2':'https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TWC.2006.04713?fields=title,authors,year,openAccessPdf,url',
 'P5_s2':'https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/WAMIC.2005.1528390?fields=title,authors,year,openAccessPdf,url',
 'P6_publisher':'https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-com.2014.0979',
 'Mei_publisher':'https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cmu2.12250',
 'Oz_dissertation':'https://surface.syr.edu/eecs_etd/59/',
 'Oz_author_index':'https://mebis.medipol.edu.tr/EgitimKadrosu?pBirimOID=10223',
 }
 for pid in ('P4','P5'):
  metadata=json.loads((OUT/f'crossref_{pid}.json').read_text())['message']
  urls[pid+'_doi']=metadata['URL']
  for n,link in enumerate(metadata.get('link',[])):urls[pid+f'_publisher_link{n}']=link['URL']
 for name,url in urls.items():
  try:
   req=urllib.request.Request(url,headers={'User-Agent':'SP-U9A-R1-research-audit/1.0'})
   with urllib.request.urlopen(req,timeout=20) as r:data=r.read();status=r.status;ctype=r.headers.get('Content-Type','');final=r.url
   suffix='.json' if 'json' in ctype else '.pdf' if data.startswith(b'%PDF') else '.html'
   path=OUT/(name+suffix);path.write_bytes(data)
   final=re.sub(r';jsessionid=[^?;/]+',';jsessionid=REDACTED',final)
   row={'id':name,'url':url,'final_url':final,'status':status,'content_type':ctype,'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest(),'pdf_valid_header':data.startswith(b'%PDF')}
   if 'openalex' in name:
    m=json.loads(data);row['oa']=m.get('open_access');row['locations']=[{k:x.get(k) for k in ('landing_page_url','pdf_url','is_oa')} for x in m.get('locations',[])]
   if '_s2' in name:row['metadata']=json.loads(data)
   attempts.append(row)
  except Exception as e:attempts.append({'id':name,'url':url,'error':str(e)})
 (OUT/'access_attempts.json').write_text(json.dumps(attempts,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(attempts,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
