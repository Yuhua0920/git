"""Bounded public metadata/source access; never imports scientific code."""
import hashlib,json,time,urllib.request,urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u9a_r1'
DOIS=['10.1109/TSP.2022.3194348','10.1109/IEEECONF56349.2022.10051921','10.1109/IEEECONF59524.2023.10477088','10.1109/TWC.2006.04713','10.1109/WAMIC.2005.1528390','10.1049/iet-com.2014.0979','10.1049/cmu2.12250']
def main():
 records=[]
 for i,doi in enumerate(DOIS,1):
  url='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')
  try:
   req=urllib.request.Request(url,headers={'User-Agent':'SP-U9A-R1-research-audit/1.0'})
   with urllib.request.urlopen(req,timeout=25) as r:data=r.read()
   dest=OUT/f'crossref_P{i}.json';dest.write_bytes(data);m=json.loads(data)['message']
   records.append({'id':f'P{i}' if i<7 else 'Mei','doi':doi,'url':url,'title':m.get('title'),'authors':m.get('author'),'published':m.get('published'),'container':m.get('container-title'),'volume':m.get('volume'),'issue':m.get('issue'),'page':m.get('page'),'links':m.get('link'),'abstract':m.get('abstract'),'sha256':hashlib.sha256(data).hexdigest()})
  except Exception as e:records.append({'doi':doi,'url':url,'error':str(e)})
  time.sleep(.2)
 (OUT/'metadata_audit.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(records,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
