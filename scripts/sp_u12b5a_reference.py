"""Acquire and pin primary reference evidence before any project MFA code."""
import sys,json,urllib.request,urllib.error,zipfile,io,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b5a_common as c
PINS={'cplx-mfa':'9f9c22a7775076fbedd0fb73929c195b0dd2aa6b','MFA_estimator':'d11915070e676579008be6f3a02b35dffb1fa189'}
def get(url):
 request=urllib.request.Request(url,headers={'User-Agent':'SP-U12B5A-reference-audit','Accept':'application/vnd.github+json' if 'api.github.com' in url else '*/*'})
 with urllib.request.urlopen(request,timeout=25) as response:return response.read()
def prepare():
 c.firewall();c.REF.mkdir(parents=True,exist_ok=True);receipts=[]
 for name,commit in PINS.items():
  old=ROOT/'outputs/sp_u3a_sources'/name;pin=c.read(old/'repository_pin.json')
  assert pin['commit']==commit
  for path,digest in pin['files'].items():assert c.sha(old/path)==digest,'OLD_REFERENCE_DRIFT'
  live={};errors=[]
  for key,url in [('repository',f'https://api.github.com/repos/benediktfesl/{name}'),('commits',f'https://api.github.com/repos/benediktfesl/{name}/commits?per_page=100'),('tags',f'https://api.github.com/repos/benediktfesl/{name}/tags')]:
   try:
    body=get(url);(c.OUT/f'{name}_{key}.json').write_bytes(body);live[key]=json.loads(body)
   except Exception as exc:errors.append(dict(url=url,type=type(exc).__name__,message=str(exc)))
  destination=c.REF/name;assert not destination.exists(),'IMMUTABLE_REFERENCE_ALREADY_EXISTS';destination.mkdir()
  try:
   archive=get(f'https://codeload.github.com/benediktfesl/{name}/zip/{commit}');(c.OUT/f'{name}_{commit}.zip').write_bytes(archive)
   with zipfile.ZipFile(io.BytesIO(archive)) as z:
    for item in z.infolist():
     if item.is_dir():continue
     parts=Path(item.filename).parts[1:];target=destination.joinpath(*parts).resolve();assert target.is_relative_to(destination.resolve())
     target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(item))
   source='Fresh GitHub commit archive'
  except Exception as exc:
   errors.append(dict(url=f'codeload:{name}:{commit}',type=type(exc).__name__,message=str(exc)))
   assert not any(destination.rglob('*')),'PARTIAL_REFERENCE_ARCHIVE_STOP'
   for path in pin['files']:
    target=destination/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old/path,target)
   source='Byte-verified existing original author commit copy; no outcome files'
  hashes={x.relative_to(destination).as_posix():c.sha(x) for x in destination.rglob('*') if x.is_file()}
  assert hashes==pin['files'],'DOWNLOADED_REFERENCE_NOT_IDENTICAL_TO_MENTOR_PIN'
  receipts.append(dict(repository=pin['repository'],owner='benediktfesl',branch=live.get('repository',{}).get('default_branch','main'),commit=commit,commit_timestamp=pin['commit_timestamp'],accessed_utc=c.utc(),license='BSD-3-Clause; cplx-mfa contains MIT-derived mofa code, see NOTICE',version='0.1.1',files=hashes,tree_sha256=c.tree(hashes),README_sha256=hashes['README.md'],source=source,current_HEAD=live.get('commits',[{}])[0].get('sha'),history_count=len(live.get('commits',[])),oldest_visible_commit_utc=live.get('commits',[{}])[-1].get('commit',{}).get('committer',{}).get('date'),tags=live.get('tags',[]),network_errors=errors,paper_period_code='2023 code not present in visible 2026 repository history; not claimed as exact 2023 implementation'))
 for name in ('paper_v2.pdf','paper_v2.txt'):
  shutil.copyfile(ROOT/'outputs/sp_u3a_sources'/name,c.REF/name)
 paper=dict(title='Low-Rank Structured MMSE Channel Estimation with Mixtures of Factor Analyzers',authors=['Benedikt Fesl','Nurettin Turan','Wolfgang Utschick'],venue='57th Asilomar Conference on Signals, Systems, and Computers (ACSSC)',year=2023,pages='375-380',DOI='10.1109/IEEECONF59524.2023.10477088',paper_version='arXiv:2304.14809v2, 2023-05-01',arxiv_url='https://arxiv.org/abs/2304.14809v2',publisher_verification='https://portal.fis.tum.de/en/publications/low-rank-structured-mmse-channel-estimation-with-mixtures-of-fact/',pdf_sha256=c.sha(c.REF/'paper_v2.pdf'),text_sha256=c.sha(c.REF/'paper_v2.txt'),supplement='No separate SI referenced by the six-page paper; none downloaded')
 c.report('reference_metadata',dict(PASS=True,paper=paper,repositories=receipts,REFERENCE_MFA_IMMUTABLE_COPY=True,reference_combined_tree_sha256=c.tree({v['repository']:v['tree_sha256'] for v in receipts}),no_auto_update=True,legacy_modified_U3_execution_code_used=False,original_reference_modified=False))
 print(json.dumps(dict(stage='B5A_reference_pin',PASS=True,commits=PINS,remote_HEAD={v['repository']:v['current_HEAD'] for v in receipts},network_errors=sum(len(v['network_errors']) for v in receipts))),flush=True)
if __name__=='__main__':prepare()
