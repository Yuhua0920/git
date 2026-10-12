"""Read-only remote provenance verification; never update immutable source copies."""
import sys,json,zipfile,io,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b5a_common as c
from scripts.sp_u12b5a_reference import PINS,get
c.firewall();metadata=c.read(ROOT/'outputs/sp_u12b5a_mfa_reference_metadata.json');result=[]
for name,commit in PINS.items():
 entry={};errors=[]
 for key,url in [('repository',f'https://api.github.com/repos/benediktfesl/{name}'),('commits',f'https://api.github.com/repos/benediktfesl/{name}/commits?per_page=100'),('tags',f'https://api.github.com/repos/benediktfesl/{name}/tags')]:
  try:
   body=get(url);(c.OUT/f'{name}_verified_{key}.json').write_bytes(body);entry[key]=json.loads(body)
  except Exception as exc:errors.append(dict(url=url,type=type(exc).__name__,message=str(exc)))
 source=next(v for v in metadata['repositories'] if v['commit']==commit)
 try:
  archive=get(f'https://codeload.github.com/benediktfesl/{name}/zip/{commit}');(c.OUT/f'{name}_verified_commit.zip').write_bytes(archive)
  with zipfile.ZipFile(io.BytesIO(archive)) as z:hashes={'/'.join(Path(i.filename).parts[1:]):hashlib.sha256(z.read(i)).hexdigest() for i in z.infolist() if not i.is_dir()}
  assert hashes==source['files'],'REMOTE_AUTHOR_COPY_DIFFERS_FROM_PIN'
  source['remote_commit_archive_exact']=True
 except Exception as exc:errors.append(dict(url='codeload:'+name,type=type(exc).__name__,message=str(exc)));source['remote_commit_archive_exact']=False
 source['current_HEAD']=entry.get('commits',[{}])[0].get('sha');source['history_count']=len(entry.get('commits',[]));source['oldest_visible_commit_utc']=entry.get('commits',[{}])[-1].get('commit',{}).get('committer',{}).get('date');source['tags']=entry.get('tags',[])
 source['fresh_remote_check_utc']=c.utc();source['remote_verification_errors']=errors
 source['paper_period_code']='No 2023 commit in complete visible history' if entry.get('commits') and len(entry['commits'])<100 and all(v['commit']['committer']['date']>'2023-12-31' for v in entry['commits']) else '2023 code availability not established'
 result.append(dict(repository=name,current_HEAD=source['current_HEAD'],remote_archive_exact=source['remote_commit_archive_exact'],history_count=source['history_count'],oldest_visible_commit_utc=source['oldest_visible_commit_utc'],errors=errors))
c.report('reference_metadata',metadata);c.write(c.OUT/'remote_verification.json',dict(records=result,original_reference_modified=False))
print(json.dumps(dict(stage='B5A_remote_reference_check',records=result),ensure_ascii=False),flush=True)
