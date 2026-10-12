"""Prepare an isolated R2 execution package; no formal science runs here."""
from pathlib import Path
import argparse, ast, csv, difflib, hashlib, json, shutil, subprocess, sys, zipfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'deployment/b5c_paired_v1';R1=ROOT/'deployment/b5c_paired_r1_v1'
NEW=ROOT/'deployment/b5c_paired_r2_v1';OUT=ROOT/'outputs/sp_u12b5c_r2'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def need(v,m):
 if not v:raise RuntimeError(m)
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def snapshot():
 previous=read(ROOT/'outputs/sp_u12b5c_r1/preservation_before.json')
 files={ROOT/n for n in previous['files']}
 for folder in (R1,ROOT/'lab_runs/b5c_paired_r1'):
  files.update(p for p in folder.rglob('*') if p.is_file())
 files.update(p for p in (ROOT/'outputs/sp_u12b5c_r1').glob('*') if p.is_file())
 records={}
 for i,q in enumerate(sorted(files)):
  records[q.relative_to(ROOT).as_posix()]={'sha256':sha(q),'bytes':q.stat().st_size}
  if i%10000==0:print('Opaque original-file hashes:',i,flush=True)
 return {'files':records,'git_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  'tracked_diff':hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=ROOT)).hexdigest(),
  'staged_diff':hashlib.sha256(subprocess.check_output(['git','diff','--cached','--binary'],cwd=ROOT)).hexdigest()}
def before():
 v=snapshot();dump(OUT/'preservation_before.json',v);print(json.dumps({'files':len(v['files']),'bytes':sum(x['bytes'] for x in v['files'].values())}))
def after():
 a=read(OUT/'preservation_before.json');z=snapshot();changed=[n for n,v in a['files'].items() if z['files'].get(n)!=v]
 additions=sorted(set(z['files'])-set(a['files']))
 ok=not changed and not additions and all(a[k]==z[k] for k in ('git_HEAD','tracked_diff','staged_diff'))
 v={'PASS':ok,'protected_files':len(a['files']),'bytes':sum(x['bytes'] for x in a['files'].values()),'changed':changed,'unexpected_additions':additions,
  'B2_B3_preservation':'all local sealed files opaque-byte SHA256; no performance decoding','original_B5C_and_R1_modified':False if ok else None,
  'manuscript_modified':False if ok else None,'production_modified':False if ok else None,'commit_created':False,'real_formal_evaluation_executed':False,
  'actual_LAB_failed_attempt_physically_verified':False}
 dump(OUT/'preservation_after.json',v);need(ok,'ORIGINAL_FILE_DRIFT');print(json.dumps(v))
def initialize():
 need(not NEW.exists(),'R2_NAMESPACE_EXISTS');NEW.mkdir(parents=True)
 for name in ('base.py','control.py','worker.py','paired_stats.py','report.py','publication_plots.py'):
  shutil.copy2(OLD/name,NEW/name)
 shutil.copy2(R1/'science.py',NEW/'science.py')
 # Share already deployed metadata; no data/model migration or duplicate manifests.
 q=NEW/'base.py';t=q.read_text(encoding='utf8')
 for a,z in [("OUT = ROOT/'lab_runs/b5c_paired'","OUT = ROOT/'lab_runs/b5c_paired_r2'"),
  ("read(HERE/'contract.json')","read(ROOT/'deployment/b5c_paired_v1/contract.json')"),
  ("read(HERE/'data_manifest.json')","read(ROOT/'deployment/b5c_paired_v1/data_manifest.json')")]:
  need(t.count(a)==1,'BASE_PATCH_ANCHOR');t=t.replace(a,z)
 q.write_text(t,encoding='utf8',newline='')
 e=read(ROOT/'outputs/sp_u12b5c_r1/identity_authenticated_source_loading.json')
 binding={'original_B5C_manifest_sha256':sha(OLD/'package_manifest.json'),'R1_manifest_sha256':sha(R1/'package_manifest.json'),
  'contract_sha256':sha(OLD/'contract.json'),'data_manifest_sha256':sha(OLD/'data_manifest.json'),
  'R1_science_sha256':sha(R1/'science.py'),'input_identity_digests':{k:e[k] for k in ('H_identity_digest','noise_identity_digest','pilot_truth_Y_identity_digest')},
  'required_R1_counts':{k:e[k] for k in ('authenticated_metadata_files','EVAL_registry_entries','raw_trajectories_checked','channel_seed_scalar_values_checked','communication_seed_scalar_values_checked','original_failure_reproductions','corrected_proof_passes','full_input_identities_checked','scored_input_identities')},
  'cells':e['cells'],'preparation_only':True,'formal_on_personal':False,'original_attempt':'lab_runs/b5c_paired','new_output':'lab_runs/b5c_paired_r2',
  'R1_job_prefix':'lab_runs/b5c_paired_r1/technical/identity_','no_approval_record_created':True}
 dump(NEW/'bindings.json',binding);print('R2 namespace initialized; no formal execution')
def manifest():
 files=dict(read(R1/'package_manifest.json')['files'])
 files['deployment/b5c_paired_r1_v1/package_manifest.json']={'sha256':sha(R1/'package_manifest.json'),'bytes':(R1/'package_manifest.json').stat().st_size}
 for q in NEW.rglob('*'):
  if q.is_file() and q.name!='package_manifest.json':files[q.relative_to(ROOT).as_posix()]={'sha256':sha(q),'bytes':q.stat().st_size}
 q=NEW/'package_manifest.json'
 if q.exists():q.unlink() # New preparation-only namespace; never old freeze files.
 dump(q,{'files':files,'scope':'Independent R2; identical R1-fixed science and original statistics; prerequisites verified before one launch'})
 print('R2 hash-bound files:',len(files))
def source_audit():
 need((NEW/'science.py').read_bytes()==(R1/'science.py').read_bytes(),'SCIENCE_DIFFERS_FROM_CERTIFIED_R1')
 original=(OLD/'science.py').read_text(encoding='utf8');current=(NEW/'science.py').read_text(encoding='utf8')
 a='expected={r:index[condition,i,-1,r] for r in ROLES[:3]}'
 z='expected={r:index[condition,i,-1 if r in ROLES[:3] else 0,r] for r in ROLES}'
 need(original.replace(a,z)==current,'MORE_THAN_SINGLE_SCIENCE_FIX')
 identical={n:sha(OLD/n)==sha(NEW/n) for n in ('paired_stats.py','report.py','publication_plots.py')};need(all(identical.values()),'STATISTICS_OR_REPORT_DRIFT')
 need(read(NEW/'bindings.json')['data_manifest_sha256']==sha(OLD/'data_manifest.json'),'DATA_BINDING_DRIFT')
 diffs=[]
 for n in ('base.py','control.py','worker.py','science.py'):
  aa=(OLD/n).read_text(encoding='utf8');zz=(NEW/n).read_text(encoding='utf8')
  diffs.extend(difflib.unified_diff(aa.splitlines(True),zz.splitlines(True),fromfile='original/'+n,tofile='R2/'+n))
 (OUT/'execution_changes.diff').write_text(''.join(diffs),encoding='utf8')
 v={'PASS':True,'science_identical_to_R1':True,'science_changes_vs_original':1,'unchanged_files':identical,
  'contract_sha256':sha(OLD/'contract.json'),'data_manifest_sha256':sha(OLD/'data_manifest.json'),
  'grid':{'conditions':5,'SNR':7,'trajectories_per_cell':1000,'frames':[4,5,6,7,8,9],'positions_per_method':210000},
  'statistics':read(OLD/'contract.json')['statistics'],'model_changed':False,'LR4_changed':False,'failed_attempt_scientific_results_used':False}
 dump(OUT/'science_preservation.json',v);print(json.dumps(v,indent=2))
def package():
 manifest();names=sorted(q.relative_to(ROOT).as_posix() for q in NEW.rglob('*') if q.is_file())
 path=OUT/'SP_U12B5C_R2_Execution.zip';need(not path.exists(),'ZIP_ALREADY_EXISTS')
 records={n:{'sha256':sha(ROOT/n),'bytes':(ROOT/n).stat().st_size} for n in names}
 with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as f:
  for n in names:f.write(ROOT/n,n)
 with zipfile.ZipFile(path) as f:
  need(f.testzip() is None,'ZIP_CRC_ERROR')
  for n,v in records.items():need(hashlib.sha256(f.read(n)).hexdigest()==v['sha256'],'ZIP_MEMBER_HASH_DRIFT')
 path.with_suffix('.zip.sha256').write_text(sha(path)+'\n',encoding='ascii')
 with (OUT/'SP_U12B5C_R2_File_Manifest.csv').open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=('relative_path','bytes','sha256'));w.writeheader()
  for n,v in records.items():w.writerow({'relative_path':n,**v})
 shutil.copy2(NEW/'Install-R2.ps1',OUT/'Install-R2.ps1')
 v={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'files':len(names),'unpacked_bytes':sum(v['bytes'] for v in records.values()),
  'data_migrated':False,'model_migrated':False,'formal_evaluation_on_personal':False,'ZIP_all_members_verified':True}
 dump(OUT/'package_report.json',v);print(json.dumps(v,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=('before','after','initialize','manifest','source-audit','package'));x=p.parse_args()
 {'before':before,'after':after,'initialize':initialize,'manifest':manifest,'source-audit':source_audit,'package':package}[x.action]()
