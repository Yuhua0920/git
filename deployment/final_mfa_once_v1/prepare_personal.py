"""Build/QA/report only. Never dispatches a real final fit."""
from pathlib import Path
import json,hashlib,csv,zipfile,shutil,sys,subprocess,time,uuid
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'lab_runs/final_mfa_once/preparation_personal'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def build():
 c=read(ROOT/'deployment/r6_formal_v1/input_contract.json');r8=read(ROOT/'deployment/r8_selection_v1/selection_contract.json')
 s=read(ROOT/'SELECTION_MANIFEST.json');digest=sha(ROOT/'SELECTION_MANIFEST.json')
 assert digest==(ROOT/'SELECTION_MANIFEST.sha256').read_text(encoding='ascii').strip()
 assert s['PASS']and s['selected']['K']==1 and s['selected']['rank']==16 and s['selected']['final_seed']==4143049057
 codes={n:h for n,h in c['frozen_hashes'].items()if n not in c['data']}
 for ns in('r6_formal_v1','r8_selection_v1'):
  m=read(ROOT/'deployment'/ns/'package_manifest.json')
  codes.update({n:v['sha256']for n,v in m['files'].items()});codes['deployment/'+ns+'/package_manifest.json']=sha(ROOT/'deployment'/ns/'package_manifest.json')
 n='deployment/r7_evidence_audit_r1_v1/source_only_loader.py';codes[n]=sha(ROOT/n)
 p={'format':'FINAL_ONLY_MINIMAL_BOUNDARY_V1','selected':next(x for x in c['candidate_order']if x['K']==1 and x['rank']==16),
    'selection_sha256':digest,'full_data':c['data']['outputs/sp_u12b5b/full_retraining.npy'],
    'R6_package_sha256':sha(ROOT/'deployment/r6_formal_v1/package_manifest.json'),'R6_input_contract_sha256':sha(ROOT/'deployment/r6_formal_v1/input_contract.json'),
    'R8_bindings':s['R8_bindings'],'code_hashes':codes,'target_denied':['outputs/sp_u12b2b_sealed','outputs/sp_u12b3g_next_sealed'],
    'pilot_order_sha256':c['pilot_order_sha256'],'noise_variances':c['noise_variances'],
    'resource_contract_sha256':sha(ROOT/'deployment/r6_formal_v1/resource_amendment.json')}
 write(HERE/'contract.json',p)
 members=[x for x in HERE.iterdir()if x.is_file()and x.name not in('package_manifest.json','prepare_personal.py')]
 manifest={'format':'FINAL_MFA_ONCE_INCREMENT_V1','files':{x.relative_to(ROOT).as_posix():{'sha256':sha(x),'bytes':x.stat().st_size}for x in sorted(members)}}
 write(HERE/'package_manifest.json',manifest);members.append(HERE/'package_manifest.json')
 dest=OUT/('package_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);dest.mkdir()
 z=dest/'SP_U12B5_FINAL_MFA_Once.zip'
 with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED)as f:
  for x in sorted(members):f.write(x,x.relative_to(ROOT).as_posix())
 digest=sha(z);(dest/(z.name+'.sha256')).write_text(digest+'  '+z.name+'\n',encoding='ascii')
 with(dest/'SP_U12B5_FINAL_File_Manifest.csv').open('x',encoding='utf8',newline='')as f:
  w=csv.writer(f);w.writerow(['path','bytes','sha256'])
  for x in sorted(members):w.writerow([x.relative_to(ROOT).as_posix(),x.stat().st_size,sha(x)])
 shutil.copy2(HERE/'Install-FinalMFA.ps1',dest/'Install-FinalMFA.ps1')
 result={'ZIP':str(z),'sha256':digest,'bytes':z.stat().st_size,'files':len(members),'installer_sha256':sha(dest/'Install-FinalMFA.ps1'),'manifest_sha256':sha(HERE/'package_manifest.json'),'selection_sha256':p['selection_sha256']}
 write(OUT/'latest_package.json',result);print(json.dumps(result,indent=2))
def preserve():
 b=read(OUT/'historical_before.json');changed=[]
 for n,v in b['files'].items():
  p=ROOT/n
  if not p.exists()or sha(p)!=v['sha256']:changed.append(n)
 now=subprocess.check_output(['git','status','--short','--untracked-files=no'],cwd=ROOT,text=True)
 result={'PASS':not changed and now==b['git_status'],'files_checked':len(b['files']),'changed':changed,'tracked_git_status_unchanged':now==b['git_status'],
         'personal_real_final_executed':False,'validation_recomputed':False,'target_access':0,'B2_B3_outcomes_opened':False}
 write(OUT/'preservation.json',result);print(json.dumps(result,indent=2));assert result['PASS']
def qa():
 pkg=read(OUT/'latest_package.json');root=ROOT/'lab_runs/final_mfa_once'/('win_'+uuid.uuid4().hex[:8])/'移植 Root';root.mkdir(parents=True)
 for n in read(HERE/'contract.json')['code_hashes']:
  p=root/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
 original={n:sha(root/n)for n in read(HERE/'contract.json')['code_hashes']}
 logs=root.parent/'qa_logs';logs.mkdir()
 def ps(label,args,expected=0):
  run=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass']+args,capture_output=True,text=True,encoding='utf8',errors='replace')
  (logs/(label+'.stdout.log')).write_text(run.stdout,encoding='utf8');(logs/(label+'.stderr.log')).write_text(run.stderr,encoding='utf8')
  assert (run.returncode==expected)if expected==0 else(run.returncode!=0),(label,run.stdout,run.stderr)
  return {'label':label,'exit_code':run.returncode,'expected_refusal':expected!=0}
 installer=str(Path(pkg['ZIP']).parent/'Install-FinalMFA.ps1');checks=[]
 checks.append(ps('install',['-File',installer,'-ProjectRoot',str(root),'-ZipPath',pkg['ZIP'],'-ExpectedSha256',pkg['sha256']]))
 checks.append(ps('install_again_refused',['-File',installer,'-ProjectRoot',str(root),'-ZipPath',pkg['ZIP'],'-ExpectedSha256',pkg['sha256']],1))
 tool=str(root/'deployment/final_mfa_once_v1/FinalMFA.ps1')
 def action(label,name,extra=(),expected=0):return ps(label,['-File',tool,'-PythonPath',sys.executable,'-Action',name]+list(extra),expected)
 checks.append(action('verify','Verify'))
 checks.append(action('start_without_confirmation_refused','StartFinal',expected=1))
 checks.append(action('selection_action_refused','CompleteSelection',expected=1))
 checks.append(action('validation_action_refused','ValidateCandidate',expected=1))
 checks.append(action('new_toy_tests','ToyTests'))
 checks.append(action('monitor_background','MonitorToy',['-Background']))
 begin=time.monotonic();done=root/'lab_runs/final_mfa_once/toy_monitor.json'
 while not done.exists()and time.monotonic()-begin<45:time.sleep(.2)
 assert done.exists(),'Background toy missing; preserve launcher logs; no retry'
 toy=read(done);assert toy['PASS']and not toy['scientific_fit_executed']
 checks.append(action('status','Status'))
 checks.append(action('export','Export'))
 unchanged={n:sha(root/n)==h for n,h in original.items()};assert all(unchanged.values())
 report={'PASS':True,'root':str(root),'checks':checks,'old_copied_dependencies_unchanged':len(unchanged),
         'background_toy':toy,'real_final_executed':False,'target_access':0,'no_preparation_arrays_copied':not(root/'outputs/sp_u12b5b/full_retraining.npy').exists()}
 write(root.parent/'windows_qa.json',report);write(OUT/'latest_windows_qa.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
def report():
 import numpy as np
 pkg=read(OUT/'latest_package.json');qa=read(OUT/'latest_windows_qa.json');pres=read(OUT/'preservation.json');p=read(HERE/'contract.json')
 s=read(ROOT/'SELECTION_MANIFEST.json');digest=sha(ROOT/'SELECTION_MANIFEST.json')
 assert digest==p['selection_sha256']==(ROOT/'SELECTION_MANIFEST.sha256').read_text(encoding='ascii').strip()
 with(ROOT/'outputs/sp_u12b5b/full_retraining.npy').open('rb')as f:
  version=np.lib.format.read_magic(f);shape,order,dtype=np.lib.format.read_array_header_1_0(f);offset=f.tell()
 assert list(shape)==[15000,4032]and dtype.str=='<c16'and not order
 manifest=read(HERE/'package_manifest.json');assert pkg['manifest_sha256']==sha(HERE/'package_manifest.json')and pkg['sha256']==sha(Path(pkg['ZIP']))
 logs=list((ROOT/'lab_runs/final_mfa_once/synthetic').rglob('pytest.log'))+list((Path(qa['root'])/'lab_runs/final_mfa_once/synthetic').rglob('pytest.log'))
 passed=[{'path':str(x),'summary':x.read_text(encoding='utf8').splitlines()[-1]}for x in logs if '20 passed'in x.read_text(encoding='utf8')]
 assert passed and qa['PASS']and pres['PASS']
 result={'verdict':'FINAL_MFA_ONCE_DEPLOYMENT_READY','package':pkg,'personal_selection_evidence':{'PASS':s['PASS'],'selected':{k:s['selected'][k]for k in('name','K','rank','seed','final_seed')},'sha256':digest,'path':str(ROOT/'SELECTION_MANIFEST.json'),'lab_live_state_not_rechecked_from_personal_host':True},
   'full_data':{'sha256':p['full_data']['sha256'],'shape':list(shape),'dtype':dtype.str,'fortran_order':order,'header_bytes':offset,'bytes':(ROOT/'outputs/sp_u12b5b/full_retraining.npy').stat().st_size},
   'tests':passed,'windows_qa':qa,'preservation':pres,
   'changed_existing_files':0,'new_deployment_files':list(manifest['files'])+['deployment/final_mfa_once_v1/package_manifest.json'],
   'new_personal_helper_not_packaged':'deployment/final_mfa_once_v1/prepare_personal.py',
   'real_final_fit_executed':False,'validation_recomputed':False,'selection_changed':False,'target_access':0,'B2_B3_outcome_reads':0,'manuscript_modified_this_task':False,'Git_commit_created':False,
   'historical_failures_and_intermediate_packages_preserved':True,
   'reuse':{'fit':'same worker.train code object; only numeric_inputs permission callback replaced by explicit final-only lease',
     'monitor':'same supervisor.run_worker code object; only worker-script directory and mutable-active-pointer write route replaced',
     'common_approval_modified':False,'author_adapter_modified':False,'old_R6_scope_modified':False,
     'new_scope':'exact authenticated source/static metadata, final-only numeric input, own new model; no candidate/validation/target data; writes only new final/job namespace and owned lock'},
   'lab_launch_prerequisites':['Verify and Preflight PASS on laboratory host','real canonical selection SHA matches pinned transferred evidence','full data SHA/header checked on lab host','no prior final, STOP or concurrent job','actual operator explicit one-time confirmation'],
   'final_model_output':'D:\\Research\\Project_MFA_Lab\\lab_runs\\r6_formal\\final\\model',
   'final_freeze_manifest':'D:\\Research\\Project_MFA_Lab\\lab_runs\\r6_formal\\FINAL_MODEL_MANIFEST.json'}
 write(OUT/'delivery_report.json',result);print(json.dumps({'verdict':result['verdict'],'package':pkg,'pytest':passed,'preservation':pres,'real_final_fit_executed':False},ensure_ascii=False,indent=2))
if __name__=='__main__':
 {'build':build,'preserve':preserve,'qa':qa,'report':report}[sys.argv[1]]()
