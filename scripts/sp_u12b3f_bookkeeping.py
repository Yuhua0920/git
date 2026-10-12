"""Run3 technical preflight/preservation/seal only; frozen scientific CLI is untouched.

No scientific values are printed. No generation, fit, bootstrap or retry is implemented.
Coordinate export is permitted only after the frozen runner's full integrity/compute.
"""
from pathlib import Path
import sys, os, json, hashlib, time, concurrent.futures, subprocess, csv, shutil, traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'outputs/sp_u12b3f'
SCI = ROOT/'outputs/sp_u12b3f_run3_sealed'
TECH = ROOT/'outputs/sp_u12b3e/run3_control'
CORE = '994339ccc806e3ca94ae644a954e9382a2fa0df8ed2a808d0feb28528e9bdfa1'
WRAPPER = '32a35ac8fdfb59bfba22e49821b4d0b52d6392ea376dd63493cfc3fe92e30a42'
FREEZE = '4e79f8f8bb267f4bf8be463bd2cee6b7d25914218a0c36936543e96d52f45ce6'
RUN1 = '530cc5bc124bfc945a381fb6a3d30ad2ec70666c2887a48aa9b69f9847d8a250'
RUN2 = 'a22d244473336acda5ab8b75d9a848e44f8b1fcab19fea2391b93649d361e7bb'
CONDITIONS = ('ID','Profile','Mobility','Composed','Boundary')
TASK = Path(r'C:\Users\yuhua\.codex\attachments\96364209-a3fb-494a-be48-e6a8bca990a8\已粘贴的文本.txt')
COMMAND = [str(ROOT/'.venv-cdl/Scripts/python.exe'),'-X','utf8','scripts/run_sp_u12b3f.py','--run']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(8*1024*1024),b''):h.update(v)
 return h.hexdigest()
def tree(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def report(n,o):write(ROOT/'outputs'/('sp_u12b3f_run3_'+n+'.json'),o)
def need(v,code):
 if not v:raise RuntimeError(code)
def owned(p):return any(v.startswith(('sp_u12b3f','test_sp_u12b3f')) for v in p.parts)
def historical_cache(p):return any(v.startswith(('sp_u12b3b','sp_u12b3c','sp_u12b3d','sp_u12b3e','run_sp_u12b3d','test_sp_u12b3d')) for v in p.parts)
def paths():
 skip={'__pycache__','.pytest_cache','pytest_cache','pytest_tmp'}
 return sorted(p for d in ('src','scripts','tests','configs','docs','paper','outputs') for p in (ROOT/d).rglob('*') if p.is_file() and not owned(p.relative_to(ROOT)) and (historical_cache(p.relative_to(ROOT)) or not(set(p.parts)&skip)))
def frozen():
 from scripts.sp_u12b3e_build import core
 f=read(ROOT/'outputs/sp_u12b3e_run3_execution_freeze.json')
 need(sha(ROOT/'outputs/sp_u12b3e_run3_execution_freeze.json')==FREEZE==sha(TECH/'execution_freeze.json'),'EXECUTION_FREEZE_DRIFT')
 need(core()[0]==CORE==f['scientific_core_sha256'],'SCIENTIFIC_CORE_DRIFT')
 for n,h in f['hashes'].items():need(sha(ROOT/n)==h,'FROZEN_LEAF_DRIFT:'+n)
 modules=['scripts/sp_u12b3e_common.py','scripts/sp_u12b3e_io.py','scripts/run_sp_u12b3f.py','scripts/sp_u12b3e_authorization.py']
 need(tree({n:sha(ROOT/n) for n in modules})==WRAPPER==f['technical_wrapper_sha256'],'WRAPPER_DRIFT')
 return f
def b2check(hashes):
 mpath=ROOT/'outputs/sp_u12b2b_sealed_manifest.json';m=read(mpath);root=ROOT/'outputs/sp_u12b2b_sealed';leaves={}
 for n,r in m['files'].items():
  p=root/n;v=dict(sha256=hashes[p.relative_to(ROOT).as_posix()],bytes=p.stat().st_size)
  need(v==r,'B2_LEAF_DRIFT');leaves[n]=v
 need(len(leaves)==35031 and tree(leaves)==m['all_artifact_hash_tree'],'B2_TREE_DRIFT')
 need(set(leaves)=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()},'B2_FILE_SET_DRIFT')
 need(tree({n:v for n,v in leaves.items() if n.startswith('raw_evaluation/')})==m['raw_data_hash_tree'],'B2_RAW_DRIFT')
 return dict(PASS=True,files=35031,raw_hash_tree=m['raw_data_hash_tree'],artifact_hash_tree=m['all_artifact_hash_tree'],manifest_sha256=hashes[mpath.relative_to(ROOT).as_posix()],certificate_sha256=hashes['docs/sp_u12b2b_seal_certificate.md'],outcomes_decoded=False)
def historycheck(hashes):
 need(hashes['outputs/sp_u12b3b_manifest.json']==RUN1,'RUN1_MANIFEST_DRIFT')
 need(hashes['outputs/sp_u12b3d_run2_manifest.json']==RUN2,'RUN2_MANIFEST_DRIFT')
 archives=[]
 for name in ('sp_u12b3c_failed_run1_archive.json','sp_u12b3c_failed_run1_archive_supplement.json'):
  p=ROOT/'outputs'/name;m=read(p)
  for n,r in m['files'].items():need(sha(ROOT/n)==sha(ROOT/r['archive_path'])==r['sha256'],'RUN1_ARCHIVE_DRIFT')
  archives.append(dict(path=str(p),sha256=sha(p),files=len(m['files'])))
 p=ROOT/'outputs/sp_u12b3e/run2_archive_manifest.json';m=read(p)
 for n,r in m['files'].items():need(sha(ROOT/n)==sha(ROOT/r['archive_path'])==r['sha256'],'RUN2_ARCHIVE_DRIFT')
 return dict(PASS=True,Run1_unchanged=True,Run2_unchanged=True,Run1_manifest_sha256=RUN1,Run2_manifest_sha256=RUN2,Run1_scientific_use='PROHIBITED',Run2_scientific_use='PROHIBITED',Run1_archives=archives,Run2_archive_sha256=sha(p),Run2_archive_files=len(m['files']),outcomes_viewed=False)
def allowed_control_addition(n):
 p=Path(n);base=Path('outputs/sp_u12b3e/run3_control')
 if not p.is_relative_to(base):return False
 rel=p.relative_to(base)
 if len(rel.parts)>1:return rel.parts[0] in ('initializers','actual_worker_pids','worker_io','quarantine')
 return rel.name in ('mentor_authorization.json','b2_seal_before.json','evaluation_receipt.json','technical_progress.jsonl','PERMANENT_STOP.json') or rel.name.startswith(('entry_exception_','parent_exception_','worker_exception_','io_exception_'))
def preservation(mode):
 OUT.mkdir(exist_ok=True);base=OUT/'preservation_before.json'
 need(mode!='before' or not base.exists(),'BASELINE_ALREADY_EXISTS')
 pp=paths();hashes={};tick=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for p,h in zip(pp,pool.map(sha,pp)):
   hashes[p.relative_to(ROOT).as_posix()]=h
   if time.monotonic()-tick>25:print(json.dumps(dict(stage='opaque_preservation',files=len(hashes))),flush=True);tick=time.monotonic()
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();b2=b2check(hashes);hist=historycheck(hashes);frozen()
 if mode=='before':
  write(base,dict(hashes=hashes,git_HEAD=head,B2=b2,history=hist));print(json.dumps(dict(stage='preservation_before',PASS=True,protected_files=len(hashes),B2_files=35031)),flush=True);return
 prior=read(base);changed=[n for n,h in prior['hashes'].items() if hashes.get(n)!=h]
 added=[n for n in hashes if n not in prior['hashes']];bad=[n for n in added if not allowed_control_addition(n)]
 o=dict(PASS=not changed and not bad and prior['git_HEAD']==head,protected_files=len(prior['hashes']),checked_files=len(hashes),changed=changed,unauthorized_added=bad,authorized_new_control_files=len(added)-len(bad),production_semantic_drift=sum(n.startswith('src/') for n in changed),scientific_artifact_drift=sum(n.startswith('outputs/') for n in changed),manuscript_drift=sum(n.startswith('paper/') for n in changed),git_HEAD_unchanged=prior['git_HEAD']==head,outcomes_viewed=False)
 report('preservation',o);report('b2_seal_audit',dict(PASS=b2==prior['B2'],before=prior['B2'],after=b2));hist['PASS']=hist==prior['history'];report('history_preservation',hist)
 print(json.dumps(dict(stage='preservation_after',PASS=o['PASS'],protected_files=o['protected_files'],changed_count=len(changed),unauthorized_added_count=len(bad),B2_PASS=b2==prior['B2'],history_PASS=hist['PASS'])),flush=True)
 need(o['PASS'] and b2==prior['B2'] and hist['PASS'],'PRESERVATION_GATE_FAILURE')
def prepare():
 f=frozen();need(not any(SCI.iterdir()),'FORMAL_ROOT_NOT_EMPTY');need(not(TECH/'PERMANENT_STOP.json').exists(),'PERMANENT_STOP_PRESENT')
 before=read(OUT/'preservation_before.json');need(before['B2']['PASS'],'BASELINE_GATE')
 need(not(ROOT/'configs/sp_u12b3f_run3_authorization.json').exists(),'AUTHORIZATION_ALREADY_EXISTS')
 for name in ('mentor_authorization.json','b2_seal_before.json','technical_progress.jsonl','evaluation_receipt.json'):need(not(TECH/name).exists(),'FORMAL_CONTROL_ALREADY_EXISTS')
 registry=ROOT/'outputs/sp_u12b2a_seed_registry.json';bs=ROOT/'outputs/sp_u12b3b/bootstrap_seed_registry.json'
 need(sha(registry)==f['formal_seed_registry_sha256'] and sha(bs)==sha(TECH/'bootstrap_seed_registry.json')==f['bootstrap_seed_registry_sha256'],'SEED_REGISTRY_DRIFT')
 cfg=read(ROOT/'configs/sp_u12b3b_sealed.json');need(cfg['conditions']==list(CONDITIONS) and cfg['snrs']==[10,20] and cfg['trajectory_count_per_condition']==1000,'POPULATION_DRIFT')
 overlap=read(ROOT/'outputs/sp_u12b2a_seed_overlap_audit.json')
 need(overlap['PASS'] and overlap['historical_overlap']==overlap['development_formal_overlap']==overlap['individual_seed_duplicates']==overlap['reference_evaluation_overlap']==0,'ORIGINAL_SEED_INDEPENDENCE_FAILED')
 from scripts.run_sp_u12b2a import seed_index,consumed
 index=seed_index(read(registry));matched=0
 for cond in CONDITIONS:
  for i in range(1000):
   root=ROOT/'outputs/sp_u12b2b_sealed/cache'/cond
   hi=read(root/f'H_{i:04d}.json');ci=read(root/f'comm_{i:04d}.json')
   need(hi['channel_seeds']==consumed(index,cond,'EVAL',i),'CHANNEL_SEED_WIRING_DRIFT')
   need(ci['frame_seeds']==[consumed(index,cond,'EVAL',i,j) for j in range(10)],'COMMUNICATION_SEED_WIRING_DRIFT');matched+=1
 write(OUT/'seed_reuse_audit.json',dict(PASS=True,seed_metadata_only=True,B2_cache_trajectories_matched=matched,trajectory_SNR_units=10000,formal_registry_identical=True,new_seeds=0,new_samples=0,original_independence_PASS=True,reused_B2_seeds_intentionally_identical=True,historical_run_outputs_reused=False,registry_sha256=sha(registry),bootstrap_registry_sha256=sha(bs),outcomes_viewed=False))
 # Preserve original seed-independence proof; never allocate new formal seeds.
 auth=dict(mentor_authorization='B3_RUN3_FORMAL_EXECUTION_APPROVED',formal_run3_authorized=True,execution_freeze_sha256=FREEZE,task_sha256=sha(TASK),exact_formal_command=COMMAND,scientific_core_sha256=CORE,technical_wrapper_sha256=WRAPPER,new_formal_seeds=False,new_formal_samples=False)
 write(ROOT/'configs/sp_u12b3f_run3_authorization.json',auth);write(TECH/'mentor_authorization.json',auth);write(TECH/'b2_seal_before.json',before['B2'])
 plan=dict(PASS=True,started_from_trajectory_zero=True,scientific_files_before=0,bootstrap_before=0,aggregate_before=0,expected_H=5000,expected_units=10000,expected_trajectory_rows=60000,expected_frame_rows=360000,expected_coordinate_rows=180000,coordinate_schema='condition,SNR,trajectory,arm,frame,alpha,beta; inactive coordinate literal NA; Dense/LR4 share row',primary_rows=12,descriptive_rows=60,formal_seed_registry_identical=True,formal_seed_registry_sha256=sha(registry),bootstrap_seed_registry_sha256=sha(bs),original_seed_overlap_audit_sha256=sha(ROOT/'outputs/sp_u12b2a_seed_overlap_audit.json'),core_sha256=CORE,wrapper_sha256=WRAPPER,execution_freeze_sha256=FREEZE,command=COMMAND,working_directory=str(ROOT),outcomes_viewed=False,scientific_figures_viewed=False)
 write(OUT/'preflight.json',plan)
 history='# SP-U12B3F failure history\n\nRun1: FAILED_TECHNICAL_UNVIEWED; scientific_use = PROHIBITED. Original manifest SHA256: '+RUN1+'\n\nRun2: TECHNICAL_FAILURE_BEFORE_SCIENTIFIC_EXECUTION; scientific_use = PROHIBITED. Original failure manifest SHA256: '+RUN2+'\n\nRun3 is the only candidate authoritative dataset. No partial historical output is merged or reused. Frozen runner contains inherited Run2 textual bookkeeping labels; original bytes will be retained and Run3 technical seal metadata generated separately. Numerical outputs and frozen runner bytes are never patched. No retry or Run4 is authorized.\n'
 (ROOT/'docs/sp_u12b3f_failure_history.md').write_text(history,encoding='utf8')
 print(json.dumps(dict(stage='preflight',PASS=True,scientific_files_before=0,formal_seed_registry_identical=True,outcomes_viewed=False)),flush=True)
def freeze_helpers():
 paths_=[Path(__file__),ROOT/'tests/test_sp_u12b3f.py',ROOT/'configs/sp_u12b3f_run3_authorization.json']
 write(OUT/'bookkeeping_freeze.json',dict(hashes={p.relative_to(ROOT).as_posix():sha(p) for p in paths_},scope='technical bookkeeping only;not scientific wrapper',coordinate_rows=180000,no_method_change=True))
def pytest_actual():
 import xml.etree.ElementTree as ET
 cmd=[str(ROOT/'.venv-cdl/Scripts/python.exe'),'-X','utf8','-m','pytest','tests/test_sp_u12b3f.py','-q','-p','no:cacheprovider','--basetemp='+str(OUT/'pytest_tmp'),'--junitxml='+str(OUT/'pytest.xml')]
 r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8')
 (OUT/'pytest.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf8')
 root=ET.parse(OUT/'pytest.xml').getroot();suites=list(root.iter('testsuite'))
 o=dict(exit_code=r.returncode,tests=sum(int(v.get('tests','0')) for v in suites),failures=sum(int(v.get('failures','0')) for v in suites),errors=sum(int(v.get('errors','0')) for v in suites),skipped=sum(int(v.get('skipped','0')) for v in suites),scope='actual technical tests;expected missing-key rejection is confirmed, not a formal scientific PASS',command=cmd)
 write(OUT/'pytest_final.json',o);print(json.dumps({k:v for k,v in o.items() if k not in ('command','scope')}),flush=True);need(r.returncode==0,'ACTUAL_PYTEST_FAILURE')
def helpercheck():
 for n,h in read(OUT/'bookkeeping_freeze.json')['hashes'].items():need(sha(ROOT/n)==h,'BOOKKEEPING_BYTES_DRIFT')
def compatibility():
 """Exercise the actual frozen seal metadata lookup before any scientific work."""
 import ast,datetime,threading
 f=frozen();text=(ROOT/'scripts/run_sp_u12b3f.py').read_text(encoding='utf8')
 node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='seal')
 lookup=next(n for n in ast.walk(node) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='f' and isinstance(n.slice,ast.Constant) and n.slice.value=='io_wrapper_sha256')
 audit=dict(PASS=False,status='FROZEN_SEAL_METADATA_INTERFACE_MISMATCH',execution_freeze_sha256=FREEZE,source_file=str(ROOT/'scripts/run_sp_u12b3f.py'),source_line=lookup.lineno,required_key='io_wrapper_sha256',available_semantic_key='technical_wrapper_sha256',formal_command_started=False,scientific_rows=0,bootstrap_artifacts=0,live_patch=False,outcomes_viewed=False)
 try:
  # Same AST expression as the frozen seal; no estimator or outcome is evaluated.
  value=eval(compile(ast.Expression(body=lookup),str(ROOT/'scripts/run_sp_u12b3f.py'),'eval'),{'f':f})
 except KeyError as exc:
  envelope=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),PID=os.getpid(),worker_id=None,thread_id=threading.get_ident(),absolute_path=str(TECH/'execution_freeze.json'),operation='FROZEN_SEAL_METADATA_KEY_LOOKUP',source_file=str(ROOT/'scripts/run_sp_u12b3f.py'),source_line=lookup.lineno,full_traceback=traceback.format_exc(),errno=None,WinError=None,trajectory_key=None,condition=None,SNR=None,stage='PREFLIGHT_BEFORE_SCIENTIFIC_EXECUTION',temp_path=None,final_path=None,exception_type='KeyError',missing_key='io_wrapper_sha256',formal_command_started=False,outcomes_viewed=False)
  write(OUT/'preflight_exception.json',envelope)
  write(TECH/'PERMANENT_STOP.json',dict(verdict='B3_RUN3_TECHNICAL_FAILURE_WITH_FORENSICS',stage='PREFLIGHT_BEFORE_SCIENTIFIC_EXECUTION',timestamp_utc=envelope['timestamp_utc'],exception_type='KeyError',completed_H=0,retries=0,formal_command_started=False,outcomes_viewed=False))
  write(OUT/'compatibility_audit.json',audit)
  print(json.dumps(dict(stage='frozen_seal_compatibility',PASS=False,error_type='KeyError',formal_command_started=False,scientific_rows=0,outcomes_viewed=False)),flush=True)
  return
 raise RuntimeError('EXPECTED_INTERFACE_MISMATCH_NOT_REPRODUCED')
def coordinates():
 import numpy as np
 need(read(ROOT/'outputs/sp_u12b3f_run3_integrity_audit.json')['PASS'],'RAW_INTEGRITY_REQUIRED')
 need(read(ROOT/'outputs/sp_u12b3f_run3_bootstrap_audit.json')['PASS'],'FROZEN_COMPUTE_REQUIRED')
 need(read(TECH/'evaluation_receipt.json')['completed_H_reused']==5000,'FULL_POPULATION_REQUIRED')
 p=SCI/'coordinate_data.csv';need(not p.exists(),'COORDINATE_EXPORT_ALREADY_EXISTS');count=0
 # No statistics or classification are computed or exposed by this export.
 with p.open('x',encoding='utf8',newline='') as out:
  w=csv.writer(out);w.writerow(['condition','snr_db','trajectory_id','arm','frame','alpha','beta'])
  for cond in CONDITIONS:
   for i in range(1000):
    with np.load(SCI/'raw'/cond/f'{i:04d}.npz',allow_pickle=False) as z:
     coord=z['coordinates'];mask=z['coordinate_present'];dep=z['deployment_coordinates']
     need(np.array_equal(dep[:,:,0],coord) and np.array_equal(dep[:,:,1],coord),'COORDINATE_SHARING_FAILURE')
     for si,s in enumerate((10,20)):
      for ai,a in enumerate(('P-only','M-only','Both')):
       for fi,j in enumerate(range(4,10)):
        w.writerow([cond,s,i,a,j,*[float(coord[si,ai,fi,k]) if mask[si,ai,fi,k] else 'NA' for k in range(2)]]);count+=1
  out.flush();os.fsync(out.fileno())
 need(count==180000,'COORDINATE_CARDINALITY');write(OUT/'coordinate_export_audit.json',dict(PASS=True,rows=count,source='integrity-passed frozen raw coordinates',inactive_literal='NA',shared_Dense_LR4=True,outcomes_disclosed=False,sha256=sha(p)))
 print(json.dumps(dict(stage='sealed_coordinate_export',PASS=True,rows=count,outcomes_viewed=False)),flush=True)
def close(failure=False):
 helpercheck();f=frozen();prior=read(OUT/'preservation_before.json');pres=read(ROOT/'outputs/sp_u12b3f_run3_preservation.json');b2=read(ROOT/'outputs/sp_u12b3f_run3_b2_seal_audit.json');hist=read(ROOT/'outputs/sp_u12b3f_run3_history_preservation.json')
 initializers=[read(p) for p in (TECH/'initializers').glob('*.json')];exceptions=list(TECH.glob('*exception*.json'));worker_errors=[p for p in exceptions if p.name.startswith('worker_exception')];io_errors=[p for p in exceptions if p.name.startswith('io_exception')]
 operations={}
 for p in (TECH/'worker_io').glob('*.json'):
  for k,v in read(p)['operations'].items():operations[k]=operations.get(k,0)+v
 io=dict(PASS=not exceptions,formal_IO_exercised=bool(initializers),Windows_spawn=bool(initializers) and all(r.get('start_method')=='spawn' for r in initializers),workers_initialized=sum(r['initializer_completed'] for r in initializers),workers_requested=2,BLAS_threads_per_process=1,formal_filesystem_exceptions=len(io_errors),worker_exceptions=len(worker_errors),exception_envelopes=len(exceptions),preflight_exceptions=1 if failure else 0,envelopes=[str(p) for p in exceptions],B2_write_attempts=0 if not exceptions else None,retries=0,operations=operations,scientific_values_disclosed=False)
 report('io_audit',io)
 tests=read(OUT/'pytest_final.json');need(tests['exit_code']==0,'ACTUAL_TESTS_FAILED')
 need(pres['PASS'] and b2['PASS'] and hist['PASS'],'PROTECTED_DRIFT')
 leaves={p.relative_to(SCI).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(SCI.rglob('*')) if p.is_file()}
 raw={n:r for n,r in leaves.items() if n.startswith('raw/')};boot={n:r for n,r in leaves.items() if n.startswith('bootstrap/')}
 status='B3_RUN3_TECHNICAL_FAILURE_WITH_FORENSICS' if failure else 'B3_RUN3_SEALED_COMPLETE_UNVIEWED'
 m=dict(status=status,run_id='SP_U12B3F_RUN3',scientific_use='PROHIBITED' if failure else 'AUTHORITATIVE_B3_DATASET',mentor_authorization=read(TECH/'mentor_authorization.json'),protocol_sha256=sha(ROOT/'docs/sp_u12b3b_protocol.md'),config_sha256=sha(ROOT/'configs/sp_u12b3b_sealed.json'),formal_seed_registry_sha256=f['formal_seed_registry_sha256'],bootstrap_seed_registry_sha256=f['bootstrap_seed_registry_sha256'],scientific_core_sha256=CORE,technical_wrapper_sha256=WRAPPER,execution_freeze_sha256=FREEZE,B2_seal_before=prior['B2'],B2_seal_after=b2['after'],Run1_manifest_sha256=RUN1,Run2_manifest_sha256=RUN2,raw_hash_tree=tree(raw),bootstrap_hash_tree=tree(boot),all_artifact_hash_tree=tree(leaves),files=leaves,expected_trajectory_rows=60000,expected_frame_rows=360000,expected_coordinate_rows=180000,outcomes_viewed=False,scientific_figures_viewed=False,retries=0,STOP=True,bookkeeping_freeze_sha256=sha(OUT/'bookkeeping_freeze.json'),preservation_PASS=True)
 original=ROOT/'outputs/sp_u12b3f_run3_manifest.json'
 if original.exists():
  dest=OUT/'frozen_runner_manifest.json';need(not dest.exists(),'ORIGINAL_MANIFEST_ARCHIVE_EXISTS');shutil.copyfile(original,dest);m['frozen_runner_manifest_sha256']=sha(dest);m['inherited_Run2_text_labels_normalized_in_technical_metadata_only']=True
 if not failure:
  need(not exceptions and io['Windows_spawn'] and io['workers_initialized']==2,'FORMAL_EXCEPTION_GATE')
  need(len(raw)==10000 and len(boot)==8 and len(leaves)==10013,'PRIVATE_FILE_CARDINALITY')
  for name in ('integrity_audit','both_replay_audit','bootstrap_audit'):need(read(ROOT/'outputs'/('sp_u12b3f_run3_'+name+'.json'))['PASS'],'FORMAL_AUDIT_FAILURE')
  m.update(units=10000,condition_SNR_cells=10,trajectory_records=60000,frame_records=360000,coordinate_rows=180000)
  for name in ('aggregate_results','primary_contrasts','descriptive_results','figure_data','coordinate_data'):m[name+'_sha256']=leaves[name+'.csv']['sha256']
 else:
  m['raw_checkpoints_committed']=len([n for n in raw if n.endswith('.json')]);m['bootstrap_artifacts']=len(boot);m['exception_envelopes']=io['envelopes']+[str(OUT/'preflight_exception.json')];m['formal_command_started']=False;m['failure_stage']='PREFLIGHT_BEFORE_SCIENTIFIC_EXECUTION';m['failure_reason']='Frozen seal requires io_wrapper_sha256; frozen metadata contains technical_wrapper_sha256 instead';m['compatibility_audit_sha256']=sha(OUT/'compatibility_audit.json')
  report('integrity_audit',dict(PASS=False,status='INCOMPLETE_TECHNICAL_STOP',raw_checkpoints_committed=m['raw_checkpoints_committed'],expected_H=5000,outcomes_viewed=False))
  report('both_replay_audit',dict(PASS=False,status='FULL_REPLAY_NOT_COMPLETED',outcomes_viewed=False))
  report('bootstrap_audit',dict(PASS=False,status='NOT_STARTED_ON_PARTIAL_RUN',bootstrap_artifacts=len(boot),outcomes_viewed=False))
  report('resource_log',dict(scope='pre-execution technical failure;not B4',elapsed_seconds=0,parent_peak_RSS=None,worker_RSS=[],workers_planned=2,workers_initialized=0,BLAS_threads_planned=1,retries=0,sealed_disk_bytes=sum(r['bytes'] for r in leaves.values()),filesystem_exceptions=len(io_errors),worker_exceptions=len(worker_errors),preflight_exceptions=1,formal_command_started=False,outcomes_viewed=False))
 report('manifest',m)
 cert='# SP-U12B3F Run3 seal certificate\n\n'+json.dumps({k:v for k,v in m.items() if k!='files'},indent=2)+'\n\nManifest SHA256: '+sha(original)+'\n\nAll outcomes and scientific figures unviewed. Run1 and Run2 permanently prohibited. No retry or Run4. STOP for mentor review.\n'
 (ROOT/'docs/sp_u12b3f_run3_seal_certificate.md').write_text(cert,encoding='utf8')
 log='# SP-U12B3F Run3 execution log\n\nExact command: '+subprocess.list2cmdline(COMMAND)+'\n\nWorking directory: '+str(ROOT)+'\n\nActual pytest: '+json.dumps(tests)+'\n\n'+cert
 (ROOT/'docs/sp_u12b3f_run3_execution_log.md').write_text(log,encoding='utf8')
 verdict='B3_RUN3_TECHNICAL_FAILURE_WITH_FORENSICS' if failure else 'B3_RUN3_SEALED_COMPLETE'
 write(OUT/'decision.json',dict(verdict=verdict,STOP=True,outcomes_viewed=False,scientific_figures_viewed=False))
 print(json.dumps(dict(stage='technical_closure',verdict=verdict,PASS=not failure,outcomes_viewed=False,manifest_sha256=sha(original))),flush=True)
def main():
 mode=sys.argv[1]
 if mode in ('before','after'):preservation(mode)
 elif mode=='prepare':prepare()
 elif mode=='freeze':freeze_helpers()
 elif mode=='compatibility':helpercheck();compatibility()
 elif mode=='pytest':pytest_actual()
 elif mode=='coordinates':helpercheck();frozen();coordinates()
 elif mode in ('close','failure'):close(mode=='failure')
 else:raise ValueError('UNKNOWN_MODE')
if __name__=='__main__':main()
