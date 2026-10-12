"""H authorization/technical integrity/export/seal only; no scientific interpretation."""
from pathlib import Path
import os,sys,json,time,hashlib,csv,subprocess,shutil,ctypes,threading,datetime,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from scripts import sp_u12b3g_schema as s
from scripts.sp_u12b3h_preserve import OUT,SCI,CONTROL,sha,read,write
CORE='994339ccc806e3ca94ae644a954e9382a2fa0df8ed2a808d0feb28528e9bdfa1'
WRAP='681609e7e367c3d80f4d879d8ce15495d519be43221a4eadb6fe7c2475caf311'
FREEZE='ba528450bc8c822759b025e71cd81fc334368feb2526307bf51b53e8f9bb4845'
TASK=Path(r'C:\Users\yuhua\.codex\attachments\279972ca-a2c4-429f-b96e-464652052378\已粘贴的文本.txt')
COND=('ID','Profile','Mobility','Composed','Boundary');ARMS=('P-only','M-only','Both');DEPLOY=('Dense','LR4')
def report(n,o):write(ROOT/'outputs'/('sp_u12b3h_'+n+'.json'),o)
def need(v,k):
 if not v:raise RuntimeError(k)
def tree(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def frozen():
 from scripts.sp_u12b3e_build import core
 f=read(s.FREEZE_PATH);s.validate_structure(f)
 need(sha(s.FREEZE_PATH)==FREEZE,'EXECUTION_FREEZE_DRIFT');need(core()[0]==CORE==f['scientific_core_sha256'],'B3_FORMAL_ABORT_SCIENTIFIC_CORE_DRIFT');need(s.wrapper_hash()==WRAP==f['technical_wrapper_sha256'],'WRAPPER_DRIFT')
 for n,h in f['hashes'].items():need(sha(ROOT/n)==h,'EXECUTION_LEAF_DRIFT:'+n)
 return f
def owncheck():
 for n,h in read(OUT/'auxiliary_freeze.json')['hashes'].items():need(sha(ROOT/n)==h,'AUXILIARY_SOURCE_DRIFT')
def prepare():
 f=frozen();s.validate_execution_freeze(f);before=read(OUT/'preservation_before.json');need(before['B2']['PASS'],'BEFORE_B2_GATE')
 need(not any(p.is_file() for p in SCI.rglob('*')),'FRESH_FORMAL_ROOT_REQUIRED')
 need(not CONTROL.exists() or not any(CONTROL.rglob('*')),'FRESH_CONTROL_REQUIRED')
 cfg=read(ROOT/'configs/sp_u12b3b_sealed.json')
 need(cfg['conditions']==list(COND) and cfg['snrs']==[10,20] and cfg['trajectory_count_per_condition']==1000 and cfg['rank']==4 and cfg['folds']==4 and cfg['window']==5,'SCIENTIFIC_CONTRACT_MISMATCH')
 from scripts.run_sp_u12b2a import seed_index,consumed
 idx=seed_index(read(ROOT/'outputs/sp_u12b2a_seed_registry.json'));n=0
 for cond in COND:
  for i in range(1000):
   base=ROOT/'outputs/sp_u12b2b_sealed/cache'/cond
   need(read(base/f'H_{i:04d}.json')['channel_seeds']==consumed(idx,cond,'EVAL',i),'CHANNEL_SEED_MISMATCH')
   need(read(base/f'comm_{i:04d}.json')['frame_seeds']==[consumed(idx,cond,'EVAL',i,j) for j in range(10)],'COMMUNICATION_SEED_MISMATCH');n+=1
 overlap=read(ROOT/'outputs/sp_u12b2a_seed_overlap_audit.json');need(overlap['PASS'] and overlap['historical_overlap']==overlap['development_formal_overlap']==overlap['individual_seed_duplicates']==0,'ORIGINAL_SEED_INDEPENDENCE_FAILURE')
 auth=dict(formal_run_authorized=True,mentor_authorization='SP_U12B3H_AUTHORIZED_FINAL_SEALED_B3_FORMAL_RUN',execution_freeze_sha256=FREEZE,scientific_core_sha256=CORE,technical_wrapper_sha256=WRAP,task_path=str(TASK),task_sha256=sha(TASK),exact_command=s.COMMAND,pre_scientific_gate=False,new_formal_seeds=False,new_formal_samples=False,formal_retries=0)
 write(CONTROL/'formal_authorization.json',auth);write(CONTROL/'mentor_authorization.json',auth);write(CONTROL/'b2_seal_before.json',before['B2'])
 plan=dict(PASS=True,scientific_files_before=0,bootstrap_before=0,aggregate_before=0,started_from_trajectory_zero=True,expected_distinct_H=5000,expected_cells=10,expected_units=10000,expected_trajectory_records=60000,expected_frame_records=360000,expected_coordinate_rows=180000,expected_primary_rows=12,expected_descriptive_rows=60,expected_bootstrap_files=8,expected_final_scientific_files=10014,raw_schema='5000 NPZ;each has trajectory_error(2,3,2),trajectory_truth(2),frame_error(2,3,2,6),coordinates(2,3,6,2) and validity/status arrays',coordinate_schema='180000 unique condition/SNR/trajectory/arm/frame rows;Dense/LR4 share coordinates;inactive field literal NA',trajectory_schema='60000 condition/SNR/trajectory/arm/deployment rows with stored error/truth energy',formal_seed_registry_identical=True,seed_metadata_trajectories_checked=n,formal_seed_registry_sha256=f['formal_seed_registry_sha256'],bootstrap_seed_registry_sha256=f['bootstrap_seed_registry_sha256'],formal_root=str(SCI),technical_control=str(CONTROL),formal_command=s.COMMAND,workers=2,BLAS_threads=1,start_method='spawn',outcomes_viewed=False)
 write(OUT/'preflight.json',plan)
 own=[Path(__file__),ROOT/'scripts/sp_u12b3h_preserve.py',ROOT/'tests/test_sp_u12b3h.py']
 write(OUT/'auxiliary_freeze.json',dict(scope='technical bookkeeping/export/resource monitor only;frozen formal wrapper unchanged',hashes={p.relative_to(ROOT).as_posix():sha(p) for p in own},authorization_sha256=sha(CONTROL/'formal_authorization.json')))
 print(json.dumps(dict(stage='H_preflight',PASS=True,seed_metadata_trajectories_checked=n,scientific_files_before=0,formal_root=str(SCI),core_sha256=CORE,wrapper_sha256=WRAP,execution_freeze_sha256=FREEZE)),flush=True)
def arm():
 owncheck();frozen();need(read(OUT/'pytest.json')['exit_code']==0,'ACTUAL_PREFLIGHT_PYTEST_REQUIRED');need(not(OUT/'formal_started.json').exists(),'FORMAL_ATTEMPT_ALREADY_STARTED')
 write(OUT/'formal_started.json',dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),unix_time=time.time(),command=s.COMMAND,attempt=1,formal_retries=0))
def mark_exit(code):write(OUT/'formal_exit.json',dict(exit_code=int(code),unix_time=time.time(),formal_retries=0))
def monitor():
 """Observe actual interpreter PIDs from existing worker handshakes."""
 class PMC(ctypes.Structure):
  _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
 k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.restype=ctypes.c_void_p;k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];k.CloseHandle.argtypes=[ctypes.c_void_p]
 ps=ctypes.WinDLL('psapi',use_last_error=True);ps.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
 peaks={};parent=None;workers=set();samples=0;tick=time.monotonic()
 def sample(pid):
  h=k.OpenProcess(0x410,False,pid)
  if not h:return None
  try:
   p=PMC();p.cb=ctypes.sizeof(p)
   if not ps.GetProcessMemoryInfo(h,ctypes.byref(p),p.cb):return None
   return dict(peak_RSS_bytes=int(p.PeakWorkingSetSize),RSS_bytes=int(p.WorkingSetSize))
  finally:k.CloseHandle(h)
 while True:
  for p in (CONTROL/'actual_worker_pids').glob('*.json'):
   r=read(p);parent=r['controller_PID'];workers.add(r['PID'])
  if parent is None:
   for p in CONTROL.glob('entry_exception_*.json'):parent=read(p)['PID']
  for pid in workers|({parent} if parent is not None else set()):
   v=sample(pid)
   if v:peaks[str(pid)]=max(peaks.get(str(pid),0),v['peak_RSS_bytes'])
  samples+=1
  write(OUT/'resource_monitor.json',dict(elapsed_seconds=time.monotonic()-tick,parent_PID=parent,worker_PIDs=sorted(workers),peak_RSS_bytes_by_PID=peaks,samples=samples,source='Win32 actual interpreter PIDs;not venv launcher',formal_retries=0,scope='engineering observation only;not B4'))
  if (OUT/'formal_exit.json').exists():return
  time.sleep(2)
def exception_files():return sorted(CONTROL.glob('*exception*.json'))
def export():
 owncheck();frozen();need(read(OUT/'formal_exit.json')['exit_code']==0,'FORMAL_EXIT_FAILURE');need(not exception_files(),'FORMAL_EXCEPTION_ENVELOPE')
 rawaudit=read(ROOT/'outputs/sp_u12b3g_next_integrity_audit.json');replay=read(ROOT/'outputs/sp_u12b3g_next_both_replay_audit.json');ba=read(ROOT/'outputs/sp_u12b3g_next_bootstrap_audit.json')
 need(rawaudit['PASS'] and rawaudit['units']==10000 and rawaudit['trajectory_rows']==60000 and ba['PASS'] and replay['PASS'],'FULL_RUN_REQUIRED_FOR_EXPORT')
 import numpy as np
 from scripts import sp_u12b3e_common as b
 from scripts.run_sp_u12b2a import seed_index,consumed
 idx=seed_index(read(ROOT/'outputs/sp_u12b2a_seed_registry.json'))
 p=SCI/'coordinate_data.csv';t=SCI/'trajectory_records.csv';need(not p.exists() and not t.exists(),'EXPORT_ALREADY_EXISTS');nc=0;nt=0;keys=set();last=time.monotonic()
 with p.open('x',encoding='utf8',newline='') as cf,t.open('x',encoding='utf8',newline='') as tf:
  cw=csv.writer(cf);tw=csv.writer(tf);cw.writerow(['condition','snr_db','trajectory_id','arm','frame','alpha','beta']);tw.writerow(['condition','snr_db','trajectory_id','arm','deployment','error_energy','truth_energy','independent_unit'])
  for cond in COND:
   for i in range(1000):
    base=SCI/'raw'/cond/f'{i:04d}.npz';proof=read(base.with_suffix('.json'));need(sha(base)==proof['artifact']['sha256'],'RAW_HASH_CHANGED')
    need(proof['channel_seeds']==consumed(idx,cond,'EVAL',i) and proof['communication_seeds']==[consumed(idx,cond,'EVAL',i,j) for j in range(10)],'FORMAL_SEED_REPLACEMENT')
    with np.load(base,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    b.validate(raw)
    for si,snr in enumerate((10,20)):
     for ai,a in enumerate(ARMS):
      for fi,j in enumerate(range(4,10)):
       cw.writerow([cond,snr,i,a,j,*[float(raw['coordinates'][si,ai,fi,k]) if raw['coordinate_present'][si,ai,fi,k] else 'NA' for k in range(2)]]);nc+=1
      for di,d in enumerate(DEPLOY):
       key=(cond,snr,i,a,d);need(key not in keys,'DUPLICATE_TRAJECTORY_KEY');keys.add(key)
       tw.writerow([*key,float(raw['trajectory_error'][si,ai,di]),float(raw['trajectory_truth'][si]),'trajectory']);nt+=1
    if time.monotonic()-last>25:print(json.dumps(dict(stage='sealed_bookkeeping',trajectory_records=nt,coordinate_rows=nc)),flush=True);last=time.monotonic()
  for f in (cf,tf):f.flush();os.fsync(f.fileno())
 need(nc==180000 and nt==60000,'EXPORT_CARDINALITY')
 checks=[]
 for rec in ba['bootstrap_receipts']:
  artifact=rec['artifact'];path=ROOT/artifact['path'];need(sha(path)==artifact['sha256'],'BOOTSTRAP_HASH_CHANGED')
  with np.load(path,allow_pickle=False) as z:
   v=z['draws'];shape=(10000,4) if rec['kind']=='PRIMARY' else (10000,2,3,2);need(v.shape==shape and np.isfinite(v).all(),'BOOTSTRAP_CARDINALITY_OR_NONFINITE')
  checks.append(dict(kind=rec['kind'],condition=rec['condition'],replicates=10000,shape=list(shape),sha256=sha(path)))
 need(len(checks)==8 and sum(v['kind']=='PRIMARY' for v in checks)==3,'BOOTSTRAP_FILE_COUNT')
 csvcounts={}
 for name,expected in [('aggregate_results',60),('primary_contrasts',12),('descriptive_results',60),('figure_data',60)]:
  with (SCI/(name+'.csv')).open(encoding='utf8',newline='') as f:
   rows=list(csv.DictReader(f));need(len(rows)==expected,'PRIVATE_TABLE_CARDINALITY');csvcounts[name]=len(rows)
   if name=='primary_contrasts':need({(r['condition'],int(r['snr_db']),r['comparison']) for r in rows}=={(c,snr,n) for c in ('Profile','Mobility','Composed') for snr in (10,20) for n in ('Both_LR4_vs_Ponly_LR4','Both_LR4_vs_Monly_LR4')},'PRIMARY_FAMILY_CHANGED')
   if name in ('descriptive_results','figure_data'):need(all(r['interval_role']=='DESCRIPTIVE_ONLY' for r in rows),'DESCRIPTIVE_SCOPE_CHANGED')
 need(sha(SCI/'descriptive_results.csv')==sha(SCI/'figure_data.csv'),'FIGURE_DATA_COPY_CHANGED')
 rawaudit.update(seed_registry_unchanged=True,seed_metadata_records_rechecked=5000,no_sample_replacement=True,coordinate_csv_rows=nc,trajectory_csv_rows=nt,bootstrap_integrity_PASS=True)
 report('integrity_audit',rawaudit);report('both_replay_audit',replay);ba.update(shape_hash_finiteness_checks=checks,private_table_counts=csvcounts);report('bootstrap_audit',ba)
 write(OUT/'export_audit.json',dict(PASS=True,trajectory_rows=nt,coordinate_rows=nc,new_scientific_analysis=False,source='stored raw coordinates and stored error/truth energies only',outcomes_disclosed=False))
 print(json.dumps(dict(stage='sealed_export',PASS=True,trajectory_records=nt,coordinate_rows=nc,bootstrap_files=8,outcomes_viewed=False)),flush=True)
def pytest_actual():
 import xml.etree.ElementTree as ET
 cmd=[str(ROOT/'.venv-cdl/Scripts/python.exe'),'-X','utf8','-m','pytest','tests/test_sp_u12b3h.py','-q','-p','no:cacheprovider','--basetemp='+str(OUT/'pytest_tmp'),'--junitxml='+str(OUT/'pytest.xml')]
 r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8');(OUT/'pytest.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf8')
 suites=list(ET.parse(OUT/'pytest.xml').getroot().iter('testsuite'));o=dict(exit_code=r.returncode,passed=sum(int(x.get('tests','0'))-int(x.get('failures','0'))-int(x.get('errors','0'))-int(x.get('skipped','0')) for x in suites),failed=sum(int(x.get('failures','0'))+int(x.get('errors','0')) for x in suites),skipped=sum(int(x.get('skipped','0')) for x in suites),command=cmd)
 write(OUT/'pytest.json',o);print(json.dumps({k:v for k,v in o.items() if k!='command'}),flush=True);need(r.returncode==0,'ACTUAL_PYTEST_FAILED')
def close(failure=False):
 owncheck();f=frozen();pres=read(ROOT/'outputs/sp_u12b3h_preservation.json');b2=read(ROOT/'outputs/sp_u12b3h_b2_seal_audit.json');hist=read(ROOT/'outputs/sp_u12b3h_history_preservation.json');need(pres['PASS'] and b2['PASS'] and hist['PASS'],'PRESERVATION_GATE')
 e=exception_files();io_errors=[p for p in e if p.name.startswith('io_exception')];worker_errors=[p for p in e if p.name.startswith('worker_exception')]
 initialized=[read(p) for p in (CONTROL/'initializers').glob('*.json')];heartbeat=[read(p) for p in (CONTROL/'heartbeats').glob('*.json')]
 operations={}
 for p in (CONTROL/'worker_io').glob('*.json'):
  for k,v in read(p)['operations'].items():operations[k]=operations.get(k,0)+v
 ia=dict(PASS=not e,workers_initialized=sum(r.get('initializer_completed',False) for r in initialized),heartbeats=len(heartbeat),Windows_spawn=bool(initialized) and all(r.get('start_method')=='spawn' for r in initialized),filesystem_exceptions=len(io_errors),worker_exceptions=len(worker_errors),all_exception_envelopes=len(e),B2_write_attempts=0 if not e else None,formal_retries=0,operations=operations,forensic_envelopes=[str(p) for p in e],outcomes_viewed=False)
 report('io_audit',ia)
 leaves={p.relative_to(SCI).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(SCI.rglob('*')) if p.is_file()}
 raw={n:v for n,v in leaves.items() if n.startswith('raw/')};boot={n:v for n,v in leaves.items() if n.startswith('bootstrap/')}
 m=dict(status='B3_FORMAL_TECHNICAL_FAILURE_WITH_FORENSICS' if failure else 'B3_SEALED_FORMAL_COMPLETE_UNVIEWED',run_id='SP_U12B3H_FINAL_FORMAL',scientific_use='PROHIBITED' if failure else 'AUTHORITATIVE_B3_DATASET',actual_scientific_root=str(SCI),frozen_technical_control=str(CONTROL),mentor_authorization=read(CONTROL/'formal_authorization.json'),protocol_sha256=f['protocol_sha256'],formal_seed_registry_sha256=f['formal_seed_registry_sha256'],bootstrap_seed_registry_sha256=f['bootstrap_seed_registry_sha256'],scientific_core_sha256=CORE,technical_wrapper_sha256=WRAP,execution_freeze_sha256=FREEZE,B2_provenance_before=b2['before'],B2_provenance_after=b2['after'],historical_failed_manifest_hashes=hist['hashes'],historical_failed_scientific_use='PROHIBITED',raw_data_hash_tree=tree(raw),bootstrap_hash_tree=tree(boot),primary_bootstrap_hash_tree=tree({n:v for n,v in boot.items() if Path(n).name.startswith('primary_')}),descriptive_bootstrap_hash_tree=tree({n:v for n,v in boot.items() if Path(n).name.startswith('descriptive_')}),all_artifact_hash_tree=tree(leaves),files=leaves,expected_trajectory_records=60000,expected_frame_records=360000,expected_coordinate_rows=180000,formal_retries=0,outcomes_viewed=False,scientific_figures_viewed=False,STOP=True)
 started=read(OUT/'formal_started.json');exit_=read(OUT/'formal_exit.json');monitor_=read(OUT/'resource_monitor.json')
 rl=dict(elapsed_command_seconds=exit_['unix_time']-started['unix_time'],workers=2,BLAS_threads=1,filesystem_exceptions=len(io_errors),worker_exceptions=len(worker_errors),formal_retries=0,sealed_disk_bytes=sum(v['bytes'] for v in leaves.values()),monitor=monitor_,scope='engineering resource log only;not B4 benchmark',outcomes_viewed=False)
 native=ROOT/'outputs/sp_u12b3g_next_resource_log.json'
 if native.exists():rl['frozen_runner_native_resources']=read(native)
 report('resource_log',rl)
 if failure:
  checkpoints=[n for n in raw if n.endswith('.json')];completed=len(checkpoints);m.update(completed_distinct_H=completed,completed_units=2*completed,completed_trajectory_records=12*completed,bootstrap_files=len(boot),forensic_envelopes=ia['forensic_envelopes'])
  need(not boot or completed==5000,'PARTIAL_RUN_BOOTSTRAP_VIOLATION')
  if not(CONTROL/'PERMANENT_STOP.json').exists():write(CONTROL/'PERMANENT_STOP.json',dict(verdict='B3_FORMAL_TECHNICAL_FAILURE_WITH_FORENSICS',stage='FORMAL_EXCEPTION',formal_retries=0,outcomes_viewed=False))
  for name in ('integrity_audit','both_replay_audit','bootstrap_audit'):
   if not(ROOT/'outputs'/('sp_u12b3h_'+name+'.json')).exists():report(name,dict(PASS=False,status='INCOMPLETE_TECHNICAL_STOP',completed_distinct_H=completed,bootstrap_files=len(boot),outcomes_viewed=False))
 else:
  need(exit_['exit_code']==0 and ia['PASS'] and ia['workers_initialized']==ia['heartbeats']==2 and ia['Windows_spawn'],'FORMAL_EXECUTION_GATE')
  need(read(OUT/'pytest.json')['exit_code']==0 and read(OUT/'pytest.json')['skipped']==0,'FINAL_PYTEST_GATE')
  for n in ('integrity_audit','both_replay_audit','bootstrap_audit'):need(read(ROOT/'outputs'/('sp_u12b3h_'+n+'.json'))['PASS'],'FORMAL_INTEGRITY_GATE')
  need(len(raw)==10000 and len(boot)==8 and len(leaves)==10014,'FINAL_FILE_CARDINALITY')
  original=ROOT/'outputs/sp_u12b3g_next_manifest.json';archive=OUT/'original_frozen_runner_manifest.json';need(not archive.exists(),'ORIGINAL_MANIFEST_COPY_EXISTS');shutil.copyfile(original,archive);need(sha(original)==sha(archive),'ORIGINAL_MANIFEST_COPY_HASH');m['original_frozen_runner_manifest_sha256']=sha(original)
  m.update(condition_SNR_cells=10,trajectory_units=10000,trajectory_estimator_records=60000,frame_estimator_records=360000,coordinate_rows=180000,primary_contrasts=12,primary_replicates_per_contrast=10000,descriptive_cells=60,descriptive_replicates_per_cell=10000,seed_registry_unchanged=True,no_sample_replacement=True)
  for name in ('aggregate_results','primary_contrasts','descriptive_results','figure_data','coordinate_data','trajectory_records'):m[name+'_sha256']=leaves[name+'.csv']['sha256']
 report('manifest',m);mp=ROOT/'outputs/sp_u12b3h_manifest.json'
 verdict='B3_FORMAL_TECHNICAL_FAILURE_WITH_FORENSICS' if failure else 'B3_SEALED_FORMAL_COMPLETE'
 cert='# SP-U12B3H seal certificate\n\n'+json.dumps({k:v for k,v in m.items() if k!='files'},indent=2)+'\n\nManifest SHA256: '+sha(mp)+'\n\nAll scientific outcomes and figures remain unviewed. Historical failed runs are permanently prohibited. No retry, resume, new samples, manuscript change, B4/B5/MFA/CDL-D or Git commit. STOP for mentor review.\n'
 (ROOT/'docs/sp_u12b3h_seal_certificate.md').write_text(cert,encoding='utf8')
 (ROOT/'docs/sp_u12b3h_execution_log.md').write_text('# SP-U12B3H execution log\n\nExact command: '+subprocess.list2cmdline(s.COMMAND)+'\nWorking directory: '+str(ROOT)+'\n\nActual pytest: '+json.dumps(read(OUT/'pytest.json'))+'\n\n'+cert,encoding='utf8')
 report('decision',dict(verdict=verdict,manifest_path=str(mp),manifest_sha256=sha(mp),actual_scientific_root=str(SCI),STOP=True,outcomes_viewed=False,scientific_figures_viewed=False,formal_retries=0))
 print(json.dumps(dict(stage='H_closure',verdict=verdict,manifest_sha256=sha(mp),STOP=True,outcomes_viewed=False)),flush=True)
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='prepare':prepare()
 elif mode=='arm':arm()
 elif mode=='exit':mark_exit(sys.argv[2])
 elif mode=='monitor':monitor()
 elif mode=='export':export()
 elif mode=='pytest':pytest_actual()
 elif mode in ('close','failure'):close(mode=='failure')
 else:raise ValueError('UNKNOWN_MODE')
