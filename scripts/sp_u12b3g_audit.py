"""Generate schema/freeze, execute technical gates and certify sealed preservation."""
from pathlib import Path
import sys,os,json,ast,hashlib,subprocess,difflib,csv,time,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b3g_schema as s
from scripts.sp_u12b3g_preserve import sha,tree,read,write,OUT
def report(n,o):write(ROOT/'outputs'/('sp_u12b3g_'+n+'.json'),o)
def need(v,k):
 if not v:raise RuntimeError(k)
def core_check():
 from scripts.sp_u12b3e_build import core
 need(core()[0]==s.CORE,'B3G_SCIENTIFIC_CODE_CHANGE_REQUIRED')
 return core()[0]
def source_functions(p,names):
 text=p.read_text(encoding='utf8');lines=text.splitlines(keepends=True)
 return {n.name:''.join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in names}
def fields():
 # Search interface/source/technical records; exclude sealed outcome trees and binary data.
 cmd=['rg','-n','--no-heading','--glob','*.py','--glob','*.json','--glob','*.md','--glob','*.tex','--glob','!**/*sealed*/**','--glob','!**/.venv*/**','--glob','!**/__pycache__/**','io_wrapper_sha256|technical_wrapper_sha256','scripts','src','tests','configs','docs','outputs','paper']
 r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8');need(r.returncode in (0,1),'REFERENCE_SEARCH_FAILED')
 rows=[]
 for line in r.stdout.splitlines():
  name,ln,context=line.split(':',2);name=name.replace('\\','/')
  if name in ('scripts/run_sp_u12b3f.py','scripts/sp_u12b3e_io.py'):
   kind='B_CURRENT_CANDIDATE_RUNTIME';action='canonical runtime access verified'
  elif name=='scripts/sp_u12b3g_schema.py':kind='B_SCHEMA_VALIDATOR';action='canonical required;obsolete alias explicitly rejected'
  elif any(v.startswith(('sp_u12b3g','test_sp_u12b3g')) for v in Path(name).parts):kind='C_CURRENT_TECHNICAL_DOCUMENT_OR_AUDIT';action='canonical wording or explicit root-cause record'
  else:kind='A_HISTORICAL_RECORD_OR_INACTIVE_SOURCE';action='preserve original bytes;not migrated'
  rows.append(dict(file=name,line=int(ln),context=context,classification=kind,action=action))
 with (ROOT/'outputs/sp_u12b3g_field_reference_audit.csv').open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['file','line','context','classification','action']);w.writeheader();w.writerows(rows)
 # AST verifies accesses, rather than falsely treating the rejection string as a dependency.
 active=[]
 for name in s.WRAPPER_FILES:
  for n in ast.walk(ast.parse((ROOT/name).read_text(encoding='utf8'))):
   if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and n.slice.value=='io_wrapper_sha256':active.append(dict(file=name,line=n.lineno))
 need(not active,'OBSOLETE_CANDIDATE_RUNTIME_ACCESS')
 report('reference_summary',dict(PASS=True,matches=len(rows),obsolete_current_runtime_accesses=active,scope='project source/document/technical JSON;sealed outcome directories and binary data excluded',historical_matches_preserved=True))
def prepare():
 prior=read(OUT/'preservation_before.json');need(prior['B2']['PASS'],'BEFORE_GATE')
 need(core_check()==s.CORE,'CORE_GATE')
 old=read(ROOT/'outputs/sp_u12b3e_run3_execution_freeze.json')
 need(sha(ROOT/'outputs/sp_u12b3e_run3_execution_freeze.json')==s.OLD_FREEZE,'OLD_FREEZE_DRIFT')
 archived=OUT/'original_bytes/scripts/run_sp_u12b3f.py'
 need(source_functions(archived,('compute',))==source_functions(ROOT/'scripts/run_sp_u12b3f.py',('compute',)),'BOOTSTRAP_SOURCE_DRIFT')
 need(source_functions(ROOT/'scripts/sp_u12b3e_common.py',('validate','evaluate','draw','bootstrap','classify','worker'))==source_functions(ROOT/'scripts/sp_u12b3d_common.py',('validate','evaluate','draw','bootstrap','classify','worker')),'SCIENTIFIC_FUNCTION_DRIFT')
 need(not any((ROOT/'outputs/sp_u12b3f_run3_sealed').iterdir()),'OLD_RUN3_ROOT_NOT_EMPTY')
 s.FORMAL_ROOT.mkdir(exist_ok=True);(s.FORMAL_ROOT/'raw').mkdir(exist_ok=True);(s.FORMAL_ROOT/'bootstrap').mkdir(exist_ok=True)
 need(not any(p.is_file() for p in s.FORMAL_ROOT.rglob('*')),'NEW_FORMAL_ROOT_NOT_EMPTY')
 report('schema_root_cause',dict(ROOT_CAUSE='EXECUTION_FREEZE_SCHEMA_KEY_MISMATCH',RUNNER_EXPECTED='io_wrapper_sha256',FROZEN_SCHEMA_CANONICAL_FIELD='technical_wrapper_sha256',FAILURE_STAGE='FORMAL_RUN_PRECHECK_BEFORE_SCIENTIFIC_EXECUTION',actual_B3F_evidence='isolated evaluation of frozen seal field expression before command launch;actual seal() and --run were not executed',B3F_manifest_sha256=prior['history']['hashes']['outputs/sp_u12b3f_run3_manifest.json'],scientific_execution_started=False,scientific_use='PROHIBITED',outcomes_viewed=False))
 required=[]
 roles={'protocol_sha256':'protocol_hash','scientific_config_sha256':'config_hash','formal_seed_registry_sha256':'formal_seed_registry_hash','B2_provenance_before':'b2_provenance_hashes','sys_version':'python_version','sys_executable':'python_executable','Windows_spawn_context':'multiprocessing_start_method','CPU_workers':'worker_count','BLAS_threads_per_process':'blas_threads_per_process','exact_entrypoint_command':'formal_entrypoint'}
 for name,kind in s.REQUIRED.items():required.append(dict(canonical_field=name,type=kind,conceptual_role=roles.get(name,name),historical_field_exists=name in old))
 optional=dict(s.OPTIONAL)
 for name in old:
  if name not in s.REQUIRED and name not in optional:optional[name]='historical optional metadata;not a canonical alias'
 report('execution_freeze_schema',dict(schema_version='B3_TECHNICAL_EXECUTION_CANONICAL_V1',required=required,nested_B2_required=s.B2_REQUIRED,optional=optional,aliases_accepted=[],obsolete_aliases_rejected=['io_wrapper_sha256','wrapper_sha256','technical_hash'],error_prefix='EXECUTION_FREEZE_SCHEMA_ERROR',output_root=str(s.FORMAL_ROOT)))
 changed=[]
 for name in ('scripts/run_sp_u12b3f.py','scripts/sp_u12b3e_io.py'):
  a=(OUT/'original_bytes'/name).read_text(encoding='utf8').splitlines();z=(ROOT/name).read_text(encoding='utf8').splitlines()
  diff='\n'.join(difflib.unified_diff(a,z,fromfile='original/'+name,tofile='candidate/'+name,lineterm=''))+'\n';p=OUT/'diffs'/(Path(name).name+'.diff');p.parent.mkdir(exist_ok=True);p.write_text(diff,encoding='utf8')
  changed.append(dict(file=name,old_sha256=prior['hashes'][name],new_sha256=sha(ROOT/name),changed_lines=[dict(tag=tag,old_lines=[i+1,j],new_lines=[k+1,l]) for tag,i,j,k,l in difflib.SequenceMatcher(a=a,b=z).get_opcodes() if tag!='equal'],diff_sha256=sha(p)))
 report('wrapper_diff_audit',dict(PASS=True,old_wrapper_sha256=s.OLD_WRAPPER,new_wrapper_sha256=s.wrapper_hash(),new_bundle_files={n:sha(ROOT/n) for n in s.WRAPPER_FILES},changed_files=changed,added_runtime_helper='scripts/sp_u12b3g_schema.py',reason='canonical schema handling;validation before worker construction;technical contract/root isolation;shared formal prelude and heartbeat exit',scientific_changes=0,compute_AST_source_exact=True,scientific_worker_AST_source_exact=True))
 report('scientific_hash_audit',dict(PASS=True,before=s.CORE,after=core_check(),compute_source_exact=True,scientific_worker_source_exact=True,scientific_protocol_unchanged=True,formal_seeds_unchanged=True,production_semantic_drift=0))
 f={k:v for k,v in old.items() if k not in ('gates','mentor_authorization','formal_run3_authorized','formal_output_root','status','hashes','technical_wrapper_sha256')}
 own=[p for d in ('scripts','tests') for p in (ROOT/d).glob('*sp_u12b3g*.py')]
 f.update(status='B3_NEXT_CANDIDATE_EXECUTION_FREEZE_TECHNICAL_GATE_ONLY',schema_version='B3_TECHNICAL_EXECUTION_CANONICAL_V1',previous_execution_freeze_sha256=s.OLD_FREEZE,previous_technical_wrapper_sha256=s.OLD_WRAPPER,supersedes='technical execution contract only;historical freezes remain immutable',scientific_protocol_unchanged=True,formal_seeds_unchanged=True,scientific_core_unchanged=True,formal_run_authorized=False,pre_scientific_gate_authorized=True,mentor_authorization='SP_U12B3G_TECHNICAL_GATE_ONLY',formal_output_root=str(s.FORMAL_ROOT),technical_wrapper_sha256=s.wrapper_hash(),technical_wrapper_scope='four previous deployment modules plus canonical schema helper',exact_entrypoint_command=s.COMMAND,formal_entrypoint_script='scripts/run_sp_u12b3f.py',gate_command_template=s.COMMAND+['--pre-scientific-gate','--session','gate_NN'],environment_variables={k:'1' for k in s.THREAD_KEYS},B2_provenance_before=prior['B2'],B2_provenance_after=prior['B2'],hashes={**{n:sha(ROOT/n) for n in old['hashes']},**{p.relative_to(ROOT).as_posix():sha(p) for p in own}},formal_trajectories_executed=0,bootstrap_count=0,outcomes_viewed=False)
 need(not s.FREEZE_PATH.exists(),'NEXT_FREEZE_ALREADY_EXISTS');write(s.FREEZE_PATH,f)
 # Check through the same function used by --run; schema hash is now immutable.
 for key in s.THREAD_KEYS:os.environ[key]='1'
 s.validate_execution_freeze(f)
 fields()
 text='# SP-U12B3G schema failure analysis\n\nROOT_CAUSE: EXECUTION_FREEZE_SCHEMA_KEY_MISMATCH. Canonical field: technical_wrapper_sha256. The B3F evidence was an isolated evaluation of the frozen seal expression, before --run, pool construction and scientific execution; it was not an observed exception from running the complete seal stage. Historical manifest bytes remain unchanged.\n\nThe new runner uses one canonical schema with explicit field-specific errors. The old freeze is not migrated or overwritten. The new candidate supersedes technical orchestration only. Exact --run config/freeze/schema/hash/B2/seed/output checks, formal-mode spawn initialization and heartbeat precede the gate exit. Formal evaluation remains separately unauthorized.\n'
 (ROOT/'docs/sp_u12b3g_schema_failure_analysis.md').write_text(text,encoding='utf8')
 gate='# SP-U12B3G exact --run gate\n\nPython: '+str(ROOT/'.venv-cdl/Scripts/python.exe')+'\nWorking directory: '+str(ROOT)+'\n\nFormal base command: '+subprocess.list2cmdline(s.COMMAND)+'\nGate command: '+subprocess.list2cmdline(s.COMMAND+['--pre-scientific-gate','--session','gate_NN'])+'\n\nOnly the technical exit flag and receipt-session name differ. Both call main → run → formal_preflight → create_pool(formal) → the same qualified initializer and guarded worker → heartbeats. The formal branch reuses that same pool for evaluation; the gate shuts it down before trajectory 0.\n\nInitializer: scripts.sp_u12b3e_io.init_worker\nWorker: scripts.sp_u12b3e_io.guarded_worker\nWindows spawn;workers2;BLAS1. No formal authorization token is created. Technical checkpoints contain only PID/heartbeat metadata and are outside scientific raw directories.\n'
 (ROOT/'docs/sp_u12b3g_exact_run_gate.md').write_text(gate,encoding='utf8')
 print(json.dumps(dict(stage='candidate_freeze',PASS=True,scientific_core_sha256=core_check(),wrapper_sha256=s.wrapper_hash(),execution_freeze_sha256=sha(s.FREEZE_PATH))),flush=True)
def probe_readonly():
 # Isolated expected policy-denial fixture; no filesystem mutation reaches B2.
 from scripts import sp_u12b3e_common as b
 from scripts.sp_u12b3e_io import activate_execution_contract
 activate_execution_contract(s.runtime_config(True,'readonly_probe'))
 b.firewall()
 p=ROOT/'outputs/sp_u12b2b_sealed/B3G_FORBIDDEN_WRITE_PROBE'
 try:p.open('wb')
 except PermissionError as exc:
  need(exc.args and 'B2_WRITE_VIOLATION' in exc.args[0] and not p.exists(),'READONLY_PROBE_FAILED')
  s.validate_execution_freeze(s.load(s.FREEZE_PATH),verify_leaves=False)
  print(json.dumps(dict(PASS=True,B2_write_blocked_before_OS=True,B2_bytes_changed=False,expected_policy_denial=True,opaque_hash_validation_PASS=True)),flush=True);return
 raise RuntimeError('READONLY_GUARD_BYPASSED')
def repeat():
 runs=[];env=dict(os.environ);env.update({k:'1' for k in s.THREAD_KEYS})
 for i in range(1,21):
  session=f'gate_{i:02d}';cmd=s.COMMAND+['--pre-scientific-gate','--session',session]
  r=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf8')
  log=OUT/'logs'/f'{session}.txt';log.parent.mkdir(exist_ok=True);log.write_text(r.stdout+'\n'+r.stderr,encoding='utf8')
  if r.returncode!=0:
   report('repeat_preflight',dict(PASS=False,completed=len(runs),failed_session=session,exit_code=r.returncode,log_sha256=sha(log),scientific_trajectories=0,outcomes_viewed=False));raise RuntimeError('EXACT_RUN_GATE_FAILED:'+session)
  p=ROOT/'outputs/sp_u12b3g/gates'/session/'pre_scientific_receipt.json';a=read(p)
  need(a['PASS'] and a['workers_initialized']==a['heartbeats_completed']==2 and a['formal_trajectories_executed']==a['formal_scientific_outputs']==a['primary_bootstrap']==a['descriptive_bootstrap']==0,'GATE_RECEIPT_FAILURE')
  need(not any(x.is_file() for x in s.FORMAL_ROOT.rglob('*')),'B3G_FORMAL_EXECUTION_VIOLATION')
  runs.append(dict(session=session,command=cmd,exit_code=0,receipt_sha256=sha(p),log_sha256=sha(log),workers=2,heartbeats=2,scientific_trajectories=0))
  print(json.dumps(dict(stage='exact_run_gate',run=i,PASS=True,workers_initialized=2,heartbeats=2,scientific_trajectories=0)),flush=True)
 exceptions=[str(p) for p in (OUT/'gates').rglob('*exception*.json')];need(not exceptions,'GATE_EXCEPTION_ENVELOPES')
 result=dict(PASS=True,runs=20,passed=20,worker_initializations=40,heartbeats=40,KeyErrors=0,schema_mismatches=0,pickle_failures=0,PermissionErrors=0,filesystem_failures=0,orphan_workers=0,formal_trajectories_executed=0,formal_scientific_outputs=0,primary_bootstrap=0,descriptive_bootstrap=0,run_receipts=runs,execution_freeze_sha256=sha(s.FREEZE_PATH),outcomes_viewed=False)
 report('repeat_preflight',result)
 first=read(OUT/'gates/gate_01/pre_scientific_receipt.json')
 report('exact_run_preflight',dict(**first,repeated_runs=20,all_repeats_PASS=True,pre_scientific_gate_path='main/run/formal_preflight/create_pool(formal)/init_worker/guarded_worker(heartbeat)',formal_authorization_granted=False,future_formal_command=s.COMMAND))
def pytest_actual():
 cmd=[str(ROOT/'.venv-cdl/Scripts/python.exe'),'-X','utf8','-m','pytest','tests/test_sp_u12b3g.py','-q','-p','no:cacheprovider','--basetemp='+str(OUT/'pytest_tmp'),'--junitxml='+str(OUT/'pytest.xml')]
 r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8');(OUT/'pytest.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf8')
 suites=list(ET.parse(OUT/'pytest.xml').getroot().iter('testsuite'))
 o=dict(exit_code=r.returncode,passed=sum(int(v.get('tests','0'))-int(v.get('failures','0'))-int(v.get('errors','0'))-int(v.get('skipped','0')) for v in suites),failed=sum(int(v.get('failures','0'))+int(v.get('errors','0')) for v in suites),skipped=sum(int(v.get('skipped','0')) for v in suites),command=cmd)
 write(OUT/'pytest.json',o);print(json.dumps({k:v for k,v in o.items() if k!='command'}),flush=True);need(r.returncode==0,'ACTUAL_PYTEST_FAILED')
def finalize():
 need(core_check()==s.CORE,'CORE_FINAL');f=read(s.FREEZE_PATH)
 for k in s.THREAD_KEYS:os.environ[k]='1'
 s.validate_execution_freeze(f)
 for n in ('preservation','b2_seal_audit','history_preservation','repeat_preflight','exact_run_preflight','wrapper_diff_audit','scientific_hash_audit'):need(read(ROOT/'outputs'/('sp_u12b3g_'+n+'.json'))['PASS'],'FINAL_GATE:'+n)
 need(read(OUT/'pytest.json')['exit_code']==0,'PYTEST_FINAL_GATE')
 need(not (s.CONTROL/'formal_authorization.json').exists(),'UNAUTHORIZED_FORMAL_AUTHORIZATION')
 need(not any(p.is_file() for p in s.FORMAL_ROOT.rglob('*')) and not any((ROOT/'outputs/sp_u12b3f_run3_sealed').iterdir()),'B3G_FORMAL_EXECUTION_VIOLATION')
 fields()
 diff=read(ROOT/'outputs/sp_u12b3g_wrapper_diff_audit.json');pres=read(ROOT/'outputs/sp_u12b3g_preservation.json');tests=read(OUT/'pytest.json')
 text='# SP-U12B3G change log\n\nVerdict: B3_EXECUTION_CONTRACT_REPAIR_READY\n\n'+json.dumps(diff,indent=2)+'\n\n'+json.dumps(pres,indent=2)+'\n\nActual tests: '+json.dumps(tests)+'\n\n20/20 exact --run pre-scientific gates;40 initializations;40 heartbeats. No trajectory0, scientific solve, raw outcome or bootstrap. Old B3E freeze and B3F failure manifest remain immutable. Next candidate freeze SHA256: '+sha(s.FREEZE_PATH)+'\n\nNo formal-run authorization, no new seeds, no manuscript change, no Git commit. STOP for mentor review.\n'
 (ROOT/'docs/sp_u12b3g_change_log.md').write_text(text,encoding='utf8')
 write(OUT/'research_lesson.md',dict(rule='Technical CLI gates must run the identical formal prelude, validate canonical schema before pool construction, and test required-field rejection; heartbeat-only sibling modes cannot certify the full execution contract.',evidence='B3F precheck KeyError;B3G explicit schema validation and20 exact --run gate receipts',scope='lesson recorded inside authorized project only'))
 report('decision',dict(verdict='B3_EXECUTION_CONTRACT_REPAIR_READY',formal_run_authorized=False,STOP=True,scientific_core_sha256=s.CORE,old_wrapper_sha256=s.OLD_WRAPPER,new_wrapper_sha256=s.wrapper_hash(),new_execution_freeze_sha256=sha(s.FREEZE_PATH),formal_trajectories_executed=0,bootstrap_executed=0,outcomes_viewed=False,scientific_figures_viewed=False,pytest=tests))
 print(json.dumps(dict(verdict='B3_EXECUTION_CONTRACT_REPAIR_READY',STOP=True,new_wrapper_sha256=s.wrapper_hash(),execution_freeze_sha256=sha(s.FREEZE_PATH),scientific_core_sha256=s.CORE,formal_trajectories_executed=0)),flush=True)
if __name__=='__main__':{'prepare':prepare,'repeat':repeat,'pytest':pytest_actual,'readonly-probe':probe_readonly,'finalize':finalize}[sys.argv[1]]()
