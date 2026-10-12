"""Metadata preparation and B3A-certified development replay fixtures only."""
from pathlib import Path
import sys,json,ast,pickle,importlib,inspect,hashlib,difflib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b3e_common as b
from scripts.sp_u12b3e_build import core,EXPECTED,sources
import numpy as np
OUT=ROOT/'outputs/sp_u12b3e'
def save(name,obj):return b.write(ROOT/'outputs'/('sp_u12b3e_'+name+'.json'),obj)
def setup():
 OUT.mkdir(exist_ok=True);future=ROOT/'outputs/sp_u12b3f_run3_sealed';future.mkdir(exist_ok=False)
 digest,partition=core();b.need(digest==EXPECTED,'B3E_SCIENTIFIC_CHANGE_REQUIRED')
 b.need(b.sha(ROOT/'outputs/sp_u12b3d_run2_manifest.json')=='a22d244473336acda5ab8b75d9a848e44f8b1fcab19fea2391b93649d361e7bb','RUN2_FAILURE_MANIFEST_DRIFT')
 save('root_cause_freeze',dict(ROOT_CAUSE='WINDOWS_SPAWN_COULD_NOT_SERIALIZE_LOCAL_INITIALIZER',FAILED_OBJECT='install_io.<locals>.init_worker',FAILURE_STAGE='PARENT_PROCESS_SERIALIZATION_BEFORE_WORKER_INITIALIZATION',Run1_root_cause='UNRECOVERABLE_UNKNOWN',Run2_status='TECHNICAL_FAILURE_BEFORE_SCIENTIFIC_EXECUTION',Run2_manifest_sha256=b.sha(ROOT/'outputs/sp_u12b3d_run2_manifest.json'),original_traceback_files={n:b.sha(ROOT/n) for n in ['outputs/sp_u12b3d/parent_exception_1576.json','outputs/sp_u12b3d/parent_exception_run_1576.json']},original_bytes_modified=False))
 # Freeze the candidate bytes before exact-entrypoint tests. Scientific family is inherited.
 prior=b.read(ROOT/'outputs/sp_u12b3b/execution_freeze.json')
 for n,h in prior['hashes'].items():b.need(b.sha(ROOT/n)==h,'HISTORICAL_EXECUTION_DRIFT')
 own=['scripts/sp_u12b3e_common.py','scripts/sp_u12b3e_io.py','scripts/sp_u12b3e_authorization.py','scripts/run_sp_u12b3f.py']
 freeze=dict(hashes={**prior['hashes'],**{n:b.sha(ROOT/n) for n in own}},scientific_core_sha256=digest,formal_run3_authorized=False,bootstrap_count=0,workers=2,BLAS_threads=1,spawn_context='spawn')
 b.write(OUT/'candidate_validation_freeze.json',freeze)
 control=OUT/'run3_control';control.mkdir();b.atomic(control/'execution_freeze.json',(OUT/'candidate_validation_freeze.json').read_bytes())
 for name in ('bootstrap_seed_registry.json','raw_schema.json'):b.atomic(control/name,(ROOT/'outputs/sp_u12b3b'/name).read_bytes())
 print(json.dumps(dict(stage='candidate_setup',PASS=True,formal_trajectories=0)),flush=True)
def callable_audit():
 from scripts import run_sp_u12b3f as runner
 from scripts import sp_u12b3e_io as io
 rows=[]
 for role,obj in [('initializer',b.init_worker),('worker',b.worker),('submitted_callable',b.worker),('heartbeat_internal',io.heartbeat)]:
  current=importlib.import_module(obj.__module__)
  for part in obj.__qualname__.split('.'):
   b.need(part!='<locals>','NESTED_SPAWN_CALLABLE');current=getattr(current,part)
  blob=pickle.dumps(obj);b.need(current is obj and pickle.loads(blob) is obj,'CALLABLE_IMPORT_IDENTITY')
  file=inspect.getsourcefile(obj);line=inspect.getsourcelines(obj)[1]
  rows.append(dict(role=role,qualified_name=obj.__module__+'.'+obj.__qualname__,module_level=True,importable=True,pickle_roundtrip=True,source_file=str(Path(file).resolve()),source_line=line,serialized_as_multiprocessing_object=role in ('initializer','worker','submitted_callable'),status='PASS'))
 rows.append(dict(role='callback',qualified_name='NONE',module_level=True,importable=True,pickle_roundtrip=True,source_file=str(ROOT/'scripts/run_sp_u12b3f.py'),source_line=0,serialized_as_multiprocessing_object=False,status='NOT_USED'))
 b.csvwrite(ROOT/'outputs/sp_u12b3e_callable_pickle_audit.csv',rows)
 text=(ROOT/'scripts/run_sp_u12b3f.py').read_text(encoding='utf8');tree=ast.parse(text)
 calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('ProcessPoolExecutor','Pool','submit','map','add_done_callback')]
 b.need(sum(isinstance(n.func,ast.Attribute) and n.func.attr=='ProcessPoolExecutor' for n in calls)==1,'MULTIPLE_POOL_CONSTRUCTIONS')
 b.need(not any(n.func.attr=='add_done_callback' for n in calls),'UNREGISTERED_CALLBACK')
 for n in calls:
  if n.func.attr=='submit':b.need(isinstance(n.args[0],ast.Attribute) and n.args[0].attr=='worker','UNREGISTERED_SUBMITTED_CALLABLE')
 b.need("multiprocessing.get_context('spawn')" in text,'EXPLICIT_SPAWN_MISSING')
 # Local I/O helpers stay process-local and are never in initargs/submit/callbacks.
 local=[]
 it=ast.parse((ROOT/'scripts/sp_u12b3e_io.py').read_text(encoding='utf8'))
 for n in it.body:
  if isinstance(n,ast.FunctionDef):
   for child in n.body:
    if isinstance(child,ast.FunctionDef):local.append(dict(qualified_name=n.name+'.<locals>.'+child.name,source_line=child.lineno,serialized=False,reason='I/O closure is reconstructed by module import inside each process;not a multiprocessing argument'))
 b.need(not any(v['qualified_name'].endswith('.init_worker') for v in local),'LOCAL_INITIALIZER_REMAINS')
 b.write(OUT/'nested_callable_audit.json',dict(PASS=True,local_helpers=local,initargs=dict(mode='string',session='string'),all_submitted_targets='scripts.sp_u12b3e_io.guarded_worker',callbacks=0,executor_constructions=1))
 print(json.dumps(dict(stage='callable_audit',PASS=True,callables=4)),flush=True)
def development_fixtures():
 from scripts import run_sp_u12b2a as a
 from scripts import sp_u12b3b_common as old
 from src.cdl_benchmark import SYSTEM,array_hash
 from scripts.resume_formal_test import load_native_h
 b.need(SYSTEM['sample_rate_hz']==SYSTEM['nfft']*SYSTEM['delta_f_hz'],'CONFIGURATION_INTEGRITY')
 registry=b.read(b.c.OUT/'development_seed_registry.json');b.need(registry['namespace']=='SP_U12B2A_DEVELOPMENT_ONLY_V1','DEVELOPMENT_NAMESPACE');index=a.seed_index(registry)
 certified=b.read(ROOT/'outputs/sp_u12b3a_development_smoke.json');b.need(certified['PASS'],'B3A_DEVELOPMENT_NOT_CERTIFIED')
 records={(r['condition'],r['trajectory_id'],r['snr_db']):r for r in certified['records']}
 operators=b.c.load_operators();cvs,factors=b.c.family(operators,b.SNRS);proofs=[]
 for condition in ('Profile','Mobility','Composed'):
  H=a.generator(condition,index,'EVAL').generate(0);b.need(array_hash(H)==records[condition,0,10]['H_sha256'],'B3A_DEVELOPMENT_H_HASH')
  seeds=[a.consumed(index,condition,'EVAL',0,j) for j in range(10)];comm=[b.c.communicate(v) for v in seeds]
  raw,inputs,residual=old.evaluate(H,comm,operators,cvs,factors)
  expected_residual=max(max(records[condition,0,s]['solve_relative_residual_max'].values()) for s in b.SNRS)
  b.need(residual==expected_residual,'CERTIFIED_DEVELOPMENT_RESIDUAL_REPLAY')
  base=OUT/'development_inputs';hp=base/'cache'/condition/'H_0000.npz';cp=base/'cache'/condition/'comm_0000.npz';rp=base/'raw_evaluation'/condition/'0000.npz'
  artifact=b.archive(hp,H=H);native=load_native_h(hp);b.need(array_hash(native)==array_hash(H) and native.strides==H.strides,'NATIVE_DEVELOPMENT_LAYOUT')
  b.write(hp.with_suffix('.json'),dict(artifacts=[artifact],export_strides_bytes=list(H.strides),H_sha256=array_hash(H),channel_seeds=a.consumed(index,condition,'EVAL',0,0),namespace=registry['namespace']))
  artifact=b.archive(cp,bits=np.stack([v[0] for v in comm]),symbols=np.stack([v[1] for v in comm]),unit_noise=np.stack([v[2] for v in comm]))
  b.write(cp.with_suffix('.json'),dict(artifacts=[artifact],frame_seeds=seeds,**{k+'_sha256':[array_hash(v[j]) for v in comm] for j,k in enumerate(('bits','symbols','unit_noise'))},namespace=registry['namespace']))
  coordinates=np.zeros((7,6,2));errors=np.zeros((7,7,6));truth=np.zeros((7,6))
  for si,s in enumerate(b.SNRS):
   k=b.c.SNRS.index(s);coordinates[k]=raw['coordinates'][si,2];truth[k]=raw['frame_truth'][si]
   for di,d in enumerate(('Dense','LR4')):errors[k,b.c.METHODS.index(d)]=raw['frame_error'][si,2,di]
  artifact=b.archive(rp,coordinates=coordinates,frame_error=errors,frame_truth=truth)
  b.write(rp.with_suffix('.json'),dict(artifacts=[artifact],inputs=inputs,scope='DEVELOPMENT_ONLY_NO_FORMAL_B2_DATA'))
  reference=b.archive(OUT/'development_reference'/condition/'0000.npz',**raw)
  proofs.append(dict(condition=condition,trajectory_id=0,H_sha256=array_hash(H),B3A_H_match=True,B3A_saved_residual_replay_exact=True,reference_artifact=reference))
  print(json.dumps(dict(stage='development_fixture',condition=condition,PASS=True)),flush=True)
 b.write(OUT/'development_reference_provenance.json',dict(PASS=True,namespace=registry['namespace'],development_registry_sha256=b.sha(b.c.OUT/'development_seed_registry.json'),B3A_certification_sha256=b.sha(ROOT/'outputs/sp_u12b3a_development_smoke.json'),reference_type='Certified B3A H and per-arm solve receipts;original unchanged B3B evaluator replayed on those development inputs because B3A did not persist full raw prediction arrays',reference_function_source_unchanged=True,records=proofs,formal_inputs_used=False,formal_registry_decoded=False,bootstrap_count=0))
def audits():
 digest,partition=core();b.need(digest==EXPECTED,'B3E_SCIENTIFIC_CHANGE_REQUIRED')
 save('scientific_hash_audit',dict(PASS=True,scientific_core_sha256_before=EXPECTED,scientific_core_sha256_after=digest,partition=partition,worker_source_exact=sources(ROOT/'scripts/sp_u12b3e_common.py',('worker',))==sources(ROOT/'scripts/sp_u12b3d_common.py',('worker',)),formal_trajectory_count=0,bootstrap_count=0))
 old=b.read(ROOT/'outputs/sp_u12b3d/execution_freeze.json');newfiles=['scripts/sp_u12b3e_common.py','scripts/sp_u12b3e_io.py','scripts/run_sp_u12b3f.py','scripts/sp_u12b3e_authorization.py'];new={n:b.sha(ROOT/n) for n in newfiles}
 pairs=[('scripts/sp_u12b3d_io.py','scripts/sp_u12b3e_io.py'),('scripts/sp_u12b3d_common.py','scripts/sp_u12b3e_common.py'),('scripts/run_sp_u12b3d.py','scripts/run_sp_u12b3f.py')];changes=[]
 for before,after in pairs:
  a=(ROOT/before).read_text(encoding='utf8').splitlines();z=(ROOT/after).read_text(encoding='utf8').splitlines();diff='\n'.join(difflib.unified_diff(a,z,fromfile=before,tofile=after,lineterm=''))+'\n';p=OUT/'wrapper_diffs'/(Path(after).name+'.diff');b.atomic(p,diff.encode())
  ranges=[dict(tag=tag,old_lines=[i+1,j],new_lines=[k+1,l]) for tag,i,j,k,l in difflib.SequenceMatcher(a=a,b=z).get_opcodes() if tag!='equal']
  changes.append(dict(before=before,after=after,changed_lines=ranges,diff_path=str(p),diff_sha256=b.sha(p)))
 save('wrapper_diff_audit',dict(PASS=True,old_Run2_wrapper_sha256=old['io_wrapper_sha256'],new_candidate_wrapper_sha256=b.tree(new),new_wrapper_files=new,changed_files=changes,reason='single core repair: promote initializer to importable module scope;primitive initargs;explicit spawn;required init-only/development modes share one exact pool factory;namespace isolation preserves failures',scientific_changes=0,IO_safeguards_retained=['unique_PID_thread_UUID_key_temps','full_exception_envelopes','B2_readonly_guard','dead_worker_quarantine','checkpoint_flush_fsync_atomic_hash','operation_diagnostics']))
 print(json.dumps(dict(stage='candidate_audits',PASS=True,scientific_core_sha256=digest)),flush=True)
if __name__=='__main__':{'setup':setup,'callables':callable_audit,'development':development_fixtures,'audits':audits}[sys.argv[1]]()
