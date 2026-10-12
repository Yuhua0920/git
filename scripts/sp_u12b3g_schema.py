"""Canonical technical execution contract. No estimator/metric/bootstrap changes."""
from pathlib import Path
import sys,os,json,hashlib,re
ROOT=Path(__file__).resolve().parents[1]
FREEZE_PATH=ROOT/'outputs/sp_u12b3g_next_execution_freeze.json'
CONTROL=ROOT/'outputs/sp_u12b3g/next_control'
FORMAL_ROOT=ROOT/'outputs/sp_u12b3g_next_sealed'
CORE='994339ccc806e3ca94ae644a954e9382a2fa0df8ed2a808d0feb28528e9bdfa1'
OLD_FREEZE='4e79f8f8bb267f4bf8be463bd2cee6b7d25914218a0c36936543e96d52f45ce6'
OLD_WRAPPER='32a35ac8fdfb59bfba22e49821b4d0b52d6392ea376dd63493cfc3fe92e30a42'
WRAPPER_FILES=('scripts/sp_u12b3e_common.py','scripts/sp_u12b3e_io.py','scripts/run_sp_u12b3f.py','scripts/sp_u12b3e_authorization.py','scripts/sp_u12b3g_schema.py')
COMMAND=[str(ROOT/'.venv-cdl/Scripts/python.exe'),'-X','utf8','scripts/run_sp_u12b3f.py','--run']
THREAD_KEYS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
REQUIRED={
 'protocol_sha256':'str','scientific_config_sha256':'str','formal_seed_registry_sha256':'str','bootstrap_seed_registry_sha256':'str',
 'B2_provenance_before':'dict','B2_provenance_after':'dict','scientific_core_sha256':'str','technical_wrapper_sha256':'str',
 'sys_version':'str','sys_executable':'str','Windows_spawn_context':'str','CPU_workers':'int','BLAS_threads_per_process':'int',
 'formal_output_root':'str','exact_entrypoint_command':'list','working_directory':'str','hashes':'dict'}
B2_REQUIRED={'PASS':'bool','files':'int','raw_hash_tree':'str','artifact_hash_tree':'str','manifest_sha256':'str','certificate_sha256':'str','outcomes_decoded':'bool'}
OPTIONAL={'status':'candidate state;not formal authorization','schema_version':'explicit technical schema version','mentor_authorization':'historical/current authorization description','formal_run_authorized':'must remain false until separate mentor authorization','previous_execution_freeze_sha256':'immutable predecessor','previous_technical_wrapper_sha256':'immutable predecessor wrapper','supersedes':'technical contract only','scientific_protocol_unchanged':'provenance','formal_seeds_unchanged':'provenance','scientific_core_unchanged':'provenance','formal_seed_registry_path':'metadata path','technical_wrapper_scope':'bundle scope','old_Run2_wrapper_sha256':'historical only','python_launcher':'engineering executable','tested_modes':'technical modes','formal_trajectories_executed':'technical evidence only','bootstrap_count':'technical evidence only','gates':'technical evidence references','outcomes_viewed':'seal policy','formal_entrypoint_script':'explicit script identity','gate_command_template':'technical exit flag only','environment_variables':'required thread settings','pre_scientific_gate_authorized':'technical-only permission','Run1_Run2_preservation':'historical provenance'}
class ExecutionFreezeSchemaError(RuntimeError):pass
def error(field,reason):raise ExecutionFreezeSchemaError('EXECUTION_FREEZE_SCHEMA_ERROR: '+field+': '+reason)
def sha(p):
 # Hash bytes through the existing opaque-read scope once the firewall is active.
 b=sys.modules.get('scripts.sp_u12b3e_common')
 if b is not None and getattr(b,'_guard',False):return b.sha(p)
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def tree(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(p):
 try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
 except (OSError,ValueError) as exc:error('execution_freeze',type(exc).__name__)
def wrapper_hash():return tree({n:sha(ROOT/n) for n in WRAPPER_FILES})
def validate_structure(f):
 if not isinstance(f,dict):error('execution_freeze','must be object')
 types={'str':str,'dict':dict,'int':int,'list':list,'bool':bool}
 def required(o,spec,prefix=''):
  for name,kind in spec.items():
   key=prefix+name
   if name not in o:error(key,'required canonical field missing')
   if o[name] is None:error(key,'required value is null')
   if type(o[name]) is not types[kind]:error(key,'expected '+kind)
   if kind in ('str','dict','list') and not o[name]:error(key,'required value empty')
 required(f,REQUIRED)
 if 'io_wrapper_sha256' in f:error('io_wrapper_sha256','obsolete alias rejected;use technical_wrapper_sha256')
 for name in ('B2_provenance_before','B2_provenance_after'):required(f[name],B2_REQUIRED,name+'.')
 for name in ('protocol_sha256','scientific_config_sha256','formal_seed_registry_sha256','bootstrap_seed_registry_sha256','scientific_core_sha256','technical_wrapper_sha256'):
  if not re.fullmatch('[0-9a-f]{64}',f[name]):error(name,'invalid SHA256')
 for name,h in f['hashes'].items():
  if not isinstance(name,str) or not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h):error('hashes','invalid leaf')
 return f
def validate_execution_freeze(f,verify_leaves=True):
 validate_structure(f)
 def equal(field,v):
  if f[field]!=v:error(field,'actual value mismatch')
 from scripts.sp_u12b3e_build import core
 equal('scientific_core_sha256',core()[0])
 if f['scientific_core_sha256']!=CORE:raise RuntimeError('B3G_SCIENTIFIC_CODE_CHANGE_REQUIRED')
 equal('technical_wrapper_sha256',wrapper_hash())
 for field,path in [('protocol_sha256','docs/sp_u12b3b_protocol.md'),('scientific_config_sha256','configs/sp_u12b3b_sealed.json'),('formal_seed_registry_sha256','outputs/sp_u12b2a_seed_registry.json'),('bootstrap_seed_registry_sha256','outputs/sp_u12b3b/bootstrap_seed_registry.json')]:equal(field,sha(ROOT/path))
 equal('CPU_workers',2);equal('BLAS_threads_per_process',1);equal('Windows_spawn_context','spawn');equal('formal_output_root',str(FORMAL_ROOT));equal('working_directory',str(ROOT));equal('sys_version',sys.version);equal('sys_executable',sys.executable);equal('exact_entrypoint_command',COMMAND)
 if Path.cwd().resolve()!=ROOT:error('working_directory','process cwd mismatch')
 for key in THREAD_KEYS:
  if os.environ.get(key)!='1':error('environment_variables.'+key,'must be 1')
 m=load(ROOT/'outputs/sp_u12b2b_sealed_manifest.json')
 actual=dict(PASS=True,files=len(m.get('files',{})),raw_hash_tree=m.get('raw_data_hash_tree'),artifact_hash_tree=m.get('all_artifact_hash_tree'),manifest_sha256=sha(ROOT/'outputs/sp_u12b2b_sealed_manifest.json'),certificate_sha256=sha(ROOT/'docs/sp_u12b2b_seal_certificate.md'),outcomes_decoded=False)
 for field in ('B2_provenance_before','B2_provenance_after'):
  if f[field]!=actual:error(field,'B2 metadata hashes/provenance mismatch')
 if actual['files']!=35031:error('B2_provenance_before.files','must be 35031')
 if verify_leaves:
  for name,h in f['hashes'].items():
   if sha(ROOT/name)!=h:error('hashes.'+name,'execution bytes changed')
 if not FORMAL_ROOT.is_dir():error('formal_output_root','directory absent')
 if any(p.is_file() for p in FORMAL_ROOT.rglob('*')):error('formal_output_root','B3G_FORMAL_EXECUTION_VIOLATION: scientific root not empty')
 return dict(PASS=True,required_fields_validated=True,core_sha256=f['scientific_core_sha256'],wrapper_sha256=f['technical_wrapper_sha256'],execution_freeze_sha256=sha(FREEZE_PATH),B2_readonly_provenance_PASS=True,formal_seeds_unchanged=True,scientific_rows=0)
def require_authorization(pre_scientific_gate,freeze=None):
 f=freeze if freeze is not None else load(FREEZE_PATH)
 if pre_scientific_gate:
  if f.get('pre_scientific_gate_authorized') is not True:error('pre_scientific_gate_authorized','technical gate not authorized')
  return
 p=CONTROL/'formal_authorization.json'
 if not p.exists():raise PermissionError('NEXT_FORMAL_MENTOR_AUTHORIZATION_PENDING')
 a=load(p)
 if a.get('formal_run_authorized') is not True or a.get('execution_freeze_sha256')!=sha(FREEZE_PATH):raise PermissionError('NEXT_FORMAL_AUTHORIZATION_NOT_BOUND_TO_FREEZE')
def runtime_config(pre_scientific_gate,session):
 if not re.fullmatch('[A-Za-z0-9_-]+',session):error('session','invalid technical identifier')
 return dict(mode='formal',session=session,execution_freeze_path=str(FREEZE_PATH),pre_scientific_gate=bool(pre_scientific_gate))
def prepare_runtime_context(g,config):
 if config['execution_freeze_path']!=str(FREEZE_PATH):error('execution_freeze_path','unexpected contract')
 f=load(FREEZE_PATH);validate_execution_freeze(f,verify_leaves=False);require_authorization(config['pre_scientific_gate'],f)
 g['TECH']=ROOT/'outputs/sp_u12b3g/gates'/config['session'] if config['pre_scientific_gate'] else CONTROL
 g['SEAL']=FORMAL_ROOT;g['B2']=ROOT/'outputs/sp_u12b2b_sealed'
 return f
