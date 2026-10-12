"""Independent Run3 candidate: exact scientific sources, one initializer fix."""
from pathlib import Path
import ast,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
CORE=('validate','evaluate','draw','bootstrap','classify');EXPECTED='994339ccc806e3ca94ae644a954e9382a2fa0df8ed2a808d0feb28528e9bdfa1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sources(p,names):
 text=p.read_text(encoding='utf8');lines=text.splitlines(keepends=True)
 return {n.name:''.join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in names}
def core():
 funcs=sources(ROOT/'scripts/sp_u12b3e_common.py',CORE);funcs.update(sources(ROOT/'scripts/run_sp_u12b3f.py',('compute',)))
 deps=['scripts/sp_u12b3a_ablation.py','src/pilot_cv.py','src/lowrank_modules.py','src/link.py','src/cdl_benchmark.py','src/frozen_affine_validation.py','scripts/sp_u12b2a_common.py']
 o=dict(function_source_hashes={n:hashlib.sha256(s.encode()).hexdigest() for n,s in funcs.items()},unchanged_scientific_dependencies={n:sha(ROOT/n) for n in deps},constants=dict(SNRS=[10,20],conditions=['ID','Profile','Mobility','Composed','Boundary'],arms=['P-only','M-only','Both'],rank=4,window=5,folds=4))
 return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest(),o
TOP_INIT='''
_initializer_original=None
_entry_config=None
def init_worker(config):
 """Importable spawn initializer; only primitive configuration is serialized."""
 global _entry_config
 g,_=_worker_context;_entry_config=dict(config)
 root=g['ROOT'];mode=config['mode'];session=config['session']
 if mode not in ('init-only','development-safe','formal') or not re.fullmatch(r'[A-Za-z0-9_-]+',session):raise ValueError('INVALID_ENTRY_CONFIGURATION')
 g['TECH']=root/'outputs/sp_u12b3e/sessions'/session if mode!='formal' else root/'outputs/sp_u12b3e/run3_control'
 g['SEAL']=root/'outputs/sp_u12b3e/development_runs'/session/'sealed' if mode=='development-safe' else root/'outputs/sp_u12b3f_run3_sealed'
 g['B2']=root/'outputs/sp_u12b3e/development_inputs' if mode=='development-safe' else root/'outputs/sp_u12b2b_sealed'
 try:
  if mode=='formal':
   from scripts.sp_u12b3e_authorization import require_formal_authorization
   require_formal_authorization()
  g['technical_context'](condition=None,trajectory_id=None,SNR=list(g['SNRS']),stage='INITIALIZER')
  parent=__import__('multiprocessing').parent_process()
  entered=dict(PID=os.getpid(),controller_PID=parent.pid if parent else os.getpid(),initializer_entered=True,initializer_completed=False,mode=mode,session=session)
  g['write'](g['TECH']/'initializers'/f'{os.getpid()}.json',entered)
  _initializer_original()
  # Opaque bytes only; no formal H, pilots, coordinates or outcomes are decoded.
  g['sha'](root/'outputs/sp_u12b2b_sealed/cache/ID/H_0999.npz')
  entered.update(initializer_completed=True,B2_readonly_guard_installed=g['_guard'],B2_readonly_opaque_access=True,technical_writer_initialized=True,Python_version=sys.version,sys_executable=sys.executable,start_method=__import__('multiprocessing').get_start_method(),BLAS_threads={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
  g['write'](g['TECH']/'initializers'/f'{os.getpid()}.json',entered)
  g['write'](g['TECH']/'actual_worker_pids'/f'{os.getpid()}.json',dict(PID=os.getpid(),parent_PID=os.getppid(),controller_PID=entered['controller_PID']))
 except Exception as exc:
  info=g['technical_exception_record'](exc)
  try:g['write'](g['TECH']/f'worker_exception_init_{os.getpid()}_{uuid.uuid4().hex}.json',info)
  except OSError:pass
  raise
def heartbeat():
 """Barrier proves both actual worker interpreters handled a submitted target."""
 g,_=_worker_context;g['write'](g['TECH']/'heartbeat_entered'/f'{os.getpid()}.json',dict(PID=os.getpid(),entered=True))
 tick=__import__('time').monotonic()
 while len(list((g['TECH']/'heartbeat_entered').glob('*.json')))<2:
  if __import__('time').monotonic()-tick>30:raise RuntimeError('TWO_WORKER_HEARTBEAT_TIMEOUT')
  __import__('time').sleep(.01)
 value=dict(PID=os.getpid(),heartbeat_returned=True,initializer_completed=True,mode=_entry_config['mode'])
 g['write'](g['TECH']/'heartbeats'/f'{os.getpid()}.json',value);return value
'''
def build():
 common=ROOT/'scripts/sp_u12b3e_common.py';io=ROOT/'scripts/sp_u12b3e_io.py'
 if common.exists() or io.exists():raise RuntimeError('CANDIDATE_ALREADY_EXISTS')
 s=(ROOT/'scripts/sp_u12b3d_common.py').read_text(encoding='utf8').replace("TECH=ROOT/'outputs/sp_u12b3d';SEAL=ROOT/'outputs/sp_u12b3d_run2_sealed';","TECH=ROOT/'outputs/sp_u12b3e/run3_control';SEAL=ROOT/'outputs/sp_u12b3f_run3_sealed';").replace('from scripts.sp_u12b3d_io import install_io','from scripts.sp_u12b3e_io import install_io')
 common.write_text(s,encoding='utf8')
 s=(ROOT/'scripts/sp_u12b3d_io.py').read_text(encoding='utf8').replace('Run2 I/O operations','Run3 candidate I/O operations').replace('sp_u12b3d_run2_','sp_u12b3e_').replace('sp_u12b3d_','sp_u12b3e_')
 s=s.replace('_worker_context=None','_worker_context=None\n'+TOP_INIT)
 s=s.replace(' g,original=_worker_context\n'," g,original=_worker_context\n if _entry_config is None:raise RuntimeError('INITIALIZER_NOT_ENTERED')\n if _entry_config['mode']=='init-only':\n  if condition!='__HEARTBEAT__':raise PermissionError('INIT_ONLY_SCIENTIFIC_WORK_FORBIDDEN')\n  return heartbeat()\n")
 start=s.index(" original_init=g['init_worker']");end=s.index(" _worker_context=(g,g['worker'])",start)
 s=s[:start]+" _initializer_original=g['init_worker']\n"+s[end:]
 s=s.replace(' global _worker_context\n',' global _worker_context,_initializer_original\n')
 # New namespace and safe fixtures only; original writer/guard algorithms are retained.
 s=s.replace("if p.is_relative_to(metadata) or p.parent==root/'outputs' and p.name.startswith('sp_u12b3e_'):return", "if p.is_relative_to(metadata) or p.is_relative_to(root/'outputs/sp_u12b3e') or p.parent==root/'outputs' and p.name.startswith('sp_u12b3e_'):return")
 s=s.replace("target.is_relative_to(metadata) or ownedpublic", "target.is_relative_to(metadata) or target.is_relative_to(root/'outputs/sp_u12b3e') or ownedpublic")
 s=s.replace("rel=p.relative_to(b2root).parts\n", "rel=p.relative_to(b2root).parts\n     if _entry_config and _entry_config['mode']=='development-safe' and b2root==real_b2:deny(p,op,'DEVELOPMENT_FORMAL_INPUT_DECODE_FORBIDDEN')\n")
 io.write_text(s,encoding='utf8')
 runner=ROOT/'scripts/run_sp_u12b3f.py'
 with runner.open('a',encoding='utf8') as f:f.write('\n'+sources(ROOT/'scripts/run_sp_u12b3d.py',('compute',))['compute']+"\nif __name__=='__main__':main()\n")
 complete_formal()
 digest,partition=core();assert digest==EXPECTED
 assert sources(common,CORE)==sources(ROOT/'scripts/sp_u12b3d_common.py',CORE)
 print(json.dumps(dict(stage='candidate_build',PASS=True,scientific_core_sha256=digest)))
def complete_formal():
 runner=ROOT/'scripts/run_sp_u12b3f.py';s=runner.read_text(encoding='utf8')
 names=('write_report','check_core','evaluation','integrity','seal','run','finalize')
 copied='\n'.join(sources(ROOT/'scripts/run_sp_u12b3d.py',names).values())
 copied=copied.replace("concurrent.futures.ProcessPoolExecutor(max_workers=2,initializer=b.init_worker)","create_pool('formal','formal')").replace('sp_u12b3d_run2','sp_u12b3f_run3').replace('sp_u12b3d_authorization','sp_u12b3e_authorization').replace('sp_u12b3d_historical_failure_disclosure','sp_u12b3e_spawn_failure_analysis')
 s=s.replace('from scripts import sp_u12b3e_common as b','from scripts import sp_u12b3e_common as b\nfrom scripts.sp_u12b3e_build import core,EXPECTED')
 s=s.replace("if __name__=='__main__':main()",copied+"\nif __name__=='__main__':main()")
 runner.write_text(s,encoding='utf8')
 io=ROOT/'scripts/sp_u12b3e_io.py';s=io.read_text(encoding='utf8')
 s=s.replace("p.name.replace('sp_u12b3b_','sp_u12b3e_',1)","p.name.replace('sp_u12b3b_','sp_u12b3f_run3_',1)")
 s=s.replace("target.name.startswith('sp_u12b3e_')","target.name.startswith(('sp_u12b3e_','sp_u12b3f_'))").replace("p.name.startswith('sp_u12b3e_')","p.name.startswith(('sp_u12b3e_','sp_u12b3f_'))")
 io.write_text(s,encoding='utf8')
if __name__=='__main__':
 import sys
 if len(sys.argv)>1 and sys.argv[1]=='complete':complete_formal()
 else:build()
