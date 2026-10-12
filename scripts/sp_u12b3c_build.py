"""Create isolated candidate wrappers; original failed bytes never change."""
from pathlib import Path
import ast,json,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sp_u12b3c'
CORE=('validate','evaluate','draw','bootstrap','classify')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def functions(p,names):
 source=p.read_text(encoding='utf8');lines=source.splitlines(keepends=True);tree=ast.parse(source)
 return {n.name:hashlib.sha256(''.join(lines[n.lineno-1:n.end_lineno]).encode()).hexdigest() for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
def save(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2)+'\n',encoding='utf8')
def run():
 if not(ROOT/'outputs/sp_u12b3c_failed_run1_archive.json').exists():raise RuntimeError('ARCHIVE_REQUIRED')
 old=ROOT/'scripts/sp_u12b3b_common.py';new=ROOT/'scripts/sp_u12b3c_run2_common.py'
 s=old.read_text(encoding='utf8').replace("TECH=ROOT/'outputs/sp_u12b3b';SEAL=ROOT/'outputs/sp_u12b3b_sealed';", "TECH=ROOT/'outputs/sp_u12b3c_run2';SEAL=ROOT/'outputs/sp_u12b3b_sealed_run2';")
 s+='\n# Technical overrides only; all scientific definitions above retain exact source.\nfrom scripts.sp_u12b3c_io import install_io\ninstall_io(globals())\n'
 new.write_text(s,encoding='utf8')
 runner=ROOT/'scripts/sp_u12b3c_run2.py';s=(ROOT/'scripts/run_sp_u12b3b.py').read_text(encoding='utf8').replace('from scripts import sp_u12b3b_common as b','from scripts import sp_u12b3c_run2_common as b')
 s=s.replace("if __name__=='__main__':main()", "if __name__=='__main__':\n from scripts.sp_u12b3c_authorization import require_formal_authorization\n require_formal_authorization(sys.argv[1])\n main()")
 s=s.replace(' except Exception as exc:\n  # No arbitrary message', " except Exception as exc:\n  try:b.write(b.TECH/f'parent_exception_{os.getpid()}.json',b.technical_exception_record(exc))\n  except OSError:pass\n  # No arbitrary message")
 runner.write_text(s,encoding='utf8')
 before=functions(old,CORE);after=functions(new,CORE);before['compute']=functions(ROOT/'scripts/run_sp_u12b3b.py',('compute',))['compute'];after['compute']=functions(runner,('compute',))['compute']
 same=before==after
 files=['scripts/sp_u12b3a_ablation.py','src/pilot_cv.py','src/lowrank_modules.py','src/link.py','src/cdl_benchmark.py','src/frozen_affine_validation.py','scripts/sp_u12b2a_common.py']
 protected={p:sha(ROOT/p) for p in files};core=dict(function_source_hashes=before,unchanged_scientific_dependencies=protected,constants=dict(SNRS=[10,20],conditions=['ID','Profile','Mobility','Composed','Boundary'],arms=['P-only','M-only','Both'],rank=4,window=5,folds=4))
 digest=hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 audit=dict(PASS=same,scientific_core_modified=False,scientific_core_sha256_before=digest,scientific_core_sha256_after=digest if same else None,before_function_source_hashes=before,after_function_source_hashes=after,dependencies=protected,partition='exact function source bytes for fitting/application/link via whole dependencies;evaluate/validate metrics;draw/bootstrap/classify;compute family/aggregation;I/O helper imports and output routing are technical',not_a_proven_root_cause_repair=True)
 save(ROOT/'outputs/sp_u12b3c_scientific_hash_audit.json',audit)
 save(ROOT/'outputs/sp_u12b3c_io_patch_audit.json',dict(status='CANDIDATE_OBSERVABILITY_AND_ISOLATION_HARDENING_ONLY',historical_root_cause_confirmed=False,scientific_changes=0,failed_source_files_modified=0,changes=['fresh independent output/control roots','per-process/thread/UUID exclusive temp names','read-only guard for open/mkdir/rename/replace/delete/touch/link/permission mutation','thread-local operation records and full PermissionError traceback','old public bookkeeping paths routed to run2 control root','explicit separate future mentor authorization required'],new_wrapper_files=[new.relative_to(ROOT).as_posix(),runner.relative_to(ROOT).as_posix(),'scripts/sp_u12b3c_io.py','scripts/sp_u12b3c_authorization.py']))
 tech=ROOT/'outputs/sp_u12b3c_run2';tech.mkdir(exist_ok=True)
 for name in ('bootstrap_seed_registry.json','raw_schema.json'):(tech/name).write_bytes((ROOT/'outputs/sp_u12b3b'/name).read_bytes())
 print(json.dumps(dict(stage='candidate_build',scientific_hash_unchanged=same,core_sha256=digest)))
 if not same:raise RuntimeError('B3C_SCIENTIFIC_CODE_CHANGE_REQUIRED')
if __name__=='__main__':run()
