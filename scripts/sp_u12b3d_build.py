"""Isolated authorized B3 Run2 technical wrappers; never rewrite historical files."""
from pathlib import Path
import ast,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
CORE=('validate','evaluate','draw','bootstrap','classify')
EXPECTED='994339ccc806e3ca94ae644a954e9382a2fa0df8ed2a808d0feb28528e9bdfa1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def function_sources(p,names):
 s=p.read_text(encoding='utf8');lines=s.splitlines(keepends=True)
 return {n.name:''.join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in names}
def core():
 funcs=function_sources(ROOT/'scripts/sp_u12b3d_common.py',CORE)
 funcs.update(function_sources(ROOT/'scripts/run_sp_u12b3d.py',('compute',)))
 deps=['scripts/sp_u12b3a_ablation.py','src/pilot_cv.py','src/lowrank_modules.py','src/link.py','src/cdl_benchmark.py','src/frozen_affine_validation.py','scripts/sp_u12b2a_common.py']
 o=dict(function_source_hashes={n:hashlib.sha256(s.encode()).hexdigest() for n,s in funcs.items()},unchanged_scientific_dependencies={n:sha(ROOT/n) for n in deps},constants=dict(SNRS=[10,20],conditions=['ID','Profile','Mobility','Composed','Boundary'],arms=['P-only','M-only','Both'],rank=4,window=5,folds=4))
 return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest(),o
def build():
 dest=ROOT/'scripts/sp_u12b3d_common.py'
 if dest.exists():raise RuntimeError('NEW_WRAPPER_ALREADY_EXISTS')
 s=(ROOT/'scripts/sp_u12b3b_common.py').read_text(encoding='utf8').replace("TECH=ROOT/'outputs/sp_u12b3b';SEAL=ROOT/'outputs/sp_u12b3b_sealed';","TECH=ROOT/'outputs/sp_u12b3d';SEAL=ROOT/'outputs/sp_u12b3d_run2_sealed';")
 s+='\n# Authorized technical overrides; preceding scientific function sources are unchanged.\nfrom scripts.sp_u12b3d_io import install_io\ninstall_io(globals())\n'
 dest.write_text(s,encoding='utf8')
 r=ROOT/'scripts/run_sp_u12b3d.py'
 with r.open('a',encoding='utf8') as f:f.write('\n'+function_sources(ROOT/'scripts/run_sp_u12b3b.py',('compute',))['compute']+'\nif __name__==\'__main__\':main()\n')
 old=function_sources(ROOT/'scripts/sp_u12b3b_common.py',CORE);new=function_sources(dest,CORE)
 assert old==new
 digest,o=core();assert digest==EXPECTED
 print(json.dumps(dict(stage='build',PASS=True,hash=digest)))
if __name__=='__main__':build()
