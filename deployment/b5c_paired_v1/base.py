"""B5C bookkeeping in a new namespace; no fit/selection entry points."""
from pathlib import Path
import hashlib, json, os, sys, time
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'lab_runs/b5c_paired'
DATA = ROOT/'lab_inputs/b5c_b2_v1'
FINAL = ROOT/'lab_runs/r6_formal/final'
def require(value, message):
    if not value: raise RuntimeError(message)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
def write(path, value, exclusive=False):
    path=Path(path); require(path.resolve().is_relative_to(OUT.resolve()),'NEW_OUTPUT_NAMESPACE_ONLY')
    path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if exclusive:
        with path.open('x',encoding='utf8') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    else:
        q=path.with_name(path.name+'.tmp'); q.write_text(data,encoding='utf8')
        for i in range(101):
            try: q.replace(path); break
            except PermissionError as e:
                if getattr(e,'winerror',None) not in (5,32,33) or i==100: raise
                time.sleep(.01) # Status publication only, never retry evaluation.
def contract(): return read(HERE/'contract.json')
def safe(base, name):
    p=(base/name).resolve()
    require(p.is_relative_to(base.resolve()) and '..' not in Path(name).parts,'UNSAFE_RELATIVE_PATH')
    return p
def verify():
    manifest=read(HERE/'package_manifest.json')
    for n,v in manifest['files'].items():
        p=safe(ROOT,n); require(p.is_file() and p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],'CODE_HASH_DRIFT:'+n)
    c=contract()
    require(c['statistics']['replicates']==100000 and c['statistics']['family_size']==35,'STATISTICAL_CONTRACT_DRIFT')
    require(c['N']==1000 and c['score_frames']==list(range(4,10)) and len(c['conditions'])==5 and len(c['snrs'])==7,'COVERAGE_DRIFT')
    return {'PASS':True,'package_manifest_sha256':sha(HERE/'package_manifest.json'),'real_evaluation_executed':False}
def data_inventory(): return read(HERE/'data_manifest.json')['files']
def verify_data():
    # Opaque byte authentication only. No formal outcome decoding in preflight.
    files=data_inventory()
    require(len(files)==30008,'EXACT_MINIMUM_B2_INVENTORY_REQUIRED')
    for n,v in files.items():
        p=safe(DATA,n); require(p.is_file() and p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],'B2_TRANSFER_HASH_MISMATCH:'+n)
    return {'PASS':True,'files':len(files),'bytes':sum(v['bytes'] for v in files.values()),'decoded_outcomes':False}
