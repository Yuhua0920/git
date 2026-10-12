"""Independent recovery namespace; original files are read-only."""
from pathlib import Path
import hashlib,json,os,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'lab_runs/b5c_paired_r2';SOURCE=OLD/'formal'
OUT=ROOT/'lab_runs/b5c_paired_r2_recovery'
def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(p):
    def pairs(items):
        d={}
        for k,v in items:require(k not in d,'DUPLICATE_JSON_FIELD:'+k);d[k]=v
        return d
    return json.loads(Path(p).read_text(encoding='utf-8-sig'),object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in ()).throw(RuntimeError('NONFINITE_JSON_CONSTANT')))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def utc():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()
def plain(p,scope=None):
    scope=ROOT if scope is None else scope
    p=Path(p);require(p.resolve().is_relative_to(scope.resolve()),'PATH_OUTSIDE_SCOPE:'+str(p))
    q=p
    while True:
        if q.exists():require(not(q.lstat().st_file_attributes&0x400),'REPARSE_PATH_REJECTED:'+str(q))
        if q==scope:break
        require(q.parent!=q,'INVALID_SCOPE');q=q.parent
    return p
def write(p,value):
    plain(OUT,ROOT)
    p=plain(p,OUT);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def status(job,state,**more):
    plain(OUT,ROOT)
    p=plain(job/'status.json',OUT);data={'state':state,'UTC':utc(),**more}
    temp=p.with_name('status.json.tmp')
    with temp.open('w',encoding='utf8') as f:f.write(json.dumps(data,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    temp.replace(p)
def bindings():return read(HERE/'bindings.json')
def verify():
    for rel,v in read(HERE/'package_manifest.json')['files'].items():
        p=plain(ROOT/rel);require(p.is_file() and p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],'RECOVERY_CODE_HASH_DRIFT:'+rel)
    for rel,v in bindings()['frozen_files'].items():
        p=plain(ROOT/rel);require(p.is_file() and p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],'FROZEN_SOURCE_DRIFT:'+rel)
    return {'PASS':True,'new_inference_calls':0,'old_status_rewritten':False}
def snapshot(folder):
    records={}
    def walk(p):
        plain(p,folder)
        for q in sorted(p.iterdir()):
            plain(q,folder)
            if q.is_dir():walk(q)
            elif q.is_file():
                s=q.stat();records[q.relative_to(folder).as_posix()]={'bytes':s.st_size,'sha256':sha(q),'mtime_ns':s.st_mtime_ns}
            else:raise RuntimeError('SPECIAL_SOURCE_FILE:'+str(q))
    require(folder.is_dir(),'LAB_ORIGINAL_RESULTS_MISSING');walk(folder);return records
def unchanged(records):require(snapshot(OLD)==records,'ORIGINAL_FILES_CHANGED_AFTER_SNAPSHOT')
def firewall(exact):
    plain(OUT,ROOT)
    exact={plain(p).resolve() for p in exact};runtimes={Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()}
    def check(raw,writing=False):
        p=Path(os.fsdecode(raw)).resolve()
        if writing:
            require(p.is_relative_to(OUT.resolve()),'RECOVERY_WRITE_DENIED:'+str(p))
            plain(Path(os.path.abspath(os.fsdecode(raw))),OUT)
        else:require(p in exact or p.is_relative_to(OUT.resolve()) or any(p.is_relative_to(q) for q in runtimes),'RECOVERY_READ_DENIED:'+str(p))
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            if os.fsdecode(args[0]).upper() in ('NUL',r'\\.\NUL'):return
            mode,flags=args[1:3];writing=isinstance(mode,str) and any(v in mode for v in 'wax+') or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            check(args[0],writing)
        elif event in ('os.remove','os.rename','os.rmdir','os.mkdir','os.utime','os.link'):
            for p in args[:2] if event in ('os.rename','os.link') else args[:1]:
                if isinstance(p,(str,bytes,os.PathLike)):check(p,True)
        elif event in ('subprocess.Popen','os.system','os.exec'):raise RuntimeError('RECOVERY_CHILD_EXECUTION_DENIED')
    sys.addaudithook(hook)
def frozen_modules():
    import importlib.util
    loaderpath=ROOT/'deployment/r7_evidence_audit_r1_v1/source_only_loader.py'
    spec=importlib.util.spec_from_file_location('_recovery_authenticated_loader',loaderpath)
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    sources=dict(bindings()['frozen_files']);sources.update(read(HERE/'package_manifest.json')['files']);loader.install(ROOT,sources)
    sys.path.insert(0,str(ROOT/'deployment/b5c_paired_r2_v1'));sys.path.insert(1,str(ROOT))
    import base,science,paired_stats,report,historical_concurrency
    return base,science,paired_stats,report,historical_concurrency
