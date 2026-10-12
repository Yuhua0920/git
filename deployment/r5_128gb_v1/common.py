"""R5-LAB deployment scope, immutable input verification and JSON bookkeeping."""
from pathlib import Path
import os,sys,json,time,hashlib,contextlib,importlib.metadata,platform
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'lab_runs/r5_128gb'
LOCAL=OUT/'preparation_personal'
ATOL=1e-10;RTOL=1e-9;GIB=2**30
BASE_MANIFEST_SHA='26316694e87b63ce198e508547e4802095c05139fb53403b0989287979288c72'
R4_SHA='3da613096e8389c2d30ecf72d37e32a371b1d320776da16de18cb95145b72762'
_opaque=False
sys.path.insert(0,str(ROOT))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def utc():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for attempt in range(101):
        try:tmp.replace(p);return
        except PermissionError as e:
            if getattr(e,'winerror',None)not in(5,32,33)or attempt==100:raise
            time.sleep(.01)
def sha(path):
    global _opaque
    old=_opaque;_opaque=True
    try:
        h=hashlib.sha256()
        with Path(path).open('rb')as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    finally:_opaque=old
def object_sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def protocol():return read(HERE/'protocol.json')

def scope():
    """Only R5 output writes; scientific numeric inputs are never opened."""
    allowed={ROOT/'outputs/sp_u12b5a/historical_reference.json'}
    runtime=Path(sys.prefix).resolve() # verified interpreter libraries, never scientific datasets
    def hook(event,args):
        if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
            writing=isinstance(mode,str)and any(v in mode for v in 'wax+')or isinstance(flags,int)and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if p.is_relative_to(ROOT):
                if writing and not p.is_relative_to(OUT):raise PermissionError('R5_HISTORICAL_WRITE_FORBIDDEN')
                if not writing and p.is_relative_to(ROOT/'outputs')and not p.is_relative_to(runtime)and not _opaque and p not in allowed:
                    raise PermissionError('R5_SCIENTIFIC_INPUT_FIREWALL')
        if event in('os.remove','os.rename','os.rmdir','os.mkdir','os.utime'):
            for raw in args[:2]if event=='os.rename'else args[:1]:
                if isinstance(raw,(str,bytes,os.PathLike)):
                    p=Path(os.fsdecode(raw)).resolve()
                    if p.is_relative_to(ROOT)and not p.is_relative_to(OUT):raise PermissionError('R5_HISTORICAL_MUTATION_FORBIDDEN')
    sys.addaudithook(hook)

def integrity(laboratory=True):
    manifest=read(HERE/'package_manifest.json');bad=[]
    for name,info in manifest['files'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(HERE)or not p.is_file()or sha(p)!=info['sha256']:bad.append(name)
    for name,h in manifest['frozen_source_hashes'].items():
        if sha(ROOT/name)!=h:bad.append(name)
    if laboratory:
        base=ROOT/'migration_manifest.json'
        if not base.is_file()or sha(base)!=BASE_MANIFEST_SHA:bad.append('migration_manifest.json')
        else:
            for name,info in read(base)['files'].items():
                p=(ROOT/name).resolve()
                if not p.is_relative_to(ROOT)or not p.is_file()or sha(p)!=info['sha256']:bad.append(name)
    assert sha(ROOT/'outputs/sp_u12b5b_r4_fidelity_manifest.json')==R4_SHA
    result={'PASS':not bad,'bad':bad,'package_manifest_SHA256':sha(HERE/'package_manifest.json'),
            'protocol_SHA256':sha(HERE/'protocol.json'),'base_manifest_SHA256':BASE_MANIFEST_SHA,
            'frozen_source_files':len(manifest['frozen_source_hashes']),'formal_training_authorized':False,
            'target_scientific_data_reads':0,'laboratory_integrity_checked':laboratory}
    if bad:raise RuntimeError('INTEGRITY_STOP:'+str(bad))
    return result

def environment():
    import numpy,scipy.linalg,sklearn.cluster # load every actual BLAS/OpenMP runtime
    spec=read(ROOT/'deployment/mfa_lab_v1/environment_source.json');versions={};bad=[]
    for row in spec['wheels']:
        try:value=importlib.metadata.version(row['package'])
        except importlib.metadata.PackageNotFoundError:value='MISSING'
        versions[row['package']]=value
        if value!=row['version']:bad.append(row['package'])
    from threadpoolctl import threadpool_info
    result={'PASS':sys.version_info[:3]==(3,13,15)and sys.maxsize>2**32 and not bad,
            'Python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'packages':versions,'mismatches':bad,'threadpools':threadpool_info(),'GPU_used':False}
    if not result['PASS']:raise RuntimeError('ENVIRONMENT_MISMATCH_STOP:'+str(result))
    return result

@contextlib.contextmanager
def threads(blas,openmp):
    import numpy,scipy.linalg,sklearn.cluster # control libraries before native work starts
    from threadpoolctl import threadpool_limits,threadpool_info
    with threadpool_limits(limits=blas,user_api='blas'),threadpool_limits(limits=openmp,user_api='openmp'):
        info=threadpool_info()
        assert any(v['user_api']=='blas'for v in info),'NO_BLAS_RUNTIME'
        assert any(v['user_api']=='openmp'for v in info),'NO_OPENMP_RUNTIME'
        assert all(v['num_threads']==(blas if v['user_api']=='blas'else openmp)for v in info if v['user_api']in('blas','openmp')),'ACTUAL_THREAD_CONFIGURATION_MISMATCH'
        yield info

def scientific_counters():
    return {'target_scientific_data_reads':0,'validation_scientific_score':0,'formal_candidates_trained':0,
            'selected_structure':'NONE','final_model':'NONE','formal_training_authorized':False}
