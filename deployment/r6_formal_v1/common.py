"""R6 deployment bookkeeping. No scientific orchestrator imports."""
from pathlib import Path
import os, sys, json, time, hashlib, contextlib, platform, importlib.metadata
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'lab_runs/r6_formal'
LOCAL = OUT / 'preparation_personal'
GIB = 2**30
sys.path.insert(0, str(ROOT))
_opaque = False
_numeric = set()
_scope_installed = False

def require(value, message):
    if not value: raise RuntimeError(message)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def write(path, value, exclusive=False):
    path = Path(path)
    require(path.resolve().is_relative_to(OUT.resolve()), 'R6_OUTPUT_ONLY')
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x', encoding='utf8') as f: f.write(text)
        return
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(text, encoding='utf8')
    for attempt in range(101):
        try: temp.replace(path); return
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5,32,33) or attempt == 100: raise
            time.sleep(.01)

def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()

def sha(path):
    """Opaque bytes only, and never an arbitrary scientific-path exemption."""
    global _opaque
    path = Path(path).resolve()
    permitted = path.is_relative_to(HERE) or path.is_relative_to(OUT) or path == ROOT/'migration_manifest.json'
    if (HERE/'input_contract.json').exists():
        contract=read(HERE/'input_contract.json')
        permitted |= path in {(ROOT/n).resolve() for n in set(contract['frozen_hashes'])|set(contract['lab_prerequisite_hashes'])}
    require(permitted, 'HASH_PATH_OUTSIDE_R6_ALLOWLIST')
    old = _opaque; _opaque = True
    try:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
        return h.hexdigest()
    finally: _opaque = old

def protocol(): return read(HERE/'resource_amendment.json')
def science(): return read(ROOT/'configs/sp_u12b5b_mfa_selection.yaml')
def registry(): return read(ROOT/'outputs/sp_u12b5b_candidate_seed_registry.json')['candidates']
def candidate(name):
    rows = [v for v in registry() if v['name'] == name]
    require(len(rows)==1, 'UNKNOWN_CANDIDATE'); return rows[0]

def npy_header(path):
    global _opaque
    path=Path(path).resolve()
    require(path in {(ROOT/n).resolve() for n in read(HERE/'input_contract.json')['data']},'HEADER_PATH_NOT_FROZEN')
    import numpy as np
    old=_opaque;_opaque=True
    try:
        with path.open('rb') as f:
            version=np.lib.format.read_magic(f)
            require(version in ((1,0),(2,0)), 'UNSUPPORTED_FROZEN_NPY_VERSION')
            shape,order,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f))
            offset=f.tell()
        require(not dtype.hasobject and path.stat().st_size==offset+int(np.prod(shape))*dtype.itemsize,'NPY_FILE_LENGTH_INVALID')
        return {'shape':list(shape),'dtype':dtype.str,'fortran_order':order,'header_bytes':offset}
    finally:_opaque=old

def scope():
    global _scope_installed
    if _scope_installed: return
    _scope_installed = True
    contract = read(HERE/'input_contract.json')
    metadata = {(ROOT/n).resolve() for n in contract['metadata_allowlist']+list(contract['lab_prerequisite_hashes'])}
    data = {(ROOT/n).resolve() for n in contract['data']}
    codes = {(ROOT/n).resolve() for n in contract['frozen_hashes'] if Path(n).suffix in ('.py','.ps1')}
    runtimes = {Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()}
    model_names={n+'.npy' for n in ('means','lambdas','psis','amps','covs','inv_covs','training_history')}
    model_paths={(OUT/'candidates'/v['name']/'train/model'/n).resolve() for v in contract['candidate_order'] for n in model_names}
    model_paths|={(OUT/'final/model'/n).resolve() for n in model_names}
    numerical_formats={'.npy','.npz','.pkl','.pickle','.h5','.hdf5','.mat'}
    def protected(path, writing=False):
        path = Path(path).resolve()
        if writing:
            require(path.is_relative_to(OUT), 'R6_HISTORICAL_OR_EXTERNAL_WRITE_FORBIDDEN'); return
        if path.is_relative_to(LOCAL):return # bounded synthetic fixtures only
        if path.is_relative_to(OUT):
            require(path.suffix.lower() not in numerical_formats or path in model_paths,'R6_UNAPPROVED_OUTPUT_NUMERICAL_READ')
            return
        if any(path.is_relative_to(p) for p in runtimes):
            require(path.suffix.lower() not in numerical_formats,'R6_RUNTIME_SCIENTIFIC_DATA_READ_FORBIDDEN')
            return
        if path.is_relative_to(HERE):
            require(path.suffix.lower() not in numerical_formats,'R6_DEPLOYMENT_SCIENTIFIC_DATA_READ_FORBIDDEN')
            return
        if path in metadata or path in codes or path == ROOT/'migration_manifest.json': return
        if path in data and (_opaque or path in _numeric): return
        # Importing source packages can legitimately read their __init__.py only
        # if that exact file was frozen by the original migration manifest.
        if path.is_relative_to(ROOT): raise PermissionError('R6_SCIENTIFIC_READ_ALLOWLIST_REJECTED:'+str(path))
        raise PermissionError('R6_EXTERNAL_READ_ALLOWLIST_REJECTED:'+str(path))
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            # Windows NUL is a device, not a mutable historical file. Python
            # platform queries legitimately open os.devnull with O_RDWR.
            if os.fsdecode(args[0]).upper() in ('NUL',r'\\.\NUL'):return
            path=Path(os.fsdecode(args[0])); mode,flags=args[1:3]
            writing = isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            protected(path,writing)
        if event in ('os.remove','os.rename','os.rmdir','os.mkdir','os.utime'):
            for raw in args[:2] if event=='os.rename' else args[:1]:
                if isinstance(raw,(str,bytes,os.PathLike)): protected(os.fsdecode(raw), True)
    sys.addaudithook(hook)

@contextlib.contextmanager
def numeric_inputs(action):
    require(action in ('train','validate','final'), 'INVALID_NUMERICAL_ROLE')
    approval()
    names = {'train':['selection_train.npy'], 'final':['full_retraining.npy'],
             'validate':['validation_truth.npy','validation_pilots.npy']}[action]
    paths = {ROOT/'outputs/sp_u12b5b'/name for name in names}
    require(not _numeric, 'NESTED_NUMERICAL_ROLE_FORBIDDEN')
    _numeric.update(paths)
    try: yield
    finally: _numeric.clear()

def integrity(laboratory=True):
    manifest = read(HERE/'package_manifest.json'); contract=read(HERE/'input_contract.json'); bad=[]
    for name,row in manifest['files'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(HERE) or not p.is_file() or sha(p)!=row['sha256']: bad.append(name)
    for name,h in contract['frozen_hashes'].items():
        if not (ROOT/name).is_file() or sha(ROOT/name)!=h: bad.append(name)
    if laboratory:
        if not (ROOT/'migration_manifest.json').exists() or sha(ROOT/'migration_manifest.json')!=contract['base_manifest_sha256']:
            bad.append('migration_manifest.json')
        for name,h in contract['lab_prerequisite_hashes'].items():
            if not (ROOT/name).is_file() or sha(ROOT/name)!=h:bad.append(name)
    require(not bad, 'R6_INTEGRITY_STOP:'+str(bad))
    return {'PASS':True,'package_manifest_SHA256':sha(HERE/'package_manifest.json'),
            'input_contract_SHA256':sha(HERE/'input_contract.json'),'resource_amendment_SHA256':sha(HERE/'resource_amendment.json'),
            'formal_authorization':'MENTOR_APPROVAL_PENDING' if not (OUT/'MENTOR_APPROVAL.json').exists() else 'ATTESTATION_PRESENT_NOT_YET_VALIDATED',
            'target_access':0}

def approval():
    require(protocol()['status']=='MENTOR_APPROVAL_PENDING', 'PROPOSAL_STATUS_DRIFT')
    path=OUT/'MENTOR_APPROVAL.json'
    require(path.exists(), 'MENTOR_APPROVAL_PENDING: FORMAL EXECUTION DISABLED')
    a=read(path)
    expected={'package_manifest_SHA256':sha(HERE/'package_manifest.json'),
              'resource_amendment_SHA256':sha(HERE/'resource_amendment.json'),
              'input_contract_SHA256':sha(HERE/'input_contract.json')}
    require(a.get('status')=='MENTOR_APPROVED' and bool(a.get('mentor_instruction_reference')) and bool(a.get('approved_UTC')),
            'MENTOR_APPROVAL_ATTESTATION_INVALID')
    require(all(a.get(k)==h for k,h in expected.items()), 'MENTOR_APPROVAL_HASH_MISMATCH')
    require(not (OUT/'STOP.json').exists() and not (OUT/'STOP_REQUESTED.json').exists(), 'PIPELINE_STOPPED_NO_RETRY')
    # A signed/frozen human instruction is required; this program neither
    # creates nor upgrades attestations. It cannot authenticate a human signer.
    return a

def environment():
    import numpy, scipy.linalg, sklearn.cluster
    spec=read(ROOT/'deployment/mfa_lab_v1/environment_source.json'); versions={}; bad=[]
    for row in spec['wheels']:
        try: actual=importlib.metadata.version(row['package'])
        except importlib.metadata.PackageNotFoundError: actual='MISSING'
        versions[row['package']]=actual
        if actual!=row['version']: bad.append(row['package'])
    require(sys.version_info[:3]==(3,13,15) and sys.maxsize>2**32 and not bad,'PINNED_ENVIRONMENT_MISMATCH:'+str(bad))
    from threadpoolctl import threadpool_info
    return {'PASS':True,'Python':sys.version,'executable':sys.executable,'packages':versions,
            'platform':platform.platform(),'threadpools':threadpool_info(),'GPU_used':False}

@contextlib.contextmanager
def threads(blas,openmp):
    import numpy, scipy.linalg, sklearn.cluster
    from threadpoolctl import threadpool_limits,threadpool_info
    with threadpool_limits(limits=blas,user_api='blas'),threadpool_limits(limits=openmp,user_api='openmp'):
        pools=threadpool_info()
        require(any(v['user_api']=='blas' for v in pools) and any(v['user_api']=='openmp' for v in pools), 'MISSING_THREAD_RUNTIME')
        require(all(v['num_threads']==(blas if v['user_api']=='blas' else openmp) for v in pools if v['user_api'] in ('blas','openmp')), 'THREAD_LIMIT_MISMATCH')
        yield pools

def scientific_counters():
    return {'target_scientific_data_reads':0,'automatic_retry':False}
