"""R1 provenance support: immutable call transactions, reconciliation and quarantine."""
from __future__ import annotations
from scripts import sp_u7b_common as original
import os,sys,json,csv,hashlib,time,contextlib,zipfile,inspect
from pathlib import Path
import numpy as np

ROOT=original.ROOT
OUT=ROOT/'outputs/sp_u7b_r1'
R1_ROOT=OUT  # permanent firewall boundary, independent of synthetic test OUT
OLD=ROOT/'outputs/sp_u7b'
PROTOCOL=original.PROTOCOL
SEEDS=OUT/'seed_manifest.csv'
NAMESPACE='SP_U7B_R1_OOD_CDLA120_V2'
RUN_ID='SP-U7B-R1-CDLA120-V2'
METHODS=original.METHODS;SNRS=original.SNRS;VARIANCES=original.VARIANCES;ROLES=original.ROLES
NEW_SOURCES=('scripts/sp_u7b_r1_common.py','scripts/run_sp_u7b_r1.py','scripts/analyze_sp_u7b_r1.py','tests/test_sp_u7b_r1.py',
             'scripts/sp_u7b_common.py','scripts/run_sp_u7b.py','scripts/analyze_sp_u7b.py','scripts/plot_sp_u7b.py','tests/test_sp_u7b.py')
utc=original.utc;resources=original.resources;load_deployables=original.load_deployables
infer_pilots=original.infer_pilots;communication=original.communication;evaluate=original.evaluate
_HASH_ONLY=False
_METADATA_ONLY=False
_GUARD_INSTALLED=False
_DENIALS=[]
_LEDGER_SNAPSHOTS={}  # verified immutable witness records, never integer counters

def sha(path):
    global _HASH_ONLY
    prior=_HASH_ONLY;_HASH_ONLY=True
    try:return original.sha(path)
    finally:_HASH_ONLY=prior

def read(path):return original.read(path)

def write(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):raise PermissionError('R1 writes confined to R1 output root')
    if isinstance(value,dict):value=dict(run_id=RUN_ID,**{k:v for k,v in value.items() if k!='run_id'})
    with path.open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def state_path():return OUT/'state.json'

def replace_state(value):
    value=dict(value,run_id=RUN_ID)
    temp=OUT/'state.pending'
    with temp.open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,state_path())

def verify(hashes):
    bad=[p for p,h in hashes.items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
    if bad:raise RuntimeError('frozen byte drift: '+repr(bad))

def verify_frozen():
    f=read(OUT/'freeze.json');verify(f['hashes'])
    if not f['R1_FREEZE'] or f['run_id']!=RUN_ID:raise RuntimeError('R1 freeze identity')
    return f

def old_tree():return {p.relative_to(OLD).as_posix():sha(p) for p in OLD.rglob('*') if p.is_file()}

def quarantine_check():
    q=read(OUT/'sp_u7b_invalidated_quarantine.json')
    if old_tree()!=q['old_tree']:raise RuntimeError('old invalidated tree drift')
    s=read(OLD/'state.json')
    if s['phase']!='INVALIDATED' or s['UNSEAL'] or (OLD/'unseal.json').exists():raise RuntimeError('permanent quarantine violated')
    return dict(old_tree_unchanged=True,old_performance_payload_reads=0,denied_old_access_attempts=len(_DENIALS),
                quarantine_sha256=sha(OUT/'sp_u7b_invalidated_quarantine.json'))

def fingerprint_quarantined(path,identities,hashes,sizes):
    path=Path(path)
    if not path.exists():return False
    st=path.stat()
    return (st.st_dev,st.st_ino) in identities or (st.st_size in sizes and sha(path) in hashes)

def guard(mode='generation'):
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:return
    q=read(OUT/'sp_u7b_invalidated_quarantine.json')
    badhash={h for p,h in q['old_tree'].items() if p.startswith(('evaluation/','matched_reference/')) and p.endswith(('.npz','.npy'))}
    badids={tuple(v) for p,v in q['old_performance_file_identities'].items() if p.startswith(('evaluation/','matched_reference/'))}
    badsizes={(OLD/p).stat().st_size for p in q['old_tree'] if p.startswith(('evaluation/','matched_reference/')) and p.endswith(('.npz','.npy'))}
    allowed={str((ROOT/p).resolve()).lower() for p in read(PROTOCOL)['deployable']['objects']}
    outputs=(ROOT/'outputs').resolve();old=OLD.resolve();r1=R1_ROOT.resolve()
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();omode=args[1];flags=args[2]
        writing=(isinstance(omode,str) and any(x in omode for x in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR))
        if writing:
            if p.is_relative_to(old):raise PermissionError('permanent old-run write quarantine')
            return
        if _HASH_ONLY:return  # opaque hashing only; no decoding path inside sha()
        deny=(p.is_relative_to(old) and p.suffix.lower() in ('.npz','.npy')) or fingerprint_quarantined(p,badids,badhash,badsizes)
        if deny:
            _DENIALS.append(str(p));raise PermissionError('invalidated-run raw quarantine: original/renamed/copied payload rejected')
        if p.is_relative_to(outputs) and not p.is_relative_to(r1) and not p.is_relative_to(old):
            if str(p).lower() in allowed or p.name.startswith('sp_u7a_') or _METADATA_ONLY:return
            raise PermissionError('historical outcome firewall: '+p.name)
    sys.addaudithook(hook);_GUARD_INSTALLED=True

def seed_rows():
    with SEEDS.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))

def seed_index():return {(r['group'],int(r['trajectory_id']),int(r['frame_id']),r['role']):int(r['seed']) for r in seed_rows()}

consumed=original.consumed

def seed_key_order():
    for i in range(1000):
        for role in ROLES[:3]:yield 'EVAL',i,-1,role
        for j in range(10):
            for role in ROLES[3:]:yield 'EVAL',i,j,role
    for i in range(500):
        for role in ROLES[:3]:yield 'MATCHED_REF',i,-1,role

def derive(group,i,j,role,counter):
    msg=f'{NAMESPACE}|{group}|{i}|{j}|{role}|{counter}'
    return int.from_bytes(hashlib.sha256(msg.encode('ascii')).digest()[:4],'big')

def old_seed_values():
    with (ROOT/'outputs/sp_u7a_future_seed_manifest.csv').open(encoding='utf8',newline='') as f:
        return {int(r['seed']) for r in csv.DictReader(f) if r['role']!='bootstrap_seed'}

def allocate_seeds():
    forbidden=set(read(ROOT/'outputs/sp_u7a_historical_seed_union.json')['forbidden_seeds'])|old_seed_values()|{764389531}
    used=set();rows=[];rejected=[]
    for group,i,j,role in seed_key_order():
        counter=0
        while True:
            value=derive(group,i,j,role,counter)
            if value not in forbidden and value not in used:break
            rejected.append(dict(group=group,trajectory_id=i,frame_id=j,role=role,counter=counter,seed=value));counter+=1
        rows.append(dict(namespace=NAMESPACE,group=group,trajectory_id=i,frame_id=j,role=role,counter=counter,seed=value));used.add(value)
    rows.append(dict(namespace=NAMESPACE,group='BOOTSTRAP',trajectory_id=-1,frame_id=-1,role='bootstrap_seed',counter=0,seed=764389531))
    with SEEDS.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    write(OUT/'allocation_audit.json',dict(namespace=NAMESPACE,allocation_counts=dict(EVAL_channel=3000,EVAL_communication=30000,MATCHED_REF_channel=1500,bootstrap=1),
        derivation='uint32_big_endian(SHA256(ASCII(namespace|group|trajectory_id|frame_id|role|counter))[0:4])',
        forbidden_count=len(forbidden),rejections=rejected,scientific_seed_old_overlap=len(used&old_seed_values()),scientific_seed_count=len(used),
        bootstrap_seed=764389531,bootstrap_intentional_reuse_only=True,seed_manifest_sha256=sha(SEEDS)))
    seed_audit();print('R1 scientific seeds allocated/frozen:34500; bootstrap unchanged; overlap0',flush=True)

def seed_audit():
    rows=seed_rows();science=rows[:-1];values=[int(r['seed']) for r in science]
    history=set(read(ROOT/'outputs/sp_u7a_historical_seed_union.json')['forbidden_seeds']);oldseeds=old_seed_values()
    expected=list(seed_key_order());actual=[(r['group'],int(r['trajectory_id']),int(r['frame_id']),r['role']) for r in science]
    if actual!=expected or len(set(values))!=34500 or set(values)&(history|oldseeds|{764389531}):raise RuntimeError('new seed identity/uniqueness/overlap')
    used=set()
    for r,(g,i,j,role) in zip(science,expected):
        counter=int(r['counter']);value=int(r['seed'])
        if r['namespace']!=NAMESPACE or derive(g,i,j,role,counter)!=value:raise RuntimeError('seed derivation drift')
        for k in range(counter):
            if derive(g,i,j,role,k) not in history|oldseeds|used|{764389531}:raise RuntimeError('nonminimal collision counter')
        used.add(value)
    b=rows[-1]
    if b['group']!='BOOTSTRAP' or int(b['seed'])!=764389531 or b['role']!='bootstrap_seed':raise RuntimeError('bootstrap drift')
    base=original.seed_audit()
    base.update(run_id=RUN_ID,namespace=NAMESPACE,manifest_sha256=sha(SEEDS),new_scientific_seed_count=34500,
                invalidated_old_scientific_seed_overlap=0,historical_overlap_count=0,bootstrap_intentional_reuse=True)
    return base

def historical_snapshot():
    # Preserve the original metadata-only projection, extending task-owned path
    # exclusions to the new isolated R1 root. No performance payload is decoded.
    global _METADATA_ONLY
    source=inspect.getsource(original.historical_snapshot)
    source=source.replace("'sp_u7b' in p.parts", "any(v.startswith('sp_u7b') for v in p.parts)")
    scope=dict(globals());exec(source,scope)
    prior=_METADATA_ONLY;_METADATA_ONLY=True
    try:return scope['historical_snapshot']()
    finally:_METADATA_ONLY=prior

def archive(path,**arrays):
    arrays=dict(arrays,run_id=np.asarray(RUN_ID))
    with Path(path).open('xb') as f:
        np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    return dict(run_id=RUN_ID,path=Path(path).relative_to(OUT).as_posix(),sha256=sha(path),bytes=Path(path).stat().st_size,
                members={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()})

def _witnesses(verify_archives=True):
    root=OUT/'calls'
    if (OUT/'transactions').exists() and any((OUT/'transactions').iterdir()):raise RuntimeError('incomplete call transaction; STOP without redraw')
    records=[]
    for group in ('MATCHED_REF','EVAL'):
        directory=root/group
        if not directory.exists():continue
        for d in sorted(directory.iterdir()):
            if not d.is_dir() or set(p.name for p in d.iterdir())!={'H.npz','call.json'}:raise RuntimeError('archive/witness pair mismatch')
            w=read(d/'call.json');p=OUT/w['output_archive_path']
            if w['run_id']!=RUN_ID or w['group']!=group or w['trajectory_id']!=int(d.name) or w['call_success'] is not True or w['states_generated']!=160:
                raise RuntimeError('call witness identity/success/states')
            if p.resolve()!= (d/'H.npz').resolve() or not p.exists() or p.stat().st_size!=w['archive']['bytes']:raise RuntimeError('witness archive missing/path/size')
            if verify_archives and sha(p)!=w['archive_SHA256']:raise RuntimeError('witness archive hash corruption')
            records.append(w)
    if len({(w['group'],w['trajectory_id']) for w in records})!=len(records):raise RuntimeError('duplicate call ledger identity')
    return records

def reconcile(*,strict=False,verify_archives=True,expected=None,_published_records=None):
    # The optional snapshot is used ONLY immediately after this process publishes
    # a successful atomic pair. All explicit reconciliation, reload/resume and
    # phase transitions reconstruct the entire ledger from disk and verify hashes.
    if _published_records is not None:
        if strict or verify_archives or expected:raise RuntimeError('full verification cannot use a publication snapshot')
        records=_published_records
    else:
        records=_witnesses(verify_archives)
        if verify_archives:_LEDGER_SNAPSHOTS[str(OUT.resolve())]=list(records)
    m=sum(w['group']=='MATCHED_REF' for w in records);e=sum(w['group']=='EVAL' for w in records)
    counters=dict(scientific_H_calls=m+e,matched_ref_H_calls=m,eval_H_calls=e)
    state=read(state_path())
    if state.get('run_id')!=RUN_ID:raise RuntimeError('state run identity')
    if strict and any(state.get(k)!=v for k,v in counters.items()):raise RuntimeError('state/witness counter mismatch')
    if expected and (m,e)!=tuple(expected):raise RuntimeError('call role counts incomplete')
    if not strict:
        state.update(counters,reconciled_utc=utc(),authority='immutable successful-call ledger / archive bijection');replace_state(state)
    return dict(counters,witness_count=len(records),archive_count=len(records),archive_bijection=True,archive_hashes_verified=verify_archives)

def transition(expected,new,**extra):
    state=read(state_path())
    if state['phase']!=expected:raise RuntimeError('illegal phase transition '+state['phase']+' -> '+new)
    reconcile(strict=True,verify_archives=True)
    state.update(phase=new,updated_utc=utc(),**extra);replace_state(state)

def commit_h(group,i,sionna_seed,condition,generate):
    if group not in ('MATCHED_REF','EVAL'):raise ValueError('undeclared call group')
    final=OUT/'calls'/group/f'{i:04d}'
    if final.exists():
        # Crash recovery can reuse ONLY a fully committed pair with verified hash.
        _witnesses(True);w=read(final/'call.json')
        if w['sionna_seed']!=sionna_seed or w['condition']!=condition:raise RuntimeError('resume identity mismatch')
        with np.load(final/'H.npz',allow_pickle=False) as f:
            if str(f['run_id'])!=RUN_ID:raise RuntimeError('archive run identity')
            saved=f['H']
        buffer=np.empty(saved.size,saved.dtype)
        h=np.ndarray(saved.shape,dtype=saved.dtype,buffer=buffer,strides=tuple(w['export_strides_bytes']));h[:]=saved
        h.flags.writeable=False;reconcile(strict=False,verify_archives=True)
        return h,w['archive']
    key=str(OUT.resolve())
    if key not in _LEDGER_SNAPSHOTS:_LEDGER_SNAPSHOTS[key]=_witnesses(True)
    if (OUT/'transactions').exists() and any((OUT/'transactions').iterdir()):raise RuntimeError('incomplete call transaction; STOP without redraw')
    if any(w['group']==group and w['trajectory_id']==i for w in _LEDGER_SNAPSHOTS[key]):raise RuntimeError('committed archive disappeared; no redraw')
    txroot=OUT/'transactions';txroot.mkdir(exist_ok=True)
    pending=txroot/f'{group}-{i:04d}';pending.mkdir(exist_ok=False)
    h=generate()
    if h.shape!=(2,2,64,160) or h.dtype!=np.complex128 or not np.isfinite(h).all():raise RuntimeError('invalid scientific H export')
    record=archive(pending/'H.npz',H=h)
    record['path']=(final/'H.npz').relative_to(OUT).as_posix()
    witness=dict(run_id=RUN_ID,group=group,trajectory_id=i,condition=condition,sionna_seed=int(sionna_seed),timestamp_utc=utc(),
        output_archive_path=record['path'],archive_SHA256=record['sha256'],archive=record,call_success=True,states_generated=160,
        H_array_sha256=__import__('src.cdl_benchmark',fromlist=['array_hash']).array_hash(h),export_strides_bytes=list(h.strides))
    write(pending/'call.json',witness)
    final.parent.mkdir(parents=True,exist_ok=True)
    os.rename(pending,final)  # pair is published by one same-volume directory rename
    published=read(final/'call.json')  # independent decoded immutable ledger record
    if published!=witness or sha(final/'H.npz')!=published['archive_SHA256']:raise RuntimeError('atomic publication verification failed')
    _LEDGER_SNAPSHOTS[key].append(published)
    reconcile(strict=False,verify_archives=False,_published_records=_LEDGER_SNAPSHOTS[key])
    h.flags.writeable=False
    return h,record

# R1 modules activate the quarantine at import, before any integrity/analysis API.
guard()
