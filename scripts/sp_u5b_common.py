"""Frozen U5A inference reuse; only the measurement mask changes."""
import os,sys,time,hashlib,subprocess
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u4a_common import sha,save,readjson,csvsave,csvread,distribution,trajectory_metric,peak_memory
from scripts.sp_u5a_common import predict
from src.pilots import PilotLayout
from src.cdl_benchmark import frames,array_hash,SYSTEM
import numpy as np
OUT=ROOT/'outputs';PREFIX='sp_u5b_';GUARD=False
OLD_HASH_ONLY={'t16e_frozen_operators.npz','t16f_frozen_candidates.npz','t16g_frozen_module_factors.npz'}

def firewall():
    global GUARD
    if GUARD:return
    base=str(OUT.resolve()).lower()+os.sep
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if not str(p).lower().startswith(base):return
        rel=p.relative_to(OUT).parts;own=rel[0].startswith(PREFIX)
        mode=args[1];flags=args[2] if isinstance(args[2],int) else 0
        writing=(isinstance(mode,str) and any(k in mode for k in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        allowed=rel[0].startswith('sp_u5a_') or (len(rel)==1 and rel[0] in OLD_HASH_ONLY)
        if not own and (not allowed or writing):raise PermissionError('SP-U5B firewall:'+str(p))
    sys.addaudithook(audit);GUARD=True

def geometry(df):
    if df not in (8,16,32):raise ValueError('frozen density')
    with np.load(OUT/f'sp_u5a_geometry_{df}.npz',allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
    pm=a['pilot_mask'];union=np.any(pm,axis=0)
    offsets=tuple(int(np.flatnonzero(np.any(pm[q],axis=1))[0]) for q in range(2))
    ts=tuple(int(v) for v in np.flatnonzero(np.any(union,axis=0)))
    layout=PilotLayout(pm,union,a['data_mask'],a['mute_mask'],a['pilot_owner'],offsets,ts)
    return SimpleNamespace(df=df,layout=layout,data=a['data'],selected=a['selected'])

def common_domain():
    gs={df:geometry(df) for df in (8,16,32)}
    txmask=np.logical_and.reduce([g.layout.data_mask for g in gs.values()])
    full=np.broadcast_to(txmask[None],(2,2,64,16)).ravel().copy();ids=np.flatnonzero(full)
    if not np.array_equal(txmask,gs[8].layout.data_mask) or txmask[0].sum()!=960 or ids.size!=3840:
        raise ValueError('actual common mask differs from task expectation; STOP')
    positions={}
    for df,g in gs.items():
        native=np.flatnonzero(g.data);pos=np.searchsorted(native,ids)
        if np.any(pos>=len(native)) or not np.array_equal(native[pos],ids):raise ValueError('target alignment')
        positions[df]=pos
    return gs,txmask,full,ids,positions

def operators(df):
    p=OUT/f'sp_u5a_operators_{df}.npz';meta=readjson(OUT/f'sp_u5a_construction_{df}.json')
    if sha(p)!=meta['operators_sha256']:raise ValueError('operator drift')
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}

def coordinate_index():
    rows=[r for r in csvread(OUT/'sp_u5a_windows.csv') if r['artifact'] in ('8','16','32')]
    index={(r['condition'],int(r['trajectory_id']),int(r['snr_db']),int(r['dF']),int(r['frame_id'])):r for r in rows}
    if len(index)!=7200:raise ValueError('coordinate population')
    return index

def freeze():
    firewall()
    if (OUT/'sp_u5b_preregistration.json').exists():raise ValueError('already frozen')
    if SYSTEM['sample_rate_hz']!=SYSTEM['nfft']*SYSTEM['delta_f_hz']:raise ValueError('sample rate mismatch')
    manifest=readjson(OUT/'sp_u5a_delivery_manifest.json')
    if sha(OUT/'sp_u5a_delivery_manifest.json')!='0948e3b546088627dc0c0f0e477e91c40674b679695fb1ebf795dd0fac61fcd2':raise ValueError('U5A seal differs')
    protected={**manifest['source_sha256'],**manifest['artifact_sha256']}
    protected['outputs/sp_u5a_delivery_manifest.json']=sha(OUT/'sp_u5a_delivery_manifest.json')
    for folder in ('src','scripts','tests','configs','docs','paper'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and 'sp_u5b' not in p.name:protected[p.relative_to(ROOT).as_posix()]=sha(p)
    for n in OLD_HASH_ONLY:protected['outputs/'+n]=sha(OUT/n)
    if any(sha(ROOT/p)!=h for p,h in manifest['source_sha256'].items()) or any(sha(ROOT/p)!=h for p,h in manifest['artifact_sha256'].items()):raise ValueError('U5A drift')
    save(OUT/'sp_u5b_before_manifest.json',protected)
    gs,mask,full,ids,pos=common_domain()
    np.savez_compressed(OUT/'sp_u5b_common_mask.npz',tx_data_mask=mask,channel_mask=full,truth_indices=ids,**{f'positions_{df}':p for df,p in pos.items()})
    save(OUT/'sp_u5b_preregistration.json',dict(config=readjson(ROOT/'configs/sp_u5b_protocol.json'),config_sha256=sha(ROOT/'configs/sp_u5b_protocol.json'),
        HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),frozen_before_common_results=True,
        common_physical_REs=int(mask[0].sum()),common_target_dimension=len(ids),common_bits_per_frame=int(mask.sum()*2),common_equals_dF8=True,
        mask_array_sha256=array_hash(full),mask_artifact_sha256=sha(OUT/'sp_u5b_common_mask.npz'),U5A_manifest_sha256=sha(OUT/'sp_u5a_delivery_manifest.json'),
        ratio_guard=float(np.sqrt(np.finfo(float).eps)),performance_threshold=None,created_unix=time.time()))
    print('SP-U5B frozen: actual intersection equals dF8;960 physical REs,3840 targets/bits; no new inputs')

def prereg():
    r=readjson(OUT/'sp_u5b_preregistration.json')
    if r['config_sha256']!=sha(ROOT/'configs/sp_u5b_protocol.json') or r['mask_artifact_sha256']!=sha(OUT/'sp_u5b_common_mask.npz'):raise ValueError('protocol/mask drift')
    return r

def protect():
    p=readjson(OUT/'sp_u5b_before_manifest.json')
    bad=[n for n,h in p.items() if sha(ROOT/n)!=h]
    if bad:raise ValueError('protected drift:'+str(bad))
    return len(p)

def block_scientific_mutations():
    # Runtime guards supplement AST auditing; no calls to generators or fitters are authorized.
    import scripts.sp_u5a_common as old
    import src.link as link
    def denied(*a,**kw):raise RuntimeError('SP-U5B forbids new random inputs/refit/rebuild')
    old.DensityCV.fit=denied;old.build_stats=denied;old.build_ops=denied
    old.StreamingStatistics.update=denied;old.StreamingStatistics.freeze=denied
    old.WeightedModule.build=denied;old.OfficialCDL.generate=denied
    link.generate_master_qpsk=denied;np.random.default_rng=denied

