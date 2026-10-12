"""SP-U4A isolated preparation and development protocol. No old outcome readers."""
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.cdl_benchmark import FrozenStatistics, StreamingStatistics, array_hash, observation_geometry
from src.lowrank_modules import WeightedModule, RankFactors, factored_affine_predict
from src.pilot_cv import PilotCV, frozen_folds, cross_pilot_operators

OUT = ROOT / 'outputs'
PREFIX = 'sp_u4a_'
HIST_STATS = {'Source': 't15a_cdl_statistics_500.npz', 'P': 't15c_cdl_statistics_CDL-B-30.npz', 'M': 't15c_cdl_statistics_CDL-C-120.npz'}
ALLOWED_OLD = set(HIST_STATS.values()) | {'t15a_fix_replay_h_hashes.csv', 't15c_cdl_channel_hashes.csv',
    't15a_cdl_manifest.csv', 't15c_cdl_manifest.csv', 't16e_frozen_operators.npz',
    't16f_frozen_candidates.npz', 't16g_frozen_module_factors.npz'}
_guard = False
_denied = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()


def readjson(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def csvsave(path, rows):
    rows = list(rows)
    if not rows: raise ValueError('empty result table')
    fields = list(rows[0])
    if any(set(r) != set(fields) for r in rows): raise ValueError('incomplete CSV schema')
    with Path(path).open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def csvread(path):
    with Path(path).open(encoding='utf8', newline='') as f: return list(csv.DictReader(f))


def config(): return readjson(ROOT/'configs/sp_u4a_protocol.json')


def firewall():
    global _guard
    if _guard: return
    outputs = str(OUT.resolve()).lower()+os.sep
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)): return
        p = Path(os.fsdecode(args[0])).resolve(); name = str(p).lower()
        if not name.startswith(outputs): return
        rel = p.relative_to(OUT).parts
        permitted = rel[0].startswith(PREFIX) or (len(rel)==1 and rel[0] in ALLOWED_OLD)
        mode = args[1] if len(args)>1 else None
        flags = args[2] if len(args)>2 and isinstance(args[2], int) else 0
        writing = (isinstance(mode,str) and any(k in mode for k in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if not permitted or (not rel[0].startswith(PREFIX) and writing):
            _denied.append(str(p)); raise PermissionError('SP-U4A output firewall blocked: '+str(p))
    sys.addaudithook(audit); _guard = True


def peak_memory():
    class PMC(ctypes.Structure):
        _fields_ = [('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
    if os.name != 'nt': return {'peak_working_set_bytes': None, 'peak_commit_bytes': None}
    p=PMC(); p.cb=ctypes.sizeof(p)
    kernel=ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype=ctypes.c_void_p
    psapi=ctypes.WinDLL('psapi'); psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(p),p.cb): raise ctypes.WinError()
    return dict(peak_working_set_bytes=int(p.PeakWorkingSetSize),peak_commit_bytes=int(p.PeakPagefileUsage))


def freeze():
    cfg=config()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=cfg['head']: raise ValueError('HEAD drift')
    path=OUT/'sp_u4a_preregistration.json'
    if path.exists(): raise ValueError('preregistration already exists')
    permutations={e: np.random.default_rng(s).permutation(500).tolist() for e,s in cfg['permutation_seeds'].items()}
    rows=[dict(environment=e,position=j,trajectory_id=i,smallest_budget=next(n for n in cfg['budgets'] if j<n)) for e,p in permutations.items() for j,i in enumerate(p)]
    csvsave(OUT/'sp_u4a_nested_ids.csv',rows)
    protect={}
    for folder in ('src','scripts','tests','configs','docs','paper'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and 'sp_u4a' not in p.name: protect[p.relative_to(ROOT).as_posix()]=sha(p)
    for f in sorted(ALLOWED_OLD): protect['outputs/'+f]=sha(OUT/f)
    save(OUT/'sp_u4a_before_manifest.json',protect)
    save(path,dict(config=cfg,config_sha256=sha(ROOT/'configs/sp_u4a_protocol.json'),permutations=permutations,
        permutation_sha256={e:array_hash(np.array(p,dtype=np.int64)) for e,p in permutations.items()},
        nested_ids_sha256=sha(OUT/'sp_u4a_nested_ids.csv'),frozen_before_nmse=True,created_unix=time.time(),
        historical_tolerance_rationale='1e-10 relative Frobenius gate on independently reordered complex128 accumulation/SVD; fixed before results; effective products only'))
    print('SP-U4A config and three nested permutations frozen before NMSE',flush=True)


def prereg():
    p=readjson(OUT/'sp_u4a_preregistration.json')
    if p['config_sha256']!=sha(ROOT/'configs/sp_u4a_protocol.json') or p['nested_ids_sha256']!=sha(OUT/'sp_u4a_nested_ids.csv'): raise ValueError('protocol drift')
    return p


def protected_files():
    # Initial snapshot also retained early U4A preparation sources. They are
    # new task-owned files; historical protection excludes that namespace.
    return {p:h for p,h in readjson(OUT/'sp_u4a_before_manifest.json').items() if 'sp_u4a' not in Path(p).name}


def development_seeds(condition,trajectory,frame=0):
    cfg=config(); role=cfg['conditions'][condition]['role']
    return {name:int(np.random.SeedSequence([*cfg['namespace_words'],role,trajectory,0 if r<=3 else frame,r]).generate_state(1)[0])
        for r,name in enumerate(('sionna_seed','torch_seed','numpy_seed','bits_seed_tx0','bits_seed_tx1','noise_seed'),1)}


def build_statistics(corpus, ids, geometry=None):
    ids=list(map(int,ids)); g=geometry or observation_geometry()
    if not ids or len(set(ids))!=len(ids) or any(i<0 or i>=500 for i in ids):
        raise ValueError('unique training IDs0..499 required')
    stream=StreamingStatistics(g.selected,g.data)
    for start in range(0,len(ids),10):
        batch=ids[start:start+10]
        stream.update(np.asarray(corpus[batch]).reshape(len(batch)*10,4096), batch)
    return stream.freeze()


def build_operators(stats, cfg=None):
    cfg=cfg or config(); g=observation_geometry(); folds,definition=frozen_folds(g.selected)
    a={'source_mean_p':stats['Source'].mean_p,'source_mean_d':stats['Source'].mean_d}
    operator_seconds=0.; compression_seconds=0.; proof=[]
    for text,sigma in cfg['noise_variances'].items():
        s=int(text); tick=time.perf_counter()
        for e in cfg['environments']:
            a[f'W{e}_{s}']=stats[e].prepare(sigma).weight
            a[f'K{e}_{s}']=np.stack(cross_pilot_operators(stats[e].rpp,sigma,folds)[0])
        a[f'Wsource_{s}']=a[f'WSource_{s}']
        for e in ('P','M'):
            a[f'DeltaW{e}_{s}']=a[f'W{e}_{s}']-a[f'WSource_{s}']
            a[f'DeltaK{e}_{s}']=a[f'K{e}_{s}']-a[f'KSource_{s}']
        operator_seconds+=time.perf_counter()-tick
        tick=time.perf_counter()
        for e in ('P','M'):
            model=WeightedModule.build(a[f'DeltaW{e}_{s}'],stats[e].rpp+sigma*np.eye(64)); f=model.factors(4)
            a[f'A_{e}_{s}']=f.A; a[f'Q_{e}_{s}']=f.Q
            proof.append(dict(module=e,snr_db=s,weighted_capture_rank4=model.capture(4),full_rank_residual=model.diagnostics()['full_rank_correction_relative_residual']))
        compression_seconds+=time.perf_counter()-tick
    return a,dict(operator_seconds=operator_seconds,compression_seconds=compression_seconds,fold_sha256=definition['sha256'],factor_diagnostics=proof)


def cv_model(a,s):
    return PilotCV(frozen_folds(observation_geometry().selected)[0],a['source_mean_p'],tuple(a[f'KSource_{s}']),tuple(a[f'DeltaKP_{s}']),tuple(a[f'DeltaKM_{s}']))


def source_predict(a,pilot,s): return a['source_mean_d']+(pilot-a['source_mean_p'])@a[f'Wsource_{s}'].T


def affine_predict(a,pilot,s,c):
    return factored_affine_predict(pilot,a['source_mean_p'],a['source_mean_d'],a[f'Wsource_{s}'],c,
        RankFactors(a[f'A_P_{s}'],a[f'Q_P_{s}']),RankFactors(a[f'A_M_{s}'],a[f'Q_M_{s}']))


def historical_arrays():
    a={}
    for filename in ('t16e_frozen_operators.npz','t16f_frozen_candidates.npz','t16g_frozen_module_factors.npz'):
        with np.load(OUT/filename,allow_pickle=False) as f:
            for k in f.files:
                if k=='metadata_json': continue
                a[k]=f[k].copy()
    for e in ('P','M'):
        for s in (10,20):
            a[f'A_{e}_{s}']=a[f'A_{e}_{s}'][:,:4]; a[f'Q_{e}_{s}']=a[f'Q_{e}_{s}'][:4]
    return a


def rel_error(x,y,epsilon=0.): return float(np.linalg.norm(x-y)/(np.linalg.norm(y)+epsilon))


def distribution(x):
    x=np.asarray(x,dtype=float)
    q=np.quantile(x,[.05,.25,.5,.75,.95])
    return dict(mean=float(x.mean()),median=float(q[2]),q25=float(q[1]),q75=float(q[3]),iqr=float(q[3]-q[1]),p05=float(q[0]),p95=float(q[4]),minimum=float(x.min()),maximum=float(x.max()))


def trajectory_metric(errors,truth):
    error=0.; energy=0.
    for e,h in zip(errors,truth,strict=True): error+=float(e); energy+=float(h)
    return dict(error_energy=error,truth_energy=energy,nmse_linear=error/energy,nmse_db=float(10*np.log10(error/energy)))


if __name__=='__main__': freeze()
