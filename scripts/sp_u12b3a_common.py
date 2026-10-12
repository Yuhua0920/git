"""B3A metadata, development-only wiring, and closed outcome firewall."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys, json, hashlib, contextlib, csv
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import sp_u12b2a_common as c
from scripts import run_sp_u12b2a as a
from scripts.sp_u12b3a_ablation import ARMS, deployed_arms, coordinate_fields
OUT = ROOT / 'outputs/sp_u12b3a'
SEAL = ROOT / 'outputs/sp_u12b2b_sealed'
MANIFEST = ROOT / 'outputs/sp_u12b2b_sealed_manifest.json'
MANIFEST_SHA = '54e5346a943e35b66e4a4f51a267b4db765430da1f27c6f0bc1bdbc95d1aa67e'
RAW_TREE = '6473a6d2a3b86b2a640249a3bc19f25002d9b1290a81af94b8a03f0c70457491'
ALL_TREE = '19f4cfe2a28059c783b125f1da475d2e7320e8d7877f6f63607c9acb844decbe'
CONDITIONS = ('Profile', 'Mobility', 'Composed')
SNRS = (10, 20)
FIELDS = ('condition', 'snr_db', 'trajectory_id', 'arm', 'deployment', 'error_energy',
          'truth_energy', 'coordinate_p', 'coordinate_m', 'solve_status')
CONTRAST_FIELDS = ('condition', 'snr_db', 'comparison', 'estimate_db', 'lower', 'upper', 'classification')
SOURCES = ('scripts/sp_u12b3a_ablation.py', 'scripts/sp_u12b3a_common.py',
           'scripts/sp_u12b3a_validation.py', 'scripts/run_sp_u12b3a.py',
           'scripts/sp_u12b3a_preserve.py', 'tests/test_sp_u12b3a.py')
_opaque = False
_guard = False

def need(value, code):
    if not value:
        raise RuntimeError(code)

def sha(path):
    global _opaque
    old = _opaque
    _opaque = True
    try:
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    finally:
        _opaque = old

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')

def csvwrite(path, rows):
    rows = list(rows)
    need(bool(rows) and all(set(row) == set(rows[0]) for row in rows), 'CSV_SCHEMA')
    with Path(path).open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def firewall():
    global _guard
    if _guard:
        return
    allowed = {MANIFEST, c.OUT / 'operators.npz', c.OUT / 'execution_freeze.json',
        c.OUT / 'development_seed_registry.json', ROOT / 'outputs/sp_u12b2a_seed_registry.json',
        ROOT / 'outputs/sp_u12b2a_formal_manifest.json'}
    allowed |= {ROOT / 'outputs' / name for name in c.OLD_OBJECTS}
    allowed = {p.resolve() for p in allowed}
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = ((isinstance(mode, str) and any(s in mode for s in 'wax+')) or
            (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        if not path.is_relative_to((ROOT / 'outputs').resolve()):
            return
        owned = path.is_relative_to(OUT.resolve()) or path.name.startswith('sp_u12b3a_')
        if writing:
            if not owned:
                raise PermissionError('HISTORICAL_WRITE_DENIED')
            return
        if _opaque:
            return  # SHA256 streaming only; no numerical decoding.
        if path.is_relative_to(SEAL.resolve()):
            raise PermissionError('B2_SCIENTIFIC_READ_DENIED')
        if not (owned or path in allowed):
            raise PermissionError('HISTORICAL_OUTCOME_READ_DENIED')
    sys.addaudithook(hook)
    _guard = True

def tree(leaves):
    return hashlib.sha256(json.dumps(leaves, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def seal_metadata():
    need(sha(MANIFEST) == MANIFEST_SHA, 'B3A_SEAL_VIOLATION_MANIFEST')
    m = read(MANIFEST)
    need(m['raw_data_hash_tree'] == RAW_TREE and m['all_artifact_hash_tree'] == ALL_TREE,
         'B3A_SEAL_VIOLATION_TREE')
    need(m['status'] == 'SEALED_COMPLETE_UNVIEWED' and not m['outcomes_viewed'], 'B2_SEAL_STATUS')
    return m

def prerequisites():
    need((OUT / 'before.json').exists(), 'PRESERVATION_BASELINE_MISSING')
    seal_metadata()
    frozen = read(c.OUT / 'execution_freeze.json')
    for path, digest in frozen['hashes'].items():
        need(sha(ROOT / path) == digest, 'B2A_FROZEN_INPUT_DRIFT')
    manifest = read(ROOT / 'outputs/sp_u12b2a_formal_manifest.json')
    need(sha(ROOT / manifest['operator_cache']['path']) == manifest['operator_cache']['sha256'], 'OPERATOR_CACHE_DRIFT')
    return manifest

def explicit_stack(cv, window):
    columns, responses = [], []
    for p in window:
        u = p - cv.source_mean_p
        for f, (o, h) in enumerate(cv.folds):
            columns.append(np.column_stack((cv.profile_delta_k[f] @ u[o], cv.mobility_delta_k[f] @ u[o])))
            responses.append(u[h] - cv.source_k[f] @ u[o])
    D, e = np.concatenate(columns), np.concatenate(responses)
    return np.vstack((D.real, D.imag)), np.r_[e.real, e.imag]

def fixture(seed=431012, mode='coupled', zero=None, frames=5):
    from src.pilot_cv import PilotCV, frozen_folds
    from src.cdl_benchmark import observation_geometry
    from src.lowrank_modules import RankFactors
    rng = np.random.default_rng(seed)
    def z(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)
    folds, _ = frozen_folds(observation_geometry().selected)
    p, m, ks = [], [], []
    for _ in folds:
        P = z((16, 48)) / 10
        M = z((16, 48)) / 10
        if mode == 'orthogonal':
            P[8:] = 0
            M[:8] = 0
        else:
            M = .6 * P + .4 * M
        if zero == 'P':
            P[:] = 0
        if zero == 'M':
            M[:] = 0
        p.append(P); m.append(M); ks.append(z((16, 48)) / 20)
    mu = z((64,)) / 10
    cv = PilotCV(folds, mu, tuple(ks), tuple(p), tuple(m))
    arrays = dict(source_mean_p=mu, source_mean_d=z((13,)), Wsource_10=z((13, 64)) / 20)
    factors = {}
    for direction in ('P', 'M'):
        A, Q = z((13, 4)) / 10, z((4, 64)) / 10
        if zero == direction:
            A[:] = 0
        factors[direction] = RankFactors(A, Q)
        arrays[f'DeltaW{direction}_10'] = A @ Q
    return cv, z((frames, 64)), arrays, factors

def planned_contrasts():
    return [dict(condition=n, snr_db=s, comparison=f'Both_LR4_vs_{arm}_LR4',
        first=('P-only/LR4' if arm == 'Ponly' else 'M-only/LR4'), second='Both/LR4', positive_means='Both has lower pooled NMSE')
        for n in CONDITIONS for s in SNRS for arm in ('Ponly', 'Monly')]

def bootstrap_dummy(error, truth, draws):
    """Schema/paired-index validation only; caller supplies small synthetic draws."""
    e, t, indices = np.asarray(error), np.asarray(truth), np.asarray(draws)
    need(e.ndim == 3 and e.shape[0:2] == (2, 3) and t.shape == (2, e.shape[2]), 'DUMMY_BOOTSTRAP_SHAPE')
    need(indices.ndim == 2 and indices.shape[1] == e.shape[2], 'DUMMY_INDEX_SHAPE')
    need(np.isfinite(e).all() and np.isfinite(t).all() and np.all(e > 0) and np.all(t > 0), 'DUMMY_ENERGY')
    pooled = np.stack([10 * np.log10(e[:, :, idx].sum(axis=-1) / t[:, idx].sum(axis=-1)[:, None]) for idx in indices])
    return np.stack((pooled[:, :, 0] - pooled[:, :, 2], pooled[:, :, 1] - pooled[:, :, 2]), axis=-1)

def future_reuse_schema():
    m = seal_metadata()
    registry = read(ROOT / 'outputs/sp_u12b2a_seed_registry.json')
    seed_index = a.seed_index(registry)
    wires = []
    for condition in CONDITIONS:
        for i in range(1000):
            H = f'cache/{condition}/H_{i:04d}.npz'
            comm = f'cache/{condition}/comm_{i:04d}.npz'
            need(H in m['files'] and comm in m['files'], 'B2_REUSE_WIRING_MISSING')
            need(all((condition, 'EVAL', i, j, role) in seed_index
                for j in range(10) for role in c.ROLES[3:]), 'B2_COMM_SEED_WIRING_MISSING')
            wires.append(dict(condition=condition, trajectory_id=i, channel_path=H,
                channel_sha256=m['files'][H]['sha256'], communication_path=comm,
                communication_sha256=m['files'][comm]['sha256']))
    write(OUT / 'future_reuse_manifest.json', dict(schema_only=True,
        B2_manifest_sha256=MANIFEST_SHA, entries=wires, scientific_values_read=False))
    schema = dict(conditions=list(CONDITIONS), snr_db=list(SNRS), arms=list(ARMS),
        deployments=['Dense', 'LR4'], trajectory_fields=list(FIELDS),
        scored_frame_extension='frame_id added to raw frame rows; trajectory energies sum frames 4–9',
        primary_key=['condition', 'snr_db', 'trajectory_id', 'arm', 'deployment'],
        null_coordinates={'P-only': ['coordinate_m'], 'M-only': ['coordinate_p'], 'Both': []},
        contrast_fields=list(CONTRAST_FIELDS), planned_contrasts=planned_contrasts(),
        bootstrap=dict(resamples=10000, family=12, confidence=.95, q=.05/24,
            unit='paired trajectory', pooling='resampled sum error / resampled sum truth before dB',
            condition_streams='independent; same index draws for all arms/deployments/SNR in condition',
            quantile_method='linear', exact_master_seed='freeze in separately authorized B3 pre-generation metadata; no channel seeds allocated'),
        evaluation_n_per_condition=1000, reference_reconstruction=False,
        future_reuse='B2 cached H and bits/symbols/unit-noise; same frozen communication-to-pilots path at10/20; no channel generation',
        ID_Boundary='descriptive only if separately authorized; not part of primary family',
        formal_B3_authorized=False, formal_bootstrap_executed=False, B2_outcomes_viewed=False)
    need(len(schema['planned_contrasts']) == 12, 'CONTRAST_FAMILY')
    write(ROOT / 'outputs/sp_u12b3a_schema.json', schema)
    return schema

def seal_audit():
    import time
    m = seal_metadata(); tick = time.monotonic()
    leaves = {}
    for i, (name, info) in enumerate(m['files'].items(), 1):
        path = SEAL / name
        need(path.is_relative_to(SEAL) and path.exists(), 'B3A_SEAL_VIOLATION_MISSING')
        digest = sha(path)
        need(digest == info['sha256'] and path.stat().st_size == info['bytes'], 'B3A_SEAL_VIOLATION_BYTES')
        leaves[name] = dict(sha256=digest, bytes=path.stat().st_size)
        if time.monotonic() - tick > 25:
            print({'seal_hashes_checked': i}, flush=True); tick = time.monotonic()
    actual = {p.relative_to(SEAL).as_posix() for p in SEAL.rglob('*') if p.is_file()}
    need(actual == set(leaves), 'B3A_SEAL_VIOLATION_POPULATION')
    need(tree(leaves) == ALL_TREE, 'B3A_SEAL_VIOLATION_TREE')
    need(tree({k: v for k, v in leaves.items() if k.startswith('raw_evaluation/')}) == RAW_TREE, 'B3A_SEAL_VIOLATION_RAW_TREE')
    audit = dict(PASS=True, seal_manifest_sha256=sha(MANIFEST), raw_hash_tree=RAW_TREE,
        all_artifact_hash_tree=ALL_TREE, files_checked=len(leaves), opaque_hash_only=True,
        B2_scientific_values_read=False, B2_outcomes_viewed=False, modifications=0)
    write(ROOT / 'outputs/sp_u12b3a_b2_seal_audit.json', audit)
    return audit
