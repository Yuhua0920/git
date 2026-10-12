"""Bounded SP-U2C conditional-noise diagnostic; no channel generator imports."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs'
IDS = tuple(range(20))
SNRS = (10, 20)
REPEATS = 20
FRAMES = tuple(range(4, 10))
HEAD = 'ca83982b1d0e4bdca0215a3ecbe3a7a8f8f8ec55'
POWER = 9.941134822523972
VARIANCES = {10: .9941134822523973, 20: .09941134822523973}
TOL = 2e-10
EPS = np.finfo(float).eps
ACCESS = []
ALLOWED = {'t16e_pilot_cv_preflight.json', 't16e_frozen_operators.npz',
           't16f_preflight.json', 't16f_frozen_candidates.npz',
           't16g_factor_preflight.json', 't16g_frozen_module_factors.npz',
           'sp_u2b_development_gate.json', 'sp_u2b_preregistration.json',
           'sp_u2b_seed_manifest.csv'}
ALLOWED |= {f'sp_u2b_development_cache/trajectory_{i:03d}.npz' for i in IDS}
_GUARDED = False


def output_guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    try:
        rel = p.relative_to(OUT.resolve()).as_posix()
    except ValueError:
        return
    flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
    writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    allowed = rel.split('/')[0].startswith('sp_u2c_') or (rel in ALLOWED and not writing)
    ACCESS.append(dict(path=rel, writing=writing, allowed=allowed))
    if not allowed:
        raise PermissionError('SP-U2C output firewall: ' + rel)


def install_guard():
    global _GUARDED
    if not _GUARDED:
        sys.addaudithook(output_guard)
        _GUARDED = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def normal(value):
    if isinstance(value, dict):
        return {str(k): normal(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normal(v) for v in value]
    if isinstance(value, np.ndarray):
        return normal(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def save(path, value):
    path = Path(path)
    if path.exists():
        raise RuntimeError('immutable output exists: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normal(value), indent=2, allow_nan=False) + '\n', encoding='utf8')


def csvsave(path, rows):
    path = Path(path)
    if path.exists() or not rows:
        raise RuntimeError('existing or empty CSV: ' + str(path))
    fields = list(rows[0])
    assert all(set(r) == set(fields) for r in rows), 'heterogeneous output schema'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def close(a, b):
    err = float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(1., float(np.linalg.norm(b))))
    if not np.isfinite(err) or err > TOL:
        raise AssertionError(('identity failure', err))
    return err


def noise_seed(i, r, j):
    if i not in IDS or r not in range(REPEATS) or j not in range(10):
        raise ValueError('outside fixed allocation')
    return 3600000000 + 10000*i + 100*r + j


def selected_ids():
    return IDS  # No input or performance parameter is accepted.


def frames(h):
    return h.reshape(2, 2, 64, 10, 16).transpose(3, 0, 1, 2, 4)


def frozen_method():
    from src.lowrank_modules import RankFactors
    from src.pilot_cv import PilotCV, frozen_folds
    from src.spatial_information import observation_geometry
    arrays, proof, factors_array = {}, {}, None
    for filename, key in (('t16e_pilot_cv_preflight.json', 'operator_artifact'),
                          ('t16f_preflight.json', 'candidate_artifact'),
                          ('t16g_factor_preflight.json', 'factor_artifact')):
        info = load(OUT/filename)[key]
        p = ROOT/info['path']
        assert sha(p) == info['sha256']
        with np.load(p, allow_pickle=False) as f:
            values = {k: f[k].copy(order='K') for k in info['array_sha256']}
        assert all(array_hash(values[k]) == h for k, h in info['array_sha256'].items())
        for v in values.values():
            v.flags.writeable = False
        if key == 'factor_artifact':
            factors_array = values
        else:
            arrays.update(values)
        proof[filename] = dict(metadata_sha256=sha(OUT/filename), **info)
    folds, definition = frozen_folds(observation_geometry().selected)
    assert definition == load(OUT/'t16e_pilot_cv_preflight.json')['pilot_fold_definition']
    cvs = {s: PilotCV(folds, arrays['source_mean_p'], tuple(arrays[f'KSource_{s}']),
                      tuple(arrays[f'DeltaKP_{s}']), tuple(arrays[f'DeltaKM_{s}'])) for s in SNRS}
    factors = {(m, s): RankFactors(factors_array[f'A_{m}_{s}'][:, :4], factors_array[f'Q_{m}_{s}'][:4])
               for m in ('P', 'M') for s in SNRS}
    assert all(f.rank == 4 for f in factors.values())
    return arrays, cvs, factors, dict(artifacts=proof, fold=definition, rank=4, window=5,
                                    variances=VARIANCES, power=POWER)


def cache(i, manifest):
    if i not in IDS:
        raise ValueError('unselected trajectory')
    record = manifest[i]
    assert record['trajectory_id'] == i and record['stage'] == 'development'
    p = OUT/f'sp_u2b_development_cache/trajectory_{i:03d}.npz'
    assert sha(p) == record['cache_sha256']
    # Never load old noisy fits, old oracle or old outcome rows.
    keys = ['H', 'pilots'] + [f'{k}_{s}_{j}' for s in SNRS for j in FRAMES for k in ('G0', 'z0', 'c0')]
    with np.load(p, allow_pickle=False) as f:
        x = {k: f[k] for k in keys}
    x['clean_pilots'] = x.pop('pilots')[0].copy()
    assert x['H'].shape == (2, 2, 64, 160) and array_hash(x['H']) == record['H_sha256']
    for a in x.values():
        a.flags.writeable = False
    return x


def fixed_inputs(i, x):
    from src.link import generate_master_qpsk, assemble_pilot_data_frame, apply_frequency_mimo_channel, PILOT_SYMBOLS
    from src.pilots import map_pilot_symbols
    from src.spatial_information import observation_geometry
    from src.estimators import pilot_ls_estimate
    g = observation_geometry()
    known = map_pilot_symbols(g.layout, PILOT_SYMBOLS)
    ys, rows = [], []
    for j, h in enumerate(frames(x['H'])):
        base = 3501000000 + i*1000
        seeds = (base+100+3*j, base+101+3*j)
        bits, symbols = generate_master_qpsk(bits_seeds=seeds)
        tx = assemble_pilot_data_frame(symbols, g.layout)
        y = apply_frequency_mimo_channel(h, tx)
        z = pilot_ls_estimate(y, known, g.layout.pilot_mask)[0].ravel()[g.selected]
        close(z, x['clean_pilots'][j])
        rows.append(dict(trajectory_id=i, frame_id=j, bits_seed_tx0=seeds[0], bits_seed_tx1=seeds[1],
                         H_frame_sha256=array_hash(h), bits_sha256=array_hash(bits), symbols_sha256=array_hash(symbols),
                         transmitted_sha256=array_hash(tx), useful_sha256=array_hash(y), clean_pilot_sha256=array_hash(z)))
        ys.append(y)
    return np.stack(ys), rows


def noisy_inputs(i, r, useful):
    from src.numerics import complex_gaussian_noise
    from src.spatial_information import observation_geometry
    from src.pilots import map_pilot_symbols
    from src.link import PILOT_SYMBOLS
    from src.estimators import pilot_ls_estimate
    g = observation_geometry()
    known = map_pilot_symbols(g.layout, PILOT_SYMBOLS)
    atoms = [complex_gaussian_noise((2,64,16), 1., np.random.default_rng(noise_seed(i,r,j))) for j in range(10)]
    pilots = np.stack([[pilot_ls_estimate(useful[j]+np.sqrt(VARIANCES[s])*atoms[j], known,
                         g.layout.pilot_mask)[0].ravel()[g.selected] for j in range(10)] for s in SNRS])
    rows = [dict(trajectory_id=i, repeat_id=r, frame_id=j, noise_seed=noise_seed(i,r,j),
                 paired_snrs='10,20', standardized_atom_sha256=array_hash(atoms[j])) for j in range(10)]
    return pilots, rows


def cosine(s, n, M):
    Es, En, dot = float(s@M@s), float(n@M@n), float(s@M@n)
    lm = float(np.linalg.norm(M,2))
    ts = 64*EPS*max(1., lm*float(s@s))
    tn = 64*EPS*max(1., lm*float(n@n))
    unsafe = Es <= ts or En <= tn
    den = float(np.sqrt(max(Es,0))*np.sqrt(max(En,0)))
    value = None if unsafe else dot/den
    if value is not None and abs(value) > 1+TOL:
        raise AssertionError('invalid metric cosine')
    return dict(alignment_cosine=value, alignment_undefined=bool(unsafe), alignment_numerator=dot,
                alignment_denominator=den, cosine_threshold_s=ts, cosine_threshold_n=tn,
                cosine_roundoff_outside_unit=bool(value is not None and abs(value)>1))


def empirical(c, c0):
    c = np.asarray(c)
    mean = c.mean(axis=0)
    centered = c-mean
    cov = centered.T@centered/(len(c)-1)
    v = float(np.mean(np.sum((c-c0)**2,axis=1)))
    bias = mean-c0
    close(v, (len(c)-1)/len(c)*np.trace(cov)+float(bias@bias))
    return dict(mean_alpha=float(mean[0]), mean_beta=float(mean[1]), bias_alpha=float(bias[0]), bias_beta=float(bias[1]),
                bias_norm=float(np.linalg.norm(bias)), V_c=v, RMS_c=float(np.sqrt(v)),
                variance_alpha=float(cov[0,0]), variance_beta=float(cov[1,1]), covariance_alpha_beta=float(cov[0,1]),
                covariance_beta_alpha=float(cov[1,0]), trace_covariance=float(np.trace(cov)))


def diagnose(i, r, pilots, x, method):
    from src.lowrank_modules import factored_affine_predict
    from src.spatial_information import observation_geometry
    from scripts.sp_u2b_diagnostic import explicit_design, geometry, channel_oracle
    arrays, cvs, factors, _ = method
    g = observation_geometry()
    rows = []
    for si, s in enumerate(SNRS):
        for j in FRAMES:
            c0 = x[f'c0_{s}_{j}']
            c, diagnostics = cvs[s].fit(pilots[si,j-4:j+1], modules=('P','M'))
            G, z = explicit_design(cvs[s],pilots[si,j-4:j+1])
            normal_error = max(close(G.T@G,diagnostics['gram']), close(G.T@z,diagnostics['rhs']))
            rank = int(np.linalg.matrix_rank(G))
            if rank != 2:
                raise AssertionError('rank failure; no fallback')
            u = pilots[si,j]-arrays['source_mean_p']
            fp,fm = factors['P',s],factors['M',s]
            X = np.column_stack((fp.apply(u),fm.apply(u)))
            base = arrays['source_mean_d']+u@arrays[f'Wsource_{s}'].T
            truth = frames(x['H'])[j].ravel()[g.data]
            oracle,M,ls_error = channel_oracle(truth,base,X)
            output_rank = int(np.linalg.matrix_rank(np.vstack((X.real,X.imag))))
            if output_rank != 2:
                raise AssertionError('oracle rank failure')
            prediction = factored_affine_predict(pilots[si,j], arrays['source_mean_p'], arrays['source_mean_d'],
                         arrays[f'Wsource_{s}'], c, fp, fm)
            pred0, predst = base+X@c0, base+X@oracle
            n,sr = c-c0,c0-oracle
            risk = lambda p: float(np.vdot(p-truth,p-truth).real/3968)
            R,R0,Ro = risk(prediction),risk(pred0),risk(predst)
            Q,C = float(n@M@n/3968),float(2*sr@M@n/3968)
            surrogate,total = float(sr@M@sr/3968),float((sr+n)@M@(sr+n)/3968)
            errors = [close(prediction-pred0,X@n),close(R-R0,Q+C),close(R0-Ro,surrogate),close(R-Ro,total)]
            scale = max(1.,abs(Q)+abs(C),R,R0)
            borderline = abs(R-R0) <= TOL*scale
            if not borderline and ((R-R0<0) != (C < -Q)):
                raise AssertionError('negative penalty equivalence')
            eig = np.linalg.eigvalsh(M)
            row = dict(trajectory_id=i,repeat_id=r,snr_db=s,frame_id=j,window_start=j-4,ndata=3968,
                       alpha0=float(c0[0]),beta0=float(c0[1]),alpha_r=float(c[0]),beta_r=float(c[1]),
                       alpha_oracle=float(oracle[0]),beta_oracle=float(oracle[1]),
                       n_alpha=float(n[0]),n_beta=float(n[1]),s_alpha=float(sr[0]),s_beta=float(sr[1]),
                       n_norm=float(np.linalg.norm(n)),M_PP=float(M[0,0]),M_PM=float(M[0,1]),M_MM=float(M[1,1]),
                       output_lambda_min=float(eig[0]),output_lambda_max=float(eig[-1]),output_condition=float(eig[-1]/eig[0]),
                       channel_strength=float(np.mean(abs(truth)**2)), Q=Q,C=C,delta_R_noise=R-R0,
                       surrogate_gap=R0-Ro,total_oracle_gap=R-Ro,R_noisy=R,R_cleanCV=R0,R_oracle=Ro,
                       identity_error_max=max(errors),normal_error=normal_error,oracle_LS_error=ls_error,
                       negative_penalty=bool(R-R0<0),negative_cross=bool(sr@M@n<0),numeric_borderline=bool(borderline),
                       fit_rank=rank,output_rank=output_rank,clean_coordinate_sha256=array_hash(c0),
                       pilot_window_sha256=array_hash(pilots[si,j-4:j+1]),prediction_sha256=array_hash(prediction),
                       current_noise_seed=noise_seed(i,r,j),window_noise_seeds=','.join(str(noise_seed(i,r,k)) for k in range(j-4,j+1)))
            row.update(cosine(sr,n,M))
            row.update({'clean_'+k:v for k,v in geometry(x[f'G0_{s}_{j}']).items()})
            nums=[v for v in row.values() if isinstance(v,(int,float)) and not isinstance(v,bool)]
            assert np.isfinite(nums).all()
            rows.append(row)
    return rows


def read_rows(path):
    with Path(path).open(encoding='utf8', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k,v in row.items():
            if v == '': row[k] = None
            elif v in ('True','False'): row[k] = v=='True'
            else:
                try: row[k] = float(v)
                except ValueError: pass
    return rows
