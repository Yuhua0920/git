"""SP-U2B diagnostic-only mathematics and guarded frozen-input access."""
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
TOL = 2e-10
HEAD = 'ca83982b1d0e4bdca0215a3ecbe3a7a8f8f8ec55'
POWER = 9.941134822523972
VARIANCES = {10: .9941134822523973, 20: .09941134822523973}
SNRS = (10, 20)
ACCESS = []
_GUARDED = False
ALLOWED = {
    't16e_pilot_cv_preflight.json', 't16e_frozen_operators.npz',
    't16f_preflight.json', 't16f_frozen_candidates.npz',
    't16g_factor_preflight.json', 't16g_frozen_module_factors.npz',
}


def output_guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    try:
        rel = path.relative_to(OUT.resolve())
    except ValueError:
        return
    leaf = rel.parts[0].lower()
    own = leaf.startswith('sp_u2b_')
    flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
    writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    allowed = own or (len(rel.parts) == 1 and leaf in ALLOWED and not writing)
    ACCESS.append(dict(path=str(rel).replace('\\', '/'), writing=writing, allowed=allowed))
    if not allowed:
        raise PermissionError('SP-U2B output firewall: ' + str(rel))


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
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    if path.exists():
        raise RuntimeError('immutable output exists: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csvsave(path, rows):
    path = Path(path)
    if path.exists():
        raise RuntimeError('immutable output exists: ' + str(path))
    if not rows:
        raise ValueError('empty CSV')
    fields = list(rows[0])
    if any(set(r) != set(fields) for r in rows):
        raise ValueError('heterogeneous CSV schema')
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def close(actual, reference):
    a, r = np.asarray(actual), np.asarray(reference)
    err = float(np.linalg.norm(a-r) / max(1., float(np.linalg.norm(r))))
    if not np.isfinite(err) or err > TOL:
        raise AssertionError(('numerical identity', err))
    return err


def seeds(stage, trajectory, frame=0):
    if stage not in ('smoke', 'development') or trajectory not in range(20 if stage == 'smoke' else 100) or frame not in range(10):
        raise ValueError('undeclared data allocation')
    base = 3500000000 + ('smoke', 'development').index(stage)*1000000 + trajectory*1000
    return dict(sionna_seed=base+1, torch_seed=base+2, numpy_seed=base+3,
                bits_seed_tx0=base+100+3*frame, bits_seed_tx1=base+101+3*frame,
                noise_seed=base+102+3*frame)


def frozen_method():
    """Read six allowlisted artifacts only; never a historical performance file."""
    from src.lowrank_modules import RankFactors
    from src.pilot_cv import PilotCV, frozen_folds
    from src.spatial_information import observation_geometry
    install_guard()
    arrays, proof = {}, {}
    parts = (('t16e_pilot_cv_preflight.json', 'operator_artifact'),
             ('t16f_preflight.json', 'candidate_artifact'),
             ('t16g_factor_preflight.json', 'factor_artifact'))
    factor_arrays = None
    for filename, key in parts:
        meta = load(OUT/filename)
        info = meta[key]
        path = ROOT/info['path']
        if sha(path) != info['sha256']:
            raise ValueError('frozen artifact byte drift')
        with np.load(path, allow_pickle=False) as f:
            value = {k: f[k].copy(order='K') for k in info['array_sha256']}
        if any(array_hash(value[k]) != h for k, h in info['array_sha256'].items()):
            raise ValueError('frozen component drift')
        for v in value.values():
            v.flags.writeable = False
        if key == 'factor_artifact':
            factor_arrays = value
        else:
            arrays.update(value)
        proof[filename] = dict(metadata_sha256=sha(OUT/filename), artifact_path=info['path'],
                               artifact_sha256=info['sha256'], array_sha256=info['array_sha256'])
    folds, definition = frozen_folds(observation_geometry().selected)
    if definition != load(OUT/'t16e_pilot_cv_preflight.json')['pilot_fold_definition']:
        raise ValueError('fold drift')
    cvs = {s: PilotCV(folds, arrays['source_mean_p'], tuple(arrays[f'KSource_{s}']),
                      tuple(arrays[f'DeltaKP_{s}']), tuple(arrays[f'DeltaKM_{s}'])) for s in SNRS}
    factors = {(m, s): RankFactors(factor_arrays[f'A_{m}_{s}'][:, :4], factor_arrays[f'Q_{m}_{s}'][:4])
               for m in ('P', 'M') for s in SNRS}
    assert arrays['source_mean_p'].shape == (64,) and arrays['source_mean_d'].shape == (3968,)
    assert all(f.rank == 4 for f in factors.values())
    return arrays, cvs, factors, dict(artifacts=proof, fold=definition, rank=4, window=5,
                                    no_refit=True, means='source', variances=VARIANCES, power=POWER)


def explicit_design(cv, pilots):
    designs, responses = [], []
    for pilot in pilots:
        u = pilot-cv.source_mean_p
        for f, (observed, hold) in enumerate(cv.folds):
            x = u[observed]
            designs.append(np.column_stack([cv.profile_delta_k[f]@x, cv.mobility_delta_k[f]@x]))
            responses.append(u[hold]-cv.source_k[f]@x)
    D, r = np.concatenate(designs), np.concatenate(responses)
    G, z = np.vstack([D.real, D.imag]), np.r_[r.real, r.imag]
    if len(pilots) == 5:
        assert G.shape == (640, 2) and z.shape == (640,)
    return G, z


def geometry(G):
    H = G.T@G
    singular = np.linalg.svd(G, compute_uv=False)
    eig = np.linalg.eigvalsh(H)
    norms = np.linalg.norm(G, axis=0)
    rank = int(np.linalg.matrix_rank(G))
    kG = float(np.linalg.cond(G))
    kH = float(np.linalg.cond(H))
    # No regularization or repair when rank deficient.
    cond_error = None if rank < 2 else abs(kH-kG*kG)/max(1., kH)
    if rank == 2 and cond_error > TOL:
        raise AssertionError('condition convention identity failed')
    return dict(rank_real=rank, sigma_min=float(singular[-1]), sigma_max=float(singular[0]),
                kappa_G=kG, gram_condition=kH, lambda_min=float(eig[0]), lambda_max=float(eig[-1]),
                column_norm_P=float(norms[0]), column_norm_M=float(norms[1]),
                rho=float(H[0, 1]/(norms[0]*norms[1])) if min(norms)>0 else None,
                determinant=float(np.linalg.det(H)), excitation_fro=float(np.linalg.norm(G)),
                condition_square_relative_error=cond_error,
                near_singular=bool(singular[-1] <= 64*np.finfo(float).eps*singular[0]))


def joint_diagnostics(G0, z0, G, z, c0, c):
    E, d = G-G0, z-z0
    e0 = z0-G0@c0
    rhs = G.T@(d-E@c0)+E.T@e0
    identity_error = close((G.T@G)@(c-c0), rhs)
    en = float(np.linalg.norm(E, 2))
    sf = np.linalg.svd(G0, compute_uv=False)
    sg = np.linalg.svd(G, compute_uv=False)
    forcing = float(np.linalg.norm(d-E@c0))
    bound = float((sg[0]*forcing + en*np.linalg.norm(e0))/sg[-1]**2)
    actual = float(np.linalg.norm(c-c0))
    tiny = 64*np.finfo(float).eps*max(1., float(np.linalg.norm(c)), float(np.linalg.norm(c0)))
    return dict(E_spectral=en, E_fro=float(np.linalg.norm(E)),
                relative_E_spectral=en/sf[0], relative_E_fro=float(np.linalg.norm(E)/np.linalg.norm(G0)),
                weyl_margin=float(sf[-1]-en), weyl_positive=bool(sf[-1]>en),
                response_perturbation_norm=float(np.linalg.norm(d)), clean_residual_norm=float(np.linalg.norm(e0)),
                joint_forcing_norm=forcing, exact_score_change_norm=float(np.linalg.norm(rhs)),
                joint_bound=bound, bound_actual_ratio=None if actual<=tiny else bound/actual,
                near_zero_coordinate_change=bool(actual<=tiny), near_zero_threshold=tiny,
                joint_identity_error=identity_error, bound_failure=bool(actual>bound+TOL*max(1.,bound)))


def channel_oracle(truth, base, X):
    """Truth-only diagnostic. Never passed to production fitting."""
    M = (X.conj().T@X).real
    b = (X.conj().T@(truth-base)).real
    c = np.linalg.solve(M, b)
    direct = np.linalg.lstsq(np.vstack([X.real, X.imag]), np.r_[(truth-base).real, (truth-base).imag], rcond=None)[0]
    err = close(c, direct)
    return c, M, err


def pair_channel(h, stage, trajectory):
    from src.cdl_benchmark import frames
    from src.spatial_information import observation_geometry
    from src.link import generate_master_qpsk, assemble_pilot_data_frame, apply_frequency_mimo_channel, PILOT_SYMBOLS
    from src.pilots import map_pilot_symbols
    from src.estimators import pilot_ls_estimate
    from src.numerics import complex_gaussian_noise
    g = observation_geometry()
    known = map_pilot_symbols(g.layout, PILOT_SYMBOLS)
    pilots = np.empty((3, 10, 64), complex)
    hashes, signal, noise, samples = [], np.zeros(2), np.zeros(2), 0
    clean_reconstruction_errors = []
    for j, truth in enumerate(frames(h)):
        seed = seeds(stage, trajectory, j)
        bits, symbols = generate_master_qpsk(bits_seeds=(seed['bits_seed_tx0'], seed['bits_seed_tx1']))
        tx = assemble_pilot_data_frame(symbols, g.layout)
        useful = apply_frequency_mimo_channel(truth, tx)
        unit = complex_gaussian_noise((2, 64, 16), 1., np.random.default_rng(seed['noise_seed']))
        for si, sigma2 in enumerate((0., VARIANCES[10], VARIANCES[20])):
            y = useful + np.sqrt(sigma2)*unit
            sparse, _ = pilot_ls_estimate(y, known, g.layout.pilot_mask)
            pilots[si, j] = sparse.ravel()[g.selected]
            hashes.append(dict(stage=stage, trajectory_id=trajectory, frame_id=j, variant=('clean','10dB','20dB')[si],
                               sigma2=sigma2, H_sha256=array_hash(truth), bits_sha256=array_hash(bits),
                               symbols_sha256=array_hash(symbols), transmitted_sha256=array_hash(tx),
                               useful_sha256=array_hash(useful), unit_noise_sha256=array_hash(unit),
                               received_sha256=array_hash(y), pilot_sha256=array_hash(pilots[si,j])))
        clean_reconstruction_errors.append(close(pilots[0,j], truth.ravel()[g.selected]))
        close((pilots[1,j]-pilots[0,j])/np.sqrt(VARIANCES[10]),
              (pilots[2,j]-pilots[0,j])/np.sqrt(VARIANCES[20]))
        if j >= 4:
            mask = g.layout.data_mask[0]
            signal += np.sum(abs(useful[:,mask])**2, axis=1)
            noise += np.sum(abs(unit[:,mask])**2, axis=1)
            samples += int(mask.sum())
    pilots.flags.writeable=False
    return pilots, hashes, dict(signal_energy_per_rx=signal.tolist(), unit_noise_energy_per_rx=noise.tolist(),
                                useful_samples_per_rx=samples,
                                clean_reconstruction_error_max=max(clean_reconstruction_errors))
