"""MR08A ten frozen estimators; preparation-only seven-SNR instantiation.

No trajectory generation, training, label/truth interface, method selection or
historical write occurs here. The caller freezes this module and the returned
asset/source identities before generating independent confirmation data.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
from time import perf_counter

import numpy as np

SNRS = (0, 5, 10, 15, 20, 25, 30)
FRAMES = tuple(range(4, 10))
METHODS = ('SOURCE', 'POOLED_NATURAL', 'PPCA_DIRECT_R32', 'GMM_PREPARED_3',
           'HARD_TW5', 'CONVEX_TW5', 'LR4_TW1', 'LR4_TW5', 'DENSE_TW5',
           'WHOLE_POOL_M2_R24')
POWER = 9.941134822523972
OLD_SNRS = (10, 20, 30)
ATOL, RTOL, SOLVE_TOL = 1e-10, 1e-10, 1e-11
STATS = ('t15a_cdl_statistics_500.npz', 't15c_cdl_statistics_CDL-B-30.npz',
         't15c_cdl_statistics_CDL-C-120.npz')
STATS_SHA = ('ce9f2d0f567e20ff98fa88e098a270bec47dafb5839e14356a98282a514b4b69',
             '8b43842c3b3738b4e403bc927b1327eb29467db9767780dd789cd080c8b81b4b',
             '8230e818870dfb6a58e95edbfc82c8ddd17748ecce7bd82417a977e92e1491e1')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def _source_module(name, path):
    """Load pure mathematical modules from their real new/historical paths."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses need their registered module
    spec.loader.exec_module(module)
    return module


def dependencies(root):
    root = Path(root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    ppca = _source_module('_mr08a_frozen_ppca', root / 'outputs/revision_mr03_ppca_capacity_controls/src_or_scripts/ppca.py')
    gmm = _source_module('_mr08a_frozen_gmm3', root / 'outputs/revision_mr03c_prepared_gmm/src/gmm3.py')
    controls = _source_module('_mr08a_frozen_mr05_controls', root / 'outputs/revision_mr05_compression_controls/src_or_scripts/controls.py')
    return ppca, gmm, controls


def close(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('FINITE_SHAPE:' + label)
    error = float(np.max(abs(a - b), initial=0))
    scale = max(float(np.max(abs(a), initial=0)), float(np.max(abs(b), initial=0)))
    if error > ATOL + RTOL * scale:
        raise ValueError('ASSET_COMPARISON:' + label)
    return dict(label=label, max_absolute=error, relative_to_max=error / max(scale, np.finfo(float).tiny),
                array_equal=bool(np.array_equal(a, b)), atol=ATOL, rtol=RTOL)


def prepare_assets(root, out):
    """Cache fixed model/statistics maps; no prediction or training is run."""
    root, out = Path(root).resolve(), Path(out).resolve()
    cache = out / 'data/estimators_7snr.npz'
    report_path = out / 'reports/model_assets.json'
    if cache.exists() or report_path.exists():
        raise FileExistsError('MODEL_CACHE_OR_REPORT_EXISTS')
    cache.parent.mkdir(parents=True, exist_ok=True)
    ppca, gmm, controls = dependencies(root)
    from src.cdl_benchmark import FrozenStatistics, observation_geometry
    from src.lowrank_modules import RankFactors
    from src.pilot_cv import frozen_folds
    begin = perf_counter()
    mr1 = root / 'outputs/revision_mr01_pooled_lmmse'
    mr3 = root / 'outputs/revision_mr03_ppca_capacity_controls'
    mr5 = root / 'outputs/revision_mr05_compression_controls'
    source_assets = {}

    def authenticate(path, expected):
        actual = sha(path)
        if actual != expected:
            raise ValueError('FROZEN_MODEL_OR_STATISTIC_HASH:' + str(path))
        source_assets[Path(path).relative_to(root).as_posix()] = actual

    b2path = root / 'outputs/sp_u12b2a/operators.npz'
    authenticate(b2path, read(root / 'outputs/sp_u12b2a/operator_anchor.json')['operator_archive']['sha256'])
    oldpath = mr1 / 'data/operators.npz'
    authenticate(oldpath, read(mr1 / 'reports/execution_freeze.json')['operator_sha256'])
    factor_path = mr5 / 'data/control_factors.npz'
    authenticate(factor_path, read(mr5 / 'reports/factor_construction.json')['cache']['sha256'])
    direct_path = mr3 / 'data/PPCA_DIRECT_R32.npz'
    authenticate(direct_path, read(mr3 / 'reports/PPCA_DIRECT_R32.json')['model_sha256'])
    pool_path = mr1 / 'data/pooled_statistics.npz'
    protected = read(mr3 / 'reports/preservation_before.json')['hashes']
    authenticate(pool_path, protected[pool_path.relative_to(root).as_posix()])
    spectrum_path = mr3 / 'data/shared_spectrum.npz'
    # Indices are derived from the frozen geometry and checked against saved
    # actual joint4032 selectors. No spectrum or PPCA model is refitted.
    geom = observation_geometry()
    selected, data = geom.selected, np.flatnonzero(geom.data)
    joint = np.union1d(selected, data)
    pidx, didx = np.searchsorted(joint, selected), np.searchsorted(joint, data)
    with np.load(spectrum_path, allow_pickle=False) as z:
        if not all(np.array_equal(z[k], v) for k, v in
                   [('joint_full', joint), ('pilot_positions', pidx), ('data_positions', didx)]):
            raise ValueError('PPCA_JOINT_INDEX_DRIFT')
    source_assets[spectrum_path.relative_to(root).as_posix()] = sha(spectrum_path)
    if len(joint) != 4032 or len(selected) != 64 or len(data) != 3968:
        raise ValueError('FROZEN_GEOMETRY_DIMENSION')
    stats = []
    for name, expected in zip(STATS, STATS_SHA):
        path = root / 'outputs' / name
        authenticate(path, expected)
        with np.load(path, allow_pickle=False) as z:
            if int(z['samples']) != 5000 or not np.array_equal(z['train_trajectory_ids'], np.arange(500)):
                raise ValueError('PREPARATION_POPULATION')
            stats.append({k: z[k] for k in ('mean_p', 'mean_d', 'rpp', 'rdp')})
    with np.load(pool_path, allow_pickle=False) as z:
        pool = {k: z[k] for k in ('mean_p', 'mean_d', 'rpp', 'rdp')}
        if int(z['samples']) != 15000:
            raise ValueError('POOLED_POPULATION')
    pool_stat = FrozenStatistics(pool['mean_p'], pool['mean_d'], pool['rpp'], pool['rdp'], (), 15000)
    with np.load(direct_path, allow_pickle=False) as z:
        if int(z['declared_rank']) != 32 or int(z['effective_rank']) != 32 or z['loading'].shape != (4032, 32):
            raise ValueError('DIRECT32_STRUCTURE')
        model = ppca.PPCAModel(z['mean'], z['loading'], float(z['nu']),
                              dict(origin='original MR03 saved direct32'))
    if model.nu != 1e-6 or model.rank != 32:
        raise ValueError('DIRECT32_RESIDUAL_OR_RANK_DRIFT')
    oldoperators = np.load(oldpath, allow_pickle=False)
    oldfactors = np.load(factor_path, allow_pickle=False)
    olddirect = np.load(mr3 / 'data/conditional_operators.npz', allow_pickle=False)
    authenticate(mr3 / 'data/conditional_operators.npz', read(mr3 / 'reports/spectral_construction.json')['operators_sha256'])
    op = {}
    diagnostics, checks = [], []
    with np.load(b2path, allow_pickle=False) as z:
        for key in ('source_mean_p', 'source_mean_d'):
            op[key] = z[key]
        for s in SNRS:
            for prefix in ('Wsource', 'WP', 'WM', 'DeltaWP', 'DeltaWM', 'KSource', 'DeltaKP', 'DeltaKM',
                           'A_P', 'Q_P', 'A_M', 'Q_M'):
                op[f'{prefix}_{s}'] = z[f'{prefix}_{s}']
    op['pool_mean_p'], op['pool_mean_d'] = pool['mean_p'], pool['mean_d']
    op['pilot_positions'], op['data_positions'], op['joint_full'] = pidx, didx, joint
    folds, fold_definition = frozen_folds(selected)
    for s in SNRS:
        sigma = float(POWER / 10 ** (s / 10))
        pooled = pool_stat.prepare(sigma)
        if pooled.solve_relative_residual > SOLVE_TOL:
            raise ValueError('POOLED_CONDITIONAL_SOLVE')
        co = ppca.conditional_operator(model, pidx, didx, sigma)
        if co.audit['solve_relative_residual'] > SOLVE_TOL:
            raise ValueError('DIRECT_CONDITIONAL_SOLVE')
        origins = [dict(name=n, mu_p=st['mean_p'], mu_d=st['mean_d'], R_pp=st['rpp'], R_dp=st['rdp'])
                   for n, st in zip(('Source', 'Profile', 'Mobility'), stats)]
        mixture = gmm.prepare_gaussian_mixture(origins, sigma)
        if not np.array_equal(mixture.priors, np.full(3, 1 / 3)):
            raise ValueError('GMM_PRIOR_DRIFT')
        op[f'Wpool_{s}'] = oldoperators[f'Wpool_{s}'] if s in OLD_SNRS else pooled.weight
        key = f'PPCA_DIRECT_R32_{s}'
        op[f'Direct_W_{s}'] = olddirect[key + '_W'] if s in OLD_SNRS else co.W
        op[f'Direct_mean_p_{s}'], op[f'Direct_mean_d_{s}'] = co.mean_p, co.mean_d
        if s in OLD_SNRS:
            checks.append(close(pooled.weight, oldoperators[f'Wpool_{s}'], f'pooled_{s}'))
            checks.append(close(co.W, olddirect[key + '_W'], f'direct32_{s}'))
            for prefix in ('Wsource', 'WP', 'WM', 'DeltaWP', 'DeltaWM', 'KSource', 'DeltaKP', 'DeltaKM',
                           'A_P', 'Q_P', 'A_M', 'Q_M'):
                checks.append(close(op[f'{prefix}_{s}'], oldoperators[f'{prefix}_{s}'], f'original_{prefix}_{s}'))
        shift = pool['mean_p'] - op['source_mean_p']
        Cwhole = pool['rpp'] + sigma * np.eye(64, dtype=np.complex128) + np.outer(shift, shift.conj())
        for qi, label in enumerate(('source', 'P', 'M')):
            if s in OLD_SNRS:
                base = f'WHOLE_POOL_M2_R24_{s}_{qi}'
                checks.append(close(Cwhole, oldfactors[base + '_C'], base + '_weight'))
                f = RankFactors(oldfactors[base + '_A'], oldfactors[base + '_Q'])
                audit = dict(reused_actual_frozen_factors=True, rank=24, weight='common Pooled source-referenced M2')
            else:
                f, audit = controls.weighted_factor(op[f'W{label}_{s}'], Cwhole, 24)
                audit['reused_actual_frozen_factors'] = False
            op[f'Whole_{s}_{qi}_A'], op[f'Whole_{s}_{qi}_Q'] = f.A, f.Q
            c = mixture.components[qi]
            close(c.W, op[f'W{label}_{s}'], f'GMM_component_conditional_{label}_{s}')
            op[f'GMM_{s}_{qi}_W'], op[f'GMM_{s}_{qi}_C'] = c.W, c.C
            op[f'GMM_{s}_{qi}_L'], op[f'GMM_{s}_{qi}_mu_p'], op[f'GMM_{s}_{qi}_mu_d'] = c.chol_lower, c.mu_p, c.mu_d
            op[f'GMM_{s}_{qi}_logdet'] = np.array(c.logdet)
            diagnostics.append(dict(snr_db=s, sigma2=sigma, origin=label, gmm=c.audit, whole24=audit))
    for archive in (oldoperators, oldfactors, olddirect):
        archive.close()
    for key, value in op.items():
        if not np.isfinite(value).all():
            raise ValueError('NONFINITE_PREPARED_CACHE:' + key)
        if value.dtype.kind == 'c' and value.dtype != np.complex128:
            raise ValueError('CACHE_COMPLEX128_REQUIRED:' + key)
    with cache.open('xb') as f:
        np.savez_compressed(f, **op)
    sources = ('src/pilot_cv.py', 'src/statistics_mixing.py', 'src/frozen_affine_validation.py',
               'src/lowrank_modules.py', 'src/cdl_benchmark.py',
               'outputs/revision_mr05_compression_controls/src_or_scripts/controls.py',
               'outputs/revision_mr03_ppca_capacity_controls/src_or_scripts/ppca.py',
               'outputs/revision_mr03c_prepared_gmm/src/gmm3.py')
    report = dict(PASS=True, methods=METHODS, snrs=SNRS, frames=FRAMES,
                  cache_path=cache.relative_to(out).as_posix(), cache_SHA256=sha(cache), cache_bytes=cache.stat().st_size,
                  source_assets=source_assets, source_code={p: sha(root / p) for p in sources},
                  fold_definition=fold_definition, diagnostics=diagnostics,
                  original_10_20_30_comparisons=checks, instantiation_seconds=perf_counter() - begin,
                  original_source_LR4_seven_SNR_cache_reused=True,
                  direct32_model_retrained=False, preparation_moments_reestimated=False,
                  new_SNR_maps_instantiated_from_fixed_assets=[0, 5, 15, 25],
                  old_whole24_factors_reused_at=list(OLD_SNRS),
                  window_Tw1_frames=1, window_Tw5_frames=5, truth_or_labels_in_predict=False,
                  predictions_run_in_preparation=0, sigma2_formula='9.941134822523972 / 10**(SNR/10)',
                  center_rules=dict(source_affine_hard_convex_whole='public Source pilot/data means',
                                    pooled='natural Pooled means', direct32='natural saved Direct32 means',
                                    GMM='three natural prepared-origin means'))
    write(report_path, report)
    return report


class TenMethodBank:
    """Current-pilot statistical maps and untouched frozen 1/5-frame fitting."""
    def __init__(self, arrays, cvs, factors, whole, gmm, controls, deployed):
        self.arrays, self.cvs, self.factors, self.whole = arrays, cvs, factors, whole
        self.gmm, self.controls, self.deployed = gmm, controls, deployed

    def predict(self, pilots):
        """Return paired estimates; enrich real failures, never retry or recover."""
        try:
            return self._predict_impl(pilots)
        except Exception as exc:
            context = dict(snr_db=None, frame_id=None, method='INPUT_OR_INITIALIZATION')
            trace = exc.__traceback__
            while trace is not None:
                local = trace.tb_frame.f_locals
                if 's' in local and isinstance(local['s'], (int, np.integer)):
                    context['snr_db'] = int(local['s'])
                if 'frame' in local and isinstance(local['frame'], (int, np.integer)):
                    context['frame_id'] = int(local['frame'])
                if 'method_context' in local:
                    context['method'] = str(local['method_context'])
                function = trace.tb_frame.f_code.co_name
                if function in ('hard_selection', 'convex_mixing'):
                    context['method'] = {'hard_selection':'HARD_TW5', 'convex_mixing':'CONVEX_TW5'}[function]
                if function == 'fit' and 'modules' in local and context['method'].startswith('FROZEN_TW5'):
                    context['method'] = 'TW5_SHARED_AFFINE_DENSE_LR4_WHOLE24'
                trace = trace.tb_next
            raise ValueError('PREDICTION_FAILURE:' + json.dumps(context, sort_keys=True) + ':' + str(exc)) from exc

    def _predict_impl(self, pilots):
        p = np.asarray(pilots)
        if p.shape != (7, 10, 64) or p.dtype != np.complex128 or not np.isfinite(p).all():
            raise ValueError('SEVEN_SNR_TEN_FRAME_COMPLEX128_PILOTS_REQUIRED')
        predictions = np.empty((7, len(METHODS), 6, 3968), np.complex128)
        coordinates5 = np.empty((7, 6, 2)); coordinates1 = np.empty_like(coordinates5)
        S5 = np.empty((7, 6, 2, 2)); b5 = np.empty((7, 6, 2)); n5 = np.empty((7, 6))
        S1 = np.empty_like(S5); b1 = np.empty_like(b5); n1 = np.empty_like(n5)
        hard = np.empty((7, 6), np.int64); convex = np.empty_like(coordinates5)
        weights = np.empty((7, 6, 3)); entropy = np.empty((7, 6)); maxw = np.empty_like(entropy)
        for si, s in enumerate(SNRS):
            cv = self.cvs[s]
            method_context = 'FROZEN_TW5_HARD_CONVEX_DENSE_LR4'
            rows = self.deployed(p[si], self.arrays, s, cv, self.factors['P', s], self.factors['M', s])
            if [r['frame_id'] for r in rows] != list(FRAMES):
                raise ValueError('ORIGINAL_FIVE_FRAME_OUTPUT_IDENTITIES')
            method_context = 'GMM_PREPARED_3'
            mix = self.gmm[s].predict(p[si, 4:10])
            predictions[si, METHODS.index('GMM_PREPARED_3')] = mix.estimate
            weights[si], entropy[si], maxw[si] = mix.posterior_weights, mix.posterior_entropy_nats, mix.max_posterior_weight
            for ri, row in enumerate(rows):
                frame = FRAMES[ri]
                if row['window_start'] != frame - 4 or row['dense_coefficients'] is not row['lr4_coefficients']:
                    raise ValueError('ORIGINAL_WINDOW_OR_SHARED_COORDINATE_DRIFT')
                c5, d5 = row['coefficients']['affine'], row['diagnostics']['affine']
                method_context = 'LR4_TW1'
                c1, d1 = cv.fit(p[si, frame:frame + 1], modules=('P', 'M'))
                for c, d, coordinates, gram, rhs, normal in ((c5, d5, coordinates5, S5, b5, n5),
                                                           (c1, d1, coordinates1, S1, b1, n1)):
                    if d['clipped'] or d['regularized'] or d['normal_equation_relative_residual'] > SOLVE_TOL:
                        raise ValueError('ORIGINAL_UNRESTRICTED_COORDINATE_SOLVE')
                    coordinates[si, ri], gram[si, ri], rhs[si, ri], normal[si, ri] = c, d['gram'], d['rhs'], d['normal_equation_relative_residual']
                mapping = {'source': 'SOURCE', 'hard': 'HARD_TW5', 'convex': 'CONVEX_TW5',
                           'dense-affine': 'DENSE_TW5', 'lr4-affine': 'LR4_TW5'}
                for old, new in mapping.items():
                    predictions[si, METHODS.index(new), ri] = row['predictions'][old]
                current = p[si, frame]
                u = current - self.arrays['source_mean_p']
                method_context = 'POOLED_NATURAL'
                predictions[si, METHODS.index('POOLED_NATURAL'), ri] = self.controls.deployed('POOLED_NATURAL', u, c5, self.arrays, self.whole[s], s)
                method_context = 'WHOLE_POOL_M2_R24'
                predictions[si, METHODS.index('WHOLE_POOL_M2_R24'), ri] = self.controls.deployed('WHOLE_POOL_M2_R24', u, c5, self.arrays, self.whole[s], s)
                from src.lowrank_modules import factored_affine_predict
                method_context = 'LR4_TW1'
                predictions[si, METHODS.index('LR4_TW1'), ri] = factored_affine_predict(current, self.arrays['source_mean_p'], self.arrays['source_mean_d'], self.arrays[f'Wsource_{s}'], c1, self.factors['P', s], self.factors['M', s])
                method_context = 'PPCA_DIRECT_R32'
                predictions[si, METHODS.index('PPCA_DIRECT_R32'), ri] = self.arrays[f'Direct_mean_d_{s}'] + self.arrays[f'Direct_W_{s}'] @ (current - self.arrays[f'Direct_mean_p_{s}'])
                hard[si, ri], convex[si, ri] = row['diagnostics']['hard']['selected_index'], row['coefficients']['convex']
        if not np.isfinite(predictions).all() or not all(np.isfinite(x).all() for x in (coordinates5, coordinates1, S5, b5, n5, S1, b1, n1, weights, entropy, maxw, convex)):
            raise ValueError('NONFINITE_PREDICTION_OR_DIAGNOSTIC')
        return dict(predictions=predictions, methods=METHODS, snrs=SNRS, frame_ids=FRAMES,
                    coordinates_tw5=coordinates5, coordinates_tw1=coordinates1,
                    gram_tw5=S5, rhs_tw5=b5, normal_residual_tw5=n5,
                    gram_tw1=S1, rhs_tw1=b1, normal_residual_tw1=n1,
                    hard_selected_index=hard, convex_coordinates=convex,
                    gmm_posterior_weights=weights, gmm_entropy_nats=entropy,
                    gmm_max_posterior_weight=maxw)


def load_assets(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    report = read(out / 'reports/model_assets.json')
    if not report['PASS'] or report['methods'] != list(METHODS) or report['snrs'] != list(SNRS):
        raise ValueError('FROZEN_METHOD_INVENTORY')
    if sha(out / report['cache_path']) != report['cache_SHA256']:
        raise ValueError('MODEL_CACHE_HASH')
    for p, h in report['source_code'].items():
        if sha(root / p) != h:
            raise ValueError('ORIGINAL_SOURCE_CHANGED:' + p)
    for p, h in report['source_assets'].items():
        if sha(root / p) != h:
            raise ValueError('ORIGINAL_FROZEN_ASSET_CHANGED:' + p)
    _, gmm, controls = dependencies(root)
    from src.pilot_cv import PilotCV, frozen_folds
    from src.cdl_benchmark import observation_geometry
    from src.lowrank_modules import RankFactors
    from src.frozen_affine_validation import deployed
    with np.load(out / report['cache_path'], allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    for array in arrays.values():
        array.flags.writeable = False
    folds, _ = frozen_folds(observation_geometry().selected)
    cvs = {s: PilotCV(folds, arrays['source_mean_p'], tuple(arrays[f'KSource_{s}']),
                      tuple(arrays[f'DeltaKP_{s}']), tuple(arrays[f'DeltaKM_{s}'])) for s in SNRS}
    factors = {(q, s): RankFactors(arrays[f'A_{q}_{s}'], arrays[f'Q_{q}_{s}']) for s in SNRS for q in ('P', 'M')}
    whole = {s: {'WHOLE_POOL_M2_R24': tuple(RankFactors(arrays[f'Whole_{s}_{q}_A'], arrays[f'Whole_{s}_{q}_Q']) for q in range(3))} for s in SNRS}
    models = {}
    for s in SNRS:
        comps = tuple(gmm.GaussianComponent(name, arrays[f'GMM_{s}_{q}_mu_p'], arrays[f'GMM_{s}_{q}_mu_d'],
                         arrays[f'GMM_{s}_{q}_C'], arrays[f'GMM_{s}_{q}_L'], arrays[f'GMM_{s}_{q}_W'],
                         float(arrays[f'GMM_{s}_{q}_logdet']), {}) for q, name in enumerate(('Source', 'Profile', 'Mobility')))
        prior = np.full(3, 1 / 3); prior.flags.writeable = False
        logprior = np.log(prior); logprior.flags.writeable = False
        models[s] = gmm.PreparedGaussianMixture(comps, prior, logprior, float(POWER / 10 ** (s / 10)), {})
    bank = TenMethodBank(arrays, cvs, factors, whole, models, controls, deployed)
    bank.asset_report = report
    return bank
