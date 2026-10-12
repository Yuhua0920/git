"""Portable package-only numerical review of MR08A sufficient statistics.

This does NOT regenerate input tensors, run inference, draw bootstrap samples,
or read omitted model/indices/bootstrap-linear files. Their full-project review
is explicitly referenced by the supplied original certification SHA256. The
package reproduces pooled energies/points and adjusted intervals from included
saved primary-difference arrays, not the omitted index-to-statistic calculation.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import tempfile
import zipfile

import numpy as np

CONDITIONS = ('ID', 'Profile', 'Mobility', 'Composed', 'Boundary')
SNRS = (0, 5, 10, 15, 20, 25, 30)
METHODS = ('SOURCE', 'POOLED_NATURAL', 'PPCA_DIRECT_R32', 'GMM_PREPARED_3',
           'HARD_TW5', 'CONVEX_TW5', 'LR4_TW1', 'LR4_TW5', 'DENSE_TW5',
           'WHOLE_POOL_M2_R24')
FAMILIES = (('A', 'POOLED_NATURAL', 'LR4_TW5'),
            ('B', 'GMM_PREPARED_3', 'LR4_TW1'),
            ('C', 'LR4_TW5', 'WHOLE_POOL_M2_R24'))
TAIL = .05 / (2 * 35)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def path_in(out, relative):
    path = (out / relative).resolve()
    path.relative_to(out)
    return path


def csv_index(path, keys):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    index = {tuple(r[k] for k in keys): r for r in rows}
    if len(rows) != len(index):
        raise ValueError('DUPLICATE_CSV_IDENTITY')
    return index


def close(actual, expected, label, report):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('FINITE_SHAPE:' + label)
    error = float(np.max(abs(a - b), initial=0))
    scale = max(float(np.max(abs(a), initial=0)), float(np.max(abs(b), initial=0)))
    if error > 1e-10 + 1e-10 * scale:
        raise ValueError('PACKAGE_NUMERICAL_MISMATCH:' + label)
    report['numerical_comparisons'] += 1
    report['maximum_absolute_difference'] = max(report['maximum_absolute_difference'], error)
    report['maximum_scaled_difference'] = max(report['maximum_scaled_difference'], error / (1 + scale))


def sorted_linear_quantiles(values):
    """Independent explicit order-statistic interpolation, not np.quantile."""
    x = np.asarray(values)
    if x.dtype != np.float64 or x.ndim != 3 or x.shape[1:] != (7, 3) or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('PRIMARY_DIFFERENCE_FINITE_FLOAT64_AXES')
    ordered = np.sort(x, axis=0)
    bounds = []
    for probability in (TAIL, 1 - TAIL):
        location = (len(x) - 1) * probability
        lower, upper = math.floor(location), math.ceil(location)
        fraction = location - lower
        bounds.append(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))
    return np.array(bounds)


def pooled_from_frames(fe, fq):
    """Independent compensated scalar sums across frames/trajectories."""
    fe, fq = np.asarray(fe), np.asarray(fq)
    if fe.dtype != np.float64 or fq.dtype != np.float64 or fe.ndim != 4 or fe.shape[1:] != (7, 10, 6) or fq.shape != (len(fe), 6):
        raise ValueError('RAW_ENERGY_FLOAT64_AXES')
    if not np.isfinite(fe).all() or not np.isfinite(fq).all() or np.any(fe < 0) or np.any(fq <= 0):
        raise ValueError('RAW_ENERGY_FINITE_OR_SIGN')
    numerator = np.array([[math.fsum(float(v) for v in fe[:, si, mi, :].ravel())
                           for mi in range(10)] for si in range(7)])
    denominator = math.fsum(float(v) for v in fq.ravel())
    linear = numerator / denominator
    if np.any(linear <= 0) or not np.isfinite(linear).all():
        raise ValueError('POOLED_NMSE_INVALID')
    db = 10 * np.log10(linear)
    difference = np.stack([db[:, METHODS.index(a)] - db[:, METHODS.index(b)]
                           for _, a, b in FAMILIES], axis=-1)
    return numerator, denominator, linear, db, difference


def _review(out, certification_sha256, *, n=1000, replicates=100000, toy=False):
    out = Path(out).resolve()
    certification_path = out / 'reports/independent_numerical_review.json'
    if sha(certification_path) != certification_sha256:
        raise ValueError('FULL_PROJECT_CERTIFICATION_SHA256_MISMATCH')
    certified = js(certification_path)
    if certified.get('PASS') is not True or certified.get('mode') != 'formal' or certified.get('physical_trajectories') != 5 * n or certified.get('bootstrap_replicates_per_condition') != replicates:
        raise ValueError('FULL_PROJECT_CERTIFICATION_SCOPE')
    if certified.get('new_draws') != 0 or certified.get('new_predictions') != 0 or certified.get('new_training') != 0:
        raise ValueError('FULL_PROJECT_REVIEW_NOT_READONLY')
    if not toy and certified.get('toy_synthetic_only') is True:
        raise ValueError('SYNTHETIC_CERTIFICATION_FORBIDDEN')
    for c in CONDITIONS:
        z = certified['conditions'][c]
        if z['physical_trajectories'] != n or z['bootstrap_draws_checked'] != replicates or z['all105family_endpoints_checked'] is not True:
            raise ValueError('FULL_PROJECT_ACTUAL_INDEX_REVIEW_INCOMPLETE')
    protocol = js(out / 'protocol.json')
    raw = js(out / 'reports/formal_receipt.json')
    analysis = js(out / 'reports/formal_analysis.json')
    if (tuple(protocol['methods']) != METHODS or tuple(protocol['snrs']) != SNRS
            or tuple(protocol['scored_frames']) != tuple(range(4, 10))
            or protocol['per_condition'] != n or protocol['physical_trajectories'] != 5 * n
            or protocol['bootstrap']['replicates'] != replicates
            or protocol['bootstrap']['tail_probability'] != TAIL):
        raise ValueError('FROZEN_PROTOCOL_SCOPE')
    for family, a, b in FAMILIES:
        p = protocol['primary_families'][family]
        if p['first'] != a or p['second'] != b or p['comparisons'] != 35:
            raise ValueError('FROZEN_FAMILY_DEFINITION')
    if raw.get('PASS') is not True or raw.get('identity_PASS') is not True or any(raw[k] for k in ('failures', 'nonfinite', 'missing', 'substituted')) or raw['smoke_mixed_into_formal'] is not False:
        raise ValueError('RAW_RECEIPT_FAILED_OR_INCOMPLETE')
    if (analysis['PASS'] is not True or analysis['mode'] != 'formal' or analysis['bootstrap_replicates'] != replicates
            or analysis['physical_trajectories'] != 5 * n or analysis['NMSE_rows'] != 350 or analysis['primary_rows'] != 105
            or analysis['no_combined105_familywise_guarantee'] is not True
            or analysis['raw_receipt_SHA256'] != sha(out / 'reports/formal_receipt.json')):
        raise ValueError('FROZEN_ANALYSIS_SCOPE_OR_BINDING')
    point_rows = csv_index(out / 'tables/formal_nmse.csv', ('condition', 'snr_db', 'method'))
    contrast_rows = csv_index(out / 'tables/formal_primary_contrasts.csv', ('family', 'condition', 'snr_db'))
    if set(point_rows) != {(c, str(s), m) for c in CONDITIONS for s in SNRS for m in METHODS} or set(contrast_rows) != {(f, c, str(s)) for f, _, _ in FAMILIES for c in CONDITIONS for s in SNRS}:
        raise ValueError('EXACT_350_NMSE_105_CONTRAST_GRID')
    entries = raw['receipts']
    index = {(r['condition'], int(r['trajectory_id'])): r for r in entries}
    if len(entries) != 5 * n or set(index) != {(c, i) for c in CONDITIONS for i in range(n)}:
        raise ValueError('EXACT_RAW_IDENTITY_GRID')
    records = {r['condition']: r for r in analysis['records']}
    if len(analysis['records']) != 5 or set(records) != set(CONDITIONS):
        raise ValueError('EXACT_BOOTSTRAP_CONDITION_GRID')
    historical = set(js(out / 'data/historical_H_hashes.json')['H_sha256'])
    smoke = js(out / 'reports/smoke_receipt.json')
    if smoke.get('PASS') is not True or smoke.get('identity_PASS') is not True:
        raise ValueError('SMOKE_IDENTITY_RECEIPT_NOT_PASS')
    forbidden = historical | {r['H_sha256'] for r in smoke['receipts']}
    seen_H, seen_identity = set(), set()
    protocol_hash = sha(out / 'protocol.json')
    registry_hash = sha(out / 'data/formal_seed_registry.json')
    registry = js(out / 'data/formal_seed_registry.json')
    seed_rows = [r for r in registry['rows'] if r['group'] == 'EVAL']
    seed_index = {(r['condition'], int(r['trajectory_id']), int(r['frame_id']), r['role']): int(r['seed']) for r in seed_rows}
    channel_roles = ('sionna_seed', 'torch_seed', 'numpy_seed')
    communication_roles = ('bits_seed_tx0', 'bits_seed_tx1', 'noise_seed')
    expected_seed_keys = {(c, i, -1, role) for c in CONDITIONS for i in range(n) for role in channel_roles}
    expected_seed_keys |= {(c, i, j, role) for c in CONDITIONS for i in range(n) for j in range(10) for role in communication_roles}
    if len(seed_rows) != len(seed_index) or set(seed_index) != expected_seed_keys:
        raise ValueError('FROZEN_SEED_REGISTRY_GRID')
    report = dict(PASS=False, scope='PACKAGE_ONLY', toy_synthetic_only=toy,
                  review_code_SHA256=sha(__file__), full_project_certification_path='reports/independent_numerical_review.json',
                  full_project_certification_SHA256=certification_sha256,
                  full_project_actual_index_review_referenced=True,
                  package_actual_bootstrap_indices_read=False, package_bootstrap_linear_read=False,
                  package_model_cache_read=False, input_tensors_regenerated=False,
                  bootstrap_distributions_recomputed_from_indices=False,
                  package_interval_route='Included saved primary differences; independently sorted-order linear interpolation',
                  point_route='Compensated scalar sums of all included raw frame energies',
                  new_predictions=0, new_draws=0, new_training=0,
                  numerical_comparisons=0, maximum_absolute_difference=0., maximum_scaled_difference=0.,
                  primary_families_separate=True, no_combined105_familywise_guarantee=True,
                  omitted_full_project_artifacts=[], conditions={})
    model_record = protocol.get('models', {})
    model_path = model_record.get('cache_path', 'data/estimators_7snr.npz')
    report['omitted_full_project_artifacts'].append(dict(kind='model_cache',
              declared_path=model_path, declared_SHA256=model_record.get('cache_SHA256'),
              present=(out / model_path).exists(), read=False,
              validation='Original execution freeze and full-project certification; not this package review'))
    for c in CONDITIONS:
        fe = np.empty((n, 7, 10, 6), np.float64)
        fq = np.empty((n, 6), np.float64)
        for i in range(n):
            entry = index[c, i]
            path = path_in(out, entry['path'])
            jp = path.with_suffix('.json')
            if sha(path) != entry['SHA256'] or sha(jp) != entry['JSON_SHA256']:
                raise ValueError('RAW_PAIR_HASH:' + c + ':' + str(i))
            meta = js(jp)
            if ((meta['mode'], meta['condition'], int(meta['trajectory_id']), meta['complete']) != ('formal', c, i, True)
                    or meta['failures'] != 0 or meta['nonfinite'] != 0
                    or meta['common_truth_for_all_methods_and_SNR'] is not True
                    or meta['raw_SHA256'] != entry['SHA256'] or meta['methods'] != list(METHODS)
                    or meta['protocol_SHA256'] != protocol_hash or meta['seed_registry_SHA256'] != registry_hash):
                raise ValueError('RAW_JSON_IDENTITY_SCOPE_BINDING:' + c + ':' + str(i))
            identity = meta['identity']
            H = meta['input']['H_sha256']
            if identity != f'{registry["namespace"]}:{c}:{i}' or identity in seen_identity or H in seen_H or H in forbidden or H != entry['H_sha256'] or len(H) != 64:
                raise ValueError('DUPLICATE_OR_HISTORICAL_PHYSICAL_IDENTITY:' + c + ':' + str(i))
            seen_identity.add(identity); seen_H.add(H)
            for j in range(10):
                expected_seeds = {role: seed_index[c, i, -1 if role in channel_roles else j, role]
                                  for role in channel_roles + communication_roles}
                if meta['input']['frame_seeds'][j] != expected_seeds:
                    raise ValueError('FRAME_SEED_MAPPING:' + c + ':' + str(i) + ':' + str(j))
                if j == 0 and meta['input']['channel_seeds'] != expected_seeds:
                    raise ValueError('SIX_FIELD_CHANNEL_SEED_METADATA_MAPPING:' + c + ':' + str(i))
            if meta['input']['actual_channel_roles'] != list(channel_roles):
                raise ValueError('CHANNEL_ROLE_DECLARATION')
            bindings = meta['input']['paired_input_hashes']
            bx = {(int(z['frame_id']), int(z['snr_db'])): z for z in bindings}
            if len(bindings) != 70 or set(bx) != {(j, s) for j in range(10) for s in SNRS}:
                raise ValueError('PAIRED_INPUT_HASH_IDENTITY_GRID')
            for j in range(10):
                for field in ('truth_frame_SHA256', 'unit_noise_SHA256', 'transmitted_symbols_SHA256', 'bits_SHA256'):
                    h = bx[j, SNRS[0]][field]
                    if len(h) != 64 or any(bx[j, s][field] != h for s in SNRS):
                        raise ValueError('CROSS_SNR_INPUT_HASH_PAIRING:' + field)
            with zipfile.ZipFile(path) as z:
                if z.testzip() is not None or len(z.namelist()) != len(set(z.namelist())):
                    raise ValueError('RAW_NPZ_CRC_OR_DUPLICATE')
            with np.load(path, allow_pickle=False) as z:
                if not (np.array_equal(z['method_names'], np.array(METHODS)) and np.array_equal(z['snr_db'], np.array(SNRS)) and np.array_equal(z['frame_ids'], np.arange(4, 10))):
                    raise ValueError('RAW_METHOD_SNR_FRAME_IDENTITY')
                if z['frame_error'].shape != (7, 10, 6) or z['frame_truth'].shape != (6,) or z['frame_error'].dtype != np.float64 or z['frame_truth'].dtype != np.float64 or z['failure_flags'].shape != (7, 10, 6) or z['failure_flags'].dtype != np.bool_ or np.any(z['failure_flags']):
                    raise ValueError('RAW_ENERGY_AXES_DTYPE_OR_FAILURE')
                # Also inspect included small numerical diagnostics, not just
                # result counts. Never interpret these as a new model audit.
                for key in z.files:
                    array = z[key]
                    if array.dtype.kind == 'O' or (array.dtype.kind in 'fci' and not np.isfinite(array).all()):
                        raise ValueError('RAW_NUMERICAL_DIAGNOSTIC_NONFINITE:' + key)
                fe[i], fq[i] = z['frame_error'], z['frame_truth']
        numerator, denominator, linear, db, delta = pooled_from_frames(fe, fq)
        record = records[c]
        difference_path = path_in(out, record['primary_differences']['path'])
        if sha(difference_path) != record['primary_differences']['SHA256']:
            raise ValueError('SAVED_PRIMARY_DIFFERENCE_HASH:' + c)
        values = np.load(difference_path, mmap_mode='r', allow_pickle=False)
        if values.shape != (replicates, 7, 3) or values.dtype != np.float64:
            raise ValueError('SAVED_PRIMARY_DIFFERENCE_AXES')
        bounds = sorted_linear_quantiles(values)
        # Record metadata for omitted assets; deliberately never open them.
        for kind in ('indices', 'bootstrap_linear'):
            r = record[kind]
            report['omitted_full_project_artifacts'].append(dict(condition=c, kind=kind,
                    declared_path=r['path'], declared_SHA256=r.get('SHA256', r.get('sha256')),
                    present=(out / r['path']).exists(), read=False,
                    validation='Original full-project actual-index certification, not this package review'))
        for si, s in enumerate(SNRS):
            for mi, method in enumerate(METHODS):
                row = point_rows[c, str(s), method]
                if row['mode'] != 'formal' or int(row['trajectory_count']) != n or int(row['scored_frames']) != n * 6:
                    raise ValueError('POOLED_NMSE_UNIT_OR_MODE')
                for field, val in (('error_energy', numerator[si, mi]), ('truth_energy', denominator),
                                   ('nmse_linear', linear[si, mi]), ('nmse_db', db[si, mi])):
                    close(float(row[field]), val, '350_NMSE_' + field, report)
            for fi, (family, first, second) in enumerate(FAMILIES):
                row = contrast_rows[family, c, str(s)]
                if row['mode'] != 'formal' or row['first'] != first or row['second'] != second or int(row['bootstrap']) != replicates or int(row['family_size']) != 35 or float(row['tail_probability']) != TAIL or int(row['trajectory_count']) != n or 'positive favors second' not in row['sign']:
                    raise ValueError('PRIMARY_FAMILY_DIRECTION_OR_UNIT')
                close(float(row['estimate_db']), delta[si, fi], '105_PRIMARY_POINT', report)
                close(float(row['CI_lower']), bounds[0, si, fi], '105_ADJUSTED_LOWER', report)
                close(float(row['CI_upper']), bounds[1, si, fi], '105_ADJUSTED_UPPER', report)
                category = 'SECOND_SUPPORTED' if bounds[0, si, fi] > 0 else 'FIRST_SUPPORTED' if bounds[1, si, fi] < 0 else 'INDETERMINATE'
                if row['classification'] != category:
                    raise ValueError('PRIMARY_CLASSIFICATION')
        report['conditions'][c] = dict(raw_hash_CRC_identity_pairs=n, common_truth_denominator_checked=True,
                       saved_primary_differences_read=int(values.size), bootstrap_indices_read=False,
                       all_primary_point_and_bounds_verified=True, identities_disjoint=True)
        del values, fe, fq
    report.update(PASS=True, physical_trajectories=len(seen_H), raw_JSON_NPZ_pairs=5 * n,
                  NMSE_rows=350, primary_contrasts=105, adjusted_endpoints=210,
                  scored_positions_per_method=5 * n * 7 * 6)
    return report


def review_package(out, full_certification_sha256):
    """Public package verification fixes real formal counts; no toy shortcut."""
    return _review(out, full_certification_sha256, n=1000, replicates=100000, toy=False)


def _toy_package(out):
    """Deterministic miniature fixture, explicitly never formal evidence."""
    n, replicates = 2, 31
    def dump(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding='utf8')
    def csvout(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    protocol = dict(methods=list(METHODS), snrs=list(SNRS), scored_frames=list(range(4, 10)),
                    per_condition=n, physical_trajectories=5*n,
                    bootstrap=dict(replicates=replicates, tail_probability=TAIL),
                    primary_families={f:dict(first=a,second=b,comparisons=35) for f,a,b in FAMILIES})
    dump(out/'protocol.json',protocol)
    seedrows=[];seed=100
    for c in CONDITIONS:
        for i in range(n):
            for frame,roles in [(-1,('sionna_seed','torch_seed','numpy_seed'))]+[(j,('bits_seed_tx0','bits_seed_tx1','noise_seed')) for j in range(10)]:
                for role in roles:
                    seed+=1;seedrows.append(dict(condition=c,trajectory_id=i,frame_id=frame,role=role,seed=seed,group='EVAL'))
    registry=dict(namespace='SYNTHETIC_PACKAGE_ONLY',rows=seedrows)
    dump(out/'data/formal_seed_registry.json',registry)
    seedix={(r['condition'],r['trajectory_id'],r['frame_id'],r['role']):r['seed'] for r in seedrows}
    dump(out/'data/historical_H_hashes.json',dict(H_sha256=['f'*64]))
    dump(out/'reports/smoke_receipt.json',dict(PASS=True,identity_PASS=True,receipts=[dict(H_sha256='e'*64)]))
    entries=[];nm=[];ct=[];records=[]
    for ci,c in enumerate(CONDITIONS):
        fe=np.empty((n,7,10,6),np.float64);fq=np.empty((n,6),np.float64)
        for i in range(n):
            fe[i]=np.arange(1.,421.).reshape(7,10,6)+(ci+i)*.1;fq[i]=np.arange(2.,8.)+i
            path=out/f'data/formal/{c}_{i:04d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,frame_error=fe[i],frame_truth=fq[i],method_names=np.array(METHODS),
                 snr_db=np.array(SNRS),frame_ids=np.arange(4,10),failure_flags=np.zeros((7,10,6),bool))
            hh=hashlib.sha256(f'SYNTHETIC:{ci}:{i}'.encode()).hexdigest()
            frame_seeds=[{role:seedix[c,i,-1 if role in ('sionna_seed','torch_seed','numpy_seed') else j,role]
                          for role in ('sionna_seed','torch_seed','numpy_seed','bits_seed_tx0','bits_seed_tx1','noise_seed')} for j in range(10)]
            bindings=[]
            for j in range(10):
                for s in SNRS:
                    bindings.append(dict(frame_id=j,snr_db=s,**{k:hashlib.sha256(f'TOY:{ci}:{i}:{j}:{k}'.encode()).hexdigest() for k in ('truth_frame_SHA256','unit_noise_SHA256','transmitted_symbols_SHA256','bits_SHA256')}))
            meta=dict(mode='formal',condition=c,trajectory_id=i,complete=True,identity=f'SYNTHETIC_PACKAGE_ONLY:{c}:{i}',
                 failures=0,nonfinite=0,common_truth_for_all_methods_and_SNR=True,raw_SHA256=sha(path),methods=list(METHODS),
                 protocol_SHA256=sha(out/'protocol.json'),seed_registry_SHA256=sha(out/'data/formal_seed_registry.json'),
                 input=dict(H_sha256=hh,frame_seeds=frame_seeds,channel_seeds=frame_seeds[0],
                            actual_channel_roles=['sionna_seed','torch_seed','numpy_seed'],paired_input_hashes=bindings))
            dump(path.with_suffix('.json'),meta)
            entries.append(dict(condition=c,trajectory_id=i,path=path.relative_to(out).as_posix(),
                           SHA256=sha(path),JSON_SHA256=sha(path.with_suffix('.json')),H_sha256=hh))
        numerator,denominator,linear,db,delta=pooled_from_frames(fe,fq)
        values=delta[None,:,:]+np.linspace(-.2,.2,replicates)[:,None,None]
        path=out/f'data/bootstrap/formal/{c}/primary_differences.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,values)
        bounds=sorted_linear_quantiles(values)
        records.append(dict(condition=c,primary_differences=dict(path=path.relative_to(out).as_posix(),SHA256=sha(path)),
                            indices=dict(path=f'data/bootstrap/formal/{c}/indices.npy',sha256='1'*64),
                            bootstrap_linear=dict(path=f'data/bootstrap/formal/{c}/bootstrap_linear.npy',SHA256='2'*64)))
        for si,s in enumerate(SNRS):
            for mi,m in enumerate(METHODS):
                nm.append(dict(mode='formal',condition=c,snr_db=s,method=m,error_energy=numerator[si,mi],truth_energy=denominator,
                               nmse_linear=linear[si,mi],nmse_db=db[si,mi],trajectory_count=n,scored_frames=n*6))
            for fi,(f,a,b) in enumerate(FAMILIES):
                lo,hi=bounds[:,si,fi]
                ct.append(dict(mode='formal',family=f,condition=c,snr_db=s,first=a,second=b,estimate_db=delta[si,fi],CI_lower=lo,CI_upper=hi,
                         classification='SECOND_SUPPORTED' if lo>0 else 'FIRST_SUPPORTED' if hi<0 else 'INDETERMINATE',
                         bootstrap=replicates,family_size=35,tail_probability=TAIL,trajectory_count=n,sign='positive favors second'))
    dump(out/'reports/formal_receipt.json',dict(PASS=True,identity_PASS=True,failures=0,nonfinite=0,missing=0,substituted=0,smoke_mixed_into_formal=False,receipts=entries))
    dump(out/'reports/formal_analysis.json',dict(PASS=True,mode='formal',bootstrap_replicates=replicates,physical_trajectories=5*n,
                NMSE_rows=350,primary_rows=105,no_combined105_familywise_guarantee=True,records=records,
                raw_receipt_SHA256=sha(out/'reports/formal_receipt.json')))
    csvout(out/'tables/formal_nmse.csv',nm);csvout(out/'tables/formal_primary_contrasts.csv',ct)
    cert=dict(PASS=True,mode='formal',physical_trajectories=5*n,bootstrap_replicates_per_condition=replicates,
              new_draws=0,new_predictions=0,new_training=0,toy_synthetic_only=True,
              conditions={c:dict(physical_trajectories=n,bootstrap_draws_checked=replicates,all105family_endpoints_checked=True) for c in CONDITIONS})
    dump(out/'reports/independent_numerical_review.json',cert)
    return sha(out/'reports/independent_numerical_review.json'),n,replicates


def toy_tests():
    x = np.arange(100000 * 21, dtype=np.float64).reshape(100000, 7, 3)
    assert np.array_equal(sorted_linear_quantiles(x), np.quantile(x, [TAIL, 1 - TAIL], axis=0, method='linear'))
    fe = np.arange(1., 3 * 7 * 10 * 6 + 1).reshape(3, 7, 10, 6)
    fq = np.arange(1., 19.).reshape(3, 6)
    _, _, got, _, _ = pooled_from_frames(fe, fq)
    assert np.allclose(got, fe.sum(axis=-1).sum(axis=0) / fq.sum(), rtol=1e-14, atol=1e-14)
    for bad in (np.full_like(fe, np.nan), -fe):
        try:
            pooled_from_frames(bad, fq)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid synthetic energies accepted')
    with tempfile.TemporaryDirectory(prefix='mr08a_package_toy_') as directory:
        root = Path(directory)
        (root / 'original').write_bytes(b'fixed bytes')
        try:
            path_in(root, '../outside')
        except ValueError:
            pass
        else:
            raise AssertionError('Package path escape accepted')
        report = dict(numerical_comparisons=0, maximum_absolute_difference=0., maximum_scaled_difference=0.)
        close(1., 1. + 1e-12, 'toy', report)
        try:
            close(1., 1.1, 'toy', report)
        except ValueError:
            pass
        else:
            raise AssertionError('Numeric mismatch accepted')
        digest,n,replicates=_toy_package(root)
        result=_review(root,digest,n=n,replicates=replicates,toy=True)
        assert result['PASS'] and result['NMSE_rows']==350 and result['primary_contrasts']==105 and result['raw_JSON_NPZ_pairs']==10
        assert all(not r['read'] and not r['present'] for r in result['omitted_full_project_artifacts'])
        try:
            _review(root,'0'*64,n=n,replicates=replicates,toy=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Certification hash mismatch accepted')
        path=root/'data/formal/ID_0000.npz';path.write_bytes(path.read_bytes()+b'mutation')
        try:
            _review(root,digest,n=n,replicates=replicates,toy=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Raw byte mutation accepted')
    return dict(PASS=True, scope='SYNTHETIC_ONLY', test_groups=9,
                actual_formal_files_read=0, new_predictions=0, new_draws=0,
                formal_review_executed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--full-certification-sha256')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()
    if not args.toy and not args.full_certification_sha256:
        parser.error('--full-certification-sha256 is required for real package review')
    result = toy_tests() if args.toy else review_package(args.out, args.full_certification_sha256)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.report:
        report = args.report.resolve()
        if report.is_relative_to((args.out / 'data').resolve()):
            raise ValueError('REPORT_MAY_NOT_WRITE_ORIGINAL_DATA')
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open('x', encoding='utf8') as f:
            f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
