"""Synthetic algebra audit of frozen pilot-CV; no saved scientific data inputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pilot_cv import PilotCV, causal_windows, cross_pilot_operators, frozen_folds
from src.lowrank_modules import RankFactors, factored_affine_predict
from src.spatial_information import observation_geometry

HEAD = 'ca83982b1d0e4bdca0215a3ecbe3a7a8f8f8ec55'
PREFIX = 'sp_u2a_'
TOL = 2e-10  # Independent contraction identity, not a deployment threshold.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def output_guard(event, args):
    """No old experiment output can become a numerical input to this audit."""
    if event != 'open' or not isinstance(args[0], (str, bytes)):
        return
    name = str(args[0]).replace('\\', '/').lower()
    root = str(ROOT).replace('\\', '/').lower() + '/'
    if name.startswith(root):
        name = name[len(root):]
    if name.startswith('outputs/'):
        leaf = name.split('/')[1]
        if not (leaf.startswith('sp_u2a_') or leaf.startswith('sp_u1_')):
            raise PermissionError('SP-U2A firewall: saved scientific output read/write forbidden')


def snapshot():
    path = ROOT / 'outputs/sp_u2a_before_manifest.json'
    if path.exists():
        raise RuntimeError('baseline snapshot already exists; refusing to replace it')
    if git('rev-parse', 'HEAD') != HEAD or git('diff', '--name-only') or git('diff', '--cached', '--name-only'):
        raise RuntimeError('accepted HEAD or tracked working tree changed')
    tracked = git('ls-files').splitlines()
    protected = [p for p in tracked if p.startswith(('src/', 'scripts/', 'tests/', 'configs/'))]
    original = [p for p in git('ls-files', '--others', '--exclude-standard').splitlines()
                if p.startswith(('docs/sp_u1_', 'outputs/sp_u1_', 'scripts/')) and 'sp_u1' in p]
    original += [p for p in git('ls-files', '--others', '--exclude-standard').splitlines()
                 if p == 'tests/test_sp_u1_upgrade.py']
    assert len(original) == 64, 'original SP-U1 inventory differs; inspect before proceeding'
    data = dict(head=HEAD, protected_sha256={p: sha(ROOT / p) for p in sorted(set(protected + original))},
                original_sp_u1_files=sorted(original), tracked_paths=tracked,
                saved_formal_outputs_hashed_or_loaded=False)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(head=HEAD, protected_files=len(data['protected_sha256']), original_sp_u1_files=len(original))))


def close(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = float(np.linalg.norm(actual - expected))
    relative = error / max(1., float(np.linalg.norm(expected)))
    assert relative <= TOL, (error, relative)
    return relative


def realify(matrix):
    return np.concatenate([matrix.real, matrix.imag], axis=0)


def family():
    """Construct PSD algebra fixtures, not empirical or CDL channel statistics."""
    index = np.arange(64)
    modes = [np.exp(2j * np.pi * k * index / 64) for k in (1, 3, 7)]
    covs = [np.eye(64) + np.outer(modes[0], modes[0].conj()) / 64]
    covs += [covs[0] + 2 * np.outer(mode, mode.conj()) / 64 for mode in modes[1:]]
    geometry = observation_geometry()
    folds, definition = frozen_folds(geometry.selected)
    kernels = [cross_pilot_operators(cov, .7, folds)[0] for cov in covs]
    mean = .01 * np.cos(index / 3) + .02j * np.sin(index / 5)
    cv = PilotCV(folds, mean, kernels[0],
                 tuple(p - s for p, s in zip(kernels[1], kernels[0])),
                 tuple(m - s for m, s in zip(kernels[2], kernels[0])))
    ij = np.arange(640).reshape(10, 64)
    pilots = mean + np.sin(ij / 9) + .7j * np.cos(ij / 13)
    return cv, pilots, definition


def maps(cv):
    identity = np.eye(64)
    u, tp, tm = [], [], []
    for f, (observed, heldout) in enumerate(cv.folds):
        u.append(identity[heldout] - cv.source_k[f] @ identity[observed])
        tp.append(cv.profile_delta_k[f] @ identity[observed])
        tm.append(cv.mobility_delta_k[f] @ identity[observed])
    return tuple(np.concatenate(v) for v in (u, tp, tm))


def expand(cv, pilots):
    columns, responses = [], []
    for pilot in pilots:
        centered = pilot - cv.source_mean_p
        for f, (observed, heldout) in enumerate(cv.folds):
            x = centered[observed]
            columns.append(np.column_stack([cv.profile_delta_k[f] @ x, cv.mobility_delta_k[f] @ x]))
            responses.append(centered[heldout] - cv.source_k[f] @ x)
    d, r = np.concatenate(columns), np.concatenate(responses)
    return realify(d), realify(r), d, r


def equivalence():
    cv, pilots, definition = family()
    window = pilots[2:7]
    g, z, d, r = expand(cv, window)
    coef, diag = cv.fit(window, modules=('P', 'M'))
    reference = np.linalg.lstsq(g, z, rcond=None)[0]  # independent QR/SVD reference, not production solve
    errors = dict(coordinate_vs_independent_lstsq=close(coef, reference),
                  gram=close(diag['gram'], g.T @ g), rhs=close(diag['rhs'], g.T @ z),
                  objective=close(diag['fitted_cv_loss'], np.linalg.norm(z - g @ coef) ** 2))
    u_map, tp, tm = maps(cv)
    for pilot in window:
        x = pilot - cv.source_mean_p
        single = expand(cv, pilot[None])
        close(single[2], np.column_stack([tp @ x, tm @ x]))
        close(single[3], u_map @ x)
    folds = []
    rx, tx, k, t = np.unravel_index(observation_geometry().selected, (2, 2, 64, 16))
    for f, (observed, heldout) in enumerate(cv.folds):
        folds.append(dict(fold=f, observed=observed.tolist(), heldout=heldout.tolist(),
                          heldout_physical=[dict(rx=int(rx[i]), tx=int(tx[i]), k=int(k[i]), t=int(t[i])) for i in heldout]))
    return dict(errors=errors, real_design_shape=list(g.shape), complex_design_shape=list(d.shape),
                response_shape=list(z.shape), gram=diag['gram'], rhs=diag['rhs'],
                coefficients=coef.tolist(), condition_gram=diag['condition_number'],
                fold_definition=definition, folds=folds,
                fixture='synthetic PSD kernel construction and fixed trigonometric received-pilot arrays')


def rank_conditioning():
    cv, pilots, _ = family()
    g, _, _, _ = expand(cv, pilots[:5])
    h = g.T @ g
    a, b, d = h[0, 0], h[0, 1], h[1, 1]
    eigen_formula = [(a + d - np.sqrt((a - d) ** 2 + 4 * b ** 2)) / 2,
                     (a + d + np.sqrt((a - d) ** 2 + 4 * b ** 2)) / 2]
    close(eigen_formula, np.linalg.eigvalsh(h))
    close(np.linalg.cond(h), np.linalg.cond(g) ** 2)
    column = np.array([1 + 2j, 3 - 1j])
    complex_matrix = np.column_stack([column, 1j * column])
    assert np.linalg.matrix_rank(complex_matrix) == 1
    assert np.linalg.matrix_rank(realify(complex_matrix)) == 2
    singular = PilotCV(cv.folds, cv.source_mean_p, cv.source_k, cv.profile_delta_k, cv.profile_delta_k)
    try:
        singular.fit(pilots[:5], modules=('P', 'M'))
    except np.linalg.LinAlgError:
        singular_raises = True
    else:
        raise AssertionError('exact duplicated columns must retain the existing solve failure')
    one, *_ = expand(cv, pilots[:1])
    assert np.linalg.eigvalsh(h)[0] >= np.linalg.eigvalsh(one.T @ one)[0] - 1e-14
    return dict(eigenvalues=eigen_formula, gram_condition=float(np.linalg.cond(h)),
                complex_rank_one_real_rank_two=True, singular_original_solve_raises=singular_raises)


def endogeneity():
    cv, _, _ = family()
    u, tp, tm = maps(cv)
    c = np.array([.4, -.3])
    f = u - c[0] * tp - c[1] * tm
    # Exactly integrate zero-mean proper noise using atoms ±sqrt(64)*e_i, ±i sqrt(64)*e_i.
    # This finite support has E[n n^H]=I and E[n n^T]=0; no Monte Carlo or channel realization.
    score = np.zeros(2)
    covariance = np.zeros((128, 128))
    for i in range(64):
        for phase in (1, -1, 1j, -1j):
            noise = np.zeros(64, complex)
            noise[i] = 8 * phase
            design = realify(np.column_stack([tp @ noise, tm @ noise]))
            residual = realify(f @ noise)
            score += design.T @ residual / 256
            covariance += np.outer(residual, residual) / 256
    expected = np.array([np.trace(t.conj().T @ f).real for t in (tp, tm)])
    close(score, expected)
    fold_expected = []
    for delta in (cv.profile_delta_k, cv.mobility_delta_k):
        fold_expected.append(-sum(np.trace(dm.conj().T @ (ks + c[0] * dp + c[1] * mm)).real
                                  for dm, ks, dp, mm in zip(delta, cv.source_k, cv.profile_delta_k, cv.mobility_delta_k)))
    close(score, fold_expected)
    ff = f @ f.conj().T
    close(covariance, .5 * np.block([[ff.real, -ff.imag], [ff.imag, ff.real]]))
    assert np.linalg.norm(score) > 0
    offdiag = covariance - np.diag(np.diag(covariance))
    assert np.linalg.norm(offdiag) > 0
    return dict(exact_noise_support_size=256, sigma2=1, score=score.tolist(),
                score_trace_identity_error=close(score, expected),
                residual_covariance_offdiagonal_norm=float(np.linalg.norm(offdiag)),
                noise_is_algebraic_moment_fixture_not_a_CDL_run=True)


def perturbation():
    cv, pilots, _ = family()
    window = pilots[:5]
    g, z, *_ = expand(cv, window)
    c = cv.fit(window, modules=('P', 'M'))[0]
    index = np.arange(320).reshape(5, 64)
    dp = 1e-4 * (np.cos(index / 3) + 1j * np.sin(index / 5))
    gp, zp, *_ = expand(cv, window + dp)
    cp = cv.fit(window + dp, modules=('P', 'M'))[0]
    e, delta_g, delta_z = z - g @ c, gp - g, zp - z
    identity = gp.T @ (delta_z - delta_g @ c) + delta_g.T @ e
    close(gp.T @ gp @ (cp - c), identity)
    smin = np.linalg.svd(gp, compute_uv=False)[-1]
    bound = (np.linalg.norm(gp, 2) * np.linalg.norm(delta_z - delta_g @ c)
             + np.linalg.norm(delta_g, 2) * np.linalg.norm(e)) / smin ** 2
    actual = float(np.linalg.norm(cp - c))
    assert actual <= bound * (1 + TOL)
    ref = np.array([.5, -.2])
    eps = z - g @ ref
    close(c - ref, np.linalg.solve(g.T @ g, g.T @ eps))
    assert np.linalg.norm(c - ref) <= np.linalg.norm(eps) / np.linalg.svd(g, compute_uv=False)[-1] * (1 + TOL)
    # No future frame can affect j=4,5,6 window fits; explicitly recompute both G and z.
    for j, window in causal_windows(pilots, 5):
        changed = pilots.copy()
        changed[j + 1:] += 1e6
        other = dict(causal_windows(changed, 5))[j]
        np.testing.assert_array_equal(window, other)
        np.testing.assert_array_equal(cv.fit(window, modules=('P', 'M'))[0], cv.fit(other, modules=('P', 'M'))[0])
    return dict(actual_coordinate_change=actual, deterministic_joint_perturbation_bound=float(bound),
                induced_design_change=float(np.linalg.norm(delta_g)), future_frames_excluded=True)


def data_fixture():
    cv, pilots, _ = family()
    n, p = 3968, 64
    row = np.arange(n)[:, None]
    col = np.arange(p)[None]
    factors = [RankFactors(.03 * (np.sin(row / 37 + np.arange(4)) + 1j * np.cos(row / 51 + np.arange(4))),
                           .03 * (np.cos(np.arange(4)[:, None] + col / 5) + 1j * np.sin(col / 9 + np.arange(4)[:, None])))
               for _ in range(2)]
    # Different origin directions; keep literal rank4 factor dimensions.
    factors[1] = RankFactors(.03 * (np.cos(row / 17 + np.arange(4)) + 1j * np.sin(row / 23 + np.arange(4))),
                            .03 * (np.sin(np.arange(4)[:, None] + col / 7) + 1j * np.cos(col / 11 + np.arange(4)[:, None])))
    b4 = [f.A @ f.Q for f in factors]  # Independent synthetic audit only, never production inference.
    dense = [b4[0] + .001 * np.sin(row / 29) @ np.cos(col / 3),
             b4[1] + .001j * np.cos(row / 31) @ np.sin(col / 4)]
    ws = .001 * (np.sin(row / 23 + col / 8) + 1j * np.cos(row / 47 + col / 5))
    mean = .01 * np.sin(np.arange(n) / 41).astype(complex)
    return cv, pilots, factors, b4, dense, ws, mean


def propagation_decomposition():
    cv, pilots, factors, b4, dense, ws, mean = data_fixture()
    g, z, *_ = expand(cv, pilots[:5])
    c = cv.fit(pilots[:5], modules=('P', 'M'))[0]
    ref = np.array([.5, -.2])
    x = pilots[4] - cv.source_mean_p
    def predict(obs, coef):
        return factored_affine_predict(obs, cv.source_mean_p, mean, ws, coef, *factors)
    xc = np.column_stack([f.apply(x) for f in factors])
    delta = c - ref
    difference = predict(pilots[4], c) - predict(pilots[4], ref)
    close(difference, xc @ delta)
    metric = (xc.conj().T @ xc).real
    energy = float(np.vdot(difference, difference).real)
    close(energy, delta @ metric @ delta)
    eps = z - g @ ref
    bound = np.linalg.eigvalsh(metric)[-1] / np.linalg.eigvalsh(g.T @ g)[0] * np.linalg.norm(eps) ** 2
    assert energy <= bound * (1 + TOL)
    dp = .0001 * (np.cos(np.arange(64)) + 1j * np.sin(np.arange(64)))
    changed = pilots[:5].copy()
    changed[-1] += dp
    newc = cv.fit(changed, modules=('P', 'M'))[0]
    dc = newc - c
    joint = (ws + sum(a * b for a, b in zip(c, b4))) @ dp + xc @ dc + sum(a * (b @ dp) for a, b in zip(dc, b4))
    close(predict(changed[-1], newc) - predict(pilots[4], c), joint)
    # Four exact vectors with nonzero explicit noise and unpredicted clean channel component.
    noise = .07 * (np.cos(np.arange(64) / 3) + 1j * np.sin(np.arange(64) / 4))
    clean = x - noise
    wd = ws + sum(a * b for a, b in zip(ref, dense))
    truth = mean + wd @ clean + .1 * np.cos(np.arange(3968) / 17)
    terms = [mean + wd @ clean - truth,
             sum(a * (b @ clean) for a, b in zip(c - ref, dense)),
             sum(a * ((b - d) @ clean) for a, b, d in zip(c, b4, dense)),
             (ws + sum(a * b for a, b in zip(c, b4))) @ noise]
    total = predict(pilots[4], c) - truth
    close(sum(terms), total)
    gram = np.array([[np.vdot(a, b) for b in terms] for a in terms])
    individual = np.diag(gram).real
    pairs = [float(2 * gram[i, j].real) for i in range(4) for j in range(i + 1, 4)]
    close(float(np.vdot(total, total).real), float(np.sum(individual) + sum(pairs)))
    assert np.linalg.norm(pairs) > 0
    assert np.linalg.norm(total) ** 2 <= 4 * np.sum(individual) * (1 + TOL)
    # Matched-reference convention D05 with deliberately nonaligned source/target means.
    target_mean_p = cv.source_mean_p + .02
    target_mean_d = mean - .03j
    wt = ws + .2 * dense[0] - .1 * dense[1]
    matched = target_mean_d + wt @ (pilots[4] - target_mean_p)
    mean_bias = mean - target_mean_d + wt @ (target_mean_p - cv.source_mean_p)
    alternative = [matched - truth,
                   mean_bias + (wd - wt) @ x,
                   sum(a * (b @ x) for a, b in zip(c - ref, dense)),
                   sum(a * ((b - d) @ x) for a, b, d in zip(c, b4, dense))]
    close(sum(alternative), total)
    return dict(data_output_size=3968, factor_shapes=[[3968, 4], [4, 64]],
                coordinate_output_energy=energy, geometry_to_output_bound=float(bound),
                output_metric=metric.tolist(), four_error_energies=individual.tolist(), six_cross_terms=pairs,
                total_error_energy=float(np.vdot(total, total).real), exact_vector_decomposition_error=close(sum(terms), total),
                matched_reference_mean_bias_identity_error=close(sum(alternative), total),
                synthetic_reference_is_not_an_estimated_CDL_oracle=True)


def oracle_distinction():
    cv, _, _ = family()
    u, tp, tm = maps(cv)
    h_p = np.array([[np.trace(a.conj().T @ b).real for b in (tp, tm)] for a in (tp, tm)])
    b_p = np.array([np.trace(a.conj().T @ u).real for a in (tp, tm)])
    c_p = np.linalg.solve(h_p, b_p)
    # Same clean-pilot law CN(0,I), same sigma2=.7 and same frozen kernels in both joint laws.
    # Two data maps B1,B2 with orthogonal output/pilot rows => H_d=(1+.7)I.
    c_data = [c_p + np.array([1., -2.]), c_p + np.array([-3., .5])]
    bp = np.zeros((2, 64), complex)
    bm = bp.copy()
    bp[0, 0], bm[1, 1] = 1, 1
    bases = (bp, bm)
    observation_covariance = 1.7 * np.eye(64)
    h_data = np.array([[np.trace(a.conj().T @ b @ observation_covariance).real for b in bases] for a in bases])
    oracle, eigenvalues, excess = [], [], []
    for c in c_data:
        clean_data_map = 1.7 * (c[0] * bp + c[1] * bm)
        cross_data_pilot = clean_data_map  # Cov(clean pilot)=I; independent observation noise.
        joint = np.block([[np.eye(64), clean_data_map.conj().T],
                          [clean_data_map, clean_data_map @ clean_data_map.conj().T + .3 * np.eye(2)]])
        eigenvalues.append(float(np.linalg.eigvalsh(joint)[0]))
        assert eigenvalues[-1] > 0
        rhs = np.array([np.trace(a.conj().T @ cross_data_pilot).real for a in bases])
        optimum = np.linalg.solve(h_data, rhs)
        oracle.append(optimum)
        excess.append(float((c_p - optimum) @ h_data @ (c_p - optimum)))
    for value, reference in zip(oracle, c_data):
        close(value, reference)
        assert np.linalg.norm(value - c_p) > 0
    assert np.linalg.norm(oracle[0] - oracle[1]) > 0
    return dict(population_pilot_coordinate=c_p.tolist(), channel_coordinates=[c.tolist() for c in oracle],
                same_received_pilot_law=True, valid_PSD_joint_laws_by_linear_generation=True,
                joint_covariance_min_eigenvalues=eigenvalues, pilot_coordinate_channel_excess_risks=excess,
                counterexample_is_general_joint_model_not_a_CDL_claim=True)


SENTINELS = dict(equation_equivalence=equivalence, real_rank_conditioning=rank_conditioning,
                 noisy_design_endogeneity=endogeneity, joint_pilot_perturbation=perturbation,
                 coordinate_propagation_and_four_errors=propagation_decomposition,
                 pilot_vs_channel_oracles=oracle_distinction)


def prior_art_audit():
    access = json.loads((ROOT / 'outputs/sp_u2a_source_access.json').read_text(encoding='utf-8'))
    assert access['current_commit'] == 'fa2647f7fd0411bdb6a33ad9705cadb134ab3793'
    assert all(r['byte_identical'] for r in access['previous_snapshot_comparison'].values())
    for record in access['sources']:
        assert sha(ROOT / record['path']) == record['sha256']
    source_root = ROOT / 'outputs/sp_u2a_sources'
    main_text = (source_root / 'A-MMSE__main.py.txt').read_text(encoding='utf-8')
    assignments = {node.targets[0].id: ast.literal_eval(node.value)
                   for node in ast.walk(ast.parse(main_text)) if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and isinstance(node.value, ast.Constant)}
    assert assignments['USE_TRANSFORMER_GLOBAL_FILTER'] is True
    model = (source_root / 'A-MMSE__AMMSE.py.txt').read_text(encoding='utf-8')
    loader = (source_root / 'A-MMSE__data.py.txt').read_text(encoding='utf-8')
    assert 'self.process_batch(self.global_token, training=False)' in model
    assert 'x_test = noisy_combined[-num_test:]' in loader
    assert 'train_test_split(noisy_combined[:-num_test]' in loader
    result = dict(date='2026-10-04', active_matrix='docs/sp_u2a_prior_art_amendment.md',
                  original_sp_u1_edited=False,
                  closest_works=[dict(id='koller2022gmm', tier=1, doi='10.1109/TSP.2022.3194348',
                                     source='https://arxiv.org/pdf/2112.12499', inspected='SecIV-A,eq11,13,15,16;Algorithm1',
                                     weights='observation-dependent posterior simplex', estimator='mixture of conditional affine LMMSE'),
                                 dict(id='fesl2023mfa', tier=1, version='2304.14809v2',
                                     source='https://arxiv.org/pdf/2304.14809v2', inspected='SecIII-A,eq9-12',
                                     weights='observation-dependent posterior simplex', estimator='mixture of conditional affine LMMSE',
                                     low_rank_object='component covariance loading'),
                                 dict(id='ozdemir2006', doi='10.1109/TWC.2006.04713',
                                     source_quality='ABSTRACT_ONLY', risk='OPEN RISK')],
                  ammse=dict(version='2506.00452v5', source='https://arxiv.org/html/2506.00452v5',
                             public_commit=access['current_commit'], inspected_files_sha256={
                                 r['path']: r['sha256'] for r in access['sources']},
                             default_global_token_verified=True, manuscript_pilot_input_vs_code_token='UNRESOLVED_DIFFERENCE',
                             test_split='chronological last4000', validation_split='random among earlier frames',
                             license=access['license'], author_code_executed=False, implementation='HOLD'),
                  novelty_clearance='NOT_ESTABLISHED', baseline_implementation_authorized=False)
    (ROOT / 'outputs/sp_u2a_prior_art_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return dict(closest_work_tier1=['koller2022gmm', 'fesl2023mfa'], ammse='v5_ONLY_HOLD')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', action='store_true')
    args = parser.parse_args()
    sys.addaudithook(output_guard)
    if args.snapshot:
        snapshot()
        return
    started = time.perf_counter()
    before = json.loads((ROOT / 'outputs/sp_u2a_before_manifest.json').read_text(encoding='utf-8'))
    results = {}
    for name, function in SENTINELS.items():
        results[name] = function()
    changed = [p for p, digest in before['protected_sha256'].items() if sha(ROOT / p) != digest]
    assert not changed, changed
    assert git('rev-parse', 'HEAD') == HEAD
    assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
    assert git('ls-files').splitlines() == before['tracked_paths']
    source_hashes = {p: sha(ROOT / p) for p in ('src/pilot_cv.py', 'src/lowrank_modules.py',
                      'src/frozen_affine_validation.py', 'src/spatial_information.py', 'src/estimators.py',
                      'scripts/validate_sp_u2a.py', 'tests/test_sp_u2a_theory.py')}
    result = dict(task='SP-U2A', head=HEAD, date='2026-10-04', numerical_identity_tolerance=TOL,
                  fixture_seed=None, split='synthetic algebra only; no development/validation/formal data',
                  channel='none', training_or_statistical_fitting='none', protected_files=len(before['protected_sha256']),
                  protected_changes=changed, source_sha256=source_hashes, sentinels=results,
                  elapsed_seconds=time.perf_counter() - started, formal_output_access_guard_enabled=True,
                  formal_outcomes_used=False, independent_CDL_evidence=False,
                  first_layer_connection_self_assessment='PASS', novel_general_theorem='NOT_ESTABLISHED',
                  next_task_authorized=False, anomalies=[], all_checks_passed=True)
    result['prior_art_amendment_audit'] = prior_art_audit()
    (ROOT / 'outputs/sp_u2a_theory_validation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(all_checks_passed=True, sentinels=list(results), protected_files=result['protected_files'],
                          elapsed_seconds=result['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
