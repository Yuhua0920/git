"""Numerical validation without outcome comparisons or scientific bootstrap."""
import numpy as np
from scripts import sp_u12b3a_common as b
from scripts.sp_u12b3a_ablation import SingleDirectionCV, solve_scalar, deployed_arms, TechnicalSolveFailure
from src.pilot_cv import deployment
from src.lowrank_modules import factored_affine_predict

def scalar_audit():
    rows = []
    for direction in ('P', 'M'):
        for n in (3, 7, 16, 31):
            for trial in range(3):
                rng = np.random.default_rng(437000 + n * 10 + trial + (0 if direction == 'P' else 10000))
                d = rng.normal(size=n) + 1j * rng.normal(size=n)
                z = rng.normal(size=n) + 1j * rng.normal(size=n)
                got = solve_scalar(float(np.vdot(d, d).real), float(np.vdot(d, z).real))
                G = np.r_[d.real, d.imag][:, None]
                response = np.r_[z.real, z.imag]
                reference = np.linalg.lstsq(G, response, rcond=None)[0]
                error = b.c.rel(got, reference)
                b.need(error <= b.c.TOL, 'SCALAR_COMPLEX_REFERENCE_MISMATCH')
                rows.append(dict(direction=direction, case=f'complex_{n}_{trial}',
                    complex_rows=n, real_rows=2*n, unknowns=1,
                    relative_reference_error=error, status='PASS'))
        for seed in range(431013, 431017):
            cv, p, _, _ = b.fixture(seed=seed)
            got, diag = SingleDirectionCV(cv, direction).fit(p)
            G, z = b.explicit_stack(cv, p)
            index = 0 if direction == 'P' else 1
            reference = np.linalg.lstsq(G[:, index:index+1], z, rcond=None)[0]
            error = b.c.rel(got, reference)
            b.need(error <= b.c.TOL and diag['normal_equation_relative_residual'] <= b.c.SOLVE_TOL,
                   'SCALAR_CROSS_PILOT_REFERENCE_MISMATCH')
            rows.append(dict(direction=direction, case=f'five_frame_{seed}', complex_rows=320,
                real_rows=640, unknowns=1, relative_reference_error=error, status='PASS'))
    return rows

def geometry_audit():
    orthogonal = {}
    cv, p, _, _ = b.fixture(mode='orthogonal')
    G, z = b.explicit_stack(cv, p)
    both, _ = cv.fit(p, modules=('P', 'M'))
    independent = np.r_[SingleDirectionCV(cv, 'P').fit(p)[0], SingleDirectionCV(cv, 'M').fit(p)[0]]
    b.need(float(G[:, 0] @ G[:, 1]) == 0 and b.c.rel(both, independent) <= b.c.TOL, 'ORTHOGONAL_GEOMETRY')
    orthogonal.update(PASS=True, real_cross_gram_exactly_zero=True, independent_scalar_coordinates_equal=True)
    cv, p, _, _ = b.fixture(mode='coupled')
    G, z = b.explicit_stack(cv, p)
    both, _ = cv.fit(p, modules=('P', 'M'))
    independent = np.r_[SingleDirectionCV(cv, 'P').fit(p)[0], SingleDirectionCV(cv, 'M').fit(p)[0]]
    cross = float(G[:, 0] @ G[:, 1])
    differences = np.abs(both - independent)
    b.need(cross != 0 and np.all(differences > 1e-12), 'COUPLED_GEOMETRY')
    coupled = dict(PASS=True, nonorthogonal_design=True, P_scalar_differs_from_Both=True,
        M_scalar_differs_from_Both=True, posthoc_zeroing_is_not_single_direction_fit=True,
        scientific_performance_values_reported=False)
    zero_cases = []
    for zero, active in (('M', 'P'), ('P', 'M')):
        cv, p, arrays, factors = b.fixture(zero=zero)
        coefficient, _ = SingleDirectionCV(cv, active).fit(p)
        G, z = b.explicit_stack(cv, p)
        active_index = 0 if active == 'P' else 1
        single_loss = float(np.linalg.norm(G[:, active_index:active_index+1] @ coefficient-z)**2)
        base = arrays['source_mean_d'] + (p[-1]-arrays['source_mean_p']) @ arrays['Wsource_10'].T
        expected = base + coefficient[0] * ((p[-1]-arrays['source_mean_p']) @ arrays[f'DeltaW{active}_10'].T)
        expected_lr = base + coefficient[0] * factors[active].apply(p[-1]-arrays['source_mean_p'])
        for arbitrary in (-3., 0., 2.5):
            cc = np.array([coefficient[0], arbitrary] if active == 'P' else [arbitrary, coefficient[0]])
            loss = float(np.linalg.norm(G @ cc-z)**2)
            pred = deployment(p[-1], arrays['source_mean_p'], arrays['source_mean_d'], arrays['Wsource_10'],
                arrays['DeltaWP_10'], arrays['DeltaWM_10'], cc, ('P', 'M'))
            lowrank = factored_affine_predict(p[-1], arrays['source_mean_p'], arrays['source_mean_d'],
                arrays['Wsource_10'], cc, factors['P'], factors['M'])
            b.need(np.isclose(loss, single_loss, rtol=b.c.TOL, atol=1e-11), 'ZERO_DIRECTION_OBJECTIVE')
            b.need(b.c.rel(pred, expected) <= b.c.TOL and b.c.rel(lowrank, expected_lr) <= b.c.TOL,
                'ZERO_DIRECTION_PREDICTION')
        rejected = False
        try:
            cv.fit(p, modules=('P', 'M'))
        except np.linalg.LinAlgError:
            rejected = True
        b.need(rejected, 'LEGACY_BOTH_SINGULAR_NOT_REJECTED')
        zero_cases.append(dict(zero_direction=zero, active_direction=active,
            objective_minimizer_set_equivalent=True, dense_prediction_equivalent=True,
            LR4_prediction_equivalent=True, legacy_Both_singular_solve_rejected=True,
            unique_Both_coordinate_returned=False, synthetic_inactive_values_checked=3,
            no_fallback=True))
    return dict(orthogonal=orthogonal, zero_direction_cases=zero_cases,
        literal_unique_Both_solve_invariance=False,
        mathematical_reason='A zero design column makes the frozen 2x2 Gram singular. Active-coordinate minimizers and predictions coincide; the unchanged solver rejects nonidentifiability.'), coupled

def development_smoke():
    from src.cdl_benchmark import frames, observation_geometry, array_hash, SYSTEM
    from src.link import assemble_pilot_data_frame, apply_frequency_mimo_channel, estimate_ls_linear_channel
    registry = b.read(b.c.OUT / 'development_seed_registry.json')
    b.need(registry['namespace'] == 'SP_U12B2A_DEVELOPMENT_ONLY_V1', 'DEVELOPMENT_NAMESPACE')
    index = b.a.seed_index(registry)
    arrays = b.c.load_operators()
    legacy = b.c.historical_operators()
    cvs, factors = b.c.family(arrays, b.SNRS)
    old_cvs, old_factors = b.c.family(legacy, b.SNRS)
    geometry = observation_geometry()
    reports, replay = [], []
    for condition in b.CONDITIONS:
        generator = b.a.generator(condition, index, 'EVAL')
        b.need(SYSTEM['sample_rate_hz'] == SYSTEM['nfft'] * SYSTEM['delta_f_hz'], 'CONFIGURATION_INTEGRITY')
        for trajectory in range(2):
            H = generator.generate(trajectory)
            truths = frames(H)
            pilots = {s: [] for s in b.SNRS}
            for j in range(10):
                seeds = b.a.consumed(index, condition, 'EVAL', trajectory, j)
                _, symbols, unit_noise = b.c.communicate(seeds)
                clean = apply_frequency_mimo_channel(truths[j], assemble_pilot_data_frame(symbols, geometry.layout))
                for s in b.SNRS:
                    observed = clean + np.sqrt(b.c.variance(s)) * unit_noise
                    sparse, _ = estimate_ls_linear_channel(observed, geometry.layout)
                    pilots[s].append(sparse.ravel()[geometry.selected])
            for s in b.SNRS:
                observed_pilots = np.stack(pilots[s])
                before = array_hash(observed_pilots)
                got = deployed_arms(observed_pilots, arrays, s, cvs[s], factors['P', s], factors['M', s])
                current = b.c.infer(observed_pilots, arrays, s, cvs, factors)
                old = b.c.infer(observed_pilots, legacy, s, old_cvs, old_factors)
                b.need([r['frame_id'] for r in got] == list(range(4, 10)), 'WARMUP_SCORE_FRAMES')
                residuals = {arm: [] for arm in ('P-only', 'M-only', 'Both')}
                max_coord = max_dense = max_lr = 0.
                for j, (row, existing, historical) in enumerate(zip(got, current, old)):
                    for arm, entry in row['arms'].items():
                        c = entry['coefficients']
                        b.need(c.shape == ((2,) if arm == 'Both' else (1,)), 'COORDINATE_SHAPE')
                        b.need(entry['dense_coefficients'] is entry['lr4_coefficients'], 'DENSE_LR4_COORDINATE_SHARING')
                        d = entry['diagnostics']
                        residuals[arm].append(d['normal_equation_relative_residual'])
                        b.need(not d['regularized'] and not d['clipped'] and d['normal_equation_relative_residual'] <= b.c.SOLVE_TOL,
                               'DEVELOPMENT_SOLVE_RESIDUAL')
                        if arm != 'Both':
                            G, z = b.explicit_stack(cvs[s], observed_pilots[j:j+5])
                            k = 0 if arm == 'P-only' else 1
                            b.need(b.c.rel(c, np.linalg.lstsq(G[:, k:k+1], z, rcond=None)[0]) <= b.c.TOL, 'DEVELOPMENT_SCALAR_REFERENCE')
                    arm = row['arms']['Both']
                    b.need(np.array_equal(arm['coefficients'], existing['coefficients']['affine']), 'B2_BOTH_COORDINATE_BITWISE_REPLAY')
                    for method, key in (('Dense', 'dense-affine'), ('LR4', 'lr4-affine')):
                        b.need(np.array_equal(arm['predictions'][method], existing['predictions'][key]), 'B2_BOTH_PREDICTION_BITWISE_REPLAY')
                    max_coord = max(max_coord, b.c.rel(arm['coefficients'], historical['coefficients']['affine']))
                    max_dense = max(max_dense, b.c.rel(arm['predictions']['Dense'], historical['predictions']['dense-affine']))
                    max_lr = max(max_lr, b.c.rel(arm['predictions']['LR4'], historical['predictions']['lr4-affine']))
                b.need(max(max_coord, max_dense, max_lr) <= b.c.TOL and array_hash(observed_pilots) == before,
                       'HISTORICAL_BOTH_REPLAY_OR_INPUT_MUTATION')
                replay.append(dict(condition=condition, snr_db=s, trajectory_id=trajectory,
                    scored_frames=6, B2_coordinate_bitwise_equal=True, B2_dense_bitwise_equal=True,
                    B2_LR4_bitwise_equal=True, historical_coordinate_relative_error=max_coord,
                    historical_dense_relative_error=max_dense, historical_LR4_relative_error=max_lr, status='PASS'))
                reports.append(dict(condition=condition, snr_db=s, trajectory_id=trajectory,
                    frames_scored=6, finite_outputs=True, coordinate_shapes={'P-only': [1], 'M-only': [1], 'Both': [2]},
                    solve_relative_residual_max={arm: float(max(v)) for arm, v in residuals.items()},
                    same_source_centering_and_residual=True, dense_LR4_coordinate_shared=True,
                    unchanged_pilot_inputs=True, scalar_reference_equivalence=True, H_sha256=array_hash(H)))
            print({'development_smoke': condition, 'trajectory_id': trajectory,
                   'snr_cells_completed': 2, 'status': 'PASS', 'performance_values_reported': False}, flush=True)
    return dict(PASS=True, namespace=registry['namespace'], conditions=list(b.CONDITIONS), snrs=list(b.SNRS),
        distinct_development_trajectories=6, condition_SNR_cells=6, trajectory_SNR_cases=12,
        scored_frame_arm_deployment_checks=432, unexpected_solve_failures=0,
        formal_B3_generated=0, formal_B3_bootstrap_executed=False, B2_outcomes_viewed=False,
        performance_values_reported=False, records=reports), replay
