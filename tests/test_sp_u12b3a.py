"""B3A tests use deterministic fixtures and public frozen metadata only."""
import ast, inspect, subprocess, sys, textwrap
import numpy as np
import pytest
from scripts import sp_u12b3a_common as b
from scripts import sp_u12b3a_validation as v
from scripts.sp_u12b3a_ablation import SingleDirectionCV, solve_scalar, TechnicalSolveFailure, deployed_arms, coordinate_fields

def test_scalar_analytic_complex_and_cross_pilot_references():
    rows = v.scalar_audit()
    assert len(rows) == 32 and {r['direction'] for r in rows} == {'P', 'M'}
    assert all(r['status'] == 'PASS' and r['unknowns'] == 1 for r in rows)

@pytest.mark.parametrize('direction', ['P', 'M'])
def test_single_direction_constructs_own_design_and_source_residual(direction, monkeypatch):
    cv, pilots, _, _ = b.fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError('BOTH_SOLVE_OR_FULL_FRAME_OBJECTIVE_USED')
    monkeypatch.setattr(type(cv), 'fit', forbidden)
    monkeypatch.setattr(type(cv), 'frame_objective', forbidden)
    fit = SingleDirectionCV(cv, direction)
    coefficient, diag = fit.fit(pilots)
    G, z = b.explicit_stack(cv, pilots)
    k = 0 if direction == 'P' else 1
    reference = np.linalg.lstsq(G[:, k:k+1], z, rcond=None)[0]
    np.testing.assert_allclose(coefficient, reference, rtol=b.c.TOL, atol=1e-11)
    assert np.asarray(diag['gram']).shape == (1, 1) and np.asarray(diag['rhs']).shape == (1,)

def test_orthogonal_coupled_and_zero_direction_mathematics():
    invariant, coupled = v.geometry_audit()
    assert invariant['orthogonal']['PASS'] and coupled['PASS']
    assert len(invariant['zero_direction_cases']) == 2
    assert all(r['objective_minimizer_set_equivalent'] and r['legacy_Both_singular_solve_rejected']
               and r['dense_prediction_equivalent'] and r['LR4_prediction_equivalent'] for r in invariant['zero_direction_cases'])
    assert not invariant['literal_unique_Both_solve_invariance']

@pytest.mark.parametrize('zero', ['P', 'M'])
def test_zero_scalar_excitation_raises_without_fallback(zero):
    cv, pilots, _, _ = b.fixture(zero=zero)
    with pytest.raises(TechnicalSolveFailure, match='ZERO_SCALAR_EXCITATION'):
        SingleDirectionCV(cv, zero).fit(pilots)

@pytest.mark.parametrize('S,rhs', [(0., 0.), (0., 1.), (-1., 1.), (np.nan, 1.), (1., np.inf)])
def test_scalar_bad_normal_equation_is_rejected(S, rhs):
    with pytest.raises(TechnicalSolveFailure):
        solve_scalar(S, rhs)

def test_positive_scalar_gram_zero_response_has_zero_coordinate():
    np.testing.assert_array_equal(solve_scalar(2., 0.), [0.])

def test_both_bitwise_replay_and_dense_LR4_shared_coordinates():
    cv, p, arrays, factors = b.fixture(frames=12)
    got = deployed_arms(p, arrays, 10, cv, factors['P'], factors['M'])
    assert len(got) == 8 and [r['frame_id'] for r in got] == list(range(4, 12))
    for row in got:
        j = row['frame_id']
        coefficients, _ = cv.fit(p[j-4:j+1], modules=('P', 'M'))
        np.testing.assert_array_equal(row['arms']['Both']['coefficients'], coefficients)
        for arm, entry in row['arms'].items():
            assert entry['dense_coefficients'] is entry['lr4_coefficients']
            assert entry['coefficients'].shape == ((2,) if arm == 'Both' else (1,))
            assert all(x.shape == (13,) and np.isfinite(x).all() for x in entry['predictions'].values())

def test_stream_warmup_and_future_frame_invariance():
    cv, p, arrays, f = b.fixture(frames=10)
    assert deployed_arms(p[:4], arrays, 10, cv, f['P'], f['M']) == []
    original = deployed_arms(p[:5], arrays, 10, cv, f['P'], f['M'])[0]
    changed = p.copy(); changed[5:] *= 1000
    current = deployed_arms(changed, arrays, 10, cv, f['P'], f['M'])[0]
    for arm in original['arms']:
        np.testing.assert_array_equal(original['arms'][arm]['coefficients'], current['arms'][arm]['coefficients'])
        for deployment in ('Dense', 'LR4'):
            np.testing.assert_array_equal(original['arms'][arm]['predictions'][deployment], current['arms'][arm]['predictions'][deployment])

@pytest.mark.parametrize('shape', [(5, 63), (5, 64, 1), (6, 64), (0, 64)])
def test_scalar_invalid_window_shapes(shape):
    cv, _, _, _ = b.fixture()
    with pytest.raises(ValueError):
        SingleDirectionCV(cv, 'P').fit(np.ones(shape))

def test_no_new_regularization_or_automatic_selection():
    from scripts import sp_u12b3a_ablation as module
    tree = ast.parse(inspect.getsource(module))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in ('pinv', 'inv', 'lstsq', 'clip') for n in calls)
    assert all(name not in inspect.signature(deployed_arms).parameters for name in ('condition', 'truth', 'target', 'oracle'))
    with pytest.raises(ValueError):
        SingleDirectionCV(b.fixture()[0], 'AUTO')

@pytest.mark.parametrize('arm,coefficients,expected', [
    ('P-only', [1.], {'coordinate_p': 1., 'coordinate_m': None}),
    ('M-only', [2.], {'coordinate_p': None, 'coordinate_m': 2.}),
    ('Both', [1., 2.], {'coordinate_p': 1., 'coordinate_m': 2.})])
def test_nullable_coordinate_schema(arm, coefficients, expected):
    assert coordinate_fields(arm, coefficients) == expected

def test_frozen_twelve_contrast_family_and_signs():
    contrasts = b.planned_contrasts()
    assert len(contrasts) == 12
    assert len({(r['condition'], r['snr_db'], r['comparison']) for r in contrasts}) == 12
    assert {r['snr_db'] for r in contrasts} == {10, 20}
    assert {r['first'] for r in contrasts} == {'P-only/LR4', 'M-only/LR4'}

def test_dummy_bootstrap_shared_draws_pool_ratio_before_dB():
    e = np.arange(1, 25, dtype=float).reshape(2, 3, 4)
    t = np.arange(2, 10, dtype=float).reshape(2, 4)
    draws = np.array([[0, 0, 1, 3], [2, 2, 2, 1]])
    got = b.bootstrap_dummy(e, t, draws)
    for k, idx in enumerate(draws):
        nmse = np.empty((2, 3))
        for s in range(2):
            for arm in range(3):
                nmse[s, arm] = 10*np.log10(sum(e[s, arm, i] for i in idx)/sum(t[s, i] for i in idx))
        np.testing.assert_allclose(got[k], np.column_stack((nmse[:, 0]-nmse[:, 2], nmse[:, 1]-nmse[:, 2])))

def test_B2_sealed_manifest_metadata_unchanged():
    m = b.seal_metadata()
    assert m['file_count'] == 35031 and m['condition_SNR_cells'] == 35

def test_firewall_denies_B2_outcomes_and_historical_writes():
    code = '''from scripts import sp_u12b3a_common as b
b.firewall()
paths = ['outputs/sp_u12b2b_sealed/aggregate_results.csv',
         'outputs/sp_u12b2b_sealed/raw_evaluation/Profile/0000.npz',
         'outputs/sp_u12b2b_sealed/cache/Profile/H_0000.npz',
         'outputs/t20b_formal_nmse.csv']
for name in paths:
    try:
        (b.ROOT/name).open('rb')
    except PermissionError:
        continue
    raise RuntimeError('FIREWALL_FAILURE')
try:
    b.MANIFEST.open('w')
except PermissionError:
    pass
else:
    raise RuntimeError('WRITE_FIREWALL_FAILURE')
assert b.seal_metadata()['status'] == 'SEALED_COMPLETE_UNVIEWED'
'''
    result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code], cwd=b.ROOT, capture_output=True, text=True)
    assert result.returncode == 0

def test_actual_development_and_seal_receipts():
    smoke = b.read(b.ROOT / 'outputs/sp_u12b3a_development_smoke.json')
    assert smoke['PASS'] and smoke['trajectory_SNR_cases'] == 12 and smoke['unexpected_solve_failures'] == 0
    assert not smoke['B2_outcomes_viewed'] and not smoke['formal_B3_bootstrap_executed']
    seal = b.read(b.ROOT / 'outputs/sp_u12b3a_b2_seal_audit.json')
    assert seal['PASS'] and seal['seal_manifest_sha256'] == b.MANIFEST_SHA
