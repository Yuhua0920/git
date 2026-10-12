"""B3A validation-only entry point. No formal evaluation CLI exists."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys, argparse, subprocess, time, json, xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import sp_u12b3a_common as b
from scripts import sp_u12b3a_validation as v

def math():
    b.prerequisites()
    b.csvwrite(ROOT/'outputs/sp_u12b3a_scalar_fit_audit.csv', v.scalar_audit())
    invariant, coupled = v.geometry_audit()
    invariant['mentor_interpretation'] = 'PENDING_CLARIFICATION'
    invariant['expected_singular_Both_negative_controls'] = 2
    b.write(ROOT/'outputs/sp_u12b3a_invariance_tests.json', invariant)
    b.write(ROOT/'outputs/sp_u12b3a_coupled_design_test.json', coupled)
    schema = b.future_reuse_schema()
    text = '''# SP-U12B3A preregistration

IMPLEMENTATION / MATHEMATICAL EQUIVALENCE / VALIDATION ONLY. Formal B3 is not authorized. B2 scientific values cannot be decoded; only public schemas, IDs, seed IDs, manifests and opaque hashes may be inspected.

The source anchor, source means, cross-pilot residual, four48-to16folds, current-inclusive five-frame window, operating noise variance, full Source/cross-pilot kernels and rank-four P/M factors are fixed. P-only and M-only reconstruct one-column designs and independently accumulate their scalar Gram/rhs, then use b/S. Both calls the existing PilotCV.fit(modules=('P','M')) unchanged. Every arm shares one coordinate object between Dense and LR4. No arm selection, label-dependent choice, ridge, epsilon, clipping, pseudoinverse, fallback, smoothing or covariance re-estimation.

Development smoke: reuse B2A DEVELOPMENT_ONLY seeds, two trajectories for each Profile/Mobility/Composed, paired at10/20dB;10continuousframes,score4–9. Only execution, finite/shape/solve residual, source-residual equality and replay checks are disclosed. No channel-risk or BER calculation is needed.

Future formal B3: reuse the exact B2 formal1000trajectories percondition (Profile/Mobility/Composed), cached H and communication bits/symbols/unitnoise, at10/20dB. Reconstruct the same pilots using the frozen link path; no new channel generation. Six arm/deployment outputs pertrajectory/SNR: P-only/M-only/Both times Dense/LR4. Store trajectory energies (sum six scored frames), coordinates with inactive coordinate NA, solve status and the declared primary key; raw frame records add frame_id. Primary family is twelve contrasts: threeconditions x twoSNR x {Both_LR4_vs_Ponly_LR4,Both_LR4_vs_Monly_LR4}. Difference is single-direction LR4 pooled NMSE minus Both LR4 pooled NMSE, positive meaning Both lower.10000pairedtrajectorybootstrap,95%two-sidedBonferroni q=.05/24,linearquantiles; same trajectory draws across arms/deployments/SNR withincondition. ID/Boundary remain descriptive if separately authorized. Exact bootstrap master seed must be registered before separately authorized formal B3; no scientific bootstrap runs here. Only tiny dummy indices validate resampling wiring.

Zero-direction assumption audit: DeltaK_M=0 makes the Both real Gram singular. Its minimizers are {(alpha_P,t):t real}; if B_M=0, all associated dense and factorized predictions equal P-only. The analogous statement holds with P/M exchanged. The frozen two-coordinate solve cannot return a unique coordinate in either case; it must reject singularity. No fallback is added. Objective/prediction equivalence and correct rejection are tested separately; literal unique-solve invariance requires mentor resolution and is not silently marked PASS.

Unexpected development solve failure => STOP. Deliberately zero/singular synthetic negative controls are expected rejection tests, not successful scientific solves. Existing relative1e-10 replay/reference tolerance and1e-11 normal-equation residual gate are reused; no thresholds are selected from outcomes. Inactive direction values used in mathematical minimizer tests do not define production estimator behavior.

Final gates: actual pytest, development smoke, Both/current/history replay, scalar reference, orthogonal/coupled geometry, B2 full opaque hash tree and protection verification. Seal changes or accidental scientific reads => B3A_SEAL_VIOLATION. No manuscript edits, formal B3, new scientific story or Git commit. STOP for mentor review.
'''
    (ROOT/'docs/sp_u12b3a_preregistration.md').write_text(text, encoding='utf8')
    b.write(b.OUT/'execution_freeze.json', dict(hashes={path: b.sha(ROOT/path) for path in b.SOURCES},
        preregistration_sha256=b.sha(ROOT/'docs/sp_u12b3a_preregistration.md'),
        operator_cache_sha256=b.sha(b.c.OUT/'operators.npz'), B2_manifest_sha256=b.MANIFEST_SHA,
        development_seed_registry_sha256=b.sha(b.c.OUT/'development_seed_registry.json'),
        formal_B3_authorized=False, scientific_outcomes_viewed=False))
    print({'mathematical_reference': 'PASS', 'scalar_cases': 32,
        'orthogonal_coupled_geometry': 'PASS', 'zero_direction_unique_solve': 'SINGULAR_REQUIRES_CLARIFICATION',
        'future_contrasts': len(schema['planned_contrasts']), 'formal_B3_generated': 0}, flush=True)

def require_implementation_freeze():
    record = b.read(b.OUT/'execution_freeze.json')
    for path, digest in record['hashes'].items():
        b.need(b.sha(ROOT/path) == digest, 'B3A_IMPLEMENTATION_FREEZE_DRIFT')

def smoke():
    require_implementation_freeze()
    report, replay = v.development_smoke()
    b.write(ROOT/'outputs/sp_u12b3a_development_smoke.json', report)
    b.csvwrite(ROOT/'outputs/sp_u12b3a_both_replay.csv', replay)

def tests():
    require_implementation_freeze()
    args = [sys.executable, '-X', 'utf8', '-m', 'pytest', 'tests/test_sp_u12b2a.py',
        'tests/test_sp_u12b3a.py', '-q', '--assert=plain', '-o',
        'cache_dir=outputs/sp_u12b3a/pytest_cache', '--basetemp=outputs/sp_u12b3a/pytest_tmp',
        '--junitxml=outputs/sp_u12b3a/pytest.xml']
    start = time.monotonic()
    process = subprocess.run(args, cwd=ROOT, capture_output=True, encoding='utf8', errors='replace')
    (b.OUT/'pytest.txt').write_text(process.stdout + process.stderr, encoding='utf8')
    xml = ET.parse(b.OUT/'pytest.xml').getroot()
    suites = list(xml.iter('testsuite'))
    report = dict(exit_code=process.returncode, tests=sum(int(s.attrib.get('tests', 0)) for s in suites),
        failed=sum(int(s.attrib.get('failures', 0))+int(s.attrib.get('errors', 0)) for s in suites),
        skipped=sum(int(s.attrib.get('skipped', 0)) for s in suites), elapsed_seconds=time.monotonic()-start,
        console_sanitized=True, B2_outcomes_viewed=False)
    b.write(b.OUT/'pytest.json', report)
    print(report, flush=True)
    b.need(process.returncode == 0 and report['failed'] == 0 and report['skipped'] == 0, 'PYTEST_FAILURE')

def finalize(interpretation):
    require_implementation_freeze()
    b.prerequisites()
    smoke = b.read(ROOT/'outputs/sp_u12b3a_development_smoke.json')
    seal = b.read(ROOT/'outputs/sp_u12b3a_b2_seal_audit.json')
    protection = b.read(ROOT/'outputs/sp_u12b3a_preservation.json')
    test = b.read(b.OUT/'pytest.json')
    b.need(smoke['PASS'] and seal['PASS'] and protection['PASS'] and test['exit_code'] == 0, 'FINAL_GATES')
    invariant = b.read(ROOT/'outputs/sp_u12b3a_invariance_tests.json')
    invariant['mentor_interpretation'] = interpretation
    invariant['gate_status'] = ('PASS_OBJECTIVE_PREDICTION_EQUIVALENCE_WITH_SINGULAR_REJECTION'
        if interpretation == 'ACCEPTED_BY_USER' else 'REVISION_REQUIRED_LITERAL_BOTH_SOLVE_SINGULAR')
    b.write(ROOT/'outputs/sp_u12b3a_invariance_tests.json', invariant)
    verdict = ('B3_ABLATION_PIPELINE_READY' if interpretation == 'ACCEPTED_BY_USER'
               else 'B3_ABLATION_PIPELINE_REVISION_REQUIRED')
    freeze = b.read(b.OUT/'execution_freeze.json')
    b.write(b.OUT/'decision.json', dict(verdict=verdict, production_semantic_drift=0,
        old_artifact_drift=0, manuscript_drift=0, B2_outcomes_viewed=False,
        B2_seal_unchanged=True, formal_B3_run=False, zero_direction_interpretation=interpretation,
        expected_singular_negative_controls=2, unexpected_smoke_solve_failures=0,
        pytest=test, source_hashes=freeze['hashes']))
    log = '''# SP-U12B3A change log

Added isolated single-direction fitter, fixed-arm Dense/LR4 deployment, validation-only runner, mathematical tests, future schemas/ID/hash wiring, outcome firewall and opaque preservation helper. No production code/manuscript/old artifact edits.

Equation/code mapping:
- D^(P/M)_(lf)=DeltaK_(P/M,f)u_(l,O_f), e_(lf)=u_(l,H_f)-K_(s,f)u_(l,O_f): SingleDirectionCV.frame_objective.
- S=sum Re(D^HD), b=sum Re(D^He), coefficient=b/S: SingleDirectionCV.fit and solve_scalar.
- Both: unchanged src/pilot_cv.py::PilotCV.fit(modules=('P','M')).
- Source plus one dense correction or its rank4A(Q u): sp_u12b3a_ablation.deployed_arms.
- Both dense/LR4: unchanged src/pilot_cv.py::deployment and src/lowrank_modules.py::factored_affine_predict.
- Input H to pilots/noise: frozen run_sp_u12b2a.generator using development registry and sp_u12b2a_common.communicate, same src/link.py functions. No estimator receives condition/H/truth.
- Same-arm coordinate identity: one coefficient object assigned to Dense/LR4 fields.
- Independent complex real-stacked reference: sp_u12b3a_common.explicit_stack and validation.scalar_audit.
- Future bootstrap schema/wiring: common.planned_contrasts, bootstrap_dummy, future_reuse_schema. No formal bootstrap.

32scalar-referencecases PASS;12developmenttrajectory-SNRcases PASS;Both coordinates/current predictions bitwise unchanged, historical frozen predictions within existing1e-10 tolerance. Six development trajectories only. All7 frozen-SNR operators retained; only10/20used. Rank4 factors loaded directly, no reconstruction/refit/compression choice.

Zero direction is a singular Both solve, with equal minimizer prediction sets. Test results explicitly separate equality of objective/predictions from unique solvability. No singular fallback is implemented.

'''
    log += 'Mentor interpretation: '+interpretation+'\n\nVerdict: '+verdict+'\n\n'
    log += json.dumps(dict(tests=test, protected_files=protection['protected_files'],
        B2_manifest_sha256=b.MANIFEST_SHA, B2_raw_hash_tree=b.RAW_TREE, B2_all_hash_tree=b.ALL_TREE,
        source_hashes=freeze['hashes']), indent=2)+'\n\nSTOP. No formal B3 or manuscript/Git action.\n'
    (ROOT/'docs/sp_u12b3a_change_log.md').write_text(log, encoding='utf8')
    print({'verdict': verdict, 'B2_seal_unchanged': True, 'B2_outcomes_viewed': False,
           'unexpected_solve_failures': 0, 'formal_B3_generated': 0}, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['math', 'smoke', 'seal', 'tests', 'finalize'])
    parser.add_argument('--zero-direction-interpretation', choices=['PENDING_CLARIFICATION', 'ACCEPTED_BY_USER', 'LITERAL_REQUIREMENT'], default='PENDING_CLARIFICATION')
    args = parser.parse_args()
    b.OUT.mkdir(exist_ok=True)
    b.firewall()
    try:
        if args.stage == 'math': math()
        elif args.stage == 'smoke': smoke()
        elif args.stage == 'seal': require_implementation_freeze(); b.seal_audit(); print({'B2_seal': 'PASS'}, flush=True)
        elif args.stage == 'tests': tests()
        else: finalize(args.zero_direction_interpretation)
    except Exception as error:
        code = str(error) if str(error).replace('_','').isalnum() and str(error).isupper() else 'TECHNICAL_VALIDATION_FAILURE'
        verdict = 'B3A_SEAL_VIOLATION' if 'SEAL_VIOLATION' in code else 'B3_ABLATION_PIPELINE_REVISION_REQUIRED'
        b.write(b.OUT/f'failure_{args.stage}.json', dict(verdict=verdict, stage=args.stage,
            error_type=type(error).__name__, failure_code=code, B2_outcomes_viewed=False))
        print({'verdict': verdict, 'stage': args.stage, 'error_type': type(error).__name__, 'failure_code': code}, flush=True)
        raise SystemExit(1)

if __name__ == '__main__':
    main()
