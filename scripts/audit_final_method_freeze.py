"""T20A governance only: immutable bytes, literature matrix and sealed design.

No scientific module import, generator, estimator evaluation or covariance fit.
NumPy is used exclusively to load/hash existing array bytes and rank4 prefixes.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
COMMIT = '456bbb64b7907186a6cabf0c7c5b61bb427a2eae'
FOLD = '16594a306157e0b6ac114100ddebbba262f8e4205a98c8f48726e5519e55ab18'
DOC_NAMES = ('final_method_definition', 'final_evidence_inventory', 'closest_work_matrix',
             'contribution_candidates', 'forbidden_claims', 'final_claim_evidence_matrix',
             'final_reviewer_red_team', 'formal_test_protocol')
DOCS = tuple(f'docs/t20a_{name}.md' for name in DOC_NAMES)
METHOD = 'Pilot-Driven Low-Rank Affine LMMSE Adaptation'
METHODS = ('Perfect CSI', 'R3 LS+linear', 'Source LMMSE', 'Hard', 'Convex',
           'Dense Affine', 'Final LR4 Affine', 'Matched LMMSE reference')
CONDITIONS = ('C30/100', 'B30/100', 'C120/100', 'B120/100', 'C30/200')
ROLES = ('sionna_seed', 'torch_seed', 'numpy_seed', 'bits_seed_tx0', 'bits_seed_tx1', 'noise_seed')
TECHNICAL_FIELDS = tuple(range(2, 17))


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')


def verify_hashes(hashes, root=ROOT):
    mismatch = [p for p, expected in hashes.items() if sha(root / p) != expected]
    if mismatch:
        raise ValueError('Historical content mismatch: ' + ', '.join(mismatch))
    return len(hashes)


def literature():
    r = yaml.safe_load((ROOT / 'configs/t20a_literature.yaml').read_text(encoding='utf8'))
    if r['search_date'] != '2026-10-03' or len(r['fields']) < 20:
        raise ValueError('Literature date/schema required')
    ids = [p['id'] for p in r['papers']]
    if len(set(ids)) != len(ids) or not set('ABCDEF').issubset(ids):
        raise ValueError('Required closest works missing/duplicated')
    for p in r['papers']:
        if len(p['values']) != len(r['fields']) or any(not isinstance(v, str) or not v.strip() for v in p['values']):
            raise ValueError('Complete explicit matrix required: ' + p['id'])
        if p['full_text'] == 'unavailable' and any(p['values'][j] != 'UNKNOWN' for j in TECHNICAL_FIELDS):
            raise ValueError('Unread full text cannot support technical fields: ' + p['id'])
        if not p['source'].startswith('https://'):
            raise ValueError('Primary source link required')
    return r


def render_matrix(review):
    lines = ['# Updated closest-work matrix', '',
        'Search cutoff:2026-10-03. Focused closest-work audit, not exhaustive coverage.', '',
        'Within the literature verified so far, selection, continuous filter combination,',
        'observation-dependent LMMSE mixtures and compact adaptation are established.',
        'UNKNOWN is an unresolved field, never evidence of absence. No cross-paper performance comparison.', '',
        'Direct Wiley/IEEE access failures are retained; publisher-indexed original passages and',
        'author manuscripts are distinguished from abstract-only records. Technical details',
        'are not inferred from titles or secondary summaries. Current T17A history is preserved.', '',
        '## Search ledger', '']
    lines += ['- ' + q for q in review['queries']]
    for p in review['papers']:
        lines += ['', f"## {p['id']} — {p['values'][0]}", '',
                  f"[Primary source]({p['source']}). {p['evidence']}", '']
        if p.get('additional_source'):
            lines += [f"[Additional original/metadata record]({p['additional_source']}).", '']
        lines += ['| Field | Verified finding |', '|---|---|']
        lines += [f'| {f} | {v} |' for f, v in zip(review['fields'], p['values'])]
    lines += ['', '## Five collision questions', '',
        '| Question | Overlap | Concrete difference | Remaining risk |', '|---|---|---|---|',
        '| Q1 Continuous Mei selection? | Held-out observation score and statistics bank | Two unrestricted real coordinates rather than a candidate argmin | HIGH; continuous extension alone is incremental, not priority evidence |',
        '| Q2 Gui LMS replaced by LMMSE? | Base plus differential filters and continuous combination | Frozen covariance operators, two-variable causal cross-pilot LS | HIGH; affine combination itself is established |',
        '| Q3 Repackaged adaptive low-rank LMMSE? | Rank-reduced statistical estimation | Present correction-only factors documented | HIGH/UNRESOLVED; Özdemir full text unavailable, distinction unconfirmed |',
        '| Q4 Classical PEFT adapter switching? | Compact library on shared base; LoRAT also transfers adapters | Current analytical pilot coordinates and linear correction operators | HIGH broad overlap; Lee/Choi technical details UNKNOWN; LoRAT learned distribution mapping differs |',
        '| Q5 A-MMSE difference only no training? | Deployable linear factors and rank trade-off | Source filter retained; two correction factors composed online by exact CV rather than a trained full filter | HIGH for generic rank/no-network claim; no matched benchmark against their implementation |', '',
        'Potential collision: true. Critical verified exact duplication: false. This is a limited evidence finding,',
        'not clearance of all prior art. C/F full-text gaps remain unresolved; no uniqueness or priority verdict.',
        'The retained candidates are scoped integration/evaluation contributions. Novelty positioning may',
        'need revision after additional access, without changing the frozen receiver or formal design.', '']
    return '\n'.join(lines)


def seeds(environment, trajectory, frame):
    """Integer reservation only; no channel/signal RNG."""
    if environment not in range(5) or trajectory not in range(600, 1600) or frame not in range(10):
        raise ValueError('Only preregistered formal allocation permitted')
    base = 2700000000 + environment * 1000000 + (trajectory - 600) * 100
    return tuple(base + (j + 1) * 10 + (frame if j >= 3 else 0) for j in range(6))


def namespace_audit():
    channel = {seeds(e, i, 0)[:3] for e in range(5) for i in range(600, 1600)}
    comm = {seeds(e, i, f)[3:] for e in range(5) for i in range(600, 1600) for f in range(10)}
    if len(channel) != 5000 or len(comm) != 50000:
        raise ValueError('Formal namespace self-collision')
    baseline = load(OUT / 't20a_history_baseline.json')
    paths = [p for p in baseline['historical_sha256'] if 'manifest' in Path(p).name and p.endswith('.csv')]
    read = []
    for p in paths:
        with (ROOT / p).open(encoding='utf8', newline='') as f:
            rows = csv.DictReader(f)
            if not set(ROLES).issubset(rows.fieldnames or ()):
                continue
            for row in rows:
                old = tuple(int(row[k]) for k in ROLES)
                if old[:3] in channel or old[3:] in comm:
                    raise ValueError('Historical seed tuple collision: ' + p)
            read.append(p)
    return dict(integer_seed_reservation_only=True, formal_channel_tuples=5000,
                formal_communication_tuples=50000, historical_manifests_checked=read,
                tuple_collision_count=0, signal_rng_calls=0, formal_manifest_materialized=False)


def frozen_arrays():
    m = load(OUT / 't18a_preflight.json')['frozen_method']
    count = 0
    rank4 = {}
    shapes = {}
    for key in ('operator_artifact', 'candidate_artifact', 'factor_artifact'):
        info = m[key]
        if sha(ROOT / info['path']) != info['sha256']:
            raise ValueError('Method artifact drift: ' + info['path'])
        with np.load(ROOT / info['path'], allow_pickle=False) as arrays:
            for name, expected in info['array_sha256'].items():
                a = arrays[name]
                if hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() != expected:
                    raise ValueError('Method array drift: ' + name)
                count += 1
            if key == 'factor_artifact':
                for mod in ('P', 'M'):
                    for snr in (5, 10, 20):
                        for part in ('A', 'Q'):
                            a = arrays[f'{part}_{mod}_{snr}']
                            a = a[:, :4] if part == 'A' else a[:4, :]
                            name = f'{mod}_{snr}_{part}'
                            h = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
                            if h != m['rank4_array_sha256'][name] or a.dtype != np.complex128:
                                raise ValueError('Rank4 prefix drift: ' + name)
                            rank4[name] = h
                            shapes[name] = list(a.shape)
    e = load(OUT / 't16e_pilot_cv_preflight.json')
    f = load(OUT / 't16e_pilot_folds.json')
    check = dict(f)
    expected = check.pop('sha256')
    digest = hashlib.sha256(json.dumps(check, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if f != e['pilot_fold_definition'] or digest != FOLD or expected != FOLD:
        raise ValueError('Frozen fold definition drift')
    if sha(OUT / 't16e_pilot_folds.json') != e['fold_artifact_sha256']:
        raise ValueError('Fold file bytes drift')
    labels = [((i // 16) % 64 // 16 + i % 16 // 4) % 4 for i in f['selected']]
    if labels != f['labels'] or any(len(v['heldout']) != 16 or len(v['observed']) != 48 for v in f['folds']):
        raise ValueError('Physical fold geometry drift')
    return dict(artifacts={k: m[k] for k in ('operator_artifact', 'candidate_artifact', 'factor_artifact')},
                arrays_verified=count, rank4_array_sha256=rank4, rank4_shapes=shapes,
                fold_sha256=digest, fold_file_sha256=e['fold_artifact_sha256'],
                covariance_fits=0, svd_calls=0, estimator_evaluations=0)


def assignment(path, name):
    for node in ast.parse(Path(path).read_text(encoding='utf8')).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Frozen assignment missing: ' + name)


def constants_audit():
    p = ROOT / 'src/frozen_affine_validation.py'
    if assignment(p, 'RANK') != 4 or assignment(p, 'POWER') != 9.941134822523972:
        raise ValueError('Rank/source power changed')
    variances = assignment(p, 'VARIANCES')
    if variances != {5: 3.1436628565989504, 10: .9941134822523973, 20: .09941134822523973}:
        raise ValueError('Source noise reference drift')
    cfg = yaml.safe_load((ROOT / 'configs/base.yaml').read_text(encoding='utf8'))['system']
    expected = dict(nt=2, nr=2, nfft=64, active_carriers=64, num_symbols=16,
                    delta_f_hz=30000, sample_rate_hz=1920000, cp_len=16,
                    fft_norm='ortho', modulation='qpsk_gray', total_power=1.0)
    if any(cfg[k] != v for k, v in expected.items()):
        raise ValueError('Frozen base system drift')
    if cfg['sample_rate_hz'] != cfg['nfft'] * cfg['delta_f_hz']:
        raise ValueError('Sampling identity violated')
    return dict(system=expected, dF=16, dT=4, rank_per_module=4, module_count=2,
                source='C30/100', development=['B30/100', 'C120/100'], causal_window=5,
                coefficient_model='two unrestricted real coordinates', means='source',
                factor_weight='respective development pilot covariance plus noise',
                cross_pilot_operators='exact uncompressed frozen T16E',
                source_noise_reference=9.941134822523972, noise_variances=variances)


def claim_states():
    d = load(OUT / 't19d_results.json')
    states = {k: d[k] for k in ('c6_claim_status', 'c7_claim_status', 'c8_claim_status',
              'c9_claim_status', 'c10_claim_status', 'c11_claim_status', 'c12_claim_status',
              't18a_materiality_status', 't19a_rank4_out_of_factor_warning')}
    expected = ['CONTRADICTED_BY_CURRENT_MECHANISM_TEST', 'CONTRADICTED/CANCELLATION_DOMINATED',
                'SUPPORTED_IN_FIXED_ENVIRONMENTS', 'INCONCLUSIVE', 'INCONCLUSIVE',
                'SUPPORTED_IN_C30_200_10_STATISTICS_REPLICATION',
                'CONTRADICTED_BY_HELDOUT_FACTOR_TRANSFER', 'PARTIAL', True]
    if list(states.values()) != expected:
        raise ValueError('Permanent historical claims changed')
    return states


def passed_summary(path, minimum):
    text = Path(path).read_text(encoding='utf8')
    lines = text.strip().splitlines()
    match = re.fullmatch(r'(\d+) passed in .+', lines[-1]) if lines else None
    if not match or int(match[1]) < minimum:
        raise ValueError('Actual complete passing pytest summary required: ' + str(path))
    return lines[-1]


def no_experiment_audit():
    tree = ast.parse(Path(__file__).read_text(encoding='utf8'))
    imports = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imports.extend(v.name for v in n.names)
        elif isinstance(n, ast.ImportFrom):
            imports.append(n.module)
        elif isinstance(n, ast.Call):
            name = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, 'id', '')
            if name in {'generate', 'generate_channel', 'fit', 'prepare', 'svd', 'eig', 'eigh', 'default_rng', 'normal', 'integers', 'zf_detect'}:
                raise ValueError('Experiment path in governance audit: ' + name)
    if any(v.startswith(('src.', 'scripts.', 'torch', 'sionna', 'scipy')) for v in imports):
        raise ValueError('Scientific import forbidden in governance audit')
    return dict(scientific_imports=[], generator_calls=0, H_generations=0,
                NMSE_evaluations=0, BER_evaluations=0, covariance_fits=0,
                model_training_calls=0, code_audit_passed=True)


def readiness_gate(*, critical_collision, core_evidence_adequate, protocol_complete,
                   executable_without_estimator_change, third_module_needed):
    method = (not critical_collision and core_evidence_adequate and
              executable_without_estimator_change and not third_module_needed)
    return method, method and protocol_complete and not critical_collision


def build_result(final=False):
    baseline = load(OUT / 't20a_history_baseline.json')
    if baseline['accepted_t19d_commit'] != COMMIT or not baseline['clean_working_tree_after_commit']:
        raise ValueError('Accepted clean T19D snapshot required')
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', COMMIT, 'HEAD'], cwd=ROOT)
    subject = subprocess.check_output(['git', 'show', '-s', '--format=%s', COMMIT], cwd=ROOT, text=True).strip()
    if subject != 'feat: reject transferable low-dimensional delay residual extension':
        raise ValueError('T19D commit subject drift')
    verify_hashes(baseline['historical_sha256'])
    old = load(OUT / 't19d_results.json')
    for key in ('frozen_history_sha256', 'numeric_artifact_sha256', 'source_sha256'):
        verify_hashes(old[key])
    # Git stores LF text; raw working-file SHA checks above retain original bytes.
    for p in (*old['source_sha256'], 'outputs/t19d_results.json'):
        blob = subprocess.check_output(['git', 'show', f'{COMMIT}:{p}'], cwd=ROOT)
        if blob.replace(b'\r\n', b'\n') != (ROOT / p).read_bytes().replace(b'\r\n', b'\n'):
            raise ValueError('Accepted Git text snapshot drift: ' + p)
    tracked = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True)
    if tracked.strip():
        raise ValueError('Tracked historical mutation: ' + tracked)
    no_experiment = no_experiment_audit()
    review = literature()
    matrix = ROOT / DOCS[2]
    rendered = render_matrix(review)
    if matrix.read_text(encoding='utf8') != rendered:
        raise ValueError('Closest-work matrix not reproducible from review input')
    for p in DOCS:
        if not (ROOT / p).read_text(encoding='utf8').strip():
            raise ValueError('Required document empty')
    arrays = frozen_arrays()
    constants = constants_audit()
    namespaces = namespace_audit()
    states = claim_states()
    subprocess.check_call(['git', 'diff', '--check'], cwd=ROOT)
    # All task files are new/untracked: also check their actual whitespace.
    sources = [*DOCS, 'configs/t20a_literature.yaml', 'scripts/audit_final_method_freeze.py', 'tests/test_final_method_freeze.py']
    for p in sources:
        if any(line.rstrip() != line for line in (ROOT / p).read_text(encoding='utf8').splitlines()):
            raise ValueError('New-file trailing whitespace: ' + p)
    startup = passed_summary(OUT / 't20a_startup_pytest.txt', 1864)
    final_summary = passed_summary(OUT / 't20a_pytest_full.txt', 1865) if final else None
    # Gate reflects scoped claims/limited literature findings, not performance alone.
    potential = True
    critical = False
    protocol_complete = all(v in (ROOT / DOCS[-1]).read_text(encoding='utf8') for v in METHODS)
    evidence_flags = dict(
        P1=load(OUT/'t15c_cdl_mismatch_results.json')['realistic_cdl_adaptation_gap_material'],
        P2=load(OUT/'t18a_results.json')['frozen_pipeline_interpolation_supported'],
        P3=load(OUT/'t16e_pilot_cv_results.json')['analytical_pilot_only_module_fitting_viable'],
        P4=load(OUT/'t16g_results.json')['lowrank_affine_modules_strong'],
        P5=load(OUT/'t18b_results.json')['c8_claim_status']=='SUPPORTED_IN_FIXED_ENVIRONMENTS',
        P6=load(OUT/'t19c_results.json')['delay_operator_deficiency_replication_supported'])
    evidence_paths = ('t15c_cdl_mismatch','t16a_lowrank','t16b_shared_basis','t16c_heldout',
        't16d_modular','t16e_pilot_cv','t16f','t16g','t17a','t17b','t18a','t18b','t19a','t19b','t19c','t19d')
    for name in evidence_paths:
        e = load(OUT / (name+'_results.json'))
        if not e['all_checks_passed'] or e['formal_test_used'] or not e['formal_test_sealed']:
            raise ValueError('Evidence acceptance/sealed contract drift: '+name)
    method_ready, paper_ready = readiness_gate(critical_collision=critical,
        core_evidence_adequate=all(evidence_flags.values()), protocol_complete=protocol_complete,
        executable_without_estimator_change=True, third_module_needed=False)
    result = dict(task='T20A', experiment_type='final method freeze, closest-work novelty gate, claim-evidence consolidation, and formal-test preregistration',
        method_candidate=METHOD, method_components_frozen=True, frozen_components=constants,
        accepted_t19d_commit=COMMIT, t19d_commit_subject=subject, clean_working_tree_after_t19d_commit=True,
        literature_gate_completed=True, literature_scope=review['search_scope'], search_cutoff=review['search_date'],
        closest_work_count=len(review['papers']), literature_full_text_unavailable=['C', 'F'],
        potential_novelty_collision=potential, critical_novelty_collision=critical,
        critical_collision_interpretation='no exact fatal duplication established in verified texts; unresolved full-text gaps remain, no priority verdict',
        retained_contribution_candidates={'C-A':'weaken', 'C-B':'retain', 'C-C':'weaken'},
        dropped_contribution_candidates=['generic affine-combination novelty', 'generic low-rank novelty', 'physical separability', 'universal generalization', 'runtime superiority', 'fully blind adaptation'],
        final_core_claims=['P1 scoped CDL covariance mismatch', 'P2 tested profile/mobility correction benefit', 'P3 tested100ns interpolation pilot-only inference', 'P4 tested interpolation compression retention and arithmetic/storage', 'P5 fixed-environment generalization and replication', 'P6 condition-specific delay boundary'],
        forbidden_claims=['universal profile/mobility separability', 'Taylor/Wiener perturbation mechanism', 'finite-shift separability', 'transferable delay residual module', 'universal environment generalization', 'actual runtime acceleration', 'fully blind adaptation', 'no target-domain offline information', 'capacity/throughput improvement', 'first/unprecedented/novel/no prior work/state-of-the-art'],
        method_freeze_criteria=dict(no_verified_fatal_collision=not critical, scoped_core_claim_evidence_adequate=all(evidence_flags.values()),
            formal_execution_requires_no_estimator_change=True, third_module_needed=False),
        scoped_claim_evidence_flags=evidence_flags,
        method_ready_to_freeze=method_ready, formal_test_protocol_complete=protocol_complete,
        paper_core_ready_for_formal_test=paper_ready,
        readiness_is_not_execution_authorization=True, formal_protocol=dict(version='T20A-v1', conditions=list(CONDITIONS),
            trajectories_per_condition=1000, trajectory_ids=[600,1599], frames_per_trajectory=10,
            scored_frames=[4,5,6,7,8,9], snr_db=[10,20], methods=list(METHODS), independent_unit='trajectory',
            bootstrap_resamples=10000, bootstrap_seed=20262001, primary_endpoint_families=3,
            primary_cell_comparisons=10, primary_CI='Bonferroni one-sided bootstrap bounds;0.005/0.995 quantiles',
            descriptive_CI='paired trajectory percentile95%', compression_margin_db=.50,
            formal_orchestration_implemented=False, formal_data_materialized=False),
        new_channel_generated=False, new_model_trained=False, formal_test_used=False, formal_test_sealed=True,
        no_experiment_audit=no_experiment, namespace_audit=namespaces, frozen_method_provenance=arrays,
        permanent_claim_states=states, historical_hashes_exact=True, historical_files_verified=len(baseline['historical_sha256']),
        historical_content_mismatches=0, no_tracked_historical_mutation=True,
        historical_baseline_sha256=sha(OUT/'t20a_history_baseline.json'),
        source_sha256={p:sha(ROOT/p) for p in sources},
        complexity=next(v for v in load(OUT/'t16g_results.json')['complexity'] if v['rank']==4),
        pytest=dict(startup=startup, full=final_summary), git_diff_check='PASS',
        completion_state='completed' if final else 'governance_audit_passed_regression_pending', all_checks_passed=bool(final and protocol_complete),
        warnings=['C/F primary full text unavailable; UNKNOWN retained',
            'Potential novelty collision and incremental positioning risk; no priority verdict',
            'Historical T18A PARTIAL and T19A rank4_out_of_factor_warning preserved',
            'Arithmetic/storage proxies exclude some operations; no wall-clock speed claim',
            'Historical Python base-location diagnostic persists although commands exit0',
            'Git autocrlf commit warnings; raw working-byte history audit separate from normalized Git text comparison',
            'Formal orchestration is a future explicit task; no formal data opened'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render-matrix', action='store_true')
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    if args.render_matrix:
        (ROOT / DOCS[2]).write_text(render_matrix(literature()), encoding='utf8')
        print('Literature matrix rendered from verified-source ledger; no experiment')
        return
    result = build_result(args.final)
    save(OUT / 't20a_results.json', result)
    print(f"T20A governance PASS: {result['closest_work_count']} closest works; {result['historical_files_verified']} historical files exact; factors/folds exact")
    print('Method ready:', result['method_ready_to_freeze'], '| Paper ready:', result['paper_core_ready_for_formal_test'])
    print('Formal test sealed; generator/NMSE/BER/fitting/training calls=0')
    print('Completion:', result['completion_state'], '| all_checks_passed:', result['all_checks_passed'])


if __name__ == '__main__':
    main()
