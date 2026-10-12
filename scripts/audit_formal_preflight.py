"""T20B read-only prerequisites and seed gate; no formal signal generation."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_final_method_freeze as frozen

COMMIT = '2b8549d7b027571f7b6a159646808b5c2c499b22'
PROTOCOL = 'docs/t20a_formal_test_protocol.md'
PROTOCOL_SHA = '196e9aeeebb2e1b4dfb9984fa35dc2187d5ab5d21a70f7c29d0713cdd705731b'
AMENDMENT = 'docs/t20b_formal_readiness_amendment.md'
RESULT = ROOT / 'outputs/t20b_preflight.json'
CONDITIONS = ('CDL-C-30-100', 'CDL-B-30-100', 'CDL-C-120-100',
              'CDL-B-120-100', 'CDL-C-30-200')


def readiness(a, b, c):
    if any(type(v) is not bool for v in (a, b, c)):
        raise ValueError('Endpoint outcomes must be Boolean PASS flags')
    return dict(paper_core_empirically_supported=a,
                paper_results_ready=a and b and c,
                paper_results_ready_partial=a and (b or c) and not (b and c))


def endpoint_status(passed):
    if type(passed) is not bool:
        raise ValueError('Boolean PASS flag required')
    return 'FORMALLY_SUPPORTED' if passed else 'NOT_SUPPORTED_BY_FORMAL_ENDPOINT'


def formal_reservation():
    """Reserve original integers only; never generate formal random tensors."""
    index = {}
    role_values = {k: set() for k in frozen.ROLES}
    for e, name in enumerate(CONDITIONS):
        for trajectory in range(600, 1600):
            for frame in range(10):
                for role, value in zip(frozen.ROLES, frozen.seeds(e, trajectory, frame)):
                    record = dict(environment=name, trajectory_id=trajectory,
                                  role=role, frame_id=None if role in frozen.ROLES[:3] else frame)
                    if value in index and index[value] != record:
                        raise ValueError('Internal formal individual seed collision')
                    index[value] = record
                    role_values[role].add(value)
    return index, role_values


def manifest_overlap(paths, formal_index):
    """Check every numeric seed column, including archived/reserved train rows."""
    collisions = []
    checked = []
    for path in paths:
        with (ROOT / path).open(encoding='utf8', newline='') as stream:
            reader = csv.DictReader(stream)
            keys = [k for k in reader.fieldnames or () if 'seed' in k.lower()]
            if not keys:
                continue
            row_count = 0
            seed_count = 0
            for row in reader:
                row_count += 1
                for key in keys:
                    try:
                        value = int(row[key])
                    except (TypeError, ValueError):
                        continue
                    seed_count += 1
                    if value in formal_index:
                        destination = formal_index[value]
                        collisions.append(dict(historical_path=path, historical_field=key,
                            historical_split=row.get('split'), historical_namespace=row.get('namespace'),
                            historical_trajectory_id=row.get('trajectory_id'),
                            historical_frame_id=row.get('frame_id'),
                            historical_communication_generated=row.get('communication_generated'),
                            seed=value, same_role=key == destination['role'], formal=destination))
            checked.append(dict(path=path, sha256=frozen.sha(ROOT / path),
                                seed_columns=keys, rows=row_count, numeric_seed_entries=seed_count))
    return dict(manifests_checked=checked, individual_seed_overlap_rows=len(collisions),
                unique_overlapping_seed_values=len({v['seed'] for v in collisions}),
                same_role_overlap_rows=sum(v['same_role'] for v in collisions),
                unique_same_role_overlapping_seed_values=len({v['seed'] for v in collisions if v['same_role']}),
                overlaps=collisions, all_individual_seeds_disjoint=not collisions)


def require_seed_independence(evidence):
    if not evidence['all_individual_seeds_disjoint']:
        raise ValueError('STOP: frozen formal seeds overlap historical manifest seeds')


def tracked_history():
    paths = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    return paths, {p: frozen.sha(ROOT / p) for p in paths}


def provenance():
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', COMMIT, 'HEAD'], cwd=ROOT)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subject = subprocess.check_output(['git', 'show', '-s', '--format=%s', COMMIT], cwd=ROOT, text=True).strip()
    if subject != 'feat: freeze final affine LMMSE method and preregister formal test':
        raise ValueError('Accepted T20A commit required')
    # Later authorized orchestration may evolve; accepted method hashes below
    # remain current-file checks, independent of historical preparation checks.
    accepted = frozen.load(ROOT / 'outputs/t20a_results.json')
    frozen.verify_hashes(accepted['source_sha256'])
    if frozen.sha(ROOT / PROTOCOL) != PROTOCOL_SHA:
        raise ValueError('Original formal protocol bytes changed')
    for path in accepted['source_sha256']:
        blob = subprocess.check_output(['git', 'show', f'{COMMIT}:{path}'], cwd=ROOT)
        if blob.replace(b'\r\n', b'\n') != (ROOT / path).read_bytes().replace(b'\r\n', b'\n'):
            raise ValueError('T20A Git snapshot mismatch: ' + path)
    baseline = frozen.load(ROOT / 'outputs/t20a_history_baseline.json')
    frozen.verify_hashes(baseline['historical_sha256'])
    return dict(T20A_commit=COMMIT, HEAD=head, T20A_subject=subject,
                clean_working_tree_after_T20A_commit=True, no_tracked_frozen_mutation=True,
                original_protocol_sha256=PROTOCOL_SHA,
                amendment_sha256=frozen.sha(ROOT / AMENDMENT),
                method_definition_sha256=accepted['source_sha256']['docs/t20a_final_method_definition.md'],
                T20A_result_sha256=frozen.sha(ROOT / 'outputs/t20a_results.json'),
                frozen_method=frozen.frozen_arrays(), historical_files_verified=len(baseline['historical_sha256']),
                historical_content_mismatches=0)


def environment():
    """Package-file audit only; do not call the generator smoke-test entrypoint."""
    from scripts.check_cdl_environment import environment_fingerprint
    old = frozen.load(ROOT / 'outputs/t15a_cdl_environment.json')
    actual = environment_fingerprint(ROOT / '.venv-cdl')
    if actual['files'] != old['isolated_payload_fingerprint']['files']:
        raise ValueError('Frozen CDL environment payload changed')
    return dict(cdl_payload=actual, historical_cdl_payload_exact=True,
                historical_environment_metadata={k: old[k] for k in
                    ('python', 'platform', 'torch', 'cuda_version', 'cuda_available', 'gpu', 'installed_versions')},
                metadata_source='frozen T15A record; current full package payload independently hashed')


def audit():
    proof = provenance()
    paths, before = tracked_history()
    index, role_values = formal_reservation()
    manifests = [p for p in paths if 'manifest' in Path(p).name.lower() and p.endswith('.csv')]
    overlap = manifest_overlap(manifests, index)
    env = environment()
    if (ROOT / 'outputs/t20b_formal_unseal.json').exists():
        raise ValueError('Preflight must precede unseal')
    frozen.verify_hashes(before)
    result = dict(task='T20B', completion_state='BLOCKED_BEFORE_UNSEAL' if overlap['overlaps'] else 'preflight_passed',
                  formal_test_used=False, formal_test_sealed=True, formal_test_unsealed=False,
                  formal_manifest_materialized=False, new_formal_H_generated=False,
                  generator_calls=0, formal_rng_calls=0, scientific_evaluations=0,
                  timestamp=datetime.now(timezone.utc).isoformat(), provenance=proof,
                  environment=env, independence_audit=overlap,
                  formal_individual_seed_count=len(index),
                  formal_seed_count_per_role={k: len(v) for k, v in role_values.items()},
                  frozen_conditions=list(CONDITIONS),
                  blocked_reason='frozen seed allocation conflicts with T20B zero-overlap requirement' if overlap['overlaps'] else None,
                  historical_sha256=before, historical_content_mismatches=0,
                  source_sha256={p: frozen.sha(ROOT / p) for p in
                      (AMENDMENT, 'scripts/audit_formal_preflight.py', 'tests/test_formal_preflight.py')},
                  all_checks_passed=overlap['all_individual_seeds_disjoint'])
    if RESULT.exists():
        raise ValueError('Existing preflight evidence must not be overwritten')
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print('Historical contents exact; original protocol/factors/folds exact; amendment recorded')
    print('Historical manifests:', len(overlap['manifests_checked']),
          '| overlapping individual values:', overlap['unique_overlapping_seed_values'],
          '| same-role values:', overlap['unique_same_role_overlapping_seed_values'])
    require_seed_independence(overlap)
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        audit()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
