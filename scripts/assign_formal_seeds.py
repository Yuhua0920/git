"""T20B seed-only governance. Hash integers; never generate formal tensors."""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from scripts import audit_formal_preflight as original
from scripts import audit_final_method_freeze as frozen
from scripts.formal_governance_snapshot import historical_hashes, COMMIT as GOVERNANCE_COMMIT, sealed_snapshot, blob, EVOLVING

MASTER = 'T20B_FORMAL_SEED_V2'
AMENDMENT = 'docs/t20b_formal_seed_governance_amendment.md'
READINESS_SHA = '58cf2cc98b8562bd823e583fe351e289221f1dabdeccda704c10a432a56da507'
MANIFEST = 'outputs/t20b_formal_seed_manifest.csv'
AUDIT = 'outputs/t20b_seed_assignment_audit.json'
FORBIDDEN = 'outputs/t20b_historical_forbidden_seeds.json'
BASELINE = 'outputs/t20b_seed_governance_history.json'
ARCHIVE = 'outputs/t20b_rejected_pre_unseal_seed_plan'
REJECTED = ARCHIVE + '/rejected_pre_unseal_seed_plan.csv'
EXECUTION = 'outputs/t20b_seed_execution_preflight.json'
FAILURE = 'outputs/t20b_seed_preflight_failure01'
FIELDS = ('environment', 'namespace', 'trajectory', 'trajectory_index', 'frame',
          'random_role', 'stream_index', 'seed_scope', 'canonical_key', 'retry_counter', 'assigned_seed')
ITEM_COUNT = 165000


def canonical_key(environment, trajectory_index, frame, role, stream_index):
    streams = {'sionna_seed': -1, 'torch_seed': -1, 'numpy_seed': -1,
               'bits_seed_tx0': 0, 'bits_seed_tx1': 1, 'noise_seed': -1}
    if (environment not in original.CONDITIONS or type(trajectory_index) is not int
            or trajectory_index not in range(1000) or type(frame) is not int
            or frame not in range(10) or role not in streams or stream_index != streams[role]
            or (role in frozen.ROLES[:3] and frame != 0)):
        raise ValueError('Only original consumed formal random-item scopes permitted')
    return json.dumps([MASTER, environment, trajectory_index, frame, role, stream_index],
                      ensure_ascii=True, separators=(',', ':'))


def candidate(key, retry_counter):
    if type(retry_counter) is not int or not 0 <= retry_counter < 2**64:
        raise ValueError('retry_counter must be an unsigned64 integer')
    digest = hashlib.sha256(key.encode('utf8') + b'\x00' + retry_counter.to_bytes(8, 'big')).digest()
    return int.from_bytes(digest[:4], 'big')


def random_items():
    for environment in original.CONDITIONS:
        for index in range(1000):
            for role in frozen.ROLES[:3]:
                yield environment, index, 0, role, -1
            for frame in range(10):
                for role, stream in zip(frozen.ROLES[3:], (0, 1, -1)):
                    yield environment, index, frame, role, stream


def namespace(environment):
    return f'FORMAL_CDL_{environment}_T20B_SEED_V2'


def allocation(forbidden, items=None, derive=candidate):
    used = set()
    rows = []
    for environment, index, frame, role, stream in random_items() if items is None else items:
        key = canonical_key(environment, index, frame, role, stream)
        retry = 0
        value = derive(key, retry)
        while value in forbidden or value in used:
            retry += 1
            value = derive(key, retry)
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError('Candidate outside frozen uint32 mapping')
        used.add(value)
        rows.append(dict(environment=environment, namespace=namespace(environment),
            trajectory=600 + index, trajectory_index=index, frame=frame,
            random_role=role, stream_index=stream,
            seed_scope='trajectory' if role in frozen.ROLES[:3] else 'frame',
            canonical_key=key, retry_counter=retry, assigned_seed=value))
    return rows


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf8')


def seed_field(key):
    key = key.lower()
    return (key in ('seed', 'seeds') or key.endswith(('_seed', '_seeds', '_seed_base'))
            or re.search(r'_seed_tx\d+$', key) is not None)


def numeric_values(value):
    if type(value) is int and value >= 0:
        yield value
    elif isinstance(value, str):
        if re.fullmatch(r'\d+', value):
            yield int(value)
        elif value.startswith(('[', '{')):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return
            yield from numeric_values(parsed)
    elif isinstance(value, (list, tuple)):
        for part in value:
            yield from numeric_values(part)
    elif isinstance(value, dict):
        for key, part in value.items():
            if not any(word in key.lower() for word in ('count', 'counter', 'tuples')):
                yield from numeric_values(part)


def record_values(value, seeds, namespaces):
    if isinstance(value, dict):
        for key, part in value.items():
            # Role-named values inside explicit seed-count reports are counts.
            if any(word in key.lower() for word in ('seed_count', 'seed_tuples')):
                continue
            if seed_field(key):
                seeds.update(numeric_values(part))
            if key.lower() in ('namespace', 'namespaces') or key.lower().endswith(('_namespace', '_namespaces')):
                def strings(item):
                    if isinstance(item, str):
                        namespaces.add(item)
                    elif isinstance(item, list):
                        for v in item:
                            strings(v)
                    elif isinstance(item, dict):
                        for v in item.values():
                            strings(v)
                strings(part)
            record_values(part, seeds, namespaces)
    elif isinstance(value, list):
        for part in value:
            record_values(part, seeds, namespaces)


def source_seed_literals(text):
    result = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and type(node.value.value) is int:
            for target in node.targets:
                name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else ''
                if seed_field(name):
                    result.update(numeric_values(node.value.value))
        elif isinstance(node, ast.Call):
            name = getattr(node.func, 'attr', getattr(node.func, 'id', ''))
            if name in ('default_rng', 'manual_seed', 'seed', 'SeedSequence') and node.args:
                try:
                    literal = ast.literal_eval(node.args[0])
                except (ValueError, TypeError):
                    continue
                result.update(numeric_values(literal))
    return result


def historical_records(paths):
    seeds = set()
    namespaces = set()
    sources = []
    for path in sorted(paths):
        full = ROOT / path
        content = blob(path) if path in EVOLVING else full.read_bytes()
        local = set()
        labels = set()
        if full.suffix.lower() == '.csv':
            with full.open(encoding='utf8', newline='') as stream:
                reader = csv.DictReader(stream)
                keys = [k for k in reader.fieldnames or () if seed_field(k) or 'namespace' in k.lower()]
                if keys:
                    for row in reader:
                        record_values({k: row[k] for k in keys}, local, labels)
        elif full.suffix.lower() in ('.json', '.yaml', '.yml'):
            text = content.decode('utf8')
            value = json.loads(text) if full.suffix.lower() == '.json' else yaml.safe_load(text)
            record_values(value, local, labels)
        elif full.suffix.lower() == '.py':
            local = source_seed_literals(content.decode('utf8'))
        else:
            continue
        sources.append(dict(path=path, sha256=hashlib.sha256(content).hexdigest(),
                            unique_seed_values=len(local), namespaces=sorted(labels)))
        seeds.update(local)
        namespaces.update(labels)
    return seeds, namespaces, sources


def archive_rejected_plan():
    dest = ROOT / ARCHIVE
    if dest.exists():
        raise ValueError('Rejected-plan archive already exists; no overwrite permitted')
    dest.mkdir()
    records = []
    for environment, index, frame, role, stream in random_items():
        e = original.CONDITIONS.index(environment)
        value = frozen.seeds(e, 600 + index, frame)[frozen.ROLES.index(role)]
        records.append(dict(environment=environment, namespace=f'FORMAL_CDL_{environment}_T20A_v1',
            trajectory=600 + index, trajectory_index=index, frame=frame, random_role=role,
            stream_index=stream, seed_scope='trajectory' if role in frozen.ROLES[:3] else 'frame',
            canonical_key=f'T20A_v1|{e}|{index}|{frame}|{role}', retry_counter=0, assigned_seed=value))
    (ROOT / REJECTED).write_bytes(csv_bytes(records))
    copied = {}
    for path in (original.PROTOCOL, original.AMENDMENT, 'outputs/t20b_preflight.json',
                 'scripts/audit_formal_preflight.py', 'tests/test_formal_preflight.py'):
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / path).read_bytes())
        copied[path] = frozen.sha(target)
        if copied[path] != frozen.sha(ROOT / path):
            raise ValueError('Rejected evidence archival content mismatch')
    metadata = dict(status='rejected_pre_unseal_seed_plan', original_plan_sha256=frozen.sha(ROOT / REJECTED),
                    consumed_items=len(records), data_generated=False, unsealed=False,
                    copied_original_sha256=copied, original_collision_evidence_sha256=frozen.sha(ROOT / 'outputs/t20b_preflight.json'))
    (dest / 'provenance.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf8')
    return metadata


def inspect_manifest(rows, forbidden, historical_namespaces):
    values = [r['assigned_seed'] for r in rows]
    keys = [r['canonical_key'] for r in rows]
    labels = {r['environment']: r['namespace'] for r in rows}
    if len(rows) != ITEM_COUNT or len(set(keys)) != ITEM_COUNT:
        raise ValueError('Exactly165000 unique consumed random items required')
    if set(labels) != set(original.CONDITIONS) or len(set(labels.values())) != 5:
        raise ValueError('Five distinct allocation namespaces required')
    overlap = set(values) & forbidden
    label_overlap = set(labels.values()) & historical_namespaces
    duplicate = len(values) - len(set(values))
    if overlap or label_overlap or duplicate:
        raise ValueError('Seed/namespace independence gate failed')
    return dict(historical_seed_intersection=0, formal_duplicate_seed_count=0,
                historical_namespace_intersection=0, formal_namespace_duplicate=0,
                formal_assigned_seed_count=len(values), namespaces=labels,
                number_of_items_requiring_retry=sum(r['retry_counter'] > 0 for r in rows),
                total_rejected_candidates=sum(r['retry_counter'] for r in rows),
                max_retry_counter=max(r['retry_counter'] for r in rows),
                globally_unique_consumed_items=True, formal_signal_rng_calls=0)


def immutable_contract():
    proof = original.provenance()
    if frozen.sha(ROOT / original.AMENDMENT) != READINESS_SHA:
        raise ValueError('Readiness amendment bytes changed')
    accepted = frozen.load(ROOT / 'outputs/t20a_results.json')
    baseline = frozen.load(ROOT / BASELINE)
    historical_hashes(baseline['historical_sha256'])
    truth = []
    for a, b, c in itertools.product((False, True), repeat=3):
        observed = original.readiness(a, b, c)
        expected = dict(paper_core_empirically_supported=a, paper_results_ready=a and b and c,
                        paper_results_ready_partial=a and (b != c))
        if observed != expected:
            raise ValueError('Readiness truth table drift')
        truth.append(dict(A=a, B=b, C=c, **observed))
    # JSON is the frozen representation: normalize integer mapping keys without
    # changing any values, tolerances or historical file/array hashes.
    return json.loads(json.dumps(dict(original_protocol_sha256=original.PROTOCOL_SHA,
                readiness_amendment_sha256=READINESS_SHA,
                seed_governance_amendment_sha256=frozen.sha(ROOT / AMENDMENT),
                frozen_method=proof['frozen_method'], method_definition_sha256=proof['method_definition_sha256'],
                formal_conditions=list(original.CONDITIONS), original_formal_contract=accepted['formal_protocol'],
                numerical_conventions=frozen.constants_audit(), readiness_truth_table=truth,
                execution_level_numerical_changes=['channel_bits_unit_noise_seed_assignment'],
                namespace_metadata_change='seed-plan version suffix only; no environment/population change',
                historical_files_verified=baseline['history_files'], historical_content_mismatches=0,
                T20A_commit=proof['T20A_commit'], HEAD=proof['HEAD'])))


def protect(path):
    if (ROOT / path).exists():
        raise ValueError('Existing seed-governance artifact protected: ' + path)


def prepare_manifest():
    for path in (MANIFEST, AUDIT, FORBIDDEN):
        protect(path)
    if (ROOT / 'outputs/t20b_formal_unseal.json').exists():
        raise ValueError('This correction must precede formal unseal')
    contract = immutable_contract()
    rejected = archive_rejected_plan()
    baseline = frozen.load(ROOT / BASELINE)
    inputs = list(baseline['historical_sha256'])
    inputs += [p.relative_to(ROOT).as_posix() for p in (ROOT / ARCHIVE).rglob('*') if p.is_file()]
    seeds, labels, sources = historical_records(inputs)
    seeds.add(20262001)  # Fixed bootstrap reservation is forbidden to data RNGs.
    evidence = dict(scope='all baseline historical records plus complete rejected plan',
                    forbidden_seeds=sorted(seeds), historical_namespaces=sorted(labels), sources=sources,
                    bootstrap_definition_unchanged=True, fixed_bootstrap_seed=20262001)
    (ROOT / FORBIDDEN).write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf8')
    rows = allocation(seeds)
    checks = inspect_manifest(rows, seeds, labels)
    content = csv_bytes(rows)
    # Independent complete reconstruction before any manifest is frozen.
    if content != csv_bytes(allocation(seeds)):
        raise ValueError('Deterministic whole-manifest reproduction failed')
    (ROOT / MANIFEST).write_bytes(content)
    result = dict(task='T20B seed-governance', phase='SEED_GATES_PASSED_REGRESSION_PENDING',
        timestamp=datetime.now(timezone.utc).isoformat(), **contract,
        master_label=MASTER, mapping='SHA256 UTF8 compact JSON key + zero separator + retry uint64 big endian; first4 digest bytes uint32',
        seed_range=[0, 2**32 - 1], formal_seed_manifest_sha256=frozen.sha(ROOT / MANIFEST),
        historical_forbidden_seed_count=len(seeds), forbidden_set_artifact=FORBIDDEN,
        forbidden_set_sha256=frozen.sha(ROOT / FORBIDDEN), rejected_original_seed_plan=rejected,
        rejected_original_seed_plan_sha256=rejected['original_plan_sha256'],
        historical_input_inventory_sha256=frozen.sha(ROOT / BASELINE),
        **checks, seed_only_change_audit_passed=True, method_hashes_exact=True,
        protocol_hashes_frozen=True, seed_manifest_frozen=True, full_pytest_passed=False,
        formal_preflight_ready=False, formal_test_used=False, formal_test_sealed=True,
        formal_test_unsealed=False, endpoint_A='UNEVALUATED', endpoint_B='UNEVALUATED', endpoint_C='UNEVALUATED',
        H_generated=False, bits_generated=False, noise_generated=False, received_grid_generated=False,
        nmse_evaluated=False, ber_evaluated=False, bootstrap_evaluated=False,
        source_sha256={p: frozen.sha(ROOT / p) for p in
            (AMENDMENT, 'scripts/assign_formal_seeds.py', 'tests/test_formal_seed_governance.py')},
        all_checks_passed=False)
    frozen.verify_hashes(baseline['historical_sha256'])
    (ROOT / AUDIT).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print('Seed gates PASS: forbidden', len(seeds), 'assigned', len(rows), 'retry items', checks['number_of_items_requiring_retry'],
          'max retry', checks['max_retry_counter'], '; formal data remain sealed', flush=True)


def validate_artifacts():
    result = frozen.load(ROOT / AUDIT)
    if (ROOT / EXECUTION).exists():
        execution = frozen.load(ROOT / EXECUTION)
        frozen.verify_hashes({FAILURE + '/' + p: h for p, h in result['source_sha256'].items()})
        historical_hashes(execution['execution_source_sha256'])
        if frozen.sha(ROOT / execution['original_audit_snapshot']) != execution['original_audit_sha256']:
            raise ValueError('Original preparation evidence changed')
    else:
        frozen.verify_hashes(result['source_sha256'])
    if frozen.sha(ROOT / MANIFEST) != result['formal_seed_manifest_sha256'] or frozen.sha(ROOT / FORBIDDEN) != result['forbidden_set_sha256']:
        raise ValueError('Frozen seed artifact bytes changed')
    if frozen.sha(ROOT / REJECTED) != result['rejected_original_seed_plan_sha256']:
        raise ValueError('Rejected original plan changed')
    old = immutable_contract()
    # HEAD in the preparation result describes its creation, not future HEAD.
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', result['HEAD'], GOVERNANCE_COMMIT], cwd=ROOT)
    old['HEAD'] = result['HEAD']
    if any(old[k] != result[k] for k in old):
        raise ValueError('Seed-only contract or historical provenance changed')
    evidence = frozen.load(ROOT / FORBIDDEN)
    historical, labels, sources = historical_records([v['path'] for v in evidence['sources']])
    historical.add(20262001)
    if evidence['forbidden_seeds'] != sorted(historical) or evidence['historical_namespaces'] != sorted(labels) or evidence['sources'] != sources:
        raise ValueError('Forbidden-set inputs or canonical set changed')
    rows = allocation(historical)
    if csv_bytes(rows) != (ROOT / MANIFEST).read_bytes():
        raise ValueError('Complete canonical manifest reconstruction mismatch')
    checks = inspect_manifest(rows, historical, labels)
    if any(checks[k] != result[k] for k in checks):
        raise ValueError('Seed/namespace/retry evidence changed')
    sealed_snapshot()
    return result


def execution_preflight():
    protect(EXECUTION)
    result = frozen.load(ROOT / AUDIT)
    snapshot = FAILURE + '/' + AUDIT
    if frozen.sha(ROOT / snapshot) != frozen.sha(ROOT / AUDIT):
        raise ValueError('Original preflight JSON must remain exact before finalization')
    frozen.verify_hashes({FAILURE + '/' + p: h for p, h in result['source_sha256'].items()})
    if frozen.sha(ROOT / MANIFEST) != result['formal_seed_manifest_sha256']:
        raise ValueError('Seed manifest changed during representation correction')
    contract = immutable_contract()
    if any(contract[k] != result[k] for k in contract):
        raise ValueError('JSON-normalized immutable contract mismatch')
    evidence = dict(original_audit_snapshot=snapshot, original_audit_sha256=frozen.sha(ROOT / snapshot),
        original_source_sha256=result['source_sha256'],
        execution_source_sha256={p: frozen.sha(ROOT / p) for p in result['source_sha256']},
        original_seed_manifest_sha256=result['formal_seed_manifest_sha256'],
        seed_manifest_unchanged=True, protocols_unchanged=True, numerical_values_unchanged=True,
        correction='JSON mapping-key representation normalization only; original evidence archived exact',
        formal_test_used=False, formal_test_sealed=True)
    (ROOT / EXECUTION).write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf8')
    print('Original preparation exact; representation correction execution snapshot frozen; manifest unchanged')


def finalize():
    result = validate_artifacts()
    if result['phase'] != 'SEED_GATES_PASSED_REGRESSION_PENDING':
        raise ValueError('Finalized evidence must not be overwritten')
    full = ROOT / 'outputs/t20b_seed_full_pytest.txt'
    related = ROOT / 'outputs/t20b_seed_related_pytest_attempt02.txt'
    summary = frozen.passed_summary(full, 1918)
    related_summary = frozen.passed_summary(related, 54)
    subprocess.check_call(['git', 'diff', '--check'], cwd=ROOT)
    for path in result['source_sha256']:
        if any(line.rstrip() != line for line in (ROOT / path).read_text(encoding='utf8').splitlines()):
            raise ValueError('New-source whitespace error')
    protect('outputs/t20b_seed_assignment_audit_before_full_pytest.json')
    (ROOT / 'outputs/t20b_seed_assignment_audit_before_full_pytest.json').write_bytes((ROOT / AUDIT).read_bytes())
    result.update(phase='PRE_UNSEAL_SEED_GOVERNANCE_COMPLETE', full_pytest_passed=True,
                  full_pytest_summary=summary, related_pytest_summary=related_summary,
                  full_pytest_sha256=frozen.sha(full), related_pytest_sha256=frozen.sha(related),
                  git_diff_check='PASS', formal_preflight_ready=True, all_checks_passed=True)
    if (ROOT / EXECUTION).exists():
        result['execution_source_sha256'] = frozen.load(ROOT / EXECUTION)['execution_source_sha256']
        result['execution_preflight_sha256'] = frozen.sha(ROOT / EXECUTION)
    (ROOT / AUDIT).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print('formal_preflight_ready=true; hashes exact; no unseal or formal scientific data', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--execution-preflight', action='store_true')
    args = parser.parse_args()
    if sum((args.prepare, args.finalize, args.execution_preflight)) > 1:
        parser.error('Choose one stage')
    if args.prepare:
        prepare_manifest()
    elif args.finalize:
        finalize()
    elif args.execution_preflight:
        execution_preflight()
    else:
        validate_artifacts()
        print('Seed manifest, forbidden set, protocol, method and history exact; no formal data')


if __name__ == '__main__':
    main()
