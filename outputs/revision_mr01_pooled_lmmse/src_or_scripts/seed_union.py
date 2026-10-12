"""Read-only MR-01 historical seed exclusions, with authenticated fixed inputs.

Default API: build_forbidden_seed_union(root) -> set[int].
With with_report=True: return a dict containing the set and its provenance.
No simulator imports, outcome reads, file writes, or RNG calls occur here.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROLES = (
    "sionna_seed", "torch_seed", "numpy_seed",
    "bits_seed_tx0", "bits_seed_tx1", "noise_seed",
)
UINT32_MAX = 2**32 - 1
UINT64_MAX = 2**64 - 1

# Hashes were computed from actual bytes and checked against frozen provenance.
INPUTS = {
    "outputs/sp_u12b2a/historical_seed_inventory.json":
        (33053124, "debd073333340b97a44554ed677ab855914ea38402758b65b810e0a2f37bd58e"),
    "outputs/sp_u12b2a_seed_registry.json":
        (44064583, "84d82c05a1757c2e3fe64afa2146c182739a92b8037eb0bd1d62ed2e1d9726a5"),
    "outputs/sp_u12b2a/development_seed_registry.json":
        (138448, "dcf2fb1857788f16675d1b6609aeb60bb15e598e4508a0747ca0a9ff4138f760"),
    "configs/sp_u12b2b_sealed.json":
        (16075, "bea89af6263722264695aad0f894f67373a81c3fe2838f2da039d016d94c19b8"),
    "outputs/sp_u12b3b/bootstrap_seed_registry.json":
        (472, "57f655de0c99efcbe54ca41fdbdc1455d859c08ba2dbc38b8e9c86f74016d862"),
    "outputs/sp_u12b5b_candidate_seed_registry.json":
        (1363, "a764f98b74b739742777889da013a6817a444037ba920db7b6dfa5895bcfba91"),
}


def _integer(value, context, maximum=UINT64_MAX):
    # Historical registry contains genuine uint64 values: never truncate them.
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("Invalid integer seed in " + context)
    return value


def _authenticated(root, name, sources):
    path = root / name
    data = path.read_bytes()
    expected_size, expected_sha = INPUTS[name]
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != expected_size or digest != expected_sha:
        raise ValueError("Frozen seed metadata byte drift: " + str(path))
    sources.append({"path": name, "absolute_path": str(path),
                    "bytes": len(data), "sha256": digest})
    return json.loads(data.decode("utf-8-sig"))


def _registry(rows, context, allocations):
    actual = set()
    keys = set()
    counts = Counter()
    for row in rows:
        value = _integer(row["seed"], context, UINT32_MAX)
        role = row["role"]
        key = (row["condition"], row["group"], row["trajectory_id"],
               row["frame_id"], role)
        if key in keys or value in actual:
            raise ValueError("Duplicate seed or registry identity: " + context)
        keys.add(key)
        actual.add(value)
        counts[role] += 1
        allocations.setdefault(role, set()).add(value)
        if role in ROLES:
            expected_frames = {-1} if role in ROLES[:3] else set(range(10))
            if row["frame_id"] not in expected_frames:
                raise ValueError("Channel/frame seed role drift: " + context)
    return actual, dict(counts)


def build_forbidden_seed_union(root, *, with_report=False):
    """Return a set forbidding every historical numeric role seed/SDK neighbor.

    Every new role must reject membership in this one any-role union. This is
    stronger than same-role or tuple-only comparisons. It includes historical
    reserved/invalidated seeds, B2 reference/eval/development seeds, B2/B3
    bootstrap seeds, and all selected/unselected B5 initialization/final seeds.
    B3 and B5C reuse B2 channel/communication inputs; no new input seeds exist.

    The frozen historical inventory already contains its adjacent SDK states.
    Later allocated values are expanded by ±1 here, covering Sionna CUDA0's
    explicit seed+1 state and conservative cross-role state collision checks.
    """
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("Pass the explicit absolute repository root")
    sources = []
    allocations = {role: set() for role in ROLES}
    hist = _authenticated(root, "outputs/sp_u12b2a/historical_seed_inventory.json", sources)
    base = {_integer(value, "historical inventory") for value in hist["forbidden_seeds"]}
    if len(base) != 1780343:
        raise ValueError("Historical seed inventory cardinality drift")
    sources[-1].update(seed_values=len(base), scope="Historical any-role union + SDK neighbors")

    reg = _authenticated(root, "outputs/sp_u12b2a_seed_registry.json", sources)
    formal, formal_counts = _registry(reg["rows"], "B2 formal registry", allocations)
    if len(formal) != 172501:
        raise ValueError("B2 formal registry cardinality drift")
    sources[-1].update(seed_values=len(formal), role_counts=formal_counts,
                       scope="B2 MATCHED_REF + EVAL + primary bootstrap")

    reg = _authenticated(root, "outputs/sp_u12b2a/development_seed_registry.json", sources)
    development, development_counts = _registry(reg["rows"], "B2A development registry", allocations)
    if len(development) != 540 or formal & development or (formal | development) & base:
        raise ValueError("Frozen B2/B2A seed independence drift")
    sources[-1].update(seed_values=len(development), role_counts=development_counts,
                       scope="B2A development smoke")

    cfg = _authenticated(root, "configs/sp_u12b2b_sealed.json", sources)
    b2_primary = _integer(cfg["primary_bootstrap"]["master_seed"], "B2 primary bootstrap")
    b2_descriptive = _integer(cfg["descriptive_bootstrap"]["seed"], "B2 descriptive bootstrap")
    if b2_primary not in formal or cfg["B2A_registry_sha256"] != INPUTS["outputs/sp_u12b2a_seed_registry.json"][1]:
        raise ValueError("B2 bootstrap/registry identity drift")
    later = {b2_descriptive}
    allocations.setdefault("descriptive_bootstrap_seed", set()).add(b2_descriptive)
    sources[-1].update(seed_values=2, new_seed_values=1, scope="B2 primary + descriptive bootstrap")

    cfg = _authenticated(root, "outputs/sp_u12b3b/bootstrap_seed_registry.json", sources)
    b3 = {_integer(cfg[key], "B3 " + key)
          for key in ("primary_master_seed", "descriptive_master_seed")}
    if len(b3) != 2 or cfg["new_channel_seeds"] != 0:
        raise ValueError("B3 declared seed/reuse drift")
    later.update(b3)
    allocations.setdefault("b3_bootstrap_seed", set()).update(b3)
    sources[-1].update(seed_values=len(b3), scope="B3 bootstrap; channel/comm reuse B2")

    cfg = _authenticated(root, "outputs/sp_u12b5b_candidate_seed_registry.json", sources)
    init = set()
    final = set()
    for row in cfg["candidates"]:
        init.add(_integer(row["seed"], "B5 candidate seed"))
        final.add(_integer(row["final_seed"], "B5 final seed"))
    if len(init) != 9 or len(final) != 9 or init & final:
        raise ValueError("B5 candidate/final seed cardinality drift")
    later.update(init | final)
    allocations.setdefault("candidate_initialization_seed", set()).update(init)
    allocations.setdefault("full_retraining_initialization_seed", set()).update(final)
    sources[-1].update(seed_values=18, scope="All nine B5 candidates + all nine reserved final fits")

    actual_union = base | formal | development | later
    if len(later) != 21 or len(actual_union) != 1953405:
        raise ValueError("Established historical seed union cardinality drift")
    # Only expand later allocations; the baseline already includes its offsets.
    added = formal | development | later
    forbidden = set(actual_union)
    for value in added:
        if value > 0:
            forbidden.add(value - 1)
        if value < UINT32_MAX:
            forbidden.add(value + 1)
    if not with_report:
        return forbidden
    return {
        "forbidden_seeds": forbidden,
        "by_role": {role: forbidden for role in ROLES},
        "role_allocations": allocations,
        "sources": sources,
        "formal_role_counts": formal_counts,
        "development_role_counts": development_counts,
        "historical_baseline_count": len(base),
        "historical_uint64_values_above_uint32": sum(value > UINT32_MAX for value in base),
        "individual_role_union_count": len(actual_union),
        "forbidden_seed_count": len(forbidden),
        "extra_derived_state_count": len(forbidden - actual_union),
        "comparison": "Every new role versus historical any-role union; no tuple-only claim",
        "SDK_state_policy": "Historical baseline includes ±1; later values expanded ±1 (CUDA0 seed+1 protected)",
        "B3_B5C_channel_communication_inputs": "Exact B2 EVAL reuse; no additional input seeds",
        "scope_limit": "Named scientific corpora and declared fit/bootstrap roles; later synthetic/engineering literals outside these registries are not claimed as exhaustively audited",
        "formal_performance_read": False,
    }


def serializable_report(report):
    """Remove Python sets while retaining all exact input provenance."""
    return {key: value for key, value in report.items()
            if key not in {"forbidden_seeds", "by_role", "role_allocations"}} | {
                "excluded_roles": list(ROLES),
                "all_roles_use_same_any_role_union": True,
                "explicit_role_allocation_counts": {
                    key: len(value) for key, value in report["role_allocations"].items()
                },
            }
