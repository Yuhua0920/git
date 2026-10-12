"""MR08 metadata-only historical seed/H exclusions; no allocation or simulation.

Reuses the authenticated MR01 any-role exclusion algorithm. The additional
registries are projected only to seed identities. CSVs are projected only to
whole-trajectory H identities, never outcome columns or frame H identities.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys

sys.dont_write_bytecode = True
U32 = 2**32 - 1
MODULE = "outputs/revision_mr01_pooled_lmmse/src_or_scripts/seed_union.py"
MODULE_HASH = "38eaac6621c73d314a9fe814b21b513b0eae455f70a2ada325abd748d44d7121"
ALLOCATOR = "scripts/run_sp_u12b2a.py"
ALLOCATOR_HASH = "1d9c62ece89727d95e70d1a553afd10f56baa91ee20865cc833f066a21d79640"
SMOKE_NAMESPACE = "SP_U17_MR08_CONFIRM_SMOKE_V1"
FORMAL_NAMESPACE = "SP_U17_MR08_CONFIRM_TNEW_V1"
EXTRA = {
 "outputs/revision_mr01_pooled_lmmse/data/seed_registry.json": "80d7af2fe65ade2ae713161132e4c2c05c30aae46e430fe5e9d4b974d66de722",
 "outputs/revision_mr02_coordinate_mechanism/data/reference_seed_registry.json": "5e0323ef7e7a47ae81e0369f6844a57a8fa75de6f184a18da566fa20591a8d7a",
 "outputs/revision_mr03_ppca_capacity_controls/data/validation_seed_registry.json": "067c190e7faf76990d177308a81f6b7ae44bce4790ae87f48f017a9c74969a9f",
 "outputs/revision_mr03_ppca_capacity_controls/data/smoke_seed_registry.json": "ea155cc1f51f0c22c7bf9faa12cc1ff5666df3a659d6d0b81b7f9aed347e6581",
}
CSV_H = {
 "outputs/t15a_fix_replay_h_hashes.csv": ("8a05aeca4fbc7ec911f6c1cb3bbc9cdb9b0e35e30ebc97fc78ffceb6eb648267", "replayed_h_sha256"),
 "outputs/t15c_cdl_channel_hashes.csv": ("3afd9c4e4da3745c2c4b131eed3b446ccccbb90611589f384d2e6d87d66cb512", "H_sha256"),
 "outputs/t16c_heldout_channel_hashes.csv": ("9d321e45f8d3f1c62e8f26dfeb7ee64fd931dcc44134b6ebab07dc2afb0cee26", "H_sha256"),
 "outputs/t18a_channel_hashes.csv": ("7855f6cc2457750491687ad9e7176867506d273c638193c75c14f95b1bf4ce14", "H_sha256"),
 "outputs/t18b_channel_hashes.csv": ("515466adf95a94f3f81f4338c6596b10bda447639525f8b0505c50fc1b6f9026", "H_sha256"),
 "outputs/t19a_channel_hashes.csv": ("9fc605f74d7aeb55c460d7daa6dcfc5c2d11ebb4d19c0d8eeb37ccdfdd89992d", "H_sha256"),
 "outputs/t19c_channel_hashes.csv": ("7a97b04b2ea3f923a8ce2f5a68ed477bc4ade868c76578a7e253af7191fea061", "H_sha256"),
 "outputs/t20b_formal_manifest.csv": ("5ab28408bfbb4d4798360784f4aa3574a3263c1037dc58a93538647a89e04a20", "H_trajectory_sha256"),
}
OTHER = {
 "outputs/revision_mr03_ppca_capacity_controls/data/historical_H_hashes.json": "53ae7fa2f83662949d8ddea3c062cae44c50d8731ef1d0f93565f6d06bdac4db",
 "outputs/revision_mr03_ppca_capacity_controls/reports/V_new_receipt.json": "ef2b730e8d94ac2964ee5a12cdf4a7a4c2a11ce840e06d1b224c26e00888bc71",
 "outputs/revision_mr03_ppca_capacity_controls/reports/smoke_receipt.json": "c61b088d1163ef2ce4c52b7a3cf5aa4034fd7fda4d56600c4c1046e76348ab03",
 "outputs/sp_u7b_r1/seed_manifest.csv": "b7f1ca11d5dfeb2100803ca16ce6b97b5787fcb64fc4471d383498d7b512b3ee",
 "outputs/sp_u7a_future_seed_manifest.csv": "dea6ddb202ef03f5a5d7914dfdf1e78d89337025026e130090f8e53338769b22",
 "outputs/t15a_cdl_manifest.csv": "fc1d3d585daaf79fa7fe7c9805262b0065bf28568db30c42f80fd7bc1dcc4979",
 "outputs/t15c_cdl_manifest.csv": "10b6d0bd16b87f3b0d1b97224c971ee3216fa6da584a259cc439acb81eddf5b8",
 "outputs/sp_u7b_r1/delivery_manifest.json": "31713d2dede5d2882d6a4a609ff786cae5680b40d69c2015b04c2febd85cfb15",
}

def sha(path):
 with Path(path).open("rb") as f:
  return hashlib.file_digest(f, "sha256").hexdigest()

def authenticated(root, name, expected, report):
 p = root/name
 actual = sha(p)
 if actual != expected:
  raise ValueError("HISTORICAL_IDENTITY_BYTE_DRIFT:" + name)
 report.append(dict(path=name, sha256=actual, bytes=p.stat().st_size))
 return p

def read(path):
 return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def h_hash(value):
 if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
  raise ValueError("WHOLE_TRAJECTORY_H_SHA256_REQUIRED")
 return value

def collect(root):
 """Return (forbidden_any_role_seeds, historical_whole_H_hashes, report).

 This function does not allocate new seeds, generate H/noise/bootstrap arrays,
 load any performance NPZ/CSV, or write historical files.
 """
 root = Path(root)
 if not root.is_absolute():
  raise ValueError("ABSOLUTE_PROJECT_ROOT_REQUIRED")
 sources = []
 modulepath = authenticated(root, MODULE, MODULE_HASH, sources)
 spec = importlib.util.spec_from_file_location("_mr08_old_seed_union", modulepath)
 module = importlib.util.module_from_spec(spec)
 spec.loader.exec_module(module)
 base = module.build_forbidden_seed_union(root, with_report=True)
 forbidden = base["forbidden_seeds"]
 sources.extend(base["sources"])
 additional = set()
 registries = []
 for name, digest in EXTRA.items():
  obj = read(authenticated(root, name, digest, sources))
  splits = obj if "rows" not in obj else {obj["namespace"]: obj}
  for split, registry in splits.items():
   rows = registry["rows"]
   values = set()
   keys = set()
   for row in rows:
    value = row["seed"]
    if type(value) is not int or not 0 <= value <= U32:
     raise ValueError("INVALID_MR_REGISTRY_SEED")
    key = (row["condition"], row["group"], row["trajectory_id"], row["frame_id"], row["role"])
    if value in values or key in keys:
     raise ValueError("DUPLICATE_MR_SEED_IDENTITY")
    values.add(value); keys.add(key)
   additional |= values
   registries.append(dict(path=name, split=split, rows=len(rows), individual_seeds=len(values), physical_trajectories=sum(row["role"] == "sionna_seed" for row in rows)))
 # Expand later MR allocations exactly as in the already trusted MR03 method.
 forbidden |= additional | {v+d for v in additional for d in (-1,1) if 0 <= v+d <= U32}
 for name, digest in OTHER.items():
  authenticated(root, name, digest, sources)
 inventory = read(root/"outputs/sp_u12b2a/historical_seed_inventory.json")
 old_sources = {row["path"]: row["sha256"] for row in inventory["sources"]}
 for name in ("outputs/sp_u7b_r1/seed_manifest.csv", "outputs/sp_u7a_future_seed_manifest.csv", "outputs/t15a_cdl_manifest.csv", "outputs/t15c_cdl_manifest.csv"):
  if old_sources.get(name) != OTHER[name]:
   raise ValueError("CDLA_OR_PREPARATION_NOT_BOUND_BY_BASE_INVENTORY")
 hset = {h_hash(h) for h in read(root/"outputs/revision_mr03_ppca_capacity_controls/data/historical_H_hashes.json")["H_sha256"]}
 h_sources = [dict(path="outputs/revision_mr03_ppca_capacity_controls/data/historical_H_hashes.json", unique_whole_H_hashes=len(hset))]
 for name, (digest, column) in CSV_H.items():
  p = authenticated(root, name, digest, sources)
  selected = set(); rows = 0
  with p.open(encoding="utf-8-sig", newline="") as f:
   reader = csv.DictReader(f)
   if column not in reader.fieldnames:
    raise ValueError("H_IDENTITY_COLUMN_MISSING:" + name)
   for row in reader:
    selected.add(h_hash(row[column])); rows += 1
  hset |= selected
  h_sources.append(dict(path=name, projected_column=column, rows=rows, unique_whole_H_hashes=len(selected), outcome_numeric_columns_read=False))
 for dataset, count in (("V_new",500), ("smoke",5)):
  name = f"outputs/revision_mr03_ppca_capacity_controls/reports/{dataset}_receipt.json"
  receipt = read(root/name)
  if receipt["PASS"] is not True or len(receipt["receipts"]) != count:
   raise ValueError("OLD_MR03_RECEIPT_COVERAGE")
  selected = {h_hash(row["H_sha256"]) for row in receipt["receipts"]}
  if len(selected) != count:
   raise ValueError("OLD_MR03_H_DUPLICATE")
  hset |= selected
  h_sources.append(dict(path=name, unique_whole_H_hashes=count, projected_column="receipts.H_sha256"))
 delivery = read(root/"outputs/sp_u7b_r1/delivery_manifest.json")["file_hashes"]
 for group, count in (("evaluation",1000), ("matched_reference",500)):
  selected = set()
  for i in range(count):
   rel = f"outputs/sp_u7b_r1/{group}/proof_{i:04d}.json"
   if rel not in delivery or sha(root/rel) != delivery[rel]:
    raise ValueError("CDLA_PROOF_IDENTITY_BYTE_DRIFT:" + rel)
   selected.add(h_hash(read(root/rel)["H_array_sha256"]))
  if len(selected) != count:
   raise ValueError("OLD_CDLA_H_DUPLICATE")
  hset |= selected
  h_sources.append(dict(path=f"outputs/sp_u7b_r1/{group}/proof_*.json", authenticated_by="delivery_manifest.json.file_hashes", projected_column="H_array_sha256", unique_whole_H_hashes=count))
 report = dict(PASS=True, historical_seed_algorithm=MODULE, sources=sources,
  base_seed_scope=module.serializable_report(base), additional_MR_individual_seed_count=len(additional),
  total_any_role_forbidden_with_SDK_neighbors=len(forbidden), registry_counts=registries,
  historical_whole_H_hash_count=len(hset), H_identity_sources=h_sources,
  physical_domains_reused=dict(MR03C=["MR01 D_eval","MR03 V_new"], MR03B=["MR01 D_eval","MR03 V_new"], MR05=["MR01 D_eval"], B3_B5C="B2 EVAL"),
  no_new_random_system=True, new_channel_calls=0, scientific_performance_read=False,
  seed_identity_scope="named authenticated scientific registries, historical reserved/invalidated values and SDK adjacent states; not an exhaustive assertion about unrecorded external RNG states",
  distribution_scope="new independent trajectories in five already studied CDL distributions; not unseen-profile/OOD confirmation")
 return forbidden, hset, report

def _exclusive_json(path, value):
 path = Path(path)
 path.parent.mkdir(parents=True, exist_ok=True)
 with path.open("x", encoding="utf8") as f:
  json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)

def prepare_seeds(root, out, smoke_per_condition=2):
 """Root's preparation API: only immutable metadata, no H/noise/draws.

 The four outputs are data/smoke_seed_registry.json,
 data/formal_seed_registry.json, data/historical_H_hashes.json and
 reports/seed_independence.json. Both registry domains have an independently
 allocated bootstrap master. Subsequent condition streams are exactly
 SeedSequence([master, condition_ordinal]); all SNR/methods share each condition's
 trajectory draws. The caller alone freezes the final science protocol.
 """
 root, out = Path(root), Path(out)
 if not root.is_absolute() or out.resolve() != (root/"outputs/revision_mr08a_independent_confirmation").resolve():
  raise ValueError("EXACT_NEW_MR08_OUTPUT_REQUIRED")
 if type(smoke_per_condition) is not int or smoke_per_condition < 1:
  raise ValueError("POSITIVE_SMOKE_PER_CONDITION_REQUIRED")
 names = ["data/smoke_seed_registry.json", "data/formal_seed_registry.json",
          "data/historical_H_hashes.json", "reports/seed_independence.json"]
 if any((out/name).exists() for name in names):
  raise ValueError("SEED_PREPARATION_OUTPUT_ALREADY_EXISTS")
 forbidden, hh, report = collect(root)
 # Pre-data conservative synthetic/scalar exclusions, not outcome-dependent.
 engineering = set(range(1001)) | {170505}
 forbidden |= engineering | {v+d for v in engineering for d in (-1,1) if 0 <= v+d <= U32}
 allocator_path = authenticated(root, ALLOCATOR, ALLOCATOR_HASH, report["sources"])
 sys.path.insert(0, str(root))
 spec = importlib.util.spec_from_file_location("_mr08_trusted_seed_allocator", allocator_path)
 module = importlib.util.module_from_spec(spec)
 spec.loader.exec_module(module)
 used = set()
 smoke = module.allocate(SMOKE_NAMESPACE, smoke_per_condition, 0, forbidden, used, True)
 formal = module.allocate(FORMAL_NAMESPACE, 1000, 0, forbidden, used, True)
 smoke_values = {row["seed"] for row in smoke["rows"]}
 formal_values = {row["seed"] for row in formal["rows"]}
 if (len(smoke_values) != 5*smoke_per_condition*33+1 or len(formal_values) != 165001
     or smoke_values & formal_values or used & forbidden
     or any(v+d in forbidden or (d and v+d in used) for v in used for d in (-1,0,1))):
  raise ValueError("NEW_SEED_ANYROLE_OR_SDK_STATE_COLLISION")
 _exclusive_json(out/names[0], smoke)
 _exclusive_json(out/names[1], formal)
 _exclusive_json(out/names[2], dict(H_sha256=sorted(hh), scope=report["H_identity_sources"],
  whole_trajectory_only=True, scientific_performance_read=False))
 report.update(
  new_namespaces=[SMOKE_NAMESPACE, FORMAL_NAMESPACE],
  allocator_source=ALLOCATOR, allocation=smoke["allocation"],
  small_scalar_engineering_exclusions=sorted(engineering),
  total_any_role_forbidden_with_SDK_neighbors=len(forbidden),
  smoke_per_condition=smoke_per_condition, formal_per_condition=1000,
  smoke_seed_count=len(smoke_values), formal_seed_count=len(formal_values),
  new_anyrole_overlap=0, SDK_neighbor_overlap=0, smoke_formal_overlap=0,
  bootstrap=dict(condition_stream="SeedSequence([master, condition ordinal])",
    same_indices_for_all_methods_and_SNR=True, independent_new_master_per_domain=True,
    smoke_master=smoke["rows"][-1]["seed"], formal_master=formal["rows"][-1]["seed"],
    bootstrap_draws_generated=False),
  metadata_hashes={name:sha(out/name) for name in names[:3]},
  helper_sha256=sha(Path(__file__)), protocol_frozen_by_caller=False)
 _exclusive_json(out/names[3], report)
 return report

def main():
 p = argparse.ArgumentParser()
 p.add_argument("--root", required=True)
 p.add_argument("--output", required=True)
 args = p.parse_args()
 _, hh, report = collect(args.root)
 out = Path(args.output)
 allowed = Path(args.root)/"outputs/revision_mr08a_independent_confirmation"
 if not out.resolve().is_relative_to(allowed.resolve()):
  raise ValueError("ONLY_NEW_MR08_OUTPUT_ALLOWED")
 out.parent.mkdir(parents=True, exist_ok=True)
 with out.open("x", encoding="utf8") as f:
  json.dump(report | {"historical_H_sha256":sorted(hh)}, f, indent=2, ensure_ascii=False, allow_nan=False)
 print(json.dumps({k:report[k] for k in ("PASS","additional_MR_individual_seed_count","total_any_role_forbidden_with_SDK_neighbors","historical_whole_H_hash_count","new_channel_calls","scientific_performance_read")}))

if __name__ == "__main__":
 main()
