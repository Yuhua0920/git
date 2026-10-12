"""T11B-FIX: verify versioned provenance and update metadata without simulation."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T11A_COMMIT = "9fe2f165f9f511c9fe1c90585f2ec5023ef3c951"
T11A_JSON = "outputs/t11a_nearest_interpolation_results.json"
T11B_JSON = "outputs/t11b_nearest_validation_results.json"
T11A_JSON_SHA256 = "060cf65ce7cbdbb88857d9db1b881b048331eafac1f1f3e1fe29f72fcf89a7e6"
HISTORICAL_PATHS = ("docs/baseline_registry.md","tests/test_nearest_interpolation.py")
PRE_FIX_PAYLOAD_SHA256 = "efbb052ce38f71ce5378ffa13909a43b77853634a55848319ea6a0ec616a1283"
METADATA_FIELDS = {"all_checks_passed","historical_t11a_commit","historical_registry_provenance_verified",
    "historical_test_provenance_verified","current_registry_updated","provenance_fix"}
PROTECTED_HASHES = {
    T11A_JSON:T11A_JSON_SHA256,
    "outputs/t11b_nearest_validation_trajectory_metrics.csv":"bb6ace1f045e863972c61d7afaeb6843685e4a898de0996011ec6bac83e00c73",
    "outputs/t11b_r0_r5_validation_baselines.csv":"970d9d5a4f6dba83ef21e8834ba6d2ba07d852333ca3dcd6781282a07150ada8",
    "outputs/t10d_lmmse_train_validation_scan.csv":"e815f93a9ec0cbd35e5f49a21804f7665a035f13f65c112308029b02a2e06c76",
    "outputs/t10d_lmmse_train_validation_trajectory_metrics.csv":"5d87c74f41a83860ca24d0bbdf04ebc912ab7d1669b0ef1e0b9261db0116bb60",
    "outputs/t10d_lmmse_train_validation_results.json":"1e19c8d0bb0f1bab1658f5cad98bc7bd29ca8704aae121d6981e014ce41bfca5",
    "outputs/t10c_train_validation_manifest.csv":"03c4fc237c374741df0d3aa43956461e4870885cf851e1f63640f00041c488f4",
    "outputs/t10c_lmmse_train_covariance.npz":"b41ec0128243d07efb7577785e798994e5d8bc00905ab40d174ff1ee0da8dc3e",
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob(root,path):
    # Capture binary stdout directly: shell text conversion would change bytes.
    return subprocess.run(["git","show",f"{T11A_COMMIT}:{path}"],cwd=root,
        check=True,capture_output=True).stdout


def verify_historical_t11a(root):
    import sys
    sys.path.insert(0,str(ROOT))
    from src.provenance import recorded_source_matches
    root = Path(root)
    if sha256_file(root/T11A_JSON)!=T11A_JSON_SHA256:
        raise ValueError("current T11A historical JSON changed")
    result = json.loads((root/T11A_JSON).read_text(encoding="utf-8"))
    expected = result["provenance"]["source_sha256"]
    historical = {p:hashlib.sha256(git_blob(root,p)).hexdigest() for p in HISTORICAL_PATHS}
    if any(historical[p]!=expected[p] for p in HISTORICAL_PATHS):
        raise ValueError("T11A commit-based historical provenance mismatch")
    if result["baseline_registry"]["path"]!=HISTORICAL_PATHS[0] or historical[HISTORICAL_PATHS[0]]!=result["baseline_registry"]["sha256"]:
        raise ValueError("T11A historical registry record mismatch")
    current = {p:expected[p] for p in expected if p not in HISTORICAL_PATHS}
    if any(not recorded_source_matches(root,p,current[p],"t11a",current_hash=sha256_file) for p in current):
        raise ValueError("current T11A core/script/config source changed")
    return {"historical_t11a_commit":T11A_COMMIT,"historical_registry_provenance_verified":True,
        "historical_test_provenance_verified":True,"historical_source_sha256":historical,
        "current_T11A_numerical_source_sha256":current,"T11A_JSON_sha256":T11A_JSON_SHA256}


def verify_current_registry(root):
    path = Path(root)/HISTORICAL_PATHS[0]
    text = path.read_text(encoding="utf-8")
    rows = [[part.strip() for part in line.split("|")[1:-1]] for line in text.splitlines()
        if any(line.startswith(f"| R{i} |") for i in range(6))]
    if len(rows)!=6 or [r[0] for r in rows]!=[f"R{i}" for i in range(6)] or any(len(r)!=5 for r in rows):
        raise ValueError("current registry must contain exact R0-R5 rows")
    for row in rows:
        status = "pilot-domain calibration available" if row[0]=="R1" else "held-out ID validation available"
        if row[4]!=status:
            raise ValueError(f"current registry validation status mismatch: {row[0]}")
    if "data-grid NMSE=N/A" not in rows[1][3] or "data BER=N/A" not in rows[1][3]:
        raise ValueError("R1 must remain pilot-only with N/A data metrics")
    if ("不是 final formal-test E1 results" not in text or "formal test remains sealed." not in text):
        raise ValueError("registry must state validation scope and sealed formal test")
    return {"current_registry_updated":True,"current_registry_sha256":sha256_file(path)}


def protected_payload_sha256(report):
    payload = {k:v for k,v in report.items() if k not in METADATA_FIELDS}
    data = json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def verify_protected_files(root):
    actual = {p:sha256_file(Path(root)/p) for p in PROTECTED_HASHES}
    if actual!=PROTECTED_HASHES:
        raise ValueError("protected T11A JSON / T11B CSV / frozen reference changed")
    return actual


def finalize(root=ROOT):
    import sys
    sys.path.insert(0,str(ROOT))
    from src.provenance import recorded_source_matches
    root = Path(root)
    before = verify_protected_files(root)
    raw = (root/T11B_JSON).read_bytes()
    report = json.loads(raw.decode("utf-8"))
    if protected_payload_sha256(report)!=PRE_FIX_PAYLOAD_SHA256:
        raise ValueError("T11B pre-fix numerical/results payload changed")
    if not all(report["checks"].values()) or not report["replay_verification"]["all_checks_passed"]:
        raise ValueError("existing T11B numerical checks must already be passed")
    for path,digest in report["provenance"]["source_sha256"].items():
        if not recorded_source_matches(root,path,digest,"t11b",current_hash=sha256_file):
            raise ValueError(f"original T11B numerical source changed: {path}")
    historical = verify_historical_t11a(root)
    registry = verify_current_registry(root)
    if "provenance_fix" in report:
        # T11C-A: the completed T11B metadata is now an immutable historical
        # artifact. Re-verify it without rewriting any recorded SHA256/bytes.
        previous = report["provenance_fix"]
        if not report["all_checks_passed"] or any(previous[k]!=v for k,v in {**historical,**registry}.items()):
            raise ValueError("completed T11B provenance gates changed")
        for path,digest in previous["fix_source_sha256"].items():
            if not recorded_source_matches(root,path,digest,"t11b",current_hash=sha256_file):
                raise ValueError(f"historical T11B fix provenance changed: {path}")
        if verify_protected_files(root)!=before or (root/T11B_JSON).read_bytes()!=raw:
            raise ValueError("historical T11B artifacts changed during verification")
        return report
    report.update({k:historical[k] for k in ("historical_t11a_commit",
        "historical_registry_provenance_verified","historical_test_provenance_verified")})
    report["current_registry_updated"] = registry["current_registry_updated"]
    after = verify_protected_files(root)
    report["provenance_fix"] = {"task":"T11B-FIX","metadata_only_update":True,"numerical_scan_executed":False,
        "original_numerical_provenance_preserved":True,**historical,**registry,
        "protected_file_sha256_before_after":{"before":before,"after":after},
        "protected_original_payload_sha256":PRE_FIX_PAYLOAD_SHA256,
        "fix_source_sha256":{p:sha256_file(root/p) for p in (
            "scripts/finalize_nearest_validation_provenance.py","tests/test_historical_provenance.py",
            "tests/test_nearest_interpolation.py","docs/baseline_registry.md")}}
    report["all_checks_passed"] = (all(report["checks"].values()) and
        report["historical_t11a_commit"]==T11A_COMMIT and
        all(report[k] for k in ("historical_registry_provenance_verified","historical_test_provenance_verified","current_registry_updated")))
    if protected_payload_sha256(report)!=PRE_FIX_PAYLOAD_SHA256:
        raise ValueError("metadata update altered original T11B payload")
    # Verify the serialized candidate before replacing the JSON. No CSV is opened
    # for writing and no channel/receiver/simulation module is imported.
    serialized = json.dumps(report,indent=2,allow_nan=False)+"\n"
    if b"\r\n" in raw:
        serialized = serialized.replace("\n","\r\n")
    candidate = json.loads(serialized)
    if protected_payload_sha256(candidate)!=PRE_FIX_PAYLOAD_SHA256 or not candidate["all_checks_passed"]:
        raise ValueError("serialized metadata candidate failed validation")
    temporary = root/"outputs/.t11b_provenance_metadata.tmp"
    temporary.write_bytes(serialized.encode("utf-8"))
    if verify_protected_files(root)!=before:
        raise ValueError("protected files changed before metadata save")
    temporary.replace(root/T11B_JSON)
    if verify_protected_files(root)!=before:
        raise ValueError("protected files changed after metadata save")
    print(json.dumps({"historical_t11a_commit":T11A_COMMIT,
        "historical_registry_provenance_verified":True,"historical_test_provenance_verified":True,
        "current_registry_updated":True,"historical_source_sha256":historical["historical_source_sha256"],
        "protected_files_unchanged":True,"original_T11B_payload_unchanged":True,
        "numerical_scan_executed":False,"all_checks_passed":report["all_checks_passed"]},indent=2))
    return report


if __name__=="__main__":
    finalize()
