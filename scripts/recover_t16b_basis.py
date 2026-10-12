"""One authorized, exact historical provenance recovery; no T16C data."""
from pathlib import Path
import hashlib
import json
import stat
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.cdl_benchmark import array_hash
from scripts.run_cdl_mismatch import load,save,sha
from scripts.run_shared_output_basis import construct
COMMIT="8f0c3a5764af3a9a0329aaa21d62ce788bd419ba"
FILES={"all_conditions":"t16b_shared_basis_all_u32.npz","CDL-B-30":"t16b_shared_basis_b30_u32.npz",
       "CDL-C-120":"t16b_shared_basis_c120_u32.npz"}


def recover():
    if Path(sys.prefix).resolve() != (ROOT/".venv-cdl").resolve(): raise ValueError("original frozen CDL environment required")
    if any((ROOT/"outputs"/p).exists() for p in FILES.values()): raise ValueError("one-time recovery artifacts already exist")
    subprocess.check_call(["git","merge-base","--is-ancestor",COMMIT,"HEAD"],cwd=ROOT)
    old=load(ROOT/"outputs/t16b_shared_basis_results.json")
    # The exact historical implementation snapshot and original checkout hashes.
    for p,h in {**old["source_sha256"],**old["numeric_artifact_sha256"],**old["frozen_history_sha256"]}.items():
        if sha(ROOT/p)!=h: raise ValueError(f"historical input changed: {p}")
    for p in old["source_sha256"]:
        blob=subprocess.check_output(["git","show",f"{COMMIT}:{p}"],cwd=ROOT)
        if blob.replace(b"\r\n",b"\n")!=(ROOT/p).read_bytes().replace(b"\r\n",b"\n"):
            raise ValueError("historical implementation snapshot mismatch")
    # Calls only original T16B construction; no channel generator, validation, or heldout input.
    bases=construct()[7]; info={}
    for scope,b in bases.items():
        u=b.output(32)
        expected=old["shared_basis_definition"][scope]
        if array_hash(u)!=expected["U32_sha256"]: raise ValueError(f"STOP: U32 hash differs for {scope}; no phase/order repair permitted")
        saved=np.array([r["singular_value"] for r in old["shared_spectrum"] if r["basis_scope"]==scope])
        if not np.array_equal(b.singular,saved): raise ValueError("historical singular values differ")
        if [list(c) for c in b.conditions]!=expected["basis_conditions"]: raise ValueError("condition order differs")
    # Write only after ALL THREE exact gates have passed.
    for scope,b in bases.items():
        p=ROOT/"outputs"/FILES[scope]; u=b.output(32)
        np.savez_compressed(p,U32=u,singular_values=b.singular,
            construction_condition_ids=np.array(json.dumps([list(c) for c in b.conditions])),
            dtype=np.array(str(u.dtype)),shape=np.array(u.shape),historical_U32_SHA256=np.array(array_hash(u)),
            source_commit=np.array(COMMIT),construction_protocol_hash=np.array(old["source_sha256"]["docs/shared_output_basis_protocol.md"]))
        p.chmod(stat.S_IREAD)
        info[scope]=dict(path=str(p.relative_to(ROOT)).replace("\\","/"),file_sha256=sha(p),U32_sha256=array_hash(u),
            Uq_sha256={str(q):array_hash(u[:,:q]) for q in (4,8,16,24)},shape=list(u.shape),dtype=str(u.dtype),
            condition_ids=[list(c) for c in b.conditions],readonly_file=True)
    proof=dict(historical_basis_artifact_was_missing=True,historical_basis_recovered_once=True,
        historical_u32_hash_exact_match=True,t16c_basis_loaded_from_frozen_artifact=False,
        recovery_definition="authorized provenance recovery of original T16B six development conditions, not T16C basis refit",
        historical_T16B_commit=COMMIT,historical_results_sha256=sha(ROOT/"outputs/t16b_shared_basis_results.json"),
        construction_protocol_hash=old["source_sha256"]["docs/shared_output_basis_protocol.md"],basis_artifacts=info,
        T16C_data_accessed=False,phase_or_order_repairs=False,all_checks_passed=True)
    save(ROOT/"outputs/t16b_basis_provenance_recovery.json",proof)
    print(json.dumps(info,indent=2));print("ALL THREE historical U32 hashes EXACT; provenance recovery PASS")


if __name__=="__main__": recover()
