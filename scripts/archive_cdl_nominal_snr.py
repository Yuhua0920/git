"""T15A-FIX preserve original nominal-transmit-SNR artifacts before correction.

This operation copies files and hashes them; it never imports a channel generator.
Original numerical JSON/CSV contents and recorded SHA256 values remain intact.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/"outputs/t15a_attempt01_nominal_tx_snr"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive():
    if ARCHIVE.exists(): raise ValueError("existing attempt01 archive protected")
    results=json.loads((ROOT/"outputs/t15a_cdl_results.json").read_text(encoding="utf8"))
    artifacts=sorted(p for p in (ROOT/"outputs").glob("t15a_*") if p.is_file())
    sources=[ROOT/p for p in results["provenance"]["source_sha256"]]
    sources += [ROOT/"scripts/audit_cdl_benchmark.py",ROOT/"docs/t15a_execution_notes.md"]
    records={}
    ARCHIVE.mkdir(parents=True)
    for p in artifacts+sources:
        relative=p.relative_to(ROOT)
        destination=ARCHIVE/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,destination)
        original=sha(p); copied=sha(destination)
        if original!=copied: raise ValueError(f"archive mismatch: {relative}")
        records[str(relative).replace("\\","/")]={"sha256":original,"bytes":p.stat().st_size}
    metadata=dict(task="T15A-FIX",status="superseded_due_to_snr_semantics",
        original_task="T15A",original_JSON_preserved_byte_for_byte=True,
        reason="attempt01 used sigma2=Ptot/10^(SNR/10), not train-ensemble receive-power reference",
        original_snr_definition="nominal transmit SNR; Ptot=1",
        corrected_snr_definition="ensemble expected useful received signal power per RX branch / complex AWGN variance",
        old_noise_variances={str(snr):1/10**(snr/10) for snr in (5,10,20)},
        original_train_mean_channel_power=results["calibration"]["mean_channel_power"],
        selected_original_trajectory_hashes=results["calibration"]["selected_trajectory_hashes"],
        original_clean_statistics=results["empirical_lmmse_definition"]["snapshots"],
        complete_raw_H_saved=results["runtime"]["complete_raw_H_saved"],
        file_hashes=records,archive_copy_verification_passed=True,
        numerical_repair_executed=False,formal_test_used=False,formal_test_sealed=True)
    (ARCHIVE/"superseded_metadata.json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps({"archive":str(ARCHIVE),"status":metadata["status"],
        "files":len(records),"archive_copy_verification_passed":True},indent=2))


if __name__=="__main__": archive()
