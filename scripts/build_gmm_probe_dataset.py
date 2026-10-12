"""Deterministic T15A replay in frozen .venv-cdl; export only the T15B NPZ."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.gmm_probe import probe_indices,pilot_unit_noise,selection_matrix
from src.cdl_benchmark import OfficialCDL,frames,array_hash,seeds,manifest_rows
from src.numerics import complex_gaussian_noise
from scripts.check_cdl_environment import environment_fingerprint,check_main_unchanged

OUT=ROOT/"outputs"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding="utf8"))
def save(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf8")


def frozen_environments():
    main=check_main_unchanged()
    old=load(OUT/"t15a_cdl_environment.json")
    current=environment_fingerprint(ROOT/".venv-cdl")
    freeze=subprocess.check_output([str(ROOT/".venv-cdl/Scripts/python.exe"),"-m","pip","freeze"],text=True)
    cdl=dict(payload_unchanged=current["files"]==old["isolated_payload_fingerprint"]["files"],
             sha256=current["sha256"],freeze_unchanged=freeze==(OUT/"t15a_cdl_packages_freeze.txt").read_text(encoding="utf8"))
    if not main["all_checks_passed"] or not cdl["payload_unchanged"] or not cdl["freeze_unchanged"]:
        raise ValueError("STOP: frozen environments changed")
    return dict(main=main,cdl=cdl,all_checks_passed=True)


def frozen_history():
    paths=subprocess.check_output(["git","ls-files"],cwd=ROOT,text=True).splitlines()
    return {p:sha(ROOT/p) for p in paths if p.startswith(("src/","scripts/","tests/","configs/","outputs/"))}


def main():
    if Path(sys.prefix).resolve()!=(ROOT/".venv-cdl").resolve(): raise ValueError("frozen .venv-cdl required")
    target=OUT/"t15b_gmm_probe_dataset.npz"
    if target.exists(): raise ValueError("existing probe dataset protected")
    gate=load(OUT/"t15b_gmm_environment.json")
    if not gate["all_checks_passed"]: raise ValueError("STOP: author package gate failed")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    subject=subprocess.check_output(["git","show","-s","--format=%s",commit],text=True).strip()
    if subject!="feat: integrate CDL-C benchmark with corrected receive-SNR semantics":
        raise ValueError("accepted T15A/FIX commit required")
    before=frozen_history(); env=frozen_environments()
    print("Frozen main/CDL environments and accepted commit PASS",flush=True)
    p,d=probe_indices();indices=np.r_[p,d]
    index_manifest=dict(order="C [rx,tx,k,t]; pilots first then stratified data",pilot_indices=p.tolist(),
                        data_indices=d.tolist(),coordinates=np.array(np.unravel_index(indices,(2,2,64,16))).T.tolist(),
                        index_sha256=array_hash(indices),rule="4x4 lower bin midpoint per link; mask-only nearest-data fallback",
                        frozen_before_generation=True,protocol_sha256=sha(ROOT/"docs/gmm_probe_protocol.md"))
    save(OUT/"t15b_gmm_probe_index_manifest.json",index_manifest)
    with (OUT/"t15a_cdl_manifest.csv").open(encoding="utf8",newline="") as f: original=list(csv.DictReader(f))
    if original!=[{k:str(v) for k,v in r.items()} for r in manifest_rows()]: raise ValueError("manifest mapping changed")
    with (OUT/"t15a_fix_replay_h_hashes.csv").open(encoding="utf8") as f: href={int(r["trajectory_id"]):r for r in csv.DictReader(f)}
    with (OUT/"t15a_fix_replay_frame_hashes.csv").open(encoding="utf8") as f: nref={(int(r["trajectory_id"]),int(r["frame_id"])):r for r in csv.DictReader(f)}
    fix=load(OUT/"t15a_cdl_results.json")
    if not fix["all_checks_passed"] or not fix["snr_semantics_repaired"]: raise ValueError("accepted FIX required")
    train=np.empty((5000,128),complex); val=np.empty((1000,128),complex);unit=np.empty((1000,64),complex)
    generator=OfficialCDL();sionna=generator.provenance()
    if sionna!=fix["sionna_provenance"]: raise ValueError("CDL provenance differs")
    hrows=[];noise_matched=0;start=time.perf_counter()
    for i in range(600):
        h=generator.generate(i);value=array_hash(h)
        if value!=href[i]["replayed_h_sha256"]: raise ValueError(f"STOP: FIX trajectory {i} H hash differs")
        historical=href[i]["historical_h_hash_available"]=="True"
        if historical and value!=href[i]["historical_h_sha256"]:raise ValueError("historical H mismatch")
        hrows.append(dict(trajectory_id=i,h_sha256=value,T15A_FIX_match=True,historical_attempt01_hash_available=historical))
        for j,frame in enumerate(frames(h)):
            row=frame.ravel()[indices]
            if i<500: train[10*i+j]=row
            else:
                v=10*(i-500)+j;val[v]=row
                noise=complex_gaussian_noise((2,64,16),1.,rng=np.random.default_rng(seeds(i,j)["noise_seed"]))
                if array_hash(noise)!=nref[i,j]["unit_noise_sha256"]: raise ValueError("FIX unit-noise replay mismatch")
                noise_matched+=1;unit[v]=pilot_unit_noise(noise)
        if (i+1)%100==0:print(f"Probe replay {i+1}/600",flush=True)
    np.savez(target,train=train,validation=val,validation_unit_pilot_noise=unit,
             train_trajectory_ids=np.repeat(np.arange(500),10),validation_trajectory_ids=np.repeat(np.arange(500,600),10),
             train_frame_ids=np.tile(np.arange(10),500),validation_frame_ids=np.tile(np.arange(10),100),
             pilot_indices=p,data_indices=d,observation_matrix=selection_matrix(),
             snr_db=np.array([5,10,20]),sigma2=np.array([r["new_sigma2"] for r in fix["noise_variances"]]))
    if before!=frozen_history(): raise ValueError("frozen historical files changed")
    report=dict(task="T15B",accepted_T15A_FIX_commit=commit,accepted_commit_subject=subject,
                environment=env,package_gate_sha256=sha(OUT/"t15b_gmm_environment.json"),
                protocol_sha256=index_manifest["protocol_sha256"],probe_index_hash=index_manifest["index_sha256"],
                probe_index_manifest="outputs/t15b_gmm_probe_index_manifest.json",dataset_path=str(target.relative_to(ROOT)).replace("\\","/"),
                dataset_sha256=sha(target),array_sha256={"train":array_hash(train),"validation":array_hash(val),"unit_pilot_noise":array_hash(unit)},
                train_shape=list(train.shape),validation_shape=list(val.shape),unit_pilot_noise_shape=list(unit.shape),
                dtype="complex128",train_trajectories=500,validation_trajectories=100,frames_per_trajectory=10,
                historical_six_h_hashes_matched=True,T15A_FIX_600_h_hashes_matched=True,
                T15A_FIX_validation_1000_unit_noise_hashes_matched=noise_matched==1000,h_hashes=hrows,
                source_manifest_sha256=sha(OUT/"t15a_cdl_manifest.csv"),sionna_provenance=sionna,
                corrected_sigma2=fix["noise_variances"],train_reference_signal_power=fix["train_reference_signal_power"],
                frozen_history_sha256=before,frozen_history_unchanged=True,replay_seconds=time.perf_counter()-start,
                formal_test_used=False,formal_test_sealed=True,all_checks_passed=True)
    save(OUT/"t15b_gmm_probe_dataset_manifest.json",report)
    print("Probe dataset saved; 600 H and 1000 validation raw unit-noise hashes exact PASS",flush=True)


if __name__=="__main__": main()
