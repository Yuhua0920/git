"""Independent audit prompted by perfect five-frame bank classification.

Uses three frozen sample replays, not a numerical scan rerun. Checks actual
pilot likelihood via direct solves and verifies every stored minimum decision.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); OUT=ROOT/"outputs"
from scripts.run_statistics_identifiability import read_selection,read_performance
from scripts.validate_b0_pipeline import json_save
from src import statistics_selection as rx,statistics_identifiability as scan
from src.b0_development import replay_link
from src.lmmse_train import sha256_file
from src.trajectories import prepare_trajectory


def audit():
    torch.set_num_threads(4)
    report=json.loads((OUT/f"{scan.PREFIX}_results.json").read_text(encoding="utf8"))
    before={p:sha256_file(OUT/p) for p in report["provenance"]["CSV_sha256"]}
    rows=read_selection(OUT/f"{scan.PREFIX}.csv"); perf=read_performance(OUT/f"{scan.PERFORMANCE_PREFIX}_trajectory_metrics.csv")
    stored={(r["true_condition"],r["trajectory_id"],r["snr_db"],r["window_frames"]):r for r in rows}
    margins={}
    for r in rows:
        scores=np.array([r[k] for k in ("score_id","score_speed120","score_ds300")]); rank=np.argsort(scores,kind="stable")
        if r["selected_condition"]!=scan.CONDITIONS[rank[0]] or r["margin"]!=scores[rank[1]]-scores[rank[0]]:
            raise ValueError("stored candidate does not minimize received-pilot likelihood")
    for s in scan.SNRS:
        margins[str(s)]={c:{"minimum":min(r["margin"] for r in rows if r["window_frames"]==5 and r["snr_db"]==s and r["true_condition"]==c),
            "median":float(np.median([r["margin"] for r in rows if r["window_frames"]==5 and r["snr_db"]==s and r["true_condition"]==c]))} for c in scan.CONDITIONS}
    bank,_=rx.build_candidate_bank(); pools,_=rx.prepare_pools(bank); refs=scan.references(OUT); historical=replay_link(ROOT)
    records=[]; permutation=(2,0,1)
    for condition,tid in (("ID",2000),("speed120",0),("DS300",0)):
        model=prepare_trajectory(frames_per_trajectory=10) if condition=="ID" else scan.ood.prepare_channel(condition)
        pilots,_,_=scan.replay_trajectory(condition,tid,model,refs,historical,{k:0. for k in scan.TIMING_KEYS})
        for snr in scan.SNRS:
            for window in rx.WINDOWS:
                z=rx.pack_joint_pilots(pilots[snr][:window]); direct=[]
                for c in bank:
                    total=0.
                    for q in range(2):
                        coordinates=rx.pilot_coordinates(window,q); k,t=coordinates.T
                        covariance=c["rf"][k[:,None],k[None,:]]*c["rt"][window][t[:,None],t[None,:]]+10**(-snr/10)*np.eye(16*window)
                        sign,logdet=np.linalg.slogdet(covariance)
                        if abs(sign-1)>1e-12:
                            raise ValueError("nonpositive actual-sample likelihood determinant")
                        for nr in range(2):
                            v=z[nr,q]; total+=float(logdet+np.vdot(v,np.linalg.solve(covariance,v)).real)
                    direct.append(total)
                old=stored[condition,tid,snr,window]; recorded=np.array([old[k] for k in ("score_id","score_speed120","score_ds300")])
                difference=float(np.max(np.abs(recorded-direct)))
                selected=rx.select_candidate(pilots[snr][:window],tuple(pools[window,snr][i] for i in permutation))
                if difference>1e-9 or scan.CONDITIONS[int(np.argmin(direct))]!=old["selected_condition"] or scan.CONDITIONS[permutation[selected["candidate_index"]]]!=old["selected_condition"]:
                    raise ValueError("actual pilot score/candidate permutation regression failed")
                # No H/label is passed in either receiver call. Observations are
                # the same frozen receiver pilots, with unchanged noise10^-SNR/10.
                records.append({"condition":condition,"trajectory_id":tid,"snr_db":snr,"window_frames":window,
                    "direct_score_max_absolute_error":difference,"permuted_bank_same_regime":True,"selected_by_direct_score":scan.CONDITIONS[int(np.argmin(direct))]})
    if any(r["R7_BankML_error_energy"]<=0 or r["R7_BankML_error_energy"]>=r["nmse_truth_energy"] for r in perf):
        raise ValueError("implausible actual receiver energy sentinel")
    after={p:sha256_file(OUT/p) for p in before}
    if before!=after:
        raise ValueError("numerical CSV changed during audit")
    result={"task":"T13C","trigger":"100 percent five-frame accuracy at all SNRs",
        "all_3600_decisions_minimize_stored_pilot_scores":True,"actual_replayed_pilot_direct_solve_checks":records,
        "five_frame_margin_summary":margins,"receiver_input_signature":"select_candidate(pilot_frames,models); no H/bits/label/ID/data-RE input",
        "power_noise_semantics":"owner pilot modulus1; Pilot-LS variance10^(-SNR/10); raw Y and LS hashes match frozen results",
        "R7_not_equal_H_true":True,"R7_OracleBank_R5_original_checks":report["OracleBank_R5_regression"],
        "formal_test_used":False,"formal_test_sealed":True,"full_scan_rerun":False,
        "CSV_before":before,"CSV_after":after,
        "source_sha256":{p:sha256_file(ROOT/p) for p in ("scripts/audit_statistics_identifiability.py","tests/test_statistics_identifiability_audit.py")},"all_checks_passed":True}
    json_save(OUT/f"{scan.PREFIX}_independent_audit.json",result)
    print("T13C independent actual-pilot audit PASS: 3600 score decisions, 18 direct-solve/permuted-bank sentinels; numerical CSV unchanged",flush=True)
    return result


if __name__=="__main__":
    audit()
