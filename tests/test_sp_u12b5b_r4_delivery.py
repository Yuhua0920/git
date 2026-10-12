"""Checks actual R4 numerical evidence, frozen byte identity and resource gates."""
from pathlib import Path
import sys,json,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u12b5b_r4_common import firewall,read,sha,OUT,LIMITS
firewall()

def test_actual_all_structure_state_stop_and_resource_evidence():
    decision=read(ROOT/'outputs/sp_u12b5b_r4_decision.json')
    assert decision['verdict']=='MFA_LOW_MEMORY_FIDELITY_READY'
    with(ROOT/'outputs/sp_u12b5b_r4_em_trajectory_parity.csv').open(encoding='utf-8-sig',newline='')as f:rows=list(csv.DictReader(f))
    assert len(rows)==16592 and all(r['PASS']=='True'for r in rows)
    assert {(int(r['K']),int(r['r']))for r in rows if r['case'].startswith('nine_')}=={(k,r)for k in(1,2,4)for r in(4,8,16)}
    assert {int(r['threads'])for r in rows}=={1,4,8,12}
    assert all(float(r['max_tolerance_ratio'])<=1 for r in rows)
    stop=read(ROOT/'outputs/sp_u12b5b_r4_stopping_parity.json')
    assert stop['PASS']and len(stop['cases'])==16
    assert all(r['reference_updates']==r['low_updates']for r in stop['cases'])
    resources=read(ROOT/'outputs/sp_u12b5b_r4_safe_resource_summary.json')
    assert resources['PASS']and resources['completed_workers']==18
    assert resources['max_peak_RSS_GiB']<6 and resources['minimum_physical_available_GiB']>2
    assert decision['formal_candidate_fits']==decision['validation_scores']==decision['target_evaluations']==0
    assert decision['selected_structure']==decision['final_model']=='NONE'
    frozen=read(OUT/'execution_freeze.json')
    assert all(sha(ROOT/n)==h for n,h in frozen['files'].items())
