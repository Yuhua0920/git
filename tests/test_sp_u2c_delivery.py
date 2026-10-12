"""Actual delivered-results checks; never silently skip absent artifacts."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2c_diagnostic as d
d.install_guard()


def test_actual_budget_and_ids():
    rows=d.read_rows(d.OUT/'sp_u2c_repeat_rows.csv')
    assert len(rows)==4800 and {a['trajectory_id'] for a in rows}==set(range(20))
    assert {(a['trajectory_id'],a['repeat_id'],a['snr_db'],a['frame_id']) for a in rows}=={
        (i,r,s,j) for i in range(20) for r in range(20) for s in (10,20) for j in range(4,10)}


def test_actual_clean_coordinate_invariance():
    rows=d.read_rows(d.OUT/'sp_u2c_repeat_rows.csv')
    for i in range(20):
        for s in (10,20):
            for j in range(4,10):
                group=[a for a in rows if (a['trajectory_id'],a['snr_db'],a['frame_id'])==(i,s,j)]
                assert len({a['clean_coordinate_sha256'] for a in group})==1


def test_actual_risk_identities():
    for a in d.read_rows(d.OUT/'sp_u2c_repeat_rows.csv'):
        d.close(a['delta_R_noise'],a['Q']+a['C'])
        d.close(a['surrogate_gap']+a['delta_R_noise'],a['total_oracle_gap'])
        if not a['numeric_borderline']:assert (a['delta_R_noise']<0)==(a['C'] < -a['Q'])
        assert a['Q']>=0 and a['fit_rank']==a['output_rank']==2


def test_actual_empirical_covariance_and_summaries():
    from scripts.analyze_sp_u2c import build_conditional
    rows=d.read_rows(d.OUT/'sp_u2c_repeat_rows.csv');saved=d.read_rows(d.OUT/'sp_u2c_conditional_windows.csv')
    rebuilt=build_conditional(rows)
    assert len(saved)==240 and all(all(v==b[k] for k,v in a.items()) for a,b in zip(rebuilt,saved))
    for a in saved:
        d.close(a['V_c'],.95*a['trace_covariance']+a['bias_norm']**2)


def test_integrity_and_replay():
    from scripts.run_sp_u2c import verify
    pre=verify();audit=d.load(d.OUT/'sp_u2c_integrity_audit.json')
    assert audit['passed'] and audit['repeat_cache_hashes_exact']==400
    assert [(a['trajectory_id'],a['repeat_id']) for a in audit['replay']]==[tuple(v) for v in pre['protocol']['replay_pairs']]
    assert sum(a['windows_replayed'] for a in audit['replay'])==36
    assert not audit['formal_read'] and audit['channel_generator_calls']==0


def test_all_six_figure_qa_and_counts():
    review=d.load(d.OUT/'sp_u2c_figure_review.json')
    assert review['passed'] and len(review['figures'])==6
    for a in review['figures']:
        assert a['collision']=='PASS' and a['physical_width_mm']==183
        assert a['alignment']==('PASS' if a['figure']=='C' else 'NOT APPLICABLE')


def test_scope_and_hypothesis_freeze():
    pre=d.load(d.OUT/'sp_u2c_preregistration.json');packet=d.load(d.OUT/'sp_u2c_advisor_review_packet.json')
    assert packet['hypotheses_verbatim']==pre['protocol']['hypotheses_verbatim']
    assert packet['decision']=='C. MIXED EFFECT' and packet['stopped']
    assert not any(packet[k] for k in ('next_task_started','production_changed','paper_changed','committed'))
