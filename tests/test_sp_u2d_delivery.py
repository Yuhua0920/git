"""Outcome delivery checks, independent of historical formal tests."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2d_diagnostic as d
d.install_guard()


def test_raw_allocation_and_symmetric_counts():
    raw=d.read_rows(d.OUT/'sp_u2d_age_rows.csv')
    assert len(raw)==24000
    keys={(a['trajectory_id'],a['repeat_id'],a['snr_db'],a['frame_id'],a['age']) for a in raw}
    assert keys=={(i,r,s,j,a) for i in d.IDS for r in d.REPEATS for s in d.SNRS for j in d.FRAMES for a in d.AGES}
    assert all(a['retained_frames']==4 and a['real_design_rows']==512 and a['removed_frame']==a['frame_id']-a['age'] for a in raw)


def test_identity_and_nonnegative_quadratic():
    for a in d.read_rows(d.OUT/'sp_u2d_age_rows.csv'):
        if not a['singular']:
            assert a['Q']>=0
            d.close(a['delta_R'],a['Q']+a['C'])
            d.close(a['delta_R'],a['R_full']-a['R_lofo'])


def test_paired_before_cluster_aggregation():
    raw=d.read_rows(d.OUT/'sp_u2d_age_rows.csv');pairs=d.read_rows(d.OUT/'sp_u2d_paired_rows.csv')
    assert len(pairs)==4800
    for k,row in enumerate(pairs):assert d.paired(raw[5*k:5*k+5])==row
    from scripts.analyze_sp_u2d import aggregate,CONTRAST_FIELDS
    assert aggregate(pairs,CONTRAST_FIELDS)==d.read_rows(d.OUT/'sp_u2d_trajectory_contrasts.csv')


def test_pre_frozen_energy_partition_preserves_repeats():
    pre=d.load(d.OUT/'sp_u2d_preregistration.json');assert pre['energy_strata_outcome_free']
    energy=d.read_rows(d.OUT/'sp_u2d_energy_manifest.csv')
    for s in d.SNRS:
        assert [sum(a['snr_db']==s and a['energy_tercile']==q for a in energy) for q in (1,2,3)]==[40,40,40]
    rows=d.read_rows(d.OUT/'sp_u2d_paired_rows.csv')
    assert all(sum(a['trajectory_id']==i and a['frame_id']==j and a['snr_db']==s for a in rows)==20 for i in d.IDS for j in d.FRAMES for s in d.SNRS)


def test_replay_and_source_input_hashes():
    from scripts.run_sp_u2d import verify
    pre=verify();audit=d.load(d.OUT/'sp_u2d_integrity_audit.json');gate=d.load(d.OUT/'sp_u2d_run_gate.json')
    assert len(pre['protected_sources'])==529 and len(pre['input_sha256'])==430
    assert sum(a['LOFO_rows'] for a in audit['replay'])==180 and all(a['all_fields_exact'] for a in audit['replay'])
    assert gate['full_coordinate_replays_bit_exact']==gate['prediction_hash_replays_exact']==4800
    assert gate['raw_sha256']==d.sha(d.OUT/'sp_u2d_age_rows.csv')


def test_five_figure_QA_and_source_integrity():
    review=d.load(d.OUT/'sp_u2d_visual_review.json');qa=d.load(d.OUT/'sp_u2d_figure_qa_execution.json')
    assert review['passed'] and review['rendered_figures_inspected']==list('ABCDE')
    assert len(qa['records'])==10 and qa['all_exit_zero']
    for f in review['figures']:
        assert f['visual_QA'].startswith('PASS') and d.sha(Path(f['source']))==f['source_sha256']
    assert review['figures'][-1]['alignment_verdict']=='PASS'


def test_evidence_packet_and_stop_are_consistent():
    packet=d.load(d.OUT/'sp_u2d_review_packet.json');pre=d.load(d.OUT/'sp_u2d_preregistration.json')
    assert packet['hypotheses_verbatim']==pre['protocol']['hypotheses_verbatim']
    assert len(packet['hypothesis_evaluation'])==5 and packet['decision'].startswith('C:')
    assert packet['no_next_task'] and not any(packet[k] for k in ('old_outcomes_used_for_design','new_channel','new_noise','production_changed','formal_read'))
    stats=d.read_rows(d.OUT/'sp_u2d_table3_paired_contrasts.csv')
    risk={a['snr_db']:a for a in stats if a['quantity']=='D_R'}
    assert risk[10]['mean']>0 and risk[20]['mean']<0
    assert 'strong A current-state gate FAIL' in packet['decision']


def test_recorded_firewall_allows_only_authorized_reads():
    for filename in ('sp_u2d_preregistration.json','sp_u2d_run_gate.json','sp_u2d_integrity_audit.json','sp_u2d_analysis.json'):
        audit=d.load(d.OUT/filename)
        assert all(a['allowed'] and (a['path'].split('/')[0].startswith('sp_u2d_') or (a['path'] in d.ALLOWED and not a['writing'])) for a in audit['access'])
