"""Bookkeeping gates; no new formal timing or scientific experiment."""
from scripts import sp_u12b4b_audit as b

def test_frozen_leaves_and_formal_dimensions():
 assert len(b.frozen_gate())==24
 c=b.r.config()
 assert c['process_blocks']==12 and c['common_measured_passes']*c['outputs_per_measured_pass']==53248
 assert c['exact_formal_command'][-1]=='--formal'

def test_B4A_and_frozen_B4B_placeholders_are_protected():
 for rel in ('outputs/sp_u12b4a_formal_freeze.json','outputs/sp_u12b4b_runtime/config.json','outputs/sp_u12b4b_runtime/environment.json','outputs/sp_u12b4b_runtime/protocol.md','scripts/run_sp_u12b4_benchmark.py','src/pilot_cv.py'):
  assert not b.new_owned(b.ROOT/rel)
 assert b.new_owned(b.B/'runtime_summary.json')

def test_initial_full_seals_and_no_scientific_decoding():
 record=b.READ(b.BASE)
 assert record['seals']['B2']['files']==35031 and record['seals']['B3']['files']==10014
 assert all(v['PASS'] and not v['outcomes_viewed'] for v in record['seals'].values())

def test_rss_aggregation_does_not_replace_interval_peak_with_lifetime_peak():
 assert 'inference_interval_sampled_peak_RSS_bytes' in b.RSS_FIELDS
 assert 'lifetime_peak_RSS_bytes' not in b.RSS_FIELDS
 assert 'peak_minus_baseline_bytes' in b.RSS_FIELDS
