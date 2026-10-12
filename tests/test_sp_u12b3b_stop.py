"""Administrative stop validity, never decode raw scientific arrays."""
from scripts import sp_u12b3b_common as b
def test_permanent_STOP_and_no_continuation():
 s=b.read(b.TECH/'PERMANENT_STOP.json');assert s['verdict']=='B3_FORMAL_TECHNICAL_FAILURE' and s['error_type']=='PermissionError' and not s['outcomes_viewed']
 try:b.check_freeze()
 except b.TechnicalFailure as e:assert str(e)=='PERMANENT_STOP'
 else:raise AssertionError('STOP_NOT_ENFORCED')
def test_frozen_runner_bytes_preserved():
 for p,h in b.read(b.TECH/'execution_freeze.json')['hashes'].items():assert b.sha(b.ROOT/p)==h
def test_preservation_and_B2_seal_complete():
 p=b.read(b.ROOT/'outputs/sp_u12b3b_preservation.json');assert p['PASS'] and p['authorized_preregistration_amendment'] and p['manuscript_drift']==p['production_semantic_drift']==p['old_artifact_drift']==0
 a=b.read(b.ROOT/'outputs/sp_u12b3b_b2_seal_audit.json');assert a['PASS'] and not a['B2_outcomes_viewed'] and a['before']['files']==a['after']['files']==35031
def test_partial_manifest_hashes_and_scope():
 m=b.read(b.ROOT/'outputs/sp_u12b3b_manifest.json');assert m['status']=='SEALED_TECHNICAL_FAILURE_UNVIEWED' and m['committed_H']==3 and m['complete_cells']==0 and not m['outcomes_viewed'] and not m['scientific_figures_viewed'] and not m['runner_patched_after_freeze'] and not m['continued_formal_run']
 for n,r in m['files'].items():assert b.sha(b.SEAL/n)==r['sha256']
 for n in ('aggregate_results.csv','primary_contrasts.csv','descriptive_results.csv','figure_data.csv'):assert not(b.SEAL/n).exists()
 assert not list((b.SEAL/'bootstrap').rglob('*.npz'))
def test_bootstrap_not_falsely_certified():
 a=b.read(b.ROOT/'outputs/sp_u12b3b_bootstrap_audit.json');assert not a['PASS'] and a['completed_primary_replicates']==a['completed_descriptive_replicates']==0
def test_partial_replay_not_falsely_global_PASS():
 a=b.read(b.ROOT/'outputs/sp_u12b3b_both_replay_audit.json');assert not a['PASS'] and a['partial_PASS'] and a['committed_H']==3 and a['max_technical_difference']==0 and not a['scientific_values_disclosed']
