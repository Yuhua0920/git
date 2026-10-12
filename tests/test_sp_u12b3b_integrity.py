"""Machine-only checks: no scientific result values are printed or asserted richly."""
from scripts import sp_u12b3b_common as b
def test_frozen_execution_bytes():b.check_freeze()
def test_full_coverage_and_no_failures():
 a=b.read(b.ROOT/'outputs/sp_u12b3b_integrity_audit.json');assert a['PASS'] and a['condition_SNR_cells']==10 and a['distinct_H_reused']==5000 and a['new_H_generated']==0 and a['units']==10000 and a['trajectory_rows']==60000 and a['frame_rows']==360000
 assert a['inactive_coordinates_NA'] and a['coordinate_sharing'] and a['no_duplicate_primary_keys'] and a['unexpected_solve_failures']==a['nonfinite_outputs']==0
def test_bitwise_Both_replay():
 a=b.read(b.ROOT/'outputs/sp_u12b3b_both_replay_audit.json');assert a['PASS'] and a['max_technical_difference']==a['tolerance']==0 and not a['B2_values_disclosed'] and not a['B3_values_disclosed']
def test_bootstrap_exact_cardinality_and_scope():
 a=b.read(b.ROOT/'outputs/sp_u12b3b_bootstrap_audit.json');assert a['PASS'] and a['primary_contrasts']==12 and a['primary_replicates_per_contrast']==10000 and a['primary_draws']==120000 and a['descriptive_points']==60 and a['descriptive_replicates_per_point']==10000
 assert a['primary_q']==.05/24 and a['quantile_method']=='linear' and not a['global_hypothesis']
def test_all_sealed_files_hash_and_bootstrap_shape():
 m=b.read(b.ROOT/'outputs/sp_u12b3b_manifest.json');assert not m['outcomes_viewed'] and not m['scientific_figures_viewed']
 for n,r in m['files'].items():assert b.sha(b.SEAL/n)==r['sha256'] and (b.SEAL/n).stat().st_size==r['bytes']
 with b.phase('INTEGRITY'):
  for p in (b.SEAL/'bootstrap').glob('*.npz'):
   x=b.arrays(p)['draws'];assert x.shape==((10000,4) if p.name.startswith('primary') else (10000,2,3,2))
def test_human_readable_schema_count_finite_only():
 import csv,numpy as np
 b.firewall()
 with b.phase('INTEGRITY'):
  for name,count in [('aggregate_results',60),('descriptive_results',60),('figure_data',60),('primary_contrasts',12)]:
   with (b.SEAL/(name+'.csv')).open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
   assert len(rows)==count
   for row in rows:
    for k in ('pooled_nmse_db','estimate_db','lower','upper'):
     if k in row:assert np.isfinite(float(row[k]))
   if name!='primary_contrasts':assert {r['condition'] for r in rows}==set(b.CONDITIONS)
   else:assert {r['condition'] for r in rows}=={'Profile','Mobility','Composed'}
def test_preservation_and_B2_before_after():
 assert b.read(b.ROOT/'outputs/sp_u12b3b_preservation.json')['PASS']
 # Hashes were recomputed opaquely by preservation; reconstruct B2 tree without decoding.
 m=b.read(b.MANIFEST);after=b.read(b.TECH/'after_hashes.json')['hashes']
 for n,r in m['files'].items():assert after['outputs/sp_u12b2b_sealed/'+n]==r['sha256']
 assert b.sha(b.MANIFEST)==b.MANIFEST_SHA and b.sha(b.CERT)==b.CERT_SHA
