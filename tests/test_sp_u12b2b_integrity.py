"""Actual sealed-artifact tests. Assert plain; emit metadata only."""
import json
from pathlib import Path
import numpy as np
from scripts import sp_u12b2b_common as b
from scripts import sp_u12b2a_common as c

def test_actual_formal_population_and_raw_integrity():
 from scripts.run_sp_u12b2b import integrity
 b.firewall()
 with b.phase('INTEGRITY'):result=integrity()
 assert result['PASS'] and result['reference_trajectories']==2500 and result['evaluation_trajectories']==5000 and result['condition_SNR_cells']==35
 assert result['raw_logical_rows']==1470000 and result['trajectory_logical_rows']==245000 and result['nonfinite_count']==result['technical_solve_failures']==0

def test_seed_registry_and_execution_preservation():
 b.check_freeze();audit=b.read(b.ROOT/'outputs/sp_u12b2b_seed_audit.json')
 assert audit['PASS'] and audit['individual_seed_duplicates']==audit['reference_evaluation_overlap']==audit['descriptive_formal_development_history_overlap']==0

def test_actual_primary_bootstrap_replicates_and_seed_stream():
 cfg=b.config();audit=b.read(b.ROOT/'outputs/sp_u12b2b_bootstrap_audit.json')
 assert audit['PASS'] and audit['primary_family_size']==35 and audit['primary_replicates_per_contrast']==100000
 with b.phase('SEAL_CHECK'):
  for condition in ('Profile','Mobility','Composed'):
   path=b.SEAL/'primary_bootstrap'/f'{condition}.npz';info=b.committed(path.with_suffix('.json'));value=b.arrays(path)['draws']
   assert info['master_seed']==cfg['primary_bootstrap']['master_seed'] and info['replicates']==100000 and value.shape==(100000,info['contrasts']) and np.isfinite(value).all()

def test_actual_descriptive_bootstrap_replicates_and_separation():
 cfg=b.config();audit=b.read(b.ROOT/'outputs/sp_u12b2b_bootstrap_audit.json')
 assert audit['descriptive_replicates_per_cell']==10000 and audit['descriptive_cells']==245 and audit['all_index_stream_hashes_unique']
 with b.phase('SEAL_CHECK'):
  for condition in c.CONDITIONS:
   path=b.SEAL/'descriptive_bootstrap'/f'{condition}.npz';info=b.committed(path.with_suffix('.json'));value=b.arrays(path)['draws']
   assert info['master_seed']==cfg['descriptive_bootstrap']['seed'] and info['master_seed']!=cfg['primary_bootstrap']['master_seed'] and info['interval_scope']=='DESCRIPTIVE_ONLY' and value.shape==(10000,7,7) and np.isfinite(value).all()

def test_manifest_hash_tree_and_all_artifacts():
 import hashlib
 manifest=b.read(b.ROOT/'outputs/sp_u12b2b_sealed_manifest.json')
 assert manifest['status']=='SEALED_COMPLETE_UNVIEWED' and manifest['condition_SNR_cells']==35 and not manifest['outcomes_viewed'] and not manifest['scientific_figures_viewed']
 for name,record in manifest['files'].items():assert b.sha(b.SEAL/name)==record['sha256']
 assert hashlib.sha256(json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==manifest['all_artifact_hash_tree']

def test_coordinate_schema_and_no_figures():
 with b.phase('SEAL_CHECK'):
  values=b.arrays(b.SEAL/'coordinate_data.npz');assert values['coordinates'].shape==(5,1000,7,6,2) and np.isfinite(values['coordinates']).all()
 assert not any(p.suffix.lower() in ('.pdf','.png','.svg','.jpg') for p in b.SEAL.rglob('*') if p.is_file())
