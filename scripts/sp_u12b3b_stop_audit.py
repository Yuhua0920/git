"""Administrative post-STOP preservation only. No evaluation, scoring or bootstrap."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b3b_common as b
def write_reports():
 stop=b.read(b.TECH/'PERMANENT_STOP.json');b.need(stop['verdict']=='B3_FORMAL_TECHNICAL_FAILURE','STOP_REQUIRED')
 f=b.read(b.TECH/'execution_freeze.json')
 for p,h in f['hashes'].items():b.need(b.sha(ROOT/p)==h,'FROZEN_BYTES_CHANGED_AFTER_STOP')
 preservation=b.read(ROOT/'outputs/sp_u12b3b_preservation.json');b.need(preservation['PASS'],'PRESERVATION_FAILURE')
 # Hash-only B2 check against the freshly recomputed preservation map.
 old=b.read(b.MANIFEST);hashes=b.read(b.TECH/'after_hashes.json')['hashes'];leaves={}
 for n,r in old['files'].items():
  digest=hashes['outputs/sp_u12b2b_sealed/'+n];b.need(digest==r['sha256'],'B2_LEAF_CHANGE');leaves[n]=dict(sha256=digest,bytes=(b.B2/n).stat().st_size)
 b.need(b.tree(leaves)==old['all_artifact_hash_tree'] and b.tree({n:r for n,r in leaves.items() if n.startswith('raw_evaluation/')})==old['raw_data_hash_tree'],'B2_TREE_CHANGE')
 b.need(b.sha(b.MANIFEST)==b.MANIFEST_SHA and b.sha(b.CERT)==b.CERT_SHA,'B2_METADATA_CHANGE')
 before=b.read(b.TECH/'b2_seal_before.json');after=dict(before,stage='after');b.write(b.TECH/'b2_seal_after.json',after);b.write(ROOT/'outputs/sp_u12b3b_b2_seal_audit.json',dict(PASS=True,before=before,after=after,B2_artifact_drift=0,B2_outcomes_viewed=False,check='all35031leafhashes freshly recomputed afterSTOP;no numerical decoding'))
 # Decode only receipts; they contain IDs/hashes/technical flags, never NMSE/gains.
 receipts=[];peaks={};seen=set()
 for p in sorted((b.SEAL/'raw').rglob('*.json')):
  r=b.read(p);key=(r['condition'],r['trajectory_id']);b.need(key not in seen,'DUPLICATE_COMMIT');seen.add(key)
  b.need(b.sha(ROOT/r['artifact']['path'])==r['artifact']['sha256'],'PARTIAL_RAW_COMMIT_HASH')
  b.need(r['execution_freeze_sha256']==b.sha(b.TECH/'execution_freeze.json'),'PARTIAL_FREEZE_BINDING')
  b.need(r['Both_current_predictions_bitwise'] and r['Both_B2_coordinates_energies_bitwise'] and r['Dense_LR4_coordinates_shared'],'PARTIAL_REPLAY')
  receipts.append(dict(condition=r['condition'],trajectory_id=r['trajectory_id'],artifact=r['artifact']));peaks[str(r['worker_pid'])]=r['peak']
 count=len(receipts);npz=list((b.SEAL/'raw').rglob('*.npz'));b.need(len(npz)==count,'UNCOMMITTED_RAW_ARCHIVE')
 intact=dict(status='PARTIAL_ONLY_NOT_FULL_PASS',full_integrity_completed=False,committed_H_reused=count,committed_units=count*2,committed_trajectory_rows=count*12,committed_frame_rows=count*72,completed_condition_SNR_cells=0,expected_cells=10,expected_H=5000,new_H_generated=0,completed_record_coordinate_sharing=True,completed_record_solver_failures=0,completed_record_nonfinite_outputs=0,uncompleted_record_status='NOT_CERTIFIED',outcomes_viewed=False,committed_receipts=receipts)
 b.write(ROOT/'outputs/sp_u12b3b_integrity_audit.json',intact)
 b.write(ROOT/'outputs/sp_u12b3b_both_replay_audit.json',dict(status='PARTIAL_RECORDS_PASS_FULL_RUN_NOT_COMPLETE',PASS=False,partial_PASS=True,committed_H=count,committed_units=count*2,committed_scored_frames=count*12,max_technical_difference=0.0,tolerance=0.0,current_prediction_replay='bitwise',B2_coordinate_energy_replay='bitwise',scientific_values_disclosed=False))
 b.write(ROOT/'outputs/sp_u12b3b_bootstrap_audit.json',dict(status='NOT_RUN_AFTER_TECHNICAL_STOP',PASS=False,planned_primary_contrasts=12,planned_primary_replicates_per_contrast=10000,completed_primary_replicates=0,planned_descriptive_points=60,planned_descriptive_replicates_per_point=10000,completed_descriptive_replicates=0,seed_registry_sha256=b.sha(b.TECH/'bootstrap_seed_registry.json'),scientific_values_disclosed=False))
 resource=dict(status='TECHNICAL_STOP',run_wall_seconds_upper_bound=(b.TECH/'PERMANENT_STOP.json').stat().st_mtime-(b.TECH/'execution_freeze.json').stat().st_mtime,worker_peaks_from_committed_receipts=peaks,CPU_workers=2,BLAS_threads=1,GPU_used=False,retries=0,resumes=0,new_H_generated=0,formal_start_count=1,bootstrap_started=False,parent_peak_unavailable=True,scope='postSTOP engineering metadata;not B4;run elapsed upper bound includes launch gap')
 b.write(ROOT/'outputs/sp_u12b3b_resource_log.json',resource)
 leaves={p.relative_to(b.SEAL).as_posix():dict(sha256=b.sha(p),bytes=p.stat().st_size) for p in sorted(b.SEAL.rglob('*')) if p.is_file()}
 manifest=dict(status='SEALED_TECHNICAL_FAILURE_UNVIEWED',verdict=stop['verdict'],protocol_sha256=b.sha(b.PROTOCOL),config_sha256=b.sha(b.CFG),execution_freeze_sha256=b.sha(b.TECH/'execution_freeze.json'),bootstrap_seed_registry_sha256=b.sha(b.TECH/'bootstrap_seed_registry.json'),B2_input_provenance='B2 immutable cache only;no new H;partial formal commits',B2_seal_before=before,B2_seal_after=after,committed_H=count,complete_cells=0,raw_hash_tree=b.tree({n:r for n,r in leaves.items() if n.startswith('raw/')}),all_artifact_hash_tree=b.tree(leaves),aggregate_sha256=None,bootstrap_hash_tree=None,primary_contrasts_sha256=None,descriptive_sha256=None,figure_data_sha256=None,scientific_output_absence_reason='not reached before permanent technical STOP;no placeholders or fabricated scientific results',files=leaves,outcomes_viewed=False,scientific_figures_viewed=False,runner_patched_after_freeze=False,continued_formal_run=False)
 b.write(ROOT/'outputs/sp_u12b3b_manifest.json',manifest)
 cert='# SP-U12B3B partial sealed technical-stop certificate\n\nVerdict: B3_FORMAL_TECHNICAL_FAILURE\nStatus: SEALED_TECHNICAL_FAILURE_UNVIEWED\n\nThe formal runner raised PermissionError; sanitized failure code does not identify the denied path. No diagnostic rerun or patch was performed. Exactly '+str(count)+' trajectories have committed raw files. No condition-SNR cell is complete. No aggregate, primary/descriptive bootstrap, intervals or figure data were generated. This certificate does not certify a complete scientific evaluation.\n\nProtocol SHA256: '+manifest['protocol_sha256']+'\nConfig SHA256: '+manifest['config_sha256']+'\nExecution freeze SHA256: '+manifest['execution_freeze_sha256']+'\nB2 manifest SHA256 before/after: '+b.MANIFEST_SHA+'\nB2 certificate SHA256 before/after: '+b.CERT_SHA+'\nB2 raw tree before/after: '+before['raw_data_hash_tree']+'\nB2 all tree before/after: '+before['all_artifact_hash_tree']+'\nB3 partial raw tree: '+manifest['raw_hash_tree']+'\nB3 partial all tree: '+manifest['all_artifact_hash_tree']+'\nB3 aggregate/bootstrap/figure hashes: NOT_GENERATED\nManifest SHA256: '+b.sha(ROOT/'outputs/sp_u12b3b_manifest.json')+'\n\nB2 outcomes viewed=false;B3 outcomes viewed=false;scientific figures viewed=false. NewH=0;no retries/resumes. Frozen runner/config/protocol bytes unchanged. B2/oldscientific/production/manuscript drift=0. Only authorized preregistration mentor correction changed an old file, with exact original bytes archived;method_change=0. STOP for mentor review.\n'
 b.atomic(ROOT/'docs/sp_u12b3b_seal_certificate.md',cert.encode())
 log='# SP-U12B3B execution log\n\n'+cert+'\nActual preflight pytest:40passed,0failed,0skipped. Formal execution attempted once. The preflight tests did not catch the PermissionError on the actual sealed worker path. Its exact origin is not established; no stronger root-cause claim is made. PostSTOP activities are limited to metadata, opaque hashes, preservation and reports, with no raw numerical decoding or scientific calculation.\n'
 b.atomic(ROOT/'docs/sp_u12b3b_execution_log.md',log.encode());b.write(b.TECH/'decision.json',dict(verdict='B3_FORMAL_TECHNICAL_FAILURE',STOP=True,committed_H=count,completed_cells=0,outcomes_viewed=False,manifest_sha256=b.sha(ROOT/'outputs/sp_u12b3b_manifest.json')))
 b.emit(stage='stop_audit',verdict='B3_FORMAL_TECHNICAL_FAILURE',trajectory_count=count,cells_completed=0,outcomes_viewed=False)
if __name__=='__main__':write_reports()
