"""Resource-only gates; no scientific inputs, metrics, model selection or output."""
from scripts.sp_u12b5b_r1_common import *
firewall()
import pytest
from scripts.sp_u12b5b_r1_audit import sizes,header
import numpy as np,ast
def test_B5A_reference_adapter_B5B_archives_unchanged():
 verified=check_frozen();assert verified['reference_tree_sha256']=='e6c27180ba2abb07c699e943ba581290371a4dea04583b19b63037c552c3db5f'
 assert verified['adapter_sha256']=='f92729e5f4b623775a1ffdd4013fe7f85d8ca86913d05383fe4274fb4c38b0cc'
def test_source_input_bytes_from_header_only():
 h=header();assert h['shape']==[12000,4032] and h['data_bytes']==774144000 and h['payload_values_read']==0 and h['header_bytes_read']==128
 assert h['Kmeans_shape']==[12000,8064] and h['Kmeans_representation_bytes']==774144000
def test_no_extra_adapter_training_copy():
 from src.baselines.mfa_reference_adapter import fit_reference
 class Recorder:
  def fit(self,values):self.values=values
 a=np.zeros((4,8),dtype=np.complex128);m=Recorder();fit_reference(m,a,4197123403)
 assert m.values is a and np.shares_memory(m.values,a)
def test_probe_seeds_separate_from_all_formal_roles():
 cfg=read(CONFIG);formal=read(ROOT/'outputs/sp_u12b5b_candidate_seed_registry.json')['candidates'];seeds={r[k] for r in formal for k in ('seed','final_seed')}
 assert not seeds&set(cfg['seeds'].values()) and len(set(cfg['seeds'].values()))==2
def test_probe_has_no_preparation_or_target_IDs():
 cfg=read(CONFIG);assert cfg['preparation_IDs']==[] and cfg['target_IDs']==[] and cfg['input'].startswith('independent shape-equivalent')
def test_probe_full_shape_dtype_no_dataset_reduction():
 cfg=read(CONFIG);assert (cfg['N'],cfg['D'],cfg['dtype'])==(12000,4032,'complex128') and cfg['maxiter'] in (1,2)
def test_worst_case_grid_scaling_exact():
 allrows=[(K,r,sizes(12000,4032,K,r)) for K in (1,2,4) for r in (4,8,16)]
 assert max(allrows,key=lambda a:a[2]['principal_live_peak_bytes'])[:2]==(4,16)
 assert sizes(12000,4032,4,16)['loading_cross_tensor_bytes']==12386304000
 assert sizes(12000,4032,1,16)['loading_cross_tensor_bytes']==12386304000
def test_allocation_bytes_are_not_claimed_as_RSS():
 model=read(ROOT/'outputs/sp_u12b5b_r1_allocation_model.json');assert 'not an unconditional RSS lower bound' in model['memory_semantics']
def test_frozen_thresholds_exact():
 limits=read(CONFIG)['limits'];assert limits['max_process_bytes']==6*2**30 and limits['min_system_available_physical_bytes']==2*2**30
 assert threshold_reason(dict(peak_working_set_bytes=6*2**30),dict(available_physical_bytes=3*2**30),limits)=='PROCESS_RSS_CAP'
 assert threshold_reason(dict(peak_working_set_bytes=1),dict(available_physical_bytes=2*2**30-1),limits)=='SYSTEM_PHYSICAL_HEADROOM'
 assert threshold_reason(dict(peak_working_set_bytes=6*2**30-1),dict(available_physical_bytes=2*2**30),limits) is None
def test_probe_worker_no_validation_selection_serialization():
 source=(ROOT/'scripts/sp_u12b5b_r1_worker.py').read_text(encoding='utf8');syntax=ast.parse(source)
 forbidden={'estimate','predict_proba','automatic_select','save_model','np.load','np.save','bootstrap'}
 calls={ast.unparse(v.func) for v in ast.walk(syntax) if isinstance(v,ast.Call)}
 assert not calls&forbidden and 'fit_reference' in calls
 cfg=read(CONFIG);assert not cfg['validation'] and not cfg['NMSE'] and not cfg['model_selection'] and not cfg['final_retraining']
def test_firewall_blocks_scientific_archives_before_open():
 for name in ('outputs/sp_u12b2b_sealed/forbidden_probe.csv','outputs/sp_u12b3g_next_sealed/forbidden_probe.csv','outputs/sp_u12b5b_candidates/K1_r4/initialization.json','outputs/sp_u12b5b/validation_truth.npy'):
  with pytest.raises(PermissionError):(ROOT/name).open('rb')
def test_sealed_manifests_and_complete_before_population():
 before=read(OUT/'preservation_before.json');assert before['seals']['B2']['files']==35031 and before['seals']['B3']['files']==10014
 assert before['seals']['B2']['PASS'] and before['seals']['B3']['PASS'] and sha(p.B2M)==p.B2SHA and sha(p.B3M)==p.B3SHA
def test_old_failed_attempt_never_resumed_or_selected():
 stop=read(ROOT/'outputs/sp_u12b5b/STOP.json');assert not stop['selection_performed'] and not stop['final_retraining_started']
 assert not (ROOT/'outputs/sp_u12b5b_final_mfa/model_manifest.json').exists()
 assert read(CONFIG)['role']=='RESOURCE_ONLY_NO_SCIENTIFIC_SELECTION_USE'
def test_actual_probe_resource_stop_and_no_model_outputs():
 file=ROOT/'outputs/sp_u12b5b_r1_resource_probe_results.json'
 if not file.exists():pytest.skip('Actual resource probe not yet executed')
 results=read(file);assert results['executed'] and len(results['records'])==1
 r=results['records'][0];assert r['actual_PID_monitored'] and r['monitor_samples']>0
 assert not r['parameters_exported'] and not r['objectives_exported'] and not r['preparation_data_used'] and not r['formal_data_used'] and not r['formal_candidate_completed']
 assert r['completed'] or r['stop_reason'] in ('SYSTEM_PHYSICAL_HEADROOM','PROCESS_RSS_CAP','RESOURCE_PROBE_TECHNICAL_TIME_CAP')
def test_actual_trace_metadata_has_no_scientific_values():
 file=OUT/'K4_r16/stages.jsonl'
 if not file.exists():pytest.skip('Actual resource-only source trace not yet executed')
 events=[json.loads(v) for v in file.read_text(encoding='utf8').splitlines()]
 assert events
 for e in events:
  assert set(e)=={'UTC','elapsed_seconds','stage','source_function','source_line','arrays_metadata_only','process','system'}
  for a in e['arrays_metadata_only']:assert set(a)=={'name','shape','dtype','nbytes'}
