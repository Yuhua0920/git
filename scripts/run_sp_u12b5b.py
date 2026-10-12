"""Sequential complete-grid selection, fresh final retraining, and mandatory STOP."""
from scripts.sp_u12b5b_common import *
firewall()
from scripts.sp_u12b5b_model import macro_score,automatic_select
from scripts.sp_u12b5b_monitor import run_worker
import argparse,re,traceback,xml.etree.ElementTree as ET

def test():
 assert not (OUT/'execution_freeze.json').exists(),'EXECUTION_ALREADY_FROZEN'
 result=subprocess.run([sys.executable,'-X','utf8','-m','scripts.sp_u12b5b_pytest'],cwd=ROOT,capture_output=True,text=True,encoding='utf8')
 (OUT/'pytest_preflight.log').write_text(result.stdout+result.stderr,encoding='utf8')
 root=ET.parse(OUT/'pytest_preflight.xml').getroot();suites=list(root.iter('testsuite'))
 counts={name:sum(int(v.attrib.get(name,0)) for v in suites) for name in ('tests','failures','errors','skipped')}
 report=dict(exit_code=result.returncode,passed=counts['tests']-counts['failures']-counts['errors']-counts['skipped'],failed=counts['failures']+counts['errors'],skipped=counts['skipped'])
 write(OUT/'pytest_preflight.json',report);print(result.stdout,flush=True)
 assert result.returncode==0 and report['failed']==0 and report['skipped']==0,'B5B_PREFLIGHT_TEST_FAILED'
 for kind in ('train','validate'):
  r=run_worker(kind,'CALIBRATION',calibration=True);assert r['completed'],'B5B_EXACT_WORKER_CALIBRATION_FAILED'
  assert r['actual_PID_handshake_used'] and r['monitor_samples']>0,'REAL_PID_MONITOR_UNVERIFIED'
 write(OUT/'exact_entrypoint_calibration.json',dict(PASS=True,scientific_candidate_fit_calls=0,benchmark_safe_synthetic_data=True,monitored_actual_PID=True,original_author_fit=True,serialization_smoke=read(OUT/'calibration_model/smoke_result.json')))

def sources():
 return ['scripts/sp_u12b5b_common.py','scripts/sp_u12b5b_model.py','scripts/sp_u12b5b_prepare.py','scripts/sp_u12b5b_worker.py','scripts/sp_u12b5b_monitor.py','scripts/sp_u12b5b_pytest.py','scripts/run_sp_u12b5b.py','tests/test_sp_u12b5b.py']
def freeze():
 assert not (OUT/'execution_freeze.json').exists();check_fidelity();tests=read(OUT/'pytest_preflight.json');assert tests['exit_code']==0 and tests['failed']==0 and tests['skipped']==0
 assert read(OUT/'exact_entrypoint_calibration.json')['PASS']
 files=sources()+['configs/sp_u12b5b_mfa_selection.yaml','outputs/sp_u12b5b_preparation_split_manifest.json','outputs/sp_u12b5b_candidate_seed_registry.json','outputs/sp_u12b5b/input_manifest.json','outputs/sp_u12b5b/pytest_preflight.json','outputs/sp_u12b5b/exact_entrypoint_calibration.json']
 hashes={n:sha(ROOT/n) for n in files};write(OUT/'execution_freeze.json',dict(hashes=hashes,B5A_fidelity_manifest_sha256=FID_SHA,config_sha256=sha(CONFIG),scientific_candidate_fits_started=0,created_UTC=utc()))
 print(json.dumps(dict(stage='B5B_execution_frozen',SHA256=sha(OUT/'execution_freeze.json'),tests=tests)),flush=True)
def check_execution():
 check_fidelity();m=read(OUT/'execution_freeze.json')
 for n,h in m['hashes'].items():assert sha(ROOT/n)==h,'B5B_EXECUTION_SOURCE_DRIFT:'+n

def tables():
 cfg=read(CONFIG);training=[];convergence=[];scores=[];cells=[]
 for row in cfg['candidate_order']:
  directory=ROOT/'outputs/sp_u12b5b_candidates'/row['name']
  tr=read(directory/'training_result.json') if (directory/'training_result.json').exists() else None
  val=read(directory/'validation_result.json') if (directory/'validation_result.json').exists() else None
  initial=read(directory/'initialization.json') if (directory/'initialization.json').exists() else None
  monitor=read(OUT/f"monitor_{row['name']}_train.json") if (OUT/f"monitor_{row['name']}_train.json").exists() else None
  entry=read(directory/'fit_entry.json') if (directory/'fit_entry.json').exists() else None
  events=[json.loads(line) for line in (directory/'iteration_events.jsonl').read_text(encoding='utf8').splitlines()] if (directory/'iteration_events.jsonl').exists() else []
  status='COMPLETE' if tr and val else 'TRAINED_VALIDATION_INCOMPLETE' if tr else 'TECHNICAL_FAILURE' if monitor and not monitor['completed'] else 'NOT_RUN'
  record=dict(candidate=row['name'],K=row['K'],rank=row['rank'],seed=row['seed'],status=status,fit_calls=entry['fit_calls'] if entry else 0,training_data_hash=read(OUT/'input_manifest.json')['files']['selection_train.npy'],initial_parameter_hash=initial['combined_hash'] if initial else None,final_parameter_hash=tr['final_parameter_hash'] if tr else None,iterations=tr['iterations'] if tr else len(events) if entry else None,final_objective=tr.get('final_objective') if tr else events[-1]['pre_update_objective'] if events else None,convergence_status=tr['convergence_status'] if tr else 'NOT_COMPLETED',wall_seconds=monitor['wall_seconds'] if monitor else None,peak_RSS_bytes=monitor['peak_RSS_bytes'] if monitor else None,stop_reason=monitor['stop_reason'] if monitor else None)
  training.append(record);convergence.append({n:record[n] for n in ('candidate','K','rank','seed','status','iterations','convergence_status')})
  score=dict(K=row['K'],rank=row['rank'],seed=row['seed'],iterations=record['iterations'],convergence_status=record['convergence_status'],**{f'Origin{o+1}_{s}_linear_pooled_NMSE':None for o in range(3) for s in (10,20)},macro_selection_score=val['macro_selection_score'] if val else None,status=status)
  if val:
   cells.extend(val['cells'])
   for c in val['cells']:
    o=cfg['canonical_origins'].index(c['origin']);score[f"Origin{o+1}_{c['snr_dB']}_linear_pooled_NMSE"]=c['linear_pooled_NMSE']
  scores.append(score)
 csvwrite(ROOT/'outputs/sp_u12b5b_candidate_training_audit.csv',training);csvwrite(ROOT/'outputs/sp_u12b5b_candidate_convergence.csv',convergence);csvwrite(ROOT/'outputs/sp_u12b5b_selection_scores.csv',scores)
 if cells:csvwrite(ROOT/'outputs/sp_u12b5b_validation_cell_metrics.csv',cells)
 else:
  csvwrite(ROOT/'outputs/sp_u12b5b_validation_cell_metrics.csv',[dict(candidate=r['name'],K=r['K'],rank=r['rank'],origin=o,snr_dB=s,error_energy=None,truth_energy=None,linear_pooled_NMSE=None,trajectories=0,scored_frames='4,5,6,7,8,9',status='NOT_COMPUTED') for r in cfg['candidate_order'] for o in cfg['canonical_origins'] for s in (10,20)])
 return training,scores,cells

def failure(reason,detail):
 status='MFA_SELECTION_TARGET_LEAKAGE' if 'TARGET_INFORMATION_FIREWALL' in str(detail) else 'MFA_SELECTION_PROTOCOL_VIOLATION' if reason=='PROTOCOL' else 'MFA_SELECTION_TECHNICAL_FAILURE'
 selection_done=(ROOT/'outputs/sp_u12b5b_selection_decision.json').exists() and read(ROOT/'outputs/sp_u12b5b_selection_decision.json')['status']=='SELECTED'
 final_started=(ROOT/'outputs/sp_u12b5b_final_mfa/fit_entry.json').exists()
 write(OUT/'STOP.json',dict(verdict=status,reason=reason,detail=detail,created_UTC=utc(),selection_performed=selection_done,final_retraining_started=final_started,remaining_candidates_not_run=not selection_done,STOP=True))
 if not (ROOT/'outputs/sp_u12b5b_selection_decision.json').exists():write(ROOT/'outputs/sp_u12b5b_selection_decision.json',dict(status='NOT_SELECTED',selected_K=None,selected_rank=None,tie_breaker_invoked=False,reason=reason))
 if not (ROOT/'outputs/sp_u12b5b_selection_manifest.json').exists():write(ROOT/'outputs/sp_u12b5b_selection_manifest.json',dict(status='SELECTION_INCOMPLETE_NO_MODEL_SELECTED',B5A_fidelity_manifest_sha256=FID_SHA,reason=reason,target_outcomes_viewed=False))
 for n in ('final_training_audit','final_smoke_test'):
  if not (ROOT/f'outputs/sp_u12b5b_{n}.json').exists():write(ROOT/f'outputs/sp_u12b5b_{n}.json',dict(status='NOT_EXECUTED',reason='selection/final pipeline stopped; no substitute model',PASS=None))
 tables();print(json.dumps(dict(verdict=status,reason=reason,STOP=True)),flush=True)

def run():
 check_execution();assert not (OUT/'RUN_STARTED.json').exists(),'FORMAL_RESTART_FORBIDDEN';cfg=read(CONFIG)
 write(OUT/'RUN_STARTED.json',dict(created_UTC=utc(),fit_policy='one fit per candidate, no restarts',target_outcomes_viewed=False))
 try:
  for row in cfg['candidate_order']:
   check_execution();train=run_worker('train',row['name'])
   if not train['completed']:failure('CANDIDATE_TRAINING_TECHNICAL_FAILURE',train);return
   validate=run_worker('validate',row['name'],budget_seconds=cfg['resource_contract']['max_candidate_seconds']-train['wall_seconds'])
   if not validate['completed']:failure('CANDIDATE_VALIDATION_TECHNICAL_FAILURE',validate);return
   tables()
  training,scores,cells=tables();assert len(cells)==54 and all(t['status']=='COMPLETE' for t in training)
  winner=automatic_select(scores);selected=next(r for r in cfg['candidate_order'] if (r['K'],r['rank'])==(winner['K'],winner['rank']))
  tied=[r for r in scores if r['macro_selection_score']==winner['macro_selection_score']]
  manifest=dict(status='SELECTION_FROZEN',B5A_fidelity_manifest_sha256=FID_SHA,execution_freeze_sha256=sha(OUT/'execution_freeze.json'),split_manifest_sha256=sha(ROOT/'outputs/sp_u12b5b_preparation_split_manifest.json'),candidate_grid=cfg['candidate_order'],candidate_model_hashes={r['candidate']:r['final_parameter_hash'] for r in training},validation_input_hash=read(OUT/'input_manifest.json')['files']['validation_pilots.npy'],validation_truth_hash=read(OUT/'input_manifest.json')['files']['validation_truth.npy'],cell_metrics_sha256=sha(ROOT/'outputs/sp_u12b5b_validation_cell_metrics.csv'),selection_scores_sha256=sha(ROOT/'outputs/sp_u12b5b_selection_scores.csv'),selected_K=winner['K'],selected_rank=winner['rank'],score=winner['macro_selection_score'],selection_rule=cfg['selection_metric'],tie_breaker=cfg['tie_breaker'],tie_breaker_invoked=len(tied)>1,target_outcomes_viewed=False)
  write(ROOT/'outputs/sp_u12b5b_selection_manifest.json',manifest)
  write(ROOT/'outputs/sp_u12b5b_selection_decision.json',dict(status='SELECTED',selected_K=winner['K'],selected_rank=winner['rank'],score=winner['macro_selection_score'],tie_breaker_invoked=len(tied)>1,final_seed=selected['final_seed'],selection_manifest_sha256=sha(ROOT/'outputs/sp_u12b5b_selection_manifest.json')))
  check_execution();final=run_worker('train','FINAL')
  if not final['completed']:failure('FINAL_RETRAINING_TECHNICAL_FAILURE',final);return
  smoke=run_worker('validate','FINAL',budget_seconds=cfg['resource_contract']['max_candidate_seconds']-final['wall_seconds'])
  if not smoke['completed']:failure('FINAL_SERIALIZATION_SMOKE_TECHNICAL_FAILURE',smoke);return
  directory=ROOT/'outputs/sp_u12b5b_final_mfa';result=read(directory/'training_result.json');artifact=read(directory/'model_artifact.json');test=read(directory/'smoke_result.json')
  write(ROOT/'outputs/sp_u12b5b_final_training_audit.json',result);write(ROOT/'outputs/sp_u12b5b_final_smoke_test.json',test)
  interface=read(ROOT/'outputs/sp_u12b5a_mfa_interface_contract.json');write(directory/'interface_contract.json',interface)
  write(directory/'inference_config.json',dict(input='received owner pilots + known symbols/geometry + known source-referenced variance',output='3968 data-domain complex channel estimate',n_summands_or_proba=1.0,python_type='float',all_components=True,normalization='none',reference_noise_variances=cfg['noise_variances'],target_information=False))
  final_manifest=dict(status='FINAL_MFA_MODEL_FROZEN',K=selected['K'],rank=selected['rank'],final_training_seed=selected['final_seed'],preparation_trajectories=1500,training_order=read(OUT/'input_manifest.json')['training_order'],training_data_manifest_hash=sha(OUT/'input_manifest.json'),training_data_hash=read(OUT/'input_manifest.json')['files']['full_retraining.npy'],reference_tree_hash=check_fidelity()['reference']['combined_tree_sha256'],adapter_hash=sha(ROOT/'src/baselines/mfa_reference_adapter.py'),parameter_hash=artifact['parameter_hash'],parameter_hashes=read(directory/'parameter_hashes.json'),selection_manifest_hash=sha(ROOT/'outputs/sp_u12b5b_selection_manifest.json'),model_artifact_hash=sha(directory/'model_artifact.json'),history_hash=sha(directory/'training_history.npy'),interface_hash=sha(directory/'interface_contract.json'),inference_config_hash=sha(directory/'inference_config.json'),software_environment=cfg['software'],target_information_used=False)
  write(directory/'model_manifest.json',final_manifest);write(OUT/'SUCCESS.json',dict(verdict='MFA_FINAL_MODEL_FROZEN',final_model_manifest_sha256=sha(directory/'model_manifest.json'),STOP=True))
  print(json.dumps(dict(verdict='MFA_FINAL_MODEL_FROZEN',selected_K=selected['K'],selected_rank=selected['rank'],STOP=True)),flush=True)
 except BaseException as error:
  failure('UNEXPECTED_EXECUTION_FAILURE',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))

def close():
 assert (OUT/'STOP.json').exists() or (OUT/'SUCCESS.json').exists();check_execution();check_fidelity()
 preserve('after')
 cfg=read(CONFIG);training,scores,cells=tables();verdict=read(OUT/'STOP.json')['verdict'] if (OUT/'STOP.json').exists() else 'MFA_FINAL_MODEL_FROZEN'
 monitors=[read(p) for p in OUT.glob('monitor_*.json')];scientific=[r for r in monitors if not r['calibration_only']]
 write(ROOT/'outputs/sp_u12b5b_resource_log.json',dict(records=monitors,scientific_records=scientific,selection_metric_uses_resources=False,system=system_memory(),memory_contract=cfg['resource_contract']))
 access=[read(p) for p in OUT.glob('worker_*_access.json')]
 write(ROOT/'outputs/sp_u12b5b_target_firewall_audit.json',dict(PASS=not DENIED and all(not r['denied'] for r in access),target_scientific_files_read=0,B2_B3_scientific_values_read=0,target_files_read_excludes_opaque_integrity_hashing=True,opaque_hash_purpose='start/end byte integrity only; no decoding or estimator/selection use',preparation_only_sources=list(PREPARATION),candidate_fit_entry_source='immutable selection_train.npy derived only from original preparation corpus',final_fit_entry_source='full_retraining.npy after selection manifest only',worker_access_logs=access,controller_allowed_reads=ACCESS,controller_denied=DENIED,pytest_prevented_probes_separate=read(OUT/'pytest_firewall_expected_denials.json')))
 decision=read(ROOT/'outputs/sp_u12b5b_selection_decision.json')
 content='# Target-blind MFA selection results\n\nVerdict: '+verdict+'\n\n'+json.dumps(dict(origins=cfg['canonical_origins'],candidate_training=training,selection_scores=scores,decision=decision,resource_records=scientific),indent=2)+'\n\nNo target performance, LR4 comparison or B2/B3 outcome was accessed. Unrun cells are null, not substitute scores.\n'
 (ROOT/'docs/sp_u12b5b_selection_results.md').write_text(content,encoding='utf8')
 final_info=read(ROOT/'outputs/sp_u12b5b_final_training_audit.json')
 (ROOT/'docs/sp_u12b5b_final_model_freeze.md').write_text('# Final MFA freeze status\n\n'+json.dumps(dict(verdict=verdict,final_training=final_info,final_model_exists=(ROOT/'outputs/sp_u12b5b_final_mfa/model_manifest.json').exists(),selection=decision),indent=2)+'\n\nSTOP. No target evaluation authorized in this task.\n',encoding='utf8')
 (ROOT/'docs/sp_u12b5b_change_log.md').write_text('# B5B change log\n\nNew B5B orchestration, tests and outputs only. Pinned author reference and B5A adapter unchanged. Reused the existing clean preparation corpus, all1500 historical channel hashes and trajectory split0–399/400–499. Precomputed common validation inputs, scored frames4–9 only. Original author fit and full float1.0 posterior remain unchanged. Initialization and per-step instrumentation capture hashes/history without modifying numerical formulas.\n\nMentor B5B amendments: valid maxiter100 outputs participate; exact floating tie picks smaller K then rank. These override older notes without editing them. Existing PLAN_B6GiB RSS/2GiB physical headroom/24h configuration resource gate stops the whole selection, never selects surviving candidates.\n\nNo formal target input/outcome, no model rescue, no manuscript edits and no Git commit.\n',encoding='utf8')
 files=[v for d in ('outputs','configs','docs','scripts','tests') for v in (ROOT/d).rglob('*') if v.is_file() and owned(v) and '__pycache__' not in v.parts]
 path=ROOT/'outputs/sp_u12b5b_manifest.json';sidecar=path.with_suffix('.sha256');files=[v for v in files if v not in (path,sidecar)]
 frozen={v.relative_to(ROOT).as_posix():sha(v) for v in sorted(files)}
 summary=dict(task='SP-U12B5B',verdict=verdict,B5A_fidelity_manifest_sha256=FID_SHA,reference_adapter_unchanged=True,preparation_split_sha256=sha(ROOT/'outputs/sp_u12b5b_preparation_split_manifest.json'),candidate_seeds_sha256=sha(ROOT/'outputs/sp_u12b5b_candidate_seed_registry.json'),execution_freeze_sha256=sha(OUT/'execution_freeze.json'),candidate_grid=cfg['candidate_order'],candidate_fit_calls=sum(r['fit_calls'] for r in training),successful_candidates=sum(r['status']=='COMPLETE' for r in training),decision=decision,final_model_frozen=(ROOT/'outputs/sp_u12b5b_final_mfa/model_manifest.json').exists(),final_model_manifest_sha256=sha(ROOT/'outputs/sp_u12b5b_final_mfa/model_manifest.json') if (ROOT/'outputs/sp_u12b5b_final_mfa/model_manifest.json').exists() else None,target_scientific_files_read=0,B2_B3_outcomes_viewed=False,target_evaluation_executed=False,LR4_MFA_target_comparison_generated=False,preservation=read(ROOT/'outputs/sp_u12b5b_preservation.json'),seals=read(ROOT/'outputs/sp_u12b5b_b2_b3_seal_audit.json'),pytest=read(OUT/'pytest_preflight.json'),files=frozen,STOP=True,next_action='Mentor review; no automatic restart/rescue/B5C')
 write(path,summary);sidecar.write_text(sha(path)+'\n',encoding='ascii')
 print(json.dumps(dict(verdict=verdict,manifest=str(path),SHA256=sha(path),successful_candidates=summary['successful_candidates'],fit_calls=summary['candidate_fit_calls'],STOP=True)),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['test','freeze','run','close']);args=parser.parse_args();globals()[args.stage]()
