"""Resource audit orchestration; never restarts scientific structure selection."""
from scripts.sp_u12b5b_r1_common import *
firewall()
import argparse,xml.etree.ElementTree as ET
def sources():
 return ['scripts/sp_u12b5b_r1_common.py','scripts/sp_u12b5b_r1_audit.py','scripts/sp_u12b5b_r1_worker.py','scripts/sp_u12b5b_r1_monitor.py','scripts/sp_u12b5b_r1_pytest.py','scripts/run_sp_u12b5b_r1.py','tests/test_sp_u12b5b_r1.py']
def tests(phase):
 result=subprocess.run([sys.executable,'-X','utf8','-m','scripts.sp_u12b5b_r1_pytest',phase],cwd=ROOT,capture_output=True,text=True,encoding='utf8');(OUT/f'pytest_{phase}.log').write_text(result.stdout+result.stderr,encoding='utf8')
 root=ET.parse(OUT/f'pytest_{phase}.xml').getroot();suites=list(root.iter('testsuite'));counts={n:sum(int(v.attrib.get(n,0)) for v in suites) for n in ('tests','failures','errors','skipped')}
 report=dict(exit_code=result.returncode,passed=counts['tests']-counts['failures']-counts['errors']-counts['skipped'],failed=counts['failures']+counts['errors'],skipped=counts['skipped']);write(OUT/f'pytest_{phase}.json',report);print(result.stdout,flush=True)
 assert report['exit_code']==0 and report['failed']==0 and (phase=='preflight' or report['skipped']==0)
def freeze():
 assert not (OUT/'execution_freeze.json').exists();check_frozen();test=read(OUT/'pytest_preflight.json');assert test['exit_code']==0
 files=sources()+['outputs/sp_u12b5b_r1_resource_probe_config.json','outputs/sp_u12b5b_r1_allocation_model.json','outputs/sp_u12b5b_r1/pytest_preflight.json']
 write(OUT/'execution_freeze.json',dict(hashes={n:sha(ROOT/n) for n in files},frozen=check_frozen(),UTC=utc(),formal_candidate_calls=0,resource_probe_calls=0));print(json.dumps(dict(stage='R1_execution_frozen',SHA256=sha(OUT/'execution_freeze.json'))),flush=True)
def verify():
 m=read(OUT/'execution_freeze.json')
 for n,h in m['hashes'].items():assert sha(ROOT/n)==h,'RESOURCE_INSTRUMENTATION_DRIFT:'+n
 check_frozen()
def probe():
 verify()
 from scripts.sp_u12b5b_r1_monitor import run_probe
 run_probe()
def close():
 verify();result=read(ROOT/'outputs/sp_u12b5b_r1_resource_probe_results.json')['records'][0];model=read(ROOT/'outputs/sp_u12b5b_r1_allocation_model.json');host=read(ROOT/'outputs/sp_u12b5b_r1_host_capacity.json');cfg=read(CONFIG)
 # Allocation bytes are not equated with RSS. A headroom stop does not prove
 # an inevitable >6GiB RSS on a different host; that remains an explicit risk.
 if result['completed']:
  verdict='MFA_RESOURCE_CAPACITY_READY';capacity='CURRENT_HOST_CAPACITY_READY';reason='Worst-case full-shape original initialization, two EM steps and public copies completed within frozen thresholds.'
 elif result['stop_reason']=='PROCESS_RSS_CAP':
  verdict='MFA_FROZEN_RESOURCE_GATE_INCOMPATIBLE';capacity='MFA_FROZEN_RESOURCE_GATE_INCOMPATIBLE';reason='Actual original-path synthetic RSS reached the unchanged process limit.'
 elif result['stop_reason']=='SYSTEM_PHYSICAL_HEADROOM':
  verdict='MFA_HOST_CAPACITY_INSUFFICIENT';capacity='CURRENT_HOST_CAPACITY_INSUFFICIENT';reason='Worst-case resource-only probe could not complete on the current research-idle host without violating physical headroom. No evidence establishes background-pressure-only or gate feasibility on another host.'
 else:
  verdict='MFA_RESOURCE_FORENSICS_INCOMPLETE';capacity='UNDETERMINED';reason='Technical probe did not reach a conclusive capacity/resource gate outcome.'
 worst=next(v for v in model['candidate_models'] if v['candidate']==model['worst_case']);resident_plan=worst['principal_live_peak_bytes']+cfg['limits']['min_system_available_physical_bytes']
 decision=dict(verdict=verdict,capacity_decision=capacity,reason=reason,worst_case=model['worst_case'],worst_case_source_shapes_unambiguous=True,probe=result,source_resident_capacity_planning_lower_bound_bytes=resident_plan,source_lower_bound_excludes_OS_extra_BLAS_and_allocator=True,source_lower_bound_is_NOT_an_RSS_proof=True,minimum_safe_launch_free_RAM='NOT_CERTIFIED: worst-case probe failed; do not restart selection',gate_incompatibility_risk='OPEN unless actual RSS limit observed; rank16 broadcast allocation alone exceeds6GiB, but allocated bytes cannot be equated unconditionally with OS RSS',background_pressure_only_proven=False,adapter_technical_duplicate_bytes_observed=0,algorithm_modified=False,formal_candidates_completed=0,validation_scores_generated=0,structure_selected=False,final_model_trained=False,target_scientific_value_reads=0,B2_B3_values_read=0,STOP=True)
 write(ROOT/'outputs/sp_u12b5b_r1_capacity_decision.json',decision)
 if not result['completed']:
  seeds=read(ROOT/'outputs/sp_u12b5b_candidate_seed_registry.json');inputs=read(ROOT/'outputs/sp_u12b5b/input_manifest.json')
  write(ROOT/'outputs/sp_u12b5b_r1_host_migration_requirements.json',dict(status='REQUIREMENTS_ONLY_NO_MIGRATION_NO_FORMAL_RESTART',recommended_physical_RAM='32GiB minimum capacity-planning target;64GiB gives more headroom for full retraining and background. Neither overrides/certifies the unchanged6GiB RSS gate.',OS='Windows64-bit preferred to preserve current environment',software=cfg['software'],BLAS_threads=1,reference_core_commit='184e97c641cd0d1383d28bce102a53b10a527c89',reference_estimator_commit='93ce8eec4770840a377581e70de14c1a447a628d',frozen=check_frozen(),seed_registry=seeds,data_hashes=inputs['files'],formal_protocol_unchanged=True,resource_limits=cfg['limits'],mandatory_before_authorization=['same reference/adapter/data bytes','reference parity','seed replay','tiny fidelity','full-scale resource smoke under unchanged limits'],BLAS_change_requires_all_parity_checks=True,GPU_conversion=False,selection_grid_restart='If newly authorized, clean restart original K1_r4 and entire9-candidate grid; old failed fit scientific use prohibited'))
 preserve('after');tests('post');verify()
 access=[read(v) for v in (OUT/'K4_r16').glob('access.json')]
 write(OUT/'firewall_audit.json',dict(PASS=not DENIED and all(not v['denied'] for v in access),controller_reads=ACCESS,controller_denied=DENIED,worker_reads=access,target_scientific_value_reads=0,B2_B3_values_read=0,preparation_payload_values_read=0,NPY_input_metadata_header_bytes_only=128,integrity_hash_byte_reads_excluded=True,pytest_blocked_probes=read(OUT/'pytest_expected_firewall_denials_post.json')))
 report=dict(decision=decision,input_memory=read(ROOT/'outputs/sp_u12b5b_r1_training_input_memory.json'),failed_run=read(OUT/'failed_timeline_summary.json'),host_capacity=host,allocation_model=model,preservation=read(ROOT/'outputs/sp_u12b5b_r1_preservation.json'),tests=read(OUT/'pytest_post.json'))
 (ROOT/'docs/sp_u12b5b_r1_resource_forensics.md').write_text('# Resource-only MFA forensics\n\n'+json.dumps(report,indent=2,ensure_ascii=False)+'\n\nSource allocation formulas and observed resource values are separate. Old likelihoods and parameters were never decoded. Exact old failure substage is UNKNOWN. The original batch EM has sequential component temporaries; K multiplies covariances, not the single D*r*N temporary.\n',encoding='utf8')
 (ROOT/'docs/sp_u12b5b_r1_capacity_recommendation.md').write_text('# Host-capacity gate\n\n'+json.dumps(decision,indent=2,ensure_ascii=False)+'\n\nNo safe B5B launch is authorized. Higher-memory capacity planning does not resolve the6GiB RSS gate by itself. Review the original implementation workspace and unchanged gate before any mentor-authorized restart. No apps/services closed, no host migration, no code optimization.\n',encoding='utf8')
 (ROOT/'docs/sp_u12b5b_r1_change_log.md').write_text('# R1 changes\n\nNew resource audit, source trace, isolated synthetic probe and tests only. Frozen author/adapter/B5B failure bytes unchanged. Synthetic input generation in small blocks creates one full complex128 ndarray; EM itself remains the original full batch code. Only the authorized resource-probe maxiter2 differs, never applied to a formal candidate. Source trace records names/shapes/dtypes/nbytes and resource counters, never scientific values.\n\nAudit preflight corrected unavailable private NumPy header reader to the public version-specific reader; the attempted audit source is archived, no payload values were read and no probe had started. Protected archives and manuscript untouched. No commit; STOP for mentor review.\n',encoding='utf8')
 paths=[v for d in ('outputs','docs','scripts','tests') for v in (ROOT/d).rglob('*') if v.is_file() and owned(v) and '__pycache__' not in v.parts]
 path=ROOT/'outputs/sp_u12b5b_r1_manifest.json';sidecar=path.with_suffix('.sha256');paths=[v for v in paths if v not in (path,sidecar)]
 m=dict(task='SP-U12B5B-R1',verdict=verdict,capacity_decision=capacity,decision_sha256=sha(ROOT/'outputs/sp_u12b5b_r1_capacity_decision.json'),frozen=check_frozen(),execution_freeze_sha256=sha(OUT/'execution_freeze.json'),probe_config_sha256=sha(CONFIG),probe_results_sha256=sha(ROOT/'outputs/sp_u12b5b_r1_resource_probe_results.json'),preservation=read(ROOT/'outputs/sp_u12b5b_r1_preservation.json'),seals=read(ROOT/'outputs/sp_u12b5b_r1_b2_b3_seal_audit.json'),pytest=read(OUT/'pytest_post.json'),formal_candidate_calls=0,validation_scores_generated=0,structure_selected=False,final_retraining=False,target_evaluation=False,target_scientific_value_reads=0,firewall_PASS=read(OUT/'firewall_audit.json')['PASS'],files={v.relative_to(ROOT).as_posix():sha(v) for v in sorted(paths)},STOP=True,next_action='Mentor review; no automatic B5B restart or rescue')
 write(path,m);sidecar.write_text(sha(path)+'\n',encoding='ascii');print(json.dumps(dict(verdict=verdict,capacity_decision=capacity,manifest=str(path),SHA256=sha(path),STOP=True)),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['test','freeze','probe','close']);args=parser.parse_args()
 if args.stage=='test':tests('preflight')
 else:globals()[args.stage]()
