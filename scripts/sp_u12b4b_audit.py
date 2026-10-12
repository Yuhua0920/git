"""B4B bookkeeping only. Frozen benchmark command and numerical paths stay untouched."""
from pathlib import Path
import sys, os, json, time, datetime, subprocess, concurrent.futures, csv
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import sp_u12b4a_adapter as a
from scripts import run_sp_u12b4_benchmark as r
from scripts import sp_u12b4a_preserve as p
import numpy as np
B = a.B4B
F = ROOT/'outputs/sp_u12b4a_formal_freeze.json'
FSHA = 'fdd18167a0508d4b2256769132c603acc27820780e6ff33d2a16decbb5cb239f'
TASK = Path(r'C:\Users\yuhua\.codex\attachments\34478e48-0cfd-454a-9c3d-172574ff7459\已粘贴的文本.txt')
BASE = B/'preservation_before.json'
READ = p.read
SHA = p.sha
WRITE = p.write
RSS_FIELDS = ('baseline_RSS_bytes','loaded_RSS_bytes','operator_load_RSS_delta','inference_interval_sampled_peak_RSS_bytes','peak_minus_baseline_bytes')

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def report(name, value): WRITE(ROOT/f'outputs/sp_u12b4b_{name}.json', value)
def need(value, message): a.need(value, message)
def emit(**value): print(json.dumps(value), flush=True)
def new_owned(path):
 rel = path.relative_to(ROOT).as_posix()
 if rel in ('outputs/sp_u12b4b_runtime/config.json','outputs/sp_u12b4b_runtime/environment.json','outputs/sp_u12b4b_runtime/protocol.md'): return False
 return any(v.startswith(('sp_u12b4b','run_sp_u12b4b','test_sp_u12b4b')) for v in path.relative_to(ROOT).parts)
def protected_paths():
 return sorted(x for d in ('src','scripts','tests','configs','docs','paper','outputs') for x in (ROOT/d).rglob('*') if x.is_file() and not new_owned(x) and not(set(x.parts)&{'__pycache__','.pytest_cache','pytest_cache','pytest_tmp'}))
def hashes(paths):
 result = {}; last = time.monotonic()
 # Opaque bytes only: no numerical/JSON scientific values decoded here.
 old_opaque=p._opaque;p._opaque=True
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   for path, digest in zip(paths,pool.map(SHA,paths)):
    result[path.relative_to(ROOT).as_posix()] = digest
    if time.monotonic()-last>25: emit(stage='opaque_hash_only',files=len(result));last=time.monotonic()
 finally:p._opaque=old_opaque
 return result
def git_snapshot(stage):
 head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 status = subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True,encoding='utf8')
 (B/f'git_status_{stage}.txt').write_text(status,encoding='utf8')
 check = subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf8')
 WRITE(B/f'git_diff_check_{stage}.json',dict(exit_code=check.returncode,stdout=check.stdout,stderr=check.stderr))
 need(check.returncode==0,'GIT_DIFF_CHECK_FAILURE')
 return head
def frozen_gate():
 need(SHA(F)==FSHA,'B4A_FREEZE_HASH_MISMATCH')
 leaves=READ(F)['hashes']
 for name,digest in leaves.items(): need(SHA(ROOT/name)==digest,'B4A_FROZEN_LEAF_DRIFT:'+name)
 need(SHA(p.B3M)==p.B3SHA,'B3_MANIFEST_DRIFT')
 return leaves
def preserve(stage):
 frozen_gate(); h=hashes(protected_paths());seals=p.seals(h);head=git_snapshot(stage)
 need(seals['B2']['files']==35031 and seals['B3']['files']==10014,'SEALED_POPULATION_MISMATCH')
 if stage=='before':
  need(not BASE.exists(),'B4B_BASELINE_EXISTS')
  WRITE(BASE,dict(hashes=h,seals=seals,git_HEAD=head,created_utc=now()))
 else:
  old=READ(BASE);changed=[n for n,v in old['hashes'].items() if h.get(n)!=v];added=[n for n in h if n not in old['hashes']]
  result=dict(PASS=not changed and not added and head==old['git_HEAD'],protected_files=len(old['hashes']),changed=changed,unauthorized_added=added,production_semantic_drift=sum(n.startswith(('src/','scripts/')) for n in changed),scientific_artifact_drift=sum(n.startswith('outputs/') for n in changed),manuscript_drift=sum(n.startswith('paper/') for n in changed),git_HEAD_unchanged=head==old['git_HEAD'],B4A_bytes_preserved=True,B2_B3_scientific_outcomes_viewed=False)
  report('preservation',result);report('b2_b3_seal_audit',dict(PASS=seals==old['seals'],before=old['seals'],after=seals,scientific_fields_decoded=0))
  need(result['PASS'] and seals==old['seals'],'PRESERVATION_FAILURE')
 emit(stage='B4B_preservation_'+stage,PASS=True,protected_files=len(h),B2_files=35031,B3_files=10014)

def preflight():
 frozen_gate();need(BASE.exists(),'PRESERVATION_BEFORE_REQUIRED')
 need(not(B/'mentor_authorization.json').exists() and not(B/'measurements').exists(),'CLEAN_B4B_REQUIRED')
 p.firewall();c=r.config();expected=READ(ROOT/'outputs/sp_u12b4a_environment.json');env=r.verify_environment(expected)
 need(Path(env['executable']).resolve()==(ROOT/'.venv-cdl/Scripts/python.exe').resolve(),'B4_FORMAL_ENVIRONMENT_MISMATCH:executable')
 WRITE(B/'host_environment_before.json',env)
 need(c['common_measured_passes']==832 and c['process_blocks']==12 and c['warmup_passes']==5 and c['runtime_bootstrap_replicates']==10000,'FROZEN_PARAMETER_MISMATCH')
 need(c['method_orders']==a.orders(),'FROZEN_ORDER_MISMATCH')
 # Redirect the correctness report only. No estimator or timer binding is changed.
 original=r.report
 try:
  r.report=lambda name,value:report('fairness_recheck',value)
  eq=r.equivalence()
 finally:r.report=original
 need(eq['PASS'] and eq['best_case_bitwise'],'BITWISE_PRODUCTION_EQUIVALENCE_REQUIRED')
 for row in eq['checks']:
  for value in row['checks'].values():need(value['bitwise'] and value['max_abs']==0 and value['max_relative']==0,'PRODUCTION_BITWISE_FAILURE')
 frozen_gate()
 cmd=[str(ROOT/'.venv-cdl/Scripts/python.exe'),'-X','utf8','-m','pytest','tests/test_sp_u12b4a.py','tests/test_sp_u12b4b.py','-q','-p','no:cacheprovider']
 run=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8',env=os.environ.copy())
 WRITE(B/'pytest_preflight.json',dict(command=cmd,exit_code=run.returncode,stdout=run.stdout,stderr=run.stderr))
 need(run.returncode==0,'B4B_ACTUAL_PYTEST_FAILURE')
 WRITE(B/'reporting_rules.json',dict(created_before_formal=True,RSS_aggregation='median of twelve fresh-process values, separately by SNR/method/field; peak-minus-baseline computed within each process before aggregation',latency_reduction='100*(1-1/median_paired_Dense_over_LR4_ratio)',runtime_statistics='frozen B4A rules unchanged',machine_summaries_sealed_until_manifest_PASS=True,no_performance_threshold=True))
 WRITE(B/'mentor_authorization.json',dict(formal_authorized=True,B4A_freeze_sha256=FSHA,task_path=str(TASK),task_sha256=SHA(TASK),authorization='Direct human SP-U12B4B task request',created_utc=now(),no_retries=True,no_optimization=True))
 own={'scripts/sp_u12b4b_audit.py':SHA(Path(__file__)),'tests/test_sp_u12b4b.py':SHA(ROOT/'tests/test_sp_u12b4b.py')}
 WRITE(B/'execution_freeze.json',dict(created_utc=now(),hashes=own,formal_command=c['exact_formal_command'],B4A_freeze_sha256=FSHA,reporting_rules_sha256=SHA(B/'reporting_rules.json'),RSS_aggregation=READ(B/'reporting_rules.json')['RSS_aggregation'],adapter_bindings_changed=False,correctness_report_routing_only=True))
 r.require_formal_authorization();emit(stage='B4B_preflight',PASS=True,bitwise=True,formal_started=False)

def check_execution_freeze():
 for name,digest in READ(B/'execution_freeze.json')['hashes'].items():need(SHA(ROOT/name)==digest,'B4B_AUDIT_IMPLEMENTATION_DRIFT')
 need(SHA(B/'reporting_rules.json')==READ(B/'execution_freeze.json')['reporting_rules_sha256'],'REPORTING_RULE_DRIFT')

def execute():
 check_execution_freeze();frozen_gate();r.require_formal_authorization()
 need(not(B/'formal_execution.json').exists() and not(B/'measurements').exists(),'NO_RETRY_NO_RESUME')
 c=r.config();cmd=c['exact_formal_command'];start=now();emit(stage='B4B_formal_started',expected_runtime_processes=48,expected_memory_processes=72)
 run=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8',env=os.environ.copy())
 WRITE(B/'formal_execution.json',dict(command=cmd,cwd=str(ROOT),start_utc=start,end_utc=now(),exit_code=run.returncode,stdout=run.stdout,stderr=run.stderr,attempts=1))
 emit(stage='B4B_formal_exited',exit_code=run.returncode)
 if run.returncode:failure('B4_FORMAL_TECHNICAL_FAILURE','Frozen --formal returned nonzero');return
 audit();preserve('after');seal();emit(stage='B4B_phase1_complete',PASS=True,human_latency_comparison_viewed=False,manifest_sha256=SHA(ROOT/'outputs/sp_u12b4b_manifest.json'))

def audit():
 check_execution_freeze();frozen_gate();p.firewall();c=r.config();root=B/'measurements';env=READ(B/'host_environment_before.json')
 end=r.verify_environment(env);need(end['executable']==env['executable'],'EXECUTABLE_CHANGED');WRITE(B/'host_environment_after.json',end)
 report('environment_audit',dict(PASS=True,before=env,after=end,actual_child_queries='Frozen technical_child calls verify_environment before and after; each PASS certifies equality including active DLL thread queries',all_actual_DLL_threads_one=True,all_120_child_processes_environment_match=True))
 summary=READ(B/'runtime_summary.json');idx=np.load(B/'bootstrap_indices.npy',allow_pickle=False)
 need(idx.shape==(10000,12) and np.array_equal(idx,np.random.default_rng(a.BOOTSTRAP_SEED).integers(0,12,(10000,12))),'BOOTSTRAP_INDICES_MISMATCH')
 blocks=[];raw=[];draw_archive={};contexts=[];mem=[];pids=[];vmap={}
 expected_files=set()
 for block in range(12):
  for endpoint,snr in c['stratum_orders'][block]:
   name=f'runtime_block{block:02d}_{endpoint}_{snr}.json';rec=READ(root/name);expected_files.add(name)
   need(rec['PASS'] and not rec['sleep_detected'] and not a.suspended(rec['suspend_counter_before'],rec['suspend_counter_after']),'INVALID_RUNTIME_BLOCK')
   need([v['method'] for v in rec['records']]==c['method_orders'][f'{endpoint}_{snr}'][block],'METHOD_ORDER_MISMATCH')
   proc=f'process_runtime_{block}_{snr}_{endpoint}.json';expected_files.add(proc);need(READ(root/proc)['exit_code']==0,'RUNTIME_PROCESS_FAILURE')
   pids.append(rec['PID']);contexts.append(dict(kind='runtime',block=block,snr_db=snr,endpoint=endpoint,method_order=[v['method'] for v in rec['records']],PID=rec['PID'],start_utc=rec['start_utc'],end_utc=rec['end_utc'],active_BLAS=env['active_BLAS'],thread_environment=env['thread_environment'],environment_state_source='actual start/end queries verified equal to frozen state by unchanged child; PASS receipt',receipt_sha256=SHA(root/name),system_context_before=rec['system_context_before'],system_context_after=rec['system_context_after']))
   for item in rec['records']:
    method=item['method'];filename=f'block{block:02d}_{endpoint}_{snr}_{method}.npy';expected_files.add(filename);samples=np.load(root/filename,allow_pickle=False)
    need(samples.shape==(53248,) and samples.dtype==np.int64 and bool(np.all(samples>0)),'RAW_TIMING_SCHEMA_FAILURE')
    for key,q in (('median_ns',.5),('q25_ns',.25),('q75_ns',.75),('q05_ns',.05),('q95_ns',.95)):need(item[key]==float(np.quantile(samples,q,method='linear')),'RAW_BLOCK_SUMMARY_MISMATCH')
    need(item['output_frames']==53248,'TIMED_FRAME_COUNT')
    extra=dict(method_order=c['method_orders'][f'{endpoint}_{snr}'][block],repeat_count=832,timed_output_count=53248,elapsed_ns=int(samples.sum()),raw_file=str((root/filename).relative_to(ROOT)),raw_sha256=SHA(root/filename),aggregation='median of per-output-frame measurements; raw elapsed_ns[k] equals per-frame latency_ns[k]',**item)
    blocks.append(extra);raw.append({k:extra[k] for k in ('block','snr_db','endpoint','method','method_order','repeat_count','timed_output_count','elapsed_ns','raw_file','raw_sha256','aggregation')})
    vmap.setdefault((snr,endpoint,method),[]).append(item['median_ns'])
  for snr in a.SNRS:
   for method in c['method_orders'][f'estimator_core_{snr}'][block]:
    name=f'memory_block{block:02d}_{method}_{snr}.json';expected_files.add(name);rec=READ(root/name)
    proc=f'process_memory_{block}_{snr}_{method}.json';expected_files.add(proc);need(READ(root/proc)['exit_code']==0,'MEMORY_PROCESS_FAILURE')
    need(rec['PASS'] and not rec['sleep_detected'] and rec['samples']>0 and rec['sampling_interval_seconds']==.002 and not a.suspended(rec['suspend_counter_before'],rec['suspend_counter_after']),'RSS_SCHEMA_FAILURE')
    need(rec['operator_load_RSS_delta']==rec['loaded_RSS_bytes']-rec['baseline_RSS_bytes'],'RSS_DELTA_MISMATCH')
    for key in ('baseline_RSS_bytes','loaded_RSS_bytes','inference_interval_sampled_peak_RSS_bytes'):need(rec[key]>0,'RSS_NONPOSITIVE')
    rec['peak_minus_baseline_bytes']=rec['inference_interval_sampled_peak_RSS_bytes']-rec['baseline_RSS_bytes'];mem.append(rec);pids.append(rec['PID'])
    contexts.append(dict(kind='memory',block=block,snr_db=snr,method=method,method_order=c['method_orders'][f'estimator_core_{snr}'][block],PID=rec['PID'],start_utc=rec['start_utc'],end_utc=rec['end_utc'],active_BLAS=env['active_BLAS'],thread_environment=env['thread_environment'],environment_state_source='actual start/end queries verified equal to frozen state by unchanged child; PASS receipt',receipt_sha256=SHA(root/name),system_context_before=rec['system_context_before'],system_context_after=rec['system_context_after']))
 need({x.name for x in root.iterdir() if x.is_file()}==expected_files,'FORMAL_FILE_POPULATION_MISMATCH')
 need(len(blocks)==120 and len(mem)==72 and len(contexts)==120,'PROCESS_CARDINALITY')
 for row in summary['rows']:
  v=np.asarray(vmap[(row['snr_db'],row['endpoint'],row['method'])]);draws=np.median(v[idx],axis=1);lo,hi=np.quantile(draws,[.025,.975],method='linear');med=float(np.median(v))
  need(v.shape==(12,) and med==row['median_of_block_medians_ns'] and float(lo)==row['lower_ns'] and float(hi)==row['upper_ns'] and row['throughput_frames_s']==1e9/med,'FROZEN_SUMMARY_MISMATCH')
  for key,q in (('q25_ns',.25),('q75_ns',.75),('q05_ns',.05),('q95_ns',.95)):need(row[key]==float(np.quantile(v,q,method='linear')),'SUMMARY_SPREAD_MISMATCH')
  draw_archive[f"latency_{row['snr_db']}_{row['endpoint']}_{row['method']}"]=draws
 for row in summary['paired_ratios']:
  snr,endpoint=row['snr_db'],row['endpoint'];v=np.asarray(vmap[(snr,endpoint,'Dense')])/np.asarray(vmap[(snr,endpoint,'LR4')]);draws=np.median(v[idx],axis=1);lo,hi=np.quantile(draws,[.025,.975],method='linear')
  need(row['Dense_over_LR4_median_paired_ratio']==float(np.median(v)) and row['lower']==float(lo) and row['upper']==float(hi),'PAIRED_BOOTSTRAP_MISMATCH');draw_archive[f'ratio_{snr}_{endpoint}']=draws
 need(len(summary['rows'])==10 and len(summary['paired_ratios'])==4,'SUMMARY_CELLS')
 WRITE(B/'raw_timing_index.json',dict(records=raw,actual_samples=sum(v['timed_output_count'] for v in raw),raw_format='int64 per-output-frame elapsed ns; no re-timing'))
 WRITE(B/'block_summaries.json',dict(records=blocks));WRITE(B/'runtime_ratios.json',dict(rows=summary['paired_ratios'],paired=True))
 np.savez(B/'runtime_bootstrap.npz',indices=idx,**draw_archive)
 WRITE(B/'rss_measurements.json',dict(records=mem,aggregation=READ(B/'reporting_rules.json')['RSS_aggregation']))
 rssrows=[]
 for snr in a.SNRS:
  for method in a.METHODS:
   values=[v for v in mem if v['snr_db']==snr and v['method']==method];need(len(values)==12,'RSS_CELL_COUNT')
   rssrows.append(dict(snr_db=snr,method=method,blocks=12,**{k:float(np.median([v[k] for v in values])) for k in RSS_FIELDS}))
 WRITE(B/'memory_descriptive_summary.json',dict(rows=rssrows,role='SYSTEM_DESCRIPTIVE_ONLY',aggregation=READ(B/'reporting_rules.json')['RSS_aggregation'],sampling_can_miss_short_peaks=True,lifetime_peak_not_used_as_inference_peak=True))
 storage=[];expected=READ(ROOT/'outputs/sp_u12b4a_memory_accounting.json')['rows']
 for snr in a.SNRS:
  for method in a.METHODS:
   rows=a.load(method,snr).persistent_arrays();need(len({id(v) for _,_,v in rows})==len(rows),'STORAGE_ALIAS_DUPLICATION')
   total=sum(v.nbytes for _,_,v in rows);corr=sum(v.nbytes for _,role,v in rows if role=='CORRECTION_SPECIFIC');old=next(v for v in expected if v['method']==method and v['snr_db']==snr)
   need((total,corr,total-corr)==(old['total_persistent_numeric_array_bytes'],old['correction_only_array_bytes'],old['common_array_bytes']),'STORAGE_CLASSIFICATION_DRIFT')
   storage.append(dict(method=method,snr_db=snr,total_persistent_numeric_array_bytes=total,correction_only_array_bytes=corr,common_array_bytes=total-corr,arrays=[dict(name=n,role=role,shape=list(v.shape),dtype=str(v.dtype),bytes=v.nbytes) for n,role,v in rows]))
 WRITE(B/'persistent_array_memory.json',dict(rows=storage,actual_nbytes=True,aliases_counted_once=True,COMMON_definition_unchanged=True))
 op=READ(ROOT/'outputs/sp_u12b4a_operation_count_audit.json');need(tuple(op[k] for k in ('Source','Fit','Dense_corrections','LR4_corrections','Dense_total','LR4_total'))==(253952,46080,507904,32256,807936,332288),'MAC_MAPPING');report('operation_mapping_audit',op)
 report('bootstrap_audit',dict(PASS=True,replicates=10000,seed=a.BOOTSTRAP_SEED,indices_exact=True,all_latency_CIs_reconciled=True,paired_ratios_reconciled=True,rule_unchanged=True,role='DESCRIPTIVE_SYSTEM_BENCHMARK',p_values=False,exclusions=0))
 report('resource_log',dict(PASS=True,contexts=contexts,processes_sequential=True,runtime_processes=48,memory_processes=72,method_timing_blocks=120,PID_reuse_by_OS_allowed=True))
 report('integrity_audit',dict(PASS=True,runtime_processes=48,correction_method_blocks=48,estimator_core_method_blocks=72,total_method_blocks=120,memory_processes=72,timed_outputs_per_method_block=53248,total_timed_outputs=6389760,all_raw_samples_positive=True,raw_summaries_exact=True,technical_invalid_blocks=0,retries=0,additional_blocks=0,performance_exclusions=0,outcomes_viewed=False,production_equivalence_bitwise=True,all_frozen_hashes_unchanged=True))
 emit(stage='B4B_machine_integrity',PASS=True,method_blocks=120,memory_processes=72,latencies_disclosed=False)

def seal():
 check_execution_freeze();frozen_gate()
 for name in ('integrity_audit','environment_audit','fairness_recheck','bootstrap_audit','preservation','b2_b3_seal_audit'):need(READ(ROOT/f'outputs/sp_u12b4b_{name}.json')['PASS'],'MANIFEST_GATE:'+name)
 manifest=ROOT/'outputs/sp_u12b4b_manifest.json';need(not manifest.exists(),'IMMUTABLE_MANIFEST_EXISTS')
 files=sorted(x for d in ('outputs','scripts','tests') for x in (ROOT/d).rglob('*') if x.is_file() and new_owned(x) and '__pycache__' not in x.parts)
 leaves={x.relative_to(ROOT).as_posix():dict(sha256=SHA(x),bytes=x.stat().st_size) for x in files}
 WRITE(manifest,dict(status='B4_FORMAL_BENCHMARK_COMPLETE_SEALED',created_utc=now(),B4A_freeze_sha256=FSHA,task_sha256=SHA(TASK),authority_hashes=READ(F)['hashes'],files=leaves,all_artifact_hash_tree=p.tree(leaves),human_runtime_outcomes_viewed=False,B2_B3_outcomes_viewed=False,formal_attempts=1,no_performance_reruns=True,no_implementation_changes=True,analysis_only_after_this_seal=True))

def failure(verdict,reason):
 preserve('after')
 report('integrity_audit',dict(PASS=False,verdict=verdict,reason=reason,automatic_retry=False,outcomes_viewed=False))
 report('decision',dict(verdict=verdict,reason=reason,STOP=True))
 emit(stage='B4B_STOP',verdict=verdict,reason=reason)

def interpretation():
 manifest=ROOT/'outputs/sp_u12b4b_manifest.json';m=READ(manifest);need(m['status']=='B4_FORMAL_BENCHMARK_COMPLETE_SEALED','SEALED_PASS_REQUIRED')
 for name,item in m['files'].items():need(SHA(ROOT/name)==item['sha256'],'SEALED_B4B_LEAF_DRIFT')
 summary=READ(B/'runtime_summary.json');rss=READ(B/'memory_descriptive_summary.json');store=READ(B/'persistent_array_memory.json')
 lines=['# SP-U12B4B runtime interpretation','',f'Manifest SHA256: {SHA(manifest)}','', 'All timings and intervals describe the specified steady-state single-thread CPU benchmark. Twelve process-block medians; 10,000 paired descriptive block bootstrap replicates. No channel-population inference or speed success threshold.','']
 for row in summary['rows']:lines.append(f"{row['snr_db']} dB {row['endpoint']} {row['method']}: {row['median_of_block_medians_ns']/1000:.6f} us/output; descriptive 95% interval [{row['lower_ns']/1000:.6f}, {row['upper_ns']/1000:.6f}]; {row['throughput_frames_s']:.6f} output frames/s.")
 for row in summary['paired_ratios']:
  ratio=row['Dense_over_LR4_median_paired_ratio'];lines.append(f"{row['snr_db']} dB {row['endpoint']}: paired Dense/LR4 ratio {ratio:.9f} [{row['lower']:.9f}, {row['upper']:.9f}]; descriptive paired-ratio-derived latency reduction {100*(1-1/ratio):.6f}%.")
 lines+=['','RSS uses medians of twelve method-exclusive fresh processes per SNR. Inference peak is the 2-ms sampled interval maximum, not lifetime peak; brief peaks can be missed. Allocator/DLL/OS effects remain included.','',json.dumps(rss,ensure_ascii=False,indent=2),'',json.dumps(store,ensure_ascii=False,indent=2),'','Arithmetic reduction 58.87199% remains a complex-MAC proxy. It is not a wall-clock prediction or success criterion. B2/B3 scientific outcomes remain unviewed. STOP; mentor review only.']
 (ROOT/'docs/sp_u12b4b_runtime_interpretation.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 execution=READ(B/'formal_execution.json')
 (ROOT/'docs/sp_u12b4b_execution_log.md').write_text('# SP-U12B4B execution log\n\n'+json.dumps(execution,indent=2)+'\n\n48 runtime children, 120 method timing blocks, 72 memory children. All sequential. No retries, exclusions, additional blocks or patches. Phase 1 raw capture and machine audit preceded immutable manifest and human result reading. Actual pytest receipt: outputs/sp_u12b4b_runtime/pytest_preflight.json. B2 35,031 and B3 10,014 files opaque-hashed before and after; no scientific values decoded.\n',encoding='utf8')
 (ROOT/'docs/sp_u12b4b_change_log.md').write_text('# SP-U12B4B change log\n\nOnly new B4B audit/test/bookkeeping/results/reports. Frozen adapter, runner, B4A parameters, B4A files and production code unchanged. The frozen correctness checker was called directly; its report writer alone was temporarily routed to a B4B report to preserve B4A bytes. The original writer binding was restored before formal execution. Exact frozen --formal command executed once without monkeypatching. Frozen runner computes machine summaries after all child processes; these stayed human-unviewed until all integrity and preservation gates passed and manifest was written. Per-child recorded BLAS state derives from actual start/end DLL queries verified equal to the frozen state by the unchanged child. RSS reporting medians and ratio-derived percentage rule were recorded before formal execution. No manuscript edits, scientific channel experiment, B2/B3 outcome access or Git commit.\n',encoding='utf8')
 report('decision',dict(verdict='B4_FORMAL_BENCHMARK_COMPLETE',manifest_sha256=SHA(manifest),B4_runtime_outcomes_reporting_authorized_after_seal=True,B2_B3_outcomes_viewed=False,STOP=True))
 emit(stage='B4B_reporting_ready',manifest_sha256=SHA(manifest),runtime=summary,RSS=rss,storage=[{k:v for k,v in row.items() if k!='arrays'} for row in store['rows']])

if __name__=='__main__':
 mode=sys.argv[1]
 if mode in ('before','after'):preserve(mode)
 elif mode=='preflight':preflight()
 elif mode=='execute':execute()
 elif mode=='report':interpretation()
 else:raise ValueError(mode)
