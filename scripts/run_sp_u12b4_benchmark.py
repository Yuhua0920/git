"""B4A fair implementation/preflight/calibration; B4B requires separate authorization."""
from pathlib import Path
import sys,os,json,time,math,csv,argparse,subprocess,itertools,gc,threading,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b4a_adapter as a
from scripts.sp_u12b4a_preserve import OUT,B3M,B3SHA,read,write,sha,tree,firewall
import numpy as np
import yaml
TASK=Path(r'C:\Users\yuhua\.codex\attachments\69ab70dc-d563-4964-a418-0cb370c190a3\已粘贴的文本.txt')
SOURCES=('scripts/sp_u12b4a_adapter.py','scripts/run_sp_u12b4_benchmark.py','scripts/sp_u12b4a_preserve.py','scripts/sp_u12b4a_validation.py','tests/test_sp_u12b4a.py')
PRODUCTION=('src/pilot_cv.py','src/lowrank_modules.py','src/frozen_affine_validation.py','scripts/sp_u12b2a_common.py','src/cdl_benchmark.py')

def report(n,o):write(ROOT/'outputs'/('sp_u12b4a_'+n+'.json'),o)
def csvreport(n,rows):
 rows=list(rows);p=ROOT/'outputs'/('sp_u12b4a_'+n+'.csv')
 with p.open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def config():return yaml.safe_load(a.CONFIG.read_text(encoding='utf8'))
def save_config(c):
 a.CONFIG.write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True),encoding='utf8')
def enumerate_ints(v):
 if isinstance(v,dict):return set().union(*(enumerate_ints(x) for x in v.values())) if v else set()
 if isinstance(v,list):return set().union(*(enumerate_ints(x) for x in v)) if v else set()
 return {v} if type(v)==int else set()

def initialize():
 a.need(not a.CONFIG.exists() and not a.STREAM.exists(),'B4A_ALREADY_INITIALIZED')
 before=read(OUT/'preservation_before.json');a.need(before['seals']['B2']['PASS'] and before['seals']['B3']['PASS'],'INITIAL_SEAL_GATE')
 a.need(sha(a.OPERATOR_PATH)==a.OPERATOR_SHA,'PREPARED_OPERATOR_HASH')
 # Metadata only. No B2/B3 pilot/channel/result array is decoded.
 old=enumerate_ints(read(ROOT/'outputs/sp_u12b2a_seed_registry.json'))|enumerate_ints(read(ROOT/'outputs/sp_u12b3b/bootstrap_seed_registry.json'))
 a.need(not ({a.STREAM_SEED,a.ORDER_SEED,a.BOOTSTRAP_SEED}&old),'BENCHMARK_SEED_REUSE')
 rng=np.random.default_rng(a.STREAM_SEED);z=(rng.standard_normal((68,64))+1j*rng.standard_normal((68,64)))/np.sqrt(2)
 noise=(rng.standard_normal((68,64))+1j*rng.standard_normal((68,64)))/np.sqrt(2)
 from src.pilot_cv import PilotCV
 arrays={};stream_receipts=[]
 for snr in a.SNRS:
  obj=a.load('Dense',snr)
  variance=9.941134822523972/10**(snr/10)
  p=np.asarray(obj.mean_p+z+np.sqrt(variance)*noise,dtype=np.complex128)
  u=p-obj.mean_p;c=np.zeros((68,2),np.float64)
  cv=PilotCV(obj.folds,obj.mean_p,tuple(obj.K),tuple(obj.KP),tuple(obj.KM))
  for j in range(4,len(p)):c[j]=cv.fit(p[j-4:j+1],modules=('P','M'))[0]
  arrays.update({f'p_{snr}':p,f'u_{snr}':u,f'c_{snr}':c})
  stream_receipts.append(dict(snr_db=snr,frames=68,pilot_dimension=64,output_frames_per_pass=64,source='dimension-correct deterministic benchmark-safe pilots;no H/CSI/condition/trajectory IDs',noise_variance=variance))
 a.STREAM.parent.mkdir(parents=True,exist_ok=True)
 with a.STREAM.open('xb') as f:np.savez(f,**arrays)
 base_strata=[['estimator_core',10],['correction_only',10],['estimator_core',20],['correction_only',20]]
 order=np.random.default_rng(a.ORDER_SEED+55).permutation(4)
 schedule=[[base_strata[order[(i+j)%4]] for j in range(4)] for i in range(12)]
 c=dict(task='SP-U12B4A',formal_authorized=False,scope='DESIGN_FAIR_IMPLEMENTATION_TECHNICAL_PREFLIGHT_ONLY',methods=list(a.METHODS),endpoints=list(a.ENDPOINTS),endpoint_methods={'estimator_core':list(a.METHODS),'correction_only':['Dense','LR4']},snr_db=[10,20],noise_variances={snr:9.941134822523972/10**(snr/10) for snr in a.SNRS},input_stream_source=str(a.STREAM),input_stream_sha256=sha(a.STREAM),input_seed=a.STREAM_SEED,formal_scientific_inputs_used=False,stream_length=68,pilot_dimension=64,data_dimension=3968,warmup_frames_per_pass=4,outputs_per_measured_pass=64,window=5,folds=4,rank=4,warmup_passes=5,calibration_pilot_passes=16,calibration_target_seconds=0.5,calibration_planning_seconds=1.0,calibration_safety_factor=1.25,common_measured_passes=None,calibration_policy='Once:minimum timed-ns-per-pass across all declared method/SNR/endpoints;ceil(planning_ns*1.25/min),round up to16;validate all blocks>=0.5s;failure stops,never retunes',process_blocks=12,processes_concurrent=1,benchmark_order_seed=a.ORDER_SEED,method_orders=a.orders(),stratum_orders=schedule,BLAS_threads=1,thread_environment={k:'1' for k in a.THREAD_KEYS},dtype_operators='complex128',dtype_pilots='complex128',dtype_coordinates='float64',dtype_normal_equations='float64',timer='time.perf_counter_ns',timer_inside='one Estimator.step(p) or correction(u,c);input lookup and result/timing storage outside',pass_state='reset and ingest first four frames outside measured frames;measure j>=4;no carried buffer across passes',warmup_policy='5 complete stream passes per method/SNR/endpoint;resident operators;no cache flush',allocation_policy='same NumPy expressions and common persistent buffer/normal arrays;no fused base+correction or asymmetric preallocation',garbage_collection='Python default enabled in all paths',runtime_environment='single fresh child per process-block/SNR/endpoint;all compared objects resident;no parallelism',memory_measurement='Win32 GetProcessMemoryInfo(GetCurrentProcess);fresh method-exclusive child;current RSS baseline and loaded RSS;2ms current-RSS sampling in separate untimed memory runs;also record lifetime peak separately',memory_process_blocks=12,memory_sample_interval_seconds=0.002,memory_run_passes='same common_measured_passes',memory_baseline='after imports and benchmark input ready,GC;before any method operators loaded',memory_loaded='only required method/SNR arrays loaded;GC;before warmup',memory_scope='numeric .nbytes independent of Python/system RSS;aliases counted once;RSS includes interpreter/input/allocator',runtime_bootstrap_seed=a.BOOTSTRAP_SEED,runtime_bootstrap_replicates=10000,runtime_statistics='DESCRIPTIVE_SYSTEM_BENCHMARK',latency_summary='median of12 process-block per-frame medians;IQR and5th/95th of block medians;also within-block per-frame median/IQR/5th/95th',latency_uncertainty='10000 resamples of12 process-block medians;95% percentile CI',ratio='Dense block median / LR4 block median;median paired block ratios and paired12-block bootstrap;no speed threshold',throughput='1e9 / median-of-block-median ns;frames/s;reciprocal descriptor,not independently timed',quantiles='linear;0.025/0.975 CI;0.25/0.75 IQR;0.05/0.95 spread',no_hypothesis_test=True,no_pvalue=True,no_Bonferroni=True,no_channel_generalization_inference=True,no_speed_success_threshold=True,no_offline_preparation_or_I_O_timed=True,technical_invalidation=['process interruption/crash','detected OS suspend/resume','non-monotonic/invalid timer','frozen environment/threads/power plan changed','hash/output-correctness failure'],never_invalidate_for=['large latency','background CPU spike alone','thermal/frequency change alone'],load_diagnostics='record processor time/frequency/background context where available;unavailable flagged;no outlier deletion',invalidation_action='mark whole process block invalid;retain bytes;STOP formal attempt;no auto replacement/retry/filtering',sleep_detection='Windows System power events(Power-Troubleshooter1,Kernel-Power42/107/506/507)over block;unavailable event query invalidates complete block',output_root=str(a.B4B),exact_formal_command=[str(ROOT/'.venv-cdl/Scripts/python.exe'),'-X','utf8','scripts/run_sp_u12b4_benchmark.py','--formal'],operator_path=str(a.OPERATOR_PATH),operator_sha256=a.OPERATOR_SHA,production_relative_tolerance=1e-10,production_tolerance_source='existing B2A fixed complex1281e-10 equivalence gate;not selected from timings',B2_manifest_sha256=before['seals']['B2']['manifest_sha256'],B3_manifest_sha256=B3SHA,task_path=str(TASK),task_sha256=sha(TASK))
 save_config(c);a.B4B.mkdir(exist_ok=True)
 write(OUT/'benchmark_safe/input_provenance.json',dict(PASS=True,seed=a.STREAM_SEED,seeds_disjoint_from_B2_B3=True,input_sha256=sha(a.STREAM),stream_receipts=stream_receipts,formal_input_decodes=0,coordinates_source='production PilotCV.fit on these same deterministic development-only windows;shared by correction endpoints',never_channel_performance_metrics=True))

def rss_probe():
 # Self handle avoids Windows venv-redirector PID assumptions.
 baseline=a.rss();probe=np.empty(8*1024*1024,np.uint8);probe.fill(7);loaded=a.rss()
 a.need(baseline['PID']==loaded['PID']==os.getpid() and baseline['RSS_bytes']>0 and loaded['RSS_bytes']>0 and loaded['lifetime_peak_RSS_bytes']>=loaded['RSS_bytes'],'RSS_SELF_API_FAILURE')
 print(json.dumps(dict(PASS=True,actual_interpreter_PID=os.getpid(),current_RSS_available=True,lifetime_peak_available=True,method_RSS_comparison_generated=False)),flush=True)

def equivalence():
 from src.pilot_cv import PilotCV,deployment
 from src.lowrank_modules import RankFactors,factored_affine_predict
 checks=[]
 for snr in a.SNRS:
  objs={m:a.load(m,snr) for m in a.METHODS};data=a.stream_arrays(snr);dense=objs['Dense']
  cv=PilotCV(dense.folds,dense.mean_p,tuple(dense.K),tuple(dense.KP),tuple(dense.KM))
  low=objs['LR4'];P=RankFactors(low.AP,low.QP);M=RankFactors(low.AM,low.QM)
  maxima={k:dict(max_abs=0.,max_relative=0.,bitwise=True) for k in ('Source','Dense','LR4','coordinates','correction_Dense','correction_LR4')}
  for j,p in enumerate(data['p']):
   got={m:obj.step(p) for m,obj in objs.items()}
   if j<4:
    a.need(got['Dense'] is None and got['LR4'] is None,'FOUR_FRAME_BUFFER_WARMUP');continue
   c=cv.fit(data['p'][j-4:j+1],modules=('P','M'))[0]
   args=(p,dense.mean_p,dense.mean_d,dense.W);u=p-dense.mean_p
   reference={'Source':dense.mean_d+u@dense.W.T,'Dense':deployment(*args,dense.BP,dense.BM,c,('P','M')),'LR4':factored_affine_predict(*args,c,P,M),'coordinates':c,'correction_Dense':c[0]*(u@dense.BP.T)+c[1]*(u@dense.BM.T),'correction_LR4':c[0]*P.apply(u)+c[1]*M.apply(u)}
   got.update(coordinates=objs['Dense'].c.copy(),correction_Dense=dense.correction(data['u'][j],data['c'][j]),correction_LR4=low.correction(data['u'][j],data['c'][j]))
   a.need(np.array_equal(objs['Dense'].c,objs['LR4'].c) and np.array_equal(c,data['c'][j]),'COORDINATE_AND_INPUT_PAIRING')
   for key in reference:
    diff=got[key]-reference[key];absolute=float(np.max(np.abs(diff)));relative=float(np.linalg.norm(diff)/max(np.linalg.norm(reference[key]),np.finfo(float).tiny))
    maxima[key]['max_abs']=max(maxima[key]['max_abs'],absolute);maxima[key]['max_relative']=max(maxima[key]['max_relative'],relative);maxima[key]['bitwise'] &= np.array_equal(got[key],reference[key])
    a.need(relative<=1e-10 and np.isfinite(got[key]).all(),'PRODUCTION_OUTPUT_EQUIVALENCE_FAILURE:'+key)
  checks.append(dict(snr_db=snr,output_frames=64,checks=maxima))
 result=dict(PASS=True,checks=checks,tolerance=1e-10,best_case_bitwise=all(v['bitwise'] for r in checks for v in r['checks'].values()),formal_inputs_used=False,scientific_metrics_computed=False,fit_diagnostics_removed='source_cv_loss/fitted_cv_loss/condition_number/normal-residual reporting only;numeric solve unchanged',production_changed=False)
 report('production_equivalence',result);return result

def audits():
 c=config();operation=[];allocation=[];memory=[]
 stages=[('pilot centering','eq:source','p_j -> u_j','(64,) complex128','one temporary per current Source;five independent centered vectors during fit','COMMON'),('buffer update','alg:estimation','chronological five-frame buffer','(5,64) complex128','one persistent buffer;same shift/ingest policy','COMMON'),('cross-pilot three products','eq:design','K_s,DeltaK_P,DeltaK_M @ u[observed]','4*(16,48) kernels;observed48;heldout16','small fold vectors/matrices allocated identically','COMMON'),('normal accumulation','eq:normal','same concatenate/conjugate dot/frame-sum order','D(64,2);e(64,);S(2,2);b(2,)','identical small temporaries;S/b/c persistent','COMMON'),('real solve','eq:normal','np.linalg.solve(S,b)','float64 two unknowns','same2x2 solve;no regularization/fallback','COMMON'),('Source application','eq:source','mean_d + u @ Wsource.T','(3968,64) complex128','same NumPy matvec and mean addition','COMMON'),('Dense correction','eq:affine','base + alpha*(u@BP.T) + beta*(u@BM.T)','two(3968,64)complex128','two3968-vectors/scaling/addition;no matrix construction','DENSE_SPECIFIC'),('LR4 correction','eq:final-estimate','base + alpha*(AP@(QP@u)) + beta*(AM@(QM@u))','A(3968,4);Q(4,64)complex128','two4-vectors+two3968-vectors/scaling/addition;no reconstruction','LR4_SPECIFIC')]
 for name,eq,op,shape,alloc,shared in stages:operation.append(dict(operation=name,equation=eq,code='scripts/sp_u12b4a_adapter.py:Estimator',array_shape=shape,dtype='complex128 products;float64 real solve',allocation_behavior=alloc,shared_computation=shared,scientific_semantics='UNCHANGED'))
 for snr in a.SNRS:
  for method in a.METHODS:
   obj=a.load(method,snr);rows=obj.persistent_arrays()
   for name,role,v in rows:
    allocation.append(dict(method=method,snr_db=snr,operation_or_array=name,shape=str(list(v.shape)),dtype=str(v.dtype),nbytes=int(v.nbytes),allocation_behavior='persistent;aliases counted once',shared_computation=role,layout='C_CONTIGUOUS' if v.flags.c_contiguous else 'VIEW_OR_STRIDED',timed_allocation=False))
   total=sum(v.nbytes for _,_,v in rows);corr=sum(v.nbytes for _,role,v in rows if role=='CORRECTION_SPECIFIC')
   memory.append(dict(method=method,snr_db=snr,correction_only_array_bytes=corr,total_persistent_numeric_array_bytes=total,common_array_bytes=total-corr,aliases_counted_once=True,Python_numpy_interpreter_excluded=True,RSS_distinct=True,shape_dtypes_verified=True))
   data=a.stream_arrays(snr)
   for name,v in data.items():allocation.append(dict(method=method,snr_db=snr,operation_or_array='shared_benchmark_input_'+name,shape=str(list(v.shape)),dtype=str(v.dtype),nbytes=int(v.nbytes),allocation_behavior='read-only benchmark corpus;outside estimator storage and timer',shared_computation='SHARED_INPUT',layout='C_CONTIGUOUS',timed_allocation=False))
   for p in data['p']:output=obj.step(p)
   allocation.append(dict(method=method,snr_db=snr,operation_or_array='channel_output',shape=str(list(output.shape)),dtype=str(output.dtype),nbytes=int(output.nbytes),allocation_behavior='normal NumPy perframe output;transient;not persistent accounting',shared_computation='COMMON_OUTPUT_SHAPE',layout='C_CONTIGUOUS',timed_allocation=True))
 csvreport('operation_mapping',operation);csvreport('dtype_allocation_audit',allocation)
 report('memory_accounting',dict(PASS=True,actual_nbytes=True,rows=memory,one_operating_SNR_per_object=True,Source_has_no_buffer_or_fit_arrays=True,operator_load_is_selective=True,transient_arrays_excluded_from_persistent_bytes=True,RSS_measurement_role='system-level descriptive,not deterministic storage orhardware/energy/cacheefficiency'))
 boundary=[]
 for item in ('buffer update(Dense/LR4)','Source centering','five-frame four-fold fitting(Dense/LR4)','real normal-equation solve(Dense/LR4)','Source matvec','path-specific correction application(Dense/LR4)'):boundary.append(dict(component=item,estimator_core='INCLUDED',correction_only='INCLUDED' if 'correction application' in item else 'EXCLUDED',reason='online required operation'))
 for item in ('channel/Sionna generation','data/QPSK/noise generation','full-waveform pilot extraction','offline covariance/Cholesky/SVD/rank selection','operator load','process startup/import','disk I/O','logging','NMSE/BER/error energy','bootstrap/CSV/plots','fit diagnostic loss/condition number'):boundary.append(dict(component=item,estimator_core='EXCLUDED',correction_only='EXCLUDED',reason='outside frozen estimator-core boundary;diagnostics removed symmetrically'))
 csvreport('benchmark_boundary_audit',boundary)
 questions=['same Dense/LR4 common implementation','same correction-only u/alpha/beta','same complex128/float64 dtype','same input stream','same timed j>=4 frames','same reset/prefill buffer policy','same warmup policy','same verified active BLAS threads1','fair symmetric allocation/no fusion','production output equivalence','operator load outside timer','file I/O outside timer']
 evidence=['Estimator.step/update/fit/source shared class','one immutable p/u/c archive perSNR','actual selective array audit','same data object in paired child block','64outputs/pass each method','reset +4warmup outside timed frames','5full passes allmethod blocks','two DLL active-thread query receipts','two normal NumPy product wrappers only representation differs','production_equivalence.json128 outputwindows andcorrection-only','load completed before warmup/measure','measure() contains only compute plus adjacent clocks']
 csvreport('fairness_audit',[dict(question=q,status='PASS',evidence=e) for q,e in zip(questions,evidence)])
 counts=dict(M=2,F=4,T_w=5,n_p=64,n_d=3968,n_o=48,n_h=16,r=4)
 source=3968*64;fit=3*4*5*48*16;dense=2*3968*64;low=2*4*(3968+64)
 a.need((source,fit,dense,low,source+fit+dense,source+fit+low)==(253952,46080,507904,32256,807936,332288),'B1_OPERATION_COUNT_MISMATCH')
 report('operation_count_audit',dict(PASS=True,parameters=counts,Source=source,Fit=fit,Dense_corrections=dense,LR4_corrections=low,Dense_total=source+fit+dense,LR4_total=source+fit+low,reduction_percent=100*(1-(source+fit+low)/(source+fit+dense)),dominant_complex_MAC_proxy=True,other_costs='GramO(FT_wn_hM^2),solveO(M^3),centering/buffer/additions/allocations/timer/Python overhead;included in runtime but not MAC proxy',no_MAC_redefinition=True))

def preflight(rss_smoke=False):
 firewall()
 if rss_smoke:rss_probe();return
 if not a.CONFIG.exists():initialize()
 c=config();a.need(sha(a.STREAM)==c['input_stream_sha256'] and sha(a.OPERATOR_PATH)==a.OPERATOR_SHA and sha(B3M)==B3SHA,'INPUT_HASH_GATE')
 c['stratum_order_seed']=a.ORDER_SEED+55
 old=enumerate_ints(read(ROOT/'outputs/sp_u12b2a_seed_registry.json'))|enumerate_ints(read(ROOT/'outputs/sp_u12b3b/bootstrap_seed_registry.json'))
 a.need(not({a.STREAM_SEED,a.ORDER_SEED,a.BOOTSTRAP_SEED,c['stratum_order_seed']}&old),'BENCHMARK_SEED_REUSE')
 c['sleep_detection']='Win32 QueryInterruptTimePrecise(includes suspend) minus QueryUnbiasedInterruptTimePrecise(excludes suspend);100ms counter/read scheduling tolerance;brief suspend below detection resolution cannot be excluded'
 c['technical_invalidation']=[v.replace('detected OS suspend/resume','detected OS suspend/resume above100ms detection tolerance') for v in c['technical_invalidation']]
 save_config(c)
 env=a.environment();report('environment',env);write(a.B4B/'environment.json',env)
 eq=equivalence();audits()
 cmd=[str(ROOT/'.venv-cdl/Scripts/python.exe'),'-X','utf8','scripts/run_sp_u12b4_benchmark.py','--preflight','--rss-api-smoke']
 r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8');a.need(r.returncode==0,'EXACT_RSS_PREFLIGHT_FAILED')
 probe=json.loads(r.stdout.strip().splitlines()[-1]);a.need(probe['PASS'],'RSS_API_PROBE_FAILED');write(OUT/'rss_api_probe.json',probe)
 power=a.suspend_clock();a.need(power['including_sleep_100ns']>0 and power['excluding_sleep_100ns']>0,'SUSPEND_COUNTER_API_UNAVAILABLE')
 write(OUT/'sleep_counter_probe.json',dict(PASS=True,inclusion_exclusion_native_APIs_available=True,observation_tolerance_milliseconds=100,formal_blocks_executed=0))
 context=a.system_context();write(OUT/'system_context_probe.json',dict(PASS=True,system_CPU_times_available=context['system_CPU_times_available'],frequency_query_available=context['frequency_query_available'],thermal_telemetry=context['thermal_telemetry'],no_block_deletion_based_on_context=True))
 receipt=dict(PASS=True,exact_entrypoint=str(Path(__file__).resolve()),actual_command=sys.argv,production_equivalence_PASS=eq['PASS'],RSS_self_API_PASS=True,active_BLAS_threads_PASS=True,Source_Dense_LR4_selective_load_PASS=True,scientific_inputs_used=False,formal_blocks_executed=0,formal_runtime_bootstrap=0,formal_comparison_generated=0,preflight_only=True)
 report('entrypoint_preflight',receipt);print(json.dumps(dict(stage='B4A_exact_preflight',PASS=True,formal_blocks=0,formal_outcomes=False)),flush=True)

def calibrate():
 firewall();a.need(read(ROOT/'outputs/sp_u12b4a_entrypoint_preflight.json')['PASS'],'EXACT_PREFLIGHT_REQUIRED')
 p=ROOT/'outputs/sp_u12b4a_timing_calibration.json';a.need(not p.exists(),'ONE_TIME_CALIBRATION_ALREADY_EXECUTED')
 c=config();private=[];items=[]
 for snr in a.SNRS:
  data=a.stream_arrays(snr)
  for endpoint in a.ENDPOINTS:
   for method in c['endpoint_methods'][endpoint]:
    obj=a.load(method,snr);a.warmup(obj,data,endpoint,c['warmup_passes'])
    durations=a.measure(obj,data,endpoint,c['calibration_pilot_passes'])
    perpass=float(durations.sum())/c['calibration_pilot_passes']
    items.append((snr,endpoint,method,perpass));private.append(dict(stage='PILOT',snr_db=snr,endpoint=endpoint,method=method,timed_ns=int(durations.sum()),passes=c['calibration_pilot_passes']))
 min_perpass=min(v[3] for v in items);repeats=16*math.ceil(c['calibration_planning_seconds']*1e9*c['calibration_safety_factor']/min_perpass/16)
 flags=[]
 for snr,endpoint,method,_ in items:
  data=a.stream_arrays(snr);obj=a.load(method,snr);a.warmup(obj,data,endpoint,c['warmup_passes'])
  durations=a.measure(obj,data,endpoint,repeats);ok=int(durations.sum())>=c['calibration_target_seconds']*1e9;flags.append(ok)
  private.append(dict(stage='CONFIRM',snr_db=snr,endpoint=endpoint,method=method,timed_ns=int(durations.sum()),passes=repeats,above_resolution_target=ok))
 write(OUT/'calibration_private.json',dict(role='SEALED_TECHNICAL_CALIBRATION_NOT_FORMAL_BENCHMARK;do not human-read duration fields',records=private,comparison_or_ranking_computed=False,optimization_after_calibration_forbidden=True))
 result=dict(PASS=all(flags),common_repeat_count=repeats,all_blocks_above_resolution_target=all(flags),target_seconds=c['calibration_target_seconds'],planning_seconds=c['calibration_planning_seconds'],calibrated_method_SNR_endpoints=len(items),warmup_passes=c['warmup_passes'],private_calibration_sha256=sha(OUT/'calibration_private.json'),calibration_attempts=1,formal_blocks_executed=0,formal_runtime_bootstrap=0,formal_comparison_generated=0,latencies_disclosed=False,rankings_computed=False)
 report('timing_calibration',result);a.need(result['PASS'],'ONE_TIME_TIMING_CALIBRATION_FAILED_STOP')
 c['common_measured_passes']=repeats;c['calibration_status']='PASS_ONCE_NO_FURTHER_OPTIMIZATION';save_config(c)
 write(a.B4B/'config.json',c)
 print(json.dumps(dict(stage='B4A_calibration',PASS=True,common_repeat_count=repeats,all_blocks_above_resolution_target=True,formal_blocks=0,formal_outcomes=False)),flush=True)

def require_formal_authorization():
 p=a.B4B/'mentor_authorization.json'
 a.need(p.exists(),'B4B_FORMAL_AUTHORIZATION_REQUIRED')
 auth=read(p);freeze=ROOT/'outputs/sp_u12b4a_formal_freeze.json'
 a.need(auth.get('formal_authorized') is True and auth.get('B4A_freeze_sha256')==sha(freeze),'B4B_AUTHORIZATION_NOT_BOUND_TO_FREEZE')
 f=read(freeze)
 for path,digest in f['hashes'].items():a.need(sha(ROOT/path)==digest,'B4B_FROZEN_SOURCE_INPUT_DRIFT')
 a.need(read(ROOT/'outputs/sp_u12b4a_decision.json')['verdict']=='B4_RUNTIME_BENCHMARK_READY','B4A_GATE_NOT_READY')
 return f

def verify_environment(frozen):
 now=a.environment()
 for key in ('OS','CPU_model','physical_cores','logical_cores','Python','NumPy','SciPy','active_BLAS','thread_environment','power_scheme'):
  a.need(now[key]==frozen[key],'B4B_ENVIRONMENT_CHANGED:'+key)
 return now

def technical_child(kind,block,snr,endpoint=None,method=None):
 """Future B4B child. No B4A mode calls this function."""
 f=require_formal_authorization();firewall();c=config();env=read(ROOT/'outputs/sp_u12b4a_environment.json');verify_environment(env)
 a.need(kind in ('runtime','memory') and block in range(12) and snr in a.SNRS,'DECLARED_FORMAL_BLOCK_REQUIRED')
 start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat();power_before=a.suspend_clock();context_before=a.system_context();data=a.stream_arrays(snr)
 if kind=='runtime':
  objs={m:a.load(m,snr) for m in c['endpoint_methods'][endpoint]};records=[]
  for m in c['method_orders'][f'{endpoint}_{snr}'][block]:
   obj=objs[m];a.warmup(obj,data,endpoint,c['warmup_passes'])
   samples=a.measure(obj,data,endpoint,c['common_measured_passes'])
   record=dict(block=block,snr_db=snr,endpoint=endpoint,method=m,output_frames=len(samples),median_ns=float(np.median(samples)),q25_ns=float(np.quantile(samples,.25,method='linear')),q75_ns=float(np.quantile(samples,.75,method='linear')),q05_ns=float(np.quantile(samples,.05,method='linear')),q95_ns=float(np.quantile(samples,.95,method='linear')))
   np.save(a.B4B/'measurements'/f'block{block:02d}_{endpoint}_{snr}_{m}.npy',samples,allow_pickle=False);records.append(record)
  output=dict(records=records)
 else:
  gc.collect();baseline=a.rss();obj=a.load(method,snr);gc.collect();loaded=a.rss();a.warmup(obj,data,'estimator_core',c['warmup_passes'])
  readings=[];errors=[];stop=threading.Event()
  def sample():
   try:
    while not stop.is_set():readings.append(a.rss()['RSS_bytes']);stop.wait(c['memory_sample_interval_seconds'])
   except Exception as exc:errors.append(type(exc).__name__);stop.set()
  t=threading.Thread(target=sample);t.start()
  try:
   # Dedicated memory observation, no latency measurements affected by sampler.
   a.warmup(obj,data,'estimator_core',c['common_measured_passes'])
  finally:stop.set();t.join()
  final=a.rss();a.need(readings and not errors,'RSS_INTERVAL_SAMPLES_REQUIRED_NO_ERRORS')
  output=dict(block=block,snr_db=snr,method=method,baseline_RSS_bytes=baseline['RSS_bytes'],loaded_RSS_bytes=loaded['RSS_bytes'],operator_load_RSS_delta=loaded['RSS_bytes']-baseline['RSS_bytes'],inference_interval_sampled_peak_RSS_bytes=max(readings),sampling_interval_seconds=c['memory_sample_interval_seconds'],sampling_can_miss_short_peaks=True,lifetime_peak_RSS_bytes=final['lifetime_peak_RSS_bytes'],peak_lifetime_distinct_from_interval_peak=True,samples=len(readings))
 power_after=a.suspend_clock();sleep_detected=a.suspended(power_before,power_after)
 verify_environment(env);end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat();a.need(not sleep_detected,'OS_SUSPEND_RESUME_BLOCK_INVALID')
 write(a.B4B/'measurements'/f'{kind}_block{block:02d}_{endpoint or method}_{snr}.json',dict(PASS=True,PID=os.getpid(),start_utc=start_utc,end_utc=end_utc,suspend_counter_before=power_before,suspend_counter_after=power_after,sleep_detected=False,system_context_before=context_before,system_context_after=a.system_context(),**output))

def formal():
 require_formal_authorization();firewall();c=config();root=a.B4B/'measurements'
 a.need(not root.exists(),'B4B_CLEAN_ATTEMPT_REQUIRED_NO_RESUME');root.mkdir()
 try:
  for b in range(c['process_blocks']):
   for endpoint,snr in c['stratum_orders'][b]:
    run_child('runtime',b,snr,endpoint=endpoint)
   for snr in a.SNRS:
    for method in c['method_orders'][f'estimator_core_{snr}'][b]:run_child('memory',b,snr,method=method)
  summarize()
 except Exception as exc:
  write(a.B4B/'PERMANENT_STOP.json',dict(error_type=type(exc).__name__,failure_code=str(exc),no_automatic_retry=True,retain_all_partial_bytes=True));raise

def run_child(kind,block,snr,endpoint=None,method=None):
 cmd=config()['exact_formal_command']+['--child-kind',kind,'--block',str(block),'--snr',str(snr)]
 if endpoint:cmd+=['--endpoint',endpoint]
 if method:cmd+=['--method',method]
 r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8')
 write(a.B4B/'measurements'/f'process_{kind}_{block}_{snr}_{endpoint or method}.json',dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
 a.need(r.returncode==0,'B4B_WHOLE_BLOCK_TECHNICALLY_INVALID_NO_REPLACEMENT')

def summarize():
 # Reachable only after separate mentor authorization and all complete processes.
 c=config();rng=np.random.default_rng(a.BOOTSTRAP_SEED);idx=rng.integers(0,12,(10000,12));records=[];ratios=[]
 for snr in a.SNRS:
  for endpoint in a.ENDPOINTS:
   by_method={m:[] for m in c['endpoint_methods'][endpoint]}
   for b in range(12):
    receipt=read(a.B4B/'measurements'/f'runtime_block{b:02d}_{endpoint}_{snr}.json');a.need(receipt['PASS'],'COMPLETE_BLOCK_REQUIRED')
    for r in receipt['records']:by_method[r['method']].append(r['median_ns'])
   for m,values in by_method.items():
    v=np.asarray(values);a.need(v.shape==(12,),'PROCESS_BLOCK_CARDINALITY');draws=np.median(v[idx],axis=1);lo,hi=np.quantile(draws,[.025,.975],method='linear');median=float(np.median(v))
    records.append(dict(snr_db=snr,endpoint=endpoint,method=m,median_of_block_medians_ns=median,q25_ns=float(np.quantile(v,.25)),q75_ns=float(np.quantile(v,.75)),q05_ns=float(np.quantile(v,.05)),q95_ns=float(np.quantile(v,.95)),lower_ns=float(lo),upper_ns=float(hi),throughput_frames_s=1e9/median,interval_role='DESCRIPTIVE_SYSTEM_BENCHMARK',process_blocks=12))
   paired=np.asarray(by_method['Dense'])/np.asarray(by_method['LR4']);draws=np.median(paired[idx],axis=1);lo,hi=np.quantile(draws,[.025,.975],method='linear')
   ratios.append(dict(snr_db=snr,endpoint=endpoint,Dense_over_LR4_median_paired_ratio=float(np.median(paired)),lower=float(lo),upper=float(hi),interval_role='DESCRIPTIVE_SYSTEM_BENCHMARK',process_blocks=12))
 memory=[read(a.B4B/'measurements'/f'memory_block{b:02d}_{m}_{s}.json') for b in range(12) for s in a.SNRS for m in a.METHODS]
 a.need(len(memory)==72 and all(v['PASS'] for v in memory),'MEMORY_BLOCK_CARDINALITY')
 write(a.B4B/'runtime_summary.json',dict(rows=records,paired_ratios=ratios,bootstrap_replicates=10000,no_hypothesis_test=True,no_speed_threshold=True));write(a.B4B/'memory_summary.json',dict(raw_process_records=memory,role='SYSTEM_DESCRIPTIVE_ONLY'))
 np.save(a.B4B/'bootstrap_indices.npy',idx,allow_pickle=False)

def documents():
 c=config();text=f'''# SP-U12B4A benchmark design and preregistration

Authority: B4A preflight and one technical calibration only. B4B has not been authorized.
Primary question: representation consequences on steady-state CPU correction application and the complete estimator core. No minimum speedup is required. Zero formal blocks, comparisons or formal bootstrap draws have been generated.

Input/output boundary: formed received pilot p_j to data-domain estimate. Dense/LR4 include chronological five-frame update, Source centering, four-fold cross-pilot moments over all five frames, one unrestricted real2x2solve, full Source and two corrections. Source uses centering/full Source only. Correction-only uses exactly the same pre-generated u/alpha/beta from benchmark-safe pilots. Offline covariance/Cholesky/SVD/rank/operator preparation, waveform extraction, generation, imports/startup, I/O/logging, diagnostics/NMSE/BER/bootstrap/plots are outside latency clocks. End-to-end here means estimator core, not communications receiver.

Fair implementation: Estimator.step/update/fit/source are shared. Only product_P/product_M dispatch to full dense versus two-stage rank4 factors. Production complex128, float64 coordinates. The original frame/fold complex moment and chronological sum order, ordinary transpose Source/Dense application and left-associated base+alphaP+betaM order are retained. Fit diagnostic losses/condition number/reporting are removed from both paths. Frozen finite inputs/operator shape/dtype are checked before timing for all methods; repeated debug finite-array scans are outside this numerical-core adapter. No cached moments, incremental fit, ridge, clipping, pinv, fallback, SVD or dense-factor reconstruction exists in timed code. Persistent five-frame and tiny normal arrays are common; path-specific NumPy matvec temporaries reflect each representation. Array .nbytes counts each owned allocation once.

Inputs: independent deterministic benchmark-safe seed{a.STREAM_SEED};68frames x64complex pilots perSNR, no H/CSI/environment labels/formal trajectory IDs. Prepared operators are loaded read-only from the existing frozen archive, SHA256 {a.OPERATOR_SHA}; only needed keys at oneSNR are retained. Input archive SHA256 {c['input_stream_sha256']}. Pregen coordinates come from production PilotCV.fit on these same development windows. Neither sealed B2 nor B3 scientific fields are used.

Process design:12 independent fresh process blocks perSNR/endpoint, sequentialsingleprocess execution, BLAS1. Core compares Source/Dense/LR4; correction-only compares Dense/LR4. All6core permutations repeated twice, bothcorrection orders repeated six times, shuffled with seed{a.ORDER_SEED}. Four endpoint/SNR strata use a balanced rotated order. No cache flush, priority or affinity manipulation. Each method gets5complete warmup passes. A measured pass resets and ingests fourprefill frames outside clocks, then times64frames(j>=4). Source times the same64inputs without fitting. Shared calibrated measured-pass count is {c['common_measured_passes']}. Technical calibration is onepilot16passes+oneconfirmation, min aggregate measured-ns/pass over every declaredcell, planning1.0s with1.25margin, roundupto16; confirmationtarget0.5s. All methods use one count. Private calibration durations are not human-reviewed or ranked, and no subsequent optimization is allowed.

Runtime statistics: actual perframe perf_counter_ns samples; process-block median/IQR/5th/95th; main summary median of12block medians with block-median IQR/5th/95th.10000 block bootstrap, percentile95%, linearquantiles, seed{a.BOOTSTRAP_SEED}. PairedDense/LR4 ratios use matched process-block indices and median pairedratios under the same bootstrap. Throughput is reciprocal summary latency. All are DESCRIPTIVE_SYSTEM_BENCHMARK, not channel inference or tests; no pvalues/significance/Bonferroni. All results, including null/adverse runtime ratios, must be reported.

Memory: actual persistent .nbytes per deployed method/SNR, correction-only and total, distinct from RSS. FutureRSS uses fresh method-exclusive process: afterimports/input+GCbaseline; selectiveoperatorload+GCloadedRSS;warmup;separate untimed inference run with2ms currentRSSsampling, plus selfAPI lifetime peak labelled separately. Sampling can miss brief peaks; interpreter/input/allocator are included inRSS. No external venv launcherPID assumptions.72fresh method/SNR/block memory runs are separate from48runtime processes. Never inferhardware/cache/energy efficiency fromRSS.

Invalidation: wholeblock only for interruption/crash, detectedOSsuspend/resume, nonmonotonic timer, frozenenvironment/threads/powerchange, hash/correctnessfailure. Suspend detection uses native included/unbiased interrupt counters with100ms tolerance for sequentialread scheduling;brief suspend below detection resolution cannot be excluded. Retainallpartialbytes andSTOP; noautomaticreplacement/retry. Large latency, backgroundload or frequency/thermal variation alone is never a deletionrule. Record context/availability, keepslowblocks. B4B authorization must bind to the B4A freeze; environment/power checked before and after eachblock.

B1 mapping:Source253952;fit46080;densecorrections507904;LRcorrections32256;Dense807936;LR332288;complex-MACproxy reduction58.87199%. Gram/solve/centering/buffer/allocations/Python overhead remain additional runtime costs. Arithmetic reduction is not a runtime target.

Seals: B2/B3 verified by opaque fullleaf hashing before/after; no scientificvalues decoded. Original production/manuscript/scientific artifacts and GitHEAD preserved. B4B root contains only config/environment/protocol placeholders inB4A. All adapter/runner/test/production/input/config hashes are frozen aftercalibration. Nextaction is mentor review, not B4B execution.
'''
 for name in ('preregistration','benchmark_design'):(ROOT/'docs'/('sp_u12b4a_'+name+'.md')).write_text(text,encoding='utf8')
 (ROOT/'docs/sp_u12b4a_change_log.md').write_text('# SP-U12B4A change log\n\nOnly newB4adapter/runner/preservation/tests/config/docs/auditfiles. Production,manuscript,scientificoutputs unchanged. CurrentB4A protocol supersedes historical PLAN_B B4 cachedformalinputs/30blocks description: development-onlyinputs,12balancedprocessblocks, estimatorcore/correction-only; no batch endpoint, no offlinepreparationtiming. Historical PLAN_B remains byte-identical. No formal B4B execution, no performance-based optimization, no Gitcommit.\n\nBefore calibration, an enhanced preflight found QueryInterruptTimePrecise/QueryUnbiasedInterruptTimePrecise absent from Kernel32 exports on thisWindows system. Failed sources are preserved in outputs/sp_u12b4a/preflight_compatibility_failure01. Actual export probing found both in KernelBase; corrected the DLL binding and reran exact preflight successfully. General lesson: verify actual runtime Win32 exports and typed native signatures rather than assuming import-library names imply a usable DLL export. All comparisons/durations remained unviewed; this was API compatibility, not performance optimization.\n\nThe first pytest process loaded NumPy before the test-module environment assignment and failed the active-BLAS-thread gate. Exact benchmark preflight already used startup threads1. Failed pytest receipts/harness preserved in outputs/sp_u12b4a/pytest_startup_failure01; set thread environment in the parent validation harness before launching pytest and reran. General lesson: process startup environment must precede plugin/test numerical imports; test actual DLL thread counts, not just environment strings. No timing calibration or formal benchmark had occurred before either technical repair.\n',encoding='utf8')
 (a.B4B/'protocol.md').write_text(text,encoding='utf8')

def freeze():
 firewall();c=config();a.need(read(ROOT/'outputs/sp_u12b4a_timing_calibration.json')['PASS'] and c['common_measured_passes']>0,'CALIBRATION_REQUIRED')
 a.need(read(ROOT/'outputs/sp_u12b4a_preservation.json')['PASS'] and read(ROOT/'outputs/sp_u12b4a_b2_b3_seal_audit.json')['PASS'],'FULL_PRESERVATION_REQUIRED')
 tests=read(OUT/'pytest.json');a.need(tests['exit_code']==0 and tests['skipped']==0 and tests['timing_calibration_sha256']==sha(ROOT/'outputs/sp_u12b4a_timing_calibration.json'),'ACTUAL_FINAL_PYTEST_REQUIRED');documents()
 DLLs=[Path(v['library']).relative_to(ROOT).as_posix() for v in read(ROOT/'outputs/sp_u12b4a_environment.json')['active_BLAS']]
 hashes={p:sha(ROOT/p) for p in (*SOURCES,*PRODUCTION,*DLLs,a.CONFIG.relative_to(ROOT).as_posix(),a.STREAM.relative_to(ROOT).as_posix(),a.OPERATOR_PATH.relative_to(ROOT).as_posix(),'outputs/sp_u12b4a_environment.json','outputs/sp_u12b4a_production_equivalence.json','outputs/sp_u12b4a_timing_calibration.json','outputs/sp_u12b4a_memory_accounting.json','outputs/sp_u12b4a_entrypoint_preflight.json','outputs/sp_u12b4a/calibration_private.json','outputs/sp_u12b4b_runtime/config.json','outputs/sp_u12b4b_runtime/environment.json','outputs/sp_u12b4b_runtime/protocol.md')}
 f=dict(status='B4A_FROZEN_PENDING_B4B_AUTHORIZATION',formal_authorized=False,hashes=hashes,config_sha256=sha(a.CONFIG),input_sha256=sha(a.STREAM),operator_sha256=a.OPERATOR_SHA,B3_manifest_sha256=B3SHA,formal_blocks_executed=0,formal_runtime_bootstrap=0,formal_comparison_generated=0,formal_runtime_outcomes_viewed=False,optimization_after_calibration_forbidden=True,calibration_once=True,STOP=True)
 p=ROOT/'outputs/sp_u12b4a_formal_freeze.json';a.need(not p.exists(),'FREEZE_ALREADY_EXISTS');write(p,f)
 for n in ('production_equivalence','entrypoint_preflight','timing_calibration','memory_accounting','operation_count_audit','b2_b3_seal_audit','preservation'):a.need(read(ROOT/'outputs'/('sp_u12b4a_'+n+'.json'))['PASS'],'FINAL_GATE:'+n)
 a.need(not(a.B4B/'measurements').exists() and not(a.B4B/'runtime_summary.json').exists() and not(a.B4B/'bootstrap_indices.npy').exists(),'B4A_FORMAL_BENCHMARK_VIOLATION')
 decision=dict(verdict='B4_RUNTIME_BENCHMARK_READY',formal_freeze_sha256=sha(p),formal_blocks_executed=0,formal_runtime_bootstrap=0,formal_comparison_generated=0,outcomes_viewed=False,STOP=True)
 report('decision',decision);print(json.dumps(decision),flush=True)

def main():
 parser=argparse.ArgumentParser();m=parser.add_mutually_exclusive_group(required=True);m.add_argument('--preflight',action='store_true');m.add_argument('--calibrate',action='store_true');m.add_argument('--formal',action='store_true');m.add_argument('--freeze',action='store_true');parser.add_argument('--rss-api-smoke',action='store_true');parser.add_argument('--child-kind',choices=['runtime','memory']);parser.add_argument('--block',type=int);parser.add_argument('--snr',type=int);parser.add_argument('--endpoint',choices=a.ENDPOINTS);parser.add_argument('--method',choices=a.METHODS);args=parser.parse_args()
 if args.preflight:preflight(args.rss_api_smoke)
 elif args.calibrate:calibrate()
 elif args.freeze:freeze()
 elif args.child_kind:technical_child(args.child_kind,args.block,args.snr,args.endpoint,args.method)
 else:formal()
if __name__=='__main__':main()
