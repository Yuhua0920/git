"""Fairness/boundary/regression gates; no formal timing or scientific generation."""
import inspect,ast,subprocess,json,collections
import numpy as np
import pytest
from scripts import sp_u12b4a_adapter as a
from scripts import run_sp_u12b4_benchmark as r
from scripts.sp_u12b4a_preserve import ROOT,OUT,read,sha,B3M,B3SHA

def test_actual_exact_preflight_and_native_RSS_receipts():
 assert read(ROOT/'outputs/sp_u12b4a_entrypoint_preflight.json')['PASS']
 p=read(OUT/'rss_api_probe.json');assert p['PASS'] and p['current_RSS_available'] and not p['method_RSS_comparison_generated']
 assert read(OUT/'sleep_counter_probe.json')['PASS']

def test_authoritative_seals_and_input_isolation():
 assert sha(B3M)==B3SHA
 base=read(OUT/'preservation_before.json');assert base['seals']['B2']['files']==35031 and base['seals']['B3']['files']==10014
 inp=read(OUT/'benchmark_safe/input_provenance.json');assert inp['PASS'] and inp['formal_input_decodes']==0 and inp['seeds_disjoint_from_B2_B3']
 assert sha(a.STREAM)==inp['input_sha256']
 assert a.STREAM.is_relative_to(OUT/'benchmark_safe')

@pytest.mark.parametrize('snr',a.SNRS)
def test_production_output_equivalence_all_steady_state_frames(snr):
 eq=read(ROOT/'outputs/sp_u12b4a_production_equivalence.json');assert eq['PASS'] and not eq['formal_inputs_used']
 row=next(x for x in eq['checks'] if x['snr_db']==snr);assert row['output_frames']==64
 assert set(row['checks'])=={'Source','Dense','LR4','coordinates','correction_Dense','correction_LR4'}
 for v in row['checks'].values():assert v['max_relative']<=eq['tolerance']

@pytest.mark.parametrize('method',a.METHODS)
@pytest.mark.parametrize('snr',a.SNRS)
def test_selective_deployment_and_persistent_accounting(method,snr):
 obj=a.load(method,snr);assert set(obj.a)==set(a.keys(method,snr))
 assert all(v.dtype==np.complex128 and not v.flags.writeable for v in obj.a.values())
 rows=obj.persistent_arrays();assert len({id(v) for _,_,v in rows})==len(rows)
 report=read(ROOT/'outputs/sp_u12b4a_memory_accounting.json')
 line=next(v for v in report['rows'] if v['method']==method and v['snr_db']==snr)
 assert line['total_persistent_numeric_array_bytes']==sum(v.nbytes for _,_,v in rows)
 if method=='Source':assert not hasattr(obj,'buffer') and not hasattr(obj,'S')
 if method=='Dense':assert not any(k.startswith(('A_','Q_')) for k in obj.a)
 if method=='LR4':assert not any(k.startswith('DeltaW') for k in obj.a)

def test_shared_fit_source_and_state_paths():
 dense=a.load('Dense',10);low=a.load('LR4',10)
 for name in ('step','update','reset','fit','source'):assert getattr(type(dense),name) is getattr(type(low),name)
 data=a.stream_arrays(10)
 for j,p in enumerate(data['p']):
  dense.step(p);low.step(p)
  assert np.array_equal(dense.buffer,low.buffer)
  if j>=4:assert np.array_equal(dense.S,low.S) and np.array_equal(dense.b,low.b) and np.array_equal(dense.c,low.c)
 dense.reset();assert dense.fill==0 and not dense.buffer.any()

def test_correction_inputs_shared_and_readonly():
 data=a.stream_arrays(20)
 assert not any(v.flags.writeable for v in data.values())
 assert data['p'].shape==data['u'].shape==(68,64) and data['c'].shape==(68,2)
 dense=a.load('Dense',20);low=a.load('LR4',20)
 assert np.array_equal(data['u'],data['p']-dense.mean_p)
 # Same argument objects; no coordinate fitting inside correction methods.
 source=inspect.getsource(a.Estimator.correction)
 assert 'fit(' not in source and 'load(' not in source
 assert dense.correction(data['u'][4],data['c'][4]).shape==low.correction(data['u'][4],data['c'][4]).shape==(3968,)

def test_singular_fit_stops_without_fallback():
 obj=a.load('Dense',10)
 for _ in range(4):assert obj.step(obj.mean_p) is None
 with pytest.raises(np.linalg.LinAlgError):obj.step(obj.mean_p)

def test_clock_brackets_only_one_computation_and_common_frame_count(monkeypatch):
 class Dummy:
  def __init__(self):self.steps=[];self.corrections=[];self.resets=0
  def reset(self):self.resets+=1
  def step(self,p):self.steps.append(int(p[0]));return 0
  def correction(self,u,c):self.corrections.append((int(u[0]),int(c[0])));return 0
 data=dict(p=np.arange(68)[:,None],u=np.arange(68)[:,None],c=np.arange(68)[:,None])
 counter=iter(range(0,10000,10));monkeypatch.setattr(a.time,'perf_counter_ns',lambda:next(counter))
 obj=Dummy();samples=a.measure(obj,data,'estimator_core',2)
 assert samples.shape==(128,) and np.all(samples==10)
 assert obj.steps==list(range(68))*2 and obj.resets==2
 obj=Dummy();samples=a.measure(obj,data,'correction_only',2)
 assert obj.corrections==[(j,j) for j in range(4,68)]*2 and not obj.steps

def test_nonmonotonic_timer_is_technical_failure(monkeypatch):
 class Dummy:
  def correction(self,*x):return None
 data=dict(p=np.zeros((5,1)),u=np.zeros((5,1)),c=np.zeros((5,1)))
 ticks=iter((10,5));monkeypatch.setattr(a.time,'perf_counter_ns',lambda:next(ticks))
 with pytest.raises(RuntimeError,match='TIMER_MONOTONICITY_FAILURE'):a.measure(Dummy(),data,'correction_only',1)

def test_timed_code_has_no_I_O_offline_metric_or_factor_reconstruction():
 source=inspect.getsource(a.Estimator)+inspect.getsource(a.measure)
 for forbidden in ('svd(','cholesky(','pinv(','clip(','open(','np.load(','np.save','print(','error_energy','BER','NMSE','linalg.cond','@self.QP@'):
  assert forbidden not in source
 assert 'base+c[0]*self.product_P(u)+c[1]*self.product_M(u)' in source
 assert 'np.linalg.solve(self.S,self.b)' in source

def test_balanced_orders_and_process_repetition_unit():
 c=r.config();assert c['process_blocks']==12 and c['processes_concurrent']==1
 for key,rows in c['method_orders'].items():
  assert len(rows)==12
  n=3 if key.startswith('estimator_core') else 2
  for pos in range(n):assert set(collections.Counter(row[pos] for row in rows).values())=={12//n}
 assert c['runtime_statistics']=='DESCRIPTIVE_SYSTEM_BENCHMARK'
 assert c['runtime_bootstrap_replicates']==10000
 assert c['no_hypothesis_test'] and c['no_pvalue'] and c['no_Bonferroni'] and c['no_speed_success_threshold']

def test_b1_operation_count_mapping():
 o=read(ROOT/'outputs/sp_u12b4a_operation_count_audit.json');assert o['PASS']
 assert (o['Source'],o['Fit'],o['Dense_corrections'],o['LR4_corrections'],o['Dense_total'],o['LR4_total'])==(253952,46080,507904,32256,807936,332288)
 assert o['no_MAC_redefinition']

def test_actual_BLAS_threads_and_native_process_memory():
 assert all(v['threads']==1 for v in a.blas_threads())
 memory=a.rss();assert memory['RSS_bytes']>0 and memory['lifetime_peak_RSS_bytes']>=memory['RSS_bytes']
 assert a.suspend_clock()['including_sleep_100ns']>0

def test_sleep_detection_and_no_latency_outlier_filter():
 start=dict(including_sleep_100ns=0,excluding_sleep_100ns=0)
 assert not a.suspended(start,dict(including_sleep_100ns=10_000_000,excluding_sleep_100ns=10_000_000))
 assert a.suspended(start,dict(including_sleep_100ns=30_000_000,excluding_sleep_100ns=10_000_000))
 c=r.config();assert 'large latency' in c['never_invalidate_for']
 assert 'retain' in c['invalidation_action'] and 'no auto' in c['invalidation_action']

def test_formal_authorization_denied_without_executing_formal_mode():
 with pytest.raises(RuntimeError,match='B4B_FORMAL_AUTHORIZATION_REQUIRED'):r.require_formal_authorization()
 assert not(a.B4B/'measurements').exists()
 assert not(a.B4B/'runtime_summary.json').exists()
 assert not(a.B4B/'memory_summary.json').exists()
 assert not(a.B4B/'bootstrap_indices.npy').exists()

def test_outcome_read_firewall_blocks_before_decode_in_fresh_process():
 code="from scripts.sp_u12b4a_preserve import firewall,B3;firewall();p=next((B3/'raw').rglob('*.npz'));p.open('rb')"
 run=subprocess.run([str(ROOT/'.venv-cdl/Scripts/python.exe'),'-X','utf8','-c',code],cwd=ROOT,capture_output=True,text=True,encoding='utf8')
 assert run.returncode!=0 and 'B4_SCIENTIFIC_OUTCOME_FIREWALL' in run.stderr

def test_actual_calibration_receipt_if_available():
 p=ROOT/'outputs/sp_u12b4a_timing_calibration.json'
 if not p.exists():pytest.skip('One-time technical calibration not executed yet;final gate requires this check')
 o=read(p);c=r.config();assert o['PASS'] and o['all_blocks_above_resolution_target']
 assert o['common_repeat_count']==c['common_measured_passes']>0 and o['common_repeat_count']%16==0
 assert o['calibrated_method_SNR_endpoints']==10 and o['calibration_attempts']==1
 assert o['formal_blocks_executed']==o['formal_runtime_bootstrap']==o['formal_comparison_generated']==0
 assert not o['latencies_disclosed'] and not o['rankings_computed']
 assert o['private_calibration_sha256']==sha(OUT/'calibration_private.json')
