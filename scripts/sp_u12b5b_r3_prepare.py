"""Freeze outcome-blind synthetic benchmark, tolerances, and portable bundle."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sp_u12b5b_r3_common import *
import random,shutil,ast
PORT=ROOT/'experiments/mfa_low_memory_prototype/portable'
def prepare():
 firewall();OUT.mkdir(exist_ok=True)
 import numpy,scipy
 sys.path.append(str(ROOT/'.venv-gmm/Lib/site-packages'))
 import sklearn
 sys.path.insert(0,str(PORT));from host_inventory import inventory
 host=inventory()
 if host['CPU_exact_model']=='13th Gen Intel(R) Core(TM) i7-13650HX':
  host['manufacturer_specification']={'source':'https://www.intel.com/content/www/us/en/products/sku/232101/intel-core-i713650hx-processor-24m-cache-up-to-4-90-ghz/specifications.html','max_turbo_GHz':4.9,'P_core_max_GHz':4.9,'E_core_max_GHz':3.6,'P_cores':6,'E_cores':8,'base_frequency':'not listed on inspected official specification page','actual_sustained_boost_verified':False}
 write(ROOT/'outputs/sp_u12b5b_r3_host_inventory.json',host)
 physical=host['topology']['physical_cores'];assert isinstance(physical,int) and physical>=1
 threads=sorted(set([n for n in (1,2,4,8,12,physical) if n<=physical]))
 from kernels import specifications
 cases=specifications();seed=4237658801;order_seed=4237658802
 cfgold=read(ROOT/'configs/sp_u12b5b_mfa_selection.yaml')
 formal={row[x] for row in cfgold['candidate_order'] for x in ('seed','final_seed')}
 used={seed,order_seed,4237658803}|{seed+100+i for i in range(len(cases))}|{4237659001+i for i in range(10)}|{4237659011+i for i in range(10)}
 assert not used&formal
 schedule=[];rng=random.Random(order_seed)
 for block in range(5):
  ordering=threads.copy();rng.shuffle(ordering);caseorder=list(range(len(cases)));rng.shuffle(caseorder)
  for t in ordering:schedule.append({'job':len(schedule),'block':block,'threads':t,'case_order':caseorder})
 cfg={'task':'SP-U12B5B-R3','role':'SYNTHETIC_TECHNICAL_ONLY','DATA_SEED':seed,'THREAD_ORDER_SEED':order_seed,'INITIALIZATION_SEED':4237658803,
 'thread_candidates':threads,'physical_core_count':physical,'logical_thread_count':host['topology']['logical_threads'],'schedule':schedule,'measurement_blocks':5,'warmups_per_kernel_per_block':1,'timer':'perf_counter_ns','cases':cases,
 'parity_tolerance':{'atol':1e-10,'rtol':1e-9,'source':'scripts/sp_u12b5a_cases.py:ATOL,RTOL; frozen before R3 tests'},
 'resource_limits':LIMITS,'sampling_seconds':.05,'worker_timeout_seconds':1800,'benchmark_process_concurrency':1,'formal_candidate_fits':0,
 'formal_seed_collision':False,'no_formal_payloads':True,'stable_measurement_rule':'five-block max/min<=2.0 for dominant projected kernels; near-tie if <=max(5%, projected relative interquartile spread), choose fewer threads',
 'parity_policy':'nonfinite, any tolerance failure, differing stop iterations/status excludes thread or prototype; no tolerance relaxation',
 'frame_inference_counts':{'task_declared_validation':16200,'source_derived_validation':9*3*2*100*6,'formal_evaluation':210000,'status':'COUNT_MISMATCH; report both validation budgets, do not amend science'}}
 write(ROOT/'outputs/sp_u12b5b_r3_thread_benchmark_config.json',cfg)
 copies={
 ROOT/'src/baselines/mfa_reference_adapter.py':PORT/'src/baselines/mfa_reference_adapter.py',
 CORE:PORT/'reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py',
 CORE.parent/'utils.py':PORT/'reference/sp_u12b5a/paper_period/cplx-mfa/utils.py',
 ROOT/'reference/sp_u12b5a/paper_period/MFA_estimator/MFA_estimator.py':PORT/'reference/sp_u12b5a/paper_period/MFA_estimator/MFA_estimator.py',
 ROOT/'outputs/sp_u12b5a/historical_reference.json':PORT/'outputs/sp_u12b5a/historical_reference.json'}
 hashes={}
 for source,target in copies.items():
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes());assert sha(source)==sha(target)
  hashes[target.relative_to(PORT).as_posix()]=sha(target)
 write(PORT/'portable_reference_hashes.json',{'copied_bytes_unchanged':True,'files':hashes,'reference_only_no_scientific_data':True})
 # Exact existing B5A analytical fixture functions, without its unrelated tasks.
 source=(ROOT/'scripts/sp_u12b5a_cases.py').read_text(encoding='utf8');parsed=ast.parse(source)
 selected=[ast.get_source_segment(source,node) for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name in ('cnormal','known_model','dense_posterior','dense_em_step')]
 assert len(selected)==4
 header='"""Exact B5A toy analytical functions; no project channel data."""\nimport numpy as np\nfrom scipy.special import logsumexp\nfrom kernels import adapter\nmake_reference_model=adapter().make_reference_model\nDATA_SEED=4190123500\nATOL=1e-10\nRTOL=1e-9\n'
 (PORT/'toy_case_reference.py').write_text(header+'\n\n'.join(selected)+'\n',encoding='utf8')
 shutil.copyfile(ROOT/'outputs/sp_u12b5b_r3_thread_benchmark_config.json',PORT/'current_host_config.json')
 (PORT/'requirements.txt').write_text('numpy==2.5.3\nscipy==1.18.1\nscikit-learn==1.9.1\nthreadpoolctl==3.7.0\npsutil>=7; platform_system != "Windows"\n',encoding='utf8')
 print(json.dumps({'stage':'R3_prepared','CPU':host['CPU_exact_model'],'physical_cores':physical,'logical_threads':host['topology']['logical_threads'],'RAM_bytes':host['memory']['total_physical_bytes'],'threads':threads,'jobs':len(schedule)}),flush=True)
if __name__=='__main__':prepare()
