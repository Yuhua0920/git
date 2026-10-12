"""R3 safety, full-batch identities, frozen residual order and thread controls."""
from pathlib import Path
import sys,os
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,scipy
sys.path.append(str(ROOT/'.venv-gmm/Lib/site-packages'))
PORT=ROOT/'experiments/mfa_low_memory_prototype/portable';sys.path.insert(0,str(PORT))
from scripts.sp_u12b5b_r3_common import sha,check_frozen,firewall,OUT,owned,LIMITS,read
from low_memory import residual_moment
from parity import expression_tests,analytical_case
from host_inventory import topology,memory,process_memory
from runner import get
from runner import clean_exit_race,write as atomic_write
from kernels import specifications,fixture
from threadpoolctl import threadpool_limits,threadpool_info
import pytest
def test_frozen_archives_and_adapter():
 with threadpool_limits(1):assert check_frozen()['adapter_sha256']=='f92729e5f4b623775a1ffdd4013fe7f85d8ca86913d05383fe4274fb4c38b0cc'
def test_exact_identities_and_analytical_author_update():
 with threadpool_limits(1):
  assert all(v['PASS'] for v in expression_tests().values())
  assert all(v['PASS'] for v in analytical_case().values())
def test_old_loading_residual_is_essential():
 rng=np.random.default_rng(4237659100);z=rng.normal(size=(3,17))+1j*rng.normal(size=(3,17));old=z*.4;w=np.ones(17)
 expected=np.real(np.dot((z-old)*z.conj(),w)/w.sum())
 assert np.allclose(residual_moment(z,old,w),expected,atol=1e-10,rtol=1e-9)
 assert not np.allclose(residual_moment(z,2*old,w),expected)
def test_effective_thread_control_restores_context():
 pools=threadpool_info();prior={v['filepath']:v['num_threads'] for v in pools}
 with threadpool_limits(2):assert all(v['num_threads']==2 for v in threadpool_info())
 assert prior=={v['filepath']:v['num_threads'] for v in threadpool_info()}
def test_thread_schedule_is_physical_and_balanced():
 cfg=get(ROOT/'outputs/sp_u12b5b_r3_thread_benchmark_config.json');physical=topology()['physical_cores']
 assert all(t<=physical for t in cfg['thread_candidates'])
 for block in range(5):assert sorted(v['threads'] for v in cfg['schedule'] if v['block']==block)==cfg['thread_candidates']
 assert cfg['parity_tolerance']=={'atol':1e-10,'rtol':1e-9,'source':'scripts/sp_u12b5a_cases.py:ATOL,RTOL; frozen before R3 tests'}
def test_fixture_pairs_equal_input_and_output():
 with threadpool_limits(1):
  for family in ('loading','latent','quadratic','residual'):
   a=next(c for c in specifications() if c['kind']==family+'_direct').copy();a.update(D=29 if family!='latent' else 0,N=127,rank=8 if family in ('loading','latent') else 0)
   b=a|{'kind':family+'_low'};f,_,m=fixture(a,4237658801,4237658803);g,_,n=fixture(b,4237658801,4237658803)
   assert m['input_hashes']==n['input_hashes']
   assert np.allclose(f(),g(),atol=1e-10,rtol=1e-9)
def test_no_formal_dimensions_and_no_production_import():
 for c in specifications():
  assert c['N']<=2048
  if c['kind']=='loading_direct':assert 16*c['D']*c['N']*c['rank']<=128*2**20
 for n in ('scripts/sp_u12b5b_worker.py','src/baselines/mfa_reference_adapter.py'):
  assert 'mfa_low_memory_prototype' not in (ROOT/n).read_text(encoding='utf8')
def test_firewall_preserves_R2_and_blocks_payloads():
 firewall()
 assert not owned(ROOT/'scripts/sp_u12b5b_r2_report.py')
 assert owned(PORT/'low_memory.py')
 with pytest.raises(PermissionError):(ROOT/'outputs/sp_u12b5b/selection_train.npy').open('rb')
 with pytest.raises(PermissionError):(ROOT/'paper/main.tex').open('w')
def test_current_safe_budget():
 assert process_memory()['peak_working_set_bytes']<LIMITS['max_process_bytes']
 assert memory()['available_physical_bytes']>=LIMITS['min_system_available_physical_bytes']
def test_atomic_monitor_marker(tmp_path):
 path=tmp_path/'marker.json';atomic_write(path,{'stage':1});assert get(path)=={'stage':1}
 atomic_write(path,{'stage':2});assert get(path)=={'stage':2};assert not list(tmp_path.glob('*.tmp'))
def test_exit_race_requires_verified_success_and_final(tmp_path):
 class Proc:
  def __init__(self,code):self.code=code
  def wait(self,timeout):assert timeout==2;return self.code
 final=tmp_path/'worker_final.json'
 assert not clean_exit_race(Proc(0),final)
 atomic_write(final,{'complete':True})
 assert clean_exit_race(Proc(0),final)
 assert not clean_exit_race(Proc(1),final)
def test_windows_sharing_retry_is_bounded_and_before_timer(tmp_path,monkeypatch):
 import runner
 replace=runner.os.replace;attempts=[]
 def simulated(source,target):
  attempts.append(1)
  if len(attempts)<4:
   error=PermissionError('synthetic transient sharing conflict');error.winerror=5;raise error
  return replace(source,target)
 monkeypatch.setattr(runner.os,'replace',simulated)
 path=tmp_path/'marker.json';atomic_write(path,{'stage':3});assert get(path)=={'stage':3};assert len(attempts)==4
def test_optional_stage_marker_does_not_disable_safety_sampling(monkeypatch):
 import runner
 def denied(path):raise PermissionError('synthetic marker sharing conflict')
 monkeypatch.setattr(runner,'get',denied)
 assert runner.optional_marker('unused') is None
 # Resource querying is independent and still available, with unchanged caps.
 assert process_memory()['peak_working_set_bytes']<LIMITS['max_process_bytes']
