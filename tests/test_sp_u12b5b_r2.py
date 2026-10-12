"""Scientific firewall and dangerous-allocation refusal tests for R2."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sp_u12b5b_r2_model import sizes, N, D
from scripts.sp_u12b5b_r2_bench import cases,validate_case
from scripts.sp_u12b5b_r2_common import ROOT,OUT,firewall,threshold_reason,LIMITS,owned
import pytest
def test_lower_bound_exact():
 assert [sizes(N,D,4,r)['LOWER_BOUND_single_tensor_bytes'] for r in (4,8,16)]==[3096576000,6193152000,12386304000]
 assert sizes(N,D,4,16)['LOWER_BOUND_single_tensor_bytes']>6*2**30
def test_lifetime_peak_not_sum_history():
 m=sizes(N,D,4,16)
 assert m['SOURCE_BASED_LIVE_ESTIMATE_bytes']==m['stage_live_bytes']['M_step_loading']
 assert m['SOURCE_BASED_LIVE_ESTIMATE_bytes']<sum(m['stage_live_bytes'].values())
def test_formal_tensor_refused():
 with pytest.raises(AssertionError):validate_case({'kind':'loading_tensor','D':4032,'N':12000,'rank':16})
 with pytest.raises(AssertionError):validate_case({'kind':'loading_tensor','D':4032,'N':1024,'rank':16})
def test_actual_schedule_bounded():
 for case in cases():validate_case(case)
 assert max(16*c['D']*c['N']*c['rank'] for c in cases() if c['kind']=='loading_tensor')<=128*2**20
def test_threshold_boundaries():
 assert threshold_reason({'peak_working_set_bytes':6*2**30},{'available_physical_bytes':3*2**30},LIMITS)=='PROCESS_RSS_CAP'
 assert threshold_reason({'peak_working_set_bytes':1},{'available_physical_bytes':2*2**30-1},LIMITS)=='SYSTEM_PHYSICAL_HEADROOM'
 assert threshold_reason({'peak_working_set_bytes':1},{'available_physical_bytes':2*2**30},LIMITS) is None
def test_firewall_blocks_historical_payload_and_write():
 firewall()
 with pytest.raises(PermissionError,match='SCIENTIFIC_VALUE'):(ROOT/'outputs/sp_u12b5b/selection_train.npy').open('rb')
 with pytest.raises(PermissionError,match='HISTORICAL_WRITE'):(ROOT/'paper/main.tex').open('w')
def test_ownership_protects_R1():
 assert owned(OUT/'pytest_tmp/a')
 assert not owned(ROOT/'scripts/sp_u12b5b_r1_worker.py')
def test_worker_no_data_loader_or_candidate_fit():
 source=(ROOT/'scripts/sp_u12b5b_r2_bench.py').read_text(encoding='utf8')
 assert 'np.load(' not in source and 'fit_reference(' not in source and 'make_reference_model(' not in source
 assert 'actual_PID' in source and 'process_handle(pid)' in source
