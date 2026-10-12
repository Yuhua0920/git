"""Isolated technical/synthetic assertions. Formal scientific values never disclosed."""
from pathlib import Path
import json,hashlib,numpy as np,pytest
from scripts import sp_u12b3d_common as b
from scripts.sp_u12b3d_build import core,EXPECTED,function_sources,CORE
ROOT=Path(__file__).resolve().parents[1]
def test_scientific_partition_exact():
 assert core()[0]==EXPECTED
 assert function_sources(ROOT/'scripts/sp_u12b3b_common.py',CORE)==function_sources(ROOT/'scripts/sp_u12b3d_common.py',CORE)
 assert function_sources(ROOT/'scripts/run_sp_u12b3b.py',('compute',))==function_sources(ROOT/'scripts/run_sp_u12b3d.py',('compute',))
def test_original_seed_config_protocol_bytes():
 assert b.sha(b.CFG)=='1b0fb501e9a0394bc180464b8e51e0af5ec0fdd4e776eef6c91fecd9f83ffb89'
 assert b.sha(b.PROTOCOL)=='4382607d7f22d2d783c449fc4f56f778f610276274745f32ec246cc39b5b2eea'
 assert b.sha(b.TECH/'bootstrap_seed_registry.json')==b.sha(ROOT/'outputs/sp_u12b3b/bootstrap_seed_registry.json')
 assert b.sha(ROOT/'outputs/sp_u12b3b_manifest.json')=='530cc5bc124bfc945a381fb6a3d30ad2ec70666c2887a48aa9b69f9847d8a250'
def test_synthetic_concurrency_crash_forensics():
 r=b.read(b.TECH/'synthetic_gates.json');assert r['PASS'] and r['actual_crash'] and r['dead_temp_quarantined'] and not r['formal_inputs_used'];assert [v['resumed'] for v in r['two_worker_passes']]==[0,100]
 assert b.read(b.TECH/'synthetic_control_archive/readonly_control.json')['expected_denials']==6
def test_exact_dispatcher_failstop_actual_worker_pids():
 r=b.read(b.TECH/'failstop_gate.json');assert r['PASS'] and r['exact_dispatcher'] and r['actual_worker_termination'] and not r['later_cell_started'] and r['retries']==0
def test_unchanged_dummy_bootstrap_and_null_classification():
 from scripts import sp_u12b3b_common as old
 E=np.arange(60,dtype=float).reshape(5,2,3,2)+1;T=np.ones((5,2));x,h=old.bootstrap(E,T,223001,0);y,h2=b.bootstrap(E,T,223001,0);assert np.array_equal(x,y) and h==h2
 assert b.classify(0,1)==b.classify(-1,0)=='INDETERMINATE'
def test_atomic_synthetic_determinism_and_integrity():
 path=b.TECH/'pytest_synthetic.npz';x=np.arange(10);a=b.archive(path,x=x);v=b.archive(path,x=x);assert a['sha256']==v['sha256'];assert np.array_equal(b.arrays(path)['x'],x)
@pytest.mark.parametrize('field',['timestamp_utc','PID','worker_id','thread_id','absolute_path','operation','source_file','source_line','full_traceback','errno','WinError','trajectory_key','condition','SNR','current_stage','temp_path','final_path'])
def test_full_envelope_fields(field):
 records=[b.read(p) for p in (b.TECH/'synthetic_control_archive').glob('worker_exception_*.json')];assert records and all(field in r for r in records)
def test_no_formal_protocol_relaxation():
 c=b.read(b.CFG);assert c['expected_units']==10000 and c['expected_trajectory_rows']==60000 and c['expected_frame_rows']==360000;assert c['primary_q']==.05/24 and c['primary_replicates']==10000
def test_postrun_receipts_if_manifest_present():
 # Preflight checks gates; final pass validates actual technical receipts without opening scientific data.
 p=ROOT/'outputs/sp_u12b3d_run2_manifest.json'
 if p.exists():
  m=b.read(p);assert m['status']=='B3_RUN2_SEALED_COMPLETE_UNVIEWED' and not m['outcomes_viewed']
  for name in ('integrity_audit','both_replay_audit','bootstrap_audit'):assert b.read(ROOT/'outputs'/('sp_u12b3d_run2_'+name+'.json'))['PASS']
  assert b.read(ROOT/'outputs/sp_u12b3d_run2_integrity_audit.json')['trajectory_rows']==60000
 else:assert b.read(b.TECH/'synthetic_gates.json')['PASS'] and not any(b.SEAL.iterdir())
