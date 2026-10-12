"""Pipeline preflight: synthetic fixtures only, no scientific CDL generation."""
import ast,json,hashlib
from pathlib import Path
import numpy as np
import pytest
import subprocess,sys
from scripts import sp_u12b2a_common as c
from scripts import run_sp_u12b2a as runner
ENGINEERING_SEEDS=(4420001,4420002,4420003)

@pytest.mark.parametrize('snr',c.SNRS)
def test_noise_calibration(snr):
 assert c.variance(snr)>0
 assert c.variance(snr)==pytest.approx(c.POWER*10**(-snr/10),rel=2e-16)
 if snr in (10,20):
  from src.frozen_affine_validation import VARIANCES
  assert c.variance(snr)==VARIANCES[snr]

def test_five_condition_and_method_inventory():
 assert tuple(c.CONDITIONS.values())==('CDL-C-30-100','CDL-B-30-100','CDL-C-120-100','CDL-B-120-100','CDL-C-30-200')
 assert c.METHODS==('LS-LIN','Source','Hard','Convex','Dense','LR4','Matched')
 assert 'Perfect CSI' not in c.METHODS
 cfg=c.read(c.CONFIG);assert cfg['sample_rate_hz']==cfg['nfft']*cfg['delta_f_hz']
 assert not cfg['formal_generation_authorized'] and cfg['rank']==4 and cfg['window_frames']==5 and cfg['folds']==4

def test_primary_family_and_signs():
 rows=c.planned_contrasts();assert len(rows)==35 and len({(r['condition'],r['snr_db'],r['comparison']) for r in rows})==35
 assert sum(r['comparison']=='source_gain' for r in rows)==21
 assert sum(r['comparison']=='convex_gain' for r in rows)==7
 assert sum(r['comparison']=='compression_penalty' for r in rows)==7
 assert all(r['first']=='LR4' and r['second']=='Dense' and r['positive_means']=='compression worse' for r in rows if r['comparison']=='compression_penalty')
 assert not any(r['condition'] in ('ID','Boundary') for r in rows)

@pytest.mark.parametrize('bounds,expected',[((.1,1),'POSITIVE'),((-1,-.1),'NEGATIVE'),((-.1,.1),'INDETERMINATE'),((0,1),'INDETERMINATE'),((-1,0),'INDETERMINATE')])
def test_classification(bounds,expected):assert c.classify(*bounds)==expected

def test_seed_registry_disjoint_and_scope_exact():
 formal=c.read(c.ROOT/'outputs/sp_u12b2a_seed_registry.json');dev=c.read(c.OUT/'development_seed_registry.json')
 f={r['seed'] for r in formal['rows']};d={r['seed'] for r in dev['rows']};history=c.read(c.OUT/'historical_seed_inventory.json')
 assert len(f)==len(formal['rows'])==172501 and len(d)==len(dev['rows'])==540
 assert not f&d and not (f|d)&set(history['forbidden_seeds'])
 assert not f&set(ENGINEERING_SEEDS)
 index=runner.seed_index(formal)
 for condition in c.CONDITIONS:
  assert len(runner.consumed(index,condition,'MATCHED_REF',0))==3
  assert len(runner.consumed(index,condition,'EVAL',0,0))==6
  for role in c.ROLES[:3]:assert runner.consumed(index,condition,'EVAL',0,0)[role]==runner.consumed(index,condition,'EVAL',0,9)[role]
 assert sum(r['role']=='noise_seed' for r in formal['rows'])==50000
 assert sum(r['group']=='MATCHED_REF' for r in formal['rows'])==7500

def test_allocator_deterministic_and_rejects_collision():
 forbidden={42,43};used=set();a=runner.allocate('UNIT_TEST_NAMESPACE',1,1,forbidden,used)
 b=runner.allocate('UNIT_TEST_NAMESPACE',1,1,forbidden,set())
 assert a==b
 values={r['seed'] for r in a['rows']};assert len(values)==180 and not values&forbidden
 collision=next(iter(values));again=runner.allocate('UNIT_TEST_NAMESPACE',1,1,forbidden|{collision},set())
 assert again['rejection_count']>0 and collision not in {r['seed'] for r in again['rows']}

def test_formal_directory_only_metadata():
 assert {p.relative_to(c.FORMAL).as_posix() for p in c.FORMAL.rglob('*') if p.is_file()}=={'config.json','seed_registry.json','schemas.json','manifest.json'}
 assert not list(c.FORMAL.rglob('*.npz')) and not list(c.FORMAL.rglob('*.csv'))
 assert c.read(c.FORMAL/'manifest.json')['clean_evaluation_generated']==0

def test_future_generation_and_unsealing_need_authorization():
 with pytest.raises(PermissionError):runner.run_future_sealed_generation()
 with pytest.raises(PermissionError):runner.analyze_future_sealed()

def test_runtime_firewall_denies_forbidden_reads_and_writes():
 source="""from scripts import sp_u12b2a_common as c
c.firewall()
attempts=[('read',c.ROOT/'outputs/t20b_formal_nmse.csv'),('read',c.ROOT/'outputs/sp_u2c_summary.json'),('read',c.FORMAL/'manifest.json'),('write',c.ROOT/'outputs/FORBIDDEN_UNIT.json'),('write',c.FORMAL/'FORBIDDEN_UNIT.json')]
with c.phase('SMOKE'):
 for mode,path in attempts:
  try:
   with path.open('r' if mode=='read' else 'x'):pass
  except PermissionError:continue
  raise AssertionError('firewall did not deny '+str(path))
print('5 forbidden actions denied')
"""
 proc=subprocess.run([sys.executable,'-X','utf8','-c',source],cwd=c.ROOT,capture_output=True,text=True)
 assert proc.returncode==0,proc.stdout+proc.stderr
 assert '5 forbidden actions denied' in proc.stdout

def test_actual_schema_writer_heterogeneous_nulls(tmp_path):
 rows=[]
 for method in c.METHODS:
  row=dict(condition='UNIT',snr_db=10,estimator=method,trajectory_id=0,frame_id=4,error_energy=1.,truth_energy=2.,bit_errors=0,bit_total=100,alpha=None,beta=None,solve_status='NOT_APPLICABLE')
  if method in ('Hard','Convex','Dense','LR4'):row.update(alpha=0.,beta=0.,solve_status='PASS' if method in ('Dense','LR4') else 'NOT_APPLICABLE')
  rows.append(row)
 path=tmp_path/'schema.csv';c.csvwrite(path,rows);actual=c.csvread(path)
 assert len(actual)==7 and set(actual[0])==set(c.RAW_FIELDS) and actual[0]['alpha']==''

def test_noise_CRN_is_scaling_only():
 from src.numerics import complex_gaussian_noise
 z=complex_gaussian_noise((2,64,16),1.,np.random.default_rng(4420003))
 for s in c.SNRS:
  y=np.sqrt(c.variance(s))*z
  np.testing.assert_allclose(y/np.sqrt(c.variance(s)),z,rtol=1e-14,atol=1e-14)

def test_frozen_estimator_source_has_no_new_behaviors():
 tree=ast.parse((c.ROOT/'src/pilot_cv.py').read_text(encoding='utf8'))
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PilotCV');fit=next(n for n in cls.body if getattr(n,'name',None)=='fit');text=ast.unparse(fit)
 assert 'np.linalg.solve(gram, rhs)' in text
 assert not any(x in text for x in ('np.clip(', 'np.linalg.pinv(', 'np.linalg.lstsq('))
 infer=ast.unparse(ast.parse((c.ROOT/'scripts/sp_u12b2a_common.py').read_text(encoding='utf8')))
 assert 'return deployed(pilots, a, s, cvs[s]' in infer

def test_synthetic_end_to_end_uses_frozen_paths_without_CDL(monkeypatch):
 from src.cdl_benchmark import OfficialCDL
 monkeypatch.setattr(OfficialCDL,'generate',lambda *a,**k:pytest.fail('scientific CDL forbidden in unit tests'))
 a=c.historical_operators();st=c.statistics(c.ROOT/'outputs'/c.STATS['Source']);matched={10:st.prepare(c.variance(10))}
 rx,tx,k,t=np.indices((2,2,64,160));h=(1+.4*rx+.3*tx+.01*k+.002*t+1j*(.2+.006*k+.004*t+.1*rx*tx)).astype(np.complex128)
 comm=[c.communicate(dict(bits_seed_tx0=ENGINEERING_SEEDS[0]+j*3,bits_seed_tx1=ENGINEERING_SEEDS[1]+j*3,noise_seed=ENGINEERING_SEEDS[2]+j*3)) for j in range(10)]
 raw,rows,evidence=c.evaluate(h,comm,a,matched,(10,),a)
 assert c.validate_raw(raw,(10,)) and len(rows)==42 and evidence['solve_success']
 assert evidence['prediction_relative_error_max']==0 and evidence['coordinate_relative_error_max']==0
 assert raw['coordinates'].shape==(1,6,2) and raw['perfect_frame_bit_errors'].shape==(1,6)
 assert 'Perfect CSI' not in {r['estimator'] for r in rows}
 e,q,b,bt=c.raw_to_trajectory(raw,(10,))
 np.testing.assert_allclose(e,raw['frame_error'].sum(axis=2));np.testing.assert_allclose(q,raw['frame_truth'].sum(axis=1))
 bad=dict(raw);bad['frame_error']=raw['frame_error'][:,:6]
 with pytest.raises(ValueError):c.validate_raw(bad,(10,))
