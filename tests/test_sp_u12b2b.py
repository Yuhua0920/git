"""Before formal H: synthetic fixtures and metadata checks only."""
import json,sys,subprocess,concurrent.futures
from pathlib import Path
import numpy as np
import pytest
from scripts import sp_u12b2b_common as b
from scripts import sp_u12b2a_common as c

def test_frozen_design_and_new_descriptive_stream():
 cfg=b.config();seed=b.read(b.ROOT/'outputs/sp_u12b2b_seed_audit.json')
 assert cfg['snr_db']==[0,5,10,15,20,25,30] and cfg['N_reference_per_condition']==500 and cfg['N_evaluation_per_condition']==1000
 assert cfg['rank']==4 and cfg['window']==5 and cfg['folds']==4 and cfg['methods']==list(c.METHODS)
 assert cfg['sample_rate_hz']==cfg['nfft']*cfg['delta_f_hz']
 assert cfg['primary_bootstrap']['replicates']==100000 and cfg['descriptive_bootstrap']['replicates']==10000
 assert cfg['primary_bootstrap']['master_seed']!=cfg['descriptive_bootstrap']['seed'] and seed['PASS']
 assert cfg['descriptive_bootstrap']['scope'].startswith('DESCRIPTIVE_ONLY') and len(cfg['primary_contrasts'])==35

def test_b2a_source_and_seed_bytes_unchanged():
 manifest=b.original_gate();assert manifest['registry_sha256']=='84d82c05a1757c2e3fe64afa2146c182739a92b8037eb0bd1d62ed2e1d9726a5'
 assert b.sha(c.OUT/'execution_freeze.json')=='4c5f9ab3e3d7b07b1462e6c2662cab938ff16b9db90cec77d1c5a69e244787cc'

def test_bootstrap_ratio_is_pooled_and_paired():
 T=np.broadcast_to(np.array([1.,9.,3.])[:,None],(3,7)).copy();E=T[:,:,None]*np.broadcast_to(np.array([1.,4.,2.])[:,None,None],(3,7,7))*np.arange(1,8)[None,None,:]
 index=np.array([[0,0,1],[1,2,2]],np.int32);actual=b.bootstrap_draw(E,T,index)
 manual=np.array([10*np.log10(E[ids].sum(axis=0)/T[ids].sum(axis=0)[:,None]) for ids in index])
 np.testing.assert_allclose(actual,manual,rtol=0,atol=1e-14)
 assert not np.allclose(actual[0],10*np.log10(E[index[0]]/T[index[0],:,None]).mean(axis=0))
 np.testing.assert_allclose(actual[:,:,1]-actual[:,:,0],10*np.log10(2),atol=1e-14)

def test_primary_descriptive_streams_and_quantiles():
 E=np.arange(1,1+9*7*7,dtype=float).reshape(9,7,7);T=np.arange(1,64,dtype=float).reshape(9,7)
 cfg=b.config();p,ph=b.bootstrap_array(E,T,cfg['primary_bootstrap']['master_seed'],1,31,c.planned_contrasts()[:7]);p2,ph2=b.bootstrap_array(E,T,cfg['primary_bootstrap']['master_seed'],1,31,c.planned_contrasts()[:7]);d,dh=b.bootstrap_array(E,T,cfg['descriptive_bootstrap']['seed'],1,31)
 assert p.shape==(31,7) and d.shape==(31,7,7) and ph==ph2 and ph!=dh
 np.testing.assert_array_equal(p,p2)
 assert np.quantile(np.array([0.,10.]),.25,method='linear')==2.5

def test_archive_and_checkpoint_are_deterministic_and_detect_corruption(tmp_path):
 data={'x':np.arange(24).reshape(2,3,4),'flags':np.array([True,False])}
 first=b.archive(tmp_path/'first.npz',**data);second=b.archive(tmp_path/'second.npz',**data)
 assert first['sha256']==second['sha256'];receipt=tmp_path/'commit.json';b.write(receipt,dict(artifacts=[first]))
 assert b.committed(receipt)['artifacts'][0]['sha256']==first['sha256']
 (tmp_path/'first.npz').write_bytes(b'corrupted')
 with pytest.raises(b.IntegrityError,match='COMMITTED_ARTIFACT_CORRUPTION'):b.committed(receipt)

def test_actual_saved_H_layout_matches_native_fixture(tmp_path):
 original=np.arange(2*2*64*160).reshape(2,2,64,160).astype(np.complex128).transpose(1,0,3,2).copy().transpose(1,0,3,2)
 path=tmp_path/'H.npz';b.archive(path,H=original);loaded=b.load_h(path,original.strides)
 np.testing.assert_array_equal(loaded,original);assert loaded.strides==original.strides

def test_serial_parallel_synthetic_scientific_path_identical(tmp_path):
 serial=b.synthetic_worker(str(tmp_path/'serial.npz'))
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
  results=list(pool.map(b.synthetic_worker,[str(tmp_path/'parallel0.npz'),str(tmp_path/'parallel1.npz')]))
 assert all(row==serial for row in results)
 assert serial['rows']==294 and serial['scientific_CDL_generated']==0
 raw=b.arrays(tmp_path/'serial.npz');assert b.validate_raw(raw,b.config()['expected_bit_total_per_frame'])
 damaged=dict(raw);damaged['method_coordinates']=raw['method_coordinates'].copy();damaged['method_coordinates'][0,4,0,0]+=1
 with pytest.raises(b.IntegrityError,match='DENSE_LR4_COORDINATE_SHARING'):b.validate_raw(damaged,b.config()['expected_bit_total_per_frame'])

def test_console_rejects_scientific_keys():
 for key in ('nmse','ber','gain','alpha','beta','classification_counts','ranking'):
  with pytest.raises(b.IntegrityError,match='CONSOLE_SCHEMA'):b.emit(**{key:1})

def test_firewall_denies_historical_outcomes_and_unscoped_sealed_reads():
 source="""from scripts import sp_u12b2b_common as b
b.firewall()
for p in (b.ROOT/'outputs/t20b_formal_nmse.csv',b.ROOT/'outputs/sp_u2c_summary.json',b.SEAL/'aggregate_results.csv'):
 try:p.open('rb')
 except PermissionError:continue
 raise AssertionError('firewall')
print('denied')
"""
 proc=subprocess.run([sys.executable,'-X','utf8','-c',source],cwd=b.ROOT,capture_output=True,text=True)
 assert proc.returncode==0 and 'denied' in proc.stdout

def test_final_schemas_and_figure_policy():
 schema=b.read(b.TECH/'schema.json')
 assert schema['raw_logical_rows']==1470000 and schema['trajectory_logical_rows']==245000 and schema['aggregate_rows']==schema['figure_rows']==245 and schema['contrast_rows']==35
 assert schema['main_figure_methods']==['LS-LIN','Source','Convex','Dense','LR4','Matched'] and schema['supplement_methods']==list(c.METHODS)
 assert schema['no_plot_rendering']
