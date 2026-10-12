"""Preflight: analytic fixtures only, no formal channel/outcome decoding."""
import numpy as np
import pytest
from scripts import sp_u12b3b_common as b
def fixture_raw():
 E=np.arange(72,dtype=float).reshape(2,3,2,6)+1;T=np.tile(np.arange(6,dtype=float)+3,(2,1));coords=np.ones((2,3,6,2));mask=np.ones(coords.shape,bool);mask[:,0,:,1]=False;mask[:,1,:,0]=False;coords[~mask]=0
 te=np.zeros((2,3,2));tt=np.zeros(2)
 for j in range(6):te+=E[:,:,:,j];tt+=T[:,j]
 return dict(frame_error=E,frame_truth=T,coordinates=coords,coordinate_present=mask,deployment_coordinates=np.repeat(coords[:,:,None],2,axis=2),solve_status=np.ones((2,3,2,6),np.int8),trajectory_error=te,trajectory_truth=tt)
def test_raw_null_shared_coordinates_and_reduction():assert b.validate(fixture_raw())
@pytest.mark.parametrize('fault',['mask','coordinate','truth','reduction','solve'])
def test_raw_corruption_rejected(fault):
 r=fixture_raw()
 if fault=='mask':r['coordinate_present'][:,0,:,1]=True
 elif fault=='coordinate':r['deployment_coordinates'][0,0,1,0,0]+=1
 elif fault=='truth':r['frame_truth'][0,0]+=1
 elif fault=='reduction':r['trajectory_error'][0,0,0]+=1
 else:r['solve_status'][0,0,0,0]=0
 with pytest.raises(b.TechnicalFailure):b.validate(r)
def test_stable_archive_roundtrip():
 p=b.TECH/'synthetic_raw.npz';r=fixture_raw();one=b.archive(p,**r);two=b.archive(p,**r);assert one['sha256']==two['sha256'];b.validate(b.arrays(p))
def test_bootstrap_pooled_ratio_paired_not_average_db():
 E=np.arange(36,dtype=float).reshape(3,2,3,2)+1;T=np.array([[1,2],[10,20],[100,200.]]);index=np.array([[0,0,2],[1,2,2]])
 got=b.draw(E,T,index)
 for k,ix in enumerate(index):np.testing.assert_allclose(got[k],10*np.log10(E[ix].sum(axis=0)/T[ix].sum(axis=0)[:,None,None]),rtol=1e-14)
def test_bootstrap_reproducible_full_replicate_schema():
 E=np.arange(60,dtype=float).reshape(5,2,3,2)+1;T=np.ones((5,2));one,h1=b.bootstrap(E,T,223001,0);two,h2=b.bootstrap(E,T,223001,0);assert one.shape==(10000,2,3,2) and h1==h2 and np.array_equal(one,two)
def test_zero_endpoints_indeterminate_and_sign():
 assert b.classify(0,1)==b.classify(-1,0)==b.classify(-1,1)=='INDETERMINATE';assert b.classify(.1,1)=='POSITIVE';assert b.classify(-2,-1)=='NEGATIVE'
def test_full_synthetic_link_evaluation_and_native_layout(monkeypatch):
 from src.cdl_benchmark import OfficialCDL
 def forbidden(*args,**kwargs):raise AssertionError('SCIENTIFIC_GENERATION_FORBIDDEN')
 monkeypatch.setattr(OfficialCDL,'generate',forbidden)
 rx,tx,k,t=np.indices((2,2,64,160));h=(1+.4*rx+.3*tx+.01*k+.002*t+1j*(.2+.006*k+.004*t+.1*rx*tx)).astype(np.complex128)
 p=b.TECH/'synthetic_H.npz';b.archive(p,H=h)
 from scripts.resume_formal_test import load_native_h
 h=load_native_h(p);a=b.c.load_operators();cv,f=b.c.family(a,b.SNRS)
 comm=[b.c.communicate(dict(bits_seed_tx0=5520001+j*3,bits_seed_tx1=5520002+j*3,noise_seed=5520003+j*3)) for j in range(10)]
 raw,inputs,residual=b.evaluate(h,comm,a,cv,f);assert b.validate(raw) and len(inputs)==20 and residual<=b.c.SOLVE_TOL
def test_frozen_config_cardinality_seeds_mentor_and_B2_hashes():
 cfg=b.read(b.CFG);assert cfg['expected_units']==5*2*1000 and cfg['expected_trajectory_rows']==60000 and cfg['expected_frame_rows']==360000
 assert cfg['primary_replicates']==cfg['descriptive_replicates']==10000 and cfg['primary_q']==.05/24
 m=b.read(b.TECH/'mentor_spec_correction.json');assert m['method_change']==0 and m['B3_ABLATION_PIPELINE_READY']
 assert b.sha(b.MANIFEST)==b.MANIFEST_SHA and b.sha(b.CERT)==b.CERT_SHA
 s=b.read(b.TECH/'bootstrap_seed_registry.json');assert s['overlap_count']==0 and s['new_channel_seeds']==0
def test_firewall_in_separate_process_denies_all_forbidden_accesses():
 import subprocess,sys
 code="""from scripts import sp_u12b3b_common as b
b.firewall()
blocked=0
for path in [b.B2/'aggregate_results.csv',b.B2/'primary_contrasts.csv',b.B2/'figure_data.csv',b.B2/'cache/ID/H_0000.npz',b.B2/'raw_evaluation/ID/0000.npz',b.SEAL/'aggregate_results.csv',b.ROOT/'outputs/sp_u2b_seed_manifest.csv']:
 try:path.open('rb')
 except PermissionError:blocked+=1
assert blocked==7
try:(b.B2/'FORBIDDEN.txt').open('wb')
except PermissionError:pass
else:raise AssertionError('WRITE_NOT_BLOCKED')
"""
 p=subprocess.run([sys.executable,'-X','utf8','-c',code],capture_output=True,text=True);assert p.returncode==0
