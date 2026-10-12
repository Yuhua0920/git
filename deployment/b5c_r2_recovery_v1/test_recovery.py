"""Synthetic stored-energy tests; never real model or formal inference."""
from pathlib import Path
from types import SimpleNamespace
import ast,csv,hashlib,json,os,shutil,subprocess,sys,zipfile
import numpy as np
import pytest
import recovery_base as r
import recovery_core as core
import control as c
sys.path.insert(0,str(r.ROOT/'deployment/b5c_paired_r2_v1'))
import science,paired_stats,report

def arrays(i=0):
    ft=np.broadcast_to(np.arange(1,7,dtype=np.float64)+i/1000,(7,6)).copy()
    fe=np.stack([ft*.3,ft*(.2+np.arange(7)[:,None]*.03)],axis=2)
    return dict(frame_error=fe,frame_truth=ft,trajectory_error=science.paired_reduction(fe),trajectory_truth=ft.sum(axis=1))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('wb') as f:np.savez(f,**v)
@pytest.fixture
def sandbox(tmp_path,monkeypatch):
    root=tmp_path/'Synthetic recovered project with spaces';root.mkdir()
    monkeypatch.setattr(r,'ROOT',root);monkeypatch.setattr(r,'OLD',root/'lab_runs/b5c_paired_r2')
    monkeypatch.setattr(r,'SOURCE',r.OLD/'formal');monkeypatch.setattr(r,'OUT',root/'lab_runs/b5c_paired_r2_recovery')
    return root

def test_original_manifest_fault_reproduced_in_independent_process():
    root=r.HERE.parents[1]
    program="""import sys,json
sys.path.insert(0,sys.argv[1]);import base as b,control as c
c.scope(b.OUT/'formal',False)
try:b.verify()
except RuntimeError as e:
 print(json.dumps({'reproduced':str(e)=='B5C_READ_FIREWALL_DENIED:'+str(b.HERE/'package_manifest.json'),'message':str(e)}))
else:raise RuntimeError('FAULT_NOT_REPRODUCED')
"""
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',program,str(root/'deployment/b5c_paired_r2_v1')],capture_output=True,text=True,encoding='utf8',timeout=30)
    assert p.returncode==0,p.stderr;assert json.loads(p.stdout)['reproduced']
@pytest.mark.parametrize('field,mode',[(f,m) for f in arrays() for m in ('nan','shape','dtype')])
def test_bad_arrays_rejected(sandbox,field,mode):
    p=r.SOURCE/'bad.npz';v=arrays()
    if mode=='nan':v[field].flat[0]=np.nan
    elif mode=='shape':v[field]=v[field][:-1]
    else:v[field]=v[field].astype(np.float32)
    save(p,v)
    with pytest.raises(RuntimeError):core.load_raw(p,science.paired_reduction)
def test_sufficient_statistics_exact(sandbox):
    p=r.SOURCE/'valid.npz';v=arrays();save(p,v)
    result=core.load_raw(p,science.paired_reduction);assert np.array_equal(result[2],v['trajectory_error'])
    v['trajectory_error'][0,1]+=1e-8;save(p,v)
    with pytest.raises(RuntimeError,match='STATISTIC'):core.load_raw(p,science.paired_reduction)
def test_truncated_and_duplicate_NPZ_rejected(sandbox):
    p=r.SOURCE/'bad.npz';save(p,arrays());data=p.read_bytes();p.write_bytes(data[:-20])
    with pytest.raises(zipfile.BadZipFile):core.load_raw(p,science.paired_reduction)
    save(p,arrays())
    with pytest.warns(UserWarning):
        with zipfile.ZipFile(p,'a') as z:z.writestr('frame_error.npy',b'FAKE')
    with pytest.raises(RuntimeError,match='DUPLICATE'):core.load_raw(p,science.paired_reduction)
def test_zip_CRC_rejected(sandbox):
    p=r.SOURCE/'bad.npz';save(p,arrays())
    with zipfile.ZipFile(p) as z:header=z.getinfo('frame_error.npy').header_offset
    data=bytearray(p.read_bytes());nl=int.from_bytes(data[header+26:header+28],'little');el=int.from_bytes(data[header+28:header+30],'little')
    data[header+30+nl+el+160]^=1;p.write_bytes(data)
    with pytest.raises((zipfile.BadZipFile,RuntimeError)):core.load_raw(p,science.paired_reduction)
def test_snapshot_detects_changed_extra_and_missing(sandbox):
    p=r.OLD/'formal/a.json';p.parent.mkdir(parents=True);p.write_text('{}')
    snap=r.snapshot(r.OLD);r.unchanged(snap);p.write_text('{"change":true}')
    with pytest.raises(RuntimeError,match='CHANGED'):r.unchanged(snap)
def test_old_writes_and_outside_reads_denied_in_child(sandbox):
    old=r.OLD/'model.npy';old.parent.mkdir(parents=True);old.write_bytes(b'OLD')
    allowed=sandbox/'allowed.json';allowed.write_text('{}');other=sandbox/'lab_inputs/forbidden.npy';other.parent.mkdir();other.write_bytes(b'NO')
    script="""import sys,json;from pathlib import Path
sys.path.insert(0,sys.argv[1]);import recovery_base as r
r.ROOT=Path(sys.argv[2]);r.OUT=r.ROOT/'lab_runs/recovery'
r.firewall([r.ROOT/'allowed.json'])
assert (r.ROOT/'allowed.json').read_text()=='{}'
for name,mode in [('lab_runs/b5c_paired_r2/model.npy','wb'),('lab_inputs/forbidden.npy','rb'),('outputs/sp_u12b3h/result.npz','rb')]:
 try:(r.ROOT/name).open(mode)
 except RuntimeError:pass
 else:raise RuntimeError('FIREWALL_BYPASS')
print('READONLY_FIREWALL_PASS')
"""
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',script,str(r.HERE),str(sandbox)],capture_output=True,text=True,encoding='utf8')
    assert p.returncode==0,p.stderr;assert old.read_bytes()==b'OLD'
def test_manifest_self_explicitly_allowed(sandbox):
    p=sandbox/'package_manifest.json';p.write_text('{}')
    script="""import sys;from pathlib import Path
sys.path.insert(0,sys.argv[1]);import recovery_base as r
r.ROOT=Path(sys.argv[2]);r.OUT=r.ROOT/'new';r.firewall([r.ROOT/'package_manifest.json'])
assert r.read(r.ROOT/'package_manifest.json')=={}
print('SELF_MANIFEST_PASS')
"""
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',script,str(r.HERE),str(sandbox)],capture_output=True,text=True,encoding='utf8')
    assert p.returncode==0,p.stderr
def test_JSON_duplicates_denied(sandbox):
    p=sandbox/'duplicate.json';p.write_text('{"x":1,"x":2}')
    with pytest.raises(RuntimeError,match='DUPLICATE_JSON'):r.read(p)
def test_inference_guard_refuses_original_call():
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',"import sys;sys.path.insert(0,sys.argv[1]);sys.path.insert(1,sys.argv[2]);import control as c,science;c.forbid_inference(science);science.evaluate(None)",str(r.HERE),str(r.ROOT/'deployment/b5c_paired_r2_v1')],capture_output=True,text=True,encoding='utf8')
    assert p.returncode!=0 and 'RECOVERY_MFA_INFERENCE_FORBIDDEN' in p.stderr
def test_gate_before_analysis_and_no_original_status_rewrite():
    tree=ast.parse((r.HERE/'control.py').read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='recover')
    text=ast.unparse(f);assert text.index('trajectory_gate')<text.index("'integrity_gate.json'")<text.index('.analyze(')
    assert 'infer(' not in text and 'evaluate(' not in text and '.OUT' in text

def test_gate_rejection_cannot_create_analysis(sandbox):
    job=r.OUT/'rejected';job.mkdir(parents=True)
    with pytest.raises(RuntimeError,match='GATE_NOT_AUTHENTICATED'):
        core.analyze(r.SOURCE,job,{},science,paired_stats,report,{'PASS':False},{})
    assert not(job/'analysis').exists()

def test_full_stored_result_recovery_and_frozen_statistics(sandbox,monkeypatch):
    p=dict(conditions=dict.fromkeys(paired_stats.CONDITIONS,'SYNTHETIC'),snrs=list(paired_stats.SNRS),score_frames=list(range(4,10)),N=1000,final_model_manifest_sha256='SYNTHETIC_MODEL',synthetic_only=True)
    def identities(i):return {(s,j):{'snr_db':s,'frame_id':j,'pilot_sha256':'SYNTHETIC_'+str(i)} for s in p['snrs'] for j in range(10)}
    def checked(condition,i,cfg,index):
        v=arrays(i);return i,condition,identities(i),v['frame_error'][:,:,0],v['frame_truth'],{'SYNTHETIC_INPUT':i},v['trajectory_error'][:,0]
    def pilots(i,condition,inputs):return None,None,arrays(i)['frame_truth'][0]
    fake=SimpleNamespace(paired_reduction=science.paired_reduction,checked_input=checked,pilot_inputs=pilots)
    for condition in p['conditions']:
        for i in range(1000):
            path=r.SOURCE/'raw'/condition/f'{i:04d}.npz';save(path,arrays(i))
            receipt=dict(condition=condition,trajectory_id=i,snrs=p['snrs'],frames=p['score_frames'],methods=['LR4','MFA'],pairing_verified=True,
                model_manifest_sha256=p['final_model_manifest_sha256'],raw_sha256=r.sha(path),original_inputs={'SYNTHETIC_INPUT':i},
                scored_pilot_and_truth_identities=[identities(i)[s,j] for s in p['snrs'] for j in p['score_frames']])
            path.with_suffix('.json').write_text(json.dumps(receipt),encoding='utf8')
    runtime=r.SOURCE/'inference_runtime.jsonl'
    with runtime.open('w',encoding='utf8') as f:
        for condition in p['conditions']:
            for i in range(1000):
                for s in p['snrs']:f.write(json.dumps(dict(condition=condition,trajectory_id=i,snr_db=s,rows=6,conditional_calls=6,observer_profile_included=True,adapter_wall_seconds=.2,posterior_setup_seconds=.1))+'\n')
    (r.SOURCE/'status.json').write_text('{"state":"FAILED","PASS":false,"synthetic_only":true}')
    snapshot=r.snapshot(r.OLD);job=r.OUT/'synthetic_full_chain';job.mkdir(parents=True);r.write(job/'original_snapshot.json',snapshot)
    gate=core.trajectory_gate(r.SOURCE,p,fake,{},{});assert gate['PASS'] and gate['MFA_scored_positions']==210000 and gate['runtime_calls']==35000 and gate['synthetic_only']
    gate['snapshot_sha256']=r.sha(job/'original_snapshot.json');r.write(job/'integrity_gate.json',gate)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=6,user_api='blas'):dest=core.analyze(r.SOURCE,job,p,science,paired_stats,report,gate,dict(peak_RSS_bytes=0,peak_private_commit_bytes=0,wall_seconds=1))
    rows=list(csv.DictReader((dest/'nmse_comparison_35_cells.csv').open()));assert len(rows)==35
    assert {v['classification'] for v in rows}=={'MFA_SUPPORTED','LR4_SUPPORTED'}
    assert all(v['bootstrap_replicates']=='100000' and v['family_size']=='35' and v['N_trajectories']=='1000' for v in rows)
    for condition in p['conditions']:
        with np.load(dest/(condition+'_bootstrap.npz'),allow_pickle=False) as z:assert z['MFA_minus_LR4_dB'].shape==(100000,7)
    r.unchanged(snapshot);assert r.read(r.SOURCE/'status.json')['state']=='FAILED'
    result={'PASS':True,'synthetic_only':True,'stored_JSON':5000,'stored_NPZ':5000,'runtime_identities':35000,'score_identities':210000,'analysis_cells':35,'bootstrap_per_condition':100000,'original_FAILED_preserved':True,'new_MFA_inference_calls':0,'fixture_job':str(job)}
    r.write(job/'synthetic_full_chain_result.json',result)
    # Persist only technical test feedback in a caller-designated scratch path.
    if os.environ.get('RECOVERY_SYNTHETIC_REPORT'):Path(os.environ['RECOVERY_SYNTHETIC_REPORT']).write_text(json.dumps(result,indent=2),encoding='utf8')
    # Negative identity, hash, missing-file, and duplicate-runtime cases are
    # checked against this same complete synthetic fixture; none rerun statistics.
    item=r.SOURCE/'raw/ID/0000.json';original=item.read_text();bad=json.loads(original);bad['trajectory_id']=1;item.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError,match='IDENTITY'):core.trajectory_gate(r.SOURCE,p,fake,{},{} )
    item.write_text(original);bad=json.loads(original);bad['raw_sha256']='0'*64;item.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError,match='HASH'):core.trajectory_gate(r.SOURCE,p,fake,{},{} )
    item.write_text(original);item.rename(item.with_suffix('.missing'))
    with pytest.raises(RuntimeError,match='INVENTORY'):core.coverage(r.SOURCE,p)
    text=runtime.read_text();runtime.write_text(text+text.splitlines(True)[0])
    with pytest.raises(RuntimeError,match='DUPLICATE'):core.runtime_grid(runtime,p)
