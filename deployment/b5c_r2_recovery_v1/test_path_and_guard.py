"""Windows reparse refusal and authenticated analysis-only process smoke."""
from pathlib import Path
import json,subprocess,sys,zipfile
import pytest
import recovery_base as r
from test_recovery import sandbox

def junction(link,target):
    quote=lambda s:"'"+str(s).replace("'","''")+"'"
    code="New-Item -ItemType Junction -Path "+quote(link)+" -Target "+quote(target)+" -ErrorAction Stop | Out-Null"
    p=subprocess.run(['powershell','-NoProfile','-Command',code],capture_output=True,text=True)
    assert p.returncode==0,p.stderr

def test_output_root_junction_rejected_without_original_write(sandbox):
    r.OLD.mkdir(parents=True);keep=r.OLD/'original.json';keep.write_bytes(b'ORIGINAL')
    r.OUT.parent.mkdir(parents=True,exist_ok=True);junction(r.OUT,r.OLD)
    with pytest.raises(RuntimeError,match='REPARSE'):r.write(r.OUT/'new.json',{})
    assert keep.read_bytes()==b'ORIGINAL' and not(r.OLD/'new.json').exists()

def test_nested_output_junction_rejected(sandbox):
    r.OUT.mkdir(parents=True);r.OLD.mkdir(parents=True);junction(r.OUT/'redirect',r.OLD)
    with pytest.raises(RuntimeError):r.write(r.OUT/'redirect/new.json',{})
    assert not(r.OLD/'new.json').exists()

def test_manifest_gate_stats_under_real_readonly_hook(tmp_path):
    # Original authenticated code, small synthetic energies; never formal data.
    program="""import sys,json;from pathlib import Path
sys.path.insert(0,sys.argv[1]);import recovery_base as r;import control
b,s,stats,report,h=r.frozen_modules()
import numpy as np;from threadpoolctl import threadpool_limits
from src.cdl_benchmark import observation_geometry,frames
from src.link import assemble_pilot_data_frame,apply_frequency_mimo_channel,estimate_ls_linear_channel
exact=[r.ROOT/n for n in r.bindings()['frozen_files']]
exact += [r.ROOT/n for n in r.read(r.HERE/'package_manifest.json')['files']]
exact += [r.HERE/'package_manifest.json']
r.OUT=Path(sys.argv[2]);r.OUT.mkdir(parents=True)
r.firewall(exact);control.forbid_inference(s)
assert b.verify()['PASS'];assert r.verify()['PASS']
e=np.ones((10,7,2),np.float64);e[:,:,1]*=2;t=np.ones((10,7),np.float64)
with threadpool_limits(limits=6,user_api='blas'):
 rows,draws=stats.summarize(e,t,0)
assert draws.shape==(100000,7) and len(rows)==7
with (r.OUT/'synthetic_bootstrap.npz').open('xb') as f:np.savez(f,draws=draws)
r.write(r.OUT/'guard_result.json',{'PASS':True,'synthetic_only':True,'new_MFA_inference_calls':0,'original_manifest_access':True,'bootstrap_replicates':100000})
print('GUARDED_AUTHENTICATED_STATS_PASS')
"""
    import uuid
    out=r.ROOT/'lab_runs'/('synthetic_guard_'+uuid.uuid4().hex[:12])
    assert not out.exists()
    p=subprocess.run([sys.executable,'-B','-X','utf8','-c',program,str(r.HERE),str(out)],capture_output=True,text=True,encoding='utf8',timeout=120)
    assert p.returncode==0,p.stdout+'\n'+p.stderr
    assert r.read(out/'guard_result.json')['PASS']

def test_status_and_export_of_independent_synthetic_attempt(sandbox,monkeypatch):
    import control as c
    assert c.status()['state']=='NOT_STARTED'
    job=r.OUT/'attempt_synthetic';job.mkdir(parents=True);r.SOURCE.mkdir(parents=True)
    (r.SOURCE/'status.json').write_text('{"state":"FAILED","synthetic_only":true}')
    r.write(r.OUT/'latest.json',{'job':job.relative_to(r.ROOT).as_posix()})
    r.write(job/'only_synthetic.json',{'synthetic_only':True});r.status(job,'STOP',PASS=False)
    assert c.status()['state']=='STOP';out=c.export()
    assert out['sha256']==r.sha(out['path'])
    with zipfile.ZipFile(out['path']) as z:
        assert json.loads(z.read('SOURCE_READONLY_FAILED_STATUS.json'))['state']=='FAILED'
        assert z.testzip() is None
    assert r.read(r.SOURCE/'status.json')['state']=='FAILED'

def test_complete_analysis_change_refused(sandbox):
    import control as c
    job=r.OUT/'attempt_synthetic';(job/'analysis').mkdir(parents=True)
    r.write(job/'analysis/analysis_manifest.json',{'files':{}})
    r.status(job,'COMPLETE',PASS=True,analysis_manifest_sha256=r.sha(job/'analysis/analysis_manifest.json'))
    assert c.check_analysis(job)['files']=={}
    (job/'analysis/analysis_manifest.json').write_text('{"files":{},"changed":true}')
    with pytest.raises(RuntimeError,match='MANIFEST_CHANGED'):c.check_analysis(job)
