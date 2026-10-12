"""Synthetic failure/resource/freeze receipts; no models or observations."""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
import recovery_base as r
import recovery_core as core
from test_recovery import sandbox,science
import r2_gate

@pytest.fixture
def receipts(sandbox,monkeypatch):
    job=r.SOURCE;job.mkdir(parents=True)
    here=sandbox/'deployment/b5c_paired_r2_v1';here.mkdir(parents=True)
    paths={'tool_manifest_sha256':here/'package_manifest.json','contract_sha256':sandbox/'deployment/b5c_paired_v1/contract.json',
           'data_manifest_sha256':sandbox/'deployment/b5c_paired_v1/data_manifest.json','model_manifest_sha256':sandbox/'lab_runs/r6_formal/FINAL_MODEL_MANIFEST.json'}
    for p in paths.values():p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    p=science.b.contract();res=p['resource'];limits={'RSS_bytes':res['RSS_GiB']*2**30,'private_commit_bytes':res['private_commit_GiB']*2**30,'min_available_physical_bytes':res['headroom_GiB']*2**30,'min_remaining_commit_bytes':res['headroom_GiB']*2**30}
    state=dict(state='FAILED',PASS=False,exit_code=1,stop_reason=None,all_descendants_kernel_contained=True,monitor_not_assumed_venv_launcher_PID=True,
               actual_interpreter_PID=1,resource_samples=1,limits=limits,blas_threads=6,openmp_threads=1,peak_RSS_bytes=1,peak_private_commit_bytes=1,
               minimum_available_physical_bytes=64*2**30,minimum_remaining_commit_bytes=64*2**30,wall_seconds=9340.88)
    freeze={k:r.sha(path) for k,path in paths.items()};freeze.update(output='lab_runs/b5c_paired_r2/formal',automatic_retry=False,
               preflight={'PASS':True,'tool_freeze_sha256':freeze['tool_manifest_sha256'],'R1_identity_audit':{'synthetic_only':True}})
    def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf8')
    put(job/'status.json',state);put(job/'exception.json',{'type':'RuntimeError','message':'B5C_READ_FIREWALL_DENIED:'+str(here/'package_manifest.json'),
               'traceback':'science.py evaluate final_identity();b.verify();b.verify_data()'})
    put(r.OLD/'STOP.json',{});put(r.OLD/'FORMAL_START_CONSUMED.json',freeze);put(job/'execution_freeze.json',freeze)
    put(job/'phase.json',dict(phase='FORMAL_EVALUATION',condition='Boundary',trajectory_id=999,MFA_predictions_complete=210000,expected=210000))
    put(job/'model_gate.json',{'identity':{'synthetic_only':True},'legal':{'PASS':True,'full_covariance_check':True},'all_components':{'PASS':True}})
    (job/'resources.jsonl').write_text(json.dumps({'process_tree':{'RSS_bytes':1,'private_commit_bytes':1},'system':{'available_physical_bytes':64*2**30,'commit_available_bytes':64*2**30}})+'\n')
    b=SimpleNamespace(HERE=here,ROOT=sandbox,contract=lambda:p)
    s=SimpleNamespace(final_identity=lambda:{'synthetic_only':True})
    class Native:
        def closed_status(self,*a):return state
        def close(self):pass
    h=SimpleNamespace(Inspection=Native)
    monkeypatch.setattr(r,'bindings',lambda:{'source_manifest_sha256':freeze['tool_manifest_sha256']})
    monkeypatch.setattr(r2_gate,'receipt_unchanged',lambda d:True)
    return b,s,h,put,state
def test_exact_failed_endpoint_and_resource_receipts_accepted(receipts):
    b,s,h,_,_=receipts;assert core.failed_end_gate(b,s,h)['state']=='FAILED'
@pytest.mark.parametrize('mode',['other_error','not_end_verify','resource_stop','peak_exceeded','sample_count','model_gate','phase_incomplete','frozen_hash','consumption_changed'])
def test_unrecoverable_end_evidence_rejected(receipts,mode):
    b,s,h,put,state=receipts
    if mode=='other_error':put(r.SOURCE/'exception.json',{'type':'RuntimeError','message':'SOME_OTHER_ERROR','traceback':''})
    elif mode=='not_end_verify':
        d=r.read(r.SOURCE/'exception.json');d['traceback']='before inference';put(r.SOURCE/'exception.json',d)
    elif mode=='resource_stop':state['stop_reason']='NATIVE_FINAL_PEAK_RSS_EXCEEDED_LIMIT';put(r.SOURCE/'status.json',state)
    elif mode=='peak_exceeded':state['peak_RSS_bytes']=49*2**30;put(r.SOURCE/'status.json',state)
    elif mode=='sample_count':state['resource_samples']=2;put(r.SOURCE/'status.json',state)
    elif mode=='model_gate':d=r.read(r.SOURCE/'model_gate.json');d['legal']['PASS']=False;put(r.SOURCE/'model_gate.json',d)
    elif mode=='phase_incomplete':d=r.read(r.SOURCE/'phase.json');d['MFA_predictions_complete']=209994;put(r.SOURCE/'phase.json',d)
    elif mode=='frozen_hash':(b.HERE/'package_manifest.json').write_text('{"changed":true}')
    elif mode=='consumption_changed':put(r.OLD/'FORMAL_START_CONSUMED.json',{})
    with pytest.raises(RuntimeError):core.failed_end_gate(b,s,h)
