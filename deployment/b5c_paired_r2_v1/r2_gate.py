"""Bind a completed, canonical R1 identity audit; no fit or outcome reads."""
from pathlib import Path
import re
import base as b
R1_FILES=('status.json','identity_report.json','original_preservation_before.json','original_preservation_after.json')
def bindings():return b.read(b.HERE/'bindings.json')
def original_inventory():
    records={}
    for folder in (b.ROOT/'deployment/b5c_paired_v1',b.ROOT/'lab_runs/b5c_paired'):
        for q in folder.rglob('*'):
            if q.is_file():records[q.relative_to(b.ROOT).as_posix()]={'sha256':b.sha(q),'bytes':q.stat().st_size}
    return records
def old_failure():
    old=b.ROOT/'lab_runs/b5c_paired'
    for n in ('STOP.json','FORMAL_START_CONSUMED.json'):
        b.require((old/n).is_file(),'ORIGINAL_FAILED_RECORD_MISSING:'+n)
    b.require(not(old/'execution.lock').exists(),'ORIGINAL_ATTEMPT_STILL_RUNNING')
def resolve_job(name):
    q=b.safe(b.ROOT,name)
    parent=b.ROOT/'lab_runs/b5c_paired_r1/technical'
    b.require(q.parent==parent.resolve() and re.fullmatch(r'identity_[0-9a-f]{12}',q.name),'CANONICAL_R1_IDENTITY_JOB_REQUIRED')
    return q
def select_job(name=None):
    if name:return resolve_job(name)
    parent=b.ROOT/'lab_runs/b5c_paired_r1/technical'
    jobs=[q for q in parent.glob('identity_*') if re.fullmatch(r'identity_[0-9a-f]{12}',q.name)]
    ready=[]
    for q in jobs:
        p=q/'status.json'
        if p.is_file():
            s=b.read(p)
            if s.get('PASS') is True and s.get('state')=='COMPLETE':ready.append(q)
    b.require(bool(ready),'LAB_R1_IDENTITY_AUDIT_COMPLETE_PASS_REQUIRED')
    # Never choose a failed/unfinished job, and record the exact chosen receipt.
    return max(ready,key=lambda q:(q/'status.json').stat().st_mtime_ns)
def check(name=None):
    freeze=bindings();old_failure()
    for n,k in (('deployment/b5c_paired_v1/package_manifest.json','original_B5C_manifest_sha256'),
                ('deployment/b5c_paired_r1_v1/package_manifest.json','R1_manifest_sha256'),
                ('deployment/b5c_paired_v1/contract.json','contract_sha256'),
                ('deployment/b5c_paired_v1/data_manifest.json','data_manifest_sha256')):
        b.require(b.sha(b.ROOT/n)==freeze[k],'PREREQUISITE_FREEZE_HASH_DRIFT:'+n)
    q=select_job(name)
    for n in R1_FILES:b.require((q/n).is_file(),'R1_AUDIT_RECORD_MISSING:'+n)
    s=b.read(q/'status.json');r=b.read(q/'identity_report.json');a=b.read(q/'original_preservation_after.json')
    b.require(s.get('PASS') is True and s.get('state')=='COMPLETE' and s.get('formal_predictions')==0 and s.get('corrected_proof_passes')==5000,'R1_STATUS_NOT_COMPLETE_PASS')
    b.require(r.get('PASS') is True and r.get('real_scientific_seed_mismatches')==0 and r.get('formal_predictions')==0 and r.get('formal_scientific_arrays_deserialized')==0,'R1_IDENTITY_REPORT_NOT_PASS')
    b.require(r.get('lab_failed_files_opaque_hashes_preserved') is True and r.get('original_H_noise_pilot_checks_unchanged') is True,'R1_PRESERVATION_OR_INPUT_CHECK_NOT_PASS')
    for k,v in freeze['required_R1_counts'].items():b.require(type(r.get(k)) is int and r[k]==v,'R1_COVERAGE_DRIFT:'+k)
    for k,v in freeze['input_identity_digests'].items():b.require(r.get(k)==v,'R1_INPUT_IDENTITY_DIGEST_DRIFT:'+k)
    b.require(r.get('cells')==freeze['cells'],'R1_35_CELL_COVERAGE_DRIFT')
    before=b.read(q/'original_preservation_before.json')
    b.require(a.get('PASS') is True and a.get('changed')==[] and a.get('files')==len(before),'R1_ORIGINAL_PRESERVATION_NOT_PASS')
    # Authenticate old logs only as opaque bytes; no scientific content decoded.
    b.require(original_inventory()==before,'ORIGINAL_FAILURE_OR_SOURCE_CHANGED_AFTER_R1')
    h={n:b.sha(q/n) for n in R1_FILES}
    return {'PASS':True,'job':q.relative_to(b.ROOT).as_posix(),'receipt_hashes':h,
        'R1_manifest_sha256':freeze['R1_manifest_sha256'],'metadata_trajectories':5000,
        'scored_input_identities':210000,'original_failed_files_preserved':len(before),'formal_predictions':0}
def receipt_unchanged(receipt):
    b.require(receipt.get('PASS') is True and receipt.get('R1_manifest_sha256')==bindings()['R1_manifest_sha256'],'R1_FROZEN_GATE_BINDING_INVALID')
    q=resolve_job(receipt['job']);h=receipt['receipt_hashes']
    b.require(set(h)==set(R1_FILES),'R1_RECEIPT_HASH_SCHEMA_DRIFT')
    for n,v in h.items():b.require(b.sha(q/n)==v,'R1_RECEIPT_CHANGED_AFTER_PREFLIGHT:'+n)
    s=b.read(q/'status.json')
    b.require(s.get('PASS') is True and s.get('state')=='COMPLETE','R1_NO_LONGER_COMPLETE_PASS')
    return True
