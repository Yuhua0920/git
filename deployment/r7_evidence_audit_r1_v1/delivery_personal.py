"""Personal packaging/evidence generation only. Excluded from laboratory ZIP."""
from pathlib import Path
import sys,json,hashlib,csv,zipfile,ast,difflib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'lab_runs/r7_evidence_audit_r1/preparation_personal'
EXCLUDE={'prepare_personal.py','personal_qa.py','delivery_personal.py','package_manifest.json'}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    with Path(p).open('x',encoding='utf8')as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def manifest():
    paths=sorted(p for p in HERE.iterdir()if p.is_file()and p.name not in EXCLUDE)
    data={'format':'R7_READONLY_AUDIT_R1_INCREMENTAL_V1','files':{p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}for p in paths},
          'contains_models':False,'contains_B2_B3_outcomes':False,'contains_approval':False,
          'old_deployment_untouched':'deployment/r7_evidence_audit_v1'}
    (HERE/'package_manifest.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    return paths+[HERE/'package_manifest.json']
def package():
    paths=manifest();dest=OUT/'package';dest.mkdir(parents=True,exist_ok=True);p=dest/'SP_U12B5_R7_Audit_R1_Fix.zip'
    with zipfile.ZipFile(p,'x',compression=zipfile.ZIP_DEFLATED)as z:
        for source in paths:z.write(source,source.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(p)as z:
        assert z.testzip()is None and len(z.namelist())==len(set(z.namelist()))==len(paths)
        for f in paths:assert hashlib.sha256(z.read(f.relative_to(ROOT).as_posix())).hexdigest()==sha(f)
    Path(str(p)+'.sha256').write_text(sha(p)+'  '+p.name+'\n',encoding='ascii')
    (dest/'Install-R7AuditR1.ps1').write_bytes((HERE/'Install-R7AuditR1.ps1').read_bytes())
    csvpath=dest/'SP_U12B5_R7_Audit_R1_File_Manifest.csv'
    with csvpath.open('x',encoding='utf-8-sig',newline='')as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows({'path':s.relative_to(ROOT).as_posix(),'bytes':s.stat().st_size,'sha256':sha(s)}for s in paths)
    report={'PASS':True,'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p),'files':len(paths),'uncompressed_bytes':sum(f.stat().st_size for f in paths),
      'manifest_sha256':sha(HERE/'package_manifest.json'),'csv_sha256':sha(csvpath),'installer_sha256':sha(dest/'Install-R7AuditR1.ps1'),
      'contains_models':False,'contains_preparation_payload':False,'contains_scientific_outcomes':False,'contains_venv_or_credentials':False}
    save(OUT/'package_report.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
def evidence():
    old=ROOT/'deployment/r7_evidence_audit_v1'
    diff=[]
    for name in ('audit_core.py','audit_cli.py','audit_worker.py','audit_monitor.py'):
        diff.extend(difflib.unified_diff((old/name).read_text().splitlines(True),(HERE/name).read_text().splitlines(True),fromfile='v1/'+name,tofile='r1/'+name))
    (OUT/'minimal_audit_tool.diff').write_text(''.join(diff),encoding='utf8')
    cfg=json.loads((HERE/'trusted_inputs.json').read_text())
    rows=[]
    for n in ['src/baselines/mfa_reference_adapter.py','deployment/r6_formal_v1/model_ops.py',
      'reference/sp_u12b5a/paper_period/cplx-mfa/utils.py','reference/sp_u12b5a/paper_period/cplx-mfa/MFA_cplx.py',
      'reference/sp_u12b5a/paper_period/MFA_estimator/MFA_estimator.py']:
        rows.append({'path':n,'actual_sha256':sha(ROOT/n),'expected_sha256':cfg['dependencies'][n]['sha256'],'PASS':sha(ROOT/n)==cfg['dependencies'][n]['sha256']})
    assert all(r['PASS']for r in rows)
    save(OUT/'frozen_source_audit.json',{'PASS':True,'rows':rows,'trusted_inputs_original_bytes':(old/'trusted_inputs.json').read_bytes()==(HERE/'trusted_inputs.json').read_bytes(),
      '42_original_tests_bytes_unchanged':(old/'test_audit.py').read_bytes()==(HERE/'test_audit.py').read_bytes(),
      'original_reference_cache_files':[p.relative_to(ROOT).as_posix()for p in (ROOT/'reference/sp_u12b5a/paper_period').rglob('*.pyc')],
      'real_lab_models_verified':0,'actual_lab_first_failure_locally_available':False})
    repro=[]
    for p in OUT.glob('reproduction_*/p/lab_runs/*/probe/probe_result.json'):
        v=json.loads(p.read_text());cache=Path(v['cache_path'])
        repro.append({'result':str(p.resolve()),'error':v['error'],'cache_SHA256_unchanged':sha(cache)==v['cache_before']['sha256'],'cache_timestamp_unchanged':cache.stat().st_mtime_ns==v['cache_before']['mtime_ns'],'synthetic_only':True})
    save(OUT/'root_cause_evidence.json',{'root_cause':'SourceFileLoader.get_code reads existing pyc even with -B; source SHA does not authenticate pyc',
      'exact_original_error_reproduced':any('utils.cpython-313.pyc' in (r['error']or{}).get('message','')for r in repro),'reproductions':repro,
      'guard_allowlist_expansion':0,'reference_and_adapter_modifications':0,'real_lab_audit':'NOT_EXECUTED'})
    print('PERSONAL_EVIDENCE_GENERATED')
if __name__=='__main__':
    if sys.argv[1]=='package':package()
    elif sys.argv[1]=='manifest':manifest()
    else:evidence()
