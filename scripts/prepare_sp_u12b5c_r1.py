"""Add-only fix package preparation; no formal evaluation entry is enabled."""
from pathlib import Path
import argparse, csv, difflib, hashlib, json, shutil, sys, zipfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'deployment/b5c_paired_v1'; NEW=ROOT/'deployment/b5c_paired_r1_v1'; OUT=ROOT/'outputs/sp_u12b5c_r1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def need(v,m):
    if not v:raise RuntimeError(m)
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def initialize():
    need(not NEW.exists(),'R1_NAMESPACE_EXISTS')
    evidence=read(OUT/'identity_original.json');need(evidence['PASS'] and evidence['real_scientific_seed_mismatches']==0,'TECHNICAL_ROOT_CAUSE_NOT_PROVEN')
    original=read(OLD/'package_manifest.json')['files'];NEW.mkdir()
    for n,v in original.items():
        q=ROOT/n;need(sha(q)==v['sha256'] and q.stat().st_size==v['bytes'],'ORIGINAL_FROZEN_DEPENDENCY_CHANGED:'+n)
        if n.startswith('deployment/b5c_paired_v1/') and q.name not in ('Install-B5C.ps1','personal_qa.py','synthetic_cost_probe.py','README_CN.md'):
            target=NEW/q.relative_to(OLD);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,target)
    changes={
      'science.py':("expected={r:index[condition,i,-1,r] for r in ROLES[:3]}","expected={r:index[condition,i,-1 if r in ROLES[:3] else 0,r] for r in ROLES}"),
      'test_b5c.py':("'channel_seeds':{r:ix['ID',0,-1,r] for r in s.ROLES[:3]}","'channel_seeds':{r:ix['ID',0,-1 if r in s.ROLES[:3] else 0,r] for r in s.ROLES}"),
      'base.py':("OUT = ROOT/'lab_runs/b5c_paired'","OUT = ROOT/'lab_runs/b5c_paired_r1'"),
      'control.py':("def start():\n    pf=preflight()","def start():\n    b.require(False,'MENTOR_NEW_ATTEMPT_REQUIRED_R1_TECHNICAL_ONLY')\n    pf=preflight()"),
      'worker.py':("    job=x.job.resolve();", "    b.require(x.action=='unit-probe','MENTOR_NEW_ATTEMPT_REQUIRED_R1_TECHNICAL_ONLY')\n    job=x.job.resolve();"),
      'B5C.ps1':("lab_runs\\b5c_paired\\controller_logs","lab_runs\\b5c_paired_r1\\controller_logs")}
    diffs=[]
    for n,(a,z) in changes.items():
        q=NEW/n;t=q.read_text(encoding='utf8');need(t.count(a)==1,'EXACT_PATCH_ANCHOR_REQUIRED:'+n)
        updated=t.replace(a,z);q.write_text(updated,encoding='utf8',newline='')
        diffs.extend(difflib.unified_diff(t.splitlines(True),updated.splitlines(True),fromfile='original/'+n,tofile='r1/'+n))
    (OUT/'minimal_fix.diff').write_text(''.join(diffs),encoding='utf8')
    # The LAB helper contains only the already reviewed metadata-audit function,
    # not personal preservation/package commands or any scientific array loader.
    source=(ROOT/'scripts/sp_u12b5c_r1_forensics.py').read_text(encoding='utf8')
    section=source[source.index('def metadata_audit('):source.index('\ndef audit():')]
    helpers='''from pathlib import Path
import hashlib, importlib.util, json, sys
sys.dont_write_bytecode=True
ROLES=('sionna_seed','torch_seed','numpy_seed','bits_seed_tx0','bits_seed_tx1','noise_seed')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def need(v,m):
 if not v:raise RuntimeError(m)
'''
    (NEW/'identity_audit.py').write_text(helpers+section,encoding='utf8')
    print(json.dumps({'created':str(NEW),'scientific_logic_changed_lines':1,'original_namespace_modified':False,'formal_run_enabled':False}))
def manifest():
    original=read(OLD/'package_manifest.json')['files'];files=dict(original)
    files['deployment/b5c_paired_v1/package_manifest.json']={'sha256':sha(OLD/'package_manifest.json'),'bytes':(OLD/'package_manifest.json').stat().st_size}
    for q in NEW.rglob('*'):
        if q.is_file() and q.name!='package_manifest.json':files[q.relative_to(ROOT).as_posix()]={'sha256':sha(q),'bytes':q.stat().st_size}
    q=NEW/'package_manifest.json'
    if q.exists():q.unlink() # Only the new, not-yet-delivered preparation namespace.
    dump(q,{'files':files,'scope':'R1 metadata-only technical correction; formal restart disabled; original B5C and failed attempt preserved'})
    print('R1 hash inventory refreshed:',len(files))
def package():
    manifest();names=sorted(q.relative_to(ROOT).as_posix() for q in NEW.rglob('*') if q.is_file())
    z=OUT/'SP_U12B5C_R1_Seed_Identity_Fix.zip';need(not z.exists(),'DELIVERY_ZIP_EXISTS')
    records={n:{'sha256':sha(ROOT/n),'bytes':(ROOT/n).stat().st_size} for n in names}
    with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as a:
        for n in names:a.write(ROOT/n,n)
    sidecar=OUT/(z.name+'.sha256');sidecar.write_text(sha(z)+'\n',encoding='ascii')
    with (OUT/'SP_U12B5C_R1_File_Manifest.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=('relative_path','bytes','sha256'));w.writeheader()
        for n,v in records.items():w.writerow({'relative_path':n,**v})
    shutil.copy2(NEW/'Install-B5C-R1.ps1',OUT/'Install-B5C-R1.ps1')
    with zipfile.ZipFile(z) as a:
        need(a.testzip() is None,'ZIP_CRC_FAILURE')
        for n,v in records.items():need(hashlib.sha256(a.read(n)).hexdigest()==v['sha256'],'ZIP_MEMBER_HASH_DRIFT')
    result={'path':str(z),'sha256':sha(z),'bytes':z.stat().st_size,'files':len(names),'unpacked_bytes':sum(v['bytes'] for v in records.values()),'all_ZIP_members_independently_verified':True,'existing_scientific_inputs_included':False,'models_included':False,'formal_restart_enabled':False}
    dump(OUT/'package_report.json',result);print(json.dumps(result,indent=2))
def refresh_identity_helper():
    source=(ROOT/'scripts/sp_u12b5c_r1_forensics.py').read_text(encoding='utf8')
    section=source[source.index('def metadata_audit('):source.index('\ndef audit():')]
    current=(NEW/'identity_audit.py').read_text(encoding='utf8')
    header=current[:current.index('def metadata_audit(')]
    (NEW/'identity_audit.py').write_text(header+section,encoding='utf8')
    print('New R1 helper refreshed; original sources unchanged')
def source_audit():
    import ast
    old=(OLD/'science.py').read_text(encoding='utf8');new=(NEW/'science.py').read_text(encoding='utf8')
    def functions(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,z=functions(old),functions(new);changed=[n for n in a if a[n]!=z[n]]
    need(changed==['proof_check'] and set(a)==set(z),'SCIENCE_FUNCTION_DRIFT')
    oldline='expected={r:index[condition,i,-1,r] for r in ROLES[:3]}'
    newline='expected={r:index[condition,i,-1 if r in ROLES[:3] else 0,r] for r in ROLES}'
    need(old.replace(oldline,newline)==new,'MORE_THAN_PROVEN_ONE_LINE_SCIENTIFIC_CHANGE')
    diffs=[];same=[]
    for q in sorted(NEW.glob('*')):
        if not q.is_file() or not (OLD/q.name).is_file():continue
        if q.suffix not in ('.py','.ps1','.md'):continue
        aa=(OLD/q.name).read_text(encoding='utf8');zz=q.read_text(encoding='utf8')
        if aa==zz:same.append(q.name)
        else:diffs.extend(difflib.unified_diff(aa.splitlines(True),zz.splitlines(True),fromfile='original/'+q.name,tofile='r1/'+q.name))
    (OUT/'all_modified_existing_code.diff').write_text(''.join(diffs),encoding='utf8')
    report={'PASS':True,'science_changed_functions':changed,'science_changed_lines':1,'unchanged_science_functions':[n for n in a if n not in changed],
      'checked_input_pilot_inputs_infer_evaluate_identical':all(a[n]==z[n] for n in ('checked_input','pilot_inputs','infer','evaluate')),
      'contract_bytes_identical':sha(OLD/'contract.json')==sha(NEW/'contract.json'),
      'data_manifest_bytes_identical':sha(OLD/'data_manifest.json')==sha(NEW/'data_manifest.json'),
      'paired_statistics_bytes_identical':sha(OLD/'paired_stats.py')==sha(NEW/'paired_stats.py'),
      'unchanged_existing_files':same,'formal_run_disabled_in_controller_and_worker':True}
    need(report['contract_bytes_identical'] and report['data_manifest_bytes_identical'] and report['paired_statistics_bytes_identical'],'PROTOCOL_OR_STATISTICS_DRIFT')
    dump(OUT/'source_audit.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=('initialize','manifest','package','source-audit','refresh-identity-helper'));x=a.parse_args();{'initialize':initialize,'manifest':manifest,'package':package,'source-audit':source_audit,'refresh-identity-helper':refresh_identity_helper}[x.action]()
