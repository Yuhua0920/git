"""Personal preparation only; never included in the incremental lab package."""
from pathlib import Path
import sys,json,hashlib,zipfile,subprocess,concurrent.futures,time,csv
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'lab_runs/r7_evidence_audit_r1/preparation_personal'
ARCHIVE=ROOT/'SP_U12B5_R7_Results_20261009_114713.zip'
EXPECTED='e94832e421d1fed8dc193e46eaf562a580bae8686fabb6a94d3e46bce280955f'
EVIDENCE=Path('C:/Users/yuhua/Documents/Codex/2026-10-09/referenced-chatgpt-conversation-this-is-an/outputs')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def preservation(mode):
    files=sorted(p for d in ('src','scripts','tests','configs','docs','paper','outputs','reference','experiments','deployment','lab_runs') for p in(ROOT/d).rglob('*')if p.is_file()
      and not p.is_relative_to(HERE)and not p.is_relative_to(OUT.parent)
      and not any(n in ('.pytest_cache','pytest_tmp')or n.startswith('.venv')for n in p.parts))
    files+=sorted(p for p in ROOT.iterdir()if p.is_file());hashes={};last=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
        for p,h in zip(files,pool.map(sha,files)):
            hashes[p.relative_to(ROOT).as_posix()]=h
            if time.monotonic()-last>25:print(json.dumps({'opaque_hash_count':len(hashes),'mode':mode}),flush=True);last=time.monotonic()
    sys.path.insert(0,str(ROOT));from scripts.sp_u12b4a_preserve import seals
    seal=seals(hashes);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain=v1','-uall'],cwd=ROOT,text=True,encoding='utf8')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/f'git_status_{mode}.txt').write_text(status,encoding='utf8')
    if mode=='before':write(OUT/'historical_before.json',{'hashes':hashes,'seals':seal,'HEAD':head})
    else:
        old=read(OUT/'historical_before.json');changed=[n for n,h in old['hashes'].items()if hashes.get(n)!=h];added=[n for n in hashes if n not in old['hashes']]
        report={'PASS':not changed and not added and seal==old['seals']and head==old['HEAD'],'files':len(old['hashes']),'changed':changed,'added_outside_namespace':added,
         'B2_B3':seal,'HEAD':head,'commit_performed':False,'outcome_values_decoded':0,'host_scope':'PERSONAL_ONLY',
         'manuscript_drift':sum(n.startswith('paper/')for n in changed),'production_drift':sum(n.startswith(('src/','scripts/'))for n in changed)}
        write(OUT/'historical_preservation.json',report);assert report['PASS'],report
    print(json.dumps({'PASS':True,'mode':mode,'files':len(hashes),'B2':seal['B2']['files'],'B3':seal['B3']['files']}),flush=True)
def bindings():
    assert sha(ARCHIVE)==EXPECTED
    with zipfile.ZipFile(ARCHIVE)as z:
        names=z.namelist();assert len(names)==len(set(names))==354;assert z.testzip()is None
        manifest=json.loads(z.read('result_manifest.json'));assert set(manifest['files'])==set(names)-{'result_manifest.json'}
        live={}
        for n,v in manifest['files'].items():
            b=z.read(n);assert len(b)==v['bytes']and hashlib.sha256(b).hexdigest()==v['sha256']
            if n.startswith('r6_candidates/'):dest='lab_runs/r6_formal/candidates/'+n.removeprefix('r6_candidates/')
            elif n.startswith('r7_sequential/'):dest='lab_runs/'+n
            else:raise AssertionError(n)
            live[dest]=v
        models={r['name']:json.loads(z.read('r6_candidates/'+r['name']+'/train/model/model_manifest.json'))for r in read(ROOT/'deployment/r6_formal_v1/input_contract.json')['candidate_order']}
    audit=read(EVIDENCE/'SP_U12B5_R7_audit_evidence.json');supp=read(EVIDENCE/'SP_U12B5_R7_supplementary_evidence.json')
    assert not audit['failed_checks']and not supp['failed_checks'];assert audit['archive']['sha256']==EXPECTED
    migration=read(ROOT/'outputs/mfa_lab_migration_v1/package/Project_MFA_Lab/migration_manifest.json')
    dependencies=dict(migration['files']);dependencies['migration_manifest.json']={'sha256':sha(ROOT/'outputs/mfa_lab_migration_v1/package/Project_MFA_Lab/migration_manifest.json')}
    for stage in ('r5_128gb_v1','r6_formal_v1','r7_sequential_v1'):
        path=ROOT/'deployment'/stage/'package_manifest.json';dependencies.update(read(path)['files']);dependencies[path.relative_to(ROOT).as_posix()]={'sha256':sha(path)}
    cfg=read(ROOT/'deployment/r6_formal_v1/input_contract.json')
    for n,h in cfg['frozen_hashes'].items():dependencies[n]={'sha256':h}
    for n,h in cfg['lab_prerequisite_hashes'].items():dependencies[n]={'sha256':h}
    for n,v in cfg['data'].items():dependencies[n]=v
    for n,v in dependencies.items():assert not n.startswith(('outputs/sp_u12b2','outputs/sp_u12b3'))
    legacy={};legacy_archives={}
    for filename,prefix,fixed in [('SP_U12B5_R5_Results_20261008_193708.zip','r5_128gb','d5472254f66f8f6afa4dbac883858f5f6273e2b9d94017e3ce5b96cc2f13da45'),('SP_U12B5_R6_Results_20261008_214935.zip','r6_formal','6bd657fc52498848a17c248c910cc613bb59eca3c419b44f9846c1156f0d8002')]:
        p=ROOT/filename;assert sha(p)==fixed
        with zipfile.ZipFile(p)as z:
            m=json.loads(z.read('result_manifest.json'));assert z.testzip()is None
            assert set(z.namelist())=={prefix+'/'+n for n in m['files']}|{'result_manifest.json'}
            for n,v in m['files'].items():
                b=z.read(prefix+'/'+n);assert len(b)==v['bytes']and hashlib.sha256(b).hexdigest()==v['sha256'];legacy['lab_runs/'+prefix+'/'+n]=v
            legacy_archives[filename]={'sha256':fixed,'leaf_count':len(m['files']),'result_manifest_sha256':hashlib.sha256(z.read('result_manifest.json')).hexdigest()}
    value={'format':'R7_READ_ONLY_EVIDENCE_INPUTS_V1','archive_SHA256':EXPECTED,'archive_members':354,'archive_leaf_count':353,
      'archive_manifest_SHA256':hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
      'live_archive_files':live,'legacy_run_files':legacy,'legacy_archives':legacy_archives,'dependencies':dependencies,'models':models,'candidate_order':cfg['candidate_order'],
      'data':cfg['data'],'resources':read(ROOT/'deployment/r6_formal_v1/resource_amendment.json'),
      'validation_scope':'INHERITED_ACCEPTED_ARCHIVE_AUDIT; identity/hash preservation only; NO_RECALCULATION',
      'B2_B3_lab_scope':'NOT_MIGRATED_BY_DESIGN; PERSONAL_SEAL_VERIFICATION_SEPARATE',
      'accepted_evidence_SHA256':{n:sha(EVIDENCE/n)for n in ('SP_U12B5_R7_audit_evidence.json','SP_U12B5_R7_supplementary_evidence.json')},
      'adopted_taskbook_amendments':['archive-prefix is not physical model path','no validation metric recalculation','no laboratory B2/B3 seal assertion','memory-only serialization','no historical absence-of-access inference from declarations']}
    # Bind the original manifest bytes, rather than a reserialized JSON representation.
    with zipfile.ZipFile(ARCHIVE)as z:value['archive_manifest_SHA256']=hashlib.sha256(z.read('result_manifest.json')).hexdigest()
    if(HERE/'trusted_inputs.json').exists():
        (OUT/'trusted_inputs_development_draft1.json').write_bytes((HERE/'trusted_inputs.json').read_bytes());(HERE/'trusted_inputs.json').unlink()
    write(HERE/'trusted_inputs.json',value)
    for n in value['accepted_evidence_SHA256']:(HERE/n).write_bytes((EVIDENCE/n).read_bytes())
    write(OUT/('archive_integrity_with_legacy.json'if(OUT/'archive_integrity.json').exists()else'archive_integrity.json'),{'PASS':True,'SHA256':EXPECTED,'bytes':ARCHIVE.stat().st_size,'members':354,'verified_leaves':353,'CRC':'PASS','metric_recalculation':False,'legacy_archives':legacy_archives})
    print(json.dumps({'PASS':True,'dependencies':len(dependencies),'live_files':len(live),'models':len(models),'bindings_bytes':(HERE/'trusted_inputs.json').stat().st_size}))
def package():
    paths=sorted(p for p in HERE.iterdir()if p.is_file()and p.name not in ('prepare_personal.py','package_manifest.json','personal_qa.py'))
    files={p.relative_to(ROOT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size}for p in paths}
    write(HERE/'package_manifest.json',{'format':'R7_READONLY_AUDIT_INCREMENTAL_V1','files':files,'contains_models':False,'contains_B2_B3_outcomes':False,'contains_approval':False})
    paths.append(HERE/'package_manifest.json');dest=OUT/'package';dest.mkdir(parents=True,exist_ok=True)
    p=dest/'SP_U12B5_R7_ReadOnly_Evidence_Audit.zip'
    with zipfile.ZipFile(p,'x',compression=zipfile.ZIP_DEFLATED)as z:
        for f in paths:z.write(f,f.relative_to(ROOT).as_posix())
    h=sha(p);Path(str(p)+'.sha256').write_text(h+'  '+p.name+'\n',encoding='ascii')
    (dest/'Install-R7Audit.ps1').write_bytes((HERE/'Install-R7Audit.ps1').read_bytes())
    with(dest/'SP_U12B5_R7_Audit_File_Manifest.csv').open('x',encoding='utf-8-sig',newline='')as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows({'path':x.relative_to(ROOT).as_posix(),'bytes':x.stat().st_size,'sha256':sha(x)}for x in paths)
    with zipfile.ZipFile(p)as z:
        assert z.testzip()is None and len(z.namelist())==len(paths)==len(set(z.namelist()))
        for f in paths:assert hashlib.sha256(z.read(f.relative_to(ROOT).as_posix())).hexdigest()==sha(f)
    result={'PASS':True,'path':str(p),'bytes':p.stat().st_size,'sha256':h,'files':len(paths),'uncompressed_bytes':sum(f.stat().st_size for f in paths),'manifest_sha256':sha(HERE/'package_manifest.json')}
    write(OUT/'package_report.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
    if sys.argv[1]in ('before','after'):preservation(sys.argv[1])
    elif sys.argv[1]=='bindings':bindings()
    else:package()
