"""Personal-side tiny fixtures and incremental, independently hashed packaging."""
from pathlib import Path
import sys,os,json,csv,zipfile,hashlib,subprocess,argparse,shutil,ast
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import ROOT,LOCAL,write,read,sha,protocol,threads,utc,environment,scientific_counters

def frozen_sources():
    names=['src/baselines/mfa_reference_adapter.py','experiments/mfa_low_memory_prototype/portable/host_inventory.py',
           'outputs/sp_u12b5a/historical_reference.json','outputs/sp_u12b5b_r4_fidelity_manifest.json',
           'deployment/mfa_lab_v1/environment_source.json','tests/test_mfa_reference.py']
    source=read(ROOT/'outputs/sp_u12b5a/historical_reference.json')
    names +=['reference/sp_u12b5a/paper_period/'+repo+'/'+n for repo,v in source.items()for n in v['files']]
    # All project Python modules are lightweight; this includes the actual imported geometry.
    base=read(ROOT/'outputs/mfa_lab_migration_v1/package/Project_MFA_Lab/migration_manifest.json')
    names +=[n for n in base['files']if n.startswith('src/')and n.endswith('.py')]
    return {n:sha(ROOT/n)for n in sorted(set(names))}

def fixtures():
    os.environ['R5_PERSONAL_SELFTEST']='1'
    from mfa_ops import fidelity_state,seed,array_hash
    import numpy as np
    from sklearn.cluster import KMeans # load both native runtimes before checking threads
    folder=HERE/'golden';folder.mkdir(exist_ok=True);rows=[]
    with threads(1,1):
        for K in protocol()['fidelity_shape']['K']:
            for r in protocol()['fidelity_shape']['rank']:
                x,state,checks=fidelity_state(K,r);assert all(v['PASS']for v in checks)
                path=folder/f'K{K}_r{r}.npz';np.savez_compressed(path,**state)
                rows.append({'K':K,'rank':r,'N':96,'D':32,'updates':2,'input_sha256':array_hash(x),
                             'initialization_seed':seed(f'fidelity_init_K{K}_r{r}'),'file':path.name,'sha256':sha(path)})
    env=environment();env.pop('executable',None);env.pop('platform',None)
    for item in env['threadpools']:item['filepath']=Path(item['filepath']).name
    write(folder/'provenance.json',{'scope':'TINY_SYNTHETIC_AUTHOR_ONLY_NO_SCIENTIFIC_INPUT',
        'environment':env,'cases':rows,'frozen_sources':frozen_sources(),'atol':1e-10,'rtol':1e-9,**scientific_counters()})
    print(json.dumps({'tiny_fixture_cases':len(rows),'full_scale_runs':0}))

def payload():
    return sorted(p for p in HERE.rglob('*')if p.is_file()and p.name!='package_manifest.json'
                  and not any(v in('__pycache__','.pytest_cache')for v in p.relative_to(HERE).parts))
def manifest():
    files={p.relative_to(ROOT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size}for p in payload()}
    value={'task':'SP-U12B5B-R5-LAB','scope':'SYNTHETIC_QUALIFICATION_ONLY',
        'base_migration_manifest_SHA256':'26316694e87b63ce198e508547e4802095c05139fb53403b0989287979288c72',
        'files':files,'frozen_source_hashes':frozen_sources(),**scientific_counters()}
    write(HERE/'package_manifest.json',value);return value

def pack():
    precheck=read(LOCAL/'local_precheck.json');preservation=read(LOCAL/'historical_preservation.json')
    assert precheck['PASS']and preservation['PASS'],'LOCAL_GATE_NOT_PASS'
    excluded=('LOCAL_PRECHECK_SUMMARY.json','HISTORICAL_PRESERVATION_SUMMARY.json')
    checked={n:h for n,h in precheck['source_payload_hashes'].items()if Path(n).name not in excluded}
    assert checked=={p.relative_to(ROOT).as_posix():sha(p)for p in payload()
        if p.name not in('LOCAL_PRECHECK_SUMMARY.json','HISTORICAL_PRESERVATION_SUMMARY.json')},'SOURCE_CHANGED_AFTER_PRECHECK'
    summary={n:precheck[n]for n in('PASS','pytest','PowerShell_parser','tiny_cross_thread_and_seed_checks',
        'relocation','no_formal_cli','personal_full_scale_runs','actual_laboratory_tests_performed')}
    # Raw local logs remain only in the independent R5 personal preparation folder.
    write(HERE/'LOCAL_PRECHECK_SUMMARY.json',summary);write(HERE/'HISTORICAL_PRESERVATION_SUMMARY.json',preservation)
    mf=manifest();files=payload()+[HERE/'package_manifest.json']
    archive=LOCAL/'SP_U12B5_R5_Lab_Test_Package.zip'
    if archive.exists():raise FileExistsError('FINAL_PACKAGE_ALREADY_EXISTS_NO_OVERWRITE')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6)as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
    h=sha(archive);archive.with_suffix('.zip.sha256').write_text(h+'  '+archive.name+'\n',encoding='ascii')
    with zipfile.ZipFile(archive)as z:
        assert len(z.namelist())==len(files)
        for p in files:assert hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()==sha(p)
    listing=LOCAL/'SP_U12B5_R5_Lab_Test_Package_files.csv'
    with listing.open('w',newline='',encoding='utf-8-sig')as f:
        w=csv.DictWriter(f,fieldnames=('path','bytes','sha256'));w.writeheader()
        w.writerows({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}for p in files)
    report={'verdict':'R5_LAB_EXECUTION_PACKAGE_READY','archive':str(archive),'bytes':archive.stat().st_size,'MiB':archive.stat().st_size/2**20,
        'file_count':len(files),'sha256':h,'file_list':str(listing),'package_manifest_SHA256':sha(HERE/'package_manifest.json'),
        'frozen_source_files':len(mf['frozen_source_hashes']),'overwritten_historical_files':0,
        'local_precheck_PASS':True,'preservation_PASS':True,'actual_laboratory_tests_performed':False,
        'personal_full_scale_runs':0,'pending':'MENTOR_REVIEW_AND_USER_MANUAL_LAB_EXECUTION',**scientific_counters()}
    write(LOCAL/'package_report.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('fixtures','manifest','pack'));a=p.parse_args()
    {'fixtures':fixtures,'manifest':manifest,'pack':pack}[a.action]()
