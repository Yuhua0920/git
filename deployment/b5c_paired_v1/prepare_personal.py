"""Prepare contracts and opaque transport. Never decode B2/B3 outcomes."""
from pathlib import Path
import argparse,csv,hashlib,json,shutil,sys,zipfile
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE)); import base as b
DEST=ROOT/'outputs/sp_u12b5c_preparation'
def dump(path,v):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def inventory():
    technical=b.read(ROOT/'outputs/sp_u12b2b_sealed_manifest.json')
    config=b.read(ROOT/'configs/sp_u12b2b_sealed.json')
    files={}; origins=technical['files']
    for condition in ('ID','Profile','Mobility','Composed','Boundary'):
        for i in range(1000):
            for ext in ('npz','json'):
                for name in (f'cache/{condition}/H_{i:04d}.{ext}',f'cache/{condition}/comm_{i:04d}.{ext}',f'raw_evaluation/{condition}/{i:04d}.{ext}'):
                    original='outputs/sp_u12b2b_sealed/'+name
                    # Technical manifest uses paths relative to the sealed root.
                    key=original if original in origins else name
                    b.require(key in origins,'B2_TECHNICAL_INVENTORY_MISSING:'+name)
                    v=origins[key]; files[original]={'sha256':v['sha256'],'bytes':v['bytes']}
    for n in ('configs/sp_u12b2b_sealed.json','configs/sp_u12b2a_snr_sweep.yaml','outputs/sp_u12b2a_seed_registry.json','outputs/sp_u12b2a/operators.npz','outputs/sp_u12b2b_sealed_manifest.json','outputs/sp_u12b2b/execution_freeze.json','outputs/sp_u12b2b/schema.json','docs/sp_u12b2b_protocol.md'):
        p=ROOT/n; files[n]={'sha256':b.sha(p),'bytes':p.stat().st_size}
    b.require(len(files)==30008,'DATA_INVENTORY_COUNT')
    b.require(config['N_evaluation_per_condition']==1000 and config['rank']==4 and config['window']==5 and config['folds']==4,'B2_PROTOCOL_CONFLICT')
    return files,config
def freeze():
    DEST.mkdir(parents=True,exist_ok=True)
    top=ROOT/'FINAL_MODEL_MANIFEST.json'; h=b.sha(top)
    b.require(h==(ROOT/'FINAL_MODEL_MANIFEST.sha256').read_text(encoding='ascii').strip(),'FINAL_MANIFEST_HASH_MISMATCH')
    m=b.read(top);s=m['selected']; b.require(m['PASS'] and s['K']==1 and s['rank']==16 and m['final_seed']==4143049057,'FINAL_IDENTITY_MISMATCH')
    files,cfg=inventory()
    dump(HERE/'data_manifest.json',{'schema':'B5C_MINIMAL_ORIGINAL_BYTES_V1','files':files,'original_performance_decoded':False})
    code={}
    names=list((ROOT/'src').rglob('*.py'))+list((ROOT/'deployment/r6_formal_v1').glob('*.py'))
    names += [ROOT/'deployment/r6_formal_v1/resource_amendment.json',ROOT/'deployment/r6_formal_v1/input_contract.json',ROOT/'experiments/mfa_low_memory_prototype/portable/host_inventory.py',ROOT/'deployment/r7_evidence_audit_r1_v1/source_only_loader.py',ROOT/'outputs/sp_u12b5a/historical_reference.json',ROOT/'reference/sp_u12b5a/paper_period/cplx-mfa/LICENSE',ROOT/'reference/sp_u12b5a/paper_period/MFA_estimator/LICENSE']
    names += [ROOT/'reference/sp_u12b5a/paper_period'/repo/n for repo,ns in [('cplx-mfa',('MFA_cplx.py','utils.py')),('MFA_estimator',('MFA_estimator.py',))] for n in ns]
    for p in names: code[p.relative_to(ROOT).as_posix()]={'sha256':b.sha(p),'bytes':p.stat().st_size}
    r6=b.read(ROOT/'deployment/r6_formal_v1/input_contract.json')['frozen_hashes']
    for n,v in code.items():
        if n in r6:b.require(v['sha256']==r6[n],'EXISTING_R6_SCIENCE_DRIFT:'+n)
    seedreg=b.read(ROOT/'outputs/sp_u12b2a_seed_registry.json')
    b.require([r['seed'] for r in seedreg['rows'] if r['role']=='bootstrap_master_seed']==[2997555629],'ORIGINAL_BOOTSTRAP_MASTER_DRIFT')
    b2a=b.read(ROOT/'configs/sp_u12b2a_snr_sweep.yaml')
    b.require(b2a['sample_rate_hz']==b2a['nfft']*b2a['delta_f_hz'] and b2a['source_receive_power']==9.941134822523972,'SAMPLING_OR_POWER_CONTRACT_DRIFT')
    contract={'schema':'FROZEN_B5C_PAIRED_V1','final_manifest_sha256':h,'final_model_manifest_sha256':m['final_model_manifest_sha256'],
       'training_result_sha256':m['training_result_sha256'],'full_training_data_sha256':m['full_data_sha256'],
       'selection_sha256':m['SELECTION_MANIFEST_SHA256'],'selected':{'K':1,'rank':16,'final_seed':4143049057,'iterations':22,'convergence':'CONVERGED'},
       'model_file_names':['means.npy','lambdas.npy','psis.npy','amps.npy','covs.npy','inv_covs.npy','training_history.npy'],
       'conditions':cfg['conditions'],'snrs':[0,5,10,15,20,25,30],'N':1000,'score_frames':list(range(4,10)),
       'power':9.941134822523972,'variance_rule':'power / 10**(snr/10)','LR4_index':5,
       'B2_methods':['LS-LIN','Source','Hard','Convex','Dense','LR4','Matched'],'rank':4,'window':5,'folds':4,
       'fold_hash':cfg['fold_hash'],'geometry_order_sha256':m['inference_config']['geometry_order_sha256'],
       'statistics':{'replicates':100000,'family_size':35,'FWER':.05,'two_sided':True,'tail_probability':.05/70,'master_seed':2997555629,'chunk':1000,'quantile_method':'linear','difference':'MFA NMSE dB minus LR4 NMSE dB; positive favors LR4'},
       'resource':{'blas':6,'openmp':1,'serial_workers':1,'RSS_GiB':48,'private_commit_GiB':64,'headroom_GiB':32,'launch_available_GiB':80,'timeout_seconds':604800},
       'runtime_endpoint':'one original adapter call per trajectory-SNR, six scored pilot rows; includes noise-dependent posterior setup; no cross-hardware speedup',
       'replay_identity':'byte-authenticated original H/comm; exact array hashes of H/truth/pilot/Y/noise vs original proof; stored LR4 frame and trajectory energies',
       'source_dependencies':code,'data_manifest_sha256':b.sha(HERE/'data_manifest.json'),'automatic_retry':False,'formal_on_personal':False}
    dump(HERE/'contract.json',contract)
    dump(DEST/'model_identity_personal.json',{'PASS':True,'top_manifest_SHA256':h,'sidecar_matches':True,'K':1,'rank':16,'final_seed':4143049057,
       'subordinate_model_manifest_bound_SHA256':m['final_model_manifest_sha256'],'seven_NPY_physically_verified_on_personal':False,'iterations_and_convergence_verified_on_personal':False,'required_lab_gate':True})
    with (DEST/'SP_U12B5C_Data_Manifest.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=('relative_path','bytes','sha256'));w.writeheader()
        for n,v in files.items():w.writerow({'relative_path':n,**v})
    print(json.dumps({'files':len(files),'bytes':sum(v['bytes'] for v in files.values()),'code_dependencies':len(code),'manifest_sha256':h}))
def manifest():
    files=dict(b.contract()['source_dependencies'])
    for p in HERE.rglob('*'):
        if p.is_file() and p.name not in ('package_manifest.json','prepare_personal.py'):
            files[p.relative_to(ROOT).as_posix()]={'sha256':b.sha(p),'bytes':p.stat().st_size}
    dump(HERE/'package_manifest.json',{'files':files,'scope':'B5C evaluation only; unchanged dependency bytes; add-only installation'})
    return files
def package():
    b.require(not(DEST/'SP_U12B5C_Execution.zip').exists(),'PACKAGE_ALREADY_EXISTS')
    files=manifest()
    files['deployment/b5c_paired_v1/package_manifest.json']={'sha256':b.sha(HERE/'package_manifest.json'),'bytes':(HERE/'package_manifest.json').stat().st_size}
    with zipfile.ZipFile(DEST/'SP_U12B5C_Execution.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,v in files.items():
            b.require(b.sha(ROOT/n)==v['sha256'],'SOURCE_CHANGED_AT_PACKAGING');z.write(ROOT/n,n)
    p=DEST/'SP_U12B5C_Execution.zip'; (DEST/(p.name+'.sha256')).write_text(b.sha(p)+'\n',encoding='ascii')
    with (DEST/'SP_U12B5C_File_Manifest.csv').open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=('relative_path','bytes','sha256'));w.writeheader()
        for n,v in files.items():w.writerow({'relative_path':n,**v})
    shutil.copy2(HERE/'Install-B5C.ps1',DEST/'Install-B5C.ps1')
    dump(DEST/'execution_package.json',{'path':str(p),'bytes':p.stat().st_size,'files':len(files),'sha256':b.sha(p),'formal_predictions':0})
    print((DEST/'execution_package.json').read_text())
def data_package():
    # The frozen tool/contract must exist before handling historical outcome bytes.
    b.verify(); files=b.data_inventory();p=DEST/'SP_U12B5C_B2_Minimal_Data.zip'
    b.require(not p.exists(),'DATA_PACKAGE_ALREADY_EXISTS')
    # np.load and JSON decoding are never used on original raw_evaluation files.
    with zipfile.ZipFile(p,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for i,(n,v) in enumerate(files.items()):
            q=ROOT/n; b.require(q.stat().st_size==v['bytes'] and b.sha(q)==v['sha256'],'ORIGINAL_BYTE_HASH_MISMATCH:'+n)
            z.write(q,n)
            if i%3000==0:print('Opaque files packaged:',i,flush=True)
    (DEST/(p.name+'.sha256')).write_text(b.sha(p)+'\n',encoding='ascii')
    dump(DEST/'data_package.json',{'path':str(p),'bytes':p.stat().st_size,'files':len(files),'sha256':b.sha(p),'decoded_performance_files':0})
    print((DEST/'data_package.json').read_text())
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=('freeze','manifest','package','data-package'));x=a.parse_args()
    {'freeze':freeze,'manifest':manifest,'package':package,'data-package':data_package}[x.action]()
