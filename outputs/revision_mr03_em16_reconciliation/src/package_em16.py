from pathlib import Path
import hashlib,json,csv,zipfile,os,subprocess,shutil,uuid
R=Path(r'D:\Users\yuhua\Desktop\Project');O=R/'outputs/revision_mr03_em16_reconciliation';S=Path(__file__).parent;MR=R/'outputs/revision_mr03_ppca_capacity_controls'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def main():
    (O/'review_reference').mkdir(exist_ok=False)
    for ds in ('V_new','D_eval'):shutil.copyfile(MR/'data'/f'bootstrap_{ds}.npz',O/'review_reference'/f'bootstrap_{ds}.npz')
    shutil.copyfile(S/'review_package_em16.py',O/'src/review_package_em16.py')
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run(['git','-C',str(R),'status','--porcelain','--untracked-files=no'],text=True,capture_output=True,env=env)
    if r.returncode:raise ValueError('GIT_READ_ONLY:'+r.stderr)
    write(O/'reports/git_status_after.json',dict(exit_code=r.returncode,stdout=r.stdout,GIT_OPTIONAL_LOCKS=0,no_commit=True,scope='Existing16 modified paper files remain; no new tracked code edits'))
    excluded={'original_model/covs.npy','original_model/inv_covs.npy','data/em16_conditional_operators.npz'}
    write(O/'reports/package_scope.json',dict(excluded=[dict(path=p,bytes=(O/p).stat().st_size,SHA256=sha(O/p)) for p in sorted(excluded)],original_transfer_ZIP_SHA256='74d2238736b4047d0dff254775e4b6cc952c2074358cccb4907ed8d5fbc6c170',original_model_preserved=True,whole_project_migration=False,small_raw_statistics_self_contained=True))
    files=sorted(p for p in O.rglob('*') if p.is_file() and p.relative_to(O).as_posix() not in excluded)
    manifest=O/'package_manifest.csv'
    with manifest.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','SHA256']);w.writeheader()
        for p in files:w.writerow(dict(path=p.relative_to(O).as_posix(),bytes=p.stat().st_size,SHA256=sha(p)))
    files.append(manifest);zp=O/'results_review.zip'
    with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,'results_review/'+p.relative_to(O).as_posix())
    zh=sha(zp)
    with (O/'results_review.zip.sha256').open('x',encoding='ascii') as f:f.write(zh+'  results_review.zip\n')
    stage=S/('mr03b_package_'+uuid.uuid4().hex[:10]);stage.mkdir()
    with zipfile.ZipFile(zp) as z:
        if z.testzip() is not None:raise ValueError('ZIP_CRC')
        for p in files:
            n='results_review/'+p.relative_to(O).as_posix()
            with z.open(n) as src:
                if hashlib.file_digest(src,'sha256').hexdigest()!=sha(p):raise ValueError('MEMBER_SHA')
        for info in z.infolist():
            dst=(stage/info.filename).resolve()
            if not dst.is_relative_to(stage.resolve()) or info.is_dir():raise ValueError('ZIP_PATH')
            dst.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,dst.open('xb') as dest:shutil.copyfileobj(src,dest)
    r=subprocess.run([str(R/'.venv-cdl/Scripts/python.exe'),'-B',str(stage/'results_review/src/review_package_em16.py'),'--out',str(stage/'results_review')],env=env,text=True,capture_output=True,timeout=120)
    if r.returncode:raise ValueError('EXTRACTED_REVIEW:'+r.stdout+r.stderr)
    review=json.loads(r.stdout.strip())
    if not review['PASS']:raise ValueError('EXTRACTED_REVIEW_NOT_PASS')
    info=dict(PASS=True,path=str(zp),bytes=zp.stat().st_size,MiB=zp.stat().st_size/1024**2,files=len(files),SHA256=zh,actual_extracted_review=review,all_members_SHA_CRC_PASS=True,extraction=str(stage),verification_record_outside_ZIP_to_avoid_circular_hash=True)
    write(O/'reports/package_verification.json',info)
    print(json.dumps(info,indent=2))
if __name__=='__main__':main()
