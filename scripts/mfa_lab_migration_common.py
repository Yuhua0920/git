"""Migration preparation: historical bytes are read-only; no formal MFA runs."""
from pathlib import Path
import sys,os,json,time,subprocess,concurrent.futures
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b4a_preserve as p
OUT=ROOT/'outputs/mfa_lab_migration_v1';DEPLOY=ROOT/'deployment/mfa_lab_v1'
read=p.read;write=p.write;sha=p.sha

def owned(path):
    return path.is_relative_to(OUT)or path.is_relative_to(DEPLOY)or path.name.startswith(('mfa_lab_migration','test_mfa_lab_migration'))

def preserve(stage):
    OUT.mkdir(parents=True,exist_ok=True);before=OUT/'preservation_before.json'
    assert stage!='before'or not before.exists()
    files=sorted(v for d in ('src','scripts','tests','configs','docs','paper','outputs','reference','experiments')
        for v in(ROOT/d).rglob('*')if v.is_file()and not owned(v)and not(set(v.parts)&{'__pycache__','.pytest_cache','pytest_cache','pytest_tmp'}))
    hashes={};tick=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
        for f,h in zip(files,pool.map(sha,files)):
            hashes[f.relative_to(ROOT).as_posix()]=h
            if time.monotonic()-tick>25:print(json.dumps({'stage':'migration_opaque_preservation','files':len(hashes)}),flush=True);tick=time.monotonic()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain=v1','-uall'],cwd=ROOT,text=True,encoding='utf8')
    (OUT/f'git_status_{stage}.txt').write_text(status,encoding='utf8')
    check=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf8')
    write(OUT/f'git_diff_check_{stage}.json',{'exit_code':check.returncode,'stdout':check.stdout,'stderr':check.stderr});assert check.returncode==0
    seals=p.seals(hashes)
    if stage=='before':write(before,{'hashes':hashes,'seals':seals,'Git_HEAD':head})
    else:
        old=read(before);changed=[n for n,h in old['hashes'].items()if hashes.get(n)!=h];added=[n for n in hashes if n not in old['hashes']]
        result={'PASS':not changed and not added and head==old['Git_HEAD']and seals==old['seals'],
            'protected_files':len(old['hashes']),'changed':changed,'unauthorized_added':added,'Git_HEAD_unchanged':head==old['Git_HEAD'],
            'B2_B3_seals':seals,'B2_B3_values_read':0,'manuscript_drift':sum(n.startswith('paper/')for n in changed),
            'historical_production_drift':sum(n.startswith(('src/','scripts/'))for n in changed),'scientific_artifact_drift':sum(n.startswith('outputs/')for n in changed)}
        write(OUT/'preservation.json',result);assert result['PASS']
    print(json.dumps({'stage':stage,'protected_files':len(hashes),'PASS':True}),flush=True)

if __name__=='__main__':preserve(sys.argv[1])
