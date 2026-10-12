"""Personal-side opaque historical preservation; never numeric outcome reads."""
from pathlib import Path
import sys,subprocess,time,concurrent.futures,json
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import ROOT,HERE,OUT,LOCAL,write,read,sha
from scripts import sp_u12b4a_preserve as legacy

def run(mode):
    LOCAL.mkdir(parents=True,exist_ok=True)
    before=LOCAL/'historical_before.json'
    assert mode!='before'or not before.exists()
    files=sorted(p for folder in('src','scripts','tests','configs','docs','paper','outputs','reference','experiments','deployment')
                 for p in(ROOT/folder).rglob('*')if p.is_file()and not p.is_relative_to(HERE)
                 and not any(v in('__pycache__','.pytest_cache','pytest_tmp')or v.startswith('.venv')for v in p.parts))
    hashes={};tick=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
        for path,h in zip(files,pool.map(sha,files)):
            hashes[path.relative_to(ROOT).as_posix()]=h
            if time.monotonic()-tick>25:
                print(json.dumps({'stage':'R5_opaque_preservation','files':len(hashes)}),flush=True);tick=time.monotonic()
    seals=legacy.seals(hashes)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain=v1','-uall'],cwd=ROOT,text=True,encoding='utf8')
    (LOCAL/f'git_status_{mode}.txt').write_text(status,encoding='utf8')
    if mode=='before':write(before,{'hashes':hashes,'B2_B3':seals,'Git_HEAD':head})
    else:
        old=read(before);changed=[n for n,h in old['hashes'].items()if hashes.get(n)!=h]
        added=[n for n in hashes if n not in old['hashes']]
        diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf8')
        result={'PASS':not changed and not added and seals==old['B2_B3']and head==old['Git_HEAD']and diff.returncode==0,
            'protected_files':len(old['hashes']),'changed':changed,'unauthorized_added':added,'B2_B3':seals,
            'Git_HEAD_unchanged':head==old['Git_HEAD'],'Git_diff_check':diff.returncode,'Git_commit':False,
            'manuscript_changes':sum(n.startswith('paper/')for n in changed),
            'reference_adapter_changes':sum(n.startswith(('reference/','src/baselines/mfa_reference_adapter'))for n in changed),
            'historical_artifact_changes':sum(n.startswith('outputs/')for n in changed),'scientific_values_decoded':0}
        write(LOCAL/'historical_preservation.json',result);assert result['PASS'],result
    print(json.dumps({'PASS':True,'stage':mode,'protected_files':len(hashes)}),flush=True)
if __name__=='__main__':run(sys.argv[1])
