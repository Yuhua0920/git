"""Personal preparation only; never included in the incremental lab package."""
from pathlib import Path
import sys,json,hashlib,zipfile,subprocess,concurrent.futures,time,csv
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'lab_runs/r8_selection/preparation_personal'
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
        name='historical_delivery_preservation.json'if mode=='delivery'else'historical_final_preservation.json'if mode=='final'else'historical_preservation.json'
        write(OUT/name,report);assert report['PASS'],report
    print(json.dumps({'PASS':True,'mode':mode,'files':len(hashes),'B2':seal['B2']['files'],'B3':seal['B3']['files']}),flush=True)
if __name__=='__main__':preservation(sys.argv[1])
