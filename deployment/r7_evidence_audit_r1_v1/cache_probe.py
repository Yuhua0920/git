"""Isolated analytic cache/firewall probe; never reads personal scientific payloads."""
from pathlib import Path
import sys,json,hashlib,shutil,py_compile,importlib.util,importlib.machinery,traceback,os,marshal,struct
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(job,mode,case='valid'):
    job=Path(job).resolve();job.mkdir(parents=True,exist_ok=False);root=job/'p';tool=root/'deployment/r7_evidence_audit_r1_v1';tool.mkdir(parents=True)
    cfg=json.loads((HERE/'trusted_inputs.json').read_text(encoding='utf8'));bound={}
    for n,v in cfg['dependencies'].items():
        if n.endswith('.py') or n in ('outputs/sp_u12b5a/historical_reference.json','deployment/mfa_lab_v1/environment_source.json'):
            src=ROOT/n;dst=root/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes());bound[n]=v
    original=ROOT/'deployment/r7_evidence_audit_v1/audit_core.py'
    for name in ('audit_core.py','synthetic.py'):
        source=(original if name=='audit_core.py' and mode=='legacy' else HERE/name)
        (tool/name).write_bytes(source.read_bytes())
    if mode=='fixed':
        for n in ('source_only_loader.py','package_manifest.json','trusted_inputs.json'):(tool/n).write_bytes((HERE/n).read_bytes())
    sys.path[:0]=[str(tool),str(root),str(root/'deployment/r6_formal_v1')]
    import audit_core as a
    a.ROOT=root;a.HERE=tool;a.OUT=root/'lab_runs/r7_evidence_audit_r1'
    # fixture dependency contract retains original hashes but only tiny code/metadata exists.
    cfg['dependencies']=bound
    if mode=='fixed':
        (tool/'trusted_inputs.json').write_text(json.dumps(cfg),encoding='utf8')
        # Authenticate probe copies independently. Never represents an installed lab package.
        manifest={'files':{p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}for p in tool.iterdir()if p.is_file() and p.name!='package_manifest.json'}}
        (tool/'package_manifest.json').write_text(json.dumps(manifest),encoding='utf8')
    import synthetic
    row={'name':'K2_r4','K':2,'rank':4,'seed':1443235906};d=root/'lab_runs/r6_formal/candidates/K2_r4/train/model';expected=synthetic.fixture(d,row)
    utils=root/'reference/sp_u12b5a/paper_period/cplx-mfa/utils.py';cache=Path(importlib.util.cache_from_source(str(utils)))
    if case!='absent':py_compile.compile(str(utils),doraise=True)
    if case=='poisoned':
        cache.write_bytes(importlib.util.MAGIC_NUMBER+struct.pack('<III',0,int(utils.stat().st_mtime),utils.stat().st_size)+marshal.dumps(compile("raise RuntimeError('UNAUTH_PYC_EXECUTED')",str(utils),'exec')))
    elif case=='corrupt':cache.write_bytes(b'UNAUTHENTICATED BROKEN CACHE')
    elif case=='wrong_source':utils.write_bytes(utils.read_bytes()+b'\n# changed source\n')
    cache_before={'sha256':sha(cache),'mtime_ns':cache.stat().st_mtime_ns}if cache.exists()else{'sha256':None,'mtime_ns':None};opened=[]
    def observer(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and not (args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):opened.append(os.fsdecode(args[0]))
    sys.addaudithook(observer)
    # Old frozen modules imported before guard, exactly as laboratory worker.
    _,common,_=a.original_modules();common.environment()
    work=a.OUT/'probe';work.mkdir(parents=True);cfg['candidate_order']=[row];cfg['live_archive_files']={x.relative_to(root).as_posix():{'sha256':sha(x)} for x in (d/'model_manifest.json',d.parent/'iteration_history.jsonl')};cfg['legacy_run_files']={}
    prep=root/'outputs/sp_u12b5b/selection_train.npy';prep.parent.mkdir(parents=True,exist_ok=True)
    import numpy as np
    np.save(prep,np.ones(2));cfg['dependencies'][prep.relative_to(root).as_posix()]={'sha256':sha(prep)}
    blocked=root/'outputs/sp_u12b2b_sealed/sentinel.txt';blocked.parent.mkdir(parents=True,exist_ok=True);blocked.write_text('SYNTHETIC ONLY')
    b3=root/'outputs/sp_u12b3g_next_sealed/sentinel.txt';b3.parent.mkdir(parents=True,exist_ok=True);b3.write_text('SYNTHETIC ONLY')
    unregistered=utils.parent/'unregistered.py';unregistered.write_text('x=1')
    a.install_guard(root,cfg,work)
    error=None;result=None
    try:
        if mode=='fixed' and case=='wrong_source':importlib.machinery.SourceFileLoader('bad',str(utils)).get_code('bad')
        result=a.model_check(d,row,expected,True)
    except BaseException as e:error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
    firewall={}
    if mode=='fixed' and error is None:
        actions={'preparation_decode':lambda:np.load(prep,allow_pickle=False),
          'B2_read':lambda:blocked.read_bytes(),'B3_read':lambda:b3.read_bytes(),
          'model_overwrite':lambda:(d/'means.npy').write_bytes(b'X'),
          'model_write_new':lambda:(d/'new.npy').write_bytes(b'X'),
          'cache_write':lambda:cache.write_bytes(b'X'),'cache_delete':lambda:cache.unlink(),
          'unregistered_source':lambda:importlib.machinery.SourceFileLoader('unknown',str(unregistered)).get_code('unknown')}
        for label,action in actions.items():
            try:action();firewall[label]={'rejected':False}
            except BaseException as e:firewall[label]={'rejected':True,'error':str(e)}
    # Opaque hashing is allowed only for explicit evidence; no numeric decoding.
    cache_after={'sha256':None,'mtime_ns':cache.stat().st_mtime_ns if cache.exists()else None}
    # cache is deliberately unregistered, so do not relax audit hook for hash read.
    state={'mode':mode,'case':case,'synthetic_only':True,'real_lab_models_verified':0,'dont_write_bytecode':sys.dont_write_bytecode,
      'loader_class':'SourceFileLoader','cache_path':str(cache),'cache_before':cache_before,'cache_after':cache_after,
      'cache_read_attempts':[x for x in opened if x==str(cache)],'error':error,'complete_synthetic_models':int(result is not None),
      'model_check':result,'firewall':firewall,'source_loader':getattr(sys.modules.get('source_only_loader'),'observations',[])}
    a.write(work/'probe_result.json',state,True)
    print(json.dumps(state,ensure_ascii=False,allow_nan=False),flush=True)
    return 0
if __name__=='__main__':sys.exit(run(sys.argv[1],sys.argv[2],sys.argv[3]if len(sys.argv)>3 else'valid'))
