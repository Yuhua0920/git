"""Outcome-blind SP-U7A identity/schema/seed audit. No simulator imports.

Numeric outcome payloads are never decoded. JSON numeric tokens are converted
only under seed fields; CSV projections only retain identity and seed columns;
NPZ numeric members are opened solely for NPY headers, never array payloads.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import io
import json
import re
import subprocess
import struct
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
STRONG = re.compile(r'\b(?:CDL[-_ ]?A|CDLA|cdl_a|A[_ -]?120)\b|profile[ _-]+A\b', re.I)
BROAD = re.compile(r'CDL[-_ ]?A|CDLA|profile[ _-]*A|A[_ -]?120', re.I)
ENCODED = re.compile(r'''["']?(?:profile|channel_profile|cdl_profile|model)["']?\s*[:=]\s*["']A["']''', re.I)
TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null|[{}\[\]:,]')
TEXT_EXT = {'.py','.json','.jsonl','.csv','.md','.tex','.txt','.log','.yaml','.yml','.toml','.ps1','.diff','.source','.html','.bib'}
IDENTITY = re.compile(r'profile|environment|condition|channel_model|namespace|split|stage|construction|purpose|model',re.I)
OUTCOME_FIELD = re.compile(r'nmse|\bber\b|(?:error|truth|signal|noise)_energy|bit_errors|total_bits|gain|penalty|metric|loss|coefficient|oracle',re.I)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def seed_key(k):
    k=k.lower()
    return (k in {'seed','seeds','forbidden_seeds','seed_literals','random_seed'} or k.endswith(('_seed','_seeds','_seed_base'))
            or k.startswith('bits_seed_tx') or k=='seed_namespace_id') and not any(x in k for x in ('count','sha','hash'))

def project_json(text):
    """Lexical projection. No json.load of a historical report or outcome.

    Strings may be identity metadata. Unselected numeric tokens are discarded
    without int/float conversion. Container ancestry preserves seed arrays.
    """
    stack=[];pending=None;identities=set();seeds=set();namespaces=set();strong=[]
    tokens=iter(TOKEN.finditer(text))
    for m in tokens:
        t=m.group()
        if t=='{': stack.append(('object',pending));pending=None
        elif t=='[': stack.append(('array',pending));pending=None
        elif t in ('}',']'): 
            if stack:stack.pop()
            pending=None
        elif t in (':',','): 
            if t==',':pending=None
        elif t.startswith('"'):
            s=json.loads(t)
            if STRONG.search(s): strong.append({'token':STRONG.search(s).group(),'package_path':bool(re.search(r'site-packages[/\\].*models[/\\].*CDL-A\.json',s,re.I))})
            # A key is recognized by the next non-whitespace character.
            p=m.end()
            while p<len(text) and text[p].isspace():p+=1
            if p<len(text) and text[p]==':':pending=s;continue
            parents=[v for _,v in stack if v is not None];keys=parents+([pending] if pending else [])
            if pending and IDENTITY.search(pending):
                if len(s)<300:identities.add((pending,s))
                if 'namespace' in pending.lower():namespaces.add(s)
            if any(seed_key(k) for k in keys) and not any(OUTCOME_FIELD.search(k) for k in keys) and re.fullmatch(r'\d+',s):seeds.add(int(s))
            pending=None
        else:
            keys=[v for _,v in stack if v is not None]+([pending] if pending else [])
            if any(seed_key(k) for k in keys) and not any(OUTCOME_FIELD.search(k) for k in keys) and re.fullmatch(r'\d+',t):seeds.add(int(t))
            pending=None
    return {'identities':sorted(identities),'seeds':seeds,'namespaces':namespaces,'strong':strong}

def csv_metadata(text):
    f=io.StringIO(text);reader=csv.DictReader(f);headers=reader.fieldnames or []
    keep=[k for k in headers if IDENTITY.search(k) or seed_key(k)]
    identities=set();seeds=set();namespaces=set();candidate=[];rows=0
    for row in reader:
        rows+=1
        for k in keep:
            v=row.get(k) or ''
            if seed_key(k) and re.fullmatch(r'\d+',v):seeds.add(int(v))
            elif IDENTITY.search(k) and v:
                identities.add((k,v))
                if 'namespace' in k.lower():namespaces.add(v)
                if STRONG.search(v) or ('profile' in k.lower() and v.upper()=='A'):candidate.append((k,v))
    return {'headers':headers,'selected_columns':keep,'row_count':rows,'identities':sorted(identities),'seeds':seeds,'namespaces':namespaces,'candidate':sorted(set(candidate))}

def archive_metadata(source):
    members=[];identities=[];seeds=set();namespaces=set();hits=[]
    with zipfile.ZipFile(source) as z:
        for info in z.infolist():
            if not info.filename.endswith('.npy'):continue
            with z.open(info) as f:
                v=np.lib.format.read_magic(f)
                if v==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f)
                elif v==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f)
                elif v==(3,0):
                    size=struct.unpack('<I',f.read(4))[0]
                    if size>10000:raise ValueError('oversized NPY header')
                    header=ast.literal_eval(f.read(size).decode('utf-8'))
                    shape,order,dtype=header['shape'],header['fortran_order'],np.dtype(header['descr'])
                else:raise ValueError('unsupported NPY header version')
            members.append({'name':info.filename,'shape':list(shape),'dtype':str(dtype),'fortran_order':order})
            stem=Path(info.filename).stem
            # Only explicit string metadata, never numeric/pickled arrays.
            if dtype.kind in 'US' and (IDENTITY.search(stem) or stem in {'metadata_json','manifest_json'}):
                if info.file_size>16*1024*1024:raise ValueError('oversized metadata: '+info.filename)
                with z.open(info) as f:a=np.lib.format.read_array(f,allow_pickle=False)
                for s in a.reshape(-1):
                    s=str(s)
                    if stem.endswith('_json'):
                        p=project_json(s);identities+=p['identities'];seeds|=p['seeds'];namespaces|=p['namespaces'];hits+=p['strong']
                    else:
                        identities.append((stem,s))
                        if STRONG.search(s) or ('profile' in stem.lower() and s.upper()=='A'):hits.append({'token':s,'package_path':False})
            elif dtype.kind=='O' and (IDENTITY.search(stem) or stem.endswith('_json')):
                raise ValueError('unsafe object metadata: '+info.filename)
    return {'members':members,'identities':identities,'seeds':seeds,'namespaces':namespaces,'strong':hits}

def source_seeds(text):
    """Literal RNG/seed declarations only; no importing historical runners."""
    found=set()
    try:tree=ast.parse(text)
    except (SyntaxError,ValueError):return found
    for n in ast.walk(tree):
        if isinstance(n,(ast.Assign,ast.AnnAssign)):
            names=[x.id for x in ast.walk(n) if isinstance(x,ast.Name)]
            if any(seed_key(k) for k in names):
                value=n.value
                for x in ast.walk(value) if value is not None else ():
                    if isinstance(x,ast.Constant) and type(x.value)==int and 0<=x.value<2**32:found.add(x.value)
        if isinstance(n,ast.Call):
            fn=n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ''
            if fn in {'default_rng','SeedSequence','manual_seed','seed'} and n.args:
                for x in ast.walk(n.args[0]):
                    if isinstance(x,ast.Constant) and type(x.value)==int and 0<=x.value<2**32:found.add(x.value)
    return found

def classified_hit(path,text,projection=None):
    """Explicit known non-exposure evidence; unexpected identity blocks gate."""
    p=path.replace('\\','/');tokens=sorted(set(m.group() for m in STRONG.finditer(text)))
    if ENCODED.search(text):tokens.append('ENCODED_PROFILE_A')
    if not tokens:return None
    if p.endswith(('test_heldout_basis.py','test_frozen_generalization.py')):
        kind='REJECTED_NAMESPACE_TEST';proof='pytest.raises(ValueError); identifier guard rejects before channel construction';level='E0'
    elif p in {'scripts/sp_u1_plans.py','docs/sp_u1_ood_study_gate.md','docs/sp_u1_advisor_review_packet.md'}:
        kind='UNEXECUTED_PROPOSAL';proof='candidate remains unverified; no generator/evaluation execution';level='E0'
    elif p in {'outputs/t14a_profile_shape_gate_results.json','scripts/run_profile_shape_gate.py'} and 'TDL-A' in text and 'CDL-A' not in text:
        kind='DISTINCT_TDL_A_MODEL';proof='T14A explicit TDL-A parametric PDP family; run_profile_shape_gate imports src.tdl_profiles, not Sionna CDL';level='E0'
    elif p=='outputs/sp_u3a_sources/dissertation_2025.txt' and tokens==['a 120']:
        kind='LITERATURE_WORD_FALSE_POSITIVE';proof='ordinary article a followed by numeric text in downloaded literature; no CDL-A condition';level='E0'
    elif p.endswith(('t15a_cdl_environment.json','t20b_preflight.json','t20b_execution_preflight.json','t20b_execution_preflight_final.json')) and projection and projection['strong'] and all(h['package_path'] for h in projection['strong']) and 'ENCODED_PROFILE_A' not in tokens:
        kind='INSTALLED_MODEL_INVENTORY';proof='all CDL-A strings are site-packages/models/CDL-A.json paths';level='E0'
    else:kind='UNRESOLVED_CANDIDATE_IDENTITY';proof='requires metadata-only manual adjudication';level='OPEN'
    return {'path':p,'tokens':tokens,'kind':kind,'candidate_exposure':level,'proof':proof,'performance_exists_for_candidate':False if level=='E0' else None}

def audit():
    paths=sorted(subprocess.check_output(['rg','--files','--hidden','-g','!.git/**','-g','!.venv*/**','-g','!**/__pycache__/**','-g','!.pytest_cache/**'],cwd=ROOT,text=True).splitlines())
    paths=[p for p in paths if not any('sp_u7a' in v.lower() or 'sp_u7b' in v.lower() for v in Path(p).parts)]
    counts=Counter();hits=[];broad=[];npz_inventory=[];issues=[];forbidden=set();namespaces=set();seed_sources=[]
    def inspect_text(path,text,scope):
        nonlocal forbidden,namespaces
        suffix=Path(path).suffix.lower();projection=None
        if suffix in {'.json','.jsonl'}:projection=project_json(text)
        elif suffix=='.csv':projection=csv_metadata(text)
        elif suffix=='.py':
            s=source_seeds(text)
            if s:forbidden|=s;seed_sources.append({'path':path,'scope':scope,'kind':'AST_literal','seed_count':len(s)})
        if projection:
            s=projection['seeds'];forbidden|=s;namespaces|=projection['namespaces']
            if s:seed_sources.append({'path':path,'scope':scope,'kind':'metadata_seed_fields','seed_count':len(s)})
            candidate=[(k,v) for k,v in projection['identities'] if (k.lower() in {'profile','channel_profile','cdl_profile','model'} and v.upper()=='A')]
            if candidate and not (path.replace('\\','/')=='outputs/t14a_profile_shape_gate_results.json' and 'TDL-A' in text and 'CDL-A' not in text):issues.append({'path':path,'scope':scope,'reason':'encoded profile A','identity':candidate})
        h=classified_hit(path,text,projection)
        if h:h['scope']=scope;hits.append(h)
        if BROAD.search(text):broad.append({'path':path,'scope':scope,'classification':h['kind'] if h else 'FALSE_POSITIVE_HASH_OR_PROFILE_ADAPTATION','matched_tokens':sorted(set(m.group() for m in BROAD.finditer(text)))[:20]})
    for j,p in enumerate(paths):
        q=ROOT/p;s=q.suffix.lower();counts['workspace_files']+=1
        if STRONG.search(p):issues.append({'path':p,'reason':'candidate filename identity requires adjudication'})
        if s in TEXT_EXT:
            counts['workspace_text']+=1
            try:inspect_text(p,q.read_text(encoding='utf-8-sig',errors='replace'),'workspace')
            except Exception as e:issues.append({'path':p,'reason':type(e).__name__})
        elif s=='.npz':
            counts['workspace_npz']+=1
            try:
                a=archive_metadata(q);forbidden|=a['seeds'];namespaces|=a['namespaces']
                npz_inventory.append({'path':p.replace('\\','/'),'members':a['members'],'identity':a['identities']})
                if a['strong']:issues.append({'path':p,'reason':'NPZ candidate identity','hits':a['strong']})
            except zipfile.BadZipFile:
                if p.replace('\\','/').endswith('t17a_startup_temp/test_frozen_basis_tamper_is_re0/outputs/t16b_shared_basis_all_u32.npz') and q.stat().st_size==8:
                    npz_inventory.append({'path':p.replace('\\','/'),'kind':'TEST_TAMPER_SENTINEL','size':8,'source':'tests/test_heldout_basis.py:53'})
                else:issues.append({'path':p,'reason':'BadZipFile'})
            except Exception as e:issues.append({'path':p,'reason':str(e)})
        else:counts['other_files_filename_inventory']+=1
        if j and j%2000==0:print('metadata coverage',j,'/',len(paths),flush=True)
    print('workspace metadata done; auditing reachable Git blobs',flush=True)
    # Index object IDs identify already-audited current NPZs without reading arrays.
    indexed={}
    for line in subprocess.check_output(['git','ls-files','-s'],cwd=ROOT,text=True).splitlines():
        fields,p=line.split('\t',1);indexed[fields.split()[1]]=p
    objects=subprocess.check_output(['git','rev-list','--objects','--all'],cwd=ROOT,text=True).splitlines()
    commits=subprocess.check_output(['git','log','--all','--format=%H %s'],cwd=ROOT,text=True).splitlines()
    counts['git_commits']=len(commits)
    commit_hits=[{'commit':x.split(' ',1)[0],'tokens':sorted(set(m.group() for m in STRONG.finditer(x)))} for x in commits if STRONG.search(x)]
    # Batch cat-file avoids Windows process launch per blob. Only selected historical
    # text/NPZ blobs not present at HEAD are requested; payload arrays stay opaque.
    chosen=[]
    for line in objects:
        if ' ' not in line:continue
        oid,p=line.split(' ',1);s=Path(p).suffix.lower()
        if s in TEXT_EXT or s=='.npz':
            if oid in indexed and (ROOT/indexed[oid]).exists():counts['git_current_blobs_covered']+=1;continue
            chosen.append((oid,p))
    if chosen:
        proc=subprocess.Popen(['git','cat-file','--batch'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        for j,(oid,p) in enumerate(chosen):
            proc.stdin.write((oid+'\n').encode());proc.stdin.flush()
            header=proc.stdout.readline().decode().split();size=int(header[2]);blob=proc.stdout.read(size);proc.stdout.read(1)
            if len(blob)!=size:raise RuntimeError('incomplete Git blob')
            if Path(p).suffix.lower()=='.npz':
                counts['git_old_npz']+=1
                try:
                    a=archive_metadata(io.BytesIO(blob));forbidden|=a['seeds'];namespaces|=a['namespaces']
                    npz_inventory.append({'path':p,'git_blob':oid,'members':a['members'],'identity':a['identities']})
                    if a['strong']:issues.append({'path':p,'blob':oid,'reason':'historical NPZ candidate identity','hits':a['strong']})
                except Exception as e:issues.append({'path':p,'blob':oid,'reason':str(e)})
            else:counts['git_old_text']+=1;inspect_text(p,blob.decode('utf-8-sig',errors='replace'),'git:'+oid)
            if j and j%100==0:print('historical metadata coverage',j,'/',len(chosen),flush=True)
        proc.stdin.close();proc.stdout.close()
        if proc.wait()!=0:raise RuntimeError('Git metadata reader failed')
    unresolved=[h for h in hits if h['candidate_exposure']=='OPEN']
    report={'task':'SP-U7A','candidate':'CDL-A/120/100','counts':dict(counts),'candidate_exposure':'E0' if not issues and not unresolved else 'UNRESOLVED',
            'verdict':'PASS' if not issues and not unresolved else 'STOP_AUDIT_INCOMPLETE','hits':hits,'commit_message_hits':commit_hits,'issues':issues,
            'unresolved_hits':unresolved,'broad_query':BROAD.pattern,'strong_query':STRONG.pattern,'encoded_query':ENCODED.pattern,
            'outcome_values_decoded':0,'numeric_npz_payloads_decoded':0,'new_channel_calls':0,
            'binary_scope':'All current/historical NPZ NPY headers + explicitly string identity metadata; other binary artifacts inventoried by filename and associated text manifests, no numerical/PDF/weight payload decoding.',
            'known_tamper_fixture':'8-byte rejected basis fixture linked to tests/test_heldout_basis.py:53; not a channel archive.'}
    write_json(OUT/'sp_u7a_exposure_audit.json',report)
    write_json(OUT/'sp_u7a_broad_hits.json',broad)
    write_json(OUT/'sp_u7a_npz_schema_inventory.json',npz_inventory)
    write_json(OUT/'sp_u7a_historical_seed_union.json',{'forbidden_seeds':sorted(forbidden),'historical_namespaces':sorted(namespaces),'sources':seed_sources,'scope':'all workspace + removed/replaced Git metadata/AST declarations; individual values, including rejected plans'})
    write_json(OUT/'sp_u7a_workspace_inventory.json',{'paths':paths,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'indexed_objects':indexed})
    print(json.dumps({'verdict':report['verdict'],'candidate_exposure':report['candidate_exposure'],'counts':report['counts'],'issue_count':len(issues),'unresolved_hit_count':len(unresolved)},ensure_ascii=False),flush=True)
    return report

if __name__=='__main__':
    import sys
    sys.exit(0 if audit()['verdict']=='PASS' else 1)
