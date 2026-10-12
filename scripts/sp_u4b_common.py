"""SP-U4B fixed subset replication, two-level summaries and isolated I/O."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.sp_u4a_common import (sha,save,readjson,csvread,csvsave,array_hash,build_statistics,
    build_operators,cv_model,source_predict,RankFactors,distribution,trajectory_metric,rel_error,observation_geometry)
OUT=ROOT/'outputs';INSTALLED=False
ALLOWED={'sp_u4a_preregistration.json','sp_u4a_clean.npy','sp_u4a_clean_gate.json','sp_u4a_delivery_manifest.json',
    'sp_u4a_operators_500.npz','sp_u4a_statistics_500.npz','sp_u4a_development_cache','sp_u4a_pair_hashes.csv',
    'sp_u4a_seed_manifest.csv','sp_u4a_trajectories.csv','sp_u4a_windows.csv','sp_u4a_development_gate.json'}
ALLOWED.update(f'sp_u4a_{kind}_{n}.npz' for kind in ('operators','statistics') for n in (50,100,250))


def artifact(replicate,budget,kind='operators'):
    if int(budget)==500 or replicate=='R0':return OUT/f'sp_u4a_{kind}_{budget}.npz'
    return OUT/f'sp_u4b_{kind}_{replicate}_{budget}.npz'


def load_arrays(replicate,budget,kind='operators'):
    with np.load(artifact(replicate,budget,kind),allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def firewall():
    global INSTALLED
    if INSTALLED:return
    base=str(OUT.resolve()).lower()+os.sep
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if not str(p).lower().startswith(base):return
        parts=p.relative_to(OUT).parts;own=parts[0].startswith('sp_u4b_')
        mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if not own and (parts[0] not in ALLOWED or write):raise PermissionError('SP-U4B firewall blocked:'+str(p))
    sys.addaudithook(audit);INSTALLED=True


def freeze():
    # Only U4A protocol/permutation metadata is read before new permutations seal.
    old=readjson(OUT/'sp_u4a_preregistration.json');perms={'R0':old['permutations']}
    seeds={'R0':old['config']['permutation_seeds']}
    for r in range(1,5):
        seeds[f'R{r}']={e:2026100500+10*r+j for j,e in enumerate(('Source','P','M'),1)}
        perms[f'R{r}']={e:np.random.default_rng(s).permutation(500).tolist() for e,s in seeds[f'R{r}'].items()}
    path=OUT/'sp_u4b_preregistration.json'
    if path.exists():raise ValueError('frozen registration exists')
    cfg=old['config']
    save(path,dict(task='SP-U4B',head=cfg['head'],config=cfg,replicates=list(perms),new_budgets=[50,100,250],common_reference_budget=500,
        permutation_seeds=seeds,permutations=perms,permutation_sha256={r:{e:array_hash(np.array(ids,np.int64)) for e,ids in p.items()} for r,p in perms.items()},
        U4A_preregistration_sha256=sha(OUT/'sp_u4a_preregistration.json'),frozen_before_scientific_result_reads=True,created_unix=time.time(),
        central_metric='median trajectory NMSE dB; B_N=median_r(m_rN-m500); V_N=IQR_r(m_rN)',
        tail_definitions='P95 full signed paired penalty and P95 conditional on penalty>0 both retained; maximum positive penalty; top5 positive IDs',
        constraints='no new evaluation H; no N500 rebuild; no extra budgets/replicates; no tuning; no formal inference; STOP after A/B/C'))
    csvsave(OUT/'sp_u4b_table1_permutations.csv',[dict(replicate=r,environment=e,seed=seeds[r][e],sha256=array_hash(np.array(p[e],np.int64))) for r,p in perms.items() for e in p])
    csvsave(OUT/'sp_u4b_nested_ids.csv',[dict(replicate=r,environment=e,position=j,trajectory_id=i) for r,p in perms.items() for e,ids in p.items() for j,i in enumerate(ids)])
    print('R0 original permutations reused; R1-R4 prescribed permutations frozen before scientific result reads',flush=True)


def prereg():
    p=readjson(OUT/'sp_u4b_preregistration.json')
    if p['U4A_preregistration_sha256']!=sha(OUT/'sp_u4a_preregistration.json'):raise ValueError('R0 protocol drift')
    return p


def integrity_baseline():
    prereg();old=readjson(OUT/'sp_u4a_delivery_manifest.json')
    for p,h in {**old['artifact_sha256'],**old['source_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError('U4A drift:'+p)
    protected={}
    for folder in ('src','scripts','tests','configs','docs','paper'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and 'sp_u4b' not in p.name:protected[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'sp_u4b_before_manifest.json',dict(protected=protected,U4A=old,U4A_manifest_sha256=sha(OUT/'sp_u4a_delivery_manifest.json')))


def signed_tail(values,ids):
    a=np.asarray(values,float);positive=a[a>0];order=np.argsort(-a,kind='stable')
    return dict(p95_penalty=float(np.quantile(a,.95)),p95_positive_penalty=float(np.quantile(positive,.95)) if len(positive) else None,
        maximum_positive_penalty=float(max(0,a.max())),positive_penalty_fraction=float(np.mean(a>0)),
        top5_positive_ids=[int(ids[j]) for j in order if a[j]>0][:5],top5_positive_values=[float(a[j]) for j in order if a[j]>0][:5])


if __name__=='__main__':freeze()
