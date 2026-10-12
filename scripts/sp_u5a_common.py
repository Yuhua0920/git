"""SP-U5A isolated geometry adapters; no historical outcome readers."""
import os,sys,json,hashlib,time,subprocess
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u4a_common import (sha,save,readjson,csvsave,csvread,distribution,trajectory_metric,rel_error,peak_memory)
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from src.pilots import build_fdm_pilot_layout,count_layout_resources
from src.cdl_benchmark import StreamingStatistics,FrozenStatistics,array_hash,SYSTEM,OfficialCDL,frames
from src.pilot_cv import PilotCV,frozen_folds
from src.lowrank_modules import WeightedModule,RankFactors
OUT=ROOT/'outputs';PREFIX='sp_u5a_'
HIST_STATS={'Source':'t15a_cdl_statistics_500.npz','P':'t15c_cdl_statistics_CDL-B-30.npz','M':'t15c_cdl_statistics_CDL-C-120.npz'}
ALLOW=set(HIST_STATS.values())|{'t16e_frozen_operators.npz','t16f_frozen_candidates.npz','t16g_frozen_module_factors.npz',
    'sp_u4a_clean.npy','sp_u4a_clean_gate.json','sp_u4a_clean_hashes.csv'}
GUARD=False
def cfg():return readjson(ROOT/'configs/sp_u5a_protocol.json')
def firewall():
    global GUARD
    if GUARD:return
    base=str(OUT.resolve()).lower()+os.sep
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if not str(p).lower().startswith(base):return
        rel=p.relative_to(OUT).parts;own=rel[0].startswith(PREFIX)
        mode=args[1];flags=args[2] if isinstance(args[2],int) else 0
        write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if not own and (len(rel)!=1 or rel[0] not in ALLOW or write):raise PermissionError('SP-U5A firewall:'+str(p))
    sys.addaudithook(audit);GUARD=True
def geometry(df):
    if df not in (8,16,32):raise ValueError('frozen dF grid')
    layout=build_fdm_pilot_layout(2,64,16,df,4)
    selected=np.flatnonzero(np.broadcast_to(layout.pilot_mask[None],(2,2,64,16)).ravel())
    data=np.broadcast_to(layout.data_mask[None],(2,2,64,16)).ravel().copy()
    return SimpleNamespace(df=df,layout=layout,selected=selected,data=data,counts=count_layout_resources(layout))
def folds_for(g):
    if g.df==16:return frozen_folds(g.selected)
    rx,tx,k,t=np.unravel_index(g.selected,(2,2,64,16));labels=(k//g.df+t//4)%4
    folds=tuple((np.flatnonzero(labels!=f),np.flatnonzero(labels==f)) for f in range(4))
    definition=dict(rule='fold=(physical_k//dF+symbol_t//4)%4',dF=g.df,selected=g.selected.tolist(),labels=labels.tolist(),
        folds=[dict(observed=o.tolist(),heldout=h.tolist()) for o,h in folds])
    definition['sha256']=hashlib.sha256(json.dumps(definition,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for o,h in folds:
        if len(h)!=len(g.selected)//4 or np.intersect1d(o,h).size:raise ValueError('fold partition')
        if any(sum((rx[h]==r)&(tx[h]==q))!=len(h)//4 for r in (0,1) for q in (0,1)):raise ValueError('link balance')
    return folds,definition
def seed_map(condition,i,j=0):
    # Exact namespace bytes determine four uint32 entropy words, independent of old run namespaces.
    words=np.frombuffer(hashlib.sha256(cfg()['namespace'].encode('utf8')).digest()[:16],dtype='<u4').tolist()
    role=cfg()['conditions'][condition]['role']
    return {n:int(np.random.SeedSequence([*words,role,i,0 if r<=3 else j,r]).generate_state(1)[0])
        for r,n in enumerate(('sionna_seed','torch_seed','numpy_seed','bits_seed_tx0','bits_seed_tx1','noise_seed'),1)}
def freeze():
    if (OUT/'sp_u5a_preregistration.json').exists():raise ValueError('already frozen')
    if SYSTEM['sample_rate_hz']!=SYSTEM['nfft']*SYSTEM['delta_f_hz']:raise ValueError('invalid sample rate')
    protected={}
    for folder in ('src','scripts','tests','configs','docs','paper'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and 'sp_u5a' not in p.name:protected[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in ALLOW:protected['outputs/'+name]=sha(OUT/name)
    save(OUT/'sp_u5a_before_manifest.json',protected)
    save(OUT/'sp_u5a_preregistration.json',dict(config=cfg(),config_sha256=sha(ROOT/'configs/sp_u5a_protocol.json'),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),created_unix=time.time(),
        frozen_before_scientific_results=True,engineering_tolerances='inherited1e-10 relative construction,1e-9 absolute replay; not performance thresholds',
        namespace_seed_derivation='first16 SHA256(namespace UTF8) bytes as4 little-endian uint32; SeedSequence(words,condition_role,trajectory,frame_or0,role)'))
    csvsave(OUT/'sp_u5a_seed_manifest.csv',[dict(namespace=cfg()['namespace'],condition=c,trajectory_id=i,frame_id=j,**seed_map(c,i,j)) for c in cfg()['conditions'] for i in range(100) for j in range(10)])
    geo=[];fc=[]
    for df in cfg()['densities']:
        g=geometry(df);fs,fd=folds_for(g);np.savez_compressed(OUT/f'sp_u5a_geometry_{df}.npz',pilot_mask=g.layout.pilot_mask,
            data_mask=g.layout.data_mask,mute_mask=g.layout.mute_mask,pilot_owner=g.layout.pilot_owner,selected=g.selected,data=g.data,
            labels=np.array(fd['labels']),**{f'observed{f}':o for f,(o,h) in enumerate(fs)},**{f'hold{f}':h for f,(o,h) in enumerate(fs)})
        save(OUT/f'sp_u5a_folds_{df}.json',fd)
        geo.append(dict(dF=df,dT=4,**{k:json.dumps(v) if isinstance(v,list) else v for k,v in g.counts.items()},
            pilot_observation_dimension=len(g.selected),target_dimension=int(g.data.sum()),muted_channel_coefficients=int(np.broadcast_to(g.layout.mute_mask[None],(2,2,64,16)).sum()),
            bits_per_frame=2*g.counts['two_stream_data_symbols'],mask_sha256=array_hash(g.layout.pilot_mask),data_sha256=array_hash(g.data),fold_sha256=fd['sha256']))
        rx,tx,k,t=np.unravel_index(g.selected,(2,2,64,16))
        for j in range(10):
            for q in (0,1):
                for f,(o,h) in enumerate(fs):fc.append(dict(dF=df,frame_id=j,TX=q,fold=f,
                    total_owner_REs=int(g.layout.pilot_mask[q].sum()),held_owner_REs=int(sum(tx[h]==q)//2),observed_owner_REs=int(sum(tx[o]==q)//2),
                    held_RX_coefficients=int(sum(tx[h]==q)),observed_RX_coefficients=int(sum(tx[o]==q))))
    csvsave(OUT/'sp_u5a_table1_geometry.csv',geo);csvsave(OUT/'sp_u5a_fold_counts.csv',fc)
    print('SP-U5A frozen seeds, geometry, power and engineering gates; no performance reads',flush=True)
def prereg():
    p=readjson(OUT/'sp_u5a_preregistration.json')
    if p['config_sha256']!=sha(ROOT/'configs/sp_u5a_protocol.json'):raise ValueError('protocol drift')
    return p
def cross_k(rpp,sigma,folds):
    result=[];res=[]
    for o,h in folds:
        a=rpp[np.ix_(o,o)]+sigma*np.eye(len(o));b=rpp[np.ix_(h,o)]
        w=cho_solve(cho_factor(a,lower=True),b.conj().T).conj().T
        r=rel_error(w@a,b)
        if r>1e-11:raise ValueError('cross pilot solve')
        result.append(w);res.append(r)
    return tuple(result),res
def build_stats(corpus,g):
    stream=StreamingStatistics(g.selected,g.data)
    for start in range(0,500,10):stream.update(np.asarray(corpus[start:start+10]).reshape(100,4096),list(range(start,start+10)))
    return stream.freeze()
def build_ops(stats,g):
    folds,_=folds_for(g);a={'source_mean_p':stats['Source'].mean_p,'source_mean_d':stats['Source'].mean_d};diags=[]
    for ss,sigma in cfg()['noise_variances'].items():
        s=int(ss)
        for e in cfg()['environments']:
            est=stats[e].prepare(sigma);a[f'W{e}_{s}']=est.weight;a[f'K{e}_{s}'],res=cross_k(stats[e].rpp,sigma,folds);a[f'K{e}_{s}']=np.stack(a[f'K{e}_{s}'])
            diags.append(dict(object=f'{e}_{s}',deployment_solve=est.solve_relative_residual,CV_solve_max=max(res)))
        a[f'Wsource_{s}']=a[f'WSource_{s}']
        for e in ('P','M'):
            delta=a[f'W{e}_{s}']-a[f'WSource_{s}'];a[f'DeltaW{e}_{s}']=delta;a[f'DeltaK{e}_{s}']=a[f'K{e}_{s}']-a[f'KSource_{s}']
            m=WeightedModule.build(delta,stats[e].rpp+sigma*np.eye(len(g.selected)));f=m.factors(4);a[f'A_{e}_{s}']=f.A;a[f'Q_{e}_{s}']=f.Q
            diags.append(dict(object=f'{e}4_{s}',weighted_capture_rank4=m.capture(4),full_rank_relative_residual=m.diagnostics()['full_rank_correction_relative_residual']))
    return a,diags
class DensityCV(PilotCV):
    """Same frozen objective/normal equation; shape validation follows actual geometry."""
    def frame_objective(self,pilots):
        p=np.asarray(pilots,np.complex128)
        if p.shape!=self.source_mean_p.shape or not np.isfinite(p).all():raise ValueError('pilot geometry mismatch')
        z=p-self.source_mean_p;columns=[];responses=[]
        for f,(o,h) in enumerate(self.folds):
            x=z[o];columns.append(np.column_stack((self.profile_delta_k[f]@x,self.mobility_delta_k[f]@x)));responses.append(z[h]-self.source_k[f]@x)
        d=np.concatenate(columns);r=np.concatenate(responses)
        return d.conj().T@d,d.conj().T@r,float(np.vdot(r,r).real)
    def fit(self,pilot_frames,*,modules):
        x=np.asarray(pilot_frames,np.complex128)
        if x.ndim!=2 or x.shape[1]!=self.source_mean_p.size or not 1<=len(x)<=5 or modules!=('P','M'):raise ValueError('frozen causal P+M fit')
        moments=[self.frame_objective(p) for p in x];gram=sum(v[0] for v in moments).real;rhs=sum(v[1] for v in moments).real
        c=np.linalg.solve(gram,rhs)
        if not np.isfinite(c).all():raise ValueError('nonfinite coordinates')
        baseline=sum(v[2] for v in moments)
        return c,dict(gram=gram.tolist(),rhs=rhs.tolist(),condition_number=float(np.linalg.cond(gram)),gram_rank=int(np.linalg.matrix_rank(gram)),
            normal_equation_relative_residual=float(np.linalg.norm(gram@c-rhs)/max(np.linalg.norm(rhs),np.finfo(float).tiny)),
            source_cv_loss=baseline,fitted_cv_loss=float(baseline-2*c@rhs+c@gram@c),window_frames=len(x),real_ls_size=2,clipped=False,regularized=False,independent_frames_assumed=False)
def cv_model(a,g,s):return DensityCV(folds_for(g)[0],a['source_mean_p'],tuple(a[f'KSource_{s}']),tuple(a[f'DeltaKP_{s}']),tuple(a[f'DeltaKM_{s}']))
def predict(a,p,s,c=None):
    z=p-a['source_mean_p'];base=a['source_mean_d']+z@a[f'Wsource_{s}'].T
    if c is None:return base
    return base+c[0]*RankFactors(a[f'A_P_{s}'],a[f'Q_P_{s}']).apply(z)+c[1]*RankFactors(a[f'A_M_{s}'],a[f'Q_M_{s}']).apply(z)
def arrays(df):
    m=readjson(OUT/f'sp_u5a_construction_{df}.json');p=OUT/f'sp_u5a_operators_{df}.npz'
    if sha(p)!=m['operators_sha256']:raise ValueError('operator hash drift')
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def historical_arrays():
    a={}
    for name in ('t16e_frozen_operators.npz','t16f_frozen_candidates.npz','t16g_frozen_module_factors.npz'):
        with np.load(OUT/name,allow_pickle=False) as f:a.update({k:f[k].copy() for k in f.files if k!='metadata_json'})
    for e in ('P','M'):
        for s in (10,20):a[f'A_{e}_{s}']=a[f'A_{e}_{s}'][:,:4];a[f'Q_{e}_{s}']=a[f'Q_{e}_{s}'][:4]
    return a
