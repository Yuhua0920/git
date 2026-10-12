"""MR03C immutable prepared-origin posterior mixture on existing development data."""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,csv,hashlib,zipfile,time,shutil,argparse,traceback
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(r'D:\Users\yuhua\Desktop\Project');OUT=ROOT/'outputs/revision_mr03c_prepared_gmm';MR=ROOT/'outputs/revision_mr03_ppca_capacity_controls';MRB=ROOT/'outputs/revision_mr03_em16_reconciliation'
sys.path[:0]=[str(ROOT),str(OUT/'src'),str(MR/'src_or_scripts')]
from gmm3 import prepare_gaussian_mixture
CONDS=('ID','Profile','Mobility','Composed','Boundary');SNRS=(10,20,30)
METHODS=('SOURCE','POOLED_NATURAL','GMM_PREPARED_3','PPCA_DIRECT_R16','PPCA_DIRECT_R32','LR4_TW1','LR4_TW5','DENSE_TW5','WHOLE_POOL_M2_R24')
STATS=('t15a_cdl_statistics_500.npz','t15c_cdl_statistics_CDL-B-30.npz','t15c_cdl_statistics_CDL-C-120.npz')
EXPECTED=('ce9f2d0f567e20ff98fa88e098a270bec47dafb5839e14356a98282a514b4b69','8b43842c3b3738b4e403bc927b1327eb29467db9767780dd789cd080c8b81b4b','8230e818870dfb6a58e95edbfc82c8ddd17748ecce7bd82417a977e92e1491e1')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def archive(p,**x):
    with Path(p).open('xb') as f:np.savez_compressed(f,**x)
def csvout(p,x):
    with Path(p).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
def close(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('FINITE_SHAPE:'+label)
    e=float(np.max(abs(a-b),initial=0));s=max(float(np.max(abs(a),initial=0)),float(np.max(abs(b),initial=0)))
    if e>1e-10+1e-9*s:raise ValueError('NUMERICAL:'+label)
    return e
def verify():
    v=read(OUT/'reports/preservation_before.json')
    for p,h in v['hashes'].items():
        if sha(ROOT/p)!=h:raise ValueError('OLD_CHANGED:'+p)
    f=read(OUT/'reports/execution_freeze.json')
    for p,h in f['hashes'].items():
        if sha(OUT/p)!=h:raise ValueError('NEW_SCIENCE_CHANGED:'+p)
    return dict(PASS=True,protected_original_files=len(v['hashes']),science_frozen_files=len(f['hashes']),modified=0)
def origins():
    rs=[]
    for n,h,name in zip(STATS,EXPECTED,('Source','Profile','Mobility')):
        p=ROOT/'outputs'/n
        if sha(p)!=h:raise ValueError('PREPARATION_HASH')
        with np.load(p,allow_pickle=False) as z:
            if int(z['samples'])!=5000 or not np.array_equal(z['train_trajectory_ids'],np.arange(500)):raise ValueError('PREPARATION_POPULATION')
            rs.append(dict(name=name,mu_p=z['mean_p'],mu_d=z['mean_d'],R_pp=z['rpp'],R_dp=z['rdp']))
    return rs
def models():
    power=read(MR/'config.json')['source_noise_power'];return {s:prepare_gaussian_mixture(origins(),power/10**(s/10)) for s in SNRS}
def prepare():
    if (OUT/'config.json').exists():raise FileExistsError('PROTOCOL_EXISTS')
    for d in ('data','tables','reports','figures','src','tests'):(OUT/d).mkdir(parents=True,exist_ok=True)
    protected=read(MRB/'reports/preservation_execution.json')['hashes'].copy()
    for p in MRB.rglob('*'):
        if p.is_file():protected[p.relative_to(ROOT).as_posix()]=sha(p)
    write(OUT/'reports/preservation_before.json',dict(hashes=protected))
    cfg=dict(task='MR03C prepared-origin Gaussian posterior mixture',origins=['Source','Profile','Mobility'],prior=[1/3,1/3,1/3],methods=METHODS,conditions=CONDS,snrs=SNRS,frames=list(range(4,10)),per_condition=100,physical_trajectories_per_dataset=500,new_GMM_positions_per_dataset=9000,datasets=['V_new','D_eval'],development_disclosed=True,new_validation=False,no_EM=True,no_statistics_refit=True,bootstrap=dict(replicates=2000,confidence=.95,adjusted=False,unit='trajectory',draws='exact original MR03 persisted indices'),posterior=dict(entropy='natural-log nats',hardlike_max_weight_threshold=.95,prepared_correspondence={'ID':0,'Profile':1,'Mobility':2},no_unique_origin=['Composed','Boundary'],correlation='trajectory level descriptive Spearman; no causality or pvalue'),smoke=dict(trajectories='i=0 from each of five conditions, both existing datasets',positions=180,reuse_smoke_results_in_full=True),timing=dict(threads=1,warmup_calls=30,windows_per_snr=100,blocks=10,order_seed=170505,window='D_eval five conditions x first20trajectories x SNR10/20/30, endingframe4',endpoint='full-estimator-core; fit+deployment for adaptive methods; current pilot only for statistical models',exclude='data generation, disk, hashing, logging, plotting; posterior diagnostics included in GMM production API'),atol=1e-10,rtol=1e-9)
    write(OUT/'config.json',cfg)
    for ds in cfg['datasets']:
        shutil.copyfile(MR/'data'/f'bootstrap_{ds}.npz',OUT/'data'/f'bootstrap_{ds}.npz')
    setup=[];g=models();kernelchecks=[]
    with np.load(ROOT/'outputs/revision_mr01_pooled_lmmse/data/operators.npz',allow_pickle=False) as z:
        for s,m in g.items():
            if not np.array_equal(m.priors,np.full(3,1/3)):raise ValueError('EQUAL_PRIOR_REQUIRED')
            setup.append(dict(snr_db=s,setup_audit=m.setup_audit))
            for label,c in zip(('source','P','M'),m.components):kernelchecks.append(dict(snr_db=s,origin=c.name,max_absolute=close(c.W,z[f'W{label}_{s}'],'COMPONENT_VS_EXISTING_LMMSE')))
    write(OUT/'reports/preparation.json',dict(PASS=True,statistics=[dict(origin=name,path='outputs/'+n,SHA256=h,trajectories=500,samples=5000) for name,n,h in zip(cfg['origins'],STATS,EXPECTED)],shapes=dict(mu_p=[64],mu_d=[3968],R_pp=[64,64],R_dp=[3968,64]),dtype='complex128',stats_reestimated=False,posterior_prior=[1/3]*3,setup=setup,existing_component_kernel_checks=kernelchecks))
    files=[OUT/'src'/n for n in ('gmm3.py','run_gmm3.py','benchmark_gmm3.py')]+list((OUT/'tests').glob('*.py'))+[OUT/'config.json',OUT/'reports/preparation.json',OUT/'reports/preservation_before.json']+[OUT/'data'/f'bootstrap_{ds}.npz' for ds in cfg['datasets']]
    write(OUT/'reports/execution_freeze.json',dict(hashes={p.relative_to(OUT).as_posix():sha(p) for p in files},first_new_scientific_prediction_started=False))
    print(json.dumps(dict(stage='prepare',PASS=True,protected=len(protected))),flush=True)
def evaluate(ds,smoke=False):
    verify();gmm=models()
    from src.cdl_benchmark import frames,observation_geometry,array_hash
    from scripts.run_sp_u12b2a import generator,seed_index
    sys.path.insert(0,str(ROOT/'outputs/revision_mr05_compression_controls/src_or_scripts'))
    import run_mr05 as old
    from scripts.sp_u12b4a_adapter import rss
    if not smoke and not read(OUT/'reports/smoke_analysis.json')['PASS']:raise ValueError('SMOKE_REQUIRED')
    mode='smoke_'+ds if smoke else ds;folder=OUT/'data'/mode;folder.mkdir(exist_ok=False)
    receipts={(x['condition'],x['trajectory_id']):x for x in read(MR/'reports'/f'{ds}_receipt.json')['receipts']};n=1 if smoke else 100;t=time.monotonic();recs=[];maximum=0
    if ds=='V_new':idx=seed_index(read(MR/'data/validation_seed_registry.json'))
    gm=observation_geometry();oldmethods=read(MR/'config.json')['methods']
    for c in CONDS:
        if ds=='V_new':gen=generator(c,idx,'EVAL')
        for i in range(n):
            path=folder/f'{c}_{i:04d}.npz';jp=path.with_suffix('.json')
            if not smoke and i==0:
                sp=OUT/'data'/('smoke_'+ds)/path.name;sj=sp.with_suffix('.json');meta=read(sj)
                if sha(sp)!=meta['raw_SHA256']:raise ValueError('SMOKE_SNAPSHOT_CHANGED')
                shutil.copyfile(sp,path);shutil.copyfile(sj,jp);recs.append(dict(condition=c,trajectory_id=i,path=path.relative_to(OUT).as_posix(),SHA256=sha(path),JSON_SHA256=sha(jp),from_smoke=True));continue
            rec=receipts[c,i];oldp=MR/rec['path'];oldj=oldp.with_suffix('.json')
            if sha(oldp)!=rec['sha256'] or sha(oldj)!=rec['json_sha256']:raise ValueError('OLD_RAW_HASH')
            meta=read(oldj)
            with np.load(oldp,allow_pickle=False) as z:raw=dict(z)
            if ds=='V_new':h=gen.generate(i)
            else:
                original,_=old.source_input(c,i);h=original['H']
                if not np.array_equal(original['pilots'],raw['pilots']):raise ValueError('OLD_PILOTS_DIFFER')
            if array_hash(h)!=meta['input']['H_sha256'] or array_hash(raw['pilots'])!=meta['input']['pilot_sha256']:raise ValueError('INPUT_IDENTITY')
            hf=frames(h);target=np.stack([hf[j].ravel()[gm.data] for j in range(4,10)])
            close(np.real(np.einsum('ij,ij->i',target.conj(),target)),raw['frame_truth'],'TRUTH_PAIR')
            e=np.empty((3,len(METHODS),6));weights=[];entropy=[];mx=[];score=[];quad=[];example=[]
            for si,s in enumerate(SNRS):
                prediction=gmm[s].predict(raw['pilots'][si,4:10]);d=prediction.estimate-target
                eg=np.real(np.einsum('ij,ij->i',d.conj(),d))
                for mi,m in enumerate(METHODS):e[si,mi]=eg if m=='GMM_PREPARED_3' else raw['frame_error'][si,oldmethods.index(m)]
                weights.append(prediction.posterior_weights);entropy.append(prediction.posterior_entropy_nats);mx.append(prediction.max_posterior_weight);score.append(prediction.log_scores);quad.append(prediction.quadratic_forms);example.append(prediction.estimate[0,:16])
            if not np.isfinite(e).all() or np.any(e<0):raise ValueError('ENERGY_INVALID')
            posterior=np.array(weights);summation=float(np.max(abs(posterior.sum(axis=-1)-1)));maximum=max(maximum,summation)
            archive(path,frame_error=e,frame_truth=raw['frame_truth'],posterior_weights=posterior,posterior_entropy_nats=np.array(entropy),max_posterior_weight=np.array(mx),log_scores=np.array(score),quadratic_forms=np.array(quad),pilots=raw['pilots'][:,4:10],snr_db=np.array(SNRS),frame_ids=np.arange(4,10),method_names=np.array(METHODS),GMM_prediction_example_frame4=np.array(example),failure_flags=np.zeros((3,6),bool))
            write(jp,dict(dataset=ds,condition=c,trajectory_id=i,identity=meta['identity'],H_SHA256=meta['input']['H_sha256'],pilot_full_SHA256=meta['input']['pilot_sha256'],scored_pilot_SHA256=array_hash(raw['pilots'][:,4:10]),old_raw_SHA256=sha(oldp),old_JSON_SHA256=sha(oldj),raw_SHA256=sha(path),methods=METHODS,complete=True,smoke=smoke,truth_and_existing_energies_exact_reuse=True,posterior_labels_used_online=False,posterior_sum_error=summation,failures=0,nonfinite=0))
            recs.append(dict(condition=c,trajectory_id=i,path=path.relative_to(OUT).as_posix(),SHA256=sha(path),JSON_SHA256=sha(jp),from_smoke=False))
            if i%25==0:print(json.dumps(dict(mode=mode,condition=c,completed=i+1,elapsed=time.monotonic()-t)),flush=True)
    write(OUT/'reports'/f'{mode}_receipt.json',dict(PASS=True,dataset=ds,namespace=mode,receipts=recs,physical_trajectories=n*5,GMM_scored_positions=n*90,existing_method_energies_reused=True,smoke_positions_reused=0 if smoke else 90,failures=0,nonfinite=0,max_weight_sum_error=maximum,elapsed_seconds=time.monotonic()-t,resource=rss(),new_dataset=False,new_noise=0))
    print(json.dumps(dict(mode=mode,PASS=True,positions=n*90,elapsed=time.monotonic()-t)),flush=True)
def corr(a,b):
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    v=float(spearmanr(a,b).statistic);return v if np.isfinite(v) else None
def analyze(smoke=False):
    verify();from independent_review_capacity import cell_statistics,linear_percentiles
    nm=[];cs=[];diag=[];traj=[];boots={};rawcount=0
    for ds in ('V_new','D_eval'):
        mode='smoke_'+ds if smoke else ds;rec=read(OUT/'reports'/f'{mode}_receipt.json')
        if not rec['PASS']:raise ValueError('RAW_INCOMPLETE')
        with np.load(OUT/'data'/f'bootstrap_{ds}.npz',allow_pickle=False) as z:draws=dict(z)
        for c in CONDS:
            E=[];Q=[];GW=[];H=[];MX=[]
            for r in rec['receipts']:
                if r['condition']!=c:continue
                p=OUT/r['path']
                if sha(p)!=r['SHA256'] or sha(p.with_suffix('.json'))!=r['JSON_SHA256']:raise ValueError('RAW_HASH')
                with np.load(p,allow_pickle=False) as z:E.append(z['frame_error']);Q.append(z['frame_truth']);GW.append(z['posterior_weights']);H.append(z['posterior_entropy_nats']);MX.append(z['max_posterior_weight'])
            E,Q,GW,H,MX=map(np.array,(E,Q,GW,H,MX));n=E.shape[0];rawcount+=n
            # Smoke tests aggregation/figure schema; one trajectory has no CI.
            dr=np.zeros((1,1),np.int64) if smoke else draws[c]
            stats=cell_statistics(E,Q,dr);b=linear_percentiles(stats['bootstrap_db']);boots[ds+'_'+c]=stats['bootstrap_linear']
            gi=METHODS.index('GMM_PREPARED_3');pi=METHODS.index('POOLED_NATURAL')
            for si,s in enumerate(SNRS):
                for mi,m in enumerate(METHODS):nm.append(dict(dataset=ds,condition=c,snr_db=s,method=m,nmse_linear=float(stats['nmse_linear'][si,mi]),nmse_db=float(stats['nmse_db'][si,mi]),CI_lower=None if smoke else float(b[0,si,mi]),CI_upper=None if smoke else float(b[1,si,mi]),trajectory_count=n,scored_frames=n*6))
                for second in ('POOLED_NATURAL','LR4_TW1','LR4_TW5','WHOLE_POOL_M2_R24','DENSE_TW5'):
                    bi=METHODS.index(second);bounds=linear_percentiles(stats['bootstrap_db'][:,si,gi]-stats['bootstrap_db'][:,si,bi])
                    cs.append(dict(dataset=ds,condition=c,snr_db=s,first='GMM_PREPARED_3',second=second,estimate_db=float(stats['nmse_db'][si,gi]-stats['nmse_db'][si,bi]),CI_lower=None if smoke else float(bounds[0]),CI_upper=None if smoke else float(bounds[1]),trajectory_count=n,bootstrap=0 if smoke else 2000,interval='SMOKE: no uncertainty inference' if smoke else 'Unadjusted exploratory 95% paired trajectory percentile; disclosed development data',sign='GMM minus comparator; positive favors comparator',GMM_worse_trajectory_fraction=float(np.mean(stats['error'][:,si,gi]>stats['error'][:,si,bi]))))
                correspondence={'ID':0,'Profile':1,'Mobility':2}.get(c);selection=GW[:,si].argmax(axis=-1);en=H[:,si].mean(axis=-1);gain=10*np.log10(stats['error'][:,si,gi]/stats['error'][:,si,pi]);mis=None if correspondence is None else (selection!=correspondence).mean(axis=-1)
                diag.append(dict(dataset=ds,condition=c,snr_db=s,mean_weight_Source=float(GW[:,si,:,0].mean()),mean_weight_Profile=float(GW[:,si,:,1].mean()),mean_weight_Mobility=float(GW[:,si,:,2].mean()),mean_entropy_nats=float(H[:,si].mean()),mean_max_weight=float(MX[:,si].mean()),hardlike_fraction_gamma_ge_095=float(np.mean(MX[:,si]>=.95)),argmax_Source_fraction=float(np.mean(selection==0)),argmax_Profile_fraction=float(np.mean(selection==1)),argmax_Mobility_fraction=float(np.mean(selection==2)),correspondence_origin=None if correspondence is None else ('Source','Profile','Mobility')[correspondence],correspondence_mismatch_frame_fraction=None if mis is None else float(mis.mean()),Spearman_entropy_vs_GMM_minus_Pooled=None if smoke else corr(en,gain),Spearman_mismatch_vs_GMM_minus_Pooled=None if smoke or mis is None else corr(mis,gain),trajectory_count=n,descriptive_only=True))
                for t in range(n):
                    for mi,m in enumerate(METHODS):traj.append(dict(dataset=ds,condition=c,trajectory_id=t,snr_db=s,method=m,error_energy=float(stats['error'][t,si,mi]),truth_energy=float(stats['truth'][t]),GMM_entropy_nats=float(en[t]),GMM_mean_max_weight=float(MX[t,si].mean()),GMM_weight_Source=float(GW[t,si,:,0].mean()),GMM_weight_Profile=float(GW[t,si,:,1].mean()),GMM_weight_Mobility=float(GW[t,si,:,2].mean())))
    prefix='smoke_' if smoke else ''
    csvout(OUT/'tables'/f'{prefix}nmse.csv',nm);csvout(OUT/'tables'/f'{prefix}paired_contrasts.csv',cs);csvout(OUT/'tables'/f'{prefix}posterior_diagnostics.csv',diag);csvout(OUT/'tables'/f'{prefix}trajectory_energies.csv',traj)
    archive(OUT/'data'/f'{prefix}bootstrap_statistics.npz',**boots)
    write(OUT/'reports'/f'{prefix}analysis.json',dict(PASS=True,smoke=smoke,NMSE_rows=len(nm),contrast_rows=len(cs),posterior_rows=len(diag),trajectory_blocks=rawcount,new_bootstrap_draws=0,existing_draws_SHA256={ds:sha(OUT/'data'/f'bootstrap_{ds}.npz') for ds in ('V_new','D_eval')},figures_can_read_same_CSV_schema=True,no_smoke_CI=True if smoke else None))
    print(json.dumps(dict(stage=prefix+'analysis',PASS=True,NMSE_rows=len(nm),contrast_rows=len(cs))),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','V_new','D_eval','analyze']);a=p.parse_args()
    try:
        if a.action=='prepare':prepare()
        elif a.action=='smoke':evaluate('V_new',True);evaluate('D_eval',True);analyze(True)
        elif a.action=='analyze':analyze()
        else:evaluate(a.action)
    except BaseException as e:
        if (OUT/'reports').exists():write(OUT/'reports'/f'{a.action}_STOP_{time.time_ns()}.json',dict(status='STOP',error=repr(e),traceback=traceback.format_exc(),historical_outputs_not_written=True))
        raise
if __name__=='__main__':main()
