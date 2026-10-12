"""Symmetric LOFO counterfactuals; frozen observations only, no generators."""
import csv
import os
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u2c_diagnostic as u
OUT=u.OUT
IDS=tuple(range(20));REPEATS=tuple(range(20));SNRS=(10,20);FRAMES=tuple(range(4,10));AGES=tuple(range(5))
HEAD=u.HEAD;TOL=u.TOL;EPS=u.EPS
sha=u.sha;array_hash=u.array_hash;load=u.load;save=u.save;csvsave=u.csvsave;close=u.close;normal=u.normal;read_rows=u.read_rows
ACCESS=[];_GUARDED=False
ALLOWED={'t16e_pilot_cv_preflight.json','t16e_frozen_operators.npz','t16f_preflight.json','t16f_frozen_candidates.npz',
         't16g_factor_preflight.json','t16g_frozen_module_factors.npz','sp_u2c_protocol.json',
         'sp_u2c_preregistration.json','sp_u2c_run_gate.json','sp_u2c_delivery_manifest.json','sp_u2c_repeat_rows.csv'}
ALLOWED.discard('sp_u2c_protocol.json')  # Protocol is under configs, not outputs.
ALLOWED|={f'sp_u2b_development_cache/trajectory_{i:03d}.npz' for i in IDS}
ALLOWED|={f'sp_u2c_repeat_cache/trajectory_{i:03d}_repeat_{r:02d}.npz' for i in IDS for r in REPEATS}


def output_guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    path=Path(os.fsdecode(args[0])).resolve()
    try:rel=path.relative_to(OUT.resolve()).as_posix()
    except ValueError:return
    flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
    writing=bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
    allowed=rel.split('/')[0].startswith('sp_u2d_') or (rel in ALLOWED and not writing)
    ACCESS.append(dict(path=rel,writing=writing,allowed=allowed))
    if not allowed:raise PermissionError('SP-U2D output firewall: '+rel)


def install_guard():
    global _GUARDED
    if not _GUARDED:sys.addaudithook(output_guard);_GUARDED=True


def allocation():
    return IDS,REPEATS,SNRS,FRAMES,AGES


def fixed_cache(i,metadata):
    assert i in IDS and metadata[i]['trajectory_id']==i
    record=metadata[i];path=OUT/f'sp_u2b_development_cache/trajectory_{i:03d}.npz'
    assert sha(path)==record['cache_sha256']
    with np.load(path,allow_pickle=False) as f:x=dict(H=f['H'],clean_pilots=f['pilots'][0].copy())
    assert x['H'].shape==(2,2,64,160) and array_hash(x['H'])==record['H_sha256']
    for v in x.values():v.flags.writeable=False
    return x


def repeat_cache(i,r,metadata):
    assert i in IDS and r in REPEATS
    record=metadata[(i,r)];path=OUT/record['path']
    assert record['trajectory_id']==i and record['repeat_id']==r and sha(path)==record['sha256']
    with np.load(path,allow_pickle=False) as f:p=f['pilots']
    assert p.shape==(2,10,64) and np.isfinite(p).all()
    p.flags.writeable=False
    return p


def production_references():
    # Only equality-reference fields; previous noise-effect outcomes are not analyzed.
    keep=('trajectory_id','repeat_id','snr_db','frame_id','alpha_r','beta_r','alpha_oracle','beta_oracle',
          'R_noisy','R_oracle','pilot_window_sha256','prediction_sha256')
    refs={}
    with (OUT/'sp_u2c_repeat_rows.csv').open(encoding='utf8',newline='') as f:
        for raw in csv.DictReader(f):
            row={k:(raw[k] if k.endswith('sha256') else float(raw[k])) for k in keep}
            key=tuple(int(row[k]) for k in keep[:4]);refs[key]=row
    assert set(refs)=={(i,r,s,j) for i in IDS for r in REPEATS for s in SNRS for j in FRAMES}
    return refs


def chronological_moments(cv,window):
    moments=[cv.frame_objective(p) for p in window]
    H=sum(v[0] for v in moments).real
    b=sum(v[1] for v in moments).real
    ageH=np.stack([v[0].real for v in moments[::-1]])
    ageb=np.stack([v[1].real for v in moments[::-1]])
    error=max(close(H,sum(ageH)),close(b,sum(ageb)))
    return H,b,ageH,ageb,error


def lofo_moments(H,b,ageH,ageb):
    results=[]
    for a in range(len(ageH)):
        hm,bm=H-ageH[a],b-ageb[a]
        rank=int(np.linalg.matrix_rank(hm))
        condition=float(np.linalg.cond(hm))
        singular=rank<2
        c=None
        if not singular:
            try:c=np.linalg.solve(hm,bm)
            except np.linalg.LinAlgError:singular=True
        if c is not None and not np.isfinite(c).all():raise AssertionError('nonfinite solve; no repair')
        results.append(dict(age=a,Hminus=hm,bminus=bm,rank=rank,condition=condition if np.isfinite(condition) else None,
                            singular=singular,c=c))
    return results


def frame_energy(pilots,mean):
    return np.sum(abs(pilots-mean)**2,axis=1)/64


def energy_manifest(fixed,mean):
    rows=[]
    for s in SNRS:
        group=[]
        for i in IDS:
            energies=frame_energy(fixed[i]['clean_pilots'],mean)
            for j in FRAMES:
                window=energies[j-4:j+1]
                assert np.all(window>0)
                ratio=float(window.max()/window.min())
                group.append(dict(trajectory_id=i,snr_db=s,frame_id=j,clean_current_energy=float(window[-1]),
                    clean_past_mean_energy=float(window[:-1].mean()),clean_current_past_ratio=float(window[-1]/window[:-1].mean()),
                    clean_max_min_ratio=ratio,similar_clean_energy=bool(ratio<=1.10),energy_tercile=0))
        # Equal-count, outcome-free stratification: clean current/past amplitude ratio only.
        order=sorted(range(len(group)),key=lambda k:(group[k]['clean_current_past_ratio'],group[k]['trajectory_id'],group[k]['frame_id']))
        for q,indices in enumerate(np.array_split(order,3),1):
            for k in indices:group[k]['energy_tercile']=q
        rows.extend(group)
    return rows


def record(i,r,s,j,pilots,x,method,reference,energy_info):
    from src.lowrank_modules import factored_affine_predict
    from src.spatial_information import observation_geometry
    from scripts.sp_u2b_diagnostic import explicit_design,channel_oracle
    arrays,cvs,factors,_=method;cv=cvs[s]
    window=pilots[j-4:j+1]
    cfull,diag=cv.fit(window,modules=('P','M'))
    assert np.array_equal(cfull,[reference['alpha_r'],reference['beta_r']]),'full-fit replay drift'
    assert array_hash(window)==reference['pilot_window_sha256']
    H,b,ageH,ageb,reconstruction=chronological_moments(cv,window)
    assert np.array_equal(H,np.asarray(diag['gram'])) and np.array_equal(b,np.asarray(diag['rhs']))
    geometry=observation_geometry();u_current=window[-1]-arrays['source_mean_p']
    fp,fm=factors['P',s],factors['M',s]
    X=np.column_stack((fp.apply(u_current),fm.apply(u_current)))
    base=arrays['source_mean_d']+u_current@arrays[f'Wsource_{s}'].T
    truth=u.frames(x['H'])[j].ravel()[geometry.data]
    oracle,M,oracle_error=channel_oracle(truth,base,X)
    assert np.array_equal(oracle,[reference['alpha_oracle'],reference['beta_oracle']])
    full_prediction=factored_affine_predict(window[-1],arrays['source_mean_p'],arrays['source_mean_d'],
                          arrays[f'Wsource_{s}'],cfull,fp,fm)
    assert array_hash(full_prediction)==reference['prediction_sha256']
    risk=lambda pred:float(np.vdot(pred-truth,pred-truth).real/3968)
    Rfull= risk(full_prediction)
    assert Rfull==reference['R_noisy']
    energies=frame_energy(window,arrays['source_mean_p'])[::-1]
    clean_energies=frame_energy(x['clean_pilots'][j-4:j+1],arrays['source_mean_p'])[::-1]
    rows=[]
    for result in lofo_moments(H,b,ageH,ageb):
        a=result['age'];hm,bm=result['Hminus'],result['bminus'];cm=result['c']
        # Common schema explicitly includes nulls for singular variants.
        row=dict(trajectory_id=i,repeat_id=r,snr_db=s,frame_id=j,age=a,removed_frame=j-a,retained_frames=4,
                 real_design_rows=512,ndata=3968,rank_Hminus=result['rank'],condition_Hminus=result['condition'],
                 singular=result['singular'],H_age_norm=float(np.linalg.norm(ageH[a],2)),H_age_fro=float(np.linalg.norm(ageH[a])),
                 b_age_norm=float(np.linalg.norm(ageb[a])),H_age_PP=float(ageH[a,0,0]),H_age_PM=float(ageH[a,0,1]),H_age_MM=float(ageH[a,1,1]),
                 b_age_P=float(ageb[a,0]),b_age_M=float(ageb[a,1]),Hminus_PP=float(hm[0,0]),Hminus_PM=float(hm[0,1]),Hminus_MM=float(hm[1,1]),
                 bminus_P=float(bm[0]),bminus_M=float(bm[1]),alpha_full=float(cfull[0]),beta_full=float(cfull[1]),
                 alpha_lofo=None,beta_lofo=None,delta_alpha=None,delta_beta=None,influence_norm=None,
                 Q=None,C=None,delta_R=None,alignment_cosine=None,alignment_undefined=True,
                 alignment_numerator=None,alignment_denominator=None,cosine_threshold_s=None,cosine_threshold_n=None,
                 cosine_roundoff_outside_unit=False,R_full=Rfull,R_lofo=None,
                 alpha_oracle=float(oracle[0]),beta_oracle=float(oracle[1]),M_PP=float(M[0,0]),M_PM=float(M[0,1]),M_MM=float(M[1,1]),
                 noisy_frame_energy=float(energies[a]),clean_frame_energy=float(clean_energies[a]),
                 clean_current_past_ratio=energy_info['clean_current_past_ratio'],clean_max_min_ratio=energy_info['clean_max_min_ratio'],
                 similar_clean_energy=energy_info['similar_clean_energy'],energy_tercile=energy_info['energy_tercile'],
                 noisy_similar_energy=bool(max(energies)/min(energies)<=1.10),
                 moment_reconstruction_error=reconstruction,moment_subtraction_error=None,direct_four_frame_error=None,
                 normal_error=None,risk_identity_error=None,propagation_error=None,oracle_LS_error=oracle_error,
                 numeric_borderline=False,pilot_window_sha256=array_hash(window),prediction_full_sha256=array_hash(full_prediction))
        subtraction=max(close(hm,H-ageH[a]),close(bm,b-ageb[a]))
        keep=np.arange(5)!=4-a
        G,z=explicit_design(cv,window[keep])
        subtraction=max(subtraction,close(hm,G.T@G),close(bm,G.T@z))
        row['moment_subtraction_error']=subtraction
        if cm is not None:
            direct,reference_diag=cv.fit(window[keep],modules=('P','M'))
            direct_error=max(close(cm,direct),close(cm,np.linalg.lstsq(G,z,rcond=None)[0]))
            delta=cfull-cm;sl=cm-oracle
            prediction=base+cm[0]*X[:,0]+cm[1]*X[:,1]
            Rminus=risk(prediction);Q=float(delta@M@delta/3968);C=float(2*sl@M@delta/3968)
            errors=max(close(Rfull-Rminus,Q+C),close(full_prediction-prediction,X@delta))
            normal_error=close(hm@cm,bm)
            align=u.cosine(sl,delta,M)
            row.update(alpha_lofo=float(cm[0]),beta_lofo=float(cm[1]),delta_alpha=float(delta[0]),delta_beta=float(delta[1]),
                       influence_norm=float(np.linalg.norm(delta)),Q=Q,C=C,delta_R=Rfull-Rminus,R_lofo=Rminus,
                       direct_four_frame_error=direct_error,normal_error=normal_error,risk_identity_error=close(Rfull-Rminus,Q+C),
                       propagation_error=close(full_prediction-prediction,X@delta),
                       numeric_borderline=bool(abs(Rfull-Rminus)<=TOL*max(1.,Rfull,Rminus,abs(Q)+abs(C))),**align)
            assert Q>=0
            if not row['numeric_borderline']:assert (Rfull-Rminus<0)==(C < -Q)
        numbers=[v for v in row.values() if isinstance(v,(int,float)) and not isinstance(v,bool)]
        assert np.isfinite(numbers).all(),'nonfinite record; no deletion'
        rows.append(row)
    return rows


def paired(rows):
    assert len(rows)==5 and [a['age'] for a in rows]==list(AGES)
    valid=all(not a['singular'] for a in rows)
    first=rows[0]
    result={k:first[k] for k in ('trajectory_id','repeat_id','snr_db','frame_id','clean_current_past_ratio',
                               'clean_max_min_ratio','similar_clean_energy','noisy_similar_energy','energy_tercile')}
    result.update(valid_all_ages=valid,D_coord=None,D_C=None,D_R=None,D_cosine=None,D_Hnorm=None,D_bnorm=None,D_noisy_energy=None)
    for key,out in [('influence_norm','D_coord'),('C','D_C'),('delta_R','D_R')]:
        if valid:result[out]=float(first[key]-np.mean([a[key] for a in rows[1:]]))
    if valid and all(not a['alignment_undefined'] for a in rows):
        result['D_cosine']=float(first['alignment_cosine']-np.mean([a['alignment_cosine'] for a in rows[1:]]))
    for key,out in [('H_age_norm','D_Hnorm'),('b_age_norm','D_bnorm'),('noisy_frame_energy','D_noisy_energy')]:
        result[out]=float(first[key]-np.mean([a[key] for a in rows[1:]]))
    return result
