"""Bounded post-hoc diagnostics on saved evidence; no scientific generation/refit."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys,csv,json,hashlib,time,zipfile
from pathlib import Path
from decimal import Decimal,localcontext
import numpy as np
from scipy.stats import spearmanr
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from scripts import sp_u8a_common as c
from src.lowrank_modules import factored_affine_predict
from mr10_detectors import linear,bit_map,zf_risk
R1=c.R1;MR2=ROOT/'outputs/revision_mr02_coordinate_mechanism'
COND=('ID','Profile','Mobility','Composed','Boundary')
METHODS=('PERFECT_ZF','PERFECT_MMSE','PERFECT_BIT_MAP','LS_ZF','LR4_ZF','LS_MMSE','LR4_MMSE')
PREF=9.941134822523972
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def table(p,rows):
    with Path(p).open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def csvrows(p):
    with Path(p).open(encoding='utf8',newline='') as f:return list(csv.DictReader(line for line in f if not line.startswith('#')))
def npz(p,**kw):
    with Path(p).open('xb') as f:np.savez_compressed(f,**kw)
def need(ok,msg):
    if not ok:raise RuntimeError(msg)
def auth(p,expected):need(sha(p)==expected,'INPUT_SHA256_DRIFT:'+str(p))
def prepare():
    protocol={'task':'MR10','role':'POST_HOC_DEVELOPMENT_DIAGNOSTICS','receiver_source':'SP-U7B-R1-CDLA120-V2',
      'trajectory_ids':list(range(100)),'snrs':[10,20],'frames':list(range(4,10)),'methods':METHODS,
      'source_reference_power':PREF,'variances':c.VARIANCES,'per_stream_data_power':.5,
      'MMSE_policy':'H^H H + sigma2/0.5 I, same policy for every supplied channel estimate',
      'MAP_policy':'16 equally probable transmit vectors; proper-complex likelihood; marginal bit posterior',
      'coordinate_source':'MR02 saved D_eval, all 9000 windows; no refit',
      'coordinate_conditions':COND,'coordinate_snrs':[10,20,30],
      'new_bootstrap_draws':0,'new_formal_family':False,'channel_or_noise_generation':False,
      'selection':'first 100 original IDs, fixed before diagnostics; not selected by performance'}
    write(OUT/'reports/protocol.json',protocol)
    manifest=read(R1/'scientific_archive_manifest.json')['hashes']
    delivery=read(R1/'delivery_manifest.json')
    paths=set()
    for i in range(100):
        for rel in (f'calls/EVAL/{i:04d}/H.npz',f'evaluation/comm_{i:04d}.npz',f'evaluation/raw_{i:04d}.npz'):
            auth(R1/rel,manifest[rel]);paths.add(R1/rel)
        for rel in (f'calls/EVAL/{i:04d}/call.json',f'evaluation/proof_{i:04d}.json'):
            p=R1/rel;auth(p,delivery['file_hashes'][p.relative_to(ROOT).as_posix()]);paths.add(p)
    for name in ('t16e_frozen_operators.npz','t16f_frozen_candidates.npz','t16g_frozen_module_factors.npz'):
        paths.add(ROOT/'outputs'/name)
    for p in (ROOT/'src').rglob('*.py'):paths.add(p)
    for p in (ROOT/'scripts').glob('sp_u7b*.py'):paths.add(p)
    paths.add(ROOT/'scripts/sp_u8a_common.py')
    for p in (ROOT/'outputs/paper_v5_candidate').rglob('*'):
        if p.is_file():paths.add(p)
    for rel in ('trajectories.csv','frames.csv','analysis.json','replay_gate.json','conditioning.csv'):
        p=ROOT/'outputs/sp_u8a'/rel
        if p.is_file():paths.add(p)
    paths.update((R1/'scientific_archive_manifest.json',R1/'delivery_manifest.json',MR2/'reports/evaluation_receipt.json',MR2/'results_review_file_manifest.csv',MR2/'data/eval_summary.npz'))
    for block in read(MR2/'reports/evaluation_receipt.json')['blocks']:
        p=MR2/block['path'];auth(p,block['sha256']);auth(p.with_suffix('.json'),block['json_sha256']);paths.update((p,p.with_suffix('.json')))
    write(OUT/'reports/preservation_before.json',{'files':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)},'scope':'all actually used historical inputs, MR02 blocks, production Python source, frozen objects and V5 candidate'})
    write(OUT/'reports/preparation.json',{'PASS':True,'receiver_input_archives':300,'receiver_metadata':200,'MR02_blocks':500})
    print('PREPARATION PASS: historical hashes authenticated; fixed scope saved',flush=True)
def receiver():
    authdict=read(OUT/'reports/preservation_before.json')['files']
    g=c.observation_geometry();mask=g.layout.data_mask[0]
    need(np.array_equal(g.layout.data_mask[0],g.layout.data_mask[1]),'DATA_MASK_NOT_SHARED')
    need(mask.sum()==992,'DATA_RE_COUNT')
    a,f,_=c.frozen.load_deployables()
    framecounts={(int(r['trajectory_id']),int(r['snr_db']),int(r['frame_id']),r['method']):int(r['bit_errors']) for r in csvrows(ROOT/'outputs/sp_u8a/frames.csv')}
    output=OUT/'data/receiver';output.mkdir(exist_ok=False)
    counter={'input_array_hash_checks':0,'prediction_hash_checks':0,'historical_frame_count_checks':0,'trajectories':0,'failures':0}
    started=time.perf_counter()
    for i in range(100):
        hf,comm,raw,proof=c.ingredients(i)
        pi={(v['snr_db'],v['frame_id']):v for v in proof['inputs']}
        po={(v['snr_db'],v['frame_id'],v['method']):v for v in proof['outputs']}
        refs=[];masks=np.empty((2,7,6,992,4),bool)
        svs=[];kaps=[];gains=[];powers=[];realpower=[];noise=[];pred_identity=[];snreff=[];berexpected=[]
        for j,h in enumerate(hf):
            bits,symbols,z=comm[j]
            x=c.assemble_pilot_data_frame(symbols,g.layout);clean=c.apply_frequency_mimo_channel(h,x)
            if j>=4:
                hr=h[:,:,mask].transpose(2,0,1)
                sv,k,gain,_,_=zf_risk(hr,c.VARIANCES[10]);svs.append(sv);kaps.append(k);gains.append(gain)
                powers.append(.5*np.mean(np.sum(abs(hr)**2,axis=2)))
                realpower.append(float(np.mean(abs(clean[:,mask])**2)))
                noise.append(z[:,mask].T)
                ref=bits[:,mask,:].transpose(1,0,2).reshape(992,4);refs.append(ref)
            for si,s in enumerate((10,20)):
                y=clean+np.sqrt(c.VARIANCES[s])*z;y.flags.writeable=False
                sparse,ls=c.estimate_ls_linear_channel(y,g.layout);pilot=sparse.ravel()[g.selected]
                inp={'H_sha256':c.array_hash(h),'bits_sha256':c.array_hash(bits),'symbols_sha256':c.array_hash(symbols),
                    'unit_noise_sha256':c.array_hash(z),'Y_sha256':c.array_hash(y),'pilot_sha256':c.array_hash(pilot)}
                for key,val in inp.items():need(val==pi[s,j][key],f'INPUT_IDENTITY:{i}:{s}:{j}:{key}');counter['input_array_hash_checks']+=1
                if j<4:continue
                ji=j-4
                coordinate=raw['coordinates'][si*6+ji,2:4]
                pred=factored_affine_predict(pilot,a['source_mean_p'],a['source_mean_d'],a[f'Wsource_{s}'],coordinate,f['P',s,4],f['M',s,4])
                lr=sparse.copy().ravel();lr[g.data]=pred;lr=lr.reshape(2,2,64,16)
                for method,v in (('LS',ls.ravel()[g.data]),('LR4',pred)):
                    hh=c.array_hash(v);need(hh==po[s,j,method]['prediction_sha256'],f'PREDICTION_DRIFT:{i}:{s}:{j}:{method}')
                    pred_identity.append({'snr_db':s,'frame_id':j,'method':method,'prediction_sha256':hh});counter['prediction_hash_checks']+=1
                yy=y[:,mask].T;lsh=ls[:,:,mask].transpose(2,0,1);lrh=lr[:,:,mask].transpose(2,0,1)
                outputs=(linear(hr,yy,c.VARIANCES[s])[0],linear(hr,yy,c.VARIANCES[s],True)[0],bit_map(hr,yy,c.VARIANCES[s])[0],
                    linear(lsh,yy,c.VARIANCES[s])[0],linear(lrh,yy,c.VARIANCES[s])[0],
                    linear(lsh,yy,c.VARIANCES[s],True)[0],linear(lrh,yy,c.VARIANCES[s],True)[0])
                for mi,v in enumerate(outputs):masks[si,mi,ji]=v!=ref
                for method,mi in (('Perfect',0),('LS',3),('LR4',4)):
                    actual=int(masks[si,mi,ji].sum());need(actual==framecounts[i,s,j,method],f'BER_REPLAY_DRIFT:{i}:{s}:{j}:{method}');counter['historical_frame_count_checks']+=1
        gains=np.array(gains);noise=np.array(noise)
        for s in (10,20):
            esn0=.5/(c.VARIANCES[s]*gains)
            from scipy.special import erfc
            snreff.append(10*np.log10(esn0));berexpected.append(.5*erfc(np.sqrt(esn0/2)))
        target_power=float(np.mean(powers))
        record={'identity':f'MR10:CDLA:original:{i}','original_run_id':proof['run_id'],'trajectory_id':i,'snrs':[10,20],
            'frames':list(range(4,10)),'methods':METHODS,'input_hashes':{k:v for k,v in authdict.items() if f'{i:04d}' in k and ('sp_u7b_r1/' in k)},
            'prediction_hashes':pred_identity,'target_receive_power_expected':target_power,'target_receive_power_realized':float(np.mean(realpower)),
            'source_reference_power':PREF,'actual_receive_snr_db':[s+10*np.log10(target_power/PREF) for s in (10,20)],
            'noise_real_second_moment':float(np.mean(noise.real**2)),'noise_imag_second_moment':float(np.mean(noise.imag**2)),
            'noise_complex_second_moment':float(np.mean(abs(noise)**2)),'noise_re_im_cross_moment':float(np.mean(noise.real*noise.imag)),
            'noise_rx_cross_moment_abs':float(abs(np.mean(noise[...,0]*noise[...,1].conj()))),
            'bit_errors':masks.sum(axis=(2,3,4)).tolist(),'total_bits_per_method_snr':23808,'finite_failures':0}
        p=output/f'{i:04d}.npz'
        npz(p,bit_error_mask=masks,reference_bits=np.array(refs),singular_values=np.array(svs),condition_number=np.array(kaps),
            zf_noise_gain=gains,zf_effective_esn0_db=np.array(snreff),conditional_zf_bit_ber=np.array(berexpected),
            frame_receive_power_expected=np.array(powers),frame_receive_power_realized=np.array(realpower),
            methods=np.array(METHODS),snrs=np.array([10,20]),frames=np.arange(4,10),trajectory_id=np.array(i))
        record['raw_sha256']=sha(p);write(p.with_suffix('.json'),record)
        counter['trajectories']+=1
        if i==0:print('REAL FIXED-ID SMOKE PASS: input/prediction hashes, saved bit masks, legacy counts exact',flush=True)
        if (i+1)%10==0:print(f'RECEIVER {i+1}/100 elapsed {time.perf_counter()-started:.1f}s',flush=True)
    write(OUT/'reports/receiver_integrity.json',{'PASS':True,**counter,'elapsed_seconds':time.perf_counter()-started,'new_inference_positions':1200,'data_REs_per_frame':992,'bits_per_SNR_method':2380800,'history_replay':'coordinates only, no refitting'})
def coordinate():
    expected={r['path']:r['sha256'] for r in csvrows(MR2/'results_review_file_manifest.csv')}
    auth(MR2/'data/eval_summary.npz',expected['data/eval_summary.npz'])
    with np.load(MR2/'data/eval_summary.npz',allow_pickle=False) as z:allz={k:z[k] for k in z.files}
    keys=('pilot_S','pilot_b','online','online_normal_residual','error','truth','pilot_min_eigen_normalized','pilot_condition')
    blocks=read(MR2/'reports/evaluation_receipt.json')['blocks'];need(len(blocks)==500,'BLOCK_COUNT')
    for b in blocks:
        ci=COND.index(b['condition']);i=b['trajectory_id'];p=MR2/b['path']
        auth(p,b['sha256']);auth(p.with_suffix('.json'),b['json_sha256']);meta=read(p.with_suffix('.json'))
        need(meta['identity']==f'MR02:evaluation:{COND[ci]}:{i}' and meta['complete'],'COORDINATE_IDENTITY')
        with zipfile.ZipFile(p) as zz:need(zz.testzip() is None,'COORDINATE_CRC')
        with np.load(p,allow_pickle=False) as z:
            for k in keys:need(np.array_equal(z[k],allz[k][ci,i]),f'SUMMARY_BLOCK_DRIFT:{b["path"]}:{k}')
    S=allz['pilot_S'];b=allz['pilot_b'];coef=allz['online']
    need(S.shape==(5,100,3,6,2,2) and b.shape==(5,100,3,6,2) and coef.shape==b.shape,'COORDINATE_SHAPE')
    need(np.isfinite(S).all() and np.isfinite(b).all() and np.isfinite(coef).all(),'NONFINITE_COORDINATE')
    eig=np.linalg.eigvalsh(S);trace=np.trace(S,axis1=-2,axis2=-1);kappa=eig[...,1]/eig[...,0]
    residual=np.linalg.norm((S@coef[...,None])[...,0]-b,axis=-1)
    backward=residual/(eig[...,1]*np.linalg.norm(coef,axis=-1)+np.linalg.norm(b,axis=-1))
    truth=np.broadcast_to(allz['truth'][:,:,None,:],(5,100,3,6))
    # Persist the exact saved sufficient quantities before producing summaries.
    npz(OUT/'data/coordinate_windows.npz',S=S,b=b,coordinates=coef,eigenvalues=eig,trace=trace,kappa=kappa,
        backward_error=backward,original_normal_residual=allz['online_normal_residual'],
        Dense_error=allz['error'][...,4],LR4_error=allz['error'][...,5],truth=truth)
    rows=[]
    for ci,co in enumerate(COND):
        for i in range(100):
            for si,s in enumerate((10,20,30)):
                for ji,j in enumerate(range(4,10)):
                    ix=(ci,i,si,ji);aa,bb=coef[ix]
                    rows.append({'condition':co,'trajectory_id':i,'snr_db':s,'frame_id':j,'lambda_min':eig[ix][0],
                      'lambda_max':eig[ix][1],'trace':trace[ix],'lambda_min_over_trace':eig[ix][0]/trace[ix],
                      'condition_number':kappa[ix],'alpha':aa,'beta':bb,'abs_alpha':abs(aa),'abs_beta':abs(bb),
                      'coordinate_norm':float(np.linalg.norm(coef[ix])),'backward_error':backward[ix],
                      'Dense_error_energy':allz['error'][ix][4],'LR4_error_energy':allz['error'][ix][5],
                      'truth_energy':truth[ix],'LR4_frame_NMSE':allz['error'][ix][5]/truth[ix]})
    table(OUT/'tables/coordinate_windows.csv',rows)
    decimals=[]
    with localcontext() as context:
        context.prec=60
        for flat in np.argsort(kappa.ravel())[-100:]:
            ix=np.unravel_index(flat,kappa.shape);mat=S[ix];rhs=b[ix]
            a1,a2,a3,a4=map(Decimal.from_float,map(float,mat.ravel()));v1,v2=map(Decimal.from_float,map(float,rhs))
            det=a1*a4-a2*a3;x1=(a4*v1-a2*v2)/det;x2=(a1*v2-a3*v1)/det
            precise=np.array([float(x1),float(x2)]);delta=np.linalg.norm(precise-coef[ix])/(1+np.linalg.norm(precise))
            decimals.append({'condition':COND[ix[0]],'trajectory_id':ix[1],'snr_db':(10,20,30)[ix[2]],'frame_id':ix[3]+4,
                'condition_number':kappa[ix],'relative_coordinate_roundoff':delta})
    table(OUT/'tables/high_precision_solve_check.csv',decimals)
    write(OUT/'reports/coordinate_integrity.json',{'PASS':True,'windows_expected':9000,'windows_saved':len(rows),
        'unique_identities':len({(r['condition'],r['trajectory_id'],r['snr_db'],r['frame_id']) for r in rows}),
        'source_blocks_authenticated':500,'nonfinite_windows':0,'nonpositive_eigenvalue_windows':int(np.sum(eig[...,0]<=0)),
        'numeric_rank_deficient_windows':int(np.sum(eig[...,0]<=2*np.finfo(float).eps*eig[...,1])),
        'max_backward_error':float(np.max(backward)),'min_normalized_eigenvalue':float(np.min(eig[...,0]/trace)),
        'max_condition_number':float(np.max(kappa)),'max_alpha_abs':float(np.max(abs(coef[...,0]))),
        'max_beta_abs':float(np.max(abs(coef[...,1]))),'max_high_precision_relative_difference':float(max(r['relative_coordinate_roundoff'] for r in decimals)),
        'claim_scope':'saved finite observed windows only; numerical conditioning is distinct from final data risk'})
    print('COORDINATE PASS: 500 blocks, all 9000 windows, 100 high-precision checks',flush=True)
def qstats(x):
    x=np.asarray(x);return {f'q{k}':float(np.quantile(x,k/100)) for k in (0,5,50,95,99,100)}
def analyze():
    need(read(OUT/'reports/receiver_integrity.json')['PASS'] and read(OUT/'reports/coordinate_integrity.json')['PASS'],'RAW_GATE')
    hist=csvrows(ROOT/'outputs/sp_u8a/trajectories.csv');need(len(hist)==10000,'HISTORY_ROWS')
    hrows=[];pairs=[]
    for s in (10,20):
        lookup={m:sorted((r for r in hist if int(r['snr_db'])==s and r['method']==m),key=lambda r:int(r['trajectory_id'])) for m in ('Perfect','LS','Source','LR4','Matched')}
        for m,rr in lookup.items():
            need(len(rr)==1000 and [int(r['trajectory_id']) for r in rr]==list(range(1000)),'HISTORY_IDENTITIES')
            err=sum(int(r['bit_errors']) for r in rr);tot=sum(int(r['total_bits']) for r in rr)
            need(tot==23808000,'HISTORY_DENOMINATOR');hrows.append({'snr_db':s,'method':m,'bit_errors':err,'total_bits':tot,'BER':err/tot,'trajectories':1000})
        perfect=np.array([int(r['bit_errors']) for r in lookup['Perfect']]);ls=np.array([int(r['bit_errors']) for r in lookup['LS']]);d=(ls-perfect)/23808
        pairs.append({'snr_db':s,'contrast':'LS_ZF_minus_Perfect_ZF','mean_BER_difference':float(d.mean()),'trajectory_SD':float(d.std(ddof=1)),
          'descriptive_SE':float(d.std(ddof=1)/np.sqrt(1000)),'LS_better_trajectories':int(np.sum(d<0)),
          'LS_worse_trajectories':int(np.sum(d>0)),'ties':int(np.sum(d==0)),**qstats(d)})
    table(OUT/'tables/historical_BER.csv',hrows);table(OUT/'tables/historical_paired_fluctuation.csv',pairs)
    raw=[];meta=[]
    for i in range(100):
        p=OUT/f'data/receiver/{i:04d}.npz';m=read(p.with_suffix('.json'));auth(p,m['raw_sha256'])
        with np.load(p,allow_pickle=False) as z:raw.append({k:z[k] for k in z.files})
        need(int(raw[-1]['trajectory_id'])==i,'ANALYSIS_IDENTITY');meta.append(m)
    masks=np.stack([r['bit_error_mask'] for r in raw]);err=masks.sum(axis=(3,4,5));n=23808
    need(err.shape==(100,2,7) and np.array_equal(err,np.array([m['bit_errors'] for m in meta])),'RAW_BIT_COUNT')
    rows=[];trajectory=[];paired=[]
    for si,s in enumerate((10,20)):
        for mi,m in enumerate(METHODS):
            e=err[:,si,mi];rows.append({'snr_db':s,'method':m,'bit_errors':int(e.sum()),'total_bits':100*n,'BER':float(e.sum()/(100*n)),'trajectories':100,'role':'POST_HOC_FIXED_100'})
            for i in range(100):trajectory.append({'trajectory_id':i,'snr_db':s,'method':m,'bit_errors':int(e[i]),'total_bits':n,'BER':float(e[i]/n)})
        for ma,mb in ((0,1),(0,2),(3,4),(5,6),(3,0)):
            d=(err[:,si,ma]-err[:,si,mb])/n
            aa=masks[:,si,ma];bb=masks[:,si,mb]
            paired.append({'snr_db':s,'contrast':METHODS[ma]+'_minus_'+METHODS[mb],
              'mean_BER_difference':float(d.mean()),'trajectory_SD':float(d.std(ddof=1)),
              'descriptive_SE':float(d.std(ddof=1)/10),'first_worse_trajectories':int(np.sum(d>0)),
              'first_better_trajectories':int(np.sum(d<0)),
              'first_wrong_second_right_bits':int(np.sum(aa & ~bb)),
              'first_right_second_wrong_bits':int(np.sum(~aa & bb))})
    table(OUT/'tables/detector_BER.csv',rows);table(OUT/'tables/detector_trajectory_errors.csv',trajectory);table(OUT/'tables/detector_paired_differences.csv',paired)
    powerrows=[];riskrows=[];correlations=[]
    for i in range(100):
        rr=raw[i];mm=meta[i]
        for si,s in enumerate((10,20)):
            analytic=float(np.mean(rr['conditional_zf_bit_ber'][si]));actual=err[i,si,0]/n
            powerrows.append({'trajectory_id':i,'snr_db':s,'target_receive_power_expected':mm['target_receive_power_expected'],
                'target_receive_power_realized':mm['target_receive_power_realized'],'source_reference_power':PREF,
                'actual_receive_snr_db':mm['actual_receive_snr_db'][si],
                'post_ZF_EsN0_db_median':float(np.median(rr['zf_effective_esn0_db'][si])),
                'conditional_ZF_expected_BER':analytic,'observed_Perfect_ZF_BER':float(actual),
                'observed_minus_conditional_expected':float(actual-analytic),
                'noise_complex_second_moment':mm['noise_complex_second_moment'],
                'noise_real_second_moment':mm['noise_real_second_moment'],'noise_imag_second_moment':mm['noise_imag_second_moment'],
                'noise_re_im_cross_moment':mm['noise_re_im_cross_moment'],'noise_RX_cross_moment_abs':mm['noise_rx_cross_moment_abs']})
    table(OUT/'tables/receive_power_SNR_trajectory.csv',powerrows)
    for name,xx in (('sigma_min',np.stack([r['singular_values'][...,1] for r in raw])),
       ('H_condition_number',np.stack([r['condition_number'] for r in raw])),
       ('ZF_noise_gain_trace',np.stack([r['zf_noise_gain'].sum(axis=-1) for r in raw]))):
        riskrows.append({'quantity':name,'snr_db':'BOTH','samples':int(xx.size),**qstats(xx)})
    for si,s in enumerate((10,20)):
        for name,xx in (('post_ZF_EsN0_db',np.stack([r['zf_effective_esn0_db'][si] for r in raw])),
          ('conditional_ZF_bit_BER',np.stack([r['conditional_zf_bit_ber'][si] for r in raw])),
          ('actual_receive_SNR_db',np.array([m['actual_receive_snr_db'][si] for m in meta]))):
            riskrows.append({'quantity':name,'snr_db':s,'samples':int(xx.size),**qstats(xx)})
        v=[r for r in powerrows if r['snr_db']==s]
        aa=np.array([r['conditional_ZF_expected_BER'] for r in v]);yy=np.array([r['observed_Perfect_ZF_BER'] for r in v])
        correlations.append({'scope':'receiver_100_trajectories','snr_db':s,'quantity':'conditional_expected_ZF_BER_vs_observed',
          'Spearman':float(spearmanr(aa,yy).statistic),'mean_expected':float(aa.mean()),'mean_observed':float(yy.mean()),
          'mean_residual':float((yy-aa).mean()),'residual_trajectory_SD':float((yy-aa).std(ddof=1))})
    table(OUT/'tables/ZF_risk_distribution.csv',riskrows);table(OUT/'tables/conditional_ZF_prediction_check.csv',correlations)
    cw=csvrows(OUT/'tables/coordinate_windows.csv');summary=[];assoc=[]
    for co in COND:
        for s in (10,20,30):
            rr=[r for r in cw if r['condition']==co and int(r['snr_db'])==s];need(len(rr)==600,'CELL_WINDOWS')
            for name in ('lambda_min','lambda_min_over_trace','condition_number','alpha','beta','abs_alpha','abs_beta','coordinate_norm','backward_error','LR4_frame_NMSE'):
                xx=np.array([float(r[name]) for r in rr]);summary.append({'condition':co,'snr_db':s,'quantity':name,'windows':600,'independent_trajectories':100,**qstats(xx)})
            y=np.array([float(r['LR4_frame_NMSE']) for r in rr])
            for name in ('condition_number','lambda_min_over_trace','coordinate_norm','abs_alpha','abs_beta'):
                xx=np.array([float(r[name]) for r in rr]);
                ty=np.array([sum(float(r['LR4_error_energy']) for r in rr if int(r['trajectory_id'])==i)/sum(float(r['truth_energy']) for r in rr if int(r['trajectory_id'])==i) for i in range(100)])
                tx=np.array([np.median([float(r[name]) for r in rr if int(r['trajectory_id'])==i]) for i in range(100)])
                assoc.append({'condition':co,'snr_db':s,'quantity':name,'window_Spearman_descriptive':float(spearmanr(xx,y).statistic),
                    'trajectory_Spearman_descriptive':float(spearmanr(tx,ty).statistic),'p_values_reported':False,'scope':'association, not causation'})
    table(OUT/'tables/coordinate_distribution.csv',summary);table(OUT/'tables/coordinate_risk_association.csv',assoc)
    write(OUT/'reports/analysis_integrity.json',{'PASS':True,'historical_trajectories':1000,'new_diagnostic_trajectories':100,
        'receiver_masks_authenticated':100,'coordinate_windows':9000,'bootstrap_executed':False,
        'receiver_nonfinite_failures':0,'missing_trajectories':0,'independent_unit':'trajectory; frames/REs/bits not treated as independent replications'})
    print('ANALYSIS PASS: historical counts, bit-mask recount, distributions; no bootstrap',flush=True)
def preserve():
    old=read(OUT/'reports/preservation_before.json')['files'];bad=[p for p,h in old.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    write(OUT/'reports/preservation_after.json',{'PASS':not bad,'files_checked':len(old),'changed_files':bad,
        'manuscript_modifications':0,'production_source_modifications':0,'historical_artifact_modifications':0})
    need(not bad,'HISTORICAL_DRIFT');print(f'PRESERVATION PASS: {len(old)} files unchanged',flush=True)
if __name__=='__main__':
    action=sys.argv[1]
    try:globals()[action]()
    except Exception as e:
        import traceback
        p=OUT/'logs'/f'{action}_exception.json'
        if not p.exists():write(p,{'exception':repr(e),'traceback':traceback.format_exc()})
        raise
