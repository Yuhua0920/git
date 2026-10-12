"""Preregistered D-only statistical analysis. Bytes frozen before first scientific H."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import sp_u7b_common as c
from scripts import sp_u7a_statistics as stats
import numpy as np
import csv
import time

def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value

def csvwrite(path,rows):
    with Path(path).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def distribution(x):
    x=np.asarray(x)
    if not np.isfinite(x).all():raise ValueError('nonfinite descriptive distribution')
    return dict(n=int(x.size),minimum=float(x.min()),p05=float(np.quantile(x,.05,method='linear')),
                median=float(np.median(x)),p95=float(np.quantile(x,.95,method='linear')),maximum=float(x.max()),mean=float(x.mean()))

def draw_table(path,n=1000,replicates=10000,seed=764389531):
    draws=np.random.Generator(np.random.PCG64(seed)).integers(0,n,size=(replicates,n))
    with Path(path).open('xb') as f:np.save(f,draws,allow_pickle=False)
    proof=dict(created_utc=c.utc(),shape=list(draws.shape),dtype=str(draws.dtype),seed=seed,bit_generator='PCG64',
               array_sha256=__import__('hashlib').sha256(draws.tobytes()).hexdigest(),file_sha256=c.sha(path),
               saved_before_first_use=True,all_methods_and_SNRs_shared=True)
    return proof

def assert_unseal():
    frozen=c.verify_frozen();s=c.read(c.state_path());u=c.read(c.OUT/'unseal.json')
    if s['phase']!='D_UNSEALED' or not s['UNSEAL'] or not u['UNSEAL'] or u['pre_unseal_manifest_sha256']!=c.sha(c.OUT/'pre_unseal_manifest.json') or u['all_code_hashes']!=frozen['hashes']:raise RuntimeError('analysis denied: valid unseal chain required')

def main():
    assert_unseal();c.guard('analysis');start=time.perf_counter()
    # Persist and hash the entire index table BEFORE loading any scientific metric.
    drawproof=draw_table(c.OUT/'bootstrap_draws.npy');c.write(c.OUT/'bootstrap_draws_manifest.json',drawproof)
    error=[];truth=[];be=[];nb=[];coordinates=[];power=[]
    for i in range(1000):
        with np.load(c.OUT/f'evaluation/raw_{i:04d}.npz',allow_pickle=False) as f:
            error.append(f['error']);truth.append(f['truth']);be.append(f['bit_errors']);nb.append(f['total_bits']);coordinates.append(f['coordinates']);power.append(f['power'])
    error,truth,be,nb,coord,power=map(np.array,(error,truth,be,nb,coordinates,power))
    if not np.array_equal(truth[:,0],truth[:,1]) or not np.array_equal(nb[:,0],nb[:,1]):raise ValueError('paired denominators')
    result=stats.paired_bootstrap(error,truth,be,nb,764389531)
    if result['indices_sha256']!=drawproof['array_sha256']:raise ValueError('bootstrap table drift')
    traj_db=10*np.log10(error/truth[...,None]);traj_ber=be/nb[...,None]
    distributions={};tails=[];table_a=[];table_b=[]
    for si,s in enumerate(c.SNRS):
        gain=traj_db[:,si,1]-traj_db[:,si,5];berdiff=traj_ber[:,si,5]-traj_ber[:,si,1]
        gain_summary=distribution(gain);gain_summary.update(negative_fraction=float(np.mean(gain<0)),nonpositive_fraction=float(np.mean(gain<=0)),worst_id=int(np.argmin(gain)),best_id=int(np.argmax(gain)))
        distributions[str(s)]={'gain':gain_summary,'ber_LR4_minus_Source':distribution(berdiff),'methods':{}}
        for mi,m in enumerate(c.METHODS):
            distributions[str(s)]['methods'][m]=dict(nmse_db=distribution(traj_db[:,si,mi]),ber=distribution(traj_ber[:,si,mi]))
            table_a.append(dict(snr_db=s,method=m,pooled_nmse_db=result['nmse_db'][si,mi],descriptive_ci_lower=result['nmse_ci'][0,si,mi],descriptive_ci_upper=result['nmse_ci'][1,si,mi],
                primary_gain_db=result['gain'][si] if m=='LR4' else '',adjusted_one_sided_LCB=result['primary_lcb'][si] if m=='LR4' else ''))
            table_b.append(dict(snr_db=s,record='BER',method=m,point=result['ber'][si,mi],ci_lower=result['ber_ci'][0,si,mi],ci_upper=result['ber_ci'][1,si,mi],units='probability'))
        for label,r in result['secondary'].items():table_b.append(dict(snr_db=s,record=label,method='',point=r['point'][si],ci_lower=r['ci'][0,si],ci_upper=r['ci'][1,si],units='dB'))
        table_b.append(dict(snr_db=s,record='BER_LR4_minus_Source',method='',point=result['ber'][si,5]-result['ber'][si,1],ci_lower=result['ber_LR4_minus_Source_ci'][0,si],ci_upper=result['ber_LR4_minus_Source_ci'][1,si],units='probability'))
        # Every trajectory is retained: ID order, no selected-example dataset.
        for i in range(1000):
            tails.append(dict(trajectory_id=i,snr_db=s,Source_nmse_db=traj_db[i,si,1],LR4_nmse_db=traj_db[i,si,5],paired_gain_db=gain[i],
                LR4_minus_Dense_db=traj_db[i,si,5]-traj_db[i,si,4],LR4_minus_Matched_db=traj_db[i,si,5]-traj_db[i,si,6],
                Source_BER=traj_ber[i,si,1],LR4_BER=traj_ber[i,si,5],BER_LR4_minus_Source=berdiff[i]))
    csvwrite(c.OUT/'table_A.csv',table_a);csvwrite(c.OUT/'table_B.csv',table_b);csvwrite(c.OUT/'full_trajectory_tails.csv',tails)
    for si,s in enumerate(c.SNRS):
        subset=coord[:,si*6:(si+1)*6,:].reshape(-1,16)
        distributions[str(s)]['coordinate']={name:distribution(subset[:,col]) for name,col in (('alpha',2),('beta',3),('norm',4),('condition',6),('normal_residual',7),('eigen_min',14),('eigen_max',15))}
        distributions[str(s)]['coordinate'].update(Gram_rank_counts={str(k):int(np.count_nonzero(subset[:,5]==k)) for k in (0,1,2)},
                worst_condition_identity={'trajectory_id':int(np.argmax(subset[:,6])//6),'frame_id':int(np.argmax(subset[:,6])%6+4)},
                largest_coordinate_identity={'trajectory_id':int(np.argmax(subset[:,4])//6),'frame_id':int(np.argmax(subset[:,4])%6+4)})
    receive_power=power[:,0]/power[:,2];unit_power=power[:,1]/power[:,2]
    poweraudit=dict(scored_frames=list(range(4,10)),useful_samples_per_trajectory=11904,P_ref=9.941134822523972,realized_power=distribution(receive_power),
        unit_noise_power=distribution(unit_power),pooled_power=float(power[:,0].sum()/power[:,2].sum()),target_normalization=False,
        nominal_noise_variance_realized_SNR={str(s):distribution(10*np.log10(receive_power/c.VARIANCES[s])) for s in c.SNRS},
        atom_energy_realized_SNR={str(s):distribution(10*np.log10(receive_power/(c.VARIANCES[s]*unit_power))) for s in c.SNRS})
    case=result['decision'][0];h1=case=='A'
    compression_points=result['secondary']['LR4_minus_Dense']['point'];compression_low=result['secondary']['LR4_minus_Dense']['ci'][0]
    berpoints=result['ber'][:,5]-result['ber'][:,1];berci=result['ber_LR4_minus_Source_ci']
    compression_status='MIXED' if np.any(compression_low>0) else 'NOT TESTED'
    # No predeclared compression noninferiority/closeness margin: do not invent one.
    claimmatrix=[dict(claim='unseen-profile transfer at tested A120/100 setting',status='SUPPORTED' if h1 else ('REJECTED' if case=='C' else 'MIXED'),evidence='independent sealed validation;2-cell Bonferroni family'),
        dict(claim='near matched reference',status='NOT TESTED',evidence='descriptive finite-reference paired gap; no preregistered closeness/equivalence threshold'),
        dict(claim='compression OOD adequacy',status=compression_status,evidence='same-coordinate paired LR4-Dense gap; no frozen noninferiority margin'),
        dict(claim='BER transfer',status='SUPPORTED' if np.all(berci[1]<0) else ('REJECTED' if np.all(berci[0]>0) else 'MIXED'),evidence='secondary descriptive paired trajectory bootstrap; not primary multiplicity-controlled'),
        dict(claim='universal OOD robustness',status='NOT TESTED',evidence='single NLOS profile/mobility/delay condition only')]
    summary=dict(task='SP-U7B',completed_utc=c.utc(),engineering_verdict='PASS',unseal_gate='13/13 PASS',protocol_deviations=[],
        N=1000,matched_reference_N=500,scored_frames=6,bootstrap=drawproof,results=plain(result),distributions=distributions,
        power_audit=poweraudit,claim_evidence=claimmatrix,H1_supported=h1,H2='Source gain does not establish near-matched; gap reported without new closeness margin',
        H3='positive LR4-Dense gap with CI excludes0 records compression limitation; no rank change',H4='all null/negative/tail evidence retained; no rescue',
        uncertainty_scope='evaluation trajectory bootstrap conditional on fixed500-trajectory Matched reference; reference-training uncertainty not resampled',
        statistics_audit={'unit':'trajectory','pooled_energy_estimand':True,'same_draws_7_methods_2_SNR':True,'replicates':10000,'family_alpha':.05,'one_sided_cell_alpha':.025,'linear_quantile':True,'no_drop':True},
        runtime_seconds=time.perf_counter()-start,resource=c.resources(),data_use='USED validation, permanently not development',next_action='STOP: wait for mentor review')
    c.write(c.OUT/'analysis.json',summary)
    from scripts.plot_sp_u7b import render
    render(c.OUT/'figures',traj_db,result)
    c.verify_frozen()
    c.transition('D_UNSEALED','D_ANALYSIS_COMPLETE')
    print('PHASE D analysis complete; case',result['decision'],'two tables and two figures; STOP waiting mentor',flush=True)

def finalize():
    """Delivery manifest after frozen-source post-run pytest and figure checks."""
    c.verify_frozen()
    if c.read(c.state_path())['phase']!='D_ANALYSIS_COMPLETE':raise RuntimeError('analysis must be complete')
    import xml.etree.ElementTree as ET
    import subprocess
    suites=[x.attrib for x in ET.parse(c.OUT/'pytest_final.xml').getroot().iter('testsuite')]
    if not suites or any(int(x.get(k,'0')) for x in suites for k in ('errors','failures','skipped')):raise RuntimeError('post-run tests required')
    for name in ('Figure_A','Figure_B'):
        if not c.read(c.OUT/f'figures/{name}_QA.json')['PASS']:raise RuntimeError('figure QA')
    protected=c.read(c.ROOT/'outputs/sp_u7a_protected_inputs.json');c.verify(protected['hashes'])
    diff=subprocess.check_output(['git','diff','--name-only'],cwd=c.ROOT,text=True).strip()
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=c.ROOT,text=True).strip()
    if diff or staged:raise RuntimeError('tracked changes forbidden')
    c.transition('D_ANALYSIS_COMPLETE','COMPLETE',mentor_review_required=True,next_task_authorized=False)
    files={p.relative_to(c.ROOT).as_posix():c.sha(p) for p in c.OUT.rglob('*') if p.is_file()}
    sources={p:c.sha(c.ROOT/p) for p in c.NEW_SOURCES}
    c.write(c.OUT/'delivery_manifest.json',dict(task='SP-U7B',verdict='PASS',created_utc=c.utc(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip(),tracked_diff_empty=True,
        source_hashes=sources,file_hashes=files,pytest_phase_A=c.read(c.OUT/'freeze.json')['pytest'],pytest_final=suites,
        scientific_case=c.read(c.OUT/'analysis.json')['results']['decision'],UNSEAL=True,gate_pass_count=13,
        no_Git_commit=True,no_manuscript_changes=True,all_prior_branches_frozen_or_stopped=True,final_stop='wait mentor review; no U7C/rescue/newprofile'))
    print('SP-U7B COMPLETE; delivery manifest generated; STOP waiting mentor review',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='finalize':finalize()
    else:main()
