"""Source audit and outcome-free fairness contract, before toy/CDL smoke."""
import csv
import importlib.metadata as metadata
import inspect
from pathlib import Path
import sys
import tomllib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u3a_common as c
import numpy as np

def main():
    c.verify();base=c.OUT/'sp_u3a_sources';repositories=[]
    for name in ('cplx-mfa','MFA_estimator'):
        pin=c.load(base/name/'repository_pin.json')
        assert all(c.sha(base/name/p)==h for p,h in pin['files'].items())
        project=tomllib.loads((base/name/'pyproject.toml').read_text(encoding='utf8'))['project']
        assert project['version']=='0.1.1' and project['license']=='BSD-3-Clause'
        license_text=(base/name/'LICENSE').read_text(encoding='utf8');assert 'Redistribution and use' in license_text
        pkg=c.load(base/('cplx_mfa_pypi.json' if name=='cplx-mfa' else 'mfa_estimator_pypi.json'))
        repositories.append(dict(**pin,version=project['version'],PyPI_version=pkg['info']['version'],
            PyPI_release_files=pkg['releases'].get('0.1.1'),dependencies=project['dependencies'],
            license_file_sha256=c.sha(base/name/'LICENSE'),tests=[p for p in pin['files'] if p.startswith('tests/')]))
    publisher=c.load(base/'publisher_metadata.json')['message']
    assert publisher['page']=='375-380' and len(publisher['author'])==3
    c.save(c.OUT/'sp_u3a_mfa_source_audit.json',dict(repositories=repositories,
      paper=dict(title=publisher['title'][0],authors=publisher['author'],venue=publisher['container-title'][0],
        DOI=publisher['DOI'],pages=publisher['page'],manuscript='arXiv:2304.14809v2',revision='2023-05-01',
        pdf_sha256=c.sha(base/'paper_v2.pdf'),publisher_metadata_sha256=c.sha(base/'publisher_metadata.json')),
      dissertation=dict(title='Generative Model-Aided Channel Estimation Design and Optimality Analysis',author='Benedikt Fesl',year=2025,
         URL='https://mediatum.ub.tum.de/doc/1748775/1748775.pdf',pdf_sha256=c.sha(base/'dissertation_2025.pdf'),
         verified_sections=['2.1 eq2.1 generalA','2.3.3 eq2.35 MFA and clean training','3.1.1 eq3.6-3.8 responsibilities/CME',
          '3.1.2 eq3.9 general-A MFA; isotropic Psi for remainder','3.2.1 invertible-A theorem scope not applicable to rectangular selector']),
      API=dict(fit='fit(X[samples,N])',estimate='estimate(y[samples,M],Cn[M,M],A[M,N],n_summands_or_proba=1.0)',
        complex=True,batch=True,general_A=True,dense_A_required=True,noise='complex covariance matrix',
        conditional_responsibilities='private _prepare_observation_model/_predict_observation_proba; predict_proba is CLEAN-prior-domain',
        default_estimate='integer1 top-component; full mixture requires float1.0',
        means='learned by default; zero_mean optional',Psi='diagonal default; ppca isotropic; lock_psis shared',
        numerical_details=['author observation precision uses hermitian pinv','safe prior weights floored at machine eps',
          'EM Psi floor1e-6','EM responsibility clipping optional, rs_clip default0',
          'max_condition_number scales random initialization; not an enforced covariance condition bound',
          'KMeans real-imag concatenation n_init1; complex random W',
          'relative lower-bound convergence after iteration>5; RuntimeWarning if max_iter exhausted']),
      paper_software_differences=['2023 y=h+n; dissertation/software support general rectangularA',
        '2023 full mixture eq11; software default integer1 truncates, overridden to float1.0',
        '2023 diagonal Psi and optional common/isotropic; dissertation experiments isotropic; software default diagonal',
        'same proper-complex low-rank-plus-diagonal family, current software numerical floors and dense caches explicit',
        'faithful equation/port audit, not exact reproduction of 2023 measured-data performance'],
      dependencies_installed=False,source_modified=False,access=c.ACCESS))
    from src.cdl_benchmark import FrozenStatistics,array_hash
    paths={'Source/CDL-C-30':'t15a_cdl_statistics_500.npz','CDL-B-30':'t15c_cdl_statistics_CDL-B-30.npz','CDL-C-120':'t15c_cdl_statistics_CDL-C-120.npz'}
    statistics={};ledger=[]
    for env,path in paths.items():
        with np.load(c.OUT/path,allow_pickle=False) as f:
            ids=tuple(map(int,f['train_trajectory_ids']));samples=int(f['samples'])
            assert ids==tuple(range(500)) and samples==5000
            statistics[env]=FrozenStatistics(*(f[k].copy() for k in ('mean_p','mean_d','rpp','rdp')),ids,samples)
        ledger.append(dict(environment=env,clean_noisy='clean',independent_trajectories=500,frames_samples=5000,
         object_constructed='source means,Rpp,Rdp,W,Ksource' if env.startswith('Source') else 'Rpp,Rdp,targetW,DeltaW,DeltaK,weighted rank4 compression',
         used_final_method=True,hyperparameter_selection=False,formal_only=False,MFA_Source_allowed=env.startswith('Source'),
         MFA_Prepared_allowed=True,evidence=path,sha256=c.sha(c.OUT/path)))
    manifests=[]
    for name in ('t15a_cdl_manifest.csv','t15c_cdl_manifest.csv','t16c_heldout_manifest.csv'):
        with (c.OUT/name).open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
        envs=sorted({r.get('shift','Source/CDL-C-30') for r in rows})
        for env in envs:
            for split in sorted({r['split'] for r in rows if r.get('shift','Source/CDL-C-30')==env}):
                group=[r for r in rows if r.get('shift','Source/CDL-C-30')==env and r['split']==split]
                manifests.append(dict(manifest=name,environment=env,split=split,
                  trajectories=len({r['trajectory_id'] for r in group}),frames=len(group),sha256=c.sha(c.OUT/name)))
    for r in manifests:
        if r['environment'] in ('CDL-C-75','CDL-B-120') or r['split']=='validation_dev':
            ledger.append(dict(environment=r['environment'],clean_noisy='clean and paired noisy evaluation',
              independent_trajectories=r['trajectories'],frames_samples=r['frames'],
              object_constructed='held-out diagnostic covariance/oracle only' if r['split']=='train_dev' else 'development evaluation; method/rank evidence, no prepared construction',
              used_final_method=False,hyperparameter_selection=r['split']=='validation_dev' and r['environment'] in ('CDL-C-75','CDL-B-120'),
              formal_only=False,MFA_Source_allowed=False,MFA_Prepared_allowed=False,evidence=r['manifest']+':'+r['split'],sha256=r['sha256']))
    ledger.append(dict(environment='C30 separate from Source',clean_noisy='N/A',independent_trajectories=0,frames_samples=0,
      object_constructed='C30 is Source alias, not an additional corpus',used_final_method=False,hyperparameter_selection=False,
      formal_only=False,MFA_Source_allowed=False,MFA_Prepared_allowed=False,evidence='src/cdl_benchmark.py SYSTEM speed30/CDL-C',sha256=c.sha(ROOT/'src/cdl_benchmark.py')))
    ledger.append(dict(environment='T20B formal / held-out delay',clean_noisy='sealed',independent_trajectories=None,frames_samples=None,
      object_constructed='reporting-only; not opened or counted',used_final_method=False,hyperparameter_selection=False,
      formal_only=True,MFA_Source_allowed=False,MFA_Prepared_allowed=False,evidence='SP-U3A firewall',sha256=None))
    pre=c.load(c.OUT/'t16e_pilot_cv_preflight.json');proof=[]
    with np.load(c.OUT/'t16e_frozen_operators.npz',allow_pickle=False) as a:
        assert c.sha(c.OUT/'t16e_frozen_operators.npz')==pre['operator_artifact']['sha256']
        assert array_hash(statistics['Source/CDL-C-30'].mean_p)==array_hash(a['source_mean_p'])
        assert array_hash(statistics['Source/CDL-C-30'].mean_d)==array_hash(a['source_mean_d'])
        for row in pre['cross_pilot_operator_provenance']:
            s=row['snr_db'];sigma=row['sigma2'];source=statistics['Source/CDL-C-30'].prepare(sigma).weight
            assert array_hash(source)==array_hash(a[f'Wsource_{s}'])
            for env,label in [('CDL-B-30','P'),('CDL-C-120','M')]:
                delta=statistics[env].prepare(sigma).weight-source
                assert array_hash(delta)==array_hash(a[f'DeltaW{label}_{s}'])
            assert row['source_rpp_sha256']==array_hash(statistics['Source/CDL-C-30'].rpp)
            assert row['profile_rpp_sha256']==array_hash(statistics['CDL-B-30'].rpp)
            assert row['mobility_rpp_sha256']==array_hash(statistics['CDL-C-120'].rpp)
            proof.append(dict(snr_db=s,sigma2=sigma,source_and_two_corrections_bit_exact=True))
    factor=c.load(c.OUT/'t16g_factor_preflight.json')
    assert factor['factor_artifact']['metadata']['development_conditions']=={'P':'CDL-B-30','M':'CDL-C-120'}
    assert c.sha(c.OUT/'t16g_frozen_module_factors.npz')==factor['factor_artifact']['sha256']
    c.csvsave(c.OUT/'sp_u3a_offline_information_ledger.csv',ledger)
    c.save(c.OUT/'sp_u3a_offline_information_ledger.json',dict(rows=ledger,manifest_counts=manifests,operator_reconstruction=proof,
      total_prepared_trajectories=1500,total_prepared_frames=15000,source_trajectories=500,source_frames=5000,
      clean_full_corpus_persisted=False,raw_corpus_recovery='future replay exact frozen train-only manifests; this task only2source trajectories',
      trajectories_independent_frames_correlated=True,formal_counts_not_read=True,access=c.ACCESS))
    fairness=[dict(variant='MFA-Source',environments='Source/CDL-C-30',trajectories=500,frames=5000,labels='none in pooled EM',
      offline_interface='full clean vectors -> learned component means/W/Psi/weights',online='current64Pilot-LS,A,Cn,trained parameters',
      temporal_history='one frame',noise='frozen source sigma2',target_supervision=False,role='source-only reference; not equal-prepared-budget superiority comparator'),
      dict(variant='MFA-Prepared',environments='Source/CDL-C-30,CDL-B-30,CDL-C-120',trajectories=1500,frames=15000,
      labels='used for balanced corpus/split only; ordinary pooled KMeans/EM sees no labels',
      offline_interface='full clean vectors -> learned component means/W/Psi/weights',online='current64Pilot-LS,A,Cn,trained parameters',
      temporal_history='one frame',noise='frozen source sigma2',target_supervision=False,role='primary matched-environment/count candidate; information interface differs'),
      dict(variant='Frozen final method',environments='Source/CDL-C-30,CDL-B-30,CDL-C-120',trajectories=1500,frames=15000,
      labels='explicit Source/P/M assignment during offline construction',offline_interface='source means/Rpp/Rdp; environment-labelled corrections; rank4 factors',
      online='64Pilot-LS/frame,known noise/operators',temporal_history='five causal frames for coordinates;current for output',
      noise='frozen source sigma2',target_supervision=False,role='unchanged; historical development decisions disclosed separately'),
      dict(variant='MFA-Oracle-Target',environments='B120',trajectories=None,frames=None,labels='target privileged',
      offline_interface='target clean corpus',online='not executed',temporal_history='not executed',noise='not executed',
      target_supervision=True,role='information-advantaged reference only; forbidden inthis task')]
    c.csvsave(c.OUT/'sp_u3a_mfa_fairness_matrix.csv',fairness);c.save(c.OUT/'sp_u3a_mfa_fairness_matrix.json',dict(rows=fairness,
       exact_information_equivalence=False,risks=['MFA raw full-vector distribution vs method Rpp/Rdp statistical interface',
        '4096 raw vectors include64 muted-channel coefficients that final method statistics do not model',
        'environment-labelled corrections vs unlabelled pooled EM','five-frame coordinate history vs current-only MFA',
        'method historical development selection vs new prepared-internal MFA selection',
        'future full training resource/convergence not established by tiny smoke']))
    protocol=dict(task='SP-U3A',training_object='C-order full4096 [rx,tx,k,t]',source_names=list(paths),
      allowed_variants=['MFA-Source','MFA-Prepared'],forbidden=['B120 clean training','C75 clean training','heldout delay','formal CSI','future upgrade validation'],
      tuning_split='within each allowed500trajectory environment:fit IDs0..399;validation400..499;split before frame extraction;all10frames grouped',
      final_refit='after selection,refit all500/environment; three independent init seeds, no target access',
      K=[1,2,4],latent_rank=[4,8,16],ppca=False,lock_psis=False,zero_mean=False,rs_clip=0.,
      max_condition_number=1e6,max_iter=200,tol=1e-4,seeds=[202603101,202603102,202603103],
      initialization='author KMeans real-imag n_init1 +author random complexW; no environment-stratified component seeding',
      selection_metric='mean clean held-out negative log likelihood per4096complex dimension;trajectory mean then equal environment mean;average3initializations',
      tie_break='score within1e-8:lower K thenlower rank; no target performance tie-break',
      no_config_substitution_on_failure=True,failures='report each seed/config failure; convergence warning after200=>ineligible;no target-assisted debugging',
      future_run_caps=dict(estimated_peak_GiB=8,max_seconds_per_fit=3600,action='stop and report resource gate;do not change dimensionality or expand search'),
      estimate_summands=1.0,upstream_source_unmodified=True,native_dense_implementation=True,
      alternative_isotropic='dissertation matched ppca=True is separately named future authorization option, not selected this round',
      future_metrics=dict(primary='data-domain NMSE,same3968mask andtrajectory aggregation',secondary='BER withsamefrozenZF detector;arithmetic/storage',
        compared_score_frames=list(range(4,10)),snrs=[10,20],sigma2={str(a['snr_db']):a['sigma2'] for a in proof}),
      toy=dict(seed=202603201,N=8,M=3,K=2,rank=2,train_samples=256,max_iter=30,tolerance=1e-10),
      smoke=dict(source_train_ids=[0,1],frames=[0,4,9],K=1,rank=2,max_iter=1,observations=2,seed=202603301,
       max_seconds=300,scope='structural gate only;expected nonconvergence warning;not scientific trained baseline'),
      theory='FREEZE',next_task_automatic=False,dependencies_installed=False)
    c.save(ROOT/'configs/sp_u3a_protocol.json',protocol)
    c.save(c.OUT/'sp_u3a_mfa_hyperparameter_protocol.json',protocol)
    inventory=dict(prior='h|k~CN(mu_k,C_k),C_k=W_kW_k^H+diag(Psi_k)',
      observation='y=A h+n,n~CN(0,Cn);Cn=sigma2I64;E|n_l|²=sigma2,real/imag variance sigma2/2',
      posterior='gamma_k(y)=pi_k CN(y;A mu_k,A C_k A^H+Cn)/sum_i pi_i CN(y;A mu_i,A C_i A^H+Cn)',
      component='mu_k+C_k A^H(A C_k A^H+Cn)^-1(y-A mu_k)',mixture='sum_k gamma_k(y)hhat_k(y)',
      density='exp(-d^H S^-1 d)/(pi^M detS); no real-Gaussian1/2 factor',
      mapping=dict(paper='eq6,10-12;A=I',dissertation='eq2.35,3.6,3.9;generalA',
        software='_update_covariances,_prepare_observation_model,_predict_observation_proba,_lmmse,estimate'),
      software_default_truncation=True,full_mixture_setting='float1.0',selector_shape=[64,4096],data_selector_shape=[3968,4096],
      full_order='index=(((rx*2+tx)*64+k)*16+t)',joint_pilot_data_dimension=4032,ignored_muted_channel_coefficients=64,
      training_vector_choice='full4096;retain full original model,do not pad pilot/data vector into4096',
      asymptotic_theorem='No rectangular-A asymptotic-optimality claim; dissertation theorem1 assumes invertibleA')
    c.save(c.OUT/'sp_u3a_mfa_equation_inventory.json',inventory)
    novelty=dict(tier='Tier1 closest work',Existing='offline prior learning,received-observation posterior responsibilities,component conditionalLMMSE,low-rank-plus-diagonal covariances',
      Missing='verified faithful performance comparison under our joint sparse MIMO-OFDM andprepared budget',
      Proposed='frozen cross-pilot real affine coordinates overprepared correction operators;fivecausalframes;rank4 correction-only output',
      Difference='unrestricted real correction coordinates versus posterior simplex;labelled statistical corrections versus learned distribution;differenttemporal interface',
      Why_it_matters='defines comparator andinformation disclosure;no novelty orsuperiority proof',
      Residual_collision_risk='GMM/MFA observation-dependent mixing is prior art;adaptive-lowrank Ozdemir ABSTRACT_ONLY/OPEN RISK remains',
      prohibited_novelty='observation-dependent LMMSE mixing itself',paper_changed=False)
    c.save(c.OUT/'sp_u3a_novelty_ledger.json',novelty)
    c.save(c.OUT/'sp_u3a_preparation_gate.json',dict(passed=True,protocol_sha256=c.sha(ROOT/'configs/sp_u3a_protocol.json'),
      old_input_sha256={p:c.sha(c.OUT/p) for p in sorted(c.ALLOWED)},protected_count=len(c.verify()['protected_sources']),
      before_toy=True,before_CDL_smoke=True,formal_read=False,old_U2_outcomes_read=False,access=c.ACCESS))
    print('PREPARED500Source+500B30+500C120;15000correlatedframes;operatorsbitexact;fairness/tuning frozenbeforetoy')

if __name__=='__main__':main()
