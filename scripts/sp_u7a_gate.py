"""SP-U7A metadata-only qualification/preregistration. Never generates a channel.

Steps: qualify (requires completed E0 exposure audit), seal (requires pytest log).
U7B execution is intentionally absent. Fixed seeds do not authorize execution.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.sp_u7a_metadata import OUT,sha,write_json
from scripts.sp_u7a_statistics import METHODS

NAMESPACE='SP_U7B_OOD_CDLA120_V1'
ROLES=('sionna_seed','torch_seed','numpy_seed','bits_seed_tx0','bits_seed_tx1','noise_seed')
INTEGRITY_GATES=('exact_condition','trajectory_count','seed_uniqueness','no_historical_overlap','method_artifact_hashes',
 'frozen_code_hashes','no_target_clean_CSI_in_deployables','matched_reference_isolated','all_trajectories_complete',
 'causal_windows','finite_outputs','no_rank_or_fallback_drift','raw_file_completeness')
ARTIFACTS=('outputs/t15a_cdl_statistics_500.npz','outputs/t15c_cdl_statistics_CDL-B-30.npz',
 'outputs/t15c_cdl_statistics_CDL-C-120.npz','outputs/t16e_frozen_operators.npz',
 'outputs/t16f_frozen_candidates.npz','outputs/t16g_frozen_module_factors.npz')
ORCHESTRATION=('scripts/run_pilot_cv.py','scripts/run_lowrank_modules.py','scripts/run_statistics_selection.py',
 'scripts/run_frozen_generalization.py','scripts/run_formal_test.py','scripts/run_cdl_mismatch.py',
 'scripts/sp_u7a_metadata.py','scripts/sp_u7a_statistics.py','scripts/sp_u7a_gate.py')

def read_owned(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def validate_unseal(checks):
    """Future executor must produce evidence for ALL gates; missing is failure."""
    if set(checks)!=set(INTEGRITY_GATES) or any(checks[k] is not True for k in INTEGRITY_GATES):raise ValueError('STOP_SEALED_INTEGRITY_FAILURE')
    return True

def seed_keys():
    keys=[]
    # Stable ordered keys; channel-role seeds are allocated once per trajectory.
    for group,n in (('EVAL',1000),('MATCHED_REF',500)):
        for i in range(n):
            for role in ROLES[:3]:keys.append((group,i,-1,role))
            if group=='EVAL':
                for j in range(10):
                    for role in ROLES[3:]:keys.append((group,i,j,role))
    keys.append(('BOOTSTRAP',-1,-1,'bootstrap_seed'))
    return keys

def allocate_seeds(forbidden):
    used=set(forbidden);rows=[];rejections=[]
    for group,i,j,role in seed_keys():
        counter=0
        while True:
            message=f'{NAMESPACE}|{group}|{i}|{j}|{role}|{counter}'
            seed=int.from_bytes(hashlib.sha256(message.encode('ascii')).digest()[:4],'big')
            if seed not in used:break
            rejections.append({'group':group,'trajectory_id':i,'frame_id':j,'role':role,'counter':counter,'rejected_seed':seed})
            counter+=1
        used.add(seed)
        rows.append({'namespace':NAMESPACE,'group':group,'trajectory_id':i,'frame_id':j,'role':role,'counter':counter,'seed':seed})
    return rows,rejections

def model_qualification():
    """Constructor metadata only, coefficient-call protected by hard sentinel."""
    import torch
    from sionna.phy.channel.tr38901 import CDL,PanelArray,models
    from sionna.phy.channel import cir_to_ofdm_channel
    if importlib.metadata.version('sionna-no-rt')!='2.2.0':raise ValueError('pinned Sionna version')
    from src.cdl_benchmark import SYSTEM,ARRAY,SYMBOL_SECONDS
    if SYSTEM['sample_rate_hz']!=SYSTEM['nfft']*SYSTEM['delta_f_hz']:raise ValueError('invalid sampling config; no silent fix')
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    call=CDL.__call__;calls=[]
    def forbidden_call(*args,**kwargs):
        calls.append(True);raise RuntimeError('U7A_CHANNEL_GENERATION_FORBIDDEN')
    CDL.__call__=forbidden_call
    try:
        array=PanelArray(num_rows_per_panel=1,num_cols_per_panel=2,polarization='single',polarization_type='V',
          antenna_pattern='38.901',carrier_frequency=3.5e9,element_horizontal_spacing=.5,element_vertical_spacing=.5,precision='double',device=device)
        model=CDL(model='A',delay_spread=100e-9,carrier_frequency=3.5e9,ut_array=array,bs_array=array,direction='downlink',
          ut_orientation=ARRAY['ut_orientation_radians'],bs_orientation=ARRAY['bs_orientation_radians'],
          ut_velocity=[120/3.6,0.,0.],min_speed=120/3.6,max_speed=120/3.6,
          normalize_delays=True,precision='double',device=device,spec_version='19.2')
        params={}
        for profile in ('A','B','C'):
            p=Path(models.parameter_file(f'CDL-{profile}.json','19.2'));spec=read_owned(p)
            params[profile]={'los':bool(spec['los']),'clusters':spec['num_clusters'],'model_file':str(p),'sha256':sha(p)}
        if model._has_los or model._num_clusters!=23 or float(model._k_factor)!=0:raise ValueError('CDL-A NLOS semantics')
        p=read_owned(params['A']['model_file'])
        powers=np.power(10.,np.array(p['powers'])/10);powers/=powers.sum()
        if not np.allclose(model._powers.detach().cpu().numpy(),powers,rtol=1e-14,atol=0):raise ValueError('PDP normalization drift')
        if not np.allclose(model._delays.detach().cpu().numpy(),np.array(p['delays'])*100e-9,rtol=1e-14,atol=0):raise ValueError('delay scaling drift')
        source_paths=[inspect.getfile(CDL),inspect.getfile(PanelArray),inspect.getfile(type(model._cir_gen)),inspect.getfile(cir_to_ofdm_channel)]
        return {'verdict':'PASS','version':'2.2.0','distribution':'sionna-no-rt','spec_version':'19.2',
          'signature':str(inspect.signature(CDL)), 'profiles':params,'device':device,
          'definition':{'profile':'A','speed_kmh':120,'nominal_delay_spread_s':100e-9,'carrier_hz':3.5e9,'direction':'downlink',
          'ut_velocity_mps':[120/3.6,0.,0.],'nt':2,'nr':2,'nfft':64,'symbols_per_frame':16,'cp':16,'delta_f_hz':30000,'sample_rate_hz':1920000,
          'dF':16,'dT':4,'modulation':'QPSK','array':dict(ARRAY,ut_velocity_mps=[120/3.6,0.,0.]),'ut_velocity_override':True},
          'normalization':{'PDP':'10^(powers_dB/10), divide by sum; total cluster power = 1',
           'delays':'stored normalized delays multiplied by 100e-9; first CDL-A delay already zero; no empirical H normalization',
           'ofdm_channel_normalize':False,'target_receive_power_renormalization':False,'k_factor':0.,'los':False},
          'sampling':{'time_states_per_trajectory':160,'one_call_per_trajectory':True,'symbol_seconds':SYMBOL_SECONDS,'state_sampling_hz':1/SYMBOL_SECONDS,
            'export_shape':[2,2,64,160],'frame_shape':[10,2,2,64,16],'export_order':'[rx,tx,k,t] C order; permute(0,1,3,2) then reshape frames',
            'ideal_sync':True,'symbol_quasi_static':True,'explicit_ICI':False},
          'precision':'torch.float64/complex128; numpy.complex128','source_hashes':{p:sha(p) for p in source_paths},
          'source_vs_docs_note':'API description says normalize_delays makes first delay zero. Actual pinned _load_parameters only multiplies stored delays by delay_spread; CDL-A table has first delay zero already. Source controls exact semantics.',
          'channel_coefficient_calls':len(calls),'OFDM_channel_calls':0,'scientific_predictions':0,
          'constructor_only_not_channel_smoke':True,
          'official_docs':'https://nvlabs.github.io/sionna/phy/api/channel/wireless/sionna.phy.channel.tr38901.CDL.html'}
    finally:CDL.__call__=call

def artifact_audit():
    result={};a={}
    for p in ARTIFACTS:
        committed=subprocess.check_output(['git','show',f'HEAD:{p}'],cwd=ROOT)
        committed_sha=hashlib.sha256(committed).hexdigest()
        if sha(ROOT/p)!=committed_sha:raise ValueError('frozen artifact differs from committed HEAD: '+p)
        result[p]={'sha256':committed_sha,'matches_committed_HEAD':True,'arrays':{}}
        with np.load(ROOT/p,allow_pickle=False) as f:
            for key in f.files:
                # Explicit frozen method/preparation artifacts, never outcome reports.
                if key=='metadata_json':continue
                v=f[key]
                result[p]['arrays'][key]={'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()}
                if p in ARTIFACTS[3:]:a[key]=v.copy()
            if p in ARTIFACTS[:3]:
                if list(f['train_trajectory_ids'])!=list(range(500)) or int(f['samples'])!=5000:raise ValueError('historical train preparation budget drift')
    prefixes={}
    for m in ('P','M'):
        for s in (10,20):
            aa=a[f'A_{m}_{s}'][:,:4];qq=a[f'Q_{m}_{s}'][:4,:]
            if aa.shape!=(3968,4) or qq.shape!=(4,64) or not np.isfinite(aa).all() or not np.isfinite(qq).all():raise ValueError('frozen rank4 prefix')
            for label,v in (('A',aa),('Q',qq)):
                prefixes[f'{m}_{s}_{label}']={'shape':list(v.shape),'sha256':hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()}
    for s in (10,20):
        for k in ('KSource','DeltaKP','DeltaKM'):
            if a[f'{k}_{s}'].shape!=(4,16,48):raise ValueError('cross-pilot kernel shape')
        for k in ('Wsource','DeltaWP','DeltaWM'):
            if a[f'{k}_{s}'].shape!=(3968,64):raise ValueError('frozen full operator shape')
    from src.pilot_cv import frozen_folds
    from src.cdl_benchmark import observation_geometry
    folds,definition=frozen_folds(observation_geometry().selected)
    if definition['sha256']!='16594a306157e0b6ac114100ddebbba262f8e4205a98c8f48726e5519e55ab18':raise ValueError('fold hash drift')
    return {'verdict':'PASS','artifacts':result,'rank4_prefixes':prefixes,'fold':definition,'source_centering':True,
      'dimensions':{'pilot':64,'data':3968,'folds':4,'heldout_per_fold':16,'observed_per_fold':48,'real_design_per_five_frame_window':[640,2]},
      'read_policy':'only six frozen method/statistics NPZ artifacts; no historical numerical outcomes; no artifact rewriting/SVD/refit'}

def protocol(bootstrap_seed):
    return {'task':'SP-U7A','future_task':'SP-U7B','namespace':NAMESPACE,'status':'PREREGISTERED_SEALED_NOT_AUTHORIZED_TO_RUN',
      'condition':{'profile':'CDL-A','speed_kmh':120,'nominal_DS_ns':100,'nt':2,'nr':2,'K':64,'T':16,'cp':16,'spacing_hz':30000,
        'carrier_hz':3500000000,'sample_rate_hz':1920000,'QPSK':True,'dF':16,'dT':4,'ideal_sync':True,'symbol_quasi_static':True,'explicit_ICI':False},
      'evaluation':{'independent_trajectories':1000,'ids':list(range(1000)),'continuous_frames':10,'scored_frames':list(range(4,10)),
        'snrs':[10,20],'methods':list(METHODS),'methods_human':['LS + frozen frequency/time interpolation','Frozen Source LMMSE','Hard candidate selection',
         'Convex combination','Dense affine adaptation','Final LR4 affine adaptation','Matched-statistics LMMSE reference'],
        'shared_across_methods_and_snrs':['H','QPSK bits/symbols','unit complex Gaussian noise atoms'],
        'raw_generation_order':'evaluation trajectory ascending; save complete H and 10 communication frames before estimators; evaluate both SNRs and all methods without outcome logging',
        'stopping':'no first-100 development, no early outcome view; interrupted run keeps all partial artifacts and no aggregate output; exact resume only with original seeds/hashes; no trajectory dropping'},
      'noise':{'P_ref':9.941134822523972,'sigma2':{'10':.9941134822523973,'20':.09941134822523973},'target_renormalization':False,
        'definition':'Y=H X+sqrt(sigma2) Z; existing communication draws unit complex noise, same Z at both SNRs',
        'realized_snr':'descriptive only; scored useful received signal energy / (sigma2 times receive sample count), plus realized atom energy audit; never used to rescale or stratify'},
      'deployable':{'objects':list(ARTIFACTS),'rank':4,'means':'frozen Source C30 means only','P_origin':'B30 versus Source C30','M_origin':'C120 versus Source C30',
        'window':5,'folds':4,'alpha_beta':'unrestricted real np.linalg.solve normal equation; no clipping/ridge/pinv/fallback',
        'fold_assignment':'(physical_k//16 + symbol_t//4) % 4, existing selected-pilot order',
        'fit_equation':'for t=j-4..j and folds f: x_o=y_o-mu_Source,o; r=y_h-mu_Source,h-KSource_f x_o; uP=DeltaKP_f x_o; uM=DeltaKM_f x_o; stack Re/Im to G (640x2), z (640); c=solve(G.T G,G.T z), retaining original frame/fold reduction order',
        'current_prediction':'mu_Source,d + WSource_s x_j + alpha_j BP_s x_j + beta_j BM_s x_j; LR4 uses A_P[:,0:4] Q_P[0:4,:], A_M[:,0:4] Q_M[0:4,:]; Dense uses full DeltaW with exactly same c_j',
        'information_budget':'past/current pilots only j-4..j; no future frames, true condition labels or target clean CSI; statistics/correction/factors/cross-pilot kernels immutable',
        'Hard':'same CV objective, candidates W0/WP/WM, frozen tie order W0 then WP then WM',
        'Convex':'same CV objective, frozen exact simplex active-set algorithm; existing Convex-specific feasibility restriction unchanged',
        'LS_and_ZF':'src.link.estimate_ls_linear_channel and existing detect_data_zf; frequency then time linear interpolation; scale sqrt(2) before frozen QPSK demodulation',
        'Dense_comparison':'compression only, not separate fitting; no inherited 0.5 dB margin'},
      'matched_reference':{'subnamespace':'MATCHED_REF','ids':list(range(500)),'independent_trajectories':500,'frames_each':10,'clean_frames':5000,
        'procedure':'src.cdl_benchmark.StreamingStatistics centered batch/Welford with population 1/5000 Rpp/Rdp; source-order [rx,tx,k,t]; no full-covariance extra budget; freeze 64/3968 means and Rpp/Rdp, prepare via existing Cholesky solve at each frozen sigma2',
        'input':'independent A120 clean CSI, reference-only; no communication/noise statistics; every frame 0..9 equally included',
        'estimator':'mu_A,d + R_A,dp (R_A,pp+sigma2 I)^(-1)(y-mu_A,p)',
        'isolation':'separate process/object/path and seeds; no means/covariance/oracle coordinates passed into deployables; statistical identity is group/id: MATCHED_REF/0..499 versus EVAL/0..999, never joined on bare integer IDs; hash and seal reference before evaluation outcome access',
        'reference_not_bayes_oracle':'finite-budget empirical matched LMMSE reference, not channel-dependent oracle or exact MMSE'},
      'statistics':{'primary_unit':'trajectory; six scored frames never treated as independent replicates','trajectory_metric':'n_i,m,s=E_i,m,s/Q_i,s; trajectory dB=10 log10(n_i,m,s)',
        'energies':'E_i=sum_j4..9 sum_data |hhat-h|^2; Q_i=sum_j4..9 sum_data |h|^2; data mask same all methods; bit counts pooled over same frames',
        'population_estimand':'NMSE_m,s=10 log10(sum_i E_i,m,s / sum_i Q_i,s); pooled energy ratio exactly as existing Paper 1 helper, NOT mean trajectory dB; trajectory distribution separately descriptive',
        'primary':'G_s=NMSE_Source,s-NMSE_LR4,s; positive favors LR4; only two primary cells',
        'bootstrap':{'replicates':10000,'seed':bootstrap_seed,'bit_generator':'numpy.random.PCG64','ids_order':'ascending 0..999',
          'draws':'one rng.integers(0,1000,size=(10000,1000)); same complete table for 7 methods and 2 SNRs; chunks100 for sums',
          'reduction':'within each draw sum sampled E and Q then ratio then dB; compute paired differences after both reductions',
          'family_alpha':.05,'family_size':2,'per_cell_one_sided_alpha':.025,'primary_LCB_quantile':.025,'quantile_method':'linear',
          'success':'strict G_s>0 and LCB_s>0 at both SNRs','descriptive_CI_quantiles':[.025,.975],
          'code_semantics_reused':'src/formal_evaluation.py pooled/bootstrap source only; old .005/.995 endpoints and 0.5 margin not imported'},
        'secondary':{'LR4_minus_Dense':'positive = compression penalty','LR4_minus_Matched':'positive = gap to finite-budget matched reference',
          'Convex_minus_LR4':'positive = improvement over Convex','BER':'pooled bit errors / pooled bits; paired LR4-Source difference, positive LR4 worse',
          'inference':'paired two-sided percentile .025/.975, descriptive only; no superiority/noninferiority margin or additional primary claims'},
        'diagnostics':['realized useful receive power/SNR','coordinate distributions','Gram eigenvalues/rank/condition numbers'],
        'zero_or_nonfinite':'no epsilon floor or missing trajectory deletion; any nonfinite NMSE/prediction/draw or invalid denominator -> sealed integrity failure; all finite-coefficient extremes retained'},
      'decisions':{'precedence':['A','C','B'], 'A':'both gains>0 and both adjusted LCB>0: limited unseen-profile transfer at tested A120/100 setting',
        'C':'if not A and any descriptive upper .975 <=0: evidence of stable nonpositive transfer at that SNR; record generalization boundary and any cross-SNR heterogeneity',
        'B':'otherwise: mixed/SNR-dependent or inconclusive transfer; crossing zero is not proof of equivalence or stable absence; no near-zero dB threshold invented',
        'H1':'positive gain hypothesis, supported only by A','H2':'positive Source gain alone never implies near-oracle; report actual matched gap CI without closeness threshold',
        'H3':'report LR4-Dense gap/CI, cannot establish expansion relative to old B120 without reusing prohibited old outcomes; no new margin',
        'H4':'negative/null evidence retained; no rescue, new correction/rank/history/clip/ridge/profile reselection'},
      'pre_unseal':{'gates':list(INTEGRITY_GATES),'phase_order':'freeze/hash/seed gate before first H; generate all complete raw/metric artifacts with no scientific console/aggregate outputs; then structural/finite integrity; only after all 13 true read/aggregate scientific outcomes',
        'evidence_required':{'exact_condition':'frozen config + pinned model metadata hashes','trajectory_count':'exact IDs 0..999; 1000 evaluation, 500 reference',
          'seed_uniqueness':'unique individual role values; exact manifest counters; both SNRs intentional shared inputs','no_historical_overlap':'frozen forbidden union plus fresh metadata-only delta audit and H hashes vs historical manifests; no replacing selected seeds',
          'method_artifact_hashes':'all six NPZ byte hashes and rank4 prefix hashes','frozen_code_hashes':'core/SDK/source hashes; future wrapper AST/source reviewed and hash-frozen before any A scientific H; plumbing only',
          'no_target_clean_CSI_in_deployables':'signature/dataflow and runtime boundary, truth only evaluator; never target labels','matched_reference_isolated':'separate object/process/output; no ref ID or covariance passed to deployable',
          'all_trajectories_complete':'exact 10-frame H/comm raw plus frames4..9 rows per SNR/method','causal_windows':'for every j in4..9 exactly j-4..j; no future',
          'finite_outputs':'predictions/coefficient/Gram/energies valid; any solver failure STOP; no coefficient magnitude exclusion',
          'no_rank_or_fallback_drift':'rank4 + original real solve, no new fallback, exact c equality Dense/LR4',
          'raw_file_completeness':'1000 H + 1000 comm archives, shape/dtype/SHA256, 14000 trajectory metric rows, 84000 frame metric rows, 20000 input rows for frames0..9 and both SNRs, 12000 coordinate/window rows for frames4..9 and both SNRs; 500 reference H archives + one ref statistics artifact; no duplicate/missing IDs'},
        'failure':'STOP without aggregate NMSE/BER access; preserve failed artifacts; never patch method or replace candidate in this task'},
      'figures_tables':{'max_figures':2,'Figure_A':'trajectory NMSE dB distributions + paired trajectory Source-LR4 gains at both SNRs, all 1000 points/ECDF no trimmed tails',
        'Figure_B':'pooled BER and descriptive CI for all seven methods at both SNRs','Table_A':'pooled NMSE, descriptive CI, primary gain and adjusted LCB',
        'Table_B':'Dense/Matched/Convex paired gaps and descriptive CI + BER','rule':'fixed method/SNR order, full data range or explicitly labeled log axes, no selective panels or extra explanatory figures'},
      'firewall':{'forbidden_outcomes':['T20B','U2','U4','U5'],'allowed_reads':['frozen method artifacts','system/method source','identity/schema/run-use metadata','seed and historical H-hash manifests'],
        'no_outcome_changes':['candidate','sample size','methods','metrics','thresholds','seeds','figure plan'],
        'all_prior_branches':'Theory/offline/pilot/final FROZEN; MFA/A-MMSE STOPPED; no manuscript or Git commit'},
      'limitations':['fixed-sample qualification is not achieved power evidence','single NLOS profile + one mobility/delay setting, ideal sync and no explicit ICI',
        'finite-budget matched reference uncertainty not estimated by evaluation-only trajectory bootstrap; gaps conditional on this fixed reference']}

def qualify():
    if (OUT/'sp_u7a_delivery_manifest.json').exists():raise RuntimeError('already sealed; no overwrite')
    exposure=read_owned(OUT/'sp_u7a_exposure_audit.json')
    if exposure['verdict']!='PASS' or exposure['candidate_exposure']!='E0':raise RuntimeError('STOP_EXPOSURE_GATE')
    model=model_qualification();write_json(OUT/'sp_u7a_model_audit.json',model)
    artifacts=artifact_audit();write_json(OUT/'sp_u7a_artifact_audit.json',artifacts)
    history=read_owned(OUT/'sp_u7a_historical_seed_union.json');forbidden=set(history['forbidden_seeds'])
    if NAMESPACE in history['historical_namespaces']:raise RuntimeError('namespace reused')
    rows,rejected=allocate_seeds(forbidden)
    values=[r['seed'] for r in rows]
    if len(rows)!=34501 or len(set(values))!=34501 or set(values)&forbidden:raise RuntimeError('seed collision')
    path=OUT/'sp_u7a_future_seed_manifest.csv'
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    seed_report={'verdict':'PASS','namespace':NAMESPACE,'mapping':'SHA256 ASCII namespace|group|trajectory_id|frame_id|role|counter; first 4 bytes unsigned big-endian uint32',
      'key_order':'EVAL ids0..999 channel roles then each frame0..9 communication roles; MATCHED_REF ids0..499 channel roles; BOOTSTRAP last',
      'counter_policy':'start 0; reject individual seed if in frozen historical forbidden union or any earlier assigned key; increment counter only for collision; no performance-based search',
      'forbidden_union_count':len(forbidden),'forbidden_union_sha256':sha(OUT/'sp_u7a_historical_seed_union.json'),
      'historical_namespace_count':len(history['historical_namespaces']),'allocation_counts':{'EVAL_channel':3000,'EVAL_communication':30000,'MATCHED_REF_channel':1500,'bootstrap':1},
      'total_unique_values':34501,'historical_overlap_count':0,'collision_rejections':rejected,'bootstrap_seed':rows[-1]['seed'],
      'manifest_sha256':sha(path),'generated_channels':0,'paired_SNR_reuse_intentional':True,
      'continuation_rule':'manifest immutable; future metadata delta collisions -> STOP rather than silent reseeding; reference and evaluation never share role values'}
    write_json(OUT/'sp_u7a_seed_audit.json',seed_report)
    spec=protocol(rows[-1]['seed']);write_json(ROOT/'configs/sp_u7a_protocol.json',spec)
    source_paths=sorted({p.relative_to(ROOT).as_posix() for p in (ROOT/'src').glob('*.py')}|set(ORCHESTRATION))
    protected={p:sha(ROOT/p) for p in source_paths+list(ARTIFACTS)}
    protected.update(model['source_hashes'])
    protected.update({v['model_file']:v['sha256'] for v in model['profiles'].values()})
    for p in (ROOT/'configs/sp_u7a_protocol.json',OUT/'sp_u7a_exposure_audit.json',OUT/'sp_u7a_historical_seed_union.json',
      OUT/'sp_u7a_future_seed_manifest.csv',OUT/'sp_u7a_seed_audit.json',OUT/'sp_u7a_model_audit.json',OUT/'sp_u7a_artifact_audit.json'):
        protected[p.relative_to(ROOT).as_posix()]=sha(p)
    write_json(OUT/'sp_u7a_protected_inputs.json',{'hashes':protected,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'old_outcome_reads':0,'generator_coefficient_calls':0,'scientific_prediction_calls':0,'future_wrapper_status':'not implemented/run; must freeze plumbing-only wrapper under this protocol before first U7B H'})
    print('QUALIFICATION_AND_PREREGISTRATION_SAVED; no U7B channels, predictions or performance generated',flush=True)

def seal():
    if (OUT/'sp_u7a_delivery_manifest.json').exists():raise RuntimeError('already sealed; no overwrite')
    log=OUT/'sp_u7a_pytest_execution.txt'
    text=log.read_text(encoding='utf-8-sig')
    if ' passed' not in text or any(x in text for x in (' failed','ERRORS',' errors',' skipped')):raise ValueError('successful actual scoped pytest required')
    protected=read_owned(OUT/'sp_u7a_protected_inputs.json')
    drift=[p for p,h in protected['hashes'].items() if sha(Path(p) if Path(p).is_absolute() else ROOT/p)!=h]
    if drift:raise ValueError('protected input drift: '+str(drift))
    subprocess.check_call(['git','diff','--check'],cwd=ROOT)
    if subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip():raise ValueError('tracked modification')
    status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
    (OUT/'sp_u7a_git_status.txt').write_text(status,encoding='utf-8')
    owned=[ROOT/'configs/sp_u7a_protocol.json',ROOT/'scripts/sp_u7a_metadata.py',ROOT/'scripts/sp_u7a_statistics.py',ROOT/'scripts/sp_u7a_gate.py',ROOT/'tests/test_sp_u7a_gate.py']
    owned+=sorted(p for p in OUT.glob('sp_u7a*') if p.is_file() and p.name not in {'sp_u7a_delivery_manifest.json','sp_u7a_seal_execution.txt'})
    owned+=sorted(p for p in (OUT/'sp_u7a_audit_failure01').rglob('*') if p.is_file())
    record={'task':'SP-U7A','verdict':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),'git_head':protected['git_head'],
      'gate_completeness':dict.fromkeys(('Exposure gate','Simulator gate','Protocol freeze','Statistics freeze','Seed freeze','Firewall'),'PASS'),
      'engineering_status':'metadata/source/API/frozen artifact/seed/statistics/firewall qualification complete',
      'scientific_status':'NOT_RUN; no channel H, LR4 predictions, matched reference construction, NMSE or BER generated',
      'future_U7B_authorized':False,'generator_coefficient_calls':0,'performance_accessed_or_generated':False,'protected_inputs_unchanged':True,
      'file_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in owned},'pytest_summary':text.strip().splitlines()[-1],
      'git_tracked_diff_empty':True,'final_stop':'Await mentor review. No U7B execution, no profile replacement, no method/baseline/paper edits, no Git commit.'}
    write_json(OUT/'sp_u7a_delivery_manifest.json',record)
    print(json.dumps({k:record[k] for k in ('task','verdict','gate_completeness','scientific_status','pytest_summary','final_stop')},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('step',choices=('qualify','seal'));args=parser.parse_args()
    (qualify if args.step=='qualify' else seal)()
