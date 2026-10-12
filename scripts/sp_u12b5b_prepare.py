"""Read only existing preparation clean channels; freeze split and CRN inputs."""
from scripts.sp_u12b5b_common import *
firewall()
from scripts.sp_u12b5b_model import np,ProjectPilotInterface,array_hash

def prepare():
 fid=check_fidelity();baseline=read(OUT/'preservation_before.json');assert baseline['seals']['B2']['PASS'] and baseline['seals']['B3']['PASS']
 assert not CONFIG.exists(),'B5B_CONFIG_ALREADY_EXISTS'
 original=read(ROOT/'configs/sp_u3a_protocol.json');cachecfg=read(ROOT/'configs/sp_u4a_protocol.json');maincfg=read(ROOT/'configs/sp_u12b2a_snr_sweep.yaml')
 assert original['tuning_split']=='within each allowed500trajectory environment:fit IDs0..399;validation400..499;split before frame extraction;all10frames grouped'
 names=original['source_names'];assert names==['Source/CDL-C-30','CDL-B-30','CDL-C-120'] and cachecfg['environments']==['Source','P','M']
 assert maincfg['sample_rate_hz']==maincfg['nfft']*maincfg['delta_f_hz'] and (maincfg['nfft'],maincfg['pilot_dimension'],maincfg['data_dimension'])==(64,64,3968)
 gate=read(ROOT/'outputs/sp_u4a_clean_gate.json');corpus=ROOT/'outputs/sp_u4a_clean.npy'
 assert gate['passed'] and gate['all_H_hashes_exact'] and sha(corpus)==gate['corpus_sha256'],'PREPARATION_CORPUS_HASH_MISMATCH'
 interface=ProjectPilotInterface();x=np.load(corpus,mmap_mode='r',allow_pickle=False)
 assert x.shape==(3,500,10,4096) and x.dtype==np.complex128
 registered=fid['frozen_policy']['future_formal_seeds']['registered']
 grid=[dict(K=r['K'],rank=r['loading_rank'],seed=r['candidate_training_initialization_seed'],final_seed=r['full_retraining_initialization_seed'],name=f"K{r['K']}_r{r['loading_rank']}") for r in registered]
 assert [(r['K'],r['rank']) for r in grid]==[(K,L) for K in (1,2,4) for L in (4,8,16)]
 cfg=dict(task='SP-U12B5B',B5A_fidelity_manifest_sha256=FID_SHA,canonical_origins=names,cache_axis_names=cachecfg['environments'],origin_physical_conditions=[dict(profile='CDL-C',speed_kmh=30,delay_spread_ns=100),dict(profile='CDL-B',speed_kmh=30,delay_spread_ns=100),dict(profile='CDL-C',speed_kmh=120,delay_spread_ns=100)],trajectory_ids=dict(training=list(range(400)),validation=list(range(400,500)),full_retraining=list(range(500))),frames=dict(training=list(range(10)),validation=list(range(4,10)),full_retraining=list(range(10))),candidate_order=grid,policy=fid['frozen_policy'],mentor_amendment='MAX_ITER_REACHED is valid if parameters/inference/serialization valid; exact tie: smaller K then smaller rank',noise_variances={s:maincfg['noise_variances'][s] for s in ('10','20')},source_power_reference=maincfg['source_receive_power'],pilot_geometry=dict(Nt=2,Nr=2,K=64,T=16,dF=16,dT=4,union_pilot_fraction='1/32',joint_dimension=4032,data_dimension=3968),selection_metric='six equally weighted origin-SNR data-domain linear pooled energy ratios; macro mean before any dB conversion',tie_breaker='EXACT float equality only; smaller K then smaller loading rank',restart_count=0,extra_model_fits=0,selection_uses_resource_metrics=False,resource_contract=dict(max_process_bytes=6*2**30,min_system_available_physical_bytes=2*2**30,max_candidate_seconds=24*3600,source='Existing frozen PLAN_B B5 resource contract; monitored STOP applies to whole grid, never select surviving candidates',BLAS_threads=1,concurrency=1),target_science_authorized=False,preparation_corpus_sha256=gate['corpus_sha256'],software=dict(Python=sys.version,NumPy=np.__version__),created_UTC=utc())
 write(CONFIG,cfg)
 split=dict(canonical_origins=names,source_configuration='configs/sp_u3a_protocol.json',source_configuration_sha256=sha(ROOT/'configs/sp_u3a_protocol.json'),rule=original['tuning_split'],origin_partitions={n:dict(train_ids=list(range(400)),validation_ids=list(range(400,500))) for n in names},unit='trajectory',train_trajectories=1200,validation_trajectories=300,full_trajectories=1500,frozen_before_any_candidate_training=True,target_IDs_used=0)
 write(ROOT/'outputs/sp_u12b5b_preparation_split_manifest.json',split)
 write(ROOT/'outputs/sp_u12b5b_candidate_seed_registry.json',dict(B5A_source_sha256=FID_SHA,candidates=grid,seeds_reused_exactly=True,seeds_regenerated=False))
 # Verify every cached trajectory against original channel hashes, not old outcomes.
 source={int(r['trajectory_id']):r['replayed_h_sha256'] for r in csvread(ROOT/'outputs/t15a_fix_replay_h_hashes.csv') if r['split']=='train_dev'}
 shift={(r['shift'],int(r['trajectory_id'])):r['H_sha256'] for r in csvread(ROOT/'outputs/t15c_cdl_channel_hashes.csv') if r['split']=='train_dev'}
 manifest_source={(int(r['trajectory_id']),int(r['frame_id'])):r for r in csvread(ROOT/'outputs/t15a_cdl_manifest.csv') if r['split']=='train_dev'}
 manifest_shift={(r['shift'],int(r['trajectory_id']),int(r['frame_id'])):r for r in csvread(ROOT/'outputs/t15c_cdl_manifest.csv') if r['split']=='train_dev'}
 OUT.mkdir(exist_ok=True)
 train=np.lib.format.open_memmap(OUT/'selection_train.npy',mode='w+',dtype=np.complex128,shape=(12000,4032))
 full=np.lib.format.open_memmap(OUT/'full_retraining.npy',mode='w+',dtype=np.complex128,shape=(15000,4032))
 truth=np.lib.format.open_memmap(OUT/'validation_truth.npy',mode='w+',dtype=np.complex128,shape=(3,100,6,3968))
 pilots=np.empty((3,2,100,6,64),dtype=np.complex128);noise_records=[];trajectory_proofs=[]
 rx,tx,k,t=np.unravel_index(interface.selected_full,(2,2,64,16),order='C')
 for o,name in enumerate(names):
  for i in range(500):
   frames=np.asarray(x[o,i]);H=frames.reshape(10,2,2,64,16).transpose(1,2,3,0,4).reshape(2,2,64,160)
   expected=source[i] if o==0 else shift[name,i];actual=array_hash(H)
   assert actual==expected,'PREPARATION_TRAJECTORY_REPLAY_HASH'
   trajectory_proofs.append(dict(origin=name,trajectory_id=i,H_sha256=actual,preparation_only=True))
   joint=frames[:,interface.joint_full]
   full[o*5000+i*10:o*5000+(i+1)*10]=joint
   if i<400:train[o*4000+i*10:o*4000+(i+1)*10]=joint
   else:
    truth[o,i-400]=frames[4:10][:,interface.data_full]
    for jj,j in enumerate(range(4,10)):
     row=manifest_source[i,j] if o==0 else manifest_shift[name,i,j]
     rng=np.random.default_rng(int(row['noise_seed']))
     unit=(rng.standard_normal((2,64,16))+1j*rng.standard_normal((2,64,16)))/np.sqrt(2)
     owner=unit[rx,k,t]/interface.owner_symbols
     for si,s in enumerate(('10','20')):pilots[o,si,i-400,jj]=joint[j,interface.pilot_positions]+np.sqrt(cfg['noise_variances'][s])*owner
     noise_records.append(dict(origin=name,trajectory_id=i,frame_id=j,seed=int(row['noise_seed']),unit_owner_divided_noise_sha256=array_hash(owner)))
  print(json.dumps(dict(stage='prepare',origin=name,clean_trajectories_verified=500)),flush=True)
 train.flush();full.flush();truth.flush();del train,full,truth,x
 with (OUT/'validation_pilots.npy').open('xb') as f:np.save(f,pilots,allow_pickle=False)
 write(OUT/'preparation_channel_proofs.json',trajectory_proofs);write(OUT/'validation_noise_registry.json',noise_records)
 inputs={n:sha(OUT/n) for n in ('selection_train.npy','full_retraining.npy','validation_truth.npy','validation_pilots.npy','validation_noise_registry.json','preparation_channel_proofs.json')}
 write(OUT/'input_manifest.json',dict(PASS=True,files=inputs,training_shape=[12000,4032],final_training_shape=[15000,4032],validation_scored_frames=list(range(4,10)),canonical_origins=names,preparation_trajectories_verified=1500,CRN='same immutable validation_pilots.npy for all nine candidates',training_order='origin source_names order, ascending trajectory ID, ascending frame0..9, frozen joint C-order',target_files_read=0))
 write(ROOT/'outputs/sp_u12b5b_formal_eval_preregistration.json',dict(status='PREREGISTRATION_ONLY_NOT_EXECUTED',conditions=['ID','Profile','Mobility','Composed','Boundary'],snr_dB=[0,5,10,15,20,25,30],trajectories_per_condition=1000,input_source='same immutable B2 formal trajectories, future authorization required',contrast='g_MFA=NMSE_MFA_dB-NMSE_LR4_dB',family_size=35,bootstrap_replicates=100000,unit='paired trajectory',interval='two-sided95% Bonferroni, linear quantiles q=.05/(2*35)',family_independent_of_B2_original_35=True,B2_thresholds_recomputed=False,target_scientific_inputs_read=0,executed=False))
 (ROOT/'docs/sp_u12b5b_selection_protocol.md').write_text('# Target-blind MFA selection protocol\n\n'+json.dumps(cfg,indent=2)+'\n\nSplit and CRN files are immutable before training. All nine structures must complete; any invalid model, training crash or whole-grid resource STOP ends selection without choosing survivors. One pooled model per configuration; no origin-label input. Validation CSI used only for scored-frame4–9 error energies, never EM or early stopping. Selection is automatic argmin of six equally weighted linear pooled NMSE cells. Exact ties choose smaller K then smaller rank. Final retraining is a new fit, with the selected structure\'s B5A-preassigned full-retraining seed, after the phase1 manifest freezes. No target inputs/results or method comparison.\n',encoding='utf8')
 write(OUT/'preparation_access_audit.json',dict(allowed_reads=ACCESS,blocked=DENIED,target_scientific_files_read=0,B2_B3_scientific_values_read=0,integrity_only_hashing_separate=True))
 print(json.dumps(dict(stage='B5B_preparation_complete',PASS=True,inputs=inputs)),flush=True)
if __name__=='__main__':prepare()
