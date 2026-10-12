"""SP-U12B2B. Scientific outputs are machine-computed and stay sealed."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,time,json,hashlib,argparse,subprocess,concurrent.futures,traceback,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import sp_u12b2b_common as b
from scripts import sp_u12b2a_common as c
from scripts import run_sp_u12b2a as a
REQUEST=Path(r'C:\Users\yuhua\.codex\attachments\d64f7aa6-0125-4b2c-909e-fc54016a795a\已粘贴的文本.txt')

def prepare():
 b.need(not b.CFG.exists(),'B2B_CONFIG_EXISTS');b.original_gate();b.TECH.mkdir(exist_ok=True)
 b.need((b.TECH/'before.json').exists(),'PROTECTION_BASELINE_MISSING')
 b.need(not b.SEAL.exists(),'SEALED_DIRECTORY_ALREADY_EXISTS')
 b.PROTOCOL.write_text(REQUEST.read_text(encoding='utf-8-sig'),encoding='utf8')
 reg=b.registry();dev=b.read(c.OUT/'development_seed_registry.json');history=b.read(c.OUT/'historical_seed_inventory.json')
 formal={r['seed'] for r in reg['rows']};development={r['seed'] for r in dev['rows']};forbidden=set(history['forbidden_seeds'])|formal|development
 b.need(len(formal)==172501 and len(development)==540 and not formal&development and not formal&set(history['forbidden_seeds']),'FORMAL_SEED_INTEGRITY')
 primary=next(r['seed'] for r in reg['rows'] if r['role']=='bootstrap_master_seed');counter=0
 while True:
  token=f'{b.sha(ROOT/"outputs/sp_u12b2a_seed_registry.json")}|SP_U12B2B_DESCRIPTIVE_BOOTSTRAP_V1|{counter}'
  descriptive=int.from_bytes(hashlib.sha256(token.encode()).digest()[:4],'big')%(2**32-3)+2
  if not any(x in forbidden for x in (descriptive-1,descriptive,descriptive+1)):break
  counter+=1
 from src.cdl_benchmark import observation_geometry,array_hash
 from src.pilot_cv import frozen_folds
 g=observation_geometry();folds,fold_definition=frozen_folds(g.selected)
 geometry=dict(selected_sha256=array_hash(g.selected),data_sha256=array_hash(g.data),pilot_mask_sha256=array_hash(g.layout.pilot_mask),data_mask_sha256=array_hash(g.layout.data_mask))
 # Bounded independent CPU workers preserve per-process BLAS1. Generator stays single/default device.
 memory=b.memory_available();workers=min(4,max(1,(os.cpu_count() or 1)//2),max(1,int(memory['available_physical_bytes']/2**30)-3))
 config=dict(task='SP-U12B2B',run_id='SP_U12B2B_FULL_SNR_SEALED_V1',authorization='User task book; machine sealed computation permitted, human/model unseal prohibited',snr_db=list(c.SNRS),noise_variances={str(s):c.variance(s) for s in c.SNRS},conditions=c.CONDITIONS,methods=list(c.METHODS),display_labels={'Dense':'Dense Affine','LR4':'LR4 Affine'},N_reference_per_condition=500,N_evaluation_per_condition=1000,frames=10,scored_frames=list(range(4,10)),rank=4,window=5,folds=4,fold_hash=hashlib.sha256(json.dumps(fold_definition,sort_keys=True).encode()).hexdigest(),fold_definition=fold_definition,pilot_geometry=geometry,pilot_geometry_hash=hashlib.sha256(json.dumps(geometry,sort_keys=True).encode()).hexdigest(),expected_bit_total_per_frame=int(2*np.count_nonzero(g.layout.data_mask)),Source_reference_power=c.POWER,source_centering='frozen Source means',operator_cache_sha256=b.sha(c.OUT/'operators.npz'),B2A_execution_freeze_sha256=b.sha(c.OUT/'execution_freeze.json'),B2A_registry_sha256=b.sha(ROOT/'outputs/sp_u12b2a_seed_registry.json'),primary_bootstrap=dict(replicates=100000,family_size=35,q=.05/70,master_seed=primary,stream='default_rng(SeedSequence([master_seed,condition_ordinal]))',chunk_size=1000,quantile_method='linear',paired_across_methods_and_SNR=True),descriptive_bootstrap=dict(replicates=10000,seed=descriptive,namespace='SP_U12B2B_DESCRIPTIVE_BOOTSTRAP_V1',collision_rejections=counter,quantiles=[.025,.975],stream='default_rng(SeedSequence([seed,condition_ordinal]))',paired_across_methods_and_SNR=True,scope='DESCRIPTIVE_ONLY; no multiplicity correction; excluded from35primarycontrasts'),worker_count=workers,generator_workers=1,BLAS_threads_per_process=1,common_random_numbers='same H/QPSK/unit CN noise across7SNR; only sigma scales; differentSNR points not independent N',statistical_unit='independent trajectory',primary_contrasts=c.planned_contrasts(),receiver='frozen unregularized2x2 ZF + GrayQPSK; Perfect CSI BER-only',sealing='Every outcome lives in sp_u12b2b_sealed; only technical receipts may be viewed',MFA_unseal_requirements=['fidelity gate complete','initialization/restart frozen','K/r selection on preparation validation only','final full500/origin retraining complete','final model hash frozen'],B3_design_unchanged=True,PAGE_BUDGET_DEBT='+5 review pages;record only',retry_policy='same run/seeds/cached artifacts; missing only; no post-result patch; corruption requires explicit same-seed recovery record')
 b.write(b.CFG,config)
 source_config=c.read(c.CONFIG)
 b.need(source_config['sample_rate_hz']==source_config['nfft']*source_config['delta_f_hz'],'SAMPLE_RATE_CONTRACT')
 config.update(sample_rate_hz=source_config['sample_rate_hz'],nfft=source_config['nfft'],delta_f_hz=source_config['delta_f_hz'],memory_preflight=memory)
 b.write(b.CFG,config)
 b.write(ROOT/'outputs/sp_u12b2b_seed_audit.json',dict(PASS=True,formal_registry_sha256=config['B2A_registry_sha256'],formal_seed_values=172501,reference_channel_role_seeds=7500,evaluation_channel_role_seeds=15000,evaluation_communication_role_seeds=150000,primary_master_seed_reused_exactly=True,descriptive_seed=descriptive,descriptive_primary_seed_overlap=False,descriptive_formal_development_history_overlap=0,reference_evaluation_overlap=0,individual_seed_duplicates=0,descriptive_namespace=config['descriptive_bootstrap']['namespace'],descriptive_seed_selection_before_scientific_generation=True))
 schema=dict(raw='One NPZ pertrajectory; axesSNR7,methods7,frames6. Condition/trajectory fixed by path+commit. frame_error/truth/bit_error/bit_total map frozen12-field logical schema; method_coordinates+coordinate_present encode nullable alpha/beta; solve_status0NA/1PASS.',raw_logical_fields=list(c.RAW_FIELDS),raw_logical_rows=1470000,trajectory_logical_rows=245000,trajectory_statistics=['trajectory_error(SNR,method)','trajectory_truth(SNR)','trajectory_bit_errors(SNR,method)','trajectory_bit_total(SNR)'],coordinate_schema='shared affine coordinates(condition,trajectory,SNR,frame,2);210000fit records;not a primaryendpoint',aggregate_fields=['condition','snr_db','estimator','pooled_nmse_db','ber','desc_ci_low','desc_ci_high','interval_scope'],aggregate_rows=245,figure_fields=['condition','snr_db','estimator','pooled_nmse_db','desc_ci_low','desc_ci_high','interval_scope'],figure_rows=245,contrast_fields=['comparison','condition','snr_db','estimate_db','simul_lower','simul_upper','classification'],contrast_rows=35,BER_reference='35PerfectCSI receiver cells;notNMSE',main_figure_methods=['LS-LIN','Source','Convex','Dense','LR4','Matched'],supplement_methods=list(c.METHODS),no_plot_rendering=True)
 b.write(b.TECH/'schema.json',schema)
 b.emit(stage='prepare',status='PASS',worker_count=workers,formal_H_generated=0,outcomes_viewed=False)

def freeze():
 b.original_gate();cfg=b.config();tests=b.read(b.TECH/'pytest_preflight.json')
 b.need(tests['exit_code']==0 and tests['failed']==0 and tests['skipped']==0,'PYTEST_PREFLIGHT_NOT_PASS')
 index=b.seed_index();runtime={}
 with b.phase('REFERENCE'):
  from src.cdl_benchmark import observation_geometry
  expected=c.read(c.OUT/'execution_freeze.json')['runtime_configurations']
  fields=('normalized_PDP_powers','scaled_cluster_delays_seconds','array_positions_m','device','internal_dtype','export_dtype','ofdm_channel_normalize','time_states_per_call','sampling_frequency_hz')
  for condition in c.CONDITIONS:
   meta=a.generator(condition,index,'MATCHED_REF').provenance()
   b.need(all(meta[key]==expected[condition][key] for key in fields),'GENERATOR_METADATA_DRIFT')
   runtime[condition]={key:meta[key] for key in fields};runtime[condition]['declared_condition']=c.CONDITIONS[condition]
 paths=[*b.SOURCE_FILES,'configs/sp_u12b2b_sealed.json','docs/sp_u12b2b_protocol.md','outputs/sp_u12b2b/schema.json','outputs/sp_u12b2b/pytest_preflight.json','outputs/sp_u12b2b/pytest_preflight.txt','outputs/sp_u12b2b/pytest_preflight.xml','outputs/sp_u12b2b/resource_amendment.json','outputs/sp_u12b2b_seed_audit.json','scripts/sp_u4a_common.py']
 hashes={path:b.sha(ROOT/path) for path in paths}
 hashes.update(c.read(c.OUT/'execution_freeze.json')['hashes'])
 hashes.update({str(c.OUT/'operators.npz'):b.sha(c.OUT/'operators.npz')})
 from importlib.metadata import version
 versions={name:version(name) for name in ('numpy','scipy','torch','sionna-no-rt')}
 b.write(b.TECH/'execution_freeze.json',dict(run_id=cfg['run_id'],hashes=hashes,protocol_sha256=b.sha(b.PROTOCOL),config_sha256=b.sha(b.CFG),seed_registry_sha256=cfg['B2A_registry_sha256'],runtime=runtime,versions=versions,python=sys.version,worker_count=cfg['worker_count'],generator_workers=1,BLAS_threads_per_process=1,created_utc=b.timestamp(),generator_calls=0,scientific_outcomes_viewed=False,pytest=tests))
 for folder in ('reference','cache','raw_evaluation','primary_bootstrap','descriptive_bootstrap'):(b.SEAL/folder).mkdir(parents=True,exist_ok=True)
 b.emit(stage='freeze',status='PASS',formal_H_generated=0,worker_count=cfg['worker_count'],outcomes_viewed=False)

def get_channel(condition,i,group,adapter,index):
 from src.cdl_benchmark import array_hash
 path=b.channel_path(group,condition,i);receipt=path.with_suffix('.json');prior=b.committed(receipt)
 if prior is not None:
  b.need(prior['channel_seeds']==a.consumed(index,condition,group,i),'CHANNEL_SEED_BINDING')
  return prior,False
 h=adapter.generate(i);record=b.archive(path,H=h)
 info=dict(condition=condition,group=group,trajectory_id=i,channel_seeds=a.consumed(index,condition,group,i),H_sha256=array_hash(h),export_strides_bytes=list(h.strides),frames=10,time_states=160,finite=True,artifacts=[record],created_utc=b.timestamp())
 # Always load through the exact persisted-layout path, even on the first run.
 restored=b.load_h(path,info['export_strides_bytes']);b.need(array_hash(restored)==info['H_sha256'],'PERSISTED_H_VALUES')
 b.write(receipt,info);return info,True

def reference(index):
 from src.cdl_benchmark import StreamingStatistics,observation_geometry,frames
 g=observation_geometry();tick=time.monotonic();last=tick;done=0;generated=0
 for condition in c.CONDITIONS:
  adapter=a.generator(condition,index,'MATCHED_REF')
  for i in range(500):
   info,new=get_channel(condition,i,'MATCHED_REF',adapter,index);done+=1;generated+=int(new)
   if time.monotonic()-last>25:b.emit(stage='reference',condition=condition,reference_count=done,elapsed_seconds=time.monotonic()-tick);last=time.monotonic()
  statfile=b.SEAL/'reference'/condition/'statistics.npz';proof=statfile.with_suffix('.json')
  if b.committed(proof) is None:
   stream=StreamingStatistics(g.selected,g.data)
   for i in range(500):
    path=b.channel_path('MATCHED_REF',condition,i);info=b.committed(path.with_suffix('.json'));h=b.load_h(path,info['export_strides_bytes']);stream.update(frames(h).reshape(10,4096),[i])
   st=stream.freeze();b.need(st.samples==5000 and st.train_trajectory_ids==tuple(range(500)),'REFERENCE_STATISTICS_POPULATION')
   for value in (st.mean_p,st.mean_d,st.rpp,st.rdp):b.need(np.isfinite(value).all(),'REFERENCE_STATISTICS_NONFINITE')
   record=b.archive(statfile,mean_p=st.mean_p,mean_d=st.mean_d,rpp=st.rpp,rdp=st.rdp,train_trajectory_ids=np.array(st.train_trajectory_ids),samples=np.array(st.samples))
   b.write(proof,dict(condition=condition,clean_trajectories=500,clean_frames=5000,source='B2A formal MATCHED_REF seeds only',artifacts=[record],created_utc=b.timestamp()))
  b.emit(stage='reference',condition=condition,reference_count=done,status='COMPLETE')
 b.write(b.TECH/'reference_receipt.json',dict(PASS=True,reference_trajectories=2500,reference_generated_this_session=generated,elapsed_seconds=time.monotonic()-tick,created_utc=b.timestamp()))

def get_communication(condition,i,index):
 from src.cdl_benchmark import array_hash
 path=b.SEAL/'cache'/condition/f'comm_{i:04d}.npz';prior=b.committed(path.with_suffix('.json'))
 seedrows=[a.consumed(index,condition,'EVAL',i,j) for j in range(10)]
 if prior is not None:b.need(prior['frame_seeds']==seedrows,'COMM_SEED_BINDING');return prior
 comm=[c.communicate(row) for row in seedrows]
 record=b.archive(path,bits=np.stack([r[0] for r in comm]),symbols=np.stack([r[1] for r in comm]),unit_noise=np.stack([r[2] for r in comm]))
 info=dict(condition=condition,trajectory_id=i,frame_seeds=seedrows,shared_across_SNR=True,bits_sha256=[array_hash(row[0]) for row in comm],symbols_sha256=[array_hash(row[1]) for row in comm],unit_noise_sha256=[array_hash(row[2]) for row in comm],artifacts=[record],created_utc=b.timestamp());b.write(path.with_suffix('.json'),info);return info

def evaluation(index):
 cfg=b.config();workers=cfg['worker_count'];tick=time.monotonic();last=tick;done=0;generated=0;worker_peaks={};resumed=0
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=b.init_eval_worker) as pool:
  pending={}
  def harvest(wait_for_one=False):
   nonlocal done,last,resumed
   if not pending:return
   finished,_=concurrent.futures.wait(pending,timeout=None if wait_for_one else 0,return_when=concurrent.futures.FIRST_COMPLETED)
   for future in finished:
    condition,i=pending.pop(future);result=future.result();b.need(result['completed'] and result['condition']==condition and result['trajectory_id']==i,'WORKER_IDENTITY')
    done+=1;resumed+=int(result['resumed']);worker_peaks[str(result['worker_pid'])]=result['peak']
   if time.monotonic()-last>25:
    b.emit(stage='evaluation',evaluation_count=done,elapsed_seconds=time.monotonic()-tick,worker_count=workers);last=time.monotonic()
  for condition in c.CONDITIONS:
   adapter=a.generator(condition,index,'EVAL')
   for i in range(1000):
    get_channel(condition,i,'EVAL',adapter,index);get_communication(condition,i,index)
    future=pool.submit(b.evaluate_worker,condition,i);pending[future]=(condition,i);harvest()
    if len(pending)>=2*workers:harvest(True)
   while pending:harvest(True)
   b.emit(stage='evaluation',condition=condition,evaluation_count=done,cells_completed=7*(list(c.CONDITIONS).index(condition)+1),status='COMPLETE')
 b.need(done==5000,'EVALUATION_POPULATION_INCOMPLETE')
 b.write(b.TECH/'evaluation_receipt.json',dict(PASS=True,evaluation_trajectories=5000,raw_logical_rows=1470000,trajectory_logical_rows=245000,condition_SNR_cells=35,elapsed_seconds=time.monotonic()-tick,worker_count=workers,worker_peaks=worker_peaks,resumed_completed_trajectories=resumed,created_utc=b.timestamp(),outcomes_viewed=False))

def run():
 b.check_freeze();b.firewall();index=b.seed_index();tick=time.monotonic()
 b.emit(stage='run',status='START',worker_count=b.config()['worker_count'],worker_pid=os.getpid(),outcomes_viewed=False)
 with b.phase('REFERENCE'):reference(index)
 with b.phase('RAW'):evaluation(index)
 with b.phase('INTEGRITY'):audit=integrity()
 b.write(ROOT/'outputs/sp_u12b2b_integrity_audit.json',audit)
 with b.phase('SEALED_COMPUTE'):compute()
 with b.phase('SEAL_CHECK'):seal_outputs()
 import torch
 gpu=dict(device=next(iter(b.read(b.TECH/'execution_freeze.json')['runtime'].values()))['device'],GPU_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,GPU_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None)
 failures=[b.read(b.TECH/'technical_failure.json')] if (b.TECH/'technical_failure.json').exists() else []
 resource=dict(run_id=b.config()['run_id'],elapsed_seconds_current_session=time.monotonic()-tick,parent_worker_pid=os.getpid(),parent_peak=b.peak(),evaluation_worker_peaks=b.read(b.TECH/'evaluation_receipt.json')['worker_peaks'],CPU_workers=b.config()['worker_count'],generator_workers=1,BLAS_threads_per_process=1,disk_bytes=sum(p.stat().st_size for p in b.SEAL.rglob('*') if p.is_file()),gpu=gpu,failures=failures,resumes=b.read(b.TECH/'evaluation_receipt.json')['resumed_completed_trajectories'],runtime_claim=False,scope='engineering log;not B4',created_utc=b.timestamp())
 b.write(ROOT/'outputs/sp_u12b2b_resource_log.json',resource)
 b.emit(stage='run',status='SEALED_COMPUTATION_COMPLETE',reference_count=2500,evaluation_count=5000,cells_completed=35,outcomes_viewed=False)

def integrity():
 cfg=b.config();index=b.seed_index();seen=set();reference_hashes=set();evaluation_hashes=set();raw_files=[];refcount=evalcount=0;peaks={}
 from src.cdl_benchmark import array_hash
 for condition in c.CONDITIONS:
  stat=b.committed((b.SEAL/'reference'/condition/'statistics.npz').with_suffix('.json'));b.need(stat is not None,'MATCHED_STATISTICS_MISSING')
  for group,total in [('MATCHED_REF',500),('EVAL',1000)]:
   for i in range(total):
    path=b.channel_path(group,condition,i);info=b.committed(path.with_suffix('.json'));b.need(info is not None,'CHANNEL_MISSING')
    b.need(info['condition']==condition and info['group']==group and info['trajectory_id']==i and info['channel_seeds']==a.consumed(index,condition,group,i),'CHANNEL_KEY_OR_SEED')
    h=b.load_h(path,info['export_strides_bytes']);digest=array_hash(h);b.need(digest==info['H_sha256'] and digest not in seen,'CHANNEL_DUPLICATION_OR_HASH');seen.add(digest)
    if group=='MATCHED_REF':reference_hashes.add(digest);refcount+=1;continue
    evaluation_hashes.add(digest);evalcount+=1
    commpath=b.SEAL/'cache'/condition/f'comm_{i:04d}.npz';comm=b.committed(commpath.with_suffix('.json'));b.need(comm is not None,'COMMUNICATION_MISSING')
    seeds=[a.consumed(index,condition,'EVAL',i,j) for j in range(10)];b.need(comm['frame_seeds']==seeds,'COMMUNICATION_SCOPE')
    z=b.arrays(commpath)
    b.need(z['bits'].shape==(10,2,64,16,2) and z['symbols'].shape==(10,2,64,16) and z['unit_noise'].shape==(10,2,64,16),'COMMUNICATION_SHAPE')
    b.need(np.isfinite(z['symbols']).all() and np.isfinite(z['unit_noise']).all(),'COMMUNICATION_NONFINITE')
    for j in range(10):
     rebuilt=c.communicate(seeds[j]);b.need(all(np.array_equal(z[key][j],value) for key,value in zip(('bits','symbols','unit_noise'),rebuilt)),'COMMUNICATION_SEED_REALIZATION')
    proof=b.committed(b.raw_receipt(condition,i));b.need(proof is not None,'RAW_MISSING')
    b.need(proof['condition']==condition and proof['trajectory_id']==i and proof['estimators']==list(c.METHODS) and proof['snr_db']==list(c.SNRS),'RAW_PRIMARY_KEYS')
    b.need(proof['H_sha256']==digest and proof['H_artifact_sha256']==info['artifacts'][0]['sha256'] and proof['communication_sha256']==comm['artifacts'][0]['sha256'],'RAW_INPUT_BINDING')
    b.need(proof['matched_reference_statistics_sha256']==stat['artifacts'][0]['sha256'],'MATCHED_SOURCE_BINDING')
    b.need(proof['rank']==4 and proof['window_frames']==5 and proof['fold_hash']==cfg['fold_hash'] and proof['pilot_geometry_hash']==cfg['pilot_geometry_hash'] and proof['operator_sha256']==cfg['operator_cache_sha256'],'METHOD_METADATA_DRIFT')
    b.need(all(proof[key] for key in ('method_pairing','CRN_pairing','dense_lr4_shared_coordinates','finite_estimates','valid_coordinate_solve')),'NUMERICAL_RECEIPT')
    b.validate_inputs(proof['inputs'])
    for entry in proof['inputs']:b.need(entry['unit_noise_sha256']==comm['unit_noise_sha256'][entry['frame_id']],'CROSS_SNR_NOISE_ID_BINDING')
    b.validate_raw(b.arrays(b.raw_path(condition,i)),cfg['expected_bit_total_per_frame']);raw_files.append(b.raw_path(condition,i).relative_to(ROOT).as_posix());peaks[str(proof['worker_pid'])]=proof['peak']
  b.emit(stage='integrity',condition=condition,reference_count=refcount,evaluation_count=evalcount,status='PASS')
 b.need(refcount==2500 and evalcount==5000 and not reference_hashes&evaluation_hashes,'FORMAL_POPULATION_SEPARATION')
 b.need(len(list((b.SEAL/'raw_evaluation').glob('*/*.npz')))==5000,'EXTRA_RAW_FILES')
 b.need(len(list((b.SEAL/'reference').glob('*/H_*.npz')))==2500 and len(list((b.SEAL/'cache').glob('*/H_*.npz')))==5000,'EXTRA_CHANNEL_FILES')
 return dict(PASS=True,reference_trajectories=2500,evaluation_trajectories=5000,condition_SNR_cells=35,raw_logical_rows=1470000,trajectory_logical_rows=245000,scored_frames=6,methods=7,unique_clean_H=7500,raw_schema_integrity=True,key_uniqueness=True,CRN_pairing=True,method_pairing=True,coordinate_sharing=True,matched_reference_separation=True,reference_evaluation_hash_overlap=0,finite_scientific_outputs=True,nonfinite_count=0,technical_solve_failures=0,bit_totals_expected=True,all_communication_seed_realizations_rebuilt=True,rank_fold_window_pilot_operator_integrity=True,preparation_statistics_hashes=b.original_gate()['preparation_statistics_hashes'],raw_file_count=5000,worker_peaks=peaks,outcomes_viewed=False,figures_viewed=False,created_utc=b.timestamp())

def compute():
 cfg=b.config();schema=b.read(b.TECH/'schema.json');aggregates=[];figure=[];contrasts=[];BERref=[];coordinates=np.empty((5,1000,7,6,2));primary_audits=[];descriptive_audits=[]
 for ordinal,condition in enumerate(c.CONDITIONS):
  E=[];T=[];BE=[];BT=[];PB=[]
  for i in range(1000):
   raw=b.arrays(b.raw_path(condition,i));E.append(raw['trajectory_error']);T.append(raw['trajectory_truth']);BE.append(raw['trajectory_bit_errors']);BT.append(raw['trajectory_bit_total']);PB.append(raw['perfect_frame_bit_errors'].sum(axis=1));coordinates[ordinal,i]=raw['coordinates']
  E=np.array(E);T=np.array(T);BE=np.array(BE);BT=np.array(BT);PB=np.array(PB)
  point=10*np.log10(E.sum(axis=0)/T.sum(axis=0)[:,None]);ber=BE.sum(axis=0)/BT.sum(axis=0)[:,None];b.need(np.isfinite(point).all() and np.isfinite(ber).all(),'AGGREGATE_NONFINITE')
  family=[r for r in cfg['primary_contrasts'] if r['condition']==condition]
  if family:
   path=b.SEAL/'primary_bootstrap'/f'{condition}.npz';receipt=path.with_suffix('.json');prior=b.committed(receipt)
   if prior is None:
    draws,indexhash=b.bootstrap_array(E,T,cfg['primary_bootstrap']['master_seed'],ordinal,100000,family,lambda completed,total:b.emit(stage='primary_bootstrap',condition=condition,replicates_completed=completed,replicates_expected=total))
    rec=b.archive(path,draws=draws);prior=dict(condition=condition,replicates=100000,contrasts=len(family),index_stream_sha256=indexhash,paired_trajectory_indices=True,seed_stream=cfg['primary_bootstrap']['stream'],condition_ordinal=ordinal,master_seed=cfg['primary_bootstrap']['master_seed'],quantile_method='linear',q=.05/70,artifacts=[rec],created_utc=b.timestamp());b.write(receipt,prior)
   else:draws=b.arrays(path)['draws']
   b.need(draws.shape==(100000,len(family)) and np.isfinite(draws).all(),'PRIMARY_BOOTSTRAP_POPULATION')
   bounds=np.quantile(draws,[.05/70,1-.05/70],axis=0,method='linear')
   for k,row in enumerate(family):
    si=c.SNRS.index(row['snr_db']);estimate=point[si,c.METHODS.index(row['first'])]-point[si,c.METHODS.index(row['second'])];lower,upper=bounds[:,k]
    contrasts.append(dict(comparison=row['comparison'],condition=condition,snr_db=row['snr_db'],estimate_db=float(estimate),simul_lower=float(lower),simul_upper=float(upper),classification=c.classify(float(lower),float(upper))))
   primary_audits.append({key:prior[key] for key in ('condition','replicates','contrasts','index_stream_sha256')})
  path=b.SEAL/'descriptive_bootstrap'/f'{condition}.npz';receipt=path.with_suffix('.json');prior=b.committed(receipt)
  if prior is None:
   draws,indexhash=b.bootstrap_array(E,T,cfg['descriptive_bootstrap']['seed'],ordinal,10000,None,lambda completed,total:b.emit(stage='descriptive_bootstrap',condition=condition,replicates_completed=completed,replicates_expected=total))
   rec=b.archive(path,draws=draws);prior=dict(condition=condition,replicates=10000,cells=49,index_stream_sha256=indexhash,paired_trajectory_indices=True,seed_stream=cfg['descriptive_bootstrap']['stream'],condition_ordinal=ordinal,master_seed=cfg['descriptive_bootstrap']['seed'],interval_scope='DESCRIPTIVE_ONLY',multiplicity_correction=False,quantile_method='linear',quantiles=[.025,.975],artifacts=[rec],created_utc=b.timestamp());b.write(receipt,prior)
  else:draws=b.arrays(path)['draws']
  b.need(draws.shape==(10000,7,7) and np.isfinite(draws).all(),'DESCRIPTIVE_BOOTSTRAP_POPULATION');bounds=np.quantile(draws,[.025,.975],axis=0,method='linear')
  for si,s in enumerate(c.SNRS):
   for mi,m in enumerate(c.METHODS):
    row=dict(condition=condition,snr_db=s,estimator=m,pooled_nmse_db=float(point[si,mi]),ber=float(ber[si,mi]),desc_ci_low=float(bounds[0,si,mi]),desc_ci_high=float(bounds[1,si,mi]),interval_scope='DESCRIPTIVE_ONLY');aggregates.append(row);figure.append({key:row[key] for key in schema['figure_fields']})
   BERref.append(dict(condition=condition,snr_db=s,receiver_reference='Perfect CSI',bit_errors=int(PB.sum(axis=0)[si]),bit_total=int(BT.sum(axis=0)[si]),ber=float(PB.sum(axis=0)[si]/BT.sum(axis=0)[si]),NMSE_estimator_path=False))
  descriptive_audits.append({key:prior[key] for key in ('condition','replicates','cells','index_stream_sha256')});b.emit(stage='sealed_compute',condition=condition,status='COMPLETE',outcomes_viewed=False)
 b.need(len(aggregates)==len(figure)==245 and len(contrasts)==35 and len(BERref)==35,'OUTCOME_ROW_CARDINALITY')
 b.csvwrite(b.SEAL/'aggregate_results.csv',aggregates,schema['aggregate_fields']);b.csvwrite(b.SEAL/'figure_data.csv',figure,schema['figure_fields']);b.csvwrite(b.SEAL/'primary_contrasts.csv',contrasts,schema['contrast_fields']);b.csvwrite(b.SEAL/'perfect_CSI_BER_reference.csv',BERref,list(BERref[0]))
 b.need(np.isfinite(coordinates).all(),'COORDINATE_ARCHIVE_NONFINITE');b.archive(b.SEAL/'coordinate_data.npz',coordinates=coordinates,conditions=np.array(list(c.CONDITIONS)),snr_db=np.array(c.SNRS),trajectory_ids=np.arange(1000),frame_ids=np.arange(4,10))
 b.write(ROOT/'outputs/sp_u12b2b_bootstrap_audit.json',dict(PASS=True,primary_complete=True,primary_replicates_per_contrast=100000,primary_family_size=35,primary_condition_streams=3,primary=primary_audits,descriptive_complete=True,descriptive_replicates_per_cell=10000,descriptive_cells=245,descriptive_condition_streams=5,descriptive=descriptive_audits,primary_descriptive_master_seed_overlap=False,all_index_stream_hashes_unique=len({r['index_stream_sha256'] for r in primary_audits+descriptive_audits})==8,quantile_method='linear',paired_statistical_unit='trajectory; pool resampled energy ratio before dB',outputs_viewed=False,created_utc=b.timestamp()))

def seal_outputs():
 # Verify machine-facing CSVs internally, emitting no scientific value or label counts.
 schema=b.read(b.TECH/'schema.json');checks=[]
 for name,fields,total in [('aggregate_results.csv',schema['aggregate_fields'],245),('figure_data.csv',schema['figure_fields'],245),('primary_contrasts.csv',schema['contrast_fields'],35)]:
  with (b.SEAL/name).open(encoding='utf8',newline='') as f:
   reader=__import__('csv').DictReader(f);b.need(reader.fieldnames==fields,'FINAL_CSV_FIELDS');rows=list(reader)
  b.need(len(rows)==total,'FINAL_CSV_ROWS');keys=[(r['condition'],r['snr_db'],r.get('estimator',r.get('comparison'))) for r in rows];b.need(len(set(keys))==total,'FINAL_CSV_KEYS')
  numeric=[key for key in fields if key in ('pooled_nmse_db','ber','desc_ci_low','desc_ci_high','estimate_db','simul_lower','simul_upper')]
  b.need(all(np.isfinite(float(r[key])) for r in rows for key in numeric),'FINAL_CSV_NONFINITE')
  if 'classification' in fields:b.need(all(r['classification'] in ('POSITIVE','NEGATIVE','INDETERMINATE') for r in rows),'CLASSIFICATION_SCHEMA')
  else:b.need(all(r['interval_scope']=='DESCRIPTIVE_ONLY' and float(r['desc_ci_low'])<=float(r['desc_ci_high']) for r in rows),'DESCRIPTIVE_SCHEMA')
  checks.append(dict(file=name,rows=total,fields=fields,finite=True,key_unique=True,sha256=b.sha(b.SEAL/name)))
 b.need(not any(p.suffix.lower() in ('.png','.pdf','.svg','.jpg') for p in b.SEAL.rglob('*') if p.is_file()),'SCIENTIFIC_FIGURE_CREATED')
 leaves={p.relative_to(b.SEAL).as_posix():dict(sha256=b.sha(p),bytes=p.stat().st_size) for p in sorted(b.SEAL.rglob('*')) if p.is_file() and not p.name.endswith('.partial')}
 b.need(not list(b.SEAL.rglob('*.partial')),'UNCOMMITTED_PARTIAL_FILES')
 tree=hashlib.sha256(json.dumps(leaves,sort_keys=True,separators=(',',':')).encode()).hexdigest();freeze=b.read(b.TECH/'execution_freeze.json')
 def subtree(prefix):return hashlib.sha256(json.dumps({k:v for k,v in leaves.items() if k.startswith(prefix)},sort_keys=True,separators=(',',':')).encode()).hexdigest()
 manifest=dict(status='SEALED_COMPLETE_UNVIEWED',run_id=b.config()['run_id'],protocol_sha256=b.sha(b.PROTOCOL),config_sha256=b.sha(b.CFG),seed_registry_sha256=freeze['seed_registry_sha256'],execution_freeze_sha256=b.sha(b.TECH/'execution_freeze.json'),B2A_execution_freeze_sha256=b.sha(c.OUT/'execution_freeze.json'),raw_data_hash_tree=subtree('raw_evaluation/'),reference_hash_tree=subtree('reference/'),primary_bootstrap_hash_tree=subtree('primary_bootstrap/'),descriptive_bootstrap_hash_tree=subtree('descriptive_bootstrap/'),aggregate_sha256=leaves['aggregate_results.csv']['sha256'],primary_contrasts_sha256=leaves['primary_contrasts.csv']['sha256'],figure_data_sha256=leaves['figure_data.csv']['sha256'],coordinate_data_sha256=leaves['coordinate_data.npz']['sha256'],all_artifact_hash_tree=tree,files=leaves,file_count=len(leaves),reference_trajectories=2500,evaluation_trajectories=5000,condition_SNR_cells=35,raw_logical_rows=1470000,trajectory_logical_rows=245000,CSV_checks=checks,primary_bootstrap_replicates=100000,descriptive_bootstrap_replicates=10000,outcomes_viewed=False,scientific_figures_viewed=False,scientific_figures_created=False,created_utc=b.timestamp(),unseal_requires=b.config()['MFA_unseal_requirements'],mentor_release_required=True,PAGE_BUDGET_DEBT='+5 review pages;untouched')
 b.write(ROOT/'outputs/sp_u12b2b_sealed_manifest.json',manifest)
 b.emit(stage='seal',status='PASS',files=len(leaves),cells_completed=35,hash=tree,outcomes_viewed=False)

def finalize():
 b.check_freeze();pres=b.read(ROOT/'outputs/sp_u12b2b_preservation.json');b.need(pres['PASS'],'FINAL_PRESERVATION')
 manifest=b.read(ROOT/'outputs/sp_u12b2b_sealed_manifest.json');audit=b.read(ROOT/'outputs/sp_u12b2b_integrity_audit.json');boot=b.read(ROOT/'outputs/sp_u12b2b_bootstrap_audit.json');tests=b.read(b.TECH/'pytest_final.json')
 b.need(audit['PASS'] and boot['PASS'] and tests['exit_code']==0 and tests['failed']==tests['skipped']==0,'FINAL_GATE')
 b.need(manifest['condition_SNR_cells']==35 and manifest['evaluation_trajectories']==5000 and manifest['reference_trajectories']==2500,'FINAL_CARDINALITY')
 for path,info in manifest['files'].items():b.need(b.sha(b.SEAL/path)==info['sha256'],'FINAL_SEAL_HASH')
 certificate='# SP-U12B2B seal certificate\n\nB2_SEALED_RUN_COMPLETE. No human/model outcome viewing.\n\n'
 for key in ('run_id','protocol_sha256','config_sha256','seed_registry_sha256','execution_freeze_sha256','raw_data_hash_tree','reference_hash_tree','aggregate_sha256','primary_contrasts_sha256','primary_bootstrap_hash_tree','descriptive_bootstrap_hash_tree','figure_data_sha256','coordinate_data_sha256','all_artifact_hash_tree','created_utc','condition_SNR_cells','reference_trajectories','evaluation_trajectories','file_count','outcomes_viewed','scientific_figures_viewed'):certificate+=key+': '+str(manifest[key])+'\n\n'
 certificate+='Sealed outcomes stay closed until all five MFA prerequisites and mentor release. No B3/B4/MFA/manuscript/Git action.\n'
 (ROOT/'docs/sp_u12b2b_seal_certificate.md').write_text(certificate,encoding='utf8')
 log='# SP-U12B2B execution log\n\nScientific methods, B2A registry/operators and original sources reused unchanged.\n\nReference2500 precedes evaluation5000. Seven paired SNR;seven estimated-channel paths;PerfectCSI BER-only.\n\nPrimary100000resamples/contrast over35family;descriptive10000resamples/cell over245cells;independent frozen stream. All outcome files are sealed. No results interpreted.\n\n'
 log+='Preflight tests: '+json.dumps(b.read(b.TECH/'pytest_preflight.json'))+'\n\nFinal integrity tests: '+json.dumps(tests)+'\n\nPreservation: '+json.dumps(pres)+'\n\nResources: '+json.dumps(b.read(ROOT/'outputs/sp_u12b2b_resource_log.json'))+'\n\n'
 log+='Protocol/config/registry/execution/data/aggregate/bootstrap/figure hashes are in the seal certificate and manifest. PAGE_BUDGET_DEBT+5reviewpages recorded only. STOP and await mentor review.\n'
 (ROOT/'docs/sp_u12b2b_execution_log.md').write_text(log,encoding='utf8')
 b.write(b.TECH/'decision.json',dict(verdict='B2_SEALED_RUN_COMPLETE',sealed_manifest_sha256=b.sha(ROOT/'outputs/sp_u12b2b_sealed_manifest.json'),seal_certificate_sha256=b.sha(ROOT/'docs/sp_u12b2b_seal_certificate.md'),reference_trajectories=2500,evaluation_trajectories=5000,condition_SNR_cells=35,outcomes_viewed=False,scientific_figures_viewed=False,production_semantic_drift=0,old_artifact_drift=0,manuscript_drift=0,created_utc=b.timestamp()))
 b.emit(stage='finalize',verdict='B2_SEALED_RUN_COMPLETE',reference_count=2500,evaluation_count=5000,cells_completed=35,outcomes_viewed=False)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','freeze','run','finalize']);stage=parser.parse_args().stage
 try:globals()[stage]()
 except BaseException as exc:
  b.TECH.mkdir(exist_ok=True)
  # No repr of arbitrary exception (which can embed scientific values), no console traceback.
  code=str(exc) if isinstance(exc,b.IntegrityError) else 'UNEXPECTED_TECHNICAL_EXCEPTION'
  code=code if code.isupper() and all(ch.isupper() or ch=='_' for ch in code) else 'TECHNICAL_GATE_FAILURE'
  record=dict(verdict='B2_SEALED_RUN_TECHNICAL_FAILURE',stage=stage,error_type=type(exc).__name__,failure_code=code,outcomes_viewed=False,scientific_figures_viewed=False,created_utc=b.timestamp())
  b.write(b.TECH/'technical_failure.json',record)
  if isinstance(exc,b.IntegrityError):b.write(b.TECH/'PERMANENT_STOP.json',record)
  # Private diagnostic bytes remain closed; only type/code may be disclosed.
  b.SEAL.mkdir(exist_ok=True)
  with b.phase('SEAL_CHECK'):b.atomic(b.SEAL/'failure_private.txt',traceback.format_exc().encode('utf8'))
  b.emit(stage=stage,verdict=record['verdict'],error_type=record['error_type'],failure_code=code,outcomes_viewed=False);raise SystemExit(1)
if __name__=='__main__':main()
