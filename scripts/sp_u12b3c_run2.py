"""SP-U12B3B: prepare/freeze, sealed run, integrity, bootstrap and certificate."""
from pathlib import Path
import sys,os,json,time,concurrent.futures,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u12b3c_run2_common as b
import numpy as np
SOURCES=('scripts/sp_u12b3b_common.py','scripts/run_sp_u12b3b.py','scripts/sp_u12b3b_preserve.py','scripts/test_sp_u12b3b.py','tests/test_sp_u12b3b.py','tests/test_sp_u12b3b_integrity.py')
TASK=Path('C:/Users/yuhua/.codex/attachments/99624850-95a1-4837-a3c2-c22bd5ff643d/已粘贴的文本.txt')
def b2seal(stage,known=None):
 b.need(b.sha(b.MANIFEST)==b.MANIFEST_SHA and b.sha(b.CERT)==b.CERT_SHA,'B2_METADATA_HASH')
 manifest=b.read(b.MANIFEST);leaves={};tick=time.monotonic()
 for name,rec in manifest['files'].items():
  p=b.B2/name;digest=known.get(p.relative_to(ROOT).as_posix()) if known else b.sha(p)
  b.need(digest==rec['sha256'] and p.stat().st_size==rec['bytes'],'B2_LEAF_HASH');leaves[name]=dict(sha256=digest,bytes=p.stat().st_size)
  if time.monotonic()-tick>25:b.emit(stage='B2_seal_check',files=len(leaves));tick=time.monotonic()
 b.need(set(leaves)=={p.relative_to(b.B2).as_posix() for p in b.B2.rglob('*') if p.is_file()},'B2_LEAF_SET')
 raw={n:r for n,r in leaves.items() if n.startswith('raw_evaluation/')}
 b.need(b.tree(leaves)==manifest['all_artifact_hash_tree'] and b.tree(raw)==manifest['raw_data_hash_tree'],'B2_TREE_HASH')
 audit=dict(PASS=True,stage=stage,files=len(leaves),manifest_sha256=b.sha(b.MANIFEST),certificate_sha256=b.sha(b.CERT),raw_data_hash_tree=b.tree(raw),all_artifact_hash_tree=b.tree(leaves),outcomes_viewed=False)
 b.write(b.TECH/f'b2_seal_{stage}.json',audit);return audit
def prepare():
 b.TECH.mkdir(exist_ok=True);b.need(not(b.TECH/'execution_freeze.json').exists(),'ALREADY_FROZEN')
 before=b.read(b.TECH/'before.json')['hashes'];b2seal('before',before)
 manifest=b.read(b.MANIFEST)
 for condition in b.CONDITIONS:
  for i in range(1000):
   for prefix in ('H','comm'):
    for suffix in ('npz','json'):b.need(f'cache/{condition}/{prefix}_{i:04d}.{suffix}' in manifest['files'],'FORMAL_CACHE_COVERAGE')
   for suffix in ('npz','json'):b.need(f'raw_evaluation/{condition}/{i:04d}.{suffix}' in manifest['files'],'B2_REPLAY_COVERAGE')
 old=ROOT/'docs/sp_u12b3a_preregistration.md';oldbytes=old.read_bytes();b.need(b.sha(old)==before[old.relative_to(ROOT).as_posix()],'PREREG_BASELINE')
 b.atomic(b.TECH/'preregistration_before.txt',oldbytes)
 correction='When one correction design/output direction is identically zero, the two-coordinate normal matrix is rank deficient. The frozen unregularized Both solver shall reject the singular system. Equivalence is assessed instead at the least-squares prediction/output-set level against the corresponding one-direction fit.'
 addition='\n\n## MENTOR_SPEC_CORRECTION — SP-U12B3B\nThe original literal zero-direction unique-solve invariance criterion is withdrawn and replaced by:\n\n'+correction+'\n\nreason = original test contradicted Proposition 2 / frozen solver semantics\nmethod_change = 0\nB3_ABLATION_PIPELINE_READY = formally accepted by the mentor. This supersedes the earlier pending technical decision; the historical decision and estimator bytes remain unchanged. Formal authorization now covers all five B2 conditions at10/20dB, including descriptive ID/Boundary, using cached inputs only.\n'
 b.atomic(old,oldbytes+addition.encode());b.write(b.TECH/'mentor_spec_correction.json',dict(type='MENTOR_SPEC_CORRECTION',reason='original test contradicted Proposition 2 / frozen solver semantics',method_change=0,criterion=correction,before_sha256=before[old.relative_to(ROOT).as_posix()],after_sha256=b.sha(old),B3_ABLATION_PIPELINE_READY=True,user_task_sha256=b.sha(TASK)))
 oldfreeze=b.read(ROOT/'outputs/sp_u12b3a/execution_freeze.json')
 for p,h in oldfreeze['hashes'].items():b.need(b.sha(ROOT/p)==h,'B3A_ESTIMATOR_DRIFT')
 b.need(b.sha(b.c.OUT/'operators.npz')==oldfreeze['operator_cache_sha256'],'OPERATOR_DRIFT')
 # Allocate bootstrap streams only from seed metadata, never outcome files.
 forbidden=set(b.read(b.c.OUT/'historical_seed_inventory.json')['forbidden_seeds'])
 def seedvalues(o):
  if isinstance(o,dict):
   for k,v in o.items():
    if 'seed' in k.lower() and isinstance(v,int):yield v
    else:yield from seedvalues(v)
  elif isinstance(o,list):
   for v in o:yield from seedvalues(v)
 for p in [ROOT/'outputs/sp_u12b2a_seed_registry.json',b.c.OUT/'development_seed_registry.json',ROOT/'configs/sp_u12b2b_sealed.json']:forbidden.update(seedvalues(b.read(p)))
 chosen=[]
 for i in range(1000):
  v=int.from_bytes(__import__('hashlib').sha256(f'SP_U12B3B_BOOTSTRAP_V1:{i}'.encode()).digest()[:4],'big')&0x7fffffff
  if v not in forbidden and v not in chosen:chosen.append(v)
  if len(chosen)==2:break
 b.need(len(chosen)==2,'SEED_ALLOCATION');registry=dict(primary_master_seed=chosen[0],descriptive_master_seed=chosen[1],derivation='SHA256 domain + first nonoverlapping positive int31;SeedSequence([master,condition ordinal])',condition_order=list(b.CONDITIONS),new_channel_seeds=0,B2_seed_registry_sha256=b.sha(ROOT/'outputs/sp_u12b2a_seed_registry.json'),overlap_count=0,forbidden_seed_count=len(forbidden));b.write(b.TECH/'bootstrap_seed_registry.json',registry)
 cfg=dict(run_id='SP_U12B3B_SEALED_ABLATION_V1',conditions=list(b.CONDITIONS),snrs=list(b.SNRS),arms=list(b.ARMS),deployments=list(b.DEPLOY),trajectory_count_per_condition=1000,frames=10,scored_frames=list(range(4,10)),rank=4,folds=4,window=5,worker_count=2,BLAS_threads=1,expected_reused_H=5000,expected_units=10000,expected_trajectory_rows=60000,expected_frame_rows=360000,expected_primary_rows=12,expected_descriptive_rows=60,primary_replicates=10000,descriptive_replicates=10000,primary_q=.05/24,descriptive_quantiles=[.025,.975],quantile_method='linear',primary_conditions=['Profile','Mobility','Composed'],Both_replay_tolerance=0.0,normal_residual_tolerance=b.c.SOLVE_TOL,operator_sha256=b.sha(b.c.OUT/'operators.npz'),bootstrap_registry_sha256=b.sha(b.TECH/'bootstrap_seed_registry.json'),new_channel_generation=False,BER_analysis=False,scientific_outcomes_viewed=False)
 b.write(b.CFG,cfg)
 b.write(b.TECH/'raw_schema.json',dict(trajectory_primary_key=['condition','snr_db','trajectory_id','arm','deployment'],frame_primary_key=['condition','snr_db','trajectory_id','arm','deployment','frame_id'],axis_order=dict(frame_error=['snr','arm','deployment','scored_frame'],frame_truth=['snr','scored_frame'],coordinates=['snr','arm','scored_frame','P_or_M'],trajectory_error=['snr','arm','deployment'],trajectory_truth=['snr']),logical_field_mapping=dict(error_energy='frame_error or sequential six-frame sum trajectory_error',truth_energy='frame_truth or sequential six-frame sum trajectory_truth',coordinate_p='coordinates[...,0] iff coordinate_present[...,0], otherwise NA',coordinate_m='coordinates[...,1] iff coordinate_present[...,1], otherwise NA',solve_status='int8=1 means PASS;no other scientific solve status accepted'),coordinate_sequence_per_trajectory=6,inactive_numeric_payload=0,inactive_NA_mask=True,condition_order=list(b.CONDITIONS),snr_order=list(b.SNRS),arm_order=list(b.ARMS),deployment_order=list(b.DEPLOY),scored_frame_order=list(range(4,10)),independent_unit='trajectory'))
 protocol=TASK.read_text(encoding='utf-8-sig')+'\n\n## Frozen implementation specification\nNative B2 H layout restored by load_native_h. Frozen link input path and variance() are reused exactly. Cached symbols and unit_noise are checked against B2 receipts. Both full predictions are bitwise replayed against c.infer; its coordinates/frame energies/truth are bitwise replayed against B2 raw. B2 aggregate/interval/bootstrap/figure artifacts may only be hashed opaquely. Six frame energies are accumulated sequentially within each trajectory. Inactive coordinate NA is encoded by an explicit boolean mask and zero numeric payload. All arms/deployments/SNR share paired trajectory bootstrap indices within condition. Primary12 family uses independent master stream from descriptive60 points,10000 draws each, linear quantiles. No global hypothesis or posthoc cell promotion. No scientific plots or BER. Workers=2,BLAS=1. Checkpoint receipts bind immutable execution freeze and committed raw hash; only missing records may run. Correctness failure permanently stops the experiment, sealed, without patch. No B4/MFA/manuscript/commit.\n'
 b.atomic(b.PROTOCOL,protocol.encode());b.emit(stage='prepare',PASS=True,rows=cfg['expected_frame_rows'],outcomes_viewed=False)
def freeze():
 tests=b.read(b.TECH/'pytest_preflight.json');b.need(tests['exit_code']==0 and tests['failed']==0 and tests['skipped']==0,'PREFLIGHT_TESTS')
 paths=list(SOURCES)+[b.CFG.relative_to(ROOT).as_posix(),b.PROTOCOL.relative_to(ROOT).as_posix(),'docs/sp_u12b3a_preregistration.md','outputs/sp_u12b3b/bootstrap_seed_registry.json','outputs/sp_u12b3b/mentor_spec_correction.json','outputs/sp_u12b3b/raw_schema.json']
 paths+=list(b.read(ROOT/'outputs/sp_u12b3a/execution_freeze.json')['hashes'])
 paths+=list(b.read(b.c.OUT/'execution_freeze.json')['hashes'])
 paths+=['outputs/sp_u12b2a/operators.npz','outputs/sp_u12b2a_seed_registry.json','outputs/sp_u12b2b_sealed_manifest.json','docs/sp_u12b2b_seal_certificate.md']
 f=dict(hashes={p:b.sha(ROOT/p) for p in sorted(set(paths))},preflight_sha256=b.sha(b.TECH/'pytest_preflight.json'),runtime=dict(python=sys.version,numpy=np.__version__,worker_count=2,BLAS_threads=1),outcomes_viewed=False)
 b.need(not(b.TECH/'execution_freeze.json').exists(),'FREEZE_EXISTS');b.write(b.TECH/'execution_freeze.json',f);b.emit(stage='freeze',PASS=True,hash=b.sha(b.TECH/'execution_freeze.json'))
def evaluation():
 tick=time.monotonic();last=tick;done=0;resumed=0;peaks={}
 with concurrent.futures.ProcessPoolExecutor(max_workers=2,initializer=b.init_worker) as pool:
  pending={}
  def harvest():
   nonlocal done,last,resumed
   completed,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in completed:
    condition,i=pending.pop(f);v=f.result();b.need(v['condition']==condition and v['trajectory_id']==i,'WORKER_ID');done+=1;resumed+=int(v['resumed']);peaks[str(v['worker_pid'])]=v['peak']
   if time.monotonic()-last>25:b.emit(stage='evaluation',trajectory_count=done,elapsed_seconds=time.monotonic()-tick,worker_count=2);last=time.monotonic()
  for condition in b.CONDITIONS:
   for i in range(1000):
    pending[pool.submit(b.worker,condition,i)]=(condition,i)
    if len(pending)>=4:harvest()
   while pending:harvest()
   b.emit(stage='evaluation',condition=condition,trajectory_count=done,cells_completed=2*(b.CONDITIONS.index(condition)+1),status='COMPLETE')
 b.need(done==5000,'TRAJECTORY_COVERAGE');b.write(b.TECH/'evaluation_receipt.json',dict(PASS=True,completed_H_reused=done,cells_completed=10,new_H_generated=0,resumes=resumed,retries=0,worker_peaks=peaks,elapsed_seconds=time.monotonic()-tick))
def integrity():
 keys=set();tick=time.monotonic();maximum=0.;leaf_count=0
 for condition in b.CONDITIONS:
  for i in range(1000):
   p=b.SEAL/'raw'/condition/f'{i:04d}.npz';proof=b.read(p.with_suffix('.json'));b.need(b.sha(p)==proof['artifact']['sha256'],'RAW_HASH');raw=b.arrays(p);b.validate(raw)
   b.need(proof['condition']==condition and proof['trajectory_id']==i,'RECEIPT_ID');b.need(proof['execution_freeze_sha256']==b.sha(b.TECH/'execution_freeze.json') and proof['operator_sha256']==b.read(b.CFG)['operator_sha256'],'RECEIPT_FROZEN_BINDING')
   b.need(proof['Dense_LR4_coordinates_shared'] and proof['Both_current_predictions_bitwise'] and proof['Both_B2_coordinates_energies_bitwise'],'REPLAY_FLAGS');maximum=max(maximum,proof['normal_residual_max'])
   b.need(proof['unexpected_solve_failures']==0 and proof['nonfinite_outputs']==0,'TECHNICAL_FAILURE')
   for s in b.SNRS:
    for arm in b.ARMS:
     for d in b.DEPLOY:
      key=(condition,s,i,arm,d);b.need(key not in keys,'DUPLICATE_PRIMARY_KEY');keys.add(key)
   leaf_count+=1
   if time.monotonic()-tick>25:b.emit(stage='integrity',trajectory_count=leaf_count);tick=time.monotonic()
 b.need(len(keys)==60000 and len(list((b.SEAL/'raw').rglob('*.npz')))==5000,'RAW_POPULATION')
 audit=dict(PASS=True,condition_SNR_cells=10,trajectories_per_cell=1000,distinct_H_reused=5000,new_H_generated=0,units=10000,trajectory_rows=len(keys),frame_rows=6*len(keys),arms=list(b.ARMS),deployments=list(b.DEPLOY),inactive_coordinates_NA=True,coordinate_sharing=True,no_duplicate_primary_keys=True,finite_energies=True,unexpected_solve_failures=0,nonfinite_outputs=0,normal_residual_max=maximum,normal_residual_tolerance=b.c.SOLVE_TOL,independent_unit='trajectory',outcomes_viewed=False)
 b.write(ROOT/'outputs/sp_u12b3b_integrity_audit.json',audit)
 b.write(ROOT/'outputs/sp_u12b3b_both_replay_audit.json',dict(PASS=True,units=10000,scored_frames=60000,Dense_LR4_full_prediction_replay='bitwise equality to frozen c.infer',B2_raw_replay='bitwise coordinates/frame error/truth energies',input_replay='H/symbols/unitnoise/pilot/Y hashes equal B2',max_technical_difference=0.0,tolerance=0.0,B2_values_disclosed=False,B3_values_disclosed=False))
def compute():
 seeds=b.read(b.TECH/'bootstrap_seed_registry.json');aggregate=[];descriptive=[];contrasts=[];bootproof=[]
 for ordinal,condition in enumerate(b.CONDITIONS):
  E=[];T=[]
  for i in range(1000):
   raw=b.arrays(b.SEAL/'raw'/condition/f'{i:04d}.npz');E.append(raw['trajectory_error']);T.append(raw['trajectory_truth'])
  E=np.stack(E);T=np.stack(T);point=10*np.log10(E.sum(axis=0)/T.sum(axis=0)[:,None,None])
  db,digest=b.bootstrap(E,T,seeds['descriptive_master_seed'],ordinal);rec=b.archive(b.SEAL/'bootstrap'/f'descriptive_{condition}.npz',draws=db);bootproof.append(dict(kind='DESCRIPTIVE_ONLY',condition=condition,replicates=10000,points=12,index_hash=digest,artifact=rec))
  for si,s in enumerate(b.SNRS):
   for ai,arm in enumerate(b.ARMS):
    for di,d in enumerate(b.DEPLOY):
     row=dict(condition=condition,snr_db=s,arm=arm,deployment=d,independent_unit='trajectory',N=1000,pooled_nmse_db=float(point[si,ai,di]));aggregate.append(row)
     lo,hi=np.quantile(db[:,si,ai,di],[.025,.975],method='linear');descriptive.append(dict(**row,lower=float(lo),upper=float(hi),interval_role='DESCRIPTIVE_ONLY',replicates=10000))
  if condition in ('Profile','Mobility','Composed'):
   primary,pdigest=b.bootstrap(E,T,seeds['primary_master_seed'],ordinal);g=np.stack([primary[:,si,ai,1]-primary[:,si,2,1] for si in range(2) for ai in range(2)],axis=1);rec=b.archive(b.SEAL/'bootstrap'/f'primary_{condition}.npz',draws=g);bootproof.append(dict(kind='PRIMARY',condition=condition,replicates=10000,contrasts=4,index_hash=pdigest,artifact=rec))
   for si,s in enumerate(b.SNRS):
    for ai,name in enumerate(('Both_LR4_vs_Ponly_LR4','Both_LR4_vs_Monly_LR4')):
     lo,hi=np.quantile(g[:,2*si+ai],[.05/24,1-.05/24],method='linear');contrasts.append(dict(condition=condition,snr_db=s,comparison=name,estimate_db=float(point[si,ai,1]-point[si,2,1]),lower=float(lo),upper=float(hi),classification=b.classify(lo,hi),family_size=12,replicates=10000,independent_unit='trajectory'))
  b.emit(stage='bootstrap',condition=condition,replicates_completed=10000,replicates_expected=10000,status='COMPLETE')
 b.need(len(aggregate)==len(descriptive)==60 and len(contrasts)==12,'TABLE_CARDINALITY')
 for rows in (aggregate,descriptive,contrasts):
  b.need(all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float))),'TABLE_NONFINITE')
 for name,rows in [('aggregate_results',aggregate),('descriptive_results',descriptive),('primary_contrasts',contrasts),('figure_data',descriptive)]:b.csvwrite(b.SEAL/(name+'.csv'),rows)
 b.write(ROOT/'outputs/sp_u12b3b_bootstrap_audit.json',dict(PASS=True,primary_contrasts=12,primary_replicates_per_contrast=10000,primary_draws=120000,descriptive_points=60,descriptive_replicates_per_point=10000,primary_q=.05/24,descriptive_quantiles=[.025,.975],quantile_method='linear',paired_unit='trajectory',paired_across='arms/deployments/SNR withincondition',seed_registry_sha256=b.sha(b.TECH/'bootstrap_seed_registry.json'),bootstrap_receipts=bootproof,global_hypothesis=False,scientific_values_disclosed=False))
def seal():
 leaves={p.relative_to(b.SEAL).as_posix():dict(sha256=b.sha(p),bytes=p.stat().st_size) for p in sorted(b.SEAL.rglob('*')) if p.is_file()}
 manifest=dict(status='SEALED_COMPLETE_UNVIEWED',run_id=b.read(b.CFG)['run_id'],protocol_sha256=b.sha(b.PROTOCOL),config_sha256=b.sha(b.CFG),execution_freeze_sha256=b.sha(b.TECH/'execution_freeze.json'),bootstrap_seed_registry_sha256=b.sha(b.TECH/'bootstrap_seed_registry.json'),B2_seed_registry_sha256=b.sha(ROOT/'outputs/sp_u12b2a_seed_registry.json'),B2_input_provenance='immutable B2 cache;5000 H and paired symbols/unitnoise;10/20dB;no new H',B2_seal_before=b.read(b.TECH/'b2_seal_before.json'),raw_hash_tree=b.tree({k:v for k,v in leaves.items() if k.startswith('raw/')}),bootstrap_hash_tree=b.tree({k:v for k,v in leaves.items() if k.startswith('bootstrap/')}),all_artifact_hash_tree=b.tree(leaves),aggregate_sha256=leaves['aggregate_results.csv']['sha256'],primary_contrasts_sha256=leaves['primary_contrasts.csv']['sha256'],descriptive_sha256=leaves['descriptive_results.csv']['sha256'],figure_data_sha256=leaves['figure_data.csv']['sha256'],files=leaves,outcomes_viewed=False,scientific_figures_viewed=False)
 b.write(ROOT/'outputs/sp_u12b3b_manifest.json',manifest)
def run():
 b.check_freeze();b.firewall();tick=time.monotonic();b.emit(stage='run',status='START',worker_count=2,outcomes_viewed=False)
 with b.phase('RAW'):evaluation()
 b.check_freeze()
 with b.phase('INTEGRITY'):integrity()
 b.check_freeze()
 with b.phase('COMPUTE'):compute()
 with b.phase('SEAL_CHECK'):seal()
 e=b.read(b.TECH/'evaluation_receipt.json');b.write(ROOT/'outputs/sp_u12b3b_resource_log.json',dict(elapsed_seconds=time.monotonic()-tick,parent_pid=os.getpid(),parent_peak=b.peak(),worker_peaks=e['worker_peaks'],CPU_workers=2,BLAS_threads=1,GPU_used=False,new_H_generated=0,retries=e['retries'],resumes=e['resumes'],sealed_disk_bytes=sum(p.stat().st_size for p in b.SEAL.rglob('*') if p.is_file()),runtime_claim=False,scope='execution resources only;not B4',outcomes_viewed=False))
 b.emit(stage='run',status='SEALED_COMPUTATION_COMPLETE',cells_completed=10,trajectory_count=5000,outcomes_viewed=False)
def finalize():
 b.check_freeze();p=b.read(ROOT/'outputs/sp_u12b3b_preservation.json');b.need(p['PASS'],'PRESERVATION_GATE');after=b2seal('after',b.read(b.TECH/'after_hashes.json')['hashes']);before=b.read(b.TECH/'b2_seal_before.json');b.need({k:v for k,v in after.items() if k!='stage'}=={k:v for k,v in before.items() if k!='stage'},'B2_SEAL_CHANGED');b.write(ROOT/'outputs/sp_u12b3b_b2_seal_audit.json',dict(PASS=True,before=before,after=after,B2_artifact_drift=0,B2_outcomes_viewed=False))
 tests=b.read(b.TECH/'pytest_final.json');b.need(tests['exit_code']==0 and tests['failed']==0 and tests['skipped']==0,'FINAL_TEST_GATE')
 m=b.read(ROOT/'outputs/sp_u12b3b_manifest.json')
 for n,r in m['files'].items():b.need(b.sha(b.SEAL/n)==r['sha256'],'FINAL_SEAL_HASH')
 m['B2_seal_after']=after;b.write(ROOT/'outputs/sp_u12b3b_manifest.json',m)
 certificate='# SP-U12B3B sealed certificate\n\nVerdict: B3_SEALED_RUN_COMPLETE\n\nProtocol SHA256: '+m['protocol_sha256']+'\nConfig SHA256: '+m['config_sha256']+'\nExecution freeze SHA256: '+m['execution_freeze_sha256']+'\nB2 provenance: '+m['B2_input_provenance']+'\nB2 raw/all trees before and after: '+before['raw_data_hash_tree']+' / '+before['all_artifact_hash_tree']+'\nB2 manifest/certificate SHA256 unchanged: '+before['manifest_sha256']+' / '+before['certificate_sha256']+'\nB3 raw tree: '+m['raw_hash_tree']+'\nB3 aggregate: '+m['aggregate_sha256']+'\nB3 primary contrasts: '+m['primary_contrasts_sha256']+'\nB3 bootstrap tree: '+m['bootstrap_hash_tree']+'\nB3 figure data: '+m['figure_data_sha256']+'\nB3 all tree: '+m['all_artifact_hash_tree']+'\nManifest SHA256: '+b.sha(ROOT/'outputs/sp_u12b3b_manifest.json')+'\n\nOutcomes viewed = false. Scientific figures viewed = false. All10cells/5000H/10000units/60000trajectoryrows/360000framerows complete. Primary12 x10000 paired draws, descriptive60 x10000 complete. Both bitwise replay PASS. No new H, no BER family, no global hypothesis. Mentor correction changes criterion only, method_change=0. Historical preregistration exact bytes archived; only authorized amended preregistration changed. Production/manuscript/scientific artifact drift=0. STOP for mentor review.\n'
 b.atomic(ROOT/'docs/sp_u12b3b_seal_certificate.md',certificate.encode());b.atomic(ROOT/'docs/sp_u12b3b_execution_log.md',('# SP-U12B3B execution log\n\n'+certificate+'\nPreflight actual pytest: '+json.dumps(b.read(b.TECH/'pytest_preflight.json'))+'\nFinal actual pytest: '+json.dumps(tests)+'\nResource log: outputs/sp_u12b3b_resource_log.json\n').encode());b.write(b.TECH/'decision.json',dict(verdict='B3_SEALED_RUN_COMPLETE',outcomes_viewed=False,manifest_sha256=b.sha(ROOT/'outputs/sp_u12b3b_manifest.json'),STOP=True));b.emit(stage='finalize',verdict='B3_SEALED_RUN_COMPLETE',PASS=True,outcomes_viewed=False)
def main():
 mode=sys.argv[1]
 try:{'prepare':prepare,'freeze':freeze,'run':run,'finalize':finalize}[mode]()
 except Exception as exc:
  try:b.write(b.TECH/f'parent_exception_{os.getpid()}.json',b.technical_exception_record(exc))
  except OSError:pass
  # No arbitrary message, traceback, arrays or outcome values reach console.
  code=str(exc) if isinstance(exc,b.TechnicalFailure) and str(exc).replace('_','').isalnum() else 'SANITIZED_TECHNICAL_FAILURE'
  if (b.TECH/'execution_freeze.json').exists():b.write(b.TECH/'PERMANENT_STOP.json',dict(verdict='B3_FORMAL_TECHNICAL_FAILURE',stage=mode,error_type=type(exc).__name__,failure_code=code,outcomes_viewed=False))
  b.emit(stage=mode,verdict='B3_FORMAL_TECHNICAL_FAILURE',error_type=type(exc).__name__,failure_code=code);raise SystemExit(1)
if __name__=='__main__':
 from scripts.sp_u12b3c_authorization import require_formal_authorization
 require_formal_authorization(sys.argv[1])
 main()
