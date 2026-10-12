"""B2A technical gate. No CLI command launches formal B2 or its bootstrap."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,time,json,csv,hashlib,argparse,subprocess,re,ast
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import sp_u12b2a_common as c

def historical_seed_inventory():
 """Seed-only lexical/CSV/AST projection, never convert outcome numerics."""
 from scripts.sp_u7a_metadata import project_json,seed_key,source_seeds
 paths=[];values=set();tick=time.monotonic()
 for folder in ('outputs','configs','scripts','tests'):
  for p in (ROOT/folder).rglob('*'):
   if not p.is_file() or '__pycache__' in p.parts or any(x.startswith(('sp_u12b2a','run_sp_u12b2a','test_sp_u12b2a')) for x in p.relative_to(ROOT).parts):continue
   selected=set()
   if p.suffix=='.csv' and any(x in p.name.lower() for x in ('seed','manifest')):
    with p.open(encoding='utf-8-sig',newline='') as f:
     rr=csv.DictReader(f);keys=[k for k in rr.fieldnames or () if seed_key(k)]
     for row in rr:
      for k in keys:
       if re.fullmatch(r'\d+',row[k] or ''):selected.add(int(row[k]))
   elif p.suffix in ('.json','.jsonl'):
    selected=project_json(p.read_text(encoding='utf-8-sig'))['seeds']
   elif p.suffix=='.py':selected=source_seeds(p.read_text(encoding='utf-8-sig'))
   elif p.suffix in ('.yaml','.yml'):
    text=p.read_text(encoding='utf-8-sig')
    for k,v in re.findall(r'^\s*([\w]+)\s*:\s*(\d+)\s*$',text,re.M):
     if seed_key(k):selected.add(int(v))
   if selected:values|=selected;paths.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=c.sha(p),seed_values=len(selected)))
   if time.monotonic()-tick>30:print({'seed_metadata_sources':len(paths),'historical_seed_values':len(values)},flush=True);tick=time.monotonic()
 # Reserve this task's synthetic unit-test RNG literals too, before formal allocation.
 for n in c.SOURCE_FILES:
  p=ROOT/n
  if p.exists():
   selected=source_seeds(p.read_text(encoding='utf-8-sig'));values|=selected
   if selected:paths.append(dict(path=n,sha256=c.sha(p),seed_values=len(selected)))
 values.update(range(4420001,4420031))  # reserve all30 unit-test communication seeds
 # Include adjacent explicit CUDA-generator states conservatively as well as allocated role seeds.
 forbidden=values|{v+1 for v in values if v<2**32-1}|{v-1 for v in values if v>0}
 c.write(c.OUT/'historical_seed_inventory.json',dict(projection='Only seed numeric fields, CSV seed columns and literal RNG/seed AST. Outcome tokens not converted.',individual_seed_count=len(values),conservative_sdk_adjacent_union_count=len(forbidden),sources=paths,forbidden_seeds=sorted(forbidden)))
 return forbidden
def allocate(namespace,n_eval,n_ref,forbidden,used,bootstrap=False):
 rows=[];rejects=0
 for condition in c.CONDITIONS:
  for group,n in (('MATCHED_REF',n_ref),('EVAL',n_eval)):
   for i in range(n):
    keys=[(-1,r) for r in c.ROLES[:3]]+([(j,r) for j in range(10) for r in c.ROLES[3:]] if group=='EVAL' else [])
    for j,r in keys:
     counter=0
     while True:
      seed=int.from_bytes(hashlib.sha256(f'{namespace}|{condition}|{group}|{i}|{j}|{r}|{counter}'.encode('ascii')).digest()[:4],'big')
      if 0<seed<2**32-1 and all(v not in forbidden and v not in used for v in (seed-1,seed,seed+1)):break
      counter+=1;rejects+=1
     rows.append(dict(namespace=namespace,condition=condition,group=group,trajectory_id=i,frame_id=j,role=r,seed=seed,retry_counter=counter));used.add(seed)
 if bootstrap:
  counter=0
  while True:
   seed=int.from_bytes(hashlib.sha256(f'{namespace}|BOOTSTRAP|{counter}'.encode()).digest()[:4],'big')
   if 0<seed<2**32-1 and all(v not in forbidden and v not in used for v in (seed-1,seed,seed+1)):break
   counter+=1;rejects+=1
  rows.append(dict(namespace=namespace,condition=None,group='BOOTSTRAP',trajectory_id=-1,frame_id=-1,role='bootstrap_master_seed',seed=seed,retry_counter=counter));used.add(seed)
 return dict(namespace=namespace,rows=rows,rejection_count=rejects,allocation='SHA256(namespace|condition|group|trajectory|frame|role|counter) uint32; rejection only for pre-data seed collisions',immutable=True,scientific_generation_started=False)
def seed_index(reg):return {(r['condition'],r['group'],r['trajectory_id'],r['frame_id'],r['role']):r['seed'] for r in reg['rows'] if r['group']!='BOOTSTRAP'}
def consumed(index,condition,group,i,j=0):return {r:index[condition,group,i,-1 if r in c.ROLES[:3] else j,r] for r in c.ROLES if (condition,group,i,-1 if r in c.ROLES[:3] else j,r) in index}
def prepare():
 if Path(sys.prefix).resolve()!=(ROOT/'.venv-cdl').resolve():raise ValueError('Frozen CDL environment required')
 if not (c.OUT/'before.json').exists():raise ValueError('Preservation baseline required')
 c.OUT.mkdir(exist_ok=True)
 forbidden=historical_seed_inventory();used=set()
 dev=allocate('SP_U12B2A_DEVELOPMENT_ONLY_V1',3,3,forbidden,used)
 formal=allocate('SP_U12B2_FULL_SNR_SEALED_V1',1000,500,forbidden,used,True)
 c.write(c.OUT/'development_seed_registry.json',dev);c.write(ROOT/'outputs/sp_u12b2a_seed_registry.json',formal)
 dv={r['seed'] for r in dev['rows']};fv={r['seed'] for r in formal['rows']}
 if len(dv)!=540 or len(fv)!=172501 or dv&fv or (dv|fv)&forbidden:raise ValueError('Formal seed overlap/cardinality')
 c.write(ROOT/'outputs/sp_u12b2a_seed_overlap_audit.json',dict(PASS=True,historical_conservative_union_size=len(forbidden),development_seed_values=540,formal_seed_values=172501,formal_channel_reference_values=7500,formal_channel_evaluation_values=15000,formal_communication_values=150000,bootstrap_master_values=1,historical_overlap=0,development_formal_overlap=0,individual_seed_duplicates=0,adjacent_sdk_state_collisions=0,reference_evaluation_overlap=0,scope='all individual roles incl reserved/invalidated historical seeds, plus conservative adjacent SDK states',historical_inventory_sha256=c.sha(c.OUT/'historical_seed_inventory.json'),registry_sha256=c.sha(ROOT/'outputs/sp_u12b2a_seed_registry.json')))
 cfg=dict(task='SP-U12B2A',authority='IMPLEMENTATION_VALIDATION_ONLY',formal_generation_authorized=False,snr_db=list(c.SNRS),source_receive_power=c.POWER,noise_variances={str(s):c.variance(s) for s in c.SNRS},noise_formula='P_rx_ref * 10^(-snr_db/10); evaluated as the frozen P_rx_ref / 10^(snr_db/10)',conditions=c.CONDITIONS,methods=list(c.METHODS),perfect_CSI='Secondary BER reference only; excluded from NMSE estimator inventory',preparation={m:'outputs/'+p for m,p in c.STATS.items()},preparation_trajectories_per_origin=500,rank=4,window_frames=5,folds=4,pilot_dimension=64,data_dimension=3968,fold_input=48,fold_heldout=16,sample_rate_hz=1920000,nfft=64,delta_f_hz=30000,n_eval_per_condition=1000,n_reference_per_condition=500,frames_per_trajectory=10,scored_frames=list(range(4,10)),cross_snr_pairing='Same clean H, same unit CN(0,I) noise and communication bits; only sigma scales. SNR points are dependent.',reference_pairing='One independent clean reference corpus per condition shared across all7SNR',bootstrap=dict(resamples=100000,family=35,two_sided=True,confidence=.95,quantile_method='linear',unit='paired trajectory',chunk_resamples=1000,condition_substreams='SeedSequence([bootstrap_master_seed,condition_ordinal]); same indices across methods/SNR',store_index_tables=False),primary_contrasts=c.planned_contrasts(),roundoff_relative_tolerance=c.TOL,roundoff_source='Existing SP-U4A fixed 1e-10 independent-complex128 effective-product gate; no threshold selected from outcomes',solve_relative_tolerance=c.SOLVE_TOL,bit_count_gate='exact integer equality',smoke=dict(eval_per_condition=3,ref_per_condition=3,snr_db=list(c.EDGE),purpose='technical finite/schema/solve checks only'),engineering_threads=1,PAGE_BUDGET_DEBT='+5 review pages relative to prior30-page build; deferred to B7/B8')
 if cfg['sample_rate_hz']!=cfg['nfft']*cfg['delta_f_hz']:raise ValueError('Sampling-rate configuration failure')
 c.write(c.CONFIG,cfg)
 schema=dict(raw_csv_fields=list(c.RAW_FIELDS),raw_null_semantics='alpha/beta null for LS-LIN/Source/Matched; fitted/restricted coordinates for the corresponding combination methods; solve_status=PASS for Dense/LR4, NOT_APPLICABLE otherwise',aggregate_fields=['condition','snr_db','estimator','pooled_nmse_db','ber','comparison','estimate_db','lower','upper','classification'],raw_npz=dict(frame_error='[7SNR,7methods,6frames], float64',frame_truth='[7SNR,6frames], float64',frame_bit_errors='[7SNR,7methods,6frames], int64',frame_bit_total='[7SNR,6frames], int64',coordinates='[7SNR,6frames,2], float64',perfect_frame_bit_errors='[7SNR,6frames], int64'),coordinate_handle='One immutable raw archive/trajectory, coordinates indexed by fixed SNR/frame; pure coordinate metadata optional',trajectory_aggregation='Sequential sum of6scored frame energies/counts; pool numerator/denominator across trajectories, then10log10; never average dB',statistical_unit='trajectory; overlapping windows, SNR points and frames are not independent evaluation N',gain_classification=dict(POSITIVE='LB>0',NEGATIVE='UB<0',INDETERMINATE='otherwise incl zero endpoint'),compression_classification='Sign/interval/magnitude only; POSITIVE means compression worse, not good',primary_family=c.planned_contrasts(),primary_family_size=35,ID_Boundary='All7SNR reported, descriptive/safety-boundary; excluded from primary family',figure_schema=dict(main_nmse_methods=['LS-LIN','Source','Convex','Dense','LR4','Matched'],main_panels=list(c.CONDITIONS),main_contrasts=['Composed convex_gain','Composed compression_penalty'],supplement_methods=list(c.METHODS),deletion_by_outcome_forbidden=True),bootstrap_generated=False)
 c.write(ROOT/'outputs/sp_u12b2a_output_schema.json',schema)
 c.FORMAL.mkdir()
 for name,obj in [('config.json',cfg),('seed_registry.json',formal),('schemas.json',schema)]:c.write(c.FORMAL/name,obj)
 for folder in ('matched_reference','evaluation','raw','analysis'): (c.FORMAL/folder).mkdir()
 c.write(c.FORMAL/'manifest.json',dict(namespace=formal['namespace'],status='SEALED_PLANNED_ONLY',N_ref=2500,N_eval=5000,clean_reference_generated=0,clean_evaluation_generated=0,raw_outcomes_generated=0,analysis_released=False,registry_sha256=c.sha(ROOT/'outputs/sp_u12b2a_seed_registry.json')))
 text='''# SP-U12B2A preregistration\n\nIMPLEMENTATION / VALIDATION ONLY. No formalB2 generation or aggregate access in this task.\n\nFrozen method: Source anchor and Source centering;500clean trajectories at each of3preparationorigins;rank4;current-inclusive5frame window;4fixed48→16folds;owner pilot geometry/order;real unconstrained solve;fullSource/fullcrosspilot kernels;correction-only compression. Only sigma² changes.\n\nSNR0:5:30 and P_rx_ref9.941134822523972. Frozen evaluation: ID=C30/100,Profile=B30/100,Mobility=C120/100,Composed=B120/100,Boundary=C30/200. For each condition500independent reference trajectories precede1000fresh evaluation trajectories;10continuousframes,score4–9. Register seeds now;do not generate.\n\nCRN: one clean channel and one standardized complex Gaussian realization per condition/trajectory/frame, plus same communication symbols, across all7SNR. Only sqrt(sigma²) changes. Noise marginals remain CN(0,sigma²I);differentSNR points are paired and dependent,not independentN. One clean reference corpus is reused at all7SNR.\n\nSevenNMSE methods:LS-LIN,Source,Hard,Convex,Dense,LR4,Matched. PerfectCSI isBER-only. Same receiver and inputs. Matched-reference statistics never enter deployment boundary.\n\n35contrasts:21Profile/Mobility/Composed Source-minus-LR4;7Composed Convex-minus-LR4;7Composed LR4-minus-Dense.100000pairedtrajectory bootstrap,35two-sidedBonferroni95%intervals,q=.05/70,linearquantiles. Same trajectory resamples across methods/SNR within condition,independent condition substreams. No new family or global weak success. ID/Boundary descriptive curves complete. Gainclassification LB>0POSITIVE,UB<0NEGATIVE,elseINDETERMINATE;zero-containingCI isnot equivalence. Compressionpositive means worse.\n\n10/20anchor audit: same old T20B cachedH andcommunication,seeds;all5000trajectories,bothSNR,all6scoreframes,sevenestimates,secondaryPerfectBER. Oldoutcomes are read only for equality checks,never method/grid/thresholdselection. Existingrelative1e-10roundoffgate,linear/normalresidual1e-11,integerbitcounts exact. Compare effective rank4 operators to avoid arbitrary SVD gauge;also report rawfactor agreement. Legacy H physicalstrides restored by frozen resume loader.\n\nSmoke:3eval+3reference percondition at0/15/30,independentdevelopmentnamespace. Onlyexecution/finite/solve/pairedinput/schema checks are disclosed;no performance conclusions. No scientific bootstrap.\n\nStop: anchor mismatch,operator/rank failure,nonfinite/solve failure,seed overlap,leakage,shape/order failure,or formalaccidentalunsealing => B2_PIPELINE_REVISION_REQUIRED,permanentSTOP. No patch-and-formalrun.\n\nB2 generation requires separate future human authorization. Aggregate results stay sealed until B3/B4/B5 dependencies and MFA finalselectedconfiguration retrained on full500/origin and frozen;the existing mentor-amended PLAN_B protocol remains authority. No B3/B4/B5/B6 in B2A.\n\nEngineering resource estimates include cache,operators,bootstrap scratch and approximate fullrun time;they are not B4 runtime findings.\n\nPAGE_BUDGET_DEBT=+5reviewpages;no manuscript changes or theory/Algorithm removal.\n'''
 (ROOT/'docs/sp_u12b2a_preregistration.md').write_text(text,encoding='utf8')
 (ROOT/'docs/sp_u12b2a_figure_spec.md').write_text('# Frozen B2 publication figure schema\n\nFive-condition NMSE-vs-SNR panels at0:5:30. Main:LS-LIN,Source,Convex,Dense,LR4,Matched. Hard appears with allsevenmethods in supplementarycurves. Composed design/compression contrast curves retain frozen signs and simultaneousintervals. ID/Boundary entire curves remain visible anddescriptive. No removal of an unfavorablemethod/cell. No scientific figure created in B2A.\n',encoding='utf8')
 print({'preparation':'PASS','formal_seeds':172501,'development_seeds':540,'formal_H_generated':0},flush=True)
def freeze():
 from scripts.run_formal_test import FormalCDL
 # Metadata-only constructor audit; no scientific H calls.
 runtime={condition:FormalCDL(name,{}).provenance() for condition,name in c.CONDITIONS.items()}
 from sionna.phy.channel.tr38901 import models
 for condition,metadata in runtime.items():
  profile='CDL-'+c.CONDITIONS[condition].split('-')[1]
  parameter=Path(models.parameter_file(profile+'.json','19.2'))
  metadata['profile']=profile
  metadata['condition']=c.CONDITIONS[condition]
  metadata['source_hashes'][str(parameter)]=c.sha(parameter)
 paths=list(c.SOURCE_FILES)+['configs/sp_u12b2a_snr_sweep.yaml','outputs/sp_u12b2a_seed_registry.json','outputs/sp_u12b2a_output_schema.json','outputs/sp_u12b2a_seed_overlap_audit.json','outputs/sp_u12b2a/development_seed_registry.json','docs/sp_u12b2a_preregistration.md','docs/sp_u12b2a_figure_spec.md']
 paths += ['outputs/'+n for n in (*c.STATS.values(),*c.OLD_OBJECTS,*c.MATCHED)]
 paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/'src').glob('*.py')]
 paths += ['scripts/run_formal_test.py','scripts/resume_formal_test.py','scripts/sp_u7a_metadata.py']
 paths += ['outputs/sp_u12b2a/run_pytest.py','outputs/sp_u12b2a/pytest_preflight.json','outputs/sp_u12b2a/pytest_preflight.txt','outputs/sp_u12b2a/pytest_preflight.xml']
 suite=c.read(c.OUT/'pytest_preflight.json')
 if suite['exit_code']!=0 or suite['failed'] or suite['skipped']:raise ValueError('Preflight pytest not clean')
 hashes={p:c.sha(ROOT/p) for p in sorted(set(paths))}
 for metadata in runtime.values():hashes.update(metadata['source_hashes'])
 c.write(c.OUT/'execution_freeze.json',dict(phase='BEFORE_OPERATOR_AND_SCIENTIFIC_VALIDATION',hashes=hashes,runtime_configurations=runtime,generator_calls=0,pytest=suite,formal_generation_authorized=False,scientific_validation_started=False))
 print({'execution_bytes_frozen':True,'formal_H_generated':0},flush=True)
def operators():
 c.require_freeze();c.firewall()
 with c.phase('OPERATORS'):
  from src.frozen_affine_validation import VARIANCES
  for s in (10,20):
   if c.variance(s)!=VARIANCES[s]:raise ValueError('10/20 noise anchor mismatch')
  stats={m:c.statistics(ROOT/'outputs'/n) for m,n in c.STATS.items()};tick=time.monotonic();a,rows,fold=c.instantiate(stats)
  old=c.historical_operators();comparisons=[]
  for s in (10,20):
   for k in ('Wsource','WP','WM','KSource','KP','KM','DeltaWP','DeltaWM','DeltaKP','DeltaKM'):
    error=c.rel(a[f'{k}_{s}'],old[f'{k}_{s}']);comparisons.append(dict(snr_db=s,object=k,relative_error=error,status='PASS' if error<=c.TOL else 'FAIL'))
   for m in ('P','M'):
    error=c.rel(a[f'A_{m}_{s}']@a[f'Q_{m}_{s}'],old[f'A_{m}_{s}']@old[f'Q_{m}_{s}']);comparisons.append(dict(snr_db=s,object=f'factor-effective-{m}',relative_error=error,raw_A_relative_error=c.rel(a[f'A_{m}_{s}'],old[f'A_{m}_{s}']),raw_Q_relative_error=c.rel(a[f'Q_{m}_{s}'],old[f'Q_{m}_{s}']),status='PASS' if error<=c.TOL else 'FAIL'))
  if any(r['status']!='PASS' for r in comparisons):raise ValueError('ANCHOR_REPLAY_FAILURE operators')
  record=c.immutable_archive(c.OUT/'operators.npz',**a)
  c.csvwrite(ROOT/'outputs/sp_u12b2a_noise_operator_audit.csv',rows)
  from scripts.sp_u4a_common import peak_memory
  c.write(c.OUT/'operator_anchor.json',dict(PASS=True,comparisons=comparisons,noise_anchors_exact=True,fold=fold,operator_archive=record,operator_seconds=time.monotonic()-tick,relative_tolerance=c.TOL,all_statistics_unchanged=True,rank_selection=False,worker_pid=os.getpid(),engineering_peak=peak_memory()))
  print({'seven_SNR_operator_gate':'PASS','origin_SNR_rows':21,'operator_cache_bytes':record['bytes']},flush=True)

def anchor():
 c.require_freeze();c.firewall()
 with c.phase('ANCHOR'):
  from scripts.resume_formal_test import load_native_h
  a=c.load_operators();old=c.historical_operators();proof=c.read(c.OUT/'operator_anchor.json');checks=[];start=time.monotonic();last=start
  label={'LS-LIN':'R3 LS+linear','Source':'Source LMMSE','Hard':'Hard','Convex':'Convex','Dense':'Dense Affine','LR4':'Final LR4 Affine','Matched':'Matched LMMSE reference'}
  nmse={(r['environment'],int(r['snr_db']),r['method']):float(r['nmse_db']) for r in c.csvread(ROOT/'outputs/t20b_formal_nmse.csv') if r['method']!='Perfect CSI'}
  ber={(r['environment'],int(r['snr_db']),r['method']):float(r['ber']) for r in c.csvread(ROOT/'outputs/t20b_formal_ber.csv')}
  # Communication cache seed identity is independently rebuilt for first/last trajectory of every condition.
  seedrows=c.csvread(ROOT/'outputs/t20b_formal_seed_manifest.csv');idx={(r['environment'],int(r['trajectory']),int(r['frame']),r['random_role']):int(r['assigned_seed']) for r in seedrows};del seedrows
  with (ROOT/'outputs/t20b_formal_frame_metrics.csv').open(encoding='utf8',newline='') as ff,(ROOT/'outputs/t20b_formal_trajectory_metrics.csv').open(encoding='utf8',newline='') as tf:
   fr=csv.DictReader(ff);tr=csv.DictReader(tf);done=0
   for condition,name in c.CONDITIONS.items():
    st=c.statistics(ROOT/'outputs'/c.MATCHED[list(c.CONDITIONS).index(condition)]);matched={s:st.prepare(c.variance(s)) for s in (10,20)}
    errors=np.zeros((1000,2,7));truth=np.zeros((1000,2));bit=np.zeros((1000,2,7),np.int64);bits=np.zeros((1000,2),np.int64);perf=np.zeros((1000,2),np.int64)
    maximum=dict(prediction=0.,coordinate=0.,frame_energy=0.,trajectory_energy=0.,pooled_nmse_absolute_db=0.,pooled_ber_absolute=0.);seed_matches=0;bit_checks=0;frame_checks=0
    for ii,i in enumerate(range(600,1600)):
     h=load_native_h(ROOT/f'outputs/t20b_formal_data/{name}_{i}.npz')
     with np.load(ROOT/f'outputs/t20b_formal_data/{name}_{i}_communication.npz',allow_pickle=False) as z:comm=list(zip(z['bits'],z['symbols'],z['unit_noise']))
     for trio in comm:
      for v in trio:v.flags.writeable=False
     if i in (600,1599):
      for j in range(10):
       seeds={r:idx[name,i,0 if r in c.ROLES[:3] else j,r] for r in c.ROLES}
       restored=c.communicate(seeds)
       if any(not np.array_equal(x,y) for x,y in zip(comm[j],restored)):raise ValueError('ANCHOR_REPLAY_FAILURE communication-seed identity')
       seed_matches+=1
     raw,_,evidence=c.evaluate(h,comm,a,matched,(10,20),old);ee,qq,bb,tt=c.raw_to_trajectory(raw,(10,20));errors[ii]=ee;truth[ii]=qq;bit[ii]=bb;bits[ii]=tt;perf[ii]=raw['perfect_frame_bit_errors'].sum(axis=1)
     maximum['prediction']=max(maximum['prediction'],evidence['prediction_relative_error_max']);maximum['coordinate']=max(maximum['coordinate'],evidence['coordinate_relative_error_max'])
     expected={}
     for _ in range(96):
      row=next(fr)
      if row['environment']!=name or int(row['trajectory_id'])!=i:raise ValueError('ANCHOR_REPLAY_FAILURE frame identity/order')
      key=(int(row['snr_db']),int(row['frame_id']),row['method'])
      if key in expected:raise ValueError('Duplicate old frame key')
      expected[key]=row
     for si,s in enumerate((10,20)):
      for ji,j in enumerate(range(4,10)):
       for mi,m in enumerate(c.METHODS):
        prior=expected[s,j,label[m]];reference=float(prior['error_energy']);er=abs(raw['frame_error'][si,mi,ji]-reference)/max(reference,np.finfo(float).tiny)
        qr=abs(raw['frame_truth'][si,ji]-float(prior['truth_energy']))/float(prior['truth_energy']);maximum['frame_energy']=max(maximum['frame_energy'],er,qr)
        if er>c.TOL or qr>c.TOL:raise ValueError('ANCHOR_REPLAY_FAILURE per-frame energy')
        if raw['frame_bit_errors'][si,mi,ji]!=int(prior['bit_errors']) or raw['frame_bit_total'][si,ji]!=int(prior['total_bits']):raise ValueError('ANCHOR_REPLAY_FAILURE frame BER integer')
        frame_checks+=1;bit_checks+=1
       prior=expected[s,j,'Perfect CSI']
       if raw['perfect_frame_bit_errors'][si,ji]!=int(prior['bit_errors']):raise ValueError('ANCHOR_REPLAY_FAILURE Perfect CSI BER')
     expected_t={}
     for _ in range(16):
      row=next(tr)
      if row['environment']!=name or int(row['trajectory_id'])!=i:raise ValueError('ANCHOR_REPLAY_FAILURE trajectory identity/order')
      expected_t[int(row['snr_db']),row['method']]=row
     for si,s in enumerate((10,20)):
      for mi,m in enumerate(c.METHODS):
       prior=expected_t[s,label[m]];er=abs(ee[si,mi]-float(prior['error_energy']))/float(prior['error_energy']);maximum['trajectory_energy']=max(maximum['trajectory_energy'],er)
       if er>c.TOL or bb[si,mi]!=int(prior['bit_errors']) or tt[si]!=int(prior['total_bits']):raise ValueError('ANCHOR_REPLAY_FAILURE trajectory metrics')
     done+=1
     if time.monotonic()-last>25:print({'anchor_replayed':done,'total_old_trajectories':5000,'old_metric_values_disclosed':False},flush=True);last=time.monotonic()
    for si,s in enumerate((10,20)):
     q=float(sum(truth[:,si]));bt=int(sum(bits[:,si]))
     for mi,m in enumerate(c.METHODS):
      pooled=float(10*np.log10(float(sum(errors[:,si,mi]))/q));b=float(sum(bit[:,si,mi])/bt)
      nd=abs(pooled-nmse[name,s,label[m]]);bd=abs(b-ber[name,s,label[m]]);maximum['pooled_nmse_absolute_db']=max(maximum['pooled_nmse_absolute_db'],nd);maximum['pooled_ber_absolute']=max(maximum['pooled_ber_absolute'],bd)
      if nd>c.TOL or bd!=0:raise ValueError('ANCHOR_REPLAY_FAILURE pooled NMSE/BER')
     if sum(perf[:,si])/bt!=ber[name,s,'Perfect CSI']:raise ValueError('ANCHOR_REPLAY_FAILURE pooled Perfect BER')
    for object_name,v in maximum.items():checks.append(dict(condition=condition,object=object_name,trajectories=1000,snr_count=2,maximum_difference=float(v),tolerance=0. if 'ber' in object_name else c.TOL,status='PASS'))
    checks.append(dict(condition=condition,object='frame_and_seed_identity',trajectories=1000,snr_count=2,maximum_difference=0.,tolerance=0.,status='PASS'))
    print({'anchor_condition':condition,'PASS':True,'frame_method_checks':frame_checks,'exact_bit_checks':bit_checks,'seed_replayed_frames':seed_matches},flush=True)
   if next(fr,None) is not None or next(tr,None) is not None:raise ValueError('Extra old anchor rows')
  c.csvwrite(ROOT/'outputs/sp_u12b2a_anchor_replay.csv',checks)
  from scripts.sp_u4a_common import peak_memory
  c.write(c.OUT/'anchor_receipt.json',dict(PASS=True,old_trajectories_replayed=5000,old_snr_count=2,seven_frame_estimates_compared=420000,raw_frame_energy_checks=420000,pooled_cells=70,secondary_Perfect_BER_cells=10,communication_seed_frames_rebuilt=100,operator_gate=proof,elapsed_seconds=time.monotonic()-start,old_outcomes_used='Only numerical replay comparisons; no method/grid/threshold selection',formal_B2_H_generated=0,scientific_new_evidence=False,worker_pid=os.getpid(),engineering_peak=peak_memory()))
  print({'anchor_replay':'PASS','old_trajectories':5000,'formal_B2_generated':0},flush=True)

def generator(condition,index,group):
 from scripts.run_formal_test import FormalCDL
 from src.cdl_benchmark import OfficialCDL
 class SweepCDL(FormalCDL):
  def __init__(self):super().__init__(c.CONDITIONS[condition],{})
  def generate(self,i):
   import src.cdl_benchmark as adapter
   old=adapter.seeds;adapter.seeds=lambda trajectory,frame=0:consumed(index,condition,group,trajectory,frame)
   try:return OfficialCDL.generate(self,i)
   finally:adapter.seeds=old
 return SweepCDL()
def build_reference(condition,index,n,save_root=None):
 from src.cdl_benchmark import StreamingStatistics,frames,observation_geometry,array_hash
 g=observation_geometry();stream=StreamingStatistics(g.selected,g.data);adapter=generator(condition,index,'MATCHED_REF');hashes=[];tick=time.monotonic()
 for i in range(n):
  h=adapter.generate(i);hashes.append(array_hash(h));stream.update(frames(h).reshape(10,4096),[i])
  if save_root is not None:
   rec=c.immutable_archive(save_root/f'{i:04d}.npz',H=h)
   c.write(save_root/f'{i:04d}.json',dict(condition=condition,group='MATCHED_REF',trajectory_id=i,seeds=consumed(index,condition,'MATCHED_REF',i),H_sha256=hashes[-1],export_strides_bytes=list(h.strides),archive=rec))
 return stream.freeze(),hashes,time.monotonic()-tick,adapter.provenance()
def smoke():
 c.require_freeze();c.firewall()
 if not c.read(c.OUT/'anchor_receipt.json')['PASS']:raise ValueError('Anchor gate prerequisite')
 with c.phase('SMOKE'):
  from src.cdl_benchmark import array_hash
  reg=c.read(c.OUT/'development_seed_registry.json');index=seed_index(reg)
  if reg['namespace']!='SP_U12B2A_DEVELOPMENT_ONLY_V1':raise ValueError('Development-only namespace')
  a=c.load_operators();records=[];seen=set();reference_seconds=evaluation_seconds=0.
  for condition in c.CONDITIONS:
   st,hashes,seconds,provenance=build_reference(condition,index,3);reference_seconds+=seconds;matched={s:st.prepare(c.variance(s)) for s in c.EDGE}
   if len(set(hashes))!=3 or seen&set(hashes):raise ValueError('Smoke reference H duplication')
   seen.update(hashes);adapter=generator(condition,index,'EVAL')
   # Same constructor/profile/array/time shape; preserve frozen default generator device.
   if adapter.provenance()['normalized_PDP_powers']!=provenance['normalized_PDP_powers']:raise ValueError('Reference/evaluation generator definition drift')
   for i in range(3):
    tick=time.monotonic();h=adapter.generate(i);hh=array_hash(h)
    if hh in seen:raise ValueError('Matched/evaluation leakage')
    seen.add(hh);comm=[c.communicate(consumed(index,condition,'EVAL',i,j)) for j in range(10)]
    raw,rows,evidence=c.evaluate(h,comm,a,matched,c.EDGE);c.validate_raw(raw,c.EDGE)
    for row in rows:
     if set(row)!=set(c.RAW_FIELDS):raise ValueError('Output CSV schema')
     row['condition']=condition;row['trajectory_id']=i
    for j in range(10):
     entries=[r for r in evidence['inputs'] if r['frame_id']==j]
     if len({r['truth_sha256'] for r in entries})!=1 or len({r['unit_noise_sha256'] for r in entries})!=1:raise ValueError('CRN input mismatch')
    # Actual writer exercised, development rows only; no aggregate table, sorting or bootstrap.
    path=c.OUT/'development'/f'{condition}_{i:04d}.npz';record=c.immutable_archive(path,**raw)
    if i==0:c.csvwrite(c.OUT/'development'/f'{condition}_schema_sample.csv',rows)
    evaluation_seconds+=time.monotonic()-tick
    records.append(dict(condition=condition,trajectory_id=i,snr_db=list(c.EDGE),finite_outputs=True,coordinate_solve='PASS',normal_residual_max=evidence['normal_residual_max'],matched_reference_path='PASS',bootstrap_input_schema='PASS',CSV_schema='PASS',CRN_pairing='PASS',dense_lr4_shared_coordinates=True,H_sha256=hh,raw_handle=record,performance_conclusions=False))
   print({'development_smoke_condition':condition,'eval_trajectories':3,'reference_trajectories':3,'finite_solve_schema':'PASS','performance_values_disclosed':False},flush=True)
  from scripts.sp_u4a_common import peak_memory
  c.write(ROOT/'outputs/sp_u12b2a_edge_smoke.json',dict(PASS=True,namespace=reg['namespace'],snrs=list(c.EDGE),evaluation_trajectories=15,matched_reference_trajectories=15,score_frame_method_rows=1890,unique_clean_H=30,records=records,reference_seconds=reference_seconds,evaluation_seconds=evaluation_seconds,bootstrap_generated=False,formal_H_generated=0,formal_outcomes_viewed=False,performance_comparisons_reported=False,worker_pid=os.getpid(),engineering_peak=peak_memory()))

def run_future_sealed_generation(authorization_path=None):
 """Not a B2A CLI action. Future approval required; writes raw, never aggregates."""
 c.require_future_authorization(authorization_path,'sealed-generation');c.firewall()
 with c.phase('SEALED_GENERATION'):
  from src.cdl_benchmark import array_hash,observation_geometry
  # Empty raw structure is a precondition; reruns cannot silently mix runs.
  if any((c.FORMAL/'raw').glob('**/*.npz')):raise RuntimeError('Formal outcomes already exist; no unreviewed resume')
  registry=c.read(c.FORMAL/'seed_registry.json');index=seed_index(registry);a=c.load_operators();references={};seen=set()
  for condition in c.CONDITIONS:
   st,hashes,seconds,provenance=build_reference(condition,index,500,c.FORMAL/'matched_reference'/condition)
   if len(set(hashes))!=500 or seen&set(hashes):raise RuntimeError('Formal reference duplication')
   seen.update(hashes);references[condition]=st
   c.immutable_archive(c.FORMAL/'matched_reference'/condition/'statistics.npz',mean_p=st.mean_p,mean_d=st.mean_d,rpp=st.rpp,rdp=st.rdp,train_trajectory_ids=np.array(st.train_trajectory_ids),samples=np.array(st.samples))
  for condition in c.CONDITIONS:
   adapter=generator(condition,index,'EVAL');matched={s:references[condition].prepare(c.variance(s)) for s in c.SNRS}
   for i in range(1000):
    h=adapter.generate(i);hh=array_hash(h)
    if hh in seen:raise RuntimeError('Formal reference/evaluation leakage or duplicate')
    seen.add(hh);comm=[c.communicate(consumed(index,condition,'EVAL',i,j)) for j in range(10)]
    folder=c.FORMAL/'evaluation'/condition
    hrec=c.immutable_archive(folder/f'H_{i:04d}.npz',H=h)
    crec=c.immutable_archive(folder/f'comm_{i:04d}.npz',bits=np.stack([v[0] for v in comm]),symbols=np.stack([v[1] for v in comm]),unit_noise=np.stack([v[2] for v in comm]))
    raw,rows,evidence=c.evaluate(h,comm,a,matched)
    rec=c.immutable_archive(c.FORMAL/'raw'/condition/f'{i:04d}.npz',**raw)
    for row in rows:row.update(condition=condition,trajectory_id=i)
    csvpath=c.FORMAL/'raw'/condition/f'{i:04d}.csv';c.csvwrite(csvpath,rows)
    c.write(folder/f'proof_{i:04d}.json',dict(condition=condition,trajectory_id=i,H_sha256=hh,export_strides_bytes=list(h.strides),channel_seeds=consumed(index,condition,'EVAL',i),frame_communication_seeds=[consumed(index,condition,'EVAL',i,j) for j in range(10)],raw=rec,raw_csv=dict(path=csvpath.relative_to(ROOT).as_posix(),sha256=c.sha(csvpath)),H=hrec,communication=crec,finite=True,solve_success=True,coordinate_handle=rec['path']+'::coordinates',input_hashes=evidence['inputs']))
  c.write(c.FORMAL/'sealed_generation_receipt.json',dict(reference_count=2500,evaluation_count=5000,unique_clean_H=7500,status='SEALED_RAW_COMPLETE',aggregate_outcomes_viewed=False,bootstrap_generated=False))

def analyze_future_sealed(authorization_path=None):
 """Future analysis requires final MFA freeze and mentor release, absent in B2A."""
 c.require_future_authorization(authorization_path,'unseal-analysis')
 # B3/B4/B5 completion is an additional release dependency from the PLAN_B protocol.
 auth=c.read(authorization_path)
 if not all(auth.get(k) for k in ('B3_frozen_paired_complete','B4_complete','B5_final_retraining_complete')):raise PermissionError('PLAN_B release dependencies incomplete')
 with c.phase('AUTHORIZED_ANALYSIS'):
  receipt=c.read(c.FORMAL/'sealed_generation_receipt.json')
  if receipt['evaluation_count']!=5000 or receipt['reference_count']!=2500:raise RuntimeError('Complete sealed population required')
  reg=c.read(c.FORMAL/'seed_registry.json');master=next(r['seed'] for r in reg['rows'] if r['role']=='bootstrap_master_seed');aggregates=[];contrasts=[];perfect_ber=[]
  q=.05/(2*35)
  for ordinal,condition in enumerate(c.CONDITIONS):
   es=[];qs=[];bes=[];bts=[];perfs=[]
   for i in range(1000):
    path=c.FORMAL/'raw'/condition/f'{i:04d}.npz';proof=c.read(c.FORMAL/'evaluation'/condition/f'proof_{i:04d}.json')
    if c.sha(path)!=proof['raw']['sha256']:raise RuntimeError('Raw frozen hash drift')
    with np.load(path,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    e,truth,bits,total=c.raw_to_trajectory(raw,c.SNRS);es.append(e);qs.append(truth);bes.append(bits);bts.append(total);perfs.append(raw['perfect_frame_bit_errors'].sum(axis=1))
   es=np.array(es);qs=np.array(qs);bes=np.array(bes);bts=np.array(bts)
   point=10*np.log10(es.sum(axis=0)/qs.sum(axis=0)[:,None]);ber=bes.sum(axis=0)/bts.sum(axis=0)[:,None]
   for si,s in enumerate(c.SNRS):
    for mi,m in enumerate(c.METHODS):aggregates.append(dict(condition=condition,snr_db=s,estimator=m,pooled_nmse_db=float(point[si,mi]),ber=float(ber[si,mi]),comparison=None,estimate_db=None,lower=None,upper=None,classification=None))
    perfect_ber.append(dict(condition=condition,snr_db=s,receiver_reference='Perfect CSI',bit_errors=int(np.array(perfs).sum(axis=0)[si]),bit_total=int(bts.sum(axis=0)[si]),ber=float(np.array(perfs).sum(axis=0)[si]/bts.sum(axis=0)[si]),NMSE_estimator_path=False))
   family=[r for r in c.planned_contrasts() if r['condition']==condition]
   if family:
    distributions=np.empty((100000,len(family)));rng=np.random.default_rng(np.random.SeedSequence([master,ordinal]));flat=es.reshape(1000,-1)
    for start in range(0,100000,1000):
     index=rng.integers(0,1000,(1000,1000),dtype=np.int32);counts=np.stack([np.bincount(row,minlength=1000) for row in index]);E=(counts@flat).reshape(1000,7,7);Q=counts@qs;draw=10*np.log10(E/Q[:,:,None])
     for k,r in enumerate(family):
      si=c.SNRS.index(r['snr_db']);first=c.METHODS.index(r['first']);second=c.METHODS.index(r['second']);distributions[start:start+1000,k]=draw[:,si,first]-draw[:,si,second]
    for k,r in enumerate(family):
     lower,upper=np.quantile(distributions[:,k],[q,1-q],method='linear');si=c.SNRS.index(r['snr_db']);value=point[si,c.METHODS.index(r['first'])]-point[si,c.METHODS.index(r['second'])]
     contrasts.append(dict(condition=condition,snr_db=r['snr_db'],estimator=None,pooled_nmse_db=None,ber=None,comparison=r['comparison'],estimate_db=float(value),lower=float(lower),upper=float(upper),classification=c.classify(float(lower),float(upper))))
   # Different condition streams independent; same counts pair allmethods/SNR.
  c.csvwrite(c.FORMAL/'analysis/aggregate.csv',aggregates);c.csvwrite(c.FORMAL/'analysis/primary_contrasts.csv',contrasts);c.csvwrite(c.FORMAL/'analysis/perfect_CSI_BER_reference.csv',perfect_ber)
  c.write(c.FORMAL/'analysis/unseal_receipt.json',dict(contrasts=35,bootstrap_resamples=100000,paired_unit='trajectory',q=q,authorized_by=c.sha(authorization_path)))

def finalize():
 c.require_freeze();c.firewall()
 with c.phase('FORMAL_METADATA'):
  op=c.read(c.OUT/'operator_anchor.json');anchor=c.read(c.OUT/'anchor_receipt.json');smoke=c.read(ROOT/'outputs/sp_u12b2a_edge_smoke.json');seeds=c.read(ROOT/'outputs/sp_u12b2a_seed_overlap_audit.json');pres=c.read(ROOT/'outputs/sp_u12b2a_preservation.json')
  if not all(x['PASS'] for x in (op,anchor,smoke,seeds)) or not pres['pass_gate']:raise RuntimeError('Incomplete final gates')
  files=[p for p in c.FORMAL.rglob('*') if p.is_file()]
  expected={'config.json','seed_registry.json','schemas.json','manifest.json'}
  if {p.relative_to(c.FORMAL).as_posix() for p in files}!=expected:raise RuntimeError('Formal outcome accidental creation/unsealing')
  from scripts.sp_u4a_common import peak_memory
  peak=peak_memory();cache_bytes=5000*(2*2*64*160*16+10*(2*64*16*2+2*64*16*16+2*64*16*16))+2500*(2*2*64*160*16)
  raw_bytes=5000*(7*7*6*8+7*6*8+7*7*6*8+7*6*8+7*6*2*8+7*6*8)
  bootstrap_scratch=1000*1000*4+1000*1000*8+100000*21*8
  numeric_peaks=[dict(stage=stage,worker_pid=entry['worker_pid'],peak=entry['engineering_peak']) for stage,entry in [('operators',op),('anchor',anchor),('smoke',smoke)]]
  seconds=2500/15*smoke['reference_seconds']+5000/15*(7/3)*smoke['evaluation_seconds']+op['operator_seconds']
  resource=dict(scope='engineering extrapolation only, not B4 measured method runtime',channel_communication_cache_bytes=cache_bytes,raw_numeric_bytes=raw_bytes,bootstrap_distribution_bytes=35*100000*8,bootstrap_peak_core_scratch_bytes=bootstrap_scratch,bootstrap_index_tables_persisted=False,operator_cache_bytes=op['operator_archive']['bytes'],disk_design_budget_GiB=12,expected_peak_RAM_GiB_range=[1,3],actual_numeric_validation_worker_peaks=numeric_peaks,finalizer_peak=peak,approximate_formal_wall_seconds=seconds,wall_time_uncertainty='At least factor2 engineering uncertainty; generation/I/O/device/load effects; fullrun inference projected from3SNR tiny smoke to7SNR',not_a_runtime_speedup_claim=True)
  c.write(ROOT/'outputs/sp_u12b2a_resource_estimate.json',resource)
  hashes={p.relative_to(ROOT).as_posix():c.sha(p) for p in files}
  f=c.read(c.OUT/'execution_freeze.json');manifest=dict(verdict='B2_PIPELINE_READY_FOR_SEALED_RUN',status='SEALED_PLANNED_ONLY',formal_H_generated=0,formal_outcomes_viewed=False,formal_bootstrap_generated=False,formal_root=str(c.FORMAL),formal_metadata_hashes=hashes,registry_sha256=c.sha(ROOT/'outputs/sp_u12b2a_seed_registry.json'),execution_freeze_sha256=c.sha(c.OUT/'execution_freeze.json'),operator_cache=op['operator_archive'],preparation_statistics_hashes={m:c.sha(ROOT/'outputs'/p) for m,p in c.STATS.items()},gates=dict(seven_snr_operator='PASS',anchor_replay='PASS',edge_smoke='PASS',seed_separation='PASS',preservation='PASS'),PAGE_BUDGET_DEBT='+5reviewpages;deferred',future_release='Separate user authorization; sealed raw first; aggregate only after B3/B4/B5 final MFA freeze and mentor release',no_B3_B4_B5_B6_execution=True)
  c.write(ROOT/'outputs/sp_u12b2a_formal_manifest.json',manifest)
  log='# SP-U12B2A change log\n\nNew sweep infrastructure only; existing src/estimators and manuscript untouched.\n\n'
  log+='Seven SNR operators:PASS. 10/20 all5000oldtrajectories replayed:PASS. Edge smoke:15developmentevaluation+15independentreference,technical checks only. Formalcache/outcomes/bootstrap generated0.\n\n'
  log+='Frozen sources/config/seeds/schemas and deployable operator hashes recorded; per-stage STOP enforced. Allseed roles audited against historical/reserved/invalidatedmetadata and developmentnamespace. Formalroot contains4metadatafiles andemptydirectories only.\n\n'
  log+='Actual pytest: '+str(f['pytest'])+'\n\nPreservation: '+str(pres)+'\n\nEngineeringresource estimate: '+str(resource)+'\n\nNo performance conclusion or runtimeclaim. PAGE_BUDGET_DEBT+5reviewpages recorded,notprocessed.\n\nStop and await mentor review.\n'
  (ROOT/'docs/sp_u12b2a_change_log.md').write_text(log,encoding='utf8')
  print({'verdict':manifest['verdict'],'formal_cache_generated':False,'formal_outcomes_viewed':False,'preserved_files':pres['protected_files']},flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','freeze','operators','anchor','smoke','finalize']);args=p.parse_args()
 try:globals()[args.stage]()
 except Exception as exc:
  c.OUT.mkdir(exist_ok=True)
  if not (c.OUT/'STOP.json').exists():c.write(c.OUT/'STOP.json',dict(verdict='B2_PIPELINE_REVISION_REQUIRED',stage=args.stage,error_type=type(exc).__name__,reason=str(exc),formal_H_generated=0,formal_outcomes_viewed=False,prohibit_patch_and_formal_run=True))
  print({'verdict':'B2_PIPELINE_REVISION_REQUIRED','stage':args.stage,'reason':str(exc)},flush=True);raise
if __name__=='__main__':main()
