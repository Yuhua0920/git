"""Original fit/inference in an isolated worker; instrumentation never alters math."""
from scripts.sp_u12b5b_common import *
firewall()
from scripts.sp_u12b5b_model import *
import argparse,types,traceback

def access_snapshot(name,kind,stage):
 write(OUT/f'worker_{name}_{kind}_access.json',dict(reads=list(ACCESS),denied=list(DENIED),target_scientific_files_read=0,B2_B3_scientific_values_read=0,phase=kind,snapshot_stage=stage,enforced_allowlist=True))

def location(kind,name):
 if name=='FINAL':return ROOT/'outputs/sp_u12b5b_final_mfa'
 if name=='CALIBRATION':return OUT/'calibration_model'
 return ROOT/'outputs/sp_u12b5b_candidates'/name

def train(name):
 cfg=read(CONFIG);inputs=read(OUT/'input_manifest.json');directory=location('train',name)
 if name=='CALIBRATION':
  K,L,seed=2,2,4190123501;data=(np.random.default_rng(4190123500).standard_normal((80,8))+1j*np.random.default_rng(4190123499).standard_normal((80,8)))/np.sqrt(2);datahash=array_hash(data);phase='BENCHMARK_SAFE_CALIBRATION'
 else:
  if name=='FINAL':
   decision=read(ROOT/'outputs/sp_u12b5b_selection_decision.json');selection=read(ROOT/'outputs/sp_u12b5b_selection_manifest.json')
   assert selection['status']=='SELECTION_FROZEN' and sha(ROOT/'outputs/sp_u12b5b_selection_manifest.json')==decision['selection_manifest_sha256']
   row=next(r for r in cfg['candidate_order'] if (r['K'],r['rank'])==(decision['selected_K'],decision['selected_rank']));filename='full_retraining.npy';seed=row['final_seed'];phase='FINAL_RETRAINING'
  else:row=next(r for r in cfg['candidate_order'] if r['name']==name);filename='selection_train.npy';seed=row['seed'];phase='CANDIDATE_TRAINING'
  K,L=row['K'],row['rank'];path=OUT/filename;assert sha(path)==inputs['files'][filename]
  data=np.load(path,allow_pickle=False);datahash=inputs['files'][filename]
  assert data.dtype==np.complex128 and data.flags.c_contiguous and data.shape==((15000 if name=='FINAL' else 12000),4032)
 model=make_reference_model(K,L);assert model.maxiter==100 and model.rs_clip==0 and model.PPCA and not model.lock_psis
 directory.mkdir(parents=True,exist_ok=True);initial={};original_init=model._initialize;original_step=model._EM_per_component
 counter=0;tick=time.monotonic()
 def instrumented_initialize(values):
  original_init(values);initial.update(parameter_hashes(model,True))
  access_snapshot(name,'train','initialized')
  write(directory/'initialization.json',dict(K=K,rank=L,seed=int(seed),parameter_hashes=initial,combined_hash=tree(initial),training_data_hash=datahash,training_order='origin / ascending trajectory / ascending frame / joint C-order',phase=phase))
  print(json.dumps(dict(stage='initialized',candidate=name,actual_PID=os.getpid(),seed=int(seed))),flush=True)
 def instrumented_step(values,ppca):
  nonlocal counter
  start=time.monotonic();original_step(values,ppca);counter+=1
  objective=float(model.logLs.sum())
  with (directory/'iteration_events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(iteration=counter,pre_update_objective=objective,step_seconds=time.monotonic()-start,elapsed_seconds=time.monotonic()-tick))+'\n')
  print(json.dumps(dict(stage='EM',candidate=name,iteration=counter,pre_update_objective=objective,elapsed_seconds=time.monotonic()-tick)),flush=True)
  access_snapshot(name,'train','completed_EM_step')
 model._initialize=instrumented_initialize;model._EM_per_component=instrumented_step
 write(directory/'fit_entry.json',dict(actual_PID=os.getpid(),K=K,rank=L,seed=int(seed),fit_calls=1,extra_restarts=0,shape=list(data.shape),training_data_hash=datahash,phase=phase,started_UTC=utc()))
 access_snapshot(name,'train','before_original_fit')
 fit_reference(model,data,seed)
 validity=parameter_validity(model);assert validity['PASS'],'INVALID_TRAINED_MODEL'
 artifact=save_model(model,directory,seed,datahash,initial)
 history=np.array(model.L_all);assert np.isfinite(history).all()
 stopping=bool(len(history)>=7 and abs((history[-1]-history[-2])/history[-1])<1e-4)
 result=dict(status='TRAINING_COMPLETE',K=K,rank=L,seed=int(seed),iterations=len(history),convergence_status='MAX_ITER_REACHED' if len(history)==100 else 'CONVERGED',stopping_criterion_satisfied=stopping,final_objective=float(history[-1]),validity=validity,initial_parameter_hash=tree(initial),final_parameter_hash=artifact['parameter_hash'],training_data_hash=datahash,wall_seconds=time.monotonic()-tick,actual_PID=os.getpid(),phase=phase,threadpools=threadpool_info(),target_information_used=False)
 write(directory/'training_result.json',result)
 print(json.dumps(dict(stage='training_complete',candidate=name,iterations=len(history),convergence_status=result['convergence_status'])),flush=True)

def validate(name):
 cfg=read(CONFIG);inputs=read(OUT/'input_manifest.json');directory=location('validate',name);model,artifact=load_model(directory)
 access_snapshot(name,'validate','model_reloaded')
 assert parameter_validity(model)['PASS'],'MODEL_RELOAD_INVALID'
 if name in ('FINAL','CALIBRATION'):
  if name=='FINAL':
   interface=ProjectPilotInterface();A=interface.A;output_positions=interface.data_positions;y=np.zeros((1,64),np.complex128);Cn=.9941134822523973*np.eye(64,dtype=complex)
  else:A=np.eye(8,dtype=complex);output_positions=np.arange(8);y=np.zeros((1,8),complex);Cn=.37*np.eye(8,dtype=complex)
  out=MFAReferenceAdapter(model,A).estimate(y,Cn)[:,output_positions]
  result=dict(PASS=bool(np.isfinite(out).all()),serialization_roundtrip=True,parameter_hash=artifact['parameter_hash'],shape=list(out.shape),dtype=str(out.dtype),input='benchmark-safe deterministic zeros; not formal or preparation-validation pilot data',all_components_float_1_0=True,target_data_read=False)
  assert result['PASS'];write(directory/'smoke_result.json',result);return
 for n in ('validation_pilots.npy','validation_truth.npy'):assert sha(OUT/n)==inputs['files'][n],'VALIDATION_INPUT_DRIFT'
 pilots=np.load(OUT/'validation_pilots.npy',mmap_mode='r',allow_pickle=False);truth=np.load(OUT/'validation_truth.npy',mmap_mode='r',allow_pickle=False)
 access_snapshot(name,'validate','preparation_validation_inputs_loaded')
 interface=ProjectPilotInterface();adapter=MFAReferenceAdapter(model,interface.A);cells=[];tick=time.monotonic()
 for o,origin in enumerate(cfg['canonical_origins']):
  T=np.sum(abs(np.asarray(truth[o]))**2,axis=2,dtype=np.float64)
  for si,s in enumerate(('10','20')):
   E=np.empty((100,6),np.float64);responsibility_residual=0.0
   for i in range(100):
    y=np.asarray(pilots[o,si,i]);estimate=adapter.estimate(y,cfg['noise_variances'][s]*np.eye(64,dtype=complex))
    resp=model.predict_proba(y)
    assert np.isfinite(estimate).all() and np.isfinite(resp).all() and np.all(resp>=0),'VALIDATION_POSTERIOR_NONFINITE_OR_INVALID'
    residual=float(np.max(abs(resp.sum(axis=1)-1)));assert residual<=1e-10,'VALIDATION_RESPONSIBILITY_NORMALIZATION'
    responsibility_residual=max(responsibility_residual,residual)
    E[i]=np.sum(abs(estimate[:,interface.data_positions]-np.asarray(truth[o,i]))**2,axis=1,dtype=np.float64)
    if (i+1)%10==0:print(json.dumps(dict(stage='validation',candidate=name,origin=origin,SNR=int(s),trajectories_completed=i+1)),flush=True)
   ratio=pooled_ratio(E,T);cells.append(dict(candidate=name,K=model.n_components,rank=model.M,origin=origin,snr_dB=int(s),error_energy=float(E.sum()),truth_energy=float(T.sum()),linear_pooled_NMSE=ratio,trajectories=100,scored_frames='4,5,6,7,8,9',max_responsibility_sum_error=responsibility_residual))
   with (directory/f'validation_o{o}_snr{s}.npz').open('xb') as f:np.savez(f,error_energy=E,truth_energy=T)
 score=macro_score([c['linear_pooled_NMSE'] for c in cells])
 csvwrite(directory/'validation_cells.csv',cells)
 write(directory/'validation_result.json',dict(PASS=True,macro_selection_score=score,cells=cells,input_hashes={n:inputs['files'][n] for n in ('validation_pilots.npy','validation_truth.npy')},serialization_roundtrip=True,wall_seconds=time.monotonic()-tick,target_data_read=False))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['train','validate']);parser.add_argument('name');args=parser.parse_args()
 directory=location(args.kind,args.name);directory.mkdir(parents=True,exist_ok=True)
 write(OUT/f'worker_{args.name}_{args.kind}_PID.json',dict(actual_PID=os.getpid(),role=args.kind,candidate=args.name,created_UTC=utc()))
 try:
  ready=OUT/f'worker_{args.name}_{args.kind}_MONITOR_READY.json';started=time.monotonic()
  while not ready.exists():
   if time.monotonic()-started>30:raise TimeoutError('ACTUAL_PID_MONITOR_NOT_ATTACHED')
   time.sleep(.05)
  with threadpool_limits(1):globals()[args.kind](args.name)
 except BaseException as error:
  write(directory/f'{args.kind}_failure.json',dict(status='MFA_SELECTION_TECHNICAL_FAILURE',exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),actual_PID=os.getpid(),phase=args.kind))
  raise
 finally:access_snapshot(args.name,args.kind,'worker_exit')
if __name__=='__main__':main()
