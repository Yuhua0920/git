"""Single Level-1 patch, exact equivalence gates, read-only corpus replay and re-gate."""
import argparse
import ast
import copy
import csv
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.append(str(ROOT/'.venv-gmm/Lib/site-packages'))
from scripts import sp_u3b_r1_common as c

ORIGINAL='''            weighted_cross_covariance = np.dot(
                zeroed_data[:, None, :] * latents[None, :, :].conj(),
                self._responsibilities[component],
            )'''
REPLACEMENT='''            weighted_latents = latents.conj() * self._responsibilities[component][None, :]
            weighted_cross_covariance = zeroed_data @ weighted_latents.T'''

def prepare():
    prior=c.verify_prior();q=c.cfg()
    original=c.OUT/'sp_u3a_sources/cplx-mfa/src/cplx_mfa/mfa.py'
    source=original.read_text(encoding='utf8');assert source.count(ORIGINAL)==1
    patched=source.replace(ORIGINAL,REPLACEMENT)
    file=c.path('patched_mfa.py');assert not file.exists();file.write_text(patched,encoding='utf8')
    diff=''.join(difflib.unified_diff(source.splitlines(keepends=True),patched.splitlines(keepends=True),
      fromfile='pinned/cplx_mfa/mfa.py',tofile='task_local/patched_mfa.py'))
    c.path('patch.diff').write_text(diff,encoding='utf8')
    # Exact scope proof: reverse the one replacement and recover original text.
    assert patched.replace(REPLACEMENT,ORIGINAL)==source
    c.save(c.path('patch_audit.json'),dict(passed=True,commit=q['author_commits']['cplx-mfa'],
      original_path=original.relative_to(ROOT).as_posix(),original_sha256=c.sha(original),
      patched_path=file.relative_to(ROOT).as_posix(),patched_sha256=c.sha(file),
      diff_sha256=c.sha(c.path('patch.diff')),changed_functions=['ComplexMFA._em_step'],
      original_expression=ORIGINAL,patched_expression=REPLACEMENT,level=1,chunking=False,
      unchanged_online_estimator=True,prior=prior,rtol=q['rtol'],atol=q['atol']))
    rows=[]
    D=4096;N=12000;b=16
    for name,s in q['sentinels'].items():
        K=s['K'];r=s['rank'];data=D*N*b;dense=K*D*D*b
        arrays=dict(training=data,responsibilities=K*N*8,log_responsibilities=K*N*8,
          log_likelihood=N*8,means=K*D*b,loadings=K*D*r*b,Psi=K*D*8,weights=K*8,
          covariance=dense,precision=dense,betas=K*r*D*b,latents=r*N*b,
          latent_outer_products=r*r*N*b,latent_covariances=r*r*N*b,
          centered_data=data,loadings_latents=data,zeroed_data=data,residual=data,
          likelihood_centered=data,likelihood_transformed=data,likelihood_product=data,
          residual_conjugate=data,residual_product=data,
          original_weighted_cross_tensor=D*r*N*b,patched_latent_conjugate=r*N*b,
          patched_weighted_latents=r*N*b,weighted_cross_output=D*r*b,
          weighted_latent_output=r*r*b,observation_buffers=0)
        # Each expression has a different lifetime. Do not sum all arrays as a peak.
        base=sum(arrays[k] for k in ('training','means','loadings','Psi','weights','covariance','precision'))
        original_lower=base+arrays['original_weighted_cross_tensor']
        patched_cross_live=base+3*data+arrays['betas']+3*r*N*b+2*r*r*N*b+D*r*b
        patched_noise_live=base+6*data+arrays['betas']+2*r*N*b+2*r*r*N*b+D*r*b+r*r*b
        # _em_step locals remain alive during _update_covariances. Original
        # _invert_covariances can have old precision + correction + negation live.
        patched_inverse_live=base+4*data+2*dense+arrays['betas']+2*r*N*b+2*r*r*N*b
        rows.append(dict(name=name,K=K,rank=r,arrays=arrays,
          original_necessary_array_lower_bound=original_lower,
          patched_cross_expression_live_estimate=patched_cross_live,
          patched_residual_expression_live_estimate=patched_noise_live,
          patched_inverse_expression_live_estimate=patched_inverse_live,
          patched_max_named_live_array_estimate=max(patched_cross_live,patched_noise_live,patched_inverse_live),
          temporary_savings=D*r*N*b-r*N*b,
          caveat='named-array lifetime estimates exclude allocator/BLAS/runtime overhead; not an observed process peak'))
    assert all(r['patched_max_named_live_array_estimate']<q['max_process_bytes'] for r in rows)
    c.save(c.path('static_audit.json'),dict(passed=True,rows=rows,
      no_D_rank_N_product_in_patch=True,still_dense_likelihood=True,still_dense_inverse=True,
      observation_buffers='not allocated by training; original online API unchanged',
      outcome='static gate permits measured smoke; margin must be validated by process working-set and commit peaks'))
    print('PREPARE PASS',[(r['name'],r['patched_max_named_live_array_estimate']/2**30) for r in rows],flush=True)

def metric(a,b):
    import numpy as np
    a=np.asarray(a);b=np.asarray(b);delta=a-b;q=c.cfg()
    return dict(shape=list(a.shape),dtype=str(a.dtype),
      max_abs=float(np.max(abs(delta))),relative_Frobenius=float(np.linalg.norm(delta)/max(np.linalg.norm(a),1e-300)),
      allclose=bool(np.allclose(a,b,rtol=q['rtol'],atol=q['atol'])))

def compare_fields(a,b,fields):
    return {k:metric(getattr(a,k),getattr(b,k)) for k in fields}

def equivalence():
    from scripts import sp_u3b_r1_mfa as p
    import numpy as np
    q=c.cfg();assert c.load(c.path('static_audit.json'))['passed']
    rng=np.random.default_rng(q['equivalence']['seed'])
    def cn(shape):return (rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2)
    contraction=[];steps=[];fits=[];passed=True
    fields=['means_','loadings_','noise_variances_','weights_','covariances_','precisions_']
    with p.threadpool_limits(limits=q['threads']):
        for D,N,r,K,uniform in [(3,7,1,1,True),(8,19,4,2,False),(23,41,16,4,False)]:
            Z=cn((D,N));L=cn((r,N));weights=rng.uniform(.02,1,(K,N));weights/=weights.sum(axis=0)
            if uniform:weights[:]=1/K
            for k in range(K):
                w=weights[k];original=np.dot(Z[:,None,:]*L[None,:,:].conj(),w)
                loop=np.zeros((D,r),complex)
                for i in range(D):
                    for j in range(r):
                        for n in range(N):loop[i,j]+=Z[i,n]*L[j,n].conjugate()*w[n]
                patched=p.patched_contraction(Z,L,w)
                er=dict(original_loop=metric(original,loop),original_patched=metric(original,patched),loop_patched=metric(loop,patched))
                assert original.shape==patched.shape==(D,r) and patched.dtype==np.complex128
                wrong_conjugation=Z@(L*w[None,:]).T
                wrong_weighting=Z@L.conj().T
                bad_conj=float(np.max(abs(patched-wrong_conjugation)))
                bad_weight=float(np.max(abs(patched-wrong_weighting)))
                assert bad_conj>1e-3
                if K>1:assert bad_weight>1e-3
                passed=passed and all(v['allclose'] for v in er.values())
                contraction.append(dict(D=D,N=N,rank=r,K=K,component=k,uniform=uniform,
                  errors=er,wrong_conjugation_error=bad_conj,wrong_weight_error=bad_weight))
        for K,r,D,N in q['equivalence']['fixtures']:
            X=cn((N,D))+.25*cn((K,D))[np.arange(N)%K]
            a=p.make(K,r,q['equivalence']['seed'],patched=False)
            b=p.make(K,r,q['equivalence']['seed'],patched=True)
            for m in (a,b):
                m._n_samples,m._n_features=X.shape;m._responsibilities=np.zeros((K,N));m._initialize(X)
            initial=compare_fields(a,b,fields)
            assert all(np.array_equal(getattr(a,k),getattr(b,k)) for k in fields)
            assert a._rng.bit_generator.state==b._rng.bit_generator.state
            a._em_step(X);b._em_step(X)
            errors=compare_fields(a,b,fields+['_responsibilities','_log_likelihoods'])
            # Objective using updated parameters in addition to the native pre-M trace.
            errors['updated_likelihood']=metric(a._calculate_probabilities(X)[0],b._calculate_probabilities(X)[0])
            passed=passed and all(v['allclose'] for v in errors.values())
            steps.append(dict(K=K,rank=r,D=D,N=N,initial_bit_exact=True,RNG_state_exact=True,errors=errors))
            a=p.make(K,r,q['equivalence']['seed'],max_iter=q['equivalence']['max_iter'],patched=False)
            b=p.make(K,r,q['equivalence']['seed'],max_iter=q['equivalence']['max_iter'],patched=True)
            warns=[]
            for m in (a,b):
                with warnings.catch_warnings(record=True) as ww:
                    warnings.simplefilter('always');m.fit(X)
                warns.append([str(w.message) for w in ww])
            errors=compare_fields(a,b,fields)
            errors['likelihood_trace']=metric(a.lower_bound_history_,b.lower_bound_history_)
            errors['posterior_responsibilities']=metric(a.predict_proba(X),b.predict_proba(X))
            A=cn((3,D))/np.sqrt(D);y=X[:4]@A.T;Cn=.37*np.eye(3)
            errors['estimator']=metric(a.estimate(y,Cn,A,1.0),b.estimate(y,Cn,A,1.0))
            errors['objective_updated']=metric(a._calculate_probabilities(X)[0],b._calculate_probabilities(X)[0])
            assert len(a.lower_bound_history_)>=5 and len(a.lower_bound_history_)==len(b.lower_bound_history_)
            assert warns[0]==warns[1]
            passed=passed and all(v['allclose'] for v in errors.values())
            fits.append(dict(K=K,rank=r,D=D,N=N,iterations=len(a.lower_bound_history_),
              original_trace=a.lower_bound_history_,patched_trace=b.lower_bound_history_,errors=errors,warnings=warns))
    c.save(c.path('equivalence.json'),dict(passed=passed,rtol=q['rtol'],atol=q['atol'],
      contraction=contraction,EM_steps=steps,fits=fits,BLAS=p.threadpool_info(),
      runtime=dict(Python=sys.version,NumPy=np.__version__,SciPy=__import__('scipy').__version__,
        sklearn=__import__('sklearn').__version__,fit_threads=q['threads']),access=c.ACCESS))
    print('EQUIVALENCE', 'PASS' if passed else 'FAIL',flush=True)
    if not passed:raise RuntimeError('PATCH SEMANTICS FAIL; no tolerance relaxation or second patch')

def corpus(kind):
    import numpy as np
    from src.cdl_benchmark import OfficialCDL,frames,array_hash,SYSTEM,seeds
    from src.cdl_mismatch import ShiftCDL,seeds as shift_seeds
    from threadpoolctl import threadpool_limits
    assert c.load(c.path('equivalence.json'))['passed']
    if kind=='full':assert c.load(c.path('medium_gate.json'))['passed']
    assert SYSTEM['sample_rate_hz']==SYSTEM['nfft']*SYSTEM['delta_f_hz']
    with (c.OUT/'t15a_fix_replay_h_hashes.csv').open(encoding='utf8',newline='') as f:
        refs={('CDL-C-30',int(r['trajectory_id'])):r['replayed_h_sha256'] for r in csv.DictReader(f)
              if r['split']=='train_dev' and int(r['trajectory_id'])<400}
    with (c.OUT/'t15c_cdl_channel_hashes.csv').open(encoding='utf8',newline='') as f:
        refs.update({(r['shift'],int(r['trajectory_id'])):r['H_sha256'] for r in csv.DictReader(f)
          if r['split']=='train_dev' and int(r['trajectory_id'])<400})
    manifests={}
    for file in ('t15a_cdl_manifest.csv','t15c_cdl_manifest.csv'):
        with (c.OUT/file).open(encoding='utf8',newline='') as f:
            for row in csv.DictReader(f):
                if row['split']=='train_dev' and int(row['trajectory_id'])<400:
                    manifests[row.get('shift','CDL-C-30'),int(row['trajectory_id']),int(row['frame_id'])]=row
    envs=['CDL-C-30'] if kind=='medium' else c.cfg()['corpus']['environments']
    count=c.cfg()['medium']['trajectories'] if kind=='medium' else 400
    target=c.path(kind+'_clean.npy');assert not target.exists()
    data=np.empty((len(envs)*count*10,4096),np.complex128,order='C')
    channel_rows=[];frame_rows=[];start=time.perf_counter();position=0
    with threadpool_limits(limits=c.cfg()['threads']):
        for env in envs:
            adapter=OfficialCDL() if env=='CDL-C-30' else ShiftCDL(env)
            for i in range(count):
                H=adapter.generate(i) if env=='CDL-C-30' else adapter.generate(i,split='train_dev')
                digest=array_hash(H);assert digest==refs[env,i],(env,i,'H replay mismatch')
                channel_rows.append(dict(environment=env,trajectory_id=i,H_sha256=digest,frozen_reference_exact=True))
                for j,frame in enumerate(frames(H)):
                    if env=='CDL-C-30':
                        used=seeds(i,j);channel=seeds(i,0)
                        for role in ('sionna_seed','torch_seed','numpy_seed'):used[role]=channel[role]
                    else:used=shift_seeds(env,'train_dev',i,j)
                    row=manifests[env,i,j];assert all(int(row[k])==v for k,v in used.items())
                    data[position]=frame.ravel(order='C')
                    frame_rows.append(dict(row=position,environment=env,trajectory_id=i,frame_id=j,
                      clean_sha256=hashlib.sha256(memoryview(data[position]).cast('B')).hexdigest()))
                    position+=1
                if (i+1)%50==0:print('CORPUS',kind,env,i+1,'/',count,flush=True)
    assert data.shape==(position,4096) and data.flags.c_contiguous and np.isfinite(data).all()
    assert len({r['H_sha256'] for r in channel_rows})==len(channel_rows)
    np.save(target,data,allow_pickle=False)
    orderfile=c.path(kind+'_order.csv');assert not orderfile.exists()
    with orderfile.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(frame_rows[0]));writer.writeheader();writer.writerows(frame_rows)
    c.save(c.path(kind+'_corpus.json'),dict(passed=True,kind=kind,samples=position,dimension=4096,
      dtype=str(data.dtype),C_contiguous=True,environments=envs,trajectories_per_environment=count,
      data_sha256=c.sha(target),order_sha256=c.sha(orderfile),channel_identity=channel_rows,
      original_reference_files={p:c.sha(c.OUT/p) for p in sorted(c.OLD)},
      ordering=c.cfg()['corpus']['order'],generation_device=adapter.device,all_frozen_H_exact=True,all_actual_seeds_exact=True,
      environment_labels_excluded_from_fit=True,generation_seconds=time.perf_counter()-start,
      process_memory=c.memory(),access=c.ACCESS))
    print('CORPUS PASS',kind,position,'all H hashes and consumed seeds exact',flush=True)

def trace_observer(log,state):
    """Read native locals only; never change a model variable or stop criterion."""
    last=[0]
    def trace(frame,event,arg):
        name=frame.f_code.co_name
        if name not in ('_run_em','_em_step'):return None
        if name=='_run_em':
            m=frame.f_locals.get('self')
            if m is not None and len(m.lower_bound_history_)>last[0]:
                last[0]=len(m.lower_bound_history_)
                row=dict(iteration_count=last[0],objective=float(m.lower_bound_history_[-1]),elapsed_seconds=time.perf_counter()-state['fit_started'])
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if event=='return':
                state['native_converged']=bool(frame.f_locals.get('converged',False))
                state['native_relative_change']=float(frame.f_locals.get('relative_change',float('inf')))
        else:
            m=frame.f_locals.get('self')
            if m is not None:
                forbidden=(m._n_features,m.latent_dim,m._n_samples)
                shapes=state.setdefault('observed_EM_array_shapes',{})
                for key,value in frame.f_locals.items():
                    shape=getattr(value,'shape',None)
                    if shape is not None:
                        shapes[key]=list(shape)
                        if tuple(shape)==forbidden:state['forbidden_tensor_seen']=True
            if event=='return':
                state['last_E_occupancy']=m._responsibilities.sum(axis=1).tolist()
        return trace
    return trace

def worker(kind):
    import os
    c.save(c.path(kind+'_process.json'),dict(actual_pid=os.getpid(),parent_pid=os.getppid()))
    from scripts import sp_u3b_r1_mfa as p
    import numpy as np
    q=c.cfg();c.frozen_sources()
    if kind=='medium':
        spec=q['medium'];corpus_kind='medium';max_iter=spec['max_iter']
    else:
        spec=q['sentinels'][kind];corpus_kind='full';max_iter=q['max_iter']
    evidence=c.load(c.path(corpus_kind+'_corpus.json'));assert evidence['passed']
    file=c.path(corpus_kind+'_clean.npy');assert c.sha(file)==evidence['data_sha256']
    data=np.load(file,allow_pickle=False);assert data.dtype==np.complex128 and data.flags.c_contiguous
    assert data.shape==(120,4096) if kind=='medium' else data.shape==(12000,4096)
    model=p.make(spec['K'],spec['rank'],q['seed'],max_iter=max_iter)
    state={};result=dict(kind=kind,K=spec['K'],rank=spec['rank'],seed=q['seed'],samples=len(data),
      dimension=4096,dtype=str(data.dtype),data_sha256=evidence['data_sha256'],patched_sha256=c.sha(c.path('patched_mfa.py')))
    tracefile=c.path(kind+'_trace.jsonl');assert not tracefile.exists()
    with p.threadpool_limits(limits=q['threads']),tracefile.open('w',encoding='utf8') as log:
        state['fit_started']=time.perf_counter()
        c.save(c.path(kind+'_fit_start.json'),dict(perf_counter=state['fit_started']))
        sys.settrace(trace_observer(log,state))
        try:
            with warnings.catch_warnings(record=True) as ww:
                warnings.simplefilter('always');model.fit(data)
            fitseconds=time.perf_counter()-state['fit_started'];sys.settrace(None)
            c.save(c.path(kind+'_fit_end.json'),dict(fit_seconds=fitseconds))
            fields=['means_','loadings_','noise_variances_','weights_','covariances_','precisions_']
            finite=all(np.isfinite(getattr(model,k)).all() for k in fields)
            weights_valid=bool(np.all(model.weights_>=0) and np.isclose(model.weights_.sum(),1,rtol=1e-12,atol=1e-12))
            result.update(fit_seconds=fitseconds,iterations=len(model.lower_bound_history_),
              native_converged=state['native_converged'],native_relative_change=state['native_relative_change'],
              termination='NATIVE_CONVERGED' if state['native_converged'] else 'NATIVE_MAX_ITER',
              objective_trace=model.lower_bound_history_,final_native_objective=float(model.lower_bound_history_[-1]),
              objective_meaning='original lower_bound_history_: E-step log likelihood before corresponding M update',
              warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in ww],
              finite_parameters=finite,weight_sum=float(model.weights_.sum()),weights=model.weights_.tolist(),
              minimum_Psi=float(model.noise_variances_.min()),maximum_Psi=float(model.noise_variances_.max()),
              loading_norms=np.linalg.norm(model.loadings_,axis=(1,2)).tolist(),
              occupancy=state.get('last_E_occupancy'),occupancy_meaning='last native E step, not re-estimated final posterior',
              observed_EM_array_shapes=state.get('observed_EM_array_shapes',{}),
              forbidden_tensor_seen=state.get('forbidden_tensor_seen',False),
              resident_model_array_bytes=sum(getattr(model,k).nbytes for k in fields),
              fit_valid=finite and weights_valid and not state.get('forbidden_tensor_seen',False) and (kind=='medium' or state['native_converged']),
              BLAS=p.threadpool_info(),memory=c.memory())
            if kind!='medium':
                modelfile=c.path(kind+'_model.npz');assert not modelfile.exists()
                np.savez(modelfile,**{k:getattr(model,k) for k in fields})
                result['model_sha256']=c.sha(modelfile)
            result['passed']=bool(result['fit_valid'])
        except Exception as exc:
            sys.settrace(None)
            result.update(passed=False,error=repr(exc),fit_seconds=time.perf_counter()-state['fit_started'],
              termination='EXCEPTION',iterations=len(model.lower_bound_history_),objective_trace=model.lower_bound_history_,memory=c.memory())
    result['access']=c.ACCESS;c.save(c.path(kind+'_worker_result.json'),result)
    print('FIT RESULT',kind,result['passed'],result.get('iterations'),result.get('fit_seconds'),flush=True)

def supervise(kind):
    assert c.load(c.path('equivalence.json'))['passed']
    c.frozen_sources();q=c.cfg()
    if kind=='A':assert c.load(c.path('medium_gate.json'))['passed']
    if kind=='B':assert c.load(c.path('A_gate.json'))['passed']
    logpath=c.path(kind+'_execution.txt');assert not logpath.exists()
    with logpath.open('w',encoding='utf8') as out:
        process=subprocess.Popen([sys.executable,str(Path(__file__)),'worker',kind],cwd=ROOT,
          stdout=out,stderr=subprocess.STDOUT,env=dict(__import__('os').environ,PYTHONDONTWRITEBYTECODE='1'))
        handle=c.open_process(process.pid);start=time.perf_counter();fitstart=None;fitend=None;actual_pid=None
        maximum=dict(peak_working_set_bytes=0,peak_commit_bytes=0);reason=None;samples=0;lastprint=start
        try:
            while process.poll() is None:
                handshake=c.path(kind+'_process.json')
                if actual_pid is None and handshake.exists():
                    try:actual_pid=c.load(handshake)['actual_pid']
                    except json.JSONDecodeError:time.sleep(q['monitor_period_seconds']);continue
                    if actual_pid!=process.pid:
                        c.close_process(handle);handle=c.open_process(actual_pid)
                try:
                    mem=c.memory(handle);samples+=1
                    for key in maximum:maximum[key]=max(maximum[key],mem[key])
                except OSError:
                    if process.poll() is not None:break
                    raise
                startfile=c.path(kind+'_fit_start.json');endfile=c.path(kind+'_fit_end.json')
                try:
                    if fitstart is None and startfile.exists():fitstart=c.load(startfile)['perf_counter']
                    if fitend is None and endfile.exists():fitend=c.load(endfile)['fit_seconds']
                except json.JSONDecodeError:time.sleep(q['monitor_period_seconds']);continue
                elapsed=0 if fitstart is None else (fitend if fitend is not None else time.perf_counter()-fitstart)
                reason=c.stop_reason(maximum,elapsed)
                if reason:
                    if actual_pid and actual_pid!=process.pid:
                        __import__('os').kill(actual_pid,__import__('signal').SIGTERM)
                    else:process.terminate()
                    process.wait();break
                if time.perf_counter()-lastprint>=30:
                    lastprint=time.perf_counter()
                    print('MONITOR',kind,'fit_s',round(elapsed,2),'WS_GiB',round(maximum['peak_working_set_bytes']/2**30,3),
                      'commit_GiB',round(maximum['peak_commit_bytes']/2**30,3),flush=True)
                time.sleep(q['monitor_period_seconds'])
            # Windows kernel peak counters persist on the retained process handle.
            mem=c.memory(handle)
            for key in maximum:maximum[key]=max(maximum[key],mem[key])
        finally:
            if process.poll() is None:
                if actual_pid and actual_pid!=process.pid:
                    try:__import__('os').kill(actual_pid,__import__('signal').SIGTERM)
                    except OSError:pass
                process.terminate();process.wait()
            c.close_process(handle)
    file=c.path(kind+'_worker_result.json');result=c.load(file) if file.exists() else None
    if result:
        for key in maximum:maximum[key]=max(maximum[key],result['memory'][key])
    seconds=result['fit_seconds'] if result else (time.perf_counter()-fitstart if fitstart else None)
    reason=reason or c.stop_reason(maximum,seconds or 0)
    passed=bool(result and result['passed'] and process.returncode==0 and reason is None)
    trace=[]
    if c.path(kind+'_trace.jsonl').exists():
        trace=[json.loads(s) for s in c.path(kind+'_trace.jsonl').read_text(encoding='utf8').splitlines() if s]
    c.save(c.path(kind+'_gate.json'),dict(passed=passed,kind=kind,reason=reason or ('VALID_FIT' if passed else 'INVALID_FIT'),
      exit_code=process.returncode,observed_peaks=maximum,monitor_samples=samples,
      launcher_pid=process.pid,actual_interpreter_pid=actual_pid,
      monitor_period_seconds=q['monitor_period_seconds'],fit_seconds=seconds,total_worker_seconds=time.perf_counter()-start,
      memory_method='Windows GetProcessMemoryInfo PROCESS_MEMORY_COUNTERS_EX: PeakWorkingSetSize and PeakPagefileUsage(process peak commit); PrivateUsage current also collected',
      objective_trace=trace,worker_result_available=bool(result),worker_result=result,
      stop_branch=not passed,patch_frozen=True))
    print('GATE',kind,'PASS' if passed else 'FAIL',reason,maximum,flush=True)
    if not passed:print('MFA FULL COMPARATOR BRANCH STOP; no patch retry or later stage',flush=True)

def tests(stage):
    command=[sys.executable,'-m','pytest','tests/test_sp_u3b_r1.py','-q','-ra','-p','no:cacheprovider',
      '--basetemp=outputs/sp_u3b_r1_pytest_tmp_'+stage]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,
      env=dict(__import__('os').environ,PYTHONDONTWRITEBYTECODE='1',SP_U3B_R1_TEST_STAGE=stage))
    log=c.path('pytest_'+stage+'.txt');assert not log.exists();log.write_text(result.stdout+result.stderr,encoding='utf8')
    print(result.stdout+result.stderr,flush=True)
    c.save(c.path('tests_'+stage+'.json'),dict(passed=result.returncode==0,command=command,exit_code=result.returncode,log_sha256=c.sha(log)))
    if result.returncode:raise RuntimeError('Tests failed; preserve result, no full fit')
    if stage=='pre':
        assert c.load(c.path('equivalence.json'))['passed']
        sources=[ROOT/'configs/sp_u3b_r1_protocol.json',ROOT/'tests/test_sp_u3b_r1.py',*ROOT.glob('scripts/*sp_u3b_r1*.py'),c.path('patched_mfa.py')]
        c.save(c.path('execution_freeze.json'),dict(hashes={p.relative_to(ROOT).as_posix():c.sha(p) for p in sources},
          before_medium=True,before_full_A=True,patch_result_independent=True,
          dependencies={n:importlib.metadata.version(n) for n in ('numpy','scipy','pytest')}))

def seal():
    prior=c.verify_prior();c.frozen_sources()
    medium=c.load(c.path('medium_gate.json')) if c.path('medium_gate.json').exists() else None
    A=c.load(c.path('A_gate.json')) if c.path('A_gate.json').exists() else None
    B=c.load(c.path('B_gate.json')) if c.path('B_gate.json').exists() else None
    passed=bool(A and A['passed'] and B and B['passed'])
    check=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True);assert check.returncode==0
    c.save(c.path('decision.json'),dict(task='SP-U3B-R1',gate='PASS' if passed else 'FAIL',
      medium='PASS' if medium and medium['passed'] else 'FAIL_OR_NOT_RUN',
      A='PASS' if A and A['passed'] else ('FAIL' if A else 'NOT_RUN'),
      B='PASS' if B and B['passed'] else ('FAIL' if B else 'NOT_RUN_DUE_GATE'),
      full_comparator_branch='await mentor authorization for SP-U3B-H' if passed else 'STOP',
      prior_integrity=prior,HEAD=c.git('rev-parse','HEAD'),tracked_diff=c.git('diff','--stat'),
      staged_diff=c.git('diff','--cached','--stat'),diff_check_exit=check.returncode,
      performance_comparison=False,B120_generated=False,formal_read=False,U2_outcomes_read=False,
      no_author_source_change=True,no_LR4_change=True,no_paper_change=True,no_commit=True,
      stop=True,access=c.ACCESS))
    artifacts={p.relative_to(ROOT).as_posix():c.sha(p) for p in sorted(c.OUT.glob('sp_u3b_r1_*')) if p.is_file()}
    c.save(c.path('delivery_manifest.json'),dict(artifacts=artifacts,execution=c.frozen_sources(),stop=True))
    print('SEALED SP-U3B-R1', 'PASS' if passed else 'FAIL',len(artifacts),'artifacts; STOP',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','equivalence','corpus','worker','run','tests','seal'])
    parser.add_argument('kind',nargs='?');args=parser.parse_args()
    if args.action=='run':supervise(args.kind)
    elif args.action in ('corpus','worker','tests'):globals()[args.action](args.kind)
    else:globals()[args.action]()
