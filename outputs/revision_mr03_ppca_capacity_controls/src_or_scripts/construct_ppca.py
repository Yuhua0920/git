"""One blocked covariance and exact shared spectra, no EM fit and no V evaluation."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='6'
os.environ['OMP_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,csv,time,ctypes,threading,traceback,hashlib
import numpy as np
from ppca import training_moments,leading_eigenpairs,direct_ppca,conditional_operator,training_logscore,blocked_residual_energy
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'outputs/revision_mr03_ppca_capacity_controls';sys.path.insert(0,str(ROOT))
from src.baselines.mfa_reference_adapter import ProjectPilotInterface
from scripts.sp_u12b4a_adapter import rss
import scipy
def blas_threads(expected=6):
    """Authenticate this construction's threads, without changing frozen B4."""
    answer=[]
    for package in (np,scipy):
        for p in (Path(package.__file__).parent.parent/(package.__name__+'.libs')).glob('*openblas*.dll'):
            lib=ctypes.CDLL(str(p))
            name=next((n for n in ('scipy_openblas_get_num_threads64_','scipy_openblas_get_num_threads','openblas_get_num_threads64_','openblas_get_num_threads') if hasattr(lib,n)),None)
            if name is None:raise ValueError('BLAS_THREAD_QUERY_UNAVAILABLE')
            f=getattr(lib,name);f.restype=ctypes.c_int
            answer.append(dict(library=str(p),threads=f(),expected_threads=expected))
    if len(answer)<2 or any(x['threads']!=expected for x in answer):raise ValueError('CONSTRUCTION_BLAS_THREAD_MISMATCH')
    return answer
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def archive(p,**x):
    with Path(p).open('xb') as f:np.savez_compressed(f,**x)
def csvout(p,rows):
    with Path(p).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
class MEMORYSTATUS(ctypes.Structure):
    _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ('ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual')]
def memory():
    m=MEMORYSTATUS();m.dwLength=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError('MEMORY_QUERY')
    return dict(total_RAM=int(m.ullTotalPhys),available_RAM=int(m.ullAvailPhys),total_commit=int(m.ullTotalPageFile),available_commit=int(m.ullAvailPageFile))
def close(a,b,label):
    a,b=np.asarray(a),np.asarray(b);err=float(np.max(np.abs(a-b)));scale=max(float(np.max(np.abs(a))),float(np.max(np.abs(b))))
    if err>1e-10+1e-10*scale:raise ValueError(label+str((err,scale)))
    return dict(max_abs=err,scaled=err/(1+scale))
def main():
    started=time.monotonic();cfg=read(OUT/'config.json');source=read(OUT/'reports/input_provenance.json');pre=memory();limits=cfg['resource'];stop=threading.Event()
    required=limits['system_headroom_bytes']+2*1024**3
    if pre['available_RAM']<required or pre['available_commit']<required:raise ValueError('INSUFFICIENT_RAM_OR_COMMIT_FOR_EXACT_COVARIANCE')
    for p,h in read(OUT/'reports/execution_freeze.json')['hashes'].items():
        if sha(OUT/p)!=h:raise ValueError('FROZEN_CONSTRUCTION_SOURCE_CHANGED:'+p)
    write(OUT/'reports/spectral_resource_preflight.json',dict(PASS=True,available=pre,resource=rss(),blas=blas_threads(),limits=limits,estimated_peak_bytes=2*1024**3,full_covariance_bytes=4032*4032*16))
    def watch():
        while not stop.wait(.5):
            try:
                usage=rss();available=memory();working=usage['RSS_bytes']
                if working>limits['RSS_limit_bytes'] or time.monotonic()-started>limits['spectral_timeout_seconds'] or min(available['available_RAM'],available['available_commit'])<limits['system_headroom_bytes']:
                    write(OUT/f'reports/spectral_resource_STOP_{time.time_ns()}.json',dict(status='STOP',usage=usage,available=available,elapsed=time.monotonic()-started,partial_results_not_used=True));os._exit(81)
            except BaseException:
                try:write(OUT/f'reports/spectral_monitor_failure_{time.time_ns()}.json',dict(status='STOP',traceback=traceback.format_exc()))
                finally:os._exit(82)
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    try:
        p=ROOT/source['training']['path']
        if sha(p)!=source['training']['sha256']:raise ValueError('FROZEN_P_SHA256')
        rows=np.load(p,mmap_mode='r',allow_pickle=False)
        if rows.shape!=(15000,4032) or rows.dtype!=np.complex128 or p.stat().st_size!=source['training']['bytes']:raise ValueError('FROZEN_P_HEADER')
        print(json.dumps(dict(stage='training_moments',N=15000,d=4032,resource=rss())),flush=True)
        mean,S,moments=training_moments(rows,block_size=256)
        rows._mmap.close();del rows
        interface=ProjectPilotInterface()
        with np.load(ROOT/'outputs/revision_mr01_pooled_lmmse/data/pooled_statistics.npz',allow_pickle=False) as z:pool=dict(z)
        checks=dict(mean_p=close(mean[interface.pilot_positions],pool['mean_p'],'TRAINING_MEAN_P'),mean_d=close(mean[interface.data_positions],pool['mean_d'],'TRAINING_MEAN_D'),rpp=close(S[np.ix_(interface.pilot_positions,interface.pilot_positions)],pool['rpp'],'TRAINING_PILOT_COVARIANCE'),rdp=close(S[np.ix_(interface.data_positions,interface.pilot_positions)],pool['rdp'],'TRAINING_DATA_PILOT_COVARIANCE'))
        print(json.dumps(dict(stage='exact_eigh',moments_seconds=moments['seconds'],training_moment_checks=checks,resource=rss())),flush=True)
        spectrum=leading_eigenpairs(S,max_rank=64)
        print(json.dumps(dict(stage='spectral_models',spectrum_audit=spectrum.audit,resource=rss())),flush=True)
        rows=np.load(p,mmap_mode='r',allow_pickle=False)
        tailstart=time.monotonic();tail=blocked_residual_energy(rows,mean,spectrum.eigenvectors,block_size=256);rawtail=float(spectrum.trace-np.sum(spectrum.eigenvalues))
        rows._mmap.close();del rows
        tailcheck=close(tail,rawtail,'R64_SPECTRAL_TAIL_INDEPENDENT_RESIDUAL')
        tail_audit=dict(blocked_residual_energy=tail,trace_minus_top64=rawtail,check=tailcheck,seconds=time.monotonic()-tailstart)
        archive(OUT/'data/shared_spectrum.npz',mean=mean,eigenvalues=spectrum.eigenvalues,eigenvectors=spectrum.eigenvectors,trace=np.array(spectrum.trace),pilot_positions=interface.pilot_positions,data_positions=interface.data_positions,joint_full=interface.joint_full)
        model_rows=[];op_rows=[];ops={};models=[]
        for r in (16,32,64):
            t=time.monotonic();model=direct_ppca(mean,spectrum,r,residual_floor=1e-6);score=training_logscore(model,mean,S)
            mpath=OUT/f'data/PPCA_DIRECT_R{r}.npz'
            archive(mpath,mean=model.mean,loading=model.loading,nu=np.array(model.nu),declared_rank=np.array(r),effective_rank=np.array(model.audit['effective_loading_rank']))
            write(OUT/f'reports/PPCA_DIRECT_R{r}.json',dict(PASS=True,model_audit=model.audit,training_score=score,model_sha256=sha(mpath),P_sha256=source['training']['sha256'],construction_seconds=time.monotonic()-t,original_EM_entity_status='PENDING_ORIGINAL_MODEL'))
            models.append(dict(method=f'PPCA_DIRECT_R{r}',path=mpath.relative_to(OUT).as_posix(),sha256=sha(mpath),bytes=mpath.stat().st_size))
            model_rows.append(dict(method=f'PPCA_DIRECT_R{r}',rank=r,effective_rank=model.audit['effective_loading_rank'],nu_raw=model.audit['raw_residual_variance'],nu=model.nu,floor_active=model.audit['floor_active'],training_average_frame_logscore=score['mean_frame_logscore'],training_average_complex_dimension_logscore=score['mean_complex_dimension_logscore'],mean_mismatch_norm=score['mean_mismatch_norm'],loading_complex_elements=model.loading.size,mean_complex_elements=model.mean.size,model_array_bytes=model.loading.nbytes+model.mean.nbytes+8,construction_seconds=time.monotonic()-t))
            for s in (10,20,30):
                sigma=cfg['source_noise_power']/10**(s/10);co=conditional_operator(model,interface.pilot_positions,interface.data_positions,sigma);key=f'PPCA_DIRECT_R{r}_{s}'
                ops[key+'_W']=co.W;ops[key+'_mean_p']=co.mean_p;ops[key+'_mean_d']=co.mean_d
                op_rows.append(dict(method=f'PPCA_DIRECT_R{r}',snr_db=s,nu=model.nu,sigma2=sigma,numerical_rank=co.audit['numerical_operator_rank'],rank_upper_bound=min(r,64),solve_relative_residual=co.audit['solve_relative_residual'],seconds=co.audit['seconds'],cached_W_bytes=co.W.nbytes,operator_singular_values=json.dumps(co.audit['operator_singular_values'])))
                if co.audit['numerical_operator_rank']>min(r,64):raise ValueError('CONDITIONAL_RANK_BOUND')
                # Single/batch/cache equality is checked without target truth.
                test=np.stack([model.mean[interface.pilot_positions],model.mean[interface.pilot_positions]+np.arange(64)/100*(1+1j)])
                direct=np.stack([co.estimate(x) for x in test]);close(co.estimate(test),direct,'CONDITIONAL_SINGLE_BATCH')
        scorevals=[r['training_average_frame_logscore'] for r in model_rows]
        if np.any(np.diff(scorevals)<-1e-8*4032):raise ValueError('NESTED_FAMILY_TRAINING_SCORE_INCONSISTENT')
        archive(OUT/'data/conditional_operators.npz',**ops);csvout(OUT/'tables/spectral_models.csv',model_rows);csvout(OUT/'tables/conditional_operators.csv',op_rows)
        # The large S was a bounded in-memory construction aid, not a new model asset.
        del S
        write(OUT/'reports/spectral_construction.json',dict(PASS=True,training=source['training'],training_moments=moments,moment_matches_MR01=checks,spectrum=spectrum.audit,independent_tail_check=tail_audit,models=models,operators_sha256=sha(OUT/'data/conditional_operators.npz'),model_count=3,EM_entity_status='PENDING_ORIGINAL_MODEL',floor_scope='independent feasible-family spectral MLE; author EM clip-per-coordinate before mean is separately documented',all_models_saved_before_V=True,elapsed_seconds=time.monotonic()-started,resource=rss(),available_after=memory(),blas=blas_threads()))
        print(json.dumps(dict(PASS=True,seconds=time.monotonic()-started,models=model_rows,resource=rss())),flush=True)
    finally:stop.set();thread.join(timeout=1)
if __name__=='__main__':
    try:main()
    except Exception as e:
        if (OUT/'reports').is_dir():write(OUT/f'reports/spectral_failure_{time.time_ns()}.json',dict(status='STOP',error=repr(e),traceback=traceback.format_exc(),original_model_unchanged=True))
        raise
