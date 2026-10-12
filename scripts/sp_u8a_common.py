"""Post-validation receiver audit. Frozen predictions replay saved coordinates only."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import csv
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import sp_u7b_common as frozen
from src.cdl_benchmark import frames, observation_geometry, array_hash, FrozenStatistics, EmpiricalEstimator
from src.link import (assemble_pilot_data_frame, apply_frequency_mimo_channel, estimate_ls_linear_channel,
    detect_data_zf, demodulate_data_bits, extract_reference_data_bits)
from src.lowrank_modules import factored_affine_predict
from src.modulation import qpsk_modulate
R1=ROOT/'outputs/sp_u7b_r1'
OUT=ROOT/'outputs/sp_u8a'
NOTICE='This diagnostic does not alter or re-test the preregistered SP-U7B-R1 primary Source-vs-LR4 NMSE result.'
METHODS=('Perfect','LS','Source','LR4','Matched')
IDS=(0,1,335,573,718,150,797)
SNRS=frozen.SNRS
VARIANCES=frozen.VARIANCES
_opaque=False
_installed=False
COUNTERS=dict(regeneration_attempts=0,coordinate_refit_attempts=0,rng_attempts=0,forbidden_reads=0)

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,value):
    value={'notice':NOTICE,**value}
    Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
    global _opaque
    old=_opaque;_opaque=True
    try:
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        return h.hexdigest()
    finally:_opaque=old
def csv_write(p,rows):
    with Path(p).open('w',encoding='utf8',newline='') as f:
        f.write('# '+NOTICE+'\n')
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def csv_read(p):
    with Path(p).open(encoding='utf8',newline='') as f:
        assert f.readline().strip()=='# '+NOTICE
        return list(csv.DictReader(f))

def install_guard():
    global _installed
    if _installed:return
    allow={str((ROOT/p).resolve()).lower() for p in read(frozen.PROTOCOL)['deployable']['objects']}
    outroot=str((ROOT/'outputs').resolve()).lower()+os.sep
    own=str(OUT.resolve()).lower()+os.sep
    used=str(R1.resolve()).lower()+os.sep
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=str(Path(os.fsdecode(args[0])).resolve()).lower()
            mode=args[1] if len(args)>1 else 'r'; flags=args[2] if len(args)>2 else 0
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR))
            if p.startswith(outroot):
                if writing and not p.startswith(own):raise PermissionError('read-only historical evidence')
                if not writing and not (p.startswith(own) or p in allow or (p.startswith(used) and (_opaque or 'seed' not in Path(p).name))):
                    COUNTERS['forbidden_reads']+=1;raise PermissionError('outcome/seed firewall')
        if event in ('os.remove','os.rename'):
            for a in args[:2 if event=='os.rename' else 1]:
                if isinstance(a,(str,bytes,os.PathLike)):
                    p=str(Path(os.fsdecode(a)).resolve()).lower()
                    if p.startswith(outroot) and not p.startswith(own):raise PermissionError('historical evidence immutable')
    sys.addaudithook(hook)
    def forbidden(key):
        def fail(*a,**k):
            COUNTERS[key]+=1;raise RuntimeError('SP-U8A forbids new scientific generation, refitting and RNG')
        return fail
    from src import link,numerics,cdl_benchmark,pilot_cv
    frozen.communication=forbidden('regeneration_attempts')
    frozen.infer_pilots=forbidden('coordinate_refit_attempts')
    link.generate_master_qpsk=forbidden('regeneration_attempts')
    numerics.complex_gaussian_noise=forbidden('regeneration_attempts')
    cdl_benchmark.OfficialCDL.generate=forbidden('regeneration_attempts')
    pilot_cv.PilotCV.fit=forbidden('coordinate_refit_attempts')
    for n in ('default_rng','seed','RandomState'):
        setattr(np.random,n,forbidden('rng_attempts'))
    _installed=True

def snapshot():
    # Hash-only preservation audit, never convert historical outcome arrays.
    import subprocess
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    paths={ROOT/p for p in tracked if (ROOT/p).is_file() and not p.startswith('outputs/')}
    paths.update(p for parent in ('scripts','tests','configs','src') for p in (ROOT/parent).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and 'sp_u8a' not in p.name)
    paths.update(p for p in R1.rglob('*') if p.is_file())
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    return dict(hashes=hashes,files=len(hashes),R1_files=sum(k.startswith('outputs/sp_u7b_r1/') for k in hashes))
def verify_unchanged():
    before=read(OUT/'before.json')['hashes']
    bad=[p for p,h in before.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    added=set(p.relative_to(ROOT).as_posix() for p in R1.rglob('*') if p.is_file())-set(before)
    if bad or added:raise RuntimeError('protected evidence byte drift '+repr((bad,sorted(added))))
    return dict(protected_files=len(before),changed=0,R1_added=0,primary_impact='NONE')

def load_methods():
    a,f,_unused=frozen.load_deployables()
    with np.load(R1/'matched_reference/statistics.npz',allow_pickle=False) as z:
        st=FrozenStatistics(z['mean_p'],z['mean_d'],z['rpp'],z['rdp'],tuple(z['train_trajectory_ids']),int(z['samples']))
        matched={s:EmpiricalEstimator(st,z[f'W_matched_{s}'],VARIANCES[s],0.) for s in SNRS}
    return a,f,matched

def ingredients(i):
    p=R1/f'calls/EVAL/{i:04d}'
    w=read(p/'call.json')
    with np.load(p/'H.npz',allow_pickle=False) as z:saved=z['H']
    h=np.ndarray(saved.shape,dtype=saved.dtype,buffer=np.empty(saved.size,saved.dtype),strides=tuple(w['export_strides_bytes']))
    h[:]=saved;h.flags.writeable=False
    if array_hash(h)!=w['H_array_sha256']:raise RuntimeError('Perfect H identity')
    with np.load(R1/f'evaluation/comm_{i:04d}.npz',allow_pickle=False) as z:
        comm=tuple(zip(z['bits'],z['symbols'],z['unit_noise']))
    with np.load(R1/f'evaluation/raw_{i:04d}.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    for trio in comm:
        for v in trio:v.flags.writeable=False
    return frames(h),comm,raw,read(R1/f'evaluation/proof_{i:04d}.json')

def replay_prediction(pilot,linear,sparse,s,coef,a,f,matched):
    # Exact frozen prediction functions and operation order; no coordinate solve.
    x=pilot-a['source_mean_p']
    pred={'LS':linear.ravel()[observation_geometry().data],
        'Source':a['source_mean_d']+x@a[f'Wsource_{s}'].T,
        'LR4':factored_affine_predict(pilot,a['source_mean_p'],a['source_mean_d'],a[f'Wsource_{s}'],coef,f['P',s,4],f['M',s,4]),
        'Matched':matched[s].estimate(pilot)}
    grids={'LS':linear}
    g=observation_geometry()
    for m in ('Source','LR4','Matched'):
        v=sparse.copy().ravel();v[g.data]=pred[m];grids[m]=v.reshape(2,2,64,16)
    return pred,grids

def count_bits(y,h,mask,ref):
    z=detect_data_zf(y,h,mask)
    b=demodulate_data_bits(z*np.sqrt(2),mask)
    return sum(int(np.count_nonzero(v!=r)) for v,r in zip(b,ref)),z

def fixtures():
    from src.detectors import zf_detect
    bit=np.array([[[[0,0]]],[[[1,1]]]],np.uint8)
    sym=np.stack([qpsk_modulate(b.ravel()).reshape(1,1) for b in bit])/np.sqrt(2)
    matrices={'identity':np.eye(2),'diagonal':np.diag([1+2j,.7-.3j]),
        'full_complex':np.array([[1+.4j,.3-.8j],[-.4+.6j,1.2+.2j]])}
    algebra=[]
    for name,m in matrices.items():
        h=m[:,:,None,None];y=apply_frequency_mimo_channel(h,sym)
        err,z=count_bits(y,h,np.ones((2,1,1),bool),tuple(b.ravel() for b in bit))
        residual=float(np.max(abs(z-sym)))
        assert err==0 and residual<1e-12
        algebra.append(dict(fixture=name,bit_errors=err,max_symbol_residual=residual))
    # Frequency/time/TX/RX signatures, explicit independent y sum and flatten roundtrip.
    k,t=np.indices((7,5));b=np.empty((2,7,5,2),np.uint8)
    for q in range(2):b[q,...,0]=((3*k+2*t+q)%5<2);b[q,...,1]=((2*k+3*t+2*q)%7<3)
    x=np.stack([qpsk_modulate(v.ravel()).reshape(7,5) for v in b])/np.sqrt(2)
    h=np.empty((2,2,7,5),complex)
    base=np.array([[1+.4j,.3-.8j],[-.4+.6j,1.2+.2j]])
    for r in range(2):
        for q in range(2):h[r,q]=base[r,q]*(1+.013*k+.019*t+.011j*(r+q+k-t))
    y=np.stack([sum(h[r,q]*x[q] for q in range(2)) for r in range(2)])
    mask=np.ones((2,7,5),bool);ref=tuple(v.ravel() for v in b)
    assert np.array_equal(h,h.ravel().reshape(h.shape))
    assert np.allclose(y,apply_frequency_mimo_channel(h,x),rtol=0,atol=1e-15)
    err,z=count_bits(y,h,mask,ref);assert err==0 and np.max(abs(z-x))<1e-12
    mutants={'TX_swap':(y,h[:,::-1]),'RX_swap':(y[::-1],h),'H_transpose':(y,h.swapaxes(0,1)),
        'H_Hermitian':(y,h.swapaxes(0,1).conj()),'KT_swap':(y.reshape(2,-1)[:,::-1].reshape(y.shape),h)}
    ordering={}
    for name,(yy,hh) in mutants.items():
        errs,zz=count_bits(yy,hh,mask,ref)
        assert errs>0 and np.max(abs(zz-x))>1e-6,(name,errs)
        ordering[name]=errs
    # Deterministic standardized Gaussian quantile atoms, no seed or random draw.
    from scipy.special import ndtri
    n=4096;u=(np.arange(n)+.5)/n
    z=(ndtri(u)+1j*ndtri(u[(np.arange(n)*73+19)%n]))/np.sqrt(2)
    z=np.stack([z,z[(np.arange(n)*37+31)%n]])[:,:,None]
    z/=np.sqrt(np.mean(abs(z)**2,axis=(1,2),keepdims=True))
    bb=np.zeros((2,n,1,2),np.uint8)
    xx=np.full((2,n,1),(1+1j)/2,complex);mask=np.ones(xx.shape,bool)
    noise=[]
    for name,m in (('identity',np.eye(2)),('diagonal',np.diag([.8+.2j,1.2-.3j]))):
        hh=np.broadcast_to(m[:,:,None,None],(2,2,n,1))
        clean=apply_frequency_mimo_channel(hh,xx);counts=[]
        for s in (10,20,30):
            # Exercise frozen SNR/P_ref convention without matching a theoretical curve.
            variance=9.941134822523972/10**(s/10)
            errs,_=count_bits(clean+np.sqrt(variance)*z,hh,mask,tuple(v.ravel() for v in bb))
            counts.append(errs);noise.append(dict(fixture=name,snr_db=s,variance=variance,
                standardized_atom_energy=float(np.mean(abs(z)**2)),bit_errors=errs,total_bits=bb.size,BER=errs/bb.size))
        assert counts[0]>counts[1]>=counts[2]
    return dict(algebra=algebra,ordering_mutation_bit_errors=ordering,noise_sweep=noise,
        production_detector='np.linalg.solve per square RE; no pseudoinverse, threshold or regularization',PASS=True)

def conditioning(h,mask):
    matrices=h[:,:,mask[0]].transpose(2,0,1)
    sv=np.linalg.svd(matrices,compute_uv=False)
    with np.errstate(divide='ignore',invalid='ignore'):
        kap=sv[:,0]/sv[:,1];log=np.log10(kap)
    # Exact same square solve as frozen detector, with identity RHS for norm diagnostic.
    eq=np.linalg.solve(matrices,np.broadcast_to(np.eye(2),matrices.shape))
    enhancement=np.sum(abs(eq)**2,axis=(1,2))
    return sv[:,0],sv[:,1],kap,log,enhancement,eq
