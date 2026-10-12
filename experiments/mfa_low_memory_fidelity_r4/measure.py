"""Bounded synthetic allocation/timing measurements, never scientific scores."""
import time, gc, tracemalloc, numpy as np
from scipy.linalg import inv
from .fidelity import low,model,initialize,synthetic

def fixture(kind,optimized,D=256,N=1024,K=4,r=16):
    rng=np.random.default_rng(42490011)
    cn=lambda shape:np.asarray((rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2),dtype=np.complex128)
    Z,H,V=cn((D,N)),cn((r,N)),cn((D,N));q=rng.uniform(.1,1,N);base=np.eye(r,dtype=np.complex128)*.7
    if kind=='loading':
        fn=(lambda:low.loading_numerator(Z,H,q)) if optimized else (lambda:np.dot(Z[:,None,:]*H[None,:,:].conj(),q))
        largest=16*r*N if optimized else 16*D*r*N;allocs=3 if optimized else 3
    elif kind=='latent':
        fn=(lambda:low.latent_moment(H,q,base)) if optimized else (lambda:np.dot(base[:,:,None]+H[:,None,:]*H[None,:,:].conj(),q))
        largest=16*r*N if optimized else 16*r*r*N;allocs=5 if optimized else 4
    elif kind=='quadratic':
        fn=(lambda:low.real_quadratic(Z,V)) if optimized else (lambda:np.real(np.sum(Z.conj()*V,axis=0)))
        largest=8*N if optimized else 16*D*N;allocs=3 if optimized else 3
    elif kind=='residual':
        fn=(lambda:low.residual_moment(Z,V,q)) if optimized else (lambda:np.real(np.dot((Z-V)*Z.conj(),q)/q.sum()))
        largest=8*D*N if optimized else 16*D*N;allocs=7 if optimized else 5
    else:raise ValueError(kind)
    return fn,{'D':D,'N':N,'K':K,'r':r,'largest_expression_temporary_bytes':largest,
        'source_named_major_allocation_count_estimate':allocs,'count_scope':'expression-level source allocations; views and internal BLAS allocations excluded'}

def measure(kind,optimized,process_memory):
    from threadpoolctl import threadpool_limits,threadpool_info
    if kind.endswith('_EM'):
        D,N,K,r=(64,256,2,8) if kind=='small_EM' else (256,1024,4,16)
        x=synthetic(N,D,K,r,42490021);seed=42490022
        def prepare():return model(K,r,optimized)
        def evaluate(m):
            from src.baselines.mfa_reference_adapter import fit_reference
            fit_reference(m,x,seed);return m
        metadata={'D':D,'N':N,'K':K,'r':r,'largest_expression_temporary_bytes':max(16*K*D*D,16*N*D) if optimized else 16*D*r*N,
            'source_named_major_allocation_count_estimate':None,'count_scope':'complete original fit; source per-stage inventory reported separately'}
    else:
        fn,metadata=fixture(kind,optimized);prepare=lambda:None;evaluate=lambda _:fn()
    rows=[]
    for block in range(3):
        obj=prepare();gc.collect();before=process_memory();t0=time.perf_counter_ns();result=evaluate(obj);elapsed=time.perf_counter_ns()-t0;after=process_memory()
        iterations=None if not kind.endswith('_EM') else len(result.L_all)
        del result,obj;gc.collect()
        # Allocation tracing is a separate untimed call to avoid benchmark contamination.
        obj=prepare();tracemalloc.start();result=evaluate(obj);live,peak=tracemalloc.get_traced_memory();snapshot=tracemalloc.take_snapshot();tracemalloc.stop()
        count=sum(s.count for s in snapshot.statistics('traceback'))
        del result,obj,snapshot;gc.collect()
        rows.append({'kind':kind,'implementation':'low_memory' if optimized else 'original','block':block,'threads':1,
            'wall_seconds':elapsed/1e9,'iterations':iterations,'RSS_before':before['working_set_bytes'],
            'RSS_after':after['working_set_bytes'],'process_peak_RSS':after['peak_working_set_bytes'],
            'private_commit_after':after['private_bytes'],'process_peak_commit':after['peak_commit_bytes'],
            'traced_live_bytes':live,'traced_peak_bytes':peak,'traced_live_allocation_blocks':count,
            'total_allocation_events':'NOT_MEASURED; tracemalloc snapshot counts live blocks only',
            'tracing_scope':'Python and NumPy registered allocations; native BLAS/allocator retention not fully observed',
            'timing_scope':'expression only or complete original fit including initialization, stop and public parameter copies; excludes diagnostics/tracing/serialization',
            'actual_threadpools':threadpool_info(),**metadata})
    return rows
