"""T18B fixed namespaces, energy-domain paired trajectory bootstrap and gates.

No estimator, covariance, basis or target optimization is defined here.
"""
import numpy as np
from src import frozen_affine_validation as old
from src.cdl_benchmark import array_hash

ENVIRONMENTS = old.INTERPOLATION
SNRS = (10, 20)
REPLICATES = tuple(range(1, 6))
TRAJECTORIES = 100
BOOTSTRAPS = 10000
BOOTSTRAP_SEED_BASE = 20261802
METHODS = old.METHODS
PRIMARY_METHODS = ('source', 'convex', 'dense-affine', 'lr4-affine', 'matched')
EXPERIMENT_TYPE = 'independent validation replication and trajectory-level uncertainty analysis of the frozen rank-4 affine LMMSE adaptation pipeline'
C8 = "Under the fixed C60/C90/B60/B90 conditions, the frozen LR4 affine estimator's performance and composed-shift advantage are stable to independent validation sampling."


def seeds(name, replicate, trajectory, frame=0):
    if name not in ENVIRONMENTS or replicate not in REPLICATES or trajectory not in range(TRAJECTORIES) or frame not in range(10):
        raise ValueError('only fixed T18B validation namespaces allowed')
    base = 2100000000 + ENVIRONMENTS.index(name)*1000000 + replicate*100000 + trajectory*100
    return {role: base + j*10 + (0 if j <= 3 else frame) for j, role in enumerate(old.ROLES, 1)}


def bootstrap_seed(name, snr):
    if name not in ENVIRONMENTS or snr not in SNRS:
        raise ValueError('undeclared bootstrap condition')
    return BOOTSTRAP_SEED_BASE + ENVIRONMENTS.index(name)*100 + snr


def manifest_rows():
    return [dict(environment=n, namespace=f'T18B_{n}_r{r}_v1', split='validation_dev',
        replicate=r, trajectory_id=i, frame_id=j, start_symbol=16*j, end_symbol=16*j+15,
        channel_seed_scope='trajectory', communication_seed_scope='frame',
        **seeds(n,r,i,j)) for n in ENVIRONMENTS for r in REPLICATES for i in range(100) for j in range(10)]


class ReplicationCDL(old.GeneralizationCDL):
    """Exact inherited CDL configuration/export, with new declared random streams."""
    def generate(self, trajectory, *, replicate):
        import torch
        from sionna.phy import config
        from sionna.phy.channel import cir_to_ofdm_channel
        s=seeds(self.name,replicate,trajectory)
        config.seed=s['sionna_seed'];torch.manual_seed(s['torch_seed'])
        self.numpy_rng=np.random.default_rng(s['numpy_seed'])
        with torch.no_grad():
            a,tau=self.model(batch_size=1,num_time_steps=160,sampling_frequency=1/old.SYMBOL_SECONDS)
            hf=cir_to_ofdm_channel(self.frequencies,a,tau,normalize=False)
        h=hf[0,0,:,0,:,:,:].permute(0,1,3,2).cpu().numpy().astype(np.complex128,copy=True)
        if h.shape!=(2,2,64,160) or not np.isfinite(h).all():raise ValueError('invalid CDL export')
        return h


def pooled(error, truth):
    error=np.asarray(error,dtype=np.float64);truth=np.asarray(truth,dtype=np.float64)
    if error.ndim!=2 or truth.shape!=(error.shape[0],) or not np.isfinite(error).all() or not np.isfinite(truth).all() or np.any(error<=0) or np.any(truth<=0):
        raise ValueError('positive finite trajectory energy matrix and shared truth required')
    # Preserve declared sequential trajectory reduction; do not average dB.
    denominator=sum(float(x) for x in truth)
    numerators=np.array([sum(float(x) for x in error[:,j]) for j in range(error.shape[1])])
    return numerators,denominator,10*np.log10(numerators/denominator)


def metric_columns(db):
    """Works for a single pooled vector or paired bootstrap rows."""
    v={m:db[...,j] for j,m in enumerate(METHODS)}
    gap=v['source']-v['matched'];capacity=v['source']-v['oracle-affine']
    if np.any(gap==0) or np.any(capacity==0):raise ValueError('undefined closure/efficiency; no silent clipping')
    return {**{f'nmse_{m.replace("-","_")}_db':v[m] for m in METHODS},
        'dense_affine_closure':(v['source']-v['dense-affine'])/gap,
        'pilotcv_efficiency':(v['source']-v['dense-affine'])/capacity,
        'compression_penalty_db':v['lr4-affine']-v['dense-affine'],
        'affine_vs_convex_gain_db':v['convex']-v['dense-affine'],
        'lr4_vs_convex_gain_db':v['convex']-v['lr4-affine']}


def paired_resamples(error,truth,indices):
    """Each row resamples whole trajectory energies with identical method indices."""
    error=np.asarray(error,dtype=np.float64);truth=np.asarray(truth,dtype=np.float64)
    pooled(error,truth)
    indices=np.asarray(indices)
    if indices.ndim!=2 or indices.shape[1]!=len(truth) or not np.issubdtype(indices.dtype,np.integer) or np.any(indices<0) or np.any(indices>=len(truth)):
        raise ValueError('one complete trajectory index row per bootstrap resample required')
    values=np.empty((len(indices),error.shape[1]))
    for start in range(0,len(indices),128):
        take=indices[start:start+128]
        values[start:start+128]=10*np.log10(error[take].sum(axis=1)/truth[take].sum(axis=1)[:,None])
    return values


def bootstrap(error,truth,name,snr):
    if np.asarray(error).shape!=(500,len(METHODS)) or np.asarray(truth).shape!=(500,):
        raise ValueError('five new replicates, 500 independent trajectories required')
    seed=bootstrap_seed(name,snr)
    indices=np.random.default_rng(seed).integers(0,500,size=(BOOTSTRAPS,500),dtype=np.int64)
    db=paired_resamples(error,truth,indices);point=metric_columns(pooled(error,truth)[2])
    rows=[]
    for metric,values in metric_columns(db).items():
        lo,hi=np.percentile(values,[2.5,97.5])
        rows.append(dict(environment=name,snr_db=snr,metric=metric,point_estimate=float(point[metric]),
            lower_2_5=float(lo),upper_97_5=float(hi),bootstrap_samples=BOOTSTRAPS,bootstrap_seed=seed,
            bootstrap_unit='trajectory',trajectories=500,confidence_interval='95_percentile',
            shared_indices_sha256=array_hash(indices)))
    return rows,dict(seed=seed,indices_sha256=array_hash(indices),shape=list(indices.shape),dtype=str(indices.dtype)),db


def passes(r):
    return r['dense_affine_closure']>=.70 and r['pilotcv_efficiency']>=.80 and r['compression_penalty_db']<=.50


def scientific_gate(replicate_metrics,pooled_metrics,intervals,weights):
    expected={(n,s,r) for n in ENVIRONMENTS for s in SNRS for r in REPLICATES}
    index={(r['environment'],r['snr_db'],r['replicate']):r for r in replicate_metrics}
    points={(r['environment'],r['snr_db']):r for r in pooled_metrics}
    cis={(r['environment'],r['snr_db'],r['metric']):r for r in intervals}
    geo={(r['environment'],r['snr_db'],r['replicate']):r for r in weights}
    if len(index)!=len(replicate_metrics) or set(index)!=expected or set(geo)!=expected or set(points)!={(n,s) for n in ENVIRONMENTS for s in SNRS}:
        raise ValueError('complete fixed replication grid required')
    composed=[(n,s) for n in ENVIRONMENTS if n.startswith('CDL-B') for s in SNRS]
    supported=all(all(index[n,s,r]['lr4_vs_convex_gain_db']>0 for r in REPLICATES)
        and cis[n,s,'lr4_vs_convex_gain_db']['lower_2_5']>0 for n,s in composed)
    unstable=any(sum(index[n,s,r]['lr4_vs_convex_gain_db']<=0 for r in REPLICATES)>=3
        or points[n,s]['lr4_vs_convex_gain_db']<=0 for n,s in composed)
    counts={f'{n}/{s}':sum(passes(index[n,s,r]) for r in REPLICATES) for n in ENVIRONMENTS for s in SNRS}
    interpolation=all(v>=4 for v in counts.values()) and all(passes(r) for r in points.values())
    # <4 passing means >=2 independent replicate failures under the original thresholds.
    performance_unstable=any(v<4 for v in counts.values())
    warning=any(cis[n,s,'compression_penalty_db']['upper_97_5']>.50 for n in ENVIRONMENTS for s in SNRS)
    geometry=all(geo[n,s,r]['outside_simplex_fraction']>=.75 for n,s in composed for r in REPLICATES)
    c8='CONTRADICTED_BY_REPLICATION' if unstable or performance_unstable else (
        'SUPPORTED_IN_FIXED_ENVIRONMENTS' if supported and interpolation and not warning else 'INCONCLUSIVE')
    return dict(positive_affine_advantage_statistically_replicated=bool(supported),
        replication_instability_observed=bool(unstable),replication_inconclusive=bool(not supported and not unstable),
        interpolation_threshold_stability=bool(interpolation),interpolation_replicate_pass_counts=counts,
        performance_replication_instability_observed=bool(performance_unstable),
        rank4_compression_uncertainty_warning=bool(warning),outside_simplex_geometry_stable=bool(geometry),
        c8_claim_status=c8,geometry_used_for_c8=False,
        post_replication_materiality_diagnostic=bool(all(points[n,s]['lr4_vs_convex_gain_db']>=1 for n,s in composed)),
        broader_factor_validation_may_be_considered=bool(supported and interpolation),
        stop_broader_factor_validation=bool(not supported),t18a_materiality_status_preserved='PARTIAL')
