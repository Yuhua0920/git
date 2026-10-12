"""T18A experiment protocol and frozen pilot-only estimator composition.

New environments are data-generation/reporting metadata only. No new correction
module is defined; deployed inference reuses T16E/F/G functions unchanged.
"""
import copy
import numpy as np
from scipy.linalg import cholesky
from src.cdl_benchmark import OfficialCDL, ARRAY, SYSTEM, SYMBOL_SECONDS
from src.pilot_cv import causal_windows, deployment
from src.statistics_mixing import objective_moments, hard_selection, convex_mixing, geometry
from src.lowrank_modules import factored_affine_predict

INTERPOLATION = ('CDL-C-60', 'CDL-C-90', 'CDL-B-60', 'CDL-B-90')
EXTRAPOLATION = ('CDL-C-180', 'CDL-B-180')
ENVIRONMENTS = INTERPOLATION + EXTRAPOLATION
SNRS = (5, 10, 20)
POWER = 9.941134822523972
VARIANCES = {5: 3.1436628565989504, 10: .9941134822523973, 20: .09941134822523973}
METHODS = ('source', 'hard', 'convex', 'dense-affine', 'lr4-affine', 'matched', 'oracle-affine')
PRIMARY_FRAMES = tuple(range(4, 10))
RANK = 4
C6 = 'CONTRADICTED_BY_CURRENT_MECHANISM_TEST'
C7 = 'CONTRADICTED/CANCELLATION_DOMINATED'
EXPERIMENT_TYPE = 'frozen-pipeline broader interpolation and extrapolation validation of pilot-driven rank-4 affine LMMSE mismatch adaptation'
ROLES = ('sionna_seed', 'torch_seed', 'numpy_seed', 'bits_seed_tx0', 'bits_seed_tx1', 'noise_seed')


def seeds(name, split, trajectory, frame=0):
    """Bijective uint32 allocation, fixed before runs; no seed search/collision.

    Channel roles are trajectory-scoped. Communication roles are frame-scoped.
    T18A environment and split ranges are disjoint; existing namespaces unchanged.
    """
    limits = {'train_dev': range(500), 'validation_dev': range(500, 600)}
    if name not in ENVIRONMENTS or split not in limits or trajectory not in limits[split] or frame not in range(10):
        raise ValueError('only six declared T18A environments and split IDs permitted')
    base = 2000000000 + ENVIRONMENTS.index(name)*1000000 + (1 if split == 'train_dev' else 2)*100000 + trajectory*100
    return {role: base + j*10 + (0 if j <= 3 else frame) for j, role in enumerate(ROLES, 1)}


def definition(name):
    if name not in ENVIRONMENTS:
        raise ValueError('undeclared environment')
    _, profile, speed = name.split('-')
    system, array = copy.deepcopy(SYSTEM), copy.deepcopy(ARRAY)
    system['speed_kmh'] = float(speed)
    array['ut_velocity_mps'] = [float(speed)/3.6, 0., 0.]
    return dict(profile=profile, system=system, array=array, ofdm_channel_normalize=False,
                dF=16, spacing_time=4, symbol_wise_abstraction=True, within_symbol_ICI=False)


def manifest_rows():
    return [dict(environment=name, namespace=f'T18A_{name}_v1', split=split,
                 trajectory_id=i, frame_id=j, start_symbol=16*j, end_symbol=16*j+15,
                 channel_seed_scope='trajectory', communication_seed_scope='frame',
                 communication_generated=(split == 'validation_dev'), **seeds(name, split, i, j))
            for name in ENVIRONMENTS for split, ids in (('train_dev', range(500)), ('validation_dev', range(500,600)))
            for i in ids for j in range(10)]


class GeneralizationCDL(OfficialCDL):
    """Original calibrated call/export; only approved profile/speed/seeds vary."""
    def __init__(self, name, device=None):
        super().__init__(device)
        from sionna.phy.channel.tr38901 import CDL
        self.name = name
        s = definition(name)
        self.model = CDL(model=s['profile'], delay_spread=100e-9, carrier_frequency=3.5e9,
            ut_array=self.array, bs_array=self.array, direction='downlink',
            ut_orientation=ARRAY['ut_orientation_radians'], bs_orientation=ARRAY['bs_orientation_radians'],
            ut_velocity=s['array']['ut_velocity_mps'], min_speed=s['system']['speed_kmh']/3.6,
            max_speed=s['system']['speed_kmh']/3.6, normalize_delays=True,
            precision='double', device=self.device, spec_version='19.2')

    def generate(self, trajectory, *, split):
        import torch
        from sionna.phy import config
        from sionna.phy.channel import cir_to_ofdm_channel
        s = seeds(self.name, split, trajectory)
        config.seed=s['sionna_seed']; torch.manual_seed(s['torch_seed'])
        self.numpy_rng=np.random.default_rng(s['numpy_seed'])
        with torch.no_grad():
            a,tau=self.model(batch_size=1,num_time_steps=160,sampling_frequency=1/SYMBOL_SECONDS)
            hf=cir_to_ofdm_channel(self.frequencies,a,tau,normalize=False)
        h=hf[0,0,:,0,:,:,:].permute(0,1,3,2).cpu().numpy().astype(np.complex128,copy=True)
        if h.shape!=(2,2,64,160) or not np.isfinite(h).all():
            raise ValueError('invalid CDL export')
        return h

    def provenance(self):
        import hashlib
        from pathlib import Path
        from sionna.phy.channel.tr38901 import models
        p=super().provenance()
        p['source_hashes']={k:v for k,v in p['source_hashes'].items() if not k.endswith('CDL-C.json')}
        path=Path(models.parameter_file(f"CDL-{definition(self.name)['profile']}.json", '19.2'))
        p['source_hashes'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        p.update(profile=f"CDL-{definition(self.name)['profile']}", environment_definition=definition(self.name))
        return p


def deployed(pilots, arrays, snr, cv, profile_factor, mobility_factor):
    """Pilot observations only; target/label/truth/oracle absent from signature."""
    if profile_factor.rank != RANK or mobility_factor.rank != RANK:
        raise ValueError('frozen rank4 required')
    result=[]
    for frame, values in causal_windows(pilots, 5):
        moments=objective_moments(cv, values)
        hard,hd=hard_selection(*moments)
        convex,cd=convex_mixing(*moments)
        affine,ad=cv.fit(values, modules=('P','M'))
        x=values[-1]-arrays['source_mean_p']
        args=(values[-1],arrays['source_mean_p'],arrays['source_mean_d'],arrays[f'Wsource_{snr}'])
        predictions=dict(source=arrays['source_mean_d']+x@arrays[f'Wsource_{snr}'].T,
            hard=arrays['source_mean_d']+x@arrays[('Wsource','WP','WM')[hd['selected_index']]+f'_{snr}'].T,
            convex=deployment(*args,arrays[f'DeltaWP_{snr}'],arrays[f'DeltaWM_{snr}'],convex,('P','M')),
            **{'dense-affine':deployment(*args,arrays[f'DeltaWP_{snr}'],arrays[f'DeltaWM_{snr}'],affine,('P','M')),
               'lr4-affine':factored_affine_predict(*args,affine,profile_factor,mobility_factor)})
        if any(not np.isfinite(p).all() for p in predictions.values()):
            raise ValueError('nonfinite deployed prediction')
        result.append(dict(frame_id=frame,window_start=frame-4,predictions=predictions,
            coefficients={'hard':hard,'convex':convex,'affine':affine},
            diagnostics={'hard':hd,'convex':cd,'affine':ad},
            dense_coefficients=affine,lr4_coefficients=affine))
    return result


def oracle_coordinates(target, arrays, snr):
    """Evaluation-only T16F two-real-coordinate target-weighted objective.

    Rank check uses the Hermitian Gram eigenvalues, avoiding any SVD path.
    Source means remain the final evaluation means, as in T16F fair controls.
    """
    weight=target.prepare(VARIANCES[snr]).weight
    lower=cholesky(target.rpp+VARIANCES[snr]*np.eye(64),lower=True)
    response=(weight-arrays[f'Wsource_{snr}'])@lower
    columns=[arrays[f'DeltaW{m}_{snr}']@lower for m in ('P','M')]
    gram=np.array([[np.vdot(a,b).real for b in columns] for a in columns])
    rhs=np.array([np.vdot(a,response).real for a in columns])
    eigen=np.linalg.eigvalsh(gram)
    if eigen[0] <= 1e-12*eigen[-1]:
        raise ValueError('oracle normal equations rank deficient; no regularization')
    coefficients=np.linalg.solve(gram,rhs)
    residual=response-sum(c*a for c,a in zip(coefficients,columns))
    return coefficients,dict(**geometry(coefficients),gram=gram.tolist(),rhs=rhs.tolist(),
        weighted_filter_excess_energy_per_frame=float(np.vdot(residual,residual).real),
        normal_relative_residual=float(np.linalg.norm(gram@coefficients-rhs)/np.linalg.norm(rhs)),
        objective='target-weighted centered filter error, not mean-bias optimization',
        source_means_used_for_actual_evaluation=True,deployable=False)


def metrics(source, matched, dense, lowrank, convex, oracle):
    if not np.isfinite([source,matched,dense,lowrank,convex,oracle]).all():
        raise ValueError('finite pooled dB NMSE required')
    gap=source-matched; recovery=source-dense; capacity=source-oracle
    return dict(matched_gap_db=gap,dense_affine_recovery_db=recovery,
        dense_affine_closure=recovery/gap if gap != 0 else None,
        oracle_affine_recovery_db=capacity,pilotcv_efficiency=recovery/capacity if capacity != 0 else None,
        compression_penalty_db=lowrank-dense,affine_vs_convex_gain_db=convex-dense,
        lr4_vs_convex_gain_db=convex-lowrank,lr4_vs_source_gain_db=source-lowrank)


def scientific_gate(rows):
    indexed={(r['environment'],r['snr_db']):r for r in rows}
    if len(rows)!=18 or set(indexed)!={(n,s) for n in ENVIRONMENTS for s in SNRS}:
        raise ValueError('all eighteen unique environment/SNR conditions required')
    primary=[indexed[n,s] for n in INTERPOLATION for s in (10,20)]
    composed=[indexed[n,s] for n in ('CDL-B-60','CDL-B-90') for s in (10,20)]
    stress=[indexed[n,s] for n in EXTRAPOLATION for s in (10,20)]
    supported=all(r['pilotcv_efficiency'] is not None and r['dense_affine_closure'] is not None
        and r['pilotcv_efficiency']>=.8 and r['compression_penalty_db']<=.5 and r['dense_affine_closure']>=.7 for r in primary)
    replicated=all(r['lr4_vs_convex_gain_db']>=1. for r in composed)
    weak=all(r['lr4_vs_convex_gain_db']<.25 for r in composed)
    return dict(frozen_pipeline_interpolation_supported=bool(supported),
        affine_composition_beyond_convex_hull_replicated=bool(replicated),
        affine_composition_replication_weak=bool(weak),affine_composition_replication_partial=bool(not replicated and not weak),
        outside_convex_hull_geometry_replicated=bool(all(r['outside_simplex_fraction']>=.75 for r in composed)),
        extrapolation_stress_promising=bool(all(r['lr4_vs_source_gain_db']>0 and r['lr4_vs_convex_gain_db']>=-.5 for r in stress)),
        extrapolation_failure_observed=bool(any(r['lr4_vs_source_gain_db']<0 for r in stress)),
        narrow_empirical_observation_downgrade=bool(not supported and not replicated),
        broader_factor_validation_may_follow=bool(supported and replicated))
