"""T19A fixed delay-factor design and descriptive failure localization.

No deployment solver, correction basis, or model is fitted in this module.
"""
import copy
import numpy as np
from src import frozen_affine_validation as old
from src.cdl_benchmark import OfficialCDL, ARRAY, SYSTEM, SYMBOL_SECONDS

ENVIRONMENTS = tuple(f'CDL-{p}-{v}-{d}' for p in ('C','B') for v in (30,90) for d in (50,200))
PURE = ENVIRONMENTS[:2]
SNRS = (10,20)
METHODS = ('source','hard','convex','dense-affine','lr4-affine','matched','oracle2-affine')
EXPERIMENT_TYPE = 'frozen two-module correction-space challenge under previously unseen nominal delay-spread factors'
C9 = 'The frozen two-module P/M correction space can represent delay-spread-induced mismatch without an explicit delay module.'


def seeds(name,split,trajectory,frame=0):
    allowed={'train_dev':range(500),'validation_dev':range(500,600)}
    if name not in ENVIRONMENTS or split not in allowed or trajectory not in allowed[split] or frame not in range(10):
        raise ValueError('only eight frozen T19A environments and declared split IDs allowed')
    base=2200000000+ENVIRONMENTS.index(name)*1000000+(1 if split=='train_dev' else 2)*100000+trajectory*100
    return {role:base+j*10+(0 if j<=3 else frame) for j,role in enumerate(old.ROLES,1)}


def manifest_rows():
    return [dict(environment=n,namespace=f'T19A_{n}_v1',split=split,trajectory_id=i,
        frame_id=j,start_symbol=16*j,end_symbol=16*j+15,channel_seed_scope='trajectory',
        communication_seed_scope='frame',communication_generated=(split=='validation_dev'),
        **seeds(n,split,i,j)) for n in ENVIRONMENTS
        for split,ids in (('train_dev',range(500)),('validation_dev',range(500,600))) for i in ids for j in range(10)]


def definition(name):
    if name not in ENVIRONMENTS:raise ValueError('undeclared delay-factor environment')
    _,profile,speed,delay=name.split('-')
    system,array=copy.deepcopy(SYSTEM),copy.deepcopy(ARRAY)
    system.update(speed_kmh=float(speed),nominal_delay_spread_seconds=int(delay)*1e-9,snr_db=list(SNRS))
    array['ut_velocity_mps']=[float(speed)/3.6,0.,0.]
    return dict(profile=profile,system=system,array=array,dF=16,spacing_time=4,
        ofdm_channel_normalize=False,symbol_wise_abstraction=True,within_symbol_ICI=False)


class FactorCDL(OfficialCDL):
    """Calibrated CDL export with only approved profile/speed/delay/seed changes."""
    def __init__(self,name,device=None):
        super().__init__(device)
        from sionna.phy.channel.tr38901 import CDL
        self.name=name;s=definition(name)
        self.model=CDL(model=s['profile'],delay_spread=s['system']['nominal_delay_spread_seconds'],
            carrier_frequency=3.5e9,ut_array=self.array,bs_array=self.array,direction='downlink',
            ut_orientation=ARRAY['ut_orientation_radians'],bs_orientation=ARRAY['bs_orientation_radians'],
            ut_velocity=s['array']['ut_velocity_mps'],min_speed=s['system']['speed_kmh']/3.6,
            max_speed=s['system']['speed_kmh']/3.6,normalize_delays=True,
            precision='double',device=self.device,spec_version='19.2')

    def generate(self,trajectory,*,split,details=False):
        import torch
        from sionna.phy import config
        from sionna.phy.channel import cir_to_ofdm_channel
        s=seeds(self.name,split,trajectory)
        config.seed=s['sionna_seed'];torch.manual_seed(s['torch_seed'])
        self.numpy_rng=np.random.default_rng(s['numpy_seed'])
        with torch.no_grad():
            a,tau=self.model(batch_size=1,num_time_steps=160,sampling_frequency=1/SYMBOL_SECONDS)
            hf=cir_to_ofdm_channel(self.frequencies,a,tau,normalize=False)
        h=hf[0,0,:,0,:,:,:].permute(0,1,3,2).cpu().numpy().astype(np.complex128,copy=True)
        if h.shape!=(2,2,64,160) or not np.isfinite(h).all():raise ValueError('invalid CDL export')
        if details:return h,a[0,0,:,0].cpu().numpy().astype(np.complex128),tau[0,0,0].cpu().numpy().astype(np.float64)
        return h

    def provenance(self):
        from pathlib import Path
        import hashlib
        from sionna.phy.channel.tr38901 import models
        result=super().provenance()
        result['source_hashes']={k:v for k,v in result['source_hashes'].items() if not k.endswith('CDL-C.json')}
        path=Path(models.parameter_file(f"CDL-{definition(self.name)['profile']}.json",'19.2'))
        result['source_hashes'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        result.update(profile=f"CDL-{definition(self.name)['profile']}",environment_definition=definition(self.name))
        return result


class EffectiveDelayAudit:
    """T15A pooled array/time CIR energy weighting; train-only, no normalization."""
    def __init__(self):self.ids=[];self.power=None;self.tau=None

    def update(self,i,a,tau):
        if i not in range(500) or i in self.ids:raise ValueError('unique train IDs only')
        a=np.asarray(a);tau=np.asarray(tau,dtype=np.float64)
        if a.ndim!=4 or a.shape[:2]!=(2,2) or tau.shape!=(a.shape[2],) or not np.isfinite(a).all() or not np.isfinite(tau).all():
            raise ValueError('finite CIR [rx,tx,path,time] and path delays required')
        if self.tau is not None and not np.array_equal(tau,self.tau):raise ValueError('path delays changed across trajectories')
        path=np.sum(abs(a)**2,axis=(0,1,3))
        self.power=path.copy() if self.power is None else self.power+path
        self.tau=tau.copy();self.ids.append(i)

    def freeze(self,nominal_ns):
        if self.ids!=list(range(500)) or self.power.sum()<=0:raise ValueError('500 ordered independent training trajectories required')
        pp=self.power/self.power.sum();mean=float(pp@self.tau)
        rms=float(np.sqrt(pp@(self.tau-mean)**2))
        return dict(nominal_delay_spread_ns=nominal_ns,empirical_array_weighted_rms_delay_ns=rms*1e9,
            empirical_mean_delay_ns=mean*1e9,normalized_empirical_path_powers=pp.tolist(),
            path_delays_seconds=self.tau.tolist(),pooled_path_energy=self.power.tolist(),
            trajectories=500,frames=5000,split='train_dev',definition='T15A pooled sum |a[rx,tx,path,time]|² across all training trajectories',audit_only=True)


def metric_columns(v):
    if set(v)!=set(METHODS) or not np.isfinite(list(v.values())).all():raise ValueError('all seven finite pooled NMSE values required')
    gap=v['source']-v['matched'];capacity=v['source']-v['oracle2-affine'];recovery=v['source']-v['dense-affine']
    issues=[]
    def ratio(numerator,denominator,name):
        if denominator<=0:
            issues.append(name+': nonpositive denominator');return None
        value=numerator/denominator
        if not np.isfinite(value):issues.append(name+': nonfinite quotient');return None
        return value
    return dict(matched_gap_db=gap,oracle2_recovery_db=capacity,deployed_recovery_db=recovery,
        representation_closure=ratio(capacity,gap,'representation_closure'),
        coordinate_efficiency=ratio(recovery,capacity,'coordinate_efficiency'),
        lr4_closure=ratio(v['source']-v['lr4-affine'],gap,'lr4_closure'),
        compression_penalty_db=v['lr4-affine']-v['dense-affine'],
        dense_vs_convex_gain_db=v['convex']-v['dense-affine'],lr4_vs_convex_gain_db=v['convex']-v['lr4-affine'],
        lr4_vs_source_gain_db=v['source']-v['lr4-affine'],ratio_anomaly=bool(issues),ratio_anomaly_reasons='; '.join(issues),
        **{f'nmse_{m.replace("-","_")}_db':x for m,x in v.items()})


def scientific_gate(rows):
    index={(r['environment'],r['snr_db']):r for r in rows}
    if len(rows)!=16 or set(index)!={(n,s) for n in ENVIRONMENTS for s in SNRS}:raise ValueError('complete sixteen-condition design required')
    pure=[index[n,s] for n in PURE for s in SNRS]
    def valid(r):return r['representation_closure'] is not None and r['coordinate_efficiency'] is not None
    sufficient=all(r['representation_closure'] is not None and r['representation_closure']>=.70 for r in pure)
    deficient=sum(valid(r) and r['representation_closure']<.50 and r['coordinate_efficiency']>=.80 for r in pure)>=3
    bottleneck=sum(valid(r) and r['representation_closure']>=.70 and r['coordinate_efficiency']<.80 for r in pure)>=3
    partial=not (sufficient or deficient or bottleneck)
    c9=('SUPPORTED_IN_TESTED_DELAY_CONDITIONS' if sufficient else 'CONTRADICTED_BY_DELAY_FACTOR_CHALLENGE' if deficient
        else 'INCONCLUSIVE_REPRESENTATION_DUE_TO_COORDINATE_BOTTLENECK' if bottleneck else 'INCONCLUSIVE')
    next_step=('No delay module; review sparse-pilot coordinate fitting.' if bottleneck else
        'No third module; consider separately assigned broader factor validation or method freeze.' if sufficient else
        'A separately assigned delay-direction necessity/minimal third-module gate may be considered.' if deficient else
        'Review the 50/200ns asymmetry before any method extension.')
    return dict(existing_pm_space_delay_representation_sufficient=bool(sufficient),delay_specific_representation_deficiency=bool(deficient),
        delay_coordinate_inference_bottleneck=bool(bottleneck),delay_factor_gate_partial=bool(partial),
        rank4_out_of_factor_warning=bool(any(r['compression_penalty_db']>1. for r in pure)),
        c9_claim_status=c9,next_step_dependency=next_step,representation_and_coordinate_gates_mutually_exclusive=False)
