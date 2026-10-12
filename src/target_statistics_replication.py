"""T19C train-only namespaces, sufficient-stat controls and fixed replication gate."""
import numpy as np
from src.cdl_benchmark import FrozenStatistics, StreamingStatistics
from src.delay_factor_challenge import FactorCDL, PURE, SNRS

ENVIRONMENTS=PURE
REPLICATES=(1,2,3,4,5)
ROLES=('sionna_seed','torch_seed','numpy_seed')
EXPERIMENT_TYPE='independent target-statistics replication of the condition-specific delay operator-space deficiency'
C11='The condition-specific C30/200/10 P/M operator-space deficiency is reproducible across independent target-statistics estimates.'


def namespace(name,replicate):
    if name not in ENVIRONMENTS or replicate not in REPLICATES:
        raise ValueError('only two pure-delay environments and five declared statistics replicates')
    return f'T19C_{name}_r{replicate}_v1'


def seeds(name,replicate,trajectory,*,split='train_dev'):
    namespace(name,replicate)
    if split!='train_dev' or trajectory not in range(500):
        raise ValueError('T19C only generates train_dev IDs0..499; validation forbidden')
    base=2300000000+ENVIRONMENTS.index(name)*10000000+(replicate-1)*1000000+trajectory*100
    return {role:base+(j+1)*10 for j,role in enumerate(ROLES)}


def manifest_rows():
    return [dict(environment=n,namespace=namespace(n,r),split='train_dev',replicate=r,
                 trajectory_id=i,frame_id=j,start_symbol=16*j,end_symbol=16*j+15,
                 channel_seed_scope='trajectory',communication_generated=False,
                 **seeds(n,r,i))
            for n in ENVIRONMENTS for r in REPLICATES for i in range(500) for j in range(10)]


class StatisticsCDL(FactorCDL):
    """Original T19A physical generator; only the explicit train seed call differs."""
    def __init__(self,name,replicate,device=None):
        seeds(name,replicate,0)
        super().__init__(name,device)
        self.replicate=replicate

    def generate(self,trajectory,*,split,details=False):
        import torch
        from sionna.phy import config
        from sionna.phy.channel import cir_to_ofdm_channel
        s=seeds(self.name,self.replicate,trajectory,split=split)
        config.seed=s['sionna_seed'];torch.manual_seed(s['torch_seed'])
        self.numpy_rng=np.random.default_rng(s['numpy_seed'])
        with torch.no_grad():
            a,tau=self.model(batch_size=1,num_time_steps=160,sampling_frequency=1/SYMBOL_SECONDS)
            hf=cir_to_ofdm_channel(self.frequencies,a,tau,normalize=False)
        h=hf[0,0,:,0,:,:,:].permute(0,1,3,2).cpu().numpy().astype(np.complex128,copy=True)
        if h.shape!=(2,2,64,160) or not np.isfinite(h).all():raise ValueError('invalid CDL export')
        if details:return h,a[0,0,:,0].cpu().numpy().astype(np.complex128),tau[0,0,0].cpu().numpy().astype(np.float64)
        return h


# Preserve the exact sampling-frequency constant used by inherited T19A code.
from src.cdl_benchmark import SYMBOL_SECONDS


def merge_sufficient_statistics(name,records):
    """Secondary 2500-trajectory control. Include between-mean corrections.

    records are exactly five (T19C namespace, frozen statistics) pairs for one
    delay. Primary estimators never receive merged statistics. No original data.
    """
    expected={namespace(name,r) for r in REPLICATES}
    if len(records)!=5 or {ns for ns,_ in records}!=expected:
        raise ValueError('five distinct new T19C namespaces required; T19A/mixed-delay pooling forbidden')
    counts=np.array([st.samples for _,st in records],np.float64)
    if not np.isfinite(counts).all() or np.any(counts<=0):raise ValueError('positive sufficient-stat counts required')
    total=int(sum(st.samples for _,st in records));fractions=counts/total
    mp=sum(w*st.mean_p for w,(_,st) in zip(fractions,records))
    md=sum(w*st.mean_d for w,(_,st) in zip(fractions,records))
    pp=np.zeros_like(records[0][1].rpp);dp=np.zeros_like(records[0][1].rdp)
    for weight,(_,st) in zip(fractions,records):
        delta_p=st.mean_p-mp;delta_d=st.mean_d-md
        pp+=weight*(st.rpp+np.outer(delta_p,delta_p.conj()))
        dp+=weight*(st.rdp+np.outer(delta_d,delta_p.conj()))
    values=[mp,md,pp,dp]
    for a in values:a.flags.writeable=False
    keys=tuple((int(ns.split('_r')[1].split('_')[0]),i) for ns,st in records for i in st.train_trajectory_ids)
    return FrozenStatistics(*values,keys,total)


def summarize(rows,original):
    expected={(n,r,s) for n in ENVIRONMENTS for r in REPLICATES for s in SNRS}
    index={(r['environment'],int(r['replicate']),int(r['snr_db'])):r for r in rows}
    if len(rows)!=20 or set(index)!=expected:raise ValueError('all twenty unique replicate/SNR captures required')
    result=[]
    for n in ENVIRONMENTS:
        for s in SNRS:
            x=np.array([index[n,r,s]['operator_span_capture_dense'] for r in REPLICATES],np.float64)
            if not np.isfinite(x).all():raise ValueError('finite captures required; no sample deletion')
            point=float(original[n,s])
            result.append(dict(environment=n,snr_db=s,replicates=5,captures=x.tolist(),
                mean=float(x.mean()),std=float(x.std(ddof=1)),std_ddof=1,median=float(np.median(x)),
                minimum=float(x.min()),maximum=float(x.max()),capture_range=float(x.max()-x.min()),
                original_t19a_capture=point,original_minus_replication_mean=point-float(x.mean()),
                original_below_replicate_min=bool(point<x.min()),original_above_replicate_max=bool(point>x.max()),
                replicates_below_original=int(np.count_nonzero(x<point)),replicates_equal_original=int(np.count_nonzero(x==point)),
                independent_unit='target-training-statistics replicate;not trajectory/frame',bootstrap=False))
    return result


def scientific_gate(rows):
    index={(v['environment'],int(v['replicate']),int(v['snr_db'])):v for v in rows}
    if len(rows)!=20 or set(index)!={(n,r,s) for n in ENVIRONMENTS for r in REPLICATES for s in SNRS}:
        raise ValueError('twenty unique independent-statistics/SNR rows required')
    primary=np.array([index['CDL-C-30-200',r,10]['operator_span_capture_dense'] for r in REPLICATES])
    if not np.isfinite(primary).all():raise ValueError('undefined primary capture:STOP')
    below=int(np.count_nonzero(primary<.50));above=int(np.count_nonzero(primary>=.70))
    supported=below>=4 and float(np.median(primary))<.50
    contradicted=above>=4
    control=[int(sum(index['CDL-C-30-50',r,s]['operator_span_capture_dense']>=.70 for r in REPLICATES)) for s in SNRS]
    pairs=[dict(replicate=r,capture_10dB=float(index['CDL-C-30-200',r,10]['operator_span_capture_dense']),
                capture_20dB=float(index['CDL-C-30-200',r,20]['operator_span_capture_dense']),
                paired_difference=float(index['CDL-C-30-200',r,20]['operator_span_capture_dense']-index['CDL-C-30-200',r,10]['operator_span_capture_dense'])) for r in REPLICATES]
    return dict(delay_operator_deficiency_replication_supported=supported,
                delay_operator_deficiency_replication_contradicted=contradicted,
                delay_operator_deficiency_replication_inconclusive=not supported and not contradicted,
                primary_below_050=below,primary_at_or_above_070=above,primary_median=float(np.median(primary)),
                delay50_control_stable=all(n>=4 for n in control),delay50_control_counts=dict(zip(map(str,SNRS),control)),
                snr_asymmetry=pairs,snr_asymmetry_direction_consistent=all(v['paired_difference']>0 for v in pairs),
                c11_claim_status=('SUPPORTED_IN_C30_200_10_STATISTICS_REPLICATION' if supported else
                                  'CONTRADICTED_BY_TARGET_STATISTICS_REPLICATION' if contradicted else 'INCONCLUSIVE'))


def calibration():
    """Independent direct centered covariance and unequal-count pooling check."""
    selected=np.array([0,2]);data=np.array([False,True,False,True,True,True])
    records=[];rows=[];max_error=0.
    for r in REPLICATES:
        k=np.arange((r+2)*6,dtype=np.float64).reshape(r+2,6)+r*10
        x=np.sin(k*.31)+r*.3+1j*(np.cos(k*.17)+r*.13)
        stream=StreamingStatistics(selected,data,dimension=6)
        stream.update(x[:1],[0]);stream.update(x[1:],[1]);st=stream.freeze()
        mean=x.mean(axis=0);xc=x-mean
        direct_pp=xc[:,selected].T@xc[:,selected].conj()/len(x)
        direct_dp=xc[:,data].T@xc[:,selected].conj()/len(x)
        max_error=max(max_error,float(np.max(abs(st.rpp-direct_pp))),float(np.max(abs(st.rdp-direct_dp))))
        records.append((namespace(ENVIRONMENTS[0],r),st));rows.append(x)
    merged=merge_sufficient_statistics(ENVIRONMENTS[0],records)
    x=np.concatenate(rows);xc=x-x.mean(axis=0)
    max_error=max(max_error,float(np.max(abs(merged.rpp-xc[:,selected].T@xc[:,selected].conj()/len(x)))),
                  float(np.max(abs(merged.rdp-xc[:,data].T@xc[:,selected].conj()/len(x)))),
                  float(np.max(abs(merged.mean_p-x.mean(axis=0)[selected]))))
    if max_error>1e-11:raise ValueError('streaming estimator / sufficient-stat aggregation calibration failed')
    return dict(passed=True,max_absolute_error=max_error,unequal_sample_counts=[len(x) for x in rows],
                total_samples=merged.samples,batch_mean_correction_verified=True,Rdd_fitted=False)
