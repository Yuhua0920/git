"""Toy gate first; tiny frozen-source replay/fit smoke only after toy PASS."""
import argparse
import importlib.metadata as metadata
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u3a_common as c

def toy():
    from scripts import sp_u3a_mfa_port as p
    c.verify();assert c.load(c.OUT/'sp_u3a_preparation_gate.json')['passed']
    result=p.toy_results();result.update(access=c.ACCESS,formal_read=False,old_U2_outcomes_read=False)
    c.save(c.OUT/'sp_u3a_mfa_toy_validation.json',result)
    print('TOY PASS',result['fixtures'],result['noise'],flush=True)

def smoke():
    from scripts import sp_u3a_mfa_port as p
    from src.cdl_benchmark import OfficialCDL,frames,array_hash,SYSTEM
    import numpy as np
    c.verify();assert c.load(c.OUT/'sp_u3a_mfa_toy_validation.json')['passed']
    assert c.load(c.OUT/'sp_u3a_tests_gate.json')['passed']
    assert c.sha(ROOT/'configs/sp_u3a_protocol.json')==c.load(c.OUT/'sp_u3a_preparation_gate.json')['protocol_sha256']
    cfg=p.protocol()['smoke'];assert SYSTEM['sample_rate_hz']==SYSTEM['nfft']*SYSTEM['delta_f_hz']
    old=c.load(c.OUT/'t15b_gmm_probe_dataset_manifest.json');refs={a['trajectory_id']:a['h_sha256'] for a in old['h_hashes'] if a['trajectory_id'] in cfg['source_train_ids']}
    start=time.perf_counter();channels=[];replay=[];adapter=OfficialCDL()
    with p.threadpool_limits(limits=4):
        for i in cfg['source_train_ids']:
            H=adapter.generate(i);assert array_hash(H)==refs[i]
            channels.extend([frames(H)[j].ravel(order='C').copy() for j in cfg['frames']])
            replay.append(dict(trajectory_id=i,H_sha256=array_hash(H),frozen_source_exact=True))
        training=np.stack(channels);np.savez(c.OUT/'sp_u3a_source_smoke_cache.npz',clean=training)
        model=p.MfaEstimator(cfg['K'],cfg['rank'],max_iter=cfg['max_iter'],random_state=cfg['seed'],verbose=False)
        fitstart=time.perf_counter()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always');model.fit(training)
        fitseconds=time.perf_counter()-fitstart;warns=[str(a.message) for a in w]
        g,A=p.mapping();proof=p.mapping_check(training[0].reshape(2,2,64,16))
        sigma=float(p.protocol()['future_metrics']['sigma2']['20']);rng=np.random.default_rng(cfg['seed'])
        # Two structurally valid observations; same tiny samples used only for smoke.
        y=training[:2]@A.T+np.sqrt(sigma)*p.cn((2,64),rng)
        tick=time.perf_counter();estimate=p.estimate_data(model,y,sigma);estseconds=time.perf_counter()-tick
        obs=model._prepare_observation_model(A,np.eye(64)*sigma);prob=model._predict_observation_proba(y,obs)
        assert estimate.shape==(2,3968) and np.isfinite(estimate).all() and np.iscomplexobj(estimate)
        assert p.error(prob.sum(axis=1),np.ones(2))<1e-12
        # Sparse selector reference avoids constructing another4096square covariance.
        Cpp=model.covariances_[0][np.ix_(g.selected,g.selected)]+sigma*np.eye(64)
        cross=model.covariances_[0][np.ix_(np.flatnonzero(g.data),g.selected)]
        reference=model.means_[0,g.data]+(cross@np.linalg.solve(Cpp,(y-model.means_[0,g.selected]).T)).T
        er=p.error(reference,estimate);assert er<1e-10
    result=dict(passed=True,scope='project-shape smoke only;not a trained scientific comparator',
      full_shape=[2,2,64,16],train_shape=list(training.shape),pilot_shape=list(y.shape),A_shape=list(A.shape),target_shape=list(estimate.shape),
      dtype=str(estimate.dtype),observations=2,trajectory_replay=replay,source_frame_ids=cfg['frames'],
      initialized_and_EM_fitted=True,EM_iterations=len(model.lower_bound_history_),EM_warnings=warns,
      converged_scientific_model=False,rank=cfg['rank'],K=cfg['K'],sigma2=sigma,
      fit_seconds=fitseconds,estimate_seconds=estseconds,total_seconds=time.perf_counter()-start,
      resident_prior_array_bytes=sum(getattr(model,k).nbytes for k in ('means_','loadings_','noise_variances_','weights_','covariances_','precisions_')),
      A_bytes=A.nbytes,clean_sample_bytes=training.nbytes,process_memory=p.process_memory(),
      finite_output=True,responsibility_sum_error=p.error(prob.sum(axis=1),np.ones(2)),direct_selector_estimate_error=er,
      mapping=proof,full_MFA_training=False,comparison_NMSE_computed=False,BER_computed=False,
      new_channel_realizations=False,frozen_train_channel_replays=2,no_target_or_formal_CSI=True,access=c.ACCESS)
    c.save(c.OUT/'sp_u3a_project_shape_smoke.json',result)
    print('PROJECT SMOKE PASS', {k:v for k,v in result.items() if k not in ('access','trajectory_replay')},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['toy','smoke']);globals()[parser.parse_args().action]()
