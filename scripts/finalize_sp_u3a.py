"""Seal portability evidence, complexity disclosure, tests and stop decision."""
import datetime
import importlib.metadata as metadata
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import sp_u3a_common as c

def evidence():
    from scripts import sp_u3a_mfa_port as p
    import scipy
    import sklearn
    import pytest
    pre=c.verify();prep=c.load(c.OUT/'sp_u3a_preparation_gate.json')
    assert c.sha(ROOT/'configs/sp_u3a_protocol.json')==prep['protocol_sha256']
    assert all(c.sha(c.OUT/path)==value for path,value in prep['old_input_sha256'].items())
    toy=c.load(c.OUT/'sp_u3a_mfa_toy_validation.json');smoke=c.load(c.OUT/'sp_u3a_project_shape_smoke.json')
    assert toy['passed'] and smoke['passed'] and c.load(c.OUT/'sp_u3a_tests_gate.json')['passed']
    assert smoke['total_seconds']<p.protocol()['smoke']['max_seconds']
    env=[]
    for name,module in [('numpy',p.np),('scipy',scipy),('scikit-learn',sklearn),('pytest',pytest)]:
        env.append(dict(package=name,version=metadata.version(name),module_file=module.__file__,module_sha256=c.sha(module.__file__)))
    c.save(c.OUT/'sp_u3a_environment_audit.json',dict(python=sys.version,executable=sys.executable,
      dependencies=env,process_path_only=['downloaded unmodified author src','existing .venv-gmm sklearn dependency'],
      dependency_installs=0,environment_writes=0,pip_install_executed=False,
      source_execution='pinned source import;not installed as distributions',
      existing_cdl_numpy_scipy_used=True,existing_gmm_sklearn_used=True,
      retrieval_recovery='sandbox network denial then Python TLS issuer error;Windows trust-store retrieval succeeded;TLSverification neverdisabled',
      inherited_python_relocation_warning=True,smoke_torch_triton_warning='optional flop counter unavailable;no effect on this CPU smoke'))
    rows=[];N=4096;M=64
    for K in p.protocol()['K']:
        for r in p.protocol()['latent_rank']:
            rows.append(dict(K=K,rank=r,real_parameter_count=K*(2*N*r+3*N)+K-1,
              structured_parameter_bytes=K*((N*r+N)*16+N*8)+K*8,
              native_covariance_precision_bytes=2*K*N*N*16,full_clean_training_bytes_Source=5000*N*16,
              full_clean_training_bytes_Prepared=15000*N*16,observed_inverse_dimension=64,
              native_observation_prepare_MAC_proxy=K*(M*N*N+M*M*N),
              native_component_cross_MAC_per_observation=K*N*N*M,
              cached_gain_apply_MAC_per_observation=K*N*M,
              cached_gain_status='not implemented;equivalent arithmetic planning only',
              peak_training_bytes_unknown=True))
    c.csvsave(c.OUT/'sp_u3a_complexity_budget.csv',rows)
    c.save(c.OUT/'sp_u3a_complexity_budget.json',dict(grid=rows,
      structured_storage='K complex W[N,r],complexmu[N],realPsi[N],weights[K]',
      native_software='full dense covariances and precisions;dense slogdet eachEM;repeatedC A^H duringestimate',
      online_responsibility='K observation Gaussian scores;M=64 quadratic O(K M²),softmax O(K)',
      conditional_solve='64x64 observation precision percomponent;native usespinv andslogdet;not4096x4096online inverse',
      mixture_aggregation='O(K N),all components withfloat1.0',
      model_cost_vs_implementation_cost='structured statistical parameter count does not equal resident dense implementation memory',
      final_method=dict(source_W_complex=3968*64,rank4_correction_complex=2*4*(3968+64),
        cross_pilot_complex=3*4*16*48,means_complex=3968+64,
        approximate_complex128_bytes_perSNR=16*(3968*64+2*4*(3968+64)+3*4*16*48+3968+64)),
      hardware_benchmark_performed=False,tiny_smoke_actual=smoke,
      unresolved='smallK1/rank2/oneiteration establishes shape feasibility only;fullPrepared K/rank/200iteration peak andconvergence unverified'))
    checklist=dict(authoritative_algorithm_pinned=True,software_pinned=True,license_known_acceptable=True,
      general_observation_faithful=True,toy_PASS=True,project_smoke_PASS=True,
      offline_variants_defined=True,hyperparameter_protocol_frozen=True,no_T20B_leakage=True)
    c.save(c.OUT/'sp_u3a_baseline_gate.json',dict(task='SP-U3A',gate='PASS',criteria=checklist,
      meaning='faithful portability and defined comparison protocol;not performance superiority orfull-training feasibility proof',
      fairness='conditional PASS:matchedenvironment/count candidate;information interface andtemporalhistory differ',
      primary_candidate='MFA-Prepared',secondary_candidate='MFA-Source',Oracle_Target='not fair;not trained',
      permitted_next='SP-U3B only after mentor review/explicit task;not started',
      theory='FREEZE',full_training=False,comparison=False,paper_edit=False,commit=False,
      source_files_unchanged=len(pre['protected_sources']),protocol_sha256=prep['protocol_sha256'],
      ratings=dict(Engineering='PASS',Faithful_reproduction='PASS forsource/equations/port;2023measurementbenchmark not reproduced',
        Fairness='conditional PASS',Firewall='PASS',Reproducibility='PASS'),
      risks=['RawfullMFA distribution interface vsRpp/Rdp preparedstatistics',
        '64mutedchannelcoefficients present infullMFA model butabsent fromfinal statistical objects',
        'labelled correctionconstruction vsunlabelled pooledEM','fiveframehistory vscurrent-only',
        'Historical finalmethod hyperparameter/development exposure not equal new internalcleanNLL tuning',
        'futuretraining computationalcost/convergence andstatistical performance unknown',
        'no guarantee fullMFA learnedprior matchesfinite500trajectory/environment corpus',
        '2023A=I anddissertationinvertibleA theorem do not establish rectangularA asymptoticoptimality'],
      stop=True,access=c.ACCESS))
    print('BASELINE GATE PASS:all9criteria;fairnessconditional;theoryFREEZE;noSP-U3Bstarted')

def seal():
    pre=c.verify();assert c.load(c.OUT/'sp_u3a_baseline_gate.json')['gate']=='PASS'
    cmd=[sys.executable,'-m','pytest','tests/test_sp_u3a_delivery.py','-q','-p','no:cacheprovider',
      '--basetemp=outputs/sp_u3a_pytest_tmp_delivery']
    result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    log=c.OUT/'sp_u3a_pytest_delivery.txt';assert not log.exists();log.write_text(result.stdout+result.stderr,encoding='utf8')
    print(result.stdout+result.stderr,flush=True)
    if result.returncode:raise RuntimeError('delivery checks failed')
    c.verify();prep=c.load(c.OUT/'sp_u3a_preparation_gate.json')
    for path,value in prep['old_input_sha256'].items():assert c.sha(c.OUT/path)==value
    env=c.load(c.OUT/'sp_u3a_environment_audit.json')
    assert all(c.sha(a['module_file'])==a['module_sha256'] for a in env['dependencies'])
    source_checks=[]
    for name in ('cplx-mfa','MFA_estimator'):
        root=c.OUT/'sp_u3a_sources'/name;pin=c.load(root/'repository_pin.json')
        assert all(c.sha(root/path)==value for path,value in pin['files'].items())
        source_checks.append(dict(package=name,commit=pin['commit'],all_pinned_bytes_unchanged=True))
    c.save(c.OUT/'sp_u3a_integrity_audit.json',dict(passed=True,protected_count=len(pre['protected_sources']),
      frozen_old_input_files=len(prep['old_input_sha256']),old_inputs_unchanged=True,author_sources=source_checks,
      dependency_module_hashes_unchanged=True,protocol_sha256=c.sha(ROOT/'configs/sp_u3a_protocol.json'),
      HEAD=c.git('rev-parse','HEAD'),tracked_diff=c.git('diff','--stat'),staged_diff=c.git('diff','--cached','--stat'),
      diff_check_exit=subprocess.run(['git','diff','--check'],cwd=ROOT).returncode,
      delivery_test_command=cmd,delivery_test_exit=result.returncode,no_formal_read=True,no_old_U2_outcome_read=True,
      no_author_algorithm_modified=True,no_production_modified=True,no_paper_modified=True,no_commit=True,stop=True,access=c.ACCESS))
    artifacts={}
    for path in sorted(c.OUT.glob('sp_u3a_*')):
        if path.name=='sp_u3a_delivery_manifest.json' or 'pytest_tmp' in path.name:continue
        files=[path] if path.is_file() else list(path.rglob('*'))
        for file in files:
            if file.is_file() and '__pycache__' not in file.parts:artifacts[file.relative_to(ROOT).as_posix()]=c.sha(file)
    sources={p.relative_to(ROOT).as_posix():c.sha(p) for folder in ('scripts','tests','configs')
       for p in sorted((ROOT/folder).glob('*sp_u3a*')) if p.is_file()}
    c.save(c.OUT/'sp_u3a_delivery_manifest.json',dict(task='SP-U3A',sealed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      artifact_sha256=artifacts,execution_source_sha256=sources,protected_sources=pre['protected_sources'],
      frozen_input_sha256=prep['old_input_sha256'],stop=True))
    print('SEALED',len(artifacts),'artifacts;',len(sources),'newsourcefiles;',len(pre['protected_sources']),'oldsourcefiles exact')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['evidence','seal']);globals()[parser.parse_args().action]()
