"""T20B immutable formal execution; after unseal any exception stops permanently."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src import formal_evaluation as metrics
from src import frozen_affine_validation as method
from src.cdl_benchmark import OfficialCDL, ARRAY, SYMBOL_SECONDS, FrozenStatistics, frames, array_hash, observation_geometry
from src.link import (assemble_pilot_data_frame, apply_frequency_mimo_channel, estimate_ls_linear_channel,
    detect_data_zf, extract_reference_data_bits, demodulate_data_bits)
from src.pilot_cv import deployment
from src.lowrank_modules import factored_affine_predict
from scripts import audit_final_method_freeze as frozen
from scripts import audit_formal_preflight as governance
from scripts import assign_formal_seeds as assignment
from scripts.formal_governance_snapshot import COMMIT, EVOLVING, historical_hashes
from scripts.run_frozen_generalization import frozen_method, historical_receiver_regression
from scripts.run_cdl_mismatch import communication
from scripts.run_pilot_cv import cv_family

OUT = ROOT/'outputs'
PREFIX = 't20b_formal_'
UNSEAL = OUT/(PREFIX+'unseal.json')
PRE = OUT/'t20b_execution_preflight_final.json'
SOURCES = ('src/formal_evaluation.py', 'scripts/run_formal_test.py', 'scripts/plot_formal_test.py',
           'scripts/formal_governance_snapshot.py', 'tests/test_formal_execution.py', *sorted(EVOLVING))
HASHES = {
    governance.PROTOCOL: '196e9aeeebb2e1b4dfb9984fa35dc2187d5ab5d21a70f7c29d0713cdd705731b',
    governance.AMENDMENT: '58cf2cc98b8562bd823e583fe351e289221f1dabdeccda704c10a432a56da507',
    assignment.AMENDMENT: 'f1d081e7ec57ec1ea64da0d5e59bbeef9f62074171571a607f91a00198db3afc',
    assignment.MANIFEST: 'b9e38d7ee153a38c226c8cfe8363cd9753c7cbaefeb8d9742d5384934454628f',
    assignment.REJECTED: '651fc74eeb14c98bedaec712cc13ba8d4901cc12d855ad5117c01beb5119f7a8'}
STATISTICS = ('outputs/t15a_cdl_statistics_500.npz', 'outputs/t15c_cdl_statistics_CDL-B-30.npz',
    'outputs/t15c_cdl_statistics_CDL-C-120.npz', 'outputs/t16c_heldout_statistics_CDL-B-120.npz',
    'outputs/t19a_matched_statistics_CDL-C-30-200.npz')
MAP = {'source':'Source LMMSE','hard':'Hard','convex':'Convex','dense-affine':'Dense Affine','lr4-affine':'Final LR4 Affine'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    """Exclusive creation: no previous evidence may be overwritten."""
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def readcsv(path):
    with Path(path).open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))


class Journal:
    def __init__(self, path, fields):
        self.stream = Path(path).open('x', encoding='utf8', newline='')
        self.writer = csv.DictWriter(self.stream, fieldnames=fields, lineterminator='\n')
        self.writer.writeheader()

    def append(self, row):
        self.writer.writerow(row)
        self.stream.flush()

    def close(self):
        self.stream.close()


def csvwrite(path, rows):
    journal = Journal(path, list(rows[0]))
    try:
        for row in rows:
            journal.append(row)
    finally:
        journal.close()


def seed_index():
    """Load the frozen allocation; never call the assignment algorithm for data."""
    frozen.verify_hashes(HASHES)
    rows = readcsv(ROOT/assignment.MANIFEST)
    index = {(r['environment'], int(r['trajectory']), int(r['frame']), r['random_role']): int(r['assigned_seed']) for r in rows}
    if len(index) != 165000:
        raise ValueError('Formal manifest cardinality drift')
    return index


def consumed(index, name, trajectory, frame):
    return {role:index[name, trajectory, 0 if role in frozen.ROLES[:3] else frame, role] for role in frozen.ROLES}


class FormalCDL(OfficialCDL):
    """Namespace/configuration wrapper around the original OfficialCDL generator."""
    def __init__(self, name, index, device=None):
        if name not in metrics.CONDITIONS:
            raise ValueError('Undeclared formal condition')
        super().__init__(device)
        from sionna.phy.channel.tr38901 import CDL
        self.name, self.index = name, index
        _, profile, speed, delay = name.split('-')
        # Preserve the actual historical float representation for each role.
        delay_seconds = 100e-9
        if int(delay)==200:
            from src.delay_factor_challenge import definition
            delay_seconds=definition(name)['system']['nominal_delay_spread_seconds']
        self.model = CDL(model=profile, delay_spread=delay_seconds, carrier_frequency=3.5e9,
            ut_array=self.array, bs_array=self.array, direction='downlink',
            ut_orientation=ARRAY['ut_orientation_radians'], bs_orientation=ARRAY['bs_orientation_radians'],
            ut_velocity=[int(speed)/3.6,0.,0.], min_speed=int(speed)/3.6, max_speed=int(speed)/3.6,
            normalize_delays=True, precision='double', device=self.device, spec_version='19.2')

    def generate(self, trajectory):
        # The numerical generator body is the original committed function.
        # Only its seed lookup is temporarily bound to the immutable manifest.
        import src.cdl_benchmark as adapter
        old = adapter.seeds
        adapter.seeds = lambda i, frame=0: consumed(self.index, self.name, i, frame)
        try:
            return super().generate(trajectory)
        finally:
            adapter.seeds = old


def frozen_statistics(path):
    with np.load(ROOT/path, allow_pickle=False) as f:
        arrays = [f[k].copy() for k in ('mean_p','mean_d','rpp','rdp')]
        ids = tuple(map(int,f['train_trajectory_ids'])); count = int(f['samples'])
    if ids != tuple(range(500)) or count != 5000:
        raise ValueError('Matched reference must be frozen clean development statistics')
    for a in arrays:
        a.flags.writeable = False
    return FrozenStatistics(*arrays,ids,count)


def configuration_sentinel(index):
    """Instantiate five old configurations; compare metadata only, no H calls."""
    from src.cdl_mismatch import ShiftCDL
    from src.heldout_basis import HeldoutCDL
    from src.delay_factor_challenge import FactorCDL
    legacy=(OfficialCDL(),ShiftCDL('CDL-B-30'),ShiftCDL('CDL-C-120'),
            HeldoutCDL('CDL-B-120'),FactorCDL('CDL-C-30-200'))
    fields=('normalized_PDP_powers','scaled_cluster_delays_seconds','array_positions_m',
            'device','internal_dtype','export_dtype','ofdm_channel_normalize','time_states_per_call','sampling_frequency_hz')
    rows=[]
    for name,old in zip(metrics.CONDITIONS,legacy):
        original=old.provenance();new=FormalCDL(name,index).provenance()
        if any(original[k]!=new[k] for k in fields):
            raise ValueError('Frozen condition metadata differs: '+name)
        rows.append(dict(environment=name,exact=True,metadata={k:original[k] for k in fields},
                         generator_library_source_hashes=original['source_hashes']))
    return dict(generator_calls=0,formal_H_generated=0,all_five_configurations_exact=True,rows=rows)


def historical_digests(paths):
    """Conservative superset of every stored SHA256, including all historical H."""
    digests = set()
    pattern = re.compile(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])')
    for p in paths:
        if Path(p).suffix.lower() in ('.json','.csv'):
            digests.update(pattern.findall((ROOT/p).read_text(encoding='utf8')))
    return sorted(digests)


def historical_generator_sentinel():
    """Replay old trajectory 500 only, never formal inputs; verify whole-link BER."""
    from src.cdl_benchmark import seeds as old_seeds
    from scripts.run_cdl_mismatch import evaluate_trajectory, source_statistics
    adapter = OfficialCDL()
    h = adapter.generate(500)
    old = next(r for r in readcsv(OUT/'t15a_fix_replay_h_hashes.csv') if int(r['trajectory_id']) == 500)
    if array_hash(h) != old['replayed_h_sha256']:
        raise ValueError('Historical CDL generator drift')
    source,_ = source_statistics()
    estimators = {(m,s):source.prepare(method.VARIANCES[s]) for m in ('R10-C-frozen','R10-matched') for s in method.SNRS}
    actual = evaluate_trajectory('historical-only',500,h,observation_geometry(),estimators,method.VARIANCES,seed_function=old_seeds)[0]
    expected = [r for r in readcsv(OUT/'t15a_cdl_trajectory_metrics.csv') if int(r['trajectory_id']) == 500]
    mapping = {'R3':'R3-LS-LIN','R10-C-frozen':'R10-500','Perfect-CSI':'R0-Perfect-CSI'}
    for row in actual:
        if row['method'] not in mapping:
            continue
        prior = next(r for r in expected if r['method'] == mapping[row['method']] and int(r['snr_db']) == row['snr_db'])
        for field in ('error_energy','truth_energy','bit_errors','total_bits','ber'):
            if str(row[field]) != prior[field]:
                raise ValueError('Historical receiver mismatch: '+field)
    # Wrapper parity on the same historical realization, not a formal seed.
    fake = {(metrics.CONDITIONS[0],500,0,k):v for k,v in old_seeds(500).items()}
    wrapped = FormalCDL(metrics.CONDITIONS[0],fake).generate(500)
    if array_hash(wrapped) != array_hash(h):
        raise ValueError('Original-body wrapper changed numerical generator')
    # Exercise the new eight-method orchestration on this historical sample,
    # then independently reproduce its Perfect/Linear/Source six-frame scores.
    arrays,family,_=frozen_method()
    comm=[communication(old_seeds(500,j)) for j in range(10)]
    matched={s:source.prepare(method.VARIANCES[s]) for s in metrics.SNRS}
    timing={k:0. for k in ('pilot_fitting_and_frozen_deployment_seconds',
        'dense_inference_recheck_seconds','lr4_inference_recheck_seconds','BER_ZF_seconds')}
    rows,_,_,_,_=evaluate(h,comm,arrays,family,{s:cv_family(arrays,s) for s in metrics.SNRS},matched,timing=timing)
    g=observation_geometry()
    for snr in metrics.SNRS:
        accum={m:dict(error_energy=0.,truth_energy=0.,bit_errors=0,total_bits=0)
               for m in ('Perfect CSI','R3 LS+linear','Source LMMSE')}
        for j in range(4,10):
            truth=frames(h)[j];bits,symbols,z=comm[j]
            clean=apply_frequency_mimo_channel(truth,assemble_pilot_data_frame(symbols,g.layout))
            y=clean+np.sqrt(method.VARIANCES[snr])*z
            sparse,linear=estimate_ls_linear_channel(y,g.layout)
            source_prediction=matched[snr].estimate(sparse.ravel()[g.selected])
            reference=extract_reference_data_bits(bits,g.layout.data_mask)
            for label,estimate in (('Perfect CSI',truth),('R3 LS+linear',linear),
                                   ('Source LMMSE',channel_grid(source_prediction,sparse,g))):
                detected=demodulate_data_bits(detect_data_zf(y,estimate,g.layout.data_mask)*np.sqrt(2),g.layout.data_mask)
                e=accum[label]
                e['error_energy']+=float(np.sum(abs(estimate.ravel()[g.data]-truth.ravel()[g.data])**2))
                e['truth_energy']+=float(np.sum(abs(truth.ravel()[g.data])**2))
                e['bit_errors']+=sum(int(np.count_nonzero(a!=b)) for a,b in zip(detected,reference))
                e['total_bits']+=sum(a.size for a in reference)
        for label,expected_values in accum.items():
            actual_row=next(r for r in rows if (r['snr_db'],r['method'])==(snr,label))
            if any(actual_row[k]!=v for k,v in expected_values.items()):
                raise ValueError('New orchestration historical six-frame receiver drift')
    return dict(trajectory=500, namespace='historical T15A', H_exact=True,
                whole_link_NMSE_BER_exact=True, wrapper_export_exact=True,
                new_orchestration_historical_six_frame_regression_exact=True,formal_H_count=0)


def preparation():
    if UNSEAL.exists() or PRE.exists():
        raise ValueError('Existing opening/preparation evidence protected')
    if Path(sys.prefix).resolve() != (ROOT/'.venv-cdl').resolve():
        raise ValueError('Frozen CDL environment required')
    frozen.verify_hashes(HASHES)
    assigned = assignment.validate_artifacts()
    if not assigned['formal_preflight_ready'] or not assigned['all_checks_passed']:
        raise ValueError('Accepted seed preflight required')
    env = governance.environment()
    arrays,family,proof = frozen_method()
    old_receiver = historical_receiver_regression(arrays,family)
    generator = historical_generator_sentinel()
    configuration = configuration_sentinel(seed_index())
    sources = {p:frozen.sha(ROOT/p) for p in SOURCES}
    # Preserve raw bytes of all current tracked immutable files. Only the four
    # stage-scoped governance sources use their accepted historical snapshot.
    paths = subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    history = {p:frozen.sha(ROOT/p) for p in paths if p not in EVOLVING}
    historical_hashes(load(ROOT/assignment.BASELINE)['historical_sha256'])
    result = dict(pre_unseal_governance_commit=COMMIT, HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        hashes=HASHES, source_sha256=sources, frozen_method=proof,
        frozen_method_definition_sha256=frozen.sha(ROOT/'docs/t20a_final_method_definition.md'),
        reference_statistics_sha256={p:frozen.sha(ROOT/p) for p in STATISTICS},
        historical_sha256=history, historical_comparison_hashes=historical_digests(paths),
        environment=env, historical_receiver=old_receiver, historical_generator=generator,
        historical_condition_configuration=configuration,
        formal_H_count=0,formal_NMSE_count=0,formal_BER_count=0,
        formal_test_used=False,formal_test_sealed=True,formal_test_unsealed=False,
        formal_seed_count=165000, historical_seed_overlap=0, formal_seed_duplicates=0, namespace_overlap=0,
        formal_preflight_ready=True,all_checks_passed=True)
    write(PRE,result)
    print('Formal execution preflight PASS; historical generator/receiver exact; formal H/NMSE/BER=0',flush=True)


def verify_preparation():
    proof = load(PRE)
    for key in ('hashes','source_sha256','reference_statistics_sha256','historical_sha256'):
        frozen.verify_hashes(proof[key])
    historical_hashes(load(ROOT/assignment.BASELINE)['historical_sha256'])
    frozen.frozen_arrays(); frozen.constants_audit()
    return proof


def open_formal():
    proof = verify_preparation()
    full = OUT/'t20b_execution_full_pytest_final.txt'
    related = OUT/'t20b_execution_related_pytest_final_v2.txt'
    summary = frozen.passed_summary(full,1946)
    frozen.passed_summary(related,50)
    subprocess.check_call(['git','diff','--check'],cwd=ROOT)
    status = subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    if status.strip():
        raise ValueError('Formal execution preparation must be committed and tree clean before opening')
    if UNSEAL.exists() or any(OUT.glob('t20b_formal_data/*')) or (OUT/(PREFIX+'trajectory_metrics.csv')).exists():
        raise ValueError('Formal signals/results already exist')
    contract = load(OUT/'t20a_results.json')['formal_protocol']
    record = dict(task='T20B',timestamp=datetime.now(timezone.utc).isoformat(),
        HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        pre_unseal_governance_commit=COMMIT, original_protocol_sha256=HASHES[governance.PROTOCOL],
        readiness_amendment_sha256=HASHES[governance.AMENDMENT],seed_governance_amendment_sha256=HASHES[assignment.AMENDMENT],
        formal_seed_manifest_sha256=HASHES[assignment.MANIFEST],rejected_original_seed_plan_sha256=HASHES[assignment.REJECTED],
        method_definition_sha256=proof['frozen_method_definition_sha256'],all_method_factor_fold_hashes=proof['frozen_method'],
        all_code_sha256=proof['source_sha256'],reference_statistics_sha256=proof['reference_statistics_sha256'],
        frozen_core_source_sha256={p:h for p,h in proof['historical_sha256'].items() if p.startswith(('src/','scripts/'))},
        frozen_generator_configurations=proof['historical_condition_configuration'],
        original_formal_contract=contract,formal_conditions=list(metrics.CONDITIONS),snr=list(metrics.SNRS),
        trajectories_per_condition=1000,frames_per_trajectory=10,scored_frames=list(range(4,10)),methods=list(metrics.METHODS),
        primary_endpoints={'A':'all six lower .005 >0','B':'both lower .005 >0','C':'both upper .995 <.50'},
        bootstrap={'resamples':10000,'seed':20262001,'unit':'trajectory','paired':True,'quantile_method':'linear'},
        source_receive_power=method.POWER,noise_variances={str(s):method.VARIANCES[s] for s in metrics.SNRS},
        seed_amendment_before_any_formal_signal_or_performance_observation=True,
        pre_unseal_formal_H_count=0,pre_unseal_formal_NMSE_count=0,pre_unseal_formal_BER_count=0,
        full_pytest_summary=summary,full_pytest_sha256=frozen.sha(full),
        execution_preflight_sha256=frozen.sha(PRE),formal_test_used=True,formal_test_sealed=False,formal_test_unsealed=True,
        permanent_data_use='test evidence only; never training/module/rank/threshold/hyperparameter/baseline selection')
    write(UNSEAL,record)
    print('FORMAL UNSEALED; immutable record saved before first formal H',flush=True)
    return record


def channel_grid(coefficients, sparse, g):
    """Data prediction plus known pilot-LS; nonobserved nondata entries not scored."""
    value = sparse.copy().ravel()
    value[g.data] = coefficients
    return value.reshape(2,2,64,16)


def evaluate(h, communication_frames, arrays, family, cvs, matched, *, timing):
    """All methods share inputs. Estimator call has no truth/label/target argument."""
    g = observation_geometry()
    truth_frames = frames(h)
    input_rows, frame_rows, weights = [], [], []
    pilot_series, cache = {}, {}
    signal = noise = 0.; count = 0
    for j,(bits,symbols,z) in enumerate(communication_frames):
        truth = truth_frames[j]
        x = assemble_pilot_data_frame(symbols,g.layout)
        clean = apply_frequency_mimo_channel(truth,x)
        data = g.layout.data_mask[0]
        signal += float(np.sum(abs(clean[:,data])**2)); noise += float(np.sum(abs(z[:,data])**2)); count += 2*int(data.sum())
        reference = extract_reference_data_bits(bits,g.layout.data_mask)
        for snr in metrics.SNRS:
            y = clean+np.sqrt(method.VARIANCES[snr])*z
            y.flags.writeable=False
            sparse,linear = estimate_ls_linear_channel(y,g.layout)
            pilot = sparse.ravel()[g.selected]; pilot.flags.writeable=False
            pilot_series.setdefault(snr,[]).append(pilot)
            cache[snr,j] = y,sparse,linear,reference
            input_rows.append(dict(frame_id=j,snr_db=snr,H_sha256=array_hash(truth),bits_sha256=array_hash(bits),
                symbols_sha256=array_hash(symbols),unit_noise_sha256=array_hash(z),Y_sha256=array_hash(y),pilot_sha256=array_hash(pilot)))
    inferred = {}
    for snr in metrics.SNRS:
        tick = time.perf_counter()
        inferred[snr] = method.deployed(np.stack(pilot_series[snr]),arrays,snr,cvs[snr],family['P',snr,4],family['M',snr,4])
        timing['pilot_fitting_and_frozen_deployment_seconds'] += time.perf_counter()-tick
        for row in inferred[snr]:
            j = row['frame_id']; truth = truth_frames[j]; y,sparse,linear,reference = cache[snr,j]
            if row['dense_coefficients'] is not row['lr4_coefficients'] or not np.array_equal(row['dense_coefficients'],row['lr4_coefficients']):
                raise ValueError('Dense/LR4 coefficient identity drift')
            alpha,beta = map(float,row['coefficients']['affine']); lam = 1-alpha-beta
            weights.append(dict(frame_id=j,snr_db=snr,alpha=alpha,beta=beta,lambda0=lam,
                outside_simplex=bool(alpha<0 or beta<0 or lam<0),dense_lr4_coefficients_exact=True,
                source_cv_loss=row['diagnostics']['affine']['source_cv_loss'],fitted_cv_loss=row['diagnostics']['affine']['fitted_cv_loss']))
            # Separate descriptive timing of identical frozen prediction calls;
            # these repeats are checked exact and never influence coefficients.
            args=(pilot_series[snr][j],arrays['source_mean_p'],arrays['source_mean_d'],arrays[f'Wsource_{snr}'])
            tick=time.perf_counter()
            dense=deployment(*args,arrays[f'DeltaWP_{snr}'],arrays[f'DeltaWM_{snr}'],row['coefficients']['affine'],('P','M'))
            timing['dense_inference_recheck_seconds']+=time.perf_counter()-tick
            tick=time.perf_counter()
            low=factored_affine_predict(*args,row['coefficients']['affine'],family['P',snr,4],family['M',snr,4])
            timing['lr4_inference_recheck_seconds']+=time.perf_counter()-tick
            if not np.array_equal(dense,row['predictions']['dense-affine']) or not np.array_equal(low,row['predictions']['lr4-affine']):
                raise ValueError('Frozen prediction exact recheck failed')
            predictions={MAP[k]:v for k,v in row['predictions'].items()}
            predictions['Perfect CSI']=truth.ravel()[g.data]
            predictions['R3 LS+linear']=linear.ravel()[g.data]
            predictions['Matched LMMSE reference']=matched[snr].estimate(pilot_series[snr][j])
            energy=float(np.sum(abs(truth.ravel()[g.data])**2)); before=array_hash(y)
            for label in metrics.METHODS:
                prediction=predictions[label]
                if not np.isfinite(prediction).all():
                    raise ValueError('Nonfinite formal prediction')
                estimate=truth if label=='Perfect CSI' else linear if label=='R3 LS+linear' else channel_grid(prediction,sparse,g)
                tick=time.perf_counter()
                bits_hat=demodulate_data_bits(detect_data_zf(y,estimate,g.layout.data_mask)*np.sqrt(2),g.layout.data_mask)
                timing['BER_ZF_seconds']+=time.perf_counter()-tick
                error=float(np.sum(abs(prediction-truth.ravel()[g.data])**2))
                bit_errors=sum(int(np.count_nonzero(a!=b)) for a,b in zip(bits_hat,reference))
                total_bits=sum(a.size for a in reference)
                frame_rows.append(dict(frame_id=j,snr_db=snr,method=label,error_energy=error,truth_energy=energy,
                    bit_errors=bit_errors,total_bits=total_bits,prediction_sha256=array_hash(prediction),
                    shared_Y_sha256=before,shared_pilot_sha256=array_hash(pilot_series[snr][j])))
                if array_hash(y)!=before:
                    raise ValueError('Paired received grid mutated')
    accum=[]
    for snr in metrics.SNRS:
        for label in metrics.METHODS:
            rows=[r for r in frame_rows if (r['snr_db'],r['method'])==(snr,label)]
            if [r['frame_id'] for r in rows]!=list(range(4,10)):
                raise ValueError('Scoring subset drift')
            # Original sequential frame reduction, without averaging dB.
            e=dict(error_energy=0.,truth_energy=0.,bit_errors=0,total_bits=0)
            for r in rows:
                for key in e:
                    e[key]+=r[key]
            accum.append(dict(snr_db=snr,method=label,**e))
    return accum,frame_rows,input_rows,weights,dict(signal_energy=signal,unit_noise_energy=noise,receive_samples=count)


def run():
    start=time.perf_counter()
    record=open_formal()
    journals=[]; generated=completed=0; seen_results=[]
    timing=dict(generation_seconds=0.,matched_reference_preparation_seconds=0.,
        pilot_fitting_and_frozen_deployment_seconds=0.,dense_inference_recheck_seconds=0.,
        lr4_inference_recheck_seconds=0.,BER_ZF_seconds=0.,bootstrap_seconds=0.,plotting_seconds=0.,audit_seconds=0.)
    try:
        proof=verify_preparation(); index=seed_index(); arrays,family,_=frozen_method()
        cvs={s:cv_family(arrays,s) for s in metrics.SNRS}
        known=set(proof['historical_comparison_hashes']); new=set()
        data_root=OUT/(PREFIX+'data'); data_root.mkdir()
        metric_fields=('environment','trajectory_id','snr_db','method','error_energy','truth_energy','bit_errors','total_bits')
        trajectory_j=Journal(OUT/(PREFIX+'trajectory_metrics.csv'),metric_fields);journals.append(trajectory_j)
        frame_j=Journal(OUT/(PREFIX+'frame_metrics.csv'),('environment','trajectory_id','frame_id','snr_db','method','error_energy','truth_energy','bit_errors','total_bits','prediction_sha256','shared_Y_sha256','shared_pilot_sha256'));journals.append(frame_j)
        input_j=Journal(OUT/(PREFIX+'input_hashes.csv'),('environment','trajectory_id','frame_id','snr_db','H_sha256','bits_sha256','symbols_sha256','unit_noise_sha256','Y_sha256','pilot_sha256'));journals.append(input_j)
        weight_j=Journal(OUT/(PREFIX+'weights.csv'),('environment','trajectory_id','frame_id','snr_db','alpha','beta','lambda0','outside_simplex','dense_lr4_coefficients_exact','source_cv_loss','fitted_cv_loss'));journals.append(weight_j)
        manifest_j=Journal(OUT/(PREFIX+'manifest.csv'),('environment','namespace','trajectory_id','frame_id','start_symbol','end_symbol',*frozen.ROLES,'H_trajectory_sha256','raw_artifact','raw_artifact_sha256'));journals.append(manifest_j)
        power_rows=[];all_rows=[];all_weights=[]
        for name,statpath in zip(metrics.CONDITIONS,STATISTICS):
            adapter=FormalCDL(name,index)
            tick=time.perf_counter();statistics=frozen_statistics(statpath)
            matched={s:statistics.prepare(method.VARIANCES[s]) for s in metrics.SNRS}
            timing['matched_reference_preparation_seconds']+=time.perf_counter()-tick
            power=dict(signal_energy=0.,unit_noise_energy=0.,receive_samples=0)
            for i in range(600,1600):
                tick=time.perf_counter();h=adapter.generate(i);generated+=1;timing['generation_seconds']+=time.perf_counter()-tick
                path=data_root/f'{name}_{i}.npz'
                # Immediately preserve generated H before any further operation.
                np.savez(path,H=h)
                hh=array_hash(h)
                if hh in known or hh in new:
                    raise ValueError('Formal H duplicate or historical intersection')
                new.add(hh)
                comm=[communication(consumed(index,name,i,j)) for j in range(10)]
                # Complete raw stream persisted before any observed performance.
                raw=data_root/f'{name}_{i}_communication.npz'
                np.savez(raw,bits=np.stack([v[0] for v in comm]),symbols=np.stack([v[1] for v in comm]),unit_noise=np.stack([v[2] for v in comm]))
                for j in range(10):
                    manifest_j.append(dict(environment=name,namespace=assignment.namespace(name),trajectory_id=i,frame_id=j,
                        start_symbol=16*j,end_symbol=16*j+15,**consumed(index,name,i,j),H_trajectory_sha256=hh,
                        raw_artifact=path.relative_to(ROOT).as_posix(),raw_artifact_sha256=frozen.sha(path)))
                before=[array_hash(h)]+[array_hash(a) for triple in comm for a in triple]
                rows,frame_rows,inputs,weights,p=evaluate(h,comm,arrays,family,cvs,matched,timing=timing)
                if before!=[array_hash(h)]+[array_hash(a) for triple in comm for a in triple]:
                    raise ValueError('Formal H/bits/noise streams mutated')
                for journal,values in ((trajectory_j,rows),(frame_j,frame_rows),(input_j,inputs),(weight_j,weights)):
                    for r in values:
                        journal.append(dict(environment=name,trajectory_id=i,**r))
                all_rows.extend(dict(environment=name,trajectory_id=i,**r) for r in rows)
                all_weights.extend(dict(environment=name,trajectory_id=i,**r) for r in weights)
                for key in power:power[key]+=p[key]
                completed+=1;seen_results.append((name,i))
                if (i-599)%25==0:
                    print(f'{name}: {i-599}/1000; total {completed}/5000; elapsed {time.perf_counter()-start:.1f}s',flush=True)
            for s in metrics.SNRS:
                signal=power['signal_energy']/power['receive_samples'];unit=power['unit_noise_energy']/power['receive_samples']
                power_rows.append(dict(environment=name,snr_db=s,ensemble_receive_power=signal,unit_noise_power=unit,
                    source_reference_power=method.POWER,sigma2=method.VARIANCES[s],
                    realised_receive_snr_db=float(10*np.log10(signal/(method.VARIANCES[s]*unit))),
                    target_power_not_used_for_noise=True))
        for journal in journals:journal.close()
        aggregate=metrics.aggregate(all_rows)
        nmse=[{k:v for k,v in r.items() if k not in ('bit_errors','total_bits','ber')} for r in aggregate]
        ber=[{k:v for k,v in r.items() if k not in ('error_energy','truth_energy','nmse_linear','nmse_db')} for r in aggregate]
        csvwrite(OUT/(PREFIX+'nmse.csv'),nmse);csvwrite(OUT/(PREFIX+'ber.csv'),ber)
        csvwrite(OUT/(PREFIX+'power_audit.csv'),power_rows)
        tick=time.perf_counter();boot=metrics.bootstrap(all_rows);timing['bootstrap_seconds']=time.perf_counter()-tick
        csvwrite(OUT/(PREFIX+'bootstrap.csv'),boot)
        endpoints=metrics.endpoints(boot);weights=metrics.weight_summary(all_weights)
        write(OUT/(PREFIX+'weight_diagnostics.json'),weights)
        from scripts.plot_formal_test import plot
        tick=time.perf_counter();plot(OUT);timing['plotting_seconds']=time.perf_counter()-tick
        tick=time.perf_counter();verify_preparation()
        if len(new)!=5000 or new&known:raise ValueError('Final H independence audit failed')
        timing['audit_seconds']=time.perf_counter()-tick;timing['total_execution_seconds']=time.perf_counter()-start
        csvwrite(OUT/(PREFIX+'runtime.csv'),[dict(stage=k,seconds=v) for k,v in timing.items()])
        result=dict(task='T20B',experiment_type='sealed formal evaluation of the frozen pilot-driven rank-4 affine LMMSE adaptation method',
            formal_test_unsealed=True,formal_test_used=True,formal_test_sealed=False,
            protocol_sha256=record['original_protocol_sha256'],readiness_amendment_sha256=record['readiness_amendment_sha256'],
            seed_governance_amendment_sha256=record['seed_governance_amendment_sha256'],formal_seed_manifest_sha256=record['formal_seed_manifest_sha256'],
            pre_unseal_governance_commit=COMMIT,execution_commit=record['HEAD'],method_sha256=record['method_definition_sha256'],
            formal_conditions=list(metrics.CONDITIONS),trajectory_count=5000,trajectories_per_condition=1000,frames_per_trajectory=10,
            snr=list(metrics.SNRS),methods=list(metrics.METHODS),nmse_results=nmse,ber_results=ber,bootstrap_results=boot,
            weight_diagnostics=weights,runtime=timing,**endpoints,boundary_results=[r for r in aggregate if r['environment']==metrics.CONDITIONS[-1]],
            power_audit=power_rows,complexity=load(OUT/'t20a_results.json')['complexity'],
            independence=dict(unique_formal_H=5000,historical_H_intersection=0,historical_all_sha256_superset_checked=len(known),
                historical_seed_overlap=0,formal_seed_duplicates=0,namespace_overlap=0),
            paired_H_bits_noise_Y_pilots=True,unit_noise_shared_across_SNR=True,dense_lr4_coefficients_exact=True,
            primary_estimator_uses_environment_label=False,primary_estimator_uses_matched_statistics=False,
            method_modified_after_unseal=False,new_model_trained=False,new_covariance_fitted=False,
            frozen_history_content_mismatch=0,completion_state='formal_numerics_complete_final_engineering_pending',
            formal_engineering_passed=False,all_checks_passed=False,
            warnings=['Fixed-environment scope only; approximate percentile-bootstrap coverage',
                'Matched is an evaluation-only target-statistics reference',
                'Offline clean development statistics are used; not fully blind adaptation',
                'Arithmetic proxies are not measured runtime speedups',
                'Frozen deployment bundle timing includes pilot fitting and all five deployed predictions; separate dense/LR4 rechecks are descriptive'],
            permanent_test_only=True,source_sha256=proof['source_sha256'])
        write(OUT/(PREFIX+'numerical_results.json'),result)
        documents(result)
        print('All formal conditions completed; endpoints:',endpoints,flush=True)
    except BaseException as exc:
        for journal in journals:
            if not journal.stream.closed:journal.close()
        write(OUT/(PREFIX+'STOP.json'),dict(error_type=type(exc).__name__,message=str(exc),
            generated_formal_trajectories=generated,completed_trajectories=completed,observed_trajectory_results=seen_results,
            formal_test_used=True,formal_test_sealed=False,formal_test_unsealed=True,
            action='STOP; no repair/restart of this same formal test namespace',timestamp=datetime.now(timezone.utc).isoformat()))
        raise


def documents(result):
    """Evidence summaries generated from formal endpoints, no manuscript prose."""
    a,b,c=(result[f'endpoint_{f}_pass'] for f in 'ABC')
    statuses={'P1':'PARTIALLY_SUPPORTED' if a else 'NOT_SUPPORTED',
        'P2':'FORMALLY_SUPPORTED' if a else 'NOT_SUPPORTED','P3':'NOT_FORMAL_ENDPOINT',
        'P4':'FORMALLY_SUPPORTED' if c else 'NOT_SUPPORTED','P5':'NOT_FORMAL_ENDPOINT','P6':'BOUNDARY_ONLY'}
    text=['# T20B formal claim evidence update','',
        'Historical T20A evidence and all previous positive/negative claim states remain unchanged.',
        'This update refers only to the frozen five conditions, not universal generalization.', '', '| Claim | Formal status | Evidence / scope |','|---|---|---|']
    explanations={'P1':'Source–Matched NMSE gap is descriptive; final efficacy is endpoint A.',
        'P2':'Endpoint A: frozen LR4 vs Source in profile/mobility/composed shifts.',
        'P3':'Pilot-only interface is engineering-verified; no oracle-coordinate efficiency endpoint.',
        'P4':'Endpoint C applies only to composed shift; other penalties remain secondary. MAC reduction is a proxy.',
        'P5':'Earlier fixed-speed replication retained; formal evaluation does not add unseen-speed conditions.',
        'P6':'Both delay SNR cells retained independently; no operator projection/refit performed.'}
    for p,s in statuses.items():text.append(f'| {p} | {s} | {explanations[p]} |')
    text+=['',f'Endpoint A/B/C: {a}/{b}/{c}.',
        'Not passing is NOT_SUPPORTED_BY_FORMAL_ENDPOINT, not a contradiction claim.',
        'Formal data are permanently test-only; never used for future tuning or selection.','']
    (ROOT/'docs/t20b_formal_claim_evidence_update.md').write_text('\n'.join(text),encoding='utf8')
    contributions={'C-A':'strengthened' if a and b else 'weakened' if a else 'unsupported',
                   'C-B':'unchanged','C-C':'strengthened' if c else 'weakened'}
    (ROOT/'docs/t20b_contribution_status.md').write_text('\n'.join(['# T20B contribution status','',
        'Only the three T20A retained candidates are assessed; no new novelty or priority claim.',
        *[f'- {k}: {v}' for k,v in contributions.items()],
        '', 'C-A is scoped continuous adaptation; C-B is analytical pilot inference with offline development information;',
        'C-C is rank4 fidelity and arithmetic/storage proxy, not runtime superiority.','']),encoding='utf8')
    attacks=[('Untouched formal test','Seed amendments and engineering preflight precede immutable unseal; all formal streams generated only after record.'),
        ('LR4 vs Source',f'Endpoint A pass={a}; each of six one-sided corrected bounds is retained.'),
        ('Affine vs Convex',f'Endpoint B pass={b}; no >=1dB materiality claim added.'),
        ('Rank4 fidelity',f'Endpoint C pass={c}; all secondary conditions retained, no rank change.'),
        ('BER versus NMSE','All eighty method/SNR/condition BER cells and paired intervals are reported; BER has no success threshold.'),
        ('Delay boundary','Both delay cells are reported independently and never merge into readiness.'),
        ('Matched oracle attribute','Matched alone receives frozen target means/statistics and is evaluation-only.'),
        ('Offline clean information','Source and two modules use clean development statistics; method is not fully blind.'),
        ('Cherry-picking','All five preregistered conditions, all 1000 trajectories each, both SNRs and eight methods retained.'),
        ('Zero method mutation','All code/factor/fold/mean/noise hashes checked before and after execution; no method mutation allowed.')]
    lines=['# T20B formal reviewer red team','']
    for title,answer in attacks:lines.extend([f'## {title}','',answer,''])
    (ROOT/'docs/t20b_formal_reviewer_red_team.md').write_text('\n'.join(lines),encoding='utf8')


def finalize():
    if (OUT/(PREFIX+'STOP.json')).exists():
        raise ValueError('Formal STOP record prohibits finalization')
    proof=verify_preparation();result=load(OUT/(PREFIX+'numerical_results.json'))
    summary=frozen.passed_summary(OUT/'t20b_formal_full_pytest.txt',1946)
    related=frozen.passed_summary(OUT/'t20b_formal_related_pytest.txt',50)
    subprocess.check_call(['git','diff','--check'],cwd=ROOT)
    paths=[p for p in OUT.glob(PREFIX+'*') if p.is_file() and p.name not in (PREFIX+'results.json',PREFIX+'output_integrity.json')]
    paths+=list((OUT/(PREFIX+'data')).glob('*.npz'))
    hashes={p.relative_to(ROOT).as_posix():frozen.sha(p) for p in paths}
    for p in ROOT.glob('docs/t20b_*status.md'):hashes[p.relative_to(ROOT).as_posix()]=frozen.sha(p)
    for p in ('docs/t20b_formal_claim_evidence_update.md','docs/t20b_formal_reviewer_red_team.md'):hashes[p]=frozen.sha(ROOT/p)
    write(OUT/(PREFIX+'output_integrity.json'),dict(hashes=hashes,files=len(hashes),mismatches=0))
    result.update(completion_state='completed',formal_engineering_passed=True,all_checks_passed=True,
        full_pytest_summary=summary,related_pytest_summary=related,git_diff_check='PASS',
        output_integrity_sha256=frozen.sha(OUT/(PREFIX+'output_integrity.json')),formal_artifact_sha256=hashes,
        historical_files_verified=len(proof['historical_sha256']),historical_unexpected_mismatch=0)
    write(OUT/(PREFIX+'results.json'),result)
    print('Formal engineering PASS; full pytest:',summary,'; history0 mismatch; scientific readiness:',result['paper_results_ready'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('preflight','run','finalize','audit'))
    args=parser.parse_args()
    if args.stage=='preflight':preparation()
    elif args.stage=='run':run()
    elif args.stage=='finalize':finalize()
    else:
        verify_preparation()
        if (OUT/(PREFIX+'results.json')).exists():frozen.verify_hashes(load(OUT/(PREFIX+'results.json'))['formal_artifact_sha256'])
        print('Formal protocol/method/history/artifact hash audit PASS')


if __name__=='__main__':
    main()
