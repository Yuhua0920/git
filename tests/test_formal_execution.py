"""Pre-unseal deterministic mathematics/receiver checks, plus actual artifact regression."""
import ast
import inspect
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from src import formal_evaluation as metrics
from src import frozen_affine_validation as frozen_method
from scripts import run_formal_test as run


def fixture_rows(conditions=metrics.CONDITIONS):
    rows=[]
    for name in conditions:
        for snr in metrics.SNRS:
            for method in metrics.METHODS:
                for i,truth in enumerate((1.,9.)):
                    ratio={'Perfect CSI':0.,'Source LMMSE':4.,'Convex':3.,'Dense Affine':1.,'Final LR4 Affine':1.05}.get(method,2.)
                    rows.append(dict(environment=name,snr_db=snr,method=method,trajectory_id=i,
                        error_energy=truth*ratio,truth_energy=truth,bit_errors=i,total_bits=10))
    return rows


def test_pooled_energy_not_average_db_or_ratios():
    rows=[dict(error_energy=1.,truth_energy=1.,bit_errors=1,total_bits=2),
          dict(error_energy=1.,truth_energy=9.,bit_errors=1,total_bits=18)]
    result=metrics.pooled(rows)
    assert result['nmse_linear']==.2 and result['ber']==.1
    assert result['nmse_db']==float(10*np.log10(.2))


def test_bootstrap_known_constant_ratio_paired_oracle():
    rows=fixture_rows()
    result=metrics.bootstrap(rows,resamples=32)
    primary=[r for r in result if r['primary_endpoint']]
    assert len(primary)==10
    source_gain=10*np.log10(4/1.05)
    for r in primary:
        if r['primary_endpoint']=='A':
            assert r['primary_lower']==pytest.approx(source_gain,abs=1e-14)
            assert r['primary_upper']==pytest.approx(source_gain,abs=1e-14)
    assert metrics.endpoints(result)['paper_results_ready']
    assert metrics.bootstrap(rows,resamples=32)==result


@pytest.mark.parametrize('a,b,c',list(itertools.product((False,True),repeat=3)))
def test_formal_readiness_truth_table_all_eight_states(a,b,c):
    rows=metrics.bootstrap(fixture_rows(),resamples=4)
    for row in rows:
        if row['primary_endpoint']:row['endpoint_cell_pass']={'A':a,'B':b,'C':c}[row['primary_endpoint']]
    result=metrics.endpoints(rows)
    assert result['paper_core_empirically_supported']==a
    assert result['paper_results_ready']==(a and b and c)
    assert result['paper_results_ready_partial']==(a and (b!=c))


def test_endpoint_equalities_fail_and_boundary_does_not_merge():
    rows=fixture_rows()
    for r in rows:
        if r['method']=='Final LR4 Affine':r['error_energy']=r['truth_energy']*4.
    result=metrics.bootstrap(rows,resamples=8)
    endpoint=metrics.endpoints(result)
    assert endpoint['endpoint_A_pass'] is False
    assert endpoint['endpoint_A_status']=='NOT_SUPPORTED_BY_FORMAL_ENDPOINT'
    assert all(r['primary_endpoint']=='' for r in result if r['environment']==metrics.CONDITIONS[-1])


def test_missing_or_duplicate_bootstrap_population_rejected():
    rows=fixture_rows()
    with pytest.raises(ValueError,match='Incomplete'):metrics.bootstrap(rows[:-1],resamples=2)
    with pytest.raises(ValueError,match='duplicate'):metrics.aggregate(rows+[rows[0]])


def test_primary_inference_signature_has_no_label_truth_or_matched():
    assert tuple(inspect.signature(frozen_method.deployed).parameters)==('pilots','arrays','snr','cv','profile_factor','mobility_factor')
    source=inspect.getsource(run.evaluate)
    tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='deployed']
    assert len(calls)==1 and len(calls[0].args)==6 and not calls[0].keywords
    assert 'matched' not in ast.unparse(calls[0]) and 'truth' not in ast.unparse(calls[0])


def test_official_generator_numerical_body_is_reused_and_restored(monkeypatch):
    import src.cdl_benchmark as benchmark
    original=benchmark.seeds
    adapter=object.__new__(run.FormalCDL);adapter.name=metrics.CONDITIONS[0]
    adapter.index={(adapter.name,600,0,k):j+1 for j,k in enumerate(run.frozen.ROLES)}
    def failing(self,trajectory):
        assert benchmark.seeds(trajectory)['sionna_seed']==1
        raise RuntimeError('sentinel')
    monkeypatch.setattr(benchmark.OfficialCDL,'generate',failing)
    with pytest.raises(RuntimeError,match='sentinel'):adapter.generate(600)
    assert benchmark.seeds is original


def test_formal_seed_lookup_is_frozen_complete_and_trajectory_scoped():
    index=run.seed_index()
    assert len(index)==165000
    for name in metrics.CONDITIONS:
        a=run.consumed(index,name,600,0);b=run.consumed(index,name,600,9)
        assert [a[k] for k in run.frozen.ROLES[:3]]==[b[k] for k in run.frozen.ROLES[:3]]
        assert all(a[k]!=b[k] for k in run.frozen.ROLES[3:])
    with pytest.raises(KeyError):run.consumed(index,metrics.CONDITIONS[0],1600,0)


def test_channel_grid_keeps_known_pilot_observations_without_truth():
    g=run.observation_geometry();sparse=np.full((2,2,64,16),np.nan+1j*np.nan)
    sparse.ravel()[g.selected]=3+2j
    prediction=np.full(3968,1-2j,np.complex128)
    grid=run.channel_grid(prediction,sparse,g)
    np.testing.assert_array_equal(grid.ravel()[g.data],prediction)
    np.testing.assert_array_equal(grid.ravel()[g.selected],3+2j)
    assert 'truth' not in inspect.signature(run.channel_grid).parameters


def test_unseal_requires_actual_passing_summary_and_clean_snapshot(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'OUT',tmp_path)
    monkeypatch.setattr(run,'verify_preparation',lambda:{})
    with pytest.raises(FileNotFoundError):run.open_formal()
    assert not (tmp_path/'t20b_formal_unseal.json').exists()


def test_exclusive_artifact_creation(tmp_path):
    path=tmp_path/'sentinel.json'
    run.write(path,{'math_fixture':True})
    old=path.read_bytes()
    with pytest.raises(FileExistsError):run.write(path,{'overwrite':True})
    assert path.read_bytes()==old


def test_bootstrap_serialization_and_plot_from_only_csv(tmp_path,monkeypatch):
    rows=metrics.bootstrap(fixture_rows(),resamples=32)
    run.csvwrite(tmp_path/'t20b_formal_bootstrap.csv',rows)
    from scripts.plot_formal_test import plot
    monkeypatch.setattr(run.FormalCDL,'generate',lambda *_:pytest.fail('generator forbidden'))
    plot(tmp_path)
    assert len(list(tmp_path.glob('*.png')))==3
    assert len(list(tmp_path.glob('*.pdf')))==3
    assert len(list(tmp_path.glob('*.svg')))==6  # figures plus QA overlays
    for p in tmp_path.glob('*.alignment.json'):
        assert p.stat().st_size>0


def test_preflight_actual_evidence_when_materialized():
    if not run.PRE.exists():
        return  # Pure math checks run first; opening itself requires actual file.
    proof=run.verify_preparation()
    assert proof['historical_receiver']['all_exact']
    assert proof['historical_generator']['H_exact']
    assert proof['formal_H_count']==proof['formal_NMSE_count']==proof['formal_BER_count']==0
    assert proof['historical_condition_configuration']['all_five_configurations_exact']


def test_formal_actual_result_regression_when_materialized():
    path=run.OUT/'t20b_formal_numerical_results.json'
    if not path.exists():
        return  # Not a scientific pass; run final audit requires actual result.
    result=run.load(path);run.verify_preparation()
    assert result['trajectory_count']==5000 and result['trajectories_per_condition']==1000
    assert result['formal_test_used'] and not result['formal_test_sealed']
    assert not result['method_modified_after_unseal'] and not result['new_covariance_fitted']
    rows=run.readcsv(run.OUT/'t20b_formal_trajectory_metrics.csv')
    assert len(rows)==80000
    numeric=[dict(**{k:v for k,v in r.items() if k not in ('snr_db','trajectory_id','error_energy','truth_energy','bit_errors','total_bits')},
        snr_db=int(r['snr_db']),trajectory_id=int(r['trajectory_id']),error_energy=float(r['error_energy']),truth_energy=float(r['truth_energy']),
        bit_errors=int(r['bit_errors']),total_bits=int(r['total_bits'])) for r in rows]
    agg=metrics.aggregate(numeric)
    for r in agg:
        expected=next(x for x in result['nmse_results'] if (x['environment'],x['snr_db'],x['method'])==(r['environment'],r['snr_db'],r['method']))
        assert r['error_energy']==expected['error_energy'] and r['truth_energy']==expected['truth_energy']
        expected=next(x for x in result['ber_results'] if (x['environment'],x['snr_db'],x['method'])==(r['environment'],r['snr_db'],r['method']))
        assert r['bit_errors']==expected['bit_errors'] and r['total_bits']==expected['total_bits']
    manifest=run.readcsv(run.OUT/'t20b_formal_manifest.csv')
    assert len(manifest)==50000 and len({r['H_trajectory_sha256'] for r in manifest})==5000
    proof=run.load(run.PRE)
    assert not {r['H_trajectory_sha256'] for r in manifest}&set(proof['historical_comparison_hashes'])
    assert metrics.endpoints(result['bootstrap_results'])['paper_results_ready']==result['paper_results_ready']
