"""T17B frozen provenance, no-generator boundary and actual-result regression."""
import ast
import inspect
import json
import numpy as np
import pytest
from pathlib import Path
from scripts import run_finite_shift_resolvent as run
from scripts import plot_finite_shift_resolvent as plot
from scripts.audit_finite_shift_resolvent import verify
from src import finite_shift_resolvent as exact
from src.cdl_benchmark import OfficialCDL,StreamingStatistics
from src.cdl_mismatch import ShiftCDL


def test_committed_T17A_and_frozen_covariance_W_provenance():
    old,d,p,v,proof=run.load_statistics();maps,w=run.prepare(d,p,v,old)
    assert len(proof)==4 and len(w)==12
    assert old['c6_claim_status']==run.C6
    assert all(x['historical_hash_exact'] for x in proof)
    assert all(x['exact_T17A_recorded_hash'] for x in w)
    assert set(v)=={5,10,20} and set(d)==set(exact.NAMES)


def test_frozen_loader_cannot_call_generator_or_refit(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('generator/statistics fit forbidden')
    for cls,names in [(OfficialCDL,('__init__','generate')),(ShiftCDL,('__init__','generate')),(StreamingStatistics,('__init__','update','freeze'))]:
        for name in names:monkeypatch.setattr(cls,name,forbidden)
    old,d,p,v,proof=run.load_statistics();run.prepare(d,p,v,old)


def test_primary_source_has_no_Taylor_inverse_or_generator_calls():
    for module in (run,exact):
        calls={getattr(n.func,'id',getattr(n.func,'attr','')) for n in ast.walk(ast.parse(inspect.getsource(module))) if isinstance(n,ast.Call)}
        assert not calls&{'generate','communication','fit','StreamingStatistics','WienerSensitivity','inv','pinv','first','second','cross'}
        updates=[n for n in ast.walk(ast.parse(inspect.getsource(module))) if isinstance(n,ast.Call)
                 and isinstance(n.func,ast.Attribute) and n.func.attr=='update']
        # These receivers are plain result dictionaries; fitting methods are
        # separately forbidden at runtime by the generator/refit sentinel.
        assert all(isinstance(n.func.value,ast.Name) and n.func.value.id in {'row','r','runtime','record'} for n in updates)
    assert 'evaluate(' not in inspect.getsource(run.preflight)


def test_mixed_optional_rows_serialize_actual_writer_without_missing_fields(tmp_path):
    path=tmp_path/'counter.csv';run.csvwrite(path,[dict(snr_db=5,available=False),dict(snr_db=10,available=True,energy=3.)])
    rows=plot.rows(path);assert rows[0]['energy']=='' and float(rows[1]['energy'])==3.
    json.dumps(dict(value=np.float64(3.).item(),available=bool(np.bool_(True))),allow_nan=False)


def plotting_fixture(tmp_path):
    terms=[];cross=[]
    for s in run.SNRS:
        c={k:np.array([[v]],complex) for k,v in zip(exact.TERMS,(1,-.5,.25,-.25))}
        r,rows=exact.weighted_bookkeeping(np.array([[.5]]),np.array([[1.]]),c,np.array([[2.]]))
        terms.extend(dict(snr_db=s,**v) for v in rows)
        terms.append(dict(snr_db=s,term='R_exact',relative_to_complete_correction=.5,normalized_energy_relative_to_complete_correction=.25))
        for i,a in enumerate(exact.TERMS):
            for j,b in enumerate(exact.TERMS):cross.append(dict(snr_db=s,term_i=a,term_j=b,
                normalized_cross_energy=r['cross_energy_matrix'][i][j]/4))
    for name,values in zip(plot.DATA,(terms,cross)):run.csvwrite(tmp_path/name,values)
    return terms,cross


def test_CSV_only_plot_QA_and_scoped_style(tmp_path):
    plotting_fixture(tmp_path);before=dict(plot.matplotlib.rcParams)
    r=plot.audit(tmp_path,tmp_path/'figures')
    assert all(r['CSV_only_rebuild'].values()) and r['source_returncode']==0
    assert dict(plot.matplotlib.rcParams)==before


def test_figure_rejects_incomplete_or_nonfinite_CSV(tmp_path):
    terms,cross=plotting_fixture(tmp_path)
    with pytest.raises(ValueError):plot.series(terms[:-1],cross)
    terms[0]['relative_to_complete_correction']=float('nan')
    with pytest.raises(ValueError):plot.series(terms,cross)


def test_actual_result_frozen_provenance_and_exact_gate():
    if not run.RESULT.exists():pytest.skip('actual T17B numerical result not yet generated')
    r=run.load(run.RESULT);assert verify(r)
    assert r['formal_test_sealed'] and r['c6_claim_status']==run.C6
    assert len(r['contribution_energies'])==15 and len(r['cross_energy_matrix'])==48
    assert len(r['interaction_spectrum'])==1920 and len(r['rank_diagnostics'])==30 and len(r['subspace_diagnostics'])==48
    assert all(v['original_reduction_exact'] for v in r['exact_affine_regression'])
    assert r['scientific_gate_thresholds']==dict(closure=.90,structural_index=2.,structural_size=.50,cancellation_index=10.,cancellation_size=1.)
    for row in r['four_term_decomposition']:
        assert abs(row['cancellation_index']+row['cross_fraction']-1)<1e-10
        assert row['energy_identity_relative_residual']<=1e-11
        assert row['four_term_relative_residual']<=1e-11
        assert all(v['action_relative_residual']<=1e-11 for v in row['resolvent_identities'])
    for row in r['subspace_diagnostics']:assert len(row['angles_degrees'])==row['top']
    for s in run.SNRS:
        for weighting in ('raw','target_weighted'):
            for k in ('R_exact',*exact.TERMS):
                data=[v for v in r['interaction_spectrum'] if (v['snr_db'],v['weighting'],v['operator'])==(s,weighting,k)]
                assert [v['mode'] for v in data]==list(range(1,65))
                e=np.array([v['singular_value']**2 for v in data])
                np.testing.assert_allclose([v['cumulative_capture'] for v in data],np.cumsum(e)/e.sum(),rtol=2e-14,atol=2e-15)
