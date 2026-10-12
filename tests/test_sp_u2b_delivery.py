"""Post-run result integrity and serializer/scope regression checks."""
import json
import numpy as np
import pytest
from scripts import sp_u2b_diagnostic as d
d.install_guard()
from scripts.analyze_sp_u2b_recovery import normalize,verify_or_create_csv
from scripts.audit_sp_u2b_recovery import long_rows,GEOMETRY_KEYS
from scripts.analyze_sp_u2b import read_rows


def test_numpy_aggregate_JSON_boundary():
    raw=dict(count=sum(np.array([1.,2.,3.])>1),scalar=np.float64(.3),nested=[np.bool_(True),np.int64(4)])
    restored=json.loads(json.dumps(normalize(raw),allow_nan=False))
    assert restored==dict(count=2,scalar=.3,nested=[True,4])


def test_existing_CSV_exact_recovery_preserves_bytes(tmp_path):
    path=tmp_path/'sp_u2b_fixture.csv'
    rows=[dict(x=1.,y=-2.),dict(x=3.,y=.4)]
    d.csvsave(path,rows)
    before=path.read_bytes()
    verify_or_create_csv(path,rows)
    assert path.read_bytes()==before
    with pytest.raises(RuntimeError,match='changed existing CSV'):
        verify_or_create_csv(path,[dict(x=9.,y=-2.)])
    assert path.read_bytes()==before


def test_actual_geometry_schema_excludes_residual():
    row=read_rows(d.OUT/'sp_u2b_development_windows.csv')[0]
    assert 'clean_residual_norm' in row
    long=long_rows([row])
    assert len(long)==2 and long[0]['clean_flag'] and not long[1]['clean_flag']
    assert set(long[0])==set(long[1])==GEOMETRY_KEYS|{'trajectory_id','frame_id','snr_db','clean_flag'}
    assert 'residual_norm' not in long[0]


def test_actual_result_counts_and_identities():
    rows=read_rows(d.OUT/'sp_u2b_development_windows.csv')
    assert len(rows)==1200 and len({r['trajectory_id'] for r in rows})==100
    assert len({(r['trajectory_id'],r['frame_id']) for r in rows})==600
    for r in rows:
        assert r['clean_rank_real']==r['noisy_rank_real']==2
        d.close(r['delta_R_total'],r['delta_R_noise']+r['delta_R_surrogate'])
        d.close(r['delta_R_noise'],r['coord_output_mse']+r['noise_penalty_cross_term'])
        d.close(r['coord_output_energy'],r['coord_output_quadratic'])
        assert not r['bound_failure'] and not r['near_zero_coordinate_change']
    # Negative findings must be retained in the delivery population.
    assert sum(r['delta_R_noise']<0 for r in rows)==815


def test_original_frozen_sources_and_recovery_CSV_integrity():
    from scripts.run_sp_u2b import verify
    verify()
    recovery=d.load(d.OUT/'sp_u2b_analysis_recovery.json')
    assert len(recovery['existing_CSVs_verified_exact'])==6
    for record in recovery['existing_CSVs_verified_exact']:
        assert d.sha(d.ROOT/record['path'])==record['sha256']


def test_guard_still_blocks_T20B_and_allows_owned_QA_temp(tmp_path):
    with pytest.raises(PermissionError):
        (d.OUT/'t20b_formal_results.json').read_bytes()
    p=tmp_path/'.figure_qa_atomic.tmp'
    p.write_text('owned',encoding='utf-8')
    assert p.read_text(encoding='utf-8')=='owned'
