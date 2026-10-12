"""Independent hand oracles for R2; existing R3 tests remain unchanged."""

import hashlib
import inspect
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.estimators import ls_linear_interpolate, pilot_ls_estimate
from src.interpolation import (
    ls_nearest_interpolate, nearest_interpolate_frequency, nearest_interpolate_time,
)
from src.link import PILOT_SYMBOLS, apply_frequency_mimo_channel
from src.pilots import build_fdm_pilot_layout, map_pilot_symbols


TIME_EXPECTED = np.array([0,0,0,4,4,4,4,8,8,8,8,12,12,12,12,12])
FREQUENCY_EXPECTED = np.array([0,0,0,4,4,4,4,8,8,8,8,8])
OFFSET_FREQUENCY_EXPECTED = np.array([2,2,2,2,2,6,6,6,6,10,10,10])


def sparse(truth, mask):
    observed = np.full(truth.shape, np.nan + 1j*np.nan, dtype=np.complex128)
    selected = np.broadcast_to(mask[None], truth.shape)
    observed[selected] = truth[selected]
    return observed


def frequency_case(offset=False):
    mask = np.zeros((1,12,1), dtype=bool)
    anchors = [2,6,10] if offset else [0,4,8]
    mask[0,anchors,0] = True
    observed = np.full((1,1,12,1), np.nan + 1j*np.nan, dtype=np.complex128)
    observed[0,0,anchors,0] = [10,20,30]
    return observed, mask


def time_case(values=(1,2,3,4)):
    observed = np.full((1,1,1,16), np.nan + 1j*np.nan, dtype=np.complex128)
    observed[0,0,0,[0,4,8,12]] = values
    return observed


def test_frequency_explicit_hand_step_with_lower_ties_and_right_edge():
    observed, mask = frequency_case()
    actual, coverage = nearest_interpolate_frequency(observed, mask)
    # Independent literals: k2 -> 0, k6 -> 4, k9..11 -> 8.
    expected = np.array([10,10,10,20,20,20,20,30,30,30,30,30])
    np.testing.assert_array_equal(actual[0,0,:,0], expected)
    assert coverage.all()


def test_frequency_explicit_left_edge_with_shifted_anchors():
    observed, mask = frequency_case(offset=True)
    actual, _ = nearest_interpolate_frequency(observed, mask)
    np.testing.assert_array_equal(actual[0,0,:,0], [10,10,10,10,10,20,20,20,20,30,30,30])


@pytest.mark.parametrize("k,expected", [(2,10),(6,20),(9,30),(11,30)])
def test_frequency_tie_and_edge_coordinates(k, expected):
    observed, mask = frequency_case()
    assert nearest_interpolate_frequency(observed, mask)[0][0,0,k,0] == expected


def test_complex_frequency_copies_one_complete_anchor():
    observed, mask = frequency_case()
    observed[0,0,[0,4,8],0] = [1+2j,-3+0.5j,7-4j]
    actual, _ = nearest_interpolate_frequency(observed, mask)
    expected = [1+2j,1+2j,1+2j,-3+0.5j,-3+0.5j,-3+0.5j,-3+0.5j,7-4j,7-4j,7-4j,7-4j,7-4j]
    np.testing.assert_array_equal(actual[0,0,:,0], expected)


def test_time_explicit_hand_step():
    actual = nearest_interpolate_time(time_case(), [0,4,8,12])
    np.testing.assert_array_equal(actual[0,0,0], [1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,4])


@pytest.mark.parametrize("t,expected", [(2,1),(6,2),(10,3),(13,4),(14,4),(15,4)])
def test_time_ties_and_trailing_edge(t, expected):
    assert nearest_interpolate_time(time_case(), [0,4,8,12])[0,0,0,t] == expected


def test_time_constant_and_complex_anchor_preservation():
    values = np.array([1+2j,-3+0.5j,7-4j,-0.25+6j])
    actual = nearest_interpolate_time(time_case(values), [0,4,8,12])
    np.testing.assert_array_equal(actual[0,0,0], values[[0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,3]])
    constant = nearest_interpolate_time(time_case([1+2j]*4), [0,4,8,12])
    np.testing.assert_array_equal(constant, np.full((1,1,1,16),1+2j))


@pytest.mark.parametrize("df", [2,4,8,16])
def test_actual_noiseless_LS_constant_full_shape_dtype_finite_and_preservation(df):
    layout = build_fdm_pilot_layout(2,64,16,df,4)
    constant = np.array([[0.8+0.2j,-0.3+0.5j],[0.4-0.1j,0.6+0.7j]])
    truth = np.broadcast_to(constant[:,:,None,None],(2,2,64,16))
    x = map_pilot_symbols(layout,PILOT_SYMBOLS)
    observed, mask = pilot_ls_estimate(apply_frequency_mimo_channel(truth,x),x,layout.pilot_mask)
    before = observed.copy()
    frequency, coverage = nearest_interpolate_frequency(observed,mask)
    actual = ls_nearest_interpolate(observed,mask)
    assert actual.shape == (2,2,64,16) and actual.dtype == np.complex128
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual,truth,rtol=0,atol=1e-14)
    np.testing.assert_allclose(frequency[...,list(layout.pilot_symbol_indices)],truth[...,list(layout.pilot_symbol_indices)],rtol=0,atol=1e-14)
    selected = np.broadcast_to(mask[None],observed.shape)
    np.testing.assert_array_equal(actual[selected],observed[selected])
    np.testing.assert_array_equal(observed,before)
    nonpilot_times = [1,2,3,5,6,7,9,10,11,13,14,15]
    assert np.isnan(frequency[...,nonpilot_times].real).all()
    assert np.isnan(frequency[...,nonpilot_times].imag).all()
    assert not coverage[...,nonpilot_times].any()


def test_physical_frequency_grid_does_not_wrap_63_to_0():
    layout = build_fdm_pilot_layout(2,64,16,4,4)
    truth = np.broadcast_to(np.arange(64)[None,None,:,None],(2,2,64,16)).astype(np.complex128)
    actual = ls_nearest_interpolate(sparse(truth,layout.pilot_mask),layout.pilot_mask)
    assert actual[0,0,63,0] == 60  # Circular nearest would choose k0.
    assert actual[0,1,0,0] == 2


@pytest.mark.parametrize("changed_tx", [0,1])
def test_TX_separation_bidirectional_contamination_negative(changed_tx):
    layout = build_fdm_pilot_layout(2,64,16,4,4)
    constants = np.array([[1+2j,-30+50j],[4-3j,80-90j]])
    truth = np.broadcast_to(constants[:,:,None,None],(2,2,64,16))
    observations = sparse(truth,layout.pilot_mask)
    original = ls_nearest_interpolate(observations,layout.pilot_mask)
    contaminated = observations.copy()
    contaminated[:,changed_tx,layout.pilot_mask[changed_tx]] += 1000+2000j
    changed = ls_nearest_interpolate(contaminated,layout.pilot_mask)
    np.testing.assert_array_equal(changed[:,1-changed_tx],original[:,1-changed_tx])
    assert not np.array_equal(changed[:,changed_tx],original[:,changed_tx])
    # A deliberately mixed-TX value at TX0 k2 differs from its correct k0 anchor.
    wrong_cross_tx_value = observations[:,1,2,0]
    assert not np.array_equal(original[:,0,2,0],wrong_cross_tx_value)


def test_affine_R3_interior_exact_R2_explicit_piecewise_constant_and_distinct():
    a = np.array([[1+2j,-1+0.25j],[0.5-1j,3-2j]])
    b = np.array([[0.125-0.0625j,-0.25+0.125j],[0.375+0.25j,0.0625-0.125j]])
    c = np.array([[0.25+0.125j,0.125-0.25j],[-0.0625+0.375j,-0.125+0.0625j]])
    truth = a[:,:,None,None]+b[:,:,None,None]*np.arange(12)[None,None,:,None]+c[:,:,None,None]*np.arange(16)[None,None,None,:]
    mask = np.zeros((2,12,16),dtype=bool)
    for t in (0,4,8,12):
        mask[0,[0,4,8],t] = True
        mask[1,[2,6,10],t] = True
    observed = sparse(truth,mask)
    r2 = ls_nearest_interpolate(observed,mask)
    r3 = ls_linear_interpolate(observed,mask)
    for q,ka in enumerate((FREQUENCY_EXPECTED,OFFSET_FREQUENCY_EXPECTED)):
        expected = a[:,q,None,None]+b[:,q,None,None]*ka[None,:,None]+c[:,q,None,None]*TIME_EXPECTED[None,None,:]
        np.testing.assert_array_equal(r2[:,q],expected)
        interior = np.arange(0,9) if q==0 else np.arange(2,11)
        np.testing.assert_allclose(r3[:,q,interior,:13],truth[:,q,interior,:13],rtol=0,atol=1e-14)
    np.testing.assert_array_equal(r2[:,0,:3,:3],np.broadcast_to(a[:,0,None,None],(2,3,3)))
    assert np.max(np.abs(r2-r3)) > 0.1


@pytest.mark.parametrize("function,parameters", [
    (nearest_interpolate_frequency,{"h_ls","pilot_mask"}),
    (nearest_interpolate_time,{"h_frequency","pilot_symbol_indices"}),
    (ls_nearest_interpolate,{"h_ls","pilot_mask"}),
])
def test_estimators_have_no_truth_or_data_input(function,parameters):
    assert set(inspect.signature(function).parameters)==parameters


@pytest.mark.parametrize("case", ["shape","real","mask_shape","mask_type","overlap","empty","missing_tx","pilot_nan","inf","nonpilot_finite"])
def test_frequency_rejects_malformed_observations(case):
    layout = build_fdm_pilot_layout(2,64,16,4,4)
    observed = sparse(np.ones((2,2,64,16),dtype=np.complex128),layout.pilot_mask)
    mask = layout.pilot_mask.copy()
    if case=="shape": observed=observed[0]
    elif case=="real": observed=observed.real
    elif case=="mask_shape": mask=mask[:,:63]
    elif case=="mask_type": mask=mask.astype(np.int8)
    elif case=="overlap": mask[1,0,0]=True
    elif case=="empty": mask[:]=False; observed[:]=np.nan+1j*np.nan
    elif case=="missing_tx": mask[1,:,4]=False; observed[:,1,:,4]=np.nan+1j*np.nan
    elif case=="pilot_nan": observed[0,0,0,0]=np.nan+1j*np.nan
    elif case=="inf": observed[0,0,0,0]=np.inf+1j
    elif case=="nonpilot_finite": observed[0,0,1,1]=3+2j
    with pytest.raises(ValueError): nearest_interpolate_frequency(observed,mask)


@pytest.mark.parametrize("anchors", [[],[0,4,4,12],[0,8,4,12],[-1,4,8,12],[0,4,8,16],[False,4,8,12],[0.,4.,8.,12.],[[0,4],[8,12]],None,[0,4,8,2**100]])
def test_time_rejects_malformed_anchor_indices(anchors):
    with pytest.raises(ValueError): nearest_interpolate_time(time_case(),anchors)


@pytest.mark.parametrize("case", ["shape","real","pilot_nan","inf","nonpilot_finite","only_real_nan"])
def test_time_rejects_malformed_grid(case):
    observed = time_case()
    if case=="shape": observed=observed[0]
    elif case=="real": observed=observed.real
    elif case=="pilot_nan": observed[...,4]=np.nan+1j*np.nan
    elif case=="inf": observed[...,4]=np.inf+0j
    elif case=="nonpilot_finite": observed[...,1]=3+2j
    elif case=="only_real_nan": observed[...,1]=complex(np.nan,0)
    with pytest.raises(ValueError): nearest_interpolate_time(observed,[0,4,8,12])


def test_complex64_observations_promote_to_complex128():
    observed, mask = frequency_case()
    actual = ls_nearest_interpolate(observed.astype(np.complex64),mask)
    assert actual.dtype == np.complex128


def test_single_frequency_and_single_time_anchor_hold_everywhere():
    observed = np.full((1,1,7,5),np.nan+1j*np.nan,dtype=np.complex128)
    mask = np.zeros((1,7,5),dtype=bool)
    mask[0,3,2] = True
    observed[0,0,3,2] = -3+0.5j
    np.testing.assert_array_equal(ls_nearest_interpolate(observed,mask),np.full(observed.shape,-3+0.5j))


def test_script_mathematical_reports_are_actual_and_passed():
    from scripts.validate_nearest_interpolation import mathematical_sentinels
    report = mathematical_sentinels()
    assert all(report["mathematical_checks"].values())
    assert report["frequency_manual_sentinel"]["expected_anchor_indices"]==[0,0,0,4,4,4,4,8,8,8,8,8]
    assert report["time_manual_sentinel"]["expected_anchor_indices"]==[0,0,0,4,4,4,4,8,8,8,8,12,12,12,12,12]
    assert report["affine_R2_vs_R3_sentinel"]["R2_hand_oracle_max_absolute_error"]==0


@pytest.fixture(scope="module")
def small_integration():
    from scripts import validate_nearest_interpolation as calibration
    from src.link import detect_data_zf
    with patch("scripts.validate_nearest_interpolation.ls_nearest_interpolate",wraps=ls_nearest_interpolate) as r2, \
            patch("scripts.validate_nearest_interpolation.ls_linear_interpolate",wraps=ls_linear_interpolate) as r3, \
            patch("src.tdl_noiseless_scan.detect_data_zf",wraps=detect_data_zf) as zf, \
            patch("src.lmmse_train.formal_trajectory",side_effect=AssertionError("calibration domain only")), \
            patch("src.lmmse_train.estimate_training_covariance",side_effect=AssertionError("no covariance estimation")), \
            patch("src.numerics.complex_gaussian_noise",side_effect=AssertionError("noise OFF")), \
            patch("src.doppler.build_tdl_a_doppler_channel",side_effect=AssertionError("use continuous trajectory")), \
            patch("numpy.linalg.inv",side_effect=AssertionError("no explicit inverse")), \
            patch("numpy.linalg.pinv",side_effect=AssertionError("no pinv")):
        report = calibration.small_tdl_integration((0,))
    return report,r2.call_args_list,r3.call_args_list,zf.call_args_list


def test_small_calibration_noiseless_ZF_no_filter_same_observations_and_Y(small_integration):
    report,r2_calls,r3_calls,zf_calls = small_integration
    assert report["all_checks_passed"] and report["trajectory_ids"]==[0]
    assert report["channel_seeds"]==[40261001]
    assert report["num_correlated_calibration_frames"]==10 and len(report["frame_seed_records"])==10
    assert report["num_filtered_trajectories"]==0 and report["numerical_failures"]==[]
    assert len(r2_calls)==len(r3_calls)==40 and len(zf_calls)==120
    for r2,r3 in zip(r2_calls,r3_calls):
        assert r2.args[0] is r3.args[0] and r2.args[1] is r3.args[1]
    for first in range(0,len(zf_calls),3):
        assert len({id(c.args[0]) for c in zf_calls[first:first+3]})==1
    for row in report["per_dF"]:
        assert row["perfect_bit_errors"]==0 and row["frames_processed"]==10
        assert row["r2_ZF_successes"]==row["r3_ZF_successes"]==row["perfect_ZF_successes"]==10


@pytest.mark.parametrize("ids", [(),(2000,),(10,),(0,0),(True,)])
def test_integration_rejects_formal_or_invalid_IDs_before_generation(ids):
    from scripts.validate_nearest_interpolation import small_tdl_integration
    with patch("scripts.validate_nearest_interpolation.prepare_trajectory") as prepare:
        with pytest.raises(ValueError,match="calibration trajectory IDs"):
            small_tdl_integration(ids)
        prepare.assert_not_called()


@pytest.fixture(scope="module")
def saved():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root/"outputs/t11a_nearest_interpolation_results.json").read_text(encoding="utf-8"))


def test_saved_task_semantics_and_frozen_R3_sources(saved):
    from scripts.validate_nearest_interpolation import T10D_COMMIT,accepted_provenance,sha256_file
    root = Path(__file__).resolve().parents[1]
    checks,frozen = accepted_provenance()
    assert all(checks.values()) and saved["all_checks_passed"] and all(saved["checks"].values())
    assert saved["provenance"]["T10D_commit"]==T10D_COMMIT
    assert saved["provenance"]["T10D_committed_cleanly_at_task_start"] is True
    assert saved["frozen_artifact_hash_before_after"]["before"]==saved["frozen_artifact_hash_before_after"]["after"]==frozen
    assert sha256_file(root/"outputs/t11a_nearest_interpolation_results.json")=="060cf65ce7cbdbb88857d9db1b881b048331eafac1f1f3e1fe29f72fcf89a7e6"
    for path,digest in saved["provenance"]["source_sha256"].items():
        if path in {"docs/baseline_registry.md","tests/test_nearest_interpolation.py"}:
            # T11B-FIX: researcher-authorized historical artifacts. Hash the
            # exact Git blob bytes; numerical sources still check current files.
            data = subprocess.run(["git","show",
                f"9fe2f165f9f511c9fe1c90585f2ec5023ef3c951:{path}"],
                cwd=root,check=True,capture_output=True).stdout
            assert hashlib.sha256(data).hexdigest()==digest
        else:
            from src.provenance import recorded_source_matches
            assert recorded_source_matches(root,path,digest,"t11a",current_hash=sha256_file)
    assert saved["r1_semantics"]["data_BER"]=="N/A"
    assert saved["r1_semantics"]["data_RE_channel_estimate"] is False
    assert saved["frequency_wraparound"] is False and saved["causal_online_tracking_claimed"] is False
    for field in ("formal_validation_scan_performed","formal_test_used","OOD_generated","performance_figures_generated","performance_ordering_required"):
        assert saved[field] is False
    assert saved["formal_test_sealed"] is True
    historical_registry = subprocess.run(["git","show",
        f"9fe2f165f9f511c9fe1c90585f2ec5023ef3c951:{saved['baseline_registry']['path']}"],
        cwd=root,check=True,capture_output=True).stdout
    assert hashlib.sha256(historical_registry).hexdigest()==saved["baseline_registry"]["sha256"]


@pytest.mark.parametrize("df,bits", [(2,307200),(4,358400),(8,384000),(16,396800)])
def test_saved_ten_trajectory_integration_counts_and_finite_diagnostics(saved,df,bits):
    report = saved["small_tdl_integration"]
    assert report["trajectory_ids"]==list(range(10))
    assert report["channel_seeds"]==list(range(40261001,40261011))
    assert report["num_independent_calibration_trajectories"]==10 and report["num_correlated_calibration_frames"]==100
    assert report["noise"]=="OFF" and report["formal_manifest_used"] is False
    assert report["num_filtered_trajectories"]==0 and report["numerical_failures"]==[]
    assert len({(r["trajectory_id"],r["frame_index"]) for r in report["frame_seed_records"]})==100
    assert len({s for r in report["frame_seed_records"] for s in r["bits_seeds"]})==200
    row = next(r for r in report["per_dF"] if r["dF"]==df)
    assert row["frames_processed"]==100 and row["total_data_bits"]==bits and row["perfect_bit_errors"]==0
    assert row["pilot_preservation_exact"] and row["full_grid_finite"]
    assert row["pilot_LS_max_absolute_error"]<=1e-12
    for method in ("r2","r3"):
        assert row[f"{method}_ZF_successes"]==100
        assert np.isfinite(row[f"{method}_data_nmse_linear"]) and np.isfinite(row[f"{method}_data_nmse_db"])
        assert row[f"{method}_data_nmse_linear"]==row[f"{method}_error_energy"]/row["nmse_truth_energy"]
