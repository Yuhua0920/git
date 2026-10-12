"""Independent D1 math, complex scaling, failure semantics and shared receiver tests."""

import ast
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy import linalg

from scripts import validate_linear_mmse_detector as calibration
from src import detectors, link
from src.detectors import LinearMMSEDetectionError, linear_mmse_detect, zf_detect

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def math_results():
    return calibration.mathematical_calibration()


@pytest.mark.parametrize("sigma2,ps", calibration.IDENTITY_CASES)
def test_identity_analytic(sigma2,ps):
    y = np.array([0.3+0.7j,-0.8+0.2j])
    actual, diag = calibration.one_re(np.eye(2),y,sigma2,ps)
    np.testing.assert_allclose(actual,ps/(ps+sigma2)*y,rtol=0,atol=calibration.ANALYTIC_ATOL)
    assert diag["regularization"]==sigma2/ps
    assert diag["cholesky_successes"]==(1 if sigma2>0 else 0)
    assert diag["cholesky_failures"]==0


def test_existing_transmitter_and_constellation_prove_half_stream_power():
    result = calibration.power_calibration()
    assert result["passed"] and result["Ptot"]==1 and result["Nt"]==2
    assert abs(result["unscaled_constellation_energy"]-1)<=calibration.ANALYTIC_ATOL
    assert result["stream_symbol_power"]==0.5
    assert result["max_power_error"]<=calibration.ANALYTIC_ATOL


def test_current_factor_two_and_wrong_sigma_only_negative_control(math_results):
    r = math_results["factor_2_sentinel"]
    assert r["factor"]==2 and r["regularization"]==0.4 and r["passed"]
    assert r["wrong_formula_difference"]>calibration.NEGATIVE_CONTROL_MIN_DIFFERENCE


def test_diagonal_complex_analytic(math_results):
    assert math_results["diagonal_analytic"]["max_absolute_error"]<=calibration.ANALYTIC_ATOL


def test_general_complex_independent_augmented_lstsq(math_results):
    r = math_results["complex_2x2_reference"]
    assert r["reference_method"]=="augmented ridge np.linalg.lstsq"
    assert r["max_absolute_error"]<=calibration.REFERENCE_ATOL


@pytest.mark.parametrize("mistake",["transpose_instead_of_Hermitian","wrong_gram_orientation","missing_RHS_conjugation"])
def test_conjugation_orientation_negative_controls(math_results,mistake):
    assert math_results["conjugation_sentinel"][mistake]["detected"]


def test_high_snr_and_exact_zero_noise_ZF(math_results):
    r = math_results["high_snr_zf_limit"]
    assert r["sigma2"]==1e-10 and r["fixed_relative_tolerance"]==1e-8
    assert r["relative_difference"]<=1e-8 and r["zero_noise_exact_ZF_match"]


def test_rank_one_regularization_and_ZF_failure(math_results):
    r = math_results["singular_channel"]
    assert r["rank"]==1 and r["ZF_failed"] and "singular channel" in r["ZF_failure_message"]
    assert r["D1_finite"] and r["diagnostics"]["cholesky_successes"]==1


def test_zero_noise_singular_has_no_pseudoinverse():
    with pytest.raises(LinearMMSEDetectionError,match="singular channel") as exc:
        calibration.one_re(np.array([[1,2],[2,4]]),calibration.Y_COMPLEX,0)
    assert exc.value.diagnostics["solve_failures"]==1
    assert not exc.value.diagnostics["pseudo_inverse_used"]


def test_near_singular_condition_diagnostic(math_results):
    r = math_results["near_singular_diagnostic"]
    assert r["rank"]==2 and r["cond_H"]>1e7
    assert r["cond_regularized"]<r["cond_HH_H"] and r["D1_finite"]
    assert not r["performance_ordering_claimed"]


def test_conditional_100000_QPSK_AWGN_symbol_mse():
    r = calibration.conditional_mse_calibration()
    theory, empirical = r["conditional_mse_theory"],r["conditional_mse_empirical"]
    assert theory["passed"] and empirical["passed"]
    assert empirical["observations"]==100000 and empirical["fixed_relative_tolerance"]==0.02
    assert empirical["relative_error"]<=0.02
    assert empirical["diagnostics"]["cholesky_successes"]==100000
    assert empirical["diagnostics"]["cholesky_failures"]==0


def test_scaled_slicer_and_perturbed_estimated_H():
    r = calibration.interface_calibration()
    assert r["hard_slicer_integration"]["passed"] and r["hard_slicer_integration"]["bit_errors"]==0
    assert r["hard_slicer_integration"]["shrinkage_max_error"]<=calibration.ANALYTIC_ATOL
    assert r["estimated_h_interface"]["passed"] and not r["estimated_h_interface"]["truth_argument_present"]
    assert "h_est" in inspect.signature(linear_mmse_detect).parameters


@pytest.mark.parametrize("sigma2",[-1,np.nan,np.inf,-np.inf,True,1j,"0.1"])
def test_invalid_noise_variance(sigma2):
    with pytest.raises(ValueError,match="noise_variance"):
        calibration.one_re(np.eye(2),calibration.Y_COMPLEX,sigma2)


@pytest.mark.parametrize("ps",[0,-0.5,np.nan,np.inf,-np.inf,False,1j,"0.5"])
def test_invalid_stream_power(ps):
    with pytest.raises(ValueError,match="stream_symbol_power"):
        calibration.one_re(np.eye(2),calibration.Y_COMPLEX,0.1,ps)


@pytest.mark.parametrize("target",["y","H"])
@pytest.mark.parametrize("value",[np.nan,np.inf,-np.inf])
def test_nonfinite_inputs(target,value):
    y = calibration.Y_COMPLEX.copy()
    h = calibration.H_COMPLEX.copy()
    if target=="y":
        y[0]=value
    else:
        h[0,0]=value
    with pytest.raises(ValueError,match="finite"):
        calibration.one_re(h,y,0.1)


@pytest.mark.parametrize("yshape,hshape",[( (2,2),(2,2,2,1)),((2,2,1),(2,2,2)),
    ((2,2,1),(2,2,3,1)),((2,2,1),(2,3,2,1)),((2,0,1),(2,2,0,1))])
def test_invalid_grid_shapes(yshape,hshape):
    with pytest.raises(ValueError):
        linear_mmse_detect(np.ones(yshape),np.ones(hshape),0.1,0.5)


def test_cholesky_failure_counts_without_fallback_or_jitter():
    y = np.tile(calibration.Y_COMPLEX[:,None,None],(1,2,1))
    h = np.tile(calibration.H_COMPLEX[:,:,None,None],(1,1,2,1))
    original = linalg.cho_factor
    calls = []
    def controlled_failure(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:
            raise np.linalg.LinAlgError("injected factorization failure")
        return original(*args,**kwargs)
    with patch.object(linalg,"cho_factor",side_effect=controlled_failure):
        with pytest.raises(LinearMMSEDetectionError,match="Cholesky failed") as exc:
            linear_mmse_detect(y,h,0.1,0.5)
    assert exc.value.diagnostics["cholesky_successes"]==1
    assert exc.value.diagnostics["cholesky_failures"]==1
    assert exc.value.diagnostics["failed_RE"]=={"k":1,"t":0}
    assert not exc.value.diagnostics["jitter_used"]


def test_production_uses_no_inverse_pseudoinverse_or_lstsq():
    with patch("numpy.linalg.inv",side_effect=AssertionError("inverse forbidden")), \
         patch("numpy.linalg.pinv",side_effect=AssertionError("pseudoinverse forbidden")), \
         patch("numpy.linalg.lstsq",side_effect=AssertionError("reference cannot enter production")):
        assert np.all(np.isfinite(calibration.one_re(calibration.H_COMPLEX,calibration.Y_COMPLEX,0.2)[0]))
    tree = ast.parse(inspect.getsource(linear_mmse_detect))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
    assert not attrs.intersection({"inv","pinv","lstsq"})


def test_wrapper_keeps_nondata_NaNs_and_uses_only_data_REs():
    y = np.tile(calibration.Y_COMPLEX[:,None,None],(1,3,2))
    h = np.tile(calibration.H_COMPLEX[:,:,None,None],(1,1,3,2))
    mask = np.ones((2,3,2),dtype=bool); mask[:,0,0]=False
    y[:,0,0]=np.nan; h[:,:,0,0]=np.nan
    out,d = link.detect_data_linear_mmse(y,h,mask,0.1,0.5,return_diagnostics=True)
    assert out.shape==(2,3,2) and np.all(np.isfinite(out[mask])) and np.all(np.isnan(out[~mask]))
    assert d["cholesky_successes"]==5
    expected = calibration.ridge_reference(calibration.H_COMPLEX,calibration.Y_COMPLEX,0.2)
    np.testing.assert_allclose(out[:,mask[0]],np.tile(expected[:,None],(1,5)),rtol=0,atol=calibration.REFERENCE_ATOL)


@pytest.mark.parametrize("mask_kind",["unequal_streams","empty","nonboolean","wrong_shape"])
def test_wrapper_rejects_invalid_data_mask(mask_kind):
    y = np.ones((2,3,2)); h = np.ones((2,2,3,2))
    mask = np.ones((2,3,2),dtype=bool)
    if mask_kind=="unequal_streams": mask[1,0,0]=False
    if mask_kind=="empty": mask[:]=False
    if mask_kind=="nonboolean": mask=mask.astype(int)
    if mask_kind=="wrong_shape": mask=mask[:,:2]
    with pytest.raises(ValueError):
        link.detect_data_linear_mmse(y,h,mask,0.1,0.5)


def test_one_full_link_R0_R3_R4_same_Y_bits_and_H_est():
    captures = []
    def observe(name,detector,y,bits,h_est):
        captures.append((name,detector,y,bits,h_est))
    calls = []
    old0,old1 = link.detect_data_zf,link.detect_data_linear_mmse
    def d0(y,h,mask):
        calls.append(("D0",y,h)); return old0(y,h,mask)
    def d1(y,h,mask,*args,**kwargs):
        calls.append(("D1",y,h)); return old1(y,h,mask,*args,**kwargs)
    with patch.object(link,"detect_data_zf",side_effect=d0),patch.object(link,"detect_data_linear_mmse",side_effect=d1):
        result = calibration.small_full_link_sentinel(observe)
    assert result["passed"] and result["trajectory_id"]==result["frame_index"]==0
    assert result["SNR_dB"]==10 and result["dF"]==16 and result["detector_calls"]==6
    for i in range(0,6,2):
        assert all(a is b for a,b in zip(captures[i][2:],captures[i+1][2:]))
        assert calls[i][1] is calls[i+1][1] and calls[i][2] is calls[i+1][2]
    assert len({id(c[2]) for c in captures})==1 and len({id(c[3]) for c in captures})==1
    assert result["frozen_before"]==result["frozen_after"]


def test_saved_t11c_a_report_and_current_sources():
    r = json.loads(calibration.RESULT_PATH.read_text(encoding="utf8"))
    assert r["task"]=="T11C-A" and r["all_checks_passed"]
    assert r["main_detector"]=="D0 ZF" and not r["main_detector_changed"]
    assert r["formal_test_sealed"] and not r["formal_test_used"]
    assert not any(r[k] for k in ["explicit_inverse_used","pseudo_inverse_used","jitter_used","validation_performance_scan_run","OOD_generated"])
    for path,digest in r["provenance"]["current_source_sha256"].items():
        assert calibration.digest(ROOT/path)==digest
    for path,digest in r["provenance"]["current_provenance_plumbing_sha256"].items():
        assert calibration.digest(ROOT/path)==digest
    for key in ("identity_analytic","factor_2_sentinel","diagonal_analytic","complex_2x2_reference",
                "conjugation_sentinel","high_snr_zf_limit","singular_channel","near_singular_diagnostic",
                "conditional_mse_theory","conditional_mse_empirical","hard_slicer_integration",
                "estimated_h_interface","small_full_link_sentinel","legacy_provenance_regression"):
        assert r[key]["passed"]
