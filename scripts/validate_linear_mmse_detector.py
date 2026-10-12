"""T11C-A: D1 mathematical calibration and one calibration-frame interface check."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy

from src.detectors import linear_mmse_detect, zf_detect
from src import link
from src.lmmse import estimate_mimo_from_pilot_ls
from src.lmmse_train import load_frozen_covariance, prepare_train_lmmse
from src.modulation import qpsk_modulate
from src.numerics import complex_gaussian_noise, noise_variance_from_snr
from src.pilots import build_fdm_pilot_layout
from src.provenance import (EXTENDED_MODULES, PROVENANCE_PLUMBING, STAGE_COMMITS, T11B_COMMIT, git_blob,
                            legacy_prefix_unchanged, recorded_source_matches)
from src.trajectories import prepare_trajectory
from scripts.validate_lmmse_train import check_config

RESULT_PATH = ROOT/"outputs/t11c_a_linear_mmse_detector_results.json"
FROZEN_PATH = ROOT/"outputs/t11c_a_frozen_artifact_hashes.json"
# All thresholds and seeds are fixed before any T11C-A sentinel execution.
ANALYTIC_ATOL = 5e-15
REFERENCE_ATOL = 5e-14
HIGH_SNR_RTOL = 1e-8
NEGATIVE_CONTROL_MIN_DIFFERENCE = 1e-3
MSE_RTOL = 0.02
MSE_OBSERVATIONS = 100000
MSE_BITS_SEED = 11261001
MSE_NOISE_SEED = 12261001
H_COMPLEX = np.array([[0.8+0.2j, -0.3+0.5j], [0.4-0.1j, 0.6+0.7j]])
Y_COMPLEX = np.array([0.7-0.3j, -0.2+0.9j])
IDENTITY_CASES = ((0.0, 0.5), (0.01, 0.5), (0.1, 0.5), (1.0, 0.5),
                  (0.2, 1.0), (0.3, 0.25), (3.0, 2.0))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def complex_record(array):
    array = np.asarray(array)
    return {"real": array.real.tolist(), "imag": array.imag.tolist()}


def freeze_accepted_outputs():
    """Capture accepted output bytes before extension, or verify an existing freeze.

    This mode never starts any channel generation or detector computation.
    A differing existing manifest is an error, never silently overwritten.
    """
    files = subprocess.check_output(["git","ls-tree","-r","--name-only",T11B_COMMIT,"outputs"],cwd=ROOT,text=True).splitlines()
    hashes = {p:digest(ROOT/p) for p in files}
    frozen = {"task":"T11C-A","accepted_t11b_commit":T11B_COMMIT,
        "purpose":"pre-extension byte hashes of all accepted tracked outputs","sha256":hashes}
    if FROZEN_PATH.exists():
        if json.loads(FROZEN_PATH.read_text(encoding="utf8"))!=frozen:
            raise ValueError("accepted output freeze changed; refusing to overwrite")
    else:
        # A first capture must see the accepted T11B output working tree.
        if subprocess.check_output(["git","diff",T11B_COMMIT,"--name-only","--","outputs"],cwd=ROOT,text=True).strip():
            raise ValueError("cannot first freeze modified historical outputs")
        FROZEN_PATH.write_text(json.dumps(frozen,indent=2)+"\n",encoding="utf8")
    print(f"Accepted tracked outputs frozen/verified: {len(hashes)} files; {FROZEN_PATH}")
    return frozen


def one_re(h, y, sigma2, ps=0.5):
    x, diag = linear_mmse_detect(np.asarray(y)[:, None, None],
        np.asarray(h)[:, :, None, None], sigma2, ps, return_diagnostics=True)
    return x[:, 0, 0], diag


def ridge_reference(h, y, regularization):
    """Independent augmented least squares; calibration only, never production."""
    h = np.asarray(h)
    a = np.concatenate((h, np.sqrt(regularization)*np.eye(h.shape[1])), axis=0)
    b = np.concatenate((y, np.zeros(h.shape[1], dtype=np.complex128)))
    return np.linalg.lstsq(a, b, rcond=None)[0]


def power_calibration():
    dibits = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
    constellation = qpsk_modulate(dibits.ravel())
    layout = build_fdm_pilot_layout(2, 64, 16, 16, 4)
    _, master = link.generate_master_qpsk(bits_seeds=(20261001, 20261002))
    x = link.assemble_pilot_data_frame(master, layout)
    measured_ps = [float(np.mean(np.abs(x[q][layout.data_mask[q]])**2)) for q in range(2)]
    constellation_ps = float(np.mean(np.abs(constellation/np.sqrt(2))**2))
    config = check_config()
    ps = config["system"]["total_power"]/config["system"]["nt"]
    err = max(abs(v-ps) for v in measured_ps+[constellation_ps])
    return {"Ptot": config["system"]["total_power"], "Nt": config["system"]["nt"],
        "stream_symbol_power": ps, "unscaled_constellation_energy": float(np.mean(np.abs(constellation)**2)),
        "scaled_constellation": complex_record(constellation/np.sqrt(2)),
        "measured_transmitter_data_power_by_stream": measured_ps,
        "measured_constellation_stream_power": constellation_ps,
        "max_power_error": err, "atol": ANALYTIC_ATOL, "passed": err <= ANALYTIC_ATOL}


def mathematical_calibration():
    identities = []
    for sigma, ps in IDENTITY_CASES:
        x, d = one_re(np.eye(2), Y_COMPLEX, sigma, ps)
        expected = ps/(ps+sigma)*Y_COMPLEX
        error = float(np.max(np.abs(x-expected)))
        identities.append({"sigma2": sigma, "Ps": ps, "max_absolute_error": error,
            "diagnostics": d, "passed": error <= ANALYTIC_ATOL})
    sigma, ps = 0.2, 0.5
    correct, d = one_re(H_COMPLEX, Y_COMPLEX, sigma, ps)
    reference = ridge_reference(H_COMPLEX, Y_COMPLEX, sigma/ps)
    wrong_scale = ridge_reference(H_COMPLEX, Y_COMPLEX, sigma)
    difference = float(np.linalg.norm(correct-wrong_scale))
    ref_error = float(np.max(np.abs(correct-reference)))
    factor2 = {"sigma2": sigma, "Ps": ps, "regularization": sigma/ps,
        "factor": 1/ps, "production": complex_record(correct), "reference": complex_record(reference),
        "wrong_sigma2_only": complex_record(wrong_scale), "wrong_formula_difference": difference,
        "negative_control_min_difference": NEGATIVE_CONTROL_MIN_DIFFERENCE,
        "passed": ref_error <= REFERENCE_ATOL and difference > NEGATIVE_CONTROL_MIN_DIFFERENCE}
    hs = np.array([0.7+0.4j, -1.1+0.3j])
    diagonal, dd = one_re(np.diag(hs), Y_COMPLEX, sigma, ps)
    diagonal_ref = hs.conj()*Y_COMPLEX/(np.abs(hs)**2+sigma/ps)
    diagonal_error = float(np.max(np.abs(diagonal-diagonal_ref)))
    hh = H_COMPLEX.conj().T
    wrong = {
        "transpose_instead_of_Hermitian": np.linalg.solve(H_COMPLEX.T@H_COMPLEX+sigma/ps*np.eye(2), H_COMPLEX.T@Y_COMPLEX),
        "wrong_gram_orientation": np.linalg.solve(H_COMPLEX@hh+sigma/ps*np.eye(2), hh@Y_COMPLEX),
        "missing_RHS_conjugation": np.linalg.solve(hh@H_COMPLEX+sigma/ps*np.eye(2), H_COMPLEX.T@Y_COMPLEX),
    }
    conjugation = {name: {"wrong_output": complex_record(value),
        "output_difference": float(np.linalg.norm(value-correct)),
        "detected": bool(np.linalg.norm(value-correct)>NEGATIVE_CONTROL_MIN_DIFFERENCE)} for name,value in wrong.items()}
    conjugation.update({"negative_control_min_difference": NEGATIVE_CONTROL_MIN_DIFFERENCE,
        "passed": all(v["detected"] for v in conjugation.values())})
    zf = zf_detect(Y_COMPLEX[:,None,None], H_COMPLEX[:,:,None,None])[:,0,0]
    high, hd = one_re(H_COMPLEX, Y_COMPLEX, 1e-10)
    high_relative = float(np.linalg.norm(high-zf)/np.linalg.norm(zf))
    zero, zd = one_re(H_COMPLEX, Y_COMPLEX, 0)
    rank1 = np.array([[1, 2], [2, 4]], dtype=np.complex128)
    try:
        zf_detect(Y_COMPLEX[:,None,None], rank1[:,:,None,None])
    except ValueError as exc:
        zf_failed, zf_message = True, str(exc)
    else:
        zf_failed, zf_message = False, None
    singular, sd = one_re(rank1, Y_COMPLEX, 0.1)
    near = np.array([[1, 1], [1, 1+1e-7]], dtype=np.complex128)
    gram = near.conj().T@near
    near_x, nd = one_re(near, Y_COMPLEX, 0.1)
    cn, cg, ca = (float(np.linalg.cond(a)) for a in (near, gram, gram+0.2*np.eye(2)))
    return {
        "identity_analytic": {"atol": ANALYTIC_ATOL, "cases": identities, "passed": all(r["passed"] for r in identities)},
        "factor_2_sentinel": factor2,
        "diagonal_analytic": {"max_absolute_error": diagonal_error, "atol": ANALYTIC_ATOL,
            "diagnostics": dd, "passed": diagonal_error <= ANALYTIC_ATOL},
        "complex_2x2_reference": {"H": complex_record(H_COMPLEX), "y": complex_record(Y_COMPLEX),
            "reference_method": "augmented ridge np.linalg.lstsq", "reference_in_production": False,
            "max_absolute_error": ref_error, "atol": REFERENCE_ATOL, "diagnostics": d,
            "passed": ref_error <= REFERENCE_ATOL},
        "conjugation_sentinel": conjugation,
        "high_snr_zf_limit": {"sigma2": 1e-10, "Ps": ps, "cond_H": float(np.linalg.cond(H_COMPLEX)),
            "relative_difference": high_relative, "fixed_relative_tolerance": HIGH_SNR_RTOL,
            "zero_noise_exact_ZF_match": bool(np.array_equal(zero,zf)), "diagnostics": hd,
            "zero_noise_diagnostics": zd, "passed": high_relative<=HIGH_SNR_RTOL and bool(np.array_equal(zero,zf))},
        "singular_channel": {"rank": int(np.linalg.matrix_rank(rank1)), "ZF_failed": zf_failed,
            "ZF_failure_message": zf_message, "D1_output": complex_record(singular), "sigma2": 0.1,
            "D1_finite": bool(np.all(np.isfinite(singular))), "diagnostics": sd,
            "passed": zf_failed and bool(np.all(np.isfinite(singular)))},
        "near_singular_diagnostic": {"rank": int(np.linalg.matrix_rank(near)), "cond_H": cn,
            "cond_HH_H": cg, "cond_regularized": ca, "sigma2": 0.1, "regularization": 0.2,
            "D1_finite": bool(np.all(np.isfinite(near_x))), "diagnostics": nd,
            "performance_ordering_claimed": False, "passed": ca<cg and bool(np.all(np.isfinite(near_x)))},
    }


def conditional_mse_calibration():
    sigma2, ps = 0.2, 0.5
    # Applying the production detector to the two observation basis vectors
    # obtains W without an inverse. The reference below uses augmented LS.
    w = np.column_stack([one_re(H_COMPLEX, np.eye(2)[:,i], sigma2, ps)[0] for i in range(2)])
    wr = np.column_stack([ridge_reference(H_COMPLEX, np.eye(2)[:,i], sigma2/ps) for i in range(2)])
    rx = ps*np.eye(2)
    residual = np.eye(2)-w@H_COMPLEX
    ce = residual@rx@residual.conj().T + sigma2*w@w.conj().T
    theory = float(np.trace(ce).real/2)
    bits = np.random.default_rng(MSE_BITS_SEED).integers(0,2,(2,MSE_OBSERVATIONS,2),dtype=np.uint8)
    x = np.stack([qpsk_modulate(stream.ravel()) for stream in bits])*np.sqrt(ps)
    noise = complex_gaussian_noise(x.shape, sigma2, np.random.default_rng(MSE_NOISE_SEED))
    y = H_COMPLEX@x+noise
    estimate, diag = linear_mmse_detect(y[...,None],
        np.broadcast_to(H_COMPLEX[:,:,None,None], (2,2,MSE_OBSERVATIONS,1)),
        sigma2, ps, return_diagnostics=True)
    empirical = float(np.mean(np.sum(np.abs(estimate[...,0]-x)**2,axis=0)/2))
    relative = abs(empirical-theory)/theory
    return {
        "conditional_mse_theory": {"sigma2": sigma2, "Ps": ps, "Rx": complex_record(rx),
            "W": complex_record(w), "Ce": complex_record(ce), "average_stream_MSE": theory,
            "W_vs_augmented_reference_max_error": float(np.max(np.abs(w-wr))),
            "passed": theory>0 and float(np.max(np.abs(w-wr)))<=REFERENCE_ATOL},
        "conditional_mse_empirical": {"observations": MSE_OBSERVATIONS,
            "bits_seed": MSE_BITS_SEED, "noise_seed": MSE_NOISE_SEED, "training_seed": None,
            "average_stream_MSE": empirical, "relative_error": relative,
            "fixed_relative_tolerance": MSE_RTOL, "diagnostics": diag, "passed": relative<=MSE_RTOL},
    }


def interface_calibration():
    # All 16 two-stream Gray-QPSK dibit combinations, controlled identity input.
    pairs = np.array([[a,b,c,d] for a,b in ((0,0),(0,1),(1,1),(1,0))
                      for c,d in ((0,0),(0,1),(1,1),(1,0))],dtype=np.uint8)
    bits = pairs.reshape(16,2,2).transpose(1,0,2)[:,:,None,:]
    unit = np.stack([qpsk_modulate(stream.ravel()).reshape(16,1) for stream in bits])
    x = unit/np.sqrt(2)
    identity = np.broadcast_to(np.eye(2)[:,:,None,None],(2,2,16,1))
    mask = np.ones(x.shape,dtype=bool)
    result, diag = link.detect_data_linear_mmse(x,identity,mask,0.2,0.5,return_diagnostics=True)
    detected = link.demodulate_data_bits(np.sqrt(2)*result,mask)
    reference = link.extract_reference_data_bits(bits,mask)
    bit_errors = sum(int(np.count_nonzero(a!=b)) for a,b in zip(detected,reference))
    expected = (0.5/0.7)*x
    h_est = H_COMPLEX + np.array([[0.1-0.05j, 0.03j], [-0.08+0.06j, -0.12j]])
    actual, _ = one_re(h_est,Y_COMPLEX,0.2)
    independent = ridge_reference(h_est,Y_COMPLEX,0.4)
    truth_output, _ = one_re(H_COMPLEX,Y_COMPLEX,0.2)
    estimate_difference = float(np.linalg.norm(actual-truth_output))
    reference_error = float(np.max(np.abs(actual-independent)))
    params = list(inspect.signature(linear_mmse_detect).parameters)
    return {
        "hard_slicer_integration": {"two_stream_constellation_pairs":16,"bit_errors":bit_errors,
            "detector_output_scale":"actual scaled x; caller restores sqrt(Nt) for existing slicer",
            "shrinkage_max_error":float(np.max(np.abs(result-expected))),
            "internal_amplitude_restoration_used":False,"diagnostics":diag,
            "passed":bit_errors==0 and bool(np.allclose(result,expected,rtol=0,atol=ANALYTIC_ATOL))},
        "estimated_h_interface": {"parameters":params,"supplied_h_est":complex_record(h_est),
            "reference_max_error":reference_error,"difference_from_H_true_output":estimate_difference,
            "truth_argument_present":False,"shared_api_for_R0_R2_R3_R4_R5":True,
            "passed":reference_error<=REFERENCE_ATOL and estimate_difference>NEGATIVE_CONTROL_MIN_DIFFERENCE
                and "h_est" in params and not any("true" in p or "perfect" in p for p in params)},
    }


def small_full_link_sentinel(observer=None):
    trajectory = prepare_trajectory().generate(0)
    h_true = trajectory.h_frames[0]
    bits_seeds = (20261001, 20261002)  # Existing T11A calibration frame-0 policy.
    noise_seed = 30261001  # Existing T09B calibration tid-0 namespace.
    bits, master = link.generate_master_qpsk(bits_seeds=bits_seeds)
    layout = build_fdm_pilot_layout(2,64,16,16,4)
    x = link.assemble_pilot_data_frame(master,layout)
    sigma2 = noise_variance_from_snr(10,1.0)
    y_clean = link.apply_frequency_mimo_channel(h_true,x)
    noise = complex_gaussian_noise(y_clean.shape,sigma2,np.random.default_rng(noise_seed))
    y = link.add_received_noise(y_clean,noise)
    h_ls,h_linear = link.estimate_ls_linear_channel(y,layout)
    manifest = ROOT/"outputs/t10c_train_validation_manifest.csv"
    covariance = ROOT/"outputs/t10c_lmmse_train_covariance.npz"
    frozen_before = {str(p.relative_to(ROOT)):digest(p) for p in (manifest,covariance)}
    frozen = load_frozen_covariance(covariance,digest(manifest))
    prepared = tuple(prepare_train_lmmse(frozen,layout.pilot_mask[q],sigma2) for q in range(2))
    h_train = estimate_mimo_from_pilot_ls(h_ls,prepared)
    ref_bits = link.extract_reference_data_bits(bits,layout.data_mask)
    before_shared = {"Y":array_digest(y),"bits":array_digest(bits)}
    records = []
    for name,h_est in (("R0_Perfect_CSI",h_true),("R3_LS_Linear",h_linear),("R4_LMMSE_Train",h_train)):
        h_digest = array_digest(h_est)
        args_seen = []
        for detector,call in (("D0_ZF",link.detect_data_zf),("D1_Linear_MMSE",link.detect_data_linear_mmse)):
            args_seen.append((y,bits,h_est))
            if observer is not None:
                observer(name,detector,y,bits,h_est)
            if detector=="D0_ZF":
                output = call(y,h_est,layout.data_mask)
                diag = {"solve_failures":0,"cholesky_successes":0,"cholesky_failures":0}
            else:
                output,diag = call(y,h_est,layout.data_mask,sigma2,0.5,return_diagnostics=True)
            bit_hat = link.demodulate_data_bits(np.sqrt(2)*output,layout.data_mask)
            errors = [int(np.count_nonzero(a!=b)) for a,b in zip(bit_hat,ref_bits)]
            valid = output.shape==(2,64,16) and bool(np.all(np.isfinite(output[layout.data_mask])))
            records.append({"estimator":name,"detector":detector,"Y_sha256":array_digest(y),
                "bits_sha256":array_digest(bits),"H_est_sha256":array_digest(h_est),
                "output_shape":list(output.shape),"data_finite":valid,
                "nondata_NaN":bool(np.all(np.isnan(output[~layout.data_mask]))),
                "diagnostic_bit_errors_by_stream":errors,"diagnostic_total_bit_errors":sum(errors),
                "data_bits":sum(b.size for b in ref_bits),"diagnostics":diag,
                "passed":valid and array_digest(h_est)==h_digest})
        if not all(a is b for a,b in zip(*args_seen)):
            raise AssertionError("detectors did not share the exact Y/bits/H_est objects")
    shared_unchanged = before_shared=={"Y":array_digest(y),"bits":array_digest(bits)}
    frozen_after = {str(p.relative_to(ROOT)):digest(p) for p in (manifest,covariance)}
    return {"namespace":"existing calibration only","trajectory_id":0,"frame_index":0,
        "channel_seed":trajectory.channel_seed,"bits_seeds":list(bits_seeds),"noise_seed":noise_seed,
        "training_seed":None,"SNR_dB":10,"dF":16,"sigma2":sigma2,"Ps":0.5,
        "independent_trajectories_used":1,"correlated_frames_used":1,
        "same_Y_bits_H_object_verified":True,"shared_inputs_unchanged":shared_unchanged,
        "covariance_reestimated":False,"frozen_before":frozen_before,"frozen_after":frozen_after,
        "estimator_preparations":2,"detector_calls":6,"performance_conclusion_claimed":False,
        "records":records,"passed":all(r["passed"] for r in records) and shared_unchanged and frozen_before==frozen_after}


def legacy_and_provenance_calibration():
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf8"))
    after = {p:digest(ROOT/p) for p in frozen["sha256"]}
    if after!=frozen["sha256"]:
        raise ValueError("accepted historical output bytes changed")
    prefixes = {p:legacy_prefix_unchanged(ROOT,p) for p in sorted(EXTENDED_MODULES)}
    prior = {}
    exec(compile(git_blob(str(ROOT),T11B_COMMIT,"src/detectors.py"),"historical_D0","exec"),prior)
    h = np.broadcast_to(H_COMPLEX[:,:,None,None],(2,2,3,2))
    y = np.broadcast_to(Y_COMPLEX[:,None,None],(2,3,2))
    old = prior["zf_detect"](y,h)
    current = zf_detect(y,h)
    equality = bool(np.array_equal(old,current))
    historical = []
    for stage,commit in STAGE_COMMITS.items():
        blobs = {p:hashlib.sha256(git_blob(str(ROOT),commit,p)).hexdigest() for p in sorted(EXTENDED_MODULES)}
        matching = []
        for name in frozen["sha256"]:
            if not name.endswith(".json") or not Path(name).name.startswith(stage+"_"):
                continue
            doc = json.loads((ROOT/name).read_text(encoding="utf8"))
            def find(value):
                if isinstance(value,dict):
                    for path,record in value.items():
                        if path in EXTENDED_MODULES and isinstance(record,str) and len(record)==64:
                            matching.append({"result":name,"path":path,"recorded_sha256":record,
                                "Git_blob_sha256":blobs[path],"passed":record==blobs[path]})
                        else:
                            find(record)
                elif isinstance(value,list):
                    for record in value:
                        find(record)
            find(doc)
        historical.append({"stage":stage,"commit":subprocess.check_output(["git","rev-parse",commit],cwd=ROOT,text=True).strip(),
            "detector_link_blob_sha256":blobs,"recorded_hash_checks":matching,
            "passed":all(r["passed"] for r in matching)})
    return {"accepted_T11B_commit":T11B_COMMIT,"legacy_prefix_byte_checks":prefixes,
        "fixed_D0_vs_Git_snapshot_exact_array_equal":equality,
        "frozen_artifacts": {"count":len(after),"before":frozen["sha256"],"after":after,"unchanged":True},
        "historical_detector_link_provenance":historical,
        "passed":all(prefixes.values()) and equality and all(r["passed"] for r in historical)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only",action="store_true",help="capture/verify historical output bytes without simulation")
    args = parser.parse_args()
    if args.freeze_only:
        return freeze_accepted_outputs()
    power = power_calibration()
    results = {**mathematical_calibration(),**conditional_mse_calibration(),**interface_calibration()}
    # Integration runs only after all small mathematical checks have passed.
    if not power["passed"] or not all(v["passed"] for v in results.values()):
        raise AssertionError("small D1 calibration failed; full-link sentinel not run")
    results["small_full_link_sentinel"] = small_full_link_sentinel()
    results["legacy_provenance_regression"] = legacy_and_provenance_calibration()
    report = {"task":"T11C-A", "experiment_type":"linear-MMSE MIMO detector mathematical, power-scaling, and interface calibration",
        "system_model":"y=H x+n; H[RX,TX,K,T], y[RX,K,T], x_hat[TX,K,T]",
        "noise_variance_definition":"complex noise variance per RX branch; E[n n^H]=sigma2 I",
        "Ptot":power["Ptot"],"Nt":power["Nt"],"stream_symbol_power":power["stream_symbol_power"],
        "regularization_definition":"noise_variance / stream_symbol_power",
        "current_regularization_factor":1/power["stream_symbol_power"],"power_scaling":power,
        **results,"explicit_inverse_used":False,"pseudo_inverse_used":False,"jitter_used":False,
        "main_detector":"D0 ZF","main_detector_changed":False,
        "formal_test_used":False,"formal_test_sealed":True,"OOD_generated":False,
        "validation_performance_scan_run":False,"warnings":[],
        "provenance":{"Python":platform.python_version(),"NumPy":np.__version__,"SciPy":scipy.__version__,
            "accepted_T11B_commit":T11B_COMMIT,"config_sha256":digest(ROOT/"configs/base.yaml"),
            "current_source_sha256":{p:digest(ROOT/p) for p in (
                "src/detectors.py","src/link.py","src/provenance.py",
                "scripts/validate_linear_mmse_detector.py","tests/test_linear_mmse_detector.py",
                "tests/test_detector_provenance.py","docs/detector_registry.md")},
            "current_provenance_plumbing_sha256":{p:digest(ROOT/p) for p in sorted(PROVENANCE_PLUMBING)}},
        "all_checks_passed":power["passed"] and all(v["passed"] for v in results.values())}
    RESULT_PATH.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps({"task":report["task"],"result_path":str(RESULT_PATH),
        "power_scaling":power,"factor_2_sentinel":results["factor_2_sentinel"],
        "conditional_mse_theory":report["conditional_mse_theory"]["average_stream_MSE"],
        "conditional_mse_empirical":report["conditional_mse_empirical"],
        "small_full_link_sentinel":report["small_full_link_sentinel"],
        "all_checks_passed":report["all_checks_passed"]},indent=2,allow_nan=False))
    if not report["all_checks_passed"]:
        raise AssertionError("T11C-A calibration failed")
    return report


if __name__=="__main__":
    main()
