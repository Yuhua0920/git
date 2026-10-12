"""T10C: training-only covariance, frozen estimator, and held-out ID audit."""

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.linalg import cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, validate_config
from src.estimators import pilot_ls_estimate
from src.link import PILOT_SYMBOLS, apply_frequency_mimo_channel
from src.lmmse import (build_frequency_covariance, build_time_covariance,
    build_full_grid_covariance, prepare_oracle_lmmse, estimate_mimo_from_pilot_ls)
from src.lmmse_train import (BATCH_SIZE, MEAN_POLICY, FLATTEN_ORDER, ensure_manifest,
    sha256_file, raw_array_sha256, estimate_training_covariance, formal_trajectory,
    covariance_statistics, save_frozen_covariance, prepare_train_lmmse)
from src.numerics import complex_gaussian_noise, noise_variance_from_snr
from src.pilots import build_fdm_pilot_layout, count_layout_resources, map_pilot_symbols
from src.trajectories import prepare_trajectory
from src.provenance import recorded_source_matches

DF_OPTIONS = (2, 4, 8, 16)
SNR_OPTIONS = (0, 5, 10, 15, 20)
CONDITIONS = ((2, 0), (4, 10), (8, 15), (16, 20))
NOISE_SEED_BASE = 70261001
NMSE_TOLERANCE_DB = 0.5  # Fixed before either train or held-out execution.
PRE_COMMIT = "c0a1ac3525cde5d90a9fe26b75b00170d2416caf"


def check_config():
    config = load_config(ROOT / "configs/base.yaml")
    validate_config(config)
    expected = {"nt":2,"nr":2,"nfft":64,"active_carriers":64,"num_symbols":16,
        "delta_f_hz":30000,"sample_rate_hz":1920000,"cp_len":16,"fft_norm":"ortho",
        "modulation":"qpsk_gray","total_power":1.0}
    if (config["system"] != expected or config["noise"]["snr_db"] != list(SNR_OPTIONS)
            or config["pilot"]["spacing_time"] != 4
            or config["pilot"]["spacing_frequency_options"] != list(DF_OPTIONS)):
        raise ValueError("T10C frozen physical system/pilot/SNR configuration conflict")
    return config


def source_checks():
    """Runtime sources remain byte-identical to the accepted prior results."""
    checks = {}
    for filename, field in (("t10b_oracle_lmmse_results.json", "source_and_reference_sha256"),
                            ("t10c_pre_multiframe_trajectory_results.json", "source_sha256")):
        old = json.loads((ROOT / "outputs" / filename).read_text(encoding="utf-8"))
        for path, digest in old["provenance"][field].items():
            if path.startswith("src/") or path == "configs/base.yaml":
                stage = "t10b" if filename.startswith("t10b") else "t10c_pre"
                checks[path] = recorded_source_matches(ROOT, path, digest, stage, current_hash=sha256_file)
    if not all(checks.values()):
        raise ValueError(f"accepted runtime source changed: {checks}")
    return checks


def mismatch_error_diagonal(estimator, r_true):
    """Offline true-distribution audit only; does not prepare the estimator."""
    p = estimator.pilot_indices
    # W = R_train_hp A_train^-1, using the existing cached Cholesky factor.
    w = cho_solve(estimator.cholesky_factor, estimator.r_hp.conj().T).conj().T
    true_hp = r_true[:, p]
    true_a = r_true[np.ix_(p,p)] + estimator.noise_variance * np.eye(p.size)
    cross = np.sum(w * true_hp.conj(), axis=1)
    ce = np.diag(r_true) - cross - cross.conj() + np.sum((w @ true_a) * w.conj(), axis=1)
    if (not np.all(np.isfinite(ce)) or np.max(np.abs(ce.imag)) > 1e-10
            or np.min(ce.real) < -1e-10):
        raise ValueError("mismatch-theory diagonal failed finite/real/nonnegative checks")
    return ce.real, float(np.max(np.abs(ce.imag)))


def covariance_audit(r_train, r_true):
    spots = [((32,0),(32,4)), ((32,0),(40,0)), ((32,0),(40,4)),
             ((32,0),(32,1)), ((32,0),(32,15)), ((32,0),(33,0)), ((32,0),(48,0))]
    entries = []
    for left,right in spots:
        i,j = left[0]*16+left[1], right[0]*16+right[1]
        a,b = r_train[i,j],r_true[i,j]
        entries.append({"left":list(left),"right":list(right),"flat_indices":[i,j],
            "train":{"real":float(a.real),"imag":float(a.imag)},
            "oracle":{"real":float(b.real),"imag":float(b.imag)},"absolute_error":float(abs(a-b))})
    delta_diag = np.diag(r_train-r_true)
    return {"relative_Frobenius_error":float(np.linalg.norm(r_train-r_true)/np.linalg.norm(r_true)),
        "diagonal_RMSE":float(np.sqrt(np.mean(np.abs(delta_diag)**2))),
        "maximum_diagonal_error":float(np.max(np.abs(delta_diag))),"fixed_entries":entries,
        "diagnostic_only":True,"performance_advantage_threshold_used":False}


def prepare_conditions(frozen, r_true, layouts):
    prepared, rows, diagnostics = {}, [], []
    with (ROOT / "outputs/t10a_lmmse_oracle_theory.csv").open(encoding="utf-8",newline="") as handle:
        previous = {(int(row["dF"]),int(row["snr_db"])):row for row in csv.DictReader(handle)}
    for df in DF_OPTIONS:
        layout = layouts[df]
        for snr in SNR_OPTIONS:
            sigma = noise_variance_from_snr(snr)
            pair = tuple(prepare_train_lmmse(frozen,layout.pilot_mask[q],sigma) for q in range(2))
            prepared[df,snr] = pair
            predicted_sum, oracle_sum, true_sum = 0.0,0.0,0.0
            for q,est in enumerate(pair):
                diagonal,imaginary = mismatch_error_diagonal(est,r_true)
                oracle = prepare_oracle_lmmse(r_true,est.pilot_indices,sigma,(64,16))
                selected = layout.data_mask[q].ravel(order="C")
                predicted_sum += float(diagonal[selected].sum())
                oracle_sum += float(oracle.posterior_diagonal[selected].sum())
                true_sum += float(np.diag(r_true).real[selected].sum())
                diagnostics.append({"dF":df,"snr_db":snr,"tx":q,**est.diagnostics,
                    "mismatch_diagonal_imaginary_max":imaginary})
            predicted,matched = predicted_sum/true_sum,oracle_sum/true_sum
            if predicted < matched - 1e-10:
                raise ValueError("Train mismatch-theory MSE is below matched Oracle MSE")
            if not np.isclose(matched,float(previous[df,snr]["posterior_nmse_data_combined"]),rtol=1e-12,atol=1e-15):
                raise ValueError("T10A matched posterior theory regression changed")
            rows.append({"dF":df,"pilot_ratio":count_layout_resources(layout)["union_ratio"],
                "snr_db":snr,"noise_variance":sigma,"oracle_nmse_data_linear":matched,
                "oracle_nmse_data_db":float(10*np.log10(matched)),
                "train_predicted_nmse_data_linear":predicted,
                "train_predicted_nmse_data_db":float(10*np.log10(predicted)),
                "train_minus_oracle_db":float(10*np.log10(predicted/matched)),
                "train_cholesky_success_tx0":pair[0].diagnostics["cholesky_success"],
                "train_cholesky_success_tx1":pair[1].diagnostics["cholesky_success"],
                "condition_number_tx0":pair[0].diagnostics["condition_number"],
                "condition_number_tx1":pair[1].diagnostics["condition_number"]})
    return prepared,rows,diagnostics


def validation_noise_seed(condition_index, trajectory_id, rx, tx):
    if condition_index not in range(4) or trajectory_id not in range(2000,2400) or rx not in range(2) or tx not in range(2):
        raise ValueError("noise seed requires a fixed condition and held-out link")
    return NOISE_SEED_BASE + 100000*condition_index + 4*trajectory_id + 2*rx + tx


def heldout_calibration(rows, model, prepared, layouts, theory, progress=None):
    if len(rows) != 400 or [r["trajectory_id"] for r in rows] != list(range(2000,2400)) or any(r["split"] != "validation_id" for r in rows):
        raise ValueError("held-out calibration requires all 400 validation manifest rows")
    energy = {key:{"data_error_energy":0.0,"data_truth_energy":0.0,"num_link_frame_observations":0} for key in CONDITIONS}
    visited = []
    for row in rows:
        h = formal_trajectory(model,row).h_frames
        visited.append(row["trajectory_id"])
        for condition_index,key in enumerate(CONDITIONS):
            layout = layouts[key[0]]
            for q,est in enumerate(prepared[key]):
                truth = h[:,:,q]  # [frame,RX,K,T]
                hp = truth.reshape(10,2,1024)[:,:,est.pilot_indices]
                noise = np.stack([complex_gaussian_noise((10,est.pilot_indices.size),est.noise_variance,
                    np.random.default_rng(validation_noise_seed(condition_index,row["trajectory_id"],r,q))) for r in range(2)],axis=1)
                prediction = est.estimate(hp+noise)
                mask = layout.data_mask[q]
                energy[key]["data_error_energy"] += float(np.sum(np.abs((prediction-truth)[:,:,mask])**2))
                energy[key]["data_truth_energy"] += float(np.sum(np.abs(truth[:,:,mask])**2))
                energy[key]["num_link_frame_observations"] += 20
        if progress and len(visited)%50 == 0:
            progress(len(visited))
    lookup = {(r["dF"],r["snr_db"]):r for r in theory}
    result = []
    for index,key in enumerate(CONDITIONS):
        e = energy[key]
        nmse = e["data_error_energy"]/e["data_truth_energy"]
        db = float(10*np.log10(nmse))
        predicted = lookup[key]["train_predicted_nmse_data_db"]
        difference = db-predicted
        result.append({"condition":chr(65+index),"dF":key[0],"snr_db":key[1],**e,
            "num_independent_validation_trajectories":400,"empirical_nmse_data_linear":nmse,
            "empirical_nmse_data_db":db,"predicted_nmse_data_db":predicted,
            "empirical_minus_theory_db":difference,"absolute_difference_db":abs(difference),
            "tolerance_db":NMSE_TOLERANCE_DB,"passed":bool(abs(difference)<=NMSE_TOLERANCE_DB)})
    return result,visited


def pilot_ls_integration(model, validation_row, layout, pair):
    if validation_row["trajectory_id"] != 2000 or validation_row["split"] != "validation_id":
        raise ValueError("actual Pilot-LS sentinel is fixed at held-out trajectory 2000")
    h = formal_trajectory(model,validation_row).h_frames[0]
    x = map_pilot_symbols(layout,PILOT_SYMBOLS)
    noise_seed = 71261001  # Separate actual-receiver integration noise namespace.
    noise = complex_gaussian_noise((2,64,16),0.1,np.random.default_rng(noise_seed))
    y = apply_frequency_mimo_channel(h,x)+noise
    ls,observed = pilot_ls_estimate(y,x,layout.pilot_mask)
    direct = np.full_like(ls,np.nan+1j*np.nan)
    observation_error = 0.0
    for q in range(2):
        mask = layout.pilot_mask[q]
        direct[:,q,mask] = h[:,q,mask]+noise[:,mask]/x[q,mask]
        observation_error = max(observation_error,float(np.max(np.abs(ls[:,q,mask]-direct[:,q,mask]))))
    estimate = estimate_mimo_from_pilot_ls(ls,pair)
    expected = estimate_mimo_from_pilot_ls(direct,pair)
    # Exact interface regression must compare identical observations. The two
    # algebraically equivalent LS constructions differ at floating-point level;
    # their solved-output difference is separately reported, not called exact.
    same_input = np.stack([pair[q].estimate(ls[:,q].reshape(2,1024)[:,pair[q].pilot_indices])
        for q in range(2)],axis=1)
    estimation_error = float(np.max(np.abs(estimate-same_input)))
    rounded_input_output_difference = float(np.max(np.abs(estimate-expected)))
    return {"trajectory_id":2000,"frame_index":0,"dF":4,"snr_db":10,"noise_seed":noise_seed,
        "noise_variance":0.1,"actual_pilot_ls_observation_max_error":observation_error,
        "train_estimate_interface_max_error":estimation_error,"estimate_shape":list(estimate.shape),
        "solved_output_difference_from_roundoff_different_observations":rounded_input_output_difference,
        "estimate_all_finite":bool(np.all(np.isfinite(estimate))),
        "unit_pilots_nonowner_zero":bool(np.all(x[layout.mute_mask]==0)),
        "observed_mask_matches":bool(np.array_equal(observed,layout.pilot_mask)),
        "comparison_policy":"machine-precision observation formula; exact same-observation production interface",
        "tolerance":1e-14,"passed":bool(observation_error<=1e-14 and estimation_error<=1e-14
            and np.all(np.isfinite(estimate)) and np.array_equal(observed,layout.pilot_mask))}


def memory_estimate():
    batch = 1000*1024*16
    covariance = 1024*1024*16
    trajectory = (2*2*23*160+2*2*64*160)*16
    packing = 40*1024*16
    factor_phase = (160*160+23*64)*16
    # During GEMM: reusable X and S, temporary conjugate X and product S.
    common = trajectory+packing+factor_phase+3*1024*16
    gemm_peak = 2*batch+2*covariance+common
    # Final symmetry diagnostics can coexist with S/raw/R/conjugate/difference.
    finalization_peak = batch+5*covariance+common
    peak = max(gemm_peak,finalization_peak)
    return {"dtype":"complex128","batch_matrix_shape":[1000,1024],"batch_matrix_bytes":batch,
        "batch_matrix_MiB":batch/2**20,"covariance_matrix_bytes":covariance,
        "covariance_matrix_MiB":covariance/2**20,"H_frames_is_view":True,
        "training_peak_explicit_buffers_estimated_bytes":peak,
        "training_peak_explicit_buffers_estimated_MiB":peak/2**20,
        "training_GEMM_buffers_estimated_MiB":gemm_peak/2**20,
        "training_finalization_buffers_estimated_MiB":finalization_peak/2**20,
        "full_observation_matrix_materialized":False,
        "avoided_full_observation_matrix_bytes":80000*1024*16,
        "estimate_scope":"conservative training ndarray buffer bound, including finalization; excludes BLAS/LAPACK workspace and Python/RSS overhead",
        "audit_phase_note":"After training, frozen R and R_true plus 40 prepared cores are retained; not included in training peak."}


def run(benchmark_only=False):
    started = perf_counter()
    config = check_config()
    old_checks = source_checks()
    outputs = ROOT / "outputs"
    manifest_path = outputs / "t10c_train_validation_manifest.csv"
    rows = ensure_manifest(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation_id"]
    model = prepare_trajectory()
    print("Benchmark: 50 train trajectories, batch_size=25",flush=True)
    bench_r,bench_mean,bench = estimate_training_covariance(train[:50],model)
    benchmark = {"num_train_trajectories":50,"num_link_frame_observations":2000,
        **bench["runtime"],"estimated_2000_accumulation_seconds":40*bench["runtime"]["total_accumulation_seconds"],
        "model_preparation":model.preparation_timing,"projection_includes_disk_io":False}
    print(json.dumps({"benchmark":benchmark,"memory":memory_estimate()},indent=2),flush=True)
    del bench_r,bench_mean
    if benchmark_only:
        return
    events = ["read_manifest","benchmark_train_only"]
    print("Formal accumulation: 2000 independent trajectory units / 80000 link-frame observations",flush=True)
    r,mean,accumulation = estimate_training_covariance(train,model,
        progress=lambda n: print(f"Train {n}/2000",flush=True) if n%100==0 else None)
    if accumulation["num_link_frame_observations"] != 80000:
        raise ValueError("formal observation count must be exactly 80000")
    events.append("estimate_R_train")
    stamp = perf_counter(); stats = covariance_statistics(r)
    psd_seconds = perf_counter()-stamp
    covariance_path = outputs / "t10c_lmmse_train_covariance.npz"
    frozen = save_frozen_covariance(covariance_path,r,mean,manifest_sha)
    events.extend(["save_R_train","compute_SHA256","mark_covariance_frozen"])
    del r,mean
    freeze_before_validation = True
    # Oracle construction starts after freeze. It only enters offline diagnostics.
    events.append("oracle_offline_audit")
    r_true = build_full_grid_covariance(build_frequency_covariance(),build_time_covariance())
    audit = covariance_audit(frozen.R_train,r_true)
    layouts = {df:build_fdm_pilot_layout(2,64,16,df,4) for df in DF_OPTIONS}
    stamp = perf_counter()
    prepared,theory,diagnostics = prepare_conditions(frozen,r_true,layouts)
    theory_seconds = perf_counter()-stamp
    events.append("evaluate_validation")
    stamp = perf_counter()
    integration = pilot_ls_integration(model,validation[0],layouts[4],prepared[4,10])
    if not integration["passed"]:
        raise RuntimeError(f"Small actual Pilot-LS integration sentinel failed: {integration}")
    print("Small actual Pilot-LS integration sentinel PASS; starting all 400 held-out trajectories",flush=True)
    heldout,visited = heldout_calibration(validation,model,prepared,layouts,theory,
        progress=lambda n: print(f"Validation {n}/400",flush=True))
    validation_seconds = perf_counter()-stamp
    hashes_after = {"covariance_file_sha256":sha256_file(covariance_path),
        "R_train_raw_array_sha256":raw_array_sha256(frozen.R_train),"manifest_sha256":sha256_file(manifest_path)}
    hashes_before = {"covariance_file_sha256":frozen.file_sha256,
        "R_train_raw_array_sha256":frozen.raw_array_sha256,"manifest_sha256":manifest_sha}
    hashes_unchanged = hashes_before == hashes_after
    checks = {"formal_manifest":len(train)==2000 and len(validation)==400,
        "independent_train_units":accumulation["num_independent_trajectory_units"]==2000,
        "80000_observations":accumulation["num_link_frame_observations"]==80000,
        "train_only_IDs":accumulation["visited_train_trajectory_ids"]==list(range(2000)),
        "validation_only_IDs":visited==list(range(2000,2400)),"frozen_before_validation":freeze_before_validation,
        "stable_hashes":hashes_unchanged,"PSD":stats["min_eigenvalue"]>=-1e-12,
        "finite_Hermitian":stats["all_finite"] and stats["hermitian_max_error"]<=1e-12,
        "40_train_Cholesky":len(diagnostics)==40 and all(d["cholesky_success"] for d in diagnostics),
        "20_theory_conditions":len(theory)==20,
        "train_theory_ge_oracle":all(r["train_predicted_nmse_data_linear"]>=r["oracle_nmse_data_linear"]-1e-10 for r in theory),
        "16000_validation_observations":all(r["num_link_frame_observations"]==16000 for r in heldout),
        "theory_vs_empirical":all(r["passed"] for r in heldout),"actual_Pilot_LS":integration["passed"],
        "accepted_runtime_sources":all(old_checks.values())}
    csv_path = outputs / "t10c_lmmse_train_theory.csv"
    with csv_path.open("w",encoding="utf-8",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(theory[0])); writer.writeheader(); writer.writerows(theory)
    source_paths = ["src/lmmse_train.py","scripts/validate_lmmse_train.py","tests/test_lmmse_train.py",
        "docs/dataset_protocol.md","tests/test_trajectories.py","configs/base.yaml",*old_checks]
    memory = memory_estimate()
    cache_arrays = [a for pair in prepared.values() for core in pair for a in (
        core.pilot_indices,core.r_hp,core.r_pp,core.observation_covariance,core.cholesky_factor[0],
        core.prior_diagonal,core.posterior_diagonal)]
    memory["40_prepared_train_cores_array_bytes"] = sum(a.nbytes for a in cache_arrays)
    memory["40_prepared_train_cores_array_MiB"] = memory["40_prepared_train_cores_array_bytes"]/2**20
    memory["audit_base_retained_covariance_and_core_MiB"] = (memory["40_prepared_train_cores_array_bytes"]+2*frozen.R_train.nbytes)/2**20
    results = {"task":"T10C","experiment_type":"training-only covariance LMMSE estimator calibration",
        "manifest":{"path":str(manifest_path),"num_rows":2400,"train_rows":2000,"validation_id_rows":400,
            "sha256":manifest_sha,"future_learning_models_reuse_same_manifest":True,"split_unit":"trajectory_id"},
        "seed_policy":{"formal_channel_seed_base":60261001,"channel_seed_rule":"60261001 + trajectory_id",
            "train_seed_range":[60261001,60263000],"validation_seed_range":[60263001,60263400],
            "baseline_calibration_channel_seed_base":40261001,"validation_noise_seed_base":NOISE_SEED_BASE,
            "validation_noise_rule":"70261001 + 100000*condition_index + 4*trajectory_id + 2*rx + tx",
            "noise_draw_shape_per_seed":"[10, actual_pilot_count_per_TX]","integration_noise_seed":71261001,
            "bits_generated":False,"training_seed_used":False},
        "num_train_trajectories":2000,"num_validation_trajectories":400,"frames_per_trajectory":10,
        "num_independent_train_trajectory_units":2000,"num_train_link_frame_observations":80000,
        "mean_policy":MEAN_POLICY,"covariance_estimator_definition":"R_train = sum(h h^H)/80000; X.T @ X.conj()",
        "flatten_order":FLATTEN_ORDER,"batch_size":BATCH_SIZE,"accumulation":accumulation,
        "covariance_statistics":stats,"empirical_mean_statistics":{k:accumulation[k] for k in (
            "mean_vector_rms_magnitude","max_absolute_mean_entry","mean_power_relative_to_channel_power")},
        "oracle_covariance_audit":audit,"covariance_frozen_before_validation":freeze_before_validation,
        "freeze_event_order":events,"covariance_sha256":frozen.file_sha256,
        "R_train_raw_array_sha256":frozen.raw_array_sha256,"hashes_before_validation":hashes_before,
        "hashes_after_validation":hashes_after,"validation_did_not_mutate_frozen_covariance":hashes_unchanged,
        "train_theoretical_conditions":theory,"heldout_validation_conditions":heldout,
        "mismatch_theory_vs_empirical":{"tolerance_db":NMSE_TOLERANCE_DB,
            "max_absolute_difference_db":max(r["absolute_difference_db"] for r in heldout),"passed":checks["theory_vs_empirical"]},
        "validation_visited_trajectory_ids":visited,"actual_pilot_ls_integration":integration,
        "solver":"src/lmmse.py: cached cho_factor + cho_solve","train_estimators_prepared":40,
        "train_cholesky_diagnostics":diagnostics,"explicit_inverse_used":False,"pseudo_inverse_used":False,
        "jitter_used":False,"diagonal_renormalization_used":False,"shrinkage_used":False,
        "oracle_blending_used":False,"eigenvalue_clipping_used":False,"rank_truncation_used":False,
        "sample_mean_subtracted":False,"per_trajectory_normalization":False,
        "formal_test_sealed":True,"test_id_generated":False,"test_ood_generated":False,"ood_dev_generated":False,
        "full_link_BER_run":False,"benchmark":benchmark,"memory":memory,
        "runtime":{"training":accumulation["runtime"],"PSD_diagnostic_seconds":psd_seconds,
            "prepare_and_theory_seconds":theory_seconds,"heldout_and_integration_seconds":validation_seconds,
            "total_seconds":perf_counter()-started},"checks":checks,"all_checks_passed":all(checks.values()),
        "config":config,"provenance":{"accepted_T10C_PRE_commit":PRE_COMMIT,
            "frozen_runtime_source_checks":old_checks,"source_sha256":{p:sha256_file(ROOT/p) for p in dict.fromkeys(source_paths)},
            "theory_csv_sha256":sha256_file(csv_path),"python_version":platform.python_version(),
            "numpy_version":np.__version__,"scipy_version":scipy.__version__},
        "warnings":["Tiny negative R_train eigenvalues are recorded without modifying R_train."] if stats["numerical_negative_count"] else []}
    result_path = outputs / "t10c_lmmse_train_results.json"
    result_path.write_text(json.dumps(results,indent=2,ensure_ascii=True,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"checks":checks,"covariance_statistics":stats,"empirical_mean":results["empirical_mean_statistics"],
        "heldout":heldout,"Pilot_LS":integration,"covariance_sha256":frozen.file_sha256,
        "R_train_raw_array_sha256":frozen.raw_array_sha256,"runtime":results["runtime"],
        "all_checks_passed":results["all_checks_passed"]},indent=2),flush=True)
    if not results["all_checks_passed"]:
        raise RuntimeError(f"T10C failed; original results saved at {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-only",action="store_true")
    run(parser.parse_args().benchmark_only)
