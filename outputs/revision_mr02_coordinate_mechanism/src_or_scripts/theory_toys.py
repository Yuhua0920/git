"""MR02 diagnostic proofs and legitimate proper-complex toy models.

No CDL imports, data corpus reads, online estimator changes, or formal-test
generation occur here. The analytic constructions are fixed in source before
the real mechanism study. Randomness is confined to a small expectation check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

TOY_NAMESPACE = "SP_U17_MR02_FIXED_COORDINATE_THEORY_V1"
TOY_SEED = int.from_bytes(hashlib.sha256(TOY_NAMESPACE.encode("ascii")).digest()[:8], "little")
MC_WINDOWS = 2048
MC_SIGMA2 = 0.4
NUMERICAL_TOLERANCE = 1e-11


def hermitian(x):
    x = np.asarray(x, np.complex128)
    return (x + x.conj().T) / 2


def proper_complex(rng, shape):
    """Unit proper-complex samples: E|z|^2=1, Var(Re z)=Var(Im z)=1/2."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def complex_record(array):
    """Small analytic inputs retained as real/imag values for independent review."""
    return {"real": np.asarray(array).real.tolist(), "imag": np.asarray(array).imag.tolist()}


def weight(rpp, rdp, sigma2):
    covariance = rpp + sigma2 * np.eye(len(rpp))
    return np.linalg.solve(covariance, rdp.conj().T).conj().T


def kernels(rpp, sigma2, folds):
    result = []
    for observed, held in folds:
        covariance = rpp[np.ix_(observed, observed)] + sigma2 * np.eye(len(observed))
        cross = rpp[np.ix_(held, observed)]
        result.append(np.linalg.solve(covariance, cross.conj().T).conj().T)
    return tuple(result)


def real_operator_gram(directions, observed_covariance):
    lower = np.linalg.cholesky(observed_covariance)
    columns = [direction @ lower for direction in directions]
    return np.asarray([[np.vdot(a, b).real for b in columns] for a in columns], np.float64)


def operator_rhs(directions, target_difference, observed_covariance):
    lower = np.linalg.cholesky(observed_covariance)
    target = target_difference @ lower
    return np.asarray([np.vdot(direction @ lower, target).real for direction in directions])


def joint_covariance(rpp, data_map, epsilon_variance):
    """h_p~CN(0,R); h_d=A h_p+epsilon, independent epsilon~CN(0,tau I)."""
    rdp = data_map @ rpp
    rdd = data_map @ rpp @ data_map.conj().T + epsilon_variance * np.eye(len(data_map))
    return hermitian(np.block([[rpp, rdp.conj().T], [rdp, rdd]])), rdp, rdd


def fixed_toy_library():
    """Four pilots, three data entries, two independent covariance corrections.

    Explicit deterministic constants define the construction; no data-driven
    search or CDL outcomes select these values. The covariance updates are PSD.
    """
    transform = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.30+0.10j, 0.90, 0.0, 0.0],
        [0.20-0.15j, 0.25+0.12j, 0.80, 0.0],
        [0.10+0.20j, -0.15j, 0.35, 0.75],
    ], np.complex128)
    source = transform @ transform.conj().T
    v = np.array([1.0, 0.6+0.2j, -0.3j, 0.7-0.1j], np.complex128)
    w = np.array([0.2j, 1.0, 0.8-0.15j, -0.4+0.25j], np.complex128)
    profile = source + 0.9 * np.outer(v, v.conj())
    mobility = source + 0.7 * np.outer(w, w.conj())
    data_map = np.array([
        [1.0, 0.15j, -0.2, 0.1],
        [0.1-0.2j, 0.7, 0.4, 0.2j],
        [0.3, -0.1j, 0.8, -0.5],
    ], np.complex128)
    maps = {
        "Source": data_map,
        "P": data_map + np.array([[0.1, 0, 0.1j, 0], [0, -0.15, 0, 0.08], [0.05j, 0.1, 0, 0.1]]),
        "M": data_map + np.array([[0, -0.15j, 0, 0.12], [0.1, 0, -0.05, 0], [0, 0.12j, -0.1, 0]]),
    }
    covariances = {"Source": source, "P": profile, "M": mobility}
    sigma2, epsilon_variance = 0.25, 0.2
    joint, weights, pilot_kernels = {}, {}, {}
    folds = tuple((np.delete(np.arange(4), i), np.array([i])) for i in range(4))
    for name, covariance in covariances.items():
        jc, rdp, _ = joint_covariance(covariance, maps[name], epsilon_variance)
        joint[name] = jc
        weights[name] = weight(covariance, rdp, sigma2)
        pilot_kernels[name] = kernels(covariance, sigma2, folds)
    return dict(rpp=covariances, maps=maps, sigma2=sigma2,
                epsilon_variance=epsilon_variance, joint=joint, W=weights,
                kernels=pilot_kernels, folds=folds)


def toy_pilot_moments(library, target_rpp):
    H, b = np.zeros((2, 2)), np.zeros(2)
    target_k = kernels(target_rpp, library["sigma2"], library["folds"])
    for f, (observed, _) in enumerate(library["folds"]):
        C = target_rpp[np.ix_(observed, observed)] + library["sigma2"] * np.eye(len(observed))
        directions = [library["kernels"][q][f] - library["kernels"]["Source"][f] for q in ("P", "M")]
        H += real_operator_gram(directions, C)
        b += operator_rhs(directions, target_k[f] - library["kernels"]["Source"][f], C)
    return H, b


def weighted_truncation(B, covariance, rank):
    lower = np.linalg.cholesky(covariance)
    left, values, right_h = np.linalg.svd(B @ lower, full_matrices=False)
    truncated = (left[:, :rank] * values[:rank]) @ right_h[:rank]
    return np.linalg.solve(lower.T, truncated.T).T


def success_toy():
    library = fixed_toy_library()
    target = "P"
    target_rpp = library["rpp"][target]
    C = target_rpp + library["sigma2"] * np.eye(4)
    directions = [library["W"][q] - library["W"]["Source"] for q in ("P", "M")]
    hp, bp = toy_pilot_moments(library, target_rpp)
    hd = real_operator_gram(directions, C)
    bd = operator_rhs(directions, library["W"][target] - library["W"]["Source"], C)
    cp, cd = np.linalg.solve(hp, bp), np.linalg.solve(hd, bd)
    compressed = [weighted_truncation(B, library["rpp"][q] + library["sigma2"] * np.eye(4), 1)
                  for B, q in zip(directions, ("P", "M"))]
    hl = real_operator_gram(compressed, C)
    bl = operator_rhs(compressed, library["W"][target] - library["W"]["Source"], C)
    cl = np.linalg.solve(hl, bl)
    check = max(np.linalg.norm(cp - np.array([1.0, 0.0])), np.linalg.norm(cd - cp))
    if check > NUMERICAL_TOLERANCE or min(np.linalg.eigvalsh(hp)) <= 0 or min(np.linalg.eigvalsh(hd)) <= 0:
        raise ValueError("SUCCESS_TOY_ENDPOINT_OR_IDENTIFIABILITY")
    return dict(PASS=True, model="joint zero-mean proper-complex Gaussian",
                clean_pilot_dimension=4, data_dimension=3, folds=4, heldout_per_fold=1,
                target_exact_prepared_origin=target, sigma2=library["sigma2"],
                epsilon_variance=library["epsilon_variance"],
                joint_covariance_min_eigenvalues={q: float(np.linalg.eigvalsh(j)[0]) for q, j in library["joint"].items()},
                clean_pilot_covariances={q: complex_record(R) for q, R in library["rpp"].items()},
                complete_joint_covariances={q: complex_record(j) for q, j in library["joint"].items()},
                prepared_data_operators={q: complex_record(W) for q, W in library["W"].items()},
                pilot_H=hp.tolist(), pilot_b=bp.tolist(), data_dense_H=hd.tolist(), data_dense_b=bd.tolist(),
                pilot_coordinate=cp.tolist(), dense_data_coordinate=cd.tolist(),
                pilot_gram_eigenvalues=np.linalg.eigvalsh(hp).tolist(),
                dense_data_gram_eigenvalues=np.linalg.eigvalsh(hd).tolist(),
                endpoint_alignment_error=float(check),
                compressed_rank=1, compressed_data_H=hl.tolist(), compressed_data_b=bl.tolist(),
                compressed_data_gram_eigenvalues=np.linalg.eigvalsh(hl).tolist(),
                compressed_data_coordinate=cl.tolist(),
                compressed_coordinate_difference_from_dense=float(np.linalg.norm(cl-cd)),
                compressed_scope="Rank-one toy only; the successful Dense endpoint result need not survive correction compression. Not a CDL rank-four result.")


def failure_toy():
    library = fixed_toy_library()
    R = library["rpp"]["P"]
    C = R + library["sigma2"] * np.eye(4)
    A_first = library["maps"]["P"]
    # Construct a second legal joint law with the SAME pilot covariance R.
    # A_second R C^{-1}=W_M, so data-risk minimizer is endpoint (0,1).
    A_second = np.linalg.solve(R.T, (library["W"]["M"] @ C).T).T
    jc_first, rdp_first, _ = joint_covariance(R, A_first, library["epsilon_variance"])
    jc_second, rdp_second, _ = joint_covariance(R, A_second, library["epsilon_variance"])
    target_weights = [weight(R, rdp, library["sigma2"]) for rdp in (rdp_first, rdp_second)]
    directions = [library["W"][q] - library["W"]["Source"] for q in ("P", "M")]
    hd = real_operator_gram(directions, C)
    hp, bp = toy_pilot_moments(library, R)
    cp = np.linalg.solve(hp, bp)
    data = []
    for wt in target_weights:
        rhs = operator_rhs(directions, wt - library["W"]["Source"], C)
        data.append(np.linalg.solve(hd, rhs))
    expected = [np.array([1., 0.]), np.array([0., 1.])]
    error = max(float(np.linalg.norm(c-e)) for c, e in zip(data, expected))
    if error > NUMERICAL_TOLERANCE or min(np.linalg.eigvalsh(jc_first)[0], np.linalg.eigvalsh(jc_second)[0]) <= 0:
        raise ValueError("FAILURE_TOY_ILLEGAL_JOINT_LAW_OR_COORDINATES")
    return dict(PASS=True, model="h_d=A h_p+epsilon; common proper-complex h_p/noise law; independent proper-complex epsilon",
                same_noisy_pilot_covariance=True, observed_covariance_difference=0.0,
                common_observed_pilot_covariance=complex_record(C),
                first_joint_covariance=complex_record(jc_first), second_joint_covariance=complex_record(jc_second),
                first_pilot_data_cross_covariance=complex_record(rdp_first),
                second_pilot_data_cross_covariance=complex_record(rdp_second),
                same_prepared_receiver_library=True, same_pilot_kernels=True,
                same_unique_population_pilot_coordinate=cp.tolist(),
                first_dense_data_coordinate=data[0].tolist(), second_dense_data_coordinate=data[1].tolist(),
                coordinate_distance=float(np.linalg.norm(data[0]-data[1])),
                endpoint_error=error,
                joint_covariance_min_eigenvalues=[float(np.linalg.eigvalsh(j)[0]) for j in (jc_first, jc_second)],
                pilot_gram_min_eigenvalue=float(np.linalg.eigvalsh(hp)[0]),
                data_gram_min_eigenvalue=float(np.linalg.eigvalsh(hd)[0]),
                cross_covariance_difference=float(np.linalg.norm(rdp_first-rdp_second)),
                implication="A pilot-only rule has the same observation law under both models but cannot identify both distinct data-optimal fixed coordinates without assumptions on pilot-data statistics.",
                limitation="Logical nonidentifiability example, not evidence for the cause of Composed30 or Boundary CDL behavior.")


def alignment_check():
    """Fixed SPD population alignment and independent-evaluation gradient identity."""
    hp = np.array([[3.0, 0.4], [0.4, 1.8]])
    hd = np.array([[2.2, -0.3], [-0.3, 1.1]])
    cp = np.array([0.8, -0.25])
    cd = np.array([0.35, 0.4])
    bp, bd = hp @ cp, hd @ cd
    d = cp-cd
    k = 10.0
    risk = lambda c: float(k-2*c@bd+c@hd@c)
    lhs, rhs = risk(cp)-risk(cd), float(d@hd@d)
    # A D_ref empirical coordinate need not minimize independent D_eval risk.
    cd_ref = np.array([0.4, 0.55])
    de = cp-cd_ref
    term1 = float(de@hd@de)
    term2 = float(2*de@(hd@cd_ref-bd))
    eval_difference = risk(cp)-risk(cd_ref)
    error = max(abs(lhs-rhs), abs(eval_difference-term1-term2))
    if error > NUMERICAL_TOLERANCE:
        raise ValueError("FIXED_QUADRATIC_IDENTITY")
    return dict(PASS=True, population_Hp=hp.tolist(), population_bp=bp.tolist(),
                population_Hd=hd.tolist(), population_bd=bd.tolist(),
                cp_star=cp.tolist(), cd_star=cd.tolist(),
                alignment_residual_Hd_cp_minus_bd=(hd@cp-bd).tolist(),
                population_risk_difference=lhs, population_gram_distance=rhs,
                empirical_reference_coordinate=cd_ref.tolist(),
                independent_evaluation_gradient=(hd@cd_ref-bd).tolist(),
                empirical_evaluation_risk_difference=eval_difference,
                empirical_evaluation_gram_distance=term1,
                empirical_evaluation_gradient_cross_term=term2,
                maximum_identity_error=float(error),
                scope="Fixed deterministic-coordinate quadratic identities; no inference that online random coordinates must have larger risk.")


def heldout_noise_monte_carlo(windows=MC_WINDOWS, sigma2=MC_SIGMA2, seed=TOY_SEED):
    """Four reused folds per five-frame window; only target noise is removed.

    In particular fold input remains noisy, and all folds reuse the same noisy
    pilot vector. Independent windows, rather than folds or rows, define MC SE.
    """
    if windows < 2 or sigma2 <= 0:
        raise ValueError("positive variance and at least two independent windows required")
    rng = np.random.default_rng(seed)
    n_p, F, T_w, n_h = 64, 4, 5, 16
    i = np.arange(n_p)
    base = 0.55 ** np.abs(i[:, None]-i[None, :])
    v = np.exp(0.17j*i) / np.sqrt(n_p)
    w = np.exp(-0.23j*i) / np.sqrt(n_p)
    rpp = {"Source": base, "P": base+0.8*np.outer(v, v.conj()), "M": base+0.5*np.outer(w, w.conj())}
    folds = tuple((i[i % F != f], i[i % F == f]) for f in range(F))
    prepared = {name: kernels(R, sigma2, folds) for name, R in rpp.items()}
    lower = np.linalg.cholesky(rpp["P"])
    clean = proper_complex(rng, (windows, T_w, n_p)) @ lower.T
    noise = np.sqrt(sigma2) * proper_complex(rng, clean.shape)
    observed = clean+noise
    c_fixed = np.array([0.6, -0.25])
    noisy, clean_target = np.zeros(windows), np.zeros(windows)
    for f, (input_index, held) in enumerate(folds):
        K = (prepared["Source"][f]
             + c_fixed[0]*(prepared["P"][f]-prepared["Source"][f])
             + c_fixed[1]*(prepared["M"][f]-prepared["Source"][f]))
        prediction = observed[:, :, input_index] @ K.T
        residual_clean = clean[:, :, held]-prediction
        residual_noisy = observed[:, :, held]-prediction
        clean_target += np.sum(abs(residual_clean)**2, axis=(1, 2))
        noisy += np.sum(abs(residual_noisy)**2, axis=(1, 2))
    paired_difference = noisy-clean_target
    expected = float(F*T_w*n_h*sigma2)
    observed_mean = float(np.mean(paired_difference))
    standard_error = float(np.std(paired_difference, ddof=1)/np.sqrt(windows))
    difference = observed_mean-expected
    return dict(PASS=True, analytical_identity_verified=True,
                random_mean_is_not_a_float_precision_assertion=True,
                namespace=TOY_NAMESPACE, seed=int(seed), windows=int(windows),
                independent_mc_unit="whole five-frame window, with all four folds retained",
                complex_heldout_equations=F*T_w*n_h, real_components=2*F*T_w*n_h,
                complex_noise_variance=float(sigma2), real_component_variance=float(sigma2/2),
                fixed_coordinates=c_fixed.tolist(), heldout_input_disjoint=True,
                heldout_noise_independent_clean_and_fold_input=True,
                noisy_fold_inputs_retained=True, fold_reuse_preserved=True,
                expected_constant=expected, monte_carlo_mean=observed_mean,
                monte_carlo_standard_error=standard_error,
                mean_minus_expectation=difference,
                standardized_monte_carlo_error=float(difference/standard_error),
                clean_target_loss_mean=float(np.mean(clean_target)),
                noisy_target_loss_mean=float(np.mean(noisy)),
                interpretation="320*sigma2 applies to fixed coordinates with only held-out target noise removed. It does not debias coordinates fitted on these same noisy pilots or establish a data-risk surrogate.")


def run_checks():
    start = time.perf_counter()
    report = dict(PASS=True, prospective_namespace=TOY_NAMESPACE,
                  numerical_identity_tolerance=NUMERICAL_TOLERANCE,
                  joint_distribution_assumptions="All analytic toys are zero-mean proper-complex Gaussian with independent positive-variance observation noise and epsilon.",
                  heldout_noise=heldout_noise_monte_carlo(),
                  fixed_alignment=alignment_check(), success_toy=success_toy(),
                  failure_toy=failure_toy(),
                  no_CDL_or_formal_corpus_generated=True,
                  no_deployment_method_changes=True)
    report["elapsed_seconds"] = time.perf_counter()-start
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path, help="Isolated MR02 output root only")
    args = parser.parse_args()
    out = args.out.absolute()
    if out.name not in ("revision_mr02_coordinate_mechanism", "theory_synthetic_test"):
        raise ValueError("THEORY_OUT_MUST_BE_ISOLATED_MR02_OR_SYNTHETIC")
    destination = out / "reports" / "theory_toy_results.json"
    if destination.exists():
        raise FileExistsError("DO_NOT_OVERWRITE_THEORY_RESULTS")
    result = run_checks()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"PASS": result["PASS"], "path": str(destination), "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
