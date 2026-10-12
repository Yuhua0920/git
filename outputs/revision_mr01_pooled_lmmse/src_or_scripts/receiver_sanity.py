"""SP-U17/MR-01 synthetic receiver convention check using frozen project APIs.

This does not generate scientific channels or evaluate any historical BER run.
Only this entry point's explicit output directory receives generated files.
Run with Python -B to leave the project source tree free of new bytecode.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys

import numpy as np


FLOAT_ATOL = 1e-12
POWER_ATOL = 1e-15
NOISE_RTOL = 0.015
NOISE_SEED = 170101
NOISE_SAMPLES = 262144
GRAY_DIBITS = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
EXPECTED_GRAY = np.array([1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j]) / np.sqrt(2.0)
ANALYTIC_H = np.array(
    [[1.0 + 0.5j, -0.25 + 0.75j], [0.4 - 0.6j, 1.2 + 0.2j]],
    dtype=np.complex128,
)


def load_apis(project_root: str | Path) -> dict:
    """Import existing source from the caller-specified project, without copying it."""
    root = Path(project_root).resolve()
    for required in ("src/modulation.py", "src/detectors.py", "configs/base.yaml"):
        if not (root / required).is_file():
            raise ValueError(f"project root is missing {required}: {root}")
    sys.dont_write_bytecode = True
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    modules = {
        name: importlib.import_module(f"src.{name}")
        for name in ("config", "modulation", "numerics", "detectors", "link", "pilots", "lmmse_train_validation")
    }
    # A cached src package from another checkout must never silently satisfy this test.
    for name, module in modules.items():
        if Path(module.__file__).resolve() != (root / "src" / f"{name}.py").resolve():
            raise ValueError(f"src.{name} resolves outside requested project root")
    return {"root": root, **modules}


def _complex_pairs(values: np.ndarray) -> list:
    a = np.asarray(values)
    return np.stack((a.real, a.imag), axis=-1).tolist()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _max_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(actual) - np.asarray(expected)) / np.linalg.norm(expected))


def system_check(apis: dict) -> dict:
    root = apis["root"]
    config = apis["config"].load_config(root / "configs/base.yaml")
    derived = apis["config"].validate_config(config)
    s = config["system"]
    expected = {
        "nt": 2, "nr": 2, "nfft": 64, "active_carriers": 64,
        "num_symbols": 16, "delta_f_hz": 30000, "sample_rate_hz": 1920000,
        "cp_len": 16, "fft_norm": "ortho", "modulation": "qpsk_gray", "total_power": 1.0,
    }
    differences = {key: {"expected": value, "actual": s.get(key)}
                   for key, value in expected.items() if s.get(key) != value}
    table_path = root / "outputs/paper_v4_2_candidate/supplement/si_system_table.tex"
    model_path = root / "outputs/paper_v4_2_candidate/sections/03_system_model.tex"
    expected_table_lines = [
        "$N_t,N_r$ & 2, 2", "$K,T$; active subcarriers & 64, 16; 64",
        "$\\Delta f$; $F_s$ & 30 kHz; 1.92 MHz",
        "CP length; FFT normalization & 16 samples; ortho",
        "Modulation; total expected power/active RE & Gray QPSK; 1.0",
        "Data-stream scaling & $1/\\sqrt{2}$",
    ]
    table_text = table_path.read_text(encoding="utf-8")
    missing_table_entries = [line for line in expected_table_lines if line not in table_text]
    return {
        "passed": not differences and not missing_table_entries,
        "base_config": s, "derived_sample_rate_hz": derived,
        "expected_current_dimensions": {"H": [2, 2, 64, 16], "X": [2, 64, 16], "Y": [2, 64, 16]},
        "config_differences": differences,
        "v4_2_table_missing_entries": missing_table_entries,
        "v4_2_system_sources": {
            "table": {"path": str(table_path), "sha256": _hash(table_path)},
            "model": {"path": str(model_path), "sha256": _hash(model_path)},
        },
        "scientific_noise_calibration_note": (
            "V4.2 scientific SNR uses fixed Source P_rx,ref=9.941134822524, "
            "sigma2=P_rx,ref*10**(-gamma_dB/10). This receiver sanity does not run that experiment. "
            "The generic noise_variance_from_snr(snr,total_power) API is checked only with an "
            "explicit synthetic reference power."
        ),
    }


def make_analytic_fixture(apis: dict) -> dict:
    """All 16 two-stream dibit combinations, with distinct TX and K/T patterns."""
    nk, ns = 64, 16
    k, t = np.indices((nk, ns))
    labels = np.stack(((t // 4 + k) % 4, (t % 4 + t // 4) % 4))
    bits = GRAY_DIBITS[labels]
    symbols = np.stack([
        apis["modulation"].qpsk_modulate(bits[q].reshape(-1)).reshape(nk, ns)
        for q in range(2)
    ])
    expected_symbols = EXPECTED_GRAY[labels]
    x = apis["numerics"].scale_data_streams(symbols.reshape(2, -1).T, nt=2, total_power=1.0).T.reshape(2, nk, ns)
    h = np.broadcast_to(ANALYTIC_H[:, :, None, None], (2, 2, nk, ns)).copy()
    y = apis["link"].apply_frequency_mimo_channel(h, x)
    return {"bits": bits, "symbols": symbols, "expected_symbols": expected_symbols, "x": x, "h": h, "y": y}


def analytic_chain_check(apis: dict) -> dict:
    f = make_analytic_fixture(apis)
    bits, s, x, h, y = (f[key] for key in ("bits", "symbols", "x", "h", "y"))
    x_hat = apis["detectors"].zf_detect(y, h)
    restored_s = np.sqrt(2.0) * x_hat
    recovered_bits = np.stack([
        apis["modulation"].qpsk_demodulate(restored_s[q].reshape(-1)).reshape(64, 16, 2)
        for q in range(2)
    ])
    # Independent formulas ensure that a shared forward/inverse convention bug cannot pass.
    expected_y = np.stack([
        ANALYTIC_H[r, 0] * x[0] + ANALYTIC_H[r, 1] * x[1] for r in range(2)
    ])
    manual_x = np.array([0.5 + 0.5j, -0.5 + 0.5j])
    manual_expected_y = np.array([0.0 + 0.25j, -0.2 + 0.4j])
    manual_y = apis["link"].apply_frequency_mimo_channel(
        ANALYTIC_H[:, :, None, None], manual_x[:, None, None]
    )[:, 0, 0]
    manual_hat = apis["detectors"].zf_detect(
        manual_y[:, None, None], ANALYTIC_H[:, :, None, None]
    )[:, 0, 0]
    stream_power_error = _max_error(np.abs(x) ** 2, np.full(x.shape, 0.5))
    total_power_error = _max_error(np.sum(np.abs(x) ** 2, axis=0), np.ones((64, 16)))
    errors = np.count_nonzero(recovered_bits != bits, axis=(1, 2, 3)).astype(int).tolist()
    transposed_hat = apis["detectors"].zf_detect(y, h.swapaxes(0, 1))
    conjugated_hat = apis["detectors"].zf_detect(y, h.conj())
    mapping_error = _max_error(s, f["expected_symbols"])
    channel_error = _max_error(y, expected_y)
    symbol_error = _max_error(x_hat, x)
    restored_error = _max_error(restored_s, s)
    manual_y_error = _max_error(manual_y, manual_expected_y)
    manual_symbol_error = _max_error(manual_hat, manual_x)
    checks = {
        "gray_mapping": mapping_error <= FLOAT_ATOL,
        "forward_RX_TX_axes": channel_error <= FLOAT_ATOL,
        "manual_analytic_y": manual_y_error <= FLOAT_ATOL,
        "manual_analytic_zf": manual_symbol_error <= FLOAT_ATOL,
        "unit_constellation_energy": _max_error(np.abs(s) ** 2, np.ones(s.shape)) <= POWER_ATOL,
        "per_stream_power_half": stream_power_error <= POWER_ATOL,
        "total_active_RE_power_one": total_power_error <= POWER_ATOL,
        "scaled_X_recovery": symbol_error <= FLOAT_ATOL,
        "restored_S_recovery": restored_error <= FLOAT_ATOL,
        "both_streams_zero_bit_errors": errors == [0, 0],
        "TX_RX_transpose_negative_control_detected": _relative_error(transposed_hat, x) > 1e-3,
        "unwanted_conjugation_negative_control_detected": _relative_error(conjugated_hat, x) > 1e-3,
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "fixture_kind": "deterministic analytic complex 2x2 full-rank matrix; no scientific channel generator",
        "H_real_imag": _complex_pairs(ANALYTIC_H), "det_H_real_imag": _complex_pairs(np.linalg.det(ANALYTIC_H)),
        "matrix_rank": int(np.linalg.matrix_rank(ANALYTIC_H)), "condition_number_2norm": float(np.linalg.cond(ANALYTIC_H)),
        "shape_H": list(h.shape), "shape_X": list(x.shape), "shape_Y": list(y.shape),
        "dtype": str(x_hat.dtype), "gray_dibits_in_quadrant_order": GRAY_DIBITS.tolist(),
        "gray_constellation_real_imag": _complex_pairs(EXPECTED_GRAY),
        "max_abs_gray_mapping_error": mapping_error, "max_abs_forward_equation_error": channel_error,
        "manual_X_real_imag": _complex_pairs(manual_x), "manual_expected_Y_real_imag": _complex_pairs(manual_expected_y),
        "max_abs_manual_Y_error": manual_y_error, "max_abs_manual_ZF_error": manual_symbol_error,
        "max_abs_X_error": symbol_error, "relative_l2_X_error": _relative_error(x_hat, x),
        "max_abs_restored_S_error": restored_error,
        "per_stream_mean_power": np.mean(np.abs(x) ** 2, axis=(1, 2)).tolist(),
        "max_abs_stream_power_error": stream_power_error, "max_abs_total_RE_power_error": total_power_error,
        "bits_by_stream": [int(bits[q].size) for q in range(2)], "bit_errors_by_stream": errors,
        "transposed_H_relative_l2_X_error": _relative_error(transposed_hat, x),
        "conjugated_H_relative_l2_X_error": _relative_error(conjugated_hat, x),
        "symbol_tolerance": {"atol": FLOAT_ATOL, "rtol": 0.0},
        "power_tolerance": {"atol": POWER_ATOL, "rtol": 0.0},
    }


def data_wrapper_check(apis: dict) -> dict:
    """Exercise existing pilot/data assembly and masked Perfect-CSI receiver."""
    f = make_analytic_fixture(apis)
    layout = apis["pilots"].build_fdm_pilot_layout(2, 64, 16, 16, 4)
    x = apis["link"].assemble_pilot_data_frame(f["symbols"], layout)
    y = apis["link"].apply_frequency_mimo_channel(f["h"], x)
    x_hat = apis["link"].detect_data_zf(y, f["h"], layout.data_mask)
    batched_hat = apis["lmmse_train_validation"].batched_data_zf(y, f["h"], layout.data_mask)
    ref = apis["link"].extract_reference_data_bits(f["bits"], layout.data_mask)
    got = apis["link"].demodulate_data_bits(np.sqrt(2.0) * x_hat, layout.data_mask)
    batched_got = apis["link"].demodulate_data_bits(np.sqrt(2.0) * batched_hat, layout.data_mask)
    # Construct reference selection independently with explicit (k,t) traversal.
    coordinates = [(k, t) for k in range(64) for t in range(16) if layout.data_mask[0, k, t]]
    independent_ref = tuple(np.concatenate([f["bits"][q, k, t] for k, t in coordinates]) for q in range(2))
    errors = [int(np.count_nonzero(a != b)) for a, b in zip(independent_ref, got)]
    batched_errors = [int(np.count_nonzero(a != b)) for a, b in zip(independent_ref, batched_got)]
    mask = layout.data_mask
    symbol_error = _max_error(x_hat[mask], x[mask])
    batched_symbol_error = _max_error(batched_hat[mask], x[mask])
    power_error = _max_error(np.sum(np.abs(x) ** 2, axis=0), np.ones((64, 16)))
    checks = {
        "common_data_mask": bool(np.array_equal(mask[0], mask[1])),
        "masked_X_recovery": symbol_error <= FLOAT_ATOL,
        "bit_selection_TX_K_T_dibit_order": errors == [0, 0],
        "reference_bit_selection_matches_explicit_K_T_traversal": all(np.array_equal(a, b) for a, b in zip(ref, independent_ref)),
        "batched_adapter_scaled_X_recovery": batched_symbol_error <= FLOAT_ATOL,
        "batched_adapter_bit_order_recovery": batched_errors == [0, 0],
        "batched_adapter_matches_individual_RE_solves": _max_error(batched_hat[mask], x_hat[mask]) <= FLOAT_ATOL,
        "nondata_symbols_are_nan": bool(np.all(np.isnan(x_hat[~mask]))),
        "owner_pilot_data_mute_total_RE_power_one": power_error <= POWER_ATOL,
        "unit_owner_pilots": bool(np.allclose(np.abs(x[layout.pilot_mask]) ** 2, 1.0, rtol=0, atol=POWER_ATOL)),
        "nonowner_pilot_mute_zero": bool(np.all(x[layout.mute_mask] == 0)),
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "pilot_union_REs": int(layout.pilot_union.sum()), "data_REs_by_stream": mask.sum(axis=(1, 2)).astype(int).tolist(),
        "bits_by_stream": [int(a.size) for a in ref], "bit_errors_by_stream": errors,
        "batched_bit_errors_by_stream": batched_errors, "max_abs_batched_X_error": batched_symbol_error,
        "max_abs_batched_vs_individual_RE_error": _max_error(batched_hat[mask], x_hat[mask]),
        "first_explicit_K_T_selected_coordinates": coordinates[:8],
        "max_abs_selected_X_error": symbol_error, "max_abs_total_RE_power_error": power_error,
        "nondata_policy": "NaN+jNaN retained and excluded from demodulation",
    }


def near_singular_check(apis: dict) -> dict:
    """Informational finite-precision behavior; no zero-error threshold is imposed."""
    x = np.array([0.5 + 0.5j, -0.5 + 0.5j], dtype=np.complex128)
    original_bits = np.array([0, 0, 1, 0], dtype=np.uint8)
    rows = []
    for epsilon in (1e-8, 1e-12, 1e-15):
        h = np.array([[1 + 0.25j, 1 + 0.25j],
                      [1 - 0.5j, 1 - 0.5j + epsilon * (1 + 1j)]], dtype=np.complex128)
        y = h @ x
        row = {"epsilon": epsilon, "H_real_imag": _complex_pairs(h),
               "condition_number_2norm": float(np.linalg.cond(h)),
               "numerical_rank_default_tolerance": int(np.linalg.matrix_rank(h)),
               "singular_values": np.linalg.svd(h, compute_uv=False).tolist(),
               "expected_analytic_nonzero_determinant_real_imag": _complex_pairs((1 + 0.25j) * epsilon * (1 + 1j))}
        try:
            hat = apis["detectors"].zf_detect(y[:, None, None], h[:, :, None, None])[:, 0, 0]
            recovered = apis["modulation"].qpsk_demodulate(np.sqrt(2.0) * hat)
            backward = float(np.linalg.norm(h @ hat - y) / (np.linalg.norm(h) * np.linalg.norm(hat) + np.linalg.norm(y)))
            row.update({"behavior": "finite_solution", "solution_real_imag": _complex_pairs(hat),
                        "max_abs_X_error": _max_error(hat, x), "relative_l2_X_error": _relative_error(hat, x),
                        "relative_normwise_backward_residual": backward,
                        "bit_errors_observed": int(np.count_nonzero(recovered != original_bits))})
        except ValueError as exc:
            row.update({"behavior": "existing_solver_rejected", "exception": str(exc)})
        rows.append(row)
    singular = np.array([[1, 2], [2, 4]], dtype=np.complex128)
    try:
        apis["detectors"].zf_detect((singular @ x)[:, None, None], singular[:, :, None, None])
        singular_result = {"rejected": False}
    except ValueError as exc:
        singular_result = {"rejected": True, "exception": str(exc)}
    return {
        "passed": singular_result["rejected"],
        "interpretation": "Near-singular rows describe floating-point conditioning only. No recovery/zero-bit-error criterion or forced correction is applied.",
        "near_singular_rows": rows, "exact_singular_control": singular_result,
        "solver_policy": "Existing np.linalg.solve; no inverse, pseudoinverse, jitter, regularization or fallback",
    }


def noise_convention_check(apis: dict) -> dict:
    """Synthetic-only noise moments, separate from the noiseless receiver check."""
    sigma2 = 0.375
    noise = apis["numerics"].complex_gaussian_noise(
        (2, NOISE_SAMPLES), sigma2, np.random.default_rng(NOISE_SEED)
    )
    real_power = np.mean(noise.real ** 2, axis=1)
    imag_power = np.mean(noise.imag ** 2, axis=1)
    complex_power = np.mean(np.abs(noise) ** 2, axis=1)
    pseudo = np.mean(noise ** 2, axis=1) / sigma2
    means = np.mean(noise, axis=1) / np.sqrt(sigma2)
    cross = np.mean(noise[0] * noise[1].conj()) / sigma2
    # Exact seeded construction verifies the convention independent of sample moments.
    rng = np.random.default_rng(NOISE_SEED)
    expected = np.sqrt(sigma2 / 2.0) * (rng.standard_normal(noise.shape) + 1j * rng.standard_normal(noise.shape))
    seeded_error = _max_error(noise, expected)
    snr_reference = 2.5  # explicit synthetic power; no Source/scientific calibration is substituted
    snr_db = 10.0
    conversion = apis["numerics"].noise_variance_from_snr(snr_db, total_power=snr_reference)
    checks = {
        "exact_seeded_sqrt_sigma2_over_2_construction": seeded_error <= POWER_ATOL,
        "complex_second_moment": bool(np.allclose(complex_power, sigma2, rtol=NOISE_RTOL, atol=0)),
        "real_second_moment": bool(np.allclose(real_power, sigma2 / 2.0, rtol=NOISE_RTOL, atol=0)),
        "imag_second_moment": bool(np.allclose(imag_power, sigma2 / 2.0, rtol=NOISE_RTOL, atol=0)),
        "proper_complex_pseudo_covariance": bool(np.all(np.abs(pseudo) < NOISE_RTOL)),
        "normalized_mean": bool(np.all(np.abs(means) < NOISE_RTOL)),
        "cross_RX_covariance": bool(abs(cross) < NOISE_RTOL),
        "generic_explicit_reference_SNR_conversion": bool(np.isclose(conversion, 0.25, rtol=0, atol=POWER_ATOL)),
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "fixture_kind": "synthetic noise draws only; no channel, detector BER or scientific experiment",
        "equation": "n=sqrt(sigma2/2)*(a+1j*b), independent a,b~N(0,1); E|n|^2=sigma2; ERe(n)^2=EIm(n)^2=sigma2/2",
        "seed": NOISE_SEED, "shape": list(noise.shape), "sigma2": sigma2,
        "empirical_complex_second_moment_by_RX": complex_power.tolist(),
        "empirical_real_second_moment_by_RX": real_power.tolist(),
        "empirical_imag_second_moment_by_RX": imag_power.tolist(),
        "normalized_mean_real_imag": _complex_pairs(means), "normalized_pseudo_covariance_real_imag": _complex_pairs(pseudo),
        "normalized_cross_RX_covariance_real_imag": _complex_pairs(cross),
        "max_abs_exact_seeded_construction_error": seeded_error,
        "moment_relative_tolerance": NOISE_RTOL,
        "generic_SNR_example": {"snr_db": snr_db, "explicit_synthetic_reference_power": snr_reference, "noise_variance": conversion},
    }


def run_sanity(project_root: str | Path) -> dict:
    apis = load_apis(project_root)
    root = apis["root"]
    source_files = [f"src/{name}.py" for name in ("config", "modulation", "numerics", "detectors", "link", "pilots", "lmmse_train_validation")]
    source_files += ["src/channels.py", "src/estimators.py", "src/interpolation.py", "configs/base.yaml"]
    before = {name: _hash(root / name) for name in source_files}
    components = {}
    for name, check in (("current_system", system_check), ("analytic_noiseless_chain", analytic_chain_check),
                        ("data_mask_wrapper", data_wrapper_check), ("conditioning_behavior", near_singular_check),
                        ("synthetic_noise_convention", noise_convention_check)):
        try:
            components[name] = check(apis)
        except Exception as exc:
            components[name] = {"passed": False, "exception_type": type(exc).__name__, "exception": str(exc)}
    after = {name: _hash(root / name) for name in source_files}
    preserved = before == after
    failed = [name for name, component in components.items() if not component["passed"]]
    if not preserved:
        failed.append("existing_source_preservation")
    return {
        "task": "SP-U17/MR-01", "scope": "minimal synthetic receiver sanity only", "project_root": str(root),
        "status": "PASS" if not failed else "STOP", "stop_required": bool(failed), "failed_checks": failed,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "executable": sys.executable, "dont_write_bytecode": sys.dont_write_bytecode},
        "api_equation_mapping": [
            {"api": "src.modulation.qpsk_modulate(bits[1D])", "equation": "S=((1-2*b0)+j*(1-2*b1))/sqrt(2); 00,01,11,10 -> I,IV,III,II"},
            {"api": "src.numerics.scale_data_streams(symbols[RE,TX],nt=2,total_power=1)", "equation": "X=S/sqrt(2); |X_q|^2=1/2; sum_q |X_q|^2=1"},
            {"api": "src.link.apply_frequency_mimo_channel(H[RX,TX,K,T],X[TX,K,T])", "equation": "Y[r,k,t]=sum_q H[r,q,k,t]*X[q,k,t]; no H transpose/conjugate"},
            {"api": "src.detectors.zf_detect(Y[RX,K,T],H[RX,TX,K,T])", "equation": "solve H[:,:,k,t]*X_hat[:,k,t]=Y[:,k,t]; returns actual scaled X_hat"},
            {"api": "src.link.detect_data_zf(Y,H,data_mask[TX,K,T])", "equation": "same solve at common data REs; nondata output NaN+jNaN"},
            {"api": "src.lmmse_train_validation.batched_data_zf(Y,H,data_mask)", "equation": "batch np.linalg.solve over common data REs; same scaled X and nondata NaNs as individual-RE zf_detect"},
            {"api": "src.link.demodulate_data_bits(sqrt(2)*X_hat,data_mask)", "equation": "restore S_hat=sqrt(2)*X_hat; b0=Re(S_hat)<0,b1=Im(S_hat)<0; TX then K then T then dibit, T varies fastest"},
            {"api": "src.numerics.complex_gaussian_noise(shape,sigma2,rng)", "equation": "sqrt(sigma2/2)*(a+j*b); E|n|^2=sigma2 and each real component variance sigma2/2"},
        ],
        "components": components, "source_sha256_before": before, "source_sha256_after": after,
        "existing_source_preserved": preserved,
        "limits": "Analytic correctness and numeric conventions only. No historical BER regeneration, CDL generation, scientific channel sampling, estimator training or algorithm change.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True, help="Existing project root containing src and configs")
    parser.add_argument("--output-dir", type=Path, required=True, help="New isolated output directory for receiver_sanity.json")
    args = parser.parse_args(argv)
    report = run_sanity(args.project_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "receiver_sanity.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "stop_required": report["stop_required"],
                      "failed_checks": report["failed_checks"], "report": str(output_path)}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
