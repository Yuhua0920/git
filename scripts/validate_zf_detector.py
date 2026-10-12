"""T06A Perfect-CSI, noiseless 2x2 spatial-stream detection calibration."""

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.channels import static_mimo_receive  # noqa: E402
from src.config import load_config, validate_config  # noqa: E402
from src.detectors import zf_detect  # noqa: E402
from src.modulation import qpsk_demodulate, qpsk_modulate  # noqa: E402
from src.numerics import scale_data_streams  # noqa: E402


def _pairs(values: np.ndarray) -> list:
    values = np.asarray(values, dtype=np.complex128)
    return np.stack((values.real, values.imag), axis=-1).tolist()


def _relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    validate_config(config)
    system = config["system"]
    for key, expected in {"nt": 2, "nr": 2, "active_carriers": 64, "num_symbols": 16,
                          "total_power": 1.0, "modulation": "qpsk_gray"}.items():
        if system[key] != expected:
            raise ValueError(f"T06A requires {key}={expected}; got {system[key]}")
    nt, nr = system["nt"], system["nr"]
    K, T = system["active_carriers"], system["num_symbols"]
    bits_seeds = [config["seed"]["base_seed"], config["seed"]["base_seed"] + 1]
    bits = [np.random.default_rng(seed).integers(0, 2, 2 * K * T, dtype=np.uint8)
            for seed in bits_seeds]
    s_grid = np.stack([qpsk_modulate(stream).reshape(K, T) for stream in bits])
    # T01 scaling uses [RE, TX]; reshape back to the unchanged [TX,K,T] order.
    x_grid = scale_data_streams(
        s_grid.reshape(nt, -1).T, nt, system["total_power"]
    ).T.reshape(nt, K, T)
    h = np.array([[0.8 + 0.2j, -0.3 + 0.5j], [0.4 - 0.1j, 0.6 + 0.7j]], dtype=np.complex128)
    # H[r,q] maps TX q to RX r. Perfect CSI is explicitly used in this task.
    h_grid = np.broadcast_to(h[:, :, None, None], (nr, nt, K, T))
    y_grid = static_mimo_receive(h, x_grid)
    x_hat = zf_detect(y_grid, h_grid)
    s_hat = np.sqrt(nt) * x_hat
    relative_error = _relative_error(x_hat, x_grid)
    symbol_power = np.mean(np.abs(s_grid) ** 2, axis=(1, 2)).tolist()
    scaled_power = np.mean(np.abs(x_grid) ** 2, axis=(1, 2)).tolist()
    total_power = float(np.mean(np.sum(np.abs(x_grid) ** 2, axis=0)))
    bit_errors = [int(np.count_nonzero(qpsk_demodulate(s_hat[q].ravel()) != bits[q]))
                  for q in range(nt)]
    bit_counts = [int(stream.size) for stream in bits]

    manual_s = np.array([qpsk_modulate(np.array([0, 0]))[0],
                         qpsk_modulate(np.array([1, 0]))[0]])
    manual_x = scale_data_streams(manual_s[None, :], nt, system["total_power"])[0]
    manual_y = static_mimo_receive(h, manual_x[:, None, None])[:, 0, 0]
    manual_x_hat = zf_detect(manual_y[:, None, None], h[:, :, None, None])[:, 0, 0]
    manual_s_hat = np.sqrt(nt) * manual_x_hat

    direct_mismatch = _relative_error(y_grid, x_grid)
    direct_bit_errors = [int(np.count_nonzero(qpsk_demodulate(y_grid[q].ravel()) != bits[q]))
                         for q in range(nt)]
    h_b = np.array([[1.1 - 0.3j, 0.2 + 0.4j], [-0.5 + 0.1j, 0.9 + 0.2j]])
    varying_h = np.empty((nr, nt, K, T), dtype=np.complex128)
    for k in range(K):
        for t in range(T):
            varying_h[:, :, k, t] = h if (k + t) % 2 == 0 else h_b
    varying_y = np.einsum("rqkt,qkt->rkt", varying_h, x_grid)
    varying_error = _relative_error(zf_detect(varying_y, varying_h), x_grid)
    wrong_static_error = _relative_error(zf_detect(varying_y, h_grid), x_grid)

    singular = np.array([[1, 2], [2, 4]], dtype=np.complex128)
    singular_rejected = False
    singular_message = ""
    try:
        zf_detect((singular @ manual_x)[:, None, None], singular[:, :, None, None])
    except ValueError as exc:
        singular_message = str(exc)
        singular_rejected = "singular" in singular_message

    # A high condition number is recorded without imposing a rank cutoff.
    ill_h = np.array([[1, 1], [1, 1 + 1e-8]], dtype=np.complex128)
    ill_x_hat = zf_detect((ill_h @ manual_x)[:, None, None], ill_h[:, :, None, None])[:, 0, 0]
    ill_condition_number = float(np.linalg.cond(ill_h))
    ill_error = _relative_error(ill_x_hat, manual_x)

    invalid_cases = {
        "y_rank": (y_grid[0], h_grid),
        "h_rank": (y_grid, h_grid[:, :, :, 0]),
        "axis_mismatch": (y_grid[:, :-1], h_grid),
        "nonsquare": (y_grid, h_grid[:, :1]),
    }
    nonfinite_y, nonfinite_h = y_grid.copy(), h_grid.copy()
    nonfinite_y[0, 0, 0] = np.nan
    nonfinite_h[0, 0, 0, 0] = np.inf
    invalid_cases["nan_y"] = (nonfinite_y, h_grid)
    invalid_cases["inf_h"] = (y_grid, nonfinite_h)
    invalid_checks = {}
    for name, (received, channel) in invalid_cases.items():
        try:
            zf_detect(received, channel)
        except ValueError:
            invalid_checks[name] = True
        else:
            invalid_checks[name] = False

    determinant = np.linalg.det(h)
    condition_number = float(np.linalg.cond(h))
    checks = {
        "nondiagonal_full_rank_channel": bool(h[0, 1] != 0 and h[1, 0] != 0 and np.linalg.matrix_rank(h) == nt),
        "unit_qpsk_symbol_power": bool(np.allclose(symbol_power, 1, rtol=0, atol=1e-12)),
        "two_stream_power_scaling": bool(np.allclose(scaled_power, 0.5, rtol=0, atol=1e-12)
                                         and abs(total_power - 1) <= 1e-12),
        "manual_re_solve": bool(np.allclose(manual_x_hat, manual_x, rtol=0, atol=1e-12)
                                and np.allclose(manual_s_hat, manual_s, rtol=0, atol=1e-12)),
        "full_grid_zf": bool(x_hat.shape == (nt, K, T) and relative_error <= 1e-12),
        "zero_bit_errors": bool(sum(bit_errors) == 0 and sum(bit_counts) == 4096),
        "direct_y_mismatch": bool(direct_mismatch > 1000 * np.finfo(np.float64).eps),
        "per_re_h_variation": bool(varying_error <= 1e-12 and wrong_static_error > 1e-2),
        "singular_h_rejection": singular_rejected,
        "ill_conditioned_channel_runs": bool(np.all(np.isfinite(ill_x_hat))),
        "input_validation": bool(all(invalid_checks.values())),
    }
    all_checks = bool(all(checks.values()))
    report = {
        "task": "T06A", "nt": nt, "nr": nr, "K": K, "T": T,
        "bits_seed_by_tx": bits_seeds,
        "calibration_H_real_imag": _pairs(h),
        "determinant_real_imag": _pairs(determinant),
        "condition_number": condition_number,
        "tx_symbol_power_by_stream": symbol_power,
        "scaled_tx_power_by_stream": scaled_power,
        "total_tx_power": total_power,
        "manual_re_coordinate": [0, 0],
        "manual_S": _pairs(manual_s), "manual_X": _pairs(manual_x),
        "manual_Y": _pairs(manual_y), "manual_X_hat": _pairs(manual_x_hat),
        "manual_recovered_S": _pairs(manual_s_hat),
        "full_grid_zf_relative_error": relative_error,
        "bit_count_by_tx": bit_counts, "bit_errors_by_tx": bit_errors,
        "total_bits": sum(bit_counts), "total_bit_errors": sum(bit_errors),
        "direct_y_relative_mismatch": direct_mismatch,
        "direct_y_bit_errors": sum(direct_bit_errors),
        "direct_y_bit_errors_by_tx": direct_bit_errors,
        "varying_h_relative_error": varying_error,
        "varying_h_wrong_static_relative_error": wrong_static_error,
        "singular_h_rejected": singular_rejected,
        "singular_h_rejection_message": singular_message,
        "ill_conditioned_condition_number": ill_condition_number,
        "ill_conditioned_relative_error": ill_error,
        "input_validation_checks": invalid_checks,
        "checks": checks, "all_checks_passed": all_checks,
    }

    print("T06A Perfect-CSI 2x2 ZF validation")
    for name, value in (("Nt", nt), ("Nr", nr), ("K", K), ("T", T)):
        print(f"{name}: {value}")
    print("Calibration H:")
    print(h)
    print(f"det(H): {determinant}")
    print(f"condition number: {condition_number:.17g}")
    for q in range(nt):
        print(f"TX{q} QPSK symbol power: {symbol_power[q]:.17g}")
    for q in range(nt):
        print(f"TX{q} scaled transmit power: {scaled_power[q]:.17g}")
    print(f"total transmit power: {total_power:.17g}")
    print("Manual RE: (k=0,t=0), fixed mapper dibits TX0=00, TX1=10")
    for name, value in (("S", manual_s), ("X", manual_x), ("Y", manual_y),
                        ("ZF X_hat", manual_x_hat), ("Recovered S", manual_s_hat)):
        print(f"{name}: {value}")
    print(f"Full-grid ZF relative symbol error: {relative_error:.17g}")
    for q in range(nt):
        print(f"TX{q} bit errors: {bit_errors[q]} / {bit_counts[q]}")
    print(f"Total bit errors: {sum(bit_errors)} / {sum(bit_counts)}")
    print(f"Direct-Y relative mismatch: {direct_mismatch:.17g}")
    print(f"Direct-Y bit errors: {sum(direct_bit_errors)}")
    print(f"Per-RE varying-H relative error: {varying_error:.17g}")
    print(f"Ill-conditioned H condition number: {ill_condition_number:.17g}")
    print(f"Ill-conditioned H relative error: {ill_error:.17g}")
    print(f"Singular-H rejection: {'PASS' if singular_rejected else 'FAIL'}")
    print(f"All checks: {'PASS' if all_checks else 'FAIL'}")
    output_path = PROJECT_ROOT / "outputs" / "t06a_zf_detector_results.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"JSON saved: {output_path}")
    if not all_checks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
