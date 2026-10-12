"""Validate T05B time interpolation and the complete noiseless LS-LIN chain."""

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.channels import static_mimo_receive  # noqa: E402
from src.config import load_config, validate_config  # noqa: E402
from src.estimators import (  # noqa: E402
    ls_linear_interpolate,
    pilot_ls_estimate,
    time_linear_interpolate_frequency_estimates,
)
from src.pilots import build_fdm_pilot_layout, map_pilot_symbols  # noqa: E402


def _pair(value: complex) -> list[float]:
    """Store a complex scalar as a JSON-compatible real/imaginary pair."""
    number = complex(value)
    return [float(number.real), float(number.imag)]


def _relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))


def _check_config(config: dict) -> tuple[int, int, int, int, int, list[int]]:
    validate_config(config)
    system = config["system"]
    pilot = config["pilot"]
    required = {"nt": 2, "nr": 2, "nfft": 64, "active_carriers": 64, "num_symbols": 16}
    for key, expected in required.items():
        if system[key] != expected:
            raise ValueError(f"T05B requires {key}={expected}; got {system[key]}")
    if pilot["spacing_time"] != 4:
        raise ValueError("T05B requires spacing_time=4 in base.yaml")
    options = pilot["spacing_frequency_options"]
    if options != [2, 4, 8, 16]:
        raise ValueError("T05B requires spacing_frequency_options=[2, 4, 8, 16]")
    return (
        system["nt"],
        system["nr"],
        system["active_carriers"],
        system["num_symbols"],
        pilot["spacing_time"],
        options,
    )


def _time_oracles(k_count: int, t_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent complex truth: frequency-varying constant and time-affine grids."""
    intercept = np.array(
        [[0.2 + 0.1j, 12 + 23j], [-8 + 4j, 24 - 15j]], dtype=np.complex128
    )
    frequency_slope = np.array(
        [[0.03 - 0.01j, 0.17 + 0.06j], [-0.11 + 0.13j, 0.23 - 0.09j]],
        dtype=np.complex128,
    )
    time_slope = np.array(
        [[0.4 - 0.2j, -1.3 + 0.8j], [2.1 + 0.5j, -0.7 - 1.2j]],
        dtype=np.complex128,
    )
    time_frequency_slope = np.array(
        [[0.002 + 0.003j, 0.011 - 0.007j], [-0.013 + 0.009j, 0.017 + 0.005j]],
        dtype=np.complex128,
    )
    frequencies = np.arange(k_count, dtype=np.float64)[None, None, :, None]
    times = np.arange(t_count, dtype=np.float64)[None, None, None, :]
    constant_by_frequency = (
        intercept[:, :, None, None] + frequency_slope[:, :, None, None] * frequencies
    )
    constant_truth = np.broadcast_to(
        constant_by_frequency, (2, 2, k_count, t_count)
    ).copy()
    affine_truth = constant_by_frequency + (
        time_slope[:, :, None, None]
        + time_frequency_slope[:, :, None, None] * frequencies
    ) * times
    return constant_truth, affine_truth


def _anchor_only(truth: np.ndarray, pilot_times: np.ndarray) -> np.ndarray:
    """Hide all nonanchor times before calling the estimator."""
    observed = np.full(truth.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    observed[..., pilot_times] = truth[..., pilot_times]
    return observed


def _constant_ls_linear_case(
    nt: int, nr: int, k_count: int, t_count: int, d_f: int, d_t: int
) -> tuple[float, bool]:
    """Run actual T04B Pilot-LS, T05A frequency, and T05B time interfaces."""
    layout = build_fdm_pilot_layout(nt, k_count, t_count, d_f, d_t)
    pilot_symbols = np.array(
        [(1 + 1j) / np.sqrt(2), (1 - 1j) / np.sqrt(2)], dtype=np.complex128
    )
    x_grid = map_pilot_symbols(layout, pilot_symbols)
    h_true = np.array(
        [[0.8 + 0.2j, -0.3 + 0.5j], [0.4 - 0.1j, 0.6 + 0.7j]],
        dtype=np.complex128,
    )
    # The physical receive model uses H_true. The estimator sees only Y, X,
    # and the pilot mask; H_true is used below solely to score its output.
    y_grid = static_mimo_receive(h_true, x_grid)
    h_ls, observed_mask = pilot_ls_estimate(y_grid, x_grid, layout.pilot_mask)
    h_base = ls_linear_interpolate(h_ls, observed_mask)
    truth_grid = np.broadcast_to(h_true[:, :, None, None], h_base.shape)
    error = _relative_error(h_base, truth_grid)
    finite = bool(
        h_base.shape == (nr, nt, k_count, t_count)
        and np.all(np.isfinite(h_base.real))
        and np.all(np.isfinite(h_base.imag))
    )
    return error, finite


def _invalid_time_inputs_rejected(valid_input: np.ndarray, pilot_times: np.ndarray) -> dict[str, bool]:
    """Exercise every T05B required malformed-input category."""
    cases: dict[str, tuple[np.ndarray, object]] = {
        "wrong_shape": (valid_input[0], pilot_times),
        "empty_pilot_times": (valid_input, []),
        "out_of_bounds_pilot_time": (valid_input, [0, 4, 8, valid_input.shape[-1]]),
        "nonincreasing_pilot_times": (valid_input, [0, 8, 4, 12]),
    }
    missing_entire_anchor = valid_input.copy()
    missing_entire_anchor[..., 4] = np.nan + 1j * np.nan
    cases["all_nan_anchor"] = (missing_entire_anchor, pilot_times)

    infinite_anchor = valid_input.copy()
    infinite_anchor[0, 0, 10, 4] = np.inf + 0j
    cases["infinite_anchor"] = (infinite_anchor, pilot_times)

    missing_single_anchor = valid_input.copy()
    missing_single_anchor[0, 0, 10, 4] = np.nan + 1j * np.nan
    cases["unexpected_missing_anchor"] = (missing_single_anchor, pilot_times)

    unexpected_nonanchor = valid_input.copy()
    unexpected_nonanchor[0, 0, 10, 1] = 3 + 2j
    cases["unexpected_nonanchor_value"] = (unexpected_nonanchor, pilot_times)

    rejected = {}
    for name, (values, indices) in cases.items():
        try:
            time_linear_interpolate_frequency_estimates(values, indices)
        except ValueError:
            rejected[name] = True
        else:
            rejected[name] = False
    return rejected


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    nt, nr, k_count, t_count, d_t, d_f_options = _check_config(config)
    d_f = 4
    layout = build_fdm_pilot_layout(nt, k_count, t_count, d_f, d_t)
    pilot_times = np.array(layout.pilot_symbol_indices, dtype=np.int64)

    constant_truth, affine_truth = _time_oracles(k_count, t_count)
    constant_input = _anchor_only(constant_truth, pilot_times)
    affine_input = _anchor_only(affine_truth, pilot_times)
    constant_full = time_linear_interpolate_frequency_estimates(
        constant_input, pilot_times
    )
    affine_full = time_linear_interpolate_frequency_estimates(affine_input, pilot_times)

    constant_error = _relative_error(constant_full, constant_truth)
    interior_times = np.arange(0, pilot_times[-1] + 1)
    affine_interior_error = _relative_error(
        affine_full[..., interior_times], affine_truth[..., interior_times]
    )

    r, q, k = 0, 0, 10
    a = affine_input[r, q, k, 0]
    b = affine_input[r, q, k, 4]
    expected_t1 = 0.75 * a + 0.25 * b
    expected_t2 = 0.5 * a + 0.5 * b
    expected_t3 = 0.25 * a + 0.75 * b
    manual = {
        "coordinate_r_q_k": [r, q, k],
        "t0": _pair(a),
        "t4": _pair(b),
        "t1": _pair(affine_full[r, q, k, 1]),
        "expected_t1": _pair(expected_t1),
        "t2": _pair(affine_full[r, q, k, 2]),
        "expected_midpoint": _pair(expected_t2),
        "t3": _pair(affine_full[r, q, k, 3]),
        "expected_t3": _pair(expected_t3),
    }
    manual_passed = bool(
        np.allclose(
            affine_full[r, q, k, 1:4],
            [expected_t1, expected_t2, expected_t3],
            atol=1e-12,
            rtol=0,
        )
    )

    last_anchor = int(pilot_times[-1])
    right_boundary = {
        str(t): _pair(affine_full[r, q, k, t])
        for t in range(last_anchor, t_count)
    }
    edge_hold_passed = bool(
        np.array_equal(
            affine_full[..., last_anchor + 1 :],
            np.broadcast_to(
                affine_full[..., last_anchor, None],
                affine_full[..., last_anchor + 1 :].shape,
            ),
        )
    )
    anchor_max_error = float(
        np.max(np.abs(affine_full[..., pilot_times] - affine_input[..., pilot_times]))
    )
    anchors_unchanged = bool(
        np.array_equal(affine_full[..., pilot_times], affine_input[..., pilot_times])
    )

    finite_mask = np.isfinite(affine_full.real) & np.isfinite(affine_full.imag)
    finite_count = int(np.count_nonzero(finite_mask))
    total_count = int(affine_full.size)
    all_finite = bool(finite_count == total_count == nr * nt * k_count * t_count)

    sentinel_value = affine_full[r, q, k, 2]
    wrong_axis_values = [
        affine_full[1, 0, k, 2],
        affine_full[0, 1, k, 2],
        affine_full[0, 0, k + 1, 2],
    ]
    wrong_axis_min_difference = float(
        min(abs(sentinel_value - wrong_value) for wrong_value in wrong_axis_values)
    )
    axis_separation_passed = bool(
        np.allclose(affine_full[..., interior_times], affine_truth[..., interior_times], atol=1e-12, rtol=0)
        and wrong_axis_min_difference > 1e-2
    )

    four_df_errors = {}
    four_df_checks = {}
    for option in d_f_options:
        error, finite = _constant_ls_linear_case(
            nt, nr, k_count, t_count, option, d_t
        )
        four_df_errors[str(option)] = error
        four_df_checks[str(option)] = bool(error <= 1e-12 and finite)
    full_ls_linear_error = four_df_errors[str(d_f)]
    invalid_checks = _invalid_time_inputs_rejected(affine_input, pilot_times)

    checks = {
        "constant_time": bool(constant_error <= 1e-12),
        "affine_time_interior": bool(affine_interior_error <= 1e-12),
        "manual_time_weights": manual_passed,
        "time_edge_hold": edge_hold_passed,
        "anchor_preservation": bool(anchors_unchanged and anchor_max_error == 0),
        "axis_separation": axis_separation_passed,
        "full_grid_finite": all_finite,
        "full_ls_linear_constant_channel": bool(full_ls_linear_error <= 1e-12),
        "four_df_compatibility": bool(all(four_df_checks.values())),
        "invalid_input_handling": bool(all(invalid_checks.values())),
    }
    all_checks_passed = bool(all(checks.values()))
    report = {
        "task": "T05B",
        "nt": nt,
        "nr": nr,
        "K": k_count,
        "T": t_count,
        "dF": d_f,
        "dT": d_t,
        "pilot_symbol_indices": pilot_times.tolist(),
        "constant_time_relative_error": constant_error,
        "affine_time_interior_relative_error": affine_interior_error,
        "manual_time_interpolation_values": manual,
        "right_time_boundary_values": right_boundary,
        "anchor_preservation_max_error": anchor_max_error,
        "finite_estimate_count": finite_count,
        "total_estimate_count": total_count,
        "all_estimates_finite": all_finite,
        "axis_separation_wrong_mix_min_difference": wrong_axis_min_difference,
        "axis_separation_sentinel_passed": axis_separation_passed,
        "full_ls_linear_constant_relative_error": full_ls_linear_error,
        "four_df_full_ls_linear_relative_errors": four_df_errors,
        "four_df_full_ls_linear_checks": four_df_checks,
        "invalid_input_checks": invalid_checks,
        "checks": checks,
        "all_checks_passed": all_checks_passed,
    }

    print("T05B time interpolation validation")
    print(f"Nt: {nt}")
    print(f"Nr: {nr}")
    print(f"K: {k_count}")
    print(f"T: {t_count}")
    print(f"pilot symbols: {pilot_times.tolist()}")
    print(f"Constant-time relative error: {constant_error:.17g}")
    print(f"Affine-time interior relative error: {affine_interior_error:.17g}")
    print("Manual coordinate:")
    print(f"(r,q,k): {(r, q, k)}")
    print(f"H(t=0): {a}")
    print(f"H(t=4): {b}")
    print(f"H(t=1): {affine_full[r, q, k, 1]}")
    print(f"expected t=1: {expected_t1}")
    print(f"H(t=2): {affine_full[r, q, k, 2]}")
    print(f"expected midpoint: {expected_t2}")
    print(f"H(t=3): {affine_full[r, q, k, 3]}")
    print(f"expected t=3: {expected_t3}")
    print("Right edge:")
    for t in range(last_anchor, t_count):
        print(f"H(t={t}): {affine_full[r, q, k, t]}")
    print(f"Anchor preservation max error: {anchor_max_error:.17g}")
    print(f"Finite estimates: {finite_count} / {total_count}")
    print(f"Axis separation sentinel: {'PASS' if axis_separation_passed else 'FAIL'}")
    print(f"Full LS-LIN constant-channel relative error: {full_ls_linear_error:.17g}")
    print(f"All four dF full LS-LIN: {'PASS' if all(four_df_checks.values()) else 'FAIL'}")
    print(f"Invalid inputs rejected: {'PASS' if all(invalid_checks.values()) else 'FAIL'}")
    print(f"All checks: {'PASS' if all_checks_passed else 'FAIL'}")

    output_path = PROJECT_ROOT / "outputs" / "t05b_time_interpolation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"JSON saved: {output_path}")
    if not all_checks_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
