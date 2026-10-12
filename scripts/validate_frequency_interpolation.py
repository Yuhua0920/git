"""Validate frequency-only interpolation of Pilot-LS observations for T05A."""

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.channels import static_mimo_receive  # noqa: E402
from src.config import load_config, validate_config  # noqa: E402
from src.estimators import frequency_linear_interpolate_ls, pilot_ls_estimate  # noqa: E402
from src.pilots import build_fdm_pilot_layout, map_pilot_symbols  # noqa: E402


def _pair(value: complex) -> list[float]:
    """Store a complex scalar as a JSON-compatible [real, imaginary] pair."""
    number = complex(value)
    return [float(number.real), float(number.imag)]


def _check_config(config: dict) -> tuple[int, int, int, int, int, list[int]]:
    validate_config(config)
    system = config["system"]
    pilot = config["pilot"]
    required = {"nt": 2, "nr": 2, "nfft": 64, "active_carriers": 64, "num_symbols": 16}
    for key, expected in required.items():
        if system[key] != expected:
            raise ValueError(f"T05A requires {key}={expected}; got {system[key]}")
    if pilot["spacing_time"] != 4:
        raise ValueError("T05A requires spacing_time=4 in base.yaml")
    options = pilot["spacing_frequency_options"]
    if options != [2, 4, 8, 16]:
        raise ValueError("T05A requires spacing_frequency_options=[2, 4, 8, 16]")
    return (
        system["nt"],
        system["nr"],
        system["active_carriers"],
        system["num_symbols"],
        pilot["spacing_time"],
        options,
    )


def _pilot_ls_for_constant(layout, h_constant: np.ndarray) -> np.ndarray:
    """Use the T04B pilot mapping, receive model, and LS estimator."""
    pilots = np.array(
        [(1 + 1j) / np.sqrt(2), (1 - 1j) / np.sqrt(2)], dtype=np.complex128
    )
    x_grid = map_pilot_symbols(layout, pilots)
    y_grid = static_mimo_receive(h_constant, x_grid)
    h_ls, estimated_mask = pilot_ls_estimate(y_grid, x_grid, layout.pilot_mask)
    if not np.array_equal(estimated_mask, layout.pilot_mask):
        raise RuntimeError("T04B Pilot-LS returned an unexpected observation mask")
    return h_ls


def _affine_truth(nr: int, nt: int, k_count: int, t_count: int) -> np.ndarray:
    """Build an independent oracle with distinct complex trends for each link."""
    if (nr, nt) != (2, 2):
        raise ValueError("T05A affine oracle requires exactly two RX and two TX antennas")
    intercepts = np.array(
        [[0.2 + 0.1j, 20 + 30j], [-0.4 + 0.5j, -18 + 24j]],
        dtype=np.complex128,
    )
    slopes = np.array(
        [[0.01 - 0.003j, -0.4 + 0.3j], [-0.015 + 0.008j, 0.28 - 0.35j]],
        dtype=np.complex128,
    )
    link_factors = np.array([[1, 2], [3, 4]], dtype=np.float64)
    time_index = np.arange(t_count, dtype=np.float64) / 4.0
    a_by_time = intercepts[:, :, None] + (
        (0.02 + 0.01j) * link_factors[:, :, None] * time_index[None, None, :]
    )
    b_by_time = slopes[:, :, None] + (
        (0.001 - 0.0002j) * link_factors[:, :, None] * time_index[None, None, :]
    )
    frequency_index = np.arange(k_count, dtype=np.float64)
    return a_by_time[:, :, None, :] + b_by_time[:, :, None, :] * frequency_index[None, None, :, None]


def _invalid_inputs_rejected(h_ls: np.ndarray, pilot_mask: np.ndarray) -> bool:
    """Exercise representative shape, dtype, missing-pilot, NaN, and Inf errors."""
    cases = []
    cases.append((h_ls[:, :, :-1, :], pilot_mask, "shape"))
    cases.append((h_ls.real, pilot_mask, "complex"))

    missing_tx = pilot_mask.copy()
    missing_tx[1, :, 0] = False
    cases.append((h_ls, missing_tx, "no pilots"))

    nan_pilot = h_ls.copy()
    nan_pilot[0, 0, 0, 0] = np.nan + 1j * np.nan
    cases.append((nan_pilot, pilot_mask, "finite"))

    infinite = h_ls.copy()
    infinite[0, 0, 1, 1] = np.inf + 0j
    cases.append((infinite, pilot_mask, "Inf"))

    for observations, mask, expected_message in cases:
        try:
            frequency_linear_interpolate_ls(observations, mask)
        except ValueError as exc:
            if expected_message not in str(exc):
                return False
        else:
            return False
    return True


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    nt, nr, k_count, t_count, d_t, d_f_options = _check_config(config)
    d_f = 4
    layout = build_fdm_pilot_layout(nt, k_count, t_count, d_f, d_t)
    pilot_times = np.array(layout.pilot_symbol_indices, dtype=np.int64)
    pilot_time_mask = np.zeros(t_count, dtype=bool)
    pilot_time_mask[pilot_times] = True
    expected_output_mask = np.broadcast_to(
        pilot_time_mask[None, None, :], (nt, k_count, t_count)
    )
    expected_output_mask_4d = np.broadcast_to(
        expected_output_mask[None, :, :, :], (nr, nt, k_count, t_count)
    )

    # The interpolation function sees only these LS observations and the mask.
    h_constant = np.array(
        [[0.8 + 0.2j, -0.3 + 0.5j], [0.4 - 0.1j, 0.6 + 0.7j]],
        dtype=np.complex128,
    )
    constant_ls = _pilot_ls_for_constant(layout, h_constant)
    constant_freq, constant_mask = frequency_linear_interpolate_ls(
        constant_ls, layout.pilot_mask
    )
    constant_truth = np.broadcast_to(h_constant[:, :, None, None], constant_freq.shape)
    constant_relative_error = float(
        np.linalg.norm(constant_freq[expected_output_mask_4d] - constant_truth[expected_output_mask_4d])
        / np.linalg.norm(constant_truth[expected_output_mask_4d])
    )

    affine_truth = _affine_truth(nr, nt, k_count, t_count)
    affine_ls = np.full(
        (nr, nt, k_count, t_count), np.nan + 1j * np.nan, dtype=np.complex128
    )
    observed_mask_4d = np.broadcast_to(layout.pilot_mask[None, :, :, :], affine_ls.shape)
    affine_ls[observed_mask_4d] = affine_truth[observed_mask_4d]
    affine_freq, affine_mask = frequency_linear_interpolate_ls(affine_ls, layout.pilot_mask)

    valid_ranges = [[0, 60], [2, 62]]
    interior_mask = np.zeros((nt, k_count, t_count), dtype=bool)
    for q, (first, last) in enumerate(valid_ranges):
        for t in pilot_times:
            interior_mask[q, first : last + 1, t] = True
    interior_mask_4d = np.broadcast_to(interior_mask[None, :, :, :], affine_freq.shape)
    interior_relative_error = float(
        np.linalg.norm(affine_freq[interior_mask_4d] - affine_truth[interior_mask_4d])
        / np.linalg.norm(affine_truth[interior_mask_4d])
    )

    # Inspect RX0, TX0, t=0 numerically; assertions below cover both RXs and all pilot times.
    h0 = affine_freq[0, 0, 0, 0]
    h4 = affine_freq[0, 0, 4, 0]
    h1 = affine_freq[0, 0, 1, 0]
    h2 = affine_freq[0, 0, 2, 0]
    midpoint = 0.5 * (h0 + h4)
    quarterpoint = 0.75 * h0 + 0.25 * h4
    manual_midpoint_values = {
        "receiver": 0,
        "transmitter": 0,
        "symbol": 0,
        "k0": _pair(h0),
        "k4": _pair(h4),
        "k1": _pair(h1),
        "expected_k1": _pair(quarterpoint),
        "k2": _pair(h2),
        "expected_k2": _pair(midpoint),
    }

    tx0_right = {str(k): _pair(affine_freq[0, 0, k, 0]) for k in range(60, 64)}
    tx1_left = {str(k): _pair(affine_freq[0, 1, k, 0]) for k in range(0, 3)}
    tx1_right = {str(k): _pair(affine_freq[0, 1, k, 0]) for k in (62, 63)}
    edge_hold_passed = True
    for t in pilot_times:
        edge_hold_passed &= bool(
            np.allclose(
                affine_freq[:, 0, 61:64, t], affine_ls[:, 0, 60, t][:, None],
                rtol=0.0, atol=1e-12,
            )
            and np.allclose(
                affine_freq[:, 1, 0:2, t], affine_ls[:, 1, 2, t][:, None],
                rtol=0.0, atol=1e-12,
            )
            and np.allclose(
                affine_freq[:, 1, 63, t], affine_ls[:, 1, 62, t],
                rtol=0.0, atol=1e-12,
            )
        )

    pilot_preservation_max_error = float(
        max(
            np.max(np.abs(constant_freq[observed_mask_4d] - constant_ls[observed_mask_4d])),
            np.max(np.abs(affine_freq[observed_mask_4d] - affine_ls[observed_mask_4d])),
        )
    )
    nonpilot_time_nan_passed = bool(
        np.array_equal(constant_mask, expected_output_mask)
        and np.array_equal(affine_mask, expected_output_mask)
        and np.all(np.isfinite(constant_freq[expected_output_mask_4d]))
        and np.all(np.isfinite(affine_freq[expected_output_mask_4d]))
        and np.all(np.isnan(constant_freq.real[~expected_output_mask_4d]))
        and np.all(np.isnan(constant_freq.imag[~expected_output_mask_4d]))
        and np.all(np.isnan(affine_freq.real[~expected_output_mask_4d]))
        and np.all(np.isnan(affine_freq.imag[~expected_output_mask_4d]))
    )

    # A wrongly merged TX sequence would replace TX0(k=2) with TX1's pilot
    # and TX1(k=4) with TX0's pilot. Both wrong values differ substantially.
    wrong_tx0_at_k2 = affine_ls[0, 1, 2, 0]
    wrong_tx1_at_k4 = affine_ls[0, 0, 4, 0]
    wrong_mix_min_error = float(
        min(
            abs(wrong_tx0_at_k2 - affine_truth[0, 0, 2, 0]),
            abs(wrong_tx1_at_k4 - affine_truth[0, 1, 4, 0]),
        )
    )
    tx_separation_sentinel_passed = bool(
        np.isclose(affine_freq[0, 0, 2, 0], affine_truth[0, 0, 2, 0], rtol=0.0, atol=1e-12)
        and np.isclose(affine_freq[0, 1, 4, 0], affine_truth[0, 1, 4, 0], rtol=0.0, atol=1e-12)
        and wrong_mix_min_error > 1.0
    )

    four_df_checks = {}
    for spacing in d_f_options:
        candidate = build_fdm_pilot_layout(nt, k_count, t_count, spacing, d_t)
        candidate_ls = _pilot_ls_for_constant(candidate, h_constant)
        candidate_freq, candidate_mask = frequency_linear_interpolate_ls(
            candidate_ls, candidate.pilot_mask
        )
        candidate_observed = np.broadcast_to(candidate.pilot_mask[None, :, :, :], candidate_ls.shape)
        four_df_checks[str(spacing)] = bool(
            candidate_freq.shape == (nr, nt, k_count, t_count)
            and candidate_mask.shape == (nt, k_count, t_count)
            and np.array_equal(candidate_mask, expected_output_mask)
            and np.array_equal(candidate_freq[candidate_observed], candidate_ls[candidate_observed])
            and np.all(np.isnan(candidate_freq.real[~expected_output_mask_4d]))
            and np.all(np.isnan(candidate_freq.imag[~expected_output_mask_4d]))
        )
    all_four_df_supported = bool(all(four_df_checks.values()))
    invalid_input_handling_passed = _invalid_inputs_rejected(affine_ls, layout.pilot_mask)

    checks = {
        "constant_channel": bool(constant_relative_error <= 1e-12),
        "complex_affine_interior": bool(interior_relative_error <= 1e-12),
        "manual_midpoint": bool(
            np.isclose(h2, midpoint, rtol=0.0, atol=1e-12)
            and np.isclose(h1, quarterpoint, rtol=0.0, atol=1e-12)
        ),
        "edge_hold": bool(edge_hold_passed),
        "pilot_preservation": bool(pilot_preservation_max_error <= 1e-12),
        "nonpilot_time_missing": nonpilot_time_nan_passed,
        "tx_separation": tx_separation_sentinel_passed,
        "four_df_layouts": all_four_df_supported,
        "invalid_input_handling": invalid_input_handling_passed,
    }
    all_checks_passed = bool(all(checks.values()))

    report = {
        "task": "T05A",
        "nt": nt,
        "nr": nr,
        "K": k_count,
        "T": t_count,
        "dF": d_f,
        "dT": d_t,
        "pilot_symbol_indices": pilot_times.astype(int).tolist(),
        "constant_channel_relative_error": constant_relative_error,
        "affine_tx0_valid_range": valid_ranges[0],
        "affine_tx1_valid_range": valid_ranges[1],
        "affine_interior_relative_error": interior_relative_error,
        "manual_midpoint_values": manual_midpoint_values,
        "tx0_right_boundary_values": tx0_right,
        "tx1_left_boundary_values": tx1_left,
        "tx1_right_boundary_values": tx1_right,
        "pilot_preservation_max_error": pilot_preservation_max_error,
        "nonpilot_time_nan_passed": nonpilot_time_nan_passed,
        "tx_separation_wrong_mix_min_error": wrong_mix_min_error,
        "tx_separation_sentinel_passed": tx_separation_sentinel_passed,
        "four_df_checks": four_df_checks,
        "all_four_df_supported": all_four_df_supported,
        "invalid_input_handling_passed": invalid_input_handling_passed,
        "checks": checks,
        "all_checks_passed": all_checks_passed,
    }

    print("T05A frequency interpolation validation")
    print(f"Nt: {nt}")
    print(f"Nr: {nr}")
    print(f"K: {k_count}")
    print(f"T: {t_count}")
    print(f"dF: {d_f}")
    print(f"pilot symbols: {pilot_times.tolist()}")
    print(f"Constant-channel relative error: {constant_relative_error:.17g}")
    print("Affine test:")
    print(f"TX0 valid interpolation range: {valid_ranges[0]}")
    print(f"TX1 valid interpolation range: {valid_ranges[1]}")
    print(f"Interior relative error: {interior_relative_error:.17g}")
    print("Manual TX0:")
    print(f"H[k=0]: {h0}")
    print(f"H[k=4]: {h4}")
    print(f"Interpolated H[k=1]: {h1}")
    print(f"Expected quarterpoint: {quarterpoint}")
    print(f"Interpolated H[k=2]: {h2}")
    print(f"Expected midpoint: {midpoint}")
    print("TX0 right edge:")
    for k in range(60, 64):
        print(f"k={k}: {affine_freq[0, 0, k, 0]}")
    print("TX1 left edge:")
    for k in range(0, 3):
        print(f"k={k}: {affine_freq[0, 1, k, 0]}")
    print("TX1 right edge:")
    for k in (62, 63):
        print(f"k={k}: {affine_freq[0, 1, k, 0]}")
    print(f"Pilot preservation max error: {pilot_preservation_max_error:.17g}")
    print(f"Nonpilot time remains NaN: {'PASS' if nonpilot_time_nan_passed else 'FAIL'}")
    print(f"TX separation sentinel: {'PASS' if tx_separation_sentinel_passed else 'FAIL'}")
    print(f"All four dF supported: {'PASS' if all_four_df_supported else 'FAIL'}")
    print(f"Invalid inputs rejected: {'PASS' if invalid_input_handling_passed else 'FAIL'}")
    print(f"All checks: {'PASS' if all_checks_passed else 'FAIL'}")

    output_path = PROJECT_ROOT / "outputs" / "t05a_frequency_interpolation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"JSON saved: {output_path}")
    if not all_checks_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
