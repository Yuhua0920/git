"""Validate noiseless pilot-only LS for the fixed T04B 2x2 MIMO example."""

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.channels import static_mimo_receive  # noqa: E402
from src.config import load_config, validate_config  # noqa: E402
from src.estimators import pilot_ls_estimate  # noqa: E402
from src.pilots import build_fdm_pilot_layout, map_pilot_symbols  # noqa: E402


def _complex_pairs(values: np.ndarray) -> list:
    """Represent complex arrays as nested [real, imaginary] JSON pairs."""
    array = np.asarray(values, dtype=np.complex128)
    return np.stack((array.real, array.imag), axis=-1).tolist()


def _check_config(config: dict) -> tuple[int, int, int, int, int, int]:
    validate_config(config)
    system = config["system"]
    pilot = config["pilot"]
    required = {"nt": 2, "nr": 2, "nfft": 64, "active_carriers": 64, "num_symbols": 16}
    for key, expected in required.items():
        if system[key] != expected:
            raise ValueError(f"T04B requires {key}={expected}; got {system[key]}")
    if pilot["spacing_time"] != 4 or 4 not in pilot["spacing_frequency_options"]:
        raise ValueError("T04B requires dF=4 and dT=4 in base.yaml")
    return (
        system["nt"],
        system["nr"],
        system["active_carriers"],
        system["num_symbols"],
        4,
        pilot["spacing_time"],
    )


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    nt, nr, k_count, t_count, d_f, d_t = _check_config(config)
    layout = build_fdm_pilot_layout(nt, k_count, t_count, d_f, d_t)

    pilots = np.array(
        [(1 + 1j) / np.sqrt(2), (1 - 1j) / np.sqrt(2)], dtype=np.complex128
    )
    x_grid = map_pilot_symbols(layout, pilots)
    # H[r, q] is the channel from TX q to RX r.
    h_true = np.array(
        [[0.8 + 0.2j, -0.3 + 0.5j], [0.4 - 0.1j, 0.6 + 0.7j]],
        dtype=np.complex128,
    )
    y_grid = static_mimo_receive(h_true, x_grid)
    h_ls, estimate_mask = pilot_ls_estimate(y_grid, x_grid, layout.pilot_mask)

    pilot_counts = np.count_nonzero(layout.pilot_mask, axis=(1, 2)).astype(int).tolist()
    scalar_count = nr * sum(pilot_counts)
    full_mask = np.broadcast_to(estimate_mask[None, :, :, :], h_ls.shape)
    true_on_grid = np.broadcast_to(h_true[:, :, None, None], h_ls.shape)
    ls_observed = h_ls[full_mask]
    truth_observed = true_on_grid[full_mask]
    relative_error = float(np.linalg.norm(ls_observed - truth_observed) / np.linalg.norm(truth_observed))
    maximum_error = float(np.max(np.abs(ls_observed - truth_observed)))

    manual = []
    for q, (k, t) in enumerate(((0, 0), (2, 0))):
        manual.append(
            {
                "coordinate": [k, t],
                "x": _complex_pairs(x_grid[:, k, t]),
                "y": _complex_pairs(y_grid[:, k, t]),
                "estimate": _complex_pairs(h_ls[:, q, k, t]),
                "truth": _complex_pairs(h_true[:, q]),
            }
        )

    contaminated_grid = x_grid.copy()
    contaminated_grid[1, 0, 0] = pilots[1]
    contaminated_y = static_mimo_receive(h_true, contaminated_grid)
    naive_estimate = contaminated_y[:, 0, 0] / pilots[0]
    contaminated_error = float(
        np.linalg.norm(naive_estimate - h_true[:, 0]) / np.linalg.norm(h_true[:, 0])
    )
    # A numerical-scale comparison, rather than a fixed communications threshold.
    contamination_sentinel_passed = bool(
        contaminated_error > 1000 * max(relative_error, np.finfo(np.float64).eps)
    )

    power_grid = np.sum(np.abs(x_grid) ** 2, axis=0)
    expected_pilot_count = [64, 64]
    checks = {
        "pilot_symbol_energy": bool(
            np.allclose(np.abs(pilots) ** 2, 1.0, rtol=0.0, atol=1e-12)
        ),
        "pilot_only_grid": bool(
            x_grid.shape == (nt, k_count, t_count)
            and np.all(x_grid[~layout.pilot_mask] == 0)
            and np.all(x_grid[layout.pilot_mask] != 0)
        ),
        "non_owner_muted": bool(
            np.all(x_grid[layout.mute_mask] == 0)
            and np.all(np.count_nonzero(x_grid, axis=0)[layout.pilot_union] == 1)
        ),
        "pilot_re_total_power": bool(
            np.allclose(power_grid[layout.pilot_union], 1.0, rtol=0.0, atol=1e-12)
        ),
        "mimo_receive_model": bool(
            y_grid.shape == (nr, k_count, t_count)
            and np.allclose(
                y_grid,
                np.tensordot(h_true, x_grid, axes=(1, 0)),
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "manual_tx0_coordinate": bool(
            np.array_equal(x_grid[:, 0, 0], [pilots[0], 0])
            and np.allclose(y_grid[:, 0, 0], h_true[:, 0] * pilots[0], rtol=0.0, atol=1e-12)
            and np.allclose(h_ls[:, 0, 0, 0], h_true[:, 0], rtol=0.0, atol=1e-12)
        ),
        "manual_tx1_coordinate": bool(
            np.array_equal(x_grid[:, 2, 0], [0, pilots[1]])
            and np.allclose(y_grid[:, 2, 0], h_true[:, 1] * pilots[1], rtol=0.0, atol=1e-12)
            and np.allclose(h_ls[:, 1, 2, 0], h_true[:, 1], rtol=0.0, atol=1e-12)
        ),
        "tx0_noiseless_ls": bool(
            np.allclose(
                h_ls[:, 0, layout.pilot_mask[0]],
                h_true[:, 0, None],
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "tx1_noiseless_ls": bool(
            np.allclose(
                h_ls[:, 1, layout.pilot_mask[1]],
                h_true[:, 1, None],
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "all_pilot_ls_error": bool(relative_error <= 1e-12),
        "nonpilot_mask": bool(
            np.array_equal(estimate_mask, layout.pilot_mask)
            and np.all(np.isnan(h_ls.real[~full_mask]))
            and np.all(np.isnan(h_ls.imag[~full_mask]))
            and np.all(np.isfinite(h_ls[full_mask]))
        ),
        "observation_count": bool(pilot_counts == expected_pilot_count and scalar_count == 256),
        "contamination_negative_test": contamination_sentinel_passed,
    }
    all_checks_passed = bool(all(checks.values()))
    report = {
        "task": "T04B",
        "nt": nt,
        "nr": nr,
        "K": k_count,
        "T": t_count,
        "dF": d_f,
        "dT": d_t,
        "pilot_symbols_real_imag": _complex_pairs(pilots),
        "true_H_real_imag": _complex_pairs(h_true),
        "pilot_count_by_tx": pilot_counts,
        "total_scalar_ls_observations": scalar_count,
        "manual_tx0_coordinate": manual[0]["coordinate"],
        "manual_tx0_x": manual[0]["x"],
        "manual_tx0_y": manual[0]["y"],
        "manual_tx0_estimate": manual[0]["estimate"],
        "manual_tx0_truth": manual[0]["truth"],
        "manual_tx1_coordinate": manual[1]["coordinate"],
        "manual_tx1_x": manual[1]["x"],
        "manual_tx1_y": manual[1]["y"],
        "manual_tx1_estimate": manual[1]["estimate"],
        "manual_tx1_truth": manual[1]["truth"],
        "pilot_ls_relative_error": relative_error,
        "pilot_ls_max_absolute_error": maximum_error,
        "contaminated_ls_error": contaminated_error,
        "contamination_sentinel_passed": contamination_sentinel_passed,
        "checks": checks,
        "all_checks_passed": all_checks_passed,
    }

    print("T04B noiseless 2x2 Pilot-LS validation")
    print(f"Nt: {nt}")
    print(f"Nr: {nr}")
    print(f"K: {k_count}")
    print(f"T: {t_count}")
    print(f"dF: {d_f}")
    print(f"dT: {d_t}")
    print(f"pilot symbols: {pilots.tolist()}")
    print("True H:")
    print(h_true)
    print(f"Pilot RE count TX0: {pilot_counts[0]}")
    print(f"Pilot RE count TX1: {pilot_counts[1]}")
    print(f"Total scalar LS observations: {scalar_count}")
    for q, (k, t) in enumerate(((0, 0), (2, 0))):
        print(f"Manual TX{q} pilot:")
        print(f"coordinate: (k={k}, t={t})")
        print(f"X: {x_grid[:, k, t]}")
        print(f"Y: {y_grid[:, k, t]}")
        print(f"LS estimate: {h_ls[:, q, k, t]}")
        print(f"truth: {h_true[:, q]}")
    print(f"Pilot LS relative error: {relative_error:.17g}")
    print(f"Pilot LS max absolute error: {maximum_error:.17g}")
    print(f"Contaminated-pilot LS error: {contaminated_error:.17g}")
    print(f"Orthogonal pilot LS: {'PASS' if checks['all_pilot_ls_error'] else 'FAIL'}")
    print(f"Contamination sentinel: {'PASS' if contamination_sentinel_passed else 'FAIL'}")
    print(f"All checks: {'PASS' if all_checks_passed else 'FAIL'}")

    output_path = PROJECT_ROOT / "outputs" / "t04b_pilot_ls_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"JSON saved: {output_path}")
    if not all_checks_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
