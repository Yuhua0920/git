"""T06B fixed frequency-selective 2x2 link, without noise or phase search."""

import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, validate_config  # noqa: E402
from src.link import (  # noqa: E402
    DELAYS, LINK_PHASES, PILOT_SYMBOLS, POWERS_DB,
    apply_frequency_mimo_channel, assemble_pilot_data_frame,
    build_calibration_mimo_channel, data_channel_nmse, demodulate_data_bits,
    detect_data_zf, estimate_ls_linear_channel, extract_reference_data_bits,
    generate_master_qpsk,
)
from src.pilots import build_fdm_pilot_layout, count_layout_resources  # noqa: E402


def _pairs(value: np.ndarray) -> list:
    array = np.asarray(value, dtype=np.complex128)
    return np.stack((array.real, array.imag), axis=-1).tolist()


def _relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))


def _bit_errors(recovered: tuple, reference: tuple) -> list[int]:
    if len(recovered) != len(reference) or any(a.shape != b.shape for a, b in zip(recovered, reference)):
        raise ValueError("recovered/reference data bit shapes must match")
    return [int(np.count_nonzero(a != b)) for a, b in zip(recovered, reference)]


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    validate_config(config)
    for key, expected in {"nt": 2, "nr": 2, "nfft": 64, "active_carriers": 64,
                          "num_symbols": 16, "total_power": 1.0,
                          "modulation": "qpsk_gray"}.items():
        if config["system"][key] != expected:
            raise ValueError(f"T06B requires {key}={expected}; got {config['system'][key]}")
    if config["pilot"]["spacing_time"] != 4 or config["pilot"]["spacing_frequency_options"] != [2, 4, 8, 16]:
        raise ValueError("T06B requires dT=4 and dF options [2,4,8,16]")
    if config["seed"]["base_seed"] != 20260930:
        raise ValueError("T06B requires fixed base bit seed 20260930")

    seeds = (20260930, 20260931)
    bits_master, s_master = generate_master_qpsk(bits_seeds=seeds)
    impulses, h_true = build_calibration_mimo_channel()
    ranks = [int(np.linalg.matrix_rank(h_true[:, :, k, 0])) for k in range(64)]
    condition_numbers = np.array([np.linalg.cond(h_true[:, :, k, 0]) for k in range(64)])
    condition_summary = {
        "min": float(np.min(condition_numbers)),
        "median": float(np.median(condition_numbers)),
        "max": float(np.max(condition_numbers)),
        "all_frequency_bins_full_rank": bool(all(rank == 2 for rank in ranks)),
    }
    if not condition_summary["all_frequency_bins_full_rank"]:
        raise ValueError("T06B FAIL: singular frequency bin in fixed calibration channel")

    print("T06B Noiseless frequency-selective 2x2 full-link validation")
    print("Nt=2, Nr=2, K=64, T=16; centered frequency order; complex128")
    print(f"Bits seeds: {seeds}; one shared master grid for all dF")
    print(f"Channel delays: {list(DELAYS)}; powers dB: {list(POWERS_DB)}")
    print(f"Impulse power by link: {np.sum(np.abs(impulses) ** 2, axis=-1).tolist()}")
    print(f"Average frequency power by link: {np.mean(np.abs(h_true[..., 0]) ** 2, axis=-1).tolist()}")
    for label in ("min", "median", "max"):
        print(f"{label} condition number: {condition_summary[label]:.17g}")
    print("All 64 H[k] full rank: PASS")

    rows, csv_rows, layout_checks = [], [], []
    manual, pilot_diagnostics = {}, []
    expected_data_counts = {2: 768, 4: 896, 8: 960, 16: 992}
    for df in config["pilot"]["spacing_frequency_options"]:
        layout = build_fdm_pilot_layout(2, 64, 16, df, 4)
        resources = count_layout_resources(layout)
        x = assemble_pilot_data_frame(s_master, layout)
        y = apply_frequency_mimo_channel(h_true, x)
        # Receiver path has no H_true, transmitted data, or master bits argument.
        h_ls, h_est = estimate_ls_linear_channel(y, layout, PILOT_SYMBOLS)
        perfect_x = detect_data_zf(y, h_true, layout.data_mask)
        estimated_x = detect_data_zf(y, h_est, layout.data_mask)
        reference_bits = extract_reference_data_bits(bits_master, layout.data_mask)
        perfect_errors = _bit_errors(demodulate_data_bits(np.sqrt(2) * perfect_x, layout.data_mask), reference_bits)
        ls_errors = _bit_errors(demodulate_data_bits(np.sqrt(2) * estimated_x, layout.data_mask), reference_bits)
        data_bits = sum(stream.size for stream in reference_bits)
        data_re_count = int(np.count_nonzero(np.all(layout.data_mask, axis=0)))
        selected_pilots = np.broadcast_to(layout.pilot_mask[None], h_true.shape)
        pilot_error = _relative_error(h_ls[selected_pilots], h_true[selected_pilots])
        nmse = data_channel_nmse(h_est, h_true, layout.data_mask)
        nmse_db = float(10 * np.log10(nmse))
        perfect_error = _relative_error(perfect_x[layout.data_mask], x[layout.data_mask])
        re_power = np.sum(np.abs(x) ** 2, axis=0)
        checks = {
            "frame_partition": bool(np.all(layout.pilot_mask.astype(int) + layout.data_mask + layout.mute_mask == 1)
                                    and not np.any(layout.data_mask[:, layout.pilot_union])
                                    and np.all(x[layout.mute_mask] == 0)),
            "data_pilot_power": bool(np.allclose(re_power, 1, rtol=0, atol=1e-12)
                                     and np.allclose(x[layout.data_mask], s_master[layout.data_mask] / np.sqrt(2), rtol=0, atol=1e-12)),
            "perfect_csi_recovery": bool(sum(perfect_errors) == 0 and perfect_error <= 1e-12),
            "pilot_ls_exact": bool(pilot_error <= 1e-12),
            "ls_linear_finite": bool(np.all(np.isfinite(h_est))),
            "data_nmse": bool(np.isfinite(nmse) and nmse >= 0 and np.isfinite(nmse_db)),
            "data_bit_accounting": bool(data_re_count == expected_data_counts[df]
                                        and data_bits == 2 * resources["two_stream_data_symbols"]
                                        and resources["two_stream_data_symbols"] == 2 * data_re_count),
        }
        layout_checks.append(checks)
        csv_row = {
            "dF": df, "pilot_ratio": resources["union_ratio"],
            "pilot_union_count": resources["union_count"], "data_re_count": data_re_count,
            "two_stream_data_symbols": resources["two_stream_data_symbols"], "data_bits": data_bits,
            "pilot_ls_relative_error": pilot_error, "nmse_data_linear": nmse, "nmse_data_db": nmse_db,
            "perfect_csi_bit_errors": sum(perfect_errors), "perfect_csi_ber": sum(perfect_errors) / data_bits,
            "ls_lin_bit_errors": sum(ls_errors), "ls_lin_ber": sum(ls_errors) / data_bits,
            "min_condition_number": condition_summary["min"],
            "median_condition_number": condition_summary["median"],
            "max_condition_number": condition_summary["max"], "all_checks_passed": bool(all(checks.values())),
        }
        csv_rows.append(csv_row)
        rows.append({**csv_row, "pilot_count_by_tx": resources["pilot_count_by_tx"],
                     "data_bits_by_tx": [int(stream.size) for stream in reference_bits],
                     "perfect_csi_bit_errors_by_tx": perfect_errors, "ls_lin_bit_errors_by_tx": ls_errors,
                     "perfect_csi_relative_symbol_error": perfect_error,
                     "data_re_total_power_min_max": [float(np.min(re_power[~layout.pilot_union])), float(np.max(re_power[~layout.pilot_union]))],
                     "pilot_re_total_power_min_max": [float(np.min(re_power[layout.pilot_union])), float(np.max(re_power[layout.pilot_union]))]})
        print(f"dF={df}: pilot ratio={resources['union_ratio']}, union={resources['union_count']}, "
              f"data RE={data_re_count}, symbols={resources['two_stream_data_symbols']}, bits={data_bits}")
        print(f"  Data/Pilot total power: {re_power[~layout.pilot_union][0]:.17g} / {re_power[layout.pilot_union][0]:.17g}")
        print(f"  Pilot-LS relative error: {pilot_error:.17g}")
        print(f"  NMSE_data: {nmse:.17g} ({nmse_db:.17g} dB)")
        print(f"  Perfect-CSI: {sum(perfect_errors)}/{data_bits} bit errors; BER={sum(perfect_errors) / data_bits:.17g}; symbol error={perfect_error:.17g}")
        print(f"  LS-LIN + ZF: {sum(ls_errors)}/{data_bits} bit errors; BER={sum(ls_errors) / data_bits:.17g}")
        print(f"  Checks: {'PASS' if csv_row['all_checks_passed'] else 'FAIL'}")
        if df == 4:
            k, t = 1, 0
            if not np.all(layout.data_mask[:, k, t]):
                raise ValueError("manual diagnostic coordinate must be a data RE")
            manual = {"dF": df, "coordinate_k_t": [k, t], "is_data_re": True,
                      "H_true": _pairs(h_true[:, :, k, t]), "H_est": _pairs(h_est[:, :, k, t]),
                      "S_master": _pairs(s_master[:, k, t]), "X": _pairs(x[:, k, t]),
                      "Y": _pairs(y[:, k, t]), "perfect_csi_X_hat": _pairs(perfect_x[:, k, t]),
                      "ls_lin_X_hat": _pairs(estimated_x[:, k, t])}
            print("Manual data RE: dF=4, (k=1,t=0), data mask confirmed")
            for label, value in (("H_true", h_true[:, :, k, t]), ("H_est", h_est[:, :, k, t]),
                                 ("S_master", s_master[:, k, t]), ("X", x[:, k, t]), ("Y", y[:, k, t]),
                                 ("Perfect-CSI X_hat", perfect_x[:, k, t]), ("LS-LIN X_hat", estimated_x[:, k, t])):
                print(f"{label}:\n{value}")
            for q, pilot_k in ((0, 0), (1, 2)):
                if not layout.pilot_mask[q, pilot_k, 0] or layout.pilot_owner[pilot_k, 0] != q:
                    raise ValueError("manual pilot coordinate owner mismatch")
                truth, estimate = h_true[:, q, pilot_k, 0], h_ls[:, q, pilot_k, 0]
                error = _relative_error(estimate, truth)
                pilot_diagnostics.append({"tx": q, "coordinate_k_t": [pilot_k, 0], "true_H_column": _pairs(truth),
                                          "ls_H_column": _pairs(estimate), "relative_error": error})
                print(f"Manual TX{q} pilot (k={pilot_k},t=0): true={truth}, LS={estimate}, relative error={error:.17g}")

    channel_checks = bool(h_true.shape == (2, 2, 64, 16)
                          and np.allclose(np.sum(np.abs(impulses) ** 2, axis=-1), 1, rtol=0, atol=1e-12)
                          and np.allclose(np.mean(np.abs(h_true[..., 0]) ** 2, axis=-1), 1, rtol=0, atol=1e-12))
    all_checks = bool(channel_checks and all(row["all_checks_passed"] for row in rows)
                      and all(item["relative_error"] <= 1e-12 for item in pilot_diagnostics))
    report = {
        "task": "T06B",
        "channel_definition": {"delays": list(DELAYS), "powers_db": list(POWERS_DB),
                               "impulse_power_by_link": np.sum(np.abs(impulses) ** 2, axis=-1).tolist(),
                               "average_frequency_power_by_link": np.mean(np.abs(h_true[..., 0]) ** 2, axis=-1).tolist(),
                               "shape": list(h_true.shape), "time_static": True,
                               "frequency_order": "fftshift ascending", "frequency_response_fft_norm": "backward",
                               "noise_added": False},
        "link_phase_definitions": {f"H{r}{q}": LINK_PHASES[r, q].tolist() for r in range(2) for q in range(2)},
        "bits_seeds": list(seeds), "pilot_symbols": _pairs(PILOT_SYMBOLS),
        "condition_number_summary": condition_summary, "layouts": rows,
        "manual_data_re_diagnostic": manual, "manual_pilot_diagnostics": pilot_diagnostics,
        "checks": {"normalized_frequency_selective_channel": channel_checks,
                   "all_H_full_rank": condition_summary["all_frequency_bins_full_rank"],
                   **{name: bool(all(check[name] for check in layout_checks)) for name in layout_checks[0]},
                   "all_four_pilot_densities": bool([row["dF"] for row in rows] == [2, 4, 8, 16])},
        "all_checks_passed": all_checks,
    }
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "t06b_noiseless_full_link.csv"
    json_path = output_dir / "t06b_noiseless_full_link_results.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"All checks: {'PASS' if all_checks else 'FAIL'}")
    print(f"CSV saved: {csv_path}")
    print(f"JSON saved: {json_path}")
    if not all_checks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
