"""T09A paired, noiseless dynamic TDL-A calibration using verified receivers."""

import re
from collections.abc import Callable

import numpy as np

from src.detectors import zf_detect
from src.doppler import build_tdl_a_doppler_channel
from src.link import (
    PILOT_SYMBOLS, apply_frequency_mimo_channel, assemble_pilot_data_frame,
    demodulate_data_bits, detect_data_zf, estimate_ls_linear_channel,
    extract_reference_data_bits, generate_master_qpsk,
)
from src.metrics import channel_error_energies, count_bit_errors
from src.pilots import build_fdm_pilot_layout, count_layout_resources


NUM_TRAJECTORIES = 200
CHANNEL_SEED_BASE = 40261001
BITS_SEED_BASE = 20261001
SPACINGS = (2, 4, 8, 16)
PILOT_TIMES = (0, 4, 8, 12)
PILOT_LS_TOLERANCE = 1e-12
EXPERIMENT_TYPE = "noiseless dynamic TDL-A full-link interpolation-floor calibration"


def complex_record(value: np.ndarray) -> dict:
    """Explicit real/imag arrays, preserving the original axis order in JSON."""
    array = np.asarray(value)
    def component(values):
        serializable = values.astype(object)
        serializable[~np.isfinite(values)] = None
        return serializable.tolist()
    return {"real": component(array.real), "imag": component(array.imag)}


class NoiselessLinkFailure(ValueError):
    """A hard failure retains trajectory, physical RE and symbol context."""

    def __init__(self, record: dict):
        self.record = record
        super().__init__(str(record))


def failure_context(reason, trajectory_id, h_true, k, t, *, dF=None, x=None,
                    h_used=None, detected=None, path=None) -> dict:
    matrix = h_true[:, :, k, t]
    condition = float(np.linalg.cond(matrix)) if np.all(np.isfinite(matrix)) else None
    return {
        "reason": reason, "trajectory_id": trajectory_id,
        "channel_seed": CHANNEL_SEED_BASE + trajectory_id, "dF": dF,
        "k": int(k), "t": int(t), "path": path,
        "condition_number": condition if condition is not None and np.isfinite(condition) else None,
        "condition_number_nonfinite": condition is not None and not np.isfinite(condition),
        "H_true": complex_record(matrix),
        "H_used": complex_record(h_used[:, :, k, t]) if h_used is not None else None,
        "symbol": complex_record(x[:, k, t]) if x is not None else None,
        "detected_symbol": complex_record(detected[:, k, t]) if detected is not None else None,
    }


def condition_numbers(h_true: np.ndarray, trajectory_id: int) -> np.ndarray:
    """Count every physical H once; no condition-number threshold or filtering."""
    if h_true.shape != (2, 2, 64, 16):
        raise ValueError("T09A H must have shape (2,2,64,16)")
    finite = np.all(np.isfinite(h_true), axis=(0, 1))
    if not np.all(finite):
        k, t = np.argwhere(~finite)[0]
        raise NoiselessLinkFailure(failure_context("nonfinite true channel", trajectory_id, h_true, k, t))
    matrices = np.transpose(h_true, (2, 3, 0, 1))
    values = np.linalg.cond(matrices)
    # Exact zero determinant / nonfinite SVD condition, not a chosen large-cond cutoff.
    bad = (np.linalg.det(matrices) == 0) | ~np.isfinite(values)
    if np.any(bad):
        k, t = np.argwhere(bad)[0]
        raise NoiselessLinkFailure(failure_context("singular true channel", trajectory_id, h_true, k, t))
    return values


def safe_detect(y, h_used, layout, h_true, x, trajectory_id, dF, path):
    """Map the detector's packed data index back to the actual (k,t)."""
    try:
        detected = detect_data_zf(y, h_used, layout.data_mask)
    except ValueError as exc:
        match = re.search(r"RE \(k=(\d+), t=\d+\)", str(exc))
        coordinates = np.argwhere(layout.data_mask[0])
        if match:
            k, t = coordinates[int(match.group(1))]
        else:
            # Recover coordinates if the verified detector reports nonfinite output.
            k, t = coordinates[0]
            for candidate_k, candidate_t in coordinates:
                matrix = h_used[:, :, candidate_k, candidate_t]
                vector = y[:, candidate_k, candidate_t]
                if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
                    k, t = candidate_k, candidate_t
                    break
                try:
                    value = np.linalg.solve(matrix, vector)
                except np.linalg.LinAlgError:
                    k, t = candidate_k, candidate_t
                    break
                if not np.all(np.isfinite(value)):
                    k, t = candidate_k, candidate_t
                    break
        raise NoiselessLinkFailure(failure_context(
            str(exc), trajectory_id, h_true, k, t, dF=dF, x=x, h_used=h_used, path=path,
        )) from exc
    return detected


def per_symbol_energies(h_est, h_true, data_mask) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool RX/TX/frequency energies, but count physical data REs once per time."""
    selected = np.broadcast_to(data_mask[None], h_true.shape)
    error = np.sum(np.where(selected, np.abs(h_est - h_true) ** 2, 0), axis=(0, 1, 2))
    truth = np.sum(np.where(selected, np.abs(h_true) ** 2, 0), axis=(0, 1, 2))
    counts = np.count_nonzero(data_mask[0], axis=0)
    return error, truth, counts


def make_symbol_rows(dF, pilot_ratio, error, truth, counts) -> list[dict]:
    """No-data symbols have undefined NMSE, represented as None (CSV blank)."""
    rows = []
    for t in range(16):
        nmse = float(error[t] / truth[t]) if counts[t] else None
        rows.append({
            "dF": dF, "pilot_ratio": pilot_ratio, "symbol_index": t,
            "is_pilot_time": t in PILOT_TIMES,
            "error_energy": float(error[t]), "truth_energy": float(truth[t]),
            "nmse_linear": nmse,
            "nmse_db": float(10 * np.log10(nmse)) if nmse is not None and nmse > 0 else None,
            "data_re_count_total": int(counts[t]),
        })
    return rows


def manual_trace(h_true, h_est, h_ls, bits, s_master, x, y, layout) -> list[dict]:
    # Fixed before the run: TX0 pilot, interior data, trailing-edge data.
    records = []
    for kind, k, t in (("pilot", 0, 0), ("interior_data", 1, 2), ("trailing_data", 1, 15)):
        truth, estimate = h_true[:, :, k, t], h_est[:, :, k, t]
        error = np.abs(estimate - truth)
        owner = int(layout.pilot_owner[k, t])
        owner_error = np.linalg.norm(h_ls[:, owner, k, t] - truth[:, owner]) if owner >= 0 else None
        # Pilot ZF is a diagnostic only; main BER never includes this RE.
        perfect_x = zf_detect(y[:, k:k+1, t:t+1], h_true[:, :, k:k+1, t:t+1])[:, 0, 0]
        ls_x = zf_detect(y[:, k:k+1, t:t+1], h_est[:, :, k:k+1, t:t+1])[:, 0, 0]
        records.append({
            "trajectory_id": 0, "channel_seed": CHANNEL_SEED_BASE, "dF": 4,
            "kind": kind, "k": k, "t": t, "pilot_owner": owner if owner >= 0 else None,
            "H_true": complex_record(truth), "H_est": complex_record(estimate),
            "absolute_channel_error": error.tolist(),
            "absolute_channel_error_frobenius": float(np.linalg.norm(estimate - truth)),
            "relative_channel_error": float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth)),
            "pilot_owner_ls_absolute_error": float(owner_error) if owner_error is not None else None,
            "pilot_owner_ls_relative_error": float(owner_error / np.linalg.norm(truth[:, owner])) if owner >= 0 else None,
            "S_master": complex_record(s_master[:, k, t]),
            "master_bits": bits[:, k, t].tolist(),
            "S_master_used_as_data": kind != "pilot",
            "X": complex_record(x[:, k, t]), "Y": complex_record(y[:, k, t]),
            "perfect_csi_X_hat": complex_record(perfect_x), "ls_lin_X_hat": complex_record(ls_x),
        })
    return records


def evaluate_layout(h_true, bits, s_master, layout, trajectory_id, dF, conditions) -> dict:
    """H_true is used by the physical channel, Perfect CSI and evaluation only."""
    x = assemble_pilot_data_frame(s_master, layout)
    y = apply_frequency_mimo_channel(h_true, x)  # Deliberately no AWGN call.
    h_ls, h_est = estimate_ls_linear_channel(y, layout, PILOT_SYMBOLS)
    if not np.all(np.isfinite(h_est)):
        k, t = np.argwhere(~np.all(np.isfinite(h_est), axis=(0, 1)))[0]
        raise NoiselessLinkFailure(failure_context(
            "nonfinite LS-LIN grid", trajectory_id, h_true, k, t, dF=dF, x=x, h_used=h_est,
        ))
    observed = np.broadcast_to(layout.pilot_mask[None], h_true.shape)
    pilot_error = float(np.sum(np.abs(h_ls[observed] - h_true[observed]) ** 2))
    pilot_truth = float(np.sum(np.abs(h_true[observed]) ** 2))
    reference = extract_reference_data_bits(bits, layout.data_mask)
    perfect_x = safe_detect(y, h_true, layout, h_true, x, trajectory_id, dF, "perfect_csi")
    perfect_bits = demodulate_data_bits(np.sqrt(2) * perfect_x, layout.data_mask)
    perfect_errors, data_bits = count_bit_errors(perfect_bits, reference)
    if perfect_errors:
        q = next(q for q in range(2) if np.any(perfect_bits[q] != reference[q]))
        bit_index = int(np.flatnonzero(perfect_bits[q] != reference[q])[0])
        k, t = np.argwhere(layout.data_mask[q])[bit_index // 2]
        record = failure_context("Perfect-CSI bit error", trajectory_id, h_true, k, t,
                                 dF=dF, x=x, h_used=h_true, detected=perfect_x, path="perfect_csi")
        record.update({"tx": q, "bit_index": bit_index, "expected_bit": int(reference[q][bit_index]),
                       "actual_bit": int(perfect_bits[q][bit_index])})
        raise NoiselessLinkFailure(record)
    if np.sqrt(pilot_error / pilot_truth) > PILOT_LS_TOLERANCE:
        delta = np.where(observed, np.abs(h_ls - h_true), 0)
        _, _, k, t = np.unravel_index(np.argmax(delta), delta.shape)
        raise NoiselessLinkFailure(failure_context(
            "Pilot-LS exceeds relative error tolerance", trajectory_id, h_true, k, t, dF=dF, x=x,
        ))
    ls_x = safe_detect(y, h_est, layout, h_true, x, trajectory_id, dF, "ls_lin")
    ls_errors, ls_data_bits = count_bit_errors(demodulate_data_bits(np.sqrt(2) * ls_x, layout.data_mask), reference)
    if ls_data_bits != data_bits:
        raise ValueError("receiver paths counted different data bits")
    error, truth = channel_error_energies(h_est, h_true, layout.data_mask)
    symbol_error, symbol_truth, symbol_counts = per_symbol_energies(h_est, h_true, layout.data_mask)
    whole_error = np.sum(np.abs(h_est - h_true) ** 2, axis=(0, 1, 2))
    whole_truth = np.sum(np.abs(h_true) ** 2, axis=(0, 1, 2))
    ratio = float(count_layout_resources(layout)["union_ratio"])
    row = {
        "trajectory_id": trajectory_id, "channel_seed": CHANNEL_SEED_BASE + trajectory_id,
        "bits_seed_tx0": BITS_SEED_BASE + 2 * trajectory_id,
        "bits_seed_tx1": BITS_SEED_BASE + 2 * trajectory_id + 1,
        "dF": dF, "pilot_ratio": ratio, "data_bits": data_bits,
        "pilot_ls_error_energy": pilot_error, "pilot_ls_truth_energy": pilot_truth,
        "nmse_error_energy": error, "nmse_truth_energy": truth,
        "perfect_csi_bit_errors": perfect_errors, "ls_lin_bit_errors": ls_errors,
        "condition_number_min": float(conditions.min()),
        "condition_number_median": float(np.median(conditions)),
        "condition_number_max": float(conditions.max()),
    }
    return {
        "row": row, "symbol_error": symbol_error, "symbol_truth": symbol_truth,
        "symbol_counts": symbol_counts, "whole_error": whole_error, "whole_truth": whole_truth,
        "manual": manual_trace(h_true, h_est, h_ls, bits, s_master, x, y, layout)
        if trajectory_id == 0 and dF == 4 else [],
    }


def run_noiseless_scan(num_trajectories: int = NUM_TRAJECTORIES,
                       progress: Callable[[int], None] | None = None) -> tuple[dict, list[dict]]:
    """One H and one master bits object per trajectory, reused across four dF."""
    if isinstance(num_trajectories, bool) or not isinstance(num_trajectories, int) or num_trajectories < 1:
        raise ValueError("num_trajectories must be a positive integer")
    layouts = {dF: build_fdm_pilot_layout(2, 64, 16, dF, 4) for dF in SPACINGS}
    rows, manual = [], []
    conditions_all = np.empty((num_trajectories, 64, 16), dtype=np.float64)
    summed = {dF: {name: np.zeros(16) for name in
                    ("symbol_error", "symbol_truth", "symbol_counts", "whole_error", "whole_truth")}
              for dF in SPACINGS}
    for trajectory_id in range(num_trajectories):
        _, h_true = build_tdl_a_doppler_channel(CHANNEL_SEED_BASE + trajectory_id)
        bits, s_master = generate_master_qpsk(bits_seeds=(
            BITS_SEED_BASE + 2 * trajectory_id, BITS_SEED_BASE + 2 * trajectory_id + 1,
        ))
        conditions = condition_numbers(h_true, trajectory_id)
        conditions_all[trajectory_id] = conditions
        for dF, layout in layouts.items():
            result = evaluate_layout(h_true, bits, s_master, layout, trajectory_id, dF, conditions)
            rows.append(result["row"])
            manual.extend(result["manual"])
            for name in summed[dF]:
                summed[dF][name] += result[name]
        if progress:
            progress(trajectory_id + 1)
    aggregate, symbols, intervals, anchors = [], [], [], []
    for dF in SPACINGS:
        group = [row for row in rows if row["dF"] == dF]
        totals = {name: sum(row[name] for row in group) for name in (
            "data_bits", "pilot_ls_error_energy", "pilot_ls_truth_energy", "nmse_error_energy",
            "nmse_truth_energy", "perfect_csi_bit_errors", "ls_lin_bit_errors",
        )}
        nmse = totals["nmse_error_energy"] / totals["nmse_truth_energy"]
        pilot_relative = float(np.sqrt(totals["pilot_ls_error_energy"] / totals["pilot_ls_truth_energy"]))
        ratio = group[0]["pilot_ratio"]
        aggregate.append({
            "dF": dF, "pilot_ratio": ratio, "num_trajectories": num_trajectories,
            "data_bits_per_trajectory": group[0]["data_bits"], "total_data_bits": totals["data_bits"],
            "pilot_ls_relative_error": pilot_relative,
            "nmse_error_energy": totals["nmse_error_energy"], "nmse_truth_energy": totals["nmse_truth_energy"],
            "nmse_data_linear": nmse, "nmse_data_db": float(10 * np.log10(nmse)),
            "perfect_csi_bit_errors": totals["perfect_csi_bit_errors"],
            "perfect_csi_ber": totals["perfect_csi_bit_errors"] / totals["data_bits"],
            "ls_lin_bit_errors": totals["ls_lin_bit_errors"], "ls_lin_ber": totals["ls_lin_bit_errors"] / totals["data_bits"],
            "all_checks_passed": pilot_relative <= PILOT_LS_TOLERANCE and totals["perfect_csi_bit_errors"] == 0,
        })
        sums = summed[dF]
        per_symbol = make_symbol_rows(dF, ratio, sums["symbol_error"], sums["symbol_truth"], sums["symbol_counts"])
        symbols.extend(per_symbol)
        for start in (1, 5, 9, 13):
            linear_mean = float(np.mean([per_symbol[t]["nmse_linear"] for t in range(start, start + 3)]))
            intervals.append({"dF": dF, "symbols": list(range(start, start + 3)),
                              "mean_symbol_nmse_linear": linear_mean,
                              "mean_symbol_nmse_db": float(10 * np.log10(linear_mean)),
                              "definition": "arithmetic mean of the three pooled per-symbol linear NMSEs, then dB"})
        for t in PILOT_TIMES:
            whole_nmse = float(sums["whole_error"][t] / sums["whole_truth"][t])
            anchors.append({"dF": dF, "symbol_index": t,
                            "whole_symbol_error_energy": float(sums["whole_error"][t]),
                            "whole_symbol_truth_energy": float(sums["whole_truth"][t]),
                            "whole_symbol_nmse_linear": whole_nmse,
                            "whole_symbol_nmse_db": float(10 * np.log10(whole_nmse))})
    worst_id, k, t = np.unravel_index(np.argmax(conditions_all), conditions_all.shape)
    percentiles = np.percentile(conditions_all, (50, 90, 99, 99.9))
    report = {
        "task": "T09A", "experiment_type": EXPERIMENT_TYPE, "num_trajectories": num_trajectories,
        "channel_definition": {"profile": "TDL-A", "num_taps": 23, "rms_delay_spread_ns": 100,
                               "carrier_frequency_hz": 3500000000, "speed_kmh": 30,
                               "temporal_model": "classical/Jakes covariance",
                               "fractional_delay_frequency_response": True, "independent_antenna_pairs": True,
                               "symbol_wise_quasi_static": True, "ici_modeled": False,
                               "H_shape": [2, 2, 64, 16], "dtype": "complex128", "awgn_added": False},
        "seed_policy": {"channel_seed_base": CHANNEL_SEED_BASE, "channel_seed": "40261001 + trajectory_id",
                        "bits_seed_base": BITS_SEED_BASE, "bits_seed_txq": "20261001 + 2*trajectory_id + q",
                        "noise_seed": None, "training_seed": None,
                        "paired_across_dF": ["H_true", "master_bits", "S_master"]},
        "aggregate_results": aggregate, "per_symbol_results": symbols,
        "boundary_diagnostics": intervals, "pilot_time_whole_symbol_diagnostics": anchors,
        "metric_definitions": {
            "nmse": "sum error energy / sum truth energy, RX/TX entries at actual data RE only",
            "pilot_ls_relative_error": "sqrt(sum pilot error energy / sum pilot truth energy)",
            "ber": "total data bit errors / total data bits",
            "data_re_count_total": "physical (k,t) data REs across trajectories; RX/TX not counted repeatedly",
            "no_data_symbol": "error=truth=count=0; undefined NMSE is CSV blank / JSON null / plot gap",
            "right_edge": "existing time interpolation holds t=12 at t=13,14,15",
        },
        "condition_number_statistics": {"count": int(conditions_all.size),
                                        **dict(zip(("p50", "p90", "p99", "p99_9"), map(float, percentiles))),
                                        "max": float(conditions_all.max()), "quantile_method": "linear",
                                        "counted_once_across_dF": True, "threshold_filtering": False},
        "maximum_condition_location": {"trajectory_id": int(worst_id), "k": int(k), "t": int(t),
                                       "channel_seed": CHANNEL_SEED_BASE + int(worst_id)},
        "manual_diagnostics": manual, "perfect_csi_zero_error": True,
        "pilot_ls_exact": all(row["pilot_ls_relative_error"] <= PILOT_LS_TOLERANCE for row in aggregate),
        "ls_lin_grid_finite": True, "num_filtered_trajectories": 0,
        "trajectory_rows": len(rows), "aggregate_rows": len(aggregate), "symbol_rows": len(symbols),
        "all_checks_passed": all(row["all_checks_passed"] for row in aggregate),
    }
    return report, rows
