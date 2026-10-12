"""T07A paired Monte Carlo scan on the unchanged T06B calibration channel."""

import numpy as np

from src.link import (
    add_received_noise, apply_frequency_mimo_channel, assemble_pilot_data_frame,
    build_calibration_mimo_channel, demodulate_data_bits, extract_reference_data_bits,
    generate_master_qpsk, receive_perfect_and_ls_linear,
)
from src.metrics import MetricAccumulator, channel_error_energies, count_bit_errors
from src.numerics import complex_gaussian_noise, noise_variance_from_snr
from src.pilots import build_fdm_pilot_layout, count_layout_resources


SNR_POINTS = (0, 5, 10, 15, 20)
DF_OPTIONS = (2, 4, 8, 16)
NUM_FRAMES = 200
BITS_SEED_BASE = 20261001
NOISE_SEED_BASE = 30261001
SEED_STRIDE = 100000


def frame_seeds(snr_index: int, frame_index: int) -> tuple[tuple[int, int], int]:
    """Disjoint seeds for each SNR/frame/TX; dF never changes a seed."""
    for name, value in (("snr_index", snr_index), ("frame_index", frame_index)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if snr_index >= len(SNR_POINTS) or 2 * frame_index + 1 >= SEED_STRIDE:
        raise ValueError("seed index exceeds its disjoint allocation")
    first = BITS_SEED_BASE + snr_index * SEED_STRIDE + 2 * frame_index
    return (first, first + 1), NOISE_SEED_BASE + snr_index * SEED_STRIDE + frame_index


def scan_schedule(num_frames: int = NUM_FRAMES):
    """Enumerate the actual 5xframesx4 evaluation coordinates."""
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or not 1 <= num_frames <= NUM_FRAMES:
        raise ValueError("num_frames must be an integer in [1,200]")
    return ((snr, frame, df) for snr in SNR_POINTS for frame in range(num_frames) for df in DF_OPTIONS)


def run_scan(num_frames: int = NUM_FRAMES, progress=None) -> tuple[list, list, list]:
    """Small runs are for tests/benchmark; the performance entry point uses 200."""
    # Validate before allocating or executing any frames.
    scan_schedule(num_frames)
    _, h_true = build_calibration_mimo_channel()
    layouts = {df: build_fdm_pilot_layout(2, 64, 16, df, 4) for df in DF_OPTIONS}
    resources = {df: count_layout_resources(layout) for df, layout in layouts.items()}
    data_bits_per_frame = {df: 2 * int(np.count_nonzero(layout.data_mask)) for df, layout in layouts.items()}
    rows, frame_rows, noise_checks = [], [], []
    for snr_index, snr in enumerate(SNR_POINTS):
        variance = noise_variance_from_snr(snr, total_power=1.0)
        accumulators = {df: MetricAccumulator() for df in DF_OPTIONS}
        noise_energy, noise_samples = 0.0, 0
        for frame_index in range(num_frames):
            bits_seeds, noise_seed = frame_seeds(snr_index, frame_index)
            bits_master, s_master = generate_master_qpsk(bits_seeds=bits_seeds)
            noise_master = complex_gaussian_noise((2, 64, 16), variance, np.random.default_rng(noise_seed))
            # Count a realization once; its reuse across dF adds no independent samples.
            noise_energy += float(np.sum(np.abs(noise_master) ** 2))
            noise_samples += noise_master.size
            for df in DF_OPTIONS:
                layout = layouts[df]
                x = assemble_pilot_data_frame(s_master, layout)
                clean_y = apply_frequency_mimo_channel(h_true, x)
                y_noisy = add_received_noise(clean_y, noise_master)
                _, h_est, perfect_x, estimated_x = receive_perfect_and_ls_linear(y_noisy, h_true, layout)
                reference_bits = extract_reference_data_bits(bits_master, layout.data_mask)
                perfect_errors, data_bits = count_bit_errors(
                    demodulate_data_bits(np.sqrt(2) * perfect_x, layout.data_mask), reference_bits
                )
                ls_errors, ls_data_bits = count_bit_errors(
                    demodulate_data_bits(np.sqrt(2) * estimated_x, layout.data_mask), reference_bits
                )
                if data_bits != data_bits_per_frame[df] or ls_data_bits != data_bits:
                    raise ValueError("data-only bit accounting mismatch")
                error_energy, truth_energy = channel_error_energies(h_est, h_true, layout.data_mask)
                accumulators[df].add(error_energy, truth_energy, perfect_errors, ls_errors, data_bits)
                frame_rows.append({
                    "frame_index": frame_index, "snr_db": snr, "dF": df,
                    "bits_seed": bits_seeds[0], "bits_seed_tx1": bits_seeds[1], "noise_seed": noise_seed,
                    "nmse_error_energy": error_energy, "nmse_truth_energy": truth_energy,
                    "perfect_csi_bit_errors": perfect_errors, "ls_lin_bit_errors": ls_errors, "data_bits": data_bits,
                })
            if progress is not None and (frame_index + 1) % 50 == 0:
                progress(f"SNR={snr} dB: {frame_index + 1}/{num_frames} paired frames")
        empirical = noise_energy / noise_samples
        relative_error = abs(empirical - variance) / variance
        noise_check = {"snr_db": snr, "target_noise_variance": variance,
                       "empirical_noise_power": empirical, "noise_relative_error": relative_error,
                       "unique_complex_noise_samples": noise_samples, "passed": bool(relative_error <= 0.02)}
        noise_checks.append(noise_check)
        for df in DF_OPTIONS:
            metrics = accumulators[df].finalize()
            checks = bool(metrics["num_frames"] == num_frames
                          and metrics["total_data_bits"] == num_frames * data_bits_per_frame[df]
                          and np.isfinite(metrics["nmse_data_linear"]) and metrics["nmse_data_linear"] >= 0
                          and np.isfinite(metrics["nmse_data_db"])
                          and 0 <= metrics["perfect_csi_ber"] <= 1 and 0 <= metrics["ls_lin_ber"] <= 1
                          and noise_check["passed"])
            rows.append({"dF": df, "pilot_ratio": resources[df]["union_ratio"], "snr_db": snr,
                         "noise_variance": variance, "data_bits_per_frame": data_bits_per_frame[df],
                         **metrics, "target_noise_variance": variance, "empirical_noise_power": empirical,
                         "noise_relative_error": relative_error, "all_checks_passed": checks})
    rows.sort(key=lambda row: (row["dF"], row["snr_db"]))
    return rows, frame_rows, noise_checks


def zero_noise_regression(t06b_baseline: dict) -> dict:
    """Reproduce the accepted T06B data and compare raw metrics without AWGN."""
    bits, symbols = generate_master_qpsk(bits_seeds=tuple(t06b_baseline["bits_seeds"]))
    _, h_true = build_calibration_mimo_channel()
    zero_noise = complex_gaussian_noise((2, 64, 16), 0, np.random.default_rng(NOISE_SEED_BASE))
    conditions = []
    for baseline in t06b_baseline["layouts"]:
        df = baseline["dF"]
        layout = build_fdm_pilot_layout(2, 64, 16, df, 4)
        x = assemble_pilot_data_frame(symbols, layout)
        y = add_received_noise(apply_frequency_mimo_channel(h_true, x), zero_noise)
        h_ls, h_est, perfect_x, estimated_x = receive_perfect_and_ls_linear(y, h_true, layout)
        reference = extract_reference_data_bits(bits, layout.data_mask)
        perfect_errors, bit_count = count_bit_errors(demodulate_data_bits(np.sqrt(2) * perfect_x, layout.data_mask), reference)
        ls_errors, _ = count_bit_errors(demodulate_data_bits(np.sqrt(2) * estimated_x, layout.data_mask), reference)
        error, truth = channel_error_energies(h_est, h_true, layout.data_mask)
        pilot_mask = np.broadcast_to(layout.pilot_mask[None], h_true.shape)
        pilot_error = float(np.linalg.norm(h_ls[pilot_mask] - h_true[pilot_mask]) / np.linalg.norm(h_true[pilot_mask]))
        nmse_matches = bool(np.isclose(error / truth, baseline["nmse_data_linear"], rtol=1e-12, atol=1e-15))
        passed = bool(perfect_errors == 0 and bit_count == baseline["data_bits"]
                      and ls_errors == baseline["ls_lin_bit_errors"] and nmse_matches and pilot_error <= 1e-12)
        conditions.append({"dF": df, "perfect_csi_bit_errors": perfect_errors,
                           "ls_lin_bit_errors": ls_errors, "data_bits": bit_count,
                           "nmse_data_linear": error / truth, "baseline_nmse_data_linear": baseline["nmse_data_linear"],
                           "pilot_ls_relative_error": pilot_error, "passed": passed})
    return {"noise_variance": 0.0, "conditions": conditions,
            "passed": bool(len(conditions) == 4 and all(row["passed"] for row in conditions))}
