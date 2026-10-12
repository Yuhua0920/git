import csv
import json
from pathlib import Path

import numpy as np
import pytest

import src.awgn_scan as scan
import src.link as link
from src.numerics import complex_gaussian_noise, noise_variance_from_snr
from src.pilots import build_fdm_pilot_layout


@pytest.fixture(scope="module")
def fixed_link():
    bits, symbols = link.generate_master_qpsk()
    _, h = link.build_calibration_mimo_channel()
    layout = build_fdm_pilot_layout(2, 64, 16, 4, 4)
    y = link.apply_frequency_mimo_channel(h, link.assemble_pilot_data_frame(symbols, layout))
    noise = complex_gaussian_noise(y.shape, 0.1, np.random.default_rng(30261001))
    return bits, h, layout, y, noise


@pytest.mark.parametrize("snr,expected", [(0, 1), (10, 0.1), (20, 0.01)])
def test_noise_variance_matches_main_snr_definition(snr, expected):
    assert noise_variance_from_snr(snr, 1) == pytest.approx(expected)


def test_noise_added_to_pilot_and_data_re(fixed_link):
    _, _, layout, y, noise = fixed_link
    noisy = link.add_received_noise(y, noise)
    np.testing.assert_allclose(noisy - y, noise, rtol=0, atol=1e-15)
    assert np.linalg.norm((noisy - y)[:, layout.pilot_union]) > 0
    assert np.linalg.norm((noisy - y)[:, ~layout.pilot_union]) > 0


def test_no_per_frame_noise_normalization(fixed_link):
    _, _, _, y, noise = fixed_link
    for scale in (0, 1, 10):
        np.testing.assert_allclose(link.add_received_noise(scale * y, noise) - scale * y,
                                   noise, rtol=0, atol=1e-14)


def test_perfect_and_ls_lin_use_same_noisy_y(fixed_link, monkeypatch):
    _, h, layout, y, noise = fixed_link
    noisy = link.add_received_noise(y, noise)
    real_estimate, real_detect = link.estimate_ls_linear_channel, link.detect_data_zf
    received_objects = []

    def estimate(received, pilot_layout):
        received_objects.append(received)
        return real_estimate(received, pilot_layout)

    def detect(received, channel, mask):
        received_objects.append(received)
        return real_detect(received, channel, mask)

    monkeypatch.setattr(link, "estimate_ls_linear_channel", estimate)
    monkeypatch.setattr(link, "detect_data_zf", detect)
    link.receive_perfect_and_ls_linear(noisy, h, layout)
    assert len(received_objects) == 3 and all(received is noisy for received in received_objects)


def test_shared_noise_across_df_for_same_frame_and_snr(monkeypatch):
    real_add, real_generate = scan.add_received_noise, scan.generate_master_qpsk
    noises, data_calls = [], []

    def add(y, noise):
        noises.append(noise)
        return real_add(y, noise)

    def generate(*args, **kwargs):
        data_calls.append(kwargs["bits_seeds"])
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(scan, "add_received_noise", add)
    monkeypatch.setattr(scan, "generate_master_qpsk", generate)
    rows, frames, _ = scan.run_scan(num_frames=1)
    assert len(data_calls) == 5 and len(noises) == 20
    for start in range(0, 20, 4):
        assert all(noises[start] is noise for noise in noises[start:start + 4])
        assert len({(r["bits_seed"], r["bits_seed_tx1"], r["noise_seed"]) for r in frames[start:start + 4]}) == 1
    assert len(rows) == 20


def test_noisy_pilot_ls_is_not_artificially_exact(fixed_link):
    _, h, layout, y, noise = fixed_link
    h_ls, _ = link.estimate_ls_linear_channel(link.add_received_noise(y, noise), layout)
    mask = np.broadcast_to(layout.pilot_mask[None], h.shape)
    assert np.linalg.norm(h_ls[mask] - h[mask]) / np.linalg.norm(h[mask]) > 1e-3
    # On unit-modulus owner pilots, the exact error is N/p.
    for q in range(2):
        selected = layout.pilot_mask[q]
        np.testing.assert_allclose(h_ls[:, q, selected] - h[:, q, selected],
                                   noise[:, selected] / link.PILOT_SYMBOLS[q], rtol=0, atol=1e-15)


def test_ber_only_uses_data_mask(fixed_link):
    bits, h, layout, y, noise = fixed_link
    _, _, perfect, estimated = link.receive_perfect_and_ls_linear(link.add_received_noise(y, noise), h, layout)
    reference = link.extract_reference_data_bits(bits, layout.data_mask)
    for recovered in (perfect, estimated):
        recovered[~layout.data_mask] = np.nan + 1j * np.nan
        actual = link.demodulate_data_bits(np.sqrt(2) * recovered, layout.data_mask)
        assert sum(stream.size for stream in actual) == sum(stream.size for stream in reference) == 3584


def test_zero_noise_regression_matches_t06b():
    path = Path(__file__).resolve().parents[1] / "outputs/t06b_noiseless_full_link_results.json"
    result = scan.zero_noise_regression(json.loads(path.read_text(encoding="utf-8")))
    assert result["passed"]
    assert [row["ls_lin_bit_errors"] for row in result["conditions"]] == [0, 0, 417, 841]


def test_noise_empirical_power():
    noise = complex_gaussian_noise((200, 2, 64, 16), 0.1, np.random.default_rng(30261001))
    assert abs(np.mean(np.abs(noise) ** 2) - 0.1) / 0.1 <= 0.02


def test_scan_has_20_conditions():
    coordinates = list(scan.scan_schedule())
    assert len({(snr, df) for snr, _, df in coordinates}) == 20


def test_frame_metrics_has_expected_rows():
    coordinates = list(scan.scan_schedule())
    assert len(coordinates) == len(set(coordinates)) == 4000
    assert all(sum(1 for s, _, d in coordinates if s == snr and d == df) == 200
               for snr in scan.SNR_POINTS for df in scan.DF_OPTIONS)


def test_seed_policy_has_no_collisions():
    seeds = [scan.frame_seeds(s, f) for s in range(5) for f in range(200)]
    bits = [seed for pair, _ in seeds for seed in pair]
    noise = [seed for _, seed in seeds]
    assert len(set(bits)) == 2000 and len(set(noise)) == 1000
    assert not set(bits) & set(noise)


def test_small_scan_uses_actual_frame_counts():
    rows, frames, _ = scan.run_scan(num_frames=2)
    assert len(rows) == 20 and len(frames) == 2 * 5 * 4
    for row in rows:
        subset = [r for r in frames if (r["dF"], r["snr_db"]) == (row["dF"], row["snr_db"])]
        assert row["num_frames"] == 2
        assert row["total_data_bits"] == sum(r["data_bits"] for r in subset)
        assert row["nmse_data_linear"] == pytest.approx(sum(r["nmse_error_energy"] for r in subset) / sum(r["nmse_truth_energy"] for r in subset))
        assert row["perfect_csi_ber"] == sum(r["perfect_csi_bit_errors"] for r in subset) / row["total_data_bits"]


@pytest.fixture(scope="module")
def saved_scan():
    root = Path(__file__).resolve().parents[1] / "outputs"
    with (root / "t07a_awgn_pilot_snr_scan.csv").open(newline="", encoding="utf-8") as handle:
        conditions = list(csv.DictReader(handle))
    with (root / "t07a_awgn_frame_metrics.csv").open(newline="", encoding="utf-8") as handle:
        frames = list(csv.DictReader(handle))
    return conditions, frames


def test_saved_scan_has_20_conditions_and_4000_frame_rows(saved_scan):
    conditions, frames = saved_scan
    assert len(conditions) == 20 and len(frames) == 4000
    assert len({(r["snr_db"], r["frame_index"], r["dF"]) for r in frames}) == 4000
    for row in conditions:
        assert int(row["num_frames"]) == 200
        assert int(row["total_data_bits"]) == 200 * int(row["data_bits_per_frame"])
        assert row["all_checks_passed"] == "True"


def test_saved_aggregation_recomputed_from_frame_rows(saved_scan):
    conditions, frames = saved_scan
    for row in conditions:
        subset = [f for f in frames if (f["dF"], f["snr_db"]) == (row["dF"], row["snr_db"])]
        assert len(subset) == 200
        error = sum(float(f["nmse_error_energy"]) for f in subset)
        truth = sum(float(f["nmse_truth_energy"]) for f in subset)
        bits = sum(int(f["data_bits"]) for f in subset)
        assert float(row["nmse_error_energy"]) == pytest.approx(error, rel=1e-12)
        assert float(row["nmse_truth_energy"]) == pytest.approx(truth, rel=1e-12)
        assert float(row["nmse_data_linear"]) == pytest.approx(error / truth, rel=1e-12)
        assert float(row["nmse_data_db"]) == pytest.approx(10 * np.log10(error / truth))
        for prefix in ("perfect_csi", "ls_lin"):
            errors = sum(int(f[f"{prefix}_bit_errors"]) for f in subset)
            assert int(row[f"{prefix}_bit_errors"]) == errors
            assert float(row[f"{prefix}_ber"]) == errors / bits


def test_saved_seed_policy_is_paired_and_reproducible(saved_scan):
    _, frames = saved_scan
    pairs = {}
    for row in frames:
        key = (int(row["snr_db"]), int(row["frame_index"]))
        seeds = (int(row["bits_seed"]), int(row["bits_seed_tx1"]), int(row["noise_seed"]))
        expected_bits, expected_noise = scan.frame_seeds(scan.SNR_POINTS.index(key[0]), key[1])
        assert seeds == (*expected_bits, expected_noise)
        pairs.setdefault(key, []).append(seeds)
    assert len(pairs) == 1000
    assert all(len(values) == 4 and len(set(values)) == 1 for values in pairs.values())
