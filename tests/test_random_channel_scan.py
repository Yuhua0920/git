import csv
import json
from pathlib import Path

import numpy as np
import pytest

import src.stochastic_scan as scan
from src.awgn_scan import DF_OPTIONS, SNR_POINTS


def test_channel_bits_noise_seed_separation():
    allocated = [scan.trajectory_seeds(t, s) for t in range(200) for s in range(5)]
    channels = {a for a, _, _ in allocated}
    bits = {seed for _, pair, _ in allocated for seed in pair}
    noise = {seed for _, _, seed in allocated}
    assert len(channels) == 200 and len(bits) == 400 and len(noise) == 1000
    assert not channels & bits and not channels & noise and not bits & noise


def test_paired_channel_data_and_noise_across_df(monkeypatch):
    builds, generated, channels, noises, frames = [], [], [], [], []
    real_build, real_data = scan.build_random_mimo_channel, scan.generate_master_qpsk
    real_apply, real_add = scan.apply_frequency_mimo_channel, scan.add_received_noise

    def build(seed):
        builds.append(seed)
        return real_build(seed)

    def data(*args, **kwargs):
        generated.append(kwargs["bits_seeds"])
        return real_data(*args, **kwargs)

    def apply(h, x):
        channels.append(h)
        frames.append(x.copy())
        return real_apply(h, x)

    def add(y, noise):
        noises.append(noise)
        return real_add(y, noise)

    monkeypatch.setattr(scan, "build_random_mimo_channel", build)
    monkeypatch.setattr(scan, "generate_master_qpsk", data)
    monkeypatch.setattr(scan, "apply_frequency_mimo_channel", apply)
    monkeypatch.setattr(scan, "add_received_noise", add)
    rows, trajectories, _ = scan.run_random_scan(num_trajectories=1)
    assert len(builds) == len(generated) == 1
    assert len(rows) == len(trajectories) == 20
    assert all(channel is channels[0] for channel in channels)
    for start in range(0, 20, 4):
        assert all(noise is noises[start] for noise in noises[start:start + 4])
        for offset, df in enumerate(DF_OPTIONS):
            layout = scan.build_fdm_pilot_layout(2, 64, 16, df, 4)
            common = scan.build_fdm_pilot_layout(2, 64, 16, 2, 4).data_mask & layout.data_mask
            np.testing.assert_array_equal(frames[start][common], frames[start + offset][common])


def test_noise_variance_independent_of_channel_power(monkeypatch):
    real_build, real_noise = scan.build_random_mimo_channel, scan.complex_gaussian_noise
    variances = []

    def noise(shape, variance, rng):
        variances.append(variance)
        return real_noise(shape, variance, rng)

    monkeypatch.setattr(scan, "complex_gaussian_noise", noise)
    scan.run_random_scan(num_trajectories=1)
    first = variances.copy()
    variances.clear()

    def scaled(seed):
        impulse, h = real_build(seed)
        return 10 * impulse, 10 * h

    monkeypatch.setattr(scan, "build_random_mimo_channel", scaled)
    scan.run_random_scan(num_trajectories=1)
    np.testing.assert_array_equal(variances, first)
    np.testing.assert_allclose(first, 10.0 ** (-np.asarray(SNR_POINTS) / 10), rtol=1e-15)


def test_condition_number_never_filters_invertible_channel(monkeypatch):
    def ill_conditioned(seed):
        impulse = np.zeros((2, 2, 6), dtype=np.complex128)
        impulse[:, :, 0] = [[1, 1], [1, 1 + 1e-8]]
        h = np.repeat(np.repeat(impulse[:, :, :1], 64, axis=2)[..., None], 16, axis=-1)
        return impulse, h

    monkeypatch.setattr(scan, "build_random_mimo_channel", ill_conditioned)
    rows, trajectories, metadata = scan.run_random_scan(num_trajectories=1)
    assert len(rows) == len(trajectories) == 20
    assert metadata["condition_number_statistics"]["max"] > 1e8
    assert metadata["filtered_trajectories"] == 0
    assert all(np.isfinite(r["nmse_data_linear"]) for r in rows)


def test_singular_true_channel_records_seed_and_frequency(monkeypatch):
    real_build = scan.build_random_mimo_channel

    def singular(seed):
        impulse, h = real_build(seed)
        h[:, :, 7, :] = np.array([[1, 2], [2, 4]])[:, :, None]
        return impulse, h

    monkeypatch.setattr(scan, "build_random_mimo_channel", singular)
    with pytest.raises(scan.ScanFailure) as caught:
        scan.run_random_scan(num_trajectories=2)
    record = caught.value.record
    assert record["trajectory_id"] == 0 and record["k"] == 7
    assert record["channel_seed"] == 40261001
    assert record["stage"] == "true_channel_rank" and record["completed_evaluations"] == 0


def test_solve_failure_records_original_re_and_stops(monkeypatch):
    def fail(*args):
        raise ValueError("singular channel at RE (k=5, t=0)")

    monkeypatch.setattr(scan, "receive_perfect_and_ls_linear", fail)
    with pytest.raises(scan.ScanFailure) as caught:
        scan.run_random_scan(num_trajectories=2)
    record = caught.value.record
    expected = np.argwhere(scan.build_fdm_pilot_layout(2, 64, 16, 2, 4).data_mask[0])[5]
    assert [record["k"], record["t"]] == expected.tolist()
    assert record["channel_seed"] == 40261001 and record["noise_seed"] == 30261001
    assert record["completed_evaluations"] == 0


def test_correct_bit_rate_proxy_formula():
    duration = scan.frame_duration_seconds()
    assert duration == pytest.approx(0.0006666666666666666)
    correct, rate = scan.correct_bit_rate_proxy(3584, 0.1, duration)
    assert correct == pytest.approx(3225.6)
    assert rate == pytest.approx(4838400)


def test_small_scan_aggregation_no_trajectory_missing():
    rows, trajectories, _ = scan.run_random_scan(num_trajectories=2)
    assert len(rows) == 20 and len(trajectories) == 40
    assert scan.verify_trajectory_aggregation(rows, trajectories, 2)
    assert {r["trajectory_id"] for r in trajectories} == {0, 1}


def test_existing_fixed_channel_regressions_preserved():
    root = Path(__file__).resolve().parents[1] / "outputs"
    baseline = json.loads((root / "t06b_noiseless_full_link_results.json").read_text(encoding="utf-8"))
    with (root / "t07a_awgn_frame_metrics.csv").open(encoding="utf-8", newline="") as handle:
        frames = list(csv.DictReader(handle))
    assert scan.fixed_channel_regressions(baseline, frames)["passed"]


def test_random_noiseless_perfect_reference():
    _, h = scan.build_random_mimo_channel(40261001)
    bits, s = scan.generate_master_qpsk(bits_seeds=(20261001, 20261002))
    layout = scan.build_fdm_pilot_layout(2, 64, 16, 4, 4)
    x = scan.assemble_pilot_data_frame(s, layout)
    y = scan.apply_frequency_mimo_channel(h, x)
    _, _, recovered, _ = scan.receive_perfect_and_ls_linear(y, h, layout)
    assert np.linalg.norm(recovered[layout.data_mask] - x[layout.data_mask]) / np.linalg.norm(x[layout.data_mask]) <= 1e-12
    errors, _ = scan.count_bit_errors(scan.demodulate_data_bits(np.sqrt(2) * recovered, layout.data_mask),
                                     scan.extract_reference_data_bits(bits, layout.data_mask))
    assert errors == 0


@pytest.fixture(scope="module")
def saved_random_scan():
    root = Path(__file__).resolve().parents[1] / "outputs"
    integer_fields = {"trajectory_id", "channel_seed", "snr_db", "dF", "num_trajectories", "bits_seed_tx0",
                      "bits_seed_tx1", "noise_seed", "data_bits", "data_bits_per_frame", "total_data_bits",
                      "perfect_csi_bit_errors", "ls_lin_bit_errors"}

    def read(name):
        with (root / name).open(encoding="utf-8", newline="") as handle:
            return [{key: (value == "True") if key == "all_checks_passed" else
                     int(value) if key in integer_fields else float(value) for key, value in row.items()}
                    for row in csv.DictReader(handle)]

    return (read("t07b_random_channel_scan.csv"), read("t07b_random_channel_trajectory_metrics.csv"),
            json.loads((root / "t07b_random_channel_scan_results.json").read_text(encoding="utf-8")))


def test_saved_200_trajectories_4000_rows_20_conditions(saved_random_scan):
    rows, trajectories, metadata = saved_random_scan
    assert len(rows) == 20 and len(trajectories) == 4000
    assert {r["trajectory_id"] for r in trajectories} == set(range(200))
    assert all(sum(r["trajectory_id"] == tid for r in trajectories) == 20 for tid in range(200))
    assert metadata["filtered_trajectories"] == 0 and metadata["numerical_failures"] == []
    for row in trajectories:
        channel, bits, noise = scan.trajectory_seeds(row["trajectory_id"], SNR_POINTS.index(row["snr_db"]))
        assert (row["channel_seed"], row["bits_seed_tx0"], row["bits_seed_tx1"], row["noise_seed"]) == (channel, *bits, noise)


def test_saved_aggregation_and_proxy_reconstructed(saved_random_scan):
    rows, trajectories, _ = saved_random_scan
    assert scan.verify_trajectory_aggregation(rows, trajectories)
    assert all(all(np.isfinite(value) for value in row.values()) for row in (*rows, *trajectories))


def test_saved_channel_statistics_count_each_trajectory_once(saved_random_scan):
    _, trajectories, metadata = saved_random_scan
    unique = {r["trajectory_id"]: r for r in trajectories}
    for r in range(2):
        for q in range(2):
            powers = np.array([row[f"channel_power_rx{r}_tx{q}"] for row in unique.values()])
            stats = metadata["channel_power_statistics"][f"rx{r}_tx{q}"]
            np.testing.assert_allclose([stats["mean"], stats["std"], stats["min"], stats["max"]],
                                       [np.mean(powers), np.std(powers, ddof=1), np.min(powers), np.max(powers)], rtol=1e-12)
            assert np.std(powers) > 0.1 and not np.allclose(powers, 1)
            for tid in range(200):
                values = {row[f"channel_power_rx{r}_tx{q}"] for row in trajectories if row["trajectory_id"] == tid}
                assert len(values) == 1


def test_saved_condition_percentiles_reproducible(saved_random_scan):
    metadata = saved_random_scan[2]
    conditions = []
    for tid in range(200):
        _, h = scan.build_random_mimo_channel(40261001 + tid)
        conditions.extend(np.linalg.cond(h[:, :, k, 0]) for k in range(64))
    stats = metadata["condition_number_statistics"]
    assert stats["bin_count"] == len(conditions) == 12800
    for percentile in (50, 90, 99):
        assert stats[f"P{percentile}"] == pytest.approx(np.percentile(conditions, percentile), rel=1e-12)
    assert stats["max"] == pytest.approx(max(conditions), rel=1e-12)
