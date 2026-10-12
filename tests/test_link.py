"""T06B integration checks with fixed channels, data, and four actual masks."""

import inspect

import numpy as np
import pytest

import src.link as link
from src.pilots import build_fdm_pilot_layout, map_pilot_symbols


@pytest.fixture(scope="module")
def calibration():
    bits, s = link.generate_master_qpsk()
    impulses, h = link.build_calibration_mimo_channel()
    return bits, s, impulses, h


@pytest.fixture(params=[2, 4, 8, 16])
def frame(request, calibration):
    bits, s, _, h = calibration
    layout = build_fdm_pilot_layout(2, 64, 16, request.param, 4)
    x = link.assemble_pilot_data_frame(s, layout)
    y = link.apply_frequency_mimo_channel(h, x)
    return request.param, bits, s, h, layout, x, y


def test_master_data_reproducible(calibration):
    bits, s, _, _ = calibration
    repeated_bits, repeated_s = link.generate_master_qpsk()
    np.testing.assert_array_equal(bits, repeated_bits)
    np.testing.assert_array_equal(s, repeated_s)
    assert bits.shape == (2, 64, 16, 2)
    for q, seed in enumerate((20260930, 20260931)):
        expected = np.random.default_rng(seed).integers(0, 2, 2048, dtype=np.uint8)
        np.testing.assert_array_equal(bits[q].ravel(), expected)
    assert not np.array_equal(bits[0], bits[1])


def test_data_re_power(frame):
    _, _, s, _, layout, x, _ = frame
    np.testing.assert_allclose(x[layout.data_mask], s[layout.data_mask] / np.sqrt(2), atol=1e-15, rtol=0)
    np.testing.assert_allclose(np.sum(np.abs(x) ** 2, axis=0)[~layout.pilot_union], 1, atol=1e-15, rtol=0)


def test_pilot_re_power(frame):
    _, _, _, _, layout, x, _ = frame
    np.testing.assert_allclose(np.sum(np.abs(x) ** 2, axis=0)[layout.pilot_union], 1, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(x[layout.mute_mask], 0)
    for q in range(2):
        np.testing.assert_array_equal(x[q][layout.pilot_mask[q]], link.PILOT_SYMBOLS[q])


def test_no_data_on_pilot_union(frame):
    _, _, _, _, layout, _, _ = frame
    assert not np.any(layout.data_mask[:, layout.pilot_union])
    assert np.all(layout.data_mask == (~layout.pilot_union)[None])
    assert np.all(layout.pilot_mask.astype(int) + layout.mute_mask + layout.data_mask == 1)


def test_four_link_frequency_selective_channel_shapes(calibration):
    _, _, impulses, h = calibration
    assert impulses.shape == (2, 2, 6)
    assert h.shape == (2, 2, 64, 16) and h.dtype == np.complex128
    np.testing.assert_array_equal(h, np.repeat(h[..., :1], 16, axis=-1))
    for r in range(2):
        for q in range(2):
            assert np.ptp(np.abs(h[r, q, :, 0])) > 0.1
            expected = np.fft.fftshift(np.fft.fft(impulses[r, q], n=64))
            np.testing.assert_array_equal(h[r, q, :, 0], expected)
    assert len({impulse.tobytes() for impulse in impulses.reshape(4, 6)}) == 4


def test_each_link_average_frequency_power(calibration):
    _, _, impulses, h = calibration
    np.testing.assert_allclose(np.sum(np.abs(impulses) ** 2, axis=-1), 1, atol=1e-15, rtol=0)
    np.testing.assert_allclose(np.mean(np.abs(h[..., 0]) ** 2, axis=-1), 1, atol=1e-15, rtol=0)


def test_all_frequency_mimo_matrices_full_rank(calibration):
    h = calibration[-1]
    assert all(np.linalg.matrix_rank(h[:, :, k, 0]) == 2 for k in range(64))


def test_perfect_csi_zero_bit_errors_all_df(frame):
    _, bits, _, h, layout, x, y = frame
    detected = link.detect_data_zf(y, h, layout.data_mask)
    assert np.linalg.norm(detected[layout.data_mask] - x[layout.data_mask]) / np.linalg.norm(x[layout.data_mask]) <= 1e-12
    recovered = link.demodulate_data_bits(np.sqrt(2) * detected, layout.data_mask)
    expected = link.extract_reference_data_bits(bits, layout.data_mask)
    for actual, reference in zip(recovered, expected):
        np.testing.assert_array_equal(actual, reference)


def test_noiseless_ls_exact_on_pilots(frame):
    _, _, _, h, layout, _, y = frame
    h_ls, _ = link.estimate_ls_linear_channel(y, layout)
    mask = np.broadcast_to(layout.pilot_mask[None], h.shape)
    assert np.linalg.norm(h_ls[mask] - h[mask]) / np.linalg.norm(h[mask]) <= 1e-12


def test_ls_linear_full_grid_finite_all_df(frame):
    _, _, _, h, layout, _, y = frame
    _, h_est = link.estimate_ls_linear_channel(y, layout)
    assert h_est.shape == h.shape and np.all(np.isfinite(h_est))


def test_nmse_data_finite_nonnegative(frame):
    _, _, _, h, layout, _, y = frame
    _, h_est = link.estimate_ls_linear_channel(y, layout)
    nmse = link.data_channel_nmse(h_est, h, layout.data_mask)
    assert np.isfinite(nmse) and nmse >= 0


def test_ber_counts_only_data_mask(frame):
    df, bits, s, h, layout, _, y = frame
    _, h_est = link.estimate_ls_linear_channel(y, layout)
    detected = link.detect_data_zf(y, h_est, layout.data_mask)
    actual = link.demodulate_data_bits(np.sqrt(2) * detected, layout.data_mask)
    reference = link.extract_reference_data_bits(bits, layout.data_mask)
    assert sum(stream.size for stream in actual) == {2: 3072, 4: 3584, 8: 3840, 16: 3968}[df]
    assert sum(stream.size for stream in reference) == 2 * np.count_nonzero(layout.data_mask)
    # Changing every excluded symbol/dibit cannot change the extracted data bits.
    changed_s, changed_bits = s.copy(), bits.copy()
    changed_s[~layout.data_mask] = np.nan + 1j * np.nan
    changed_bits[~layout.data_mask] ^= 1
    for before, after in zip(link.demodulate_data_bits(s, layout.data_mask), link.demodulate_data_bits(changed_s, layout.data_mask)):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(reference, link.extract_reference_data_bits(changed_bits, layout.data_mask)):
        np.testing.assert_array_equal(before, after)


def test_estimator_does_not_receive_true_h(frame, monkeypatch):
    _, _, _, _, layout, _, y = frame
    assert tuple(inspect.signature(link.estimate_ls_linear_channel).parameters) == ("y_grid", "layout", "pilot_symbols")
    real_ls = link.pilot_ls_estimate
    calls = []

    def checked_ls(received, known_pilots, mask):
        calls.append(True)
        assert received is y
        np.testing.assert_array_equal(known_pilots, map_pilot_symbols(layout, link.PILOT_SYMBOLS))
        assert not np.any(known_pilots[layout.data_mask])
        np.testing.assert_array_equal(mask, layout.pilot_mask)
        return real_ls(received, known_pilots, mask)

    monkeypatch.setattr(link, "pilot_ls_estimate", checked_ls)
    _, estimate = link.estimate_ls_linear_channel(y, layout)
    assert calls == [True]
    monkeypatch.setattr(link, "pilot_ls_estimate", real_ls)
    changed_y = y.copy()
    changed_y[:, ~layout.pilot_union] = 123 + 456j
    _, changed_estimate = link.estimate_ls_linear_channel(changed_y, layout)
    np.testing.assert_array_equal(estimate, changed_estimate)


def test_data_nmse_excludes_pilot_union(frame):
    _, _, _, h, layout, _, _ = frame
    estimate = h.copy()
    estimate[:, ~layout.data_mask] += 100
    assert link.data_channel_nmse(estimate, h, layout.data_mask) == 0
    estimate = 2 * h
    assert link.data_channel_nmse(estimate, h, layout.data_mask) == pytest.approx(1)


def test_channel_application_manual_re(frame):
    _, _, _, h, _, x, y = frame
    np.testing.assert_allclose(y[:, 1, 0], h[:, :, 1, 0] @ x[:, 1, 0], atol=1e-15, rtol=0)


def test_master_data_paired_across_layouts(calibration):
    _, s, _, _ = calibration
    layouts = [build_fdm_pilot_layout(2, 64, 16, df, 4) for df in (2, 4, 8, 16)]
    frames = [link.assemble_pilot_data_frame(s, layout) for layout in layouts]
    for layout, frame in zip(layouts[1:], frames[1:]):
        common = layouts[0].data_mask & layout.data_mask
        np.testing.assert_array_equal(frames[0][common], frame[common])
