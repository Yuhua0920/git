"""T06A Perfect-CSI, noiseless two-stream ZF calibration."""

import numpy as np
import pytest

from src.channels import static_mimo_receive
from src.detectors import zf_detect
from src.modulation import qpsk_demodulate, qpsk_modulate
from src.numerics import scale_data_streams


# H[r,q] maps TX q to RX r; both cross-stream coefficients are nonzero.
H = np.array([[0.8 + 0.2j, -0.3 + 0.5j], [0.4 - 0.1j, 0.6 + 0.7j]])
H_B = np.array([[1.1 - 0.3j, 0.2 + 0.4j], [-0.5 + 0.1j, 0.9 + 0.2j]])


@pytest.fixture
def link():
    bits = [np.random.default_rng(seed).integers(0, 2, 2048, dtype=np.uint8)
            for seed in (20260930, 20260931)]
    s_grid = np.stack([qpsk_modulate(stream).reshape(64, 16) for stream in bits])
    x_grid = scale_data_streams(s_grid.reshape(2, -1).T, 2).T.reshape(2, 64, 16)
    h_grid = np.broadcast_to(H[:, :, None, None], (2, 2, 64, 16))
    y_grid = static_mimo_receive(H, x_grid)
    return bits, s_grid, x_grid, y_grid, h_grid


def test_static_nondiagonal_mimo_zf_exact(link):
    _, _, x, y, h = link
    detected = zf_detect(y, h)
    assert detected.shape == (2, 64, 16)
    assert detected.dtype == np.complex128
    assert np.linalg.norm(detected - x) / np.linalg.norm(x) <= 1e-12


def test_zf_qpsk_zero_bit_errors(link):
    bits, s, x, y, h = link
    recovered_s = np.sqrt(2) * zf_detect(y, h)
    np.testing.assert_allclose(recovered_s, s, rtol=0, atol=1e-12)
    for q in range(2):
        recovered_bits = qpsk_demodulate(recovered_s[q].ravel())
        assert np.count_nonzero(recovered_bits != bits[q]) == 0


def test_two_stream_total_power(link):
    _, s, x, _, _ = link
    np.testing.assert_allclose(np.mean(np.abs(s) ** 2, axis=(1, 2)), [1, 1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.mean(np.abs(x) ** 2, axis=(1, 2)), [0.5, 0.5], rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.sum(np.abs(x) ** 2, axis=0), 1, rtol=0, atol=1e-15)


def test_manual_single_re_zf():
    # Reuse mapper: 00 -> (+1+j)/sqrt(2), 10 -> (-1+j)/sqrt(2).
    s = np.array([qpsk_modulate(np.array([0, 0]))[0], qpsk_modulate(np.array([1, 0]))[0]])
    x = s / np.sqrt(2)
    y = H @ x
    np.testing.assert_allclose(y, [0.2 + 0.1j, -0.4 + 0.1j], rtol=0, atol=1e-15)
    recovered = zf_detect(y[:, None, None], H[:, :, None, None])[:, 0, 0]
    np.testing.assert_allclose(recovered, x, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.sqrt(2) * recovered, s, rtol=0, atol=1e-15)


def test_nondiagonal_channel_required():
    assert H[0, 1] != 0 and H[1, 0] != 0
    assert np.linalg.matrix_rank(H) == 2


def test_grid_varying_h_uses_per_re_channel(link):
    _, _, x, _, _ = link
    h = np.empty((2, 2, 64, 16), dtype=np.complex128)
    for k in range(64):
        for t in range(16):
            h[:, :, k, t] = H if (k + t) % 2 == 0 else H_B
    y = np.einsum("rqkt,qkt->rkt", h, x)
    assert np.linalg.norm(zf_detect(y, h) - x) / np.linalg.norm(x) <= 1e-12
    wrong = zf_detect(y, np.broadcast_to(H[:, :, None, None], h.shape))
    assert np.linalg.norm(wrong - x) / np.linalg.norm(x) > 1e-2


def test_singular_h_rejected(link):
    _, _, _, y, h = link
    singular = h.copy()
    singular[:, :, 7, 3] = [[1, 2], [2, 4]]
    with pytest.raises(ValueError, match=r"singular channel.*k=7, t=3"):
        zf_detect(y, singular)


@pytest.mark.parametrize("case", ["y_rank", "h_rank", "nr", "k", "t", "nonsquare", "empty"])
def test_invalid_shapes_rejected(link, case):
    _, _, _, y, h = link
    if case == "y_rank":
        y = y[0]
    elif case == "h_rank":
        h = h[:, :, :, 0]
    elif case == "nr":
        y = y[:1]
    elif case == "k":
        y = y[:, :-1]
    elif case == "t":
        y = y[:, :, :-1]
    elif case == "nonsquare":
        h = h[:, :1]
    else:
        y = y[:, :0]
    with pytest.raises(ValueError):
        zf_detect(y, h)


@pytest.mark.parametrize("input_name", ["y", "h"])
@pytest.mark.parametrize("value", [np.nan + 0j, np.inf + 0j, complex(0, np.inf)])
def test_nonfinite_inputs_rejected(link, input_name, value):
    _, _, _, y, h = link
    y, h = y.copy(), h.copy()
    if input_name == "y":
        y[0, 0, 0] = value
    else:
        h[0, 0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        zf_detect(y, h)


def test_direct_y_is_not_same_as_transmitted_streams(link):
    bits, _, x, y, _ = link
    assert np.linalg.norm(y - x) / np.linalg.norm(x) > 1e-2
    direct_errors = sum(np.count_nonzero(qpsk_demodulate(y[q].ravel()) != bits[q]) for q in range(2))
    assert direct_errors > 0


def test_ill_conditioned_but_invertible_channel_runs():
    h = np.array([[1, 1], [1, 1 + 1e-8]], dtype=np.complex128)
    x = np.array([0.5 + 0.5j, -0.5 + 0.5j])
    assert np.linalg.cond(h) > 1e6
    recovered = zf_detect((h @ x)[:, None, None], h[:, :, None, None])
    assert np.all(np.isfinite(recovered))
