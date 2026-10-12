import numpy as np
import pytest

from src.metrics import MetricAccumulator, channel_error_energies, count_bit_errors


def test_nmse_aggregate_energy_formula():
    metrics = MetricAccumulator()
    metrics.add(1, 1, 0, 0, 2)
    metrics.add(9, 100, 0, 0, 8)
    result = metrics.finalize()
    assert result["nmse_data_linear"] == pytest.approx(10 / 101)
    assert result["nmse_data_db"] == pytest.approx(10 * np.log10(10 / 101))
    assert result["nmse_data_linear"] != pytest.approx((1 + 9 / 100) / 2)


def test_ber_aggregate_count_formula():
    metrics = MetricAccumulator()
    metrics.add(1, 1, 1, 2, 2)
    metrics.add(1, 1, 0, 1, 8)
    result = metrics.finalize()
    assert result["total_data_bits"] == 10
    assert result["perfect_csi_ber"] == 0.1
    assert result["ls_lin_ber"] == 0.3
    assert result["perfect_csi_ber"] != (0.5 + 0) / 2


def test_channel_energies_only_use_data_mask():
    truth = np.ones((2, 2, 3, 1), dtype=np.complex128)
    estimate = truth.copy()
    mask = np.ones((2, 3, 1), dtype=bool)
    mask[:, 0] = False
    estimate[:, :, 0] = np.nan + 1j * np.nan
    estimate[:, :, 1:] += 1j
    assert channel_error_energies(estimate, truth, mask) == (8, 8)


def test_bit_error_actual_counts():
    assert count_bit_errors((np.array([1, 0]), np.array([1])),
                            (np.array([0, 0]), np.array([1]))) == (1, 3)
    with pytest.raises(ValueError, match="matching"):
        count_bit_errors((np.array([0]),), (np.array([0, 1]),))


@pytest.mark.parametrize("values", [(np.nan, 1, 0, 0, 2), (1, 0, 0, 0, 2),
                                     (1, 1, 3, 0, 2), (1, 1, 0, 0, 0)])
def test_invalid_accumulation_rejected(values):
    with pytest.raises(ValueError):
        MetricAccumulator().add(*values)


def test_empty_and_zero_error_accumulator():
    with pytest.raises(ValueError, match="empty"):
        MetricAccumulator().finalize()
    metrics = MetricAccumulator()
    metrics.add(0, 1, 0, 0, 10)
    result = metrics.finalize()
    assert result["nmse_data_linear"] == 0
    assert result["nmse_data_db"] == float("-inf")
    assert result["perfect_csi_ber"] == 0


def test_zero_ber_plotting_preserves_raw_values():
    from scripts.plot_awgn_pilot_snr_scan import ber_plot_values
    raw = np.array([0.0, 0.1])
    plotted, zero = ber_plot_values(raw, np.array([100, 200]))
    np.testing.assert_array_equal(raw, [0, 0.1])
    np.testing.assert_array_equal(plotted, [0.005, 0.1])
    np.testing.assert_array_equal(zero, [True, False])
