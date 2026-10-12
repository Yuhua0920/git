"""Small formatting calibration precedes actual frozen-evidence table build."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_paper_tables import metric_table, number, longtable, endpoint_rows


@pytest.mark.parametrize("value,metric,expected", [(None, "nmse_db", "N/A"), ("", "nmse_db", "N/A"),
    ("-22.08329845911522", "nmse_db", "-22.083"), ("0.05214083501344086", "ber", "0.05214")])
def test_frozen_number_format(value, metric, expected):
    assert number(value, metric) == expected


def test_missing_perfect_csi_db_is_not_finite_or_invented():
    row = {"environment": "CDL-C-30-100", "snr_db": "10", "method": "Perfect CSI", "nmse_db": ""}
    ci = {("CDL-C-30-100", 10, "Perfect CSI", "nmse_db"): {"ci95_lower": "", "ci95_upper": ""}}
    rows, cells = metric_table([row], ci, "nmse_db")
    assert rows[0][-2:] == ("N/A", "N/A")
    assert cells[0]["raw_value"] == ""


def test_missing_interval_raises_instead_of_silent_drop():
    with pytest.raises(KeyError):
        metric_table([{"environment": "x", "snr_db": "10", "method": "Source LMMSE"}], {}, "ber")


def test_longtable_small_calibration_retains_all_rows():
    text = longtable(["A", "B"], [("one", "2"), ("three", "4")], "Test", "tab:test", "lr", "Note")
    assert "one & 2" in text and "three & 4" in text
    assert text.count(r"\end{longtable}") == 1


@pytest.mark.parametrize("family,bound,rule", [("A", "6.455", "lower > 0"),
                                             ("B", "6.455", "lower > 0"),
                                             ("C", "6.891", "upper < 0.50")])
def test_endpoint_uses_registered_direction_when_both_bounds_exist(family, bound, rule):
    row = {"primary_endpoint": family, "primary_lower": "6.455461391885126",
           "primary_upper": "6.891308696159531", "environment": "CDL-B-30-100",
           "snr_db": "10", "estimate": "6.672091949268225", "endpoint_cell_pass": "True"}
    assert endpoint_rows([row])[0][4:6] == (bound, rule)
