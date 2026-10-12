import pandas as pd
import pytest

from credit_risk import split as S


def _frame():
    dates = pd.to_datetime(["2012-06-01", "2013-01-01", "2014-12-01", "2015-01-01", "2015-12-01"])
    return pd.DataFrame({
        "issue_d": dates,
        "term_months": [36, 36, 60, 36, 36],
        "default": [0, 1, 0, 0, 1],
    })


def test_split_is_strictly_out_of_time():
    train, test = S.time_split(_frame())
    assert train["issue_d"].max() < test["issue_d"].min()
    assert train["issue_d"].min() >= S.WINDOW_START
    assert len(train) == 2 and len(test) == 2


def test_split_rejects_bad_window():
    with pytest.raises(ValueError):
        S.time_split(_frame(), pd.Timestamp("2015-01-01"), pd.Timestamp("2014-01-01"))


def test_summarize():
    train, _ = S.time_split(_frame())
    s = S.summarize(train)
    assert s["issue_d_min"] == "2013-01" and s["issue_d_max"] == "2014-12"
    assert s["rows"] == 2 and s["defaults"] == 1 and s["default_rate"] == 0.5
    assert s["rows_by_term_months"] == {"36": 1, "60": 1}
