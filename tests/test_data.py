import pandas as pd
import pytest

from credit_risk import data
from credit_risk import features as F


def test_parse_emp_length():
    s = pd.Series(["< 1 year", "1 year", "5 years", "10+ years", None])
    out = data.parse_emp_length(s).tolist()
    assert out[:4] == [0, 1, 5, 10]
    assert pd.isna(out[4])


def test_add_engineered_features():
    df = pd.DataFrame({
        "term": [" 36 months", " 60 months"],
        "emp_length": ["2 years", None],
        "earliest_cr_line": ["Jan-2000", "Jun-2010"],
        "issue_d": pd.to_datetime(["2013-03-01", "2014-06-01"]),
    })
    out = data.add_engineered_features(df)
    assert out["term_months"].tolist() == [36, 60]
    assert out["emp_length_years"].iloc[0] == 2
    assert pd.isna(out["emp_length_years"].iloc[1])
    assert out["credit_history_months"].tolist() == [158, 48]


def test_load_raw_reads_only_used_columns_and_drops_footer(tmp_path):
    cols = F.RAW_COLUMNS_USED + list(F.EXCLUDED)
    rows = [{c: "" for c in cols} for _ in range(2)]
    rows[0]["issue_d"] = "Dec-2015"
    rows[0]["loan_amnt"] = "1000"
    path = tmp_path / "raw.csv"
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)

    df = data.load_raw(path)
    assert len(df) == 1
    assert set(df.columns) == set(F.RAW_COLUMNS_USED)
    assert df["issue_d"].iloc[0] == pd.Timestamp("2015-12-01")


def test_load_raw_rejects_unaccounted_column(tmp_path):
    cols = F.RAW_COLUMNS_USED + list(F.EXCLUDED) + ["surprise"]
    path = tmp_path / "raw.csv"
    pd.DataFrame(columns=cols).to_csv(path, index=False)
    with pytest.raises(ValueError, match="surprise"):
        data.load_raw(path)
