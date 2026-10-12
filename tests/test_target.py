import pandas as pd

from credit_risk import target as T


def _frame(issue, term, status):
    return pd.DataFrame({
        "issue_d": pd.to_datetime(issue),
        "term_months": term,
        "loan_status": status,
    })


def test_is_matured_uses_term_and_buffer():
    df = _frame(
        ["2015-12-01", "2016-01-01", "2013-12-01", "2014-01-01"],
        [36, 36, 60, 60],
        ["Fully Paid"] * 4,
    )
    snapshot = pd.Timestamp("2019-03-01")
    assert T.is_matured(df, snapshot, 3).tolist() == [True, False, True, False]


def test_missing_term_is_never_matured():
    df = _frame(["2010-01-01"], [float("nan")], ["Fully Paid"])
    assert not T.is_matured(df).iloc[0]


def test_build_target_keeps_only_final_statuses():
    statuses = [
        "Fully Paid", "Charged Off", "Default", "Current", "In Grace Period",
        "Late (31-120 days)", "Does not meet the credit policy. Status:Charged Off",
    ]
    df = _frame(["2013-01-01"] * len(statuses), [36] * len(statuses), statuses)
    out, counts = T.build_target(df)
    assert out["loan_status"].tolist() == ["Fully Paid", "Charged Off", "Default"]
    assert out[T.TARGET].tolist() == [0, 1, 1]
    assert counts["rows_out"] == 3
    assert sum(counts["dropped_non_final_status"].values()) == 4


def test_build_target_drops_unmatured_loans():
    df = _frame(["2013-01-01", "2018-06-01"], [36, 36], ["Charged Off", "Charged Off"])
    out, counts = T.build_target(df)
    assert len(out) == 1
    assert counts["rows_matured"] == 1
