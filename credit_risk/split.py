"""Out-of-time train/test split by issue date."""

import pandas as pd

from credit_risk.features import DATE_COLUMN
from credit_risk.target import TARGET

# Most bureau attributes (tot_cur_bal, mort_acc, num_*, ...) are 100% missing
# before mid-2012, so the modelling window starts in 2013.
WINDOW_START = pd.Timestamp("2013-01-01")
TEST_START = pd.Timestamp("2015-01-01")


def time_split(df, window_start=WINDOW_START, test_start=TEST_START):
    """Train on loans issued in [window_start, test_start), test on loans issued from test_start."""
    if test_start <= window_start:
        raise ValueError("test_start must be after window_start")
    issue = df[DATE_COLUMN]
    train = df[(issue >= window_start) & (issue < test_start)].reset_index(drop=True)
    test = df[issue >= test_start].reset_index(drop=True)
    return train, test


def summarize(df):
    issue = df[DATE_COLUMN]
    return {
        "issue_d_min": issue.min().strftime("%Y-%m"),
        "issue_d_max": issue.max().strftime("%Y-%m"),
        "rows": int(len(df)),
        "defaults": int(df[TARGET].sum()),
        "default_rate": round(float(df[TARGET].mean()), 4),
        "rows_by_term_months": {str(int(k)): int(v) for k, v in df["term_months"].value_counts().sort_index().items()},
    }
