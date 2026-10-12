"""Load the raw Lending Club file and derive engineered application-time features."""

import pandas as pd

from credit_risk.features import DATE_COLUMN, RAW_COLUMNS_USED, check_columns

DATE_FORMAT = "%b-%Y"  # e.g. "Dec-2015"


def load_raw(path):
    """Read only the columns the model needs, after checking the full header is accounted for."""
    header = pd.read_csv(path, nrows=0).columns
    check_columns(header)
    df = pd.read_csv(path, usecols=RAW_COLUMNS_USED, low_memory=False)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format=DATE_FORMAT, errors="coerce")
    # The file ends with summary/footer rows that have no issue date.
    return df.dropna(subset=[DATE_COLUMN]).reset_index(drop=True)


def parse_emp_length(s):
    """'< 1 year' -> 0, '1 year' -> 1, ..., '10+ years' -> 10, missing stays missing."""
    s = s.astype("string").str.strip()
    s = s.str.replace("< 1", "0", regex=False)
    return pd.to_numeric(s.str.extract(r"(\d+)", expand=False), errors="coerce")


def add_engineered_features(df):
    df = df.copy()
    df["term_months"] = pd.to_numeric(
        df["term"].astype("string").str.extract(r"(\d+)", expand=False), errors="coerce"
    )
    df["emp_length_years"] = parse_emp_length(df["emp_length"])
    first_credit = pd.to_datetime(df["earliest_cr_line"], format=DATE_FORMAT, errors="coerce")
    issue = df[DATE_COLUMN]
    df["credit_history_months"] = (issue.dt.year - first_credit.dt.year) * 12 + (
        issue.dt.month - first_credit.dt.month
    )
    return df
