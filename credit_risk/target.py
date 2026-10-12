"""Target definition: default = charged off, on loans whose outcome is final."""

import pandas as pd

from credit_risk.features import DATE_COLUMN, TARGET_COLUMN

TARGET = "default"

DEFAULT_STATUSES = {"Charged Off", "Default"}
NON_DEFAULT_STATUSES = {"Fully Paid"}
# Everything else is dropped: "Current", "In Grace Period" and "Late (...)" are
# unresolved; "Does not meet the credit policy. Status:..." loans were issued
# under an older policy (2007-2010) and are not comparable.

# The latest payment month in accepted_2007_to_2018Q4.csv is Mar-2019. This
# fixes the data cut-off; it is not used as a feature.
SNAPSHOT_DATE = pd.Timestamp("2019-03-01")
# Loans need their full term plus this buffer before the snapshot so that
# late final payments and charge-offs have been recorded.
MATURITY_BUFFER_MONTHS = 3


def is_matured(df, snapshot=SNAPSHOT_DATE, buffer_months=MATURITY_BUFFER_MONTHS):
    """True when issue date + term + buffer is on or before the snapshot.

    Without this, recent cohorts only contain loans that resolved early, which
    are disproportionately early defaults and early payoffs.
    """
    issue = df[DATE_COLUMN]
    end_month = issue.dt.year * 12 + issue.dt.month + df["term_months"] + buffer_months
    # A missing term gives NaN, which compares False: never treated as matured.
    return end_month <= snapshot.year * 12 + snapshot.month


def build_target(df, snapshot=SNAPSHOT_DATE, buffer_months=MATURITY_BUFFER_MONTHS):
    """Keep matured loans with a final status and add the binary target.

    Returns the filtered frame and a dict of row counts for each step.
    """
    counts = {"rows_in": len(df)}
    matured = is_matured(df, snapshot, buffer_months)
    df = df[matured]
    counts["rows_matured"] = len(df)

    status = df[TARGET_COLUMN]
    final = status.isin(DEFAULT_STATUSES | NON_DEFAULT_STATUSES)
    counts["dropped_non_final_status"] = {
        str(k): int(v) for k, v in status[~final].value_counts(dropna=False).items()
    }
    df = df[final].copy()
    df[TARGET] = df[TARGET_COLUMN].isin(DEFAULT_STATUSES).astype(int)
    counts["rows_out"] = len(df)
    return df.reset_index(drop=True), counts
