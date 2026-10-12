import pytest

from credit_risk import features as F

# Columns that leak the outcome; none may ever be a model input.
LEAKY = {
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_amnt", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low", "funded_amnt", "funded_amnt_inv",
    "loan_status",
}


def test_no_leaky_feature_used():
    assert not LEAKY & set(F.MODEL_FEATURES)
    assert not LEAKY & set(F.SOURCE_COLUMNS)


def test_used_and_excluded_are_disjoint():
    assert not set(F.RAW_COLUMNS_USED) & set(F.EXCLUDED)


def test_spec_covers_all_151_raw_columns():
    assert len(set(F.RAW_COLUMNS_USED) | set(F.EXCLUDED)) == 151


def test_every_exclusion_has_a_reason():
    assert all(isinstance(r, str) and r.strip() for r in F.EXCLUDED.values())


def test_no_duplicate_features():
    assert len(F.MODEL_FEATURES) == len(set(F.MODEL_FEATURES))


def test_check_columns_rejects_unknown_and_missing():
    header = F.RAW_COLUMNS_USED + list(F.EXCLUDED)
    F.check_columns(header)
    with pytest.raises(ValueError, match="new_col"):
        F.check_columns(header + ["new_col"])
    with pytest.raises(ValueError, match="loan_amnt"):
        F.check_columns([c for c in header if c != "loan_amnt"])
