"""Feature specification.

Every column of the raw Lending Club file (accepted_2007_to_2018Q4.csv,
151 columns) is either used as a model input, used to derive one, used for
the target/split, or excluded with a reason. ``check_columns`` enforces
this against the actual file header so a new column cannot slip in silently.
"""

# Raw numeric columns used as-is. All are known when the application is
# decided: applicant-supplied, credit-bureau data pulled at application, or
# Lending Club's own pricing (int_rate, installment) set before issuance.
NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "collections_12_mths_ex_med",
    "mths_since_last_major_derog",
    "acc_now_delinq",
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "chargeoff_within_12_mths",
    "delinq_amnt",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "mths_since_recent_bc",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_inq",
    "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "pub_rec_bankruptcies",
    "tax_liens",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
]

# Numeric features derived from raw application-time columns (see data.py).
ENGINEERED_FEATURES = {
    "term_months": "term parsed to 36 or 60",
    "emp_length_years": "emp_length parsed to 0-10 ('< 1 year' -> 0, '10+ years' -> 10)",
    "credit_history_months": "months between earliest_cr_line and issue_d",
}

CATEGORICAL_FEATURES = [
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
]

MODEL_FEATURES = NUMERIC_FEATURES + list(ENGINEERED_FEATURES) + CATEGORICAL_FEATURES

# Lending Club's own risk pricing. Known before issuance, but it encodes LC's
# internal model; the sensitivity run drops it to measure what the model adds.
# installment is a function of loan_amnt, term and int_rate, so it goes too.
PRICING_FEATURES = ["int_rate", "installment", "sub_grade"]

# Raw columns consumed to build engineered features (not model inputs themselves).
SOURCE_COLUMNS = {
    "term": "source of term_months",
    "emp_length": "source of emp_length_years",
    "earliest_cr_line": "source of credit_history_months",
}

# Raw columns used for the target and the time split, never as features.
TARGET_COLUMN = "loan_status"
DATE_COLUMN = "issue_d"

_POST_OUTCOME = "post-outcome: recorded after the loan was issued and reflects repayment behaviour (target leakage)"
_HARDSHIP = "post-outcome: hardship programme data, only exists after the borrower got into difficulty"
_SETTLEMENT = "post-outcome: debt settlement data, only exists after default"
_JOINT = (
    "joint applications only appear from late 2015 and are absent from the "
    "training window, so the model cannot learn from them"
)
_LATE_BUREAU = (
    "bureau attribute only reported for loans issued from late 2015 "
    "(>=95% missing up to 2015), absent from the training window"
)

EXCLUDED = {
    # Identifiers / free text
    "id": "identifier, no predictive meaning",
    "member_id": "identifier, 100% missing",
    "url": "listing URL, identifier",
    "emp_title": "free text with very high cardinality; would need NLP and risks proxying protected attributes",
    "desc": "free text, mostly missing after 2013",
    "title": "free text loan title, duplicates purpose",
    # Geography
    "zip_code": "geographic; fair-lending risk as a proxy for protected attributes",
    "addr_state": "geographic; fair-lending risk as a proxy for protected attributes",
    # Redundant / constant
    "grade": "redundant: first letter of sub_grade",
    "fico_range_high": "redundant: equals fico_range_low + 4 in >99.9% of rows",
    "policy_code": "constant (1) for every loan in the public file",
    # Known only after approval / at funding
    "funded_amnt": "set when the loan is funded, after the application; near-duplicate of loan_amnt",
    "funded_amnt_inv": "amount funded by investors, determined after listing",
    "initial_list_status": "Lending Club's whole/fractional listing decision after approval, not an applicant attribute",
    "disbursement_method": "disbursement choice made at issuance; DirectPay only appears from 2016",
    "pymnt_plan": _POST_OUTCOME,
    # Post-outcome repayment fields (the original leakage)
    "out_prncp": _POST_OUTCOME,
    "out_prncp_inv": _POST_OUTCOME,
    "total_pymnt": _POST_OUTCOME,
    "total_pymnt_inv": _POST_OUTCOME,
    "total_rec_prncp": _POST_OUTCOME,
    "total_rec_int": _POST_OUTCOME,
    "total_rec_late_fee": _POST_OUTCOME,
    "recoveries": _POST_OUTCOME,
    "collection_recovery_fee": _POST_OUTCOME,
    "last_pymnt_d": _POST_OUTCOME,
    "last_pymnt_amnt": _POST_OUTCOME,
    "next_pymnt_d": _POST_OUTCOME,
    "last_credit_pull_d": _POST_OUTCOME,
    "last_fico_range_high": "post-outcome: FICO refreshed after issuance (target leakage)",
    "last_fico_range_low": "post-outcome: FICO refreshed after issuance (target leakage)",
    "debt_settlement_flag": _SETTLEMENT,
    "debt_settlement_flag_date": _SETTLEMENT,
    "settlement_status": _SETTLEMENT,
    "settlement_date": _SETTLEMENT,
    "settlement_amount": _SETTLEMENT,
    "settlement_percentage": _SETTLEMENT,
    "settlement_term": _SETTLEMENT,
    "hardship_flag": _HARDSHIP,
    "hardship_type": _HARDSHIP,
    "hardship_reason": _HARDSHIP,
    "hardship_status": _HARDSHIP,
    "deferral_term": _HARDSHIP,
    "hardship_amount": _HARDSHIP,
    "hardship_start_date": _HARDSHIP,
    "hardship_end_date": _HARDSHIP,
    "payment_plan_start_date": _HARDSHIP,
    "hardship_length": _HARDSHIP,
    "hardship_dpd": _HARDSHIP,
    "hardship_loan_status": _HARDSHIP,
    "orig_projected_additional_accrued_interest": _HARDSHIP,
    "hardship_payoff_balance_amount": _HARDSHIP,
    "hardship_last_payment_amount": _HARDSHIP,
    # Joint-application fields
    "application_type": _JOINT,
    "annual_inc_joint": _JOINT,
    "dti_joint": _JOINT,
    "verification_status_joint": _JOINT,
    "revol_bal_joint": _JOINT,
    "sec_app_fico_range_low": _JOINT,
    "sec_app_fico_range_high": _JOINT,
    "sec_app_earliest_cr_line": _JOINT,
    "sec_app_inq_last_6mths": _JOINT,
    "sec_app_mort_acc": _JOINT,
    "sec_app_open_acc": _JOINT,
    "sec_app_revol_util": _JOINT,
    "sec_app_open_act_il": _JOINT,
    "sec_app_num_rev_accts": _JOINT,
    "sec_app_chargeoff_within_12_mths": _JOINT,
    "sec_app_collections_12_mths_ex_med": _JOINT,
    "sec_app_mths_since_last_major_derog": _JOINT,
    # Bureau attributes introduced late 2015
    "open_acc_6m": _LATE_BUREAU,
    "open_act_il": _LATE_BUREAU,
    "open_il_12m": _LATE_BUREAU,
    "open_il_24m": _LATE_BUREAU,
    "mths_since_rcnt_il": _LATE_BUREAU,
    "total_bal_il": _LATE_BUREAU,
    "il_util": _LATE_BUREAU,
    "open_rv_12m": _LATE_BUREAU,
    "open_rv_24m": _LATE_BUREAU,
    "max_bal_bc": _LATE_BUREAU,
    "all_util": _LATE_BUREAU,
    "inq_fi": _LATE_BUREAU,
    "total_cu_tl": _LATE_BUREAU,
    "inq_last_12m": _LATE_BUREAU,
}

# Columns the loader must read from the raw file.
RAW_COLUMNS_USED = NUMERIC_FEATURES + CATEGORICAL_FEATURES + list(SOURCE_COLUMNS) + [TARGET_COLUMN, DATE_COLUMN]


def check_columns(header):
    """Fail if any raw column is neither used nor excluded with a reason, or if a used column is missing."""
    header = set(header)
    accounted = set(RAW_COLUMNS_USED) | set(EXCLUDED)
    unaccounted = sorted(header - accounted)
    missing = sorted(set(RAW_COLUMNS_USED) - header)
    if unaccounted or missing:
        raise ValueError(f"Unaccounted columns: {unaccounted}; missing required columns: {missing}")
