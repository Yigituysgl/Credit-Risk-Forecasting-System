import json

import numpy as np
import pandas as pd

import train
from credit_risk import features as F


def _raw_csv(path, n=600):
    """Synthetic file in the raw Lending Club layout (all 151 columns)."""
    rng = np.random.default_rng(1)
    cols = F.RAW_COLUMNS_USED + list(F.EXCLUDED)
    df = pd.DataFrame({c: [""] * n for c in cols})
    for c in F.NUMERIC_FEATURES:
        df[c] = rng.normal(size=n)
    months = pd.date_range("2013-01-01", "2015-12-01", freq="MS")
    df["issue_d"] = months.strftime("%b-%Y")[rng.integers(0, len(months), size=n)]
    df["term"] = " 36 months"
    df["emp_length"] = rng.choice(["< 1 year", "3 years", "10+ years"], size=n)
    df["earliest_cr_line"] = "Jan-2000"
    df["sub_grade"] = rng.choice(["A1", "B2", "C3"], size=n)
    df["home_ownership"] = rng.choice(["RENT", "MORTGAGE"], size=n)
    df["verification_status"] = rng.choice(["Verified", "Not Verified"], size=n)
    df["purpose"] = rng.choice(["debt_consolidation", "credit_card"], size=n)
    p = 1 / (1 + np.exp(-2 * df["int_rate"]))
    df["loan_status"] = np.where(rng.random(n) < p, "Charged Off", "Fully Paid")
    df.loc[:9, "loan_status"] = "Current"
    df.to_csv(path, index=False)


def test_without_pricing_set_drops_exactly_pricing_features():
    full = set(sum(train.FEATURE_SETS["full"], []))
    reduced = set(sum(train.FEATURE_SETS["without_pricing"], []))
    assert full == set(F.MODEL_FEATURES)
    assert full - reduced == set(F.PRICING_FEATURES)


def test_train_end_to_end(tmp_path):
    data = tmp_path / "raw.csv"
    _raw_csv(data)
    report_path = tmp_path / "reports" / "metrics.json"
    models_dir = tmp_path / "models"

    train.main(["--data", str(data), "--report", str(report_path), "--models-dir", str(models_dir)])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    split = report["split"]
    assert split["fit"]["issue_d_max"] < split["calibration"]["issue_d_min"]
    assert split["calibration"]["issue_d_max"] < split["test"]["issue_d_min"]
    assert report["data"]["filtering"]["dropped_non_final_status"] == {"Current": 10}
    for set_name in train.FEATURE_SETS:
        for name in ("xgboost", "logistic_regression"):
            res = report["models"][set_name][name]
            for stage in ("uncalibrated", "calibrated"):
                assert set(res[stage]) >= {"roc_auc", "pr_auc", "brier", "ks", "mean_predicted_pd"}
            assert (models_dir / set_name / f"{name}.joblib").exists()
            assert (models_dir / set_name / f"{name}_calibrated.joblib").exists()
    assert set(report["excluded"]) == set(F.EXCLUDED)
    assert str(tmp_path) not in report_path.read_text(encoding="utf-8")
