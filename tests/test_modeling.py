import numpy as np
import pytest

from credit_risk import modeling as M
from credit_risk.features import MODEL_FEATURES


def test_ks_statistic_bounds():
    y = np.array([0, 0, 1, 1])
    assert M.ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert M.ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.0


def test_evaluate_keys():
    out = M.evaluate(np.array([0, 1, 0, 1]), np.array([0.2, 0.7, 0.4, 0.6]))
    assert set(out) == {"roc_auc", "pr_auc", "brier", "ks", "mean_predicted_pd"}
    assert out["roc_auc"] == 1.0


@pytest.mark.parametrize("name", ["logistic_regression", "xgboost"])
def test_models_fit_and_learn_signal(synthetic_inputs, name):
    df = synthetic_inputs
    train, test = df.iloc[:300], df.iloc[300:]
    model = M.build_models()[name]
    model.fit(train[MODEL_FEATURES], train["default"])
    p = model.predict_proba(test[MODEL_FEATURES])[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert M.evaluate(test["default"], p)["roc_auc"] > 0.7


def test_calibration_preserves_ranking_and_moves_mean(synthetic_inputs):
    df = synthetic_inputs
    fit, cal, test = df.iloc[:200], df.iloc[200:300], df.iloc[300:]
    model = M.build_models()["xgboost"].fit(fit[MODEL_FEATURES], fit["default"])
    calibrated = M.calibrate(model, cal[MODEL_FEATURES], cal["default"])
    p_raw = model.predict_proba(test[MODEL_FEATURES])[:, 1]
    p_cal = calibrated.predict_proba(test[MODEL_FEATURES])[:, 1]
    assert M.evaluate(test["default"], p_raw)["roc_auc"] == M.evaluate(test["default"], p_cal)["roc_auc"]
    assert not np.allclose(p_raw, p_cal)


def test_build_models_with_feature_subset(synthetic_inputs):
    df = synthetic_inputs
    numeric, categorical = ["loan_amnt", "fico_range_low"], ["purpose"]
    model = M.build_models(numeric, categorical)["logistic_regression"]
    model.fit(df[numeric + categorical], df["default"])
    assert model.named_steps["pre"].n_features_in_ == 3


def test_preprocessor_ignores_unseen_category(synthetic_inputs):
    df = synthetic_inputs
    model = M.build_models()["logistic_regression"].fit(df[MODEL_FEATURES], df["default"])
    row = df[MODEL_FEATURES].iloc[[0]].copy()
    row["sub_grade"] = "G5"
    assert model.predict_proba(row).shape == (1, 2)
