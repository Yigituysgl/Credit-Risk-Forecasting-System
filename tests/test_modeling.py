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


def test_preprocessor_ignores_unseen_category(synthetic_inputs):
    df = synthetic_inputs
    model = M.build_models()["logistic_regression"].fit(df[MODEL_FEATURES], df["default"])
    row = df[MODEL_FEATURES].iloc[[0]].copy()
    row["sub_grade"] = "G5"
    assert model.predict_proba(row).shape == (1, 2)
