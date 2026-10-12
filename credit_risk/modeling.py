"""Model pipelines and evaluation metrics.

All preprocessing (imputation, scaling, encoding) lives inside the sklearn
Pipeline, so it is fit on the training rows only.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from credit_risk.features import CATEGORICAL_FEATURES, ENGINEERED_FEATURES, NUMERIC_FEATURES

NUMERIC_INPUTS = NUMERIC_FEATURES + list(ENGINEERED_FEATURES)

# Fixed up front, not tuned on the test set.
XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}
LOGREG_PARAMS = {"C": 1.0, "max_iter": 2000}


def _to_object(X):
    # Uniform missing-value handling for pandas string columns.
    return X.astype(object).where(X.notna(), np.nan)


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("to_object", FunctionTransformer(_to_object, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, NUMERIC_INPUTS), ("cat", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
    )


def build_models():
    return {
        "logistic_regression": Pipeline([
            ("pre", build_preprocessor()),
            ("model", LogisticRegression(**LOGREG_PARAMS)),
        ]),
        "xgboost": Pipeline([
            ("pre", build_preprocessor()),
            ("model", XGBClassifier(**XGB_PARAMS)),
        ]),
    }


def ks_statistic(y_true, y_score):
    """Kolmogorov-Smirnov: max distance between the score CDFs of defaults and non-defaults."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(y_true, y_score):
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "brier": round(float(brier_score_loss(y_true, y_score)), 4),
        "ks": round(ks_statistic(y_true, y_score), 4),
        "mean_predicted_pd": round(float(np.mean(y_score)), 4),
    }
