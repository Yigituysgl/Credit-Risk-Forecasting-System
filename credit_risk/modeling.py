"""Model pipelines and evaluation metrics.

All preprocessing (imputation, scaling, encoding) lives inside the sklearn
Pipeline, so it is fit on the training rows only.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
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


def build_preprocessor(numeric_inputs=NUMERIC_INPUTS, categorical_inputs=CATEGORICAL_FEATURES):
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
        [("num", numeric, numeric_inputs), ("cat", categorical, categorical_inputs)],
        remainder="drop",
    )


def build_models(numeric_inputs=NUMERIC_INPUTS, categorical_inputs=CATEGORICAL_FEATURES):
    return {
        "logistic_regression": Pipeline([
            ("pre", build_preprocessor(numeric_inputs, categorical_inputs)),
            ("model", LogisticRegression(**LOGREG_PARAMS)),
        ]),
        "xgboost": Pipeline([
            ("pre", build_preprocessor(numeric_inputs, categorical_inputs)),
            ("model", XGBClassifier(**XGB_PARAMS)),
        ]),
    }


def calibrate(model, X_cal, y_cal):
    """Platt (sigmoid) calibration of an already-fitted model on a held-out slice.

    Sigmoid is monotonic, so ranking metrics (ROC AUC, KS) are unchanged and
    only the probability scale moves.
    """
    return CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid").fit(X_cal, y_cal)


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
