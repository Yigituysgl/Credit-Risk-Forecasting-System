"""Train and evaluate the default model on the Lending Club accepted-loans file.

Usage:
    python train.py --data path/to/accepted_2007_to_2018Q4.csv

Writes reports/metrics.json and the fitted pipelines under models/
(git-ignored).
"""

import argparse
import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import xgboost

from credit_risk import features as F
from credit_risk.data import add_engineered_features, load_raw
from credit_risk.modeling import LOGREG_PARAMS, XGB_PARAMS, build_models, evaluate
from credit_risk.split import TEST_START, WINDOW_START, summarize, time_split
from credit_risk.target import MATURITY_BUFFER_MONTHS, SNAPSHOT_DATE, TARGET, build_target


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="path to accepted_2007_to_2018Q4.csv")
    p.add_argument("--models-dir", default=Path("models"), type=Path)
    p.add_argument("--report", default=Path("reports/metrics.json"), type=Path)
    p.add_argument("--window-start", default=WINDOW_START, type=pd.Timestamp)
    p.add_argument("--test-start", default=TEST_START, type=pd.Timestamp)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"Loading {args.data.name} ...")
    raw = add_engineered_features(load_raw(args.data))
    df, target_counts = build_target(raw)
    train, test = time_split(df, args.window_start, args.test_start)
    if train.empty or test.empty:
        raise ValueError("Empty train or test set; check --window-start/--test-start")
    assert train[F.DATE_COLUMN].max() < test[F.DATE_COLUMN].min()

    split = {"train": summarize(train), "test": summarize(test)}
    print(json.dumps(split, indent=2))

    X_train, y_train = train[F.MODEL_FEATURES], train[TARGET]
    X_test, y_test = test[F.MODEL_FEATURES], test[TARGET]

    args.models_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, model in build_models().items():
        print(f"Fitting {name} on {len(X_train):,} rows ...")
        model.fit(X_train, y_train)
        metrics = evaluate(y_test, model.predict_proba(X_test)[:, 1])
        metrics["train_roc_auc"] = evaluate(y_train, model.predict_proba(X_train)[:, 1])["roc_auc"]
        results[name] = metrics
        print(f"  test: {metrics}")
        joblib.dump(model, args.models_dir / f"{name}.joblib")

    report = {
        "data": {
            "source_file": args.data.name,
            "snapshot_date": SNAPSHOT_DATE.strftime("%Y-%m"),
            "maturity_buffer_months": MATURITY_BUFFER_MONTHS,
            "window_start": args.window_start.strftime("%Y-%m"),
            "test_start": args.test_start.strftime("%Y-%m"),
            "target": "1 = Charged Off or Default, 0 = Fully Paid (matured loans only)",
            "filtering": target_counts,
        },
        "split": split,
        "models": results,
        "params": {"xgboost": XGB_PARAMS, "logistic_regression": LOGREG_PARAMS},
        "features": {
            "numeric": F.NUMERIC_FEATURES,
            "engineered": F.ENGINEERED_FEATURES,
            "categorical": F.CATEGORICAL_FEATURES,
            "source_columns": F.SOURCE_COLUMNS,
        },
        "excluded": F.EXCLUDED,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}")
    return report


if __name__ == "__main__":
    main()
