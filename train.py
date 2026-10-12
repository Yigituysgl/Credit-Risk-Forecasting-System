"""Train and evaluate the default model on the Lending Club accepted-loans file.

Usage:
    python train.py --data path/to/accepted_2007_to_2018Q4.csv

Models are fit on the early part of the training period, calibrated on the
later part, and evaluated on the out-of-time test period. Writes
reports/metrics.json and the fitted pipelines under models/ (git-ignored).
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
from credit_risk.dataset import load_splits
from credit_risk.modeling import LOGREG_PARAMS, NUMERIC_INPUTS, XGB_PARAMS, build_models, calibrate, evaluate
from credit_risk.split import CALIBRATION_START, TEST_START, WINDOW_START, summarize
from credit_risk.target import MATURITY_BUFFER_MONTHS, SNAPSHOT_DATE, TARGET

FEATURE_SETS = {
    "full": (NUMERIC_INPUTS, F.CATEGORICAL_FEATURES),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="path to accepted_2007_to_2018Q4.csv")
    p.add_argument("--models-dir", default=Path("models"), type=Path)
    p.add_argument("--report", default=Path("reports/metrics.json"), type=Path)
    p.add_argument("--window-start", default=WINDOW_START, type=pd.Timestamp)
    p.add_argument("--calibration-start", default=CALIBRATION_START, type=pd.Timestamp)
    p.add_argument("--test-start", default=TEST_START, type=pd.Timestamp)
    return p.parse_args(argv)


def fit_feature_set(splits, numeric, categorical, out_dir):
    """Fit, calibrate, save and evaluate both models on one feature set."""
    fit, cal, test = splits["fit"], splits["calibration"], splits["test"]
    cols = numeric + categorical
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, model in build_models(numeric, categorical).items():
        print(f"  {name}: fitting on {len(fit):,} rows, calibrating on {len(cal):,} rows ...")
        model.fit(fit[cols], fit[TARGET])
        calibrated = calibrate(model, cal[cols], cal[TARGET])
        joblib.dump(model, out_dir / f"{name}.joblib")
        joblib.dump(calibrated, out_dir / f"{name}_calibrated.joblib")
        results[name] = {
            "uncalibrated": evaluate(test[TARGET], model.predict_proba(test[cols])[:, 1]),
            "calibrated": evaluate(test[TARGET], calibrated.predict_proba(test[cols])[:, 1]),
            "fit_roc_auc": evaluate(fit[TARGET], model.predict_proba(fit[cols])[:, 1])["roc_auc"],
        }
        print(f"    test uncalibrated: {results[name]['uncalibrated']}")
        print(f"    test calibrated:   {results[name]['calibrated']}")
    return results


def main(argv=None):
    args = parse_args(argv)

    print(f"Loading {args.data.name} ...")
    splits, target_counts = load_splits(args.data, args.window_start, args.calibration_start, args.test_start)
    split_summary = {k: summarize(v) for k, v in splits.items()}
    print(json.dumps(split_summary, indent=2))

    results = {}
    for set_name, (numeric, categorical) in FEATURE_SETS.items():
        print(f"Feature set '{set_name}' ({len(numeric) + len(categorical)} features)")
        results[set_name] = fit_feature_set(splits, numeric, categorical, args.models_dir / set_name)

    report = {
        "data": {
            "source_file": args.data.name,
            "snapshot_date": SNAPSHOT_DATE.strftime("%Y-%m"),
            "maturity_buffer_months": MATURITY_BUFFER_MONTHS,
            "window_start": args.window_start.strftime("%Y-%m"),
            "calibration_start": args.calibration_start.strftime("%Y-%m"),
            "test_start": args.test_start.strftime("%Y-%m"),
            "target": "1 = Charged Off or Default, 0 = Fully Paid (matured loans only)",
            "filtering": target_counts,
        },
        "split": split_summary,
        "evaluation": "metrics on the test period; calibration = Platt (sigmoid) fit on the calibration period",
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
