import json

import pytest

import make_report
import train
from test_train import _raw_csv

CHARTS = ["model_comparison.png", "calibration.png", "default_rate_by_decile.png", "sensitivity.png"]


@pytest.fixture
def trained(tmp_path):
    data = tmp_path / "raw.csv"
    _raw_csv(data, n=1500)
    metrics, models = tmp_path / "metrics.json", tmp_path / "models"
    train.main(["--data", str(data), "--report", str(metrics), "--models-dir", str(models)])
    return data, metrics, models


def test_report_writes_all_charts(trained, tmp_path):
    data, metrics, models = trained
    out = tmp_path / "docs"
    make_report.main(["--data", str(data), "--metrics", str(metrics), "--models-dir", str(models), "--out-dir", str(out)])
    for name in CHARTS:
        assert (out / name).stat().st_size > 10_000


def test_report_rejects_stale_metrics(trained, tmp_path):
    data, metrics, models = trained
    report = json.loads(metrics.read_text(encoding="utf-8"))
    report["models"]["full"]["xgboost"]["calibrated"]["roc_auc"] += 0.1
    metrics.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        make_report.main(["--data", str(data), "--metrics", str(metrics), "--models-dir", str(models),
                          "--out-dir", str(tmp_path / "docs")])
