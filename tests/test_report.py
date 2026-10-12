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
    out, evaluation = tmp_path / "docs", tmp_path / "evaluation.json"
    make_report.main(["--data", str(data), "--metrics", str(metrics), "--models-dir", str(models),
                      "--out-dir", str(out), "--evaluation", str(evaluation)])
    for name in CHARTS:
        assert (out / name).stat().st_size > 10_000
    ev = json.loads(evaluation.read_text(encoding="utf-8"))
    assert [d["decile"] for d in ev["risk_deciles"]] == list(range(1, 11))
    assert sum(d["loans"] for d in ev["risk_deciles"]) == json.loads(metrics.read_text())["split"]["test"]["rows"]
    assert ev["top_to_bottom_decile_default_ratio"] > 0
    assert set(ev["default_rate_by_grade_36_month_loans"]) == {"A", "B", "C"}


def test_report_rejects_stale_metrics(trained, tmp_path):
    data, metrics, models = trained
    report = json.loads(metrics.read_text(encoding="utf-8"))
    report["models"]["full"]["xgboost"]["calibrated"]["roc_auc"] += 0.1
    metrics.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        make_report.main(["--data", str(data), "--metrics", str(metrics), "--models-dir", str(models),
                          "--out-dir", str(tmp_path / "docs"), "--evaluation", str(tmp_path / "evaluation.json")])
