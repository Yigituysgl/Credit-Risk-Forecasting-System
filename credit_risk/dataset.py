"""Raw file -> fit / calibration / test frames. Shared by train.py and make_report.py."""

from credit_risk.data import add_engineered_features, load_raw
from credit_risk.features import DATE_COLUMN
from credit_risk.split import CALIBRATION_START, TEST_START, WINDOW_START, calibration_split, time_split
from credit_risk.target import build_target


def load_splits(path, window_start=WINDOW_START, calibration_start=CALIBRATION_START, test_start=TEST_START):
    """Return ({"fit", "calibration", "test"} frames, target filtering counts)."""
    df, target_counts = build_target(add_engineered_features(load_raw(path)))
    train, test = time_split(df, window_start, test_start)
    if train.empty or test.empty:
        raise ValueError("Empty train or test set; check the split dates")
    fit, calibration = calibration_split(train, calibration_start)
    assert fit[DATE_COLUMN].max() < calibration[DATE_COLUMN].min()
    assert calibration[DATE_COLUMN].max() < test[DATE_COLUMN].min()
    return {"fit": fit, "calibration": calibration, "test": test}, target_counts
