import numpy as np
import pandas as pd
import pytest

from credit_risk import features as F


@pytest.fixture
def synthetic_inputs():
    """Model-ready frame with all model features, some missing values and a learnable target."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({c: rng.normal(size=n) for c in F.NUMERIC_FEATURES + list(F.ENGINEERED_FEATURES)})
    df["sub_grade"] = rng.choice(["A1", "B2", "C3", "D4"], size=n)
    df["home_ownership"] = rng.choice(["RENT", "OWN", "MORTGAGE"], size=n)
    df["verification_status"] = rng.choice(["Verified", "Not Verified"], size=n)
    df["purpose"] = rng.choice(["debt_consolidation", "credit_card", None], size=n)
    df.loc[::7, "dti"] = np.nan
    logit = 1.5 * df["int_rate"] - 1.0 * df["fico_range_low"]
    df["default"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df
