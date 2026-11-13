import numpy as np
import pandas as pd

from src.models import p2_gbt


def _table(n=300):
    rng = np.random.default_rng(0)
    t = pd.DataFrame({c: rng.normal(size=n) for c in p2_gbt.WINDOW_COLS})
    t["nearest_team"] = rng.integers(0, 2, n)
    t["period"] = np.r_[np.ones(n // 2), np.full(n - n // 2, 2)]
    return t


def test_default_matches_v1_columns():
    t = _table()
    X = p2_gbt.features(t)
    assert "ball_margin_m_past5s" in X and "ball_margin_m_next5s" in X
    assert not any("sd" in c or "diff" in c for c in X.columns)


def test_std_diff_windows_inside_period():
    t = _table()
    X = p2_gbt.features(t, windows=(1, 5), stats=("mean", "std", "diff"))
    c = "ball_margin_m"
    for s in (1, 5):
        assert np.allclose(X[c + "_diff{}s".format(s)], X[c + "_next{}s".format(s)] - X[c + "_past{}s".format(s)])
        assert c + "_pastsd{}s".format(s) in X
    # first frame of period 2: past window starts fresh there, not in period 1
    i = 150
    assert np.isclose(X[c + "_past5s"].iloc[i], t[c].iloc[i])
    assert np.isclose(X[c + "_next1s"].iloc[i - 1], t[c].iloc[i - 1])
