import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.models import p4_gru


def test_p4_gru_end_to_end_on_synthetic_halves():
    rng = np.random.default_rng(0)
    n = 400
    period = np.r_[np.ones(230, int), np.full(n - 230, 2)]
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    X.loc[rng.random(n) < 0.3, "b"] = np.nan
    labels = np.where(X["a"] > 0.5, "H", np.where(X["a"] < -0.5, "A", "D"))

    for s, e in p4_gru.windows(period, 50, 25):
        assert len(set(period[s:e])) == 1, "window crosses halftime"

    scaler = p4_gru.fit_scaler(X)
    assert scaler["masked"] == ["b"]
    x = p4_gru.transform(X, scaler)
    assert x.shape == (n, 3) and np.isfinite(x).all()

    seq = [(x, p4_gru.encode(labels), period)]
    model = p4_gru.train(seq, seq, length=50, max_epochs=2, log=lambda *_: None)
    proba = p4_gru.predict_proba(model, x, period, length=50, step=2)
    assert proba.shape == (n, 3)
    assert np.allclose(proba.sum(1), 1, atol=1e-5)

    again = p4_gru.predict_proba(p4_gru.train(seq, seq, length=50, max_epochs=2, log=lambda *_: None),
                                 x, period, length=50, step=2)
    assert np.array_equal(proba, again), "same seed must give identical output (RULES.md rule 3)"
