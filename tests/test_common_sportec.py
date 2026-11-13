import numpy as np

from src.common import sportec


def test_tune_penalty_prefers_smoothing_when_probabilities_flicker():
    n = 3000
    truth = np.array(["H"] * n)
    p = np.full((n, 3), 0.1)
    p[:, 0] = 0.8
    p[::7] = [0.1, 0.8, 0.1]                      # frequent wrong-team blips
    table = {"period": np.ones(n)}
    best, curve = sportec.tune_penalty({"m": p}, {"m": (table, {"adjusted": truth}, 105.0)})
    assert best > 0 and curve[best] >= curve[0]
