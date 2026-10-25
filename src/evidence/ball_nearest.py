"""Baseline ball evidence: the team of the track nearest the ball (the official baseline's signal).

Evidence contract: frame, score_H, score_A, score_D, confidence. Scores are NaN where the module has
no opinion; this one never speaks about D (a missing ball is not dead ball).
"""

import numpy as np
import pandas as pd


def score(table):
    nearest = table["nearest_team"].to_numpy()
    h = np.where(nearest == 0, 1.0, np.where(nearest == 1, 0.0, np.nan))
    return pd.DataFrame({"frame": table["frame"], "score_H": h, "score_A": 1.0 - h,
                         "score_D": np.nan, "confidence": table["ball_visible"].astype(float)})
