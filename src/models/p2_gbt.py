"""P2: gradient-boosted trees (sklearn HistGradientBoostingClassifier) on the frame table.

v0: the per-frame model inputs `ft.model_columns()` (NaN kept: the trees learn a branch for missing).
v1: v0 plus past / next 5 s means of a few ball, shape, dead-ball and generated-tag columns, computed inside each
    period, so a tree sees what happened just before and just after the frame.

Same code for Sportec and BAR-ATH: features depend only on the frame table. Labels never enter `features`.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import frame_table as ft

CLASSES = np.array(["H", "A", "D"])
WINDOW = 125                                   # 5 s at 25 fps
WINDOW_COLS = ("ball_visible", "ball_home_adv_m", "ball_margin_m", "home_ball_dist_m", "away_ball_dist_m",
               "home_cen_m", "away_cen_m", "home_width_m", "away_width_m", "home_speed_ms", "away_speed_ms",
               "ball_speed_p90_ms", "pl_slow_share", "gtag_poss")


def features(table, version="v1", windows=(5,), stats=("mean",)):
    """windows: seconds for past/next rolling windows. stats: "mean" (default), "std", "diff" (next mean - past mean)."""
    X = table[ft.model_columns(list(table.columns))].astype("float32")
    if version == "v0":
        return X
    nt = table["nearest_team"].to_numpy()
    src = table[list(WINDOW_COLS)].astype("float32").copy()
    src["nearest_signed"] = np.where(nt == 0, 1.0, np.where(nt == 1, -1.0, np.nan))
    extra = {}
    for _, part in src.groupby(table["period"].to_numpy(), sort=False):
        for sec in windows:
            w = int(sec * WINDOW / 5)
            rp, rn = part.rolling(w, min_periods=1), part.iloc[::-1].rolling(w, min_periods=1)
            past, nxt = rp.mean(), rn.mean().iloc[::-1]
            cols = {}
            if "mean" in stats:
                cols.update({"_past{}s": past, "_next{}s": nxt})
            if "std" in stats:
                cols.update({"_pastsd{}s": rp.std(), "_nextsd{}s": rn.std().iloc[::-1]})
            if "diff" in stats:
                cols["_diff{}s"] = nxt - past
            for suf, df in cols.items():
                for c in part.columns:
                    extra.setdefault(c + suf.format(sec), []).append(df[c])
    extra = pd.DataFrame({k: pd.concat(v).sort_index() for k, v in extra.items()}).astype("float32")
    return pd.concat([X, extra.set_index(X.index)], axis=1)


PARAMS = dict(max_iter=500, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True,
              n_iter_no_change=20)


def train(X, y, X_val, y_val, seed=0, **params):
    model = HistGradientBoostingClassifier(**{**PARAMS, **params}, random_state=seed)
    model.fit(X, y, X_val=X_val, y_val=y_val)
    return model


def predict_proba(model, X):
    """[frames, 3] columns in H, A, D order."""
    p = model.predict_proba(X)
    order = [list(model.classes_).index(c) for c in CLASSES]
    return p[:, order]
