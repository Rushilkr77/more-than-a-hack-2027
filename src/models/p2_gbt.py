"""P2: gradient-boosted trees over the selected common flat input view."""

import inspect

import numpy as np
import pandas as pd

from src.features import classification_inputs as ci


CLASSES = np.array(["H", "A", "D"])
WINDOW_COLS = (
    "ball_visible", "ball_home_adv_m", "ball_margin_m", "home_ball_dist_m", "away_ball_dist_m",
    "home_cen_m", "away_cen_m", "home_width_m", "away_width_m", "home_speed_ms", "away_speed_ms",
    "ball_speed_p90_ms", "pl_slow_share", "dead_player_conf", "gtag_poss",
)
PARAMS = dict(max_iter=500, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True,
              n_iter_no_change=20)


def features(inputs: ci.ClassificationInputs, version="v1", windows=(5,), stats=("mean",), fps=25):
    """P2's old within-period rolling features over an explicit flat input policy."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    X = inputs.frame_features.astype("float32")
    if version == "v0":
        return X
    if version != "v1":
        raise ValueError("unknown P2 feature version {!r}".format(version))
    sources = [column for column in WINDOW_COLS if column in X]
    src = X[sources]
    extra = {}
    for _, part in src.groupby(inputs.period, sort=False):
        for seconds in windows:
            window = int(seconds * fps)
            if window <= 0:
                raise ValueError("window durations must be positive")
            past = part.rolling(window, min_periods=1)
            next_ = part.iloc[::-1].rolling(window, min_periods=1)
            values = {}
            if "mean" in stats:
                values.update({"_past{}s": past.mean(), "_next{}s": next_.mean().iloc[::-1]})
            if "std" in stats:
                values.update({"_pastsd{}s": past.std(), "_nextsd{}s": next_.std().iloc[::-1]})
            if "diff" in stats:
                values["_diff{}s"] = next_.mean().iloc[::-1] - past.mean()
            for suffix, frame in values.items():
                for column in part:
                    extra.setdefault(column + suffix.format(seconds), []).append(frame[column])
    if not extra:
        return X
    rolled = pd.DataFrame({name: pd.concat(values).sort_index() for name, values in extra.items()})
    return pd.concat([X, rolled.set_index(X.index)], axis=1).astype("float32")


def train(X, y, X_val, y_val, seed=0, **params):
    from sklearn.ensemble import HistGradientBoostingClassifier

    columns = X.columns[X.notna().any()].tolist()
    if not columns:
        raise ValueError("P2 cannot train when every selected feature is missing")
    model = HistGradientBoostingClassifier(**{**PARAMS, **params}, random_state=seed)
    if "X_val" in inspect.signature(model.fit).parameters:
        model.fit(X[columns], y, X_val=X_val[columns], y_val=y_val)
        model.p2_validation_mode_ = "explicit_grouped_validation"
    else:
        model.fit(X[columns], y)
        model.p2_validation_mode_ = "internal_train_fraction"
    model.p2_columns_ = columns
    return model


def predict_proba(model, X):
    """Return P(H), P(A), P(D), regardless of sklearn's class ordering."""
    probabilities = model.predict_proba(X[model.p2_columns_])
    return probabilities[:, [list(model.classes_).index(label) for label in CLASSES]]
