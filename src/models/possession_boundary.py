"""Training-only boundary scores from the shared offline tracking features."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import classification_inputs as ci


COLUMNS = (
    "ball_visible", "ball_margin_m", "seconds_since_ball_seen",
    "seconds_until_ball_seen", "gap_length_s", "ball_range_back_1s_m",
    "ball_range_fwd_1s_m", "ball_speed_p90_ms", "ball_dist_touchline_m",
    "ball_dist_goalline_m", "ball_dist_centre_m", "pl_slow_share",
    "pl_fast_share", "dead_ball_conf", "dead_player_conf",
)
FIT_STEP = 5
TARGET_RADIUS_S = 0.2
CONTEXT_S = 0.4
MAX_COST_SHIFT = 10.0


@dataclass(frozen=True)
class BoundaryModel:
    classifier: HistGradientBoostingClassifier
    prevalence: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(values)[:, 1]


def features(inputs: ci.ClassificationInputs, fps: int) -> np.ndarray:
    """Use short tracking context on both sides of a potential switch."""
    selected = inputs.frame_features.loc[:, list(COLUMNS)].to_numpy(dtype=np.float32, copy=True)
    period = np.asarray(inputs.period)
    step = max(1, round(CONTEXT_S * fps))
    left, right = selected.copy(), selected.copy()
    for start, end in _runs(period):
        positions = np.arange(start, end)
        left[start:end] = selected[np.maximum(positions - step, start)]
        right[start:end] = selected[np.minimum(positions + step, end - 1)]
    change = np.abs(right - left)
    margin = COLUMNS.index("ball_margin_m")
    for table in (selected, left, right):
        table[:, margin] = np.abs(table[:, margin])
    matrix = np.concatenate((left, selected, right, change), axis=1)
    return np.where(np.isfinite(matrix), matrix, np.nan)


def _runs(period: np.ndarray) -> Sequence[Tuple[int, int]]:
    edges = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1, len(period)]
    return list(zip(edges[:-1], edges[1:]))


def targets(labels: np.ndarray, period: np.ndarray, fps: int) -> np.ndarray:
    """Return 1 near a graded state change, 0 elsewhere, -1 for invalid edges."""
    if labels.ndim != 1 or period.shape != labels.shape:
        raise ValueError("boundary labels and periods must align on one frame axis")
    graded = np.isin(labels, ("H", "A", "D"))
    radius = round(TARGET_RADIUS_S * fps)
    out = np.full(len(labels), -1, dtype=np.int8)
    for start, end in _runs(period):
        mask = graded[start:end]
        cuts = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        for a, b in zip(cuts[::2] + start, cuts[1::2] + start):
            out[a + 1:b] = 0
            switches = a + 1 + np.flatnonzero(labels[a + 1:b] != labels[a:b - 1])
            for switch in switches:
                out[max(a + 1, switch - radius):min(b, switch + radius + 1)] = 1
    return out


def fit(samples: Sequence[Tuple[np.ndarray, np.ndarray]]) -> BoundaryModel:
    """Fixed native missing-value tree model, trained only on supplied matches."""
    x = np.concatenate([values for values, _ in samples])
    y = np.concatenate([target for _, target in samples])
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("boundary fit requires both binary classes")
    model = HistGradientBoostingClassifier(
        max_iter=120, max_leaf_nodes=15, learning_rate=0.1,
        l2_regularization=1.0, early_stopping=False, random_state=0,
    )
    model.fit(x, y)
    return BoundaryModel(model, float(y.mean()))


def switch_cost(probability: np.ndarray, prevalence: float, penalty: float) -> np.ndarray:
    """Discount fixed cost by the boundary log-odds gain over its train prior."""
    if not 0 < prevalence < 1 or not np.isfinite(penalty) or penalty < 0:
        raise ValueError("switch costs need a valid train prior and non-negative penalty")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("boundary probabilities must be finite and in [0, 1]")
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    gain = np.log(probability / (1 - probability)) - np.log(prevalence / (1 - prevalence))
    return np.maximum(0, penalty - np.clip(gain, -MAX_COST_SHIFT, MAX_COST_SHIFT))
