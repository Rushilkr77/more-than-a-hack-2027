"""Shared smoother for P2-P5: 3-state Viterbi over per-frame P(H), P(A), P(D).

Maximises sum(log P(label)) minus a penalty per label change, separately inside each period, so a possession never
spans halftime. Penalties are tuned on the validation matches against labels_adjusted (organiser possession grain).

- switch_penalty: cost of any change (default, one dial). 0 is plain argmax; bigger gives fewer, longer spells.
- team_penalty / dead_penalty: optional separate costs for H<->A and for live<->D changes (default: switch_penalty).
- min_frames: optional minimum spell length (a spell can only end after this many frames; the last spell of a period
  may be shorter). 1 = no minimum.
- frame_switch_cost: optional cost of changing into each frame; period starts ignore this cost.
"""

from typing import Optional

import numpy as np

LABELS = np.array(["H", "A", "D"])


def _penalties(switch_penalty, team_penalty=None, dead_penalty=None):
    t = switch_penalty if team_penalty is None else team_penalty
    d = switch_penalty if dead_penalty is None else dead_penalty
    return np.array([[0, t, d], [t, 0, d], [d, d, 0]], float)


def _viterbi(logp, P):
    n, k = logp.shape
    back = np.zeros((n, k), dtype=np.int8)
    score = logp[0].copy()
    for t in range(1, n):
        cand = score[:, None] - (P[t] if P.ndim == 3 else P)  # [from, to]
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logp[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _viterbi_min(logp, P, L):
    """Expanded states (label, age) with age capped at L-1; a switch is allowed only from age L-1."""
    n, k = logp.shape
    S = k * L
    back = np.zeros((n, S), dtype=np.int16)
    idx = np.arange(S).reshape(k, L)
    score = np.full((k, L), -np.inf)
    score[:, 0] = logp[0]
    for t in range(1, n):
        new = np.empty((k, L))
        b = np.empty((k, L), dtype=np.int16)
        new[:, 1:], b[:, 1:] = score[:, :-1], idx[:, :-1]               # age + 1
        stay = score[:, L - 1] > score[:, L - 2]                           # mature spell continues
        new[:, L - 1] = np.where(stay, score[:, L - 1], score[:, L - 2])
        b[:, L - 1] = np.where(stay, idx[:, L - 1], idx[:, L - 2])
        cand = score[:, L - 1][:, None] - P + np.where(np.eye(k, dtype=bool), -np.inf, 0)  # switch to a new label
        new[:, 0], b[:, 0] = cand.max(axis=0), idx[cand.argmax(axis=0), L - 1]
        score = new + logp[t][:, None]
        back[t] = b.ravel()
    path = np.empty(n, dtype=np.int16)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return (path // L).astype(np.int8)


def smooth(proba, period, switch_penalty, team_penalty=None, dead_penalty=None, min_frames=1,
           *, frame_switch_cost: Optional[np.ndarray] = None) -> np.ndarray:
    """proba: [frames, 3] columns H, A, D. period: [frames]. Returns an array of 'H' / 'A' / 'D'."""
    logp = np.log(np.clip(np.asarray(proba, float), 1e-6, 1.0))
    P = _penalties(switch_penalty, team_penalty, dead_penalty)
    period = np.asarray(period)
    if frame_switch_cost is not None:
        costs = np.asarray(frame_switch_cost, dtype=float)
        if costs.shape != (len(logp),) or not np.isfinite(costs).all() or (costs < 0).any():
            raise ValueError("frame_switch_cost must have one finite, non-negative value per frame")
        if min_frames != 1 or team_penalty is not None or dead_penalty is not None:
            raise ValueError("frame_switch_cost requires the default transition and duration options")
        P = costs[:, None, None] * _penalties(1)
    if not len(logp):
        return LABELS[:0]
    out = np.empty(len(logp), dtype=np.int8)
    edges = np.r_[0, np.flatnonzero(np.diff(period)) + 1, len(period)]
    for a, b in zip(edges[:-1], edges[1:]):
        transitions = P[a:b] if frame_switch_cost is not None else P
        out[a:b] = (_viterbi_min(logp[a:b], transitions, min_frames)
                    if min_frames > 1 else _viterbi(logp[a:b], transitions))
    return LABELS[out]
