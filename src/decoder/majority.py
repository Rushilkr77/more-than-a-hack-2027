"""Baseline decoder: centred weighted vote over +/- `window` frames, D where nobody has a vote.

Vote weight is the evidence `weight` column when present, else 1 wherever a score exists (so ball_nearest
reproduces the official baseline exactly). Optional `min_anchor`: if the evidence has an `anchor` column
(weight contributed by non-filled, trusted observations), a window needs at least that much anchor weight
to vote at all; otherwise it is D. This stops filled-in evidence from deciding a stretch on its own.

`tie`: "home" gives exact vote ties to home (the starter's rule; keeps the published baseline byte-identical);
"previous" holds the last H/A label instead, so no team is favoured (use this for new configs).
"""

import numpy as np


def _window_sum(x, window):
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    return c[np.minimum(len(x), i + window + 1)] - c[np.maximum(0, i - window)]


def decode(evidence, window, min_anchor=0.0, tie="home"):
    h, a = evidence["score_H"].to_numpy(), evidence["score_A"].to_numpy()
    codes = np.where(h > a, 0, np.where(a > h, 1, -1))                  # NaN compares False -> no vote
    w = evidence["weight"].to_numpy(float) if "weight" in evidence else np.ones(len(h))
    w = np.where(codes >= 0, np.nan_to_num(w), 0.0)
    home, away = _window_sum(w * (codes == 0), window), _window_sum(w * (codes == 1), window)

    labels = np.full(len(h), "D", dtype="<U1")
    voting = (home + away) > 0
    if min_anchor > 0 and "anchor" in evidence:
        voting &= _window_sum(np.nan_to_num(evidence["anchor"].to_numpy(float)), window) >= min_anchor
    labels[voting] = np.where(home[voting] >= away[voting], "H", "A")
    if tie == "previous":
        tied = voting & (home == away)
        if tied.any():
            held = np.where(voting & ~tied, labels, "")                # H/A where decided
            idx = np.maximum.accumulate(np.where(held != "", np.arange(len(h)), -1))
            prev = np.where(idx >= 0, held[np.maximum(idx, 0)], "D")
            labels[tied] = prev[tied]
    elif tie != "home":
        raise ValueError("tie must be 'home' or 'previous'")
    return labels
