"""Optional ball evidence: tactical ball where visible, broadcast-ball fills (tiers 1-2) where it is not.

NOT used by the default build. Every row says where its evidence came from, so downstream code can down-weight,
exclude or audit the filled stretches.

Columns (evidence contract first, then supplementary):
  score_H, score_A, score_D  nearest-team evidence (NaN = no opinion); never speaks about D
  confidence                 1.0 where this module has a vote, else 0.0 (same meaning as ball_nearest)
  weight                     vote weight: 1.0 tactical ball; broadcast fill = alignment conf (tier 1: 0.75-1.00,
                             tier 2: 0.50-0.75); 0.0 no vote
  anchor                     1.0 where the vote is a tactical observation, 0.0 for fills / no vote
  ball_source                0 tactical, 1 broadcast fill tier 1, 2 broadcast fill tier 2, -1 no ball evidence
  align_conf                 alignment confidence of the fill (NaN for tactical / no vote)

Caveats: fill agreement was measured where both balls are visible (easier than real gaps); about half of the
tier-2 fills have no ball-agreement evidence in their window; use `anchor` / min_anchor in the decoder so fills
cannot decide a stretch alone. See scripts/validate_fill.py for the burst-mask check.
"""

import numpy as np
import pandas as pd

from src.features import broadcast_xcheck as bx
from src.features import frame_table as ft


def nearest_team(tc, frames, ball_x, ball_y):
    """0 home / 1 away / -1 unknown: team of the tactical track nearest the given ball position."""
    out = np.full(len(frames), -1, dtype=np.int8)
    if len(frames) == 0:
        return out
    d = np.hypot(tc["x"][frames] - ball_x[:, None], tc["y"][frames] - ball_y[:, None])
    with np.errstate(all="ignore"):
        dh = np.nanmin(np.where(tc["team"][None] == 0, d, np.nan), axis=1)
        da = np.nanmin(np.where(tc["team"][None] == 1, d, np.nan), axis=1)
    out[np.isfinite(dh) & (~np.isfinite(da) | (dh <= da))] = 0
    out[np.isfinite(da) & (~np.isfinite(dh) | (da < dh))] = 1
    return out


def score(table, tiers=(1, 2), xc=None, tc=None, broadcast_dir=None):
    tc = tc if tc is not None else ft.load_slots()
    xc = xc if xc is not None else bx.load_or_build(table, tc, broadcast_dir)
    n = len(table)
    nearest = table["nearest_team"].to_numpy().astype(np.int8).copy()
    source = np.where(table["ball_visible"].to_numpy() & (nearest >= 0), 0, -1).astype(np.int8)
    weight = np.where(source == 0, 1.0, 0.0)
    align = np.full(n, np.nan)

    bx_, by_ = xc["bc_ball_x_m"].to_numpy(), xc["bc_ball_y_m"].to_numpy()
    tier = xc["tier"].to_numpy()
    fill = np.where(~table["ball_visible"].to_numpy() & np.isfinite(bx_) & np.isin(tier, list(tiers)))[0]
    team = nearest_team(tc, fill, bx_[fill], by_[fill])
    fill, team = fill[team >= 0], team[team >= 0]
    nearest[fill], source[fill] = team, tier[fill]
    weight[fill] = align[fill] = xc["conf"].to_numpy()[fill]

    voting = source >= 0
    h = np.where(voting, (nearest == 0).astype(float), np.nan)
    return pd.DataFrame({"frame": table["frame"], "score_H": h, "score_A": 1.0 - h, "score_D": np.nan,
                         "confidence": voting.astype(float), "weight": weight,
                         "anchor": (source == 0).astype(float), "ball_source": source, "align_conf": align})
