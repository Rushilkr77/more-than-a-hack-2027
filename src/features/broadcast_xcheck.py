"""Cross-check the broadcast ATD against the (reference) tactical ATD and grade how far it can be trusted.

Findings that drive this module (re-derivable with `refit_offsets`):
  * Both ATDs use the same 0-1 pitch coordinates, so player configurations can be matched directly.
  * The broadcast is a shifted copy of the tactical timeline with ONE constant offset per half:
    tactical = broadcast + OFFSETS[0] (first half) / OFFSETS[1] (second half). Halftime is cut out.
  * Frames that break the offset (replays, cuts) show up as player configurations that do not match.

Per tactical frame we grade, over a +/-WINDOW frame window (spell-level, not per-frame):
  tier 1  verified alignment, ball agreement observed and good          conf 0.75..1.00
  tier 2  mostly aligned, or aligned but ball agreement unobserved      conf 0.50..0.75
  tier 3  large mismatch -> filling gaps from the broadcast is a risk   conf 0.25..0.50
  tier 4  alignment cannot be verified / no mapped broadcast frame      conf NaN (no opinion)
Tiers are ordinal grades from measured agreement, not calibrated probabilities.
"""

import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

from games import GAME, GAMES, SOURCES, asset_paths
from src.data import atd

OFFSETS = (4014, 26920)        # tactical = broadcast + OFFSET, first / second half
SPLIT_T = 100000               # tactical frames before this use OFFSETS[0] (halftime gap is 90652..110807)
N_BC = SOURCES["broadcast"]["n_video_frames"]
MIN_TRACKS = 6                 # visible broadcast tracks needed to judge a player-configuration match
PLAYER_THR_M = 3.0             # mean nearest-neighbour distance below this = match (null 1st pct is ~4.9 m)
BALL_THR_M = 5.0               # co-visible balls closer than this support alignment (null: 1.3% under 5 m)
WINDOW = 125                   # +/- frames (5 s) over which evidence is pooled
MIN_EVIDENCE = 25              # frames of evidence needed in a window to grade it at all
TIER1_SCORE, TIER2_SCORE, TIER1_BALL = 0.9, 0.6, 0.8
TIER_NAMES = {1: "verified", 2: "partial", 3: "mismatch", 4: "unverifiable"}


def broadcast_paths(broadcast_dir=None, data_dir=None):
    """(meta, raw) paths. Accepts the bundle layout or a folder (e.g. a Drive download) holding the files."""
    stem = SOURCES["broadcast"]["stem"]
    want = (stem + "_atd.xml", stem + "_atd.txt")
    candidates = [broadcast_dir] if broadcast_dir else [os.path.dirname(asset_paths(data_dir=data_dir, source="broadcast")["atd_raw"])]
    for d in list(candidates) + [p for c in candidates for p in glob.glob(os.path.join(c, "*"))]:
        if d and all(os.path.isfile(os.path.join(d, w)) for w in want):
            return tuple(os.path.join(d, w) for w in want)
    raise IOError("broadcast ATD not found under {}".format(candidates[0]))


def pair_cost(bc, tc, b, t, chunk=6000):
    """Per pair (broadcast frame b, tactical frame t): mean over visible broadcast tracks of the distance
    (m, capped at 10) to the nearest same-team tactical track. NaN if t is out of range or b has no tracks.
    Returns (cost, n_visible_broadcast_tracks)."""
    b, t = np.asarray(b), np.asarray(t)
    cost, n = np.full(len(b), np.nan), np.zeros(len(b), dtype=int)
    same = bc["team"][:, None] == tc["team"][None, :]
    for s in range(0, len(b), chunk):
        bb, tq = b[s:s + chunk], t[s:s + chunk]
        ok = (tq >= 0) & (tq < len(tc["x"]))
        tq = np.where(ok, tq, 0)
        bx, by = bc["x"][bb], bc["y"][bb]
        tx, ty = tc["x"][tq].astype(np.float32), tc["y"][tq].astype(np.float32)
        d = np.hypot(bx[:, :, None] - tx[:, None, :], by[:, :, None] - ty[:, None, :])
        d = np.where(np.isfinite(d) & same[None], d, 50.0)
        nn = np.minimum(d.min(axis=2), 10.0)
        vis = np.isfinite(bx)
        k = vis.sum(1)
        c = np.where(vis, nn, 0).sum(1) / np.maximum(k, 1)
        c[~ok | (k == 0)] = np.nan
        cost[s:s + chunk], n[s:s + chunk] = c, k
    return cost, n


def refit_offsets(bc, tc, centre=OFFSETS, half_width=6, step=7):
    """Re-derive each half's offset: the shift minimising mean player-configuration cost on frames that match."""
    n_tracks = np.isfinite(bc["x"]).sum(1)
    out = []
    for (lo, hi), centre_o in zip(((3000, 82000), (90000, 164000)), centre):
        bs = np.arange(lo, hi, step)
        bs = bs[n_tracks[bs] >= 10]
        offs = np.arange(centre_o - half_width, centre_o + half_width + 1)
        c, _ = pair_cost(bc, tc, np.repeat(bs, len(offs)), np.repeat(bs, len(offs)) + np.tile(offs, len(bs)))
        c = c.reshape(len(bs), len(offs))
        keep = np.nanmin(np.where(np.isnan(c), 99, c), axis=1) < 1.5
        out.append(int(offs[np.nanargmin(np.nanmean(c[keep], axis=0))]))
    return tuple(out)


def grade_windows(matched, evaluable, close, co_visible, window=None):
    """Tier (1-4) and confidence per tactical frame from boolean evidence arrays on the tactical timeline."""
    window = WINDOW if window is None else window
    def pooled(x):
        c = np.concatenate([[0], np.cumsum(x, dtype=np.int64)])
        i = np.arange(len(x))
        return c[np.minimum(len(x), i + window + 1)] - c[np.maximum(0, i - window)]
    n_p, m_p, n_b, m_b = pooled(evaluable), pooled(matched & evaluable), pooled(co_visible), pooled(close & co_visible)
    n = n_p + n_b
    score = np.divide(m_p + m_b, n, out=np.full(len(n), np.nan), where=n > 0)
    ball_ok = np.divide(m_b, n_b, out=np.full(len(n), np.nan), where=n_b > 0)

    tier = np.full(len(n), 4, dtype=np.int8)
    graded = n >= MIN_EVIDENCE
    tier[graded & (score < TIER2_SCORE)] = 3
    tier[graded & (score >= TIER2_SCORE)] = 2
    tier[graded & (score >= TIER1_SCORE) & (n_b >= 10) & (ball_ok >= TIER1_BALL)] = 1
    s = np.nan_to_num(score)
    conf = np.full(len(n), np.nan)
    conf = np.where(tier == 1, 0.75 + 0.25 * np.clip((s - TIER1_SCORE) / (1 - TIER1_SCORE), 0, 1), conf)
    conf = np.where(tier == 2, 0.50 + 0.25 * np.clip((s - TIER2_SCORE) / (TIER1_SCORE - TIER2_SCORE), 0, 1), conf)
    conf = np.where(tier == 3, 0.25 + 0.25 * np.clip(s / TIER2_SCORE, 0, 1), conf)
    return tier, conf, score, n


def player_evidence(tc, bc, n_t):
    """Mapped broadcast frame per tactical frame and the player-configuration cost there. Independent of the
    tactical ball, so burst-mask validation can compute it once and reuse it."""
    t = np.arange(n_t)
    b = t - np.where(t < SPLIT_T, OFFSETS[0], OFFSETS[1])
    mapped = (b >= 0) & (b < N_BC)
    bm = np.where(mapped, b, 0)
    cost, n_vis = pair_cost(bc, tc, bm, t)
    return {"b": b, "bm": bm, "mapped": mapped, "cost": cost, "n_vis": n_vis}


def build(table, tc, bc, player=None):
    """DataFrame keyed by tactical `frame`: mapped broadcast frame/ball, evidence, tier, conf, fill_ok."""
    n_t = len(table)
    t = np.arange(n_t)
    p = player or player_evidence(tc, bc, n_t)
    b, bm, mapped, cost, n_vis = p["b"], p["bm"], p["mapped"], p["cost"], p["n_vis"]
    evaluable = mapped & (n_vis >= MIN_TRACKS)
    matched = evaluable & (cost < PLAYER_THR_M)

    bx = np.where(mapped, bc["ball_x"][bm], np.nan)
    by = np.where(mapped, bc["ball_y"][bm], np.nan)
    tx, ty = table["ball_x_m"].to_numpy(np.float64), table["ball_y_m"].to_numpy(np.float64)
    co_visible = np.isfinite(bx) & np.isfinite(tx)
    ball_err = np.where(co_visible, np.hypot(bx - tx, by - ty), np.nan)
    close = co_visible & (ball_err < BALL_THR_M)

    tier, conf, score, n_ev = grade_windows(matched, evaluable, close, co_visible)
    tier[~mapped] = 4
    conf[~mapped] = np.nan
    return pd.DataFrame({
        "frame": t, "bc_frame": np.where(mapped, b, -1), "bc_ball_x_m": bx, "bc_ball_y_m": by,
        "player_cost_m": cost, "player_matched": matched, "ball_err_m": ball_err,
        "evidence_n": n_ev, "score": score, "tier": tier, "conf": conf,
        # fillable = tactical ball missing, broadcast ball present, alignment graded tier 1 or 2
        "fill_ok": (~table["ball_visible"].to_numpy()) & np.isfinite(bx) & (tier <= 2),
    })


def matrix(table, xc):
    """Tier x gap-status counts (graded frames only): the confidence matrix."""
    g = table["graded"].to_numpy()
    tactical = table["ball_visible"].to_numpy()
    bc_ball = np.isfinite(xc["bc_ball_x_m"].to_numpy())
    status = np.where(tactical, "tactical ball present",
                      np.where(bc_ball, "gap, broadcast ball available", "gap, no broadcast ball"))
    m = pd.crosstab(xc["tier"][g].map(TIER_NAMES), pd.Series(status[g], index=xc.index[g], name="status"))
    return m.reindex([TIER_NAMES[i] for i in (1, 2, 3, 4)]).fillna(0).astype(int)


# -- cache -------------------------------------------------------------------------------------
CODE_FILES = ("src/features/broadcast_xcheck.py", "src/data/atd.py")


def _cache_paths():
    from src.features import frame_table as ft
    d = ft.cache_dir()
    return os.path.join(d, "broadcast_xcheck_v0.parquet"), os.path.join(d, "broadcast_xcheck_v0.meta.json")


def _fingerprint():
    """Code of this module + the frame table it was graded against (its meta already hashes tactical inputs)."""
    from src.features import frame_table as ft
    with open(ft._paths()[2]) as f:
        return ft.code_sha256(CODE_FILES, extra=f.read())


def is_fresh(broadcast_dir=None):
    from src.features import frame_table as ft
    path, meta_path = _cache_paths()
    if not (os.path.isfile(path) and os.path.isfile(meta_path)):
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("fingerprint") != _fingerprint():
        return False
    try:
        return meta["inputs_sha256"] == {os.path.basename(p): ft.sha256_file(p) for p in broadcast_paths(broadcast_dir)}
    except OSError:
        warnings.warn("broadcast inputs not available; cross-check cache checked against code only")
        return True


def load_or_build(table, tc=None, broadcast_dir=None):
    """The per-frame cross-check table; rebuilt (needs the broadcast ATD files) if missing or stale."""
    from src.features import frame_table as ft
    path, meta_path = _cache_paths()
    if is_fresh(broadcast_dir):
        return pd.read_parquet(path)
    meta_file, raw_file = broadcast_paths(broadcast_dir)
    tc = tc or ft.load_slots()
    bc = atd.load_slots(meta_file, raw_file, N_BC, GAMES[GAME]["pitch"])
    xc = build(table, tc, bc)
    os.makedirs(ft.cache_dir(), exist_ok=True)
    xc.to_parquet(path, index=False)
    with open(meta_path, "w") as f:
        json.dump({"fingerprint": _fingerprint(),
                   "inputs_sha256": {os.path.basename(p): ft.sha256_file(p) for p in (meta_file, raw_file)}}, f, indent=2)
    return xc
