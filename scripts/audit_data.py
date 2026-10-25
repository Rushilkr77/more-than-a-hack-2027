"""Re-runnable data audit: prints facts downstream modules should know. Writes nothing.

    python scripts/audit_data.py
"""

import math
import os
import random
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths  # noqa: E402
from src.features import frame_table as ft  # noqa: E402
from starter import load_smart_tagging as lst  # noqa: E402


def raw_spot_check(t, paths, n=400, seed=1):
    """Re-parse random raw lines with plain python (no numpy) and compare with the cached table."""
    L, W = GAMES[GAME]["pitch"]
    root = ET.parse(paths["atd_meta"]).getroot()
    first = next(iter(root.iter("Team"))).get("id")
    slot_team = [0 if p.get("teamId") == first else 1 for p in root.iter("Player")]
    want = set(random.Random(seed).sample(range(len(t)), n))
    bad = 0
    with open(paths["atd_raw"]) as f:
        for i, line in enumerate(f):
            if i not in want:
                continue
            fid, players, ball = line.rstrip("\n").split(":")
            pts = []
            for s, g in enumerate(players.split(";")):
                x, y, _ = (float(v) for v in g.split(","))
                if not math.isnan(x):
                    pts.append((x * L, y * W, slot_team[s]))
            bx, by, _ = (float(v) for v in ball.split(","))
            r = t.iloc[i]
            ok = int(fid) == i + 1 and r.n_home == sum(p[2] == 0 for p in pts) and r.ball_visible == (not math.isnan(bx))
            if not math.isnan(bx) and ok:
                d = [(math.hypot(p[0] - bx * L, p[1] - by * W), p[2]) for p in pts]
                ok = abs(r.ball_x_m - bx * L) < 1e-3 and ((not d) or min(d)[1] == r.nearest_team)
            bad += not ok
    return n, bad


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    L, W = GAMES[GAME]["pitch"]
    t, sl, paths = ft.load(), ft.load_slots(), asset_paths()
    g, vis = t.graded.to_numpy(), t.ball_visible.to_numpy()
    print("== loader vs independent raw re-parse: %d frames checked, %d mismatches" % raw_spot_check(t, paths))
    print("== shape: %d rows x %d cols; graded %d; ball visible all %.1f%% graded %.1f%%" % (
        len(t), t.shape[1], g.sum(), 100 * vis.mean(), 100 * vis[g].mean()))

    print("\n== orientation (XML attack_direction_first_half vs measured mean x)")
    for half, (a, b) in ((1, (11185, 90651)), (2, (110808, 190197))):
        s = t.iloc[a:b + 1]
        print("   half %d: home attacks right = %s ; home mean x %.1f, away %.1f" % (half, bool(s.home_attacks_right.iloc[0]), s.home_cx_m.mean(), s.away_cx_m.mean()))

    print("\n== track quality")
    print("   tracks >11: home %.1f%% away %.1f%% of graded; slot obs outside pitch: %d" % (
        100 * (t.n_home[g] > 11).mean(), 100 * (t.n_away[g] > 11).mean(),
        int(((sl["x"] < -1) | (sl["x"] > L + 1) | (sl["y"] < -1) | (sl["y"] > W + 1)).sum())))
    print("   shape columns usable (>=7 on-pitch tracks): home %.1f%% away %.1f%% of graded; mean shape_conf %.2f / %.2f" % (
        100 * t.home_dline_m[g].notna().mean(), 100 * t.away_dline_m[g].notna().mean(), t.home_shape_conf[g].mean(), t.away_shape_conf[g].mean()))

    print("\n== ball")
    bx, by = t.ball_x_m.to_numpy(), t.ball_y_m.to_numpy()
    out = g & vis & ((bx < -2) | (bx > L + 2) | (by < -2) | (by > W + 2))
    print("   >2 m outside pitch: %d of %d graded observations (%.2f%%)" % (out.sum(), (g & vis).sum(), 100 * out.sum() / (g & vis).sum()))
    sp = t.ball_speed_atd_ms.to_numpy()
    print("   ATD ball speed missing on %.1f%% of ball-visible frames (do not use)" % (100 * np.isnan(sp[vis]).mean()))
    gaps = np.diff(np.concatenate([[0], (~vis & g).astype(int), [0]]))
    ln = np.where(gaps == -1)[0] - np.where(gaps == 1)[0]
    print("   ball-missing runs (graded): %d, median %.1f s, longest %.0f s; %.0f%% of missing frames are in runs > 10 s" % (
        len(ln), np.median(ln) / 25, ln.max() / 25, 100 * ln[ln > 250].sum() / ln.sum()))
    m = g & vis & t.ball_margin_m.notna().to_numpy()
    print("   nearest-player margin < 1 m on %.0f%% of visible graded frames (contested); use ball_margin_m as confidence" % (100 * (t.ball_margin_m[m] < 1).mean()))

    print("\n== time base: ball at throw-in starts should sit at a touchline")
    pos = []
    for e in lst.load():
        if (e["code"] or "").lower() == "set pieces" and (e["tags"].get("Type") or "").lower() == "throw-in":
            a, b = e["start_frame"], min(e["end_frame"], e["start_frame"] + 125)
            v = vis[a:b]
            if v.any():
                pos.append(np.nanmedian(by[a:b][v]))
    d = np.minimum(np.array(pos), W - np.array(pos))
    print("   %d throw-ins with ball: %.0f%% within 3 m of a touchline" % (len(pos), 100 * (d < 3).mean()))

    print("\n== shape columns vs Smart Tagging phases (tagged team's mean defensive line / centroid, m from own goal)")
    rows = {}
    for e in lst.load():
        team = ft.TEAM_TAG.get((e["tags"].get("Team") or "").strip().lower())
        a, b = e["start_frame"], e["end_frame"]
        if team is None or b <= a or not g[a:b].all() or (e["code"] or "").upper() == "SET PIECES":
            continue
        k = "home" if team == 0 else "away"
        rows.setdefault(e["code"].upper(), []).append((np.nanmean(t["%s_dline_m" % k][a:b]), np.nanmean(t["%s_cen_m" % k][a:b])))
    tab = pd.DataFrame({c: np.nanmean(v, axis=0) for c, v in rows.items()}, index=["dline", "cen"]).T.sort_values("dline")
    print(tab.round(0).astype(int).to_string())
    print("\n== weak possession hint coverage: non-zero on %.1f%% of graded frames" % (100 * (t.tag_poss_hint[g] != 0).mean()))
    print("\n== team style (median over graded half; metres from own goal)")
    st = ft.team_style(t)
    print(st[st.scope == "half"].round(1).drop(columns="scope").to_string(index=False))


if __name__ == "__main__":
    main()
