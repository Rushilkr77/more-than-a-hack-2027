"""Burst-mask validation of broadcast fills. Hides the tactical ball in random 0.5-10 s bursts, re-grades with
only the remaining evidence, fills tiers 1-2 from the broadcast, and measures how well each variant recovers
the unmasked baseline decode.

    python scripts/validate_fill.py [--seeds 5] [--mask-share 0.2] [--broadcast-dir DIR]

Reference = smooth50 decode with the full tactical ball (NOT true possession: no full-match truth exists).
Random bursts are easier than real, missing-not-at-random gaps, and tactical-visible frames have the broadcast
ball far more often (~79%) than real gaps do (~13%): read coverage and per-fill precision, not just the net score.
"""

import argparse
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES  # noqa: E402
from src.data import atd  # noqa: E402
from src.decoder import majority  # noqa: E402
from src.evidence import ball_filled, ball_nearest  # noqa: E402
from src.features import broadcast_xcheck as bx  # noqa: E402
from src.features import frame_table as ft  # noqa: E402

VARIANTS = (("masked baseline", None, 0), ("fill t1, weight only", (1,), 0), ("fill t1, min_anchor=1", (1,), 1),
            ("fill t1+2, weight only", (1, 2), 0), ("fill t1+2, min_anchor=1", (1, 2), 1))


def burst_mask(table, rng, share, lo=12, hi=250):
    """Bool mask of frames to hide: random bursts inside graded, ball-visible stretches, ~share of such frames."""
    ok = table["graded"].to_numpy() & table["ball_visible"].to_numpy()
    mask, target, n = np.zeros(len(table), dtype=bool), share * ok.sum(), len(table)
    while (mask & ok).sum() < target:
        length = int(rng.integers(lo, hi + 1))
        start = int(rng.integers(0, n - length))
        seg = slice(start, start + length)
        if ok[seg].mean() > 0.9:
            mask[seg] = True
    return mask & ok


def masked_copy(table, mask):
    m = table.copy()
    m.loc[mask, ["ball_visible"]] = False
    m.loc[mask, ["ball_x_m", "ball_y_m", "ball_speed_atd_ms"]] = np.nan
    m.loc[mask, "nearest_team"] = -1
    m.loc[mask, ["ball_margin_m", "ball_home_adv_m"]] = np.nan
    return m


def codes(labels, graded):
    c = pm.label_codes(labels)
    return np.where(graded, c, 3)            # ungraded -> X, ignored by the scorer


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--mask-share", type=float, default=0.2)
    ap.add_argument("--broadcast-dir")
    args = ap.parse_args()

    table, tc = ft.load(), ft.load_slots()
    meta, raw = bx.broadcast_paths(args.broadcast_dir)
    bc = atd.load_slots(meta, raw, bx.N_BC, GAMES[GAME]["pitch"])
    player = bx.player_evidence(tc, bc, len(table))
    g = table["graded"].to_numpy()
    truth = majority.decode(ball_nearest.score(table), 50)
    gt = codes(truth, g)

    rows = {v[0]: [] for v in VARIANTS}
    fills = {1: [], 2: []}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        mask = burst_mask(table, rng, args.mask_share)
        mt = masked_copy(table, mask)
        xc = bx.build(mt, tc, bc, player=player)
        for name, tiers, anchor in VARIANTS:
            ev = ball_nearest.score(mt) if tiers is None else ball_filled.score(mt, tiers, xc, tc)
            lab = majority.decode(ev, 50, anchor)
            res = pm.evaluate(gt, codes(lab, g))
            rows[name].append((res["spell_f1_mean"], res["macro_f1"], (lab[mask] == truth[mask]).mean(), res["n_pred_spells"],
                               res["per_class"]["H"]["recall"], res["per_class"]["A"]["recall"]))
            if tiers == (1, 2) and anchor == 0:
                src = ev["ball_source"].to_numpy()
                nearest_true = table["nearest_team"].to_numpy()
                nearest_fill = np.where(ev["score_H"].to_numpy() == 1, 0, np.where(ev["score_A"].to_numpy() == 1, 1, -1))
                for tier in (1, 2):
                    sel = mask & (src == tier) & (nearest_true >= 0)
                    fills[tier].append((sel.sum() / max(1, mask.sum()), (nearest_fill[sel] == nearest_true[sel]).mean() if sel.any() else np.nan))

    print("burst-mask validation: {} seeds, ~{:.0f}% of ball-visible graded frames hidden in 0.5-10 s bursts".format(
        args.seeds, 100 * args.mask_share))
    print("reference = unmasked smooth50 decode ({} graded spells)\n".format(res["n_gt_spells"]))
    print("{:<26} {:>14} {:>14} {:>16} {:>9} {:>7} {:>7}".format("variant", "spellF1", "macroF1", "acc in bursts", "spells", "H rec", "A rec"))
    for name, _, _ in VARIANTS:
        a = np.array(rows[name])
        print("{:<26} {:>7.4f}±{:.4f} {:>7.4f}±{:.4f} {:>9.4f}±{:.4f} {:>9.0f} {:>7.3f} {:>7.3f}".format(
            name, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std(), a[:, 2].mean(), a[:, 2].std(), a[:, 3].mean(),
            a[:, 4].mean(), a[:, 5].mean()))
    base = np.array(rows["masked baseline"])
    print("\npaired change vs masked baseline (same bursts), spellF1 / macroF1 / acc in bursts; seeds improved on spellF1:")
    for name, _, _ in VARIANTS[1:]:
        d = np.array(rows[name]) - base
        print("{:<26} {:+.4f}±{:.4f}  {:+.4f}  {:+.4f}   {}/{}".format(
            name, d[:, 0].mean(), d[:, 0].std(), d[:, 1].mean(), d[:, 2].mean(), int((d[:, 0] > 0).sum()), len(d)))
    print()
    for tier in (1, 2):
        a = np.array(fills[tier])
        print("tier {} fills inside bursts: cover {:.1f}% of hidden frames; fill picks the true nearest team {:.1f}% of the time".format(
            tier, 100 * np.nanmean(a[:, 0]), 100 * np.nanmean(a[:, 1])))
