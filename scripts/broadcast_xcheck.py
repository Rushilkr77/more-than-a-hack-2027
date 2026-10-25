"""Cross-check broadcast vs tactical ATD: refresh the cached confidence table, print the matrix and fill
risk, write the list of stretches that would be filled (outputs/broadcast_fill_segments.csv).

    python scripts/broadcast_xcheck.py [--broadcast-dir DIR] [--refit]
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

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


def n_spells(labels):
    codes = pm.label_codes(pd.Series(labels))
    return len(pm.possession_spells(codes, pm.valid_runs(codes)))


def fill_segments(table, ev):
    """Contiguous runs of filled frames per source tier: where broadcast evidence entered."""
    rows = []
    src, weight, graded = ev["ball_source"].to_numpy(), ev["weight"].to_numpy(), table["graded"].to_numpy()
    for tier in (1, 2):
        edges = np.diff(np.concatenate([[0], (src == tier).astype(int), [0]]))
        for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1):
            rows.append({"start_frame": a, "end_frame": b, "n_frames": b - a + 1, "tier": tier,
                         "mean_weight": round(float(weight[a:b + 1].mean()), 3),
                         "graded": bool(graded[a:b + 1].all())})
    return pd.DataFrame(rows).sort_values("start_frame").reset_index(drop=True)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--broadcast-dir")
    ap.add_argument("--refit", action="store_true", help="re-derive the per-half offsets and compare")
    args = ap.parse_args()

    table, tc = ft.load(), ft.load_slots()
    if args.refit:
        meta, raw = bx.broadcast_paths(args.broadcast_dir)
        bc = atd.load_slots(meta, raw, bx.N_BC, GAMES[GAME]["pitch"])
        print("refit offsets {} (module constants {})".format(bx.refit_offsets(bc, tc), bx.OFFSETS))
    xc = bx.load_or_build(table, tc, args.broadcast_dir)

    g = table["graded"].to_numpy()
    print("confidence matrix (graded frames: {})".format(int(g.sum())))
    m = bx.matrix(table, xc)
    m["total"] = m.sum(axis=1)
    m.loc["total"] = m.sum()
    print(m.to_string())
    print("tier share of graded frames:", (xc["tier"][g].value_counts(normalize=True).sort_index() * 100).round(1).to_dict())

    # risk: where both balls are visible, does the broadcast ball pick the same nearest team?
    co = g & table["ball_visible"].to_numpy() & np.isfinite(xc["bc_ball_x_m"].to_numpy())
    f = np.where(co)[0]
    nt_bc = ball_filled.nearest_team(tc, f, xc["bc_ball_x_m"].to_numpy()[f], xc["bc_ball_y_m"].to_numpy()[f])
    nt_tac = table["nearest_team"].to_numpy()[f]
    both = (nt_bc >= 0) & (nt_tac >= 0)
    print("nearest-team agreement, broadcast ball vs tactical ball (co-visible graded frames; easier than real gaps)")
    for tier in (1, 2, 3, 4):
        mk = both & (xc["tier"].to_numpy()[f] == tier)
        if mk.any():
            print("  tier {} ({:<12}) n={:6d}  agree {:.1f}%  median ball error {:.2f} m".format(
                tier, bx.TIER_NAMES[tier], int(mk.sum()), 100 * (nt_bc[mk] == nt_tac[mk]).mean(),
                np.nanmedian(xc["ball_err_m"].to_numpy()[f][mk])))

    ev = ball_filled.score(table, (1, 2), xc, tc)
    seg = fill_segments(table, ev)
    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    seg.to_csv(os.path.join(out, "broadcast_fill_segments.csv"), index=False)
    print("\nfill segments: {} written to outputs/broadcast_fill_segments.csv ({} frames, graded {})".format(
        len(seg), int(seg.n_frames.sum()), int(seg.loc[seg.graded, "n_frames"].sum())))
    print(seg.groupby("tier").n_frames.agg(["count", "sum", "median", "max"]).to_string())

    base = majority.decode(ball_nearest.score(table), 50)
    for name, tiers, anchor in (("tier1", (1,), 1), ("tier1+2", (1, 2), 1)):
        lab = majority.decode(ball_filled.score(table, tiers, xc, tc), 50, anchor)
        print("fill {:<8} min_anchor={}: labels changed on {:.2f}% of graded frames; spells {} -> {}; D {} -> {}".format(
            name, anchor, 100 * (base[g] != lab[g]).mean(), n_spells(base[g]), n_spells(lab[g]),
            int((base[g] == "D").sum()), int((lab[g] == "D").sum())))
