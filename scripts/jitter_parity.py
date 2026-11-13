"""Tracker jitter parity: BAR-ATH vs clean Sportec vs damaged bundles. Inputs only, no labels.

    python scripts/jitter_parity.py [MATCH ...]

Player jitter = median |second difference| of x per visible track (m per frame^2): white noise of sd s adds about
2.4*s here, smooth (low-frequency) error adds almost nothing. Ball: median and p90 step speed between consecutive
visible frames (steps > 40 m/s dropped as jumps).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_sportec_bundles as bb  # noqa: E402
from src.features import frame_table as ft  # noqa: E402


def stats(x, bx, by, fps=25):
    d2 = np.abs(x[2:] - 2 * x[1:-1] + x[:-2])
    d2 = d2[np.isfinite(d2)]
    step = np.hypot(np.diff(bx), np.diff(by)) * fps
    step = step[np.isfinite(step) & (step < 40)]
    return {"player_d2_m": float(np.median(d2)), "player_d2_p90": float(np.percentile(d2, 90)),
            "ball_step_med_ms": float(np.median(step)), "ball_step_p90_ms": float(np.percentile(step, 90))}


def show(name, s):
    print("{:28s} ".format(name) + "  ".join("{} {:.4f}".format(k, v) for k, v in s.items()), flush=True)


def main():
    ms = sys.argv[1:] or ["J03WMX"]
    t = ft.load()
    g = t["graded"].to_numpy()
    sl = ft.load_slots()
    show("BAR-ATH", stats(sl["x"][g], t["ball_x_m"].to_numpy()[g], t["ball_y_m"].to_numpy()[g]))
    for m in ms:
        info, _, pos = bb.match_files(m)
        home, _, L, W = bb.read_info(info)
        c = bb.assemble(bb.read_positions(pos), home, L, W)
        show(m + " clean", stats(c["x"], c["ball"]["x"], c["ball"]["y"]))
        bt, bs = ft.bundle_table(os.path.join(bb.OUT_DIR, m))
        show(m + " damaged", stats(bs["x"], bt["ball_x_m"].to_numpy(), bt["ball_y_m"].to_numpy()))


if __name__ == "__main__":
    main()
