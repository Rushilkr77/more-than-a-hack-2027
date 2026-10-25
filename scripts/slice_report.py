"""Where do the scores come from? Score a prediction against labels overall and sliced by the reliability markers.

    python scripts/slice_report.py --pred outputs/x.csv --gt labels.csv     # labels for a VALIDATION match with the same frame table
    python scripts/slice_report.py --pred outputs/filled_t12_smooth50.csv --proxy outputs/baseline_smooth50.csv

--proxy uses another prediction as the reference (e.g. fills vs baseline): a difference report, NOT accuracy.
Never point --gt at the provided match's private labels; the public five minutes are an alignment check only.
Spell metrics are global (spells do not slice); per-slice numbers are frame-level recall/accuracy and share of all errors.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from src.features import frame_table as ft  # noqa: E402


def slices(table):
    """{slice name: {bucket: bool mask}} from the reliability markers in the frame table."""
    q = lambda col, edges, names: {n: ((table[col] >= lo) & (table[col] < hi)).to_numpy() for (lo, hi), n in zip(edges, names)}  # noqa: E731
    conf = np.minimum(table["home_shape_conf"], table["away_shape_conf"])
    out = {
        "period": {"half 1": (table.period == 1).to_numpy(), "half 2": (table.period == 2).to_numpy()},
        "tactical ball": {"visible": table.ball_visible.to_numpy(), "missing": ~table.ball_visible.to_numpy()},
        "ball margin (m)": {**q("ball_margin_m", [(0, 1), (1, 4), (4, 1e9)], ["<1 contested", "1-4", ">4 clear"]),
                            "no ball": table.ball_margin_m.isna().to_numpy()},
        "shape confidence (min of teams)": {"low <0.5": (conf < 0.5).to_numpy(), "mid 0.5-0.9": ((conf >= 0.5) & (conf < 0.9)).to_numpy(),
                                            "high >=0.9": (conf >= 0.9).to_numpy()},
        "weak possession hint": {"home": (table.tag_poss_hint == 1).to_numpy(), "away": (table.tag_poss_hint == -1).to_numpy(),
                                 "none/conflict": (table.tag_poss_hint == 0).to_numpy()},
    }
    return out


def slice_scores(table, gt, pred):
    """Overall metrics plus a per-slice table. gt/pred: arrays of 'H','A','D' (or 'X') per frame."""
    g = table["graded"].to_numpy() & np.isin(gt, list("HAD"))
    res = pm.evaluate(np.where(g, pm.label_codes(gt), 3), np.where(g, pm.label_codes(pred), 3))
    wrong = g & (gt != pred)
    rows = []
    for name, buckets in slices(table).items():
        for bucket, m in buckets.items():
            m = m & g
            if not m.any():
                continue
            rec = lambda c: float(((pred == c) & (gt == c) & m).sum() / max(1, ((gt == c) & m).sum()))  # noqa: E731
            rows.append({"slice": name, "bucket": bucket, "frames": int(m.sum()), "accuracy": float((gt == pred)[m].mean()),
                         "H recall": rec("H"), "A recall": rec("A"), "D recall": rec("D"), "share of errors": float(wrong[m].sum() / max(1, wrong.sum()))})
    return res, pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt")
    ap.add_argument("--proxy")
    args = ap.parse_args()
    if bool(args.gt) == bool(args.proxy):
        sys.exit("give exactly one of --gt or --proxy")
    read = lambda p: pm.read_label_csv(p)["label"].astype(str).str.strip().str.upper().to_numpy()  # noqa: E731
    table = ft.load()
    res, tab = slice_scores(table, read(args.gt or args.proxy), read(args.pred))
    print("{}: spellF1 {:.4f}  macroF1 {:.4f}  spells pred {} vs ref {}  home share err {:+.3f}".format(
        "vs proxy (difference, not accuracy)" if args.proxy else "vs labels", res["spell_f1_mean"], res["macro_f1"],
        res["n_pred_spells"], res["n_gt_spells"], res["share_err"]))
    print(tab.round(3).to_string(index=False))
