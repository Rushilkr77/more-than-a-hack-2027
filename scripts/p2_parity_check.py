"""Inputs-only parity + BAR-ATH output shape for P2 variants. No BAR-ATH labels are read."""
import sys, os
ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scripts")
import numpy as np, pandas as pd
import train_p2 as tp
from src.features import frame_table as ft
from src.models import p2_gbt
from src.common.sportec import mirrored, split as sportec_split
from src.decoder.viterbi import smooth
from evaluation import possession_metrics as pm

split = sportec_split()
data = {m: tp.load(m) for m in split["train"] + split["validate"]}
bar = ft.load()
g = bar["graded"].to_numpy()

for v in ["v3_jumps", "h_sd_win_1_5_15"]:
    Xb = tp.feats(bar, v)
    Xs = pd.concat([tp.feats(data[m][0], v) for m in split["validate"]])
    Xbg = Xb[g]
    lo, hi = Xs.quantile(0.01), Xs.quantile(0.99)
    out = ((Xbg < lo) | (Xbg > hi)).mean()
    iqr = (Xs.quantile(0.75) - Xs.quantile(0.25)).replace(0, np.nan)
    shift = ((Xbg.median() - Xs.median()).abs() / iqr)
    r = pd.DataFrame({"outside_p1_p99": out, "median_shift_iqr": shift}).sort_values("outside_p1_p99", ascending=False)
    print("\n==", v, "features", Xb.shape[1], "| mean share outside Sportec p1-p99 (expect ~0.02):", round(out.mean(), 3),
          "| features >10% outside:", int((out > 0.10).sum()))
    print(r.head(8).round(3).to_string())
    # train (same recipe) and predict BAR-ATH
    o = tp.OPTS.get(v, {})
    Xs_, ys_ = [], []
    for m in split["train"]:
        t, lab, L = data[m]
        for tab, y in ((t, lab["raw"]), mirrored(t, lab["raw"], L)):
            Xs_.append(tp.feats(tab, v).iloc[::5]); ys_.append(y[::5])
    Xv = pd.concat([tp.feats(data[m][0], v).iloc[::5] for m in split["validate"]]); yv = np.concatenate([data[m][1]["raw"][::5] for m in split["validate"]])
    model = p2_gbt.train(pd.concat(Xs_), np.concatenate(ys_), Xv, yv, **o.get("params", {}))
    p = p2_gbt.predict_proba(model, Xb)
    lab = smooth(p, bar["period"], 20)
    codes = pm.label_codes(pd.Series(lab).where(g, "X"))
    runs = [(a, b) for a, b in pm.valid_runs(codes)]
    st = pm.possession_stats(codes, runs)
    share = pd.Series(lab[g]).value_counts(normalize=True).round(3).to_dict()
    conf = p[g].max(axis=1)
    print("BAR-ATH output: spells", st["n_spells"], "mean s", round(st["mean_duration_s"], 1), "shares", share,
          "| mean max-prob", round(conf.mean(), 3))
    pv = np.concatenate([p2_gbt.predict_proba(model, tp.feats(data[m][0], v)) for m in split["validate"]])
    print("Sportec val     : mean max-prob", round(pv.max(axis=1).mean(), 3))
