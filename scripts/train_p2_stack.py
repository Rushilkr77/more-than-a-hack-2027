"""P2 two-stage stacking: a second GBT reads the first model's probabilities over time.

    python scripts/train_p2_stack.py [--base v3_jumps]

Stage 1 = the base variant. Its training-match probabilities are out-of-fold (leave one training match out, 4 fits),
so stage 2 never learns from probabilities a model produced on its own training frames. Stage 2 inputs: stage-1
P(H), P(A), P(D) plus their past/next means over 1, 5 and 15 s (within each period), plus the base features.
Validation probabilities come from stage 1 trained on all 4 training matches (same as train_p2.py), so the validation
matches stay out of every fit except early stopping. Output: outputs/p2/<base>_stack/ in the train_p2.py layout.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_p2 as tp  # noqa: E402
from src.common.sportec import mirrored, split as sportec_split, tune_penalty  # noqa: E402
from src.decoder.viterbi import smooth  # noqa: E402
from src.models import p2_gbt  # noqa: E402

WINDOWS = (1, 5, 15)


def proba_feats(proba, period):
    df = pd.DataFrame(proba, columns=["pH", "pA", "pD"]).astype("float32")
    out = [df]
    cols = {}
    for _, part in df.groupby(np.asarray(period), sort=False):
        for sec in WINDOWS:
            w = sec * 25
            past = part.rolling(w, min_periods=1).mean()
            nxt = part.iloc[::-1].rolling(w, min_periods=1).mean().iloc[::-1]
            for c in part.columns:
                cols.setdefault("{}_past{}s".format(c, sec), []).append(past[c])
                cols.setdefault("{}_next{}s".format(c, sec), []).append(nxt[c])
    out.append(pd.DataFrame({k: pd.concat(v).sort_index() for k, v in cols.items()}).astype("float32"))
    return pd.concat(out, axis=1)


def fit_stage1(variant, train_ms, data, Xv, yv):
    Xs, ys = [], []
    for m in train_ms:
        table, lab, length = data[m]
        for tab, y in ((table, lab["raw"]), mirrored(table, lab["raw"], length)):
            Xs.append(tp.feats(tab, variant).iloc[::tp.STEP])
            ys.append(y[::tp.STEP])
    return p2_gbt.train(pd.concat(Xs, ignore_index=True), np.concatenate(ys), Xv, yv,
                        **tp.OPTS.get(variant, {}).get("params", {}))


def swap_proba(p):
    return p[:, [1, 0, 2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v3_jumps")
    a = ap.parse_args()
    split = sportec_split()
    data = {m: tp.load(m) for m in split["train"] + split["validate"]}      # the test match is never loaded
    base_X = {m: tp.feats(data[m][0], a.base) for m in data}
    Xv = pd.concat([base_X[m].iloc[::tp.STEP] for m in split["validate"]], ignore_index=True)
    yv = np.concatenate([data[m][1]["raw"][::tp.STEP] for m in split["validate"]])

    # stage 1: out-of-fold on training matches (plus the mirrored copy, scored by the same fold model)
    oof = {}
    for m in split["train"]:
        model = fit_stage1(a.base, [k for k in split["train"] if k != m], data, Xv, yv)
        table, lab, length = data[m]
        mt, _ = mirrored(table, lab["raw"], length)
        oof[m] = (p2_gbt.predict_proba(model, base_X[m]), p2_gbt.predict_proba(model, tp.feats(mt, a.base)), mt)
        print("fold", m, "iterations", model.n_iter_, flush=True)
    full = fit_stage1(a.base, split["train"], data, Xv, yv)
    pv = {m: p2_gbt.predict_proba(full, base_X[m]) for m in split["validate"]}

    # stage 2
    def s2(m, proba, X):
        return pd.concat([X.reset_index(drop=True), proba_feats(proba, data[m][0]["period"]).reset_index(drop=True)],
                         axis=1)

    Xs, ys = [], []
    for m in split["train"]:
        p, pm_, mt = oof[m]
        y = data[m][1]["raw"]
        ym = np.where(y == "H", "A", np.where(y == "A", "H", y))
        Xs += [s2(m, p, base_X[m]).iloc[::tp.STEP], s2(m, pm_, tp.feats(mt, a.base)).iloc[::tp.STEP]]
        ys += [y[::tp.STEP], ym[::tp.STEP]]
    Xv2 = {m: s2(m, pv[m], base_X[m]) for m in split["validate"]}
    model2 = p2_gbt.train(pd.concat(Xs, ignore_index=True), np.concatenate(ys),
                          pd.concat([Xv2[m].iloc[::tp.STEP] for m in split["validate"]], ignore_index=True), yv)
    proba = {m: p2_gbt.predict_proba(model2, Xv2[m]) for m in split["validate"]}

    name = a.base + "_stack"
    best, curve = tune_penalty(proba, data)
    rows = []
    for m in proba:
        raw = p2_gbt.CLASSES[proba[m].argmax(axis=1)]
        sm = smooth(proba[m], data[m][0]["period"], best)
        tp.write(os.path.join(tp.OUT, name, "raw", m + ".csv"), raw)
        tp.write(os.path.join(tp.OUT, name, "smoothed", m + ".csv"), sm)
        os.makedirs(os.path.join(tp.OUT, name, "proba"), exist_ok=True)
        np.save(os.path.join(tp.OUT, name, "proba", m + ".npy"), proba[m].astype(np.float32))
        np.save(os.path.join(tp.OUT, name, "proba", m + "_period.npy"), data[m][0]["period"].to_numpy())
        for kind, pred in (("model only", raw), ("smoothed", sm)):
            rows.append({"match": m, "output": kind, **tp.score(data[m][1]["adjusted"], pred)})
    meta = {"version": name, "stage2_features": int(Xv2[split["validate"][0]].shape[1]),
            "stage2_iterations": int(model2.n_iter_), "switch_penalty": best,
            "penalty_curve": {str(k): round(v, 4) for k, v in curve.items()}}
    json.dump(meta, open(os.path.join(tp.OUT, name, "meta.json"), "w"), indent=2)
    print(json.dumps(meta))
    print(pd.DataFrame(rows).groupby("output")[["spellF1", "macroF1", "spells", "mean_s"]].mean())


if __name__ == "__main__":
    main()
