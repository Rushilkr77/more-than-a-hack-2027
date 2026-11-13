"""Train the P2 baseline (gradient-boosted trees) on the damaged Sportec bundles and score it on the validation matches.

    python scripts/train_p2.py                               # final model h_sd_win_1_5_15
    python scripts/train_p2.py --version v2_no_gtag          # one variant (see VARIANTS)
    python scripts/train_p2.py --compare v1 v2_no_gtag       # keep-rule test on saved smoothed predictions

Train: configs/sportec_split.json "train" on labels_raw (every 5th frame, plus a home/away-mirrored copy).
Validate: "validate" matches. Early stopping on raw labels; the smoother's switch penalty is picked by mean spellF1
against labels_adjusted. The test match is never loaded here (scored once, at the end of P2-P5).

Writes outputs/p2/<version>/raw/<M>.csv (model argmax) and outputs/p2/<version>/smoothed/<M>.csv for the validation
matches, ready for: python scripts/diagnostics_round1.py --pred .../smoothed --model-only .../raw
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evaluation import possession_metrics as pm  # noqa: E402
from src.common import keep_rule  # noqa: E402
from src.common.sportec import LABELS, PENALTIES, load, mirrored, score, split as sportec_split, tune_penalty, write  # noqa: E402,F401  common layer
from src.decoder.viterbi import smooth  # noqa: E402
from src.models import p2_gbt  # noqa: E402
OUT = os.path.join(ROOT, "outputs", "p2")
STEP = 5
SPEED = ("speed_ms", "pl_slow_share", "pl_fast_share", "dead_player_conf")
WIDTH_CEN = ("width_m", "cen_m", "cx_m", "cy_m")
# name -> (feature version, substrings of columns to drop). One change per variant, judged by the keep rule.
VARIANTS = {
    "v0": ("v0", ()),
    "v1": ("v1", ()),
    "v2_no_gtag": ("v1", ("gtag_",)),
    "v2_no_speed": ("v1", SPEED),
    "v2_no_width_cen": ("v1", WIDTH_CEN),
    "v3_jumps": ("v1", ("gtag_",)),            # = v2_no_gtag retrained on bundles with tracker ball teleports
    "v4_no_speed": ("v1", ("gtag_",) + SPEED),
    "v4_no_width_cen": ("v1", ("gtag_",) + WIDTH_CEN),
}
# Round 3: variants of the final v3_jumps, one change each. Options: params (HistGB kwargs), step, mirror, label
# ("raw" / "adjusted"), windows (seconds), stats ("mean", "std", "diff").
GT = ("v1", ("gtag_",))
OPTS = {
    "g_lr05": {"params": {"learning_rate": 0.05, "max_iter": 1000}},
    "g_leaves15": {"params": {"max_leaf_nodes": 15}},
    "g_leaves63": {"params": {"max_leaf_nodes": 63}},
    "g_minleaf200": {"params": {"min_samples_leaf": 200}},
    "g_l2_10": {"params": {"l2_regularization": 10.0}},
    "g_balanced": {"params": {"class_weight": "balanced"}},
    "g_step2": {"step": 2},
    "g_nomirror": {"mirror": False},
    "g_adjusted": {"label": "adjusted"},
    "g_win_1_5_15": {"windows": (1, 5, 15)},
    "g_win_std_diff": {"stats": ("mean", "std", "diff")},
    # round 3b: add-ons on top of the strongest single change (g_win_std_diff)
    "h_sd_minleaf200": {"stats": ("mean", "std", "diff"), "params": {"min_samples_leaf": 200}},
    "h_sd_win_1_5_15": {"stats": ("mean", "std", "diff"), "windows": (1, 5, 15)},
}
VARIANTS.update({k: GT for k in OPTS})
FINAL = "h_sd_win_1_5_15"           # docs/models/P2.md: final approach, why, risks


def feats(table, variant):
    version, drop = VARIANTS[variant]
    o = OPTS.get(variant, {})
    X = p2_gbt.features(table, version, o.get("windows", (5,)), o.get("stats", ("mean",)))
    return X[[c for c in X.columns if not any(d in c for d in drop)]]


def run(variant, split, data):
    version = variant
    o = OPTS.get(variant, {})
    step, lk = o.get("step", STEP), o.get("label", "raw")
    t0 = time.time()
    Xs, ys = [], []
    for m in split["train"]:
        table, lab, length = data[m]
        copies = [(table, lab[lk])] + ([mirrored(table, lab[lk], length)] if o.get("mirror", True) else [])
        for tab, y in copies:
            Xs.append(feats(tab, variant).iloc[::step])
            ys.append(y[::step])
    X, y = pd.concat(Xs, ignore_index=True), np.concatenate(ys)
    val = {m: (feats(data[m][0], variant), data[m]) for m in split["validate"]}
    Xv = pd.concat([v[0].iloc[::STEP] for v in val.values()], ignore_index=True)
    yv = np.concatenate([v[1][1][lk][::STEP] for v in val.values()])
    model = p2_gbt.train(X, y, Xv, yv, **o.get("params", {}))
    proba = {m: p2_gbt.predict_proba(model, Xm) for m, (Xm, _) in val.items()}

    best, curve = tune_penalty(proba, data)                # common smoother, tuned on validation
    rows = []
    for m in val:
        raw_pred = p2_gbt.CLASSES[proba[m].argmax(axis=1)]
        sm = smooth(proba[m], data[m][0]["period"], best)
        write(os.path.join(OUT, version, "raw", m + ".csv"), raw_pred)
        write(os.path.join(OUT, version, "smoothed", m + ".csv"), sm)
        os.makedirs(os.path.join(OUT, version, "proba"), exist_ok=True)       # for smoother tuning without retraining
        np.save(os.path.join(OUT, version, "proba", m + ".npy"), proba[m].astype(np.float32))
        np.save(os.path.join(OUT, version, "proba", m + "_period.npy"), data[m][0]["period"].to_numpy())
        for kind, pred in (("model only", raw_pred), ("smoothed", sm)):
            rows.append({"version": version, "match": m, "output": kind, **score(data[m][1]["adjusted"], pred)})
    meta = {"version": version, "features": X.shape[1], "train_rows": len(X), "iterations": int(model.n_iter_),
            "switch_penalty": best, "penalty_curve": {str(k): round(v, 4) for k, v in curve.items()},
            "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, version, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=sorted(VARIANTS), action="append")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "CANDIDATE"))
    args = ap.parse_args()
    split = sportec_split()
    if args.compare:
        return compare(*args.compare, split)
    versions = args.version or [FINAL]
    data = {m: load(m) for m in split["train"] + split["validate"]}     # the test match is never loaded
    rows = []
    for v in versions:
        r, meta = run(v, split, data)
        rows += r
        print(json.dumps(meta))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))
    print("\nmean over validation matches:")
    print(df.groupby(["version", "output"])[["spellF1", "macroF1", "spells", "true_spells", "mean_s"]].mean()
          .to_string(float_format=lambda x: "{:.3f}".format(x)))


def compare(base, cand, split):
    gains = []
    for m in split["validate"]:
        gt = pm.label_codes(pd.read_csv(os.path.join(LABELS, m, "labels_adjusted.csv"))["label"])
        b, c = (pm.label_codes(pd.read_csv(os.path.join(OUT, v, "smoothed", m + ".csv"))["label"]) for v in (base, cand))
        gains.append(keep_rule.chunk_gains(gt, b, c))
    r = keep_rule.keep_test(np.concatenate(gains))
    print("{} -> {}: mean chunk spellF1 gain {:+.3f}, 95% CI [{:+.3f}, {:+.3f}] over {} chunks -> {}".format(
        base, cand, r["mean_gain"], r["ci95"][0], r["ci95"][1], r["chunks"], "KEEP" if r["keep"] else "no (keep base)"))
    return r


if __name__ == "__main__":
    main()
