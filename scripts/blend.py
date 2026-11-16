"""Blend saved per-frame probabilities of several runs, then smooth once with the common smoother.

Each member is a run folder holding proba_<match>.npy ([frames, 3] = P(H), P(A), P(D)), as written by
scripts/train_p2.py and scripts/train_p4.py (--export adds the test match and the public BAR-ATH window).

    python scripts/blend.py --name p2p4 --members outputs/p2/h_sd_win_1_5_15_ens outputs/p4/04_g0_w30_s5_aug2_ens
    python scripts/blend.py --compare outputs/p4/04_g0_w30_s5_aug2_ens outputs/blend/p2p4
    python scripts/blend.py --name p2p4 --members ... --final      # held-out test match, allowed once

The penalty is tuned on Sportec validation only. BAR-ATH is report-only (a guard against collapse, never a selector).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation import possession_metrics as pm
from src.common import keep_rule
from src.common import sportec
from src.decoder.viterbi import LABELS, smooth
from src.features import frame_table as ft

OUT = os.path.join(ROOT, "outputs", "blend")
BAR_ATH = "BAR-ATH"
BAR_ATH_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
TEST_MARKER = os.path.join(OUT, "TEST_SCORED.json")


def blend(probas, weights=None, mean="arith"):
    """Weighted average of [frames, 3] probability arrays; 'geo' averages log-probabilities and renormalises."""
    w = np.ones(len(probas)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    stack = np.stack([np.asarray(p, float) for p in probas])
    if mean == "arith":
        return np.tensordot(w, stack, axes=1)
    out = np.exp(np.tensordot(w, np.log(np.clip(stack, 1e-6, 1.0)), axes=1))
    return out / out.sum(axis=1, keepdims=True)


def member_proba(members, match, weights, mean):
    return blend([np.load(os.path.join(m, "proba_" + match + ".npy")) for m in members], weights, mean)


def report(gt, proba, period, penalty, kw, folder, match, interval=False):
    rows = {}
    for output, labels in (("model only", LABELS[proba.argmax(axis=1)]), ("smoothed", smooth(proba, period, penalty, **kw))):
        sportec.write(os.path.join(folder, "raw" if output == "model only" else "smoothed", match + ".csv"), labels)
        rows[output] = sportec.score(gt, labels)
        if interval:
            codes, pred = pm.label_codes(gt), pm.label_codes(labels)
            rows[output]["spellF1_low"], rows[output]["spellF1_high"] = pm.bootstrap_spell_f1(
                codes, pred, pm.valid_runs(codes), 1000)
    return rows


def run(args):
    split = sportec.split()
    folder = os.path.join(OUT, args.name)
    kw = {k: v for k, v in (("team_penalty", args.team_penalty), ("dead_penalty", args.dead_penalty)) if v is not None}
    if args.min_frames > 1:
        kw["min_frames"] = args.min_frames
    data = {m: sportec.load(m) for m in split["validate"]}
    probas = {m: member_proba(args.members, m, args.weights, args.mean) for m in data}
    curve = {lam: float(np.mean([sportec.score(data[m].labels["adjusted"],
                                               smooth(probas[m], data[m].table["period"], lam, **kw))["spellF1"]
                                 for m in probas])) for lam in args.penalties}
    penalty = max(curve, key=curve.get)
    results = {m: report(data[m].labels["adjusted"], probas[m], data[m].table["period"], penalty, kw, folder, m)
               for m in probas}
    meta = {"version": args.name, "members": [os.path.relpath(os.path.abspath(m), ROOT) for m in args.members],
            "weights": args.weights, "mean": args.mean, "smoother": kw, "switch_penalty": penalty,
            "penalty_curve": {str(k): round(v, 4) for k, v in curve.items()},
            "validation": {out: {k: float(np.mean([results[m][out][k] for m in results]))
                                 for k in ("spellF1", "macroF1", "spells", "true_spells")}
                           for out in ("model only", "smoothed")},
            "validation_per_match": results}
    if all(os.path.exists(os.path.join(m, "proba_" + BAR_ATH + ".npy")) for m in args.members):
        meta["bar_ath_public_report_only"] = report(
            pd.read_csv(BAR_ATH_GT)["label"].to_numpy(), member_proba(args.members, BAR_ATH, args.weights, args.mean),
            ft.load()["period"], penalty, kw, folder, BAR_ATH, interval=True)
    if args.final:
        if os.path.exists(TEST_MARKER):
            sys.exit("the test match was already scored: " + open(TEST_MARKER).read())
        meta["test"] = {}
        for m in split["test"]:
            match = sportec.load(m)
            meta["test"][m] = report(match.labels["adjusted"], member_proba(args.members, m, args.weights, args.mean),
                                     match.table["period"], penalty, kw, os.path.join(folder, "test"), m)
        with open(TEST_MARKER, "w") as file:
            json.dump({"version": args.name, "test": meta["test"]}, file, indent=2, default=float)
    with open(os.path.join(folder, "meta.json"), "w") as file:
        json.dump(meta, file, indent=2, default=float)
    show = {"validation": meta["validation"], **{k: meta[k] for k in ("bar_ath_public_report_only", "test") if k in meta}}
    print(json.dumps({"version": args.name, "switch_penalty": penalty, "penalty_curve": meta["penalty_curve"]}))
    for part, value in show.items():
        for key, rows in (value.items() if part == "test" else [(part, value)]):
            print(pd.DataFrame(rows).T.rename_axis(key).to_string(float_format=lambda v: "{:.3f}".format(v)))
    return meta


def compare(base, candidate):
    """keep_rule on the smoothed validation output of two run folders."""
    gains = []
    for m in sportec.split()["validate"]:
        truth = pm.label_codes(pd.read_csv(os.path.join(sportec.LABELS, m, "labels_adjusted.csv"))["label"])
        predictions = [pm.label_codes(pd.read_csv(os.path.join(v, "smoothed", m + ".csv"))["label"])
                       for v in (base, candidate)]
        gains.append(keep_rule.chunk_gains(truth, *predictions))
    result = keep_rule.keep_test(np.concatenate(gains))
    print("{} -> {}: {}".format(base, candidate, result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--members", nargs="+", help="run folders with proba_<match>.npy")
    parser.add_argument("--weights", nargs="+", type=float, help="one per member (default equal)")
    parser.add_argument("--mean", choices=("arith", "geo"), default="arith")
    parser.add_argument("--penalties", nargs="+", type=float, default=list(sportec.PENALTIES))
    parser.add_argument("--team-penalty", type=float)
    parser.add_argument("--dead-penalty", type=float)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--final", action="store_true", help="score the held-out test match (refused a second time)")
    parser.add_argument("--compare", nargs=2, metavar=("BASE", "CANDIDATE"))
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.name or not args.members:
        parser.error("--name and --members are required")
    run(args)


if __name__ == "__main__":
    main()
