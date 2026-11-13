"""Smoother iterations for P2 on saved validation probabilities (no retraining). Validation matches only.

    python scripts/tune_smoother_p2.py --variant h_sd_win_1_5_15

Step 2: grid over separate H<->A and live<->D penalties, keep-rule test vs the single-penalty smoother.
Step 3: minimum spell length (1, 2, 3 s) on top of the step-2 winner, penalties re-scaled, keep-rule test.
Step 4: D-class bias (log P(D) + b before decoding, i.e. a dead-ball prior shift), keep-rule test vs current.
Step 5: finer switch-penalty grid around the current one, keep-rule test vs current.
Scored on labels_adjusted. Writes outputs/p2/<variant>/smoother.json.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evaluation import possession_metrics as pm  # noqa: E402
from src.common.sportec import PENALTIES  # noqa: E402
from src.decoder.viterbi import smooth  # noqa: E402
from src.common import keep_rule  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "p2")
LABELS = os.path.join(ROOT, "data", "sportec_labels")
FPS = 25


def run_cfg(data, cfg):
    return {m: pm.label_codes(smooth(p, per, **cfg)) for m, (p, per, _) in data.items()}


def mean_f1(data, preds):
    out = []
    for m, (_, _, gt) in data.items():
        out.append(pm.spell_scores(gt, preds[m], pm.valid_runs(gt))["mean_f1"])
    return float(np.mean(out))


def test(data, base, cand):
    g = np.concatenate([keep_rule.chunk_gains(data[m][2], base[m], cand[m]) for m in data])
    return keep_rule.keep_test(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="h_sd_win_1_5_15")
    a = ap.parse_args()
    d = os.path.join(OUT, a.variant)
    split = json.load(open(os.path.join(ROOT, "configs", "sportec_split.json")))
    data = {m: (np.load(os.path.join(d, "proba", m + ".npy")), np.load(os.path.join(d, "proba", m + "_period.npy")),
                pm.label_codes(pd.read_csv(os.path.join(LABELS, m, "labels_adjusted.csv"))["label"]))
            for m in split["validate"]}
    lam = json.load(open(os.path.join(d, "meta.json")))["switch_penalty"]
    log = []

    base_cfg = {"switch_penalty": lam}
    base = run_cfg(data, base_cfg)
    log.append({"step": "0 current", "cfg": base_cfg, "spellF1": mean_f1(data, base)})
    print(log[-1], flush=True)

    # step 2: separate penalties
    grid = {}
    for t, dd in itertools.product(PENALTIES, PENALTIES):
        cfg = {"switch_penalty": 0, "team_penalty": t, "dead_penalty": dd}
        grid[(t, dd)] = mean_f1(data, run_cfg(data, cfg))
    t, dd = max(grid, key=grid.get)
    s2_cfg = {"switch_penalty": 0, "team_penalty": t, "dead_penalty": dd}
    s2 = run_cfg(data, s2_cfg)
    r = test(data, base, s2)
    log.append({"step": "2 separate penalties", "cfg": s2_cfg, "spellF1": grid[(t, dd)], **r,
                "grid": {"{},{}".format(*k): round(v, 4) for k, v in grid.items()}})
    print({k: v for k, v in log[-1].items() if k != "grid"}, flush=True)
    win_cfg, win = (s2_cfg, s2) if r["keep"] else (base_cfg, base)   # tie: keep the simpler single penalty

    # step 3: minimum spell length on top of the winner; penalties re-scaled since the minimum takes part of their job
    for sec in (1, 2, 3):
        best = None
        for scale in (1.0, 0.5, 0.25, 0.0):
            cfg = {k: (v * scale if k != "min_frames" else v) for k, v in win_cfg.items()}
            cfg["min_frames"] = sec * FPS
            f = mean_f1(data, run_cfg(data, cfg))
            if best is None or f > best[0]:
                best = (f, cfg)
        cand = run_cfg(data, best[1])
        r = test(data, win, cand)
        log.append({"step": "3 min {} s".format(sec), "cfg": best[1], "spellF1": best[0], **r})
        print(log[-1], flush=True)

    # step 4: dead-ball prior shift on top of the current smoother
    best = None
    for b in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        biased = {m: (p * np.array([1, 1, np.exp(b)]), per, gt) for m, (p, per, gt) in data.items()}
        preds = run_cfg(biased, base_cfg)
        f = mean_f1(data, preds)
        if best is None or f > best[0]:
            best = (f, b, preds)
    r = test(data, base, best[2])
    log.append({"step": "4 D bias", "cfg": {**base_cfg, "d_log_bias": best[1]}, "spellF1": best[0], **r})
    print(log[-1], flush=True)

    # step 5: finer penalty grid
    fine = {lp: mean_f1(data, run_cfg(data, {"switch_penalty": lp})) for lp in (12, 15, 17, 20, 23, 26, 30, 35)}
    lp = max(fine, key=fine.get)
    r = test(data, base, run_cfg(data, {"switch_penalty": lp}))
    log.append({"step": "5 fine penalty", "cfg": {"switch_penalty": lp}, "spellF1": fine[lp], **r,
                "grid": {str(k): round(v, 4) for k, v in fine.items()}})
    print(log[-1], flush=True)

    with open(os.path.join(d, "smoother.json"), "w") as f:
        json.dump(log, f, indent=2, default=float)


if __name__ == "__main__":
    main()
