"""Official baseline vs P2 on the same data: damaged Sportec validation matches, and the BAR-ATH public 5 minutes.

    python scripts/eval_baseline_p2.py

Baseline = src/evidence/ball_nearest + src/decoder/majority (the starter's nearest-player vote), fixed window +/-50
frames (smooth50), not tuned. P2 = final model h_sd_win_1_5_15, retrained here (same recipe as train_p2.py, ~35 s) with
the saved switch penalty. Sportec is scored on labels_adjusted, as train_p2.py does.

BAR-ATH is inference only. The 5 minute public ground truth holds 17 possessions, so that row is a sanity check, never
a number to tune against (EVALUATION.md). The test match J03WQQ is not loaded.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_p2 as tp  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402
from src.common.sportec import baseline, mirrored, split as sportec_split  # noqa: E402
from src.decoder import majority  # noqa: E402
from src.decoder.viterbi import smooth  # noqa: E402
from src.evidence import ball_nearest  # noqa: E402
from src.features import frame_table as ft  # noqa: E402
from src.models import p2_gbt  # noqa: E402

V = tp.FINAL


def bar_detail(gt_labels, preds, draws=2000):
    """BAR-ATH 5 min: bootstrap interval on spellF1, then each true possession next to what the system did with it."""
    gt = pm.label_codes(gt_labels)
    runs = pm.valid_runs(gt)
    t0 = runs[0][0]
    gts = pm.possession_spells(gt, runs)
    print("\n== bootstrap: resample the {} true possessions, {} draws, 95% interval for spellF1".format(len(gts), draws))
    codes = {name: pm.label_codes(p) for name, p in preds.items()}
    for name, c in codes.items():
        lo, hi = pm.bootstrap_spell_f1(gt, c, runs, draws)
        print("   {:<18} {:.3f}  [{:.3f}, {:.3f}]".format(name, pm.spell_scores(gt, c, runs)["mean_f1"], lo, hi))

    print("\n== frame confusion, rows = truth, cols = P2 (H A D)")
    print(pm.scored_confusion(pm.confusion(gt, codes["P2"]))[:, :3])

    p2 = codes["P2"]
    ps = pm.possession_spells(p2, runs)
    print("\n== each true possession vs P2 (t = seconds into the 5 min; IoU = best same-team P2 spell)")
    print("   #    t_start   dur  team | P2 labels over the span: H%  A%  D% | IoU   matched at 0.3/0.5/0.7")
    for i, (a, b, k) in enumerate(gts):
        share = np.bincount(p2[a:b], minlength=4)[:3] / float(b - a)
        ious = [x for x in pm.spell_overlaps(gts, ps) if x[1] == i]
        best = max([x[0] for x in ious], default=0.0)
        ok = ["Y" if best >= t else "n" for t in (0.3, 0.5, 0.7)]
        print("  {:>2} {:>8.1f} {:>6.1f}   {}  |                          {:>3.0f} {:>3.0f} {:>3.0f} | {:.2f}  {}".format(
            i + 1, (a - t0) / 25.0, (b - a) / 25.0, "HA"[k], *(100 * share), best, "/".join(ok)))
    hit = {j for _, _, j in pm.spell_overlaps(gts, ps)}
    extra = [ps[j] for j in range(len(ps)) if j not in hit]
    print("   P2 spells that overlap no true possession of the same team: {} {}".format(
        len(extra), [("HA"[k], round((a - t0) / 25.0, 1), round((b - a) / 25.0, 1)) for a, b, k in extra]))


def main():
    split = sportec_split()
    penalty = json.load(open(os.path.join(tp.OUT, V, "meta.json")))["switch_penalty"]
    data = {m: tp.load(m) for m in split["train"] + split["validate"]}

    # train P2 exactly as train_p2.run does
    Xs, ys = [], []
    for m in split["train"]:
        table, lab, length = data[m]
        for tab, y in ((table, lab["raw"]), mirrored(table, lab["raw"], length)):
            Xs.append(tp.feats(tab, V).iloc[::tp.STEP])
            ys.append(y[::tp.STEP])
    Xv = pd.concat([tp.feats(data[m][0], V).iloc[::tp.STEP] for m in split["validate"]])
    yv = np.concatenate([data[m][1]["raw"][::tp.STEP] for m in split["validate"]])
    model = p2_gbt.train(pd.concat(Xs, ignore_index=True), np.concatenate(ys), Xv, yv)

    rows = []
    for m in split["validate"]:
        table, lab, _ = data[m]
        proba = p2_gbt.predict_proba(model, tp.feats(table, V))
        p2 = smooth(proba, table["period"], penalty)
        for name, pred in (("baseline raw", majority.decode(ball_nearest.score(table), 0)),
                           ("baseline smooth50", baseline(table)),
                           ("P2 model only", p2_gbt.CLASSES[proba.argmax(axis=1)]), ("P2", p2)):
            rows.append({"data": "Sportec " + m, "system": name, **tp.score(lab["adjusted"], pred)})

    bar = ft.load()
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pd.read_csv(gt_path)["label"].to_numpy()
    assert len(gt) == len(bar), (len(gt), len(bar))
    proba_bar = p2_gbt.predict_proba(model, tp.feats(bar, V))
    p2_bar = smooth(proba_bar, bar["period"], penalty)
    base_bar = baseline(bar)
    starter = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")
    if os.path.exists(starter):
        same = (pd.read_csv(starter)["label"].to_numpy() == base_bar).mean()
        print("check: baseline here vs starter baseline_smooth50.csv, frame agreement {:.4f}".format(same))
    for name, pred in (("baseline raw", majority.decode(ball_nearest.score(bar), 0)), ("baseline smooth50", base_bar),
                       ("P2 model only", p2_gbt.CLASSES[proba_bar.argmax(axis=1)]), ("P2", p2_bar)):
        rows.append({"data": "BAR-ATH public 5 min", "system": name, **tp.score(gt, pred)})

    bar_detail(gt, {"baseline smooth50": base_bar, "P2": p2_bar})

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))
    out = os.path.join(tp.OUT, "eval_baseline_p2.csv")
    df.to_csv(out, index=False)
    print("->", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
