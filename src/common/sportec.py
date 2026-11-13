"""Shared train/validate plumbing for P2-P5, so every approach sees the same data, split, smoother and scorer.

What is common (do not fork it; if it must change, change it here and tell everyone):
- data: `load(match)` reads the damaged bundles and labels built by scripts/build_sportec_bundles.py
- split: configs/sportec_split.json (the test match is never loaded here)
- mirroring: `mirrored` (home/away swapped copy for training)
- smoother: src/decoder/viterbi.py `smooth`, with its switch penalty tuned by `tune_penalty`
- scorer: `score` (spellF1 and macroF1 against labels_adjusted, from evaluation/possession_metrics.py)
- keep rule: src/common/keep_rule.py
- baseline: `baseline` (the organiser's nearest-player vote, smooth50)

What is per approach: the model, its inputs and how it is trained. Own smoother experiments are welcome, but report
the common-smoother number as the headline so approaches stay comparable. See docs/common/COMMON_LAYER.md.
"""

import json
import os

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from src.decoder import majority
from src.decoder.viterbi import smooth
from src.evidence import ball_nearest
from src.features import frame_table as ft

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUNDLES = os.path.join(ROOT, "data", "sportec_bundles")
LABELS = os.path.join(ROOT, "data", "sportec_labels")
PENALTIES = (0, 2, 5, 10, 20, 40, 80, 160)     # the common smoother-penalty grid (nats per label change)
BASELINE_WINDOW = 50                           # official smooth50, fixed


def split():
    with open(os.path.join(ROOT, "configs", "sportec_split.json")) as f:
        return json.load(f)                    # keys: train, validate, test


def load(match):
    """(frame table, {"raw": labels, "adjusted": labels}, pitch length m). raw trains, adjusted tunes and scores."""
    table, _ = ft.bundle_table(os.path.join(BUNDLES, match))
    length = ft.bundle_args(os.path.join(BUNDLES, match, "tactical", match + "_tactical_atd.xml"))["pitch"][0]
    lab = {k: pd.read_csv(os.path.join(LABELS, match, "labels_{}.csv".format(k)))["label"].to_numpy()
           for k in ("raw", "adjusted")}
    return table, lab, length


def mirrored(table, labels, length):
    """Home/away swapped copy for augmentation: features from ft.swap_sides, H and A labels exchanged."""
    swapped = np.where(labels == "H", "A", np.where(labels == "A", "H", labels))
    return ft.swap_sides(table, length=length), swapped


def score(gt_labels, pred_labels):
    gt, pred = pm.label_codes(gt_labels), pm.label_codes(pred_labels)
    runs = pm.valid_runs(gt)
    st = pm.possession_stats(pred, runs)
    return {"spellF1": pm.spell_scores(gt, pred, runs)["mean_f1"],
            "macroF1": pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred))),
            "spells": st["n_spells"], "mean_s": st["mean_duration_s"],
            "true_spells": pm.possession_stats(gt, runs)["n_spells"]}


def tune_penalty(proba, data, penalties=PENALTIES):
    """Pick the smoother's switch penalty by mean spellF1 on the validation matches (labels_adjusted).

    proba / data: {match: [frames, 3] P(H), P(A), P(D)} / {match: (table, labels, length)}. Returns (best, curve)."""
    curve = {lam: float(np.mean([score(data[m][1]["adjusted"], smooth(proba[m], data[m][0]["period"], lam))["spellF1"]
                                 for m in proba])) for lam in penalties}
    return max(curve, key=curve.get), curve


def baseline(table):
    """Organiser baseline: nearest-player vote over +/- 50 frames (reproduces baseline_smooth50.csv)."""
    return majority.decode(ball_nearest.score(table), BASELINE_WINDOW)


def write(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(path, index=False)
