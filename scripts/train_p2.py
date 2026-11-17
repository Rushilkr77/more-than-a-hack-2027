"""Train P2's GBT on the common leak-safe input contract.

The P2 recipe remains: train on raw labels every fifth frame, add a mirrored
copy, early-stop on validation, then tune the common Viterbi penalty against
adjusted validation labels.  The only input path is ``classification_inputs``.
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.common import keep_rule
from src.common import sportec
from evaluation import possession_metrics as pm
from src.decoder.viterbi import smooth
from src.features import classification_inputs as ci
from src.features import frame_table as ft
from src.features import model_inputs as mi
from src.models import p2_gbt

import build_p4_aug


OUT = os.path.join(ROOT, "outputs", "p2")
STEP = 5
BAR_ATH_META = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.xml")
P2_POLICY = ci.InputPolicy(schema=mi.SCHEMA_PHASE, temporal_mode=ci.TemporalMode.OFFLINE_ACAUSAL)
SPEED = ("speed_ms", "pl_slow_share", "pl_fast_share", "dead_player_conf")
WIDTH_CEN = ("width_m", "cen_m", "cx_m", "cy_m")
VARIANTS = {
    "v0": ("v0", ()),
    "v1": ("v1", ()),
    "v2_no_gtag": ("v1", ("gtag_",)),
    "v2_no_speed": ("v1", SPEED),
    "v2_no_width_cen": ("v1", WIDTH_CEN),
    "v3_jumps": ("v1", ("gtag_",)),
}
OPTS = {
    "g_win_1_5_15": {"windows": (1, 5, 15)},
    "g_win_std_diff": {"stats": ("mean", "std", "diff")},
    "h_sd_win_1_5_15": {"windows": (1, 5, 15), "stats": ("mean", "std", "diff")},
    "i_sd_win_1_5_10_15": {"windows": (1, 5, 10, 15), "stats": ("mean", "std", "diff")},
}
VARIANTS.update({name: ("v1", ("gtag_",)) for name in OPTS})
FINAL = "h_sd_win_1_5_15"


def feats(match: sportec.MatchData, variant: str) -> pd.DataFrame:
    """Build P2 features only from the selected shared flat view."""
    return features(sportec.classification_inputs(match, P2_POLICY), variant, match.fps)


def features(inputs: ci.ClassificationInputs, variant: str, fps) -> pd.DataFrame:
    version, drop = VARIANTS[variant]
    options = OPTS.get(variant, {})
    X = p2_gbt.features(inputs, version=version, windows=options.get("windows", (5,)),
                         stats=options.get("stats", ("mean",)), fps=fps)
    return X[[column for column in X if not any(token in column for token in drop)]]


def load_aug(k, match: sportec.MatchData, name) -> sportec.MatchData:
    """Extra damaged copy k of a training match (scripts/build_p4_aug.py); labels are the original match's."""
    table, slots = ft.bundle_table(build_p4_aug.bundle_dir(k, name))
    return sportec.MatchData(table=table, slots=slots, labels=match.labels, pitch=match.pitch, fps=match.fps)


def export(models, variant, split, folder):
    """Save P(H), P(A), P(D) for the held-out test match and the public BAR-ATH window, for scripts/blend.py.
    Nothing is scored here: the test match is scored once, by blend.py --final."""
    for name in split["test"]:
        X = feats(sportec.load(name), variant)
        np.save(os.path.join(folder, "proba_" + name + ".npy"),
                np.mean([p2_gbt.predict_proba(model, X) for model in models], axis=0))
    root = ET.parse(BAR_ATH_META).getroot()
    pitch = (float(root.findtext(".//FieldSize/Width")), float(root.findtext(".//FieldSize/Height")))
    X = features(ci.make_classification_inputs(ft.load(), ft.load_slots(), pitch, P2_POLICY, fps=25), variant, 25)
    np.save(os.path.join(folder, "proba_BAR-ATH.npy"),
            np.mean([p2_gbt.predict_proba(model, X) for model in models], axis=0))


def run(variant, split, data, seeds=(0,), aug=0, export_probas=False):
    t0 = time.time()
    tag = variant + ("_aug{}".format(aug) if aug else "")
    tag += "" if len(seeds) == 1 else "_ens" if len(seeds) == 3 else "_ens{}".format(len(seeds))
    Xs, ys = [], []
    for name in split["train"]:
        copies = [data[name]] + [load_aug(k, data[name], name) for k in range(1, aug + 1)]
        for match in copies + [sportec.mirrored(copy) for copy in copies]:
            Xs.append(feats(match, variant).iloc[::STEP])
            ys.append(match.labels["raw"][::STEP])
    X = pd.concat(Xs, ignore_index=True)
    y = np.concatenate(ys)
    validation = {name: feats(data[name], variant) for name in split["validate"]}
    X_val = pd.concat([X_match.iloc[::STEP] for X_match in validation.values()], ignore_index=True)
    y_val = np.concatenate([data[name].labels["raw"][::STEP] for name in split["validate"]])
    models = [p2_gbt.train(X, y, X_val, y_val, seed=seed) for seed in seeds]
    # seeds > 1: average P(H), P(A), P(D) across seeds before smoothing (the final-submission form)
    probabilities = {name: np.mean([p2_gbt.predict_proba(model, X_match) for model in models], axis=0)
                     for name, X_match in validation.items()}

    penalty, curve = sportec.tune_penalty(probabilities, data)
    rows = []
    for name, proba in probabilities.items():
        raw = p2_gbt.CLASSES[proba.argmax(axis=1)]
        smoothed = smooth(proba, data[name].table["period"], penalty)
        sportec.write(os.path.join(OUT, tag, "raw", name + ".csv"), raw)
        sportec.write(os.path.join(OUT, tag, "smoothed", name + ".csv"), smoothed)
        np.save(os.path.join(OUT, tag, "proba_" + name + ".npy"), proba)
        for output, labels in (("model only", raw), ("smoothed", smoothed)):
            rows.append({"version": tag, "match": name, "output": output,
                         **sportec.score(data[name].labels["adjusted"], labels)})
    metadata = {
        "version": tag, "features": X.shape[1], "train_rows": len(X), "seeds": list(seeds), "aug_copies": aug,
        "iterations": [int(model.n_iter_) for model in models],
        "switch_penalty": penalty, "penalty_curve": {str(key): round(value, 4) for key, value in curve.items()},
        "seconds": round(time.time() - t0, 1),
    }
    if export_probas:
        export(models, variant, split, os.path.join(OUT, tag))
    with open(os.path.join(OUT, tag, "meta.json"), "w") as file:
        json.dump(metadata, file, indent=2)
    return rows, metadata


def compare(base, candidate, split):
    gains = []
    for name in split["validate"]:
        truth = pm.label_codes(pd.read_csv(os.path.join(sportec.LABELS, name, "labels_adjusted.csv"))["label"])
        predictions = [pm.label_codes(pd.read_csv(os.path.join(OUT, version, "smoothed", name + ".csv"))["label"])
                       for version in (base, candidate)]
        gains.append(keep_rule.chunk_gains(truth, *predictions))
    result = keep_rule.keep_test(np.concatenate(gains))
    print("{} -> {}: {}".format(base, candidate, result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=sorted(VARIANTS), action="append")
    parser.add_argument("--ensemble", action="store_true", help="average probabilities over seeds 0, 1, 2")
    parser.add_argument("--seeds", type=int, default=3, help="with --ensemble: number of seeds (0..n-1)")
    parser.add_argument("--aug", type=int, default=0, help="extra damaged training copies (build_p4_aug.py)")
    parser.add_argument("--export", action="store_true",
                        help="also save test-match and BAR-ATH probabilities for scripts/blend.py (not scored)")
    parser.add_argument("--compare", nargs=2, metavar=("BASE", "CANDIDATE"))
    args = parser.parse_args()
    split = sportec.split()
    if args.compare:
        return compare(*args.compare, split)
    data = {name: sportec.load(name) for name in split["train"] + split["validate"]}
    rows = []
    for variant in args.version or [FINAL]:
        result, metadata = run(variant, split, data, tuple(range(args.seeds)) if args.ensemble else (0,),
                               args.aug, args.export)
        rows += result
        print(json.dumps(metadata))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))


if __name__ == "__main__":
    main()
