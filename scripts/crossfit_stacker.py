"""Cross-fit one fixed linear stacker across the two Sportec validation matches."""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation import possession_metrics as pm
from src.common import keep_rule, sportec
from src.decoder.viterbi import smooth
from src.models import linear_stacker


DEFAULT_MEMBERS = (
    os.path.join(ROOT, "outputs", "p2", "h_sd_win_1_5_15_ens"),
    os.path.join(ROOT, "outputs", "p4", "04_g0_w30_s5_aug2_ens"),
    os.path.join(ROOT, "outputs", "p4", "04_g0_w20_s5_aug2_ens"),
    os.path.join(ROOT, "outputs", "p3_10x10_501", "schema_04"),
)
DEFAULT_INCUMBENT = os.path.join(ROOT, "outputs", "blend", "audit_p2_p4w30_p4w20")
DEFAULT_POINT_LEADER = os.path.join(ROOT, "outputs", "blend", "audit_p2_p4w30_p4w20_p3")
OUT = os.path.join(ROOT, "outputs", "stacking")


def _features(members, match):
    return linear_stacker.stack_features(
        [np.load(os.path.join(member, "proba_" + match + ".npy")) for member in members]
    )


def _mean_metrics(probabilities, data, penalty=None):
    rows = {}
    for name, proba in probabilities.items():
        labels = np.asarray(linear_stacker.CLASSES)[proba.argmax(axis=1)]
        if penalty is not None:
            labels = smooth(proba, data[name].table["period"], penalty)
        rows[name] = sportec.score(data[name].labels["adjusted"], labels)
    mean = {
        key: float(np.mean([row[key] for row in rows.values()]))
        for key in ("spellF1", "macroF1", "spells", "true_spells")
    }
    return rows, mean


def _comparison(folder, predictions, data):
    gains = {}
    for name, candidate in predictions.items():
        truth = pm.label_codes(data[name].labels["adjusted"])
        baseline = pm.label_codes(
            pd.read_csv(os.path.join(folder, "smoothed", name + ".csv"))["label"]
        )
        gains[name] = keep_rule.chunk_gains(truth, baseline, pm.label_codes(candidate))
    return {
        "aggregate": keep_rule.keep_test(np.concatenate(list(gains.values()))),
        "per_match_mean": {name: float(values.mean()) for name, values in gains.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="crossfit_p2_p4w30_p4w20_p3")
    parser.add_argument("--members", nargs="+", default=list(DEFAULT_MEMBERS))
    parser.add_argument("--incumbent", default=DEFAULT_INCUMBENT)
    parser.add_argument("--point-leader", default=DEFAULT_POINT_LEADER)
    args = parser.parse_args()

    split = sportec.split()
    validation_names = list(split["validate"])
    if len(validation_names) != 2:
        raise ValueError("the locked match-cross-fit protocol requires exactly two validation matches")
    data = {name: sportec.load(name) for name in validation_names}
    features = {name: _features(args.members, name) for name in validation_names}

    probabilities = {}
    fold_models = {}
    for held_out in validation_names:
        fit_name = next(name for name in validation_names if name != held_out)
        model = linear_stacker.LinearStacker.fit(
            features[fit_name], data[fit_name].labels["adjusted"], regularization_c=1.0
        )
        probabilities[held_out] = model.predict_proba(features[held_out])
        fold_models[held_out] = {
            "fit_match": fit_name,
            "iterations": model.iterations,
        }

    curve = {
        penalty: float(np.mean([
            sportec.score(
                data[name].labels["adjusted"],
                smooth(probabilities[name], data[name].table["period"], penalty),
            )["spellF1"]
            for name in validation_names
        ]))
        for penalty in sportec.PENALTIES
    }
    penalty = max(curve, key=curve.get)
    raw_rows, raw_mean = _mean_metrics(probabilities, data)
    smoothed_rows, smoothed_mean = _mean_metrics(probabilities, data, penalty)

    directory = os.path.join(OUT, args.name)
    smoothed_predictions = {}
    os.makedirs(directory, exist_ok=True)
    for name, proba in probabilities.items():
        np.save(os.path.join(directory, "proba_" + name + ".npy"), proba)
        raw = np.asarray(linear_stacker.CLASSES)[proba.argmax(axis=1)]
        decoded = smooth(proba, data[name].table["period"], penalty)
        smoothed_predictions[name] = decoded
        sportec.write(os.path.join(directory, "raw", name + ".csv"), raw)
        sportec.write(os.path.join(directory, "smoothed", name + ".csv"), decoded)

    final_model = linear_stacker.LinearStacker.fit(
        np.vstack([features[name] for name in validation_names]),
        np.concatenate([data[name].labels["adjusted"] for name in validation_names]),
        regularization_c=1.0,
    )
    final_model.save(os.path.join(directory, "deployment_stacker.npz"))

    result = {
        "version": args.name,
        "method": "two-fold match-cross-fitted standardized multinomial logistic stacking",
        "members": [os.path.relpath(os.path.abspath(member), ROOT) for member in args.members],
        "features": "12 clipped member log probabilities; no match identity",
        "regularization_c": 1.0,
        "class_weight": None,
        "folds": fold_models,
        "validation_matches": validation_names,
        "test_matches_loaded": [],
        "switch_penalty": penalty,
        "penalty_curve": {str(key): round(value, 6) for key, value in curve.items()},
        "raw": raw_mean,
        "smoothed": smoothed_mean,
        "validation_per_match": {
            name: {"raw": raw_rows[name], "smoothed": smoothed_rows[name]}
            for name in validation_names
        },
        "comparison_to_incumbent": _comparison(args.incumbent, smoothed_predictions, data),
        "comparison_to_equal_four_member": _comparison(args.point_leader, smoothed_predictions, data),
        "deployment_artifact_fit_after_crossfit": {
            "matches": validation_names,
            "iterations": final_model.iterations,
            "scored_in_sample": False,
        },
    }
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=float)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
