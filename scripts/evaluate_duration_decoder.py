"""Evaluate one training-only semi-Markov decoder on a frozen run."""

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
from src.decoder import semi_markov


OUT = os.path.join(ROOT, "outputs", "duration")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", default=os.path.join(ROOT, "outputs", "blend", "audit_p2_p4w30_p4w20")
    )
    parser.add_argument("--name", default="incumbent_semi_markov")
    parser.add_argument("--token-frames", type=int, default=5)
    args = parser.parse_args()

    split = sportec.split()
    train = {match: sportec.load(match) for match in split["train"]}
    validation = {match: sportec.load(match) for match in split["validate"]}
    model = semi_markov.fit_duration_model(
        [
            (match.labels["raw"], match.table["period"].to_numpy())
            for match in train.values()
        ],
        token_frames=args.token_frames,
    )

    directory = os.path.join(OUT, args.name)
    rows = {}
    gains = []
    for name, match in validation.items():
        probability = np.load(os.path.join(args.run, "proba_" + name + ".npy"))
        prediction = semi_markov.decode(
            probability, match.table["period"].to_numpy(), model
        )
        sportec.write(os.path.join(directory, "smoothed", name + ".csv"), prediction)
        rows[name] = sportec.score(match.labels["adjusted"], prediction)

        truth_codes = pm.label_codes(match.labels["adjusted"])
        baseline_codes = pm.label_codes(
            pd.read_csv(os.path.join(args.run, "smoothed", name + ".csv"))["label"]
        )
        candidate_codes = pm.label_codes(prediction)
        gains.append(
            keep_rule.chunk_gains(truth_codes, baseline_codes, candidate_codes)
        )

    result = {
        "version": args.name,
        "base_run": os.path.relpath(os.path.abspath(args.run), ROOT),
        "training_matches": list(split["train"]),
        "validation_matches": list(split["validate"]),
        "test_matches_loaded": [],
        "token_frames": model.token_frames,
        "classes": list(model.classes),
        "max_duration_tokens": model.max_duration_tokens,
        "duration_mu": model.duration_mu.tolist(),
        "duration_sigma": model.duration_sigma.tolist(),
        "initial_probability": np.exp(model.initial_log_probability).tolist(),
        "transition_probability": np.exp(model.transition_log_probability).tolist(),
        "validation_per_match": rows,
        "validation_mean": {
            key: float(np.mean([row[key] for row in rows.values()]))
            for key in ("spellF1", "macroF1", "spells", "true_spells")
        },
        "keep_rule": keep_rule.keep_test(np.concatenate(gains)),
        "keep_rule_per_match_mean": {
            name: float(gain.mean())
            for name, gain in zip(validation, gains)
        },
    }
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=float)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
