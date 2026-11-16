"""Train and evaluate P5 on the fixed common Sportec protocol.

Defaults implement the frozen P5 v0 contract in docs/models/P5_PLAN.md.
BAR-ATH and the held-out test match are intentionally absent from this first
runner; validation evidence must be recorded before either is loaded.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src.common import sportec
from src.decoder.viterbi import smooth
from src.features import classification_inputs as ci
from src.features import frame_table as ft
from src.features import model_inputs as mi
from src.models import p5_transformer as p5

import build_p5_aug


OUT = os.path.join(ROOT, "outputs", "p5")
P2_PROBABILITIES = os.path.join(
    ROOT, "outputs", "p2", "h_sd_win_1_5_15", "proba"
)
POLICY = ci.InputPolicy(
    schema=mi.SCHEMA_PHASE,
    temporal_mode=ci.TemporalMode.OFFLINE_ACAUSAL,
)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=ROOT,
            text=True,
        ).strip()
    )


def feature_frame(inputs: ci.ClassificationInputs) -> pd.DataFrame:
    """P5 uses schema 04 but rejects the complete generated-tag block."""

    columns = [
        column
        for column in inputs.frame_features.columns
        if not column.startswith("gtag_")
    ]
    if any(column.startswith("gtag_") for column in columns):
        raise RuntimeError("gtag columns crossed the P5 boundary")
    return inputs.frame_features.loc[:, columns]


def load_augmented(
    copy_index: int, match: sportec.MatchData, match_name: str
) -> sportec.MatchData:
    path = build_p5_aug.bundle_dir(copy_index, match_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            "missing P5 augmentation {}; run scripts/build_p5_aug.py first".format(path)
        )
    table, slots = ft.bundle_table(path)
    return sportec.MatchData(
        table=table,
        slots=slots,
        labels=match.labels,
        pitch=match.pitch,
        fps=match.fps,
    )


def match_inputs(match: sportec.MatchData) -> Tuple[pd.DataFrame, np.ndarray]:
    inputs = sportec.classification_inputs(match, POLICY)
    return feature_frame(inputs), inputs.period


def visibility_report(
    truth: np.ndarray, prediction: np.ndarray, visible: np.ndarray
) -> Dict[str, Dict[str, object]]:
    rows: Dict[str, Dict[str, object]] = {}
    for name, keep in (("visible", visible), ("missing", ~visible)):
        selected_truth = truth[keep]
        selected_prediction = prediction[keep]
        live = np.isin(selected_truth, ("H", "A"))
        rows[name] = {
            "share": round(float(keep.mean()), 3),
            "live_accuracy": (
                round(float((selected_truth[live] == selected_prediction[live]).mean()), 3)
                if live.any()
                else None
            ),
            "dead_recall": (
                round(
                    float(
                        (selected_prediction[selected_truth == "D"] == "D").mean()
                    ),
                    3,
                )
                if (selected_truth == "D").any()
                else None
            ),
        }
    return rows


def run_name(args: argparse.Namespace, suffix: str) -> str:
    return "local_global_w{}_s{}_aug{}_{}".format(
        args.window, args.step, args.aug, suffix
    )


def train_seed(
    args: argparse.Namespace,
    split: Dict[str, Sequence[str]],
    data: Dict[str, sportec.MatchData],
    seed: int,
) -> List[Dict[str, object]]:
    started = time.time()
    length = args.window * 25 // args.step
    if length > args.max_tokens:
        raise ValueError("window/step produces more tokens than max_tokens")

    raw_train: List[Tuple[pd.DataFrame, np.ndarray, np.ndarray]] = []
    for match_name in split["train"]:
        base = data[match_name]
        copies = [base] + [
            load_augmented(copy_index, base, match_name)
            for copy_index in range(1, args.aug + 1)
        ]
        for match in copies + [sportec.mirrored(copy) for copy in copies]:
            frame, period = match_inputs(match)
            raw_train.append(
                (
                    frame.iloc[:: args.step],
                    match.labels["raw"][:: args.step],
                    period[:: args.step],
                )
            )

    scaler = p5.fit_scaler(
        pd.concat([frame for frame, _, _ in raw_train], ignore_index=True)
    )
    training = [
        p5.SequenceExample(
            values=p5.transform(frame, scaler),
            labels=p5.encode(labels),
            period=period,
        )
        for frame, labels, period in raw_train
    ]

    validation: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    validation_examples: List[p5.SequenceExample] = []
    for match_name in split["validate"]:
        frame, period = match_inputs(data[match_name])
        transformed = p5.transform(frame, scaler)
        validation[match_name] = (transformed, period)
        validation_examples.append(
            p5.SequenceExample(
                values=transformed[:: args.step],
                labels=p5.encode(data[match_name].labels["raw"][:: args.step]),
                period=period[:: args.step],
            )
        )

    config = p5.TransformerConfig(
        input_features=training[0].values.shape[1],
        model_width=args.width,
        heads=args.heads,
        layers=args.layers,
        feedforward_width=args.feedforward,
        dropout=args.dropout,
        max_tokens=args.max_tokens,
    )
    model, summary = p5.train(
        training,
        validation_examples,
        config=config,
        length=length,
        seed=seed,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        log=lambda message: print(message, flush=True),
    )
    probabilities = {
        name: p5.predict_proba(
            model, values, period, length=length, step=args.step
        )
        for name, (values, period) in validation.items()
    }
    penalty, curve = sportec.tune_penalty(probabilities, data)
    name = run_name(args, "seed{}".format(seed))
    directory = os.path.join(OUT, name)
    rows: List[Dict[str, object]] = []
    visibility: Dict[str, Dict[str, Dict[str, object]]] = {}
    for match_name, probability in probabilities.items():
        raw = np.asarray(p5.CLASSES)[probability.argmax(axis=1)]
        smoothed = smooth(probability, data[match_name].table["period"], penalty)
        sportec.write(os.path.join(directory, "raw", match_name + ".csv"), raw)
        sportec.write(
            os.path.join(directory, "smoothed", match_name + ".csv"), smoothed
        )
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "proba_" + match_name + ".npy"), probability)
        for output, labels in (("model only", raw), ("smoothed", smoothed)):
            rows.append(
                {
                    "version": name,
                    "match": match_name,
                    "output": output,
                    **sportec.score(data[match_name].labels["adjusted"], labels),
                }
            )
        visibility[match_name] = visibility_report(
            data[match_name].labels["adjusted"],
            smoothed,
            data[match_name].table["ball_visible"].to_numpy(dtype=bool),
        )

    metadata = {
        "version": name,
        "commit": git_commit(),
        "working_tree_dirty": git_dirty(),
        "schema": mi.SCHEMA_PHASE,
        "generated_tags": False,
        "temporal_mode": ci.TemporalMode.OFFLINE_ACAUSAL.value,
        "window_seconds": args.window,
        "step": args.step,
        "augmented_train_copies": args.aug,
        "test_matches_loaded": [],
        "config": asdict(config),
        "training": asdict(summary),
        "train_rows": int(sum(len(example.values) for example in training)),
        "switch_penalty": penalty,
        "penalty_curve": {str(key): round(value, 6) for key, value in curve.items()},
        "visibility": visibility,
        "torch": torch.__version__,
        "seconds": round(time.time() - started, 1),
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler": scaler,
            "metadata": metadata,
        },
        os.path.join(directory, "model.pt"),
    )
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=float)
    return rows


def ensemble(
    args: argparse.Namespace,
    split: Dict[str, Sequence[str]],
    data: Dict[str, sportec.MatchData],
    seeds: Sequence[int],
) -> List[Dict[str, object]]:
    members = [
        os.path.join(OUT, run_name(args, "seed{}".format(seed))) for seed in seeds
    ]
    probabilities = {
        match: np.mean(
            [
                np.load(os.path.join(member, "proba_" + match + ".npy"))
                for member in members
            ],
            axis=0,
        )
        for match in split["validate"]
    }
    penalty, curve = sportec.tune_penalty(probabilities, data)
    name = run_name(args, "ensemble")
    directory = os.path.join(OUT, name)
    rows: List[Dict[str, object]] = []
    for match, probability in probabilities.items():
        raw = np.asarray(p5.CLASSES)[probability.argmax(axis=1)]
        smoothed = smooth(probability, data[match].table["period"], penalty)
        sportec.write(os.path.join(directory, "raw", match + ".csv"), raw)
        sportec.write(os.path.join(directory, "smoothed", match + ".csv"), smoothed)
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "proba_" + match + ".npy"), probability)
        for output, labels in (("model only", raw), ("smoothed", smoothed)):
            rows.append(
                {
                    "version": name,
                    "match": match,
                    "output": output,
                    **sportec.score(data[match].labels["adjusted"], labels),
                }
            )
    metadata = {
        "version": name,
        "members": [os.path.basename(member) for member in members],
        "seeds": list(seeds),
        "switch_penalty": penalty,
        "penalty_curve": {str(key): round(value, 6) for key, value in curve.items()},
        "test_matches_loaded": [],
    }
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return rows


def hybrid_p2_p5(
    args: argparse.Namespace,
    split: Dict[str, Sequence[str]],
    data: Dict[str, sportec.MatchData],
    seeds: Sequence[int],
) -> List[Dict[str, object]]:
    """Evaluate the single pre-declared 50/50 P2/P5 probability blend."""

    p5_members = [
        os.path.join(OUT, run_name(args, "seed{}".format(seed))) for seed in seeds
    ]
    probabilities: Dict[str, np.ndarray] = {}
    for match in split["validate"]:
        p5_probability = np.mean(
            [
                np.load(os.path.join(member, "proba_" + match + ".npy"))
                for member in p5_members
            ],
            axis=0,
        )
        p2_probability = np.load(os.path.join(P2_PROBABILITIES, match + ".npy"))
        probabilities[match] = p5.equal_probability_blend(
            p2_probability, p5_probability
        )

    penalty, curve = sportec.tune_penalty(probabilities, data)
    name = "p2_p5_equal_w{}_s{}_aug{}".format(args.window, args.step, args.aug)
    directory = os.path.join(OUT, name)
    rows: List[Dict[str, object]] = []
    visibility: Dict[str, Dict[str, Dict[str, object]]] = {}
    for match, probability in probabilities.items():
        raw = np.asarray(p5.CLASSES)[probability.argmax(axis=1)]
        smoothed = smooth(probability, data[match].table["period"], penalty)
        sportec.write(os.path.join(directory, "raw", match + ".csv"), raw)
        sportec.write(os.path.join(directory, "smoothed", match + ".csv"), smoothed)
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "proba_" + match + ".npy"), probability)
        for output, labels in (("model only", raw), ("smoothed", smoothed)):
            rows.append(
                {
                    "version": name,
                    "match": match,
                    "output": output,
                    **sportec.score(data[match].labels["adjusted"], labels),
                }
            )
        visibility[match] = visibility_report(
            data[match].labels["adjusted"],
            smoothed,
            data[match].table["ball_visible"].to_numpy(dtype=bool),
        )
    metadata = {
        "version": name,
        "rule": "0.5 * frozen P2 probability + 0.5 * three-seed P5 probability",
        "p2_probability_directory": os.path.relpath(P2_PROBABILITIES, ROOT),
        "p5_members": [os.path.basename(member) for member in p5_members],
        "seeds": list(seeds),
        "switch_penalty": penalty,
        "penalty_curve": {
            str(key): round(value, 6) for key, value in curve.items()
        },
        "visibility": visibility,
        "test_matches_loaded": [],
    }
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--hybrid-p2", action="store_true")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--aug", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    split = sportec.split()
    data = {
        match: sportec.load(match)
        for match in list(split["train"]) + list(split["validate"])
    }
    seeds = args.seed or [0, 1, 2]
    if args.hybrid_p2:
        rows = hybrid_p2_p5(args, split, data, seeds)
    elif args.ensemble:
        rows = ensemble(args, split, data, seeds)
    else:
        rows = []
        for seed in seeds:
            rows.extend(train_seed(args, split, data, seed))
    print(
        pd.DataFrame(rows).to_string(
            index=False, float_format=lambda value: "{:.3f}".format(value)
        )
    )


if __name__ == "__main__":
    main()
