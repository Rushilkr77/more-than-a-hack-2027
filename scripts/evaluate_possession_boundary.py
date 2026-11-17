"""Compare fixed and learned switch costs on identical validation probabilities."""

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Sequence, Tuple

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import possession_metrics as pm
from scripts.blend import blend
from src.common import keep_rule, sportec
from src.decoder.viterbi import smooth
from src.features import classification_inputs as ci, model_inputs as mi
from src.models import possession_boundary as boundary

POLICY = ci.InputPolicy(mi.SCHEMA_BALL, ci.TemporalMode.OFFLINE_ACAUSAL)
PENALTY = 20


def load_members(members: Sequence[Path], match: str) -> Tuple[np.ndarray, dict]:
    arrays = []
    hashes = {}
    for member in members:
        path = member / ("proba_" + match + ".npy")
        payload = path.read_bytes()
        hashes[member.name] = hashlib.sha256(payload).hexdigest()
        values = np.load(BytesIO(payload), allow_pickle=False)
        if (values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all()
                or (values < 0).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("Expected finite H/A/D probability rows in {}".format(path))
        arrays.append(values)
    return blend(arrays), hashes


def train_match(name: str) -> Tuple[np.ndarray, np.ndarray]:
    match = sportec.load(name)
    inputs = sportec.classification_inputs(match, POLICY)
    target = boundary.targets(match.labels["raw"], inputs.period, match.fps)[::boundary.FIT_STEP]
    values = boundary.features(inputs, match.fps)[::boundary.FIT_STEP]
    graded = target >= 0
    return values[graded], target[graded]


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def evaluate(members: Sequence[Path], output: Path) -> dict:
    if not members or len({member.name for member in members}) != len(members):
        raise ValueError("Provide one or more distinct member directory names")
    split = sportec.split()
    predictions = {name: load_members(members, name) for name in split["validate"]}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "events.jsonl").open("x", encoding="utf-8") as log:
        def record(event: str, **fields: object) -> None:
            row = json.dumps({"event": event, **fields}, sort_keys=True)
            print(row, flush=True)
            log.write(row + "\n")
            log.flush()

        run = {
            "commit": git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(git("status", "--porcelain")),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "train": split["train"],
            "validate": split["validate"],
            "test_matches_loaded": [],
            "members": [os.path.relpath(member, ROOT) for member in members],
            "probability_hashes": {name: hashes for name, (_, hashes) in predictions.items()},
            "base_penalty": PENALTY,
            "schema": POLICY.schema,
            "target_radius_seconds": boundary.TARGET_RADIUS_S,
            "context_seconds": boundary.CONTEXT_S,
            "fit_step_frames": boundary.FIT_STEP,
            "max_cost_shift": boundary.MAX_COST_SHIFT,
        }
        record("started", **run)
        samples = []
        for name in split["train"]:
            values, target = train_match(name)
            samples.append((values, target))
            record("training_match", match=name, rows=len(target), positives=int(target.sum()))
        model = boundary.fit(samples)
        record("model_fit", prevalence=model.prevalence)

        metrics = {}
        gains = []
        for name, (proba, _) in predictions.items():
            match = sportec.load(name)
            inputs = sportec.classification_inputs(match, POLICY)
            if len(proba) != len(inputs.period):
                raise ValueError("{} probability rows do not match tracking".format(name))
            probability = model.predict(boundary.features(inputs, match.fps))
            costs = boundary.switch_cost(probability, model.prevalence, PENALTY)
            base = smooth(proba, inputs.period, PENALTY)
            candidate = smooth(proba, inputs.period, PENALTY, frame_switch_cost=costs)
            truth = match.labels["adjusted"]
            truth_codes = pm.label_codes(truth)
            paired = keep_rule.chunk_gains(truth_codes, pm.label_codes(base), pm.label_codes(candidate))
            gains.append(paired)
            target = boundary.targets(truth, inputs.period, match.fps)
            graded = target >= 0
            metrics[name] = {
                "baseline": sportec.score(truth, base),
                "boundary": sportec.score(truth, candidate),
                "boundary_average_precision": float(average_precision_score(target[graded], probability[graded])),
                "boundary_prevalence": float(target[graded].mean()),
                "chunk_gain": float(paired.mean()),
            }
            sportec.write(str(output / "baseline" / (name + ".csv")), base)
            sportec.write(str(output / "boundary" / (name + ".csv")), candidate)
            record("validation_match", match=name, **metrics[name])
        paired_result = keep_rule.keep_test(np.concatenate(gains))
        direction = all(row["boundary"]["spellF1"] >= row["baseline"]["spellF1"] for row in metrics.values())
        result = {
            **run,
            "training_prevalence": model.prevalence,
            "per_match": metrics,
            "keep_rule": paired_result,
            "nonnegative_on_both_matches": direction,
            "keep": paired_result["keep"] and direction,
        }
        (output / "meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        record("finished", keep_rule=paired_result, nonnegative_on_both_matches=direction, keep=result["keep"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=Path, nargs="+", required=True, help="Folders with proba_<match>.npy")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "possession_boundary" / "v1")
    args = parser.parse_args()
    evaluate(args.members, args.output)


if __name__ == "__main__":
    main()
