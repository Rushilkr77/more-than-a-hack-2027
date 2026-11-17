"""Post-selection report for the already-viewed Sportec test match; never tunes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import possession_metrics as pm
from scripts.blend import blend
from src.common import keep_rule, sportec
from src.decoder.viterbi import smooth


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(manifest_path: Path, output: Path) -> dict:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    candidate = manifest["candidate"]
    members = [ROOT / member for member in candidate["members"]]
    split = sportec.split()
    if len(split["test"]) != 1:
        raise ValueError("the frozen report requires exactly one configured Sportec test match")
    name = split["test"][0]
    paths = [member / ("proba_" + name + ".npy") for member in members]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen test probabilities: {}".format(missing))
    arrays = [np.load(path, allow_pickle=False) for path in paths]
    match = sportec.load(name)
    penalty = candidate["switch_penalty"]
    control = smooth(blend(arrays[:3]), match.table["period"], penalty)
    selected = smooth(blend(arrays), match.table["period"], penalty)
    truth = match.labels["adjusted"]
    gains = keep_rule.chunk_gains(
        pm.label_codes(truth), pm.label_codes(control), pm.label_codes(selected)
    )
    result = {
        "report_only": True,
        "selection_changed": False,
        "test_previously_viewed": True,
        "match": name,
        "members": candidate["members"],
        "probability_sha256": {
            member: _hash(path) for member, path in zip(candidate["members"], paths)
        },
        "switch_penalty": penalty,
        "control_three_member": sportec.score(truth, control),
        "selected_four_member": sportec.score(truth, selected),
        "selected_vs_control_keep_rule": keep_rule.keep_test(gains),
        "test_matches_loaded": [name],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs" / "best_model_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "blend" / "frozen_test_report.json",
    )
    args = parser.parse_args()
    report(args.manifest, args.output)


if __name__ == "__main__":
    main()
