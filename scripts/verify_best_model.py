"""Verify frozen member hashes and recompute the selected validation result."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.blend import blend
from src.common import sportec
from src.decoder.viterbi import smooth


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_probability_path(root: Path, member: str, match: str) -> Path:
    root = root.resolve()
    path = (root / member / ("proba_" + match + ".npy")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("member probability path escapes the repository root") from error
    return path


def load_verified_probabilities(manifest: dict, root: Path) -> dict:
    candidate = manifest["candidate"]
    members = candidate["members"]
    expected = candidate["validation_probability_sha256"]
    if not members or set(expected) != set(candidate["validation_per_match_spellF1"]):
        raise ValueError("manifest members and validation matches must be complete")
    loaded = {}
    for match, hashes in expected.items():
        if set(hashes) != set(members):
            raise ValueError("manifest must record every member hash for {}".format(match))
        arrays = []
        for member in members:
            path = _safe_probability_path(root, member, match)
            if not path.is_file():
                raise FileNotFoundError(
                    "missing frozen probability {}; regenerate the declared member first".format(path)
                )
            actual = sha256(path)
            if actual != hashes[member]:
                raise ValueError(
                    "probability hash mismatch for {} {}: expected {}, got {}"
                    .format(match, member, hashes[member], actual)
                )
            arrays.append(np.load(path, allow_pickle=False))
        loaded[match] = arrays
    return loaded


def verify(manifest_path: Path, root: Path = ROOT) -> dict:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    candidate = manifest["candidate"]
    probabilities = load_verified_probabilities(manifest, root)
    split = sportec.split()
    if set(probabilities) != set(split["validate"]):
        raise ValueError("frozen manifest must cover exactly the configured validation matches")

    per_match = {}
    for match_name in split["validate"]:
        match = sportec.load(match_name)
        combined = blend(probabilities[match_name])
        prediction = smooth(
            combined,
            match.table["period"],
            candidate["switch_penalty"],
        )
        score = sportec.score(match.labels["adjusted"], prediction)
        expected_spell = candidate["validation_per_match_spellF1"][match_name]
        expected_macro = candidate["validation_per_match_macroF1"][match_name]
        if not np.isclose(score["spellF1"], expected_spell, atol=1e-12, rtol=0):
            raise AssertionError("{} spellF1 changed: {} != {}".format(match_name, score["spellF1"], expected_spell))
        if not np.isclose(score["macroF1"], expected_macro, atol=1e-12, rtol=0):
            raise AssertionError("{} macroF1 changed: {} != {}".format(match_name, score["macroF1"], expected_macro))
        per_match[match_name] = score

    mean_spell = float(np.mean([row["spellF1"] for row in per_match.values()]))
    mean_macro = float(np.mean([row["macroF1"] for row in per_match.values()]))
    if not np.isclose(mean_spell, candidate["validation_spellF1"], atol=1e-12, rtol=0):
        raise AssertionError("mean spellF1 changed")
    if not np.isclose(mean_macro, candidate["validation_macroF1"], atol=1e-12, rtol=0):
        raise AssertionError("mean macroF1 changed")
    result = {
        "verified": True,
        "candidate": candidate["name"],
        "members": candidate["members"],
        "validation_per_match": per_match,
        "validation_mean": {"spellF1": mean_spell, "macroF1": mean_macro},
        "test_matches_loaded": [],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs" / "best_model_selection.json",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest), indent=2, default=float))


if __name__ == "__main__":
    main()
