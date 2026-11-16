"""Reproduce P5 validation scores and common keep-rule comparisons."""

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation import possession_metrics as pm
from src.common import keep_rule, sportec
from src.decoder.viterbi import smooth
from src.models import p5_transformer as p5


P2 = os.path.join(ROOT, "outputs", "p2", "h_sd_win_1_5_15")
P5 = os.path.join(ROOT, "outputs", "p5", "local_global_w10_s5_aug2_ensemble")
HYBRID = os.path.join(ROOT, "outputs", "p5", "p2_p5_equal_w10_s5_aug2")


def _prediction(directory: str, match: str) -> np.ndarray:
    path = os.path.join(directory, "smoothed", match + ".csv")
    return pd.read_csv(path)["label"].to_numpy()


def _probability(directory: str, match: str) -> np.ndarray:
    direct = os.path.join(directory, "proba_" + match + ".npy")
    nested = os.path.join(directory, "proba", match + ".npy")
    return np.load(direct if os.path.exists(direct) else nested)


def _scores(
    directory: str,
    matches: List[str],
    data: Dict[str, sportec.MatchData],
    fixed_penalty: Optional[int] = None,
) -> Dict[str, object]:
    rows: Dict[str, Dict[str, object]] = {}
    for match in matches:
        prediction = (
            _prediction(directory, match)
            if fixed_penalty is None
            else smooth(
                _probability(directory, match),
                data[match].table["period"],
                fixed_penalty,
            )
        )
        rows[match] = sportec.score(data[match].labels["adjusted"], prediction)
    return {
        "matches": rows,
        "mean_spellF1": float(np.mean([row["spellF1"] for row in rows.values()])),
        "mean_macroF1": float(np.mean([row["macroF1"] for row in rows.values()])),
    }


def _keep(
    baseline: str,
    candidate: str,
    matches: List[str],
    data: Dict[str, sportec.MatchData],
) -> Dict[str, object]:
    gains = []
    per_match: Dict[str, float] = {}
    for match in matches:
        truth = pm.label_codes(data[match].labels["adjusted"])
        base = pm.label_codes(_prediction(baseline, match))
        cand = pm.label_codes(_prediction(candidate, match))
        match_gains = keep_rule.chunk_gains(truth, base, cand)
        gains.append(match_gains)
        per_match[match] = float(match_gains.mean())
    return {**keep_rule.keep_test(np.concatenate(gains)), "per_match_mean_gain": per_match}


def main() -> None:
    split = sportec.split()
    matches = list(split["validate"])
    data = {match: sportec.load(match) for match in matches}
    report = {
        "tuned": {
            "p2": _scores(P2, matches, data),
            "p5": _scores(P5, matches, data),
            "hybrid": _scores(HYBRID, matches, data),
        },
        "fixed_penalty_20": {
            "p2": _scores(P2, matches, data, 20),
            "p5": _scores(P5, matches, data, 20),
            "hybrid": _scores(HYBRID, matches, data, 20),
        },
        "keep_rule": {
            "p2_to_p5": _keep(P2, P5, matches, data),
            "p2_to_hybrid": _keep(P2, HYBRID, matches, data),
            "p5_to_hybrid": _keep(P5, HYBRID, matches, data),
        },
    }
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
