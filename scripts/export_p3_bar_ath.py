"""Export frozen P3 probabilities for the report-only BAR-ATH public window."""

import argparse
import json
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import train_p3
from games import GAME, GAMES
from src.common import sportec
from src.features import frame_table as ft
from src.models import p3_tcn


def export(config_path: str, artifact_dir: str, data_dir=None) -> dict:
    config = p3_tcn.load_config(config_path)
    model, schema, scaler, penalty = p3_tcn.load_artifact(artifact_dir, config)
    table = ft.load(data_dir=data_dir)
    placeholder = np.full(len(table), "D")
    match = sportec.MatchData(
        table=table,
        slots=ft.load_slots(data_dir=data_dir),
        labels={"raw": placeholder, "adjusted": placeholder},
        pitch=GAMES[GAME]["pitch"],
        fps=25,
    )
    values, periods = p3_tcn.feature_matrix(match, config, schema)
    device = train_p3._device()
    probability = train_p3._probabilities(
        model.to(device), scaler.transform(values), periods, config, device
    )
    path = os.path.join(artifact_dir, "proba_BAR-ATH.npy")
    np.save(path, probability)
    result = {
        "artifact_dir": os.path.relpath(artifact_dir, ROOT),
        "schema": schema,
        "device": str(device),
        "frames": len(probability),
        "switch_penalty": penalty,
        "output": os.path.relpath(path, ROOT),
        "labels_loaded": False,
    }
    print(json.dumps(result, indent=2))
    return result


def export_sportec_test(config_path: str, artifact_dir: str) -> dict:
    """Export the frozen artifact on the already-viewed Sportec test without scoring."""
    config = p3_tcn.load_config(config_path)
    model, schema, scaler, penalty = p3_tcn.load_artifact(artifact_dir, config)
    split = sportec.split()
    device = train_p3._device()
    outputs = {}
    for name in split["test"]:
        match = sportec.load(name)
        values, periods = p3_tcn.feature_matrix(match, config, schema)
        probability = train_p3._probabilities(
            model.to(device), scaler.transform(values), periods, config, device
        )
        path = os.path.join(artifact_dir, "proba_" + name + ".npy")
        np.save(path, probability)
        outputs[name] = {
            "frames": len(probability),
            "output": os.path.relpath(path, ROOT),
        }
    result = {
        "artifact_dir": os.path.relpath(artifact_dir, ROOT),
        "schema": schema,
        "device": str(device),
        "switch_penalty": penalty,
        "outputs": outputs,
        "labels_scored": False,
        "test_previously_viewed": True,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "p3_tcn_10x10.json"))
    parser.add_argument(
        "--artifact-dir",
        default=os.path.join(ROOT, "outputs", "p3_10x10_501", "schema_04"),
    )
    parser.add_argument("--data-dir", help="BAR-ATH bundle root; defaults to MTAH_DATA_DIR or ./data")
    parser.add_argument("--sportec-test", action="store_true", help="export the already-viewed Sportec test instead")
    args = parser.parse_args()
    if args.sportec_test:
        export_sportec_test(args.config, args.artifact_dir)
    else:
        export(args.config, args.artifact_dir, args.data_dir)


if __name__ == "__main__":
    main()
