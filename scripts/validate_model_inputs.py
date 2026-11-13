#!/usr/bin/env python3
"""Validate all Sportec model inputs without fitting or evaluating a model.

This is the hand-off gate between the data/input layer and model training.  It
builds each schema through the same public function, verifies the fixed split,
checks label alignment separately, and fails if the input contract drifts.
"""

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.features import frame_table as ft  # noqa: E402
from src.features import classification_inputs as ci  # noqa: E402
from src.features import model_inputs as mi  # noqa: E402


def _matches(split: Mapping[str, List[str]]) -> List[str]:
    matches: List[str] = []
    for group in ("train", "validate", "test"):
        if group not in split:
            raise ValueError("split is missing {!r}".format(group))
        matches.extend(split[group])
    if len(matches) != len(set(matches)):
        raise ValueError("a match occurs in more than one split")
    return matches


def _update_profile(
    profiles: MutableMapping[str, Dict[str, object]],
    inputs: pd.DataFrame,
) -> None:
    for name in inputs.columns:
        values = inputs[name].dropna()
        profile = profiles.setdefault(
            name,
            {"dtype": str(inputs[name].dtype), "non_null": 0, "min": None, "max": None},
        )
        if profile["dtype"] != str(inputs[name].dtype):
            raise ValueError(
                "dtype drift for {}: {} versus {}".format(name, profile["dtype"], inputs[name].dtype)
            )
        profile["non_null"] = int(profile["non_null"]) + len(values)
        if values.empty:
            continue
        low, high = values.min(), values.max()
        profile["min"] = low if profile["min"] is None else min(profile["min"], low)
        profile["max"] = high if profile["max"] is None else max(profile["max"], high)


def _validate_labels(labels_path: str, table: pd.DataFrame) -> None:
    labels = pd.read_csv(labels_path)
    if list(labels.columns) != ["frame", "label"]:
        raise ValueError("{} must contain exactly frame,label".format(labels_path))
    if len(labels) != len(table):
        raise ValueError("{} has {} rows; expected {}".format(labels_path, len(labels), len(table)))
    if not labels["frame"].equals(table["frame"].astype(labels["frame"].dtype)):
        raise ValueError("{} frames do not align with the frame table".format(labels_path))
    invalid = sorted(set(labels["label"].dropna()) - {"H", "A", "D"})
    if invalid:
        raise ValueError("{} contains invalid labels: {}".format(labels_path, invalid))


def validate(
    split_path: str,
    bundles_dir: str,
    labels_dir: str,
    schemas: Iterable[str],
) -> Dict[str, object]:
    with open(split_path, encoding="utf-8") as handle:
        split = json.load(handle)
    matches = _matches(split)
    selected = list(schemas)
    for schema in selected:
        mi.schema_columns(schema)

    expected_dtypes: Dict[str, pd.Series] = {}
    profiles: Dict[str, Dict[str, Dict[str, object]]] = {schema: {} for schema in selected}
    rows: Dict[str, int] = {}
    structured: Dict[str, Dict[str, object]] = {}

    for match in matches:
        bundle = os.path.join(bundles_dir, match)
        meta = os.path.join(bundle, "tactical", match + "_tactical_atd.xml")
        args = ft.bundle_args(meta)
        table, slots = ft.bundle_table(bundle)
        rows[match] = len(table)

        for label_name in ("labels_raw.csv", "labels_adjusted.csv"):
            _validate_labels(os.path.join(labels_dir, match, label_name), table)

        for schema in selected:
            inputs = mi.make_inputs(table, args["pitch"], schema, slots=slots, fps=args["fps"])
            if len(inputs) != len(table) or not inputs.index.equals(pd.Index(table["frame"], name="frame_id")):
                raise ValueError("{} {} row identity drift".format(match, schema))
            if tuple(inputs.columns) != mi.schema_columns(schema):
                raise ValueError("{} {} column-order drift".format(match, schema))
            if any(name.startswith("tag_") for name in inputs.columns):
                raise ValueError("{} {} exposes provider Smart Tagging".format(match, schema))
            if schema == mi.SCHEMA_PHASE and not set(mi.GENERATED_TAG_COLUMNS).issubset(inputs.columns):
                raise ValueError("{} schema 04 is missing generated tracking-only tags".format(match))

            dtypes = inputs.dtypes.astype(str)
            if schema not in expected_dtypes:
                expected_dtypes[schema] = dtypes
            elif not dtypes.equals(expected_dtypes[schema]):
                raise ValueError("{} {} dtype contract differs from the first match".format(match, schema))
            _update_profile(profiles[schema], inputs)

        objects = ci.make_object_inputs(
            table,
            slots,
            args["pitch"],
            ci.TemporalMode.OFFLINE_ACAUSAL,
            fps=args["fps"],
        )
        if not objects.frame_id.shape == (len(table),):
            raise ValueError("{} structured frame identity drift".format(match))
        if objects.players.values.shape[:2] != slots["x"].shape:
            raise ValueError("{} structured player shape drift".format(match))
        structured[match] = {
            "slots": int(objects.players.values.shape[1]),
            "player_observed_share": float(objects.players.observed.mean()),
            "ball_observed_share": float(objects.ball.observed.mean()),
        }

        print("ok {:7s} {:6,d} frames".format(match, len(table)))

    all_missing = {
        schema: [name for name, profile in columns.items() if profile["non_null"] == 0]
        for schema, columns in profiles.items()
    }
    constants = {
        schema: [
            name
            for name, profile in columns.items()
            if profile["non_null"] > 0 and profile["min"] == profile["max"]
        ]
        for schema, columns in profiles.items()
    }
    if any(all_missing.values()):
        raise ValueError("all-missing input columns: {}".format(all_missing))
    if any(constants.values()):
        raise ValueError("constant input columns across all matches: {}".format(constants))

    return {
        "matches": matches,
        "rows": rows,
        "schemas": {schema: len(mi.schema_columns(schema)) for schema in selected},
        "structured": {
            "player_channels": list(ci.PLAYER_VALUE_CHANNELS),
            "ball_channels": list(ci.BALL_VALUE_CHANNELS),
            "matches": structured,
            "temporal_mode": ci.TemporalMode.OFFLINE_ACAUSAL.value,
        },
        "provider_tag_columns": [],
        "all_missing_columns": all_missing,
        "constant_columns": constants,
        "status": "ready_for_model_training",
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=os.path.join(ROOT, "configs", "sportec_split.json"))
    parser.add_argument("--bundles", default=os.path.join(ROOT, "data", "sportec_bundles"))
    parser.add_argument("--labels", default=os.path.join(ROOT, "data", "sportec_labels"))
    parser.add_argument("--schema", action="append", choices=mi.SCHEMA_NAMES, dest="schemas")
    args = parser.parse_args(argv)
    report = validate(args.split, args.bundles, args.labels, args.schemas or mi.SCHEMA_NAMES)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
