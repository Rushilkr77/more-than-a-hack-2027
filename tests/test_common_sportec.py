import numpy as np
import pandas as pd

from src.common import sportec
from src.features import classification_inputs as ci
from src.features import model_inputs as mi
from tests.test_model_inputs import _slots, _table


def test_load_retains_complete_tracking_contract(monkeypatch, tmp_path):
    table = pd.DataFrame({"frame": [0, 1]})
    slots = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "y": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        "s": np.ones((2, 2), dtype=np.float32),
        "team": np.array([0, 1], dtype=np.int8),
    }
    labels_dir = tmp_path / "labels" / "M"
    labels_dir.mkdir(parents=True)
    pd.DataFrame({"label": ["H", "A"]}).to_csv(labels_dir / "labels_raw.csv", index=False)
    pd.DataFrame({"label": ["H", "D"]}).to_csv(labels_dir / "labels_adjusted.csv", index=False)
    monkeypatch.setattr(sportec, "BUNDLES", str(tmp_path / "bundles"))
    monkeypatch.setattr(sportec, "LABELS", str(tmp_path / "labels"))
    monkeypatch.setattr(sportec.ft, "bundle_table", lambda _: (table, slots))
    monkeypatch.setattr(sportec.ft, "bundle_args", lambda _: {"pitch": (105.0, 68.0), "fps": 25})

    loaded = sportec.load("M")

    assert loaded.table is table
    assert loaded.slots is slots
    assert loaded.pitch == (105.0, 68.0)
    assert loaded.fps == 25
    assert loaded.labels["raw"].tolist() == ["H", "A"]
    assert loaded.labels["adjusted"].tolist() == ["H", "D"]


def test_classification_inputs_builds_schema4_from_match_contract():
    table = _table()
    slots = _slots()
    match = sportec.MatchData(
        table=table,
        slots=slots,
        labels={"raw": np.array(["H"] * len(table)), "adjusted": np.array(["H"] * len(table))},
        pitch=(105.0, 68.0),
        fps=1,
    )

    result = sportec.classification_inputs(
        match,
        ci.InputPolicy(schema=mi.SCHEMA_PHASE, temporal_mode=ci.TemporalMode.OFFLINE_ACAUSAL),
    )

    assert tuple(result.frame_features.columns) == mi.schema_columns(mi.SCHEMA_PHASE)
    assert result.players.values.shape[:2] == slots["x"].shape
    assert result.ball.values.shape[0] == len(table)


def test_classification_inputs_never_forwards_labels(monkeypatch):
    table = _table()
    slots = _slots()
    labels = {"raw": np.array(["H"] * len(table)), "adjusted": np.array(["D"] * len(table))}
    match = sportec.MatchData(table, slots, labels, (105.0, 68.0), 1)
    policy = ci.InputPolicy(schema=mi.SCHEMA_CORE, temporal_mode=ci.TemporalMode.CAUSAL)
    received = {}

    def build(table_arg, slots_arg, pitch_arg, policy_arg, fps):
        received.update(table=table_arg, slots=slots_arg, pitch=pitch_arg, policy=policy_arg, fps=fps)
        return "inputs"

    monkeypatch.setattr(sportec.ci, "make_classification_inputs", build)

    assert sportec.classification_inputs(match, policy) == "inputs"
    assert received["table"] is table
    assert received["slots"] is slots
    assert received["pitch"] == (105.0, 68.0)
    assert received["policy"] is policy
    assert received["fps"] == 1


def test_tune_penalty_prefers_smoothing_when_probabilities_flicker():
    n = 3000
    truth = np.array(["H"] * n)
    p = np.full((n, 3), 0.1)
    p[:, 0] = 0.8
    p[::7] = [0.1, 0.8, 0.1]                      # frequent wrong-team blips
    match = sportec.MatchData(
        table=pd.DataFrame({"period": np.ones(n)}),
        slots={},
        labels={"raw": truth, "adjusted": truth},
        pitch=(105.0, 68.0),
        fps=25,
    )
    best, curve = sportec.tune_penalty({"m": p}, {"m": match})
    assert best > 0 and curve[best] >= curve[0]
