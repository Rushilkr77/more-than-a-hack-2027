import numpy as np

from scripts import train_p2
from src.common import sportec
from src.features import classification_inputs as ci
from src.features import model_inputs as mi
from src.models import p2_gbt
from tests.test_model_inputs import _slots, _table


def _inputs(n=300):
    table = _table(n)
    table["period"] = np.r_[np.ones(n // 2), np.full(n - n // 2, 2)]
    return ci.make_classification_inputs(
        table,
        _slots(n),
        (105.0, 68.0),
        ci.InputPolicy(schema=mi.SCHEMA_PHASE, temporal_mode=ci.TemporalMode.OFFLINE_ACAUSAL),
        fps=25,
    )


def test_p2_uses_the_selected_flat_view_and_respects_periods():
    inputs = _inputs()
    X = p2_gbt.features(inputs, windows=(1, 5), stats=("mean", "std", "diff"))

    assert set(inputs.frame_features.columns).issubset(X.columns)
    assert "nearest_team" not in X
    assert "ball_margin_m_diff1s" in X
    assert np.isclose(X["ball_margin_m_past5s"].iloc[150], inputs.frame_features["ball_margin_m"].iloc[150])


def test_p2_v0_is_exactly_the_selected_flat_view():
    inputs = _inputs()

    assert p2_gbt.features(inputs, version="v0").equals(inputs.frame_features.astype("float32"))


def test_p2_builds_features_through_the_common_input_layer(monkeypatch):
    inputs = _inputs()
    match = sportec.MatchData(table=None, slots={}, labels={}, pitch=(105.0, 68.0), fps=25)
    received = []
    monkeypatch.setattr(train_p2.sportec, "classification_inputs",
                        lambda match_arg, policy: received.append((match_arg, policy)) or inputs)

    X = train_p2.feats(match, "v3_jumps")

    assert received == [(match, train_p2.P2_POLICY)]
    assert not any(column.startswith("gtag_") for column in X)


def test_p2_training_ignores_only_columns_missing_in_every_training_row():
    inputs = _inputs()
    X = p2_gbt.features(inputs)
    labels = np.array(["H", "A", "D"] * 100)

    model = p2_gbt.train(X.iloc[:240], labels[:240], X.iloc[240:], labels[240:], max_iter=3)

    assert "left_anchor_margin_m" not in model.p2_columns_
    assert p2_gbt.predict_proba(model, X.iloc[240:]).shape == (60, 3)
