"""Contract tests for the four leak-safe model-input schemas."""

import numpy as np
import pandas as pd
import pytest

from src.features import model_inputs as mi


def _table(n=8):
    frame = np.arange(n, dtype=np.int32)
    period = np.where(frame < n // 2, 1, 2).astype(np.int8)
    data = {
        "frame": frame,
        "period": period,
        "home_attacks_right": np.ones(n, dtype=bool),
        "n_home": np.full(n, 2, dtype=np.int8),
        "n_away": np.full(n, 2, dtype=np.int8),
        "home_shape_conf": np.ones(n, dtype=np.float32),
        "away_shape_conf": np.ones(n, dtype=np.float32),
        "home_dline_m": frame.astype(np.float32),
        "home_cen_m": (frame + 10).astype(np.float32),
        "home_fwd_m": (frame + 20).astype(np.float32),
        "home_line_gap_m": (frame + 20).astype(np.float32),
        "home_width_m": (frame + 30).astype(np.float32),
        "home_depth_m": (frame + 40).astype(np.float32),
        "home_speed_ms": (frame + 1).astype(np.float32),
        "away_dline_m": frame.astype(np.float32),
        "away_cen_m": (frame + 10).astype(np.float32),
        "away_fwd_m": (frame + 20).astype(np.float32),
        "away_line_gap_m": (frame + 20).astype(np.float32),
        "away_width_m": (frame + 30).astype(np.float32),
        "away_depth_m": (frame + 40).astype(np.float32),
        "away_speed_ms": (frame + 1).astype(np.float32),
        "ball_visible": np.ones(n, dtype=bool),
        "ball_x_m": np.full(n, 50.0, dtype=np.float32),
        "ball_y_m": np.full(n, 34.0, dtype=np.float32),
        "ball_home_adv_m": np.full(n, 50.0, dtype=np.float32),
        "home_ball_dist_m": np.full(n, 2.0, dtype=np.float32),
        "away_ball_dist_m": np.full(n, 4.0, dtype=np.float32),
        "ball_margin_m": np.full(n, 2.0, dtype=np.float32),
        "nearest_team": np.zeros(n, dtype=np.int8),
    }
    for name in mi.DEAD_BALL_COLUMNS:
        data[name] = np.zeros(n, dtype=np.float32)
    for name in mi.GENERATED_TAG_COLUMNS:
        data[name] = np.zeros(n, dtype=np.int8)
    return pd.DataFrame(data)


def _slots(n=8):
    return {
        "x": np.tile(np.array([52.0, 40.0, 48.0, 60.0], dtype=np.float32), (n, 1)),
        "y": np.full((n, 4), 34.0, dtype=np.float32),
        "s": np.ones((n, 4), dtype=np.float32),
        "team": np.array([0, 0, 1, 1], dtype=np.int8),
    }


@pytest.mark.parametrize("schema", mi.SCHEMA_NAMES)
def test_exact_allow_list_and_frame_identity(schema):
    table = _table()
    table["label"] = "H"
    table["tag_poss_hint"] = 1
    slots = _slots() if schema in (mi.SCHEMA_BALL, mi.SCHEMA_PHASE) else None

    result = mi.make_inputs(table, (105.0, 68.0), schema, slots=slots, fps=1)

    assert tuple(result.columns) == mi.schema_columns(schema)
    assert result.index.name == "frame_id"
    assert result.index.tolist() == table["frame"].tolist()
    assert "frame" not in result.columns
    assert "label" not in result.columns
    assert not any(name.startswith("tag_") for name in result.columns)
    assert mi.standalone_frame(result).columns[0] == "frame_id"


def test_phase_schema_uses_only_tracking_generated_tags():
    table = _table()
    result = mi.make_inputs(table, (105.0, 68.0), mi.SCHEMA_PHASE, slots=_slots(), fps=1)

    assert set(mi.GENERATED_TAG_COLUMNS).issubset(result.columns)
    assert not any(name.startswith("tag_") for name in result.columns)


def test_changes_never_cross_period_boundary():
    table = _table()
    result = mi.make_inputs(table, (105.0, 68.0), mi.SCHEMA_DYNAMICS, fps=1)

    change = result["home_cen_change_3s"]
    assert np.isnan(change.iloc[4:7]).all()
    assert change.iloc[3] == pytest.approx(3.0)
    assert change.iloc[7] == pytest.approx(3.0)


def test_ball_gap_anchors_do_not_cross_period_boundary():
    table = _table(6)
    table["ball_visible"] = [True, False, False, False, False, True]
    table.loc[~table["ball_visible"], ["ball_x_m", "ball_y_m", "ball_home_adv_m"]] = np.nan
    result = mi.make_inputs(table, (105.0, 68.0), mi.SCHEMA_BALL, slots=_slots(6), fps=1)

    assert result.loc[1, "left_anchor_margin_m"] == pytest.approx(2.0)
    assert np.isnan(result.loc[1, "right_anchor_margin_m"])
    assert np.isnan(result.loc[3, "left_anchor_margin_m"])
    assert result.loc[3, "right_anchor_margin_m"] == pytest.approx(2.0)
    assert result.loc[1, "gap_length_s"] == pytest.approx(2.0)
    assert result.loc[3, "gap_length_s"] == pytest.approx(2.0)
    assert np.isnan(result.loc[3, "seconds_since_ball_seen"])
    assert np.isnan(result.loc[2, "seconds_until_ball_seen"])


def test_player_ball_geometry_uses_anonymous_slots():
    result = mi.make_inputs(_table(), (105.0, 68.0), mi.SCHEMA_BALL, slots=_slots(), fps=1)

    assert result.iloc[0]["home_tracks_ahead_ball_count"] == 1
    assert result.iloc[0]["home_tracks_behind_ball_count"] == 1
    assert result.iloc[0]["away_tracks_ahead_ball_count"] == 1
    assert result.iloc[0]["away_tracks_behind_ball_count"] == 1
    assert result.iloc[0]["home_local_support_count_5m"] == 1
    assert result.iloc[0]["away_local_support_count_5m"] == 1


def test_validation_fails_closed():
    with pytest.raises(ValueError, match="unknown input schema"):
        mi.make_inputs(_table(), (105.0, 68.0), "unknown")
    with pytest.raises(ValueError, match="require anonymous tracking slot arrays"):
        mi.make_inputs(_table(), (105.0, 68.0), mi.SCHEMA_BALL)
    with pytest.raises(ValueError, match="fps must be positive"):
        mi.make_inputs(_table(), (105.0, 68.0), mi.SCHEMA_CORE, fps=0)
    with pytest.raises(ValueError, match="pitch must be a positive"):
        mi.make_inputs(_table(), (0.0, 68.0), mi.SCHEMA_CORE)
